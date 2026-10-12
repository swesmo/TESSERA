// Runs single building blocks of the clustering methods ported to TEE (public/js/segmentation.js)
// on inputs from files, so that the Python tests can compare them with the Python reference.
//   node port_blocks.mjs ward  <cent.f32> <counts.i32> <K0> <k> <dim> <out.f32>
//   node port_blocks.mjs icm   <D.f32> <labels.i32> <W> <H> <k> <beta> <out.i32>
//   node port_blocks.mjs gauss <emb.f32> <W> <H> <dim> <sx> <sy> <out.f32>
//   node port_blocks.mjs run   <emb.f32> <W> <H> <dim> <k> <method> <seeds,comma> <out.i32>
// 'run' is a whole clustering run with the options of buildSegOptions(); for the metre -> pixel
// conversion of 'gauss' it uses the geotransform of the Gothenburg viewport.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SEG = [process.env.TEE_SEG, path.join(HERE, '../../../TEE/public/js/segmentation.js'),
    path.join(HERE, '../../../public/js/segmentation.js')].find((p) => p && fs.existsSync(p));
if (!SEG) { console.error('segmentation.js not found (set TEE_SEG)'); process.exit(2); }
const src = fs.readFileSync(SEG, 'utf8');
const grab = (name) => {
    const i = src.indexOf(`function ${name}(`); let depth = 0, started = false, j = i;
    for (;;) { const c = src[j]; if (c === '{') { depth++; started = true; } else if (c === '}') { depth--; if (started && depth === 0) return src.slice(i, j + 1); } j++; }
};
const api = new Function(grab('mulberry32') + '\n' + grab('buildSample') + '\n' + src.match(/const SEG_METHODS = \{[\s\S]*?\n\};/)[0] + '\n' +
    grab('pixelSizeMetres') + '\n' + grab('buildSegOptions') + '\nreturn { mulberry32, buildSample, buildSegOptions };')();
const workerSrc = src.match(/new Blob\(\[`([\s\S]*?)`\], \{type/)[1].replace('${mulberry32.toString()}', grab('mulberry32'));
const fn = new Function('self', workerSrc + '; return { wardMerge, gaussFilter, icmSmooth };')({});
const f32 = (file) => { const b = fs.readFileSync(file); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
const i32 = (file) => { const b = fs.readFileSync(file); return new Int32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
const write = (file, arr) => fs.writeFileSync(file, Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
const GBG = { a: 0.00011696237616832214, c: 11.932478203520972, e: -0.00011696237616832214, f: 57.73132766836905 };

const [cmd, ...a] = process.argv.slice(2);
if (cmd === 'ward') {
    write(a[5], fn.wardMerge(f32(a[0]), +a[2], i32(a[1]), +a[3], +a[4]));
} else if (cmd === 'icm') {
    const lab = i32(a[1]);
    fn.icmSmooth(f32(a[0]), lab, +a[2], +a[3], +a[4], +a[5], 5);
    write(a[6], lab);
} else if (cmd === 'gauss') {
    const emb = f32(a[0]);
    fn.gaussFilter(emb, +a[1], +a[2], +a[3], +a[4], +a[5]);
    write(a[6], emb);
} else if (cmd === 'run') {
    const emb = f32(a[0]), [W, H, dim, k] = [+a[1], +a[2], +a[3], +a[4]], N = W * H, seeds = a[6].split(',').map(Number);
    const all = new Int32Array(N * seeds.length);
    seeds.forEach((seed, i) => {
        let out; const self = { postMessage: (m) => { out = m; } };
        new Function('self', workerSrc)(self);
        const { opts, sampleSize } = api.buildSegOptions(a[5], k, N, seed >>> 0, { w: W, h: H }, GBG);
        const sample = api.buildSample(N, sampleSize, api.mulberry32(seed >>> 0));
        self.onmessage({ data: { vectors: emb.slice().buffer, N, dim, k, maxIter: 20, sampleIdx: sample ? sample.buffer : null, seed: seed >>> 0, opts } });
        all.set(new Int32Array(out.assignments), i * N);
    });
    write(a[7], all);
} else { console.error('unknown command'); process.exit(2); }
