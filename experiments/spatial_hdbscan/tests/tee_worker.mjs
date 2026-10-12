// Runs TEE's own k-means worker (extracted from public/js/segmentation.js) on a float32 file.
//   node tee_worker.mjs <vectors.f32> <N> <dim> <k> <seeds, comma-separated> <out.i32>
// Writes the assignments of all seeds one after the other as Int32.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SEG = [process.env.TEE_SEG, path.join(HERE, '../../../TEE/public/js/segmentation.js'),
    path.join(HERE, '../../../public/js/segmentation.js')].find((p) => p && fs.existsSync(p));
if (!SEG) { console.error('segmentation.js not found (set TEE_SEG)'); process.exit(2); }
const [file, N, dim, k, seeds, out] = [process.argv[2], +process.argv[3], +process.argv[4], +process.argv[5], process.argv[6].split(',').map(Number), process.argv[7]];
const buf = fs.readFileSync(file);
const emb = new Float32Array(buf.buffer, buf.byteOffset, N * dim);
const src = fs.readFileSync(SEG, 'utf8');
const mulb = src.match(/function mulberry32\(seed\) \{[\s\S]*?\n\}/)[0];
const workerSrc = src.match(/new Blob\(\[`([\s\S]*?)`\], \{type/)[1].replace('${mulberry32.toString()}', mulb);
const mulberry32 = new Function(mulb + '; return mulberry32;')();
const buildSample = new Function(src.match(/function buildSample\(N, size, rand\) \{[\s\S]*?\n\}/)[0] + '; return buildSample;')();
const all = new Int32Array(N * seeds.length);
seeds.forEach((seed, i) => {
    let res;
    const self = { postMessage: (m) => { res = m; } };
    new Function('self', workerSrc)(self);
    const sample = buildSample(N, Math.min(Math.max(5000, k * 500), N), mulberry32(seed >>> 0));
    self.onmessage({ data: { vectors: emb.slice().buffer, N, dim, k, maxIter: 20, sampleIdx: sample ? sample.buffer : null, seed: seed >>> 0 } });
    all.set(new Int32Array(res.assignments), i * N);
});
fs.writeFileSync(out, Buffer.from(all.buffer));
