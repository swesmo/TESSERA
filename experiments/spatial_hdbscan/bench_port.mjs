// Runtime of the clustering methods as ported to TEE (Christopher, 11 Oct 2026).
// Runs the worker code of TEE/public/js/segmentation.js itself (the code that ships), in Node on
// one thread, which corresponds to TEE's single Web Worker. One timed run = what happens after
// the user presses "Go": drawing the samples on the main thread plus the whole worker job. The
// de-quantised copy of the embeddings that TEE makes before every run is the same for all
// methods and is not timed.
//
//   node bench_port.mjs [gbg|aut] [reps=15] [only]
// only: regular expression; time just the methods whose name matches and merge them into the
// existing result file.
//
// Input: ~/tee_data/harness-cache/gbg.f32 (385 x 720) or aut2022.f32 (1128 x 766), raw float32
// (see README). Writes results/bench_port_<dataset>.json with every single timing in ms.
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SEG = [process.env.TEE_SEG, path.join(HERE, '../../TEE/public/js/segmentation.js'),
    path.join(HERE, '../../public/js/segmentation.js')].find((p) => p && fs.existsSync(p));
const which = process.argv[2] || 'gbg';
const reps = +(process.argv[3] || 15);
const only = process.argv[4] ? new RegExp(process.argv[4]) : null;
const DS = { gbg: { file: 'gbg.f32', H: 385, W: 720, px: [6.95, 13.0] }, aut: { file: 'aut2022.f32', H: 1128, W: 766, px: [10, 10] } }[which];
const DIM = 128, { H, W } = DS, N = H * W;

const buf = fs.readFileSync(path.join(os.homedir(), 'tee_data/harness-cache', DS.file));
const emb = new Float32Array(buf.buffer, buf.byteOffset, N * DIM);

const src = fs.readFileSync(SEG, 'utf8');
const grab = (name) => {
    const i = src.indexOf(`function ${name}(`); let depth = 0, started = false, j = i;
    for (;;) { const c = src[j]; if (c === '{') { depth++; started = true; } else if (c === '}') { depth--; if (started && depth === 0) return src.slice(i, j + 1); } j++; }
};
const mulberry32 = new Function(grab('mulberry32') + '; return mulberry32;')();
const buildSample = new Function(grab('buildSample') + '; return buildSample;')();
const workerSrc = src.match(/new Blob\(\[`([\s\S]*?)`\], \{type/)[1].replace('${mulberry32.toString()}', grab('mulberry32'));
const fn = new Function('self', workerSrc + '; return { fitKMeans, sampleCounts, gaussFilter };')({});

// One clustering run. m: {K0, cOver (samples per over-cluster), fullIters, beta, sigmaM}.
// Mirrors buildSegOptions() in segmentation.js, with the sample factor as a parameter.
function run(k, seed, m, vectors) {
    const t0 = performance.now();
    const opts = {};
    let sampleSize = Math.min(Math.max(5000, k * 500), N);
    if (m.K0 > k) {
        opts.K0 = m.K0;
        const refine = buildSample(N, sampleSize, mulberry32((seed + 1000) >>> 0));
        if (refine) opts.refineIdx = refine.buffer;
        sampleSize = Math.min(Math.max(5000, m.K0 * (m.cOver || 300)), N);
    }
    if (m.fullIters) opts.fullIters = m.fullIters;
    if (m.beta || m.sigmaM) { opts.gridW = W; opts.gridH = H; }
    if (m.beta) opts.beta = m.beta;
    if (m.sigmaM) opts.sigma = [m.sigmaM / DS.px[0], m.sigmaM / DS.px[1]];
    const sample = buildSample(N, sampleSize, mulberry32(seed >>> 0));
    let out; const self = { postMessage: (r) => { out = r; } };
    new Function('self', workerSrc)(self);
    self.onmessage({ data: { vectors, N, dim: DIM, k, maxIter: 20, sampleIdx: sample ? sample.buffer : null, seed: seed >>> 0, opts } });
    return performance.now() - t0;
}

// R restarts on one sample: R k-means++ / Lloyd fits, keep the one with the lowest inertia on
// the sample, then one assignment pass (timed as a plain k-means run from that fit's seed).
function runRestarts(k, seed, R, vectors) {
    const t0 = performance.now();
    const S = Math.min(Math.max(5000, k * 500), N);
    const sample = buildSample(N, S, mulberry32(seed >>> 0));
    const rand = mulberry32((seed >>> 0) ^ 0x9e3779b9);
    let best = Infinity;
    for (let r = 0; r < R; r++) {
        const C = fn.fitKMeans(emb, N, DIM, k, 20, sample, rand);
        let inertia = 0;
        for (let si = 0; si < S; si++) {
            const base = sample[si] * DIM; let bd = Infinity;
            for (let c = 0; c < k; c++) { let d2 = 0; for (let d = 0; d < DIM; d++) { const v = emb[base + d] - C[c * DIM + d]; d2 += v * v; } if (d2 < bd) bd = d2; }
            inertia += bd;
        }
        if (inertia < best) best = inertia;
    }
    const tFits = performance.now() - t0;
    // final assignment pass = a plain run minus its own fit; measure a plain run and subtract one fit
    const tOneFit = tFits / R;
    return tFits + run(k, seed, {}, vectors) - tOneFit;
}

const METHODS = {
    'baseline': { },
    'baseline + MRF 0.25': { beta: 0.25 },
    'baseline + 1 full Lloyd': { fullIters: 1 },
    'restarts 3': { restarts: 3 },
    'restarts 10': { restarts: 10 },
    'over32 S=300K0': { K0: 32 },
    'over32 S=500K0': { K0: 32, cOver: 500 },
    'over64 S=300K0': { K0: 64 },
    'over32 S=300K0 + 1 full Lloyd': { K0: 32, fullIters: 1 },
    'over32 S=300K0 + MRF 0.25': { K0: 32, beta: 0.25 },
    'Gauss 10 m + over32 S=300K0': { K0: 32, sigmaM: 10 },
    'Gauss 10 m + over32 S=300K0 + MRF 0.25': { K0: 32, sigmaM: 10, beta: 0.25 },
    'Gauss 10 m + baseline': { sigmaM: 10 },
};
const stats = (ts) => { const s = ts.slice().sort((a, b) => a - b), q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))]; return { median: q(0.5), p25: q(0.25), p75: q(0.75), min: s[0], max: s[s.length - 1] }; };

const resFile = path.join(HERE, `results/bench_port_${which}.json`);
const res = only && fs.existsSync(resFile) ? JSON.parse(fs.readFileSync(resFile, 'utf8'))
    : { dataset: which, N, H, W, reps, node: process.version, cpu: os.cpus()[0].model, date: new Date().toISOString().slice(0, 10), k: {} };
run(5, 1, {}, emb.slice().buffer); run(5, 2, { K0: 32 }, emb.slice().buffer);       // warm-up (JIT)
for (const k of [5, 17]) {
    res.k[k] = res.k[k] || {};
    for (const [name, m] of Object.entries(METHODS)) {
        if (only && !only.test(name)) continue;
        const ts = [];
        for (let r = 0; r < reps; r++) {
            const vectors = emb.slice().buffer;                                       // not timed
            ts.push(m.restarts ? runRestarts(k, 42 + r, m.restarts, vectors) : run(k, 42 + r, m, vectors));
        }
        res.k[k][name] = { ms: ts.map((v) => +v.toFixed(1)), ...stats(ts) };
        const s = res.k[k][name];
        console.log(`k=${String(k).padEnd(2)} ${name.padEnd(42)} median ${s.median.toFixed(0).padStart(5)} ms  [${s.p25.toFixed(0)}-${s.p75.toFixed(0)}]  min ${s.min.toFixed(0)} max ${s.max.toFixed(0)}`);
    }
}
if (!only) {   // the Gaussian filter alone (does not depend on k or the seed)
    const ts = [];
    for (let r = 0; r < Math.min(reps, 5); r++) { const c = emb.slice(); const t0 = performance.now(); fn.gaussFilter(c, W, H, DIM, 10 / DS.px[0], 10 / DS.px[1]); ts.push(performance.now() - t0); }
    res.gauss_filter_ms = { ms: ts.map((v) => +v.toFixed(1)), ...stats(ts) };
    console.log(`Gaussian filter 10 m alone: median ${res.gauss_filter_ms.median.toFixed(0)} ms`);
}
fs.writeFileSync(resFile, JSON.stringify(res, null, 1));
