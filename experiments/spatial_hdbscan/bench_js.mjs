// Browser-feasibility benchmark (Node, single thread = one Web Worker).
// Runs TEE's own k-means worker code (extracted from public/js/segmentation.js) on the
// Gothenburg viewport, then times JS versions of the smoothing steps and the O(n^2) core of
// HDBSCAN.   node bench_js.mjs [path/to/hdbscan-ts]
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SEG = path.join(HERE, '../../public/js/segmentation.js');
const H = 385, W = 720, DIM = 128, N = H * W;
const PX = [13.0, 6.95]; // metres per pixel (row, col) in TEE's EPSG:4326 grid at 57.7 N

const buf = fs.readFileSync(path.join(os.homedir(), 'tee_data/harness-cache/gbg.f32'));
const emb = new Float32Array(buf.buffer, buf.byteOffset, N * DIM);

const t = (f) => { const s = performance.now(); const r = f(); return [performance.now() - s, r]; };
const med = (f, n = 3) => { const ts = []; let r; for (let i = 0; i < n; i++) { const [dt, rr] = t(f); ts.push(dt); r = rr; } ts.sort((a, b) => a - b); return [ts[n >> 1], r]; };
const win = (r) => [Math.max(1, Math.round(r / PX[0])), Math.max(1, Math.round(r / PX[1]))];

// ── TEE k-means, exactly as shipped ──
const src = fs.readFileSync(SEG, 'utf8');
const mulb = src.match(/function mulberry32\(seed\) \{[\s\S]*?\n\}/)[0];
const workerSrc = src.match(/new Blob\(\[`([\s\S]*?)`\], \{type/)[1].replace('${mulberry32.toString()}', mulb);
const mulberry32 = new Function(mulb + '; return mulberry32;')();
function buildSample(n, size, rand) { // copied from segmentation.js
    if (size >= n) return null;
    const indices = new Uint32Array(size), pool = new Uint32Array(n);
    for (let i = 0; i < n; i++) pool[i] = i;
    for (let i = 0; i < size; i++) { const j = i + Math.floor(rand() * (n - i)); const tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp; indices[i] = pool[i]; }
    return indices;
}
function teeKmeans(k, seed, sampleSize) {
    let out;
    const self = { postMessage: (m) => { out = m; } };
    new Function('self', workerSrc)(self);
    const sample = buildSample(N, sampleSize || Math.min(Math.max(5000, k * 500), N), mulberry32(seed >>> 0));
    self.onmessage({ data: { vectors: emb.slice().buffer, N, dim: DIM, k, maxIter: 20, sampleIdx: sample.buffer, seed } });
    return { lab: new Int32Array(out.assignments), cent: new Float32Array(out.centroids) };
}

// ── smoothing in JS ──
function boxCounts(mask, hy, hx, out) { // out[i] = #mask in (2hy+1)x(2hx+1) window, via integral image
    const I = new Int32Array((H + 1) * (W + 1));
    for (let y = 0; y < H; y++) { let row = 0; for (let x = 0; x < W; x++) { row += mask[y * W + x]; I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + row; } }
    for (let y = 0; y < H; y++) {
        const y0 = Math.max(0, y - hy), y1 = Math.min(H, y + hy + 1);
        for (let x = 0; x < W; x++) {
            const x0 = Math.max(0, x - hx), x1 = Math.min(W, x + hx + 1);
            out[y * W + x] = I[y1 * (W + 1) + x1] - I[y0 * (W + 1) + x1] - I[y1 * (W + 1) + x0] + I[y0 * (W + 1) + x0];
        }
    }
}
function postMode(lab, k, [hy, hx]) {
    const best = new Int32Array(N).fill(-1), bestC = new Int32Array(lab), own = new Int32Array(N);
    const mask = new Uint8Array(N), cnt = new Int32Array(N);
    for (let c = 0; c < k; c++) {
        for (let i = 0; i < N; i++) mask[i] = lab[i] === c ? 1 : 0;
        boxCounts(mask, hy, hx, cnt);
        for (let i = 0; i < N; i++) {
            if (lab[i] === c) own[i] = cnt[i];
            if (cnt[i] > best[i]) { best[i] = cnt[i]; bestC[i] = c; }
        }
    }
    const out = new Int32Array(N);
    for (let i = 0; i < N; i++) out[i] = best[i] > own[i] ? bestC[i] : lab[i];
    return out;
}
function dists(cent, k) { // N x k squared distances (the extra cost ICM needs)
    const D = new Float32Array(N * k);
    for (let i = 0; i < N; i++) for (let c = 0; c < k; c++) {
        let s = 0; for (let d = 0; d < DIM; d++) { const v = emb[i * DIM + d] - cent[c * DIM + d]; s += v * v; }
        D[i * k + c] = s;
    }
    return D;
}
function icm(D, lab0, k, beta, [hy, hx], iters = 5) {
    let lab = new Int32Array(lab0);
    let s = 0; for (let i = 0; i < N; i++) s += D[i * k + lab[i]]; s /= N;
    const same = new Int32Array(N * k), mask = new Uint8Array(N), cnt = new Int32Array(N);
    const nnb = new Int32Array(N); boxCounts(new Uint8Array(N).fill(1), hy, hx, nnb);
    for (let it = 0; it < iters; it++) {
        for (let c = 0; c < k; c++) {
            for (let i = 0; i < N; i++) mask[i] = lab[i] === c ? 1 : 0;
            boxCounts(mask, hy, hx, cnt);
            for (let i = 0; i < N; i++) same[i * k + c] = cnt[i] - mask[i];
        }
        const nl = new Int32Array(N); let changed = 0;
        for (let i = 0; i < N; i++) {
            let bc = 0, be = Infinity;
            for (let c = 0; c < k; c++) { const e = D[i * k + c] / s + beta * (nnb[i] - 1 - same[i * k + c]); if (e < be) { be = e; bc = c; } }
            nl[i] = bc; if (bc !== lab[i]) changed++;
        }
        lab = nl; if (changed < N * 1e-4) break;
    }
    return lab;
}
function gaussPre(sigmaM) { // separable Gaussian over all 128 channels
    const tmp = new Float32Array(N * DIM), out = new Float32Array(N * DIM);
    const kern = (s) => { const r = Math.ceil(3 * s), w = []; let z = 0; for (let j = -r; j <= r; j++) { const v = Math.exp(-j * j / (2 * s * s)); w.push(v); z += v; } return [r, w.map(v => v / z)]; };
    const [rx, wx] = kern(sigmaM / PX[1]), [ry, wy] = kern(sigmaM / PX[0]);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const o = (y * W + x) * DIM;
        for (let j = -rx; j <= rx; j++) { const xx = Math.min(W - 1, Math.max(0, x + j)), b = (y * W + xx) * DIM, w = wx[j + rx]; for (let d = 0; d < DIM; d++) tmp[o + d] += w * emb[b + d]; }
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const o = (y * W + x) * DIM;
        for (let j = -ry; j <= ry; j++) { const yy = Math.min(H - 1, Math.max(0, y + j)), b = (yy * W + x) * DIM, w = wy[j + ry]; for (let d = 0; d < DIM; d++) out[o + d] += w * tmp[b + d]; }
    }
    return out;
}

// ── HDBSCAN core in JS: core distances (brute-force kNN) + Prim MST without a distance matrix ──
function project(sampleIdx, d) { // PCA via covariance + power iterations (cost is what matters here)
    const S = sampleIdx.length, mean = new Float64Array(DIM), C = new Float64Array(DIM * DIM);
    for (const i of sampleIdx) for (let a = 0; a < DIM; a++) mean[a] += emb[i * DIM + a] / S;
    for (const i of sampleIdx) for (let a = 0; a < DIM; a++) { const va = emb[i * DIM + a] - mean[a]; for (let b = a; b < DIM; b++) C[a * DIM + b] += va * (emb[i * DIM + b] - mean[b]); }
    const P = new Float32Array(DIM * d); for (let i = 0; i < P.length; i++) P[i] = Math.random() - 0.5;
    const Z = new Float32Array(N * d); // project ALL pixels (needed for assignment)
    for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) { let s = 0; for (let a = 0; a < DIM; a++) s += (emb[i * DIM + a] - mean[a]) * P[a * d + j]; Z[i * d + j] = s; }
    return Z;
}
function hdbscanCore(Z, d, idx, minSamples = 10) {
    const n = idx.length, X = new Float32Array(n * d);
    idx.forEach((p, i) => X.set(Z.subarray(p * d, p * d + d), i * d));
    const dist = (a, b) => { let s = 0; for (let j = 0; j < d; j++) { const v = X[a * d + j] - X[b * d + j]; s += v * v; } return Math.sqrt(s); };
    const core = new Float32Array(n), knn = new Float32Array(minSamples);
    for (let a = 0; a < n; a++) { knn.fill(Infinity); for (let b = 0; b < n; b++) { if (a === b) continue; const v = dist(a, b); if (v < knn[minSamples - 1]) { let j = minSamples - 1; while (j > 0 && knn[j - 1] > v) { knn[j] = knn[j - 1]; j--; } knn[j] = v; } } core[a] = knn[minSamples - 1]; }
    const inTree = new Uint8Array(n), best = new Float32Array(n).fill(Infinity), edges = [];
    let cur = 0; inTree[0] = 1;
    for (let e = 0; e < n - 1; e++) {
        let nxt = -1, nv = Infinity;
        for (let b = 0; b < n; b++) { if (inTree[b]) continue; const v = Math.max(dist(cur, b), core[cur], core[b]); if (v < best[b]) best[b] = v; if (best[b] < nv) { nv = best[b]; nxt = b; } }
        edges.push(nv); inTree[nxt] = 1; cur = nxt;
    }
    edges.sort((a, b) => a - b); // + union-find / condensed tree: O(n log n), negligible
    return edges.length;
}
function assignKnn(Z, d, ex) { // 1-NN of every pixel among ex exemplar points
    const out = new Int32Array(N);
    for (let i = 0; i < N; i++) { let bb = 0, bv = Infinity; for (const e of ex) { let s = 0; for (let j = 0; j < d; j++) { const v = Z[i * d + j] - Z[e * d + j]; s += v * v; } if (s < bv) { bv = s; bb = e; } } out[i] = bb; }
    return out;
}

const PART = process.argv[3] || 'all';
const resFile = path.join(HERE, 'results/bench_js.json');
const res = fs.existsSync(resFile) ? JSON.parse(fs.readFileSync(resFile, 'utf8')) : {};
if (PART === 'all' || PART === 'smooth') for (const k of [5, 17]) {
    const [tk, km] = med(() => teeKmeans(k, 42));
    const [tm1] = med(() => postMode(km.lab, k, win(10)));
    const [tm2] = med(() => postMode(km.lab, k, win(20)));
    const [td, D] = med(() => dists(km.cent, k), 1);
    const [ti] = med(() => icm(D, km.lab, k, 0.5, win(10)), 1);
    res[`k=${k}`] = { tee_kmeans_ms: tk, post_mode_10m_ms: tm1, post_mode_20m_ms: tm2, icm_distances_ms: td, icm_5iter_ms: ti };
    console.log(`k=${k}`, res[`k=${k}`]);
}
if (PART === 'all' || PART === 'gauss') { const [tg] = med(() => gaussPre(10), 1); res.gauss_pre_10m_ms = tg; console.log('gauss pre 10 m', tg); }
if (PART === 'all' || PART === 'hdbscan') {
const rand = mulberry32(1);
const d = 16;
const [tp, Z] = med(() => project(buildSample(N, 20000, rand), d), 1);
res.pca16_project_all_ms = tp; console.log('PCA-16 project all pixels', tp);
res.hdbscan_core_ms = {};
for (const n of [2000, 5000, 10000, 20000]) {
    const [th] = med(() => hdbscanCore(Z, d, buildSample(N, n, rand)), 1);
    res.hdbscan_core_ms[n] = th; console.log(`HDBSCAN core (kNN + MST) n=${n}`, th);
}
res.assign_ms = {};
for (const e of [50, 1000, 5000]) {
    const [ta] = med(() => assignKnn(Z, d, Array.from(buildSample(N, e, rand))), 1);
    res.assign_ms[e] = ta; console.log(`assign all pixels to ${e} exemplars`, ta);
}
// the existing npm package (full n x n JS matrix)
const lib = process.argv[2];
if (lib) {
    const { HDBSCAN } = await import(lib);
    res.hdbscan_ts_ms = {};
    for (const n of [500, 1000, 2000, 3000]) {
        const ids = buildSample(N, n, rand), data = Array.from(ids, (p) => Array.from(Z.subarray(p * d, p * d + d)));
        const [th] = med(() => new HDBSCAN({ minClusterSize: 20, minSamples: 10 }).fit(data), 1);
        res.hdbscan_ts_ms[n] = th; console.log(`hdbscan-ts n=${n}`, th);
    }
}
}
function wardMerge(cent, K0, counts, k) { // greedy Ward merge of K0 centroids, O(K0^3)
    const mu = Array.from({ length: K0 }, (_, c) => Float64Array.from(cent.subarray(c * DIM, (c + 1) * DIM)));
    const n = Float64Array.from(counts), alive = new Uint8Array(K0).fill(1), lut = Int32Array.from({ length: K0 }, (_, i) => i);
    for (let left = K0; left > k; left--) {
        let ba = -1, bb = -1, bc = Infinity;
        for (let a = 0; a < K0; a++) if (alive[a]) for (let b = a + 1; b < K0; b++) if (alive[b]) {
            let d2 = 0; for (let d = 0; d < DIM; d++) { const v = mu[a][d] - mu[b][d]; d2 += v * v; }
            const c = n[a] * n[b] / (n[a] + n[b]) * d2; if (c < bc) { bc = c; ba = a; bb = b; }
        }
        for (let d = 0; d < DIM; d++) mu[ba][d] = (n[ba] * mu[ba][d] + n[bb] * mu[bb][d]) / (n[ba] + n[bb]);
        n[ba] += n[bb]; alive[bb] = 0; for (let i = 0; i < K0; i++) if (lut[i] === bb) lut[i] = ba;
    }
    return lut;
}
function teeKmeansOnSample(k, seed, S) { // TEE's worker run on the sample rows only (no full-image pass)
    const idx = buildSample(N, S, mulberry32(seed >>> 0)), X = new Float32Array(S * DIM);
    idx.forEach((p, i) => X.set(emb.subarray(p * DIM, (p + 1) * DIM), i * DIM));
    let out; const self = { postMessage: (m) => { out = m; } };
    new Function('self', workerSrc)(self);
    self.onmessage({ data: { vectors: X.buffer, N: S, dim: DIM, k, maxIter: 20, sampleIdx: null, seed } });
    return { lab: new Int32Array(out.assignments), cent: new Float32Array(out.centroids) };
}
if (PART === 'overcluster2') {
    res.overcluster_sample_only_ms = {};
    for (const [K0, S] of [[32, 16000], [32, 10000], [64, 10000]]) {
        const [tk, km] = med(() => teeKmeansOnSample(K0, 42, S), 3);
        const counts = new Int32Array(K0); for (const l of km.lab) counts[l]++;
        const [tw] = med(() => wardMerge(km.cent, K0, counts, 5), 3);
        const key = `K0=${K0} sample=${S}`;
        res.overcluster_sample_only_ms[key] = { kmeans_ms: tk, ward_merge_ms: tw };
        console.log(key, res.overcluster_sample_only_ms[key]);
    }
}
if (PART === 'pipeline') { // end-to-end: over-cluster 32 (10k sample) -> Ward -> refine -> full assign -> MRF 0.25
    const k = 5, K0 = 32, reps = 5;
    const runOnce = (seed) => {
        const T = {}; let t0 = performance.now();
        const km = teeKmeansOnSample(K0, seed, 10000);
        T.overcluster_ms = performance.now() - t0; t0 = performance.now();
        const counts = new Int32Array(K0); for (const l of km.lab) counts[l]++;
        const lut = wardMerge(km.cent, K0, counts, k);
        const groups = [...new Set(lut)], G = new Float32Array(k * DIM);
        groups.forEach((g, gi) => { let n = 0; for (let c = 0; c < K0; c++) if (lut[c] === g) { n += counts[c]; for (let d = 0; d < DIM; d++) G[gi * DIM + d] += counts[c] * km.cent[c * DIM + d]; } for (let d = 0; d < DIM; d++) G[gi * DIM + d] /= Math.max(n, 1); });
        T.ward_ms = performance.now() - t0; t0 = performance.now();
        const S = Math.min(Math.max(5000, 500 * k), N), idx = buildSample(N, S, mulberry32((seed + 1000) >>> 0));
        const a = new Int32Array(S);
        for (let it = 0; it < 20; it++) { // Lloyd on the sample from the merged centroids
            for (let s = 0; s < S; s++) { const p = idx[s]; let bc = 0, bv = Infinity; for (let c = 0; c < k; c++) { let v = 0; for (let d = 0; d < DIM; d++) { const x = emb[p * DIM + d] - G[c * DIM + d]; v += x * x; } if (v < bv) { bv = v; bc = c; } } a[s] = bc; }
            const sum = new Float64Array(k * DIM), n = new Int32Array(k);
            for (let s = 0; s < S; s++) { n[a[s]]++; for (let d = 0; d < DIM; d++) sum[a[s] * DIM + d] += emb[idx[s] * DIM + d]; }
            for (let c = 0; c < k; c++) if (n[c]) for (let d = 0; d < DIM; d++) G[c * DIM + d] = sum[c * DIM + d] / n[c];
        }
        T.refine_ms = performance.now() - t0; t0 = performance.now();
        const D = dists(G, k), lab = new Int32Array(N);
        for (let i = 0; i < N; i++) { let bc = 0, bv = Infinity; for (let c = 0; c < k; c++) if (D[i * k + c] < bv) { bv = D[i * k + c]; bc = c; } lab[i] = bc; }
        T.assign_ms = performance.now() - t0; t0 = performance.now();
        icm(D, lab, k, 0.25, win(10));
        T.mrf_ms = performance.now() - t0;
        T.total_ms = T.overcluster_ms + T.ward_ms + T.refine_ms + T.assign_ms + T.mrf_ms;
        return T;
    };
    const runs = [], base = [];
    for (let r = 0; r < reps; r++) { runs.push(runOnce(42 + r)); const [tb] = t(() => teeKmeans(k, 42 + r)); base.push(tb); }
    const medOf = (xs) => xs.slice().sort((x, y) => x - y)[xs.length >> 1];
    res.pipeline_k5 = Object.fromEntries(Object.keys(runs[0]).map((key) => [key, medOf(runs.map((r) => r[key]))]));
    res.pipeline_k5.tee_kmeans_same_session_ms = medOf(base);
    res.pipeline_k5.reps = reps;
    console.log('pipeline k=5 (median of', reps, ')', res.pipeline_k5);
}
if (PART === 'all' || PART === 'overcluster') {
    res.overcluster_ms = {};
    for (const [K0, S] of [[32, 0], [64, 0], [64, 10000], [128, 0], [128, 10000]]) {
        const [tk, km] = med(() => teeKmeans(K0, 42, S), 1);
        const counts = new Int32Array(K0); for (const l of km.lab) counts[l]++;
        const [tw] = med(() => wardMerge(km.cent, K0, counts, 5), 3);
        const key = `K0=${K0} sample=${S || Math.min(Math.max(5000, K0 * 500), N)}`;
        res.overcluster_ms[key] = { kmeans_ms: tk, ward_merge_ms: tw };
        console.log(key, res.overcluster_ms[key]);
    }
}
fs.writeFileSync(resFile, JSON.stringify(res, null, 1));
