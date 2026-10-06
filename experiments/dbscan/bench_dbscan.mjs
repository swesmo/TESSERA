// Browser-feasibility of DBSCAN (Node, single thread = one Web Worker), Gothenburg viewport.
// Uses the sample, PCA and eps exported by export_js_params.py, so the JS result can be checked
// against sklearn on the same input.      node bench_dbscan.mjs
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const H = 385, W = 720, DIM = 128, N = H * W;
const buf = fs.readFileSync(path.join(os.homedir(), 'tee_data/harness-cache/gbg.f32'));
const emb = new Float32Array(buf.buffer, buf.byteOffset, N * DIM);
const P = JSON.parse(fs.readFileSync(path.join(HERE, 'results/js_params.json'), 'utf8'));
const now = () => performance.now();

function projectAll(mean, comp, d) { // PCA projection of every pixel (needed for the assignment step)
    const Z = new Float32Array(N * d), C = Float32Array.from(comp.flat()), M = Float32Array.from(mean);
    for (let i = 0; i < N; i++) for (let j = 0; j < d; j++) {
        let s = 0; for (let a = 0; a < DIM; a++) s += (emb[i * DIM + a] - M[a]) * C[j * DIM + a]; Z[i * d + j] = s;
    }
    return Z;
}

// Plain DBSCAN, brute-force region queries (no index: kd-trees do not help at d >= 16).
// Pass 1 counts neighbours (core test), pass 2 expands clusters from core points (BFS).
function dbscan(X, n, d, eps, minPts) {
    const e2 = eps * eps, cnt = new Int32Array(n);
    for (let a = 0; a < n; a++) {
        let c = 1; // the point itself, as in sklearn
        for (let b = 0; b < n; b++) { if (a === b) continue; let s = 0; for (let j = 0; j < d; j++) { const v = X[a * d + j] - X[b * d + j]; s += v * v; } if (s <= e2) c++; }
        cnt[a] = c;
    }
    const lab = new Int32Array(n).fill(-1), queue = new Int32Array(n);
    let C = 0;
    for (let a = 0; a < n; a++) {
        if (lab[a] !== -1 || cnt[a] < minPts) continue;
        let head = 0, tail = 0; queue[tail++] = a; lab[a] = C;
        while (head < tail) {
            const p = queue[head++];
            if (cnt[p] < minPts) continue; // border point: do not expand
            for (let b = 0; b < n; b++) {
                if (lab[b] !== -1) continue;
                let s = 0; for (let j = 0; j < d; j++) { const v = X[p * d + j] - X[b * d + j]; s += v * v; }
                if (s <= e2) { lab[b] = C; queue[tail++] = b; }
            }
        }
        C++;
    }
    let noise = 0; for (let a = 0; a < n; a++) if (lab[a] < 0) noise++;
    return { clusters: C, noise: noise / n, core: cnt.reduce((s, c) => s + (c >= minPts), 0) };
}

const res = {};
for (const d of [8, 16]) {
    const p0 = P[`n20000_d${d}`];
    let t = now(); const Z = projectAll(p0.mean, p0.comp, d); res[`pca${d}_project_all_ms`] = now() - t;
    console.log(`PCA-${d} project all ${N} px`, res[`pca${d}_project_all_ms`].toFixed(0), 'ms');
    for (const n of [2000, 5000, 10000, 20000]) {
        const p = P[`n${n}_d${d}`];
        // project the sample with ITS OWN PCA (as in Python)
        const C = Float32Array.from(p.comp.flat()), M = Float32Array.from(p.mean), X = new Float32Array(n * d);
        p.idx.forEach((pix, i) => { for (let j = 0; j < d; j++) { let s = 0; for (let a = 0; a < DIM; a++) s += (emb[pix * DIM + a] - M[a]) * C[j * DIM + a]; X[i * d + j] = s; } });
        t = now(); const r = dbscan(X, n, d, p.eps, p.minPts); const dt = now() - t;
        res[`n${n}_d${d}`] = { dbscan_ms: dt, clusters: r.clusters, noise: r.noise, sk_clusters: p.sk_clusters, sk_noise: p.sk_noise, eps: p.eps, minPts: p.minPts };
        console.log(`DBSCAN n=${n} d=${d}`, dt.toFixed(0), 'ms', 'clusters', r.clusters, 'noise', r.noise.toFixed(4), '| sklearn', p.sk_clusters, p.sk_noise);
    }
}
fs.writeFileSync(path.join(HERE, 'results/bench_dbscan_js.json'), JSON.stringify(res, null, 1));
