"""DBSCAN on TESSERA embeddings (Christopher, 3 Oct 2026).

    ~/tee-local/venv/bin/python run_dbscan.py [gbg|aut22] [n_seeds]

Pipeline (one run = one seed):
  1. draw a 20k pixel sample (like HDBSCAN on 30 Sep; TEE's k-means sample is 5k-8.5k),
  2. PCA to d dims fitted on the sample (d = 128 means raw embeddings, no PCA),
  3. DBSCAN on the sample with minPts = 2d (Sander et al. 1998 / Schubert et al. 2017 heuristic)
     and eps taken from the k-distance curve: eps = q-quantile of the distance to the
     minPts-th neighbour, so exactly a share q of the sample are core points; "knee" = the point
     of the sorted k-distance curve furthest below its chord,
  4. extend to all N pixels: nearest core point in PCA space; within eps -> its cluster (border
     point), otherwise noise ("dbscan"); variant "nearest" gives every pixel the nearest core's
     cluster (a second pass that removes all noise, Romaric's question on 30 Sep).
Compared with TEE's k-means at the same number of clusters.
Writes results/dbscan_<tag>.json, results/maps_dbscan_<tag>.npz, results/kdist_<tag>.json.
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spatial_hdbscan"))
import tsh  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

which = sys.argv[1] if len(sys.argv) > 1 else "gbg"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ds = tsh.gothenburg() if which == "gbg" else tsh.austria(year=2022)
k_base = 5 if which == "gbg" else 17
E, V = ds["emb"], ds["valid"]
H, W, _ = E.shape
X = E[V]
N = len(X)
SAMPLE = 20_000
DIMS = (8, 16, 32, 128)
QS = (0.5, 0.75, 0.9, 0.95, 0.99, "knee")
seeds = [42] + list(range(1, n_seeds))


def grid(l):
    g = np.full((H, W), -1, np.int64)
    g[V] = l
    return g


def knee(sorted_y):
    """Index of the point of an increasing curve that lies furthest below its chord."""
    x = np.linspace(0, 1, len(sorted_y))
    y = (sorted_y - sorted_y[0]) / max(sorted_y[-1] - sorted_y[0], 1e-12)
    return int(np.argmax(x - y))


def nn1(Zc, Z):
    nn = NearestNeighbors(n_neighbors=1).fit(Zc)
    d, i = [], []
    for s in range(0, len(Z), 100_000):
        a, b = nn.kneighbors(Z[s:s + 100_000])
        d.append(a[:, 0]); i.append(b[:, 0])
    return np.concatenate(d), np.concatenate(i)


def relabel(l):
    """Noise (-1) -> its own label id so that label metrics can be computed."""
    l = l.copy()
    l[l < 0] = l.max() + 1
    return l


def describe(labs, infos, noise_is_label=True):
    """Aggregate metrics over seeds. labs: list of (N,) arrays with -1 = noise."""
    r = dict(seeds=len(labs))
    r["k_found"] = float(np.mean([len(np.unique(l[l >= 0])) for l in labs]))
    r["noise"] = float(np.mean([(l < 0).mean() for l in labs]))
    r["max_share"] = float(np.mean([np.bincount(l[l >= 0]).max() / len(l) if (l >= 0).any() else 0 for l in labs]))
    shares = [np.bincount(l[l >= 0]) / len(l) for l in labs]
    r["k_eff"] = float(np.mean([np.exp(-(p[p > 0] / p.sum() * np.log(p[p > 0] / p.sum())).sum()) if p.sum() else 0
                                for p in shares]))
    rl = [relabel(l) for l in labs]
    r["ARI_mean"], r["ARI_min"] = tsh.stability(rl)
    grids = [grid(l) for l in rl]
    r["coherence"] = float(np.mean([tsh.coherence(g, V) for g in grids]))
    if "fid" in ds:
        fm = [tsh.field_metrics(g, ds["fid"], ds["cls"]) for g in grids]
        for key in fm[0]:
            r[key] = float(np.mean([f[key] for f in fm]))
        # agreement on the pixels DBSCAN keeps (noise excluded) and how much it keeps
        m = ds["fid"][V] > 0
        cls = ds["cls"][V]
        from sklearn.metrics import adjusted_mutual_info_score as ami
        kept = [(l >= 0) & m for l in labs]
        r["AMI_kept"] = float(np.mean([ami(cls[k], l[k]) if k.sum() > 100 and len(np.unique(l[k])) > 1 else 0
                                       for l, k in zip(labs, kept)]))
        r["labelled_kept"] = float(np.mean([k.sum() / m.sum() for k in kept]))
    for key in infos[0]:
        if isinstance(infos[0][key], (int, float)):
            r[key] = float(np.mean([i[key] for i in infos]))
    return r


res, maps, kd_curves = {}, {}, {}
runs = {}  # (d, q) -> list of (lab_dbscan, lab_nearest, info)
km_cache = {}
for s in seeds:
    rng = np.random.default_rng(s)
    idx = rng.choice(N, min(SAMPLE, N), replace=False)
    for d in DIMS:
        t0 = time.time()
        if d < X.shape[1]:
            pca = PCA(n_components=d, random_state=s).fit(X[idx])
            Zs = pca.transform(X[idx]).astype(np.float32)
            Z = pca.transform(X).astype(np.float32)
            evr = float(pca.explained_variance_ratio_.sum())
        else:
            Zs, Z, evr = X[idx], X, 1.0
        t_pca = time.time() - t0
        m = 2 * d
        t0 = time.time()
        kd = NearestNeighbors(n_neighbors=m).fit(Zs).kneighbors(Zs)[0][:, m - 1]  # self counts, as in DBSCAN
        t_kd = time.time() - t0
        kds = np.sort(kd)
        if s == 42:
            kd_curves[f"pca{d}"] = dict(q=np.linspace(0, 1, 201).tolist(),
                                        kdist=np.quantile(kds, np.linspace(0, 1, 201)).tolist(),
                                        knee_q=knee(kds) / len(kds), minPts=m)
        for q in QS:
            qq = knee(kds) / len(kds) if q == "knee" else q
            eps = float(np.quantile(kds, qq))
            t0 = time.time()
            db = DBSCAN(eps=eps, min_samples=m, n_jobs=-1).fit(Zs)
            t_db = time.time() - t0
            hl = db.labels_
            core = db.core_sample_indices_
            t0 = time.time()
            if len(core) and hl.max() >= 0:
                dist, j = nn1(Zs[core], Z)
                near = hl[core][j]
                lab = np.where(dist <= eps, near, -1)
            else:
                near = np.zeros(N, int); lab = np.full(N, -1)
            t_assign = time.time() - t0
            info = dict(eps=eps, q=qq, minPts=m, evr=evr, t_pca=t_pca, t_kdist=t_kd, t_dbscan=t_db,
                        t_assign=t_assign, noise_sample=float((hl < 0).mean()), core_share=len(core) / len(Zs),
                        n_clusters_sample=int(hl.max() + 1))
            runs.setdefault((d, q), []).append((lab.astype(np.int16), near.astype(np.int16), info))
        print(f"seed {s} d={d} done", flush=True)

for (d, q), rr in runs.items():
    name = f"DBSCAN pca{d} q={q}" if d < 128 else f"DBSCAN raw128 q={q}"
    labs = [a for a, _, _ in rr]; nears = [b for _, b, _ in rr]; infos = [c for _, _, c in rr]
    res[name] = describe(labs, infos)
    res[name + " | noise->nearest"] = describe(nears, infos)
    # k-means with the same number of clusters (rounded mean over seeds), TEE algorithm, 1 run
    kk = int(round(res[name]["k_found"]))
    if kk >= 2:
        if kk not in km_cache:
            kl = [tsh.sqd(X, tsh.tee_kmeans(X, kk, s)).argmin(1) for s in seeds]
            km_cache[kk] = describe(kl, [dict()])
        res[name + " | kmeans same k"] = dict(km_cache[kk], k=kk)
    maps[name] = grid(labs[0]).astype(np.int16)
    print(f"{name:28s} " + " ".join(f"{a}={b:.3f}" for a, b in res[name].items() if isinstance(b, float)), flush=True)

# baseline k-means at the usual k
kl = [tsh.sqd(X, tsh.tee_kmeans(X, k_base, s)).argmin(1) for s in seeds]
res[f"kmeans (TEE) k={k_base}"] = describe(kl, [dict()])
maps[f"kmeans (TEE) k={k_base}"] = grid(kl[0]).astype(np.int16)

tag = which
(OUT / f"dbscan_{tag}.json").write_text(json.dumps(res, indent=1))
(OUT / f"kdist_{tag}.json").write_text(json.dumps(kd_curves, indent=1))
np.savez_compressed(OUT / f"maps_dbscan_{tag}.npz", **{n.replace("/", "_"): m for n, m in maps.items()})
print("written", OUT / f"dbscan_{tag}.json")
