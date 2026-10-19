"""Over-cluster (k-means with K0 >> k) and merge the K0 centroids to k groups (Ward / average
linkage), or let HDBSCAN on the K0 centroids choose the number of groups itself.
    [YEAR=2022] ~/tee-local/venv/bin/python run_overcluster.py [gbg|aut] [n_seeds] [only]
only: run just the variants whose name contains one of the '|'-separated substrings and merge
them into the existing results, e.g. "HDBSCAN|kmeans (TEE)".
Writes results/overcluster_<tag>.json and maps_overcluster_<tag>.npz."""
import json, os, sys, time
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
import tsh
from run_experiments import OUT

which = sys.argv[1] if len(sys.argv) > 1 else "gbg"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
YEAR = int(os.environ.get("YEAR", 2024))
ds = tsh.gothenburg() if which == "gbg" else tsh.austria(year=YEAR)
k = 5 if which == "gbg" else 17
E, V, px = ds["emb"], ds["valid"], ds["px_m"]
H, W, _ = E.shape
X = E[V]
t0 = time.time(); Xg = tsh.pre_gauss(E, V, 10, px)[V]; t_gauss = time.time() - t0
seeds = [42] + list(range(1, n_seeds))


def grid(l):
    g = np.zeros((H, W), np.int64); g[V] = l; return g


def overmerge(Xin, s, K0, method="ward", gamma=0.0, sample=None, n_init=1):
    t = time.time()
    C = tsh.tee_kmeans(Xin, K0, s, sample=sample, n_init=n_init)
    lab0 = tsh.sqd(Xin, C).argmin(1)
    t_fit = time.time() - t
    t = time.time()
    n = np.bincount(lab0, minlength=K0)
    A = tsh.adjacency(grid(lab0), V, K0) if gamma else None
    lut = tsh.merge_tree(C, n, k, method, A, gamma)
    return lut[lab0], dict(t_fit=t_fit, t_merge=time.time() - t)


def lloyd_from_groups(Xin, s, C, n, lut, iters=20):
    """Start from the pixel-weighted means of the merged centroid groups, run Lloyd on a sample of
    max(5000, 500 g) pixels (g = number of groups), return squared distances of all N to the result."""
    g_n = lut.max() + 1
    G = np.stack([(C[lut == g] * n[lut == g, None]).sum(0) / max(n[lut == g].sum(), 1) for g in range(g_n)]).astype(np.float32)
    rng = np.random.default_rng(s + 1000)
    Xs = Xin[rng.choice(len(Xin), min(max(5000, 500 * g_n), len(Xin)), replace=False)]
    for _ in range(iters):
        a = tsh.sqd(Xs, G).argmin(1)
        G = np.stack([Xs[a == g].mean(0) if (a == g).any() else G[g] for g in range(g_n)]).astype(np.float32)
    return tsh.sqd(Xin, G)


_over = {}


def over_cached(Xin, s, K0, sample=10_000):
    """TEE k-means with K0 clusters (10k sample) + assignment of all N. Cached per (K0, seed) because
    all HDBSCAN merge variants start from it; t_fit is the time of the first (real) computation."""
    key = (id(Xin), K0, s, sample)
    if key not in _over:
        t = time.time()
        C = tsh.tee_kmeans(Xin, K0, s, sample=sample)
        lab0 = tsh.sqd(Xin, C).argmin(1)
        _over[key] = (C, lab0, np.bincount(lab0, minlength=K0), time.time() - t)
    return _over[key]


def hdb_merge(Xin, s, K0, space="raw", weight="unw", mcs=2, sel="leaf", ms=None, iters=0, sample=10_000):
    """Over-cluster (TEE k-means, K0, 10k sample), then merge the K0 centroids with HDBSCAN instead
    of Ward, so HDBSCAN picks the number of groups.
    space:  'raw' = 128-d centroids; 'pca16' = PCA-16 fitted on the same 10k pixel sample.
    weight: 'unw' = one point per centroid; 'rep' = centroid c repeated round(1000 n_c / N) times
            (>= 1), because sklearn's HDBSCAN has no sample weights.
    ms:     min_samples (None = sklearn default = mcs).
    Noise centroids get the group of their nearest non-noise centroid (same space); if HDBSCAN finds
    no cluster at all, everything is one group. iters > 0: Lloyd refine from the merged group means."""
    C, lab0, n, t_fit = over_cached(Xin, s, K0, sample)
    t = time.time()
    F = C.astype(np.float64)
    if space == "pca16":
        idx = np.random.default_rng(s).choice(len(Xin), min(sample, len(Xin)), replace=False)  # = k-means sample
        F = PCA(16, random_state=s).fit(Xin[idx]).transform(C)
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=sel, copy=True)
    if weight == "rep":
        rep = np.maximum(1, np.round(1000 * n / n.sum())).astype(int)
        hp = hdb.fit_predict(np.repeat(F, rep, 0))
        owner = np.repeat(np.arange(K0), rep)
        h = np.array([np.bincount(hp[owner == c] + 1).argmax() - 1 for c in range(K0)])  # copies share one label
    else:
        h = hdb.fit_predict(F)
    noise = h < 0
    if noise.all():
        lut = np.zeros(K0, np.int64)
    else:
        if noise.any():
            h = h.copy()
            h[noise] = h[~noise][((F[noise, None] - F[None, ~noise]) ** 2).sum(-1).argmin(1)]
        lut = np.unique(h, return_inverse=True)[1].astype(np.int64)
    info = dict(t_fit=t_fit, t_merge=time.time() - t, k_hdb=float(0 if noise.all() else h.max() + 1),
                noise_c=float(noise.mean()), noise_px=float(n[noise].sum() / n.sum()), all_noise=float(noise.all()))
    if not iters:
        return lut[lab0], info
    t = time.time()
    lab = lloyd_from_groups(Xin, s, C, n, lut, iters).argmin(1)
    info["t_refine"] = time.time() - t
    return lab, info


def refine(Xin, s, K0, iters=20, sample=None, beta=0.0):
    """Over-cluster + Ward merge, then Lloyd on the sample starting from the merged group means."""
    t = time.time()
    C = tsh.tee_kmeans(Xin, K0, s, sample=sample)
    lab0 = tsh.sqd(Xin, C).argmin(1)
    n = np.bincount(lab0, minlength=K0)
    lut = tsh.merge_tree(C, n, k, "ward")
    D = lloyd_from_groups(Xin, s, C, n, lut, iters)
    if not beta:
        return D.argmin(1), dict(t_fit=time.time() - t, t_merge=0.0)
    Dg = np.zeros((H, W, k), np.float32); Dg[V] = D
    lab = tsh.icm_potts(Dg, grid(D.argmin(1)), V, beta, tsh.window(px, 10))
    return lab[V], dict(t_fit=time.time() - t, t_merge=0.0)


def kmeans(Xin, s, n_init):
    t = time.time()
    C = tsh.tee_kmeans(Xin, k, s, n_init=n_init)
    return tsh.sqd(Xin, C).argmin(1), dict(t_fit=time.time() - t, t_merge=0.0)


M = {
    "kmeans (TEE)": lambda s: kmeans(X, s, 1),
    "kmeans n_init=10": lambda s: kmeans(X, s, 10),
    "over K0=32 + Ward": lambda s: overmerge(X, s, 32),
    "over K0=64 + Ward": lambda s: overmerge(X, s, 64),
    "over K0=128 + Ward": lambda s: overmerge(X, s, 128),
    "over K0=64 + average link": lambda s: overmerge(X, s, 64, "average"),
    "over K0=64 + spatial Ward g=1": lambda s: overmerge(X, s, 64, gamma=1.0),
    "over K0=64 + spatial Ward g=4": lambda s: overmerge(X, s, 64, gamma=4.0),
    "over K0=64 + Ward, sample 10k": lambda s: overmerge(X, s, 64, sample=10_000),
    "over K0=64 + Ward, n_init=3": lambda s: overmerge(X, s, 64, n_init=3),
    "over K0=32 + Ward + refine": lambda s: refine(X, s, 32),
    "over K0=64 + Ward + refine": lambda s: refine(X, s, 64),
    "over K0=32 + Ward + refine, sample 10k": lambda s: refine(X, s, 32, sample=10_000),
    "over K0=32 + Ward + refine + MRF 0.25": lambda s: refine(X, s, 32, sample=10_000, beta=0.25),
    "over K0=32 + Ward + refine + MRF 0.5": lambda s: refine(X, s, 32, sample=10_000, beta=0.5),
    "Gauss 10m + kmeans n_init=10": lambda s: kmeans(Xg, s, 10),
    "Gauss 10m + over K0=64 + Ward": lambda s: overmerge(Xg, s, 64),
    "Gauss 10m + over K0=32 + Ward + refine": lambda s: refine(Xg, s, 32, sample=10_000),
}
# HDBSCAN as the merge step (6 Oct). 'rep' with mcs 2/3/5 counts copies, so any centroid with >= mcs
# copies has core distance 0 and becomes its own cluster; 'rep ms=100' keeps min_samples above every
# centroid's copy count (largest seen: 94 of ~1000) so the weighting can act on the density.
for K0 in (32, 64, 128):
    for space in ("raw", "pca16"):
        for w, ms, mcss in (("unw", None, (2, 3, 5)), ("rep", None, (2, 3, 5)), ("rep", 100, (20, 50))):
            for mcs in mcss:
                for sel in ("leaf", "eom"):
                    for it in ((0, 20) if w == "unw" else (0,)):
                        name = (f"over K0={K0} + HDBSCAN {space} {w}{f' ms={ms}' if ms else ''} mcs={mcs} {sel}"
                                + (" + refine" if it else ""))
                        M[name] = lambda s, a=(K0, space, w, mcs, sel, ms, it): hdb_merge(
                            X, s, a[0], space=a[1], weight=a[2], mcs=a[3], sel=a[4], ms=a[5], iters=a[6])


def inertia(l):
    """Within-cluster sum of squares on the raw embeddings (all N), per pixel."""
    tot = 0.0
    for c in np.unique(l):
        Z = X[l == c].astype(np.float64)
        tot += ((Z - Z.mean(0)) ** 2).sum()
    return tot / len(l)


only = sys.argv[3] if len(sys.argv) > 3 else None
tag = which if which == "gbg" else f"aut{YEAR}"
res, maps = {}, {}
if only:
    res = json.loads((OUT / f"overcluster_{tag}.json").read_text())
    maps = dict(np.load(OUT / f"maps_overcluster_{tag}.npz"))
    M = {n: f for n, f in M.items() if any(p in n for p in only.split("|"))}
for name, fn in M.items():
    labs, infos = [], []
    for s in seeds:
        l, info = fn(s)
        labs.append(l); infos.append(info)
    grids = [grid(l) for l in labs]
    maps[name] = grids[0].astype(np.int16)
    r = dict(seeds=len(seeds), k_found=float(np.mean([len(np.unique(l)) for l in labs])))
    r["coherence"] = float(np.mean([tsh.coherence(g, V) for g in grids]))
    r["ARI_mean"], r["ARI_min"] = tsh.stability(labs)
    r["max_share"] = float(np.mean([np.bincount(l).max() / len(l) for l in labs]))
    r["min_share"] = float(np.mean([np.bincount(l)[np.bincount(l) > 0].min() / len(l) for l in labs]))
    r["inertia"] = float(np.mean([inertia(l) for l in labs[:5]]))
    ks = [len(np.unique(l)) for l in labs]
    r["k_min"], r["k_max"] = float(min(ks)), float(max(ks))
    r["k_eff"] = float(np.mean([np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()) for p in (np.bincount(l) / len(l) for l in labs)]))
    for key in infos[0]:
        r[key] = float(np.mean([i[key] for i in infos]))
    if name.startswith("Gauss"):
        r["t_pre"] = t_gauss
    if "fid" in ds:
        fm = [tsh.field_metrics(g, ds["fid"], ds["cls"]) for g in grids]
        em = [tsh.edge_adherence(g, ds["fid"]) for g in grids]
        for key in fm[0]:
            r[key] = float(np.mean([f[key] for f in fm]))
        for key in em[0]:
            r[key] = float(np.mean([e[key] for e in em]))
    res[name] = r
    print(f"{name:34s} " + " ".join(f"{a}={b:.3f}" for a, b in r.items() if isinstance(b, float)), flush=True)
(OUT / f"overcluster_{tag}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / f"maps_overcluster_{tag}.npz", **maps)
