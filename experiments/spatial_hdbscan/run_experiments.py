"""Run all smoothing / HDBSCAN variants over several seeds and write results/<dataset>.json
plus seed-42 label maps (results/maps_<dataset>.npz).

    ~/tee-local/venv/bin/python run_experiments.py [gbg|aut] [n_seeds] [year]
"""
import json, sys, time
from pathlib import Path

import numpy as np
import tsh

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def methods(ds, k):
    """name -> function(seed) returning (label grid, info). Smoothing radii are in metres."""
    E, V, px = ds["emb"], ds["valid"], ds["px_m"]
    X = E[V]
    H, W, _ = E.shape
    cache = {}

    def grid(lab_flat):
        g = np.zeros((H, W), np.int64)
        g[V] = lab_flat
        return g

    def base(seed, Xin=X, n_init=1):
        key = (id(Xin), seed, n_init)
        if key not in cache:
            C = tsh.tee_kmeans(Xin, k, seed, n_init=n_init)
            D = tsh.sqd(Xin, C)
            Dg = np.zeros((H, W, k), np.float32)
            Dg[V] = D
            cache[key] = (grid(D.argmin(1)), Dg)
        return cache[key]

    smoothed = {}

    def pre(name, fn):
        if name not in smoothed:
            t = time.time()
            smoothed[name] = (fn()[V], time.time() - t)
        return smoothed[name]

    def timed(f):
        def run(seed):
            t = time.time()
            lab, extra = f(seed)
            return lab, dict(t=time.time() - t, **extra)
        return run

    B = 10  # smoothing is applied on top of k-means with 10 restarts (stable base)
    M = {}
    M["kmeans (TEE)"] = timed(lambda s: (base(s)[0], {}))
    M["kmeans n_init=10"] = timed(lambda s: (base(s, n_init=B)[0], {}))
    M["TEE kmeans + post mode r=20m"] = timed(lambda s: (tsh.post_mode(base(s)[0], V, k, tsh.window(px, 20)), {}))
    M["TEE kmeans + ICM beta=0.5"] = timed(lambda s: (tsh.icm_potts(base(s)[1], base(s)[0], V, 0.5, tsh.window(px, 10)), {}))
    for r, it in ((10, 1), (20, 1), (20, 3), (40, 1)):
        hw = tsh.window(px, r)
        M[f"post mode r={r}m x{it}"] = timed(lambda s, hw=hw, it=it: (tsh.post_mode(base(s, n_init=B)[0], V, k, hw, it), {}))
    for sig in (5, 10, 20):
        def f(s, sig=sig):
            Xs, tp = pre(f"g{sig}", lambda: tsh.pre_gauss(E, V, sig, px))
            return base(s, Xs, n_init=B)[0], dict(t_pre=tp)
        M[f"pre gauss sigma={sig}m"] = timed(f)
    for r, it in ((10, 2), (20, 1)):
        def f(s, r=r, it=it):
            Xs, tp = pre(f"b{r}{it}", lambda: tsh.pre_bilateral(E, V, tsh.window(px, r), 1.0, it))
            return base(s, Xs, n_init=B)[0], dict(t_pre=tp)
        M[f"pre bilateral r={r}m x{it}"] = timed(f)
    for beta in (0.25, 0.5, 1.0):
        M[f"MRF/ICM beta={beta}"] = timed(lambda s, b=beta: (tsh.icm_potts(base(s, n_init=B)[1], base(s, n_init=B)[0], V, b, tsh.window(px, 10)), {}))

    def bil_icm(s):
        Xs, tp = pre("b102", lambda: tsh.pre_bilateral(E, V, tsh.window(px, 10), 1.0, 2))
        lab, D = base(s, Xs, n_init=B)
        return tsh.icm_potts(D, lab, V, 0.5, tsh.window(px, 10)), dict(t_pre=tp)
    M["bilateral r=10m x2 + ICM 0.5"] = timed(bil_icm)

    n = len(X)
    for d, mcs_div, red in ((8, 200, "pca"), (16, 200, "pca"), (32, 200, "pca"), (16, 50, "pca"),
                            (16, 1000, "pca"), (8, 200, "umap")):
        def f(s, d=d, mcs_div=mcs_div, red=red):
            lab, info = tsh.hdbscan_pipeline(X, s, d=d, sample=20_000, min_cluster_size=20_000 // mcs_div,
                                             reducer=red)
            return grid(lab), info
        M[f"HDBSCAN {red}{d} mcs={20_000 // mcs_div}"] = timed(f)
    return M


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "gbg"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    year = int(sys.argv[3]) if len(sys.argv) > 3 else 2024
    ds = tsh.gothenburg() if which == "gbg" else tsh.austria(year=year)
    k = 5 if which == "gbg" else 17
    V = ds["valid"]
    seeds = [42] + list(range(1, n_seeds))
    res, maps = {}, {}
    for name, fn in methods(ds, k).items():
        if "umap" in name and n_seeds > 5:
            use = seeds[:5]
        else:
            use = seeds
        labs, infos = [], []
        for s in use:
            lab, info = fn(s)
            labs.append(lab[V])
            infos.append(info)
            if s == 42:
                maps[name] = lab.astype(np.int16)
        r = dict(seeds=len(use), k_found=float(np.mean([len(np.unique(l)) for l in labs])))
        grids = []
        for l in labs:
            g = np.zeros(V.shape, np.int64)
            g[V] = l
            grids.append(g)
        r["coherence"] = float(np.mean([tsh.coherence(g, V) for g in grids]))
        r["ARI_mean"], r["ARI_min"] = tsh.stability(labs)
        r["min_share"] = float(np.mean([np.bincount(l)[np.bincount(l) > 0].min() / len(l) for l in labs]))
        r["max_share"] = float(np.mean([np.bincount(l).max() / len(l) for l in labs]))
        # effective number of clusters = exp(entropy of cluster shares)
        r["k_eff"] = float(np.mean([np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()) for p in
                                    (np.bincount(l) / len(l) for l in labs)]))
        for key in infos[0]:
            r[key] = float(np.mean([i[key] for i in infos]))
        if "fid" in ds:
            fm = [tsh.field_metrics(g, ds["fid"], ds["cls"]) for g in grids]
            for key in fm[0]:
                r[key] = float(np.mean([f[key] for f in fm]))
        res[name] = r
        print(f"{name:34s} " + " ".join(f"{a}={b:.3f}" for a, b in r.items() if isinstance(b, float)), flush=True)
    tag = which if which == "gbg" else f"aut{year}"
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / f"maps_{tag}.npz", **{n.replace("/", "_"): m for n, m in maps.items()})


if __name__ == "__main__":
    main()
