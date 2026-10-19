"""HDBSCAN follow-up: 'leaf' vs 'eom' cluster selection and kNN vs centroid assignment,
compared with k-means (10 restarts) at the SAME number of clusters.
    ~/tee-local/venv/bin/python run_hdbscan_extra.py [gbg|aut] [n_seeds]"""
import json, sys
import numpy as np
import tsh
from run_experiments import OUT

which = sys.argv[1] if len(sys.argv) > 1 else "gbg"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ds = tsh.gothenburg() if which == "gbg" else tsh.austria()
V = ds["valid"]; X = ds["emb"][V]
seeds = [42] + list(range(1, n_seeds))
res, maps = {}, {}
cfgs = [("eom", 100, "knn"), ("leaf", 100, "knn"), ("leaf", 100, "centroid"), ("leaf", 400, "knn")]
for sel, mcs, asg in cfgs:
    name = f"HDBSCAN pca16 {sel} mcs={mcs} assign={asg}"
    labs, infos, kms = [], [], []
    for s in seeds:
        lab, info = tsh.hdbscan_pipeline(X, s, d=16, sample=20_000, min_cluster_size=mcs, selection=sel, assign=asg)
        labs.append(lab); infos.append(info)
        # k-means with the same number of clusters, for a fair comparison
        k = max(2, len(np.unique(lab)))
        C = tsh.tee_kmeans(X, k, s, n_init=10)
        kms.append(tsh.sqd(X, C).argmin(1))
    for tag, L in ((name, labs), (f"  k-means n_init=10, same k", kms)):
        r = dict(k_found=float(np.mean([len(np.unique(l)) for l in L])))
        grids = []
        for l in L:
            g = np.zeros(V.shape, np.int64); g[V] = l; grids.append(g)
        r["coherence"] = float(np.mean([tsh.coherence(g, V) for g in grids]))
        r["ARI_mean"], r["ARI_min"] = tsh.stability(L)
        r["max_share"] = float(np.mean([np.bincount(l).max() / len(l) for l in L]))
        r["k_eff"] = float(np.mean([np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()) for p in (np.bincount(l) / len(l) for l in L)]))
        if tag == name:
            for key in ("noise", "t_fit", "t_assign"):
                r[key] = float(np.mean([i[key] for i in infos]))
        if "fid" in ds:
            fm = [tsh.field_metrics(g, ds["fid"], ds["cls"]) for g in grids]
            for key in fm[0]:
                r[key] = float(np.mean([f[key] for f in fm]))
        key = tag if tag == name else f"{name} | k-means same k"
        res[key] = r
        maps[key] = grids[0].astype(np.int16)
        print(f"{key:60s} " + " ".join(f"{a}={b:.3f}" for a, b in r.items()), flush=True)
tag = which if which == "gbg" else "aut2024"
(OUT / f"hdbscan_extra_{tag}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / f"maps_hdbscan_extra_{tag}.npz", **{k.replace("/", "_").replace("|", "vs"): m for k, m in maps.items()})
