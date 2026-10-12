"""Spectral clustering of a pixel sample: robustness checks (Christopher, 12 Oct 2026).
Follows up on the 'spectral' family of run_final.py, where spectral clustering of 5,000 sample
pixels with nearest-mean assignment raised AMI and ARI on the Austrian tile but collapsed on
Gothenburg with k = 5.

    [YEAR=2022] [GBG_LABELS=labels.geojson TAG=_labels] \
        ~/tee-local/venv/bin/python -W ignore run_spectral.py [gbg|aut] [n_seeds=10] [all|west|east]

Varies the number of graph neighbours, the sample size and k, adds MRF smoothing, and reports for
every setting AMI / ARI (if labels), stability, the largest cluster and the number of connected
components of the k-NN graph. Writes results/spectral_<tag>.json.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph

import tsh

OUT = Path(__file__).resolve().parent / "results"
which = sys.argv[1] if len(sys.argv) > 1 else "gbg"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
PART = sys.argv[3] if len(sys.argv) > 3 else "all"
ds = tsh.gothenburg() if which == "gbg" else tsh.austria(year=int(os.environ.get("YEAR", 2022)))
if which == "gbg" and os.environ.get("GBG_LABELS"):  # hand-made labels exported from TEE (GeoJSON)
    gt = json.loads((Path.home() / "tee_data/vectors/Gteborg/2024/metadata.json").read_text())["geotransform"]
    ds["cls"], ds["fid"], _ = tsh.load_geojson_labels(os.environ["GBG_LABELS"], ds["valid"].shape, gt)
if PART in ("west", "east"):
    half = ds["emb"].shape[1] // 2
    sl = slice(0, half) if PART == "west" else slice(half, None)
    for key in ("emb", "valid", "fid", "cls"):
        if key in ds:
            ds[key] = ds[key][:, sl]
E, V, px = ds["emb"], ds["valid"], ds["px_m"]
H, W, _ = E.shape
X = E[V]
LAB = "fid" in ds
seeds = [42] + list(range(1, n_seeds))
tag = which + ("" if PART == "all" else f"_{PART}") + os.environ.get("TAG", "")


def grid(l):
    g = np.zeros((H, W), np.int64); g[V] = l; return g


def spectral(k, s, S=5000, nn=15, beta=0.0):
    idx, lab_s = tsh.spectral_sample(X, k, s, S, nn)
    M = tsh.group_means(X[idx], lab_s, int(lab_s.max()) + 1)
    D = tsh.sqd(X, M)
    lab = D.argmin(1)
    if beta:
        Dg = np.zeros((H, W, D.shape[1]), np.float32); Dg[V] = D
        lab = tsh.icm_potts(Dg, grid(lab), V, beta, tsh.window(px, 10))[V]
    comp = connected_components(kneighbors_graph(X[idx], nn, include_self=False), directed=False)[0]
    return lab, comp


def kmeans(k, s, beta=0.0, over=False):
    C = tsh.over_ward_refine(X, k, s) if over else tsh.tee_kmeans(X, k, s)
    D = tsh.sqd(X, C)
    lab = D.argmin(1)
    if beta:
        Dg = np.zeros((H, W, k), np.float32); Dg[V] = D
        lab = tsh.icm_potts(Dg, grid(lab), V, beta, tsh.window(px, 10))[V]
    return lab, 0


ks = (5, 8, 12, 17) if which == "gbg" else (17, 8, 5)
k0 = ks[0]
runs = {}
for k in ks:
    runs[f"k={k} baseline"] = lambda s, k=k: kmeans(k, s)
    runs[f"k={k} over32 + MRF 0.25 (proposed)"] = lambda s, k=k: kmeans(k, s, 0.25, True)
    runs[f"k={k} spectral S=5000 nn=15"] = lambda s, k=k: spectral(k, s)
    runs[f"k={k} spectral S=5000 nn=15 + MRF 0.25"] = lambda s, k=k: spectral(k, s, beta=0.25)
for nn in (8, 30, 60, 120):
    runs[f"k={k0} spectral S=5000 nn={nn}"] = lambda s, nn=nn: spectral(k0, s, nn=nn)
for S in (2000, 10000):
    runs[f"k={k0} spectral S={S} nn=15"] = lambda s, S=S: spectral(k0, s, S=S)

res = {}
for name, fn in runs.items():
    t0 = time.time()
    labs, comps = zip(*(fn(s) for s in seeds))
    dt = (time.time() - t0) / len(seeds)
    P = tsh.pair_matrix(labs)
    r = dict(stab=float(np.nanmean(P)), stab_min=float(np.nanmin(P)), comps=float(np.mean(comps)), t_py=dt,
             max_share=[float(np.bincount(l).max() / len(l)) for l in labs])
    if LAB:
        m = ds["fid"] > 0
        gs = [grid(l) for l in labs]
        r["AMI"] = [float(adjusted_mutual_info_score(ds["cls"][m], g[m])) for g in gs]
        r["ARI_gt"] = [float(tsh.ari(ds["cls"][m], g[m])) for g in gs]
        r["speckle"] = [tsh.edge_adherence(g, ds["fid"])["d_in"] for g in gs]
    res[name] = r
    print(f"{name:42s} stab {r['stab']:.2f} (min {r['stab_min']:.2f})  largest {100 * np.mean(r['max_share']):3.0f} % (max {100 * np.max(r['max_share']):3.0f} %)"
          + (f"  AMI {np.mean(r['AMI']):.3f}+-{np.std(r['AMI'], ddof=1):.3f}  ARI {np.mean(r['ARI_gt']):.3f}  speckle {100 * np.mean(r['speckle']):.1f} %" if LAB else "")
          + f"  graph components {r['comps']:.1f}  {dt:.1f} s", flush=True)
    (OUT / f"spectral_{tag}.json").write_text(json.dumps(dict(dataset=ds["name"], part=PART, seeds=seeds, methods=res)))
