"""Final method table (Christopher, 11 Oct 2026): every method over many seeds, with the value of
every metric stored PER SEED, so that the report can give mean +- std and paired differences to
the baseline. Also the parameter sweeps (sigma, beta, K0, n_init, over-clustering sample) and
the configurations that were still open (full-N Lloyd, PCA, superpixels).

    [YEAR=2022] [K=17] [GBG_LABELS=export.geojson] \
        ~/tee-local/venv/bin/python run_final.py [gbg|aut] [n_seeds] [all|west|east] [only]

part  west / east = left / right half of the Austrian tile (hold-out check for the parameters).
only  '|'-separated family names (base, gauss, over, gover, fast, iters, pca, sp, key); the result is
      merged into the existing file. 'key' (the four main methods, for runs with many seeds) only
      runs on request; use it with TAG=_100seeds so that it gets its own file.
Writes results/final_<tag>.json and results/maps_final_<tag>.npz (seed 42).

Differences to run_overcluster.py (30 Sep - 6 Oct), so that Python measures what the JavaScript
port does: the Ward weights are the SAMPLE counts of the over-clustering (no extra pass over all
pixels with K0 centroids), and the refinement stops like TEE's k-means (< 0.5 % changes, at most
20 iterations). 'over32 legacy' is the old variant and shows that this changes nothing.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

import tsh

OUT = Path(__file__).resolve().parent / "results"
which = sys.argv[1] if len(sys.argv) > 1 else "gbg"
n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
PART = sys.argv[3] if len(sys.argv) > 3 else "all"
ONLY = sys.argv[4].split("|") if len(sys.argv) > 4 else None
YEAR = int(os.environ.get("YEAR", 2022))

ds = tsh.gothenburg() if which == "gbg" else tsh.austria(year=YEAR)
if which == "gbg" and os.environ.get("GBG_LABELS"):
    gt = json.loads((Path.home() / "tee_data/vectors/Gteborg/2024/metadata.json").read_text())["geotransform"]
    ds["cls"], ds["fid"], ds["class_names"] = tsh.load_geojson_labels(os.environ["GBG_LABELS"], ds["valid"].shape, gt)
if PART in ("west", "east"):
    half = ds["emb"].shape[1] // 2
    sl = slice(0, half) if PART == "west" else slice(half, None)
    for key in ("emb", "valid", "fid", "cls"):
        if key in ds:
            ds[key] = ds[key][:, sl]
k = int(os.environ.get("K", 5 if which == "gbg" else 17))
E, V, px = ds["emb"], ds["valid"], ds["px_m"]
H, W, DIM = E.shape
X = E[V]
N = len(X)
LAB = "fid" in ds
seeds = [42] + list(range(1, n_seeds))
tag = (which if which == "gbg" else f"aut{YEAR}") + ("" if PART == "all" else f"_{PART}") + (
    "" if "K" not in os.environ else f"_k{k}") + os.environ.get("TAG", "")
print(f"{ds['name']} part={PART} N={N} k={k} seeds={len(seeds)} labelled={int((ds['fid'] > 0).sum()) if LAB else 0}", flush=True)


# ───────────────────────── building blocks ─────────────────────────

def grid(l):
    g = np.zeros((H, W), np.int64); g[V] = l; return g


def gridD(D):
    Dg = np.zeros((H, W, D.shape[1]), np.float32); Dg[V] = D; return Dg


def mrf(D, lab, beta):
    return tsh.icm_potts(gridD(D), grid(lab), V, beta, tsh.window(px, 10))[V]


def majority(lab, r):
    return tsh.post_mode(grid(lab), V, k, tsh.window(px, r))[V]


def over(Xin, s, K0, c_over=300, c_ref=500, S_over=None, legacy=False):
    return tsh.over_ward_refine(Xin, k, s, K0, c_over, c_ref, S_over, legacy)


# ───────────────────────── families: seed -> {name: labels} ─────────────────────────
# A family computes everything that shares one expensive step (one k-means fit, one pre-filter).
# t[name] = seconds of the clustering itself in Python (without metrics and without the
# once-per-viewport pre-filter); only for orientation, the runtimes of the report are JavaScript.

def fam_base(s, out, t):
    for n_init in (1, 2, 3, 5, 10):
        t0 = time.time()
        C = tsh.tee_kmeans(X, k, s, n_init=n_init)
        D = tsh.sqd(X, C); lab = D.argmin(1)
        name = "baseline" if n_init == 1 else f"restarts {n_init}"
        out[name], t[name] = lab, time.time() - t0
        if n_init in (1, 10):
            for beta in ((0.25,) if n_init == 1 else (0.1, 0.25, 0.5, 1.0)):
                t1 = time.time(); out[f"{name} + MRF {beta}"] = mrf(D, lab, beta)
                t[f"{name} + MRF {beta}"] = t[name] + time.time() - t1
            for r in ((20,) if n_init == 1 else (10, 20)):
                t1 = time.time(); out[f"{name} + majority {r} m"] = majority(lab, r)
                t[f"{name} + majority {r} m"] = t[name] + time.time() - t1
            for it in ((1, 2) if n_init == 1 else (1,)):
                t1 = time.time(); out[f"{name} + {it} full Lloyd"] = tsh.sqd(X, tsh.lloyd_full(X, C, it)).argmin(1)
                t[f"{name} + {it} full Lloyd"] = t[name] + time.time() - t1


_pre = {}


def gauss(sig):
    if sig not in _pre:
        _pre.clear()  # keep one filtered copy in memory
        t0 = time.time(); _pre[sig] = (tsh.pre_gauss(E, V, sig, px)[V], time.time() - t0)
    return _pre[sig]


def make_fam_gauss(sig):
    def fam(s, out, t):
        Xg, _ = gauss(sig)
        for n_init in ((1, 10) if sig == 10 else (10,)):
            t0 = time.time()
            C = tsh.tee_kmeans(Xg, k, s, n_init=n_init)
            D = tsh.sqd(Xg, C); lab = D.argmin(1)
            name = f"Gauss {sig:g} m + " + ("baseline" if n_init == 1 else "restarts 10")
            out[name], t[name] = lab, time.time() - t0
            if sig == 10 and n_init == 10:
                t1 = time.time(); out[name + " + MRF 0.25"] = mrf(D, lab, 0.25)
                t[name + " + MRF 0.25"] = t[name] + time.time() - t1
    return fam


def fam_over(s, out, t):
    for K0 in (24, 32, 48, 64, 96, 128):
        if K0 <= k:
            continue
        for c in (150, 300, 500):
            t0 = time.time()
            G = over(X, s, K0, c)
            D = tsh.sqd(X, G); lab = D.argmin(1)
            name = f"over{K0} S={c}K0"
            out[name], t[name] = lab, time.time() - t0
            if K0 == 32 and c in (300, 500):
                t1 = time.time(); out[name + " + MRF 0.25"] = mrf(D, lab, 0.25)
                t[name + " + MRF 0.25"] = t[name] + time.time() - t1
                t1 = time.time(); out[name + " + 1 full Lloyd"] = tsh.sqd(X, tsh.lloyd_full(X, G, 1)).argmin(1)
                t[name + " + 1 full Lloyd"] = t[name] + time.time() - t1
                if max(5000, 300 * k) < max(5000, 500 * k):  # reduced rule also for the refinement
                    t1 = time.time(); out[name + ", refine 300k"] = tsh.sqd(X, over(X, s, K0, c, 300)).argmin(1)
                    t[name + ", refine 300k"] = time.time() - t1
    t0 = time.time(); out["over32 legacy (10k, all-N counts, 20 it)"] = tsh.sqd(X, over(X, s, 32, S_over=10_000, legacy=True)).argmin(1)
    t["over32 legacy (10k, all-N counts, 20 it)"] = time.time() - t0


def fam_gover(s, out, t):
    Xg, _ = gauss(10)
    for K0, c in ((32, 300), (32, 500), (64, 300)):
        if K0 <= k:
            continue
        t0 = time.time()
        D = tsh.sqd(Xg, over(Xg, s, K0, c)); lab = D.argmin(1)
        name = f"Gauss 10 m + over{K0} S={c}K0"
        out[name], t[name] = lab, time.time() - t0
        if (K0, c) == (32, 300):
            t1 = time.time(); out[name + " + MRF 0.25"] = mrf(D, lab, 0.25)
            t[name + " + MRF 0.25"] = t[name] + time.time() - t1


_pca = {}


def fam_pca(s, out, t):
    if "b" not in _pca:  # one basis per viewport, fitted on a fixed 20k sample
        Xs = X[np.random.default_rng(12345).choice(N, min(20_000, N), replace=False)]
        _pca["b"] = {d: tsh.pca_basis(Xs, d) for d in (8, 16, 32, 64)}
        _pca["std"] = Xs.std(0) + 1e-6
        print("PCA variance kept:", {d: round(b[2], 3) for d, b in _pca["b"].items()}, flush=True)
    for d, (mu, P, _) in _pca["b"].items():
        Z = (X - mu) @ P
        t0 = time.time(); out[f"PCA-{d} baseline"] = tsh.sqd(Z, tsh.tee_kmeans(Z, k, s)).argmin(1)
        t[f"PCA-{d} baseline"] = time.time() - t0
        if d == 16 and k < 32:
            t0 = time.time(); out["PCA-16 over32 S=300K0"] = tsh.sqd(Z, over(Z, s, 32, 300)).argmin(1)
            t["PCA-16 over32 S=300K0"] = time.time() - t0
    Zs = X / _pca["std"]
    t0 = time.time(); out["z-score baseline"] = tsh.sqd(Zs, tsh.tee_kmeans(Zs, k, s)).argmin(1)
    t["z-score baseline"] = time.time() - t0


_sp = {}
SP_GRID = [(30, 0.5), (30, 1.0), (50, 1.0), (30, 1e6)]  # (edge length in m, compactness); 1e6 = regular blocks


def fam_sp(s, out, t):
    for size_m, c_rel in SP_GRID:
        key = (size_m, c_rel)
        if key not in _sp:
            t0 = time.time()
            sp = tsh.superpixels(E, V, px, size_m, c_rel)[V]
            M = tsh.group_means(X, sp, sp.max() + 1)
            _sp[key] = (sp, M)
            print(f"superpixels {size_m} m c={c_rel}: {len(M)} segments, {time.time() - t0:.1f} s", flush=True)
        sp, M = _sp[key]
        base = f"superpixels {size_m} m c={c_rel:g}" if c_rel < 1e5 else f"blocks {size_m} m"
        t0 = time.time(); out[base + " + baseline"] = tsh.sqd(M, tsh.tee_kmeans(M, k, s)).argmin(1)[sp]
        t[base + " + baseline"] = time.time() - t0
        if k < 32:
            t0 = time.time(); out[base + " + over32 S=300K0"] = tsh.sqd(M, over(M, s, 32, 300)).argmin(1)[sp]
            t[base + " + over32 S=300K0"] = time.time() - t0


def fam_fast(s, out, t):
    """Over-clustering fitted in a low-dimensional projection of the sample (cheap in the browser)."""
    for K0, d, proj in ((32, 8, "pca"), (32, 16, "pca"), (32, 32, "pca"), (32, 16, "rp"), (32, 32, "rp"), (64, 16, "pca")):
        if K0 <= k:
            continue
        t0 = time.time()
        G = tsh.over_ward_refine_proj(X, k, s, K0, d, proj)
        name = f"over{K0} S=300K0, fit in {'PCA' if proj == 'pca' else 'RP'}-{d}"
        out[name], t[name] = tsh.sqd(X, G).argmin(1), time.time() - t0
        if (K0, d, proj) == (32, 16, "pca"):
            t1 = time.time(); out[name + " + 1 full Lloyd"] = tsh.sqd(X, tsh.lloyd_full(X, G, 1)).argmin(1)
            t[name + " + 1 full Lloyd"] = t[name] + time.time() - t1


def fam_key(s, out, t):
    """The four main methods only, for runs with many seeds (stability estimate with a tight interval)."""
    for name, fit in (("baseline", lambda: tsh.tee_kmeans(X, k, s)), ("restarts 10", lambda: tsh.tee_kmeans(X, k, s, n_init=10)),
                      ("over32 S=300K0", lambda: over(X, s, 32, 300)),
                      ("over32 S=300K0, fit in PCA-16", lambda: tsh.over_ward_refine_proj(X, k, s, 32, 16, "pca"))):
        t0 = time.time(); out[name] = tsh.sqd(X, fit()).argmin(1); t[name] = time.time() - t0


def fam_iters(s, out, t):
    """How many Lloyd iterations does the over-clustering need? (It is merged and refined anyway.)"""
    for it in (1, 2, 3, 5, 10):
        t0 = time.time()
        G = tsh.over_ward_refine(X, k, s, 32, 300, over_iter=it)
        out[f"over32 S=300K0, {it} it"], t[f"over32 S=300K0, {it} it"] = tsh.sqd(X, G).argmin(1), time.time() - t0
    for c in (150, 500):
        t0 = time.time()
        G = tsh.over_ward_refine(X, k, s, 32, c, over_iter=5)
        out[f"over32 S={c}K0, 5 it"], t[f"over32 S={c}K0, 5 it"] = tsh.sqd(X, G).argmin(1), time.time() - t0


FAMILIES = {"key": fam_key, "fast": fam_fast, "iters": fam_iters, "base": fam_base, "over": fam_over, "pca": fam_pca, "sp": fam_sp,
            **{f"gauss{sig:g}": make_fam_gauss(sig) for sig in (5, 7.5, 10, 15, 20)}, "gover": fam_gover}


# ───────────────────────── metrics ─────────────────────────

def score(lab):
    g = grid(lab)
    cnt = np.bincount(lab)
    r = dict(coherence=float(tsh.coherence(g, V)), k_found=int((cnt > 0).sum()), max_share=float(cnt.max() / len(lab)))
    if LAB:
        m = ds["fid"] > 0
        r["AMI"] = float(adjusted_mutual_info_score(ds["cls"][m], g[m]))
        r["ARI_gt"] = float(tsh.ari(ds["cls"][m], g[m]))
        fm = tsh.field_metrics(g, ds["fid"], ds["cls"])
        ea = tsh.edge_adherence(g, ds["fid"])
        r.update(homog=float(fm["field_homog"]), speckle=ea["d_in"], edge=ea["d_out"])
    return r


res_file, map_file = OUT / f"final_{tag}.json", OUT / f"maps_final_{tag}.npz"
res = json.loads(res_file.read_text()) if ONLY and res_file.exists() else {"methods": {}}
maps = dict(np.load(map_file)) if ONLY and map_file.exists() else {}
res["meta"] = dict(dataset=ds["name"], part=PART, N=int(N), k=k, seeds=seeds, shape=[H, W], px_m=list(px),
                   labelled=int((ds["fid"] > 0).sum()) if LAB else 0, date=time.strftime("%Y-%m-%d"))
for fname, fam in FAMILIES.items():
    if ONLY and not any(o == fname or (o == "gauss" and fname.startswith("gauss")) for o in ONLY):
        continue
    if fname == "key" and not ONLY:  # only on request
        continue
    t_fam = time.time()
    labs, per_seed, times = {}, {}, {}
    for s in seeds:
        out, t = {}, {}
        fam(s, out, t)
        for name, lab in out.items():
            lab = lab.astype(np.int16)
            labs.setdefault(name, []).append(lab)
            for key, v in score(lab).items():
                per_seed.setdefault(name, {}).setdefault(key, []).append(v)
            times.setdefault(name, []).append(t[name])
            if s == 42:
                maps[name] = grid(lab).astype(np.int8)
    for name in labs:
        P = tsh.pair_matrix(labs[name])
        r = dict(per_seed=per_seed[name], pairs=np.round(P, 5).tolist(), t_py=times[name])
        if fname.startswith("gauss") or fname == "gover":
            r["t_pre_py"] = _pre[next(iter(_pre))][1]
        res["methods"][name] = r
        ps = per_seed[name]
        print(f"{name:44s} stab {np.nanmean(P):.3f} (min {np.nanmin(P):.3f})"
              + (f"  AMI {np.mean(ps['AMI']):.3f}+-{np.std(ps['AMI'], ddof=1):.3f}  ARIgt {np.mean(ps['ARI_gt']):.3f}"
                 f"  speckle {np.mean(ps['speckle']):.3f}" if LAB else f"  coh {np.mean(ps['coherence']):.3f}")
              + f"  t {np.median(times[name]):.2f}s", flush=True)
    res_file.write_text(json.dumps(res))
    np.savez_compressed(map_file, **maps)
    print(f"-- family {fname} done in {time.time() - t_fam:.0f} s", flush=True)
