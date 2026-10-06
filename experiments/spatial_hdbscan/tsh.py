"""Spatial smoothing + HDBSCAN experiments on TESSERA embeddings (Christopher, Sep/Oct 2026).

Everything works on a 2-D grid: emb is (H, W, D) float32, valid is (H, W) bool.
Run with TEE's venv:  ~/tee-local/venv/bin/python run_experiments.py

Data
  gothenburg()      TEE viewport "Gteborg" 2024 (EPSG:4326 grid, ~7 m x 13 m pixels, no labels yet)
  austria(tile)     GeoTessera tile (EPSG:32633, 10 m) from TEE's compute cache + rasterised
                    INVEKOS 2022 fields (field id, 17 classes).
"""
import glob, json, sys, time
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier

HOME = Path.home()
TILES = HOME / "tee_data/compute-cache/tessera-eval/tiles"
CACHE = HOME / "tee_data/harness-cache"
CACHE.mkdir(exist_ok=True)
ROOT = Path(__file__).resolve().parents[2]  # repo root
AUSTRIA_ZIP = ROOT / "austria.zip"


# ───────────────────────── data ─────────────────────────

def gothenburg():
    f = CACHE / "gbg2024_emb.npy"
    if not f.exists():
        sys.path.insert(0, str(ROOT / "experiments/seed_instability"))
        from make_seed_maps import load_embeddings  # faithful RVQ decode as in TEE
        emb, H, W = load_embeddings()
        np.save(f, emb.reshape(H, W, -1))
    G = np.load(f)
    return dict(name="Gothenburg 2024", emb=G, valid=np.ones(G.shape[:2], bool),
                px_m=(13.0, 6.95))  # (row, col) pixel size in metres


def _austria_labels(transform, shape):
    import geopandas as gpd, zipfile, csv, io
    from rasterio.features import rasterize
    g = gpd.read_file(f"zip://{AUSTRIA_ZIP}!austrian_crop_17classes/austrian_crop_17classes.shp")
    with zipfile.ZipFile(AUSTRIA_ZIP) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(
            "austrian_crop_17classes/austrian_crop_17classes_label_mapping.csv"), "utf-8")))
    code2cls = {int(c): int(r["label_id"]) for r in rows for c in r["source_snar_codes"].split("|")}
    names = {int(r["label_id"]): r["label_name_en"] for r in rows}
    g["cls"] = g["NVC"].astype(int).map(code2cls)
    g = g[g.cls.notna()].reset_index(drop=True)
    fid = rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry)), out_shape=shape,
                    transform=transform, fill=0, dtype="int32")
    lut = np.concatenate([[0], g.cls.values.astype(np.int16)])
    return fid, lut[fid], names


def austria(tile="grid_16.75_48.35", year=2024):
    f = CACHE / f"austria_{tile}_{year}.npz"
    if not f.exists():
        import rasterio
        d = TILES / f"global_0.1_degree_representation/{year}/{tile}"
        q = np.load(d / f"{tile}.npy")
        s = np.load(d / f"{tile}_scales.npy")
        emb = (q.astype(np.float32) * s[..., None]).astype(np.float32)
        valid = np.isfinite(s) & (s > 0)
        emb[~valid] = 0
        with rasterio.open(TILES / f"global_0.1_degree_tiff_all/{tile}.tiff") as r:
            fid, cls, _ = _austria_labels(r.transform, r.shape)
        np.savez(f, emb=emb, valid=valid, fid=fid, cls=cls)
    z = np.load(f)
    return dict(name=f"Austria {tile} {year}", emb=z["emb"], valid=z["valid"],
                fid=z["fid"], cls=z["cls"], px_m=(10.0, 10.0))


# ───────────────────────── baseline: TEE k-means ─────────────────────────

def sqd(X, C):
    """Squared Euclidean distances (n, k), float32, chunked."""
    out = np.empty((len(X), len(C)), np.float32)
    cc = (C.astype(np.float64) ** 2).sum(1)
    for s in range(0, len(X), 200_000):
        x = X[s:s + 200_000]
        out[s:s + 200_000] = np.maximum((x.astype(np.float64) ** 2).sum(1)[:, None] - 2 * x @ C.T + cc, 0)
    return out


def tee_kmeans(X, k, seed, sample=None, max_iter=20, n_init=1):
    """TEE's algorithm (segmentation.js): sample max(5000, 500k), k-means++ once, Lloyd on the
    sample (<=20 it, stop <0.5 % changes), one assignment pass over all N. numpy RNG instead of
    mulberry32. Returns centroids (k, D)."""
    rng = np.random.default_rng(seed)
    N = len(X)
    S = min(sample or max(5000, 500 * k), N)
    Xs = X[rng.choice(N, S, replace=False)].astype(np.float32)
    best = None
    for _ in range(n_init):
        C = np.empty((k, X.shape[1]), np.float32)
        C[0] = Xs[rng.integers(S)]
        mind = np.full(S, np.inf)
        for c in range(1, k):
            mind = np.minimum(mind, sqd(Xs, C[c - 1:c])[:, 0])
            C[c] = Xs[min(np.searchsorted(np.cumsum(mind), rng.random() * mind.sum()), S - 1)]
        a = np.zeros(S, int)
        for _ in range(max_iter):
            D = sqd(Xs, C)
            new = D.argmin(1)
            changed, a = (new != a).sum(), new
            for c in range(k):
                m = a == c
                C[c] = Xs[rng.integers(S)] if not m.any() else Xs[m].mean(0)
            if changed < 0.005 * S:
                break
        inertia = sqd(Xs, C).min(1).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C.copy())
    return best[1]


# ───────────────────────── spatial smoothing ─────────────────────────

def window(px_m, radius_m):
    """Half-window sizes (rows, cols) so that the window covers ~radius_m in both directions."""
    return max(1, round(radius_m / px_m[0])), max(1, round(radius_m / px_m[1]))


def box_sum(a, hw):
    return ndimage.uniform_filter(a, size=(2 * hw[0] + 1, 2 * hw[1] + 1), mode="nearest") * (
        (2 * hw[0] + 1) * (2 * hw[1] + 1))


def post_mode(lab, valid, k, hw, iters=1):
    """Majority (mode) filter on the label map. Ties keep the current label."""
    lab = lab.copy()
    for _ in range(iters):
        cnt = np.stack([box_sum(((lab == c) & valid).astype(np.float32), hw) for c in range(k)], -1)
        own = np.take_along_axis(cnt, lab[..., None], -1)[..., 0]
        best = cnt.argmax(-1)
        new = np.where(cnt.max(-1) > own + 1e-6, best, lab)
        lab = np.where(valid, new, lab)
    return lab


def pre_gauss(emb, valid, sigma_m, px_m):
    """Gaussian filter on the embeddings (per channel), sigma given in metres."""
    s = (sigma_m / px_m[0], sigma_m / px_m[1], 0)
    w = ndimage.gaussian_filter(valid.astype(np.float32), s[:2])
    out = ndimage.gaussian_filter(emb * valid[..., None], s) / np.maximum(w, 1e-6)[..., None]
    return out.astype(np.float32)


def pre_bilateral(emb, valid, hw, h_rel=1.0, iters=1):
    """Edge-aware smoothing: average of neighbours in a window, weighted by
    exp(-||x_i - x_j||^2 / h^2), h = h_rel * median neighbour distance. Keeps field edges."""
    E = emb.copy()
    H, W, _ = E.shape
    for _ in range(iters):
        d_ref = np.median(((E[:, 1:] - E[:, :-1]) ** 2).sum(-1)[valid[:, 1:] & valid[:, :-1]])
        h2 = (h_rel ** 2) * d_ref
        acc = np.zeros_like(E, dtype=np.float32)
        wsum = np.zeros((H, W), np.float32)
        P = np.pad(E, ((hw[0], hw[0]), (hw[1], hw[1]), (0, 0)), mode="edge")
        V = np.pad(valid, ((hw[0], hw[0]), (hw[1], hw[1])), mode="constant")
        for dy in range(-hw[0], hw[0] + 1):
            for dx in range(-hw[1], hw[1] + 1):
                Q = P[hw[0] + dy:hw[0] + dy + H, hw[1] + dx:hw[1] + dx + W]
                w = np.exp(-((Q - E) ** 2).sum(-1) / h2) * V[hw[0] + dy:hw[0] + dy + H, hw[1] + dx:hw[1] + dx + W]
                acc += w[..., None] * Q
                wsum += w
        E = acc / np.maximum(wsum, 1e-6)[..., None]
    return E.astype(np.float32)


def icm_potts(D, lab, valid, beta, hw=(1, 1), iters=5):
    """MRF / Potts smoothing with ICM: each pixel takes argmin_c  D[c]/s + beta * (#neighbours != c).
    D = squared distances to the centroids (H, W, k); s = mean nearest distance (makes beta
    scale-free). hw = neighbourhood half-window (rows, cols)."""
    k = D.shape[-1]
    s = D.min(-1)[valid].mean()
    n_nb = box_sum(valid.astype(np.float32), hw) - 1
    lab = lab.copy()
    for _ in range(iters):
        same = np.stack([box_sum(((lab == c) & valid).astype(np.float32), hw) - (lab == c) for c in range(k)], -1)
        E = D / s + beta * (n_nb[..., None] - same)
        new = E.argmin(-1)
        changed = ((new != lab) & valid).mean()
        lab = np.where(valid, new, lab)
        if changed < 1e-4:
            break
    return lab


# ───────────────────────── HDBSCAN ─────────────────────────

def hdbscan_pipeline(X, seed, d=16, sample=20_000, min_cluster_size=None, min_samples=10,
                     reducer="pca", knn=5, assign="knn", selection="eom"):
    """PCA (or UMAP) on a sample -> HDBSCAN on the sample -> assign all N pixels by
    k-NN vote over the non-noise sample points (in PCA space). Returns labels (N,) and info."""
    rng = np.random.default_rng(seed)
    N = len(X)
    idx = rng.choice(N, min(sample, N), replace=False)
    t0 = time.time()
    pca = PCA(n_components=d if reducer == "pca" else 32, random_state=seed).fit(X[idx])
    Zs = pca.transform(X[idx])
    if reducer == "umap":
        import umap
        Zc = umap.UMAP(n_components=d, n_neighbors=30, min_dist=0.0, random_state=seed).fit_transform(Zs)
    else:
        Zc = Zs
    mcs = min_cluster_size or max(20, len(idx) // 200)
    hl = HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, copy=True, cluster_selection_method=selection).fit_predict(Zc)
    t_fit = time.time() - t0
    keep = hl >= 0
    t0 = time.time()
    Z = pca.transform(X)
    if assign == "knn":
        clf = KNeighborsClassifier(n_neighbors=knn).fit(Zs[keep], hl[keep])
        lab = np.concatenate([clf.predict(Z[s:s + 100_000]) for s in range(0, N, 100_000)])
    else:  # nearest cluster mean in PCA space: cheap enough for the browser, but Voronoi-shaped
        C = np.stack([Zs[hl == c].mean(0) for c in range(hl.max() + 1)])
        lab = sqd(Z.astype(np.float32), C.astype(np.float32)).argmin(1)
    t_assign = time.time() - t0
    return lab, dict(n_clusters=int(hl.max() + 1), noise=float((~keep).mean()), t_fit=t_fit,
                     t_assign=t_assign, min_cluster_size=mcs)


# ───────────────────────── metrics ─────────────────────────

def coherence(lab, valid):
    """Share of 4-neighbour pairs (both valid) with the same label."""
    h = valid[:, 1:] & valid[:, :-1]
    v = valid[1:] & valid[:-1]
    same = (lab[:, 1:] == lab[:, :-1])[h].sum() + (lab[1:] == lab[:-1])[v].sum()
    return same / (h.sum() + v.sum())


def field_metrics(lab, fid, cls):
    """Supervised metrics on labelled pixels: AMI/NMI vs crop class, field homogeneity
    (pixel-weighted share of the majority cluster inside each field), and boundary contrast
    (share of 4-neighbour pairs across two different labelled fields that get different clusters)."""
    m = fid > 0
    out = dict(AMI=adjusted_mutual_info_score(cls[m], lab[m]),
               NMI=normalized_mutual_info_score(cls[m], lab[m]))
    f, l = fid[m], lab[m]
    k = l.max() + 1
    pair = f.astype(np.int64) * k + l
    u, c = np.unique(pair, return_counts=True)
    fu = u // k
    maj = np.zeros(fid.max() + 1)
    np.maximum.at(maj, fu, c)
    size = np.bincount(f, minlength=fid.max() + 1)
    out["field_homog"] = maj.sum() / size.sum()
    small = (size > 0) & (size <= 100)          # fields <= 1 ha
    out["field_homog_small"] = maj[small].sum() / size[small].sum()
    diff = []
    for a, b, la, lb in ((fid[:, 1:], fid[:, :-1], lab[:, 1:], lab[:, :-1]), (fid[1:], fid[:-1], lab[1:], lab[:-1])):
        cross = (a > 0) & (b > 0) & (a != b)
        diff.append((la != lb)[cross])
    out["boundary_contrast"] = np.concatenate(diff).mean()
    return out


def stability(labs):
    """Mean and min pairwise ARI over a list of label arrays."""
    aris = [adjusted_rand_score(labs[i].ravel(), labs[j].ravel())
            for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(aris)), float(np.min(aris))


def edge_adherence(lab, fid):
    """Where do cluster edges fall? d_in = share of 4-neighbour pairs INSIDE one field that get
    different clusters (speckle, want low); d_out = same for pairs ACROSS two different labelled
    fields (real edges, want high). ratio = d_out / d_in (higher = edges follow field boundaries)."""
    din, dout = [], []
    for a, b, la, lb in ((fid[:, 1:], fid[:, :-1], lab[:, 1:], lab[:, :-1]), (fid[1:], fid[:-1], lab[1:], lab[:-1])):
        d = la != lb
        din.append(d[(a > 0) & (a == b)])
        dout.append(d[(a > 0) & (b > 0) & (a != b)])
    d_in, d_out = np.concatenate(din).mean(), np.concatenate(dout).mean()
    return dict(d_in=float(d_in), d_out=float(d_out), ratio=float(d_out / max(d_in, 1e-9)))


# ───────────────────────── over-cluster + merge ─────────────────────────

def adjacency(lab, valid, K):
    """A[a, b] = number of 4-neighbour pixel pairs with labels a and b (a != b), symmetric."""
    A = np.zeros((K, K))
    for a, b, m in ((lab[:, 1:], lab[:, :-1], valid[:, 1:] & valid[:, :-1]), (lab[1:], lab[:-1], valid[1:] & valid[:-1])):
        d = m & (a != b)
        np.add.at(A, (a[d], b[d]), 1)
    return A + A.T


def merge_tree(C, n, k, method="ward", A=None, gamma=0.0):
    """Greedy agglomerative merge of K0 centroids C (K0, D) with pixel counts n down to k groups.
    ward:    cost(a,b) = n_a n_b / (n_a + n_b) * ||mu_a - mu_b||^2  (increase in inertia)
    average: mean pairwise centroid distance between the groups (unweighted)
    gamma>0: Ward cost divided by (1 + gamma * s_ab), s_ab = shared boundary / sqrt(P_a P_b),
             i.e. clusters that touch a lot on the map merge earlier.
    Returns lut (K0,) -> group id in 0..k-1."""
    K0 = len(C)
    mu = C.astype(np.float64).copy()
    cnt = n.astype(np.float64).copy()
    groups = {i: [i] for i in range(K0)}
    A = None if A is None else A.astype(np.float64).copy()
    D0 = ((C[:, None, :].astype(np.float64) - C[None]) ** 2).sum(-1)

    def cost(a, b):
        if method == "average":
            return np.sqrt(D0[np.ix_(groups[a], groups[b])]).mean()
        c = cnt[a] * cnt[b] / (cnt[a] + cnt[b]) * ((mu[a] - mu[b]) ** 2).sum()
        if gamma and A is not None:
            P = A.sum(1)
            s = A[a, b] / np.sqrt(max(P[a] * P[b], 1e-9))
            c /= 1 + gamma * s
        return c

    while len(groups) > k:
        keys = list(groups)
        if method == "ward":  # vectorised over all active pairs
            ki = np.array(keys)
            m, c = mu[ki], cnt[ki]
            Cm = c[:, None] * c[None] / (c[:, None] + c[None]) * ((m[:, None] - m[None]) ** 2).sum(-1)
            if gamma and A is not None:
                P = A.sum(1)[ki]
                Cm = Cm / (1 + gamma * A[np.ix_(ki, ki)] / np.sqrt(np.maximum(P[:, None] * P[None], 1e-9)))
            np.fill_diagonal(Cm, np.inf)
            i, j = np.unravel_index(Cm.argmin(), Cm.shape)
            a, b = int(ki[min(i, j)]), int(ki[max(i, j)])
        else:
            best = min(((cost(a, b), a, b) for i, a in enumerate(keys) for b in keys[i + 1:]))
            _, a, b = best
        mu[a] = (cnt[a] * mu[a] + cnt[b] * mu[b]) / (cnt[a] + cnt[b])
        cnt[a] += cnt[b]
        groups[a] += groups.pop(b)
        if A is not None:
            A[a] += A[b]; A[:, a] += A[:, b]; A[a, a] = 0
            A[b] = 0; A[:, b] = 0
    lut = np.empty(K0, int)
    for g, (_, members) in enumerate(sorted(groups.items())):
        lut[members] = g
    return lut
