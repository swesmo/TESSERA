"""Export PCA parameters, sample indices and eps values for bench_dbscan.mjs (Gothenburg, seed 42),
plus the sklearn reference result on the same sample so the JS DBSCAN can be checked.
    ~/tee-local/venv/bin/python export_js_params.py"""
import json, sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spatial_hdbscan"))
import tsh  # noqa: E402


def knee(sorted_y):
    """Same as run_dbscan.knee: point of the increasing curve furthest below its chord."""
    x = np.linspace(0, 1, len(sorted_y))
    y = (sorted_y - sorted_y[0]) / max(sorted_y[-1] - sorted_y[0], 1e-12)
    return int(np.argmax(x - y))

ds = tsh.gothenburg()
X = ds["emb"].reshape(-1, 128)
rng = np.random.default_rng(42)
out = {}
for n in (2000, 5000, 10000, 20000):
    idx = rng.choice(len(X), n, replace=False)
    for d in (8, 16):
        pca = PCA(n_components=d, random_state=42).fit(X[idx])
        Zs = pca.transform(X[idx])
        m = 2 * d
        kd = np.sort(NearestNeighbors(n_neighbors=m).fit(Zs).kneighbors(Zs)[0][:, m - 1])
        eps = float(kd[knee(kd)])
        db = DBSCAN(eps=eps, min_samples=m).fit(Zs)
        out[f"n{n}_d{d}"] = dict(idx=idx.tolist(), mean=pca.mean_.tolist(), comp=pca.components_.tolist(),
                                 eps=eps, minPts=m, sk_clusters=int(db.labels_.max() + 1),
                                 sk_noise=float((db.labels_ < 0).mean()))
        print(n, d, eps, out[f"n{n}_d{d}"]["sk_clusters"], out[f"n{n}_d{d}"]["sk_noise"], flush=True)
(Path(__file__).resolve().parent / "results/js_params.json").write_text(json.dumps(out))
