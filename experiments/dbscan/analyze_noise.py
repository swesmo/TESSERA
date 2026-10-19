"""Where are DBSCAN's noise pixels? (Romaric's question on 30 Sep: discard them as bad data, or are
they real pixels that TEE must still label?)    ~/tee-local/venv/bin/python analyze_noise.py [gbg|aut22]

For every DBSCAN configuration of run_dbscan.py (seed-42 maps) with 2-80 % noise:
  - noise rate on k-means cluster boundaries vs interiors (4-neighbourhood, TEE k-means seed 42),
  - noise rate per quartile of local heterogeneity (mean embedding distance to the 4 neighbours:
    high = mixed pixel / edge, low = homogeneous patch),
  - noise rate per k-means cluster,
  - Austria only: inside field interiors vs field edges vs outside the INVEKOS fields, and per crop class.
Writes results/noise_<tag>.json.
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spatial_hdbscan"))
import tsh  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
which = sys.argv[1] if len(sys.argv) > 1 else "gbg"
ds = tsh.gothenburg() if which == "gbg" else tsh.austria(year=2022)
k = 5 if which == "gbg" else 17
E, V = ds["emb"], ds["valid"]
H, W, _ = E.shape
maps = np.load(OUT / f"maps_dbscan_{which}.npz")
km = maps[f"kmeans (TEE) k={k}"].astype(np.int64)


def nb_any(cond_fn, a):
    """Pixels where cond_fn(a[p], a[q]) holds for at least one 4-neighbour q."""
    out = np.zeros(a.shape, bool)
    for sl_p, sl_q in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                       ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        c = cond_fn(a[sl_p], a[sl_q])
        out[sl_p] |= c
        out[sl_q] |= c
    return out


# local heterogeneity: mean Euclidean distance to the valid 4-neighbours
het = np.zeros((H, W)); cnt = np.zeros((H, W))
for dy, dx in ((0, 1), (1, 0)):
    a = E[:H - dy, :W - dx]; b = E[dy:, dx:]
    d = np.sqrt(((a - b) ** 2).sum(-1))
    het[:H - dy, :W - dx] += d; cnt[:H - dy, :W - dx] += 1
    het[dy:, dx:] += d; cnt[dy:, dx:] += 1
het /= np.maximum(cnt, 1)
q = np.quantile(het[V], [0.25, 0.5, 0.75])
het_q = np.digitize(het, q)  # 0..3

km_edge = nb_any(lambda x, y: x != y, km) & V
res = dict(het_quartile_edges=q.tolist(), km_edge_share=float(km_edge[V].mean()))
if "fid" in ds:
    fid = ds["fid"]
    f_edge = nb_any(lambda x, y: x != y, fid) & (fid > 0)
    zones = {"field interior": (fid > 0) & ~f_edge, "field edge": f_edge, "outside fields": fid == 0}
    res["zone_share"] = {z: float(m[V].mean()) for z, m in zones.items()}

for name in maps.files:
    if not name.startswith("DBSCAN"):
        continue
    lab = maps[name]
    noise = (lab < 0) & V
    r = float(noise[V].mean())
    if not 0.02 <= r <= 0.8:
        continue
    o = dict(noise=r,
             noise_on_kmeans_edges=float(noise[km_edge].mean()),
             noise_in_kmeans_interior=float(noise[V & ~km_edge].mean()),
             noise_by_het_quartile=[float(noise[V & (het_q == i)].mean()) for i in range(4)],
             noise_by_kmeans_cluster={int(c): float(noise[V & (km == c)].mean()) for c in range(k)},
             kmeans_cluster_share={int(c): float((km[V] == c).mean()) for c in range(k)})
    if "fid" in ds:
        o["noise_by_zone"] = {z: float(noise[m & V].mean()) for z, m in zones.items()}
        cls = ds["cls"]
        o["noise_by_class"] = {int(c): float(noise[(cls == c) & (fid > 0)].mean()) for c in np.unique(cls[fid > 0])}
    res[name] = o
    print(f"{name:26s} noise={r:.3f} edges={o['noise_on_kmeans_edges']:.3f} interior={o['noise_in_kmeans_interior']:.3f} "
          f"het-quartiles={[round(x, 3) for x in o['noise_by_het_quartile']]}"
          + (f" zones={ {z: round(v, 3) for z, v in o['noise_by_zone'].items()} }" if "fid" in ds else ""), flush=True)
(OUT / f"noise_{which}.json").write_text(json.dumps(res, indent=1))
