"""Map figures for the report (seed 42).
Gothenburg: colour maps (k = 5), clusters matched to 'kmeans n_init=10' by the Hungarian method.
Austria: edge maps on a crop - grey = field boundaries (INVEKOS 2022), dark = cluster boundaries.
    ~/tee-local/venv/bin/python make_figures.py
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import tsh

R = Path(__file__).resolve().parent / "results"
FIG = R / "fig"
FIG.mkdir(exist_ok=True)
PAL = np.array([[0x2a, 0x78, 0xd6], [0xeb, 0x68, 0x34], [0x1b, 0xaf, 0x7a], [0xed, 0xa1, 0x00],
                [0xe8, 0x7b, 0xa4], [0x00, 0x83, 0x00], [0x4a, 0x3a, 0xa7], [0xe3, 0x49, 0x48]], np.uint8)
GREY = np.array([0x9a, 0x99, 0x94], np.uint8)


def match(lab, ref):
    """Relabel lab so its clusters get the colour of the best-overlapping ref cluster."""
    ka, kb = lab.max() + 1, ref.max() + 1
    C = np.zeros((ka, max(kb, ka)), int)
    np.add.at(C, (lab.ravel(), ref.ravel()), 1)
    r, c = linear_sum_assignment(-C)
    lut = np.arange(ka)
    lut[r] = c
    return lut[lab]


def colour(lab):
    img = np.where(lab[..., None] < len(PAL), PAL[np.minimum(lab, len(PAL) - 1)], GREY)
    return img.astype(np.uint8)


def gbg():
    m = np.load(R / "maps_gbg.npz")
    ref = m["kmeans n_init=10"].astype(int)
    names = {"kmeans (TEE)": "tee", "kmeans n_init=10": "ninit", "post mode r=20m x1": "mode20",
             "pre gauss sigma=10m": "gauss10", "MRF_ICM beta=0.5": "icm05", "pre bilateral r=10m x2": "bil10",
             "HDBSCAN pca16 mcs=100": "hdb"}
    for n, f in names.items():
        lab = match(m[n].astype(int), ref)
        img = colour(lab)
        # TEE pixels are ~7 m x 13 m: stretch to the true (square) 5 x 5 km shape
        Image.fromarray(img).resize((720, 720), Image.NEAREST).save(FIG / f"gbg_{f}.png")
        crop = img[120:240, 250:475]  # ~1.6 km x 1.6 km around the river / city centre
        Image.fromarray(crop).resize((480, 480), Image.NEAREST).save(FIG / f"gbg_{f}_zoom.png")


def aut():
    m = np.load(R / "maps_aut2024.npz")
    ds = tsh.austria()
    fid = ds["fid"]
    y0, x0, s = 300, 250, 320
    F = fid[y0:y0 + s, x0:x0 + s]
    fb = np.zeros_like(F, bool)
    fb[:, 1:] |= F[:, 1:] != F[:, :-1]
    fb[1:] |= F[1:] != F[:-1]
    for n, f in {"kmeans (TEE)": "tee", "kmeans n_init=10": "ninit", "post mode r=20m x1": "mode20",
                 "pre gauss sigma=10m": "gauss10", "MRF_ICM beta=0.5": "icm05",
                 "pre bilateral r=10m x2": "bil10", "HDBSCAN pca16 mcs=100": "hdb"}.items():
        L = m[n][y0:y0 + s, x0:x0 + s]
        cb = np.zeros_like(L, bool)
        cb[:, 1:] |= L[:, 1:] != L[:, :-1]
        cb[1:] |= L[1:] != L[:-1]
        img = np.full(L.shape + (3,), 252, np.uint8)
        img[fb] = (0x86, 0xb6, 0xef)
        img[cb] = (0x1a, 0x1a, 0x19)
        Image.fromarray(img).resize((640, 640), Image.NEAREST).save(FIG / f"aut_{f}_edges.png")


def overcluster():
    """Maps for the over-cluster + merge section (seed 42)."""
    g = np.load(R / "maps_overcluster_gbg.npz")
    ref = np.load(R / "maps_gbg.npz")["kmeans n_init=10"].astype(int)
    for n, f in {"over K0=64 + Ward": "over64", "over K0=64 + average link": "overavg",
                 "over K0=32 + Ward + refine + MRF 0.25": "overbest"}.items():
        img = colour(match(g[n].astype(int), ref))
        Image.fromarray(img[120:240, 250:475]).resize((480, 480), Image.NEAREST).save(FIG / f"gbg_{f}_zoom.png")
    a = np.load(R / "maps_overcluster_aut2024.npz")
    fid = tsh.austria()["fid"]
    y0, x0, s = 300, 250, 320
    F = fid[y0:y0 + s, x0:x0 + s]
    fb = np.zeros_like(F, bool); fb[:, 1:] |= F[:, 1:] != F[:, :-1]; fb[1:] |= F[1:] != F[:-1]
    L = a["over K0=32 + Ward + refine + MRF 0.25"][y0:y0 + s, x0:x0 + s]
    cb = np.zeros_like(L, bool); cb[:, 1:] |= L[:, 1:] != L[:, :-1]; cb[1:] |= L[1:] != L[:-1]
    img = np.full(L.shape + (3,), 252, np.uint8); img[fb] = (0x86, 0xb6, 0xef); img[cb] = (0x1a, 0x1a, 0x19)
    Image.fromarray(img).resize((640, 640), Image.NEAREST).save(FIG / "aut_overbest_edges.png")


if __name__ == "__main__":
    gbg()
    aut()
    overcluster()
    print(sorted(p.name for p in FIG.iterdir()))
