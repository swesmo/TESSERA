"""Recreate the two seed maps on slide 02 (Gothenburg 2024, k = 5, seed 42 vs seed 7).

Faithful NumPy port of TEE's browser k-means (public/js/segmentation.js) and RVQ
reconstruction (public/js/vq_reconstruct.js). Checked on 26/27 Sep 2026: the cluster
shares match TEE's Auto-label panel exactly (seed 42: 23.5/14.9/30.9/9.5/21.2 %,
seed 7: 21.4/6.5/15.0/26.8/30.3 %).

Needs the viewport cached by TEE in ~/tee_data (open it once in TEE) and numpy + Pillow:
    ~/tee-local/venv/bin/python make_seed_maps.py            # seeds 42 and 7
    ~/tee-local/venv/bin/python make_seed_maps.py 42 13      # any two seeds
Writes map_seed<A>.png / map_seed<B>.png next to this script and prints the shares
and the fraction of pixels that change cluster.
"""
import gzip, io, itertools, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

VP = Path.home() / "tee_data/vectors/Gteborg/2024"
OUT = Path(__file__).resolve().parent
K = 5
M32 = 0xFFFFFFFF
# slide palette; cluster ids of the second run are matched to the first run's
PALETTE = np.array([[0x55, 0x60, 0x6E], [0x6E, 0x8B, 0x3D], [0xD9, 0xCB, 0x9A],
                    [0xE4, 0x57, 0x2E], [0xA9, 0xB3, 0xBC]], np.uint8)


def load_npy(name):
    with gzip.open(VP / name, "rb") as f:
        return np.load(io.BytesIO(f.read()))


def mulberry32(seed):
    a = seed & M32

    def rand():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & M32)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return rand


def load_embeddings():
    """RVQ codebooks + indices -> uint8 mosaic -> dequantised float32, as TEE does it."""
    meta = json.loads((VP / "vq_metadata.json").read_text())
    t, H, W = meta["tile_size"], *meta["output_shape"]
    nR, nC = meta["n_tile_rows"], meta["n_tile_cols"]
    top, left = meta.get("crop_offset", [0, 0])
    fullH, fullW = meta.get("mosaic_shape", meta["output_shape"])

    def decode(u8, sc):
        mn, mx = sc[:, :, 0].astype(np.float64), sc[:, :, 1].astype(np.float64)
        span = mx - mn
        span[span == 0] = 1.0
        return (mn[:, None, :] + (u8 / 255.0) * span[:, None, :]).astype(np.float32)

    cb1 = decode(load_npy("codebooks1_uint8.npy.gz"), load_npy("codebooks1_scales.npy.gz"))
    cb2 = decode(load_npy("codebooks2_uint8.npy.gz"), load_npy("codebooks2_scales.npy.gz"))
    idx1 = load_npy("indices1.npy.gz").reshape(-1)[: H * W]
    idx2 = load_npy("indices2.npy.gz").reshape(-1)[: H * W]

    def tile_of(p, n, full):
        return np.where(p >= full - t, n - 1, p // t)

    ly, lx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    tile = (tile_of(top + ly, nR, fullH) * nC + tile_of(left + lx, nC, fullW)).reshape(-1)
    v = (cb1[tile, idx1].astype(np.float64) + cb2[tile, idx2]).astype(np.float32)
    dmin, dmax = v.min(0), v.max(0)
    dscale = (dmax.astype(np.float64) - dmin).astype(np.float32)
    dscale[dscale == 0] = 1
    q = np.clip(np.floor((v.astype(np.float64) - dmin) / dscale * 255 + 0.5), 0, 255)
    scale = ((dmax.astype(np.float64) - dmin) / 255).astype(np.float32)
    return (q * scale + dmin).astype(np.float32), H, W


def sqdist(X, c):
    d = X.astype(np.float64) - c.astype(np.float64)
    return np.einsum("ij,ij->i", d, d)


def tee_kmeans(emb, k, seed, max_iter=20):
    N, dim = emb.shape
    S = min(max(5000, k * 500), N)
    rs = mulberry32(seed)  # buildSample: partial Fisher-Yates
    pool = np.arange(N)
    for i in range(S):
        j = i + int(rs() * (N - i))
        pool[i], pool[j] = pool[j], pool[i]
    X = emb[pool[:S]]
    rand = mulberry32((seed & M32) ^ 0x9E3779B9)
    cent = np.zeros((k, dim), np.float32)
    cent[0] = X[int(rand() * S)]
    mind = np.full(S, np.inf)
    for c in range(1, k):  # k-means++ on the sample
        mind = np.minimum(mind, sqdist(X, cent[c - 1]))
        r, chosen = rand() * float(np.cumsum(mind)[-1]), 0
        for si in range(S):
            r -= mind[si]
            if r <= 0:
                chosen = si
                break
        cent[c] = X[chosen]
    assign = np.zeros(S, np.int64)
    for _ in range(max_iter):  # Lloyd on the sample, stop below 0.5 % changes
        new = np.stack([sqdist(X, cent[c]) for c in range(k)], 1).argmin(1)
        changed, assign = int((new != assign).sum()), new
        for c in range(k):
            m = assign == c
            cent[c] = X[int(rand() * S)] if not m.any() else (X[m].astype(np.float64).sum(0) / m.sum()).astype(np.float32)
        if changed < S * 0.005:
            break
    return np.concatenate([np.stack([sqdist(emb[s:s + 50000], cent[c]) for c in range(k)], 1).argmin(1)
                           for s in range(0, N, 50000)])  # one pass over all N


def main():
    seeds = [int(s) for s in sys.argv[1:3]] or [42, 7]
    emb, H, W = load_embeddings()
    a, b = (tee_kmeans(emb, K, s) for s in seeds)
    C = np.zeros((K, K), int)
    np.add.at(C, (a, b), 1)
    best = max(itertools.permutations(range(K)), key=lambda p: sum(C[p[j], j] for j in range(K)))
    b = np.array(best)[b]  # relabel run B to the matching clusters of run A
    for s, lab in zip(seeds, (a, b)):
        shares = np.bincount(lab, minlength=K) / len(lab) * 100
        print(f"seed {s}: " + " / ".join(f"{x:.1f}" for x in shares) + " %")
        # pixels are square in degrees (~7 m x 13 m here); stretch to the true 5 x 5 km shape
        Image.fromarray(PALETTE[lab.reshape(H, W)]).resize((1080, 1080), Image.NEAREST).save(OUT / f"map_seed{s}.png")
    print(f"pixels that change cluster: {(a != b).mean() * 100:.1f} %")


if __name__ == "__main__":
    main()
