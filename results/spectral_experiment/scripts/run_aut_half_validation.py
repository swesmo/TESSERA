#!/usr/bin/env python3

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph

import tsh


PART = sys.argv[1] if len(sys.argv) > 1 else None

if PART not in ("west", "east"):
    raise SystemExit("Usage: python run_aut_half_validation.py west|east")

K = 17
S_SPECTRAL = 5000
SEEDS = [42] + list(range(1, 10))

# Only the configurations worth validating spatially
NEIGHBORS = [5, 10, 15, 60]


# ------------------------------------------------------------
# DATA + EXACT CHRISTOPHER SPLIT
# ------------------------------------------------------------

ds = tsh.austria(year=2022)

half = ds["emb"].shape[1] // 2

sl = (
    slice(0, half)
    if PART == "west"
    else slice(half, None)
)

E = ds["emb"][:, sl]
V = ds["valid"][:, sl]
FID = ds["fid"][:, sl]
CLS = ds["cls"][:, sl]

X = E[V]

mask = (FID > 0) & V
gt = CLS[mask]

OUT = Path(f"results/aut2022_spectral_validation_{PART}.json")


print("=" * 80)
print(f"AUSTRIA 2022 — {PART.upper()} HALF")
print("=" * 80)

print("Shape:", E.shape)
print("Valid pixels:", int(V.sum()))
print("Labelled pixels:", int(mask.sum()))
print(f"GT coverage: {mask.sum()/V.sum()*100:.2f}%")
print("k:", K)
print("K-means sample:", max(5000, 500*K))
print("Spectral sample:", S_SPECTRAL)
print("Seeds:", SEEDS)
print("Spectral n_neighbors:", NEIGHBORS)
print()


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def scores(labels):
    grid = np.full(V.shape, -1, dtype=np.int16)
    grid[V] = labels

    pred = grid[mask]

    return (
        float(adjusted_rand_score(gt, pred)),
        float(adjusted_mutual_info_score(gt, pred)),
    )


def stability(labels_by_seed):
    vals = []

    for i in range(len(labels_by_seed)):
        for j in range(i + 1, len(labels_by_seed)):
            vals.append(
                adjusted_rand_score(
                    labels_by_seed[i],
                    labels_by_seed[j],
                )
            )

    vals = np.asarray(vals)

    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "min": float(vals.min()),
    }


def summarise(name, labels_by_seed, ari, ami, runtimes,
              largest, components=None):

    stab = stability(labels_by_seed)

    r = {
        "method": name,

        "ari_mean": float(np.mean(ari)),
        "ari_std": float(np.std(ari, ddof=1)),

        "ami_mean": float(np.mean(ami)),
        "ami_std": float(np.std(ami, ddof=1)),

        "stability_mean": stab["mean"],
        "stability_std": stab["std"],
        "stability_min": stab["min"],

        "largest_cluster_mean": float(np.mean(largest)),
        "runtime_mean_s": float(np.mean(runtimes)),

        "per_seed": [
            {
                "seed": int(SEEDS[i]),
                "ari": float(ari[i]),
                "ami": float(ami[i]),
                "runtime_s": float(runtimes[i]),
                "largest_cluster": float(largest[i]),
            }
            for i in range(len(SEEDS))
        ],
    }

    if components is not None:
        r["graph_components_mean"] = float(np.mean(components))
        r["graph_components_max"] = int(np.max(components))

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    print(f"ARI:       {r['ari_mean']:.3f} +- {r['ari_std']:.3f}")
    print(f"AMI:       {r['ami_mean']:.3f} +- {r['ami_std']:.3f}")
    print(
        f"Stability: {r['stability_mean']:.3f} "
        f"+- {r['stability_std']:.3f}"
    )
    print(f"Stab min:  {r['stability_min']:.3f}")
    print(f"Largest:   {r['largest_cluster_mean']*100:.1f}%")
    print(f"Runtime:   {r['runtime_mean_s']:.3f}s")

    if components is not None:
        print(
            f"Graph comp:{r['graph_components_mean']:.1f} "
            f"(max {r['graph_components_max']})"
        )

    return r


results = {}


# ------------------------------------------------------------
# K-MEANS BASELINE
# ------------------------------------------------------------

print("\n" + "#" * 80)
print("TEE K-MEANS")
print("#" * 80)

labs = []
aris = []
amis = []
times = []
largest = []

for seed in SEEDS:

    t0 = time.perf_counter()

    C = tsh.tee_kmeans(
        X,
        k=K,
        seed=seed,
    )

    labels = tsh.sqd(X, C).argmin(1)

    dt = time.perf_counter() - t0

    ari, ami = scores(labels)

    shares = np.bincount(labels, minlength=K) / len(labels)

    labs.append(labels.astype(np.int16))
    aris.append(ari)
    amis.append(ami)
    times.append(dt)
    largest.append(float(shares.max()))

    print(
        f"seed={seed:3d} | "
        f"ARI={ari:.3f} | "
        f"AMI={ami:.3f} | "
        f"largest={shares.max()*100:5.1f}% | "
        f"time={dt:.2f}s"
    )

results["kmeans"] = summarise(
    "TEE K-means",
    labs, aris, amis, times, largest
)


# ------------------------------------------------------------
# SPECTRAL
# ------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    message="Graph is not fully connected",
)

for nn in NEIGHBORS:

    print("\n" + "#" * 80)
    print(f"SPECTRAL nn={nn}")
    print("#" * 80)

    labs = []
    aris = []
    amis = []
    times = []
    largest = []
    components = []

    for seed in SEEDS:

        t0 = time.perf_counter()

        idx, labels_sample = tsh.spectral_sample(
            X,
            k=K,
            seed=seed,
            S=S_SPECTRAL,
            n_neighbors=nn,
        )

        means = tsh.group_means(
            X[idx],
            labels_sample,
            K,
        )

        labels = tsh.sqd(X, means).argmin(1)

        dt = time.perf_counter() - t0

        ari, ami = scores(labels)

        shares = np.bincount(labels, minlength=K) / len(labels)

        graph = kneighbors_graph(
            X[idx],
            nn,
            include_self=False,
        )

        ncomp = connected_components(
            graph,
            directed=False,
        )[0]

        labs.append(labels.astype(np.int16))
        aris.append(ari)
        amis.append(ami)
        times.append(dt)
        largest.append(float(shares.max()))
        components.append(ncomp)

        print(
            f"seed={seed:3d} | "
            f"ARI={ari:.3f} | "
            f"AMI={ami:.3f} | "
            f"comp={ncomp:2d} | "
            f"largest={shares.max()*100:5.1f}% | "
            f"time={dt:.2f}s"
        )

    results[f"spectral_nn{nn}"] = summarise(
        f"Spectral nn={nn}",
        labs,
        aris,
        amis,
        times,
        largest,
        components,
    )


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("=" * 100)
print(f"FINAL SUMMARY — AUSTRIA {PART.upper()}")
print("=" * 100)

print(
    f"{'method':20s} "
    f"{'ARI':>8s} "
    f"{'AMI':>8s} "
    f"{'stability':>10s} "
    f"{'stab_min':>9s} "
    f"{'largest':>9s} "
    f"{'runtime':>9s}"
)

for r in results.values():

    print(
        f"{r['method']:20s} "
        f"{r['ari_mean']:8.3f} "
        f"{r['ami_mean']:8.3f} "
        f"{r['stability_mean']:10.3f} "
        f"{r['stability_min']:9.3f} "
        f"{r['largest_cluster_mean']*100:8.1f}% "
        f"{r['runtime_mean_s']:8.3f}s"
    )


OUT.write_text(
    json.dumps(
        {
            "dataset": ds["name"],
            "part": PART,
            "shape": list(E.shape),
            "valid_pixels": int(V.sum()),
            "labelled_pixels": int(mask.sum()),
            "k": K,
            "seeds": SEEDS,
            "neighbors": NEIGHBORS,
            "results": results,
        },
        indent=2,
    )
)

print()
print("Saved:", OUT)
