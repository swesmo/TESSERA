#!/usr/bin/env python3

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import tsh


K = 5
S = 5000
SEEDS = [42] + list(range(1, 10))
NEIGHBORS = [5, 10, 15, 20, 30, 40, 60, 120]

GT_FILE = Path.home() / "tee_data/harness-cache/gbg_manual_gt.npy"
OUT = Path("results/gbg_kmeans_vs_spectral_gt.json")


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

ds = tsh.gothenburg()

E = ds["emb"]
V = ds["valid"]
X = E[V]

gt = np.load(GT_FILE)

if gt.shape != V.shape:
    raise RuntimeError(
        f"GT shape {gt.shape} != viewport shape {V.shape}"
    )

mask = (gt > 0) & V
gt_labels = gt[mask]

print("=" * 80)
print("GOTHENBURG: K-MEANS vs SPECTRAL WITH MANUAL GROUND TRUTH")
print("=" * 80)

print("Dataset:", ds["name"])
print("Shape:", E.shape)
print("Valid pixels:", len(X))
print("Labelled pixels:", int(mask.sum()))
print(f"GT coverage: {mask.sum() / V.sum() * 100:.2f}%")
print("k:", K)
print("Sample size:", S)
print("Seeds:", SEEDS)
print("Spectral n_neighbors:", NEIGHBORS)
print()


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def scores(labels):
    grid = np.full(V.shape, -1, dtype=np.int32)
    grid[V] = labels

    pred = grid[mask]

    return (
        float(adjusted_rand_score(gt_labels, pred)),
        float(adjusted_mutual_info_score(gt_labels, pred)),
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
        "max": float(vals.max()),
        "n_pairs": int(len(vals)),
    }


def summarise(name, labels_by_seed, ari, ami, runtimes, largest):
    stab = stability(labels_by_seed)

    result = {
        "method": name,

        "ari_mean": float(np.mean(ari)),
        "ari_std": float(np.std(ari, ddof=1)),

        "ami_mean": float(np.mean(ami)),
        "ami_std": float(np.std(ami, ddof=1)),

        "stability_mean": stab["mean"],
        "stability_std": stab["std"],
        "stability_min": stab["min"],
        "stability_max": stab["max"],
        "stability_pairs": stab["n_pairs"],

        "runtime_mean_s": float(np.mean(runtimes)),
        "runtime_median_s": float(np.median(runtimes)),

        "largest_cluster_mean": float(np.mean(largest)),
        "largest_cluster_max": float(np.max(largest)),

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

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)
    print(
        f"ARI:       {result['ari_mean']:.3f} "
        f"+- {result['ari_std']:.3f}"
    )
    print(
        f"AMI:       {result['ami_mean']:.3f} "
        f"+- {result['ami_std']:.3f}"
    )
    print(
        f"Stability: {result['stability_mean']:.3f} "
        f"+- {result['stability_std']:.3f}"
    )
    print(
        f"Stab min:  {result['stability_min']:.3f}"
    )
    print(
        f"Largest:   {result['largest_cluster_mean'] * 100:.1f}%"
    )
    print(
        f"Runtime:   {result['runtime_mean_s']:.3f}s"
    )

    return result


results = {}


# ------------------------------------------------------------
# TEE K-MEANS
# ------------------------------------------------------------

print("\n" + "#" * 80)
print("TEE K-MEANS")
print("#" * 80)

labels_by_seed = []
aris = []
amis = []
runtimes = []
largest = []

for seed in SEEDS:

    t0 = time.perf_counter()

    C = tsh.tee_kmeans(
        X,
        k=K,
        seed=seed,
        sample=S,
    )

    labels = tsh.sqd(X, C).argmin(1)

    dt = time.perf_counter() - t0

    ari, ami = scores(labels)

    shares = (
        np.bincount(labels, minlength=K)
        / len(labels)
    )

    labels_by_seed.append(labels)
    aris.append(ari)
    amis.append(ami)
    runtimes.append(dt)
    largest.append(float(shares.max()))

    print(
        f"seed={seed:3d} | "
        f"ARI={ari:.3f} | "
        f"AMI={ami:.3f} | "
        f"largest={shares.max()*100:5.1f}% | "
        f"time={dt:.3f}s"
    )

results["kmeans"] = summarise(
    "TEE K-means",
    labels_by_seed,
    aris,
    amis,
    runtimes,
    largest,
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
    print(f"SPECTRAL n_neighbors={nn}")
    print("#" * 80)

    labels_by_seed = []
    aris = []
    amis = []
    runtimes = []
    largest = []

    for seed in SEEDS:

        t0 = time.perf_counter()

        idx, labels_sample = tsh.spectral_sample(
            X,
            k=K,
            seed=seed,
            S=S,
            n_neighbors=nn,
        )

        means = tsh.group_means(
            X[idx],
            labels_sample,
            K,
        )

        labels = tsh.sqd(
            X,
            means,
        ).argmin(1)

        dt = time.perf_counter() - t0

        ari, ami = scores(labels)

        shares = (
            np.bincount(labels, minlength=K)
            / len(labels)
        )

        labels_by_seed.append(labels)
        aris.append(ari)
        amis.append(ami)
        runtimes.append(dt)
        largest.append(float(shares.max()))

        print(
            f"seed={seed:3d} | "
            f"ARI={ari:.3f} | "
            f"AMI={ami:.3f} | "
            f"largest={shares.max()*100:5.1f}% | "
            f"time={dt:.3f}s"
        )

    results[f"spectral_nn{nn}"] = summarise(
        f"Spectral nn={nn}",
        labels_by_seed,
        aris,
        amis,
        runtimes,
        largest,
    )


# ------------------------------------------------------------
# FINAL TABLE
# ------------------------------------------------------------

print()
print("=" * 100)
print("FINAL SUMMARY")
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

for key, r in results.items():
    print(
        f"{r['method']:20s} "
        f"{r['ari_mean']:8.3f} "
        f"{r['ami_mean']:8.3f} "
        f"{r['stability_mean']:10.3f} "
        f"{r['stability_min']:9.3f} "
        f"{r['largest_cluster_mean']*100:8.1f}% "
        f"{r['runtime_mean_s']:8.3f}s"
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

OUT.write_text(
    json.dumps(
        {
            "dataset": ds["name"],
            "shape": list(E.shape),
            "valid_pixels": int(V.sum()),
            "labelled_pixels": int(mask.sum()),
            "gt_coverage": float(mask.sum() / V.sum()),
            "k": K,
            "sample_size": S,
            "seeds": SEEDS,
            "neighbors": NEIGHBORS,
            "results": results,
        },
        indent=2,
    )
)

print()
print("Saved:", OUT)
