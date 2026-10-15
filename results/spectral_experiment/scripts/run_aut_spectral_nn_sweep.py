#!/usr/bin/env python3

import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph

import tsh


K = 17
S_SPECTRAL = 5000
SEEDS = [42] + list(range(1, 10))
NEIGHBORS = [5, 10, 15, 20, 30, 40, 60, 120]

OUT = Path("results/aut2022_kmeans_vs_spectral_nn.json")


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

ds = tsh.austria(year=2022)

E = ds["emb"]
V = ds["valid"]
X = E[V]

mask = (ds["fid"] > 0) & V
gt = ds["cls"][mask]

print("=" * 80)
print("AUSTRIA 2022: K-MEANS vs SPECTRAL n_neighbors")
print("=" * 80)

print("Dataset:", ds["name"])
print("Shape:", E.shape)
print("Valid pixels:", len(X))
print("Labelled pixels:", int(mask.sum()))
print(f"GT coverage: {mask.sum()/V.sum()*100:.2f}%")
print("k:", K)
print("TEE K-means sample:", max(5000, 500*K))
print("Spectral sample:", S_SPECTRAL)
print("Seeds:", SEEDS)
print("n_neighbors:", NEIGHBORS)
print()


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
        "max": float(vals.max()),
        "n_pairs": len(vals),
    }


def summarise(name, labels_by_seed, ari, ami, runtimes, largest,
              components=None):

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
        "stability_max": stab["max"],

        "largest_cluster_mean": float(np.mean(largest)),
        "largest_cluster_max": float(np.max(largest)),

        "runtime_mean_s": float(np.mean(runtimes)),
        "runtime_median_s": float(np.median(runtimes)),

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
# TEE K-MEANS BASELINE
# ------------------------------------------------------------

print("\n" + "#" * 80)
print("TEE K-MEANS")
print("#" * 80)

labels_by_seed = []
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

    labels_by_seed.append(labels.astype(np.int16))
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
    labels_by_seed,
    aris,
    amis,
    times,
    largest,
)


# ------------------------------------------------------------
# SPECTRAL SWEEP
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

        n_components = connected_components(
            graph,
            directed=False,
        )[0]

        labels_by_seed.append(labels.astype(np.int16))
        aris.append(ari)
        amis.append(ami)
        times.append(dt)
        largest.append(float(shares.max()))
        components.append(n_components)

        print(
            f"seed={seed:3d} | "
            f"ARI={ari:.3f} | "
            f"AMI={ami:.3f} | "
            f"comp={n_components:3d} | "
            f"largest={shares.max()*100:5.1f}% | "
            f"time={dt:.2f}s"
        )

    results[f"spectral_nn{nn}"] = summarise(
        f"Spectral nn={nn}",
        labels_by_seed,
        aris,
        amis,
        times,
        largest,
        components,
    )


# ------------------------------------------------------------
# FINAL TABLE
# ------------------------------------------------------------

print()
print("=" * 110)
print("FINAL SUMMARY — AUSTRIA 2022")
print("=" * 110)

print(
    f"{'method':20s} "
    f"{'ARI':>8s} "
    f"{'AMI':>8s} "
    f"{'stability':>10s} "
    f"{'stab_min':>9s} "
    f"{'largest':>9s} "
    f"{'runtime':>9s} "
    f"{'components':>11s}"
)

for key, r in results.items():

    comp = (
        f"{r['graph_components_mean']:.1f}"
        if "graph_components_mean" in r
        else "-"
    )

    print(
        f"{r['method']:20s} "
        f"{r['ari_mean']:8.3f} "
        f"{r['ami_mean']:8.3f} "
        f"{r['stability_mean']:10.3f} "
        f"{r['stability_min']:9.3f} "
        f"{r['largest_cluster_mean']*100:8.1f}% "
        f"{r['runtime_mean_s']:8.3f}s "
        f"{comp:>11s}"
    )


OUT.write_text(
    json.dumps(
        {
            "dataset": ds["name"],
            "shape": list(E.shape),
            "valid_pixels": int(V.sum()),
            "labelled_pixels": int(mask.sum()),
            "k": K,
            "kmeans_sample": max(5000, 500*K),
            "spectral_sample": S_SPECTRAL,
            "seeds": SEEDS,
            "neighbors": NEIGHBORS,
            "results": results,
        },
        indent=2,
    )
)

print()
print("Saved:", OUT)
