#!/usr/bin/env python3

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.neighbors import kneighbors_graph

import tsh


OUT = Path(__file__).resolve().parent / "results"

K = 5
S = 5000

NEIGHBORS = [
    5,
    10,
    15,
    20,
    30,
    40,
    60,
    120,
]

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10

# Same convention used in Christopher's experiments:
# seed 42 + seeds 1...(n_seeds-1)
SEEDS = [42] + list(range(1, n_seeds))


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

ds = tsh.gothenburg()

E = ds["emb"]
V = ds["valid"]

H, W, _ = E.shape

X = E[V]


def grid(labels):
    g = np.zeros((H, W), np.int64)
    g[V] = labels
    return g


# ------------------------------------------------------------
# OPTIONAL GOTHENBURG GROUND TRUTH
# ------------------------------------------------------------

HAS_LABELS = False

if os.environ.get("GBG_LABELS"):

    metadata = (
        Path.home()
        / "tee_data/vectors/Gteborg/2024/metadata.json"
    )

    gt = json.loads(
        metadata.read_text()
    )["geotransform"]

    ds["cls"], ds["fid"], _ = tsh.load_geojson_labels(
        os.environ["GBG_LABELS"],
        V.shape,
        gt,
    )

    HAS_LABELS = True


print("=" * 80)
print("SPECTRAL n_neighbors SWEEP")
print("=" * 80)

print("Dataset:", ds["name"])
print("Shape:", E.shape)
print("Valid pixels:", len(X))
print("k:", K)
print("Sample size:", S)
print("n_neighbors:", NEIGHBORS)
print("Seeds:", SEEDS)
print("Ground truth:", HAS_LABELS)

print()


# ------------------------------------------------------------
# BENCHMARK
# ------------------------------------------------------------

results = {}

warnings.filterwarnings(
    "ignore",
    message="Graph is not fully connected",
)


for nn in NEIGHBORS:

    print()
    print("#" * 80)
    print(f"n_neighbors = {nn}")
    print("#" * 80)

    labels_by_seed = []

    components = []
    runtimes = []
    largest_shares = []
    cluster_shares = []

    ari_gt = []
    ami_gt = []
    speckle = []


    for seed in SEEDS:

        t0 = time.perf_counter()

        # ----------------------------------------------------
        # Spectral clustering on sample
        # ----------------------------------------------------

        idx, labels_sample = tsh.spectral_sample(
            X,
            k=K,
            seed=seed,
            S=S,
            n_neighbors=nn,
        )


        # ----------------------------------------------------
        # Convert spectral clusters to 128-D cluster means
        # ----------------------------------------------------

        means = tsh.group_means(
            X[idx],
            labels_sample,
            int(labels_sample.max()) + 1,
        )


        # ----------------------------------------------------
        # Assign all viewport pixels to nearest cluster mean
        # ----------------------------------------------------

        labels_all = tsh.sqd(
            X,
            means,
        ).argmin(1)


        runtime = time.perf_counter() - t0


        # ----------------------------------------------------
        # Connectivity of sampled k-NN graph
        # ----------------------------------------------------

        graph = kneighbors_graph(
            X[idx],
            nn,
            include_self=False,
        )

        n_components = connected_components(
            graph,
            directed=False,
        )[0]


        # ----------------------------------------------------
        # Cluster balance
        # ----------------------------------------------------

        shares = (
            np.bincount(
                labels_all,
                minlength=K,
            )
            / len(labels_all)
        )


        labels_by_seed.append(labels_all)

        components.append(n_components)
        runtimes.append(runtime)
        cluster_shares.append(shares.tolist())
        largest_shares.append(float(shares.max()))


        # ----------------------------------------------------
        # Ground-truth metrics, if labels are available
        # ----------------------------------------------------

        if HAS_LABELS:

            g = grid(labels_all)

            mask = ds["fid"] > 0

            ari_gt.append(
                float(
                    tsh.ari(
                        ds["cls"][mask],
                        g[mask],
                    )
                )
            )

            ami_gt.append(
                float(
                    adjusted_mutual_info_score(
                        ds["cls"][mask],
                        g[mask],
                    )
                )
            )

            speckle.append(
                float(
                    tsh.edge_adherence(
                        g,
                        ds["fid"],
                    )["d_in"]
                )
            )


        print(
            f"seed={seed:3d} | "
            f"components={n_components:3d} | "
            f"largest={shares.max()*100:6.2f}% | "
            f"time={runtime:5.2f}s"
        )


    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    pair_matrix = tsh.pair_matrix(
        labels_by_seed
    )

    stability_values = pair_matrix[
        np.triu_indices(
            len(SEEDS),
            k=1,
        )
    ]

    result = {
        "n_neighbors": nn,
        "k": K,
        "sample_size": S,
        "seeds": SEEDS,

        "stability_mean":
            float(np.mean(stability_values)),

        "stability_std":
            float(np.std(stability_values, ddof=1)),

        "stability_min":
            float(np.min(stability_values)),

        "graph_components_mean":
            float(np.mean(components)),

        "graph_components_max":
            int(np.max(components)),

        "largest_cluster_mean":
            float(np.mean(largest_shares)),

        "largest_cluster_max":
            float(np.max(largest_shares)),

        "runtime_mean_s":
            float(np.mean(runtimes)),

        "runtime_median_s":
            float(np.median(runtimes)),

        "cluster_shares":
            cluster_shares,
    }


    if HAS_LABELS:

        result.update({
            "ari_mean":
                float(np.mean(ari_gt)),

            "ari_std":
                float(np.std(ari_gt, ddof=1)),

            "ami_mean":
                float(np.mean(ami_gt)),

            "ami_std":
                float(np.std(ami_gt, ddof=1)),

            "speckle_mean":
                float(np.mean(speckle)),
        })


    results[str(nn)] = result


    print()
    print(
        f"SUMMARY nn={nn}"
    )

    print(
        f"  stability: "
        f"{result['stability_mean']:.3f} "
        f"+- {result['stability_std']:.3f}"
    )

    print(
        f"  graph components: "
        f"{result['graph_components_mean']:.1f}"
    )

    print(
        f"  largest cluster: "
        f"{result['largest_cluster_mean']*100:.1f}%"
    )

    print(
        f"  runtime: "
        f"{result['runtime_mean_s']:.2f}s"
    )

    if HAS_LABELS:

        print(
            f"  ARI: "
            f"{result['ari_mean']:.3f}"
        )

        print(
            f"  AMI: "
            f"{result['ami_mean']:.3f}"
        )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

output = OUT / "spectral_nn_sweep_gbg.json"

output.write_text(
    json.dumps(
        {
            "dataset": ds["name"],
            "shape": list(E.shape),
            "valid_pixels": int(len(X)),
            "k": K,
            "sample_size": S,
            "neighbors": NEIGHBORS,
            "seeds": SEEDS,
            "ground_truth": HAS_LABELS,
            "results": results,
        },
        indent=2,
    )
)

print()
print("=" * 80)
print("FINISHED")
print("=" * 80)
print("Results:", output)
