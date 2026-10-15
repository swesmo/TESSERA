#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

import tsh


S = 5000
MAX_NN = 120
SEEDS = list(range(1001, 1011))

OUT = Path("results/spectral_connectivity_thresholds.json")
OUT.parent.mkdir(exist_ok=True)


def get_dataset(name):
    if name == "gothenburg":
        ds = tsh.gothenburg()
        X = ds["emb"][ds["valid"]]
        return X

    ds = tsh.austria(year=2022)

    if name == "austria":
        E = ds["emb"]
        V = ds["valid"]

    elif name in ("west", "east"):
        half = ds["emb"].shape[1] // 2

        sl = (
            slice(0, half)
            if name == "west"
            else slice(half, None)
        )

        E = ds["emb"][:, sl]
        V = ds["valid"][:, sl]

    else:
        raise ValueError(name)

    return E[V]


def graph_components(neighbor_indices, n_neighbors):
    n = neighbor_indices.shape[0]

    rows = np.repeat(
        np.arange(n),
        n_neighbors,
    )

    cols = neighbor_indices[:, :n_neighbors].reshape(-1)

    data = np.ones(len(rows), dtype=np.uint8)

    graph = csr_matrix(
        (data, (rows, cols)),
        shape=(n, n),
    )

    return connected_components(
        graph,
        directed=False,
        return_labels=False,
    )


def connectivity_threshold(X, seed):
    rng = np.random.default_rng(seed)

    size = min(S, len(X))

    idx = rng.choice(
        len(X),
        size,
        replace=False,
    )

    Xs = X[idx]

    # Compute the first MAX_NN nearest neighbours only once.
    model = NearestNeighbors(
        n_neighbors=MAX_NN,
        metric="euclidean",
        n_jobs=-1,
    )

    model.fit(Xs)

    # X=None means the point itself is excluded from its neighbours.
    neighbors = model.kneighbors(
        X=None,
        return_distance=False,
    )

    # Check that our search range is sufficient.
    comp_max = graph_components(
        neighbors,
        MAX_NN,
    )

    if comp_max != 1:
        return None, comp_max

    # Connectivity is monotonic:
    # once connected, increasing n_neighbors cannot disconnect it.
    lo = 1
    hi = MAX_NN

    while lo < hi:
        mid = (lo + hi) // 2

        c = graph_components(
            neighbors,
            mid,
        )

        if c == 1:
            hi = mid
        else:
            lo = mid + 1

    threshold = lo

    before = (
        graph_components(neighbors, threshold - 1)
        if threshold > 1
        else None
    )

    return threshold, before


datasets = [
    "austria",
    "west",
    "east",
    "gothenburg",
]

results = {}

print("=" * 80)
print("SPECTRAL GRAPH CONNECTIVITY DIAGNOSTIC")
print("=" * 80)
print("Sample size:", S)
print("Seeds:", SEEDS)
print("Maximum n_neighbors tested:", MAX_NN)
print()


for name in datasets:

    X = get_dataset(name)

    print()
    print("#" * 80)
    print(name.upper())
    print("#" * 80)
    print("N:", len(X))

    thresholds = []

    per_seed = []

    for seed in SEEDS:

        threshold, previous_components = connectivity_threshold(
            X,
            seed,
        )

        per_seed.append({
            "seed": seed,
            "minimum_connected_nn": threshold,
            "components_at_previous_nn": previous_components,
        })

        if threshold is not None:
            thresholds.append(threshold)

            print(
                f"seed={seed:3d} | "
                f"minimum connected nn={threshold:3d} | "
                f"components at nn={threshold-1:3d}: "
                f"{previous_components}"
            )
        else:
            print(
                f"seed={seed:3d} | "
                f"NOT connected by nn={MAX_NN}"
            )

    thresholds_np = np.asarray(
        thresholds,
        dtype=float,
    )

    summary = {
        "N": int(len(X)),
        "sample_size": min(S, len(X)),
        "per_seed": per_seed,
        "mean": float(thresholds_np.mean()),
        "median": float(np.median(thresholds_np)),
        "min": int(thresholds_np.min()),
        "max": int(thresholds_np.max()),
    }

    results[name] = summary

    print()
    print("SUMMARY")
    print(
        f"minimum connected nn: "
        f"mean={summary['mean']:.2f}, "
        f"median={summary['median']:.1f}, "
        f"range={summary['min']}--{summary['max']}"
    )

    # Show what simple adaptive multipliers would produce.
    print("candidate adaptive values from median:")

    for alpha in [1.0, 1.5, 2.0]:
        candidate = int(
            np.ceil(alpha * summary["median"])
        )

        print(
            f"  alpha={alpha:.1f}: "
            f"nn={candidate}"
        )


OUT.write_text(
    json.dumps(
        {
            "sample_size": S,
            "max_nn": MAX_NN,
            "seeds": SEEDS,
            "results": results,
        },
        indent=2,
    )
)

print()
print("=" * 80)
print("Saved:", OUT)
print("=" * 80)
