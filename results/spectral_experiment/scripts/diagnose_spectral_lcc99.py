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
TARGET_SHARE = 0.99
SEEDS = list(range(1001, 1011))

OUT = Path("results/spectral_lcc99_thresholds.json")
OUT.parent.mkdir(exist_ok=True)


def get_dataset(name):

    if name == "gothenburg":
        ds = tsh.gothenburg()
        return ds["emb"][ds["valid"]]

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


def component_stats(neighbours, nn):

    n = neighbours.shape[0]

    rows = np.repeat(
        np.arange(n),
        nn,
    )

    cols = neighbours[:, :nn].reshape(-1)

    data = np.ones(
        len(rows),
        dtype=np.uint8,
    )

    graph = csr_matrix(
        (data, (rows, cols)),
        shape=(n, n),
    )

    n_components, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    counts = np.bincount(labels)

    largest_share = counts.max() / n

    return n_components, float(largest_share)


def find_lcc99_threshold(X, seed):

    rng = np.random.default_rng(seed)

    idx = rng.choice(
        len(X),
        min(S, len(X)),
        replace=False,
    )

    Xs = X[idx].astype(np.float32)

    model = NearestNeighbors(
        n_neighbors=MAX_NN,
        metric="euclidean",
        n_jobs=-1,
    )

    model.fit(Xs)

    neighbours = model.kneighbors(
        X=None,
        return_distance=False,
    )

    # Check that MAX_NN is enough
    _, share_max = component_stats(
        neighbours,
        MAX_NN,
    )

    if share_max < TARGET_SHARE:
        return None

    # Monotonic binary search:
    # once >=99% is in the largest component,
    # adding neighbours cannot reduce that share.
    lo = 1
    hi = MAX_NN

    while lo < hi:

        mid = (lo + hi) // 2

        _, share = component_stats(
            neighbours,
            mid,
        )

        if share >= TARGET_SHARE:
            hi = mid
        else:
            lo = mid + 1

    threshold = lo

    ncomp, share = component_stats(
        neighbours,
        threshold,
    )

    if threshold > 1:
        prev_comp, prev_share = component_stats(
            neighbours,
            threshold - 1,
        )
    else:
        prev_comp = None
        prev_share = None

    return {
        "threshold": int(threshold),
        "components": int(ncomp),
        "largest_share": float(share),
        "previous_components": (
            int(prev_comp)
            if prev_comp is not None
            else None
        ),
        "previous_largest_share": (
            float(prev_share)
            if prev_share is not None
            else None
        ),
    }


datasets = [
    "austria",
    "west",
    "east",
    "gothenburg",
]

results = {}

print("=" * 85)
print("SPECTRAL 99% LARGEST-COMPONENT DIAGNOSTIC")
print("=" * 85)
print("Sample size:", S)
print("Target largest-component share:", TARGET_SHARE)
print("Seeds:", SEEDS)
print()


for name in datasets:

    X = get_dataset(name)

    print()
    print("#" * 85)
    print(name.upper())
    print("#" * 85)
    print("N:", len(X))

    thresholds = []
    per_seed = []

    for seed in SEEDS:

        r = find_lcc99_threshold(
            X,
            seed,
        )

        if r is None:

            print(
                f"seed={seed} | "
                f"99% NOT reached by nn={MAX_NN}"
            )

            per_seed.append({
                "seed": seed,
                "threshold": None,
            })

            continue

        thresholds.append(
            r["threshold"]
        )

        per_seed.append({
            "seed": seed,
            **r,
        })

        print(
            f"seed={seed} | "
            f"n99={r['threshold']:3d} | "
            f"LCC={r['largest_share']*100:6.2f}% | "
            f"components={r['components']:3d} | "
            f"previous LCC="
            f"{r['previous_largest_share']*100:6.2f}%"
        )

    a = np.asarray(
        thresholds,
        dtype=float,
    )

    summary = {
        "N": int(len(X)),
        "sample_size": min(S, len(X)),
        "target_share": TARGET_SHARE,
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)),
        "min": int(a.min()),
        "max": int(a.max()),
        "per_seed": per_seed,
    }

    results[name] = summary

    print()
    print("SUMMARY")
    print(
        f"n99: mean={summary['mean']:.2f}, "
        f"median={summary['median']:.1f}, "
        f"std={summary['std']:.2f}, "
        f"range={summary['min']}--{summary['max']}"
    )

    print("candidate values:")

    for alpha in [1.0, 1.5, 2.0, 2.5]:

        nn = int(
            np.ceil(
                alpha * summary["median"]
            )
        )

        print(
            f"  alpha={alpha:.1f}: nn={nn}"
        )


OUT.write_text(
    json.dumps(
        {
            "sample_size": S,
            "target_share": TARGET_SHARE,
            "max_nn": MAX_NN,
            "seeds": SEEDS,
            "results": results,
        },
        indent=2,
    )
)

print()
print("=" * 85)
print("Saved:", OUT)
print("=" * 85)
