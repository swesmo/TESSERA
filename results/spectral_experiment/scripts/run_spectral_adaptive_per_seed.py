#!/usr/bin/env python3

import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors

import tsh


ALPHA = 2.5
S = 5000
MAX_NN = 120
SEEDS = [42] + list(range(1, 10))

OUT = Path("results/spectral_adaptive_per_seed.json")
OUT.parent.mkdir(exist_ok=True)


# ============================================================
# DATA
# ============================================================

def prepare_dataset(name):

    if name == "gothenburg":

        ds = tsh.gothenburg()

        E = ds["emb"]
        V = ds["valid"]

        gt_grid = np.load(
            Path.home()
            / "tee_data/harness-cache/gbg_manual_gt.npy"
        )

        mask = (gt_grid > 0) & V
        gt = gt_grid[mask]

        return {
            "name": name,
            "E": E,
            "V": V,
            "X": E[V],
            "mask": mask,
            "gt": gt,
            "k": 5,
        }

    ds = tsh.austria(year=2022)

    if name == "austria":

        E = ds["emb"]
        V = ds["valid"]
        FID = ds["fid"]
        CLS = ds["cls"]

    elif name in ("west", "east"):

        half = ds["emb"].shape[1] // 2

        sl = (
            slice(0, half)
            if name == "west"
            else slice(half, None)
        )

        E = ds["emb"][:, sl]
        V = ds["valid"][:, sl]
        FID = ds["fid"][:, sl]
        CLS = ds["cls"][:, sl]

    else:
        raise ValueError(name)

    mask = (FID > 0) & V
    gt = CLS[mask]

    return {
        "name": name,
        "E": E,
        "V": V,
        "X": E[V],
        "mask": mask,
        "gt": gt,
        "k": 17,
    }


# ============================================================
# CONNECTIVITY
# ============================================================

def graph_components(neighbour_indices, nn):

    n = neighbour_indices.shape[0]

    rows = np.repeat(
        np.arange(n),
        nn,
    )

    cols = neighbour_indices[:, :nn].reshape(-1)

    data = np.ones(
        len(rows),
        dtype=np.uint8,
    )

    graph = csr_matrix(
        (data, (rows, cols)),
        shape=(n, n),
    )

    return connected_components(
        graph,
        directed=False,
        return_labels=False,
    )


def find_connectivity_threshold(Xs):

    # Build the neighbour structure once.
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

    # Safety check
    if graph_components(neighbours, MAX_NN) != 1:
        raise RuntimeError(
            f"Graph still disconnected at nn={MAX_NN}"
        )

    # Connectivity is monotonic, so use binary search.
    lo = 1
    hi = MAX_NN

    while lo < hi:

        mid = (lo + hi) // 2

        ncomp = graph_components(
            neighbours,
            mid,
        )

        if ncomp == 1:
            hi = mid
        else:
            lo = mid + 1

    return lo


# ============================================================
# METRICS
# ============================================================

def ground_truth_scores(dataset, labels):

    grid = np.full(
        dataset["V"].shape,
        -1,
        dtype=np.int16,
    )

    grid[dataset["V"]] = labels

    pred = grid[dataset["mask"]]

    ari = adjusted_rand_score(
        dataset["gt"],
        pred,
    )

    ami = adjusted_mutual_info_score(
        dataset["gt"],
        pred,
    )

    return float(ari), float(ami)


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
        "pairs": int(len(vals)),
    }


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(name):

    ds = prepare_dataset(name)

    X = ds["X"]
    k = ds["k"]

    print()
    print("#" * 90)
    print(name.upper())
    print("#" * 90)

    print("N:", len(X))
    print("k:", k)
    print("sample size:", S)
    print("alpha:", ALPHA)
    print()

    per_seed = []
    labels_by_seed = []

    for seed in SEEDS:

        total_start = time.perf_counter()

        # ----------------------------------------------------
        # 1. SAME SAMPLE used for connectivity and Spectral
        # ----------------------------------------------------

        rng = np.random.default_rng(seed)

        idx = rng.choice(
            len(X),
            min(S, len(X)),
            replace=False,
        )

        Xs = X[idx].astype(np.float32)


        # ----------------------------------------------------
        # 2. FIND CONNECTIVITY THRESHOLD
        # ----------------------------------------------------

        t0 = time.perf_counter()

        n_conn = find_connectivity_threshold(
            Xs
        )

        connectivity_time = (
            time.perf_counter() - t0
        )


        # ----------------------------------------------------
        # 3. ADAPTIVE n_neighbors
        # ----------------------------------------------------

        nn = int(
            np.ceil(
                ALPHA * n_conn
            )
        )

        nn = max(1, min(nn, len(Xs) - 1))


        # ----------------------------------------------------
        # 4. SPECTRAL FIT ON SAME SAMPLE
        # ----------------------------------------------------

        t0 = time.perf_counter()

        labels_sample = tsh.spectral_groups(
            Xs,
            k=k,
            seed=seed,
            n_neighbors=nn,
        )

        means = tsh.group_means(
            Xs,
            labels_sample,
            k,
        )

        spectral_fit_time = (
            time.perf_counter() - t0
        )


        # ----------------------------------------------------
        # 5. ASSIGN ALL PIXELS
        # ----------------------------------------------------

        t0 = time.perf_counter()

        labels = tsh.sqd(
            X,
            means,
        ).argmin(1)

        assignment_time = (
            time.perf_counter() - t0
        )

        total_time = (
            time.perf_counter() - total_start
        )


        # ----------------------------------------------------
        # 6. METRICS
        # ----------------------------------------------------

        ari, ami = ground_truth_scores(
            ds,
            labels,
        )

        shares = (
            np.bincount(
                labels,
                minlength=k,
            )
            / len(labels)
        )

        largest = float(
            shares.max()
        )

        labels_by_seed.append(
            labels.astype(np.int16)
        )

        result = {
            "seed": int(seed),

            "connectivity_nn":
                int(n_conn),

            "adaptive_nn":
                int(nn),

            "ari":
                ari,

            "ami":
                ami,

            "largest_cluster":
                largest,

            "connectivity_time_s":
                float(connectivity_time),

            "spectral_fit_time_s":
                float(spectral_fit_time),

            "assignment_time_s":
                float(assignment_time),

            "total_time_s":
                float(total_time),
        }

        per_seed.append(result)

        print(
            f"seed={seed:3d} | "
            f"n_conn={n_conn:2d} | "
            f"nn={nn:3d} | "
            f"ARI={ari:.3f} | "
            f"AMI={ami:.3f} | "
            f"largest={largest*100:5.1f}% | "
            f"conn={connectivity_time:.2f}s | "
            f"fit={spectral_fit_time:.2f}s | "
            f"assign={assignment_time:.2f}s | "
            f"total={total_time:.2f}s"
        )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    stab = stability(
        labels_by_seed
    )

    n_conn_values = np.array([
        r["connectivity_nn"]
        for r in per_seed
    ])

    nn_values = np.array([
        r["adaptive_nn"]
        for r in per_seed
    ])

    ari_values = np.array([
        r["ari"]
        for r in per_seed
    ])

    ami_values = np.array([
        r["ami"]
        for r in per_seed
    ])

    connectivity_times = np.array([
        r["connectivity_time_s"]
        for r in per_seed
    ])

    fit_times = np.array([
        r["spectral_fit_time_s"]
        for r in per_seed
    ])

    assignment_times = np.array([
        r["assignment_time_s"]
        for r in per_seed
    ])

    total_times = np.array([
        r["total_time_s"]
        for r in per_seed
    ])

    summary = {
        "k":
            k,

        "sample_size":
            S,

        "alpha":
            ALPHA,

        "connectivity_nn_mean":
            float(n_conn_values.mean()),

        "connectivity_nn_median":
            float(np.median(n_conn_values)),

        "connectivity_nn_min":
            int(n_conn_values.min()),

        "connectivity_nn_max":
            int(n_conn_values.max()),

        "adaptive_nn_mean":
            float(nn_values.mean()),

        "adaptive_nn_median":
            float(np.median(nn_values)),

        "adaptive_nn_min":
            int(nn_values.min()),

        "adaptive_nn_max":
            int(nn_values.max()),

        "ari_mean":
            float(ari_values.mean()),

        "ari_std":
            float(ari_values.std(ddof=1)),

        "ami_mean":
            float(ami_values.mean()),

        "ami_std":
            float(ami_values.std(ddof=1)),

        "stability_mean":
            stab["mean"],

        "stability_std":
            stab["std"],

        "stability_min":
            stab["min"],

        "connectivity_time_mean_s":
            float(connectivity_times.mean()),

        "spectral_fit_time_mean_s":
            float(fit_times.mean()),

        "assignment_time_mean_s":
            float(assignment_times.mean()),

        "total_time_mean_s":
            float(total_times.mean()),

        "per_seed":
            per_seed,
    }

    print()
    print("-" * 90)
    print("SUMMARY")
    print("-" * 90)

    print(
        f"n_conn:    "
        f"{summary['connectivity_nn_mean']:.1f} "
        f"(range "
        f"{summary['connectivity_nn_min']}--"
        f"{summary['connectivity_nn_max']})"
    )

    print(
        f"adaptive nn:"
        f" {summary['adaptive_nn_mean']:.1f} "
        f"(range "
        f"{summary['adaptive_nn_min']}--"
        f"{summary['adaptive_nn_max']})"
    )

    print(
        f"ARI:       "
        f"{summary['ari_mean']:.3f} "
        f"+- {summary['ari_std']:.3f}"
    )

    print(
        f"AMI:       "
        f"{summary['ami_mean']:.3f} "
        f"+- {summary['ami_std']:.3f}"
    )

    print(
        f"Stability: "
        f"{summary['stability_mean']:.3f} "
        f"+- {summary['stability_std']:.3f}"
    )

    print(
        f"Stab min:  "
        f"{summary['stability_min']:.3f}"
    )

    print(
        f"Connectivity time: "
        f"{summary['connectivity_time_mean_s']:.3f}s"
    )

    print(
        f"Spectral fit time: "
        f"{summary['spectral_fit_time_mean_s']:.3f}s"
    )

    print(
        f"Assignment time:   "
        f"{summary['assignment_time_mean_s']:.3f}s"
    )

    print(
        f"TOTAL time:        "
        f"{summary['total_time_mean_s']:.3f}s"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

results = {}

for name in [
    "austria",
    "west",
    "east",
    "gothenburg",
]:

    results[name] = run_dataset(
        name
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 115)
print("FINAL PER-SEED ADAPTIVE SUMMARY")
print("=" * 115)

print(
    f"{'dataset':12s} "
    f"{'conn':>8s} "
    f"{'nn':>8s} "
    f"{'ARI':>8s} "
    f"{'AMI':>8s} "
    f"{'stab':>8s} "
    f"{'conn t':>9s} "
    f"{'fit t':>9s} "
    f"{'total':>9s}"
)

for name, r in results.items():

    print(
        f"{name:12s} "
        f"{r['connectivity_nn_mean']:8.1f} "
        f"{r['adaptive_nn_mean']:8.1f} "
        f"{r['ari_mean']:8.3f} "
        f"{r['ami_mean']:8.3f} "
        f"{r['stability_mean']:8.3f} "
        f"{r['connectivity_time_mean_s']:8.3f}s "
        f"{r['spectral_fit_time_mean_s']:8.3f}s "
        f"{r['total_time_mean_s']:8.3f}s"
    )


OUT.write_text(
    json.dumps(
        {
            "alpha": ALPHA,
            "sample_size": S,
            "max_nn": MAX_NN,
            "seeds": SEEDS,
            "results": results,
        },
        indent=2,
    )
)

print()
print("Saved:", OUT)
