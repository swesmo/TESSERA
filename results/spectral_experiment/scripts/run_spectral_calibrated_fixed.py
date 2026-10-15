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

CALIBRATION_SEEDS = [1001, 1002, 1003]
EVALUATION_SEEDS = [42] + list(range(1, 10))

OUT = Path("results/spectral_calibrated_fixed.json")
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


def connectivity_threshold(X, seed):

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

    if graph_components(neighbours, MAX_NN) != 1:
        raise RuntimeError(
            f"Graph disconnected at nn={MAX_NN}"
        )

    lo = 1
    hi = MAX_NN

    while lo < hi:

        mid = (lo + hi) // 2

        if graph_components(neighbours, mid) == 1:
            hi = mid
        else:
            lo = mid + 1

    return lo


# ============================================================
# METRICS
# ============================================================

def gt_scores(ds, labels):

    grid = np.full(
        ds["V"].shape,
        -1,
        dtype=np.int16,
    )

    grid[ds["V"]] = labels

    pred = grid[ds["mask"]]

    return (
        float(
            adjusted_rand_score(
                ds["gt"],
                pred,
            )
        ),
        float(
            adjusted_mutual_info_score(
                ds["gt"],
                pred,
            )
        ),
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
        "pairs": int(len(vals)),
    }


# ============================================================
# ONE DATASET
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
    print("calibration seeds:", CALIBRATION_SEEDS)
    print("evaluation seeds:", EVALUATION_SEEDS)


    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    print()
    print("CALIBRATION")

    calibration_start = time.perf_counter()

    thresholds = []

    for seed in CALIBRATION_SEEDS:

        t0 = time.perf_counter()

        n_conn = connectivity_threshold(
            X,
            seed,
        )

        dt = time.perf_counter() - t0

        thresholds.append(n_conn)

        print(
            f"seed={seed} | "
            f"n_conn={n_conn} | "
            f"time={dt:.3f}s"
        )

    calibration_time = (
        time.perf_counter() - calibration_start
    )

    median_conn = float(
        np.median(thresholds)
    )

    fixed_nn = int(
        np.ceil(
            ALPHA * median_conn
        )
    )

    print()
    print(
        f"calibration thresholds: {thresholds}"
    )

    print(
        f"median n_conn: {median_conn:.1f}"
    )

    print(
        f"FIXED adaptive nn: {fixed_nn}"
    )

    print(
        f"one-time calibration cost: "
        f"{calibration_time:.3f}s"
    )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print()
    print("EVALUATION")

    labels_runs = []
    per_seed = []

    for seed in EVALUATION_SEEDS:

        total_start = time.perf_counter()

        # Spectral sample + fit
        fit_start = time.perf_counter()

        idx, labels_sample = tsh.spectral_sample(
            X,
            k=k,
            seed=seed,
            S=S,
            n_neighbors=fixed_nn,
        )

        means = tsh.group_means(
            X[idx],
            labels_sample,
            k,
        )

        fit_time = (
            time.perf_counter() - fit_start
        )

        # Assignment
        assign_start = time.perf_counter()

        labels = tsh.sqd(
            X,
            means,
        ).argmin(1)

        assignment_time = (
            time.perf_counter() - assign_start
        )

        total_time = (
            time.perf_counter() - total_start
        )

        ari, ami = gt_scores(
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

        labels_runs.append(
            labels.astype(np.int16)
        )

        per_seed.append({
            "seed":
                int(seed),

            "ari":
                ari,

            "ami":
                ami,

            "largest_cluster":
                largest,

            "fit_time_s":
                float(fit_time),

            "assignment_time_s":
                float(assignment_time),

            "total_time_s":
                float(total_time),
        })

        print(
            f"seed={seed:3d} | "
            f"ARI={ari:.3f} | "
            f"AMI={ami:.3f} | "
            f"largest={largest*100:5.1f}% | "
            f"fit={fit_time:.2f}s | "
            f"assign={assignment_time:.2f}s | "
            f"total={total_time:.2f}s"
        )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    aris = np.array([
        r["ari"] for r in per_seed
    ])

    amis = np.array([
        r["ami"] for r in per_seed
    ])

    fit_times = np.array([
        r["fit_time_s"] for r in per_seed
    ])

    assignment_times = np.array([
        r["assignment_time_s"]
        for r in per_seed
    ])

    total_times = np.array([
        r["total_time_s"] for r in per_seed
    ])

    largest = np.array([
        r["largest_cluster"]
        for r in per_seed
    ])

    stab = stability(
        labels_runs
    )

    summary = {
        "k":
            k,

        "sample_size":
            S,

        "alpha":
            ALPHA,

        "calibration_seeds":
            CALIBRATION_SEEDS,

        "calibration_thresholds":
            thresholds,

        "median_connectivity_nn":
            median_conn,

        "fixed_adaptive_nn":
            fixed_nn,

        "calibration_time_s":
            float(calibration_time),

        "ari_mean":
            float(aris.mean()),

        "ari_std":
            float(aris.std(ddof=1)),

        "ami_mean":
            float(amis.mean()),

        "ami_std":
            float(amis.std(ddof=1)),

        "stability_mean":
            stab["mean"],

        "stability_std":
            stab["std"],

        "stability_min":
            stab["min"],

        "largest_cluster_mean":
            float(largest.mean()),

        "fit_time_mean_s":
            float(fit_times.mean()),

        "assignment_time_mean_s":
            float(assignment_times.mean()),

        "run_time_mean_s":
            float(total_times.mean()),

        "per_seed":
            per_seed,
    }

    print()
    print("-" * 90)
    print("SUMMARY")
    print("-" * 90)

    print(
        f"calibration n_conn: "
        f"{thresholds}"
    )

    print(
        f"median n_conn:       "
        f"{median_conn:.1f}"
    )

    print(
        f"fixed adaptive nn:   "
        f"{fixed_nn}"
    )

    print(
        f"ARI:                 "
        f"{summary['ari_mean']:.3f} "
        f"+- {summary['ari_std']:.3f}"
    )

    print(
        f"AMI:                 "
        f"{summary['ami_mean']:.3f} "
        f"+- {summary['ami_std']:.3f}"
    )

    print(
        f"Stability:           "
        f"{summary['stability_mean']:.3f} "
        f"+- {summary['stability_std']:.3f}"
    )

    print(
        f"Stability min:       "
        f"{summary['stability_min']:.3f}"
    )

    print(
        f"Calibration cost:    "
        f"{calibration_time:.3f}s once"
    )

    print(
        f"Fit time/run:        "
        f"{summary['fit_time_mean_s']:.3f}s"
    )

    print(
        f"Assignment/run:      "
        f"{summary['assignment_time_mean_s']:.3f}s"
    )

    print(
        f"Runtime/run:         "
        f"{summary['run_time_mean_s']:.3f}s"
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

    results[name] = run_dataset(name)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 120)
print("FINAL CALIBRATED-FIXED SPECTRAL SUMMARY")
print("=" * 120)

print(
    f"{'dataset':12s} "
    f"{'calib':>12s} "
    f"{'nn':>5s} "
    f"{'ARI':>7s} "
    f"{'AMI':>7s} "
    f"{'stab':>7s} "
    f"{'calib t':>9s} "
    f"{'run t':>9s}"
)

for name, r in results.items():

    thresholds = ",".join(
        str(x)
        for x in r["calibration_thresholds"]
    )

    print(
        f"{name:12s} "
        f"{thresholds:>12s} "
        f"{r['fixed_adaptive_nn']:5d} "
        f"{r['ari_mean']:7.3f} "
        f"{r['ami_mean']:7.3f} "
        f"{r['stability_mean']:7.3f} "
        f"{r['calibration_time_s']:8.3f}s "
        f"{r['run_time_mean_s']:8.3f}s"
    )


OUT.write_text(
    json.dumps(
        {
            "alpha":
                ALPHA,

            "sample_size":
                S,

            "calibration_seeds":
                CALIBRATION_SEEDS,

            "evaluation_seeds":
                EVALUATION_SEEDS,

            "results":
                results,
        },
        indent=2,
    )
)

print()
print("Saved:", OUT)
