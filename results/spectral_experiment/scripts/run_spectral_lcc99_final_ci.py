#!/usr/bin/env python3

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import t
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
)

import tsh


S = 5000
SEEDS = [42] + list(range(1, 10))

FIXED_NN = {
    "austria": 8,
    "west": 10,
    "east": 10,
    "gothenburg": 45,
}

BASELINE_FILES = {
    "austria":
        "results/aut2022_kmeans_vs_spectral_nn.json",

    "west":
        "results/aut2022_spectral_validation_west.json",

    "east":
        "results/aut2022_spectral_validation_east.json",

    "gothenburg":
        "results/gbg_kmeans_vs_spectral_gt.json",
}

OUT = Path(
    "results/spectral_lcc99_final_with_ci.json"
)


# ============================================================
# DATA
# ============================================================

def prepare(name):

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

        return E, V, mask, gt, 5


    ds = tsh.austria(year=2022)

    if name == "austria":

        E = ds["emb"]
        V = ds["valid"]
        FID = ds["fid"]
        CLS = ds["cls"]

    else:

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

    mask = (FID > 0) & V
    gt = CLS[mask]

    return E, V, mask, gt, 17


# ============================================================
# STATISTICS
# ============================================================

def paired_ci(diff):

    diff = np.asarray(
        diff,
        dtype=float,
    )

    n = len(diff)

    mean = float(
        diff.mean()
    )

    sd = float(
        diff.std(ddof=1)
    )

    se = sd / np.sqrt(n)

    crit = t.ppf(
        0.975,
        n - 1,
    )

    lo = mean - crit * se
    hi = mean + crit * se

    return {
        "mean_difference": mean,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def stability(runs):

    vals = []

    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):

            vals.append(
                adjusted_rand_score(
                    runs[i],
                    runs[j],
                )
            )

    vals = np.asarray(vals)

    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "min": float(vals.min()),
    }


# ============================================================
# EXPERIMENT
# ============================================================

results = {}

for name in [
    "austria",
    "west",
    "east",
    "gothenburg",
]:

    E, V, mask, gt, k = prepare(name)

    X = E[V]

    nn = FIXED_NN[name]

    print()
    print("=" * 90)
    print(
        f"{name.upper()} | FINAL LCC99 nn={nn}"
    )
    print("=" * 90)

    spectral_runs = []
    spectral_per_seed = []

    for seed in SEEDS:

        t0 = time.perf_counter()

        idx, lab_s = tsh.spectral_sample(
            X,
            k=k,
            seed=seed,
            S=S,
            n_neighbors=nn,
        )

        means = tsh.group_means(
            X[idx],
            lab_s,
            k,
        )

        labels = tsh.sqd(
            X,
            means,
        ).argmin(1)

        runtime = (
            time.perf_counter() - t0
        )

        grid = np.full(
            V.shape,
            -1,
            dtype=np.int16,
        )

        grid[V] = labels

        pred = grid[mask]

        ari = float(
            adjusted_rand_score(
                gt,
                pred,
            )
        )

        ami = float(
            adjusted_mutual_info_score(
                gt,
                pred,
            )
        )

        spectral_runs.append(
            labels.astype(np.int16)
        )

        spectral_per_seed.append({
            "seed": int(seed),
            "ari": ari,
            "ami": ami,
            "runtime_s": float(runtime),
        })

        print(
            f"seed={seed:3d} | "
            f"ARI={ari:.4f} | "
            f"AMI={ami:.4f} | "
            f"time={runtime:.3f}s"
        )


    # --------------------------------------------------------
    # LOAD EXISTING BASELINE
    # --------------------------------------------------------

    baseline_json = json.load(
        open(BASELINE_FILES[name])
    )

    baseline_per_seed = (
        baseline_json["results"]
        ["kmeans"]
        ["per_seed"]
    )

    # Make sure seeds match
    baseline_by_seed = {
        int(r["seed"]): r
        for r in baseline_per_seed
    }

    spectral_by_seed = {
        int(r["seed"]): r
        for r in spectral_per_seed
    }


    baseline_ari = np.array([
        baseline_by_seed[s]["ari"]
        for s in SEEDS
    ])

    baseline_ami = np.array([
        baseline_by_seed[s]["ami"]
        for s in SEEDS
    ])

    spectral_ari = np.array([
        spectral_by_seed[s]["ari"]
        for s in SEEDS
    ])

    spectral_ami = np.array([
        spectral_by_seed[s]["ami"]
        for s in SEEDS
    ])


    # --------------------------------------------------------
    # PAIRED DIFFERENCES
    # Spectral - baseline
    # --------------------------------------------------------

    ari_diff = (
        spectral_ari
        - baseline_ari
    )

    ami_diff = (
        spectral_ami
        - baseline_ami
    )

    ari_ci = paired_ci(
        ari_diff
    )

    ami_ci = paired_ci(
        ami_diff
    )

    stab = stability(
        spectral_runs
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "nn":
            nn,

        "baseline_ari_mean":
            float(baseline_ari.mean()),

        "spectral_ari_mean":
            float(spectral_ari.mean()),

        "ari_difference":
            ari_ci,

        "baseline_ami_mean":
            float(baseline_ami.mean()),

        "spectral_ami_mean":
            float(spectral_ami.mean()),

        "ami_difference":
            ami_ci,

        "spectral_stability_mean":
            stab["mean"],

        "spectral_stability_std":
            stab["std"],

        "spectral_stability_min":
            stab["min"],

        "spectral_runtime_mean_s":
            float(
                np.mean([
                    r["runtime_s"]
                    for r in spectral_per_seed
                ])
            ),

        "spectral_per_seed":
            spectral_per_seed,
    }

    results[name] = summary


    print()
    print("-" * 90)
    print("PAIRED RESULT")
    print("-" * 90)

    print(
        f"ARI: baseline "
        f"{summary['baseline_ari_mean']:.4f} "
        f"-> spectral "
        f"{summary['spectral_ari_mean']:.4f}"
    )

    print(
        f"ΔARI = "
        f"{ari_ci['mean_difference']:+.4f} "
        f"["
        f"{ari_ci['ci95_low']:+.4f}, "
        f"{ari_ci['ci95_high']:+.4f}"
        f"]"
    )

    print()

    print(
        f"AMI: baseline "
        f"{summary['baseline_ami_mean']:.4f} "
        f"-> spectral "
        f"{summary['spectral_ami_mean']:.4f}"
    )

    print(
        f"ΔAMI = "
        f"{ami_ci['mean_difference']:+.4f} "
        f"["
        f"{ami_ci['ci95_low']:+.4f}, "
        f"{ami_ci['ci95_high']:+.4f}"
        f"]"
    )

    print()

    print(
        f"Spectral stability: "
        f"{stab['mean']:.3f} "
        f"+- {stab['std']:.3f}"
    )

    print(
        f"Spectral stability min: "
        f"{stab['min']:.3f}"
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 115)
print("FINAL LCC99 PAIRED SUMMARY")
print("=" * 115)

print(
    f"{'dataset':12s} "
    f"{'nn':>4s} "
    f"{'ARI base':>9s} "
    f"{'ARI spec':>9s} "
    f"{'ΔARI':>9s} "
    f"{'AMI base':>9s} "
    f"{'AMI spec':>9s} "
    f"{'ΔAMI':>9s} "
    f"{'stab':>7s}"
)

for name, r in results.items():

    print(
        f"{name:12s} "
        f"{r['nn']:4d} "
        f"{r['baseline_ari_mean']:9.3f} "
        f"{r['spectral_ari_mean']:9.3f} "
        f"{r['ari_difference']['mean_difference']:+9.3f} "
        f"{r['baseline_ami_mean']:9.3f} "
        f"{r['spectral_ami_mean']:9.3f} "
        f"{r['ami_difference']['mean_difference']:+9.3f} "
        f"{r['spectral_stability_mean']:7.3f}"
    )


OUT.write_text(
    json.dumps(
        {
            "selection_rule":
                "nn = ceil(2.5 * median n99); "
                "n99 is the smallest neighbourhood for which "
                "the largest connected component contains >=99% "
                "of the 5000-point calibration sample; "
                "calibration seeds 1001--1010",

            "sample_size":
                S,

            "evaluation_seeds":
                SEEDS,

            "fixed_nn":
                FIXED_NN,

            "results":
                results,
        },
        indent=2,
    )
)

print()
print("Saved:", OUT)
