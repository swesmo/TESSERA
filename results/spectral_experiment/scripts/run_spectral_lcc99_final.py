#!/usr/bin/env python3

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import tsh


S = 5000
SEEDS = [42] + list(range(1, 10))

# Selected without using evaluation labels:
# nn = ceil(2.5 * median n99)
# n99 estimated from independent calibration seeds 1001--1010
FIXED_NN = {
    "austria": 8,
    "west": 10,
    "east": 10,
    "gothenburg": 45,
}

OUT = Path("results/spectral_lcc99_final.json")


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

    return (
        float(vals.mean()),
        float(vals.std(ddof=1)),
        float(vals.min()),
    )


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
    print("=" * 80)
    print(f"{name.upper()} | nn={nn}")
    print("=" * 80)

    print("N:", len(X))
    print("k:", k)
    print("sample size:", S)

    runs = []
    aris = []
    amis = []
    times = []

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

        dt = time.perf_counter() - t0

        grid = np.full(
            V.shape,
            -1,
            dtype=np.int16,
        )

        grid[V] = labels

        pred = grid[mask]

        ari = adjusted_rand_score(
            gt,
            pred,
        )

        ami = adjusted_mutual_info_score(
            gt,
            pred,
        )

        runs.append(
            labels.astype(np.int16)
        )

        aris.append(ari)
        amis.append(ami)
        times.append(dt)

        print(
            f"seed={seed:3d} | "
            f"ARI={ari:.3f} | "
            f"AMI={ami:.3f} | "
            f"time={dt:.3f}s"
        )

    stab, stab_sd, stab_min = stability(runs)

    result = {
        "nn": nn,

        "ari_mean":
            float(np.mean(aris)),

        "ari_std":
            float(np.std(aris, ddof=1)),

        "ami_mean":
            float(np.mean(amis)),

        "ami_std":
            float(np.std(amis, ddof=1)),

        "stability_mean":
            stab,

        "stability_std":
            stab_sd,

        "stability_min":
            stab_min,

        "runtime_mean_s":
            float(np.mean(times)),
    }

    results[name] = result

    print()
    print("SUMMARY")
    print(
        f"nn={nn} | "
        f"ARI={result['ari_mean']:.3f} +- {result['ari_std']:.3f} | "
        f"AMI={result['ami_mean']:.3f} +- {result['ami_std']:.3f} | "
        f"stability={result['stability_mean']:.3f} +- {result['stability_std']:.3f} | "
        f"stab min={result['stability_min']:.3f} | "
        f"runtime={result['runtime_mean_s']:.3f}s"
    )


OUT.write_text(
    json.dumps(
        {
            "selection_rule":
                "nn = ceil(2.5 * median n99); "
                "n99 calibrated using seeds 1001--1010",

            "sample_size":
                S,

            "fixed_nn":
                FIXED_NN,

            "evaluation_seeds":
                SEEDS,

            "results":
                results,
        },
        indent=2,
    )
)


print()
print("=" * 90)
print("FINAL SUMMARY")
print("=" * 90)

for name, r in results.items():

    print(
        f"{name:12s} "
        f"nn={r['nn']:2d} | "
        f"ARI={r['ari_mean']:.3f} | "
        f"AMI={r['ami_mean']:.3f} | "
        f"stab={r['stability_mean']:.3f} | "
        f"time={r['runtime_mean_s']:.3f}s"
    )

print()
print("Saved:", OUT)
