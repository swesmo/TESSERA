#!/usr/bin/env python3

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import tsh


ALPHA = 2.5
S = 5000
SEEDS = [42] + list(range(1, 10))

CONNECTIVITY_FILE = Path(
    "results/spectral_connectivity_thresholds.json"
)

OUT = Path(
    "results/spectral_adaptive_alpha2_5.json"
)

conn = json.load(open(CONNECTIVITY_FILE))


def stability(labels):
    vals = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            vals.append(
                adjusted_rand_score(
                    labels[i],
                    labels[j],
                )
            )

    vals = np.asarray(vals)

    return (
        float(vals.mean()),
        float(vals.std(ddof=1)),
        float(vals.min()),
    )


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

        return E, V, mask, gt

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

    return E, V, mask, gt


results = {}

print("=" * 85)
print("ADAPTIVE SPECTRAL CLUSTERING TEST")
print("=" * 85)
print("alpha:", ALPHA)
print()


for name in [
    "austria",
    "west",
    "east",
    "gothenburg",
]:

    median_conn = conn["results"][name]["median"]

    nn = int(
        np.ceil(
            ALPHA * median_conn
        )
    )

    E, V, mask, gt = prepare(name)

    X = E[V]

    k = (
        5
        if name == "gothenburg"
        else 17
    )

    print()
    print("#" * 85)
    print(name.upper())
    print("#" * 85)

    print(
        f"median n_conn={median_conn:.1f} | "
        f"alpha={ALPHA} | "
        f"adaptive nn={nn}"
    )

    labels_runs = []
    ari_values = []
    ami_values = []
    runtimes = []
    largest = []

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

        shares = (
            np.bincount(
                labels,
                minlength=k,
            )
            / len(labels)
        )

        labels_runs.append(
            labels.astype(np.int16)
        )

        ari_values.append(ari)
        ami_values.append(ami)
        runtimes.append(dt)
        largest.append(shares.max())

        print(
            f"seed={seed:3d} | "
            f"ARI={ari:.3f} | "
            f"AMI={ami:.3f} | "
            f"largest={shares.max()*100:5.1f}% | "
            f"time={dt:.2f}s"
        )

    stab_mean, stab_std, stab_min = stability(
        labels_runs
    )

    r = {
        "median_connectivity_nn":
            float(median_conn),

        "alpha":
            ALPHA,

        "adaptive_nn":
            nn,

        "ari_mean":
            float(np.mean(ari_values)),

        "ari_std":
            float(np.std(ari_values, ddof=1)),

        "ami_mean":
            float(np.mean(ami_values)),

        "ami_std":
            float(np.std(ami_values, ddof=1)),

        "stability_mean":
            stab_mean,

        "stability_std":
            stab_std,

        "stability_min":
            stab_min,

        "runtime_mean_s":
            float(np.mean(runtimes)),

        "largest_cluster_mean":
            float(np.mean(largest)),
    }

    results[name] = r

    print()
    print("SUMMARY")
    print(f"adaptive nn: {nn}")
    print(
        f"ARI:       {r['ari_mean']:.3f} "
        f"+- {r['ari_std']:.3f}"
    )
    print(
        f"AMI:       {r['ami_mean']:.3f} "
        f"+- {r['ami_std']:.3f}"
    )
    print(
        f"Stability: {r['stability_mean']:.3f} "
        f"+- {r['stability_std']:.3f}"
    )
    print(
        f"Stab min:  {r['stability_min']:.3f}"
    )
    print(
        f"Runtime:   {r['runtime_mean_s']:.3f}s"
    )


OUT.write_text(
    json.dumps(
        {
            "alpha": ALPHA,
            "sample_size": S,
            "seeds": SEEDS,
            "results": results,
        },
        indent=2,
    )
)

print()
print("=" * 85)
print("FINAL ADAPTIVE SUMMARY")
print("=" * 85)

for name, r in results.items():

    print(
        f"{name:12s} "
        f"nn={r['adaptive_nn']:3d} | "
        f"ARI={r['ari_mean']:.3f} | "
        f"AMI={r['ami_mean']:.3f} | "
        f"stab={r['stability_mean']:.3f} | "
        f"time={r['runtime_mean_s']:.3f}s"
    )

print()
print("Saved:", OUT)
