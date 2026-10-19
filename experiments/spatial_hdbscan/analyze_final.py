"""Summarise results/final_<tag>.json (written by run_final.py): mean and standard deviation over
the seeds, paired differences to the baseline, and a bootstrap interval for the stability.

    ~/tee-local/venv/bin/python analyze_final.py [tag ...]      (default: aut2022 gbg)

Writes results/final_<tag>.csv (all methods) and prints / writes results/final_summary.md.

Statistics
  per-seed metrics (AMI, ARI vs labels, speckle, ...): mean and sample standard deviation over the
      seeds. Difference to the baseline: paired by seed, mean with a 95 % t-interval, the number of
      seeds on which the method is better, and the p-value of a Wilcoxon signed-rank test.
  stability: mean and minimum of the ARI between all pairs of seeds. The pairs are not
      independent, so the 95 % interval is a bootstrap over SEEDS (2,000 resamples; pairs of a seed
      with itself are left out). The interval of the difference to the baseline resamples the same
      seeds for both methods.
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "results"
B = 2000


def load(tag):
    return json.loads((OUT / f"final_{tag}.json").read_text())


def boot_idx(n, seed=0):
    return np.random.default_rng(seed).integers(0, n, (B, n))


def stab(P, idx):
    P = np.asarray(P, float)
    b = np.array([np.nanmean(P[np.ix_(i, i)]) for i in idx])
    return float(np.nanmean(P)), float(np.nanmin(P)), b


def paired(x, base, higher_better=True):
    d = np.asarray(x) - np.asarray(base)
    n = len(d)
    if np.allclose(d, 0):
        return dict(diff=0.0, lo=0.0, hi=0.0, better=0, p=1.0)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return dict(diff=float(d.mean()), lo=float(d.mean() - half), hi=float(d.mean() + half),
                better=int((d > 0).sum() if higher_better else (d < 0).sum()), p=float(stats.wilcoxon(d).pvalue))


def summarise(tag):
    R = load(tag)
    M, meta = R["methods"], R["meta"]
    n = len(meta["seeds"])
    idx = boot_idx(n)
    base = M["baseline"]
    _, _, b_base = stab(base["pairs"], idx)
    rows = []
    for name, r in M.items():
        ps = r["per_seed"]
        s_mean, s_min, b = stab(r["pairs"], idx)
        row = dict(method=name, stab=s_mean, stab_min=s_min, stab_lo=float(np.percentile(b, 2.5)),
                   stab_hi=float(np.percentile(b, 97.5)), dstab=s_mean - stab(base["pairs"], idx)[0],
                   dstab_lo=float(np.percentile(b - b_base, 2.5)), dstab_hi=float(np.percentile(b - b_base, 97.5)),
                   coherence=float(np.mean(ps["coherence"])), t_py=float(np.median(r["t_py"])))
        for key, hb in (("AMI", True), ("ARI_gt", True), ("homog", True), ("speckle", False), ("edge", True)):
            if key in ps:
                row[key], row[key + "_sd"] = float(np.mean(ps[key])), float(np.std(ps[key], ddof=1))
                for a, v in paired(ps[key], base["per_seed"][key], hb).items():
                    row[f"{key}_{a}"] = v
        rows.append(row)
    with open(OUT / f"final_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return meta, {r["method"]: r for r in rows}


MAIN = ["baseline", "restarts 3", "restarts 10", "baseline + 1 full Lloyd", "over32 S=300K0", "over32 S=500K0",
        "over32 S=300K0 + 1 full Lloyd", "over32 S=300K0, fit in PCA-16", "over32 S=300K0, fit in PCA-16 + 1 full Lloyd",
        "Gauss 10 m + baseline", "Gauss 10 m + restarts 10", "Gauss 10 m + over32 S=300K0", "restarts 10 + MRF 0.25",
        "restarts 10 + majority 20 m", "over32 S=300K0 + MRF 0.25", "Gauss 10 m + over32 S=300K0 + MRF 0.25",
        "superpixels 30 m c=1 + over32 S=300K0", "blocks 30 m + over32 S=300K0", "PCA-16 baseline", "z-score baseline"]


def table(meta, S, names, labelled):
    out = []
    if labelled:
        out.append("| Method | AMI (mean ± sd) | ΔAMI vs baseline [95 %] | better / p | ARI vs labels | Δ ARI [95 %] | Stability [95 %] (min) | Homog. | Speckle | Edge kept |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
    else:
        out.append("| Method | Stability [95 %] (min) | Δ stability vs baseline [95 %] | Coherence |")
        out.append("|---|---|---|---|")
    n = len(meta["seeds"])
    for name in names:
        if name not in S:
            continue
        r = S[name]
        st = f"{r['stab']:.3f} [{r['stab_lo']:.2f}, {r['stab_hi']:.2f}] ({r['stab_min']:.2f})"
        if labelled:
            out.append(f"| {name} | {r['AMI']:.3f} ± {r['AMI_sd']:.3f} | {r['AMI_diff']:+.3f} [{r['AMI_lo']:+.3f}, {r['AMI_hi']:+.3f}] | "
                       f"{r['AMI_better']}/{n}, p = {r['AMI_p']:.2g} | {r['ARI_gt']:.3f} ± {r['ARI_gt_sd']:.3f} | "
                       f"{r['ARI_gt_diff']:+.3f} [{r['ARI_gt_lo']:+.3f}, {r['ARI_gt_hi']:+.3f}] | {st} | {100 * r['homog']:.1f} % | "
                       f"{100 * r['speckle']:.1f} % | {100 * r['edge']:.1f} % |")
        else:
            out.append(f"| {name} | {st} | {r['dstab']:+.3f} [{r['dstab_lo']:+.3f}, {r['dstab_hi']:+.3f}] | {r['coherence']:.3f} |")
    return "\n".join(out)


def sweep(S, title, names, cols):
    out = [f"**{title}**", "", "| Setting | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for label, name in names:
        if name in S:
            out.append(f"| {label} | " + " | ".join(f(S[name]) for _, f in cols) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    tags = sys.argv[1:] or ["aut2022", "gbg"]
    md = ["# Final method table: summary (generated by analyze_final.py)", ""]
    for tag in tags:
        if not (OUT / f"final_{tag}.json").exists():
            continue
        meta, S = summarise(tag)
        lab = meta["labelled"] > 0
        md += [f"## {meta['dataset']} ({meta['part']}), k = {meta['k']}, N = {meta['N']:,}, {len(meta['seeds'])} seeds"
               + (f", {meta['labelled']:,} labelled pixels" if lab else ", no labels"), "", table(meta, S, MAIN, lab), ""]
        f3 = lambda key: (lambda r: f"{r[key]:.3f}")
        pct = lambda key: (lambda r: f"{100 * r[key]:.1f} %")
        st = ("Stability (min)", lambda r: f"{r['stab']:.3f} ({r['stab_min']:.2f})")
        q = ([("AMI", lambda r: f"{r['AMI']:.3f} ± {r['AMI_sd']:.3f}"), ("ARI vs labels", f3("ARI_gt")), ("Speckle", pct("speckle")),
              ("Edge kept", pct("edge"))] if lab else [("Coherence", f3("coherence"))])
        md += [sweep(S, "Restarts (n_init)", [("1 (baseline)", "baseline")] + [(str(n), f"restarts {n}") for n in (2, 3, 5, 10)], [st] + q[:2]), "",
               sweep(S, "Gaussian pre-filter, sigma (on 10 restarts)", [("none", "restarts 10")] + [(f"{s:g} m", f"Gauss {s:g} m + restarts 10") for s in (5, 7.5, 10, 15, 20)], [st] + q), "",
               sweep(S, "MRF, beta (on 10 restarts)", [("none", "restarts 10")] + [(str(b), f"restarts 10 + MRF {b}") for b in (0.1, 0.25, 0.5, 1.0)], [st] + q), "",
               sweep(S, "Majority filter, radius (on 10 restarts)", [("none", "restarts 10"), ("10 m", "restarts 10 + majority 10 m"), ("20 m", "restarts 10 + majority 20 m")], [st] + q), "",
               sweep(S, "Over-clustering: K0 and sample per over-cluster", [(f"K0 = {K0}, S = {c} K0", f"over{K0} S={c}K0") for K0 in (24, 32, 48, 64, 96, 128) for c in (150, 300, 500)], [st] + q[:2]), "",
               sweep(S, "Over-clustering 32: variants", [(n, n) for n in S if n.startswith("over32") or n.startswith("over64 S=300K0, fit")], [st] + q[:2]), "",
               sweep(S, "Dimension reduction and superpixels", [(n, n) for n in S if n.startswith(("PCA", "z-score", "superpixels", "blocks"))], [st] + q), ""]
    (OUT / "final_summary.md").write_text("\n".join(md))
    print("\n".join(md))
