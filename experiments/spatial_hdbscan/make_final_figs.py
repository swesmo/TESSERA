"""Figures for the final report (Christopher, 11 Oct 2026), from results/final_*.json,
results/bench_port_*.json and the label maps written by port_maps.mjs.

    ~/tee-local/venv/bin/python make_final_figs.py [tradeoff] [sensitivity] [maps] [speckle]

Writes results/fig/final_<name>.pdf and .png.

Colours: baseline in grey (context); the method families use the first three slots of a palette
that was checked for colour-vision deficiencies (blue, orange, aqua), and every family also has
its own marker shape, so the figures stay readable in greyscale print.
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import analyze_final as af

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = OUT / "fig"
FIG.mkdir(exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300})

# family -> (colour, marker, label)
FAM = {"base": (MUTED, "s", "TEE baseline (one $k$-means run)"),
       "stab": (BLUE, "o", "Stabilised: restarts or over-clustering"),
       "smooth": (ORANGE, "^", "Smoothed: one run + filter"),
       "both": (AQUA, "D", "Over-clustering + smoothing")}
# method -> family
METHODS = {
    "baseline": "base",
    "restarts 3": "stab",
    "restarts 10": "stab",
    "over32 S=300K0": "stab",
    "Gauss 10 m + baseline": "smooth",
    "baseline + MRF 0.25": "smooth",
    "over32 S=300K0 + MRF 0.25": "both",
    "Gauss 10 m + over32 S=300K0": "both",
}
# Direct labels per panel: method -> (text, offset in points, horizontal alignment). The
# unlabelled orange triangle next to the baseline is the baseline with the MRF.
LABELS = [
    {"baseline": ("baseline", (5, -9), "left"), "baseline + MRF 0.25": ("MRF", (-5, 3), "right"),
     "Gauss 10 m + baseline": ("Gaussian", (5, 3), "left"), "restarts 10": ("10 restarts", (4, -11), "right"),
     "over32 S=300K0": ("over-cluster 32", (5, -10), "left"), "over32 S=300K0 + MRF 0.25": ("proposed", (5, 3), "left"),
     "Gauss 10 m + over32 S=300K0": ("over 32 + Gaussian", (5, 3), "left")},
    {"baseline": ("baseline", (-5, -9), "right"), "Gauss 10 m + baseline": ("Gaussian", (5, -8), "left"),
     "restarts 3": ("3 restarts", (5, -8), "left"), "restarts 10": ("10 restarts", (-5, -10), "right"),
     "over32 S=300K0": ("over-cluster 32", (-5, 4), "right"), "over32 S=300K0 + MRF 0.25": ("proposed", (5, -9), "left"),
     "Gauss 10 m + over32 S=300K0": ("over 32 + Gaussian", (0, 7), "center")},
    {"baseline": ("baseline", (6, -9), "left"), "Gauss 10 m + baseline": ("Gaussian", (6, -3), "left"),
     "restarts 3": ("3 restarts", (6, -3), "left"), "restarts 10": ("10 restarts", (-5, -9), "right"),
     "over32 S=300K0": ("over-cluster 32", (-4, 6), "right"), "over32 S=300K0 + MRF 0.25": ("proposed", (7, -9), "left"),
     "Gauss 10 m + over32 S=300K0": ("over 32 + Gaussian", (0, 7), "center")},
]


def save(fig, name):
    fig.savefig(FIG / f"final_{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"final_{name}.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", FIG / f"final_{name}.pdf")


def bench(which, k):
    f = OUT / f"bench_port_{which}.json"
    return json.loads(f.read_text())["k"][str(k)] if f.exists() else {}


# ───────────────────────── quality against runtime ─────────────────────────

def fig_tradeoff():
    """One point per method: JavaScript time of one clustering run against its quality."""
    import matplotlib.patheffects as pe
    _, A = af.summarise("aut2022")
    _, G = af.summarise("gbg")
    panels = [("(a) Austrian tile, $k = 17$: label agreement", A, bench("aut", 17), "AMI", "AMI with the 17 crop classes"),
              ("(b) Austrian tile, $k = 17$: stability", A, bench("aut", 17), "stab", "Mean ARI between seeds"),
              ("(c) Gothenburg viewport, $k = 5$: stability", G, bench("gbg", 5), "stab", "Mean ARI between seeds")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75))
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for ax, labels, (title, S, T, key, ylabel) in zip(axes, LABELS, panels):
        for name, fam in METHODS.items():
            if name not in S or name not in T:
                continue
            r, t = S[name], T[name]
            x = t["median"] / 1000
            y = r[key]
            yerr = [[r["AMI_sd"]], [r["AMI_sd"]]] if key == "AMI" else [[y - r["stab_lo"]], [r["stab_hi"] - y]]
            col, mk, _ = FAM[fam]
            ax.errorbar(x, y, yerr=yerr, xerr=[[x - t["p25"] / 1000], [t["p75"] / 1000 - x]], fmt="none", ecolor=col, elinewidth=0.6, alpha=0.45, zorder=2)
            ax.plot(x, y, mk, color=col, ms=5.5, mec="white", mew=0.8, zorder=3)
            if name in labels:
                text, off, ha = labels[name]
                ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=6.3, color=INK, ha=ha, zorder=5, path_effects=halo)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Time of one clustering run in JavaScript (s)")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.16, y=0.14)
        ax.set_xlim(left=0)
    handles = [plt.Line2D([], [], marker=m, color=c, ls="none", ms=5.5, mec="white", mew=0.8, label=lab) for c, m, lab in FAM.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout(w_pad=1.4)
    save(fig, "tradeoff")


# ───────────────────────── parameter choices ─────────────────────────

def fig_sensitivity():
    _, A = af.summarise("aut2022")
    halves = {h: af.summarise(f"aut2022_{h}")[1] for h in ("west", "east") if (OUT / f"final_aut2022_{h}.json").exists()}
    _, G = af.summarise("gbg")
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.05))

    def sweep(ax, xs, names, key, chosen, xlabel, title, ylabel):
        for h, S in halves.items():
            ys = [S[n][key] for n in names]
            ax.plot(range(len(xs)), ys, "-", color=MUTED, lw=0.9, alpha=0.7, zorder=2)
            ax.annotate(f"{h} half", (len(xs) - 1, ys[-1]), xytext=(3, 0), textcoords="offset points", fontsize=6, color=MUTED, va="center")
        ys = np.array([A[n][key] for n in names]); sd = np.array([A[n][key + "_sd"] for n in names])
        ax.errorbar(range(len(xs)), ys, yerr=sd, color=BLUE, lw=1.4, marker="o", ms=4, mec="white", mew=0.7, elinewidth=0.7, capsize=0, zorder=3)
        ax.annotate("full tile", (len(xs) - 1, ys[-1]), xytext=(3, 0), textcoords="offset points", fontsize=6, color=INK, va="center")
        i = xs.index(chosen)
        ax.plot(i, ys[i], "o", ms=8, mfc="none", mec=INK, mew=0.9, zorder=4)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
        ax.set_xlim(-0.3, len(xs) - 1 + 1.25)

    sweep(axes[0], ["0", "5", "7.5", "10", "15", "20"], ["restarts 10"] + [f"Gauss {s:g} m + restarts 10" for s in (5, 7.5, 10, 15, 20)],
          "AMI", "10", r"Gaussian $\sigma$ (m)", "(a) Pre-filter width", "AMI with crop classes")
    sweep(axes[1], ["0", "0.1", "0.25", "0.5", "1"], ["restarts 10"] + [f"restarts 10 + MRF {b}" for b in (0.1, 0.25, 0.5, 1.0)],
          "AMI", "0.25", r"MRF weight $\beta$", "(b) MRF weight", "AMI with crop classes")

    ax = axes[2]  # stability against K0 for three sample sizes
    K0s = [24, 32, 48, 64, 96, 128]
    for (c, col, mk) in ((150, AQUA, "D"), (300, BLUE, "o"), (500, ORANGE, "^")):
        ys = [A[f"over{K0} S={c}K0"]["stab"] for K0 in K0s]
        ax.plot(range(len(K0s)), ys, "-", color=col, lw=1.2, marker=mk, ms=4, mec="white", mew=0.7, label=f"$S = {c}\\,K_0$", zorder=3)
    ax.axhline(A["baseline"]["stab"], color=MUTED, lw=0.9, zorder=2)
    ax.annotate("TEE baseline", (0, A["baseline"]["stab"]), xytext=(0, 3), textcoords="offset points", fontsize=6, color=MUTED, ha="left")
    ax.plot(K0s.index(32), A["over32 S=300K0"]["stab"], "o", ms=8, mfc="none", mec=INK, mew=0.9, zorder=4)
    ax.set_ylim(0.57, 0.78)
    ax.set_xticks(range(len(K0s))); ax.set_xticklabels(K0s)
    ax.set_xlabel("Over-clusters $K_0$"); ax.set_ylabel("Mean ARI between seeds"); ax.set_title("(c) Over-clustering", loc="left")
    ax.legend(loc="center", ncol=1, bbox_to_anchor=(0.72, 0.36), handletextpad=0.3, handlelength=1.4, fontsize=6, labelspacing=0.25)

    ax = axes[3]  # restarts: stability against n_init on both datasets, over-clustering as the last tick
    ns = [1, 2, 3, 5, 10]
    names = ["baseline"] + [f"restarts {n}" for n in ns[1:]]
    xo = len(ns) + 0.3
    for S, col, mk, lab in ((G, ORANGE, "^", "Gothenburg, $k = 5$"), (A, BLUE, "o", "Austria, $k = 17$")):
        ys = [S[n]["stab"] for n in names]
        lo = [S[n]["stab"] - S[n]["stab_lo"] for n in names]; hi = [S[n]["stab_hi"] - S[n]["stab"] for n in names]
        ax.errorbar(range(len(ns)), ys, yerr=[lo, hi], color=col, lw=1.2, marker=mk, ms=4, mec="white", mew=0.7, elinewidth=0.7, label=lab, zorder=3)
        o = S["over32 S=300K0"]
        ax.errorbar([xo], [o["stab"]], yerr=[[o["stab"] - o["stab_lo"]], [o["stab_hi"] - o["stab"]]], color=col, marker=mk, ms=4, mec="white",
                    mew=0.7, elinewidth=0.7, ls="none", zorder=4)
    ax.set_xticks(list(range(len(ns))) + [xo]); ax.set_xticklabels([str(n) for n in ns] + ["OC"])
    ax.set_xlim(-0.35, xo + 0.4)
    ax.set_ylim(0.52, 1.0)
    ax.set_xlabel("Restarts (OC: over-clustering)"); ax.set_ylabel("Mean ARI between seeds"); ax.set_title("(d) Restarts", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.03), handletextpad=0.3, fontsize=6, labelspacing=0.25)
    fig.tight_layout(w_pad=0.9)
    save(fig, "sensitivity")


# ───────────────────────── maps ─────────────────────────

def match(ref, lab, k):
    """Rename the clusters of `lab` so that they overlap best with `ref` (Hungarian matching)."""
    C = np.zeros((k, k))
    np.add.at(C, (ref.ravel(), lab.ravel()), 1)
    r, c = linear_sum_assignment(-C)
    lut = np.empty(k, int); lut[c] = r
    return lut[lab]


def fig_maps():
    """Gothenburg, two seeds, TEE's own worker code: the current k-means and the 'Stable' method."""
    f = OUT / "port_maps_gbg.json"
    meta = json.loads(f.read_text())
    H, W, k = meta["H"], meta["W"], meta["k"]
    labs = np.fromfile(OUT / "port_maps_gbg.i32", "<i4").reshape(len(meta["runs"]), H, W)
    runs = {(r["method"], r["seed"]): labs[i] for i, r in enumerate(meta["runs"])}
    sa, sb = meta["seeds"]
    cmap = matplotlib.colors.ListedColormap([BLUE, ORANGE, AQUA, YELLOW, MAGENTA][:k])
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.98))
    ref = runs[("kmeans", sa)]
    panels = [("kmeans", sa, "(a) TEE $k$-means, seed %d"), ("kmeans", sb, "(b) TEE $k$-means, seed %d"),
              ("stable", sa, "(c) Over-clustering, seed %d"), ("stable", sb, "(d) Over-clustering, seed %d")]
    for ax, (method, seed, title) in zip(axes, panels):
        ax.imshow(match(ref, runs[(method, seed)], k), cmap=cmap, vmin=-0.5, vmax=k - 0.5, interpolation="nearest", aspect=13.0 / 6.95)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title % seed, loc="left")
    for method in ("kmeans", "stable", "smooth", "gauss"):
        a, b = runs[(method, sa)], runs[(method, sb)]
        print(f"{method}: {100 * float((a != match(a, b, k)).mean()):.1f} % of the pixels differ between seeds {sa} and {sb}")
    fig.tight_layout(w_pad=0.5)
    save(fig, "maps_gbg")


def fig_speckle():
    """Austrian tile, seed 42: pixels that get another cluster than a neighbouring pixel of the same
    field (what the speckle metric counts), for the baseline and the two smoothed pipelines."""
    import tsh
    ds = tsh.austria(year=2022)
    fid = ds["fid"]
    maps = np.load(OUT / "maps_final_aut2022.npz")
    S = af.summarise("aut2022")[1]
    y0, x0, n = 420, 150, 300                                      # 3 km x 3 km with many fields
    sl = (slice(y0, y0 + n), slice(x0, x0 + n))
    methods = [("baseline", "(a) TEE baseline"), ("over32 S=300K0 + MRF 0.25", "(b) Over-clustering + MRF (proposed)"),
               ("Gauss 10 m + over32 S=300K0", "(c) Over-clustering + Gaussian pre-filter")]
    edge = np.zeros(fid.shape, bool)                               # field outlines
    edge[:, 1:] |= (fid[:, 1:] != fid[:, :-1]) & ((fid[:, 1:] > 0) | (fid[:, :-1] > 0))
    edge[1:] |= (fid[1:] != fid[:-1]) & ((fid[1:] > 0) | (fid[:-1] > 0))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.62))
    for ax, (name, title) in zip(axes, methods):
        lab = maps[name].astype(np.int64)
        differs = np.zeros(fid.shape, bool)                        # differs from a 4-neighbour of the same field
        h = (fid[:, 1:] > 0) & (fid[:, 1:] == fid[:, :-1]) & (lab[:, 1:] != lab[:, :-1])
        v = (fid[1:] > 0) & (fid[1:] == fid[:-1]) & (lab[1:] != lab[:-1])
        differs[:, 1:] |= h; differs[:, :-1] |= h; differs[1:] |= v; differs[:-1] |= v
        img = np.zeros(fid.shape, np.int8)                         # 0 outside fields, 1 field, 2 differs
        img[fid > 0] = 1
        img[differs] = 2
        rgb = np.array([[252, 252, 251], [228, 227, 223], [235, 104, 52]], np.uint8)[img]
        rgb[edge & (img != 2)] = (150, 149, 144)
        ax.imshow(rgb[sl], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ea = tsh.edge_adherence(lab[sl], fid[sl])
        ax.set_title(title, loc="left")
        print(f"{name}: speckle in the crop, seed 42: {100 * ea['d_in']:.1f} %; whole tile, 20 seeds: {100 * S[name]['speckle']:.1f} %")
    fig.tight_layout(w_pad=0.5)
    save(fig, "speckle_aut")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["tradeoff", "sensitivity", "maps", "speckle"]
    for name in todo:
        {"tradeoff": fig_tradeoff, "sensitivity": fig_sensitivity, "maps": fig_maps, "speckle": fig_speckle}[name]()
