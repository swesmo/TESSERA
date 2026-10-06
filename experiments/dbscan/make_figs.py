"""Report figure: DBSCAN red-flag check (Schubert et al. 2017) on both datasets + Gothenburg map.
    ~/tee-local/venv/bin/python make_figs.py   -> fig/dbscan_redflags.pdf (+ .png preview)"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
(HERE / "fig").mkdir(exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = {8: "#86b6ef", 16: "#3987e5", 32: "#1c5cab", 128: "#0d366b"}   # ordinal blue ramp (validated)
MARK = {8: "o", 16: "s", 32: "^", 128: "D"}
QS = ["0.5", "0.75", "0.9", "0.95", "0.99", "knee"]
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6})


def redflag_panel(ax, tag, title):
    r = json.loads((R / f"dbscan_{tag}.json").read_text())
    ax.add_patch(Rectangle((1, 0), 29, 50, facecolor="#f0efec", edgecolor="none", zorder=0))
    ax.text(15.5, 3, "acceptable\n(noise 1–30 %,\nlargest ≤ 50 %)", ha="center", va="bottom", fontsize=6, color=INK2)
    for d in (8, 16, 32, 128):
        pts = []
        for q in QS:
            name = f"DBSCAN pca{d} q={q}" if d < 128 else f"DBSCAN raw128 q={q}"
            v = r[name]
            noise = 100 * v["noise"]
            largest = 100 * v["max_share"] / max(1 - v["noise"], 1e-9)
            pts.append((float(v["q"]) if q == "knee" else float(q), noise, largest, q == "knee"))
        pts.sort()
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        ax.plot(xs, ys, color=RAMP[d], lw=1.2, zorder=2)
        ax.scatter(xs, ys, s=16, marker=MARK[d], color=RAMP[d], edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"PCA {d}" if d < 128 else "raw 128-d")
        kx, ky = [(p[1], p[2]) for p in pts if p[3]][0]
        ax.scatter([kx], [ky], s=46, marker=MARK[d], facecolor="none", edgecolor=INK, linewidth=0.7, zorder=4)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102)
    ax.set_xticks([0, 1, 10, 30, 100]); ax.set_xticklabels(["0", "1", "10", "30", "100"])
    ax.set_xlabel("noise pixels (%)")
    ax.set_ylabel("largest cluster (% of clustered pixels)")
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 0.95]))
redflag_panel(axs[0], "gbg", "(a) Gothenburg 2024, 5 seeds")
redflag_panel(axs[1], "aut22", "(b) Austria tile 2022, 5 seeds")
axs[1].set_ylabel("")
h, l = axs[0].get_legend_handles_labels()
h.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=6))
l.append("ε at k-dist knee")
axs[0].legend(h, l, loc="upper left", bbox_to_anchor=(0.0, 0.76), fontsize=6, frameon=True, framealpha=0.95, edgecolor=GRID, handlelength=1.5)

m = np.load(R / "maps_dbscan_gbg.npz")
db = m["DBSCAN pca8 q=knee"]
u, c = np.unique(db[db >= 0], return_counts=True)
big, river = u[np.argsort(c)[-1]], u[np.argsort(c)[-2]]
cat = np.full(db.shape, 2)          # 2 = other small clusters
cat[db == big] = 0; cat[db == river] = 1; cat[db < 0] = 3
cmap = ListedColormap(["#dcdbd6", "#2a78d6", "#eb6834", "#0b0b0b"])
axs[2].imshow(cat, cmap=cmap, vmin=0, vmax=3, interpolation="nearest", aspect=13.0 / 6.95)
axs[2].set_axis_off()
axs[2].set_title("(c) Gothenburg, PCA 8, ε at knee", fontsize=7.5, color=INK, loc="left")
sh = lambda x: f"{100 * x:.0f} %"
axs[2].legend(handles=[Patch(color="#dcdbd6", label=f"largest cluster ({sh((db == big).mean())})"),
                       Patch(color="#2a78d6", label=f"2nd cluster: river ({sh((db == river).mean())})"),
                       Patch(color="#eb6834", label=f"{len(u) - 2} small clusters ({sh(((db >= 0) & (db != big) & (db != river)).mean())})"),
                       Patch(color="#0b0b0b", label=f"noise ({sh((db < 0).mean())})")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=5.8, frameon=False, ncol=2,
              handlelength=1, columnspacing=0.8)
fig.tight_layout(w_pad=1.2)
fig.savefig(HERE / "fig/dbscan_redflags.pdf", bbox_inches="tight")
fig.savefig(HERE / "fig/dbscan_redflags.png", dpi=200, bbox_inches="tight")
print("ok")
