"""Build results/report.html from the JSON results and figures (all numbers come from files).
    ~/tee-local/venv/bin/python build_report.py
"""
import base64, json
from html import escape
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
J = lambda n: json.loads((R / n).read_text()) if (R / n).exists() else {}
gbg, aut, edges, js = J("gbg.json"), J("aut2024.json"), J("aut2024_edges_seed42.json"), J("bench_js.json")
hx_g, hx_a = J("hdbscan_extra_gbg.json"), J("hdbscan_extra_aut2024.json")
ov_g, ov_a, ov_e = J("overcluster_gbg.json"), J("overcluster_aut2024.json"), J("overcluster_aut2024_edges_seed42.json")
BEST = "over K0=32 + Ward + refine + MRF 0.25"


def img(name, alt):
    b = base64.b64encode((R / "fig" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{escape(alt)}" loading="lazy">'


def fam(name):
    if "over K0" in name:
        return "merge"
    if name.startswith("HDBSCAN"):
        return "hdb"
    if name.startswith("kmeans"):
        return "base"
    return "smooth"


def f(x, d=2):
    return f"{x:.{d}f}"


def pct(x):
    return f"{100 * x:.0f} %"


# ── chart 1: Austria speckle vs edge placement (seed 42) ──
def scatter():
    pts = [(n.replace("MRF_ICM", "MRF/ICM"), v["d_in"], v["ratio"]) for n, v in edges.items()
           if not n.startswith("HDBSCAN pca16 mcs=20")]
    pts += [(n, v["d_in"], v["ratio"]) for n, v in ov_e.items() if "over K0" in n]
    W, H, L, B, T, Rr = 640, 380, 58, 46, 16, 16
    x0, x1, y0, y1 = 0.0, 0.28, 1.4, 2.05
    sx = lambda x: L + (x - x0) / (x1 - x0) * (W - L - Rr)
    sy = lambda y: T + (y1 - y) / (y1 - y0) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Scatter: speckle inside fields versus edge placement, per method">']
    for gx in (0, 0.05, 0.10, 0.15, 0.20, 0.25):
        out.append(f'<line x1="{sx(gx):.1f}" x2="{sx(gx):.1f}" y1="{T}" y2="{H - B}" class="grid"/>'
                   f'<text x="{sx(gx):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{int(gx * 100)} %</text>')
    for gy in (1.4, 1.6, 1.8, 2.0):
        out.append(f'<line x1="{L}" x2="{W - Rr}" y1="{sy(gy):.1f}" y2="{sy(gy):.1f}" class="grid"/>'
                   f'<text x="{L - 8}" y="{sy(gy) + 4:.1f}" class="tick" text-anchor="end">{gy:.1f}×</text>')
    out.append(f'<text x="{(L + W - Rr) / 2}" y="{H - 6}" class="axis" text-anchor="middle">Speckle: neighbours inside one field with different clusters (lower is better)</text>')
    out.append(f'<text transform="translate(14 {(T + H - B) / 2}) rotate(-90)" class="axis" text-anchor="middle">Edge ratio (higher is better)</text>')
    show = {"kmeans (TEE)": ("TEE today", -10, -10), "post mode r=20m x1": ("majority 20 m", 8, 14),
            "pre gauss sigma=10m": ("Gaussian 10 m", 8, -8), "MRF/ICM beta=0.25": ("MRF β 0.25", -10, 16),
            "pre bilateral r=10m x2": ("bilateral 10 m", 8, -8), "post mode r=40m x1": ("majority 40 m", 8, -8),
            "HDBSCAN pca16 mcs=100": ("HDBSCAN (fewer clusters)", 8, 14),
            BEST: ("recommended: merge + refine + MRF", -10, 18), "over K0=64 + average link": ("average link", 8, 14)}
    for n, x, y in pts:
        c = fam(n)
        out.append(f'<g class="pt {c}"><circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5.5"/>'
                   f'<title>{escape(n)}: speckle {pct(x)}, edge ratio {y:.2f}×</title></g>')
        if n in show:
            lab, dx, dy = show[n]
            anchor = "end" if dx < 0 else "start"
            out.append(f'<text x="{sx(x) + dx:.1f}" y="{sy(y) + dy:.1f}" class="lbl" text-anchor="{anchor}">{escape(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


# ── chart 2: stability dot + range ──
def stab(res, keys, title):
    W, rowh, L = 640, 26, 250
    H = 34 + rowh * len(keys) + 24
    sx = lambda v: L + v * (W - L - 20)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{sx(g):.1f}" x2="{sx(g):.1f}" y1="24" y2="{H - 24}" class="grid"/>'
                   f'<text x="{sx(g):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{g:.2f}</text>')
    out.append(f'<text x="{L}" y="14" class="axis">Adjusted Rand index between runs with different seeds: dot = mean, bar = worst pair</text>')
    for i, (k, lab) in enumerate(keys):
        v = res[k]
        y = 34 + i * rowh + rowh / 2
        c = fam(k)
        out.append(f'<text x="{L - 10}" y="{y + 4:.1f}" class="rowlbl" text-anchor="end">{escape(lab)}</text>'
                   f'<g class="pt {c}"><line x1="{sx(max(v["ARI_min"], 0)):.1f}" x2="{sx(v["ARI_mean"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="rng"/>'
                   f'<circle cx="{sx(v["ARI_mean"]):.1f}" cy="{y:.1f}" r="5.5"/>'
                   f'<title>{escape(k)}: mean ARI {v["ARI_mean"]:.2f}, worst {v["ARI_min"]:.2f}, {v["seeds"]} seeds</title></g>')
    out.append("</svg>")
    return "".join(out)


STAB_KEYS = [("kmeans (TEE)", "TEE k-means today"), ("kmeans n_init=10", "k-means, 10 restarts"),
             ("TEE kmeans + post mode r=20m", "TEE today + majority 20 m"),
             ("post mode r=20m x1", "restarts + majority 20 m"), ("pre gauss sigma=10m", "restarts + Gaussian 10 m"),
             ("pre bilateral r=10m x2", "restarts + bilateral 10 m"), ("MRF/ICM beta=0.5", "restarts + MRF β 0.5"),
             ("HDBSCAN pca16 mcs=100", "HDBSCAN (PCA 16)")]


def aut_table():
    rows = [("kmeans (TEE)", "TEE k-means today"), ("kmeans n_init=10", "k-means, 10 restarts"),
            ("post mode r=10m x1", "Majority filter 10 m"), ("post mode r=20m x1", "Majority filter 20 m"),
            ("post mode r=40m x1", "Majority filter 40 m"), ("pre gauss sigma=10m", "Gaussian pre-filter 10 m"),
            ("pre gauss sigma=20m", "Gaussian pre-filter 20 m"), ("pre bilateral r=10m x2", "Bilateral pre-filter 10 m"),
            ("MRF/ICM beta=0.25", "MRF / Potts, β 0.25"), ("MRF/ICM beta=0.5", "MRF / Potts, β 0.5"),
            ("MRF/ICM beta=1.0", "MRF / Potts, β 1.0"), ("bilateral r=10m x2 + ICM 0.5", "Bilateral + MRF β 0.5")]
    h = ["<table><thead><tr><th>Method (Austria, k = 17)</th><th>Field homogeneity</th><th>Speckle in fields</th>"
         "<th>Edge ratio</th><th>Stability (ARI)</th><th>Python time</th></tr></thead><tbody>"]
    for k, lab in rows:
        v, e = aut[k], edges[k.replace("/", "_")]
        t = v["t"] + v.get("t_pre", 0)
        h.append(f'<tr class="{fam(k)}"><td>{lab}</td><td>{pct(v["field_homog"])}</td><td>{pct(e["d_in"])}</td>'
                 f'<td>{e["ratio"]:.2f}×</td><td>{v["ARI_mean"]:.2f}</td><td>{t:.2f} s</td></tr>')
    h.append("</tbody></table>")
    return "".join(h)


def hdb_table():
    rows = [(n, v) for n, v in gbg.items() if n.startswith("HDBSCAN")]
    h = ["<table><thead><tr><th>HDBSCAN setting</th><th>Clusters</th><th>Largest cluster</th><th>Effective k</th>"
         "<th>Noise in sample</th><th>ARI</th><th>Clusters (Austria)</th><th>Largest (Austria)</th><th>Noise (Austria)</th></tr></thead><tbody>"]
    for n, v in rows:
        a = aut.get(n, {})
        h.append(f"<tr><td>{escape(n.replace('HDBSCAN ', ''))}</td><td>{v['k_found']:.0f}</td><td>{pct(v['max_share'])}</td>"
                 f"<td>{v['k_eff']:.1f}</td><td>{pct(v['noise'])}</td><td>{v['ARI_mean']:.2f}</td>"
                 f"<td>{a.get('k_found', float('nan')):.0f}</td><td>{pct(a.get('max_share', 0))}</td><td>{pct(a.get('noise', 0))}</td></tr>")
    h.append("</tbody></table>")
    return "".join(h)


def hdb_extra_table():
    if not hx_g:
        return "<p class='note'>Follow-up run missing.</p>"
    h = ["<table><thead><tr><th>Setting (PCA 16, sample 20 000)</th><th>Dataset</th><th>Clusters</th><th>Largest</th>"
         "<th>ARI</th><th>Speckle-free (coherence)</th><th>Field homog.</th></tr></thead><tbody>"]
    for ds, res in (("Gothenburg", hx_g), ("Austria", hx_a)):
        for n, v in res.items():
            lab = n.replace("HDBSCAN pca16 ", "HDBSCAN ").replace(" | k-means same k", " → k-means, same k")
            cls = "base" if "k-means" in n else "hdb"
            fh = pct(v["field_homog"]) if "field_homog" in v else "–"
            h.append(f'<tr class="{cls}"><td>{escape(lab)}</td><td>{ds}</td><td>{v["k_found"]:.0f}</td><td>{pct(v["max_share"])}</td>'
                     f'<td>{v["ARI_mean"]:.2f}</td><td>{pct(v["coherence"])}</td><td>{fh}</td></tr>')
    h.append("</tbody></table>")
    return "".join(h)


OV_ROWS = [("kmeans (TEE)", "TEE k-means today"), ("kmeans n_init=10", "k-means, 10 restarts"),
           ("over K0=32 + Ward", "Over-cluster 32 → Ward"), ("over K0=64 + Ward", "Over-cluster 64 → Ward"),
           ("over K0=128 + Ward", "Over-cluster 128 → Ward"), ("over K0=64 + average link", "Over-cluster 64 → average link"),
           ("over K0=64 + spatial Ward g=1", "Over-cluster 64 → spatial Ward γ 1"),
           ("over K0=64 + spatial Ward g=4", "Over-cluster 64 → spatial Ward γ 4"),
           ("over K0=32 + Ward + refine", "Over-cluster 32 → Ward → refine"),
           ("over K0=32 + Ward + refine, sample 10k", "… same, over-cluster on 10 000 px"),
           ("over K0=32 + Ward + refine + MRF 0.25", "… + MRF β 0.25 (recommended)"),
           ("Gauss 10m + over K0=32 + Ward + refine", "Gaussian 10 m + over-cluster 32 → Ward → refine")]


def ov_table():
    ig, ia = ov_g["kmeans n_init=10"]["inertia"], ov_a["kmeans n_init=10"]["inertia"]
    h = ["<table><thead><tr><th>Method</th><th>ARI Gbg</th><th>ARI Aut</th><th>Worst ARI Aut</th><th>Fit vs restarts Gbg / Aut</th>"
         "<th>Speckle Aut</th><th>Edge ratio Aut</th><th>Python time Aut</th></tr></thead><tbody>"]
    for k, lab in OV_ROWS:
        g, a = ov_g[k], ov_a[k]
        t = a["t_fit"] + a["t_merge"] + a.get("t_pre", 0)
        rel = lambda x, ref: f"{100 * (x / ref - 1):+.1f} %"
        h.append(f'<tr class="{fam(k)}"><td>{escape(lab)}</td><td>{g["ARI_mean"]:.2f}</td><td>{a["ARI_mean"]:.2f}</td>'
                 f'<td>{a["ARI_min"]:.2f}</td><td>{rel(g["inertia"], ig)} / {rel(a["inertia"], ia)}</td>'
                 f'<td>{pct(a["d_in"])}</td><td>{a["ratio"]:.2f}×</td><td>{t:.2f} s</td></tr>')
    h.append("</tbody></table>")
    return "".join(h)


OV_STAB = [("kmeans (TEE)", "TEE k-means today"), ("kmeans n_init=10", "k-means, 10 restarts"),
           ("over K0=32 + Ward", "over-cluster 32 → Ward"), ("over K0=64 + Ward", "over-cluster 64 → Ward"),
           ("over K0=64 + average link", "over-cluster 64 → average link"),
           ("over K0=32 + Ward + refine, sample 10k", "over-cluster 32 → Ward → refine"),
           ("over K0=32 + Ward + refine + MRF 0.25", "… + MRF β 0.25")]


def js_table():
    k5, k17 = js.get("k=5", {}), js.get("k=17", {})
    hc, asg, lib = js.get("hdbscan_core_ms", {}), js.get("assign_ms", {}), js.get("hdbscan_ts_ms", {})
    ms = lambda x: "–" if x is None else (f"{x:,.0f} ms" if x < 10_000 else f"{x / 1000:,.0f} s")
    rows = [("TEE k-means as shipped (reference)", k5.get("tee_kmeans_ms"), k17.get("tee_kmeans_ms")),
            ("Majority filter, 10 m window", k5.get("post_mode_10m_ms"), k17.get("post_mode_10m_ms")),
            ("Majority filter, 20 m window", k5.get("post_mode_20m_ms"), k17.get("post_mode_20m_ms")),
            ("MRF / Potts, 5 ICM sweeps", k5.get("icm_5iter_ms"), k17.get("icm_5iter_ms")),
            ("  + all pixel-to-centroid distances (free if the worker keeps them)", k5.get("icm_distances_ms"), k17.get("icm_distances_ms")),
            ("Gaussian pre-filter 10 m, all 128 channels", js.get("gauss_pre_10m_ms"), js.get("gauss_pre_10m_ms"))]
    h = ["<table><thead><tr><th>Step (Gothenburg viewport, 277 200 pixels, one thread)</th><th>k = 5</th><th>k = 17</th></tr></thead><tbody>"]
    for lab, a, b in rows:
        h.append(f"<tr><td>{escape(lab)}</td><td>{ms(a)}</td><td>{ms(b)}</td></tr>")
    h.append("</tbody></table>")
    h.append("<table><thead><tr><th>HDBSCAN step in JavaScript</th><th>Sample size</th><th>Time</th></tr></thead><tbody>")
    h.append(f"<tr><td>PCA to 16 dimensions, project all pixels</td><td>20 000 fit</td><td>{ms(js.get('pca16_project_all_ms'))}</td></tr>")
    for n, v in hc.items():
        h.append(f"<tr><td>Core distances + minimum spanning tree (no n×n matrix)</td><td>{int(n):,}</td><td>{ms(v)}</td></tr>")
    for n, v in asg.items():
        h.append(f"<tr><td>Assign every pixel to its nearest of n exemplars</td><td>{int(n):,}</td><td>{ms(v)}</td></tr>")
    for n, v in lib.items():
        h.append(f"<tr><td><code>hdbscan-ts</code> 1.0.17 (npm), full fit</td><td>{int(n):,}</td><td>{ms(v)}</td></tr>")
    h.append("</tbody></table>")
    oc, os_ = js.get("overcluster_ms", {}), js.get("overcluster_sample_only_ms", {})
    h.append("<table><thead><tr><th>Over-cluster step in JavaScript (TEE worker code)</th><th>Setting</th><th>Time</th></tr></thead><tbody>")
    for key, v in oc.items():
        h.append(f"<tr><td>k-means with K0 centroids incl. full-image assignment</td><td>{key.replace('sample=', 'sample ')}</td><td>{ms(v['kmeans_ms'])}</td></tr>")
    for key, v in os_.items():
        h.append(f"<tr><td>k-means with K0 centroids on the sample only</td><td>{key.replace('sample=', 'sample ')}</td><td>{ms(v['kmeans_ms'])}</td></tr>")
    for key, v in oc.items():
        if "10000" in key:
            h.append(f"<tr><td>Ward merge of the K0 centroids down to 5</td><td>{key.split(' ')[0]}</td><td>{v['ward_merge_ms']:.1f} ms</td></tr>")
    h.append("</tbody></table>")
    return "".join(h).replace(",", " ")


t = lambda k: gbg[k]
base, rst = aut["kmeans (TEE)"], aut["kmeans n_init=10"]
g_base, g_rst = gbg["kmeans (TEE)"], gbg["kmeans n_init=10"]
e_base, e_g10, e_m20, e_icm = edges["kmeans (TEE)"], edges["pre gauss sigma=10m"], edges["post mode r=20m x1"], edges["MRF_ICM beta=0.25"]
hd = gbg["HDBSCAN pca16 mcs=100"]

html = (Path(__file__).resolve().parent / "report_template.html").read_text()
subs = {
    "SCATTER": scatter(),
    "STAB_GBG": stab(gbg, STAB_KEYS, "Stability across seeds, Gothenburg"),
    "STAB_AUT": stab(aut, STAB_KEYS, "Stability across seeds, Austria"),
    "AUT_TABLE": aut_table(), "HDB_TABLE": hdb_table(), "HDB_EXTRA": hdb_extra_table(), "JS_TABLE": js_table(),
    "G_TEE": img("gbg_tee_zoom.png", "Gothenburg centre, TEE k-means today"),
    "G_NINIT": img("gbg_ninit_zoom.png", "Gothenburg centre, k-means with 10 restarts"),
    "G_MODE": img("gbg_mode20_zoom.png", "Gothenburg centre, majority filter 20 m"),
    "G_GAUSS": img("gbg_gauss10_zoom.png", "Gothenburg centre, Gaussian pre-filter 10 m"),
    "G_ICM": img("gbg_icm05_zoom.png", "Gothenburg centre, MRF beta 0.5"),
    "G_HDB": img("gbg_hdb_zoom.png", "Gothenburg centre, HDBSCAN"),
    "A_TEE": img("aut_tee_edges.png", "Austria crop, cluster edges of TEE k-means"),
    "A_GAUSS": img("aut_gauss10_edges.png", "Austria crop, cluster edges after Gaussian pre-filter 10 m"),
    "A_MODE": img("aut_mode20_edges.png", "Austria crop, cluster edges after majority filter 20 m"),
    "N_SPECKLE_BASE": pct(e_base["d_in"]), "N_SPECKLE_G10": pct(e_g10["d_in"]), "N_SPECKLE_M20": pct(e_m20["d_in"]),
    "N_SPECKLE_ICM": pct(e_icm["d_in"]),
    "N_RATIO_BASE": f"{e_base['ratio']:.2f}", "N_RATIO_G10": f"{e_g10['ratio']:.2f}", "N_RATIO_M20": f"{e_m20['ratio']:.2f}",
    "N_RATIO_ICM": f"{e_icm['ratio']:.2f}",
    "N_FH_BASE": pct(base["field_homog"]), "N_FH_M20": pct(aut["post mode r=20m x1"]["field_homog"]),
    "N_FH_G10": pct(aut["pre gauss sigma=10m"]["field_homog"]),
    "N_ARI_G_BASE": f(g_base["ARI_mean"]), "N_ARI_G_BASE_MIN": f(g_base["ARI_min"]), "N_ARI_G_RST": f(g_rst["ARI_mean"]),
    "N_ARI_A_BASE": f(base["ARI_mean"]), "N_ARI_A_RST": f(rst["ARI_mean"]),
    "N_ARI_G_MODE_TEE": f(gbg["TEE kmeans + post mode r=20m"]["ARI_mean"]),
    "N_HDB_MAX": pct(hd["max_share"]), "N_HDB_K": f"{hd['k_found']:.0f}", "N_HDB_NOISE": pct(hd["noise"]),
    "N_HDB_A_NOISE": pct(aut["HDBSCAN pca16 mcs=100"]["noise"]), "N_HDB_A_MAX": pct(aut["HDBSCAN pca16 mcs=100"]["max_share"]),
    "N_HDB_A_ARIMIN": f(aut["HDBSCAN pca16 mcs=100"]["ARI_min"]),
    "N_JS_KM5": f"{js['k=5']['tee_kmeans_ms']:.0f}", "N_JS_KM17": f"{js['k=17']['tee_kmeans_ms']:.0f}",
    "N_JS_MODE5": f"{js['k=5']['post_mode_20m_ms']:.0f}", "N_JS_MODE17": f"{js['k=17']['post_mode_20m_ms']:.0f}",
    "N_JS_GAUSS": f"{js['gauss_pre_10m_ms']:.0f}",
    "N_JS_H10": f"{js['hdbscan_core_ms']['10000'] / 1000:.1f} s", "N_JS_H20": f"{js['hdbscan_core_ms']['20000'] / 1000:.1f} s",
    "N_JS_TS2": f"{js['hdbscan_ts_ms']['2000'] / 60000:.1f} minutes",
    "OV_TABLE": ov_table(),
    "OV_STAB_GBG": stab(ov_g, OV_STAB, "Stability of over-cluster variants, Gothenburg"),
    "OV_STAB_AUT": stab(ov_a, OV_STAB, "Stability of over-cluster variants, Austria"),
    "G_OV64": img("gbg_over64_zoom.png", "Gothenburg centre, over-cluster 64 then Ward"),
    "G_OVAVG": img("gbg_overavg_zoom.png", "Gothenburg centre, over-cluster 64 then average linkage"),
    "G_OVBEST": img("gbg_overbest_zoom.png", "Gothenburg centre, over-cluster 32, Ward, refine, MRF"),
    "A_OVBEST": img("aut_overbest_edges.png", "Austria crop, cluster edges of the recommended pipeline"),
    "N_BEST_ARI_G": f(ov_g[BEST]["ARI_mean"]), "N_BEST_ARI_A": f(ov_a[BEST]["ARI_mean"]),
    "N_BEST_ARI_G_MIN": f(ov_g[BEST]["ARI_min"]), "N_BEST_SPECKLE": pct(ov_a[BEST]["d_in"]),
    "N_BEST_RATIO": f"{ov_a[BEST]['ratio']:.2f}", "N_BEST_FH": pct(ov_a[BEST]["field_homog"]),
    "N_REF_ARI_G": f(ov_g["over K0=32 + Ward + refine, sample 10k"]["ARI_mean"]),
    "N_REF_ARI_A": f(ov_a["over K0=32 + Ward + refine, sample 10k"]["ARI_mean"]),
    "N_OV32_ARI_G": f(ov_g["over K0=32 + Ward"]["ARI_mean"]), "N_OV32_ARI_A": f(ov_a["over K0=32 + Ward"]["ARI_mean"]),
    "N_OV128_ARI_A": f(ov_a["over K0=128 + Ward"]["ARI_mean"]),
    "N_AVG_MAX_G": pct(ov_g["over K0=64 + average link"]["max_share"]),
    "N_JS_OV32_S": f"{js['overcluster_sample_only_ms']['K0=32 sample=10000']['kmeans_ms']:.0f}",
    "N_JS_OV32_FULL": f"{js['overcluster_ms']['K0=32 sample=16000']['kmeans_ms'] / 1000:.1f}",
    "N_JS_WARD64": f"{js['overcluster_ms']['K0=64 sample=10000']['ward_merge_ms']:.0f}",
    "N_SEEDS_G": str(g_base["seeds"]), "N_SEEDS_A": str(base["seeds"]),
}
for k, v in subs.items():
    html = html.replace("{{" + k + "}}", v)
assert "{{" not in html, html[html.index("{{"):html.index("{{") + 40]
(R / "report.html").write_text(html)
print("wrote", R / "report.html", f"{len(html) / 1e6:.1f} MB")
