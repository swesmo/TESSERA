# Clustering experiments (Christopher)

Python/Node experiments on TEE's k-means and the extensions in Section 6 of the report.
All numbers below are measured with these scripts; full write-up in
`spatial_hdbscan/results/report.html`.

| Folder | What |
|---|---|
| `seed_instability/` | NumPy port of TEE's browser k-means (`public/js/segmentation.js`) and RVQ decode; seed 42 vs 7 maps (midterm slide 02) and pairwise disagreement over seeds 1-20 |
| `spatial_hdbscan/` | Harness `tsh.py` (data loading, TEE baseline, metrics), spatial smoothing (Gauss, bilateral, majority, MRF/ICM), restarts, HDBSCAN, over-cluster + Ward merge + refine, JS timing benchmark |
| `dbscan/` | DBSCAN on PCA 8/16/32/128 with minPts = 2d and eps from the k-distance curve, noise analysis, JS benchmark |

Each folder keeps its outputs in `results/` (JSON metrics, logs, `.npz` label maps, figures).

The methods that came out of these experiments are also in TEE itself: the auto-label panel has
a "method" selector (`public/js/segmentation.js`, `SEG_METHODS`): k-means (unchanged default),
Stable (over-cluster 32 + Ward merge + refine), Stable + smooth (+ MRF, beta 0.25) and
Stable + Gaussian (10 m pre-filter). Tests: `node validation/test_segmentation_methods.mjs`.

## Setup

Run with TEE's local venv (numpy, scipy, scikit-learn, geopandas, rasterio, Pillow):
`~/tee-local/venv/bin/python <script>.py`. Node scripts: `node <script>.mjs`.

Data is read from TEE's local cache in `~/tee_data`; preprocessed arrays are written to
`~/tee_data/harness-cache/`.

- **Gothenburg 2024**: open the viewport "Gteborg" (year 2024) once in TEE so that
  `~/tee_data/vectors/Gteborg/2024/` exists. No labels yet.
- **Austria**: tile `grid_16.75_48.35` in `~/tee_data/compute-cache/tessera-eval/tiles/`
  (2024 via TEE's compute cache, 2022 downloaded with `geotessera`, 110 MB). Labels come
  from `austria.zip` in the repo root (INVEKOS 2022, 17 classes). Use 2022 embeddings:
  with 2024 the labels do not match (crop rotation, AMI 0.08 vs 0.384).
- `bench_js.mjs` and `bench_dbscan.mjs` read `~/tee_data/harness-cache/gbg.f32`, the raw
  float32 bytes of `gbg2024_emb.npy` (written by `tsh.gothenburg()`). Create it once with
  `python -c "import numpy as np, pathlib; d = pathlib.Path.home()/'tee_data/harness-cache'; np.load(d/'gbg2024_emb.npy').astype('<f4').tofile(d/'gbg.f32')"`.

## Run

```
cd seed_instability && python make_seed_maps.py [seedA seedB]; python seed_stability.py
cd spatial_hdbscan  && python run_experiments.py [gbg|aut] [n_seeds] [year]
                       YEAR=2022 python run_overcluster.py [gbg|aut] [n_seeds]
                       python run_hdbscan_extra.py [gbg|aut] [n_seeds]
                       python make_figures.py && python build_report.py
                       node bench_js.mjs
cd dbscan           && python run_dbscan.py [gbg|aut22] [n_seeds]
                       python analyze_noise.py [gbg|aut22]; python make_figs.py
                       python export_js_params.py && node bench_dbscan.mjs
```

## Final method table (11 Oct 2026)

`spatial_hdbscan/run_final.py` re-runs every method with 20 seeds and stores every metric per
seed (`results/final_<tag>.json`): TEE baseline, restarts, the smoothing sweeps (Gaussian sigma,
MRF beta, majority radius), over-clustering (K0 x sample size, iterations, fit in a PCA
projection), full-N Lloyd, PCA / z-score, superpixels. Tags: `aut2022`, `gbg`, `aut2022_west` /
`_east` (the two halves of the tile, clustered separately: hold-out check of the filter sizes),
`*_100seeds` (the four main methods with 100 seeds).

```
cd spatial_hdbscan
YEAR=2022 python run_final.py aut 20            # all families, ~15 min
python run_final.py gbg 20
YEAR=2022 python run_final.py aut 20 west       # and: east
TAG=_100seeds python run_final.py gbg 100 all key
python analyze_final.py                         # mean +- sd, paired differences, bootstrap -> results/final_summary.md
python -c "import numpy as np, pathlib; d = pathlib.Path.home()/'tee_data/harness-cache'; np.load(d/'austria_grid_16.75_48.35_2022.npz')['emb'].astype('<f4').tofile(d/'aut2022.f32')"
node bench_port.mjs gbg 15; node --max-old-space-size=8192 bench_port.mjs aut 9   # runtime of the ported worker
node port_maps.mjs                              # label maps from TEE's worker for the map figure
node bench_browser.mjs                          # the same methods timed in the TEE viewer in (headless) Chrome
python -W ignore run_spectral.py aut 10         # spectral clustering of a sample: neighbours, sample size, k (also: gbg, west, east)
python make_final_figs.py                       # results/fig/final_*.pdf
python make_report_tables.py                    # LaTeX rows of the report tables
python -m unittest discover -s tests -v         # 25 tests, see below
```

Write the logs outside OneDrive (e.g. `> ~/tee_data/harness-cache/logs/x.txt`): a process that
keeps a log file open inside the synced folder dies silently when OneDrive touches the file.

Gothenburg labels: `python merge_labels.py labels.geojson EXPORT [EXPORT ...]` combines the label
exports of several people (TEE's shapefile ZIP, e.g. `gothenburg.zip` on main, or GeoJSON) into
one file. It reports the pixels per class, pixels that two exports label differently, and how
much of each assigned region (a feature named after team members) is labelled. Then
`GBG_LABELS=labels.geojson K=<number of classes> TAG=_labels python run_final.py gbg 20`
(also `run_spectral.py gbg`) computes AMI / ARI on the labelled pixels; `tsh.load_geojson_labels`
rasterises the labels onto the viewport grid (checked pixel-exactly with a file in TEE's export
format). Everyone should create the viewport from `gothenburg_viewport.geojson`, so that all
exports and the cached embeddings share one pixel grid.

Only hand-drawn polygons are ground truth. TEE's exports mix three kinds of features (column
`type`): `polygon` (drawn by hand), `similarity` (pixels TEE matched to a seed pixel by embedding
distance) and, with "classified pixels" ticked, `classified` (TEE's nearest-centroid
classification of every pixel). The last two are derived from the embeddings that the clustering
is scored on, so scoring against them is circular. Keep `type == "polygon"` before merging:
`python -c "import geopandas as g; d = g.read_file('export.geojson'); d[d.type == 'polygon'][['name', 'geometry']].to_file('hand.geojson', driver='GeoJSON')"`.

Results on the partial labels of 12 Oct (PRELIMINARY: 5 classes, viewport shifted by ~450 m
against the one the embeddings were cached for; re-run when the labelling is complete):
- `results/final_gbg_labels_hand.*`: hand-drawn polygons only, 62,142 pixels = 22 % of the
  viewport. k = 5, 20 seeds: baseline AMI 0.467 +- 0.033 / ARI 0.468 / stability 0.63;
  over-cluster 32 + MRF 0.497 +- 0.001 / 0.490 / 0.95. Use these.
- `results/final_gbg_labels.*`, `results/spectral_gbg_labels.json`: scored on all 81,295 labelled
  pixels, i.e. INCLUDING 28,855 similarity-matched FOREST pixels. The scores are inflated
  (baseline AMI 0.521); kept only for comparison.

Tests (`spatial_hdbscan/tests/test_tsh.py`): small inputs whose result can be worked out by
hand (three groups of five points, Ward merges of three and four centroids, one isolated pixel
for the MRF and the majority filter, an impulse for the Gaussian filter, a two-field grid for
the field metrics), and comparisons with TEE's own worker code run in Node: the line-by-line
Python port (`tsh.tee_kmeans_exact`, TEE's mulberry32) returns the same cluster for every pixel,
the harness (`tsh.tee_kmeans`, NumPy generator) has the same stability within sampling noise,
and the JavaScript port of Ward merge / MRF / Gaussian filter / "Stable" matches the Python code.

## Main findings

Numbers from the final runs of 11 Oct 2026 (20 seeds, 100 for the stability of the main methods;
runtime = TEE's worker code in Node, median per run). Details: `spatial_hdbscan/results/final_summary.md`.

- TEE k-means is seed-sensitive. TEE's own worker on Gothenburg, k = 5, seeds 1-100: two seeds put
  21.6 % of the pixels into different clusters on average (max 57 %), mean ARI 0.67. (Seeds 1-20
  alone give 12 % / 37 % / 0.79, the numbers of the midterm slides.)
- Over-cluster 32 (sample 300 K0) -> Ward merge -> refine is the most stable method: mean ARI
  between seeds 0.62 -> 0.96 (Gothenburg) and 0.60 -> 0.73 (Austria 2022); 1.8 % of the pixels
  differ between two seeds with the ported code. 10 restarts: 0.94 / 0.67. K0 (24-128) and the
  sample size (150 / 300 / 500 K0) make no difference within +-0.03.
- Smoothing roughly halves speckle (22.6 % -> 13-14 %) and raises the AMI with the crop classes
  from 0.380 to 0.401 (MRF, beta 0.25) or 0.410 (Gaussian 10 m); the ARI is unchanged (MRF) or
  slightly lower (Gaussian). sigma = 10 m is best on the tile and on both halves separately.
- Proposed for TEE: over-cluster 32 + Ward + refine + MRF 0.25. Runtime on the Gothenburg viewport:
  0.74 s (k = 5) and 1.10 s (k = 17), against 0.21 s and 0.71 s for the current k-means.
  In the TEE viewer in Chrome 154 (run start until the cluster map is drawn): 0.77 s and 1.13 s against
  0.27 s and 0.73 s; same cluster sizes as in Node.
- No gain: Lloyd iterations on all pixels, k-means on PCA or z-scored embeddings, SLIC
  superpixels, HDBSCAN as the merge step (its number of groups follows K0, not the data).
- HDBSCAN on pixels gives one 93 % cluster or mostly noise; DBSCAN is degenerate on both datasets
  (noise = mixed / edge pixels). Both dropped.
- Spectral clustering (12 Oct, Python only, 10 seeds): spectral clustering of a 5,000-pixel sample (15-NN
  graph) with nearest-mean assignment raises AMI / ARI on the Austrian tile from 0.384 / 0.241 to 0.428 / 0.308
  at k = 17 (0.443 / 0.319 with the MRF; confirmed on both halves and at k = 8, no gain at k = 5). On
  Gothenburg the 15-NN graph is disconnected and one cluster keeps 82 % of the viewport at k = 5; with >= 60
  neighbours it is balanced and stable, but that setting removes the gain on the Austrian tile. Open: a
  neighbour rule that works on both, the Gothenburg labels, and the cost in the browser.
