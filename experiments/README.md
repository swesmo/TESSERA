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

## Main findings

- TEE k-means is seed-sensitive: Gothenburg k=5, seeds 1-20, mean pairwise disagreement 12 %,
  worst 37 %, mean ARI 0.79 (min 0.49).
- Stability comes from restarts (ARI 0.63 -> 0.93), not smoothing. Over-cluster 32 (10k
  sample) -> Ward -> refine is the most stable (ARI Gothenburg 0.96, Austria 2024 0.83).
- Smoothing roughly halves speckle at 10-500 ms in JS. On Austria 2022 the best combination
  is a Gauss 10 m pre-filter + over-cluster/Ward/refine (AMI 0.409, ARI 0.59 -> 0.76).
- HDBSCAN gives one 93 % cluster or mostly noise; DBSCAN is degenerate on both datasets
  (noise = mixed/edge pixels). Both dropped.
