# TEE clustering benchmark results

This directory contains the experimental results produced while benchmarking and improving the clustering procedure used by the TESSERA Embeddings Explorer (TEE).

The experiments investigate three main directions:

1. reducing the sample size used by the current TEE K-means implementation;
2. replacing standard K-means with MiniBatch K-means;
3. evaluating Spectral Clustering as an alternative graph-based clustering method.

The main evaluation criteria are clustering quality, seed stability, and computational cost.

---

# 1. Sample-size benchmark

## Dataset

The sample-size experiments use the Austrian Crop benchmark:

- Year: 2022
- 17 ground-truth crop classes
- 42,789 labelled polygons
- 8,243,147 labelled pixels
- 128-dimensional TESSERA embeddings

## Experimental protocol

Each configuration was evaluated using 10 random seeds.

Clustering stability was evaluated using Adjusted Rand Index (ARI) between every pair of seed runs, resulting in 45 pairwise comparisons per configuration.

The original TEE sampling policy is:

```text
S = min(max(5000, 500*k), N)
```

where:

- `k` is the requested number of clusters;
- `N` is the number of available embeddings;
- `S` is the number of embeddings used to fit K-means.

The investigated efficiency-oriented policy is:

```text
S = min(max(5000, 300*k), N)
```

The goal of this experiment was to determine whether the fitting sample could be reduced without substantially changing clustering quality or stability.

## experiments/

Contains the unique raw experimental results.

### runs.csv

One row for every individual K-means run.

### stability_pairs.csv

Pairwise stability comparisons between seed runs.

### summary.csv

Mean and standard deviation for each `(k, sample_size)` configuration.

### manifest.csv

Identifies the source and experimental role of every configuration.

### metadata.json

Dataset and experimental configuration information.

## analysis/sample_size_vs_tee_baseline/

Contains the analysis used to determine how far the sample size can be reduced while retaining clustering performance relative to the original TEE policy.

## analysis/policy_300k_vs_500k/

Contains the direct comparison between the original `500*k` sampling policy and the candidate `300*k` policy.

The `300*k` policy should be interpreted as a promising efficiency-oriented alternative rather than as a universally optimal replacement.

For `k > 10`, the new rule reduces the fitting sample size by 40%.

Across the tested configurations, ARI, AMI, and seed stability generally remain close to those obtained using the original TEE rule. The main deviations occur at some individual values of `k`, particularly around `k = 20` for ARI and `k = 50` for stability.

The experiments therefore suggest that the current TEE sampling policy is conservative and that a smaller fitting sample can provide a useful quality-runtime trade-off.

---

# 2. MiniBatch K-means benchmark

A second benchmark was performed to investigate whether MiniBatch K-means could provide a more computationally efficient alternative to the standard K-means algorithm currently used in TEE.

The main objective was to determine whether MiniBatch K-means could preserve clustering quality and stability comparable to standard K-means while reducing fitting and total clustering time.

Unlike standard K-means, which updates the cluster centroids using the full fitting sample at each iteration, MiniBatch K-means updates them using smaller random batches of samples.

This can reduce computational cost for large datasets, but may also introduce additional stochasticity into the clustering.

## Experimental protocol

The original TEE sampling policy was kept unchanged:

```text
S = min(max(5000, 500*k), N)
```

This means that the experiment evaluates only the effect of replacing standard K-means with MiniBatch K-means, without simultaneously changing the number of embeddings used for fitting.

The following values of `k` were tested:

```text
k = 10, 17, 20, 30, 40, 50
```

For MiniBatch K-means, the following batch sizes were evaluated:

```text
batch_size = 256, 512, 1024, 2048
```

Each configuration was evaluated using 10 random seeds.

For each seed, standard K-means and MiniBatch K-means were evaluated using:

- the same value of `k`;
- the same TEE fitting sample size;
- the same sampled embeddings;
- the same random seed;
- the same K-means++ initial cluster centres.

Using the same fitting sample and initialization makes the comparison between the two algorithms more controlled.

Standard K-means used Lloyd iterations, while MiniBatch K-means updated the centroids using the specified mini-batch size.

The initial MiniBatch experiments used:

```text
max_no_improvement = 10
```

which allows MiniBatch K-means to stop when no improvement is observed for 10 consecutive mini-batches.

With 6 values of `k`, 4 MiniBatch configurations, one standard K-means baseline, and 10 random seeds, the main benchmark contains:

```text
6 * 5 * 10 = 300 runs
```

## Evaluation metrics

The same evaluation criteria used in the sample-size benchmark were retained in order to make the experiments directly comparable.

Clustering quality was evaluated using:

- Adjusted Rand Index (ARI) against the ground truth;
- Adjusted Mutual Information (AMI) against the ground truth;
- clustering distortion.

Clustering stability was evaluated using Adjusted Rand Index between every pair of seed runs.

With 10 seeds, each configuration produces 45 pairwise stability comparisons.

Computational performance was evaluated using:

- fitting time;
- final assignment time;
- total clustering time.

The number of MiniBatch iterations and mini-batch update steps was also recorded to analyse convergence behaviour.

For comparison with standard K-means, ARI, AMI, and stability are reported as retention percentages relative to the corresponding K-means baseline for the same value of `k`.

## experiments/minibatch_vs_kmeans/

Contains the complete experimental results of the MiniBatch K-means benchmark.

### config.json

Configuration used for the benchmark, including the tested values of `k`, batch sizes, seeds, and sampling policy.

### runs.csv

One row for every individual standard K-means and MiniBatch K-means run.

It contains clustering quality metrics, runtime measurements, distortion, and convergence information.

### stability_pairs.csv

Pairwise ARI comparisons between clustering results obtained using different random seeds.

### summary.csv

Mean and standard deviation of the measured metrics for each combination of algorithm, `k`, and batch size.

## analysis/minibatch_vs_kmeans/batch_size_comparison.csv

Contains the direct comparison between each MiniBatch configuration and the corresponding standard K-means baseline.

The results show that MiniBatch K-means generally preserves ground-truth clustering quality reasonably well.

ARI and AMI remain close to the standard K-means baseline for most tested configurations.

However, MiniBatch K-means produces substantially lower clustering stability across random seeds.

In particular, the smallest tested batch size:

```text
batch_size = 256
```

provided the most promising overall results, but retained only approximately 72--80% of the stability of standard K-means across the tested values of `k`.

Increasing the batch size did not consistently recover this loss of stability.

The experiment also showed little computational advantage from MiniBatch K-means.

Under the current TEE pipeline, standard K-means is already fitted on a relatively small sampled subset of the complete dataset:

```text
approximately 5,000--25,000 embeddings
```

depending on `k`.

As a result, the standard K-means fitting stage is already very fast, leaving limited opportunity for MiniBatch K-means to reduce runtime.

Furthermore, the final assignment stage still assigns all pixels to the resulting cluster centres regardless of the fitting algorithm.

Therefore, any reduction in MiniBatch fitting time has only a limited effect on total clustering time.

These results motivated an additional experiment to determine whether the lower stability of MiniBatch K-means was caused by its early-stopping criterion.

---

# 3. MiniBatch early-stopping test

The main MiniBatch benchmark used:

```text
max_no_improvement = 10
```

The benchmark showed that MiniBatch K-means preserved ARI and AMI relatively well but produced considerably lower stability across random seeds.

One possible explanation was that MiniBatch K-means was stopping before the centroids had sufficiently converged.

A second experiment was therefore performed to test whether allowing MiniBatch K-means to continue for longer could recover the stability of standard K-means.

## Experimental protocol

The batch size was fixed to the most promising value from the first experiment:

```text
batch_size = 256
```

The following early-stopping settings were compared:

```text
max_no_improvement = 10
max_no_improvement = 50
max_no_improvement = None
```

The experiment was performed for:

```text
k = 20, 40
```

These values were selected to examine the behaviour at two different fitting sample sizes:

```text
k = 20 -> S = 10000
k = 40 -> S = 20000
```

As in the main benchmark, 10 random seeds were used for each configuration.

A standard K-means baseline was also executed for each value of `k` in the same experimental environment.

The experiment therefore contains:

```text
2 * 4 * 10 = 80 runs
```

where the four configurations for each `k` consist of:

- one standard K-means baseline;
- MiniBatch with `max_no_improvement = 10`;
- MiniBatch with `max_no_improvement = 50`;
- MiniBatch with early stopping disabled.

## experiments/minibatch_vs_kmeans/early_stopping_test/

Contains the complete results of the early-stopping experiment.

### config.json

Configuration of the early-stopping experiment.

### runs.csv

Individual results for each seed and early-stopping configuration, including quality, runtime, distortion, number of iterations, and number of MiniBatch update steps.

### stability_pairs.csv

Pairwise stability comparisons between seed runs.

### summary.csv

Mean and standard deviation of all metrics for each early-stopping configuration.

## analysis/minibatch_vs_kmeans/early_stopping_comparison.csv

Contains the direct comparison of the three MiniBatch early-stopping settings with standard K-means.

Increasing `max_no_improvement` from 10 to 50 produced only a small improvement in clustering stability.

Disabling the early-stopping criterion entirely allowed MiniBatch K-means to perform substantially more updates and improved stability further.

For `k = 20`, stability retention increased approximately from:

```text
77.1% -> 79.2% -> 88.1%
```

for `max_no_improvement = 10`, `50`, and `None`, respectively.

For `k = 40`, the corresponding values were approximately:

```text
79.6% -> 81.2% -> 87.2%
```

However, stability still remained below the standard K-means baseline.

At the same time, computational cost increased substantially.

Without early stopping, MiniBatch K-means required much longer fitting times, eliminating the computational motivation for using MiniBatch in TEE.

For the tested configurations, disabling early stopping made the MiniBatch fitting stage approximately 29 times slower than standard K-means at `k = 20` and approximately 65 times slower at `k = 40`.

The experiment therefore indicates that early stopping contributes to the lower stability observed in MiniBatch K-means, but does not fully explain it.

The stochastic mini-batch updates themselves also appear to contribute to the stronger dependence on the random seed.

Overall, for the current TEE sampling strategy, MiniBatch K-means does not provide a favourable trade-off over standard K-means.

It preserves ARI and AMI reasonably well, but produces lower clustering stability without a meaningful runtime advantage.

Allowing MiniBatch K-means to converge for longer partially improves stability, but at a substantially higher computational cost.

For this reason, standard K-means remains preferable to MiniBatch K-means for the current TEE clustering pipeline.

---

# 4. Spectral Clustering benchmark

Spectral Clustering was evaluated as a graph-based alternative to the centroid-based K-means procedure currently used by TEE.

The motivation was to investigate whether the local structure of the TESSERA embedding space could be represented more effectively using a nearest-neighbour graph, potentially improving semantic clustering quality and reducing sensitivity to the random seed.

---

## Spectral Clustering method

Spectral Clustering is applied to a random sample of the TESSERA embeddings rather than directly to all pixels.

Constructing a nearest-neighbour graph over the complete image would be too expensive for interactive use, particularly for datasets containing hundreds of thousands of 128-dimensional embeddings.

For all Spectral Clustering experiments, the sample size is fixed to:

```text
S = 5000
```

Each sampled embedding is represented as a node in a k-nearest-neighbour graph constructed in the original 128-dimensional TESSERA embedding space.

Each node is connected to its `n_neighbors` nearest neighbours according to Euclidean distance.

The parameter `n_neighbors` therefore controls the density of the similarity graph:

- small values preserve only highly local relationships;
- larger values create a denser graph and connect more distant regions of the embedding space.

Spectral Clustering is performed using scikit-learn with:

```text
affinity = nearest_neighbors
assign_labels = cluster_qr
```

The number of output clusters remains fixed to the value requested by the user.

For the benchmark datasets:

```text
Austria:    k = 17
Gothenburg: k = 5
```

Spectral Clustering does not directly provide a prediction operation for points that were not included in the fitted graph.

To extend the sample clustering to the complete image, the mean 128-dimensional embedding of each spectral cluster is computed.

All remaining pixels are then assigned to the nearest cluster mean using squared Euclidean distance.

The complete pipeline is therefore:

```text
1. Randomly sample 5,000 embeddings.
2. Construct a nearest-neighbour graph.
3. Apply Spectral Clustering to the sample.
4. Compute the mean embedding of each spectral cluster.
5. Assign every pixel to the nearest cluster mean.
```

This preserves the same high-level structure as TEE's current K-means procedure: fit the clustering on a sample, then perform one final assignment pass over the complete image.

---

## Spectral benchmark datasets

Two datasets are used.

### Austria

The Austrian Spectral benchmark uses:

```text
tile = grid_16.75_48.35
year = 2022
```

The 2022 embedding year is used in order to match the year of the INVEKOS crop ground truth.

Dataset properties:

```text
Embedding grid:         1128 x 766 x 128
Valid pixels:           864,048
Labelled pixels:        438,419
Ground-truth classes:   17
Pixel resolution:       10 m
```

The ground truth is obtained from `austria.zip` and rasterised onto the embedding grid.

For spatial robustness checks, the tile is also divided vertically into two independent regions:

```text
west = left half of the tile
east = right half of the tile
```

The two halves are clustered independently rather than being used only as evaluation subsets of a clustering fitted on the complete tile.

### Gothenburg

The Gothenburg benchmark uses the `Gteborg` viewport with 2024 TESSERA embeddings.

Dataset properties:

```text
Embedding grid:         353 x 661 x 128
Pixels:                 233,333
Labelled pixels:        62,072
Ground-truth coverage:  26.6%
Ground-truth classes:   5
```

The five manually labelled classes are:

```text
RIVER
FOREST
FOOTBALL FIELD
BUILDINGS
STREET
```

Only manually drawn polygons are used as ground truth:

```text
type = polygon
```

Regions generated using TEE's similarity functionality are excluded:

```text
type = similarity
```

because these regions are themselves derived from the TESSERA embeddings and would make the evaluation circular.

The single point annotation is excluded, as is the polygon labelled:

```text
ELIAS and CHRISTOFER
```

which does not belong to the five benchmark classes.

After filtering, the Gothenburg ground truth contains:

```text
948 manual polygons
62,072 labelled pixels
```

---

## Spectral evaluation protocol

The final evaluation uses ten random seeds:

```text
[42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

The same seeds are used for the corresponding K-means baseline whenever paired comparisons are reported.

### Ground-truth quality

Clustering quality is measured on labelled pixels using:

- Adjusted Rand Index (ARI);
- Adjusted Mutual Information (AMI).

Both metrics are invariant to cluster identifiers and therefore do not require an explicit correspondence between cluster numbers and ground-truth class numbers.

### Seed stability

Seed stability is measured as the pairwise ARI between the complete cluster maps produced by different random seeds.

With 10 seeds, each configuration produces:

```text
45 pairwise comparisons
```

The experiments report:

- mean pairwise ARI;
- standard deviation;
- minimum pairwise ARI.

### Additional diagnostics

The Spectral experiments also record:

- runtime;
- largest-cluster share;
- number of connected components in the sampled nearest-neighbour graph.

Graph connectivity is particularly important because a highly disconnected graph can cause Spectral Clustering to produce unstable or highly unbalanced partitions.

Where appropriate, differences in ARI and AMI relative to the K-means baseline are computed seed-by-seed and reported with 95% confidence intervals.

---

# 5. Spectral fixed-neighbourhood sweep

The first Spectral experiment investigates sensitivity to the graph neighbourhood size.

The following values are evaluated:

```text
n_neighbors = 5, 10, 15, 20, 30, 40, 60, 120
```

The sample size remains fixed at:

```text
S = 5000
```

and each configuration is evaluated using the same ten seeds.

The experiment reveals that the appropriate graph density is strongly dataset-dependent.

---

## Austria fixed-neighbourhood results

The TEE K-means baseline on the Austrian tile gives approximately:

```text
ARI       = 0.241
AMI       = 0.384
Stability = 0.592
```

Sparse Spectral graphs substantially improve agreement with the crop labels.

For example:

```text
n_neighbors = 5

ARI       = 0.433
AMI       = 0.425
Stability = 0.648
```

The highest mean ARI is obtained with this very sparse graph.

The ARI values across the neighbourhood sweep are approximately:

```text
nn = 5      ARI = 0.433
nn = 10     ARI = 0.326
nn = 15     ARI = 0.308
nn = 20     ARI = 0.283
nn = 30     ARI = 0.258
nn = 40     ARI = 0.254
nn = 60     ARI = 0.237
nn = 120    ARI = 0.225
```

Neighbourhood sizes around 10--15 provide a more balanced trade-off between semantic quality and seed stability.

For example:

```text
nn = 10
ARI       = 0.326
AMI       = 0.428
Stability = 0.707

nn = 15
ARI       = 0.308
AMI       = 0.428
Stability = 0.741
```

As the neighbourhood becomes larger, the semantic-quality advantage over the K-means baseline progressively disappears.

---

## Gothenburg fixed-neighbourhood results

Gothenburg behaves very differently.

Small neighbourhood sizes frequently produce disconnected graphs and highly unbalanced clusterings.

For example:

```text
n_neighbors = 15
```

gives approximately:

```text
ARI             = 0.266
AMI             = 0.430
Stability       = 0.519
Largest cluster = 76.6%
```

The sampled graph is disconnected and some seeds place almost the entire viewport into a single cluster.

The clustering becomes substantially more stable as the graph density increases.

At:

```text
n_neighbors = 60
```

the result is approximately:

```text
ARI       = 0.432
AMI       = 0.481
Stability = 0.941
```

compared with the K-means baseline:

```text
ARI       = 0.448
AMI       = 0.435
Stability = 0.555
```

The graph is connected for all tested seeds and the largest cluster contains approximately 40% of the viewport rather than dominating the result.

Increasing `n_neighbors` further to 120 does not provide a clear benefit and increases runtime.

The initial sweep therefore shows:

```text
Austria:
small neighbourhood sizes perform best.

Gothenburg:
considerably larger neighbourhood sizes are required.
```

A single fixed `n_neighbors` value is therefore not suitable for both datasets.

---

# 6. Spectral paired comparisons

Paired-by-seed comparisons confirm the behaviour observed in the initial sweep.

For Gothenburg with `n_neighbors = 60`:

```text
Delta ARI = -0.0155
95% CI    = [-0.0424, +0.0115]

Delta AMI = +0.0462
95% CI    = [+0.0192, +0.0733]
```

The ARI difference is small and its interval includes zero, while the AMI improvement is consistently positive.

For Austria with `n_neighbors = 5`:

```text
Delta ARI = +0.1922
95% CI    = [+0.1421, +0.2423]

Delta AMI = +0.0411
95% CI    = [+0.0292, +0.0530]
```

Neighbourhoods of 10, 15, and 20 also produce positive paired differences in both ARI and AMI.

At larger neighbourhood sizes, the quality improvement progressively disappears.

---

# 7. Spectral spatial validation on Austria

The main candidate neighbourhood sizes were also evaluated independently on the western and eastern halves of the Austrian tile.

The K-means baselines are:

```text
West
ARI       = 0.211
AMI       = 0.371
Stability = 0.594

East
ARI       = 0.232
AMI       = 0.381
Stability = 0.601
```

With:

```text
n_neighbors = 10
```

Spectral Clustering gives:

```text
West
ARI       = 0.298
AMI       = 0.416
Stability = 0.706

East
ARI       = 0.399
AMI       = 0.443
Stability = 0.637
```

Paired comparisons for the western half give:

```text
Delta ARI = +0.0877
95% CI    = [+0.0514, +0.1239]

Delta AMI = +0.0444
95% CI    = [+0.0271, +0.0616]
```

For the eastern half:

```text
Delta ARI = +0.1671
95% CI    = [+0.1031, +0.2311]

Delta AMI = +0.0621
95% CI    = [+0.0477, +0.0765]
```

Both ARI and AMI therefore improve in both spatial halves.

This indicates that the Austrian result is not limited to the particular composition of the complete tile.

---

# 8. Spectral graph-connectivity analysis

Because the fixed-neighbourhood sweep shows that graph density must change between datasets, several label-free connectivity diagnostics were investigated.

The objective was to determine whether `n_neighbors` could be selected directly from the geometry of the sampled embeddings rather than from ground-truth performance.

---

## Complete-connectivity threshold

The first diagnostic defines the connectivity threshold as the minimum value of `n_neighbors` for which the complete 5,000-point sampled graph becomes connected.

Using ten independent calibration samples gives:

```text
Austria
mean   = 7.50
median = 7.5
range  = 5--10

Austria West
mean   = 6.20
median = 6
range  = 5--9

Austria East
mean   = 5.40
median = 5.5
range  = 4--6

Gothenburg
mean   = 21.50
median = 21
range  = 16--27
```

This clearly captures the structural difference between the datasets.

Gothenburg requires a substantially denser graph before complete connectivity is reached.

However, complete connectivity is sensitive to very small isolated groups of sampled points.

For example, the Austrian threshold varies from 5 to 10 even though most of the graph may already belong to one large component.

Complete connectivity was therefore considered too sensitive to use directly as the final automatic parameter-selection criterion.

---

# 9. Per-run adaptive Spectral experiment

A first adaptive strategy selected `n_neighbors` independently for every clustering run.

For each random sample:

1. the minimum fully connected neighbourhood was measured;
2. the Spectral neighbourhood was selected proportionally to that value;
3. Spectral Clustering was performed on the same sample.

This successfully produced much smaller neighbourhoods on Austria than on Gothenburg.

However, changing the random seed then changed both:

- the sampled embeddings;
- the Spectral `n_neighbors` parameter.

This introduced an additional source of variability.

For example, on Austria West the estimated full-connectivity threshold varied from 5 to 12, resulting in neighbourhood sizes between approximately 13 and 30.

The per-seed adaptive strategy therefore reduced reproducibility relative to using one fixed neighbourhood for the complete dataset.

The experiment indicated that adaptation should be performed once at the dataset or viewport level rather than independently for every clustering run.

---

# 10. Dataset-level Spectral calibration

A second adaptive experiment estimated the connectivity threshold using separate calibration samples and then kept the selected neighbourhood fixed across all evaluation seeds.

The calibration seeds were separated from the evaluation seeds.

Calibration seeds:

```text
1001, 1002, ...
```

Evaluation seeds:

```text
42, 1, 2, ..., 9
```

Using only three calibration samples remained unreliable when full graph connectivity was used.

For example, on the complete Austrian tile, three calibration samples produced:

```text
7, 10, 9
```

giving a median connectivity threshold of 9 and selecting a much denser graph than the neighbourhoods preferred by the initial sweep.

A more robust graph statistic was therefore required.

---

# 11. 99% largest-connected-component criterion

A new connectivity measure was introduced to reduce sensitivity to small isolated groups.

Let `n99` denote the smallest neighbourhood size for which at least 99% of the 5,000 sampled embeddings belong to the largest connected component.

In other words:

```text
n99 = minimum n_neighbors such that
      largest connected component >= 99% of sample
```

Unlike complete connectivity, this criterion allows a small number of isolated embeddings to remain outside the main graph.

The estimate is computed using ten independent calibration seeds:

```text
1001, 1002, ..., 1010
```

These seeds are not used in the final clustering evaluation.

The resulting distributions are:

```text
Austria
mean   = 3.20
median = 3
std    = 0.42
range  = 3--4

Austria West
mean   = 3.80
median = 4
std    = 0.79
range  = 3--5

Austria East
mean   = 3.60
median = 4
std    = 0.52
range  = 3--4

Gothenburg
mean   = 18.10
median = 18
std    = 2.56
range  = 14--22
```

The `n99` statistic is substantially more stable than the full-connectivity threshold, especially on the Austrian dataset.

It also preserves the important structural difference between the two datasets: Gothenburg requires a much denser graph than Austria.

---

# 12. Final connectivity-based Spectral heuristic

The final candidate rule is:

```text
n_neighbors = ceil(2.5 * median(n99))
```

where the median is calculated across the ten independent calibration samples.

This gives:

```text
Austria
median n99 = 3
n_neighbors = 8

Austria West
median n99 = 4
n_neighbors = 10

Austria East
median n99 = 4
n_neighbors = 10

Gothenburg
median n99 = 18
n_neighbors = 45
```

The factor `2.5` is motivated by the exploratory fixed-neighbourhood sweep.

The experiments showed that a graph moderately denser than the minimum required to connect almost all sampled embeddings generally provides a better quality-stability compromise.

However, `2.5` should be interpreted as an exploratory heuristic rather than as a universally optimised constant.

The final parameter-selection procedure has three important properties:

- it does not use the evaluation ground-truth labels;
- it uses calibration seeds that are independent of the evaluation seeds;
- once calibrated, `n_neighbors` remains fixed across all evaluation runs of the same dataset.

---

# 13. Final Spectral results

The final `n99`-based neighbourhoods are evaluated using the same ten evaluation seeds as the corresponding TEE K-means baselines.

| Dataset | n_neighbors | K-means ARI | Spectral ARI | Delta ARI | K-means AMI | Spectral AMI | Delta AMI | Spectral stability |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Austria | 8 | 0.241 | 0.334 | +0.093 | 0.384 | 0.428 | +0.044 | 0.729 |
| Austria West | 10 | 0.211 | 0.298 | +0.088 | 0.371 | 0.416 | +0.044 | 0.706 |
| Austria East | 10 | 0.232 | 0.399 | +0.167 | 0.381 | 0.443 | +0.062 | 0.637 |
| Gothenburg | 45 | 0.448 | 0.422 | -0.026 | 0.435 | 0.474 | +0.039 | 0.921 |

The paired-by-seed 95% confidence intervals are:

### Austria

```text
Delta ARI = +0.0935
95% CI    = [+0.0611, +0.1258]

Delta AMI = +0.0442
95% CI    = [+0.0290, +0.0595]
```

### Austria West

```text
Delta ARI = +0.0877
95% CI    = [+0.0514, +0.1239]

Delta AMI = +0.0444
95% CI    = [+0.0271, +0.0616]
```

### Austria East

```text
Delta ARI = +0.1671
95% CI    = [+0.1031, +0.2311]

Delta AMI = +0.0621
95% CI    = [+0.0477, +0.0765]
```

### Gothenburg

```text
Delta ARI = -0.0258
95% CI    = [-0.0513, -0.0004]

Delta AMI = +0.0392
95% CI    = [+0.0131, +0.0654]
```

---

## Interpretation of the final Spectral results

On the Austrian crop benchmark, the final connectivity-calibrated Spectral configuration improves both ground-truth metrics.

ARI increases from:

```text
0.241 -> 0.334
```

and AMI increases from:

```text
0.384 -> 0.428
```

Seed stability also improves:

```text
0.592 -> 0.729
```

The improvement is preserved when the western and eastern halves are clustered independently.

This provides evidence that the improvement is not restricted to one particular spatial composition of the Austrian tile.

The Gothenburg result is different.

AMI improves from:

```text
0.435 -> 0.474
```

and seed stability increases substantially from:

```text
0.555 -> 0.921
```

However, ARI decreases slightly:

```text
0.448 -> 0.422
```

with a paired difference of:

```text
Delta ARI = -0.0258
95% CI    = [-0.0513, -0.0004]
```

Spectral Clustering therefore provides a clear semantic-quality improvement on the Austrian crop benchmark and a large reproducibility improvement on Gothenburg, but does not improve every ground-truth metric simultaneously on the urban viewport.

---

# 14. Spectral runtime

The final Spectral configurations take approximately:

```text
Austria, nn = 8
1.19 s

Austria West, nn = 10
0.93 s

Austria East, nn = 10
0.91 s

Gothenburg, nn = 45
1.13 s
```

These timings were measured using the Python experimental implementation.

They should not be compared directly with the final JavaScript/browser timings of the production TEE implementation unless Spectral Clustering is ported and benchmarked under the same environment.

The graph-connectivity calibration is intended as a one-time operation for a dataset or viewport rather than as a cost repeated for every clustering run.

The ten-sample `n99` diagnostic over all four evaluated datasets completed in approximately:

```text
14 s
```

in the Python experimental environment.

---

# 15. Spectral result files

The Spectral experiment results are stored under:

```text
results/spectral_experiment/results/
```

The main files are:

```text
spectral_nn_sweep_gbg.json
kmeans_baseline_current_gbg.json
gbg_kmeans_vs_spectral_gt.json
aut2022_kmeans_vs_spectral_nn.json
aut2022_spectral_validation_west.json
aut2022_spectral_validation_east.json
spectral_connectivity_thresholds.json
spectral_adaptive_alpha2_5.json
spectral_adaptive_per_seed.json
spectral_calibrated_fixed.json
spectral_lcc99_thresholds.json
spectral_lcc99_final.json
spectral_lcc99_final_with_ci.json
```

The most important files for the final result are:

### spectral_lcc99_thresholds.json

Contains the independent `n99` connectivity calibration using seeds 1001--1010.

### spectral_lcc99_final.json

Contains the final evaluation of the neighbourhood sizes selected using the `n99` heuristic.

### spectral_lcc99_final_with_ci.json

Contains:

- per-seed Spectral results;
- comparison with the corresponding K-means baseline;
- paired ARI differences;
- paired AMI differences;
- 95% confidence intervals;
- final stability measurements.

This is the main file containing the final Spectral result.

---

# 16. Spectral scripts

The scripts used for the Spectral experiments are stored under:

```text
results/spectral_experiment/scripts/
```

The scripts include:

```text
run_spectral_nn_sweep.py
run_gbg_gt_comparison.py
run_aut_spectral_nn_sweep.py
run_aut_half_validation.py
diagnose_spectral_connectivity.py
diagnose_spectral_lcc99.py
run_spectral_adaptive_test.py
run_spectral_adaptive_per_seed.py
run_spectral_calibrated_fixed.py
run_spectral_lcc99_final.py
run_spectral_lcc99_final_ci.py
tsh.py
```

The main scripts required to reproduce the final adaptive-neighbourhood result are:

```text
diagnose_spectral_lcc99.py
run_spectral_lcc99_final.py
run_spectral_lcc99_final_ci.py
```

`diagnose_spectral_lcc99.py` estimates the connectivity statistic used to select `n_neighbors`.

`run_spectral_lcc99_final.py` evaluates the final selected neighbourhoods.

`run_spectral_lcc99_final_ci.py` performs the final paired comparison with the K-means baseline and calculates the 95% confidence intervals.

`tsh.py` contains the common experimental functions used for:

- dataset loading;
- TEE K-means baseline reproduction;
- squared-distance calculations;
- Spectral sample clustering;
- cluster-mean construction;
- spatial evaluation utilities.

---

# 17. Spectral limitations

The Spectral experiments have several limitations.

First, the scaling factor:

```text
2.5
```

used in the final adaptive rule was motivated by the exploratory neighbourhood sweep.

It has not been independently validated as a universal constant.

The final parameter-selection strategy should therefore be described as a connectivity-based heuristic rather than as an optimal automatic rule.

Second, only two geographic datasets are evaluated:

- one Austrian agricultural tile;
- one Gothenburg urban viewport.

Additional regions would be required to determine whether the relationship between `n99` and an effective Spectral neighbourhood generalises to other landscapes.

Third, the Gothenburg ground truth covers only:

```text
26.6%
```

of the viewport.

ARI and AMI therefore evaluate the manually labelled subset, whereas stability is evaluated on the complete cluster map.

Fourth, the Spectral implementation is currently evaluated in Python.

Production integration into TEE would require an implementation suitable for the browser/worker environment and a runtime comparison under the same conditions as the existing TEE clustering methods.

Finally, Spectral Clustering still requires a user-specified number of clusters `k`.

The method changes how the partition is produced but does not solve the separate problem of automatically selecting the number of clusters.

---

# Overall conclusions

The experiments lead to three main conclusions.

## Sample size

TEE's original sampling rule:

```text
S = min(max(5000, 500*k), N)
```

appears conservative.

Reducing the proportional term from `500*k` to `300*k` substantially reduces the fitting sample while generally preserving similar clustering quality and stability.

The `300*k` rule is therefore a promising efficiency-oriented alternative, although some individual configurations show larger deviations and additional validation would be required before treating it as a universal replacement.

## MiniBatch K-means

MiniBatch K-means does not provide a favourable quality-runtime trade-off under the current TEE pipeline.

ARI and AMI remain close to standard K-means, but seed stability is considerably lower.

Because TEE already performs K-means on a relatively small sampled subset, MiniBatch provides little runtime advantage.

Allowing MiniBatch to run for substantially longer improves stability only partially and removes its computational motivation.

Standard K-means therefore remains preferable to MiniBatch K-means in the current pipeline.

## Spectral Clustering

Spectral Clustering shows that the structure of the TESSERA embedding graph contains useful information that is not fully captured by the current centroid-based K-means partition.

On the Austrian crop benchmark, the final connectivity-calibrated Spectral configuration improves:

- ARI;
- AMI;
- seed stability.

The improvement is preserved when the western and eastern halves of the tile are clustered independently.

On Gothenburg, Spectral Clustering substantially improves reproducibility and AMI, although this comes with a small reduction in ARI.

The experiments also show that a single fixed nearest-neighbour graph density does not generalise across the two datasets.

Sparse graphs work well for the Austrian agricultural embeddings, while the heterogeneous Gothenburg viewport requires a considerably denser graph.

The `n99` connectivity statistic provides a label-free way to detect this difference and to select neighbourhood sizes in the useful range for both datasets.

However, the scaling factor used to transform `n99` into the final `n_neighbors` value remains exploratory.

Further evaluation on additional regions and a production implementation would therefore be required before adopting the connectivity-based Spectral heuristic as the default TEE clustering algorithm.