# TEE clustering benchmark results

This directory contains the consolidated results used for the
sample-size analysis on the Austrian Crop dataset.

## Dataset

- Austrian Crop dataset
- Year: 2022
- 17 ground-truth crop classes
- 8,243,147 labelled pixels
- 128-dimensional TESSERA embeddings

## Experimental protocol

Each configuration was evaluated using 10 random seeds
(seed 1 through 10).

Clustering stability was evaluated using Adjusted Rand Index
between every pair of seed runs, resulting in 45 pairwise
comparisons per configuration.

The original TEE sampling policy is:

    S = min(max(5000, 500*k), N)

The investigated efficiency-oriented policy is:

    S = min(max(5000, 300*k), N)

## experiments/

Contains all unique raw experimental results.

- runs.csv:
  one row for every individual K-means run.

- stability_pairs.csv:
  pairwise stability comparisons between seed runs.

- summary.csv:
  mean and standard deviation for each (k, sample_size)
  configuration.

- manifest.csv:
  identifies the source and experimental role of every
  configuration.

- metadata.json:
  dataset and experimental configuration information.

## analysis/sample_size_vs_tee_baseline/

Contains the analysis used to determine how far sample size
can be reduced while retaining clustering performance relative
to the original TEE policy.

## analysis/policy_300k_vs_500k/

Direct comparison between the original 500*k policy and the
candidate 300*k policy.

The 300*k policy should be interpreted as a promising
efficiency-oriented alternative rather than a universally
optimal replacement. In the tested configurations it reduced
the sample size substantially while generally retaining similar
quality and stability, with a notable limitation around k=20.

## **MiniBatch K-means benchmark**

A second benchmark was performed to investigate whether MiniBatch K-means
could provide a more computationally efficient alternative to the standard
K-means algorithm currently used in TEE.

The main objective of this experiment was to determine whether MiniBatch
K-means could preserve clustering quality and stability comparable to
standard K-means while reducing fitting and total clustering time.

Unlike standard K-means, which updates the cluster centroids using the full
fitting sample at each iteration, MiniBatch K-means updates them using smaller
random batches of samples. This can reduce computational cost for large
datasets, but may also introduce additional stochasticity in the clustering.

### **Experimental protocol**

The original TEE sampling policy was kept unchanged:

    S = min(max(5000, 500*k), N)

This means that the experiment evaluates only the effect of replacing
standard K-means with MiniBatch K-means, without simultaneously changing
the number of embeddings used for fitting.

The following values of k were tested:

    k = 10, 17, 20, 30, 40, 50

For MiniBatch K-means, the following batch sizes were evaluated:

    batch_size = 256, 512, 1024, 2048

Each configuration was evaluated using 10 random seeds
(seed 1 through 10).

For each seed, standard K-means and MiniBatch K-means were evaluated using:

- the same value of k
- the same TEE fitting sample size
- the same sampled embeddings
- the same random seed
- the same k-means++ initial cluster centers

Using the same fitting sample and initialization makes the comparison
between the two algorithms more controlled.

Standard K-means used Lloyd iterations, while MiniBatch K-means updated
the centroids using the specified mini-batch size.

The initial MiniBatch experiments used:

    max_no_improvement = 10

which allows MiniBatch K-means to stop when no improvement is observed
for 10 consecutive mini-batches.

With 6 values of k, 4 MiniBatch configurations, one standard K-means
baseline, and 10 random seeds, the main benchmark contains:

    6 * 5 * 10 = 300 runs

### **Evaluation metrics**

The same evaluation criteria used in the sample-size benchmark were retained
in order to make the experiments directly comparable.

Clustering quality was evaluated using:

- Adjusted Rand Index (ARI) against the ground truth
- Adjusted Mutual Information (AMI) against the ground truth
- clustering distortion

Clustering stability was evaluated using Adjusted Rand Index between every
pair of seed runs. With 10 seeds, each configuration produces 45 pairwise
stability comparisons.

Computational performance was evaluated using:

- fitting time
- final assignment time
- total clustering time

The number of MiniBatch iterations and mini-batch update steps was also
recorded to analyse its convergence behaviour.

For the comparison with standard K-means, ARI, AMI, and stability are
reported as retention percentages relative to the corresponding K-means
baseline for the same value of k.

---

### **experiments/minibatch_vs_kmeans/**

Contains the complete experimental results of the MiniBatch K-means
benchmark.

- `config.json`:
  configuration used for the benchmark, including the tested values of k,
  batch sizes, seeds, and sampling policy.

- `runs.csv`:
  one row for every individual standard K-means and MiniBatch K-means run.
  It contains clustering quality metrics, runtime measurements, distortion,
  and convergence information.

- `stability_pairs.csv`:
  pairwise ARI comparisons between clustering results obtained using
  different random seeds.

- `summary.csv`:
  mean and standard deviation of the measured metrics for each combination
  of algorithm, k, and batch size.

### **analysis/minibatch_vs_kmeans/batch_size_comparison.csv**

Contains the direct comparison between each MiniBatch configuration and
the corresponding standard K-means baseline.

The results show that MiniBatch K-means generally preserves ground-truth
clustering quality well. ARI and AMI remain close to the standard K-means
baseline for most tested configurations.

However, MiniBatch K-means produces substantially lower clustering stability
across random seeds. In particular, the smallest tested batch size
(`batch_size = 256`), which provided the most promising overall results,
retained approximately 72--80% of the stability of standard K-means across
the tested values of k.

Increasing the batch size did not consistently recover this loss of
stability.

The experiment also showed little computational advantage from MiniBatch
K-means. Under the current TEE pipeline, standard K-means is already fitted
on a relatively small sampled subset of the complete dataset
(approximately 5,000--25,000 embeddings depending on k). As a result,
the standard K-means fitting stage is already very fast, leaving limited
opportunity for MiniBatch K-means to reduce runtime.

Furthermore, the final assignment stage still assigns all labelled pixels
to the resulting cluster centres regardless of the fitting algorithm.
Therefore, any reduction in MiniBatch fitting time has only a limited
effect on the total clustering time.

These results motivated an additional experiment to determine whether the
lower stability of MiniBatch K-means was caused by its early-stopping
criterion.

---

## **MiniBatch early-stopping test**

The main MiniBatch benchmark used:

    max_no_improvement = 10

The benchmark showed that MiniBatch K-means preserved ARI and AMI relatively
well but produced considerably lower stability across random seeds.

One possible explanation was that MiniBatch K-means was stopping before
the centroids had sufficiently converged.

A second experiment was therefore performed to test whether allowing
MiniBatch K-means to continue for longer could recover the stability of
standard K-means.

### **Experimental protocol**

The batch size was fixed to the most promising value from the first
experiment:

    batch_size = 256

The following early-stopping settings were compared:

    max_no_improvement = 10
    max_no_improvement = 50
    max_no_improvement = None

The experiment was performed for:

    k = 20, 40

The values k = 20 and k = 40 were selected to examine the behaviour at
two different fitting sample sizes:

    k = 20 -> S = 10000
    k = 40 -> S = 20000

As in the main benchmark, 10 random seeds were used for each configuration.

A standard K-means baseline was also executed for each value of k in the
same experimental environment.

The experiment therefore contains:

    2 * 4 * 10 = 80 runs

where the four configurations for each k consist of one standard K-means
baseline and three MiniBatch early-stopping settings.

### **experiments/minibatch_vs_kmeans/early_stopping_test/**

Contains the complete results of the early-stopping experiment.

- `config.json`:
  configuration of the early-stopping experiment.

- `runs.csv`:
  individual results for each seed and early-stopping configuration,
  including quality, runtime, distortion, number of iterations, and
  number of MiniBatch update steps.

- `stability_pairs.csv`:
  pairwise stability comparisons between seed runs.

- `summary.csv`:
  mean and standard deviation of all metrics for each early-stopping
  configuration.

### **analysis/minibatch_vs_kmeans/early_stopping_comparison.csv**

Contains the direct comparison of the three MiniBatch early-stopping
settings with standard K-means.

Increasing `max_no_improvement` from 10 to 50 produced only a small
improvement in clustering stability.

Disabling this early-stopping criterion entirely (`max_no_improvement=None`)
allowed MiniBatch K-means to perform substantially more updates and improved
stability further. However, stability still remained below the standard
K-means baseline.

At the same time, the computational cost increased substantially. Without
early stopping, MiniBatch K-means required much longer fitting times,
eliminating the computational motivation for using MiniBatch in TEE.

The experiment therefore indicates that early stopping contributes to the
lower stability observed in MiniBatch K-means, but does not fully explain it.
The stochastic mini-batch updates themselves also make the final clustering
more dependent on the random seed.

Overall, for the current TEE sampling strategy, MiniBatch K-means does not
provide a favourable trade-off over standard K-means. It preserves ARI and
AMI reasonably well, but produces lower clustering stability without a
meaningful runtime advantage. Allowing MiniBatch K-means to converge for
longer partially improves stability, but at a substantially higher
computational cost.

For this reason, standard K-means remains the preferred algorithm for the
current TEE clustering pipeline.