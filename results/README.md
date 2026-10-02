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
