"""Numbers in the slide-02 footnote: how many pixels change cluster between two seeds.

Runs TEE's k-means (see make_seed_maps.py) for seeds 1-20 on Gothenburg 2024, k = 5,
matches the clusters of every pair of runs (best one-to-one mapping) and reports the
share of pixels whose cluster differs. TEE itself has no such comparison; it only shows
the cluster shares of one run in the Auto-label panel.
    ~/tee-local/venv/bin/python seed_stability.py
"""
import itertools
import numpy as np
from make_seed_maps import K, load_embeddings, tee_kmeans


def changed_share(a, b):
    C = np.zeros((K, K), int)
    np.add.at(C, (a, b), 1)
    best = max(sum(C[p[j], j] for j in range(K)) for p in itertools.permutations(range(K)))
    return 1 - best / len(a)


emb, _, _ = load_embeddings()
labs = {s: tee_kmeans(emb, K, s) for s in list(range(1, 21)) + [42]}
pairs = [changed_share(labs[a], labs[b]) for a, b in itertools.combinations(range(1, 21), 2)]
print(f"seeds 1-20, {len(pairs)} pairs: mean {np.mean(pairs) * 100:.1f} %, worst {np.max(pairs) * 100:.1f} %")
print(f"seed 42 vs seed 7: {changed_share(labs[42], labs[7]) * 100:.1f} %")
