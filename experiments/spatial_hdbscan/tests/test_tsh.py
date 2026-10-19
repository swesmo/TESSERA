"""Small, hand-checkable tests of the experiment code (suggested by Romaric on 6 Oct 2026: tiny
inputs whose expected result can be worked out on paper), plus a check of the Python k-means
against TEE's own JavaScript worker.

    cd experiments/spatial_hdbscan && ~/tee-local/venv/bin/python -m unittest discover -s tests -v

The JavaScript checks need `node` and TEE's public/js/segmentation.js; the last one also needs the
Gothenburg viewport in ~/tee_data. They are skipped when these are missing.
"""
import json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tsh  # noqa: E402

SEG = next((p for p in (HERE / "../../../TEE/public/js/segmentation.js", HERE / "../../../public/js/segmentation.js")
            if p.exists()), None)
NODE = shutil.which("node")


def blobs(seed=0, n=5, spread=0.1):
    """15 points in three tight groups around (0, 0), (10, 0) and (0, 10)."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
    X = np.concatenate([c + spread * rng.standard_normal((n, 2)) for c in centres]).astype(np.float32)
    return X, np.repeat(np.arange(3), n), centres


class Metrics(unittest.TestCase):
    def test_ari_hand_values(self):
        self.assertEqual(tsh.ari([0, 0, 1, 1], [1, 1, 0, 0]), 1.0)            # same partition, renamed
        # crossing partitions of 4 points: no pair agrees. pairs(a) = pairs(b) = 2 of 6,
        # expected = 2 * 2 / 6, max = 2  ->  (0 - 2/3) / (2 - 2/3) = -0.5
        self.assertAlmostEqual(tsh.ari([0, 0, 1, 1], [0, 1, 0, 1]), -0.5)

    def test_ari_equals_sklearn(self):
        rng = np.random.default_rng(1)
        for ka, kb in ((2, 2), (5, 17), (17, 17)):
            a = rng.integers(0, ka, 20_000)
            b = (a + rng.integers(0, 3, 20_000)) % kb
            self.assertAlmostEqual(tsh.ari(a, b), adjusted_rand_score(a, b), places=12)

    def test_stability_pair_matrix(self):
        labs = [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1])]
        P = tsh.pair_matrix(labs)
        self.assertTrue(np.isnan(np.diag(P)).all())
        self.assertEqual(P[0, 1], 1.0)
        self.assertAlmostEqual(P[0, 2], -0.5)
        self.assertAlmostEqual(tsh.stability(labs)[0], np.nanmean(P))

    def test_field_metrics_hand_example(self):
        # two fields of 4 pixels (left = class 1, right = class 2); one pixel of the left field
        # falls into the wrong cluster
        fid = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        lab = np.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        fm = tsh.field_metrics(lab, fid, fid)
        self.assertAlmostEqual(fm["field_homog"], 7 / 8)          # (3 + 4) / 8 pixels in the majority cluster
        ea = tsh.edge_adherence(lab, fid)
        self.assertAlmostEqual(ea["d_in"], 2 / 8)                 # 8 neighbour pairs inside a field, 2 differ
        self.assertAlmostEqual(ea["d_out"], 1 / 2)                # 2 pairs across the field edge, 1 differs
        self.assertAlmostEqual(tsh.coherence(lab, np.ones_like(lab, bool)), 7 / 10)


class KMeans(unittest.TestCase):
    def test_three_blobs_every_seed(self):
        X, truth, centres = blobs()
        for seed in range(20):
            C = tsh.tee_kmeans(X, 3, seed)
            lab = tsh.sqd(X, C).argmin(1)
            self.assertEqual(tsh.ari(lab, truth), 1.0, f"seed {seed}")
            # every centroid is the mean of its 5 points
            for c in range(3):
                np.testing.assert_allclose(C[c], X[lab == c].mean(0), atol=1e-5)

    def test_sample_rule(self):
        # S = min(max(5000, 500 k), N): with N < 5000 all points are used, so the result must not
        # depend on the `sample` argument
        X, _, _ = blobs()
        np.testing.assert_array_equal(tsh.tee_kmeans(X, 3, 3), tsh.tee_kmeans(X, 3, 3, sample=10_000))

    def test_restarts_never_worse(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((3000, 4)).astype(np.float32)
        for seed in range(5):
            i1 = tsh.sqd(X, tsh.tee_kmeans(X, 6, seed, n_init=1)).min(1).sum()
            i10 = tsh.sqd(X, tsh.tee_kmeans(X, 6, seed, n_init=10)).min(1).sum()
            self.assertLessEqual(i10, i1 * (1 + 1e-6))               # the first restart is the single run

    def test_full_lloyd_step(self):
        X = np.array([[0], [2], [10], [14]], np.float32)
        C = tsh.lloyd_full(X, np.array([[1], [9]], np.float32), 1)
        np.testing.assert_allclose(C.ravel(), [1, 12])               # means of {0, 2} and {10, 14}
        np.testing.assert_allclose(tsh.group_means(X, np.array([0, 0, 0, 0]), 2, C).ravel(), [6.5, 12])  # empty keeps old

    def test_exact_port_equals_numpy_version_on_blobs(self):
        X, truth, _ = blobs()
        for seed in (1, 7, 42):
            lab, _ = tsh.tee_kmeans_exact(X, 3, seed)
            self.assertEqual(tsh.ari(lab, truth), 1.0)


class WardMerge(unittest.TestCase):
    def test_unweighted_line(self):
        # centroids at 0, 1, 10, 12: cost(0,1) = 1*1/2 * 1 = 0.5 is the cheapest merge, then
        # cost(10,12) = 2 against cost({0,1},10) = 2*1/3 * 9.5^2 = 60.2
        C = np.array([[0], [1], [10], [12]], np.float32)
        lut = tsh.merge_tree(C, np.ones(4), 2)
        self.assertEqual(lut[0], lut[1]); self.assertEqual(lut[2], lut[3]); self.assertNotEqual(lut[0], lut[2])

    def test_pixel_counts_change_the_merge(self):
        # centroids at 0, 3, 5. Equal counts: cost(3,5) = 2 < cost(0,3) = 4.5 -> {3,5} merge.
        # Counts 1, 100, 100: cost(0,3) = 100/101 * 9 = 8.9 < cost(3,5) = 50 * 4 = 200 -> {0,3} merge.
        C = np.array([[0], [3], [5]], np.float32)
        lut = tsh.merge_tree(C, np.array([1, 1, 1]), 2)
        self.assertEqual(lut[1], lut[2]); self.assertNotEqual(lut[0], lut[1])
        lut = tsh.merge_tree(C, np.array([1, 100, 100]), 2)
        self.assertEqual(lut[0], lut[1]); self.assertNotEqual(lut[1], lut[2])

    def test_over_cluster_merge_refine_recovers_blobs(self):
        X, truth, centres = blobs(n=40)
        for seed in range(10):
            G = tsh.over_ward_refine(X, 3, seed, K0=9)
            lab = tsh.sqd(X, G).argmin(1)
            self.assertEqual(tsh.ari(lab, truth), 1.0, f"seed {seed}")
            np.testing.assert_allclose(np.sort(G.sum(1)), np.sort(centres.sum(1)), atol=0.1)


class Smoothing(unittest.TestCase):
    def setUp(self):
        self.valid = np.ones((7, 7), bool)

    def test_icm_removes_isolated_pixel_but_keeps_edge(self):
        # left 3 columns belong to cluster 0, right 4 to cluster 1: distance 1 to the own centroid,
        # 3 to the other one. One pixel deep inside the right part is slightly closer to cluster 0
        # (1.0 against 1.2). icm_potts divides the distances by their mean (here 1), so this pixel
        # flips as soon as beta * 8 neighbours > 0.2, i.e. beta > 0.025.
        D = np.empty((7, 7, 2), np.float32)
        D[:, :3] = (1, 3); D[:, 3:] = (3, 1)
        D[3, 5] = (1.0, 1.2)
        lab = D.argmin(-1)
        self.assertEqual(lab[3, 5], 0)
        np.testing.assert_array_equal(tsh.icm_potts(D, lab, self.valid, 0.0), lab)      # beta = 0: nothing changes
        np.testing.assert_array_equal(tsh.icm_potts(D, lab, self.valid, 0.02), lab)     # 0.16 < 0.2: still nothing
        out = tsh.icm_potts(D, lab, self.valid, 0.25)
        self.assertEqual(out[3, 5], 1)                                                   # 8 neighbours outvote it
        out[3, 5] = lab[3, 5]
        np.testing.assert_array_equal(out, lab)                                          # the edge stays

    def test_majority_filter(self):
        lab = np.zeros((7, 7), np.int64); lab[:, 4:] = 1; lab[3, 1] = 1                 # edge + one speckle pixel
        out = tsh.post_mode(lab, self.valid, 2, (1, 1))
        self.assertEqual(out[3, 1], 0)
        out[3, 1] = 1
        np.testing.assert_array_equal(out, lab)

    def test_gauss_impulse_and_constant(self):
        emb = np.zeros((21, 21, 1), np.float32); emb[10, 10, 0] = 1
        valid = np.ones((21, 21), bool)
        out = tsh.pre_gauss(emb, valid, 10, (10.0, 10.0))[..., 0]                       # sigma = 1 pixel
        self.assertAlmostEqual(float(out.sum()), 1.0, places=4)                          # mass is kept
        self.assertEqual(np.unravel_index(out.argmax(), out.shape), (10, 10))
        np.testing.assert_allclose(out, out.T, atol=1e-7)
        self.assertAlmostEqual(float(out[10, 11] / out[10, 10]), np.exp(-0.5), places=2)  # exp(-d^2 / 2 sigma^2)
        const = np.full((9, 9, 3), 4.0, np.float32)
        hole = np.ones((9, 9), bool); hole[4, 4] = False                                 # invalid pixels are ignored
        np.testing.assert_allclose(tsh.pre_gauss(const, hole, 10, (10.0, 10.0))[hole], 4.0, rtol=1e-5)

    def test_gauss_uses_metres(self):
        # pixels of 13 m x 7 m (Gothenburg): the filter is wider in columns than in rows
        emb = np.zeros((41, 41, 1), np.float32); emb[20, 20, 0] = 1
        out = tsh.pre_gauss(emb, np.ones((41, 41), bool), 10, (13.0, 6.95))[..., 0]
        self.assertGreater(out[20, 21], out[21, 20])
        self.assertEqual(tsh.window((13.0, 6.95), 20), (2, 3))


class Other(unittest.TestCase):
    def test_pca_finds_the_line(self):
        rng = np.random.default_rng(0)
        t = rng.standard_normal(500)
        X = (np.outer(t, [3, 4, 0]) + 0.01 * rng.standard_normal((500, 3)) + [5, 5, 5]).astype(np.float32)
        mu, P, share = tsh.pca_basis(X, 1)
        np.testing.assert_allclose(np.abs(P[:, 0]), [0.6, 0.8, 0.0], atol=0.01)
        self.assertGreater(share, 0.999)
        np.testing.assert_allclose(mu, [5, 5, 5], atol=0.5)

    def test_superpixels_follow_an_edge(self):
        emb = np.zeros((20, 30, 4), np.float32); emb[:, 13:] = 5                        # edge between columns 12 and 13
        emb += 0.01 * np.random.default_rng(0).standard_normal(emb.shape).astype(np.float32)
        sp = tsh.superpixels(emb, np.ones((20, 30), bool), (10.0, 10.0), 50, 1.0)
        self.assertGreater(sp.max(), 5)
        left, right = set(sp[:, :13].ravel()), set(sp[:, 13:].ravel())
        self.assertFalse(left & right)                                                   # no superpixel crosses it

    def test_geojson_labels(self):
        gt = dict(a=0.0001, c=11.0, e=-0.0001, f=57.0)

        def rect(px0, py0, px1, py1):  # pixel rectangle -> lon/lat ring
            x0, x1, y0, y1 = gt["c"] + px0 * gt["a"], gt["c"] + px1 * gt["a"], gt["f"] + py0 * gt["e"], gt["f"] + py1 * gt["e"]
            return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]

        fc = dict(type="FeatureCollection", features=[
            dict(type="Feature", properties=dict(name="water"), geometry=dict(type="Polygon", coordinates=rect(0, 0, 4, 3))),
            dict(type="Feature", properties=dict(name="forest"), geometry=dict(type="Polygon", coordinates=rect(6, 2, 10, 8))),
            dict(type="Feature", properties=dict(name="water"), geometry=dict(type="Polygon", coordinates=rect(0, 6, 2, 8))),
            dict(type="Feature", properties=dict(), geometry=dict(type="Polygon", coordinates=rect(0, 0, 10, 8)))])  # no name: ignored
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "labels.geojson"
            f.write_text(json.dumps(fc))
            cls, fid, names = tsh.load_geojson_labels(f, (8, 10), gt)
        self.assertEqual(names, ["water", "forest"])
        self.assertEqual(int((cls == 1).sum()), 4 * 3 + 2 * 2)
        self.assertEqual(int((cls == 2).sum()), 4 * 6)
        self.assertEqual(int((cls == 0).sum()), 80 - 16 - 24)
        self.assertEqual(len(np.unique(fid[fid > 0])), 3)
        self.assertTrue((cls[0:3, 0:4] == 1).all() and (cls[2:8, 6:10] == 2).all())


@unittest.skipUnless(NODE and SEG, "needs node and TEE's segmentation.js")
class AgainstTeeWorker(unittest.TestCase):
    """The reference is TEE's own worker code, run in Node."""

    def js_labels(self, X, k, seeds):
        with tempfile.TemporaryDirectory() as d:
            X.astype("<f4").tofile(Path(d) / "x.f32")
            subprocess.run([NODE, str(HERE / "tee_worker.mjs"), str(Path(d) / "x.f32"), str(len(X)), str(X.shape[1]), str(k),
                            ",".join(map(str, seeds)), str(Path(d) / "out.i32")], check=True,
                           env={**os.environ, "TEE_SEG": str(SEG)})
            return np.fromfile(Path(d) / "out.i32", "<i4").reshape(len(seeds), len(X))

    def test_exact_port_gives_the_same_labels(self):
        # 6 overlapping Gaussian groups in 8-d, more points than the sample (5000), so that the
        # Fisher-Yates sample, k-means++ and the Lloyd iterations are all exercised
        rng = np.random.default_rng(3)
        X = (rng.standard_normal((6, 8)) * 2)[rng.integers(0, 6, 12_000)] + rng.standard_normal((12_000, 8))
        X = X.astype(np.float32)
        seeds = [1, 7, 42]
        js = self.js_labels(X, 6, seeds)
        for s, ref in zip(seeds, js):
            lab, _ = tsh.tee_kmeans_exact(X, 6, s)
            self.assertGreaterEqual((lab == ref).mean(), 0.9999, f"seed {s}")           # same cluster ids, not only ARI

    @unittest.skipUnless((Path.home() / "tee_data/harness-cache/gbg2024_emb.npy").exists(), "needs the Gothenburg cache")
    def test_gothenburg_shares_and_seed_statistics(self):
        G = np.load(Path.home() / "tee_data/harness-cache/gbg2024_emb.npy")
        X = G.reshape(-1, G.shape[-1])
        # (a) exact port = TEE worker on the real viewport
        js = self.js_labels(X, 5, [42, 7])
        for s, ref in zip((42, 7), js):
            lab, _ = tsh.tee_kmeans_exact(X, 5, s)
            self.assertGreaterEqual((lab == ref).mean(), 0.9999, f"seed {s}")
        # the cluster shares TEE's Auto-label panel showed on 26 Sep (CLAUDE.md), sorted
        np.testing.assert_allclose(np.sort(np.bincount(js[0]) / len(X) * 100), np.sort([23.5, 14.9, 30.9, 9.5, 21.2]), atol=0.06)
        # (b) the harness (NumPy generator instead of mulberry32) is the same algorithm with other
        # random numbers: over 30 seeds each, the mean pairwise ARI must agree within sampling noise
        seeds = list(range(1, 31))
        Pj = tsh.pair_matrix(list(self.js_labels(X, 5, seeds)))
        Pn = tsh.pair_matrix([tsh.sqd(X, tsh.tee_kmeans(X, 5, s)).argmin(1) for s in seeds])
        boot = []
        rng = np.random.default_rng(0)
        for _ in range(2000):                                                            # bootstrap over seeds
            i, j = rng.integers(0, 30, 30), rng.integers(0, 30, 30)
            boot.append(np.nanmean(Pj[np.ix_(i, i)]) - np.nanmean(Pn[np.ix_(j, j)]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n    mean pairwise ARI, 30 seeds: TEE worker {np.nanmean(Pj):.3f}, harness {np.nanmean(Pn):.3f}; "
              f"95 % interval of the difference [{lo:+.3f}, {hi:+.3f}]")
        self.assertTrue(lo <= 0 <= hi)


@unittest.skipUnless(NODE and SEG, "needs node and TEE's segmentation.js")
class PortAgainstPython(unittest.TestCase):
    """The methods as ported to TEE's worker (JavaScript) against the Python code that produced
    the numbers of the report."""

    def node(self, *args):
        subprocess.run([NODE, str(HERE / "port_blocks.mjs"), *map(str, args)], check=True, env={**os.environ, "TEE_SEG": str(SEG)})

    def test_ward_merge(self):
        rng = np.random.default_rng(0)
        K0, k, dim = 32, 5, 128
        C = rng.standard_normal((K0, dim)).astype(np.float32) * 3
        n = rng.integers(1, 600, K0).astype(np.int32)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); C.tofile(d / "c.f32"); n.tofile(d / "n.i32")
            self.node("ward", d / "c.f32", d / "n.i32", K0, k, dim, d / "out.f32")
            js = np.fromfile(d / "out.f32", "<f4").reshape(k, dim)
        lut = tsh.merge_tree(C, n, k, "ward")
        py = np.stack([(C[lut == g] * n[lut == g, None]).sum(0) / n[lut == g].sum() for g in range(k)])
        # same groups; the order of the groups may differ
        order_js, order_py = np.lexsort(js.T[::-1]), np.lexsort(py.T[::-1])
        np.testing.assert_allclose(js[order_js], py[order_py], atol=1e-4)

    def test_icm(self):
        # two smooth regions plus noise in the distances, as after a k-means assignment
        rng = np.random.default_rng(1)
        H, W, k = 60, 80, 4
        region = (np.add.outer(np.arange(H) // 20, np.arange(W) // 27)) % k
        D = (2 + rng.random((H, W, k))).astype(np.float32)
        D[np.arange(H)[:, None], np.arange(W)[None], region] -= 0.8
        lab = D.argmin(-1).astype(np.int32)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); D.tofile(d / "D.f32"); lab.tofile(d / "lab.i32")
            self.node("icm", d / "D.f32", d / "lab.i32", W, H, k, 0.25, d / "out.i32")
            js = np.fromfile(d / "out.i32", "<i4").reshape(H, W)
        py = tsh.icm_potts(D, lab.astype(np.int64), np.ones((H, W), bool), 0.25, (1, 1))
        self.assertGreater((py != lab).mean(), 0.05)                                     # the smoothing does something
        # identical away from the image border (Python replicates the edge pixels, the port counts
        # only existing neighbours), and almost identical overall
        self.assertEqual(int((js[6:-6, 6:-6] != py[6:-6, 6:-6]).sum()), 0)
        self.assertGreater((js == py).mean(), 0.995)

    def test_gauss_filter(self):
        rng = np.random.default_rng(2)
        H, W, dim = 40, 50, 6
        E = rng.standard_normal((H, W, dim)).astype(np.float32)
        px = (13.0, 6.95)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); E.tofile(d / "e.f32")
            self.node("gauss", d / "e.f32", W, H, dim, 10 / px[1], 10 / px[0], d / "out.f32")
            js = np.fromfile(d / "out.f32", "<f4").reshape(H, W, dim)
        py = tsh.pre_gauss(E, np.ones((H, W), bool), 10, px)
        # the port cuts the kernel at 3 sigma (scipy: 4 sigma) and replicates the edge pixels
        # (scipy: reflects them), so compare the interior closely and the whole image loosely
        np.testing.assert_allclose(js[8:-8, 8:-8], py[8:-8, 8:-8], atol=0.02)
        self.assertGreater(np.corrcoef(js.ravel(), py.ravel())[0, 1], 0.99)

    @unittest.skipUnless((Path.home() / "tee_data/harness-cache/gbg.f32").exists(), "needs the Gothenburg cache")
    def test_stable_on_gothenburg(self):
        # Same algorithm, other random numbers: the stability over 20 seeds must agree within
        # sampling noise, and the two implementations must find the same partition.
        f = Path.home() / "tee_data/harness-cache/gbg.f32"
        H, W, dim, k = 385, 720, 128, 5
        seeds = list(range(1, 21))
        with tempfile.TemporaryDirectory() as d:
            self.node("run", f, W, H, dim, k, "stable", ",".join(map(str, seeds)), Path(d) / "out.i32")
            js = list(np.fromfile(Path(d) / "out.i32", "<i4").reshape(len(seeds), H * W))
        X = np.fromfile(f, "<f4").reshape(H * W, dim)
        py = [tsh.sqd(X, tsh.over_ward_refine(X, k, s)).argmin(1) for s in seeds]
        Pj, Pp = tsh.pair_matrix(js), tsh.pair_matrix(py)
        cross = np.mean([tsh.ari(a, b) for a in js[:10] for b in py[:10]])
        print(f"\n    'Stable' on Gothenburg, 20 seeds: mean pairwise ARI port {np.nanmean(Pj):.3f}, Python {np.nanmean(Pp):.3f}; "
              f"port against Python {cross:.3f}")
        self.assertGreater(np.nanmean(Pj), 0.9)
        self.assertLess(abs(np.nanmean(Pj) - np.nanmean(Pp)), 0.04)
        self.assertGreater(cross, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
