"""Tests for benchmark/tuning/recluster.py -- Table A's common reclustering.

``micro_centers`` is a best-effort accessor over every protocol algorithm's
micro-cluster level; ``common_recluster`` runs the same k-means/DBSCAN pair
over whichever centres it returns and scores them against the window's own
labels. See .superpowers/sdd/OTB_01_PLAN/task-5-brief.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.algorithms.registry import accepts_seed, get_algorithm
from benchmark.tuning.recluster import common_recluster, micro_centers


def _get(algo, **kw):
    # Some registered factories (DenStream, DBSTREAM, SOStream, BIRCH,
    # RepStream) build models whose __init__ has no `seed` parameter and no
    # **kwargs catch-all, so passing seed=1 unconditionally raises a
    # TypeError. accepts_seed() is the same guard conf_pipeline/run.py and
    # harness.py already use for this.
    if accepts_seed(algo):
        kw.setdefault("seed", 1)
    return get_algorithm(algo, **kw)


def _two_blobs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n):
        c = i % 2
        xs.append({"a": 5.0 * c + rng.normal(0, 0.1), "b": rng.normal(0, 0.1)})
        ys.append(c)
    return xs, ys


@pytest.mark.parametrize(
    "algo,kw",
    [
        ("DenStream", dict(epsilon=0.5, mu=2, beta=0.75, n_samples_init=50)),
        ("CluStream", dict(n_macro_clusters=2, max_micro_clusters=10)),
        ("DBSTREAM", dict(clustering_threshold=0.5)),
        ("SOStream", dict(merge_threshold=0.5)),
        ("BIRCH", dict(threshold=0.3)),
        ("STREAMKMeans", dict(chunk_size=20, n_clusters=2)),
        ("HPStream", dict(max_clusters=2, n_samples_init=50, n_projected_dimensions=2)),
        ("RepStream", dict(k=3)),
    ],
)
def test_micro_centers_are_exposed_for_every_protocol_algorithm(algo, kw):
    ad = _get(algo, **kw)
    xs, ys = _two_blobs()
    for x in xs:
        ad.learn_only(x)
    got = micro_centers(ad.model)
    assert got is not None
    C, w, keys = got
    assert C.shape[1] == 2 and keys == ["a", "b"] and len(w) == len(C) >= 2 and (w > 0).all()


def test_micro_centers_is_none_before_the_model_has_any():
    assert micro_centers(get_algorithm("DenStream", n_samples_init=50).model) is None


def test_common_recluster_recovers_two_blobs_from_many_micro_centres():
    xs, ys = _two_blobs()
    C = np.array([[0, 0], [0.1, 0], [5, 0], [5.1, 0], [5, 0.1]], float)
    w = np.ones(5)
    out = common_recluster(C, w, ["a", "b"], xs, ys, eps=1.0)
    assert out["ari_A_kmeans"] > 0.95 and out["ari_A_dbscan"] > 0.95
    assert out["n_micro_centers"] == 5 and out["ssq_micro"] > 0


def test_common_recluster_handles_missing_centres_and_single_class_windows():
    xs, ys = _two_blobs()
    assert common_recluster(None, None, None, xs, ys, eps=1.0) == {
        "ari_A_kmeans": None,
        "ari_A_dbscan": None,
        "ssq_micro": None,
        "n_micro_centers": None,
    }
    out = common_recluster(np.zeros((3, 2)), np.ones(3), ["a", "b"], xs, [0] * len(ys), eps=1.0)
    assert out["ari_A_kmeans"] is None and out["n_micro_centers"] == 3


def test_common_recluster_handles_string_labels_like_kdd99_and_insects():
    # KDD99 and every INSECTS variant carry string labels ("normal", "dos",
    # ...); `y >= 0` on an array holding those crashes, and the earlier
    # `-1 if v is None else v` sentinel scheme cannot even build such an
    # array. Table A must still produce a real ARI on these folds.
    xs, ys = _two_blobs()
    labels = ["normal", "dos"]
    ys = [labels[i % 2] for i in range(len(ys))]
    C = np.array([[0, 0], [0.1, 0], [5, 0], [5.1, 0], [5, 0.1]], float)
    out = common_recluster(C, np.ones(5), ["a", "b"], xs, ys, eps=1.0)
    assert out["ari_A_kmeans"] > 0.95 and out["ari_A_dbscan"] > 0.95


def test_common_recluster_treats_negative_one_as_a_real_class():
    # batch_external() never special-cases -1 -- only `None` means
    # "unlabelled" -- so common_recluster must not either.
    xs, ys = _two_blobs()
    ys = [-1 if v == 0 else 1 for v in ys]
    C = np.array([[0, 0], [0.1, 0], [5, 0], [5.1, 0], [5, 0.1]], float)
    out = common_recluster(C, np.ones(5), ["a", "b"], xs, ys, eps=1.0)
    assert out["ari_A_kmeans"] > 0.95 and out["ari_A_dbscan"] > 0.95


def test_common_recluster_scales_to_thousands_of_centres_without_a_dense_cube():
    # The naive (h, m, d) broadcast at m=10_000, h=1000, d=41 is gigabytes;
    # pairwise_distances_argmin_min is O(h*m). Finishing at m=5000, h=1000,
    # d=40 without allocating that cube is the check; wall time is not.
    rng = np.random.default_rng(0)
    m, h, d = 5000, 1000, 40
    keys = [f"f{i}" for i in range(d)]
    C = rng.normal(size=(m, d))
    xs = [dict(zip(keys, row)) for row in rng.normal(size=(h, d))]
    ys = [i % 2 for i in range(h)]
    out = common_recluster(C, np.ones(m), keys, xs, ys, eps=1.0)
    assert set(out) == {"ari_A_kmeans", "ari_A_dbscan", "ssq_micro", "n_micro_centers"}
    assert out["n_micro_centers"] == 5000 and out["ssq_micro"] > 0
