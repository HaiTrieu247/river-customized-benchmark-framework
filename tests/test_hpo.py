"""Tests for the HPO pipeline: the trace metric, the rules, and the space.

These deliberately avoid touching any real dataset, so the suite stays runnable
without downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.algorithms.registry import get_algorithm
from benchmark.datasets.synthetic import StreamConfig, SyntheticStream
from benchmark.evaluation.clustering_quality import StreamingMATRTrace
from benchmark.evaluation.runner import run_once
from hpo_pipeline import rules, space
from hpo_pipeline._common import config_id


def _dataset(n=3000, k=4, d=3, seed=1):
    return SyntheticStream(StreamConfig(n_samples=n, n_clusters=k, n_features=d, seed=seed))


# --- the trace accumulator -------------------------------------------------
def test_matr_trace_matches_brute_force():
    """The shortcut that makes P4a affordable, checked against the definition."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    labels = rng.integers(0, 7, size=2000)

    metric = StreamingMATRTrace()
    for x, k in zip(X, labels):
        metric.update({f"f{j}": v for j, v in enumerate(x)}, int(k))

    brute = 0.0
    for k in set(labels.tolist()):
        points = X[labels == k]
        brute += ((points - points.mean(axis=0)) ** 2).sum()
    brute *= -2.0

    assert abs(metric.get() - brute) / abs(brute) < 1e-9


def test_matr_trace_needs_no_centers():
    """It must be defined for the clusterers that expose no centres."""
    metric = StreamingMATRTrace()
    assert metric.requires_centers is False
    metric.update({"a": 1.0, "b": 2.0}, 0)
    metric.update({"a": 3.0, "b": 4.0}, 0)
    # two points, one cluster: WCSS = 4, trace = -8
    assert metric.get() == pytest.approx(-8.0)


def test_matr_trace_reaches_the_runner_without_centers():
    adapter = get_algorithm("DBSTREAM", scaler="none", clustering_threshold=0.2)

    class _NoCenters:
        def __init__(self, inner):
            self.inner = inner

        def learn_one(self, x):
            self.inner.learn_one(x)

        def predict_one(self, x):
            return self.inner.predict_one(x)

    adapter.model = _NoCenters(adapter.model)
    result = run_once(
        adapter, _dataset(), n_checkpoints=3, window_size=500,
        external_metrics=("ari",), internal_metrics=("silhouette", "ssq", "matr_trace"),
    )
    assert adapter.centers() is None
    assert result.quality["matr_trace"] < 0


# --- score_from ------------------------------------------------------------
def test_score_from_learns_from_zero_but_scores_from_n0():
    adapter = get_algorithm("DBSTREAM", scaler="none", clustering_threshold=0.2)
    result = run_once(
        adapter, _dataset(), n_checkpoints=3, window_size=500,
        external_metrics=("ari",), score_from=1000,
    )
    assert result.score_from == 1000
    assert result.n_samples == 3000  # the whole stream was still consumed
    assert result.windows[0].start == 1000


def test_cluster_size_diagnostics_are_recorded():
    adapter = get_algorithm("KMeans", scaler="none", n_clusters=4)
    result = run_once(
        adapter, _dataset(), n_checkpoints=3, window_size=500, external_metrics=("ari",)
    )
    diag = result.cluster_size_diagnostics
    assert diag["n_predicted_clusters"] >= 1
    assert 0.0 <= diag["largest_cluster_share"] <= 1.0
    assert 0.0 <= diag["dust_cluster_share"] <= 1.0


# --- the selection rules ---------------------------------------------------
def test_r_parsimony_picks_the_coarsest_acceptable_level():
    # rises then plateaus: with a loose phi the plateau's start is enough
    curve = np.array([0.0, 0.5, 0.9, 0.98, 1.0, 1.0, 1.0, 1.0])
    assert rules.r_parsimony(curve, 0.20) == 2
    assert rules.r_parsimony(curve, 0.01) == 4
    # argmax would always take the finest end of a plateau
    assert rules.r_max(curve) == 4


def test_r_max_degenerates_on_a_monotone_criterion():
    """Written down in advance: a negated WCSS rises with the cluster count."""
    curve = np.arange(8, dtype=float)
    assert rules.r_max(curve) == 7
    assert rules.r_parsimony(curve, 0.20) < 7


def test_r_knee_finds_the_elbow():
    curve = np.array([0.0, 0.6, 0.85, 0.93, 0.96, 0.98, 0.99, 1.0])
    assert 0 < rules.r_knee(curve) < 4


def test_flat_curves_fall_back_to_the_coarsest():
    flat = np.ones(8)
    assert rules.r_parsimony(flat, 0.05) == 0
    assert rules.r_knee(flat) == 0


def test_orient_flips_only_the_smaller_is_better_criteria():
    vals = [1.0, 2.0]
    assert list(rules.orient(vals, "matr_trace_win")) == [1.0, 2.0]
    assert list(rules.orient(vals, "silhouette")) == [-1.0, -2.0]
    assert list(rules.orient(vals, "ssq")) == [-1.0, -2.0]


# --- the search space ------------------------------------------------------
def test_granularity_runs_coarse_to_fine_in_both_families():
    """g=1 is coarsest everywhere; thresholds must therefore descend."""
    counts = [space.granularity_value("KMeans", i) for i in range(1, 9)]
    assert counts == sorted(counts)

    thresholds = [space.granularity_value("DBSTREAM", i) for i in range(1, 9)]
    assert thresholds == sorted(thresholds, reverse=True)

    alphas = [space.granularity_value("RepStream", i) for i in range(1, 9)]
    assert alphas == sorted(alphas, reverse=True)


def test_space_is_eight_by_eight_plus_the_default():
    built = space.build_space(["DBSTREAM"])["DBSTREAM"]
    tuned = [c for c in built if not c["is_default"]]
    assert len(tuned) == 64
    assert len({c["config_id"] for c in built}) == len(built)
    assert sum(c["is_default"] for c in built) == 1
    assert {c["granularity_level"] for c in tuned} == set(range(1, 9))
    assert {c["theta"] for c in tuned} == set(range(1, 9))


def test_clustream_shape_points_cover_the_whole_ladder():
    """Rejection at generation, not clipping at run time."""
    for point in space.shape_points("CluStream"):
        assert point["max_micro_clusters"] >= max(space.COUNTS)


def test_denstream_shape_points_carry_the_pinned_beta():
    """A dropped dimension must be pinned, not handed to River's default.

    River's DenStream defaults to beta=0.75, which sits outside the [0.2, 0.6]
    band Fig. 20 verified. Removing beta from the search without pinning it would
    have silently moved every run there.
    """
    for point in space.shape_points("DenStream"):
        assert point["beta"] == space.FIXED["DenStream"]["beta"]
        # River's pruning period is log(beta*mu / (beta*mu - 1)).
        assert point["beta"] * point["mu"] > 1


def test_clustream_shape_points_carry_the_pinned_time_gap():
    for point in space.shape_points("CluStream"):
        assert point["time_gap"] == space.FIXED["CluStream"]["time_gap"]


def test_pinned_values_reach_the_resolved_parameters():
    """A pin is only real if it survives into what the model is built with."""
    candidate = next(
        c for c in space.build_space(["DenStream"])["DenStream"] if not c["is_default"]
    )
    params = space.resolve(candidate, {"s": 1.0})
    assert params["beta"] == space.FIXED["DenStream"]["beta"]
    assert "stream_speed" in params


def test_config_id_is_stable_across_folds_but_params_are_not():
    candidate = space.build_space(["DBSTREAM"])["DBSTREAM"][0]
    coarse = space.resolve(candidate, {"s": 1.0})
    fine = space.resolve(candidate, {"s": 100.0})
    # same config, different absolute threshold per fold
    assert coarse["clustering_threshold"] != fine["clustering_threshold"]
    assert fine["clustering_threshold"] == pytest.approx(
        100.0 * coarse["clustering_threshold"]
    )
    # ...and the id does not move, which is what P2 and P5 depend on
    again = config_id(
        candidate["algo"], candidate["granularity_level"], candidate["g"], candidate["shape"]
    )
    assert again == candidate["config_id"]


def test_count_axis_resolves_to_integers():
    candidate = next(
        c for c in space.build_space(["KMeans"])["KMeans"] if not c["is_default"]
    )
    params = space.resolve(candidate, {"s": 3.7})
    assert isinstance(params["n_clusters"], int)
