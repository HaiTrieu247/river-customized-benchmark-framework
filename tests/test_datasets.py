"""Tests for the stream generators and dataset adapters."""

from __future__ import annotations

import collections

import pytest

from benchmark.datasets import IterableDataset, from_river_dataset, make_scenario
from benchmark.datasets.synthetic import SCENARIOS, StreamConfig, SyntheticStream


def _labels(dataset):
    return [y for _, y in dataset]


# --- geometry ---------------------------------------------------------------
def test_separation_is_the_difficulty_knob_and_binds_at_every_dimensionality():
    """A minimum-distance constraint stops binding as d grows; a derived radius does not.

    This is the whole reason `separation` sets the radius instead of merely
    bounding centre distance: the unit cube gets roomier with d, so uniformly
    drawn centres are always far apart and every layout comes out equally easy.
    """
    radii = {}
    for d in (2, 10, 50):
        for sep in (1.0, 3.0):
            stream = SyntheticStream(
                StreamConfig(n_samples=10, n_features=d, n_clusters=5, separation=sep, seed=1)
            )
            list(stream)  # layout is built on iteration
            radii[(d, sep)] = stream._base_r

    for d in (2, 10, 50):
        # tighter separation => bigger clusters relative to the same layout
        assert radii[(d, 1.0)] > radii[(d, 3.0)]
        assert radii[(d, 1.0)] == pytest.approx(3.0 * radii[(d, 3.0)], rel=1e-9)


def test_separation_none_uses_cluster_radius_literally():
    stream = SyntheticStream(
        StreamConfig(n_samples=10, n_clusters=4, separation=None, cluster_radius=0.05, seed=1)
    )
    list(stream)
    assert stream._base_r == 0.05


def test_overlapping_clusters_are_harder_than_separated_ones():
    from benchmark.experiments import BenchmarkConfig, Setting, run_setting

    config = BenchmarkConfig(n_checkpoints=3, window_size=1000)
    setting = Setting("STREAMKMeans", params={"n_clusters": 5, "chunk_size": 500},
                      scaler="standard")
    scores = {}
    for sep in (1.0, 6.0):
        stream = SyntheticStream(
            StreamConfig(n_samples=6000, n_features=10, n_clusters=5, separation=sep, seed=1)
        )
        scores[sep] = run_setting(setting, stream, config, seed=1).quality["ari"]
    assert scores[6.0] > scores[1.0]


# --- composition and dynamics ----------------------------------------------
def test_noise_is_unlabelled_so_metrics_skip_it():
    stream = SyntheticStream(
        StreamConfig(n_samples=4000, n_clusters=4, noise=0.2, seed=1)
    )
    labels = _labels(stream)
    n_noise = sum(1 for y in labels if y is None)
    assert 0.15 * len(labels) < n_noise < 0.25 * len(labels)


def test_weight_range_produces_imbalanced_clusters():
    balanced = collections.Counter(
        _labels(SyntheticStream(StreamConfig(n_samples=8000, n_clusters=5, seed=1)))
    )
    skewed = collections.Counter(
        _labels(SyntheticStream(
            StreamConfig(n_samples=8000, n_clusters=5, weight_range=0.9, seed=1)
        ))
    )

    def spread(counter):
        counts = sorted(counter.values())
        return counts[-1] / counts[0]

    assert spread(skewed) > spread(balanced)


def test_events_change_the_live_cluster_count():
    stream = SyntheticStream(StreamConfig(
        n_samples=40_000, n_clusters=5, n_features=10,
        drift_speed=0.001, event_frequency=2000, n_clusters_range=3, seed=1,
    ))
    labels = _labels(stream)
    # new ids appear as clusters are created or split
    assert len(set(labels)) > 5
    # and the number alive at any moment stays near the configured range
    per_block = [len(set(labels[i:i + 4000])) for i in range(0, 40_000, 4000)]
    assert all(2 <= n <= 12 for n in per_block), per_block


def test_drift_moves_the_clusters():
    import numpy as np

    def centroid_of_first_cluster(drift_speed):
        stream = SyntheticStream(StreamConfig(
            n_samples=8000, n_clusters=3, n_features=5,
            drift_speed=drift_speed, drift_every=50, seed=1,
        ))
        rows = [(x, y) for x, y in stream if y == 0]
        early = np.array([list(x.values()) for x, _ in rows[:200]]).mean(axis=0)
        late = np.array([list(x.values()) for x, _ in rows[-200:]]).mean(axis=0)
        return float(np.linalg.norm(late - early))

    assert centroid_of_first_cluster(0.0) < 0.05          # static: stays put
    assert centroid_of_first_cluster(0.004) > 0.10        # drifting: moves away


# --- ordering ---------------------------------------------------------------
def test_iid_ordering_keeps_windows_class_mixed_and_blocked_ordering_does_not():
    """The property that decides whether per-window metrics mean anything."""
    from benchmark.algorithms.registry import get_algorithm
    from benchmark.evaluation.runner import run_once

    for scenario, expected_share in (("static_easy", 0.0), ("block_ordered", 1.0)):
        stream = make_scenario(scenario, n_samples=8000, n_clusters=4, seed=1)
        res = run_once(get_algorithm("DBSTREAM", scaler="standard"), stream,
                       n_checkpoints=3, window_size=1000)
        share = res.window_label_diagnostics["single_class_window_share"]
        if expected_share == 0.0:
            assert share == 0.0
        else:
            assert share > 0.5


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_scenario_produces_a_usable_stream(scenario):
    stream = make_scenario(scenario, n_samples=3000, n_clusters=4, n_features=6, seed=1)
    labels = _labels(stream)
    assert len(labels) == 3000
    assert stream.n_features == 6
    assert len({y for y in labels if y is not None}) >= 2
    # deterministic given the seed
    assert _labels(make_scenario(scenario, n_samples=3000, n_clusters=4,
                                 n_features=6, seed=1)) == labels


# --- river adapter ----------------------------------------------------------
class _FakeRiverDataset:
    """Mimics a river dataset that yields its numeric columns as strings."""

    n_samples = 6

    def __iter__(self):
        for i in range(self.n_samples):
            yield {"a": str(i * 1.5), "b": i, "name": "categorical", "flag": True}, str(i % 2)


def test_from_river_dataset_coerces_numeric_strings_and_drops_categoricals():
    """River ships several datasets whose numbers arrive as strings.

    An `isinstance(v, Number)` filter drops every one of them and yields an
    empty stream, so the adapter coerces instead of type-checking.
    """
    dataset = from_river_dataset(_FakeRiverDataset(), name="fake")
    assert isinstance(dataset, IterableDataset)
    assert dataset.n_features == 3  # a, b, flag -- 'name' dropped
    rows = list(dataset)
    assert len(rows) == 6
    x, y = rows[2]
    assert x == {"a": 3.0, "b": 2.0, "flag": 1.0}
    assert y == "0"
    # re-iterable: every setting must see the identical stream
    assert [r[1] for r in dataset] == [r[1] for r in rows]


def test_from_river_dataset_respects_n_samples_and_reports_length():
    dataset = from_river_dataset(_FakeRiverDataset(), name="fake", n_samples=4)
    assert len(list(dataset)) == 4
    assert len(dataset) == 4


def test_from_river_dataset_rejects_a_fully_categorical_stream():
    class _AllStrings:
        n_samples = 3

        def __iter__(self):
            for _ in range(3):
                yield {"a": "x", "b": "y"}, "1"

    with pytest.raises(ValueError, match="non-numeric"):
        from_river_dataset(_AllStrings(), name="bad")
