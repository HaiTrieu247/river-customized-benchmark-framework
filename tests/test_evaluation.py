"""Tests for the evaluation layer (metrics, runner) and the harness."""

from __future__ import annotations

from benchmark.algorithms.registry import get_algorithm
from benchmark.datasets.synthetic import StreamConfig, SyntheticStream
from benchmark.evaluation import (
    ami,
    ari,
    calinski_harabasz,
    davies_bouldin,
    nmi,
    purity,
    silhouette,
)
from benchmark.evaluation.runner import run_once
from benchmark.experiments import BenchmarkConfig, Setting, run_benchmark


def _dataset(n=4000, k=3, seed=1):
    return SyntheticStream(StreamConfig(n_samples=n, n_clusters=k, seed=seed))


def test_external_metrics_perfect_and_missing():
    yt = ["a", "a", "b", "b"]
    assert ari(yt, [0, 0, 1, 1]) == 1.0
    assert nmi(yt, [0, 0, 1, 1]) == 1.0
    assert ami(yt, [0, 0, 1, 1]) == 1.0
    assert purity(yt, [0, 0, 1, 1]) == 1.0
    assert ari([None, None], [0, 1]) is None  # unlabeled dropped


def test_internal_metrics_on_separated_points():
    # two well-separated 1-D blobs
    x = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]
    labels = [0, 0, 0, 1, 1, 1]
    assert silhouette(x, labels) > 0.8
    assert davies_bouldin(x, labels) is not None and davies_bouldin(x, labels) >= 0
    assert calinski_harabasz(x, labels) > 0
    # a single cluster => internal metrics undefined
    assert silhouette(x, [0, 0, 0, 0, 0, 0]) is None


def test_run_once_efficiency_and_stream_quality():
    adapter = get_algorithm("DBSTREAM", scaler="standard")
    res = run_once(adapter, _dataset(), n_checkpoints=5, window_size=500)
    assert res.n_samples == 4000
    assert res.n_features == 2
    assert res.efficiency.throughput_sps > 0
    assert "ari" in res.quality  # whole-stream value
    # time-aware summary present with mean/median/final
    assert "ari" in res.quality_stream
    assert set(res.quality_stream["ari"]) == {"mean", "median", "final"}
    assert len(res.windows) >= 7  # 4000 / 500
    # checkpoints monotonic in samples seen
    ns = [c.n_seen for c in res.checkpoints]
    assert ns == sorted(ns)


def test_internal_metrics_requested_appear_in_windows():
    adapter = get_algorithm("STREAMKMeans", n_clusters=3, scaler="standard")
    res = run_once(
        adapter, _dataset(n=2000), window_size=500,
        internal_metrics=("silhouette",),
    )
    assert "silhouette" in res.quality_stream


def test_run_benchmark_multiple_settings_same_dataset():
    dataset = _dataset(n=3000, k=4)
    config = BenchmarkConfig(
        settings=[
            Setting("DBSTREAM", scaler="standard"),
            Setting("STREAMKMeans", params={"n_clusters": 4}),
            Setting("CluStream", params={"n_macro_clusters": 4}),
        ],
        seeds=(1,),
        n_checkpoints=4,
    )
    results = run_benchmark(dataset, config)
    assert len(results) == 3
    for r in results:
        assert r.n_samples == 3000
        assert r.efficiency.total_runtime_s > 0


def test_unlabeled_dataset_skips_external_quality():
    from benchmark.datasets.base import IterableDataset

    base = _dataset(n=800)
    unlabeled = IterableDataset(
        lambda: ((x, None) for x, _ in base), name="unlabeled",
        n_features=base.n_features, has_labels=False,
    )
    adapter = get_algorithm("DBSTREAM")
    res = run_once(adapter, unlabeled, n_checkpoints=3)
    assert res.quality == {}  # no labels -> no external quality
    assert res.efficiency.total_runtime_s > 0
