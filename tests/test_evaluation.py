"""Tests for the evaluation layer (metrics, adaptation, runner)."""

from __future__ import annotations

from benchmark.algorithms.registry import get_algorithm
from benchmark.datasets.synthetic import scenarios
from benchmark.evaluation import ari, nmi, purity
from benchmark.evaluation.adaptation import compute_adaptation
from benchmark.evaluation.runner import WindowRecord, run_experiment
from benchmark.ground_truth.events import DriftEvent, DriftType


def test_quality_metrics_perfect_and_random():
    yt = ["a", "a", "b", "b"]
    assert ari(yt, [0, 0, 1, 1]) == 1.0
    assert nmi(yt, [0, 0, 1, 1]) == 1.0
    assert purity(yt, [0, 0, 1, 1]) == 1.0
    # None labels (noise) are dropped, not scored
    assert ari([None, None], [0, 1]) is None


def test_compute_adaptation_detects_drop_and_recovery():
    # synthetic quality curve: high, drop at drift, recover afterwards
    records = []
    curve = [0.9, 0.92, 0.9, 0.4, 0.6, 0.8, 0.9, 0.91]
    for i, q in enumerate(curve):
        rec = WindowRecord(start=i * 1000, end=i * 1000 + 999)
        rec.metrics["ari"] = q
        records.append(rec)
    event = DriftEvent(1, DriftType.MOVEMENT, start=3000, end=3000)
    m = compute_adaptation(records, event, metric="ari", recovery_fraction=0.95)
    assert m.quality_before is not None and m.quality_before > 0.85
    assert m.quality_min < 0.5
    assert m.performance_drop > 0.4
    assert m.recovered
    assert m.recovery_time is not None and m.recovery_time > 0


def test_run_experiment_stationary_high_quality():
    stream, gt = scenarios.stationary(n_samples=4000, seed=1)
    adapter = get_algorithm("DBSTREAM")
    res = run_experiment(adapter, stream, gt, window_size=500)
    _, ys = res.quality_curve("ari")
    assert ys, "expected a quality curve"
    assert max(ys) > 0.8
    assert res.efficiency.throughput_sps > 0


def test_run_experiment_records_drift_adaptation():
    stream, gt = scenarios.abrupt_movement(n_samples=6000, seed=1)
    adapter = get_algorithm("DBSTREAM")
    res = run_experiment(adapter, stream, gt, window_size=500)
    assert len(res.adaptation) == 1
    assert res.adaptation[0].drift_type == "movement"
