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
from benchmark.experiments import BenchmarkConfig, Setting, run_benchmark, run_setting


def _dataset(n=4000, k=3, seed=1, d=2):
    # d is pinned rather than left to the default so these tests keep asserting
    # what they mean when the generator default changes.
    return SyntheticStream(
        StreamConfig(n_samples=n, n_clusters=k, n_features=d, seed=seed)
    )


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


# --- scaler configuration ---------------------------------------------------
def test_make_scaler_forwards_params_and_rejects_typos():
    import pytest
    from river import preprocessing

    from benchmark.algorithms.registry import make_scaler

    plain = make_scaler("standard")
    assert isinstance(plain, preprocessing.StandardScaler)
    assert plain.window_size is None  # accumulates over the whole stream

    rolling = make_scaler("standard", window_size=500)
    assert rolling.window_size == 500  # tracks drift instead

    fading = make_scaler("adaptive_standard", fading_factor=0.05)
    assert fading.fading_factor == 0.05

    assert make_scaler("none") is None
    with pytest.raises(TypeError):
        make_scaler("standard", windwo_size=500)  # typo must not pass silently


def test_scaler_params_reach_the_result_and_the_label():
    setting = Setting("DBSTREAM", scaler="standard", scaler_params={"window_size": 400})
    assert "window_size=400" in setting.display()

    config = BenchmarkConfig(settings=[setting], n_checkpoints=3, window_size=500)
    (result,) = run_benchmark(_dataset(n=2000), config)
    assert result.scaler == "standard"
    assert result.scaler_params == {"window_size": 400}
    assert result.to_dict()["scaler_parameters"] == {"window_size": 400}


def test_rolling_scaler_forgets_a_shifted_stream_and_cumulative_one_does_not():
    """The reason window_size matters: only the rolling scaler re-centres on drift."""
    from benchmark.algorithms.registry import make_scaler

    cumulative = make_scaler("standard")
    rolling = make_scaler("standard", window_size=100)
    fading = make_scaler("adaptive_standard", fading_factor=0.1)

    # 1000 samples around 0, then a hard shift to a mean of 50
    stream = [{"a": float(i % 3)} for i in range(1000)] + [{"a": 50.0 + i % 3} for i in range(300)]
    out = {}
    for name, scaler in (("cum", cumulative), ("roll", rolling), ("fade", fading)):
        for x in stream:
            scaler.learn_one(x)
            out[name] = scaler.transform_one(x)["a"]

    # After the shift the rolling and fading scalers have re-centred (|z| small),
    # while the cumulative one still measures against the pre-shift statistics.
    assert abs(out["roll"]) < 2
    assert abs(out["fade"]) < 2
    assert abs(out["cum"]) > abs(out["roll"])


# --- evaluation protocol ----------------------------------------------------
def test_window_summary_uses_per_window_values_not_the_cumulative_curve():
    adapter = get_algorithm("DBSTREAM", scaler="standard")
    res = run_once(adapter, _dataset(n=4000), n_checkpoints=4, window_size=500)

    # both views are exposed and they are genuinely different series
    per_window = [w.window_metrics["ari"] for w in res.windows]
    cumulative = [w.metrics["ari"] for w in res.windows]
    assert per_window != cumulative

    # quality_stream summarises the per-window series (the time-aware protocol)
    assert res.quality_stream["ari"]["final"] == per_window[-1]
    assert res.quality_cumulative_stream["ari"]["final"] == cumulative[-1]
    # the cumulative curve ends at the whole-stream score
    assert abs(cumulative[-1] - res.quality["ari"]) < 1e-9


def test_memory_is_probed_only_at_checkpoints():
    """The probe walks the whole model graph; per-sample calls made runs unusable."""
    adapter = get_algorithm("STREAMKMeans", n_clusters=3, scaler="standard")
    calls = {"n": 0}
    real = adapter.memory_bytes

    def counted():
        calls["n"] += 1
        return real()

    adapter.memory_bytes = counted
    res = run_once(adapter, _dataset(n=4000), n_checkpoints=5, window_size=1000)
    assert calls["n"] == len(res.checkpoints)
    assert calls["n"] <= 6  # n_checkpoints (+ at most one final snapshot)


def test_seed_is_injected_only_where_the_constructor_accepts_one():
    from benchmark.algorithms.registry import accepts_seed

    assert accepts_seed("STREAMKMeans") is True
    assert accepts_seed("DBSTREAM") is False

    dataset = _dataset(n=1500)
    config = BenchmarkConfig(n_checkpoints=2, window_size=500)
    seeded = run_setting(Setting("STREAMKMeans", params={"n_clusters": 3}), dataset, config, seed=7)
    assert seeded.params["seed"] == 7
    assert seeded.seed == 7

    unseeded = run_setting(Setting("DBSTREAM"), dataset, config, seed=7)
    assert "seed" not in unseeded.params
    assert unseeded.seed == 7  # still recorded on the run


def test_bad_algorithm_params_are_not_swallowed():
    import pytest

    with pytest.raises(TypeError):
        run_setting(
            Setting("DBSTREAM", params={"nonexistent_param": 1}),
            _dataset(n=500), BenchmarkConfig(), seed=1,
        )


def test_cold_start_fallback_predictions_are_counted():
    # STREAMKMeans has no center until its first chunk is complete
    adapter = get_algorithm("STREAMKMeans", n_clusters=3, chunk_size=200, scaler="standard")
    res = run_once(adapter, _dataset(n=2000), n_checkpoints=2, window_size=500)
    assert res.n_fallback_predictions >= 1
    assert res.to_dict()["n_fallback_predictions"] == res.n_fallback_predictions


def test_environment_record_is_captured():
    from benchmark.reports import capture_environment, format_environment

    env = capture_environment()
    assert env["python"]
    assert env["os"]
    assert env["libraries"]["river"] or env.get("river_version")
    assert "Key" in format_environment(env)


def test_rolling_standard_scaler_never_emits_non_real_values():
    """A constant feature drives the rolling variance slightly negative.

    ``v ** 0.5`` on a negative float is complex in Python, and a complex feature
    only fails much later, inside the clusterer's distance function. Guard it
    here: the scaler must always hand the model real numbers.
    """
    from benchmark.algorithms.registry import make_scaler

    scaler = make_scaler("standard", window_size=100)
    for i in range(5000):
        # 'varying' moves, 'constant' never does
        x = {"varying": float(i % 7), "constant": 0.0}
        scaler.learn_one(x)
        out = scaler.transform_one(x)
        assert all(isinstance(v, float) for v in out.values()), (i, out)


def test_single_class_windows_are_flagged_as_degenerate():
    """A window holding one class scores a free 1.0 — the report must say so."""
    from benchmark.datasets.base import IterableDataset

    base = _dataset(n=4000, k=4)
    samples = list(base)

    # block-ordered stream: every 1000-sample window holds exactly one class,
    # the shape KDDCup99 arrives in
    blocked = IterableDataset(
        lambda: ((x, f"class{i // 1000}") for i, (x, _) in enumerate(samples)),
        name="blocked", n_features=base.n_features, has_labels=True,
    )
    res = run_once(get_algorithm("DBSTREAM", scaler="standard"), blocked,
                   n_checkpoints=4, window_size=1000)
    diag = res.window_label_diagnostics
    assert diag["single_class_window_share"] == 1.0
    assert all(w.is_degenerate for w in res.windows)
    # On such a window ARI is only a "did you lump it?" test: 1.0 for lumping,
    # 0.0 for any split — never a measure of how good the clustering is.
    from river import metrics

    def _ari(y_true, y_pred):
        m = metrics.AdjustedRand()
        for t, p in zip(y_true, y_pred):
            m.update(t, p)
        return m.get()

    assert _ari(["A"] * 100, [0] * 100) == 1.0
    assert _ari(["A"] * 100, [i % 2 for i in range(100)]) == 0.0
    assert _ari(["A"] * 100, [i % 10 for i in range(100)]) == 0.0

    # a stream whose windows are class-mixed is not flagged
    mixed = IterableDataset(
        lambda: ((x, f"class{i % 4}") for i, (x, _) in enumerate(samples)),
        name="mixed", n_features=base.n_features, has_labels=True,
    )
    res2 = run_once(get_algorithm("DBSTREAM", scaler="standard"), mixed,
                    n_checkpoints=4, window_size=1000)
    assert res2.window_label_diagnostics["single_class_window_share"] == 0.0
    assert res2.window_label_diagnostics["n_windows"] == 4
