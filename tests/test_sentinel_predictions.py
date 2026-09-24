"""BUG-6: DenStream's ``-1`` sentinel is not a cluster id.

The adapter counts it always and suppresses it only on request, so the default
measurement stays River's own behaviour and the alternative is explicit.
"""
from __future__ import annotations

import pytest

from benchmark.algorithms.adapter import SENTINEL_LABELS, ClustererAdapter


class FakeDenStream:
    """Stands in for River's DenStream: returns the sentinel until it has seen
    ``ready_after`` points, then a real cluster id."""

    def __init__(self, ready_after=3):
        self.ready_after = ready_after
        self.seen = 0

    def learn_one(self, x, w=None):
        self.seen += 1

    def predict_one(self, x, w=None):
        return -1 if self.seen < self.ready_after else 7


def adapter(**kw):
    a = ClustererAdapter(name="DenStream", model=FakeDenStream(), **kw)
    # the lookup is keyed on the River class name
    SENTINEL_LABELS.setdefault("FakeDenStream", (-1,))
    return a


def test_denstream_is_the_registered_sentinel_owner():
    assert SENTINEL_LABELS["DenStream"] == (-1,)


def test_sentinels_are_counted_and_by_default_still_returned():
    """Default = River's behaviour, so the pilot's existing numbers stay
    reproducible; the count is what makes the degeneracy visible."""
    a = adapter()
    preds = [a.process_one({"v": i}) for i in range(6)]
    assert a.n_sentinel_predictions == 3
    assert preds[:3] == [-1, -1, -1]
    assert preds[3:] == [7, 7, 7]


def test_opting_in_returns_none_so_the_runner_skips_the_sample():
    a = adapter(drop_sentinel_predictions=True)
    preds = [a.process_one({"v": i}) for i in range(6)]
    assert a.n_sentinel_predictions == 3
    assert preds[:3] == [None, None, None]
    assert preds[3:] == [7, 7, 7]


def test_a_real_label_is_never_counted_as_a_sentinel():
    a = adapter()
    for i in range(10):
        a.process_one({"v": i})
    assert a.n_sentinel_predictions == 3   # not 10


def test_sentinels_are_not_confused_with_the_cold_start_fallback():
    a = adapter()
    for i in range(6):
        a.process_one({"v": i})
    assert a.n_fallback_predictions == 0


def test_other_algorithms_are_untouched_by_the_workaround():
    """-1 is a legitimate label for a model not in the table."""
    class MinusOneIsReal:
        def learn_one(self, x, w=None):
            pass

        def predict_one(self, x, w=None):
            return -1

    a = ClustererAdapter(name="Other", model=MinusOneIsReal(),
                         drop_sentinel_predictions=True)
    assert a.process_one({"v": 1}) == -1
    assert a.n_sentinel_predictions == 0


def test_real_denstream_at_a_tiny_epsilon_emits_sentinels():
    """Not a mock: the condition really occurs in River."""
    import random

    from river import cluster

    model = cluster.DenStream(epsilon=1e-4, mu=2, beta=0.75,
                              n_samples_init=10, stream_speed=10)
    a = ClustererAdapter(name="DenStream", model=model)
    rng = random.Random(0)
    for _ in range(200):
        a.process_one({"x": rng.random() * 1000, "y": rng.random() * 1000})
    assert a.n_sentinel_predictions > 0


def test_the_count_reaches_the_run_result():
    from benchmark.evaluation.runner import RunResult

    from benchmark.evaluation.efficiency import EfficiencyMetrics

    result = RunResult(algorithm="DenStream", dataset="d", scaler="none", params={},
                       n_samples=1, n_features=1,
                       efficiency=EfficiencyMetrics(total_runtime_s=1.0, n_samples=1),
                       n_sentinel_predictions=42)
    assert result.to_dict()["n_sentinel_predictions"] == 42
