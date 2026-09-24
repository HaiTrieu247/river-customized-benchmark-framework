"""Macro purity: the papers' definition, and how it differs from the micro one.

DenStream (SDM 2006, Sec. 5) reports ``pur = (sum_i |C_i^d| / |C_i|) / K`` --
an unweighted mean over clusters. The framework's existing ``purity`` is the
sample-weighted version. These tests pin both, and pin the case that separates
them, because the whole reason for carrying two columns is that they disagree.
"""
from __future__ import annotations

import pytest

from benchmark.evaluation.clustering_quality import (
    EXTERNAL_METRICS,
    StreamingMacroPurity,
    StreamingPurity,
    build_external_metrics,
)


def feed(metric, pairs):
    for y_true, y_pred in pairs:
        metric.update(y_true, y_pred)
    return metric.get()


def test_perfect_partition_is_one_both_ways():
    pairs = [("a", 0)] * 10 + [("b", 1)] * 10
    assert feed(StreamingMacroPurity(), pairs) == pytest.approx(1.0)
    assert feed(StreamingPurity(), pairs) == pytest.approx(1.0)


def test_single_cluster_is_the_majority_share_both_ways():
    pairs = [("a", 0)] * 7 + [("b", 0)] * 3
    assert feed(StreamingMacroPurity(), pairs) == pytest.approx(0.7)
    assert feed(StreamingPurity(), pairs) == pytest.approx(0.7)


def test_macro_and_micro_diverge_when_cluster_sizes_are_uneven():
    """One big mixed cluster plus one tiny pure one.

    micro = (60 + 1) / 101 = 0.604 -- the tiny cluster barely counts.
    macro = (0.6 + 1.0) / 2 = 0.8  -- it counts as much as the big one.
    """
    pairs = [("a", 0)] * 60 + [("b", 0)] * 40 + [("c", 1)] * 1
    assert feed(StreamingPurity(), pairs) == pytest.approx(61 / 101)
    assert feed(StreamingMacroPurity(), pairs) == pytest.approx(0.8)


def test_macro_rewards_fragmentation_harder_than_micro():
    """Splitting a mixed cluster into singletons sends macro to 1, micro stays."""
    mixed = [("a", 0)] * 5 + [("b", 0)] * 5
    split = [("a", i) for i in range(5)] + [("b", 5 + i) for i in range(5)]
    assert feed(StreamingMacroPurity(), mixed) == pytest.approx(0.5)
    assert feed(StreamingMacroPurity(), split) == pytest.approx(1.0)
    # micro also rises here, but it is the macro number that hits the ceiling;
    # this is why neither is readable without the cluster count beside it.
    assert feed(StreamingPurity(), mixed) == pytest.approx(0.5)
    assert feed(StreamingPurity(), split) == pytest.approx(1.0)


def test_denstream_worked_example():
    """Three clusters of different sizes, checked against the paper's formula.

    C0: 8 of class a, 2 of class b -> 0.8
    C1: 3 of class b, 3 of class c -> 0.5
    C2: 1 of class c               -> 1.0
    pur = (0.8 + 0.5 + 1.0) / 3 = 0.766...
    """
    pairs = ([("a", 0)] * 8 + [("b", 0)] * 2
             + [("b", 1)] * 3 + [("c", 1)] * 3
             + [("c", 2)] * 1)
    assert feed(StreamingMacroPurity(), pairs) == pytest.approx((0.8 + 0.5 + 1.0) / 3)


def test_empty_metric_is_zero_not_a_crash():
    assert StreamingMacroPurity().get() == 0.0


def test_revert_undoes_update():
    m = StreamingMacroPurity()
    feed(m, [("a", 0)] * 4 + [("b", 1)] * 4)
    m.update("b", 0)
    assert m.get() == pytest.approx((0.8 + 1.0) / 2)
    m.revert("b", 0)
    assert m.get() == pytest.approx(1.0)


def test_registered_in_the_external_metric_set():
    assert "purity_macro" in EXTERNAL_METRICS
    built = build_external_metrics()
    assert isinstance(built["purity_macro"], StreamingMacroPurity)
    assert isinstance(built["purity"], StreamingPurity)


def test_pipeline_metric_lists_carry_it():
    from hpo_pipeline import _common
    from hpo_pipeline import final_eval

    assert "purity_macro" in _common.EXTERNAL_METRICS
    assert "purity_macro" in final_eval.EXTERNAL_METRICS
