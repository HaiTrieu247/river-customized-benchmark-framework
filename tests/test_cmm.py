"""CMM (benchmark.evaluation.cmm): parity with MOA's own implementation, and the properties
the measure is built to have."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from benchmark.algorithms.adapter import build_river_path

build_river_path()  # benchmark.evaluation imports river from the sibling tree

from benchmark.evaluation import cmm as M  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "cmm_moa_cases.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_every_reading_matches_moa(case):
    """All eight readings of MOA's collection, for each case MOA scored (tests/fixtures,
    produced by MOA 2024.07.0). The port reproduces MOA's arithmetic order, so the match is
    exact on every case so far; 1e-12 leaves room for a last-bit difference in pow."""
    window = M.CMMWindow.build(case["X"], case["y"], case["weights"])
    got = window.measures(case["pred"])
    for name, expected in case["moa"].items():
        if expected is None:  # MOA's NaN
            assert math.isnan(got[name]), name
        else:
            assert got[name] == pytest.approx(expected, abs=1e-12), name


def test_default_weights_are_moas_batch_fading():
    """``EvaluateClustering`` (BatchCmd) fades with rate ln(1/0.01)/h used as a power of 2,
    so the oldest point of a window weighs 2 ** -ln(100) (about 0.041), not 0.01."""
    w = M.fading_weights(1000)
    assert w[-1] == 1.0
    assert w[0] == pytest.approx(2 ** (-math.log(100) * 999 / 1000), rel=1e-12)
    assert np.all(np.diff(w) > 0)
    vis = M.fading_weights(1000, rule="visualizer")
    assert vis[0] == pytest.approx(0.01 ** (999 / 1000), rel=1e-12)
    with pytest.raises(ValueError):
        M.fading_weights(10, rule="other")


def _blobs(seed=0, n_per=40, sd=0.2):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, sd, size=(n_per, 2)) for c in (0.0, 6.0, 12.0)])
    y = np.repeat([0, 1, 2], n_per)
    return X, y


def test_a_perfect_partition_scores_one_and_a_split_class_costs_nothing():
    """CMM maps clusters to classes, so splitting a class into several clusters is no fault
    (Kremer et al. 2011, Sec. 4), while merging two classes is."""
    X, y = _blobs()
    w = M.CMMWindow.build(X, y)
    assert w.score(y) == 1.0
    split = y.copy()
    split[(y == 0) & (np.arange(len(y)) % 2 == 0)] = 9
    assert w.score(split) == 1.0
    merged = np.where(y == 2, 1, y)
    assert w.score(merged) < 0.8
    assert w.score(np.zeros(len(y), int)) < w.score(merged)


def test_missed_points_cost_one_minus_exp_minus_lambda_when_clusters_are_labels():
    """A label-defined cluster is not a sphere, so MOA's hull distance is 1 and a missed
    point's penalty is (1 - e^-lambda) times its connectivity."""
    X, y = _blobs()
    w = M.CMMWindow.build(X, y)
    # every penalty is (1 - e^-1) times the connectivity the ratio divides by
    assert w.score([None] * len(y)) == pytest.approx(math.exp(-1.0), rel=1e-12)
    w2 = M.CMMWindow.build(X, y, lambda_missed=3.0)
    assert w2.score([None] * len(y)) < w.score([None] * len(y))


def test_sentinel_labels_count_as_unassigned():
    X, y = _blobs()
    pred = list(y)
    for i in range(0, len(pred), 5):
        pred[i] = -1
    w = M.CMMWindow.build(X, y)
    as_none = [None if p == -1 else p for p in pred]
    assert w.score(pred, noise_labels=frozenset({-1})) == w.score(as_none)
    assert w.score(pred) != w.score(as_none)  # without noise_labels, -1 is a cluster


def test_undefined_windows_read_none_not_a_score():
    w = M.CMMWindow.build(np.zeros((0, 3)), [])
    assert w.score([]) is None
    assert M.cmm(np.zeros((0, 3)), [], []) is None
    # every class a singleton: no point has a neighbour in its class, connectivity 0
    # everywhere, and MOA's ratio has nothing to divide by (MOA reports 1)
    X = np.arange(12.0).reshape(4, 3)
    w = M.CMMWindow.build(X, [0, 1, 2, 3])
    assert w.measures([0, 0, 1, 1])["cmm"] == 1.0
    assert w.score([0, 0, 1, 1]) is None
    # zero weights likewise
    X, y = _blobs()
    assert M.CMMWindow.build(X, y, weights=np.zeros(len(y))).score(y) is None


def test_dict_rows_and_the_convenience_function_agree_with_arrays():
    X, y = _blobs(seed=3)
    rng = np.random.default_rng(1)
    pred = rng.integers(0, 4, len(y))
    rows = [{"b": float(r[1]), "a": float(r[0])} for r in X]  # read in sorted key order
    a = M.CMMWindow.build(X, y).score(pred)
    assert M.CMMWindow.build(rows, y).score(pred) == a
    assert M.cmm(X, y, pred) == a


def test_ground_truth_noise_is_none():
    rng = np.random.default_rng(2)
    X, y = _blobs(seed=2)
    noise = rng.uniform(-2, 14, size=(20, 2))
    Xn = np.vstack([X, noise])
    yn = list(y) + [None] * 20
    w = M.CMMWindow.build(Xn, yn)
    perfect = list(y) + [None] * 20
    assert w.score(perfect) == 1.0  # noise left out of every cluster is no fault
    swallowed = list(y) + [0] * 20
    assert w.score(swallowed) < 1.0


def test_options_switch_off_the_merge_and_the_model_excuse():
    """Two heavily overlapping classes are merged by MOA's analysis, so mixing them is no
    fault; with the merge off it is."""
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(0, 1, size=(60, 2)), rng.normal(0.3, 1, size=(60, 2)),
                   rng.normal(12, 1, size=(60, 2))])
    y = np.repeat([0, 1, 2], 60)
    mixed = np.where(y == 2, 2, rng.integers(0, 2, len(y)))
    w = M.CMMWindow.build(X, y)
    assert w.n_gt0 == 2 and w.score(mixed) == 1.0
    strict = M.CMMWindow.build(X, y, enable_class_merge=False, enable_model_error=False)
    assert strict.n_gt0 == 3 and strict.score(mixed) < 1.0


def test_lengths_must_match():
    X, y = _blobs()
    with pytest.raises(ValueError):
        M.CMMWindow.build(X, y[:-1])
    with pytest.raises(ValueError):
        M.CMMWindow.build(X, y).score(y[:-1])
    with pytest.raises(ValueError):
        M.CMMWindow.build(X, y, weights=np.ones(3))


@pytest.mark.parametrize("seed", range(12))
def test_the_fast_enclosing_ball_is_moas_ball(seed):
    """``_meb`` (Fischer-Gaertner-Kutz) against the line-for-line port of MOA's Miniball,
    including duplicate points and fewer points than dimensions."""
    rng = np.random.default_rng(seed)
    n, d = int(rng.integers(1, 80)), int(rng.integers(1, 30))
    P = rng.normal(size=(n, d)) * rng.uniform(0.1, 5, size=d)
    if seed % 3 == 0:
        P = np.round(P)
    if seed % 4 == 0 and n > 3:
        P[: n // 2] = P[0]
    c_fast, r_fast = M._meb(P)
    c_moa, r_moa = M._miniball_moa(P)
    scale = max(math.sqrt(r_moa), 1e-12)
    assert math.sqrt(r_fast) == pytest.approx(math.sqrt(r_moa), rel=1e-9, abs=1e-12)
    assert np.linalg.norm(c_fast - c_moa) <= 1e-7 * scale
    assert np.sqrt(((P - c_fast) ** 2).sum(axis=1)).max() <= math.sqrt(r_fast) * (1 + 1e-12) + 1e-300
