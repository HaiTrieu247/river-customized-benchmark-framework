"""Window buffer, batch silhouette and the run_once predictions collector."""

from __future__ import annotations

import random

import pytest

from benchmark.algorithms.registry import get_algorithm
from benchmark.datasets.base import IterableDataset
from benchmark.evaluation.runner import run_once
from benchmark.tuning.window import UNDEFINED_SCORE, WindowBuffer, batch_silhouette, warm_start


def _two_blobs(n: int, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        c = i % 2
        rows.append(({"a": 5.0 * c + rng.gauss(0, 0.1), "b": rng.gauss(0, 0.1)}, c))
    return rows


def test_run_once_collects_one_prediction_per_sample_in_order():
    rows = _two_blobs(40)
    ds = IterableDataset(lambda: iter(rows), name="blobs", n_features=2, has_labels=True)
    adapter = get_algorithm("KMeans", n_clusters=2, seed=1)

    preds: list = []
    res = run_once(adapter, ds, window_size=20, n_checkpoints=1, predictions=preds)

    assert res.n_samples == 40
    assert len(preds) == 40
    # The collector must see exactly what the metrics saw: after the run the
    # adapter classifies each blob consistently, so the last window's labels
    # alternate between two ids just like the ground truth does.
    tail = preds[-20:]
    assert len(set(tail)) == 2
    assert tail[0::2] == [tail[0]] * 10 and tail[1::2] == [tail[1]] * 10


# --- batch silhouette ---------------------------------------------------------


def test_batch_silhouette_rewards_the_true_partition_of_two_blobs():
    rows = _two_blobs(60)
    xs = [x for x, _ in rows]
    labels = [y for _, y in rows]
    score, defined = batch_silhouette(xs, labels)
    assert defined is True
    assert 0.9 < score <= 1.0


def test_batch_silhouette_is_undefined_for_a_single_label():
    rows = _two_blobs(60)
    xs = [x for x, _ in rows]
    score, defined = batch_silhouette(xs, [0] * 60)
    assert defined is False
    assert score == UNDEFINED_SCORE == -1.0


def test_batch_silhouette_is_undefined_when_every_point_is_its_own_cluster():
    rows = _two_blobs(10)
    xs = [x for x, _ in rows]
    score, defined = batch_silhouette(xs, list(range(10)))
    assert defined is False
    assert score == UNDEFINED_SCORE


def test_batch_silhouette_ignores_none_predictions_but_scores_the_rest():
    rows = _two_blobs(60)
    xs = [x for x, _ in rows]
    labels = [y for _, y in rows]
    labels[0] = None  # a fallback prediction: no cluster id at all
    score, defined = batch_silhouette(xs, labels)
    assert defined is True
    assert 0.9 < score <= 1.0


# --- window buffer -------------------------------------------------------------


def test_window_buffer_fills_to_h_then_reports_full():
    buf = WindowBuffer(h=3)
    rows = _two_blobs(5)
    assert buf.push(*rows[0]) is False
    assert buf.push(*rows[1]) is False
    assert buf.push(*rows[2]) is True
    assert buf.is_full and len(buf) == 3
    with pytest.raises(ValueError):
        buf.push(*rows[3])


def test_window_buffer_dataset_is_reiterable_and_never_hands_out_its_own_dicts():
    buf = WindowBuffer(h=4)
    rows = _two_blobs(4)
    for x, y in rows:
        buf.push(x, y)
    ds = buf.as_dataset(name="w")
    first = [(x, y) for x, y in ds]
    second = [(x, y) for x, y in ds]
    assert [y for _, y in first] == [y for _, y in second] == [0, 1, 0, 1]
    assert ds.has_labels is True and len(ds) == 4
    # River clusterers keep a reference to the dict they are given (BUG-3), so
    # every member of the ensemble must receive its own copy of each point.
    first[0][0]["a"] = 999.0
    assert buf.xs[0]["a"] != 999.0
    assert second[0][0]["a"] != 999.0


def test_window_buffer_clear_resets_it_and_keeps_h():
    buf = WindowBuffer(h=2)
    rows = _two_blobs(2)
    for x, y in rows:
        buf.push(x, y)
    buf.clear()
    assert len(buf) == 0 and not buf.is_full and buf.h == 2


def test_warm_start_trains_a_fresh_model_on_the_buffer_without_predicting():
    buf = WindowBuffer(h=40)
    for x, y in _two_blobs(40):
        buf.push(x, y)
    adapter = get_algorithm("KMeans", n_clusters=2, seed=1)
    n = warm_start(adapter, buf)
    assert n == 40
    assert adapter.n_fallback_predictions == 0
    # After replaying the two blobs the model separates them.
    left = adapter.predict_one({"a": 0.0, "b": 0.0})
    right = adapter.predict_one({"a": 5.0, "b": 0.0})
    assert left != right
