"""Tests for the synthetic stream generator."""

from __future__ import annotations

import numpy as np

from benchmark.datasets.synthetic import scenarios
from benchmark.datasets.synthetic.generator import StreamConfig, SyntheticStream
from benchmark.ground_truth import GroundTruthTracker


def test_stream_reproducible_with_seed():
    _, gt = scenarios.stationary(n_samples=500, seed=7)
    s1 = list(SyntheticStream(gt, StreamConfig(seed=7)))
    s2 = list(SyntheticStream(gt, StreamConfig(seed=7)))
    assert [x for x, _, _ in s1] == [x for x, _, _ in s2]


def test_stream_length_and_labels():
    stream, gt = scenarios.stationary(n_samples=600, seed=1)
    items = list(stream)
    assert len(items) == 600
    labels = {y for _, y, _ in items}
    assert labels == {"C1", "C2", "C3"}


def test_gradual_movement_interpolates_center():
    t = GroundTruthTracker(1000, 2, seed=1)
    t.add_cluster("C1", (0.0, 0.0), spread=0.01)
    t.move("C1", (10.0, 0.0), start=200, end=400)
    gt = t.build()
    stream = SyntheticStream(gt, StreamConfig(seed=1))
    # midpoint of the gradual interval => center ~ (5, 0)
    mid_states = stream._effective_states(300)
    cx = mid_states[0].center[0]
    assert 4.0 < cx < 6.0


def test_noise_produces_unlabeled_samples():
    stream, gt = scenarios.noise_and_drift(n_samples=2000, seed=1)
    labels = [y for _, y, _ in stream]
    assert any(y is None for y in labels)  # background noise present
