"""Tests for the ground-truth layer (schema, events, tracker)."""

from __future__ import annotations

import pytest

from benchmark.ground_truth import DriftType, GroundTruthTracker
from benchmark.ground_truth.events import DriftEvent


def test_drift_event_validation():
    with pytest.raises(ValueError):
        DriftEvent(1, DriftType.MOVEMENT, start=100, end=50)


def test_drift_event_gradual_flag():
    abrupt = DriftEvent(1, DriftType.MOVEMENT, start=10, end=10)
    gradual = DriftEvent(2, DriftType.MOVEMENT, start=10, end=20)
    assert not abrupt.is_gradual and abrupt.duration == 0
    assert gradual.is_gradual and gradual.duration == 10


def test_tracker_movement_emits_event_and_state():
    t = GroundTruthTracker(1000, 2, seed=1)
    t.add_cluster("C1", (0.0, 0.0))
    t.move("C1", (3.0, 4.0), start=500)
    gt = t.build()
    assert len(gt.drift_events) == 1
    ev = gt.drift_events[0]
    assert ev.type == DriftType.MOVEMENT
    assert ev.magnitude == pytest.approx(5.0)  # 3-4-5 triangle
    # state before/after the drift
    assert gt.active_states_at(499)[0].center == (0.0, 0.0)
    assert gt.active_states_at(501)[0].center == (3.0, 4.0)


def test_tracker_birth_death():
    t = GroundTruthTracker(1000, 2, seed=1)
    t.add_cluster("C1", (0.0, 0.0))
    t.birth("C2", (5.0, 5.0), at=300)
    t.death("C1", at=600)
    gt = t.build()
    assert {e.type for e in gt.drift_events} == {DriftType.BIRTH, DriftType.DEATH}
    labels_before = set(gt.true_labels_at(100))
    labels_mid = set(gt.true_labels_at(400))
    labels_end = set(gt.true_labels_at(700))
    assert labels_before == {"C1"}
    assert labels_mid == {"C1", "C2"}
    assert labels_end == {"C2"}


def test_tracker_split_and_merge():
    t = GroundTruthTracker(1000, 2, seed=1)
    t.add_cluster("C1", (0.0, 0.0))
    t.split("C1", {"C1a": (-2.0, 0.0), "C1b": (2.0, 0.0)}, at=400)
    gt = t.build()
    assert set(gt.true_labels_at(500)) == {"C1a", "C1b"}
    assert set(gt.true_labels_at(300)) == {"C1"}
