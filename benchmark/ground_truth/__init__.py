"""Ground-truth layer: cluster states, drift events and the evolution tracker.

Ground truth describes *how the cluster structure changes over time* — not just
"drift happened at sample N". It is used **only** for evaluation and must never
leak into the model (Principle 1).
"""

from __future__ import annotations

from .events import DriftEvent, DriftType
from .schema import ClusterState, StreamGroundTruth
from .tracker import GroundTruthTracker

__all__ = [
    "ClusterState",
    "StreamGroundTruth",
    "DriftEvent",
    "DriftType",
    "GroundTruthTracker",
]
