"""Cluster-state representation and per-stream ground truth (§3, §4, §7)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from .events import DriftEvent


@dataclass(frozen=True)
class ClusterState:
    """State of one ground-truth cluster over a time interval (§4).

    A cluster is a spherical Gaussian blob: ``center`` +/- ``spread`` (std-dev),
    carrying a ``weight`` (relative sampling probability) and an ``active`` flag.
    States are snapshots; the evolution of a cluster is a sequence of states with
    increasing ``t_start``.
    """

    cluster_id: str
    center: tuple[float, ...]
    spread: float = 1.0
    weight: float = 1.0
    active: bool = True
    t_start: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", tuple(float(x) for x in self.center))
        if self.spread < 0:
            raise ValueError("spread must be >= 0")
        if self.weight < 0:
            raise ValueError("weight must be >= 0")

    @property
    def n_features(self) -> int:
        return len(self.center)

    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float)


@dataclass
class StreamGroundTruth:
    """Complete ground truth for one synthetic stream.

    Holds the drift events and a per-cluster timeline of :class:`ClusterState`
    snapshots. This is the reference the evaluator uses; the model never sees it
    (Principle 1).
    """

    n_samples: int
    n_features: int
    seed: int
    drift_events: list[DriftEvent] = field(default_factory=list)
    # cluster_id -> list of states sorted by t_start
    timelines: dict[str, list[ClusterState]] = field(default_factory=dict)
    name: str = "unnamed"

    # -- construction helpers ------------------------------------------------
    def add_state(self, state: ClusterState) -> None:
        line = self.timelines.setdefault(state.cluster_id, [])
        line.append(state)
        line.sort(key=lambda s: s.t_start)

    def add_event(self, event: DriftEvent) -> None:
        self.drift_events.append(event)
        self.drift_events.sort(key=lambda e: e.start)

    # -- queries -------------------------------------------------------------
    def active_states_at(self, t: int) -> list[ClusterState]:
        """Return the active cluster states in effect at sample index ``t``."""
        out: list[ClusterState] = []
        for line in self.timelines.values():
            starts = [s.t_start for s in line]
            idx = bisect.bisect_right(starts, t) - 1
            if idx < 0:
                continue
            state = line[idx]
            if state.active:
                out.append(state)
        return out

    def true_labels_at(self, t: int) -> list[str]:
        """Ids of the clusters that are active at time ``t`` (label space)."""
        return [s.cluster_id for s in self.active_states_at(t)]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "seed": self.seed,
            "drifts": [e.to_dict() for e in self.drift_events],
            "timelines": {
                cid: [
                    {
                        "center": list(s.center),
                        "spread": s.spread,
                        "weight": s.weight,
                        "active": s.active,
                        "t_start": s.t_start,
                    }
                    for s in states
                ]
                for cid, states in self.timelines.items()
            },
        }
