"""Helper to build a :class:`StreamGroundTruth` incrementally (§4, tracker.py).

The tracker turns a sequence of high-level structural operations (move / birth /
death / split / merge / recur) into (a) the cluster-state timeline and (b) the
matching :class:`DriftEvent` metadata, keeping the two consistent.
"""

from __future__ import annotations

from .events import DriftEvent, DriftType
from .schema import ClusterState, StreamGroundTruth


class GroundTruthTracker:
    """Fluent builder for ground-truth cluster evolution."""

    def __init__(self, n_samples: int, n_features: int, seed: int, name: str = "unnamed"):
        self.gt = StreamGroundTruth(
            n_samples=n_samples, n_features=n_features, seed=seed, name=name
        )
        self._next_drift_id = 1
        # current live state per cluster id
        self._current: dict[str, ClusterState] = {}

    # -- initial layout ------------------------------------------------------
    def add_cluster(
        self,
        cluster_id: str,
        center: tuple[float, ...],
        spread: float = 1.0,
        weight: float = 1.0,
        t_start: int = 0,
    ) -> "GroundTruthTracker":
        state = ClusterState(cluster_id, center, spread, weight, True, t_start)
        self.gt.add_state(state)
        self._current[cluster_id] = state
        return self

    def _emit(self, drift_type: DriftType, start: int, end: int, affected, magnitude=None, note="") -> None:
        self.gt.add_event(
            DriftEvent(
                drift_id=self._next_drift_id,
                type=drift_type,
                start=start,
                end=end,
                affected_clusters=tuple(affected),
                magnitude=magnitude,
                note=note,
            )
        )
        self._next_drift_id += 1

    # -- drift operations ----------------------------------------------------
    def move(
        self,
        cluster_id: str,
        new_center: tuple[float, ...],
        start: int,
        end: int | None = None,
    ) -> "GroundTruthTracker":
        """Move a cluster to ``new_center`` (§5.1). ``end>start`` => gradual (§5.7)."""
        end = start if end is None else end
        cur = self._current[cluster_id]
        magnitude = float(
            sum((a - b) ** 2 for a, b in zip(new_center, cur.center)) ** 0.5
        )
        new_state = ClusterState(
            cluster_id, new_center, cur.spread, cur.weight, True, end
        )
        self.gt.add_state(new_state)
        self._current[cluster_id] = new_state
        self._emit(DriftType.MOVEMENT, start, end, [cluster_id], magnitude)
        return self

    def birth(
        self,
        cluster_id: str,
        center: tuple[float, ...],
        at: int,
        spread: float = 1.0,
        weight: float = 1.0,
    ) -> "GroundTruthTracker":
        """A new cluster appears (§5.2)."""
        state = ClusterState(cluster_id, center, spread, weight, True, at)
        self.gt.add_state(state)
        self._current[cluster_id] = state
        self._emit(DriftType.BIRTH, at, at, [cluster_id])
        return self

    def death(self, cluster_id: str, at: int) -> "GroundTruthTracker":
        """An existing cluster disappears (§5.3)."""
        cur = self._current[cluster_id]
        dead = ClusterState(cluster_id, cur.center, cur.spread, 0.0, False, at)
        self.gt.add_state(dead)
        self._current.pop(cluster_id, None)
        self._emit(DriftType.DEATH, at, at, [cluster_id])
        return self

    def split(
        self,
        cluster_id: str,
        children: dict[str, tuple[float, ...]],
        at: int,
    ) -> "GroundTruthTracker":
        """Split one cluster into several children (§5.4)."""
        cur = self._current[cluster_id]
        # parent dies
        self.gt.add_state(
            ClusterState(cluster_id, cur.center, cur.spread, 0.0, False, at)
        )
        self._current.pop(cluster_id, None)
        share = cur.weight / max(len(children), 1)
        for child_id, center in children.items():
            child = ClusterState(child_id, center, cur.spread, share, True, at)
            self.gt.add_state(child)
            self._current[child_id] = child
        self._emit(DriftType.SPLIT, at, at, [cluster_id, *children.keys()])
        return self

    def merge(
        self,
        parents: list[str],
        target_id: str,
        center: tuple[float, ...],
        at: int,
    ) -> "GroundTruthTracker":
        """Merge several clusters into one (§5.5)."""
        total_w = 0.0
        spread = 1.0
        for pid in parents:
            cur = self._current.get(pid)
            if cur is None:
                continue
            total_w += cur.weight
            spread = cur.spread
            self.gt.add_state(
                ClusterState(pid, cur.center, cur.spread, 0.0, False, at)
            )
            self._current.pop(pid, None)
        merged = ClusterState(target_id, center, spread, total_w or 1.0, True, at)
        self.gt.add_state(merged)
        self._current[target_id] = merged
        self._emit(DriftType.MERGE, at, at, [*parents, target_id])
        return self

    def recur(
        self,
        cluster_id: str,
        center: tuple[float, ...],
        start: int,
        end: int | None = None,
    ) -> "GroundTruthTracker":
        """A previous concept/position returns (§5.6)."""
        end = start if end is None else end
        cur = self._current[cluster_id]
        new_state = ClusterState(cluster_id, center, cur.spread, cur.weight, True, end)
        self.gt.add_state(new_state)
        self._current[cluster_id] = new_state
        self._emit(DriftType.RECURRING, start, end, [cluster_id])
        return self

    def build(self) -> StreamGroundTruth:
        return self.gt
