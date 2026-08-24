"""Drift event taxonomy and schema (direction doc §5, §7)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class DriftType(str, enum.Enum):
    """Taxonomy of cluster-structure changes the framework supports (§5)."""

    MOVEMENT = "movement"  # a cluster changes position
    BIRTH = "birth"  # a new cluster appears
    DEATH = "death"  # an existing cluster disappears
    SPLIT = "split"  # one cluster splits into several
    MERGE = "merge"  # several clusters merge into one
    RECURRING = "recurring"  # a previous concept returns

    def __str__(self) -> str:  # nicer repr in tables / reports
        return self.value


@dataclass(frozen=True)
class DriftEvent:
    """A single drift event in ground-truth metadata (§7).

    ``start == end`` denotes an abrupt drift; ``start < end`` a gradual drift
    spanning an interval (§5.7 — never force every drift to one timestamp).
    """

    drift_id: int
    type: DriftType
    start: int
    end: int
    affected_clusters: tuple[str, ...] = ()
    magnitude: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"drift {self.drift_id}: end ({self.end}) < start ({self.start})"
            )
        if not isinstance(self.type, DriftType):
            object.__setattr__(self, "type", DriftType(self.type))
        # normalise affected_clusters to a tuple so the event stays hashable
        object.__setattr__(self, "affected_clusters", tuple(self.affected_clusters))

    @property
    def is_gradual(self) -> bool:
        return self.end > self.start

    @property
    def duration(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "id": self.drift_id,
            "type": str(self.type),
            "start": self.start,
            "end": self.end,
            "affected_clusters": list(self.affected_clusters),
            "magnitude": self.magnitude,
            "note": self.note,
        }
