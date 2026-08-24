"""Pre-built benchmark scenarios — the S1..S10 matrix (§16).

Each factory returns ``(SyntheticStream, StreamGroundTruth)`` for a fixed seed so
every algorithm is run on the *same* stream (§14). Scale is kept modest by
default so the whole matrix runs quickly; pass ``n_samples`` to enlarge.
"""

from __future__ import annotations

from typing import Callable

from ...ground_truth.tracker import GroundTruthTracker
from .generator import StreamConfig, SyntheticStream


def _stream(gt, noise: float = 0.0, noise_scale: float = 20.0) -> SyntheticStream:
    cfg = StreamConfig(
        n_samples=gt.n_samples,
        n_features=gt.n_features,
        noise=noise,
        noise_scale=noise_scale,
        seed=gt.seed,
    )
    return SyntheticStream(gt, cfg)


def stationary(n_samples: int = 20_000, seed: int = 42):
    """S1 — no drift. Baseline (§17)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S1_stationary")
    t.add_cluster("C1", (0.0, 0.0), spread=1.0)
    t.add_cluster("C2", (10.0, 10.0), spread=1.0)
    t.add_cluster("C3", (0.0, 10.0), spread=1.0)
    gt = t.build()
    return _stream(gt), gt


def abrupt_movement(n_samples: int = 20_000, seed: int = 42):
    """S2 — abrupt cluster movement at the midpoint (tracking ability)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S2_abrupt_movement")
    t.add_cluster("C1", (0.0, 0.0), spread=1.0)
    t.add_cluster("C2", (10.0, 10.0), spread=1.0)
    t.move("C1", (8.0, 0.0), start=n_samples // 2)
    gt = t.build()
    return _stream(gt), gt


def gradual_movement(n_samples: int = 20_000, seed: int = 42):
    """S3 — gradual cluster movement over an interval (§5.7)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S3_gradual_movement")
    t.add_cluster("C1", (0.0, 0.0), spread=1.0)
    t.add_cluster("C2", (10.0, 10.0), spread=1.0)
    lo = int(n_samples * 0.4)
    hi = int(n_samples * 0.6)
    t.move("C1", (8.0, 0.0), start=lo, end=hi)
    gt = t.build()
    return _stream(gt), gt


def birth(n_samples: int = 20_000, seed: int = 42):
    """S4 — a new cluster appears (§5.2)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S4_birth")
    t.add_cluster("C1", (0.0, 0.0), spread=1.0)
    t.add_cluster("C2", (10.0, 10.0), spread=1.0)
    t.birth("C3", (0.0, 10.0), at=n_samples // 2)
    gt = t.build()
    return _stream(gt), gt


def death(n_samples: int = 20_000, seed: int = 42):
    """S5 — an existing cluster disappears (§5.3)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S5_death")
    t.add_cluster("C1", (0.0, 0.0), spread=1.0)
    t.add_cluster("C2", (10.0, 10.0), spread=1.0)
    t.add_cluster("C3", (0.0, 10.0), spread=1.0)
    t.death("C3", at=n_samples // 2)
    gt = t.build()
    return _stream(gt), gt


def split(n_samples: int = 20_000, seed: int = 42):
    """S6 — one cluster splits into two (§5.4)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S6_split")
    t.add_cluster("C1", (0.0, 0.0), spread=1.2)
    t.add_cluster("C2", (10.0, 10.0), spread=1.0)
    t.split("C1", {"C1a": (-4.0, 0.0), "C1b": (4.0, 0.0)}, at=n_samples // 2)
    gt = t.build()
    return _stream(gt), gt


def merge(n_samples: int = 20_000, seed: int = 42):
    """S7 — two clusters merge into one (§5.5)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S7_merge")
    t.add_cluster("C1", (-4.0, 0.0), spread=1.0)
    t.add_cluster("C2", (4.0, 0.0), spread=1.0)
    t.add_cluster("C3", (10.0, 10.0), spread=1.0)
    t.merge(["C1", "C2"], "C12", (0.0, 0.0), at=n_samples // 2)
    gt = t.build()
    return _stream(gt), gt


def recurring(n_samples: int = 24_000, seed: int = 42):
    """S8 — a concept returns: (0,0) -> (10,10) -> (0,0) (§5.6)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S8_recurring")
    t.add_cluster("C1", (0.0, 0.0), spread=1.0)
    t.add_cluster("C2", (20.0, 20.0), spread=1.0)
    t.move("C1", (10.0, 10.0), start=n_samples // 3)
    t.recur("C1", (0.0, 0.0), start=2 * n_samples // 3)
    gt = t.build()
    return _stream(gt), gt


def multiple_drift(n_samples: int = 40_000, seed: int = 42):
    """S9 — several drift events across the stream (§6 timeline)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S9_multiple_drift")
    t.add_cluster("C1", (0.0, 0.0), spread=1.0)
    t.add_cluster("C2", (10.0, 10.0), spread=1.0)
    q = n_samples // 5
    t.move("C1", (5.0, 5.0), start=q)  # movement
    t.birth("C3", (0.0, 10.0), at=2 * q)  # birth
    t.death("C2", at=3 * q)  # death
    t.merge(["C1", "C3"], "C13", (2.5, 7.5), at=4 * q)  # merge
    gt = t.build()
    return _stream(gt), gt


def noise_and_drift(n_samples: int = 20_000, seed: int = 42):
    """S10 — abrupt movement with background noise (robustness)."""
    t = GroundTruthTracker(n_samples, 2, seed, name="S10_noise_drift")
    t.add_cluster("C1", (0.0, 0.0), spread=1.0)
    t.add_cluster("C2", (10.0, 10.0), spread=1.0)
    t.move("C1", (8.0, 0.0), start=n_samples // 2)
    gt = t.build()
    return _stream(gt, noise=0.10), gt


#: name -> factory, used by experiments and the scenario matrix (§16)
SCENARIOS: dict[str, Callable] = {
    "S1_stationary": stationary,
    "S2_abrupt_movement": abrupt_movement,
    "S3_gradual_movement": gradual_movement,
    "S4_birth": birth,
    "S5_death": death,
    "S6_split": split,
    "S7_merge": merge,
    "S8_recurring": recurring,
    "S9_multiple_drift": multiple_drift,
    "S10_noise_drift": noise_and_drift,
}
