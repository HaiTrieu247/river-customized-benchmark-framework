"""Materialise a data stream from a :class:`StreamGroundTruth` (§6).

The generator samples one point at a time from the active ground-truth clusters
(spherical Gaussian blobs), so it can be consumed exactly like a live stream:
``for x, y_true, t in stream``. During a *gradual* drift interval the affected
cluster's center is linearly interpolated between its pre- and post-drift
position (§5.7), instead of snapping at a single timestamp.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from ...ground_truth.events import DriftType
from ...ground_truth.schema import ClusterState, StreamGroundTruth


@dataclass
class StreamConfig:
    """High-level knobs for a synthetic stream (§6)."""

    n_samples: int = 100_000
    n_features: int = 2
    noise: float = 0.0  # fraction of samples drawn as uniform background noise
    noise_scale: float = 20.0  # half-range of the uniform noise box per feature
    seed: int = 42


class SyntheticStream:
    """Iterable stream backed by ground truth.

    Parameters
    ----------
    ground_truth:
        The :class:`StreamGroundTruth` describing cluster evolution + drifts.
    config:
        Sampling configuration (noise, seed). ``n_samples``/``n_features`` are
        taken from ``ground_truth`` to stay consistent.
    """

    def __init__(self, ground_truth: StreamGroundTruth, config: StreamConfig | None = None):
        self.gt = ground_truth
        self.config = config or StreamConfig(
            n_samples=ground_truth.n_samples,
            n_features=ground_truth.n_features,
            seed=ground_truth.seed,
        )
        # keep the two in sync
        self.config.n_samples = ground_truth.n_samples
        self.config.n_features = ground_truth.n_features
        self._rng = np.random.default_rng(self.config.seed)
        self._feature_names = [f"x{i}" for i in range(self.config.n_features)]
        # precompute gradual-movement interpolation plans
        self._gradual = self._build_gradual_plans()

    # -- gradual interpolation ----------------------------------------------
    def _state_before(self, cluster_id: str, t: int) -> ClusterState | None:
        line = self.gt.timelines.get(cluster_id, [])
        starts = [s.t_start for s in line]
        idx = bisect.bisect_left(starts, t) - 1
        return line[idx] if idx >= 0 else None

    def _state_at_start(self, cluster_id: str, t: int) -> ClusterState | None:
        line = self.gt.timelines.get(cluster_id, [])
        for s in line:
            if s.t_start == t:
                return s
        return None

    def _build_gradual_plans(self) -> list[dict]:
        plans = []
        for ev in self.gt.drift_events:
            if not ev.is_gradual:
                continue
            if ev.type not in (DriftType.MOVEMENT, DriftType.RECURRING):
                continue
            for cid in ev.affected_clusters:
                src = self._state_before(cid, ev.start)
                dst = self._state_at_start(cid, ev.end)
                if src is None or dst is None:
                    continue
                plans.append(
                    {
                        "cluster_id": cid,
                        "start": ev.start,
                        "end": ev.end,
                        "from": src.center_array(),
                        "to": dst.center_array(),
                        "spread": src.spread,
                        "weight": src.weight,
                    }
                )
        return plans

    def _effective_states(self, t: int) -> list[ClusterState]:
        """Active states at ``t`` with gradual-drift centers interpolated."""
        states = {s.cluster_id: s for s in self.gt.active_states_at(t)}
        for plan in self._gradual:
            if plan["start"] <= t < plan["end"]:
                frac = (t - plan["start"]) / max(plan["end"] - plan["start"], 1)
                center = plan["from"] + frac * (plan["to"] - plan["from"])
                states[plan["cluster_id"]] = ClusterState(
                    plan["cluster_id"],
                    tuple(center.tolist()),
                    plan["spread"],
                    plan["weight"],
                    True,
                    t,
                )
        return list(states.values())

    # -- sampling ------------------------------------------------------------
    def _sample_point(self, t: int) -> tuple[dict, str | None]:
        states = self._effective_states(t)
        if not states:
            # no active cluster: emit background noise
            return self._noise_point(), None

        if self.config.noise > 0 and self._rng.random() < self.config.noise:
            return self._noise_point(), None

        weights = np.array([s.weight for s in states], dtype=float)
        weights = weights / weights.sum()
        idx = int(self._rng.choice(len(states), p=weights))
        state = states[idx]
        center = state.center_array()
        point = center + self._rng.normal(0.0, state.spread, size=self.config.n_features)
        x = {name: float(v) for name, v in zip(self._feature_names, point)}
        return x, state.cluster_id

    def _noise_point(self) -> dict:
        lo, hi = -self.config.noise_scale, self.config.noise_scale
        vals = self._rng.uniform(lo, hi, size=self.config.n_features)
        return {name: float(v) for name, v in zip(self._feature_names, vals)}

    # -- iteration protocol --------------------------------------------------
    def __iter__(self):
        for t in range(self.config.n_samples):
            x, y = self._sample_point(t)
            yield x, y, t

    def __len__(self) -> int:
        return self.config.n_samples

    def take(self, n: int):
        """Yield the first ``n`` samples (handy for smoke tests)."""
        for i, item in enumerate(self):
            if i >= n:
                break
            yield item
