"""Drift-adaptation metrics (§12) — the central group of the benchmark.

All metrics are derived from a quality-over-time curve (a list of window
records) plus the ground-truth drift events. They quantify *how the model
behaves around a drift*: how far quality drops, how long it takes to recover,
how quickly it starts adapting, and the total quality lost during adaptation.

Because streaming clusterers need not expose a drift detector (§9), everything
here is measured from quality dynamics, not from ``detect_drift()`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ground_truth.events import DriftEvent

# np.trapz was removed in NumPy 2.x in favour of np.trapezoid.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


@dataclass
class AdaptationMetrics:
    """Per-drift adaptation summary (§12.1-§12.5)."""

    drift_id: int
    drift_type: str
    drift_start: int
    drift_end: int

    quality_before: float | None  # mean quality in the pre-drift window
    quality_min: float | None  # worst quality after drift start
    performance_drop: float | None  # before - min (§12.1)
    recovery_time: int | None  # samples to reach recovery threshold (§12.2)
    adaptation_delay: int | None  # samples until quality first improves (§12.3)
    post_drift_quality: float | None  # stabilised quality after drift (§12.4)
    area_under_recovery: float | None  # quality lost during adaptation (§12.5)
    recovered: bool

    def to_dict(self) -> dict:
        return {
            "drift_id": self.drift_id,
            "drift_type": self.drift_type,
            "drift_start": self.drift_start,
            "drift_end": self.drift_end,
            "quality_before": _r(self.quality_before),
            "quality_min": _r(self.quality_min),
            "performance_drop": _r(self.performance_drop),
            "recovery_time": self.recovery_time,
            "adaptation_delay": self.adaptation_delay,
            "post_drift_quality": _r(self.post_drift_quality),
            "area_under_recovery": _r(self.area_under_recovery),
            "recovered": self.recovered,
        }


def _r(v, ndigits: int = 4):
    return round(v, ndigits) if v is not None else None


def _series(records, metric: str):
    """Extract (time, value) points for ``metric`` from window records, sorted."""
    pts = []
    for r in records:
        v = r.metrics.get(metric)
        if v is not None:
            pts.append((r.center, float(v)))
    pts.sort(key=lambda p: p[0])
    times = np.array([p[0] for p in pts])
    vals = np.array([p[1] for p in pts])
    return times, vals


def compute_adaptation(
    records,
    event: DriftEvent,
    metric: str = "ari",
    recovery_fraction: float = 0.95,
    next_drift_start: int | None = None,
) -> AdaptationMetrics:
    """Compute adaptation metrics for a single drift ``event``.

    Parameters
    ----------
    records:
        Window records (each with ``.center`` sample index and ``.metrics``).
    event:
        The ground-truth drift event to analyse.
    metric:
        Which quality metric to track (default ``"ari"``).
    recovery_fraction:
        Fraction of the pre-drift quality that counts as "recovered" (§12.2).
    next_drift_start:
        Start of the following drift, so this event's window doesn't bleed into
        the next one. ``None`` => end of stream.
    """
    times, vals = _series(records, metric)
    if len(times) == 0:
        return _empty(event)

    horizon = next_drift_start if next_drift_start is not None else int(times.max()) + 1

    before_mask = times < event.start
    after_mask = (times >= event.start) & (times < horizon)

    quality_before = float(vals[before_mask].mean()) if before_mask.any() else None
    after_times = times[after_mask]
    after_vals = vals[after_mask]

    if after_vals.size == 0:
        return _empty(event, quality_before=quality_before)

    quality_min = float(after_vals.min())
    performance_drop = (
        quality_before - quality_min if quality_before is not None else None
    )

    # post-drift quality: mean of the last third of the post-drift window (§12.4)
    tail = after_vals[max(len(after_vals) * 2 // 3, len(after_vals) - 1):]
    post_drift_quality = float(tail.mean()) if tail.size else float(after_vals[-1])

    # recovery time (§12.2): first time quality reaches recovery threshold
    recovery_time = None
    recovered = False
    if quality_before is not None:
        threshold = recovery_fraction * quality_before
        drop_idx = int(np.argmin(after_vals))
        for i in range(drop_idx, len(after_vals)):
            if after_vals[i] >= threshold:
                recovery_time = int(after_times[i] - event.start)
                recovered = True
                break

    # adaptation delay (§12.3): first improvement after the minimum
    adaptation_delay = None
    drop_idx = int(np.argmin(after_vals))
    for i in range(drop_idx + 1, len(after_vals)):
        if after_vals[i] > after_vals[drop_idx]:
            adaptation_delay = int(after_times[i] - event.start)
            break

    # area under recovery curve (§12.5): quality lost vs pre-drift baseline,
    # integrated over the post-drift window (higher => more/longer degradation)
    area_under_recovery = None
    if quality_before is not None and after_times.size >= 2:
        deficit = np.clip(quality_before - after_vals, 0.0, None)
        area_under_recovery = float(_trapezoid(deficit, after_times))

    return AdaptationMetrics(
        drift_id=event.drift_id,
        drift_type=str(event.type),
        drift_start=event.start,
        drift_end=event.end,
        quality_before=quality_before,
        quality_min=quality_min,
        performance_drop=performance_drop,
        recovery_time=recovery_time,
        adaptation_delay=adaptation_delay,
        post_drift_quality=post_drift_quality,
        area_under_recovery=area_under_recovery,
        recovered=recovered,
    )


def _empty(event: DriftEvent, quality_before=None) -> AdaptationMetrics:
    return AdaptationMetrics(
        drift_id=event.drift_id,
        drift_type=str(event.type),
        drift_start=event.start,
        drift_end=event.end,
        quality_before=quality_before,
        quality_min=None,
        performance_drop=None,
        recovery_time=None,
        adaptation_delay=None,
        post_drift_quality=None,
        area_under_recovery=None,
        recovered=False,
    )
