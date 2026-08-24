"""Plots (§23): quality-over-time, drift timeline, recovery-curve zoom.

Uses a non-interactive matplotlib backend so it works headless (CI / scripts).
Each function returns the matplotlib ``Figure`` so callers can save or show it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _mark_drifts(ax, ground_truth):
    for ev in ground_truth.drift_events:
        if ev.is_gradual:
            ax.axvspan(ev.start, ev.end, color="red", alpha=0.10)
        ax.axvline(ev.start, color="red", ls="--", alpha=0.6, lw=1)
        ax.annotate(
            str(ev.type),
            xy=(ev.start, ax.get_ylim()[1]),
            xytext=(0, -10),
            textcoords="offset points",
            fontsize=7,
            color="red",
            rotation=90,
            va="top",
        )


def plot_quality_over_time(results, metric: str = "ari", title: str | None = None, ax=None):
    """Quality vs time for one or more algorithms on the same scenario (§23)."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    gt = None
    for r in results:
        xs, ys = r.quality_curve(metric)
        ax.plot(xs, ys, marker="o", ms=3, lw=1.5, label=r.algorithm)
        gt = r.ground_truth
    if gt is not None:
        _mark_drifts(ax, gt)

    ax.set_xlabel("sample")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title or f"{metric.upper()} over time")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_drift_timeline(ground_truth, ax=None):
    """Timeline marking each drift event across the stream (§23)."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 2))
    else:
        fig = ax.figure

    ax.hlines(0, 0, ground_truth.n_samples, color="black", lw=1)
    for ev in ground_truth.drift_events:
        ax.plot(ev.start, 0, "rv", ms=9)
        if ev.is_gradual:
            ax.hlines(0, ev.start, ev.end, color="red", lw=4, alpha=0.4)
        ax.annotate(
            f"#{ev.drift_id} {ev.type}",
            xy=(ev.start, 0),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=8,
            ha="center",
            color="red",
        )
    ax.set_yticks([])
    ax.set_xlim(0, ground_truth.n_samples)
    ax.set_xlabel("sample")
    ax.set_title(f"Drift timeline — {ground_truth.name}")
    return fig


def plot_recovery_curve(results, drift_index: int = 0, metric: str = "ari",
                        pad: int = 3000, ax=None):
    """Zoom on the region around one drift event to inspect recovery (§23)."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    gt = results[0].ground_truth
    if drift_index >= len(gt.drift_events):
        ax.set_title("no such drift")
        return fig
    ev = gt.drift_events[drift_index]
    lo, hi = ev.start - pad, ev.end + pad

    for r in results:
        xs, ys = r.quality_curve(metric)
        pts = [(x, y) for x, y in zip(xs, ys) if lo <= x <= hi]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, label=r.algorithm)
    ax.axvline(ev.start, color="red", ls="--", alpha=0.7)
    if ev.is_gradual:
        ax.axvspan(ev.start, ev.end, color="red", alpha=0.10)
    ax.set_xlabel("sample")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Recovery around drift #{ev.drift_id} ({ev.type})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return fig
