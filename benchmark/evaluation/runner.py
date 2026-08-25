"""Online benchmark runner.

Streams a dataset through one :class:`ClustererAdapter` following the streaming
protocol (scale -> ``predict_one`` -> ``learn_one``) and records:

* **efficiency** — cumulative runtime, throughput, model memory and cluster
  counts sampled at checkpoints as the stream grows (so scalability is visible,
  not just a single final number);
* **quality** — external metrics (ARI / NMI / AMI / purity) when the dataset is
  labeled, and internal metrics (silhouette / Davies-Bouldin / Calinski-
  Harabasz) when requested. Evaluated **per time window** so we can report the
  mean / median over the stream (more informative than the final score alone),
  plus the final-window and whole-stream values.

Labels are used exclusively for scoring; the model only ever sees the (scaled)
feature dict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..algorithms.adapter import ClustererAdapter
from ..datasets.base import Dataset
from . import clustering_quality as cq
from .efficiency import EfficiencyMetrics


@dataclass
class Checkpoint:
    """A single (n_seen, time, memory, clusters) measurement."""

    n_seen: int
    cumulative_runtime_s: float
    memory_bytes: int | None
    n_micro_clusters: int | None
    n_macro_clusters: int | None

    def to_dict(self) -> dict:
        return {
            "n_seen": self.n_seen,
            "runtime_s": round(self.cumulative_runtime_s, 5),
            "memory_bytes": self.memory_bytes,
            "n_micro_clusters": self.n_micro_clusters,
            "n_macro_clusters": self.n_macro_clusters,
        }


@dataclass
class WindowQuality:
    """Quality metrics for one evaluation window (time-aware evaluation)."""

    start: int
    end: int
    metrics: dict = field(default_factory=dict)

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    def to_dict(self) -> dict:
        d = {"start": self.start, "end": self.end}
        d.update({k: (round(v, 4) if v is not None else None) for k, v in self.metrics.items()})
        return d


@dataclass
class RunResult:
    """Everything produced by one (algorithm+setting, dataset) run."""

    algorithm: str
    dataset: str
    scaler: str
    params: dict
    n_samples: int
    n_features: int | None
    efficiency: EfficiencyMetrics
    # whole-stream quality: metric -> value (computed over all labeled samples)
    quality: dict = field(default_factory=dict)
    # time-aware summary: metric -> {"mean","median","final"} over windows
    quality_stream: dict = field(default_factory=dict)
    windows: list[WindowQuality] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def quality_curve(self, metric: str):
        xs, ys = [], []
        for w in self.windows:
            v = w.metrics.get(metric)
            if v is not None:
                xs.append(w.center)
                ys.append(v)
        return xs, ys

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "scaler": self.scaler,
            "parameters": self.params,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "efficiency": self.efficiency.to_dict(),
            "quality": {k: (round(v, 4) if v is not None else None) for k, v in self.quality.items()},
            "quality_stream": {
                m: {k: (round(v, 4) if v is not None else None) for k, v in stats.items()}
                for m, stats in self.quality_stream.items()
            },
            "windows": [w.to_dict() for w in self.windows],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
        }


def _summarize(values) -> dict:
    """mean / median / final over a list of per-window metric values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "median": None, "final": None}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "final": float(vals[-1]),
    }


def run_once(
    adapter: ClustererAdapter,
    dataset: Dataset,
    n_samples: int | None = None,
    n_checkpoints: int = 20,
    window_size: int = 1000,
    external_metrics: tuple[str, ...] = ("ari", "nmi", "ami", "purity"),
    internal_metrics: tuple[str, ...] = (),
) -> RunResult:
    """Run one algorithm/setting over one dataset and measure it.

    Parameters
    ----------
    adapter:
        The clusterer (+ optional scaler) to benchmark.
    dataset:
        A re-iterable :class:`Dataset` yielding ``(x, y)``.
    n_samples:
        Cap the number of streamed samples (``None`` => whole dataset).
    n_checkpoints:
        Number of evenly-spaced *efficiency* measurements across the stream.
    window_size:
        Size of each *quality* evaluation window (time-aware evaluation).
    external_metrics:
        Metrics needing labels; used only when the dataset is labeled.
    internal_metrics:
        Label-free metrics (silhouette / davies_bouldin / calinski_harabasz).
        Computed per window from the feature points; empty => skip (they are the
        expensive ones, so opt in explicitly).
    """
    ext = {m: cq.EXTERNAL_FNS[m] for m in external_metrics if m in cq.EXTERNAL_FNS}
    inter = {m: cq.INTERNAL_FNS[m] for m in internal_metrics if m in cq.INTERNAL_FNS}
    want_internal = bool(inter)

    if n_samples is not None:
        length = n_samples
    else:
        try:
            length = len(dataset)
        except TypeError:  # __len__ returned None (unknown length)
            length = None
    checkpoint_every = max(length // n_checkpoints, 1) if length else None

    # whole-stream accumulators (only kept when labeled, to bound memory)
    y_true_all: list = []
    y_pred_all: list = []
    mc_samples: list[int] = []

    # per-window buffers
    windows: list[WindowQuality] = []
    win_start = 0
    win_yt: list = []
    win_yp: list = []
    win_x: list = []

    checkpoints: list[Checkpoint] = []
    total_runtime = 0.0
    peak_mem = 0
    max_mc = 0
    n_seen = 0

    def eff_snapshot():
        nonlocal max_mc
        mc = adapter.n_micro_clusters()
        mac = adapter.n_macro_clusters()
        mem = adapter.memory_bytes()
        if mc is not None:
            mc_samples.append(mc)
            max_mc = max(max_mc, mc)
        checkpoints.append(Checkpoint(n_seen, total_runtime, mem, mc, mac))

    def flush_window(start: int, end: int):
        rec = WindowQuality(start=start, end=end)
        if dataset.has_labels and win_yt:
            for name, fn in ext.items():
                rec.metrics[name] = fn(win_yt, win_yp)
        if want_internal and win_x:
            for name, fn in inter.items():
                rec.metrics[name] = fn(win_x, win_yp)
        if rec.metrics:
            windows.append(rec)

    for x, y in dataset:
        if n_samples is not None and n_seen >= n_samples:
            break

        # quality-window boundary
        if n_seen > 0 and n_seen % window_size == 0:
            flush_window(win_start, n_seen - 1)
            win_start = n_seen
            win_yt, win_yp, win_x = [], [], []

        t0 = time.perf_counter()
        y_pred = adapter.process_one(x)  # scale -> predict -> learn
        total_runtime += time.perf_counter() - t0
        n_seen += 1

        if dataset.has_labels:
            y_true_all.append(y)
            y_pred_all.append(y_pred)
            win_yt.append(y)
            win_yp.append(y_pred)
        elif want_internal:
            win_yp.append(y_pred)
        if want_internal:
            win_x.append(list(x.values()))

        mem = adapter.memory_bytes()
        if mem:
            peak_mem = max(peak_mem, mem)

        if checkpoint_every and n_seen % checkpoint_every == 0:
            eff_snapshot()

    # final partial window + final efficiency checkpoint
    if win_yp or win_yt:
        flush_window(win_start, max(n_seen - 1, win_start))
    if not checkpoints or checkpoints[-1].n_seen != n_seen:
        eff_snapshot()

    # whole-stream external quality (single value over all labeled samples)
    quality: dict = {}
    if dataset.has_labels and y_true_all:
        for name, fn in ext.items():
            quality[name] = fn(y_true_all, y_pred_all)

    # time-aware summary: mean / median / final per metric over windows
    quality_stream: dict = {}
    metric_names = set()
    for w in windows:
        metric_names.update(w.metrics)
    for m in metric_names:
        quality_stream[m] = _summarize([w.metrics.get(m) for w in windows])

    efficiency = EfficiencyMetrics(
        total_runtime_s=total_runtime,
        n_samples=n_seen,
        peak_memory_bytes=peak_mem or None,
        mean_micro_clusters=(float(np.mean(mc_samples)) if mc_samples else None),
        max_micro_clusters=(max_mc or None),
        final_macro_clusters=adapter.n_macro_clusters(),
    )

    return RunResult(
        algorithm=adapter.name,
        dataset=dataset.name,
        scaler=adapter.scaler_name,
        params=adapter.params,
        n_samples=n_seen,
        n_features=getattr(dataset, "n_features", None),
        efficiency=efficiency,
        quality=quality,
        quality_stream=quality_stream,
        windows=windows,
        checkpoints=checkpoints,
    )
