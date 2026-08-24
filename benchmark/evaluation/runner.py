"""Online benchmark runner (§14, §15).

Streams a synthetic stream through one :class:`ClustererAdapter`, following the
streaming protocol (``predict_one`` *then* ``learn_one``), and records quality +
efficiency per time window. The window records feed the adaptation and reporting
layers. Ground truth is used only here, for scoring — never passed to the model
(Principle 1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..algorithms.adapter import ClustererAdapter
from ..datasets.synthetic.generator import SyntheticStream
from ..ground_truth.schema import StreamGroundTruth
from . import clustering_quality as cq
from .adaptation import compute_adaptation
from .efficiency import EfficiencyMetrics


@dataclass
class WindowRecord:
    """Metrics for one evaluation window (§15)."""

    start: int
    end: int
    metrics: dict = field(default_factory=dict)
    n_micro_clusters: int | None = None
    runtime_s: float = 0.0
    memory_bytes: int | None = None

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    def to_dict(self) -> dict:
        d = {
            "start": self.start,
            "end": self.end,
            "n_micro_clusters": self.n_micro_clusters,
            "runtime": round(self.runtime_s, 5),
            "memory": self.memory_bytes,
        }
        d.update({k: (round(v, 4) if v is not None else None) for k, v in self.metrics.items()})
        return d


@dataclass
class ExperimentResult:
    """Everything produced by one (algorithm, stream, seed) run (§22)."""

    algorithm: str
    scenario: str
    seed: int
    params: dict
    windows: list[WindowRecord]
    efficiency: EfficiencyMetrics
    adaptation: list  # list[AdaptationMetrics]
    ground_truth: StreamGroundTruth

    def quality_curve(self, metric: str = "ari"):
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
            "scenario": self.scenario,
            "seed": self.seed,
            "parameters": self.params,
            "stream": {
                "n_samples": self.ground_truth.n_samples,
                "n_features": self.ground_truth.n_features,
            },
            "drifts": [e.to_dict() for e in self.ground_truth.drift_events],
            "efficiency": self.efficiency.to_dict(),
            "adaptation": [a.to_dict() for a in self.adaptation],
            "windows": [w.to_dict() for w in self.windows],
        }


def run_experiment(
    adapter: ClustererAdapter,
    stream: SyntheticStream,
    ground_truth: StreamGroundTruth,
    window_size: int = 1000,
    metrics: tuple[str, ...] = ("ari", "nmi", "purity"),
    recovery_fraction: float = 0.95,
    scenario_name: str | None = None,
    keep_points_for_silhouette: bool = False,
) -> ExperimentResult:
    """Run one algorithm over one stream and score it (§14, §15, §12)."""
    metric_fns = {
        "ari": cq.ari,
        "nmi": cq.nmi,
        "purity": cq.purity,
    }
    selected = {m: metric_fns[m] for m in metrics if m in metric_fns}

    windows: list[WindowRecord] = []
    y_true_buf: list = []
    y_pred_buf: list = []
    x_buf: list = []
    mc_samples: list[int] = []

    win_start = 0
    win_runtime = 0.0
    total_runtime = 0.0
    peak_mem = 0
    max_mc = 0

    def flush(w_start: int, w_end: int, runtime_s: float):
        nonlocal max_mc
        rec = WindowRecord(start=w_start, end=w_end, runtime_s=runtime_s)
        for name, fn in selected.items():
            rec.metrics[name] = fn(y_true_buf, y_pred_buf)
        if keep_points_for_silhouette and "silhouette" in metrics:
            rec.metrics["silhouette"] = cq.silhouette(x_buf, y_pred_buf)
        mc = adapter.n_micro_clusters()
        rec.n_micro_clusters = mc
        if mc is not None:
            mc_samples.append(mc)
            max_mc = max(max_mc, mc)
        rec.memory_bytes = adapter.memory_bytes()
        windows.append(rec)

    for x, y, t in stream:
        # window boundary
        if t > 0 and t % window_size == 0:
            flush(win_start, t - 1, win_runtime)
            win_start = t
            win_runtime = 0.0
            y_true_buf = []
            y_pred_buf = []
            x_buf = []

        t0 = time.perf_counter()
        y_pred = adapter.predict_one(x)  # test-then-train (§core concepts)
        adapter.learn_one(x)
        dt = time.perf_counter() - t0

        win_runtime += dt
        total_runtime += dt
        y_true_buf.append(y)
        y_pred_buf.append(y_pred)
        if keep_points_for_silhouette:
            x_buf.append(list(x.values()))
        mem = adapter.memory_bytes()
        if mem:
            peak_mem = max(peak_mem, mem)

    # final partial window
    if y_true_buf:
        flush(win_start, win_start + len(y_true_buf) - 1, win_runtime)

    efficiency = EfficiencyMetrics(
        total_runtime_s=total_runtime,
        n_samples=ground_truth.n_samples,
        peak_memory_bytes=peak_mem or None,
        mean_micro_clusters=(float(np.mean(mc_samples)) if mc_samples else None),
        max_micro_clusters=(max_mc or None),
    )

    # adaptation per drift, using the primary metric (first requested)
    primary = metrics[0] if metrics else "ari"
    events = ground_truth.drift_events
    adaptation = []
    for i, ev in enumerate(events):
        nxt = events[i + 1].start if i + 1 < len(events) else None
        adaptation.append(
            compute_adaptation(
                windows, ev, metric=primary,
                recovery_fraction=recovery_fraction, next_drift_start=nxt,
            )
        )

    return ExperimentResult(
        algorithm=adapter.name,
        scenario=scenario_name or ground_truth.name,
        seed=ground_truth.seed,
        params=adapter.params,
        windows=windows,
        efficiency=efficiency,
        adaptation=adaptation,
        ground_truth=ground_truth,
    )
