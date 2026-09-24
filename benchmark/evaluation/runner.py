"""Online benchmark runner.

Streams a dataset through one :class:`ClustererAdapter` following the streaming
protocol (scale -> ``predict_one`` -> ``learn_one``) and records:

* **efficiency** — cumulative runtime, throughput, model memory and cluster
  counts sampled at checkpoints as the stream grows (so scalability is visible,
  not just a single final number);
* **quality** — external metrics (ARI / NMI / AMI / purity) when the dataset is
  labeled, and internal metrics (river's streaming silhouette) when requested.

Quality is tracked two ways at once, because they answer different questions:

* **cumulative** — the metric from the start of the stream up to each window
  boundary. This is the curve that converges; its last value is the whole-stream
  score.
* **per window** — the metric computed *only* over the samples inside each
  window, with the metric object reset at every boundary. This is the
  time-aware view the evaluation protocol asks for, and it is what
  ``quality_stream`` (mean / median / final) summarises. Averaging the
  cumulative curve instead would just re-weight the early stream rather than
  measure behaviour over time.

Labels are used exclusively for scoring; the model only ever sees the (scaled)
feature dict.
"""

from __future__ import annotations

from collections.abc import Callable

import collections
import time
from dataclasses import dataclass, field

import numpy as np

from ..algorithms.adapter import ClustererAdapter
from ..datasets.base import Dataset
from . import clustering_quality as cq
from .efficiency import EfficiencyMetrics

try:  # optional: process-level RSS alongside river's model-only accounting
    import psutil

    _PROCESS = psutil.Process()
except Exception:  # pragma: no cover - psutil not installed
    _PROCESS = None


def _process_rss() -> int | None:
    if _PROCESS is None:
        return None
    try:
        return int(_PROCESS.memory_info().rss)
    except Exception:  # pragma: no cover
        return None


@dataclass
class Checkpoint:
    """A single (n_seen, time, memory, clusters) measurement."""

    n_seen: int
    cumulative_runtime_s: float
    memory_bytes: int | None
    n_micro_clusters: int | None
    n_macro_clusters: int | None
    process_rss_bytes: int | None = None
    #: Structure summary for clusterers that expose one (ODAC: cluster count,
    #: active clusters, tree height). ``None`` for the rest.
    structure: dict | None = None

    def to_dict(self) -> dict:
        return {
            "n_seen": self.n_seen,
            "runtime_s": round(self.cumulative_runtime_s, 5),
            "memory_bytes": self.memory_bytes,
            "process_rss_bytes": self.process_rss_bytes,
            "n_micro_clusters": self.n_micro_clusters,
            "n_macro_clusters": self.n_macro_clusters,
            "structure": self.structure,
        }


@dataclass
class WindowQuality:
    """Quality for one evaluation window.

    ``metrics`` holds the *cumulative* values (stream start -> window end);
    ``window_metrics`` holds the values computed strictly inside the window.
    """

    start: int
    end: int
    metrics: dict = field(default_factory=dict)
    window_metrics: dict = field(default_factory=dict)
    #: Distinct ground-truth classes inside this window.
    n_true_classes: int | None = None
    #: Share of the window held by its most common class, in [0, 1].
    majority_class_share: float | None = None

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    @property
    def is_degenerate(self) -> bool:
        """True when the window holds a single class.

        A per-window external metric degenerates on such a window: with one
        true class, ARI collapses to a "did you lump the whole window into one
        cluster?" test — 1.0 for lumping, 0.0 for any split at all, whatever
        the clustering is actually worth. Streams that arrive block-ordered by
        class (KDDCup99 is one) produce a lot of these, and a model that
        under-segments then scores well for the wrong reason.
        """
        return self.n_true_classes is not None and self.n_true_classes <= 1

    def to_dict(self) -> dict:
        d = {"start": self.start, "end": self.end}
        d.update({k: (round(v, 4) if v is not None else None) for k, v in self.metrics.items()})
        d["window_metrics"] = {
            k: (round(v, 4) if v is not None else None) for k, v in self.window_metrics.items()
        }
        d["n_true_classes"] = self.n_true_classes
        d["majority_class_share"] = (
            round(self.majority_class_share, 4) if self.majority_class_share is not None else None
        )
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
    scaler_params: dict = field(default_factory=dict)
    # whole-stream quality: metric -> value (computed over all scored samples)
    quality: dict = field(default_factory=dict)
    # time-aware summary over PER-WINDOW values: metric -> {"mean","median","final"}
    quality_stream: dict = field(default_factory=dict)
    # the same summary over the CUMULATIVE curve, kept for reference
    quality_cumulative_stream: dict = field(default_factory=dict)
    windows: list[WindowQuality] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    #: predictions served by the cold-start fallback (model not yet initialised)
    n_fallback_predictions: int = 0
    #: predictions that were a sentinel rather than a cluster id -- the model
    #: held no clusters at all (RIVER_BUGS.md BUG-6). A run with this near
    #: `n_samples` produced no clustering, and its speed is the speed of that.
    n_sentinel_predictions: int = 0
    #: How trustworthy the per-window scores are on this stream: share of
    #: windows holding a single class, and the median majority-class share.
    #: See :attr:`WindowQuality.is_degenerate`.
    window_label_diagnostics: dict = field(default_factory=dict)
    #: Whether the algorithm assigns each sample to a cluster. ``False`` for
    #: ODAC, which clusters variables; external quality is then undefined and
    #: `quality` is empty by design rather than by failure.
    partitions_samples: bool = True
    #: Final structure summary, for clusterers that expose one.
    structure: dict | None = None
    #: evaluation window size actually used (part of the protocol, so recorded)
    window_size: int | None = None
    seed: int | None = None
    #: First stream index that was scored. The model still learns from sample 0;
    #: only the metrics start here. Recorded so a row describes itself.
    score_from: int = 0
    #: Size distribution of the *predicted* clusters: how many were produced,
    #: what share of them held under 0.1% of the scored samples
    #: (``dust_cluster_share``), and what share of samples fell in the biggest
    #: one (``largest_cluster_share``). The two catch opposite degeneracies --
    #: shattering into dust, and lumping everything together -- that a single
    #: ARI value hides.
    cluster_size_diagnostics: dict = field(default_factory=dict)

    def quality_curve(self, metric: str):
        """Cumulative metric vs stream position (the converging curve)."""
        xs, ys = [], []
        for w in self.windows:
            v = w.metrics.get(metric)
            if v is not None:
                xs.append(w.center)
                ys.append(v)
        return xs, ys

    def window_quality_curve(self, metric: str):
        """Per-window metric vs stream position (the time-aware curve)."""
        xs, ys = [], []
        for w in self.windows:
            v = w.window_metrics.get(metric)
            if v is not None:
                xs.append(w.center)
                ys.append(v)
        return xs, ys

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "scaler": self.scaler,
            "scaler_parameters": self.scaler_params,
            "parameters": self.params,
            "seed": self.seed,
            "window_size": self.window_size,
            "score_from": self.score_from,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_fallback_predictions": self.n_fallback_predictions,
            "n_sentinel_predictions": self.n_sentinel_predictions,
            "partitions_samples": self.partitions_samples,
            "structure": self.structure,
            "window_label_diagnostics": self.window_label_diagnostics,
            "cluster_size_diagnostics": self.cluster_size_diagnostics,
            "efficiency": self.efficiency.to_dict(),
            "quality": {
                k: (round(v, 4) if v is not None else None) for k, v in self.quality.items()
            },
            "quality_stream": {
                m: {k: (round(v, 4) if v is not None else None) for k, v in stats.items()}
                for m, stats in self.quality_stream.items()
            },
            "quality_cumulative_stream": {
                m: {k: (round(v, 4) if v is not None else None) for k, v in stats.items()}
                for m, stats in self.quality_cumulative_stream.items()
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


def _summarize_all(windows, attr: str) -> dict:
    names: set[str] = set()
    for w in windows:
        names.update(getattr(w, attr))
    return {m: _summarize([getattr(w, attr).get(m) for w in windows]) for m in names}


#: How often :func:`run_once` calls its ``progress`` callback, in samples.
PROGRESS_EVERY = 512


def run_once(
    adapter: ClustererAdapter,
    dataset: Dataset,
    n_samples: int | None = None,
    n_checkpoints: int = 20,
    window_size: int = 1000,
    external_metrics: tuple[str, ...] = ("ari", "nmi", "ami", "purity"),
    internal_metrics: tuple[str, ...] = (),
    seed: int | None = None,
    score_from: int = 0,
    progress: Callable[[int, int | None, int | None], None] | None = None,
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
        Memory is probed only here: the probe walks the model's whole object
        graph, so it costs orders of magnitude more than processing a sample.
        Peak memory is therefore a max over checkpoints — raise this for a
        tighter bound on models that prune, such as DenStream and DBSTREAM.
    window_size:
        Size of each *quality* evaluation window (time-aware evaluation).
    external_metrics:
        Metrics needing labels; used only when the dataset is labeled.
    internal_metrics:
        Label-free metrics (river's streaming ``silhouette``). They need the
        model's centers, so clusterers that expose none report no value.
    seed:
        Recorded on the result for reproducibility; the adapter is already built.
    score_from:
        Stream index at which *scoring* starts. The model still learns from
        sample 0 -- it is an online model and there is no reason to throw the
        head of the stream away -- but no metric is updated before this index,
        and the evaluation windows are measured from it. This separates "what
        the model has seen" from "what it is judged on", which is what tuning on
        a prefix and scoring on the remainder requires. The stream order itself
        is untouched.
    progress:
        Optional ``(n_seen, length, n_micro_clusters)`` callback, invoked every
        :data:`PROGRESS_EVERY` samples from outside the timed region, so a
        caller can draw a progress bar without inflating the measured runtime.
        ``length`` is ``None`` when the stream length is unknown, and
        ``n_micro_clusters`` is ``None`` for a model that exposes none.
    """
    labeled = dataset.has_labels
    # An algorithm that does not partition samples has no per-sample label to
    # compare against ground truth, so external and internal metrics alike are
    # undefined for it -- ODAC clusters variables, not observations.
    partitions = adapter.partitions_samples
    want_external = bool(external_metrics) and labeled and partitions
    want_internal = bool(internal_metrics) and partitions

    def new_external():
        if not want_external:
            return {}
        return {m: o for m, o in cq.build_external_metrics().items() if m in external_metrics}

    def new_internal():
        if not want_internal:
            return {}
        return {m: o for m, o in cq.build_internal_metrics().items() if m in internal_metrics}

    # cumulative metric objects (never reset) + per-window ones (reset each window)
    cum_ext, cum_int = new_external(), new_internal()
    win_ext, win_int = new_external(), new_internal()

    if n_samples is not None:
        length = n_samples
    else:
        try:
            length = len(dataset)
        except TypeError:  # __len__ returned None (unknown length)
            length = None
    checkpoint_every = max(length // n_checkpoints, 1) if length else None

    mc_samples: list[int] = []

    windows: list[WindowQuality] = []
    win_start = score_from
    win_labels: collections.Counter = collections.Counter()
    #: Predicted cluster -> how many scored samples landed in it.
    pred_sizes: collections.Counter = collections.Counter()

    checkpoints: list[Checkpoint] = []
    total_runtime = 0.0
    peak_mem = 0
    peak_rss = 0
    max_mc = 0
    n_seen = 0

    def eff_snapshot():
        """Sample efficiency. Expensive (memory probe) -> checkpoints only."""
        nonlocal max_mc, peak_mem, peak_rss
        mc = adapter.n_micro_clusters()
        mac = adapter.n_macro_clusters()
        mem = adapter.memory_bytes()
        rss = _process_rss()
        if mc is not None:
            mc_samples.append(mc)
            max_mc = max(max_mc, mc)
        if mem:
            peak_mem = max(peak_mem, mem)
        if rss:
            peak_rss = max(peak_rss, rss)
        checkpoints.append(
            Checkpoint(n_seen, total_runtime, mem, mc, mac, rss, adapter.structure())
        )

    def flush_window(start: int, end: int):
        rec = WindowQuality(start=start, end=end)
        total = sum(win_labels.values())
        if total:
            rec.n_true_classes = len(win_labels)
            rec.majority_class_share = max(win_labels.values()) / total
        for name, metric in cum_ext.items():
            rec.metrics[name] = metric.get()
        for name, metric in cum_int.items():
            rec.metrics[name] = metric.get()
        for name, metric in win_ext.items():
            rec.window_metrics[name] = metric.get()
        for name, metric in win_int.items():
            rec.window_metrics[name] = metric.get()
        if rec.metrics or rec.window_metrics:
            windows.append(rec)

    for x, y in dataset:
        if n_samples is not None and n_seen >= n_samples:
            break

        # quality-window boundary: flush, then reset only the per-window metrics
        if n_seen > score_from and (n_seen - score_from) % window_size == 0:
            flush_window(win_start, n_seen - 1)
            win_start = n_seen
            win_ext, win_int = new_external(), new_internal()
            win_labels.clear()

        t0 = time.perf_counter()
        y_pred = adapter.process_one(x)  # scale -> predict -> learn
        total_runtime += time.perf_counter() - t0
        n_seen += 1

        # Outside the timed region on purpose: reporting progress must not be
        # charged to the algorithm. The modulo keeps the per-sample cost to one
        # integer operation; the callback itself throttles on wall-clock.
        if progress is not None and n_seen % PROGRESS_EVERY == 0:
            progress(n_seen, length, adapter.n_micro_clusters())

        # `n_seen` has already been incremented, so the sample just processed
        # sits at index `n_seen - 1`.
        scoring = n_seen > score_from
        if scoring and y_pred is not None:
            pred_sizes[y_pred] += 1

        # Samples with no ground truth (e.g. injected noise) are skipped by the
        # external metrics rather than counted as a class of their own.
        if scoring and want_external and y is not None and y_pred is not None:
            win_labels[y] += 1
            for metric in cum_ext.values():
                metric.update(y, y_pred)
            for metric in win_ext.values():
                metric.update(y, y_pred)
        if scoring and want_internal:
            centers = adapter.centers()
            # Score in the model's own (scaled) space, where its centers live.
            x_model = adapter.last_x_scaled if adapter.last_x_scaled is not None else x
            # Most internal metrics compare the point against the model's
            # centers and can do nothing without them. `matr_trace` needs only
            # the predicted label, so gating every internal metric on `centers`
            # would leave it empty for exactly the clusterers it exists to
            # cover. Each metric declares which it is.
            for metrics_group in (cum_int, win_int):
                for metric in metrics_group.values():
                    if centers or not getattr(metric, "requires_centers", True):
                        metric.update(x_model, y_pred, centers)

        if checkpoint_every and n_seen % checkpoint_every == 0:
            eff_snapshot()

    # final partial window + final efficiency checkpoint
    if n_seen > win_start:
        flush_window(win_start, max(n_seen - 1, win_start))
    if not checkpoints or checkpoints[-1].n_seen != n_seen:
        eff_snapshot()

    # Whole-stream quality (single value over every scored sample).
    #
    # `adapter.partitions_samples` is re-read here rather than trusted from the
    # top of the run: a model can reveal mid-stream that it has no per-sample
    # assignment (ODAC raises NotImplementedError on the first predict). The
    # metric objects would then have received no updates at all and would still
    # report a value -- ARI returns 1.0 on an empty contingency table by its
    # 0/0 convention -- so a perfect score would be recorded for a model that
    # never predicted anything. Drop the metrics instead.
    if adapter.partitions_samples:
        quality: dict = {name: metric.get() for name, metric in cum_ext.items()}
        quality.update({name: metric.get() for name, metric in cum_int.items()})
    else:
        # Emptying the window list also empties the window diagnostics computed
        # just below, which is correct: there were no per-window scores.
        quality = {}
        windows = []

    # How much the per-window scores can be trusted on this particular stream.
    diagnostics: dict = {}
    scored = [w for w in windows if w.n_true_classes is not None]
    if scored:
        shares = sorted(w.majority_class_share for w in scored)
        degenerate = sum(w.is_degenerate for w in scored)
        diagnostics = {
            "n_windows": len(scored),
            "n_single_class_windows": degenerate,
            "single_class_window_share": round(degenerate / len(scored), 4),
            "median_majority_class_share": round(shares[len(shares) // 2], 4),
        }

    # Shape of the predicted partition. Two cheap degeneracies hide behind an
    # ARI value: shattering the stream into clusters too small to mean anything,
    # and dropping nearly everything into one cluster.
    cluster_sizes: dict = {}
    if adapter.partitions_samples and pred_sizes:
        total_assigned = sum(pred_sizes.values())
        dust_floor = 0.001 * total_assigned
        n_dust = sum(1 for count in pred_sizes.values() if count < dust_floor)
        cluster_sizes = {
            "n_predicted_clusters": len(pred_sizes),
            "n_dust_clusters": n_dust,
            "dust_cluster_share": round(n_dust / len(pred_sizes), 4),
            "largest_cluster_share": round(max(pred_sizes.values()) / total_assigned, 4),
        }

    efficiency = EfficiencyMetrics(
        total_runtime_s=total_runtime,
        n_samples=n_seen,
        peak_memory_bytes=peak_mem or None,
        peak_process_rss_bytes=peak_rss or None,
        mean_micro_clusters=(float(np.mean(mc_samples)) if mc_samples else None),
        max_micro_clusters=(max_mc or None),
        final_macro_clusters=adapter.n_macro_clusters(),
    )

    return RunResult(
        algorithm=adapter.name,
        dataset=dataset.name,
        scaler=adapter.scaler_name,
        scaler_params=dict(adapter.scaler_params),
        params=adapter.params,
        n_samples=n_seen,
        n_features=getattr(dataset, "n_features", None),
        efficiency=efficiency,
        quality=quality,
        quality_stream=_summarize_all(windows, "window_metrics"),
        quality_cumulative_stream=_summarize_all(windows, "metrics"),
        windows=windows,
        checkpoints=checkpoints,
        n_fallback_predictions=adapter.n_fallback_predictions,
        n_sentinel_predictions=adapter.n_sentinel_predictions,
        partitions_samples=adapter.partitions_samples,
        structure=adapter.structure(),
        window_label_diagnostics=diagnostics,
        cluster_size_diagnostics=cluster_sizes,
        window_size=window_size,
        seed=seed,
        score_from=score_from,
    )
