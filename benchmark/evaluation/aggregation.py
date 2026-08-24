"""Aggregate repeated runs across seeds -> mean +/- std (§20)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Aggregate:
    mean: float
    std: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f}"

    def to_dict(self) -> dict:
        return {"mean": round(self.mean, 4), "std": round(self.std, 4), "n": self.n}


def _agg(values) -> Aggregate | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return Aggregate(float(arr.mean()), float(arr.std(ddof=0)), len(vals))


def aggregate_efficiency(results) -> dict:
    """Aggregate efficiency across a list of ExperimentResult (same algo/scenario)."""
    keys = ["total_runtime_s", "throughput_sps", "ms_per_sample", "peak_memory_mb", "mean_micro_clusters"]
    buckets = defaultdict(list)
    for r in results:
        d = r.efficiency.to_dict()
        for k in keys:
            buckets[k].append(d.get(k))
    return {k: (_agg(v).to_dict() if _agg(v) else None) for k, v in buckets.items()}


def aggregate_adaptation(results) -> dict:
    """Aggregate per-drift adaptation metrics across seeds, keyed by drift_id."""
    by_drift: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    meta: dict[int, dict] = {}
    fields = [
        "quality_before", "quality_min", "performance_drop", "recovery_time",
        "adaptation_delay", "post_drift_quality", "area_under_recovery",
    ]
    for r in results:
        for a in r.adaptation:
            d = a.to_dict()
            did = d["drift_id"]
            meta[did] = {"drift_type": d["drift_type"], "drift_start": d["drift_start"]}
            for f in fields:
                by_drift[did][f].append(d.get(f))
            by_drift[did]["_recovered"].append(1.0 if a.recovered else 0.0)
    out = {}
    for did, cols in by_drift.items():
        entry = dict(meta[did])
        for f in fields:
            agg = _agg(cols[f])
            entry[f] = agg.to_dict() if agg else None
        rec = _agg(cols["_recovered"])
        entry["recovery_rate"] = round(rec.mean, 3) if rec else None
        out[did] = entry
    return out


def aggregate_quality_curves(results, metric: str = "ari"):
    """Average quality curves aligned on window center (§23 plotting)."""
    series = defaultdict(list)
    for r in results:
        for w in r.windows:
            v = w.metrics.get(metric)
            if v is not None:
                series[w.center].append(v)
    centers = sorted(series)
    means = [float(np.mean(series[c])) for c in centers]
    stds = [float(np.std(series[c], ddof=0)) for c in centers]
    return centers, means, stds
