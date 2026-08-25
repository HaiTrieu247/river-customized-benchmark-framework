"""Aggregate repeated runs (across seeds) into mean +/- std.

Streaming clusterers can depend on seed / initialisation, so a robust benchmark
repeats each (algorithm, setting, dataset) run over several seeds and reports
mean +/- std rather than a single number.
"""

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


_EFF_KEYS = [
    "total_runtime_s", "throughput_sps", "ms_per_sample", "peak_memory_mb",
    "mean_micro_clusters", "max_micro_clusters", "final_macro_clusters",
]


def aggregate_runs(results) -> dict:
    """Aggregate a list of RunResult (same algorithm/setting/dataset) over seeds.

    Returns aggregated efficiency, whole-stream quality, and the mean-over-stream
    summary (mean/median/final per metric) — each as mean/std/n across seeds.
    """
    eff = defaultdict(list)
    qual = defaultdict(list)
    stream = defaultdict(lambda: defaultdict(list))  # metric -> stat -> [values]
    for r in results:
        d = r.efficiency.to_dict()
        for k in _EFF_KEYS:
            eff[k].append(d.get(k))
        for k, v in r.quality.items():
            qual[k].append(v)
        for m, stats in r.quality_stream.items():
            for stat, v in stats.items():
                stream[m][stat].append(v)

    return {
        "n_runs": len(results),
        "efficiency": {k: (_agg(v).to_dict() if _agg(v) else None) for k, v in eff.items()},
        "quality": {k: (_agg(v).to_dict() if _agg(v) else None) for k, v in qual.items()},
        "quality_stream": {
            m: {stat: (_agg(v).to_dict() if _agg(v) else None) for stat, v in stats.items()}
            for m, stats in stream.items()
        },
    }
