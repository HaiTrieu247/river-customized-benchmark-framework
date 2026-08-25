"""Efficiency metrics: runtime, throughput, memory, cluster counts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EfficiencyMetrics:
    """Aggregate efficiency numbers for one run."""

    total_runtime_s: float
    n_samples: int
    peak_memory_bytes: int | None = None
    mean_micro_clusters: float | None = None
    max_micro_clusters: int | None = None
    final_macro_clusters: int | None = None

    @property
    def throughput_sps(self) -> float:
        """Samples per second."""
        return self.n_samples / self.total_runtime_s if self.total_runtime_s > 0 else float("nan")

    @property
    def ms_per_sample(self) -> float:
        return (self.total_runtime_s * 1000.0) / self.n_samples if self.n_samples else float("nan")

    @property
    def peak_memory_mb(self) -> float | None:
        return self.peak_memory_bytes / (1024 * 1024) if self.peak_memory_bytes else None

    def to_dict(self) -> dict:
        return {
            "total_runtime_s": round(self.total_runtime_s, 4),
            "throughput_sps": round(self.throughput_sps, 2),
            "ms_per_sample": round(self.ms_per_sample, 5),
            "peak_memory_mb": round(self.peak_memory_mb, 3) if self.peak_memory_mb else None,
            "mean_micro_clusters": (
                round(self.mean_micro_clusters, 2) if self.mean_micro_clusters is not None else None
            ),
            "max_micro_clusters": self.max_micro_clusters,
            "final_macro_clusters": self.final_macro_clusters,
        }
