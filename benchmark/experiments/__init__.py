"""Experiment orchestration: run many settings on the same dataset."""

from __future__ import annotations

from .harness import (
    BenchmarkConfig,
    Setting,
    run_benchmark,
    run_benchmark_repeated,
    run_setting,
)
from .scalability import (
    scalability_table,
    sweep_features,
    sweep_samples,
)

__all__ = [
    "Setting",
    "BenchmarkConfig",
    "run_setting",
    "run_benchmark",
    "run_benchmark_repeated",
    "sweep_samples",
    "sweep_features",
    "scalability_table",
]
