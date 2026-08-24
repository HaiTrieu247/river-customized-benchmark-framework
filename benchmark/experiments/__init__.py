"""Experiment orchestration (§14, §20) and the scenario matrix driver (§16)."""

from __future__ import annotations

from .harness import (
    BenchmarkConfig,
    run_matrix,
    run_scenario,
    run_scenario_repeated,
)

__all__ = [
    "BenchmarkConfig",
    "run_scenario",
    "run_scenario_repeated",
    "run_matrix",
]
