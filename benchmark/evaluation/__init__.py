"""Evaluation: cluster quality, drift adaptation, efficiency, aggregation (§10)."""

from __future__ import annotations

from .adaptation import AdaptationMetrics, compute_adaptation
from .clustering_quality import ari, nmi, purity
from .efficiency import EfficiencyMetrics
from .runner import WindowRecord, run_experiment

__all__ = [
    "ari",
    "nmi",
    "purity",
    "AdaptationMetrics",
    "compute_adaptation",
    "EfficiencyMetrics",
    "WindowRecord",
    "run_experiment",
]
