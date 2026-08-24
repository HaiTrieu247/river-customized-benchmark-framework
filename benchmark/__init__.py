"""Streaming clustering benchmark framework under concept drift.

Built around the idea that, for streaming clustering, the benchmark must answer
*"which algorithm adapts best when the cluster structure changes over time, at a
reasonable compute/memory cost?"* rather than only *"which has the highest final
clustering quality?"*.

The design follows ``streaming_clustering_concept_drift_benchmark_direction.md``:

    Synthetic Stream -> Ground Truth Cluster Evolution -> Drift Events / States
        -> Streaming Model -> Observed Clusters -> Evaluator
        -> (Cluster Quality, Drift Adaptation, Efficiency)
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
