"""Streaming clustering benchmark framework.

Run many River clustering algorithms — in several settings (params + scaler) —
on the *same* dataset, and compare them on resource cost (execution time, RAM,
micro/macro-cluster counts, throughput) and, when the dataset is labeled, on
clustering quality (ARI / NMI / purity).

    Dataset -> [ algorithm + params + scaler ] x N -> Runner -> Report
                                                        |
                          (runtime, memory, cluster counts, quality)
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.2.0"
