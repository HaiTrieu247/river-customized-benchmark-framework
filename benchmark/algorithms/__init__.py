"""Adapters over River clusterers with a uniform benchmark interface.

The adapter requires only ``learn_one``/``predict_one`` and best-effort reports
micro/macro-cluster counts and model memory. An optional online scaler can be
chained in front of any clusterer.
"""

from __future__ import annotations

from .adapter import ClustererAdapter, build_river_path, make_adapter
from .registry import (
    ALGORITHMS,
    SCALERS,
    discover_clusterers,
    get_algorithm,
    make_scaler,
)

__all__ = [
    "ClustererAdapter",
    "make_adapter",
    "build_river_path",
    "ALGORITHMS",
    "SCALERS",
    "get_algorithm",
    "make_scaler",
    "discover_clusterers",
]
