"""Adapters over River clusterers with a uniform benchmark interface.

Not every streaming clusterer exposes a drift detector (§9); the adapter only
requires ``learn_one``/``predict_one`` and best-effort reports micro-cluster
count and centers when the underlying model exposes them.
"""

from __future__ import annotations

from .adapter import ClustererAdapter, build_river_path, make_adapter
from .registry import ALGORITHMS, get_algorithm

__all__ = [
    "ClustererAdapter",
    "make_adapter",
    "build_river_path",
    "ALGORITHMS",
    "get_algorithm",
]
