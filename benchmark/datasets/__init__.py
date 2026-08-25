"""Datasets: synthetic static clusters + real-world loaders (KDD)."""

from __future__ import annotations

from .base import Dataset, IterableDataset
from .real import load_kddcup98, load_kddcup99
from .synthetic import StreamConfig, SyntheticStream

__all__ = [
    "Dataset",
    "IterableDataset",
    "StreamConfig",
    "SyntheticStream",
    "load_kddcup99",
    "load_kddcup98",
]
