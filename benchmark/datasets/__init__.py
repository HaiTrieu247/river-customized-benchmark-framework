"""Datasets: synthetic clustering streams + real-world loaders."""

from __future__ import annotations

from .base import Dataset, IterableDataset
from .real import (
    INSECTS_VARIANTS,
    from_river_dataset,
    load_covtype,
    load_electricity,
    load_insects,
    load_kddcup98,
    load_kddcup99,
    load_keystroke,
    load_shuttle,
    load_sms_spam,
)
from .synthetic import SCENARIOS, StreamConfig, SyntheticStream, make_scenario

__all__ = [
    "Dataset",
    "IterableDataset",
    # synthetic
    "StreamConfig",
    "SyntheticStream",
    "SCENARIOS",
    "make_scenario",
    # real
    "load_kddcup99",
    "load_kddcup98",
    "load_covtype",
    "load_insects",
    "INSECTS_VARIANTS",
    "load_shuttle",
    "load_sms_spam",
    "load_keystroke",
    "load_electricity",
    "from_river_dataset",
]
