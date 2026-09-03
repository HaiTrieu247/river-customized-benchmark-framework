"""Synthetic clustering-stream generator and its named scenarios."""

from __future__ import annotations

from .generator import EVENTS, SCENARIOS, StreamConfig, SyntheticStream, make_scenario
from .paper_benchmarks import (
    make_dstream_30k_nonconvex,
    make_dstream_85k_dynamic,
    make_scalability_dataset,
)

__all__ = [
    "StreamConfig",
    "SyntheticStream",
    "SCENARIOS",
    "make_scenario",
    "EVENTS",
    "make_dstream_30k_nonconvex",
    "make_dstream_85k_dynamic",
    "make_scalability_dataset",
]
