"""Synthetic clustering-stream generator and its named scenarios."""

from __future__ import annotations

from .generator import EVENTS, SCENARIOS, StreamConfig, SyntheticStream, make_scenario

__all__ = ["StreamConfig", "SyntheticStream", "SCENARIOS", "make_scenario", "EVENTS"]
