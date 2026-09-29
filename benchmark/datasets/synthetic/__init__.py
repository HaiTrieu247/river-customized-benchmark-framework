"""Synthetic clustering-stream generator and its named scenarios."""

from __future__ import annotations

from .birch import (
    BIRCH_BASE_WORKLOAD,
    BIRCH_PAPER_REFERENCE,
    BirchGeneratorConfig,
    make_birch_dataset,
    make_ds1,
    make_ds2,
    make_ds3,
    weighted_average_diameter,
)
from .generator import EVENTS, SCENARIOS, StreamConfig, SyntheticStream, make_scenario
from .paper_benchmarks import (
    make_dstream_30k_nonconvex,
    make_dstream_85k_dynamic,
    make_scalability_dataset,
)
from .param_drift import PARAM_DRIFT_SCENARIOS, ParamDriftStream, make_param_drift
from .repstream import (
    DS1_NAMES,
    DS2_NAMES,
    SCALE_DS1,
    SCALE_DS2,
    make_repstream_ds1,
    make_repstream_ds2,
)

__all__ = [
    "StreamConfig",
    "SyntheticStream",
    "SCENARIOS",
    "make_scenario",
    "EVENTS",
    "ParamDriftStream",
    "PARAM_DRIFT_SCENARIOS",
    "make_param_drift",
    "make_dstream_30k_nonconvex",
    "make_dstream_85k_dynamic",
    "make_scalability_dataset",
    "BirchGeneratorConfig",
    "make_birch_dataset",
    "make_ds1",
    "make_ds2",
    "make_ds3",
    "BIRCH_BASE_WORKLOAD",
    "BIRCH_PAPER_REFERENCE",
    "weighted_average_diameter",
    "make_repstream_ds1",
    "make_repstream_ds2",
    "DS1_NAMES",
    "DS2_NAMES",
    "SCALE_DS1",
    "SCALE_DS2",
]
