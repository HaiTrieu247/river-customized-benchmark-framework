"""Paper benchmark reproductions framework and registry."""

from .base import (
    BasePaperBenchmark,
    PaperFigureSpec,
    get_paper,
    list_paper_names,
    list_papers,
    register_paper,
)
from .dstream_vs_clustream import DStream2007Benchmark
from .repstream import RepStream2009Benchmark

__all__ = [
    "BasePaperBenchmark",
    "PaperFigureSpec",
    "register_paper",
    "get_paper",
    "list_papers",
    "list_paper_names",
    "DStream2007Benchmark",
    "RepStream2009Benchmark",
]
