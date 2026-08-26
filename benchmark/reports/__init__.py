"""Reporting: comparison tables and plots."""

from __future__ import annotations

from .environment import capture_environment, format_environment
from .tables import (
    efficiency_table,
    efficiency_table_agg,
    format_table,
    quality_table,
    quality_table_agg,
    stream_quality_table,
    stream_quality_table_agg,
    structure_table,
    window_diagnostics_table,
)

__all__ = [
    "capture_environment",
    "format_environment",
    "format_table",
    "efficiency_table",
    "quality_table",
    "stream_quality_table",
    "window_diagnostics_table",
    "structure_table",
    "efficiency_table_agg",
    "quality_table_agg",
    "stream_quality_table_agg",
]

# Plotting depends on matplotlib; import lazily so the core stays import-light.
try:  # pragma: no cover - optional dependency guard
    from .plots import (  # noqa: F401
        plot_clusters_vs_samples,
        plot_memory_vs_samples,
        plot_quality_bars,
        plot_runtime_vs_samples,
        plot_quality_curve,
        plot_window_quality_curve,
        plot_quality_vs_throughput,
        plot_throughput_vs_samples,
    )

    __all__ += [
        "plot_runtime_vs_samples",
        "plot_memory_vs_samples",
        "plot_clusters_vs_samples",
        "plot_quality_bars",
        "plot_quality_curve",
        "plot_window_quality_curve",
        "plot_quality_vs_throughput",
        "plot_throughput_vs_samples",
    ]
except Exception:  # matplotlib missing
    pass
