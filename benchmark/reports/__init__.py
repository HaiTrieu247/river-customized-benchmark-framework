"""Reporting: comparison tables (§18) and plots (§23)."""

from __future__ import annotations

from .tables import (
    adaptation_table,
    before_during_after_table,
    efficiency_table,
    format_table,
)

__all__ = [
    "adaptation_table",
    "before_during_after_table",
    "efficiency_table",
    "format_table",
]

# Plotting depends on matplotlib; import lazily so the core stays import-light.
try:  # pragma: no cover - optional dependency guard
    from .plots import (  # noqa: F401
        plot_drift_timeline,
        plot_quality_over_time,
        plot_recovery_curve,
    )

    __all__ += ["plot_quality_over_time", "plot_drift_timeline", "plot_recovery_curve"]
except Exception:  # matplotlib missing
    pass
