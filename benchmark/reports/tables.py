"""Comparison tables (§18) — plain text / markdown, no heavy deps.

The key table is the Before / Min / Recovery / After breakdown per drift (§18),
which is far more informative than a single final-ARI column.
"""

from __future__ import annotations

import numpy as np


def format_table(headers, rows, align: str = "left") -> str:
    """Render a simple markdown table from headers + row tuples."""
    cols = [str(h) for h in headers]
    str_rows = [[_fmt(c) for c in row] for row in rows]
    widths = [len(h) for h in cols]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(cols)) + " |"
    out = [head, sep]
    for row in str_rows:
        out.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |")
    return "\n".join(out)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def before_during_after_table(results, metric: str = "ari") -> str:
    """Per-algorithm Before / Min / Recovery / After table for one scenario (§18).

    ``results`` is a list of ExperimentResult on the *same* scenario. Uses the
    first drift event of each result (extend as needed for multi-drift streams).
    """
    headers = ["Algorithm", f"Before {metric.upper()}", f"Min {metric.upper()}",
               "Recovery", f"After {metric.upper()}", "Recovered"]
    rows = []
    for r in results:
        if not r.adaptation:
            rows.append([r.algorithm, None, None, None, None, "n/a"])
            continue
        a = r.adaptation[0]
        rows.append([
            r.algorithm,
            a.quality_before,
            a.quality_min,
            a.recovery_time,
            a.post_drift_quality,
            "yes" if a.recovered else "no",
        ])
    return format_table(headers, rows)


def adaptation_table(results, metric: str = "ari") -> str:
    """All drifts x all algorithms: performance drop, recovery, delay, AUC (§12)."""
    headers = ["Algorithm", "Drift", "Type", "Drop", "Recovery", "Delay", "PostQ", "AUC"]
    rows = []
    for r in results:
        for a in r.adaptation:
            rows.append([
                r.algorithm, a.drift_id, a.drift_type, a.performance_drop,
                a.recovery_time, a.adaptation_delay, a.post_drift_quality,
                a.area_under_recovery,
            ])
    return format_table(headers, rows)


def efficiency_table(results) -> str:
    """Runtime / throughput / memory / micro-cluster comparison (§13)."""
    headers = ["Algorithm", "Runtime (s)", "Throughput (sps)", "ms/sample",
               "Memory (MB)", "Mean MC", "Max MC"]
    rows = []
    for r in results:
        e = r.efficiency.to_dict()
        rows.append([
            r.algorithm, e["total_runtime_s"], e["throughput_sps"], e["ms_per_sample"],
            e["peak_memory_mb"], e["mean_micro_clusters"], e["max_micro_clusters"],
        ])
    return format_table(headers, rows)


def quality_summary_table(results, metric: str = "ari") -> str:
    """Mean / min / final quality over the whole stream, per algorithm."""
    headers = ["Algorithm", f"Mean {metric.upper()}", f"Min {metric.upper()}", f"Final {metric.upper()}"]
    rows = []
    for r in results:
        _, ys = r.quality_curve(metric)
        if ys:
            rows.append([r.algorithm, float(np.mean(ys)), float(np.min(ys)), ys[-1]])
        else:
            rows.append([r.algorithm, None, None, None])
    return format_table(headers, rows)
