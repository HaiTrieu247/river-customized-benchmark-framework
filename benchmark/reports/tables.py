"""Comparison tables (plain markdown, no heavy deps).

Tables:

* **efficiency** — runtime / throughput / memory / cluster counts;
* **quality** — whole-stream ARI / NMI / AMI / purity (labeled datasets);
* **stream-quality** — mean / median / final of a metric over evaluation windows
  (time-aware evaluation, the top-priority quality criterion).

Each has a flat (single-seed) form and an aggregated (mean ± std over seeds)
form that consumes a ``run_benchmark_repeated`` digest.
"""

from __future__ import annotations


def format_table(headers, rows) -> str:
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


# --- flat (single-seed) tables ---------------------------------------------
def efficiency_table(results) -> str:
    """Runtime / throughput / memory / cluster comparison for a list of RunResult.

    ``Model MB`` is river's model-only footprint (the comparable number);
    ``RSS MB`` is whole-process resident memory, shown for context.
    """
    headers = ["Setting", "Samples", "Runtime (s)", "Throughput (sps)", "ms/sample",
               "Model MB", "RSS MB", "Mean MC", "Max MC", "Macro K", "Cold-start"]
    rows = []
    for r in results:
        e = r.efficiency.to_dict()
        rows.append([
            r.algorithm, r.n_samples, e["total_runtime_s"], e["throughput_sps"],
            e["ms_per_sample"], e["peak_memory_mb"], e.get("peak_process_rss_mb"),
            e["mean_micro_clusters"], e["max_micro_clusters"], e["final_macro_clusters"],
            r.n_fallback_predictions,
        ])
    return format_table(headers, rows)


def quality_table(results, metrics=("ari", "nmi", "ami", "purity")) -> str:
    """Whole-stream quality comparison; only meaningful for labeled datasets."""
    headers = ["Setting"] + [m.upper() for m in metrics]
    rows = []
    for r in results:
        rows.append([r.algorithm] + [r.quality.get(m) for m in metrics])
    return format_table(headers, rows)


def stream_quality_table(results, metric="ari", cumulative=False) -> str:
    """Time-aware summary of one metric: mean / median / final over windows.

    By default this summarises the **per-window** values (each window scored on
    its own samples), which is the time-aware protocol number. Pass
    ``cumulative=True`` to summarise the converging from-stream-start curve
    instead.
    """
    source = "quality_cumulative_stream" if cumulative else "quality_stream"
    kind = "cum." if cumulative else "win."
    headers = ["Setting", f"Mean {metric.upper()} ({kind})",
               f"Median {metric.upper()} ({kind})", f"Final {metric.upper()} ({kind})"]
    rows = []
    for r in results:
        s = getattr(r, source).get(metric, {})
        rows.append([r.algorithm, s.get("mean"), s.get("median"), s.get("final")])
    return format_table(headers, rows)


# --- aggregated (repeated-run) tables --------------------------------------
def _agg_cell(agg: dict | None) -> str:
    if not agg:
        return "-"
    return f"{agg['mean']:.4f} ± {agg['std']:.4f}"


def efficiency_table_agg(digest: dict) -> str:
    """Aggregated efficiency table from run_benchmark_repeated digest."""
    headers = ["Setting", "Runs", "Runtime (s)", "Throughput (sps)", "Model MB",
               "Mean MC", "Max MC", "Macro K"]
    rows = []
    for label, block in digest.items():
        agg = block["aggregate"]
        e = agg["efficiency"]
        rows.append([
            label, agg["n_runs"], _agg_cell(e["total_runtime_s"]),
            _agg_cell(e["throughput_sps"]), _agg_cell(e["peak_memory_mb"]),
            _agg_cell(e["mean_micro_clusters"]), _agg_cell(e["max_micro_clusters"]),
            _agg_cell(e["final_macro_clusters"]),
        ])
    return format_table(headers, rows)


def quality_table_agg(digest: dict, metrics=("ari", "nmi", "ami", "purity")) -> str:
    """Aggregated whole-stream quality (mean ± std over seeds)."""
    headers = ["Setting", "Runs"] + [m.upper() for m in metrics]
    rows = []
    for label, block in digest.items():
        agg = block["aggregate"]
        row = [label, agg["n_runs"]]
        for m in metrics:
            row.append(_agg_cell(agg["quality"].get(m)))
        rows.append(row)
    return format_table(headers, rows)


def stream_quality_table_agg(digest: dict, metric="ari") -> str:
    """Aggregated time-aware summary over per-window values (mean ± std over seeds)."""
    headers = ["Setting", "Runs", f"Mean {metric.upper()}", f"Median {metric.upper()}",
               f"Final {metric.upper()}"]
    rows = []
    for label, block in digest.items():
        agg = block["aggregate"]
        s = agg.get("quality_stream", {}).get(metric, {})
        rows.append([
            label, agg["n_runs"], _agg_cell(s.get("mean")),
            _agg_cell(s.get("median")), _agg_cell(s.get("final")),
        ])
    return format_table(headers, rows)


def window_diagnostics_table(results) -> str:
    """How much the per-window scores can be trusted on this stream.

    On a window holding a single ground-truth class, ARI degenerates into a
    "did you lump the whole window together?" test: 1.0 for lumping, 0.0 for any
    split, whatever the clustering is worth. A stream that arrives block-ordered
    by class therefore rewards under-segmentation, and this table makes that
    visible instead of letting it pass as a good result.
    """
    d = next((r.window_label_diagnostics for r in results if r.window_label_diagnostics), None)
    if not d:
        return "(no labeled windows)"
    headers = ["Windows", "Single-class windows", "Share", "Median majority-class share"]
    rows = [[
        d["n_windows"], d["n_single_class_windows"],
        f"{d['single_class_window_share']:.0%}", d["median_majority_class_share"],
    ]]
    return format_table(headers, rows)


def structure_table(results) -> str:
    """Structure summary for clusterers that do not partition samples.

    ODAC clusters the *variables*, so ARI/NMI/AMI/purity are undefined for it and
    it is absent from the quality tables. What it does produce is a hierarchy,
    and its size and depth are the meaningful outputs — reported here so the run
    is still comparable on something rather than silently dropped.
    """
    rows = []
    for r in results:
        if r.partitions_samples or not r.structure:
            continue
        rows.append([
            r.algorithm,
            r.structure.get("n_clusters"),
            r.structure.get("n_active_clusters"),
            r.structure.get("height"),
            round(r.efficiency.throughput_sps, 1),
        ])
    if not rows:
        return "(no structure-only algorithms in this run)"
    headers = ["Setting", "Variable clusters", "Active", "Tree height", "Throughput (sps)"]
    return format_table(headers, rows)
