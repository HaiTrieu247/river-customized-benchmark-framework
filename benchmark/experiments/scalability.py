"""Scalability experiments: how each algorithm scales with the stream.

Two sweeps, matching the evaluation criteria:

* **vs #samples** — runtime / throughput / memory / quality as the stream grows
  (10K -> 100K -> ...);
* **vs #features** — the same measures as dimensionality grows (2D -> 10D -> ...).

Each cell is a full :class:`RunResult`, so any metric can be tabulated/plotted.
"""

from __future__ import annotations

from ..datasets.synthetic import StreamConfig, SyntheticStream
from ..evaluation.runner import RunResult, run_once
from .harness import Setting, run_setting, BenchmarkConfig


def _run(setting: Setting, dataset, config: BenchmarkConfig, seed: int) -> RunResult:
    return run_setting(setting, dataset, config, seed)


def sweep_samples(
    settings: list[Setting],
    sizes=(10_000, 50_000, 100_000),
    n_clusters: int = 5,
    n_features: int = 2,
    seed: int = 1,
    window_size: int = 1000,
) -> dict:
    """Runtime/memory/quality vs number of samples.

    Returns ``{setting_label: [(n_samples, RunResult), ...]}``.
    """
    config = BenchmarkConfig(window_size=window_size)
    out: dict[str, list] = {s.display(): [] for s in settings}
    for n in sizes:
        dataset = SyntheticStream(
            StreamConfig(n_samples=n, n_features=n_features, n_clusters=n_clusters, seed=42)
        )
        for s in settings:
            res = _run(s, dataset, config, seed)
            out[s.display()].append((n, res))
    return out


def sweep_features(
    settings: list[Setting],
    dims=(2, 10, 50, 100),
    n_samples: int = 20_000,
    n_clusters: int = 5,
    seed: int = 1,
    window_size: int = 1000,
) -> dict:
    """Runtime/memory/quality vs dimensionality.

    Returns ``{setting_label: [(n_features, RunResult), ...]}``.
    """
    config = BenchmarkConfig(window_size=window_size)
    out: dict[str, list] = {s.display(): [] for s in settings}
    for d in dims:
        dataset = SyntheticStream(
            StreamConfig(n_samples=n_samples, n_features=d, n_clusters=n_clusters, seed=42)
        )
        for s in settings:
            res = _run(s, dataset, config, seed)
            out[s.display()].append((d, res))
    return out


def scalability_table(sweep: dict, axis_label: str, metric: str = "ari") -> str:
    """Tabulate a sweep: one row per (setting, axis value)."""
    from ..reports.tables import format_table

    headers = ["Setting", axis_label, "Runtime (s)", "Throughput (sps)",
               "Memory (MB)", "Mean MC", f"Mean {metric.upper()}"]
    rows = []
    for label, cells in sweep.items():
        for axis_val, res in cells:
            e = res.efficiency.to_dict()
            mean_q = res.quality_stream.get(metric, {}).get("mean")
            rows.append([
                label, axis_val, e["total_runtime_s"], e["throughput_sps"],
                e["peak_memory_mb"], e["mean_micro_clusters"], mean_q,
            ])
    return format_table(headers, rows)
