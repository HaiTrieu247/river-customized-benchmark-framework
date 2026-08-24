"""Phase 6 — efficiency & scalability benchmark (§13, roadmap Phase 6).

Measures runtime / throughput / memory / micro-cluster count as the stream size
grows, to expose scalability behaviour (§13 "Scalability").

Run directly::

    python -m benchmark.experiments.efficiency
"""

from __future__ import annotations

from ..algorithms.registry import get_algorithm
from ..datasets.synthetic.scenarios import stationary
from ..evaluation.runner import run_experiment
from ..reports.tables import format_table


def scalability(algorithms=("CluStream", "DenStream", "DBSTREAM", "STREAMKMeans"),
                sizes=(5_000, 10_000, 20_000), seed: int = 1) -> str:
    """Throughput / memory vs stream size, per algorithm (§13)."""
    headers = ["Algorithm", "n_samples", "Runtime (s)", "Throughput (sps)", "Memory (MB)", "Mean MC"]
    rows = []
    for alg in algorithms:
        for n in sizes:
            stream, gt = stationary(n_samples=n, seed=seed)
            adapter = get_algorithm(alg)
            res = run_experiment(adapter, stream, gt, window_size=1000, metrics=("ari",))
            e = res.efficiency.to_dict()
            rows.append([alg, n, e["total_runtime_s"], e["throughput_sps"],
                         e["peak_memory_mb"], e["mean_micro_clusters"]])
    return format_table(headers, rows)


def main() -> None:
    print("# Efficiency & scalability (§13)\n")
    print(scalability())


if __name__ == "__main__":
    main()
