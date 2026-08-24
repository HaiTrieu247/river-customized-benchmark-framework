"""Phase 2 — stationary baseline experiment (§17, roadmap Phase 2).

Run directly::

    python -m benchmark.experiments.stationary
"""

from __future__ import annotations

from ..reports.tables import efficiency_table, quality_summary_table
from .harness import BenchmarkConfig, run_scenario


def main() -> None:
    config = BenchmarkConfig(seeds=(1,), n_samples=10_000, window_size=1000)
    results = run_scenario("S1_stationary", config)
    print("# Stationary baseline (S1)\n")
    print("## Cluster quality\n")
    print(quality_summary_table(results))
    print("\n## Efficiency\n")
    print(efficiency_table(results))


if __name__ == "__main__":
    main()
