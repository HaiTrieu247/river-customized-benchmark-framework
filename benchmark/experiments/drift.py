"""Phase 3-4 — synthetic drift + adaptation experiments (§12, §18, roadmap).

Run directly::

    python -m benchmark.experiments.drift                 # abrupt movement (S2)
    python -m benchmark.experiments.drift S6_split        # any scenario id
"""

from __future__ import annotations

import sys

from ..reports.tables import (
    adaptation_table,
    before_during_after_table,
    quality_summary_table,
)
from .harness import BenchmarkConfig, run_scenario


def main(scenario: str = "S2_abrupt_movement") -> None:
    config = BenchmarkConfig(seeds=(1,), n_samples=20_000, window_size=1000)
    results = run_scenario(scenario, config)
    print(f"# Drift adaptation — {scenario}\n")
    print("## Quality summary\n")
    print(quality_summary_table(results))
    print("\n## Before / Min / Recovery / After (§18)\n")
    print(before_during_after_table(results))
    print("\n## Adaptation metrics (§12)\n")
    print(adaptation_table(results))


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "S2_abrupt_movement"
    main(scenario)
