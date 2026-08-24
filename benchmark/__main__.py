"""Command-line entry point for the benchmark framework.

Examples::

    python -m benchmark stationary
    python -m benchmark drift S6_split
    python -m benchmark efficiency
    python -m benchmark calibrate
    python -m benchmark matrix --scenarios S1_stationary,S2_abrupt_movement
    python -m benchmark matrix --out results.json
"""

from __future__ import annotations

import argparse
import json

from .datasets.synthetic.scenarios import SCENARIOS
from .experiments import BenchmarkConfig, run_matrix, run_scenario
from .experiments.efficiency import main as efficiency_main
from .experiments.hyperparameter import main as calibrate_main
from .experiments.stationary import main as stationary_main
from .reports.tables import (
    adaptation_table,
    before_during_after_table,
    efficiency_table,
    quality_summary_table,
)


def _cmd_stationary(_args) -> None:
    stationary_main()


def _cmd_drift(args) -> None:
    config = BenchmarkConfig(seeds=(args.seed,), n_samples=args.n_samples, window_size=args.window)
    results = run_scenario(args.scenario, config)
    print(f"# Drift adaptation — {args.scenario}\n")
    print("## Quality summary\n")
    print(quality_summary_table(results))
    print("\n## Before / Min / Recovery / After (§18)\n")
    print(before_during_after_table(results))
    print("\n## Adaptation metrics (§12)\n")
    print(adaptation_table(results))
    print("\n## Efficiency (§13)\n")
    print(efficiency_table(results))


def _cmd_efficiency(_args) -> None:
    efficiency_main()


def _cmd_calibrate(_args) -> None:
    calibrate_main()


def _cmd_matrix(args) -> None:
    scenarios = tuple(args.scenarios.split(",")) if args.scenarios else None
    config = BenchmarkConfig(
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        n_samples=args.n_samples,
        window_size=args.window,
    )
    matrix = run_matrix(scenarios, config)

    # Build a JSON-serialisable digest (drop raw ExperimentResult objects).
    digest = {}
    for scen, block in matrix.items():
        digest[scen] = {
            "seeds": block["seeds"],
            "algorithms": {
                alg: {
                    "efficiency": data["efficiency"],
                    "adaptation": data["adaptation"],
                    "n_runs": data["n_runs"],
                }
                for alg, data in block["algorithms"].items()
            },
        }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(digest, fh, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(digest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchmark", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("stationary", help="Phase 2 baseline (§17)").set_defaults(func=_cmd_stationary)

    d = sub.add_parser("drift", help="Phase 3-4 drift adaptation (§12,§18)")
    d.add_argument("scenario", nargs="?", default="S2_abrupt_movement", choices=list(SCENARIOS))
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--n-samples", dest="n_samples", type=int, default=20_000)
    d.add_argument("--window", type=int, default=1000)
    d.set_defaults(func=_cmd_drift)

    sub.add_parser("efficiency", help="Phase 6 efficiency/scalability (§13)").set_defaults(func=_cmd_efficiency)
    sub.add_parser("calibrate", help="Phase 5 hyperparameter protocol (§19)").set_defaults(func=_cmd_calibrate)

    m = sub.add_parser("matrix", help="Run the S1..S10 matrix over seeds (§16,§20)")
    m.add_argument("--scenarios", default=None, help="comma-separated scenario ids")
    m.add_argument("--seeds", default="1,2,3")
    m.add_argument("--n-samples", dest="n_samples", type=int, default=None)
    m.add_argument("--window", type=int, default=1000)
    m.add_argument("--out", default=None, help="write JSON digest to this path")
    m.set_defaults(func=_cmd_matrix)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
