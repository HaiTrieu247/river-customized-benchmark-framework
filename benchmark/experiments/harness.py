"""Benchmark harness (§14, §16, §20).

Enforces the protocol from the direction doc:

* every algorithm runs on the *same* stream for a given (scenario, seed) (§14);
* runs are repeated over multiple seeds and aggregated to mean +/- std (§20);
* nothing about the ground truth is passed to the model (Principle 1) — the
  harness only hands the model the feature dict, and keeps ground truth for the
  evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..algorithms.registry import get_algorithm
from ..datasets.synthetic.scenarios import SCENARIOS
from ..evaluation.aggregation import (
    aggregate_adaptation,
    aggregate_efficiency,
    aggregate_quality_curves,
)
from ..evaluation.runner import ExperimentResult, run_experiment


@dataclass
class BenchmarkConfig:
    """Fixed experimental conditions for reproducibility (§14, Principle 5)."""

    algorithms: tuple[str, ...] = ("CluStream", "DenStream", "DBSTREAM", "STREAMKMeans")
    seeds: tuple[int, ...] = (1, 2, 3)
    n_samples: int | None = None  # None => each scenario's default size
    window_size: int = 1000
    metrics: tuple[str, ...] = ("ari", "nmi", "purity")
    recovery_fraction: float = 0.95
    # per-algorithm param overrides, e.g. {"DenStream": {"epsilon": 3.0}}
    overrides: dict = field(default_factory=dict)


def _make_stream(scenario: str, seed: int, n_samples: int | None):
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; available: {sorted(SCENARIOS)}")
    factory = SCENARIOS[scenario]
    kwargs = {"seed": seed}
    if n_samples is not None:
        kwargs["n_samples"] = n_samples
    return factory(**kwargs)


def run_scenario(
    scenario: str,
    config: BenchmarkConfig | None = None,
    seed: int | None = None,
) -> list[ExperimentResult]:
    """Run every configured algorithm once on ``scenario`` (single seed).

    All algorithms share the identical stream for the chosen seed (§14).
    """
    config = config or BenchmarkConfig()
    seed = config.seeds[0] if seed is None else seed
    results: list[ExperimentResult] = []
    for alg in config.algorithms:
        # rebuild the stream per algorithm so each consumes it from the start,
        # but with the same seed => identical sample sequence (§14).
        stream, gt = _make_stream(scenario, seed, config.n_samples)
        overrides = config.overrides.get(alg, {})
        adapter = get_algorithm(alg, **overrides)
        results.append(
            run_experiment(
                adapter, stream, gt,
                window_size=config.window_size,
                metrics=config.metrics,
                recovery_fraction=config.recovery_fraction,
                scenario_name=scenario,
            )
        )
    return results


def run_scenario_repeated(
    scenario: str,
    config: BenchmarkConfig | None = None,
) -> dict:
    """Run ``scenario`` over all seeds and aggregate (§20).

    Returns a dict with raw results per seed plus aggregated efficiency,
    adaptation and averaged quality curves per algorithm.
    """
    config = config or BenchmarkConfig()
    per_alg: dict[str, list[ExperimentResult]] = {a: [] for a in config.algorithms}
    for seed in config.seeds:
        for r in run_scenario(scenario, config, seed=seed):
            per_alg[r.algorithm].append(r)

    summary = {}
    for alg, runs in per_alg.items():
        centers, means, stds = aggregate_quality_curves(runs, metric=config.metrics[0])
        summary[alg] = {
            "efficiency": aggregate_efficiency(runs),
            "adaptation": aggregate_adaptation(runs),
            "quality_curve": {"center": centers, "mean": means, "std": stds},
            "n_runs": len(runs),
        }
    return {"scenario": scenario, "seeds": list(config.seeds), "algorithms": summary,
            "raw": per_alg}


def run_matrix(
    scenarios: tuple[str, ...] | None = None,
    config: BenchmarkConfig | None = None,
) -> dict:
    """Run the whole scenario matrix (§16), repeated over seeds (§20)."""
    config = config or BenchmarkConfig()
    scenarios = scenarios or tuple(SCENARIOS.keys())
    return {s: run_scenario_repeated(s, config) for s in scenarios}
