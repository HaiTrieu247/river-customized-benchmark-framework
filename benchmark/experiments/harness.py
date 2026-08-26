"""Benchmark harness: run many algorithm settings on the same dataset.

The core idea of this framework: define a dataset and a list of *settings*
(algorithm + params + scaler), then run them all on the identical stream and
compare execution time / RAM / cluster counts (and quality when labels exist).

Reproducibility rules enforced:

* every setting consumes the *same* dataset stream from the start (no shuffling);
* runs can be repeated over multiple seeds and aggregated to mean +/- std;
* each setting records its full params + scaler so results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..algorithms.registry import get_algorithm
from ..datasets.base import Dataset
from ..evaluation.aggregation import aggregate_runs
from ..evaluation.runner import RunResult, run_once


@dataclass
class Setting:
    """One benchmark configuration: an algorithm + params + scaler.

    ``label`` is what shows up in reports; defaults to a readable combination of
    algorithm and scaler. Add several settings of the same algorithm (different
    params/scaler) to compare configurations side by side.
    """

    algorithm: str
    params: dict = field(default_factory=dict)
    scaler: str = "none"
    scaler_params: dict = field(default_factory=dict)
    label: str | None = None

    def display(self) -> str:
        if self.label:
            return self.label
        if self.scaler == "none":
            return self.algorithm
        detail = ",".join(f"{k}={v}" for k, v in sorted(self.scaler_params.items()))
        return f"{self.algorithm} [{self.scaler}{'/' + detail if detail else ''}]"


@dataclass
class BenchmarkConfig:
    """Fixed experimental conditions for a benchmark run."""

    settings: list[Setting] = field(default_factory=list)
    seeds: tuple[int, ...] = (1,)
    n_samples: int | None = None
    n_checkpoints: int = 20
    window_size: int = 1000
    external_metrics: tuple[str, ...] = ("ari", "nmi", "ami", "purity")
    internal_metrics: tuple[str, ...] = ()


def run_setting(
    setting: Setting,
    dataset: Dataset,
    config: BenchmarkConfig,
    seed: int,
) -> RunResult:
    """Run a single setting once on ``dataset`` with the given seed."""
    from ..algorithms.registry import accepts_seed  # local import to avoid cycle

    params = dict(setting.params)
    # Inject the run seed only into algorithms whose constructor declares one.
    # Deciding by signature (rather than constructing and catching TypeError)
    # keeps a genuine bad-parameter error visible instead of silently retrying
    # with a different configuration than the one that was asked for.
    if accepts_seed(setting.algorithm):
        params.setdefault("seed", seed)

    adapter = get_algorithm(
        setting.algorithm, display_name=setting.display(),
        scaler=setting.scaler, scaler_params=setting.scaler_params, **params,
    )
    return run_once(
        adapter, dataset,
        n_samples=config.n_samples,
        n_checkpoints=config.n_checkpoints,
        window_size=config.window_size,
        external_metrics=config.external_metrics,
        internal_metrics=config.internal_metrics,
        seed=seed,
    )


def run_benchmark(dataset: Dataset, config: BenchmarkConfig) -> list[RunResult]:
    """Run every setting once (first seed) on ``dataset``. Returns flat results."""
    seed = config.seeds[0]
    return [run_setting(s, dataset, config, seed) for s in config.settings]


def run_benchmark_repeated(dataset: Dataset, config: BenchmarkConfig) -> dict:
    """Run every setting over all seeds and aggregate to mean +/- std.

    Returns ``{setting_label: {"runs": [...], "aggregate": {...}}}``.
    """
    out: dict[str, dict] = {}
    for setting in config.settings:
        runs = [run_setting(setting, dataset, config, seed) for seed in config.seeds]
        out[setting.display()] = {"runs": runs, "aggregate": aggregate_runs(runs)}
    return out
