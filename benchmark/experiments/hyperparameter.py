"""Phase 5 — hyperparameter protocol (§19).

The rule (Principle 4): do NOT pick the best params per test stream. Instead
select params on *calibration* streams, freeze them, then evaluate on disjoint
*test* streams. This module implements a small grid search on the calibration
set and returns frozen params ready to feed into ``BenchmarkConfig.overrides``.

Run directly::

    python -m benchmark.experiments.hyperparameter
"""

from __future__ import annotations

import itertools

import numpy as np

from ..algorithms.registry import get_algorithm
from ..evaluation.runner import run_experiment
from .harness import _make_stream


def _score_on_streams(alg: str, params: dict, streams, window_size: int, metric: str) -> float:
    """Mean quality of ``alg(params)`` averaged over the calibration streams."""
    scores = []
    for scenario, seed, n_samples in streams:
        stream, gt = _make_stream(scenario, seed, n_samples)
        adapter = get_algorithm(alg, **params)
        res = run_experiment(adapter, stream, gt, window_size=window_size, metrics=(metric,))
        _, ys = res.quality_curve(metric)
        if ys:
            scores.append(float(np.mean(ys)))
    return float(np.mean(scores)) if scores else float("-inf")


def grid_search(
    alg: str,
    grid: dict[str, list],
    calibration_streams,
    window_size: int = 1000,
    metric: str = "ari",
) -> tuple[dict, float]:
    """Exhaustive grid search over ``grid`` on the calibration streams (§19).

    Returns ``(best_params, best_score)``. ``calibration_streams`` is a list of
    ``(scenario, seed, n_samples)`` tuples.
    """
    keys = list(grid)
    best_params: dict = {}
    best_score = float("-inf")
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        score = _score_on_streams(alg, params, calibration_streams, window_size, metric)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score


#: default calibration grids (small, illustrative — extend as needed)
DEFAULT_GRIDS: dict[str, dict[str, list]] = {
    "DBSTREAM": {"clustering_threshold": [1.0, 2.0, 3.0], "fading_factor": [0.01, 0.05]},
    "DenStream": {"epsilon": [3.0, 4.0, 5.0], "mu": [2.5, 5.0]},
    "CluStream": {"max_micro_clusters": [30, 50], "n_macro_clusters": [3]},
    "STREAMKMeans": {"n_clusters": [2, 3, 4], "chunk_size": [50]},
}


def calibrate(
    algorithms=("DBSTREAM", "DenStream", "CluStream", "STREAMKMeans"),
    calibration_streams=None,
    window_size: int = 1000,
    metric: str = "ari",
) -> dict[str, dict]:
    """Calibrate every algorithm and return frozen overrides per algorithm."""
    if calibration_streams is None:
        # calibration split: S1..S3 (Principle 4 — disjoint from test S4..S7)
        calibration_streams = [
            ("S1_stationary", 1, 8000),
            ("S2_abrupt_movement", 1, 8000),
            ("S3_gradual_movement", 1, 8000),
        ]
    frozen = {}
    for alg in algorithms:
        grid = DEFAULT_GRIDS.get(alg, {})
        if not grid:
            frozen[alg] = {}
            continue
        params, score = grid_search(alg, grid, calibration_streams, window_size, metric)
        frozen[alg] = params
        print(f"{alg:14s} best {metric}={score:.4f} params={params}")
    return frozen


def main() -> None:
    print("# Hyperparameter calibration (§19)\n")
    print("Calibration streams: S1, S2, S3  |  Test streams: S4..S7\n")
    frozen = calibrate()
    print("\nFrozen params (feed into BenchmarkConfig.overrides):")
    print(frozen)


if __name__ == "__main__":
    main()
