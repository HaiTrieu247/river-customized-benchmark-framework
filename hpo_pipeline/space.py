"""Stage 3 -- SPACE: a scale-free candidate set, stratified on resolution.

In clustering one parameter dominates the metric in a way that has no analogue
in classification: get the resolution wrong and no other dimension rescues the
run. Spending a uniform Sobol budget as though every dimension were equal is
therefore the wrong design. Instead the space is a **full 8 x 8 factorial**:

    Lambda_a = { (g_i, theta_j) : i = 1..8, j = 1..8 }      64 per algorithm

``g`` walks the resolution axis and ``theta`` indexes 8 Sobol points over the
remaining shape dimensions. Every resolution level is tried with all 8 shape
points and the other way round, so each shape point carries a complete
resolution response curve -- which is exactly what Stage 6's P2R reads.

Direction convention, and the single easiest thing to get wrong here:

    g = 1  ->  COARSEST  (fewest clusters)
    g = 8  ->  FINEST    (most clusters)

For a count parameter the value rises with ``g``. For a distance threshold it
**falls**: a smaller threshold is a tighter one, which yields more clusters and
so is *finer*. The `c` and `alpha` ladders below are therefore written in
coarse-to-fine order rather than in ascending numeric order. Get this wrong and
``R_parsimony`` in Stage 6, which picks "the coarsest level still acceptable",
picks backwards on exactly half the pool.

Gate: 64 configs per algorithm, every one resolvable on every fold, and the
inter-parameter constraints satisfied by rejection at generation time rather
than by clipping at run time -- clipping would make ``config_id`` describe
something other than what ran.
"""

from __future__ import annotations

import argparse

import numpy as np

from ._common import (
    N_GRANULARITY,
    N_SHAPE,
    PROBE_JSON,
    SCALER,
    SCALER_PARAMS,
    SPACE_JSON,
    SPACE_SEED,
    config_id,
    read_json,
    require,
    write_json,
)

#: algorithm -> (parameter carrying the resolution axis, how it is resolved)
GRAN: dict[str, tuple[str, str]] = {
    "CluStream": ("n_macro_clusters", "count"),
    "KMeans": ("n_clusters", "count"),
    "STREAMKMeans": ("n_clusters", "count"),
    "DenStream": ("epsilon", "scaled"),
    "DBSTREAM": ("clustering_threshold", "scaled"),
    "SOStream": ("merge_threshold", "scaled"),
    "BIRCH": ("threshold", "scaled"),
    "ImprovedBIRCH": ("initial_threshold", "scaled"),
    # alpha already divides by each representative's own local density, so it is
    # dimensionless as it stands and needs no s(D).
    "RepStream": ("alpha", "relative"),
}

#: Coarse -> fine. Ascending, because more clusters is finer.
COUNTS = [2, 3, 5, 9, 14, 23, 37, 60]
#: Coarse -> fine. DESCENDING, because a smaller threshold is a tighter one.
CSCALE = [2.00, 1.18, 0.70, 0.41, 0.24, 0.14, 0.085, 0.05]
#: Coarse -> fine. DESCENDING: a larger alpha satisfies the density-related
#: condition more easily, giving more links, fewer components, a coarser result.
#: This one is inferred rather than read off, so Stage 5 check 8 verifies it
#: against the data instead of trusting the reasoning.
ALPHAS = [3.00, 2.70, 2.45, 2.10, 1.80, 1.50, 1.25, 1.05]


# --- one-dimensional maps from [0,1) ---------------------------------------
def _lu(lo, hi):
    """Log-uniform: a multiplicative-scale parameter sampled on its own scale."""
    return lambda u: float(lo * (hi / lo) ** u)


def _u(lo, hi):
    return lambda u: float(lo + (hi - lo) * u)


def _li(lo, hi):
    return lambda u: int(round(lo * (hi / lo) ** u))


def _choice(values):
    return lambda u: values[min(int(u * len(values)), len(values) - 1)]


#: The SHAPE dimensions only. The resolution parameter of each algorithm is
#: swept over its 8 fixed levels and never drawn from here.
SHAPE: dict[str, dict] = {
    "CluStream": {
        "max_micro_clusters": _li(25, 800),
        "time_gap": _li(50, 5000),
    },
    "KMeans": {
        "halflife": _lu(0.01, 0.9),
    },
    "STREAMKMeans": {
        "chunk_size": _li(16, 1024),
    },
    "BIRCH": {
        "branching_factor": _choice([10, 25, 50, 100]),
        "max_entries": _li(100, 4000),
    },
    "ImprovedBIRCH": {
        "modifying_factor": _u(1.1, 4.0),
        "branching_factor": _choice([10, 25, 50, 100]),
        "max_entries": _li(100, 4000),
    },
    "DenStream": {
        "mu": _lu(1.5, 12),
        "beta": _u(0.2, 1.0),
        "decaying_factor": _lu(0.05, 1.0),
        "n_samples_init": _choice([200, 500, 1000, 2000]),
    },
    "DBSTREAM": {
        "fading_factor": _lu(0.001, 0.1),
        "intersection_factor": _u(0.05, 0.6),
        "minimum_weight": _lu(0.5, 8),
        "cleanup_interval": _li(2, 100),
    },
    "SOStream": {
        "alpha": _u(0.02, 0.5),
        "min_pts": _li(2, 12),
        "fading_factor": _lu(0.001, 0.1),
    },
    "RepStream": {
        # The k of the mutual-kNN graph. NOT a cluster count.
        "k": _li(3, 20),
        "decay_rate": _lu(0.9, 0.9999),
        "max_vertices": _li(50, 1000),
        "repository_fraction": _u(0.2, 0.8),
    },
}


def _accepts(algo: str, shape: dict) -> bool:
    """Inter-parameter constraints, checked against the whole resolution ladder.

    A shape point is shared by all 8 resolution levels, so it is only admissible
    if it holds at every one of them -- for CluStream that means the micro
    budget must cover the largest macro count on the ladder.
    """
    if algo == "CluStream":
        return shape["max_micro_clusters"] >= max(COUNTS)
    return True


def shape_points(algo: str, n: int = N_SHAPE, seed: int = SPACE_SEED) -> list[dict]:
    """``n`` Sobol points over the shape dimensions, rejecting invalid ones."""
    from scipy.stats import qmc

    dims = sorted(SHAPE[algo])
    if not dims:
        return [{} for _ in range(n)]

    engine = qmc.Sobol(d=len(dims), scramble=True, seed=seed)
    out: list[dict] = []
    # Draw in powers of two so Sobol keeps its balance properties, and take the
    # first n points that satisfy the constraints.
    drawn = engine.random(max(n, 16))
    while True:
        for row in drawn:
            point = {d: SHAPE[algo][d](u) for d, u in zip(dims, row)}
            if _accepts(algo, point):
                out.append(point)
                if len(out) == n:
                    return out
        drawn = engine.random(max(n, 16))


def granularity_value(algo: str, level: int) -> float:
    """The dimensionless value on the resolution axis at ``level`` (1..8)."""
    kind = GRAN[algo][1]
    if kind == "count":
        return float(COUNTS[level - 1])
    if kind == "scaled":
        return float(CSCALE[level - 1])
    return float(ALPHAS[level - 1])


def build_space(algos: list[str]) -> dict:
    """The full candidate set: 8 x 8 per algorithm, plus River's own default."""
    from benchmark.algorithms.registry import ALGORITHMS

    space: dict[str, list[dict]] = {}
    for algo in algos:
        shapes = shape_points(algo)
        candidates = []
        for level in range(1, N_GRANULARITY + 1):
            g = granularity_value(algo, level)
            for theta, shape in enumerate(shapes, start=1):
                candidates.append(
                    {
                        "algo": algo,
                        "granularity_level": level,
                        "theta": theta,
                        "g": g,
                        "shape": shape,
                        "config_id": config_id(algo, level, g, shape),
                        "scaler": SCALER,
                        "scaler_params": dict(SCALER_PARAMS),
                        "is_default": False,
                    }
                )

        # River's constructor default enters Lambda_a as one more candidate, so
        # P0 is read from the same matrix under the same seeds and the same
        # measurement as everything else. Run it down a separate path and the
        # P1 - P0 gap would also contain differences in how it was measured.
        _, defaults = ALGORITHMS[algo].factory()
        candidates.append(
            {
                "algo": algo,
                "granularity_level": None,
                "theta": None,
                "g": None,
                "shape": dict(defaults),
                "config_id": config_id(algo, None, None, dict(defaults)),
                "scaler": SCALER,
                "scaler_params": dict(SCALER_PARAMS),
                "is_default": True,
            }
        )
        space[algo] = candidates
    return space


def resolve(candidate: dict, probe_of_fold: dict) -> dict:
    """Dimensionless spec -> absolute params for ONE fold."""
    algo = candidate["algo"]
    if candidate.get("is_default"):
        return dict(candidate["shape"])

    name, kind = GRAN[algo]
    params = dict(candidate["shape"])
    g = candidate["g"]
    params[name] = g * probe_of_fold["s"] if kind == "scaled" else g
    if kind == "count":
        params[name] = int(round(params[name]))
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", default="")
    args = parser.parse_args()

    require(PROBE_JSON, "Stage 2 (probe.py)")
    probes = read_json(PROBE_JSON)

    algos = (
        [a.strip() for a in args.algos.split(",") if a.strip()]
        if args.algos
        else list(GRAN)
    )
    space = build_space(algos)
    write_json(SPACE_JSON, space)

    failures = []
    print(f"{'algorithm':15s} {'shape dims':>10} {'configs':>8} {'per dim':>8}")
    for algo, candidates in space.items():
        tuned = [c for c in candidates if not c["is_default"]]
        dims = len(SHAPE[algo])
        if len(tuned) != N_GRANULARITY * N_SHAPE:
            failures.append(f"{algo}: {len(tuned)} tuned configs, expected 64")
        ids = {c["config_id"] for c in candidates}
        if len(ids) != len(candidates):
            failures.append(f"{algo}: config_id collision")
        for candidate in candidates:
            for fold, entry in probes.items():
                try:
                    resolve(candidate, entry)
                except Exception as exc:
                    failures.append(f"{algo} on {fold}: {type(exc).__name__}: {exc}")
        # Coverage density is a real limit of the equal-budget design, so it is
        # printed rather than hidden: read P1 as "the ceiling at 64 trials".
        print(
            f"{algo:15s} {dims:>10} {len(tuned):>8} "
            f"{(N_SHAPE / dims if dims else float('nan')):>8.2f}"
        )

    print(f"wrote {SPACE_JSON}")
    if failures:
        for line in failures:
            print(f"  FAIL {line}")
        raise SystemExit("GATE FAILED")
    print("GATE PASSED")


if __name__ == "__main__":
    main()
