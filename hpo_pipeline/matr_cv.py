"""Stage 7b -- P4a: MATR + MATR-CV, choosing both axes without labels.

Fan et al. give two algorithms doing two different jobs, and this stage uses
both:

    inner loop (MATR, Alg. 1)    -> theta*, the SHAPE, by a plain argmax
    outer loop (MATR-CV, Alg. 2) -> g_hat, the RESOLUTION, by the parsimony rule

Why one axis needs a ``Delta`` and the other does not is not a convention, it is
the asymmetry in the assumptions of their Theorem 6: under-estimating the
cluster count loses trace outright, while over-estimating it can still *score
higher* than the truth. So a plain argmax on the resolution axis always splits
too finely -- which is exactly why Alg. 2 exists -- whereas on the shape axis
the plain argmax is correct (Theorem 1).

The split is interleaved by index rather than cut in time. After shuffling any
cut gives two samples from one distribution, but keeping the interleaved form
means the same code also works on a non-shuffled branch.

Restricted to the centroid-based family. For the density-based methods the weak
assortativity assumption does not hold, and the result would describe MATR
rather than the algorithm.

One caveat that must travel with the output: in raw units the WCSS is swallowed
by the dominant feature just as ARI is, so on a 1-D dominated fold this stage
picks a resolution for the structure of a single axis.
"""

from __future__ import annotations

import argparse

import numpy as np

from ._common import (
    FOLDS,
    MATRCV_JSONL,
    PROBE_JSON,
    SEEDS,
    SPACE_JSON,
    append_jsonl,
    read_json,
    require,
    shuffle_seed_for,
)

#: Centroid-based only, on purpose.
CENTROID_FAMILY = (
    "CluStream",
    "KMeans",
    "STREAMKMeans",
    "BIRCH",
    "ImprovedBIRCH",
    "SOStream",
)

N_SPLITS = 5
HOLDOUT_EVERY = 10
#: Reported as a sweep, because if g_hat jumps around with Delta then Delta is a
#: hyper-hyperparameter and that has to be said out loud.
DELTAS = (0.01, 0.02, 0.05, 0.10, 0.20)


def trace_on_holdout(algo, params, fold, seed, residue, n_samples, scaler, scaler_params):
    """Learn on Q1, score the trace on Q2.

    The similarity measure is the raw squared distance and never depends on the
    parameter being tuned; only the partition changes with the configuration.
    Letting the criterion depend on the tuned parameter would make it
    self-referential and void the guarantees.
    """
    from benchmark.algorithms.registry import get_algorithm
    from benchmark.evaluation.clustering_quality import StreamingMATRTrace

    from .stream import shuffled_stream

    adapter = get_algorithm(
        algo, scaler=scaler, scaler_params=scaler_params, **params
    )
    trace = StreamingMATRTrace()
    dataset = shuffled_stream(fold, shuffle_seed_for(seed))

    for i, (x, _) in enumerate(dataset):
        if n_samples is not None and i >= n_samples:
            break
        if i % HOLDOUT_EVERY == residue:
            y_pred = adapter.predict_one(adapter._scale(x))
            if y_pred is not None:
                trace.update(x, y_pred)
        else:
            adapter.process_one(x)
    return trace.get()


def parsimony(values, delta_frac):
    """min{ g : l_g >= max l - Delta }, with Delta a fraction of the range."""
    values = np.asarray(values, dtype=float)
    span = float(values.max() - values.min())
    if span <= 0:
        return 1
    keep = np.flatnonzero(values >= values.max() - delta_frac * span)
    return int(keep.min()) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", default=",".join(CENTROID_FAMILY))
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    require(SPACE_JSON, "Stage 3 (space.py)")
    require(PROBE_JSON, "Stage 2 (probe.py)")
    space = read_json(SPACE_JSON)
    probes = read_json(PROBE_JSON)

    from ._common import load_matrix
    from .space import resolve

    matrix = load_matrix()
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]

    for algo in algos:
        if algo not in space:
            continue
        candidates = [c for c in space[algo] if not c["is_default"]]
        for fold in folds:
            # --- inner loop: MATR picks the shape, by plain argmax on the
            # trace already sitting in the matrix.
            sub = matrix[
                (matrix.algo == algo)
                & (matrix.dataset == fold)
                & (matrix.status == "ok")
                & matrix.feasible
                & matrix.theta.notna()
            ]
            if sub.empty or "matr_trace_win" not in sub.columns:
                continue
            by_theta = sub.groupby("theta")["matr_trace_win"].mean().dropna()
            if by_theta.empty:
                continue
            theta_star = int(by_theta.idxmax())

            # --- outer loop: MATR-CV picks the resolution
            levels = sorted({int(c["granularity_level"]) for c in candidates})
            curves = np.full((len(levels), args.splits), np.nan)
            for j in range(args.splits):
                for li, level in enumerate(levels):
                    candidate = next(
                        c
                        for c in candidates
                        if int(c["granularity_level"]) == level
                        and int(c["theta"]) == theta_star
                    )
                    params = resolve(candidate, probes[fold])
                    try:
                        curves[li, j] = trace_on_holdout(
                            algo,
                            params,
                            fold,
                            args.seed,
                            residue=j % HOLDOUT_EVERY,
                            n_samples=args.n_samples,
                            scaler=candidate["scaler"],
                            scaler_params=candidate["scaler_params"],
                        )
                    except Exception:
                        continue

            usable = ~np.isnan(curves).all(axis=0)
            if not usable.any():
                continue

            row = {
                "algo": algo,
                "dataset": fold,
                "seed": args.seed,
                "theta_star": theta_star,
                "n_samples": args.n_samples,
                "levels": levels,
                "trace_curves": curves.tolist(),
            }
            for delta in DELTAS:
                picks = [
                    parsimony(curves[:, j], delta)
                    for j in range(args.splits)
                    if not np.isnan(curves[:, j]).all()
                ]
                row[f"g_hat_delta_{delta}"] = int(np.median(picks)) if picks else None
            # The headline value, with the sweep kept beside it.
            row["g_hat"] = row.get("g_hat_delta_0.05")
            row["g_hat_argmax"] = (
                int(np.nanargmax(np.nanmean(curves, axis=1)) + 1)
                if usable.any()
                else None
            )
            append_jsonl(MATRCV_JSONL, row)
            print(
                f"{algo:14s} {fold:16s} theta*={theta_star} "
                f"g_hat={row['g_hat']} argmax={row['g_hat_argmax']} "
                f"sweep={[row.get(f'g_hat_delta_{d}') for d in DELTAS]}"
            )

    print(f"wrote {MATRCV_JSONL}")


if __name__ == "__main__":
    main()
