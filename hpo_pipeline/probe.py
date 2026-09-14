"""Stage 2 -- PROBE: measure each fold's raw distance scale.

With no scaler, ``epsilon = 0.5`` means nothing: its unit is the unit of
whichever feature dominates, and that differs by orders of magnitude between
folds. ``s(D)`` is the only thing that makes a distance threshold comparable
across datasets, which makes this stage load-bearing rather than optional.

``s(D)`` is the median distance to the 4th nearest neighbour, measured on a
prefix of the *shuffled* stream. After shuffling a prefix is a representative
sample, so the statistic is both cheap and unbiased. A prefix rather than the
whole stream is also what a real system has when it must pick parameters, so a
parameterisation built on it is one that can actually be deployed.

The stage also measures how far the raw-units decision goes: with one feature
holding almost all the variance, every distance-based method is solving a
one-dimensional problem, and that has to be published next to the scores rather
than discovered by a reader.

Gate: ``s > 0`` and finite on every fold, and the probe sees at least 80% of the
classes -- which is the evidence that shuffling worked.
"""

from __future__ import annotations

import argparse

import numpy as np

from ._common import FOLDS, PROBE_JSON, SEEDS, shuffle_seed_for, write_json
from .stream import load_arrays, stats_of

K_NN = 4
N_PROBE = 5000
SUBSAMPLE = 2000


def probe(X, y, k_nn: int = K_NN, n_probe: int = N_PROBE, seed: int = 0) -> dict:
    """Raw-scale statistics of one fold, from a prefix of the shuffled stream."""
    Xp = np.asarray(X[:n_probe], dtype=float)
    if len(Xp) < 2:
        raise SystemExit("probe needs at least 2 samples")

    var = Xp.var(axis=0)
    total = float(var.sum()) or 1.0

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xp), size=min(SUBSAMPLE, len(Xp)), replace=False)
    Xs = Xp[idx]
    sq = (Xs**2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Xs @ Xs.T, 0.0))
    np.fill_diagonal(D, np.inf)
    knn = np.sort(D, axis=1)[:, k_nn - 1]
    offdiag = D[np.isfinite(D)]

    labels = [v for v in y[:n_probe] if v is not None]
    return {
        "d": int(Xp.shape[1]),
        "n_probe": int(len(Xp)),
        # The unit every distance threshold is expressed in.
        "s": float(np.median(knn)),
        "knn_q05": float(np.quantile(knn, 0.05)),
        "knn_q95": float(np.quantile(knn, 0.95)),
        "knn_spread": float(np.quantile(knn, 0.95) / max(np.quantile(knn, 0.05), 1e-12)),
        "pairwise_median": float(np.median(offdiag)),
        # How far one feature dominates the distance. Mandatory alongside every
        # score table in raw-units mode.
        "top1_variance_share": float(var.max() / total),
        "top3_variance_share": float(np.sort(var)[-3:].sum() / total),
        # Participation ratio: equals d when the variance is spread evenly and
        # 1 when a single feature swallows everything.
        "effective_dim": float(total**2 / float((var**2).sum() or 1.0)),
        "n_classes_probe": len(set(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    args = parser.parse_args()

    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    shuffle_seed = shuffle_seed_for(args.seed)

    out: dict[str, dict] = {}
    failures: list[str] = []
    print(
        f"{'fold':16s} {'s(D)':>12} {'knn_spread':>10} {'top1_var':>9} "
        f"{'eff_dim':>8} {'k_probe':>7} {'k_total':>7}"
    )
    for fold in folds:
        X, y = load_arrays(fold, shuffle_seed)
        stats = stats_of(fold)
        entry = probe(X, y, seed=args.seed)
        entry["fold"] = fold
        entry["shuffle_seed"] = shuffle_seed
        entry["n_classes_total"] = stats["n_classes"]
        entry["n_samples"] = stats["n_samples"]
        # A property of the data in raw-units mode, not a defect -- but on such
        # a fold "A beats B" only ever describes behaviour along one axis.
        entry["one_d_dominated"] = bool(entry["top1_variance_share"] > 0.9)
        out[fold] = entry

        print(
            f"{fold:16s} {entry['s']:12.5g} {entry['knn_spread']:10.2f} "
            f"{entry['top1_variance_share']:9.3f} {entry['effective_dim']:8.2f} "
            f"{entry['n_classes_probe']:7d} {stats['n_classes']:7d}"
            + ("   <- 1-D dominated" if entry["one_d_dominated"] else "")
        )

        if not np.isfinite(entry["s"]) or entry["s"] <= 0:
            failures.append(f"{fold}: s={entry['s']} is not a usable unit")
        if entry["n_classes_probe"] < 0.8 * stats["n_classes"]:
            failures.append(
                f"{fold}: probe saw {entry['n_classes_probe']} of {stats['n_classes']} "
                "classes -- the shuffle did not work"
            )

    write_json(PROBE_JSON, out)
    print(f"wrote {PROBE_JSON}")

    if failures:
        for line in failures:
            print(f"  FAIL {line}")
        raise SystemExit("GATE FAILED")
    print("GATE PASSED")


if __name__ == "__main__":
    main()
