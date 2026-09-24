"""Stage 6 -- VIEWS: every protocol as a reading of the matrix. No compute.

P0 through P5 differ only in *which cell of the matrix they are allowed to
look at*, which is the entire point of building one matrix and reading it many
times. P2R adds the readings that transfer a rule instead of a value.

This stage refuses to run until Stage 5 has passed, because a number read off an
unhealthy matrix is worse than no number.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ._common import (
    CRITERIA,
    FOLD_GROUPS,
    FOLDS,
    HEALTH_MD,
    MATRIX_JSONL,
    VIEWS_DIR,
    load_matrix,
    require,
)
from .rules import RULES, orient

OBJ = "ari"


def _ok(df):
    return df[(df.status == "ok") & df.feasible]


def _mbar(df):
    """M-bar[algo, config_id, fold]: the objective, averaged over seeds."""
    return _ok(df).groupby(["algo", "config_id", "dataset"])[OBJ].mean()


def panel(df):
    """[algo, theta, granularity_level, fold] -> objective + every criterion.

    This is the shape P2R needs: one row per point of a resolution curve.
    """
    sub = _ok(df)
    sub = sub[sub.granularity_level.notna() & sub.theta.notna()]
    agg = {OBJ: (OBJ, "mean")}
    for criterion in CRITERIA:
        if criterion in sub.columns:
            agg[criterion] = (criterion, "mean")
    return sub.groupby(["algo", "theta", "granularity_level", "dataset"]).agg(**agg)


# --- P0 .. P5 --------------------------------------------------------------
def p0(df):
    """River's constructor default, run through the identical measurement."""
    defaults = df[df.is_default.fillna(False).astype(bool)]
    if defaults.empty:
        return pd.Series(dtype=float)
    return _mbar(defaults).groupby(["algo", "dataset"]).max()


def p1(df):
    """Oracle: argmax on the test fold itself. A ceiling, and NOT fair."""
    return _mbar(df).groupby(["algo", "dataset"]).max()


def p1_holdout(df):
    """Oracle, but selected on seeds {1,2} and scored on seed {3}.

    This removes the winner's curse (picking a lucky draw) without removing the
    oracle-ness, which is a different bias. Because each seed also carries its
    own stream permutation, the gap it measures includes the variance due to
    stream order.
    """
    selection = _mbar(df[df.seed.isin([1, 2])])
    scoring = _mbar(df[df.seed == 3])
    out = {}
    for (algo, fold), grp in selection.groupby(level=["algo", "dataset"]):
        if grp.empty or grp.isna().all():
            continue
        best_config = grp.idxmax()[1]
        out[(algo, fold)] = scoring.get((algo, best_config, fold), np.nan)
    return pd.Series(out)


def p2(df, folds=FOLDS, groups=None, normalize_by=None):
    """Leave-one-dataset-out. ``groups`` turns it into leave-one-GROUP-out.

    The synthetic folds share a generator and are not independent of one
    another, so plain LODO over all nine is optimistic. Holding out a whole
    group is the honest transfer number.

    ``normalize_by`` (a P1 series) rescales each fold before averaging;
    otherwise an easy fold dominates the choice.
    """
    m = _mbar(df)
    key = (lambda d: groups[d]) if groups else (lambda d: d)
    out = {}
    for algo in m.index.get_level_values("algo").unique():
        ma = m.xs(algo, level="algo")
        for held in {key(d) for d in folds}:
            opt = [d for d in folds if key(d) != held]
            sub = ma[ma.index.get_level_values("dataset").isin(opt)]
            if sub.empty:
                continue
            if normalize_by is not None:
                denom = normalize_by.reindex(
                    [(algo, d) for d in sub.index.get_level_values("dataset")]
                ).to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    sub = sub / denom
            # Eligible only if feasible on EVERY optimisation fold. The test
            # fold is never consulted.
            counts = sub.groupby("config_id").size()
            eligible = counts[counts == len(opt)].index
            sub = sub[sub.index.get_level_values("config_id").isin(eligible)]
            if sub.empty:
                continue
            best = sub.groupby("config_id").mean().idxmax()
            for fold in [f for f in folds if key(f) == held]:
                out[(algo, fold)] = ma.get((best, fold), np.nan)
    return pd.Series(out)


def p5(df):
    """Tuned default: the single best configuration across every fold.

    This is a *product* -- a proposed new default -- not an evaluation
    protocol. Its unbiased score is p2(). Note it is dimensionless: proposing it
    upstream means proposing the rule for computing s(D) alongside it, or fixing
    an absolute value and saying which scale it is tied to.
    """
    m = _mbar(df)
    out = {}
    for algo in m.index.get_level_values("algo").unique():
        ma = m.xs(algo, level="algo")
        by_config = ma.groupby("config_id").mean()
        if by_config.empty or by_config.isna().all():
            continue
        best = by_config.idxmax()
        out[algo] = {
            "config_id": best,
            "mean": float(by_config.max()),
            "per_fold": ma.xs(best, level="config_id").to_dict(),
        }
    return out


# --- P2R -------------------------------------------------------------------
def _curve(pan, algo, theta, fold, criterion):
    """The 8 criterion values along g, seed-averaged and sign-corrected."""
    try:
        sub = pan.xs((algo, theta), level=("algo", "theta"))
    except KeyError:
        return None
    try:
        sub = sub.xs(fold, level="dataset")
    except KeyError:
        return None
    if criterion not in sub.columns:
        return None
    series = sub[criterion].groupby("granularity_level").mean().reindex(range(1, 9))
    values = series.to_numpy(dtype=float)
    # A missing cell is a broken curve, not a shorter one -- and `inf` counts as
    # missing here. River's streaming silhouette returns `inf` when a cluster
    # has no neighbour to compare against, which `isna` does not catch; left in,
    # it turns the curve's range into NaN and every rule reading it into noise.
    if not np.isfinite(values).all():
        return None
    return orient(values, criterion)


def _score(pan, algo, theta, level, fold):
    try:
        return float(pan.loc[(algo, theta, float(level), fold), OBJ])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _best_over_g(pan, algo, theta, fold):
    try:
        sub = pan.xs((algo, theta), level=("algo", "theta")).xs(fold, level="dataset")
    except KeyError:
        return np.nan
    return float(sub[OBJ].max()) if not sub.empty else np.nan


def p4(pan, matrcv_rows, delta_key: str = "g_hat") -> pd.Series:
    """P4a: the cell MATR + MATR-CV chose without ever seeing a label.

    Stage 7b picks both axes label-free -- ``theta_star`` by MATR (Alg. 1) and
    the resolution level by MATR-CV (Alg. 2) -- and writes them to
    ``matrcv.jsonl``. This reads that file and scores the cell it names out of
    the same matrix every other protocol is read from, so ``P1 - P4`` is a
    comparison of two readings and not of two measurement paths.

    Scoring with ARI is the point, not a contradiction: the tuner is blind, the
    evaluator is not. Grading a label-free choice with a label-free criterion
    would only show that the criterion agrees with itself. Fan et al. do the
    same -- MATR tunes without labels, and the result is reported in NMI.

    ``delta_key`` selects which column of the parsimony sweep to read
    (``g_hat`` is the headline delta; ``g_hat_argmax`` is plain argmax, the
    strawman Theorem 6 predicts will over-split).
    """
    out = {}
    for row in matrcv_rows:
        level = row.get(delta_key)
        if level is None:
            continue
        out[(row["algo"], row["dataset"])] = _score(
            pan, row["algo"], row["theta_star"], level, row["dataset"]
        )
    return pd.Series(out, dtype=float)


def p2_rule(pan, folds, rule_name, criterion, groups=None):
    """P2R: transfer the shape by LODO, re-pick the resolution by rule R."""
    rule, phis = RULES[rule_name]
    key = (lambda d: groups[d]) if groups else (lambda d: d)
    out = {}
    algos = pan.index.get_level_values("algo").unique()
    thetas = sorted(pan.index.get_level_values("theta").unique())

    for algo in algos:
        for held in {key(d) for d in folds}:
            opt = [d for d in folds if key(d) != held]

            # (1) shape: ordinary LODO on the best-over-g score
            def shape_quality(theta):
                vals = [_best_over_g(pan, algo, theta, d) for d in opt]
                vals = [v for v in vals if np.isfinite(v)]
                return float(np.mean(vals)) if vals else -np.inf

            theta_star = max(thetas, key=shape_quality)
            if not np.isfinite(shape_quality(theta_star)):
                continue

            # (2) the rule's own parameter: also LODO
            def rule_quality(phi):
                vals = []
                for d in opt:
                    curve = _curve(pan, algo, theta_star, d, criterion)
                    if curve is None:
                        continue
                    picked = rule(curve, phi) + 1
                    vals.append(_score(pan, algo, theta_star, picked, d))
                vals = [v for v in vals if np.isfinite(v)]
                return float(np.mean(vals)) if vals else -np.inf

            phi_star = max(phis, key=rule_quality)

            # (3) apply the rule on the HELD-OUT fold, label-free only
            for fold in [f for f in folds if key(f) == held]:
                curve = _curve(pan, algo, theta_star, fold, criterion)
                out[(algo, fold)] = (
                    np.nan
                    if curve is None
                    else _score(pan, algo, theta_star, rule(curve, phi_star) + 1, fold)
                )
    return pd.Series(out, dtype=float)


# --- output ----------------------------------------------------------------
def _table(series: pd.Series) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame()
    frame = series.copy()
    frame.index = pd.MultiIndex.from_tuples(list(frame.index), names=["algo", "dataset"])
    return frame.unstack("dataset")


def _health_ok() -> bool:
    if not HEALTH_MD.exists():
        return False
    return "**VERDICT: PASS**" in HEALTH_MD.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="read the matrix even though Stage 5 has not passed (smoke tests)",
    )
    args = parser.parse_args()

    require(MATRIX_JSONL, "Stage 4 (build_matrix.py)")
    if not _health_ok() and not args.force:
        raise SystemExit(
            "Stage 5 has not passed -- run health.py first, or pass --force "
            "if you are knowingly inspecting an incomplete matrix"
        )

    df = load_matrix()
    folds = [f for f in FOLDS if f in set(df.dataset)]
    VIEWS_DIR.mkdir(parents=True, exist_ok=True)

    ceiling = p1(df)
    written = []
    for name, series in [
        ("p0", p0(df)),
        ("p1", ceiling),
        ("p1_holdout", p1_holdout(df)),
        ("p2", p2(df, folds)),
        ("p2_normalised", p2(df, folds, normalize_by=ceiling)),
        ("p2_cross_group", p2(df, folds, groups=FOLD_GROUPS)),
    ]:
        table = _table(series)
        path = VIEWS_DIR / f"{name}.csv"
        table.to_csv(path)
        written.append(path.name)
        print(f"\n== {name} ==")
        print(table.round(4).to_string() if not table.empty else "(empty)")

    pan = panel(df)
    for rule_name in RULES:
        for criterion in CRITERIA:
            series = p2_rule(pan, folds, rule_name, criterion, groups=None)
            table = _table(series)
            if table.empty:
                continue
            path = VIEWS_DIR / f"p2r_{rule_name}_{criterion}.csv"
            table.to_csv(path)
            written.append(path.name)

    tuned = p5(df)
    pd.DataFrame(
        [{"algo": a, "config_id": v["config_id"], "mean": v["mean"]} for a, v in tuned.items()]
    ).to_csv(VIEWS_DIR / "p5.csv", index=False)
    written.append("p5.csv")

    print(f"\nwrote {len(written)} tables to {VIEWS_DIR}")


if __name__ == "__main__":
    main()
