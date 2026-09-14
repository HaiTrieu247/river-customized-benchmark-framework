"""Stage 8 -- ANALYZE: the gaps, the rule table, the budget curve, the tests.

Five distances, each normalised by P1:

    P1 - P0            how arbitrary River's default is
    P1 - P1_holdout    winner's curse along the seed / stream-order axis
    P1 - P3(40%)       winner's curse along the data axis, plus the oracle gap
    P1 - P2            the part that does not transfer between datasets
    P1 - P4            the price of having no labels

Two analyses here are not decoration.

The **budget curve** turns a bias into a measurement. Equal candidate counts are
equal *budget*, not equal *coverage density*: with one shape dimension KMeans
gets 8 samples per dimension and with four DenStream gets 2. Plotting the
expected best-of-b against b says how much each algorithm is still losing to a
finite budget, and makes the claim precise -- "at a budget of 64 trials, A beats
B", not "A is better than B".

The **variance decomposition** is a test of P2R's own assumption. An 8x8 full
factorial splits the variance into the resolution axis, the shape axis and their
interaction. A large interaction means the best shape depends on the resolution,
which undermines "transfer theta*, then re-pick g" -- so it must be checked
before the P2R table is read, not after.

A warning that has to be printed rather than glossed: with 9 algorithms and 9
folds the Nemenyi critical difference is about 4 ranks on a 9-rank scale, so
almost no pair separates statistically. Effect sizes with bootstrap intervals
are the honest reading at this D.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from ._common import (
    FOLD_GROUPS,
    FOLDS,
    MATRIX_JSONL,
    PREFIX_JSONL,
    REPORT_DIR,
    VIEWS_DIR,
    load_matrix,
    read_jsonl,
    require,
)
from .rules import RULES
from .views import _table, p0, p1, p1_holdout, p2, p2_rule, panel

#: Studentised range values at alpha=0.05 for the Nemenyi test, by k.
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}

BUDGETS = (4, 8, 16, 32, 64)


def critical_difference(k: int, d: int) -> float:
    """Nemenyi CD. Prints wide at small d, which is the point."""
    q = _Q05.get(k)
    if q is None or d == 0:
        return float("nan")
    return float(q * np.sqrt(k * (k + 1) / (6.0 * d)))


def p1_at_budget(scores, b, n_draws=2000, rng=None):
    """Expected best score when only ``b`` of the candidates may be tried."""
    scores = np.asarray([s for s in scores if np.isfinite(s)], dtype=float)
    if len(scores) == 0:
        return float("nan")
    if b >= len(scores):
        return float(scores.max())
    rng = rng or np.random.default_rng(0)
    return float(
        np.mean([scores[rng.choice(len(scores), b, replace=False)].max() for _ in range(n_draws)])
    )


def budget_curve(df) -> pd.DataFrame:
    ok = df[(df.status == "ok") & df.feasible]
    per_config = ok.groupby(["algo", "dataset", "config_id"])["ari"].mean()
    rows = []
    for (algo, fold), grp in per_config.groupby(level=["algo", "dataset"]):
        scores = grp.to_numpy()
        row = {"algo": algo, "dataset": fold, "n_candidates": len(scores)}
        for b in BUDGETS:
            row[f"p1_at_{b}"] = p1_at_budget(scores, b)
        # What is still being lost to a finite budget.
        row["slope_at_max"] = row[f"p1_at_{BUDGETS[-1]}"] - row[f"p1_at_{BUDGETS[-2]}"]
        rows.append(row)
    return pd.DataFrame(rows)


def variance_decomposition(df) -> pd.DataFrame:
    """Split ARI variance into resolution, shape and interaction."""
    ok = df[(df.status == "ok") & df.feasible]
    ok = ok[ok.granularity_level.notna() & ok.theta.notna()]
    rows = []
    for (algo, fold), grp in ok.groupby(["algo", "dataset"]):
        cell = grp.groupby(["granularity_level", "theta"])["ari"].mean()
        if cell.size < 4 or cell.isna().all():
            continue
        table = cell.unstack("theta")
        grand = float(np.nanmean(table.to_numpy()))
        row_means = np.nanmean(table.to_numpy(), axis=1)
        col_means = np.nanmean(table.to_numpy(), axis=0)
        n_g, n_t = table.shape
        ss_g = float(n_t * np.nansum((row_means - grand) ** 2))
        ss_theta = float(n_g * np.nansum((col_means - grand) ** 2))
        resid = table.to_numpy() - row_means[:, None] - col_means[None, :] + grand
        ss_gxt = float(np.nansum(resid**2))
        total = ss_g + ss_theta + ss_gxt
        if total <= 0:
            continue
        rows.append(
            {
                "algo": algo,
                "dataset": fold,
                "ss_g_share": ss_g / total,
                "ss_theta_share": ss_theta / total,
                "ss_interaction_share": ss_gxt / total,
            }
        )
    return pd.DataFrame(rows)


def hypervolume(df) -> pd.DataFrame:
    """2^3 objective summary: ARI up, runtime down, memory down.

    Computed per (algorithm, fold) over its whole candidate set, so it describes
    the ALGORITHM rather than how well it was tuned.
    """
    ok = df[(df.status == "ok") & df.feasible]
    ref_time, ref_mem = 900.0, 2048.0
    rows = []
    for (algo, fold), grp in ok.groupby(["algo", "dataset"]):
        pts = grp[["ari", "runtime_s", "peak_memory_mb"]].dropna()
        if pts.empty:
            continue
        best = 0.0
        for _, p in pts.iterrows():
            vol = (
                max(p["ari"], 0.0)
                * max(ref_time - p["runtime_s"], 0.0) / ref_time
                * max(ref_mem - p["peak_memory_mb"], 0.0) / ref_mem
            )
            best = max(best, vol)
        rows.append({"algo": algo, "dataset": fold, "hypervolume": best})
    return pd.DataFrame(rows)


def friedman(table: pd.DataFrame):
    """Friedman over an algo x fold table, plus the critical difference."""
    from scipy.stats import friedmanchisquare

    clean = table.dropna(axis=1, how="any")
    if clean.shape[0] < 3 or clean.shape[1] < 3:
        return None
    ranks = clean.rank(axis=0, ascending=False)
    try:
        stat, p = friedmanchisquare(*[clean.iloc[i].to_numpy() for i in range(clean.shape[0])])
    except ValueError:
        return None
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "mean_ranks": ranks.mean(axis=1).to_dict(),
        "critical_difference": critical_difference(clean.shape[0], clean.shape[1]),
        "n_algorithms": clean.shape[0],
        "n_folds": clean.shape[1],
    }


def _gap(ceiling: pd.Series, other: pd.Series, label: str) -> dict:
    joined = pd.DataFrame({"p1": ceiling, "other": other}).dropna()
    if joined.empty:
        return {"comparison": label, "mean_absolute_gap": None, "mean_relative_gap": None}
    absolute = (joined.p1 - joined.other).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = ((joined.p1 - joined.other) / joined.p1.replace(0, np.nan)).mean()
    return {
        "comparison": label,
        "mean_absolute_gap": float(absolute),
        "mean_relative_gap": float(relative) if np.isfinite(relative) else None,
        "n_cells": int(len(joined)),
    }


def _md(frame, index_label: str | None = None) -> str:
    """Render a frame as markdown using the framework's own table formatter.

    pandas' ``to_markdown`` needs ``tabulate``, which is not a dependency of
    this repo; ``format_table`` is already here and already formats floats and
    missing values the way the other reports do.
    """
    from benchmark.reports import format_table

    frame = frame.copy()
    if index_label is not None:
        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: index_label})
    headers = [str(c) for c in frame.columns]
    rows = [list(row) for row in frame.to_numpy()]
    return format_table(headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    require(MATRIX_JSONL, "Stage 4 (build_matrix.py)")
    df = load_matrix()
    folds = [f for f in FOLDS if f in set(df.dataset)]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ceiling = p1(df)
    protocols = {
        "P0": p0(df),
        "P1": ceiling,
        "P1_holdout": p1_holdout(df),
        "P2": p2(df, folds),
        "P2_cross_group": p2(df, folds, groups=FOLD_GROUPS),
    }

    # --- 1. the gaps
    gaps = [
        _gap(ceiling, protocols["P0"], "P1 - P0"),
        _gap(ceiling, protocols["P1_holdout"], "P1 - P1_holdout"),
        _gap(ceiling, protocols["P2"], "P1 - P2"),
        _gap(ceiling, protocols["P2_cross_group"], "P1 - P2_cross_group"),
    ]
    prefix_rows = list(read_jsonl(PREFIX_JSONL))
    if prefix_rows:
        pf = pd.DataFrame(prefix_rows)
        pf = pf[pf.status == "ok"]
        for origin, grp in pf.groupby("origin"):
            series = grp.groupby(["algo", "dataset"])["ari_tail"].mean()
            gaps.append(_gap(ceiling, series, f"P1 - P3({origin:.0%})"))
    gaps_df = pd.DataFrame(gaps)
    gaps_df.to_csv(REPORT_DIR / "gaps.csv", index=False)

    # --- 2. the P2R table: which rule transfers
    pan = panel(df)
    p2r_rows = []
    baseline = protocols["P2"]
    for rule_name in RULES:
        for criterion in ("matr_trace_win", "silhouette", "ssq"):
            series = p2_rule(pan, folds, rule_name, criterion)
            if series.empty or series.isna().all():
                continue
            gap = _gap(ceiling, series, f"{rule_name}/{criterion}")
            gap["rule"] = rule_name
            gap["criterion"] = criterion
            p2r_rows.append(gap)
    # R_const is standard LODO and belongs in the same table, not beside it.
    const = _gap(ceiling, baseline, "R_const/-")
    const.update({"rule": "R_const", "criterion": "(transfers the value)"})
    p2r_rows.append(const)
    p2r_df = pd.DataFrame(p2r_rows)
    p2r_df.to_csv(REPORT_DIR / "p2r_regret.csv", index=False)

    # --- 3. budget curve, variance decomposition, Pareto
    budget = budget_curve(df)
    budget.to_csv(REPORT_DIR / "budget_curve.csv", index=False)
    variance = variance_decomposition(df)
    variance.to_csv(REPORT_DIR / "variance_decomposition.csv", index=False)
    hv = hypervolume(df)
    hv.to_csv(REPORT_DIR / "hypervolume.csv", index=False)

    # --- 4. statistics, per protocol
    stats = {}
    for name, series in protocols.items():
        table = _table(series)
        if table.empty:
            continue
        result = friedman(table)
        if result:
            stats[name] = result

    # --- 5. the report
    lines = ["# Stage 8 -- analysis\n\n"]
    lines.append("## 1. Gaps (normalised by P1)\n\n")
    lines.append(_md(gaps_df) + "\n\n")

    lines.append("## 2. P2R -- which selection rule transfers\n\n")
    lines.append("Mean LODO regret against P1. `R_const` is standard LODO.\n\n")
    if not p2r_df.empty:
        pivot = p2r_df.pivot_table(
            index="rule", columns="criterion", values="mean_absolute_gap"
        )
        lines.append(_md(pivot.round(4), "rule") + "\n\n")
    lines.append(
        "Read by row for which rule wins: if `R_parsimony` beats `R_const`, "
        "transferring a *rule* beats transferring a *value*, and what should be "
        "proposed upstream is a selection policy rather than a default. Read by "
        "column for which criterion is usable: if every rule is poor under every "
        "criterion, that is direct evidence for the claim that no label-free "
        "criterion works on these datasets.\n\n"
    )

    lines.append("## 3. Budget curve\n\n")
    if not budget.empty:
        per_algo = budget.groupby("algo")[[f"p1_at_{b}" for b in BUDGETS] + ["slope_at_max"]].mean()
        lines.append(_md(per_algo.round(4), "algo") + "\n\n")
        lines.append(
            "A curve already flat at small b means the space is covered and P1 is "
            "a real ceiling; one still rising at the largest budget means P1 is a "
            "floor. `slope_at_max` is what each algorithm still loses to the "
            "budget.\n\n"
        )

    lines.append("## 4. Parameter importance (tests P2R's assumption)\n\n")
    if not variance.empty:
        shares = (
            variance.groupby("algo")[
                ["ss_g_share", "ss_theta_share", "ss_interaction_share"]
            ]
            .mean()
            .round(4)
        )
        lines.append(_md(shares, "algo") + "\n\n")
        lines.append(
            "A large `ss_interaction_share` means the best shape depends on the "
            "resolution, which threatens P2R's premise that theta* can be "
            "transferred and g re-picked independently. Check this before "
            "reading the P2R table.\n\n"
        )

    lines.append("## 5. Statistical tests\n\n")
    for name, result in stats.items():
        lines.append(
            f"**{name}**: chi2={result['statistic']:.3f}, p={result['p_value']:.4g}, "
            f"A={result['n_algorithms']}, D={result['n_folds']}, "
            f"CD={result['critical_difference']:.3f}\n\n"
        )
    if stats:
        any_result = next(iter(stats.values()))
        lines.append(
            f"With A={any_result['n_algorithms']} and D={any_result['n_folds']} the "
            f"critical difference is {any_result['critical_difference']:.2f} ranks on "
            f"a {any_result['n_algorithms']}-rank scale. State plainly that almost no "
            "pair separates at this D, and prefer effect sizes with bootstrap "
            "intervals over the binary test.\n\n"
        )

    lines.append("## 6. Diagnostics that travel with every score\n\n")
    diag = (
        df[df.status == "ok"]
        .groupby("dataset")[["top1_variance_share", "effective_dim"]]
        .mean()
    )
    if not diag.empty:
        diag["1-D dominated"] = diag.top1_variance_share > 0.9
        lines.append(_md(diag.round(4), "dataset") + "\n\n")

    (REPORT_DIR / "report.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    print(f"wrote {REPORT_DIR}/report.md and 5 CSVs")


if __name__ == "__main__":
    main()
