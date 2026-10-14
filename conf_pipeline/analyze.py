"""Cross-fold statistics and the paper's figures, from the tables of ``report.py``.

The headline question is whether the *ranking of algorithms* depends on the
tuning regime (the arm). Everything here starts from
``report.summary_table`` (seed means per (algo, fold, arm)) and reads one
quality metric, by default ``lag_ari_win_tail_mean``: the lagged ARI over the
tail windows every arm shares. A cell whose ``all_seeds_ok`` is False (one of
its arm's seeds did not finish; spec 5.1) counts as no reading in the rankings,
Kendall's tau, win/tie/loss and the Pareto table; a summary without that column
is read as it is.

- ``rank_matrix``       folds x algos, rank 1 = best, a missing reading ranks last
- ``kendall_matrix``    arms x arms, mean over folds of Kendall's tau between rankings
- ``degenerate_table``  per arm, the cells whose configuration produced no usable partition
- ``friedman_nemenyi``  Friedman's test over folds and the Nemenyi critical difference
- ``win_tie_loss``      every algo against ``default``, ties inside the seed-noise band
- ``pareto_table``      quality against incumbent throughput, with a mean per-fold hypervolume
- ``fidelity``          does a capped run rank algorithms like the full-length run?
- ``impl_table``        spec 4.6 / T7: an implementation variant against its River reference
- ``write_analysis``    all of the above as CSVs, ``analysis.md`` and figures F1-F4

An implementation variant (``_common.IMPL_VARIANTS``, e.g. DenStreamPatchedFast
for DenStream) is the same algorithm under another implementation, so it is
never ranked with the protocol algorithms: every function above except
``impl_table`` leaves it out.

Run from the repo root::

    python -m conf_pipeline.analyze [--out-dir conf_pipeline/out] [--full-out-dir ...]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import _common as C
from . import report

TAIL = report.TAIL
GROUP = ["algo", "fold", "arm"]
COST = "incumbent_throughput_sps"

#: Critical values q_alpha of the Nemenyi test at alpha = 0.05, by number of
#: methods k (Demsar 2006, Table 5a: the studentized range statistic / sqrt(2)).
Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

WTL_COLS = ["algo", "wins", "ties", "losses", "n_folds", "worst_gap"]
PARETO_COLS = ["algo", "arm", "ari", "sps", "hv", "n_folds"]
FIDELITY_COLS = ["arm", "spearman", "n"]
FRIEDMAN_COLS = ["arm", "n_folds", "k", "statistic", "pvalue", "cd"]
IMPL_COLS = ["variant", "reference", "arm", "fold", "reference_value", "variant_value", "delta",
             "rank_reference", "rank_variant", "rank_change"]
#: Protocol order of the arms, for tables and colours; anything else follows, sorted.
ARM_ORDER: tuple[str, ...] = C.ARMS + C.OPTIONAL_ARMS + C.EXTRA_ARMS
#: Float slack on the noise-band comparison (0.6 - 0.55 is 0.04999999999999993).
_EPS = 1e-12

# Figure palette: categorical slots in fixed order (colour follows the arm, never
# its rank), a one-hue sequential ramp and a blue <-> red diverging pair around gray.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
_DIVERGING = ("#e34948", "#f0efec", "#2a78d6")
_INK, _INK_2, _MUTED, _GRID, _AXIS, _MISSING = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec",
)


def sd_column(metric: str) -> str:
    """The seed-sd column ``report.summary_table`` writes beside ``metric``."""
    return f"{metric.removesuffix('_mean')}_sd" if metric.endswith("_mean") else f"{metric}_sd"


# --- helpers ---------------------------------------------------------------------------------

def _has_group(summary: pd.DataFrame | None) -> bool:
    return summary is not None and not summary.empty and set(GROUP) <= set(summary.columns)


def _algos(summary: pd.DataFrame | None) -> list[str]:
    """Every protocol algo in ``summary``, over all arms, sorted (no implementation variant)."""
    if summary is None or "algo" not in summary:
        return []
    return sorted(set(summary["algo"].dropna().unique()) - set(C.IMPL_VARIANTS))


def _protocol(summary: pd.DataFrame | None) -> pd.DataFrame | None:
    """``summary`` without the rows of implementation variants."""
    if summary is None or "algo" not in summary:
        return summary
    return summary[~summary["algo"].isin(list(C.IMPL_VARIANTS))]


def _arms_in(summary: pd.DataFrame | None) -> list[str]:
    if summary is None or "arm" not in summary:
        return []
    present = set(summary["arm"].dropna())
    return [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))


def _arm_key(arm: str) -> tuple:
    return (ARM_ORDER.index(arm), "") if arm in ARM_ORDER else (len(ARM_ORDER), arm)


def _cells(summary: pd.DataFrame | None, arm: str, col: str) -> pd.Series:
    """``col`` of ``arm`` as a float Series indexed by (fold, algo); a missing reading dropped."""
    empty = pd.Series(
        dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["fold", "algo"]),
    )
    if not _has_group(summary) or col not in summary:
        return empty
    sub = summary[summary["arm"] == arm]
    vals = pd.to_numeric(sub[col], errors="coerce")
    keep = vals.notna().to_numpy()
    if not keep.any():
        return empty
    index = pd.MultiIndex.from_arrays(
        [sub["fold"].to_numpy()[keep], sub["algo"].to_numpy()[keep]], names=["fold", "algo"],
    )
    return pd.Series(vals.to_numpy(dtype=float)[keep], index=index).groupby(level=["fold", "algo"]).mean()


def _wide(summary: pd.DataFrame | None, arm: str, metric: str, algos: list[str]) -> pd.DataFrame:
    """folds x algos of ``metric`` under ``arm``: the folds where the arm has a reading."""
    cols = pd.Index(algos, name="algo")
    v = _cells(summary, arm, metric)
    if v.empty:
        return pd.DataFrame(index=pd.Index([], name="fold"), columns=cols, dtype=float)
    wide = v.unstack("algo").reindex(columns=cols).sort_index()
    wide.index.name = "fold"
    return wide


def _complete(summary: pd.DataFrame | None) -> pd.DataFrame | None:
    """``summary`` without the cells whose ``all_seeds_ok`` is False (spec 5.1: a
    cell is valid only when every seed of its arm finished). Only an explicit
    False drops a cell; a summary without the column is returned as it is."""
    if summary is None or "all_seeds_ok" not in summary:
        return summary
    incomplete = np.array([isinstance(v, (bool, np.bool_)) and not v for v in summary["all_seeds_ok"]],
                          dtype=bool)
    return summary[~incomplete]


def _ranks(wide: pd.DataFrame) -> pd.DataFrame:
    """Rank 1 = highest; ties averaged; a missing reading ranks ``k`` (the worst)."""
    return wide.rank(axis=1, method="average", ascending=False).fillna(float(wide.shape[1]))


# --- rankings --------------------------------------------------------------------------------

def rank_matrix(summary: pd.DataFrame, arm: str, metric: str = TAIL) -> pd.DataFrame:
    """Per fold, the rank of every algorithm under ``arm`` (rank 1 = highest ``metric``).

    Index: the folds where ``arm`` has at least one reading, sorted. Columns:
    every algo in ``summary`` over all arms, sorted, so each arm's matrix has
    the same columns. Ties take the average rank. An algorithm without a
    reading on a fold (no row, a failed run, a None metric, a cell missing a
    seed) ranks ``k = number of algos``: the worst.
    """
    return _ranks(_wide(_complete(summary), arm, metric, _algos(summary)))


def kendall_matrix(summary: pd.DataFrame, arms, metric: str = TAIL,
                   complete_only: bool = False) -> pd.DataFrame:
    """arms x arms: Kendall's tau-b between two arms' rank vectors, mean over folds.

    Folds: those both arms rank, less any where fewer than two algorithms have
    a reading in either arm (no ranking to compare). Identical rank vectors read
    tau = 1; a fold where only one of the two vectors is constant has no tau and
    is skipped. NaN when no fold is left. Symmetric, unit diagonal.

    By default an algorithm without a reading ranks last (``rank_matrix``), so
    a run that failed or timed out under one arm only -- the costly online
    arms -- moves the ranking of that arm alone and lowers tau with no change
    in quality. ``complete_only=True`` compares, on each fold, only the
    algorithms both arms have a reading for, ranked among themselves, and
    skips a fold where fewer than two are shared.
    """
    arms = list(dict.fromkeys(arms))
    algos = _algos(summary)
    wide = {a: _wide(_complete(summary), a, metric, algos) for a in arms}
    ranks = {a: _ranks(w) for a, w in wide.items()}
    present = {a: w.notna().sum(axis=1) for a, w in wide.items()}
    index = pd.Index(arms, name="arm")
    out = pd.DataFrame(np.eye(len(arms)), index=index, columns=index.copy())
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            taus = []
            for fold in ranks[a].index.intersection(ranks[b].index):
                if present[a][fold] < 2 or present[b][fold] < 2:
                    continue
                if complete_only:
                    both = wide[a].loc[fold].notna() & wide[b].loc[fold].notna()
                    if both.sum() < 2:
                        continue
                    x = _ranks(wide[a].loc[[fold], both]).iloc[0].to_numpy()
                    y = _ranks(wide[b].loc[[fold], both]).iloc[0].to_numpy()
                else:
                    x, y = ranks[a].loc[fold].to_numpy(), ranks[b].loc[fold].to_numpy()
                if np.array_equal(x, y):
                    taus.append(1.0)
                elif np.ptp(x) > 0 and np.ptp(y) > 0:
                    taus.append(float(stats.kendalltau(x, y).statistic))
            tau = float(np.mean(taus)) if taus else np.nan
            out.loc[a, b] = tau
            out.loc[b, a] = tau
    return out


def friedman_nemenyi(ranks: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman's test over the folds of a ``rank_matrix``, and the Nemenyi CD.

    ``statistic``/``pvalue``: ``scipy.stats.friedmanchisquare`` over the k
    algorithms (blocks = folds); NaN when k < 3, N < 2, or every fold ties every
    algorithm. ``cd = q_alpha(k) * sqrt(k (k+1) / (6 N))``: two algorithms whose
    average ranks differ by more than ``cd`` differ at ``alpha``; NaN for k
    outside 2..10 (the tabulated range). Only alpha = 0.05 is tabulated.
    """
    if not math.isclose(alpha, 0.05):
        raise ValueError(f"only alpha = 0.05 is tabulated (Q_ALPHA_05), got {alpha}")
    ranks = ranks.astype(float)
    n, k = ranks.shape
    avg = pd.Series(ranks.mean(axis=0), index=ranks.columns, dtype=float)
    statistic = pvalue = np.nan
    values = ranks.to_numpy()
    if k >= 3 and n >= 2 and (np.ptp(values, axis=1) > 0).any():
        res = stats.friedmanchisquare(*[ranks[c].to_numpy() for c in ranks.columns])
        statistic, pvalue = float(res.statistic), float(res.pvalue)
    q = Q_ALPHA_05.get(k)
    cd = float(q * math.sqrt(k * (k + 1) / (6 * n))) if q is not None and n > 0 else np.nan
    return {"statistic": statistic, "pvalue": pvalue, "avg_rank": avg, "cd": cd,
            "n_folds": int(n), "k": int(k)}


#: A cell whose lagged incumbent had no defined partition on more than this share of its
#: windows is degenerate. A runnable placeholder: pass ``threshold`` to change it.
DEGENERATE_SHARE = 0.5


def degenerate_table(summary: pd.DataFrame, threshold: float = DEGENERATE_SHARE) -> pd.DataFrame:
    """Per arm, how many (algo, fold) cells are degenerate: ``lag_undefined_share > threshold``.

    Such a cell reads ARI near 0 because the configuration produced no usable
    partition (one label, or the -1 sentinel), not because it clustered badly;
    a ranking under an arm with many of them -- typically ``default`` -- ranks
    broken configurations, and Kendall tau against it should be read beside
    this count. Over the cells with a reading of ``lag_undefined_share``; a
    cell missing a seed is left out (``_complete``). Implementation variants
    are left out, as everywhere outside ``impl_table``.
    """
    cols = ["arm", "n_cells", "n_degenerate", "share"]
    summary = _complete(_protocol(summary))
    if not _has_group(summary) or "lag_undefined_share" not in summary:
        return pd.DataFrame(columns=cols)
    rows = []
    for arm in _arms_in(summary):
        v = pd.to_numeric(summary.loc[summary["arm"] == arm, "lag_undefined_share"], errors="coerce").dropna()
        n_bad = int((v > threshold + _EPS).sum())
        rows.append({"arm": arm, "n_cells": int(len(v)), "n_degenerate": n_bad,
                     "share": n_bad / len(v) if len(v) else np.nan})
    return pd.DataFrame(rows, columns=cols)


# --- arm against default ---------------------------------------------------------------------

def win_tie_loss(summary: pd.DataFrame, arm: str, metric: str = TAIL) -> pd.DataFrame:
    """Per algo, how often ``arm`` beats ``default`` by more than the seed noise.

    Over the folds where both arms have ``metric`` (a cell missing a seed has
    none, see ``_complete``): ``gap = arm - default``;
    ``band = max(sd_arm, sd_default)`` on that fold (the seed sd beside the
    metric; a missing sd, e.g. a single seed, reads 0); win if ``gap > band``,
    loss if ``gap < -band``, tie otherwise. ``worst_gap`` is the smallest gap
    (NaN without a comparable fold).
    """
    algos = _algos(summary)
    if not algos:
        return _typed_wtl(pd.DataFrame(columns=WTL_COLS))
    sd = sd_column(metric)
    valid = _complete(summary)
    a, d = _cells(valid, arm, metric).to_dict(), _cells(valid, "default", metric).to_dict()
    sa, sdd = _cells(valid, arm, sd).to_dict(), _cells(valid, "default", sd).to_dict()
    rows = []
    for algo in algos:
        wins = ties = losses = 0
        gaps = []
        for key in sorted(k for k in a if k[1] == algo and k in d):
            gap = a[key] - d[key]
            band = max(sa.get(key, 0.0), sdd.get(key, 0.0))
            gaps.append(gap)
            if gap > band + _EPS:
                wins += 1
            elif gap < -band - _EPS:
                losses += 1
            else:
                ties += 1
        rows.append({"algo": algo, "wins": wins, "ties": ties, "losses": losses,
                     "n_folds": len(gaps), "worst_gap": min(gaps) if gaps else np.nan})
    return _typed_wtl(pd.DataFrame(rows, columns=WTL_COLS))


def _typed_wtl(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({"algo": object, "wins": int, "ties": int, "losses": int, "n_folds": int,
                      "worst_gap": float})


# --- quality against cost --------------------------------------------------------------------

def pareto_table(summary: pd.DataFrame, metric: str = TAIL, cost: str = COST) -> pd.DataFrame:
    """Per (algo, arm): mean ``metric`` (``ari``), mean ``cost`` (``sps``) and ``hv``.

    ``hv``: on every fold, ``metric`` and ``log10(cost)`` are min-max normalised
    over all (algo, arm) rows of that fold (an axis on which the fold's rows all
    agree maps to 1: nobody is worse), giving the pair one point
    ``(ari_norm, logsps_norm)`` per fold whose dominated area, reference (0, 0),
    is ``ari_norm * logsps_norm``; ``hv`` is the mean of that area over the
    folds. It is the typical fold's reading: a pair best on both axes on one
    fold and at the origin on another reads 0.5, not 1. Rows without ``metric``,
    with ``cost <= 0`` (no log) or missing a seed (``all_seeds_ok`` False) are
    left out before the normalisation; ``n_folds`` counts the folds that remain.
    """
    if not _has_group(summary) or metric not in summary or cost not in summary:
        return _typed_pareto(pd.DataFrame(columns=PARETO_COLS))
    summary = _complete(_protocol(summary))
    df = summary[GROUP].assign(
        ari=pd.to_numeric(summary[metric], errors="coerce"),
        sps=pd.to_numeric(summary[cost], errors="coerce"),
    )
    df = df[df["ari"].notna() & (df["sps"] > 0)]
    if df.empty:
        return _typed_pareto(pd.DataFrame(columns=PARETO_COLS))
    df = df.assign(area=_minmax(df["ari"], df["fold"]) * _minmax(np.log10(df["sps"]), df["fold"]))
    rows = []
    for (algo, arm), g in df.groupby(["algo", "arm"], sort=False):
        rows.append({
            "algo": algo, "arm": arm, "ari": float(g["ari"].mean()), "sps": float(g["sps"].mean()),
            "hv": float(np.clip(g["area"].mean(), 0.0, 1.0)), "n_folds": int(g["fold"].nunique()),
        })
    rows.sort(key=lambda r: (r["algo"], _arm_key(r["arm"])))
    return _typed_pareto(pd.DataFrame(rows, columns=PARETO_COLS))


def _typed_pareto(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({"algo": object, "arm": object, "ari": float, "sps": float, "hv": float,
                      "n_folds": int})


def _minmax(values: pd.Series, fold: pd.Series) -> np.ndarray:
    """Per-fold min-max to [0, 1]; a fold whose values all agree maps to 1."""
    lo = values.groupby(fold).transform("min").to_numpy(dtype=float)
    hi = values.groupby(fold).transform("max").to_numpy(dtype=float)
    v = values.to_numpy(dtype=float)
    rng = hi - lo
    return np.where(rng > 0, (v - lo) / np.where(rng > 0, rng, 1.0), 1.0)


# --- capped run against the full-length run ----------------------------------------------------

def fidelity(summary_cap: pd.DataFrame, summary_full: pd.DataFrame, arm: str,
             metric: str = TAIL) -> dict:
    """Spearman between the per-algo means of ``metric`` in a capped and a full run.

    Restricted to ``arm`` and to the folds present in both, and within them to
    the (algo, fold) cells both runs have: an algorithm that failed a fold in
    one run is compared on the same folds in the other, not on a different set.
    ``n`` = algos compared; ``spearman`` is NaN when ``n < 3`` or either side's
    means are all equal (no ranking).
    """
    cap, full = _cells(_protocol(summary_cap), arm, metric), _cells(_protocol(summary_full), arm, metric)
    common = cap.index.intersection(full.index)
    if common.empty:
        return {"spearman": np.nan, "n": 0}
    c = cap.loc[common].groupby(level="algo").mean()
    f = full.loc[common].groupby(level="algo").mean().reindex(c.index)
    n = int(len(c))
    if n < 3 or c.nunique() < 2 or f.nunique() < 2:
        return {"spearman": np.nan, "n": n}
    return {"spearman": float(stats.spearmanr(c.to_numpy(), f.to_numpy()).statistic), "n": n}


# --- implementation variants ---------------------------------------------------------------------

def impl_table(summary: pd.DataFrame, metric: str = TAIL) -> pd.DataFrame:
    """Spec 4.6 / T7: every implementation variant against the River class it re-implements.

    One row per (variant, arm, fold) where both the variant and its reference
    have ``metric`` (a cell missing a seed has none, see ``_complete``):
    ``delta = variant - reference``; ``rank_reference`` is the reference's rank
    among the protocol algorithms on that fold (``rank_matrix``'s rule);
    ``rank_variant`` the rank it would take with its reading replaced by the
    variant's; ``rank_change = rank_reference - rank_variant`` (positive: the
    variant ranks better). Empty, with the declared columns, without a variant.
    """
    rows = []
    complete = _complete(summary)
    algos = _algos(summary)
    for variant, reference in C.IMPL_VARIANTS.items():
        if reference not in algos:
            continue
        for arm in _arms_in(summary):
            wide = _wide(complete, arm, metric, [*algos, variant])
            for fold, row in wide.iterrows():
                ref, var = row[reference], row[variant]
                if pd.isna(ref) or pd.isna(var):
                    continue
                base = row[algos].to_frame().T
                swapped = base.copy()
                swapped[reference] = var
                r_ref = float(_ranks(base)[reference].iloc[0])
                r_var = float(_ranks(swapped)[reference].iloc[0])
                rows.append({
                    "variant": variant, "reference": reference, "arm": arm, "fold": fold,
                    "reference_value": float(ref), "variant_value": float(var), "delta": float(var - ref),
                    "rank_reference": r_ref, "rank_variant": r_var, "rank_change": r_ref - r_var,
                })
    rows.sort(key=lambda r: (r["variant"], _arm_key(r["arm"]), r["fold"]))
    return pd.DataFrame(rows, columns=IMPL_COLS)


# --- the report --------------------------------------------------------------------------------

def write_analysis(out_dir: Path, full_out_dir: Path | None = None) -> list[Path]:
    """``<out_dir>/report/``: ``analysis.md``, the CSVs behind it, and figures F1-F4.

    Figures need matplotlib and are skipped without it.
    """
    out_dir = Path(out_dir)
    rep = out_dir / "report"
    rep.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary = report.summary_table(out_dir)
    arms = _arms_in(summary)
    algos = _algos(summary)

    ranks = {arm: rank_matrix(summary, arm) for arm in arms}
    fried = {arm: friedman_nemenyi(ranks[arm]) for arm in arms}
    for arm in arms:
        written.append(_csv(ranks[arm], rep / f"ranks_{arm}.csv", index=True))
    kendall = kendall_matrix(summary, arms)
    written.append(_csv(kendall, rep / "kendall.csv", index=True))
    kendall_both = kendall_matrix(summary, arms, complete_only=True)
    written.append(_csv(kendall_both, rep / "kendall_complete.csv", index=True))
    degenerate = degenerate_table(summary)
    written.append(_csv(degenerate, rep / "degenerate.csv"))
    fried_table = pd.DataFrame(
        [{"arm": arm, **{c: fried[arm][c] for c in FRIEDMAN_COLS[1:]},
          **{f"avg_rank_{a}": fried[arm]["avg_rank"].get(a, np.nan) for a in algos}} for arm in arms],
        columns=FRIEDMAN_COLS + [f"avg_rank_{a}" for a in algos],
    )
    written.append(_csv(fried_table, rep / "friedman.csv"))

    wtl = {}
    if "default" in arms:
        for arm in arms:
            if arm != "default":
                wtl[arm] = win_tie_loss(summary, arm)
                written.append(_csv(wtl[arm], rep / f"win_tie_loss_{arm}.csv"))
    pareto = pareto_table(summary)
    written.append(_csv(pareto, rep / "pareto.csv"))

    fid = None
    if full_out_dir is not None:
        full = report.summary_table(Path(full_out_dir))
        both = [a for a in arms if a in set(_arms_in(full))]
        fid = pd.DataFrame([{"arm": a, **fidelity(summary, full, a)} for a in both],
                           columns=FIDELITY_COLS)
        written.append(_csv(fid, rep / "fidelity.csv"))

    impl = impl_table(summary)
    if not impl.empty:
        written.append(_csv(impl, rep / "impl_T7.csv"))

    figures = _figures(out_dir, rep, summary, arms, kendall, kendall_both)
    md = rep / "analysis.md"
    md.write_text(_markdown(summary, arms, algos, fried, kendall, wtl, pareto, fid, figures,
                            full_out_dir, impl, kendall_both, degenerate), encoding="utf-8")
    return written + [md] + figures


def _csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    df.to_csv(path, index=index)
    return path


def _fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}f}"
    return str(v)


def _md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(v, digits) for v in row) + " |"
              for row in df.itertuples(index=False, name=None)]
    return "\n".join(lines)


def _pvalue(p) -> str:
    if p is None or np.isnan(p):
        return "—"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _markdown(summary, arms, algos, fried, kendall, wtl, pareto, fid, figures, full_out_dir,
              impl=None, kendall_both=None, degenerate=None) -> str:
    n_folds = summary["fold"].nunique() if "fold" in summary else 0
    lines = ["# conf_pipeline analysis", ""]
    lines.append(
        f"Metric: `{TAIL}`, the lagged ARI over the tail windows every arm shares, as the "
        f"seed mean per (algo, fold, arm) of `report.summary_table`. {len(algos)} algorithms, "
        f"{n_folds} folds, {len(arms)} arms ({', '.join(f'`{a}`' for a in arms) or 'none'})."
    )

    lines += ["", "## Rankings by arm", ""]
    lines.append(
        "On every fold the algorithms are ranked by the metric (rank 1 = highest, ties "
        "averaged); an algorithm without a reading on a fold (no row, no finished seed) takes "
        "rank k, the worst. Only folds where the arm has at least one reading count. Friedman's "
        "test is over those folds (undefined for k < 3 or N < 2); the Nemenyi critical "
        "difference is `CD = q_0.05(k) * sqrt(k (k+1) / (6 N))` (Demšar 2006; undefined for k "
        "outside 2..10): two algorithms whose average ranks differ by more than CD differ at "
        "α = 0.05. A cell counts only when every seed its arm was run with finished "
        "(`all_seeds_ok`, spec §5.1): otherwise it is read as no reading, so it takes rank k and "
        "is left out of Kendall τ, win/tie/loss and the Pareto table."
    )
    lines.append("")
    if arms and algos:
        avg = pd.DataFrame({"algo": algos, **{a: [fried[a]["avg_rank"].get(g, np.nan) for g in algos]
                                              for a in arms}})
        lines.append(_md_table(avg, digits=2))
        lines.append("")
        tests = pd.DataFrame([{
            "arm": a, "N folds": fried[a]["n_folds"], "k": fried[a]["k"],
            "Friedman χ²": fried[a]["statistic"], "p": _pvalue(fried[a]["pvalue"]),
            "CD": fried[a]["cd"],
        } for a in arms])
        lines.append(_md_table(tests, digits=3))
    else:
        lines.append("no runs to rank")

    lines += ["", "## Ranking agreement between arms (Kendall τ)", ""]
    lines.append(
        "Kendall's τ-b between two arms' rank vectors on each fold, averaged over the folds "
        "both arms rank; a fold where fewer than two algorithms have a reading in either arm "
        "is skipped. τ = 1: the same order; 0: unrelated; −1: reversed. — : no fold left."
    )
    lines.append("")
    if len(arms) > 1:
        lines.append(_md_table(kendall.reset_index(), digits=3))
        if kendall_both is not None:
            lines += ["", (
                "The same on complete cases only: on each fold, the algorithms both arms have a "
                "reading for, ranked among themselves (a fold sharing fewer than two is skipped). "
                "Above, an algorithm that failed or timed out under one arm ranks last in that arm "
                "alone, which lowers τ without any change in quality; a gap between the two tables "
                "is that effect."
            ), ""]
            lines.append(_md_table(kendall_both.reset_index(), digits=3))
    else:
        lines.append("fewer than two arms: nothing to compare")
    if degenerate is not None and not degenerate.empty:
        lines += ["", (
            f"Degenerate cells per arm: `lag_undefined_share > {DEGENERATE_SHARE}`, the lagged "
            "incumbent produced no usable partition (one label, or the -1 sentinel) on most "
            "windows. Such a cell reads ARI near 0 because the configuration is broken, not "
            "because it clusters badly, so τ against an arm with many of them compares a "
            "ranking of broken configurations."
        ), ""]
        lines.append(_md_table(degenerate, digits=3))

    lines += ["", "## Win/tie/loss vs default", ""]
    lines.append(
        "Per algorithm, over the folds where both the arm and `default` have a reading: a win "
        "when `arm - default` exceeds the noise band `max(sd_arm, sd_default)` on that fold "
        "(seed sd; a missing sd, as with one seed, reads 0), a loss when it is below "
        "`-band`, a tie otherwise. `worst_gap` is the most negative `arm - default`."
    )
    lines.append("")
    if wtl:
        table = pd.concat(
            [w.assign(arm=a)[["arm", *WTL_COLS]] for a, w in wtl.items()], ignore_index=True,
        )
        lines.append(_md_table(table, digits=3))
    elif "default" not in arms:
        lines.append("no `default` arm in these runs")
    else:
        lines.append("no arm besides `default`")

    lines += ["", "## Pareto (ARI vs throughput)", ""]
    lines.append(
        "`ari`: mean over folds of the metric; `sps`: mean incumbent throughput (samples/s). "
        "`hv`: on every fold the metric and log10 throughput are min–max normalised over all "
        "(algo, arm) rows of that fold (an axis on which they all agree maps to 1), and the "
        "pair's point on that fold dominates the area `ari_norm * logsps_norm` of the unit "
        "square (reference (0, 0)); `hv` is the mean of that area over the folds, the typical "
        "fold's reading (best on both axes on one fold and at the origin on another reads "
        "0.5). Rows without a throughput are left out of all three."
    )
    lines.append("")
    lines.append(_md_table(pareto, digits=3) if not pareto.empty else "no row with both readings")

    if fid is not None:
        lines += ["", "## Fidelity cap vs full", ""]
        lines.append(
            f"Spearman correlation, per arm, between the per-algorithm means of the metric in "
            f"this run and in the full-length run (`{full_out_dir}`), over the (algo, fold) "
            "cells both runs have. `n`: algorithms compared; — : undefined (n < 3, or a side "
            "whose means are all equal)."
        )
        lines.append("")
        lines.append(_md_table(fid, digits=3) if not fid.empty else "no arm in both runs")

    if impl is not None and not impl.empty:
        lines += ["", "## Implementation variants (T7)", ""]
        lines.append(
            "Each variant is the same algorithm as its reference under another implementation "
            "(spec §4.6), so it is ranked with nobody above. Per arm, over the folds where both "
            "have a reading: `mean_delta` is the mean of `variant - reference`; `n_rank_changes` "
            "counts the folds where the reference's rank among the protocol algorithms would "
            "change if its reading were the variant's. Per-fold rows: `impl_T7.csv`."
        )
        lines.append("")
        agg = (impl.groupby(["variant", "reference", "arm"], sort=False)
               .agg(n_folds=("fold", "nunique"), mean_delta=("delta", "mean"),
                    n_rank_changes=("rank_change", lambda r: int((r != 0).sum())))
               .reset_index())
        lines.append(_md_table(agg, digits=3))

    lines += ["", "## Figures", ""]
    if figures:
        lines += [f"- `{p.name}`: {_figure_caption(p.name)}" for p in figures]
    else:
        lines.append("none (no runs, or matplotlib is not installed)")
    return "\n".join(lines) + "\n"


def _figure_caption(name: str) -> str:
    if name.startswith("F1_heatmap_"):
        return f"`{TAIL}` per algo x fold, arm `{name[len('F1_heatmap_'):-4]}` (— : no reading)"
    if name == "F2_kendall_complete.png":
        return "mean Kendall τ between the arms' rankings, complete cases only"
    if name == "F2_kendall.png":
        return "mean Kendall τ between the arms' rankings"
    if name == "F3_regret.png":
        return ("R_search_lag / R_proxy_lag per algo on the tail windows (labels given the silhouette's "
                "one-window delay), mean over folds, arm `confstream`")
    if name.startswith("F4_drift_"):
        return f"lagged ARI per window around the drift points of `{name[len('F4_drift_'):-4]}`"
    return ""


# --- figures -----------------------------------------------------------------------------------

def _pyplot():
    """``matplotlib.pyplot`` on the Agg backend, or None when matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib is optional (requirements.txt)
        return None
    return plt


def _style() -> dict:
    return {
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": _AXIS, "axes.labelcolor": _INK_2, "text.color": _INK,
        "xtick.color": _INK_2, "ytick.color": _INK_2, "grid.color": _GRID,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.6,
    }


def _arm_color(arm: str) -> str:
    if arm in ARM_ORDER:
        return _SERIES[ARM_ORDER.index(arm) % len(_SERIES)]
    return _MUTED


def _figures(out_dir: Path, rep: Path, summary: pd.DataFrame, arms: list[str],
             kendall: pd.DataFrame, kendall_both: pd.DataFrame | None = None) -> list[Path]:
    plt = _pyplot()
    if plt is None or not arms:
        return []
    written: list[Path] = []
    with plt.rc_context(_style()):
        written += _f1_heatmaps(plt, rep, summary, arms)
        if len(arms) > 1:
            written.append(_f2_kendall(plt, rep, kendall))
            if kendall_both is not None:
                written.append(_f2_kendall(plt, rep, kendall_both, complete_only=True))
        p = _f3_regret(plt, rep, report.regret_table(out_dir))
        if p is not None:
            written.append(p)
        written += _f4_drift(plt, out_dir, rep)
    return written


def _save(plt, fig, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _f1_heatmaps(plt, rep: Path, summary: pd.DataFrame, arms: list[str]) -> list[Path]:
    from matplotlib.colors import LinearSegmentedColormap

    algos = _algos(summary)
    folds = sorted(summary["fold"].dropna().unique())
    grids = {a: _wide(summary, a, TAIL, algos).reindex(index=folds).T for a in arms}
    finite = np.concatenate([g.to_numpy(dtype=float).ravel() for g in grids.values()])
    finite = finite[np.isfinite(finite)]
    vmin = min(0.0, float(finite.min())) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    vmax = vmax if vmax > vmin else vmin + 1.0
    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL).with_extremes(bad=_MISSING)
    out = []
    for arm in arms:
        grid = grids[arm].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(1.6 + 0.55 * max(len(folds), 1), 1.2 + 0.42 * max(len(algos), 1)))
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                dark = np.isfinite(v) and (v - vmin) / (vmax - vmin) > 0.55
                ax.text(j, i, "—" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if dark else _INK)
        ax.set_xticks(range(len(folds)), folds, rotation=45, ha="right")
        ax.set_yticks(range(len(algos)), algos)
        ax.set_title(f"Lagged tail ARI, arm {arm}")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.set_label(TAIL, fontsize=8)
        out.append(_save(plt, fig, rep / f"F1_heatmap_{arm}.png"))
    return out


def _f2_kendall(plt, rep: Path, kendall: pd.DataFrame, complete_only: bool = False) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    arms = list(kendall.index)
    grid = kendall.to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("div_red_blue", _DIVERGING).with_extremes(bad=_MISSING)
    size = 1.6 + 0.9 * len(arms)
    fig, ax = plt.subplots(figsize=(size + 0.8, size))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-1.0, vmax=1.0)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, "—" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if np.isfinite(v) and abs(v) > 0.6 else _INK)
    ax.set_xticks(range(len(arms)), arms, rotation=30, ha="right")
    ax.set_yticks(range(len(arms)), arms)
    ax.set_title("Ranking agreement between arms (mean Kendall τ over folds"
                 + (", complete cases)" if complete_only else ")"))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label("τ", fontsize=8)
    return _save(plt, fig, rep / ("F2_kendall_complete.png" if complete_only else "F2_kendall.png"))


#: The same-window decomposition (``a_star`` picked with the window's own labels: a bound).
REGRET_TERMS = ("R_lag", "R_proxy", "R_search")
#: The decomposition F3 plots: labels given the silhouette's one-window delay.
REGRET_TERMS_LAG = ("R_search_lag", "R_proxy_lag")


def regret_means(regret: pd.DataFrame, arm: str = "confstream", region: str = "tail",
                 terms: tuple[str, ...] = REGRET_TERMS) -> pd.DataFrame:
    """algos x ``terms``: the mean over folds of ``report.regret_table``'s
    ``region`` rows (the tail by default, the primary region) under ``arm``.

    NaN where no fold has the term (or the table has no such column); empty
    without a row. A table without a ``region`` column (one written before the
    split) is read whole.
    """
    terms = list(terms)
    empty = pd.DataFrame(columns=terms, index=pd.Index([], name="algo"), dtype=float)
    if regret is None or regret.empty or "arm" not in regret:
        return empty
    keep = regret["arm"] == arm
    if "region" in regret:
        keep &= regret["region"] == region
    rows = regret[keep]
    if rows.empty:
        return empty
    values = {t: pd.to_numeric(rows[t], errors="coerce") if t in rows else np.nan for t in terms}
    return rows[["algo"]].assign(**values).groupby("algo", sort=True)[terms].mean()


def _f3_regret(plt, rep: Path, regret: pd.DataFrame, arm: str = "confstream",
               terms: tuple[str, ...] = REGRET_TERMS_LAG) -> Path | None:
    means = regret_means(regret, arm, terms=terms)
    if means.empty:
        return None
    algos = list(means.index)
    x = np.arange(len(algos))
    fig, ax = plt.subplots(figsize=(2.2 + 0.8 * len(algos), 3.6))
    # positive parts stack up from 0, negative parts down from 0
    pos, neg = np.zeros(len(algos)), np.zeros(len(algos))
    handles, missing = [], []
    for slot, term in enumerate(terms):
        vals = means[term].to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            missing.append(term)
        vals = np.nan_to_num(vals, nan=0.0)
        for part, base in ((np.clip(vals, 0, None), pos), (np.clip(vals, None, 0), neg)):
            nz = part != 0      # a zero-height bar would pin the y limits at its base
            if nz.any():
                ax.bar(x[nz], part[nz], 0.6, bottom=base[nz], color=_SERIES[slot],
                       edgecolor="white", linewidth=1)
        pos, neg = pos + np.clip(vals, 0, None), neg + np.clip(vals, None, 0)
        label = f"{term} (n/a)" if term in missing else term
        handles.append(plt.Rectangle((0, 0), 1, 1, color=_SERIES[slot], label=label))
    net = np.nansum(means.to_numpy(dtype=float), axis=1)
    ax.scatter(x, net, marker="D", s=22, color=_INK, zorder=3)
    handles.append(plt.Line2D([], [], marker="D", linestyle="", color=_INK, label="sum of available terms"))
    lo, hi = min(0.0, neg.min(), net.min()), max(0.0, pos.max(), net.max())
    pad = 0.08 * (hi - lo) if hi > lo else 0.01
    ax.set_ylim(lo - pad if lo < 0 else 0.0, hi + pad)
    ax.axhline(0, color=_AXIS, linewidth=1)
    ax.set_xticks(x, algos, rotation=30, ha="right")
    ax.set_ylabel("regret (ARI)")
    title = f"Gap to the static ceiling on the tail, arm {arm} (mean over folds)"
    if missing:
        title += f"\n{', '.join(missing)} not available: no tail window / member log / static oracle"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(plt, fig, rep / "F3_regret.png")


def _cid(v) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else int(f)


def _drift_folds(runs: pd.DataFrame) -> dict[str, list[int]]:
    """fold -> drift windows (``drift_point // h``) of the ``ok`` run rows that carry any."""
    out: dict[str, set[int]] = {}
    if runs.empty or "drift_points" not in runs:
        return {}
    for row in runs[runs["status"] == "ok"].to_dict("records"):
        pts = row.get("drift_points")
        if not isinstance(pts, (list, tuple)) or not pts:
            continue
        h = _cid(row.get("h")) or C.H
        out.setdefault(row["fold"], set()).update(int(p) // h for p in pts)
    return {f: sorted(w) for f, w in sorted(out.items())}


def _f4_drift(plt, out_dir: Path, rep: Path) -> list[Path]:
    from matplotlib.ticker import MaxNLocator

    runs = report.current_runs(out_dir)
    folds = _drift_folds(runs)
    if not folds:
        return []
    win = report.load_windows(out_dir)
    if win.empty or "lag_ari_win" not in win:
        return []
    ok = runs[runs["status"] == "ok"]
    current = {
        (r["algo"], r["fold"], r["seed"], r["arm"], _cid(r.get("config_id")))
        for r in ok.to_dict("records")
    }
    out = []
    for fold, drift_windows in folds.items():
        g = win[win["fold"] == fold]
        if g.empty:
            continue
        cids = g["config_id"] if "config_id" in g else pd.Series(None, index=g.index, dtype=object)
        keep = np.array([(a, fold, s, arm, _cid(c)) in current
                         for a, s, arm, c in zip(g["algo"], g["seed"], g["arm"], cids)], dtype=bool)
        g = g[keep].assign(cid=[_cid(c) for c in cids[keep]],
                           lag=pd.to_numeric(g.loc[keep, "lag_ari_win"], errors="coerce"))
        g = g.drop_duplicates(["algo", "seed", "arm", "cid", "window"], keep="last")
        curve = g.groupby(["algo", "arm", "window"], sort=True)["lag"].mean().dropna()
        if curve.empty:
            continue
        pairs = set(curve.index.droplevel("window"))
        algos = sorted(curve.index.get_level_values("algo").unique())
        arms = sorted(curve.index.get_level_values("arm").unique(), key=_arm_key)
        ncols = min(4, len(algos))
        nrows = math.ceil(len(algos) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.5 * nrows + 0.5),
                                 sharex=True, sharey=True, squeeze=False, layout="constrained")
        for ax, algo in zip(axes.ravel(), algos):
            for w in drift_windows:
                ax.axvline(w, color=_MUTED, linestyle="--", linewidth=1)
            for arm in arms:
                if (algo, arm) in pairs:
                    s = curve.loc[(algo, arm)]
                    ax.plot(s.index, s.to_numpy(), color=_arm_color(arm), label=arm)
            ax.set_title(algo)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes.ravel()[len(algos):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("lagged ARI")
        for j in range(ncols):
            # the lowest visible panel of each column carries the window axis
            last = axes[max(i for i in range(nrows) if i * ncols + j < len(algos)), j]
            last.xaxis.set_tick_params(labelbottom=True)
            last.set_xlabel("window")
        handles = [plt.Line2D([], [], color=_arm_color(a), label=a) for a in arms]
        handles.append(plt.Line2D([], [], color=_MUTED, linestyle="--", label="drift window"))
        fig.suptitle(f"{fold}: lagged ARI per window (mean over seeds)", fontsize=10)
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        out.append(_save(plt, fig, rep / f"F4_drift_{fold}.png"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(C.OUT))
    parser.add_argument("--full-out-dir", default=None,
                        help="a full-length run's out dir: adds the cap-vs-full fidelity table")
    args = parser.parse_args()
    full = Path(args.full_out_dir) if args.full_out_dir else None
    for p in write_analysis(Path(args.out_dir), full_out_dir=full):
        print(p)


if __name__ == "__main__":
    main()
