"""What choosing the algorithm by its defaults costs once it is tuned online (plan P1.1).

A ranking of algorithms under River's defaults and one under online tuning can disagree.
This module turns that disagreement into the consequence a user meets: on each stream D,
pick ``a_default = argmax_a A_default(a, D)`` and read

    regret(D) = max_a A_target(a, D) - A_target(a_default, D)

the tail ARI given up under the target regime by having picked by defaults, against the
best algorithm under that regime. ``A`` is the tail ARI ``lag_ari_win_tail_mean`` (mean of
the lagged per-window ARI over windows >= the run's ``tail_from``), averaged over seeds.
The target is ``confstream`` (online tuning, primary), then ``frozen_prefix`` and
``oracle_static`` as sensitivity targets. Only the eight protocol algorithms compete; an
implementation variant (``_common.IMPL_VARIANTS``) is never a candidate. Every algorithm
competes on every fold: the tunable gate of ``report.headroom_table`` says whether one
algorithm's configuration can be improved, which has no bearing on choosing between them.

Per (fold, target) the module reads, from ``runs.jsonl`` (and ``windows.jsonl`` for the
split in time):

- ``regret_exp/best/worst``  the regret of a uniform pick from the selector's pick set, and
  its best and worst case. The pick set is every algorithm whose selecting score is within
  ``band`` of the top: ``band = 0`` is the exact top (ties within 1e-12 shared openly),
  ``band = b`` (``--band``, default ``_common.MIN_CEILING_GAIN``, the study's pre-declared
  smallest effect of interest) treats every algorithm within b of the top as practically
  tied, so a default ranking separated by less than b is not taken at its word.
- ``top1_agree``  the probability that the pick is one of the target's exact best (ties at
  the target's top all count as best).
- ``p_regret_gt_b``  the probability that the pick loses more than b, the ``--band`` value
  (column ``b``), on the exact rows as on the band rows; its mean over folds is the share of
  folds whose regret exceeds b. With ``--band 0`` it is ``1 - top1_agree``.
- ``flat_within_b``  every algorithm the selector scores is within b of its top: the
  selector's ranking separates nothing at the practical scale, and its band pick set is the
  random pick. Not read for a rank score.
- ``topk_overlap``  ``|top-k(selector) & top-k(target)| / k`` (``--top-k``, default 3), in
  expectation over a uniform breaking of the ties at the k-th seat on each side; read on
  the algorithms with both readings, undefined with fewer than k of them.
- ``tau``  Kendall's tau-b between the selector's scores and the target's readings on the
  algorithms with both; undefined, with the reason in ``tau_note``, when fewer than three
  algorithms have both or either side is constant. Neither top-k nor tau depends on the
  band; each band row repeats them.

Selectors, each a score per algorithm whose pick set is read as above:

- ``default``  the selecting arm's reading on the fold itself.
- ``random``  every algorithm tied: expected regret ``max - mean`` over the algorithms, top-k
  overlap k/n, no tau.
- ``lobo_default`` / ``lobo_target``  a global choice made on other data: the algorithm with
  the best mean reading (of ``default``, of the target) over the folds of every *other*
  source block (``_common.source_block``: the seven INSECTS folds are one block, so an
  INSECTS fold is held out with all of them). The mean is over folds, the plan's wording;
  the test fold never enters it.
- ``lobo_default_rank`` / ``lobo_target_rank``  the same with the mean rank over those folds
  (rank 1 best, ties averaged) as the score, a sensitivity: a mean of ARIs is ruled by the
  folds with the widest spread, a mean rank is not. The band, in ARI units, is not applied
  to a rank score: its pick set is the exact best mean rank at every band.

A global choice reads the same-data seed means of the other blocks in every variant below:
it never sees the test fold, so it is out of sample on it whichever way that fold is split.

Variants (``selection_regret.csv``'s ``variant``):

- ``same_data``  selection and evaluation on the same runs (every seed, whole tail): a
  **descriptive selection regret**, not an out-of-sample estimate. The target's maximum
  over eight noisy seed means is biased upward (the winner's curse), which inflates the
  regret; the bias is larger where the algorithms are close.
- ``heldout_seeds``  out of sample in seeds: select on the default arm's seed(s)
  ``--select-seeds`` (1), evaluate every target on ``--eval-seeds`` (2, 3), its maximum over
  those seeds too. ``oracle_static`` was rerun on seeds 2-4 (its configuration was picked
  on seed 1 of the sweep), so it is read on seeds 2-3 as well, leaving seed 4 out, so that
  every target averages the same number of seeds and its maximum carries the same noise.
  Five algorithms take no seed (BIRCH, DBSTREAM, DenStream, RepStream, SOStream): their
  default readings repeat exactly over seeds, so for them this split holds out only the
  target's tuner noise; the report lists the algorithms whose default readings repeat.
- ``prefix_suffix``  out of sample in time: each run's tail windows are split in two by
  window index (the first half, then the rest: an odd count gives the suffix the extra
  window); select on the default arm's mean lagged ARI over the prefix, evaluate the
  target on the suffix. Every job of a fold sees the same windows, so the halves cover the
  same stretch of the stream for every algorithm and arm. A half with no defined window (a
  run of single-class windows) has no reading.

Neither split generalises to new synthetic streams: the synthetic folds were generated
from one data seed, so that would need runs on new data-generator seeds, which the logs do
not have.

Aggregation (``groups.csv``): per stream group (``real``, ``param_drift``, ``other_synth``,
``all``) and per source block, fold-weighted means (``_fw``) and block-weighted ones
(``_bw``: the mean within each block, then over blocks), with ``n_folds`` and ``n_blocks`` of
the folds where the regret is defined, and ``n_tau_folds``/``n_topk_folds`` for the two
readings that can be undefined on their own; ``regret_exp_med``/``regret_exp_max`` are the
median and worst fold, ``n_flat_folds`` counts the folds where ``flat_within_b``. An
undefined reading is left out of a mean and of its denominator; a group with no reading
reads NaN.

Another selecting arm (``--selecting-arm``, e.g. ``scaled_default`` for E1: River's defaults with
every absolute-distance parameter in units of s) replaces ``default`` throughout: its own pick
and its global choices are named after it (``scaled_default``, ``lobo_scaled_default``,
``lobo_scaled_default_rank``; the default arm's labels are the ones above), and its outputs go
to ``<out-dir>/report/selection_<arm>/`` so the published ``selection/`` is never overwritten.
An out dir without the arm writes the files with no row and the reason. ``run.py`` makes no
``scaled_default`` run of an algorithm without an absolute-distance parameter (HPStream,
RepStream): that configuration is its default, so its ``default`` runs stand in for it
(``IDENTICAL_TO``, ``fill_identical``) and every algorithm stays a candidate; the report names
the algorithms copied, ``--no-fill-identical`` leaves them out. The copies are made only
when the out dir ran the arm at all.

Outputs, in ``<out-dir>/report/selection/``: ``selection_regret.csv`` (one row per variant x
target x band x selector x fold), ``groups.csv``, ``selection.md`` and
``F_selection_regret.png`` (per-fold tau beside the regret of the default pick under
``confstream``). No tuning run is made.

``--lobo`` writes the mechanism campaign's selection reading instead (``lobo_real``): real sources
only, each held out with its whole block, every training block weighted once (variants first), one
training panel and candidate set for ``lobo_default``, ``lobo_scaled_default``, ``lobo_target`` and the
exact uniform expectation, ties to the name that sorts first, and the primary difference
``Q_confstream,test(a_lobo_target) - Q_confstream,test(a_lobo_scaled_default)`` with per-source rows and
leave-one-source-out sensitivity. The panel it expects is read from a plan before any result
(``read_lobo_plan``: ``--campaign-config``, the dir's campaign manifest, or its run manifest), and a
held-out block whose training panel misses a regime, a job or a common window chooses nothing and says
why (``lobo_status.csv``); the mean over blocks stays undefined unless every expected block is whole.
``--reduced-panel`` adds the sensitivity that excludes what is missing, apart. Under the split seed policy
the held-out-seed variant of ``write_selection`` is read only on seeds given explicitly
(``resolve_heldout_seeds``).

Run from the repo root::

    python -m conf_pipeline.selection --out-dir conf_pipeline/out/full [--band 0.05] [--top-k 3]
    python -m conf_pipeline.selection --out-dir <dir> --selecting-arm scaled_default
    python -m conf_pipeline.selection --out-dir <dir> --lobo [--metric nmi] [--campaign-config <json>] [--reduced-panel]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from . import report

#: The arm whose ranking a user would pick by (``--selecting-arm``; E1 reads ``scaled_default``),
#: and the regimes the pick is read under.
SELECTING_ARM = "default"
TARGETS: tuple[str, ...] = ("confstream", "frozen_prefix", "oracle_static")
PRIMARY_TARGET = "confstream"
ARMS: tuple[str, ...] = (SELECTING_ARM,) + TARGETS
#: A selecting arm whose run ``run.py`` skips for an algorithm because it would repeat another
#: arm's configuration: ``scaled_default`` of an algorithm without an absolute-distance
#: parameter (HPStream, RepStream) is its ``default``. That arm's runs stand in for it.
IDENTICAL_TO: dict[str, str] = {"scaled_default": "default"}
#: How each selecting arm is named in the report's sentences.
ARM_WORDS: dict[str, str] = {
    "scaled_default": "River's defaults with every absolute-distance parameter in units of s",
}
#: Placeholders the CLI exposes: the top-k overlap's k, the practical-effect band (the
#: study's pre-declared smallest effect of interest), and the seeds of the held-out split.
TOP_K = 3
BAND = C.MIN_CEILING_GAIN
SELECT_SEEDS: tuple[int, ...] = (1,)
EVAL_SEEDS: tuple[int, ...] = (2, 3)
#: Why the held-out-seed split is not read on a split-policy dir unless its seeds are given: the legacy
#: 1/2/3 are job seeds of the shared policy, and a fixed arm's seed there is not a tuner seed.
NO_HELDOUT_SEEDS_NOTE = ("split seed policy: the held-out-seed split needs --select-seeds and --eval-seeds given "
                         "explicitly (tuner seeds); the legacy 1/2/3 are not assumed")


def resolve_heldout_seeds(runs: pd.DataFrame, select_seeds=None, eval_seeds=None) -> tuple:
    """The held-out-seed split's seeds: as given; when not given, the legacy ``SELECT_SEEDS``/``EVAL_SEEDS``
    for a shared-policy dir and None (the split is not read, ``NO_HELDOUT_SEEDS_NOTE``) for a split one."""
    if select_seeds is not None and eval_seeds is not None:
        return tuple(select_seeds), tuple(eval_seeds)
    if report.is_split(runs):
        return None, None
    return tuple(select_seeds or SELECT_SEEDS), tuple(eval_seeds or EVAL_SEEDS)


def _seeds_text(seeds) -> str:
    return ",".join(map(str, seeds)) if seeds is not None else "(not given)"


def selectors(selecting_arm: str = SELECTING_ARM) -> tuple[str, ...]:
    """The selectors of a run of this module: the selecting arm's own pick (named after the
    arm), the random pick, and the global choices made from the selecting arm and from the
    target, by mean score and by mean rank."""
    a = selecting_arm
    return (a, "random", f"lobo_{a}", "lobo_target", f"lobo_{a}_rank", "lobo_target_rank")


SELECTORS: tuple[str, ...] = selectors(SELECTING_ARM)
VARIANTS: tuple[str, ...] = ("same_data", "heldout_seeds", "prefix_suffix")
HALVES: tuple[str, ...] = ("prefix", "suffix")
ALL = "all"
TAIL = report.TAIL
#: Two readings closer than this are tied (float noise of a mean, not a difference).
_EPS = 1e-12
#: Kendall reads values rounded to the tie tolerance, so its ties are the pick set's.
_ROUND = 12

ROW_COLS = [
    "variant", "target", "band", "selector", "fold", "group", "block", "n_algos", "picked", "n_picked",
    "select_top", "flat_within_b", "n_train_folds", "target_best", "target_best_algos", "regret_exp",
    "regret_best", "regret_worst", "b", "top1_agree", "p_regret_gt_b", "topk_overlap", "topk_note", "tau",
    "tau_note", "note",
]
#: Readings ``aggregate`` averages, fold- and block-weighted.
STATS: tuple[str, ...] = (
    "regret_exp", "regret_best", "regret_worst", "top1_agree", "p_regret_gt_b", "topk_overlap", "tau",
)


# --- the pick set and what it reads -------------------------------------------------------------

def _clean(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce").dropna()


def pick_set(scores: pd.Series, band: float) -> list[str]:
    """The algorithms whose score is within ``band`` of the top (the exact top at 0, ties
    within ``_EPS`` included), sorted by name; empty without a score."""
    s = _clean(scores)
    if s.empty:
        return []
    top = float(s.max())
    return sorted(str(a) for a, v in s.items() if v >= top - band - _EPS)


def pick_readings(pick: list[str], target: pd.Series, b: float) -> dict:
    """The target's readings of a uniform pick from ``pick``.

    ``target``: algo -> the target's reading on the fold. The regret of an algorithm is
    the target's best minus its reading; ``regret_exp`` is the mean over ``pick``,
    ``regret_best``/``regret_worst`` its min/max, ``top1_agree`` the share of ``pick`` among
    the target's exact best, ``p_regret_gt_b`` the share losing more than ``b``.
    Undefined (NaN, the reason in ``note``) when the target has no reading on the fold,
    nothing was picked, or a picked algorithm has no target reading: a pick that failed
    under the target is not dropped in favour of the others.
    """
    t = _clean(target)
    out = {"target_best": np.nan, "target_best_algos": None, "regret_exp": np.nan, "regret_best": np.nan,
           "regret_worst": np.nan, "top1_agree": np.nan, "p_regret_gt_b": np.nan, "note": None}
    if t.empty:
        out["note"] = "no target reading on this fold"
        return out
    best = float(t.max())
    out["target_best"] = best
    out["target_best_algos"] = ";".join(sorted(str(a) for a, v in t.items() if v >= best - _EPS))
    if not pick:
        out["note"] = "the selector picked nothing (no selecting reading)"
        return out
    missing = [a for a in pick if a not in t.index]
    if missing:
        out["note"] = f"no target reading of {', '.join(missing)}"
        return out
    reg = best - t[list(pick)].to_numpy(dtype=float)
    out.update(regret_exp=float(reg.mean()), regret_best=float(reg.min()), regret_worst=float(reg.max()),
               top1_agree=float(np.mean(reg <= _EPS)), p_regret_gt_b=float(np.mean(reg > b + _EPS)))
    return out


def inclusion(scores: pd.Series, k: int) -> pd.Series:
    """Per algorithm, the probability of a seat in the top ``k`` when ties are broken
    uniformly: 1 strictly inside, 0 strictly outside, and the free seats shared among the
    algorithms tied at the boundary. Sums to ``min(k, n)``."""
    s = _clean(scores)
    v = s.to_numpy(dtype=float)
    out = {}
    for a, x in zip(s.index, v):
        above = int(np.sum(v > x + _EPS))
        tied = int(np.sum(np.abs(v - x) <= _EPS))
        out[a] = float(min(1.0, max(0.0, (k - above) / tied)))
    return pd.Series(out, dtype=float)


def _both(scores: pd.Series, target: pd.Series) -> tuple[pd.Series, pd.Series]:
    s, t = _clean(scores), _clean(target)
    common = s.index.intersection(t.index)
    return s[common], t[common]


def topk_overlap(scores: pd.Series, target: pd.Series, k: int) -> tuple[float, str | None]:
    """Expected ``|top-k(scores) & top-k(target)| / k`` over the algorithms with both
    readings, ties broken uniformly and independently on each side; NaN with the reason
    when fewer than ``k`` algorithms have both."""
    s, t = _both(scores, target)
    if len(s) < k:
        return np.nan, f"fewer than {k} algorithms with both readings ({len(s)})"
    return float((inclusion(s, k) * inclusion(t, k)).sum() / k), None


def kendall(scores: pd.Series, target: pd.Series) -> tuple[float, str | None]:
    """Kendall's tau-b between the selector's scores and the target's readings on the
    algorithms with both; NaN with the reason when fewer than three have both or either
    side is constant (a constant side has no ranking to agree with)."""
    from scipy import stats

    s, t = _both(scores, target)
    if len(s) < 3:
        return np.nan, f"fewer than 3 algorithms with both readings ({len(s)})"
    x, y = np.round(s.to_numpy(dtype=float), _ROUND), np.round(t.to_numpy(dtype=float), _ROUND)
    if np.ptp(x) == 0:
        return np.nan, "the selector's scores are constant (no ranking)"
    if np.ptp(y) == 0:
        return np.nan, "the target's readings are constant (no ranking)"
    return float(stats.kendalltau(x, y).statistic), None


# --- global choices made on other blocks -----------------------------------------------------------

def lobo_scores(train: pd.DataFrame, fold: str, by: str = "score") -> tuple[pd.Series, list[str]]:
    """The score of every algorithm over the folds of ``train`` (folds x algos) outside
    ``fold``'s source block, and those folds. ``by="score"``: the mean reading over them;
    ``by="rank"``: minus the mean rank (rank 1 = best within a fold, ties averaged), so the
    highest score wins either way. An algorithm is averaged over the folds it has a reading
    on. Empty when no fold lies outside the block."""
    block = C.source_block(fold)
    folds = [str(f) for f in train.index if C.source_block(f) != block]
    if not folds:
        return pd.Series(dtype=float), []
    sub = train.loc[folds].apply(pd.to_numeric, errors="coerce")
    if by == "score":
        scores = sub.mean(axis=0, skipna=True)
    elif by == "rank":
        scores = -sub.rank(axis=1, ascending=False, method="average").mean(axis=0, skipna=True)
    else:
        raise ValueError(f"by must be 'score' or 'rank', got {by!r}")
    return scores.dropna().astype(float), folds


# --- one variant's per-fold table --------------------------------------------------------------------

def _row_of(frame: pd.DataFrame | None, fold: str) -> pd.Series:
    if frame is None or fold not in frame.index:
        return pd.Series(dtype=float)
    return _clean(frame.loc[fold])


def _fold_key(fold: str) -> tuple:
    group = C.stream_group(fold)
    return (C.STREAM_GROUPS.index(group), C.FOLDS.index(fold) if fold in C.FOLDS else len(C.FOLDS), fold)


def _notes(*parts) -> str | None:
    parts = [p for p in parts if p]
    return "; ".join(parts) if parts else None


def _flat(scores: pd.Series, b: float):
    """Every scored algorithm within ``b`` of the top; None without two scores."""
    s = _clean(scores)
    return bool(float(s.max() - s.min()) <= b + _EPS) if len(s) > 1 else None


def fold_selection(variant: str, sel: pd.DataFrame, targets: dict[str, pd.DataFrame],
                   train: dict[str, pd.DataFrame], band: float = BAND, k: int = TOP_K,
                   selecting_arm: str = SELECTING_ARM) -> pd.DataFrame:
    """One row per (target, band, selector, fold) of one variant, ``band`` being 0 (exact
    ties) and, when ``band > 0``, ``band`` itself (column ``band``: the pick set's width).
    Every row reads its pick against the same ``b = band`` (column ``b``).

    ``sel``: folds x algos of the selecting arm's readings; ``targets``: target -> folds x
    algos of its readings; ``train``: arm -> folds x algos of the same-data readings the
    global choices are made from (``selecting_arm`` and each target). Folds: those ``sel`` or
    a target has; a fold without a target reading keeps its row, undefined, with the reason.
    Selector labels: ``selectors(selecting_arm)``.
    """
    if band < 0:
        raise ValueError(f"band must be >= 0, got {band}")
    bands = (0.0, float(band)) if band > 0 else (0.0,)
    rows = []
    for target, tgt in targets.items():
        folds = sorted(set(map(str, sel.index)) | set(map(str, tgt.index)), key=_fold_key)
        for fold in folds:
            t = _row_of(tgt, fold)
            s = _row_of(sel, fold)
            universe = sorted(set(map(str, s.index)) | set(map(str, t.index)))
            own, _, lobo_own, lobo_t, lobo_own_rank, lobo_t_rank = selectors(selecting_arm)
            scores = {own: (s, None, None), "random": (pd.Series(0.0, index=universe), None, None)}
            for name, arm, by in ((lobo_own, selecting_arm, "score"), (lobo_t, target, "score"),
                                  (lobo_own_rank, selecting_arm, "rank"), (lobo_t_rank, target, "rank")):
                sc, used = lobo_scores(train[arm], fold, by=by) if arm in train else (pd.Series(dtype=float), [])
                scores[name] = (sc, len(used), None if used else "no fold outside the held-out block")
            for width in bands:
                for selector in selectors(selecting_arm):
                    sc, n_train, why = scores[selector]
                    rank_score = selector.endswith("_rank")
                    pick = pick_set(sc, 0.0 if rank_score else width)
                    read = pick_readings(pick, t, band)
                    topk, topk_note = topk_overlap(sc, t, k)
                    tau, tau_note = kendall(sc, t)
                    sc_clean = _clean(sc)
                    rows.append({
                        "variant": variant, "target": target, "band": float(width), "selector": selector,
                        "fold": fold, "group": C.stream_group(fold), "block": C.source_block(fold),
                        "n_algos": int(len(_both(sc, t)[0])), "picked": ";".join(pick), "n_picked": len(pick),
                        "select_top": (float(sc_clean.max()) if len(sc_clean) and selector != "random"
                                       else np.nan),
                        "flat_within_b": None if rank_score else _flat(sc, band),
                        "n_train_folds": n_train,
                        "target_best": read["target_best"], "target_best_algos": read["target_best_algos"],
                        "regret_exp": read["regret_exp"], "regret_best": read["regret_best"],
                        "regret_worst": read["regret_worst"], "b": float(band), "top1_agree": read["top1_agree"],
                        "p_regret_gt_b": read["p_regret_gt_b"], "topk_overlap": topk,
                        "topk_note": topk_note, "tau": tau, "tau_note": tau_note,
                        "note": _notes(why, read["note"],
                                       "band not applied to a rank score" if rank_score and width > 0 else None),
                    })
    return pd.DataFrame(rows, columns=ROW_COLS)


# --- aggregation ---------------------------------------------------------------------------------------

GROUP_KEYS = ["variant", "target", "band", "selector"]


def _weighted(sub: pd.DataFrame, stat: str) -> tuple[float, float, int, int]:
    """Fold-weighted mean, block-weighted mean, and the folds and blocks with a reading."""
    v = sub[["block", stat]].copy()
    v[stat] = pd.to_numeric(v[stat], errors="coerce")
    v = v.dropna(subset=[stat])
    if v.empty:
        return np.nan, np.nan, 0, 0
    per_block = v.groupby("block")[stat].mean()
    return float(v[stat].mean()), float(per_block.mean()), int(len(v)), int(len(per_block))


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, target, band, selector): one row per stream group (and ``all``) and
    per source block, each reading of ``STATS`` fold-weighted (``_fw``) and block-weighted
    (``_bw``), ``regret_exp_med``/``regret_exp_max`` the median and the worst fold.
    ``n_folds``/``n_blocks`` count the folds and blocks where the regret is defined,
    ``n_tau_*``/``n_topk_folds`` those where tau and the top-k overlap are, ``n_flat_folds``
    those where every scored algorithm is within b of the selector's top."""
    out = []
    if rows.empty:
        return pd.DataFrame(out)
    for key, g in rows.groupby(GROUP_KEYS, sort=False):
        blocks = sorted(set(g["block"]))
        units = [("group", grp, g if grp == ALL else g[g["group"] == grp]) for grp in (*C.STREAM_GROUPS, ALL)]
        units += [("block", b, g[g["block"] == b]) for b in blocks]
        for level, unit, sub in units:
            row = dict(zip(GROUP_KEYS, key), level=level, unit=unit)
            for stat in STATS:
                fw, bw, n_f, n_b = _weighted(sub, stat)
                row[f"{stat}_fw"], row[f"{stat}_bw"] = fw, bw
                if stat == "regret_exp":
                    row["n_folds"], row["n_blocks"] = n_f, n_b
                elif stat == "tau":
                    row["n_tau_folds"], row["n_tau_blocks"] = n_f, n_b
                elif stat == "topk_overlap":
                    row["n_topk_folds"] = n_f
            reg = pd.to_numeric(sub["regret_exp"], errors="coerce").dropna()
            row["regret_exp_med"] = float(reg.median()) if len(reg) else np.nan
            row["regret_exp_max"] = float(reg.max()) if len(reg) else np.nan
            row["n_flat_folds"] = int(sum(v is True or v == True for v in sub["flat_within_b"]))  # noqa: E712
            out.append(row)
    lead = [*GROUP_KEYS, "level", "unit", "n_folds", "n_blocks", "n_tau_folds", "n_tau_blocks", "n_topk_folds",
            "n_flat_folds"]
    df = pd.DataFrame(out)
    return df[lead + [c for c in df.columns if c not in lead]]


# --- the readings each variant selects and evaluates on ---------------------------------------------

def seed_values(runs: pd.DataFrame, arms=ARMS) -> pd.DataFrame:
    """One row per ok run of a protocol algorithm under ``arms``: algo, fold, arm, seed and
    ``value``, the run's tail ARI. Runs without a tail reading are left out."""
    cols = ["algo", "fold", "arm", "seed", "value"]
    if runs.empty or TAIL not in runs:
        return pd.DataFrame(columns=cols)
    ok = runs[(runs["status"] == "ok") & runs["arm"].isin(list(arms)) & ~runs["algo"].isin(list(C.variant_algos()))]
    out = ok[["algo", "fold", "arm", "seed"]].copy()
    out["value"] = pd.to_numeric(ok[TAIL], errors="coerce")
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    # Under the split seed policy a fixed arm's mean stands beside every tuner seed (report.replicate_view).
    if report.is_split(runs):
        out = report.replicate_view(out, split=True)[cols]
    return out


def wide(values: pd.DataFrame, arm: str, seeds=None) -> pd.DataFrame:
    """folds x algos: the mean of ``value`` over the runs of ``arm`` on ``seeds`` (every seed
    when None). Columns: every algorithm in ``values``, so a missing cell reads NaN."""
    algos = sorted(set(values["algo"])) if len(values) else []
    sub = values[values["arm"] == arm]
    if seeds is not None:
        wanted = {int(s) for s in seeds}
        sub = sub[sub["seed"].map(lambda v: report._seed_key(v) in wanted)]
    if sub.empty:
        out = pd.DataFrame(index=pd.Index([], name="fold"), columns=algos, dtype=float)
    else:
        out = sub.pivot_table(index="fold", columns="algo", values="value", aggfunc="mean").reindex(columns=algos)
    out.columns.name = None
    out.index.name = "fold"
    return out.astype(float)


def half_values(windows: pd.DataFrame, tail_from: dict) -> pd.DataFrame:
    """Per run (algo, fold, arm, seed) and half: the mean ``lag_ari_win`` over the first half
    of its tail windows (``prefix``) and over the rest (``suffix``). The tail is the run's
    windows >= its ``tail_from`` (``{(algo, fold, seed, arm): int}``, else
    ``FREEZE_AFTER_WINDOW``); its first ``n // 2`` windows are the prefix. Windows without a
    reading (single-class) count in the split but not in the means; a half with no reading
    has no row."""
    cols = ["algo", "fold", "arm", "seed", "half", "value"]
    if windows is None or windows.empty:
        return pd.DataFrame(columns=cols)
    df = windows[["algo", "fold", "arm", "seed", "window", "lag_ari_win"]].copy()
    df["window"] = pd.to_numeric(df["window"], errors="coerce")
    df["lag_ari_win"] = pd.to_numeric(df["lag_ari_win"], errors="coerce")
    tail_from = {(a, f, report._seed_key(sd), arm): v for (a, f, sd, arm), v in tail_from.items()}
    start = [int(tail_from.get((a, f, report._seed_key(sd), arm), C.FREEZE_AFTER_WINDOW))
             for a, f, sd, arm in zip(df["algo"], df["fold"], df["seed"], df["arm"])]
    df = df[df["window"] >= np.array(start, dtype=float)]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.drop_duplicates(["algo", "fold", "arm", "seed", "window"])
    job = ["algo", "fold", "arm", "seed"]
    # A deterministic fixed run has no seed (None under the split policy): kept as a group of its own.
    pos = df.groupby(job, dropna=False)["window"].rank(method="first") - 1
    n = df.groupby(job, dropna=False)["window"].transform("size")
    df["half"] = np.where(pos < (n // 2), HALVES[0], HALVES[1])
    out = df.groupby(job + ["half"], sort=True, dropna=False)["lag_ari_win"].mean().rename("value").reset_index()
    return out.dropna(subset=["value"])[cols].reset_index(drop=True)


def variant_tables(values: pd.DataFrame, halves: pd.DataFrame | None, targets=TARGETS,
                   select_seeds=SELECT_SEEDS, eval_seeds=EVAL_SEEDS, selecting_arm: str = SELECTING_ARM) -> dict:
    """variant -> ``(sel, {target: folds x algos}, note)``: what each variant selects on and
    evaluates on. ``sel`` is None, with the reason in ``note``, for a variant the logs do
    not support; a target absent from the runs is left out of the dict. ``select_seeds`` or
    ``eval_seeds`` None: the held-out-seed split is not read (``resolve_heldout_seeds``)."""
    seeds_given = select_seeds is not None and eval_seeds is not None
    if seeds_given and set(map(int, select_seeds)) & set(map(int, eval_seeds)):
        raise ValueError(f"selection seeds {tuple(select_seeds)} and evaluation seeds {tuple(eval_seeds)} overlap")
    present = set(values["arm"]) if len(values) else set()
    out = {}
    if selecting_arm not in present:
        reason = f"no `{selecting_arm}` run to select on"
        return {v: (None, {}, reason) for v in VARIANTS}
    if not any(t in present for t in targets):
        reason = f"no target run ({', '.join(f'`{t}`' for t in targets)})"
        return {v: (None, {}, reason) for v in VARIANTS}
    out["same_data"] = (wide(values, selecting_arm), {t: wide(values, t) for t in targets if t in present}, None)

    if not seeds_given:
        out["heldout_seeds"] = (None, {}, NO_HELDOUT_SEEDS_NOTE)
    else:
        sel = wide(values, selecting_arm, select_seeds)
        tgts = {t: wide(values, t, eval_seeds) for t in targets if t in present}
        tgts = {t: w for t, w in tgts.items() if not w.empty}
        if sel.empty or not tgts:
            out["heldout_seeds"] = (None, {}, f"no `{selecting_arm}` run on seeds {tuple(select_seeds)} or no "
                                              f"target run on seeds {tuple(eval_seeds)}")
        else:
            out["heldout_seeds"] = (sel, tgts, None)

    if halves is None:
        out["prefix_suffix"] = (None, {}, "windows.jsonl not found: no per-window reading to split in time")
    else:
        pre, suf = halves[halves["half"] == HALVES[0]], halves[halves["half"] == HALVES[1]]
        sel = wide(pre, selecting_arm)
        tgts = {t: wide(suf, t) for t in targets if t in present}
        tgts = {t: w for t, w in tgts.items() if not w.empty}
        if sel.empty or not tgts:
            out["prefix_suffix"] = (None, {}, "no run has a reading on both halves of its tail")
        else:
            out["prefix_suffix"] = (sel, tgts, None)
    return out


# --- loading -----------------------------------------------------------------------------------------

WINDOW_KEYS = ("algo", "fold", "seed", "arm", "config_id", "window", "lag_ari_win")


def load_windows(out_dir: Path, runs: pd.DataFrame, arms=ARMS) -> pd.DataFrame | None:
    """The ``WINDOW_KEYS`` of ``windows.jsonl`` for the current ok runs of protocol algorithms
    under ``arms`` (for ``oracle_static``, only the reruns of the configuration
    ``oracle_static.json`` names, as ``report.current_runs`` keeps them); None when the file
    is missing. Only these columns are kept: the file is large."""
    path = Path(out_dir) / C.WINDOWS_JSONL.name
    if not path.exists():
        return None
    ok = runs[(runs["status"] == "ok") & runs["arm"].isin(list(arms))] if not runs.empty else runs
    cid = ok["config_id"] if "config_id" in ok else pd.Series([None] * len(ok), index=ok.index)
    # A deterministic fixed run has no seed (None/NaN under the split policy): keyed as None, never cast.
    jobs = {(a, f, report._seed_key(s), arm, report._cid(c)) for a, f, s, arm, c in
            zip(ok["algo"], ok["fold"], ok["seed"], ok["arm"], cid) if a not in C.variant_algos()}
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind", "window") != "window" or r.get("arm") not in arms:
                continue
            key = (r.get("algo"), r.get("fold"), report._seed_key(r.get("seed")), r.get("arm"),
                   report._cid(r.get("config_id")))
            if key in jobs:
                rows.append(tuple(r.get(k) for k in WINDOW_KEYS))
    return pd.DataFrame(rows, columns=list(WINDOW_KEYS))


def repeating_defaults(values: pd.DataFrame, arm: str = SELECTING_ARM) -> list[str]:
    """Algorithms whose ``arm`` reading is the same on every seed of every fold: for them a
    split by seeds holds out nothing on the selecting side."""
    d = values[values["arm"] == arm]
    out = []
    for algo, g in d.groupby("algo", sort=True):
        spread = g.groupby("fold")["value"].agg(lambda v: float(np.ptp(v.to_numpy(dtype=float))))
        if g.groupby("fold")["seed"].nunique().min() > 1 and (spread <= _EPS).all():
            out.append(str(algo))
    return out


def _repeats_other(selecting_arm: str, algo: str) -> bool:
    """Whether ``run.py`` skips ``selecting_arm`` for ``algo`` because the configuration would be
    ``IDENTICAL_TO[selecting_arm]``'s: for ``scaled_default``, an algorithm with no absolute-distance
    parameter. An algorithm outside the protocol's spaces never qualifies."""
    if selecting_arm != "scaled_default":
        return False
    from .spaces import SPACES, distance_params

    return algo in SPACES and not distance_params(algo)


def fill_identical(frame: pd.DataFrame, selecting_arm: str) -> tuple[pd.DataFrame, list[str]]:
    """``frame`` (rows with algo, fold, arm, seed) with the selecting arm's missing runs of an
    algorithm whose selecting configuration is another arm's (``IDENTICAL_TO``) copied from that
    arm, and the algorithms filled. A run the selecting arm has is never replaced, and an
    out dir that never ran the selecting arm gets no arm made of copies alone."""
    other = IDENTICAL_TO.get(selecting_arm)
    if other is None or frame.empty or not (frame["arm"] == selecting_arm).any():
        return frame, []
    own = frame[frame["arm"] == selecting_arm]
    have = set(zip(own["algo"], own["fold"], own["seed"]))
    src = frame[frame["arm"] == other]
    src = src[np.array([_repeats_other(selecting_arm, a) and (a, f, sd) not in have
                        for a, f, sd in zip(src["algo"], src["fold"], src["seed"])], dtype=bool)]
    if src.empty:
        return frame, []
    return (pd.concat([frame, src.assign(arm=selecting_arm)], ignore_index=True),
            sorted(map(str, set(src["algo"]))))


# --- putting it together ---------------------------------------------------------------------------

def report_dir_name(selecting_arm: str = SELECTING_ARM) -> str:
    """``selection`` for the default arm, ``selection_<arm>`` for any other, so a reading with
    another selecting arm never overwrites the published one."""
    return "selection" if selecting_arm == SELECTING_ARM else f"selection_{selecting_arm}"


def write_selection(out_dir: Path, band: float = BAND, top_k: int = TOP_K, select_seeds=None,
                    eval_seeds=None, rep_dir: Path | None = None, selecting_arm: str = SELECTING_ARM,
                    fill: bool = True) -> list[Path]:
    """Every variant on ``out_dir``: ``selection_regret.csv``, ``groups.csv``, ``selection.md``
    and the figure in ``<rep>/<report_dir_name(selecting_arm)>/``, ``<rep>`` being
    ``<out_dir>/report`` unless ``rep_dir``. ``fill``: copy the runs ``fill_identical`` names."""
    out_dir = Path(out_dir)
    rep = (Path(rep_dir) if rep_dir is not None else out_dir / C.REPORT_DIR.name) / report_dir_name(selecting_arm)
    if band < 0:
        raise ValueError(f"band must be >= 0, got {band}")
    rep.mkdir(parents=True, exist_ok=True)

    read = [selecting_arm]
    if fill and selecting_arm in IDENTICAL_TO:
        read.append(IDENTICAL_TO[selecting_arm])  # the arm the copies come from
    arms = tuple(dict.fromkeys((*read, *TARGETS)))
    runs = report.current_runs(out_dir)
    select_seeds, eval_seeds = resolve_heldout_seeds(runs, select_seeds, eval_seeds)
    values = seed_values(runs, arms)
    windows = load_windows(out_dir, runs, arms)
    halves = half_values(windows, report._tail_from_by_job(runs)) if windows is not None else None
    filled: list[str] = []
    if fill:
        values, filled = fill_identical(values, selecting_arm)
        if halves is not None:
            halves, _ = fill_identical(halves, selecting_arm)
    tables = variant_tables(values, halves, TARGETS, select_seeds, eval_seeds, selecting_arm)
    train = {arm: wide(values, arm) for arm in (selecting_arm, *TARGETS)}
    parts = [fold_selection(v, sel, tgts, train, float(band), top_k, selecting_arm)
             for v, (sel, tgts, _) in tables.items() if sel is not None]
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ROW_COLS)
    groups = aggregate(rows)

    paths = [rep / "selection_regret.csv", rep / "groups.csv"]
    rows.to_csv(paths[0], index=False)
    groups.to_csv(paths[1], index=False)
    fig = _figure(rep, rows, float(band), selecting_arm=selecting_arm)
    md = _markdown(out_dir, values, tables, rows, groups, float(band), top_k, select_seeds, eval_seeds, fig,
                   selecting_arm, filled, fill)
    (rep / "selection.md").write_text(md, encoding="utf-8")
    paths.append(rep / "selection.md")
    if fig is not None:
        paths.append(fig)
    return paths


# --- markdown ----------------------------------------------------------------------------------------

def _table(df: pd.DataFrame, digits: int = 3) -> str:
    from . import analyze

    return analyze._md_table(df, digits) if len(df) else "_(no rows)_"


def _g(groups: pd.DataFrame, **match) -> pd.DataFrame:
    sub = groups
    for k, v in match.items():
        sub = sub[np.isclose(sub[k].astype(float), v)] if k == "band" else sub[sub[k] == v]
    return sub


def _cell(groups: pd.DataFrame, col: str, **match) -> float:
    sub = _g(groups, **match)
    return float(sub.iloc[0][col]) if len(sub) and pd.notna(sub.iloc[0][col]) else np.nan


def _num(v, digits: int = 3) -> str:
    from . import analyze

    return analyze._fmt(v, digits)


def _words(arm: str) -> dict:
    """How the report names the selecting arm: the default arm keeps the wording of the published
    report, any other is named by its code."""
    if arm == "default":
        return {"by": "defaults", "under": "the defaults", "adj": "default", "desc": "River's defaults"}
    return {"by": f"`{arm}`", "under": f"`{arm}`", "adj": arm, "desc": ARM_WORDS.get(arm, f"the `{arm}` arm")}


def _headline(groups: pd.DataFrame, variant: str, target: str, band: float, arm: str = SELECTING_ARM) -> pd.DataFrame:
    """Per group: the denominators and the expected regret of every selector, fold- and
    block-weighted, and the selecting arm's pick's median fold."""
    out = []
    for grp in (*C.STREAM_GROUPS, ALL):
        base = dict(variant=variant, target=target, band=0.0, level="group", unit=grp)
        row = {"group": grp, "n_folds": _cell(groups, "n_folds", selector=arm, **base),
               "n_blocks": _cell(groups, "n_blocks", selector=arm, **base)}
        for sel in (arm, "random", f"lobo_{arm}", "lobo_target"):
            row[f"{sel} fw"] = _cell(groups, "regret_exp_fw", selector=sel, **base)
            row[f"{sel} bw"] = _cell(groups, "regret_exp_bw", selector=sel, **base)
            if sel == arm:
                row[f"{arm} median"] = _cell(groups, "regret_exp_med", selector=sel, **base)
        out.append(row)
    df = pd.DataFrame(out)
    for c in ("n_folds", "n_blocks"):
        df[c] = df[c].astype("Int64")
    return df


def _agreement(groups: pd.DataFrame, variant: str, target: str, band: float, k: int,
               arm: str = SELECTING_ARM) -> pd.DataFrame:
    """Per group: the selecting arm's pick's agreement with the target, its share of folds losing
    more than b (exact pick, and the band's pick set), and the random pick's readings."""
    out = []
    for grp in (*C.STREAM_GROUPS, ALL):
        base = dict(variant=variant, target=target, level="group", unit=grp, selector=arm)
        rnd = {**base, "selector": "random"}
        row = {"group": grp,
               "top1 fw": _cell(groups, "top1_agree_fw", band=0.0, **base),
               f"top-{k} fw": _cell(groups, "topk_overlap_fw", band=0.0, **base),
               "tau fw": _cell(groups, "tau_fw", band=0.0, **base),
               "tau bw": _cell(groups, "tau_bw", band=0.0, **base),
               "n tau folds": _cell(groups, "n_tau_folds", band=0.0, **base),
               f"share > {band:g}": _cell(groups, "p_regret_gt_b_fw", band=0.0, **base)}
        if band > 0:
            row.update({f"flat within {band:g}": _cell(groups, "n_flat_folds", band=0.0, **base),
                        f"band {band:g}: share > {band:g}": _cell(groups, "p_regret_gt_b_fw", band=band, **base),
                        f"band {band:g}: exp fw": _cell(groups, "regret_exp_fw", band=band, **base),
                        f"band {band:g}: worst fw": _cell(groups, "regret_worst_fw", band=band, **base)})
        row.update({"random top1": _cell(groups, "top1_agree_fw", band=0.0, **rnd),
                    f"random top-{k}": _cell(groups, "topk_overlap_fw", band=0.0, **rnd),
                    f"random share > {band:g}": _cell(groups, "p_regret_gt_b_fw", band=0.0, **rnd)})
        out.append(row)
    df = pd.DataFrame(out)
    for c in ("n tau folds", f"flat within {band:g}"):
        if c in df:
            df[c] = df[c].astype("Int64")
    return df


def _per_fold(rows: pd.DataFrame, variant: str, target: str, band: float, arm: str = SELECTING_ARM) -> pd.DataFrame:
    r = rows[(rows["variant"] == variant) & (rows["target"] == target)]
    ex = r[np.isclose(r["band"], 0.0)].set_index(["selector", "fold"])
    bd = r[np.isclose(r["band"], band)].set_index(["selector", "fold"])
    out = []
    for fold in sorted(set(r["fold"]), key=_fold_key):
        d = ex.loc[(arm, fold)]
        row = {"fold": fold, "group": d["group"], f"{arm} pick": d["picked"], f"{target} best": d["target_best_algos"],
               "regret": d["regret_exp"], "regret worst": d["regret_worst"]}
        if band > 0:
            b = bd.loc[(arm, fold)]
            row.update({f"band n picked": b["n_picked"], "band exp": b["regret_exp"], "band worst": b["regret_worst"]})
        row.update({"random": ex.loc[("random", fold)]["regret_exp"],
                    "lobo_target pick": ex.loc[("lobo_target", fold)]["picked"],
                    "lobo_target regret": ex.loc[("lobo_target", fold)]["regret_exp"],
                    "tau": d["tau"]})
        out.append(row)
    return pd.DataFrame(out)


def _blocks_table(groups: pd.DataFrame, rows: pd.DataFrame, variant: str, target: str,
                  arm: str = SELECTING_ARM) -> pd.DataFrame:
    g = _g(groups, variant=variant, target=target, band=0.0, level="block")
    group_of = {}
    for b, grp in zip(rows["block"], rows["group"]):
        group_of.setdefault(b, grp)
    out = []
    for block in sorted(set(g["unit"]), key=lambda b: (C.STREAM_GROUPS.index(group_of[b]), b)):
        base = dict(variant=variant, target=target, band=0.0, level="block", unit=block)
        out.append({"block": block, "group": group_of[block],
                    "n_folds": _cell(groups, "n_folds", selector=arm, **base),
                    arm: _cell(groups, "regret_exp_fw", selector=arm, **base),
                    "random": _cell(groups, "regret_exp_fw", selector="random", **base),
                    f"lobo_{arm}": _cell(groups, "regret_exp_fw", selector=f"lobo_{arm}", **base),
                    "lobo_target": _cell(groups, "regret_exp_fw", selector="lobo_target", **base),
                    "top1": _cell(groups, "top1_agree_fw", selector=arm, **base),
                    "tau": _cell(groups, "tau_fw", selector=arm, **base)})
    df = pd.DataFrame(out)
    if len(df):
        df["n_folds"] = df["n_folds"].astype("Int64")
    return df


def _targets_table(groups: pd.DataFrame, variant: str, arm: str = SELECTING_ARM) -> pd.DataFrame:
    out = []
    for target in TARGETS:
        for grp in (*C.STREAM_GROUPS, ALL):
            base = dict(variant=variant, target=target, band=0.0, level="group", unit=grp)
            if not len(_g(groups, selector=arm, **base)):
                continue
            out.append({"target": target, "group": grp,
                        "n_folds": _cell(groups, "n_folds", selector=arm, **base),
                        f"{arm} fw": _cell(groups, "regret_exp_fw", selector=arm, **base),
                        f"{arm} bw": _cell(groups, "regret_exp_bw", selector=arm, **base),
                        "random fw": _cell(groups, "regret_exp_fw", selector="random", **base),
                        f"lobo_{arm} fw": _cell(groups, "regret_exp_fw", selector=f"lobo_{arm}", **base),
                        "lobo_target fw": _cell(groups, "regret_exp_fw", selector="lobo_target", **base),
                        "top1 fw": _cell(groups, "top1_agree_fw", selector=arm, **base),
                        "tau fw": _cell(groups, "tau_fw", selector=arm, **base)})
    df = pd.DataFrame(out)
    if len(df):
        df["n_folds"] = df["n_folds"].astype("Int64")
    return df


def _rank_table(groups: pd.DataFrame, variant: str, target: str, arm: str = SELECTING_ARM) -> pd.DataFrame:
    out = []
    for grp in (*C.STREAM_GROUPS, ALL):
        base = dict(variant=variant, target=target, band=0.0, level="group", unit=grp)
        out.append({"group": grp, **{f"{s} fw": _cell(groups, "regret_exp_fw", selector=s, **base)
                                     for s in (f"lobo_{arm}", f"lobo_{arm}_rank", "lobo_target",
                                               "lobo_target_rank")},
                    **{f"{s} bw": _cell(groups, "regret_exp_bw", selector=s, **base)
                       for s in (f"lobo_{arm}", f"lobo_{arm}_rank", "lobo_target", "lobo_target_rank")}})
    return pd.DataFrame(out)


def _variants_table(groups: pd.DataFrame, target: str, band: float, arm: str = SELECTING_ARM) -> pd.DataFrame:
    out = []
    for variant in VARIANTS:
        for grp in (*C.STREAM_GROUPS, ALL):
            base = dict(variant=variant, target=target, band=0.0, level="group", unit=grp)
            if not len(_g(groups, selector=arm, **base)):
                continue
            out.append({"variant": variant, "group": grp,
                        "n_folds": _cell(groups, "n_folds", selector=arm, **base),
                        "n_blocks": _cell(groups, "n_blocks", selector=arm, **base),
                        f"{arm} fw": _cell(groups, "regret_exp_fw", selector=arm, **base),
                        f"{arm} bw": _cell(groups, "regret_exp_bw", selector=arm, **base),
                        f"{arm} median": _cell(groups, "regret_exp_med", selector=arm, **base),
                        f"share > {band:g}": _cell(groups, "p_regret_gt_b_fw", selector=arm, **base),
                        "random fw": _cell(groups, "regret_exp_fw", selector="random", **base),
                        f"lobo_{arm} fw": _cell(groups, "regret_exp_fw", selector=f"lobo_{arm}", **base),
                        "lobo_target fw": _cell(groups, "regret_exp_fw", selector="lobo_target", **base),
                        "top1 fw": _cell(groups, "top1_agree_fw", selector=arm, **base),
                        "tau fw": _cell(groups, "tau_fw", selector=arm, **base)})
    df = pd.DataFrame(out)
    if len(df):
        for c in ("n_folds", "n_blocks"):
            df[c] = df[c].astype("Int64")
    return df


def _expected_count(rows: pd.DataFrame, col: str) -> tuple[float, int]:
    v = pd.to_numeric(rows[col], errors="coerce")
    return float(v.sum()), int(v.notna().sum())


def _reading(groups: pd.DataFrame, rows: pd.DataFrame, band: float, variant: str = "same_data",
             target: str = PRIMARY_TARGET, arm: str = SELECTING_ARM) -> list[str]:
    """The plan's decision, as what the numbers show: per group, the regret of the default
    pick against b and against the two baselines. Small or large is read against b on both
    weightings; when they disagree the reading says so instead of choosing one."""
    w = _words(arm)
    lines = []
    r_all = rows[(rows["variant"] == variant) & (rows["target"] == target) & (rows["selector"] == arm)]
    for grp in (*C.STREAM_GROUPS, ALL):
        base = dict(variant=variant, target=target, band=0.0, level="group", unit=grp)
        n_f = _cell(groups, "n_folds", selector=arm, **base)
        n_b = _cell(groups, "n_blocks", selector=arm, **base)
        if np.isnan(n_f) or not n_f:
            lines.append(f"- `{grp}`: no fold with a defined regret.")
            continue
        fw = _cell(groups, "regret_exp_fw", selector=arm, **base)
        bw = _cell(groups, "regret_exp_bw", selector=arm, **base)
        med = _cell(groups, "regret_exp_med", selector=arm, **base)
        rnd = _cell(groups, "regret_exp_fw", selector="random", **base)
        lt = _cell(groups, "regret_exp_fw", selector="lobo_target", **base)
        ld = _cell(groups, "regret_exp_fw", selector=f"lobo_{arm}", **base)
        r = r_all if grp == ALL else r_all[r_all["group"] == grp]
        exact = r[np.isclose(r["band"], 0.0)]
        over, n_def = _expected_count(exact, "p_regret_gt_b")
        text = (f"- `{grp}` ({int(n_f)} folds, {int(n_b)} blocks): picking by {w['by']} gives up {_num(fw)} tail ARI "
                f"per fold under `{target}` ({_num(bw)} block-weighted, median fold {_num(med)}); the loss exceeds "
                f"b = {band:g} on {over:.1f} of {n_def} folds (in expectation over ties).")
        if band > 0:
            flat = int(sum(v is True or v == True for v in exact["flat_within_b"]))  # noqa: E712
            wide_r = r[np.isclose(r["band"], band)]
            over_b, _ = _expected_count(wide_r, "p_regret_gt_b")
            if flat:
                text += (f" On {flat} of the {n_def} folds every algorithm is within b of the top under "
                         f"{w['under']}, so there the {w['adj']} ranking separates nothing at that scale; counting "
                         f"every algorithm within b of the {w['adj']} top as tied, the loss exceeds b on {over_b:.1f} "
                         "folds.")
            else:
                text += f" No fold has every algorithm within b of the top under {w['under']}."
        cmp_rnd = "less than" if fw < rnd - _EPS else "no less than"
        text += (f" The fold-weighted loss is {cmp_rnd} that of a random pick ({_num(rnd)}); a global choice made "
                 f"on the other blocks loses {_num(lt)} when made from `{target}` and {_num(ld)} when made from "
                 f"{w['under']}.")
        if fw > band and bw > band:
            text += " Both means are above b, so by the plan's rule this group reads as regret large."
        elif fw <= band and bw <= band:
            text += " Both means are at or below b, so by the plan's rule this group reads as regret small."
        else:
            hi = "fold-weighted" if fw > band else "block-weighted"
            lo = "block-weighted" if fw > band else "fold-weighted"
            text += (f" The {hi} mean is above b and the {lo} one is not, so the plan's rule gives no single "
                     "reading for this group.")
        lines.append(text)
    return lines


def _count_word(n: int) -> str:
    return {8: "eight"}.get(int(n), str(int(n)))


def _fill_lines(arm: str, filled, fill: bool) -> list[str]:
    """The decision on copied runs, stated only when it applies to the selecting arm."""
    if arm not in IDENTICAL_TO:
        return []
    other = IDENTICAL_TO[arm]
    if not fill:
        return [f"- Copies are off (`--no-fill-identical`): an algorithm without a `{arm}` run (none is made for an "
                f"algorithm whose `{arm}` configuration is its `{other}` one) is not a candidate under `{arm}`."]
    if not filled:
        return []
    one = len(filled) == 1
    has, them, their = ("has", "it", "its") if one else ("have", "them", "their")
    return [f"- {', '.join(filled)} {has} no absolute-distance parameter, so `run.py` makes no `{arm}` run of "
            f"{them}: {their} `{arm}` configuration is {their} `{other}` one. {their.capitalize()} `{other}` runs "
            f"stand in for it, so every algorithm stays a candidate (`--no-fill-identical` leaves {them} out)."]


def _markdown(out_dir, values, tables, rows, groups, band, k, select_seeds, eval_seeds, fig,
              arm: str = SELECTING_ARM, filled=(), fill: bool = True) -> str:
    n_algos, n_folds = values["algo"].nunique(), values["fold"].nunique()
    n_blocks = len({C.source_block(f) for f in values["fold"]}) if len(values) else 0
    w = _words(arm)
    seeds = {a: sorted(map(int, set(values.loc[values["arm"] == a, "seed"]))) for a in (arm, *TARGETS)}
    seed_txt = ", ".join(f"{a} {','.join(map(str, s))}" for a, s in seeds.items() if s)
    ran = [v for v, (sel, _, _) in tables.items() if sel is not None]
    skipped = {v: note for v, (sel, _, note) in tables.items() if sel is None}
    repeating = repeating_defaults(values, arm)
    if rows.empty:
        lines = [f"# Algorithm selection under {w['by']}, read under online tuning (P1.1)", "",
                 f"Read from `{out_dir}`: no variant could run, so there is no selection regret to report; "
                 "every reading is undefined.", ""]
        lines += [f"- `{v}` did not run: {skipped[v]}." for v in VARIANTS if v in skipped]
        return "\n".join(lines + [""])
    lines = [
        f"# Algorithm selection under {w['by']}, read under online tuning (P1.1)",
        "",
        f"Read from `{out_dir}`: {n_algos} protocol algorithms, {n_folds} folds in {n_blocks} source blocks "
        f"(every INSECTS variant is one block), seeds per arm: {seed_txt}. Implementation variants are not "
        "candidates. Quality is the tail ARI (`lag_ari_win_tail_mean`, the lagged per-window ARI averaged over "
        "windows >= each run's `tail_from`), averaged over seeds. No tuning run was made for this report.",
        "",
        "## What is measured",
        "",
        "On each fold the selector picks an algorithm and the pick is read under a target regime: "
        "`regret = max over algorithms of the target's reading - the target's reading of the pick`. The selector "
        f"of interest is `{arm}`: the algorithm that is best under {w['desc']} on that fold. The primary "
        f"target is `{PRIMARY_TARGET}` (online tuning); `frozen_prefix` and `oracle_static` are sensitivity targets.",
        "",
        "- Ties are public. The pick set at band 0 is every algorithm tied at the top (within 1e-12); "
        "`regret_exp` is the regret of a uniform pick from it, `regret_best`/`regret_worst` its best and worst case. "
        f"At band b = {band:g} (`--band`; the default is the study's pre-declared smallest effect of interest on "
        "tail ARI) every algorithm within b of the top is practically tied and enters the pick set.",
        f"- `top1_agree` is the probability that the pick is among the target's best; `p_regret_gt_b` the "
        f"probability that it loses more than b = {band:g}, read the same way for the exact pick and for the band's "
        "pick set. Their means over folds are the top-1 agreement and the share of folds whose regret exceeds b. "
        "`flat_within_b`: every algorithm is within b of the selector's top, so its ranking separates nothing at "
        "the practical scale and the band's pick set is the random pick.",
        f"- `topk_overlap` = |top-{k}(selector) & top-{k}(target)| / {k}, in expectation over a uniform breaking "
        "of ties at the boundary seat. `tau` is Kendall's tau-b between the selector's scores and the target's "
        "readings, on the algorithms with both; undefined (with the reason) for fewer than 3 such algorithms or "
        "a constant side. Neither depends on the band.",
        f"- Baselines. `random`: a uniform pick over all algorithms (expected regret = max - mean). `lobo_{arm}` "
        "and `lobo_target`: a global choice made on other data, the algorithm with the best mean reading (under "
        f"{w['under']}, under the target) over the folds of every other source block; an INSECTS fold is held "
        "out with all seven INSECTS folds. `lobo_*_rank`: the same by mean rank, a sensitivity. No baseline is "
        "chosen on the test fold.",
        "- Aggregation. `fw` is the mean over the folds of a group, `bw` the mean within each source block, then "
        "over blocks, so the seven INSECTS folds count once. `n_folds`/`n_blocks` are the denominators; an "
        "undefined reading is in neither.",
        "",
        "The `same_data` numbers are a **descriptive selection regret**, not an out-of-sample estimate: the "
        f"selection and the evaluation read the same runs, and the target's maximum over {_count_word(n_algos)} "
        "noisy seed means is "
        "biased upward (the winner's curse), which inflates the regret most where algorithms are close.",
        "",
        "## Decisions taken where the plan is silent",
        "",
        "- Every protocol algorithm is a candidate on every fold; the tunable gate of `report.headroom_table` "
        "concerns one algorithm's configurations and is not applied.",
        "- A pick without a target reading makes the regret undefined (with the reason); it is not replaced by "
        "the next algorithm.",
        "- The global choice averages over folds (the plan's wording), not over blocks; the mean rank is the "
        "sensitivity. It reads the same-data seed means of the other blocks in every variant: it never sees "
        "the test fold, so it is out of sample on it whichever way that fold is split.",
        "- The band is applied to ARI scores only; a rank score's pick set is its exact best at every band.",
        f"- Held-out seeds: select on `{arm}` seed(s) {_seeds_text(select_seeds)}, evaluate every "
        f"target on seeds {_seeds_text(eval_seeds)} (its maximum too). `oracle_static`, rerun on seeds 2-4, "
        "is read on the same evaluation seeds so every target averages the same number of seeds; seed 4 is left out.",
        "- Prefix/suffix: each run's tail windows are split at their middle by window index (the suffix takes the "
        "odd one); single-class windows count in the split but not in the means.",
        *_fill_lines(arm, filled, fill),
        "",
        f"## Headline: picking by {w['by']}, read under `confstream` (same data, exact ties)",
        "",
        "Expected regret (tail ARI) of each selector; `fw` fold-weighted, `bw` block-weighted.",
        "",
        _table(_headline(groups, "same_data", PRIMARY_TARGET, band, arm)),
        "",
        f"Agreement of the {w['adj']} pick with `{PRIMARY_TARGET}`: top-1 agreement, top-{k} overlap, Kendall's tau "
        f"and the share of folds whose regret exceeds b = {band:g}. The band columns are the sensitivity: how many "
        f"folds have every algorithm within b of the top under {w['under']}, and the share above b and the expected "
        f"and worst regret when every algorithm within b of the {w['adj']} top counts as tied. The random pick's "
        "readings are beside them.",
        "",
        _table(_agreement(groups, "same_data", PRIMARY_TARGET, band, k, arm)),
        "",
        "## What the numbers show",
        "",
        "The plan's decision: a small regret lowers the role of the headline ranking; a large one is direct "
        "evidence that evaluating per regime matters in practice. With the pre-declared band b as the line "
        "between small and large:",
        "",
        *_reading(groups, rows, band, arm=arm),
        "",
        "These are same-data readings; the out-of-sample variants below say how much of them survives a split.",
        "",
        f"## Per fold (same data, target `{PRIMARY_TARGET}`)",
        "",
        "`regret`/`regret worst`: exact ties; `band *`: the band's pick set; `random`: expected regret of a uniform "
        f"pick; `lobo_target`: the global choice made from the other blocks' online readings; `tau`: {w['adj']} "
        "against online ranking.",
        "",
        _table(_per_fold(rows, "same_data", PRIMARY_TARGET, band, arm)) if "same_data" in ran else "_(no rows)_",
        "",
        f"## Per source block (same data, target `{PRIMARY_TARGET}`, exact ties)",
        "",
        "A block with several folds (INSECTS) is the mean over its folds; the others are single folds.",
        "",
        _table(_blocks_table(groups, rows, "same_data", PRIMARY_TARGET, arm)),
        "",
        "## Sensitivity targets (same data, exact ties)",
        "",
        _table(_targets_table(groups, "same_data", arm)),
        "",
        "## Global choice by mean score and by mean rank (same data, exact ties, target `confstream`)",
        "",
        _table(_rank_table(groups, "same_data", PRIMARY_TARGET, arm)),
        "",
        "## Out-of-sample variants (target `confstream`)",
        "",
        f"`heldout_seeds` is out of sample in seeds: selection on `{arm}` seed(s) "
        f"{_seeds_text(select_seeds)}, evaluation on seeds {_seeds_text(eval_seeds)}. "
        "`prefix_suffix` is out of sample in time: selection on the first half of each run's tail windows, "
        "evaluation on the second.",
        "",
        _table(_variants_table(groups, PRIMARY_TARGET, band, arm)),
        "",
    ]
    for v in VARIANTS:
        if v in skipped:
            lines.append(f"- `{v}` did not run: {skipped[v]}.")
    if repeating:
        lines.append(f"- The `{arm}` readings of {', '.join(repeating)} repeat exactly over seeds on every "
                     "fold (these algorithms take no seed): for them the seed split holds out only the target's "
                     "tuner noise, not the selecting side's.")
    lines += [
        "- Neither split generalises to new synthetic streams: every synthetic fold was generated from one data "
        "seed, so that would need runs on new data-generator seeds, which these logs do not have.",
        "",
    ]
    if fig is not None:
        lines += ["## Figure", "",
                  f"`{fig.name}`: per fold, Kendall's tau-b between the {w['adj']} and the `{PRIMARY_TARGET}` "
                  f"rankings (left) beside the regret of the {w['adj']} pick under `{PRIMARY_TARGET}` "
                  "(right: dot exact ties, "
                  f"hollow diamond the band-{band:g} pick set, bar the random pick), same data, coloured by stream "
                  "group.", ""]
    lines += ["Files: `selection_regret.csv` (one row per variant x target x band x selector x fold), `groups.csv` "
              "(per stream group and per source block).", ""]
    return "\n".join(lines)


# --- figure ------------------------------------------------------------------------------------------

def _figure(rep: Path, rows: pd.DataFrame, band: float, variant: str = "same_data",
            target: str = PRIMARY_TARGET, selecting_arm: str = SELECTING_ARM) -> Path | None:
    """Per fold, tau (selecting arm against target) beside the regret of the selecting arm's pick."""
    from . import analyze

    arm, adj = selecting_arm, _words(selecting_arm)["adj"]
    plt = analyze._pyplot()
    r = rows[(rows["variant"] == variant) & (rows["target"] == target)] if len(rows) else rows
    if plt is None or r.empty:
        return None
    ex = r[np.isclose(r["band"], 0.0)].set_index(["selector", "fold"])
    bd = r[np.isclose(r["band"], band)].set_index(["selector", "fold"])
    folds = sorted(set(r["fold"]), key=lambda f: (C.STREAM_GROUPS.index(C.stream_group(f)),
                                                  -(ex.loc[(arm, f), "regret_exp"]
                                                    if pd.notna(ex.loc[(arm, f), "regret_exp"]) else -1), f))
    y = np.arange(len(folds))[::-1]
    colors = {g: analyze._SERIES[i] for i, g in enumerate(C.STREAM_GROUPS)}
    with plt.rc_context(analyze._style()):
        fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(10.5, 1.4 + 0.3 * len(folds)), sharey=True,
                                         gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.08})
        reg_max = 0.0
        for yi, fold in zip(y, folds):
            d = ex.loc[(arm, fold)]
            col = colors[C.stream_group(fold)]
            if pd.notna(d["tau"]):
                ax_t.plot([0, d["tau"]], [yi, yi], color=col, linewidth=1.2, alpha=0.5)
                ax_t.scatter([d["tau"]], [yi], s=34, color=col, edgecolor="white", linewidth=1, zorder=3)
            else:
                ax_t.text(0, yi, "undefined", ha="center", va="center", fontsize=7, color=analyze._MUTED)
            rnd = ex.loc[("random", fold), "regret_exp"]
            if pd.notna(rnd):
                ax_r.scatter([rnd], [yi], marker="|", s=90, color=analyze._MUTED, zorder=2)
                reg_max = max(reg_max, float(rnd))
            if pd.notna(d["regret_exp"]):
                if d["regret_worst"] - d["regret_best"] > _EPS:
                    ax_r.plot([d["regret_best"], d["regret_worst"]], [yi, yi], color=col, linewidth=2)
                ax_r.scatter([d["regret_exp"]], [yi], s=34, color=col, edgecolor="white", linewidth=1, zorder=4)
                reg_max = max(reg_max, float(d["regret_worst"]))
            if band > 0 and (arm, fold) in bd.index and pd.notna(bd.loc[(arm, fold), "regret_exp"]):
                v = bd.loc[(arm, fold), "regret_exp"]
                ax_r.scatter([v], [yi], marker="D", s=26, facecolor="none", edgecolor=col, linewidth=1.2, zorder=3)
                reg_max = max(reg_max, float(v))
        groups = [C.stream_group(f) for f in folds]
        for i in range(1, len(folds)):
            if groups[i] != groups[i - 1]:
                for ax in (ax_t, ax_r):
                    ax.axhline(y[i] + 0.5, color=analyze._GRID, linewidth=1)
        ax_t.axvline(0, color=analyze._AXIS, linewidth=1)
        ax_t.set_xlim(-1.05, 1.05)
        ax_t.set_yticks(y, folds)
        ax_t.set_xlabel("Kendall tau-b")
        ax_t.set_title(f"Ranking agreement, {arm} vs {target}")
        if band > 0:
            ax_r.axvline(band, color=analyze._INK_2, linewidth=1, linestyle=(0, (4, 3)))
            ax_r.text(band, y.max() + 0.7, f" b = {band:g}", fontsize=7, color=analyze._INK_2, va="bottom")
        ax_r.set_xlim(-0.01, max(reg_max * 1.08, band * 1.5, 0.05))
        ax_r.set_xlabel("tail ARI given up")
        ax_r.set_title(f"Regret of the {adj} pick under {target}")
        for ax in (ax_t, ax_r):
            ax.grid(axis="x", alpha=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[g], label=g) for g in C.STREAM_GROUPS
                   if g in groups]
        handles += [plt.Line2D([], [], marker="o", linestyle="", color=analyze._INK_2, label=f"{adj} pick (exact)"),
                    plt.Line2D([], [], marker="D", linestyle="", markerfacecolor="none",
                               markeredgecolor=analyze._INK_2, label=f"band-{band:g} pick set (expected)"),
                    plt.Line2D([], [], marker="|", linestyle="", markersize=10, color=analyze._MUTED,
                               label="random pick (expected)")]
        if band <= 0:
            handles = [h for h in handles if not h.get_label().startswith("band")]
        ax_r.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        path = rep / "F_selection_regret.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return path


# --- LOBO on real sources, each source block weighted once (the campaign's primary selection reading) ---
#
# ``write_selection`` above is the published P1.1 reading (fold-weighted global choices over every
# stream group). ``lobo_real`` is the mechanism campaign's: the global choice is made from the real
# sources other than the held-out one, every training source block weighs the same (its variants are
# averaged first), every selector reads the same training panel (the same sources, variants, windows
# and candidate algorithms), and the held-out source is never read before the choice is made.

#: The selectors of ``lobo_real`` and the regime each one picks by (None: a uniform pick).
LOBO_SELECTORS: dict[str, str | None] = {
    "lobo_default": "default", "lobo_scaled_default": "scaled_default", "lobo_target": PRIMARY_TARGET, "random": None,
}
#: The regimes the training panel must hold for an algorithm to be a candidate.
LOBO_REGIMES: tuple[str, ...] = ("default", "scaled_default", PRIMARY_TARGET)
#: The primary difference: ``Q_online,test(a_lobo_target) - Q_online,test(a_lobo_scaled_default)``.
LOBO_PRIMARY: tuple[str, str] = ("lobo_target", "lobo_scaled_default")
#: Further paired differences reported beside it.
LOBO_CONTRASTS: tuple[tuple[str, str], ...] = (
    LOBO_PRIMARY, ("lobo_target", "lobo_default"), ("lobo_scaled_default", "lobo_default"), ("lobo_target", "random"),
)
#: Two training scores closer than this are tied; a tie goes to the algorithm whose name sorts first.
LOBO_TIE_TOL = 1e-12
LOBO_TIE_POLICY = (f"highest training score; scores within {LOBO_TIE_TOL:g} are tied and the tie goes to the "
                   "algorithm name that sorts first; the held-out source is never read")
LOBO_FOLD_COLS = [
    "analysis", "variant", "held_out", "selector", "regime", "panel_status", "pick", "n_tied", "tied", "band_set",
    "train_score", "n_candidates", "candidates", "excluded", "expected_train_blocks", "incomplete_train_blocks",
    "n_train_blocks", "train_blocks", "excluded_train_blocks", "n_train_folds", "train_windows",
    "excluded_train_folds", "missing", "test_status", "test_missing", "test_folds", "test_windows", "q_test",
    "best_test", "best_test_algos", "regret", "n_test_defined", "note",
]
LOBO_STATUS_COLS = [
    "variant", "analysis", "plan_source", "analysis_status", "n_blocks_expected", "n_blocks_eligible",
    "n_blocks_test_available", "expected_blocks", "incomplete_blocks", "blocks_without_test", "missing_regimes",
    "missing", "n_windows_partial_replicates", "partial_replicates", "reason",
]
#: What a reduced-panel summary row says it is: never the primary reading.
REDUCED_STATUS = "sensitivity: reduced panel"
#: A leave-one-source-out row of a variant whose primary reading is not eligible: not computed.
LOSO_NOT_READ = "not read: primary not eligible"


# --- the expected panel: read from the plan, before any result ------------------------------------

def lobo_plan(algos, folds, *, source: str, regimes=LOBO_REGIMES, jobs: dict | None = None,
              planned=None) -> dict:
    """The panel the LOBO reading expects, from a plan rather than from the results on file:
    ``source`` (where the plan was read), ``algos``, ``blocks`` (real source block -> its folds) and
    ``cells``: (algo, real fold, regime) -> ``{"jobs": [job identities] or None, "via": regime, "planned": bool}``.
    ``jobs`` maps a cell to the job identities the plan runs for it (None: the plan names cells only, and
    every run of the cell on file must be ok). ``scaled_default`` of an algorithm without an absolute-distance
    parameter reads its ``default`` cell (``via``), as ``run.py`` skips it as identical. ``planned``: the
    (algo, fold, arm) cells the plan runs (None: every cell); one it does not is required all the same.
    ``problems``: why the plan cannot carry a primary reading at all (a label variant beside its reference;
    ``read_lobo_plan`` adds a config's protocol mismatches)."""
    from .spaces import distance_params

    real = sorted({f for f in folds if _real(f)}, key=_fold_key)
    blocks: dict[str, list[str]] = {}
    for f in real:
        blocks.setdefault(C.source_block(f), []).append(f)
    algos = sorted({a for a in algos if a not in C.IMPL_VARIANTS})
    problems = [f"{v} and {ref} are both candidates: {v} is {ref} under other labels, so the same model would "
                f"count twice; a plan reads one of them" for v, ref in sorted(C.LABEL_VARIANTS.items())
                if v in algos and ref in algos]
    cells = {}
    for a in algos:
        for f in real:
            for r in regimes:
                via = "default" if r == "scaled_default" and not distance_params(a) else r
                cells[(a, f, r)] = {"via": via, "jobs": None if jobs is None else list(jobs.get((a, f, via), [])),
                                    "planned": planned is None or (a, f, via) in planned}
    return {"source": source, "algos": algos, "blocks": blocks, "cells": cells, "problems": problems}


def plan_from_inventory(inv: dict, source: str) -> dict:
    """``lobo_plan`` from a campaign inventory (``campaign.inventory``: ``jobs``, ``skipped``): the job
    identities of every cell, the algorithms and real folds it plans."""
    from .analysis_support import seed_key

    jobs: dict[tuple, list] = {}
    for j in inv.get("jobs", []):
        if j["arm"] in LOBO_REGIMES:
            ident = (j["algo"], j["fold"], j["arm"], seed_key(j.get("seed")), seed_key(j.get("model_seed")),
                     seed_key(j.get("config_id")))
            jobs.setdefault((j["algo"], j["fold"], j["arm"]), []).append(ident)
    every = list(inv.get("jobs", [])) + list(inv.get("skipped", []))
    algos = {j["algo"] for j in every}
    folds = {j["fold"] for j in every}
    return lobo_plan(algos, folds, source=source, jobs=jobs, planned=set(jobs))


def read_lobo_plan(out_dir: Path, config: Path | None = None) -> dict | None:
    """The expected panel of ``out_dir``: a campaign config given (``config``), else the campaign manifest
    in the dir (its inventory names every job), else the run manifest (algorithms x streams x arms, cell
    level). None when the dir holds no plan at all."""
    from . import campaign as CP
    from . import manifest as M

    out_dir = Path(out_dir)
    if config is not None:
        cfg = CP.load_config(Path(config))
        plan = plan_from_inventory(CP.inventory(cfg, out_dir), source=f"campaign config {config}")
        plan["problems"] = plan["problems"] + plan_protocol_problems(cfg, out_dir)
        return plan
    camp = CP.load_manifest(out_dir)
    if camp is not None:
        return plan_from_inventory(camp.get("inventory", {}), source=f"campaign manifest {out_dir / CP.CAMPAIGN_MANIFEST}")
    rm = M.load(out_dir)
    if rm is None:
        return None
    folds = {str(k).split("/", 1)[0] for k in rm.get("streams", {})}
    arms = set(rm.get("arms", {}))
    planned = {(a, f, r) for a in rm.get("algorithms", {}) for f in folds for r in arms}
    return lobo_plan(rm.get("algorithms", {}), folds, source=f"run manifest {out_dir / M.MANIFEST} (cell level)",
                     planned=planned)


#: Campaign protocol keys the run manifest records under the same name (``manifest.json``'s ``protocol``).
_PLAN_PROTOCOL_KEYS = ("h", "freeze_after_window", "n_samples", "half_life_frac", "lam", "data_seed", "seed_policy",
                       "max_memory_mb", "max_micro_clusters")


def plan_protocol_problems(cfg: dict, out_dir: Path) -> list[str]:
    """Why a campaign config given from outside may not be the plan of ``out_dir``'s results: a job is
    named by (algo, fold, arm, seeds, config) only, so a dir run under another protocol would otherwise
    satisfy it. Compared: the dir's campaign manifest's config hash, and the run manifest's protocol (the
    keys both record, and the runtime gate). A dir without a run manifest cannot be checked."""
    from . import campaign as CP
    from . import manifest as M

    out = []
    camp = CP.load_manifest(out_dir)
    if camp is not None and camp.get("config_sha256") != CP.config_sha256(cfg):
        out.append(f"the dir's campaign manifest was made from another config ({str(camp.get('config_sha256'))[:12]})")
    rm = M.load(out_dir)
    if rm is None:
        return out + ["the dir has no run manifest: its protocol cannot be checked against the plan"]
    have, want = rm.get("protocol", {}), cfg["protocol"]
    for k in _PLAN_PROTOCOL_KEYS:
        if k in have and k in want and have[k] != want[k]:
            out.append(f"protocol.{k}: the dir ran {have[k]!r}, the plan says {want[k]!r}")
    if "runtime_factor" in have and (have["runtime_factor"] is not None) != bool(want.get("runtime_gate")):
        out.append(f"runtime gate: the dir ran runtime_factor={have['runtime_factor']!r}, the plan says "
                   f"runtime_gate={want.get('runtime_gate')!r}")
    # what each arm ran under (``manifest.json``'s ``arms``): none of it is in a job's identity either
    for arm in cfg.get("arms", []):
        got = rm.get("arms", {}).get(arm)
        if got is None:
            continue
        expect = {"log_signals": sorted(want.get("signals") or []), "signal_noise": want.get("signal_noise"),
                  "cmm": bool(want.get("cmm"))}
        seen = {"log_signals": sorted(got.get("log_signals") or []), "signal_noise": got.get("signal_noise"),
                "cmm": "cmm" in (got.get("external_metrics") or [])}
        if arm not in C.FIXED_ARMS:
            expect.update(e_size=want.get("e_size"), e_test=want.get("e_test"), p_reset=want.get("p_reset"),
                          tie_break=want.get("tie_break"))
            seen.update({k: got.get(k) for k in ("e_size", "e_test", "p_reset", "tie_break")})
        for k, v in expect.items():
            if k in ("signal_noise",) and seen[k] is None:
                continue
            if seen[k] != v:
                out.append(f"arms.{arm}.{k}: the dir ran {seen[k]!r}, the plan says {v!r}")
        if (got.get("runtime_factor") is not None) != bool(want.get("runtime_gate")):
            out.append(f"arms.{arm}.runtime_factor: the dir ran {got.get('runtime_factor')!r}, the plan says "
                       f"runtime_gate={want.get('runtime_gate')!r}")
    return out


def _latest_status(runs: pd.DataFrame) -> dict[tuple, str]:
    """Job identity (``analysis_support.IDENTITY``) -> the status of its latest attempt; skips left out."""
    from .analysis_support import IDENTITY, seed_key

    latest: dict[tuple, str] = {}
    if len(runs):
        for r in runs.to_dict("records"):
            if r.get("status") == "skipped":
                continue
            latest[tuple(seed_key(r[k]) if k in ("seed", "model_seed", "config_id") else r[k] for k in IDENTITY)] = (
                r.get("status"))
    return latest


def cell_jobs(plan: dict, runs: pd.DataFrame) -> dict[tuple, list[tuple]]:
    """(algo, fold, regime) -> the job identities the cell needs (arm: the regime it is read through).
    A plan with job lists (a campaign's) names them. A plan that names cells only (a run manifest's, which
    records the union of the seeds over all jobs, not per algorithm) expects, for each (algo, arm), every
    replicate (seed, model seed, config) that pair ran on any planned real fold: a replicate one fold
    lacks is then a missing job, though a replicate run nowhere cannot be told from one never planned."""
    folds = {f for fs in plan["blocks"].values() for f in fs}
    reps: dict[tuple, set] = {}
    for a, f, arm, *rep in _latest_status(runs):
        if f in folds:
            reps.setdefault((a, arm), set()).add(tuple(rep))
    out = {}
    for (a, f, r), cell in plan["cells"].items():
        via = cell["via"]
        if cell["jobs"] is not None:
            out[(a, f, r)] = [tuple(j) for j in cell["jobs"]]
        else:
            out[(a, f, r)] = [(a, f, via, *rep) for rep in sorted(reps.get((a, via), ()), key=repr)]
    return out


def cell_gaps(plan: dict, runs: pd.DataFrame) -> dict[tuple, str]:
    """(algo, fold, regime) -> why the plan's cell has no usable job: ``not planned``, ``not on file`` or
    the statuses of its jobs that are not ok (``error``, ``timeout``, ``oom``, ``not on file``) with
    their count (``cell_jobs``). A cell whose jobs are all ok is absent. Never read as a score."""
    latest = _latest_status(runs)
    jobs = cell_jobs(plan, runs)
    out = {}
    for (a, f, r), cell in plan["cells"].items():
        via = cell["via"]
        if not cell["planned"]:
            out[(a, f, r)] = "not planned"
            continue
        statuses = [latest.get(j, "not on file") for j in jobs[(a, f, r)]]
        if not statuses:
            out[(a, f, r)] = "not on file"
            continue
        bad = [s for s in statuses if s != "ok"]
        if bad:
            kinds = sorted(set(bad))
            out[(a, f, r)] = (", ".join(kinds) if len(kinds) > 1 else kinds[0]) + f" ({len(bad)} of {len(statuses)} jobs)"
            if via != r:
                out[(a, f, r)] += f", read through {via}"
    return out


def planned_windows(tw: pd.DataFrame, plan: dict, runs: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    """``tw`` kept to the plan's jobs (``cell_jobs``; a run the plan does not name is never averaged in) and
    to the windows every job of the cell reads (a window some replicate of the cell does not read is left
    out of that cell, never averaged over the replicates that do); and per (algo, fold, regime) how many
    windows that left out. A row ``fill_identical_windows`` copied stands for the job it was copied from."""
    from .analysis_support import seed_key

    if tw.empty:
        return tw, {}
    jobs = cell_jobs(plan, runs)
    wanted = {j for js in jobs.values() for j in js}
    src = tw["filled_from"] if "filled_from" in tw else pd.Series([None] * len(tw), index=tw.index)
    keys = [(a, f, s if isinstance(s, str) else arm, seed_key(sd), seed_key(ms), seed_key(cid))
            for a, f, arm, s, sd, ms, cid in zip(tw["algo"], tw["fold"], tw["arm"], src, tw["seed"],
                                                  tw["model_seed"], tw["config_id"])]
    sub = tw[[k in wanted for k in keys]]
    sub = sub[sub[metric].notna()]
    n = sub.groupby(["algo", "fold", "arm", "window"]).size().to_dict()
    need = {k: len(v) for k, v in jobs.items()}
    full = [n[(a, f, r, w)] == need.get((a, f, r), -1) for a, f, r, w in zip(sub["algo"], sub["fold"], sub["arm"],
                                                                             sub["window"])]
    kept = sub[full]
    partial = {}
    for (a, f, r), g in sub[[not x for x in full]].groupby(["algo", "fold", "arm"]):
        partial[(a, f, r)] = int(g["window"].nunique())
    return kept, partial


def regime_window_values(tw: pd.DataFrame, regime: str, metric: str) -> dict[tuple, pd.Series]:
    """(algo, fold) -> window -> ``regime``'s reading of ``metric``: the mean over its replicates on the
    window (``analysis_support.arm_windows``: tuner seeds of an online arm, the fixed runs of a fixed one)."""
    from .analysis_support import arm_windows

    aw = arm_windows(tw, regime, metric)
    out = {}
    for (algo, fold), g in aw.groupby(["algo", "fold"], sort=True):
        out[(algo, fold)] = g.groupby("window")["value"].mean().sort_index()
    return out


def _lobo_pick(scores: dict[str, float], band: float) -> dict:
    """The pick of a score dict under ``LOBO_TIE_POLICY``, the algorithms tied with it, and the band set."""
    if not scores:
        return {"pick": None, "tied": [], "band_set": [], "train_score": np.nan}
    top = max(scores.values())
    tied = sorted(a for a, v in scores.items() if v >= top - LOBO_TIE_TOL)
    return {"pick": tied[0], "tied": tied, "train_score": float(top),
            "band_set": sorted(a for a, v in scores.items() if v >= top - band - LOBO_TIE_TOL)}


def _half(windows: list, half: str | None) -> list:
    if half is None:
        return windows
    k = len(windows) // 2
    return windows[:k] if half == "prefix" else windows[k:]


def _has(values: dict, regime: str, algo: str, fold: str) -> bool:
    s = values.get(regime, {}).get((algo, fold))
    return s is not None and not s.dropna().empty


def lobo_once(values: dict[str, dict[tuple, pd.Series]], blocks: dict[str, list[str]], algos: list[str],
              held_out: str, band: float = BAND, test_half: str | None = None, variant: str = "same_data",
              gaps: dict[tuple, str] | None = None, via: dict[tuple, str] | None = None, reduced: bool = False,
              plan_known: bool = True, plan_problems=()) -> tuple[list[dict], list[dict]]:
    """One held-out source block: ``(rows, training_scores)``.

    ``blocks`` and ``algos`` are the plan's (``lobo_plan``), never what the results hold, and ``gaps``
    the plan's cells without a usable job (``cell_gaps``). The training panel is every fold of every
    other block, every algorithm and every regime of ``LOBO_REGIMES``. In the primary reading a cell of
    it without an ok job or without a tail reading, or a training fold without a window every algorithm
    reads under every regime, makes the block's panel ``ineligible``: its rows choose nothing and say
    why (``missing``, ``incomplete_train_blocks``). ``reduced`` (a sensitivity, asked for by name)
    instead excludes such an algorithm for every selector alike and leaves such a fold out, and says
    what it left out (``excluded``, ``excluded_train_blocks``, ``excluded_train_folds``).
    ``plan_known`` False (no plan on file) makes every primary panel ineligible; a ``plan_problems`` entry
    (``plan_protocol_problems``: the plan is not the dir's protocol) makes the reduced panel ineligible too.

    On each training fold the windows are those every candidate reads under every regime. A regime's
    training score of an algorithm is the mean over training blocks of the mean over the block's folds;
    ties follow ``LOBO_TIE_POLICY``. The test reads ``PRIMARY_TARGET`` on the held-out block only, on the
    windows every candidate reads there (``test_half``: the first or second half of them). A candidate
    whose test job is missing or failed, or that has no reading there, leaves the block's test quality
    ``unavailable`` with the reason (``test_missing``); the choice and the candidates stay what the
    training data made them, and nothing is imputed.
    """
    gaps, via = gaps or {}, via or {}
    regimes = list(LOBO_REGIMES)
    train_blocks = {b: fs for b, fs in blocks.items() if b != held_out}
    train_folds = [f for b in sorted(train_blocks) for f in train_blocks[b]]
    block_of = {f: b for b, fs in train_blocks.items() for f in fs}

    def gap(a, f, r):
        why = gaps.get((a, f, r))
        if why is None and not _has(values, r, a, f):
            why = "ok job without a tail reading" + (f" (read through {via[(a, f, r)]})"
                                                     if via.get((a, f, r), r) != r else "")
        return why

    missing, incomplete, bad_algo = [], set(), {}
    if not plan_known:
        missing.append("no plan on file: the expected panel cannot be checked")
    missing += [f"plan: {why}" for why in plan_problems]
    for f in train_folds:
        for a in algos:
            for r in regimes:
                why = gap(a, f, r)
                if why is not None:
                    missing.append(f"{a}/{f}/{r}: {why}")
                    incomplete.add(block_of[f])
                    bad_algo.setdefault(a, f"{r} on {f}: {why}")
    excluded = bad_algo if reduced else {}
    cands = [a for a in algos if a not in excluded]
    support, dropped = {}, []
    for f in train_folds:
        sets = [set(values[r][(a, f)].dropna().index) for a in cands for r in regimes if _has(values, r, a, f)]
        complete = bool(cands) and len(sets) == len(cands) * len(regimes)
        common = set.intersection(*sets) if complete else set()
        if common:
            support[f] = sorted(common)
        else:
            dropped.append(f)
            if complete:
                missing.append(f"{f}: no tail window every candidate reads under every regime")
                incomplete.add(block_of[f])
    choose = (reduced and not plan_problems) or not missing
    panel_status = "eligible" if not missing else ("reduced" if reduced and choose else "ineligible")
    scores_rows = []
    block_scores: dict[str, dict[str, dict[str, float]]] = {r: {a: {} for a in cands} for r in regimes}
    if choose:
        for r in regimes:
            for a in cands:
                for b, fs in sorted(train_blocks.items()):
                    qs = [float(values[r][(a, f)].loc[support[f]].mean()) for f in fs if f in support]
                    if qs:
                        block_scores[r][a][b] = float(np.mean(qs))
                        scores_rows.append({"analysis": "reduced_panel" if reduced else "primary", "variant": variant,
                                            "held_out": held_out, "regime": r, "algo": a, "train_block": b,
                                            "n_folds": len(qs), "score": block_scores[r][a][b]})
    used_blocks = sorted({b for r in regimes for a in cands for b in block_scores[r][a]})

    test_folds = blocks.get(held_out, [])
    tgt = values.get(PRIMARY_TARGET, {})
    test_missing = [] if test_folds else ["the held-out source has no real fold in the plan"]
    for f in test_folds:
        for a in cands:
            why = gap(a, f, PRIMARY_TARGET)
            if why is not None:
                test_missing.append(f"{a}/{f}/{PRIMARY_TARGET}: {why}")
    q_fold: dict[str, dict[str, float]] = {a: {} for a in cands}
    test_windows = 0
    if not test_missing and cands:
        for f in test_folds:
            common = set.intersection(*[set(tgt[(a, f)].dropna().index) for a in cands])
            wins = _half(sorted(common), test_half)
            if not wins:
                test_missing.append(f"{f}: no {PRIMARY_TARGET} window every candidate reads"
                                    + (f" ({test_half} half)" if test_half else ""))
                continue
            test_windows += len(wins)
            for a in cands:
                q_fold[a][f] = float(tgt[(a, f)].loc[wins].mean())
    test_ok = bool(cands) and not test_missing
    q_test = {a: (float(np.mean([q_fold[a][f] for f in test_folds])) if test_ok else np.nan) for a in cands}
    defined = {a: v for a, v in q_test.items() if not np.isnan(v)}
    best = max(defined.values()) if defined else np.nan
    best_algos = sorted(a for a, v in defined.items() if v >= best - LOBO_TIE_TOL) if defined else []
    rows = []
    base = {"analysis": "reduced_panel" if reduced else "primary", "variant": variant, "held_out": held_out,
            "panel_status": panel_status, "n_candidates": len(cands), "candidates": ";".join(cands),
            "excluded": ";".join(f"{a} ({why})" for a, why in sorted(excluded.items())),
            "expected_train_blocks": ";".join(sorted(train_blocks)),
            "incomplete_train_blocks": ";".join(sorted(incomplete)),
            "n_train_blocks": len(used_blocks), "train_blocks": ";".join(used_blocks),
            "excluded_train_blocks": ";".join(sorted(set(train_blocks) - set(used_blocks))),
            "n_train_folds": len(support) if choose else 0,
            "train_windows": int(sum(len(v) for v in support.values())) if choose else 0,
            "excluded_train_folds": ";".join(dropped), "missing": "; ".join(missing),
            "test_status": "available" if test_ok else "unavailable", "test_missing": "; ".join(test_missing),
            "test_folds": ";".join(test_folds), "test_windows": test_windows, "best_test": best,
            "best_test_algos": ";".join(best_algos), "n_test_defined": len(defined)}
    for selector, regime in LOBO_SELECTORS.items():
        row = {**base, "selector": selector, "regime": regime}
        notes = []
        if not choose:
            row.update(pick=None, n_tied=None, tied=None, band_set=None, train_score=np.nan, q_test=np.nan)
            notes.append("primary analysis not eligible: the expected training panel is incomplete (see missing); "
                         "nothing is chosen from a narrowed panel")
        elif selector == "random":
            row.update(pick=None, n_tied=len(cands), tied=";".join(cands), band_set=";".join(cands),
                       train_score=np.nan)
            row["q_test"] = float(np.mean(list(defined.values()))) if test_ok else np.nan
        else:
            scores = {a: float(np.mean(list(block_scores[regime][a].values())))
                      for a in cands if block_scores[regime][a]}
            p = _lobo_pick(scores, band)
            row.update(pick=p["pick"], n_tied=len(p["tied"]), tied=";".join(p["tied"]),
                       band_set=";".join(p["band_set"]), train_score=p["train_score"])
            row["q_test"] = q_test.get(p["pick"], np.nan) if p["pick"] is not None else np.nan
            if p["pick"] is None:
                notes.append("no candidate")
        if choose and not test_ok:
            notes.append("test quality unavailable, the choice kept: " + "; ".join(test_missing or ["no candidate"]))
        row["regret"] = (best - row["q_test"]) if not np.isnan(row["q_test"]) and not np.isnan(best) else np.nan
        row["note"] = "; ".join(notes) or None
        rows.append(row)
    return rows, scores_rows


def _lobo_differences(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, held-out block): the paired test differences ``LOBO_CONTRASTS``, each regret, and
    the block's panel and test status."""
    out = []
    for (variant, held), g in rows.groupby(["variant", "held_out"], sort=True):
        q = dict(zip(g["selector"], g["q_test"]))
        picks = dict(zip(g["selector"], g["pick"]))
        row = {"variant": variant, "held_out": held, "panel_status": g["panel_status"].iloc[0],
               "test_status": g["test_status"].iloc[0],
               **{f"pick_{s}": picks.get(s) for s in LOBO_SELECTORS if s != "random"}}
        for a, b in LOBO_CONTRASTS:
            qa, qb = q.get(a, np.nan), q.get(b, np.nan)
            row[f"{a}_minus_{b}"] = (qa - qb) if not (pd.isna(qa) or pd.isna(qb)) else np.nan
        for s in LOBO_SELECTORS:
            row[f"regret_{s}"] = g.loc[g["selector"] == s, "regret"].iloc[0] if (g["selector"] == s).any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def _lobo_summary(diffs: pd.DataFrame, threshold: float = BAND, label: str = "all sources",
                  n_expected: int | None = None, reduced: bool = False) -> list[dict]:
    """The mean over held-out blocks of every difference and regret. The primary reading is
    ``eligible`` only when every expected block (``n_expected``) has a complete training panel and a
    test reading; otherwise its numbers are left undefined and ``analysis_status`` says so. A reduced
    panel (``reduced``) is summarised over the blocks it defines and labelled ``REDUCED_STATUS``."""
    from .analysis_support import block_bootstrap

    out = []
    for variant, g in diffs.groupby("variant", sort=True):
        n_exp = len(g) if n_expected is None else int(n_expected)
        whole = (len(g) == n_exp and (g["panel_status"] == "eligible").all() and (g["test_status"] == "available").all())
        status = REDUCED_STATUS if reduced else ("eligible" if whole else "ineligible")
        read = reduced or whole
        reason = None if whole else (
            f"{int((g['panel_status'] != 'eligible').sum())} of {n_exp} held-out blocks lack a complete training "
            f"panel, {int((g['test_status'] != 'available').sum())} lack a test reading")
        for col in [f"{a}_minus_{b}" for a, b in LOBO_CONTRASTS] + [f"regret_{s}" for s in LOBO_SELECTORS]:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            row = {"variant": variant, "analysis": label, "difference": col, "analysis_status": status,
                   "n_blocks_expected": n_exp, "n_blocks": int(len(v)), "n_blocks_undefined": int(g[col].isna().sum()),
                   "status_reason": reason}
            if read:
                row.update(mean=float(v.mean()) if len(v) else np.nan, median=float(v.median()) if len(v) else np.nan)
                if not col.startswith("regret_"):
                    lo, hi = block_bootstrap(v)
                    row.update(ci_low=lo, ci_high=hi, n_above_0=int((v > _EPS).sum()), n_below_0=int((v < -_EPS).sum()),
                               n_above_threshold=int((v > threshold + _EPS).sum()),
                               n_below_threshold=int((v < -threshold - _EPS).sum()))
            else:
                row.update(mean=np.nan, median=np.nan)
            out.append(row)
    return out


def _lobo_status(rows: pd.DataFrame, plan: dict, blocks: dict, analysis: str, reduced: bool,
                 partial: dict | None = None) -> list[dict]:
    """One row per variant: whether the reading is eligible, over which expected blocks, and what is missing;
    ``partial`` (``planned_windows``): the windows left out of a cell because a replicate did not read them."""
    partial = partial or {}
    part_text = "; ".join(f"{a}/{f}/{r}: {n}" for (a, f, r), n in sorted(partial.items()))
    out = []
    for variant, g in rows.groupby("variant", sort=True):
        per = g.groupby("held_out", sort=True).first()
        miss = sorted({m for col in ("missing", "test_missing") for v in g[col].dropna() for m in str(v).split("; ") if m})
        regimes = sorted({m.split(":", 1)[0].split("/")[2] for m in miss if m.count("/") >= 2 and ":" in m})
        incomplete = sorted({b for v in per["incomplete_train_blocks"].dropna() for b in str(v).split(";") if b})
        no_test = sorted(per.index[per["test_status"] != "available"])
        n_ok = int((per["panel_status"] == "eligible").sum())
        n_test = int((per["test_status"] == "available").sum())
        whole = n_ok == len(blocks) and n_test == len(blocks) and len(per) == len(blocks)
        out.append({
            "variant": variant, "analysis": analysis, "plan_source": plan["source"],
            "analysis_status": REDUCED_STATUS if reduced else ("eligible" if whole else "ineligible"),
            "n_blocks_expected": len(blocks), "n_blocks_eligible": n_ok, "n_blocks_test_available": n_test,
            "expected_blocks": ";".join(sorted(blocks)), "incomplete_blocks": ";".join(incomplete),
            "blocks_without_test": ";".join(no_test), "missing_regimes": ";".join(regimes), "missing": "; ".join(miss),
            "n_windows_partial_replicates": int(sum(partial.values())), "partial_replicates": part_text,
            "reason": ("every expected block has a complete training panel and a test reading" if whole else
                       f"{len(blocks) - n_ok} of {len(blocks)} held-out blocks lack a complete training panel, "
                       f"{len(blocks) - n_test} lack a test reading"),
        })
    return out


def _observed_plan(tw: pd.DataFrame) -> dict:
    folds = sorted({f for f in tw["fold"]}) if len(tw) else []
    algos = sorted({a for a in tw["algo"]}) if len(tw) else []
    return lobo_plan(algos, folds, source="none: read off the results on file (no plan given)")


def lobo_real(tw: pd.DataFrame, metric: str = "lag_ari_win", band: float = BAND, folds=None,
              sensitivity_halves: bool = True, plan: dict | None = None, runs: pd.DataFrame | None = None,
              reduced_panel: bool = False) -> dict[str, pd.DataFrame]:
    """The LOBO reading of tail windows ``tw`` (``analysis_support.tail_windows``, identical scaled
    defaults filled): real folds only, each held out with its whole source block.

    ``plan`` (``lobo_plan``, ``read_lobo_plan``) names the blocks, folds, algorithms and regimes the
    reading expects and, with ``runs`` (``analysis_support.run_table``), the jobs it needs; without a
    plan the panel is read off the results and the primary reading is never eligible.

    Returns ``folds`` (one row per variant x held-out block x selector), ``training_scores``,
    ``differences`` (per held-out block), ``summary`` (mean over blocks, undefined unless the primary
    reading is eligible), ``status`` (per variant: eligible or not, expected and incomplete blocks,
    missing regimes and jobs) and ``loso`` (the summary again with each source removed from the plan,
    Shuttle included). Variants: ``same_data`` (primary: every tail window of the held-out source) and,
    with ``sensitivity_halves``, ``suffix_eval`` (the second half of its common windows only).
    ``reduced_panel`` adds ``reduced``: the same tables on the panel that excludes what is missing, a
    sensitivity that never replaces the primary reading."""
    known = plan is not None
    plan = plan if known else _observed_plan(tw)
    blocks = {b: [f for f in fs if folds is None or f in set(folds)] for b, fs in plan["blocks"].items()}
    blocks = {b: fs for b, fs in blocks.items() if fs}
    algos = list(plan["algos"])
    gaps = cell_gaps(plan, runs) if runs is not None else {}
    partial: dict = {}
    if runs is not None:
        tw, partial = planned_windows(tw, plan, runs, metric)
    via = {k: c["via"] for k, c in plan["cells"].items()}
    values = {r: regime_window_values(tw, r, metric) for r in LOBO_REGIMES if (tw["arm"] == r).any()} if len(tw) else {}
    variants = [("same_data", None)] + ([("suffix_eval", "suffix")] if sensitivity_halves else [])

    def run(bl: dict[str, list[str]], reduced: bool = False):
        rows, scores = [], []
        for variant, half in variants:
            for held in sorted(bl):
                r, s = lobo_once(values, bl, algos, held, band, half, variant, gaps, via, reduced, known,
                                 plan.get("problems", ()))
                rows += r
                scores += s
        df = pd.DataFrame(rows).reindex(columns=LOBO_FOLD_COLS)
        df["n_tied"] = pd.array(df["n_tied"], dtype="Int64")      # None where nothing was chosen
        return df, pd.DataFrame(scores)

    def tables(bl, reduced=False):
        rows, scores = run(bl, reduced)
        diffs = _lobo_differences(rows) if len(rows) else pd.DataFrame()
        summary = pd.DataFrame(_lobo_summary(diffs, band, n_expected=len(bl), reduced=reduced) if len(diffs) else [])
        status = pd.DataFrame(_lobo_status(rows, plan, bl, "reduced_panel" if reduced else "primary", reduced, partial)
                              if len(rows) else [], columns=LOBO_STATUS_COLS)
        return rows, scores, diffs, summary, status

    rows, scores, diffs, summary, status = tables(blocks)
    eligible = set(status.loc[status["analysis_status"] == "eligible", "variant"]) if len(status) else set()
    loso = []
    for gone in sorted(blocks):
        rest = {b: fs for b, fs in blocks.items() if b != gone}
        r2, _ = run(rest)
        if len(r2):
            label = f"without {gone}" + (" (drop Shuttle)" if gone == "shuttle" else "")
            for row in _lobo_summary(_lobo_differences(r2), band, label, n_expected=len(rest)):
                if row["variant"] not in eligible:
                    # leaving out the block that holds the gap would be a reduced panel under another name
                    row = {k: (np.nan if k in ("mean", "median", "ci_low", "ci_high") else v) for k, v in row.items()
                           if k not in ("n_above_0", "n_below_0", "n_above_threshold", "n_below_threshold")}
                    row.update(analysis_status=LOSO_NOT_READ, status_reason="the primary reading is not eligible")
                loso.append(row)
    out = {"folds": rows, "training_scores": scores, "differences": diffs, "summary": summary,
           "status": status, "loso": pd.DataFrame(loso)}
    if reduced_panel:
        r_rows, r_scores, r_diffs, r_summary, r_status = tables(blocks, reduced=True)
        out["reduced"] = {"folds": r_rows, "training_scores": r_scores, "differences": r_diffs,
                          "summary": r_summary, "status": r_status}
    return out


def _real(fold: str) -> bool:
    try:
        return C.stream_group(fold) == "real"
    except ValueError:
        return False


def _plan_expected_jobs(plan: dict) -> dict[str, int] | None:
    """Per arm, the job identities the plan runs for the LOBO regimes (a cell read through another regime
    counted once, under the arm that runs it); None when the plan names cells only."""
    seen: dict[str, set] = {}
    for (a, f, r), cell in plan["cells"].items():
        if cell["jobs"] is None:
            return None
        if cell["planned"] and cell["via"] == r:
            seen.setdefault(r, set()).update(cell["jobs"])
    return {arm: len(v) for arm, v in sorted(seen.items())}


def write_lobo(out_dir: Path, rep_dir: Path | None = None, metric: str = "ari", band: float = BAND,
               fill: bool = True, config: Path | None = None, reduced_panel: bool = False) -> list[Path]:
    """``lobo_real`` on ``out_dir``'s latest runs, into ``<rep>/lobo[_<metric>]/``: ``lobo_status.csv``,
    ``lobo_folds.csv``, ``lobo_training_scores.csv``, ``lobo_differences.csv``, ``lobo_summary.csv``,
    ``lobo_loso.csv``, ``coverage.csv`` and ``lobo.md``.

    The expected panel is read before any result (``read_lobo_plan``: ``config``, a campaign config, else
    the dir's campaign manifest, else its run manifest); without one the primary reading is never
    eligible. ``fill``: ``scaled_default`` of an algorithm without a distance parameter reads its
    ``default`` windows (``analysis_support.fill_identical_windows``), before any panel.
    ``reduced_panel`` (a sensitivity, asked for by name) also writes the reduced-panel tables into
    ``<rep>/reduced_panel/``; they never replace the primary ones."""
    from . import analysis_support as AS

    out_dir = Path(out_dir)
    col = AS.METRICS[metric]
    rep = (Path(rep_dir) if rep_dir is not None else out_dir / C.REPORT_DIR.name) / ("lobo" if metric == "ari" else f"lobo_{metric}")
    rep.mkdir(parents=True, exist_ok=True)
    plan = read_lobo_plan(out_dir, config)
    runs = AS.run_table(out_dir, arms=LOBO_REGIMES)
    tw = AS.tail_windows(out_dir, runs, metrics=(col,))
    filled: list[str] = []
    if fill:
        tw, filled = AS.fill_identical_windows(tw)
    res = lobo_real(tw, col, band, plan=plan, runs=runs, reduced_panel=reduced_panel)
    paths = []
    for name in ("status", "folds", "training_scores", "differences", "summary", "loso"):
        path = rep / f"lobo_{name}.csv"
        res[name].to_csv(path, index=False)
        paths.append(path)
    if reduced_panel:
        red = rep / "reduced_panel"
        red.mkdir(exist_ok=True)
        for name in ("status", "folds", "training_scores", "differences", "summary"):
            path = red / f"lobo_{name}.csv"
            res["reduced"][name].to_csv(path, index=False)
            paths.append(path)
    real = {f for fs in plan["blocks"].values() for f in fs} if plan is not None else {f for f in runs["fold"] if _real(f)}
    cov_runs = runs[runs["fold"].isin(real)] if len(runs) else runs
    cov_tw = tw[tw["fold"].isin(real) & (tw["filled_from"].isna() if "filled_from" in tw else True)] if len(tw) else tw
    cov = AS.coverage(cov_runs, cov_tw, expected=_plan_expected_jobs(plan) if plan is not None else None)
    cov.to_csv(rep / "coverage.csv", index=False)
    paths.append(rep / "coverage.csv")
    md = rep / "lobo.md"
    md.write_text(_lobo_markdown(out_dir, res, metric, band, filled, fill, runs, plan), encoding="utf-8")
    paths.append(md)
    return paths


def _status_lines(status: pd.DataFrame, heading: str) -> list[str]:
    """The eligibility of each variant, in words, with what the panel expected and what it lacks."""
    lines = [heading, ""]
    if status.empty:
        return lines + ["No real source block in the plan: nothing to read.", ""]
    for st in status.to_dict("records"):
        lines.append(f"- `{st['variant']}`: **{st['analysis_status']}**. {st['reason']}. Expected blocks: "
                     f"{st['expected_blocks'] or 'none'} ({st['n_blocks_expected']}); with a complete training panel: "
                     f"{st['n_blocks_eligible']}; with a test reading: {st['n_blocks_test_available']}.")
        if st["incomplete_blocks"]:
            lines.append(f"  - Training blocks with a gap: {st['incomplete_blocks']}.")
        if st["blocks_without_test"]:
            lines.append(f"  - Held-out blocks without a test reading: {st['blocks_without_test']}.")
        if st["missing_regimes"]:
            lines.append(f"  - Regimes with a missing cell: {st['missing_regimes']}.")
        if st.get("n_windows_partial_replicates"):
            lines.append(f"  - Windows left out of a cell because one of its replicates did not read them: "
                         f"{st['n_windows_partial_replicates']} ({st['partial_replicates']}).")
        if st["missing"]:
            miss = str(st["missing"]).split("; ")
            more = f" (and {len(miss) - 12} more, in `lobo_status.csv`)" if len(miss) > 12 else ""
            lines.append(f"  - Missing: {'; '.join(miss[:12])}{more}.")
    return lines + [""]


def _n_tuner_seeds(runs: pd.DataFrame) -> int:
    """Distinct tuner seeds of the online runs (a fixed run's model seed is no tuner seed)."""
    if not len(runs):
        return 0
    col = runs["tuner_seed"] if "tuner_seed" in runs else runs.loc[~runs["arm"].isin(list(C.FIXED_ARMS)), "seed"]
    return int(pd.Series([v for v in col if v is not None and not (isinstance(v, float) and np.isnan(v))]).nunique())


def _lobo_markdown(out_dir, res, metric, band, filled, fill, runs, plan=None) -> str:
    summ, diffs, rows = res["summary"], res["differences"], res["folds"]
    primary = f"{LOBO_PRIMARY[0]}_minus_{LOBO_PRIMARY[1]}"
    lines = [
        f"# LOBO selection on real sources ({metric.upper()}, tail windows)", "",
        f"Out dir `{out_dir}`. Each real source block is held out whole; the global choice is made from the "
        "other real blocks only, each block weighing the same (its variants averaged first), and every selector "
        "reads the same training panel: the same blocks, folds, windows and candidate algorithms. The held-out "
        f"source is read only after the choice, under `{PRIMARY_TARGET}`. Tie policy: {LOBO_TIE_POLICY}. "
        "`random` is the exact expectation of a uniform pick over the candidates.", "",
        f"Expected panel read from: {plan['source'] if plan is not None else 'no plan on file (the primary reading cannot be eligible)'}. "
        "A held-out block whose training panel misses a cell (a regime, a job, or a window every candidate reads) "
        "chooses nothing; a missing or failed test job keeps the choice and leaves its test quality unavailable. "
        "Nothing is imputed, and no number of a narrowed panel stands in for the primary reading.", "",
        f"Primary difference: `Q_{PRIMARY_TARGET},test(a_lobo_target) - Q_{PRIMARY_TARGET},test(a_lobo_scaled_default)` "
        f"(`{primary}`), per held-out block, then the mean over blocks. Seeds never add a block: {_n_tuner_seeds(runs)} "
        f"tuner seeds feed {diffs['held_out'].nunique() if len(diffs) else 0} blocks. With so few blocks the "
        "interval is coarse (resampled blocks, three or more); no difference here is read as equivalence when "
        "it is not significant.", "",
    ]
    if fill:
        lines.append(f"- `scaled_default` read from `default` (no absolute-distance parameter): "
                     f"{', '.join(filled) if filled else 'none'}.")
    lines.append(f"- Practical threshold: {band:g} {metric.upper()}.")
    lines.append("")
    lines += _status_lines(res["status"], "## Status of the primary reading")
    if len(summ):
        whole = (summ["analysis_status"] == "eligible").all()
        lines += ["## Mean over held-out blocks" + ("" if whole else " (not eligible: means left undefined)"), "",
                  _table(summ), ""]
    if len(diffs):
        lines += ["## Per held-out source", "", _table(diffs), ""]
    if len(rows):
        keep = ["variant", "held_out", "selector", "panel_status", "pick", "n_tied", "tied", "band_set", "train_score",
                "n_candidates", "excluded", "n_train_blocks", "n_train_folds", "train_windows", "test_status",
                "test_windows", "q_test", "regret", "note"]
        lines += ["## Choices per fold", "", _table(rows[keep]), ""]
    if len(res["loso"]):
        lo = res["loso"]
        lines += ["## Leave one source out (the whole analysis rerun without it)", "",
                  _table(lo[lo["difference"] == primary]), ""]
    if "reduced" in res:
        red = res["reduced"]
        lines += ["## Sensitivity: reduced panel (asked for with `--reduced-panel`; not the primary reading)", "",
                  "Each held-out block's panel leaves out what is missing: an algorithm with a gap is excluded for "
                  "every selector alike, a training fold without a common window is dropped. Its tables are in "
                  "`reduced_panel/`; none of its numbers replaces a primary one.", ""]
        lines += _status_lines(red["status"], "### Status")
        if len(red["summary"]):
            lines += ["### Mean over the blocks it defines", "", _table(red["summary"]), ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(C.OUT))
    parser.add_argument("--band", type=float, default=BAND,
                        help="practical-effect band in tail ARI (0 = exact ties only); default the study's "
                             "pre-declared smallest effect of interest")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--select-seeds", default=None,
                        help="selecting-arm seeds the held-out split selects on (default: 1 on a shared-policy dir; "
                             "on a split-policy dir the split is not read unless both seed lists are given)")
    parser.add_argument("--eval-seeds", default=None,
                        help="seeds the held-out split evaluates every target on (default: 2,3 on a shared-policy dir)")
    parser.add_argument("--selecting-arm", default=SELECTING_ARM,
                        help="the arm whose per-fold best is picked, e.g. scaled_default (E1); any arm other than "
                             "default writes to <out-dir>/report/selection_<arm>/")
    parser.add_argument("--no-fill-identical", action="store_true",
                        help="do not let an arm's runs stand in for a selecting arm run.py skipped as identical "
                             "(scaled_default of HPStream and RepStream is their default)")
    parser.add_argument("--lobo", action="store_true",
                        help="write the campaign's LOBO reading instead (real sources only, each source block "
                             "weighted once, one training panel for every selector) into <out-dir>/report/lobo/")
    parser.add_argument("--metric", default="ari", choices=("ari", "nmi"),
                        help="with --lobo: the lagged reading (ARI primary, NMI a sensitivity, into report/lobo_nmi/)")
    parser.add_argument("--campaign-config", default=None,
                        help="with --lobo: read the expected panel from this campaign config (default: the dir's "
                             "campaign manifest, else its run manifest; with neither the primary reading is never eligible)")
    parser.add_argument("--reduced-panel", action="store_true",
                        help="with --lobo: also write the reduced-panel sensitivity (what is missing excluded for every "
                             "selector alike) into report/lobo/reduced_panel/; it never replaces the primary reading")
    args = parser.parse_args(argv)
    if args.lobo:
        for p in write_lobo(Path(args.out_dir), metric=args.metric, band=args.band, fill=not args.no_fill_identical,
                            config=Path(args.campaign_config) if args.campaign_config else None,
                            reduced_panel=args.reduced_panel):
            print(p)
        return
    seeds = lambda s: tuple(int(x) for x in s.split(",") if x.strip()) if s else None  # noqa: E731
    for p in write_selection(Path(args.out_dir), band=args.band, top_k=args.top_k,
                             select_seeds=seeds(args.select_seeds), eval_seeds=seeds(args.eval_seeds),
                             selecting_arm=args.selecting_arm, fill=not args.no_fill_identical):
        print(p)


if __name__ == "__main__":
    main()
