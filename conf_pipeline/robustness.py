"""How far the full run's conclusions hold: analyses of the existing logs, no new tuning run.

Every number here is read from what ``run.py`` and ``sweep.py`` already wrote (and, for
A5, from ``label_reach.jsonl``, see :mod:`conf_pipeline.label_reach`). The published
tables of ``report.py`` and ``analyze.py`` are left as they are; this module asks how
much they depend on the measure, the reference and the seeds.

- group aggregations  capture ratio and the selection/search split per stream group
- A1  rankings under ARI, NMI and purity, and how well they agree
- A2  a reference selected on the even tail windows and read on the odd ones
- A3  bootstrap intervals (folds within groups, then seeds) and Holm-corrected tests
- A4  the windows the single-class rule removes, and the rankings without KDD99
- the usable-stream table, which joins A1, A4 and A5
- P1.2  the statistical unit and the measure:

  - the result script's checks (``result/otb_full/report/make_robustness_extra.py``) as
    functions: tau between arms with intervals, the INSECTS campaign as one block, tau(default, ·)
    after the declared weak-default filters, rankings by single Sobol points, leaving one real
    stream or one source block out, the per-stream split and the selection share's interval;
    :mod:`conf_pipeline.robustness_check` compares them with the printed file
  - a resampling unit for the bootstraps (``UNITS``: folds, or source blocks that bring all
    their folds) and seed pairing by seed number, the reference's reruns drawn apart
  - the online arm's paired effects per unit (blocks averaged first), with intervals and
    Wilcoxon over the units with Holm
  - NMI under two reference constructions: (a) every ARI selection kept and read in NMI, (b) the
    reference and the labels' pick within the ensemble reselected by NMI (``build_constructions``)
  - ``conclusions.csv``: conclusion x measure x unit x reference construction, each ratio with its
    numerator, denominator and n, undefined below ``MIN_DENOMINATOR``, never clipped

Run from the repo root::

    python -m conf_pipeline.robustness --out-dir conf_pipeline/out/full
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from . import report

#: The deployable arms whose share of the attainable gain is reported.
CAPTURE_ARMS: tuple[str, ...] = ("confstream", "frozen_prefix")
#: The pooled row every per-group table ends with.
ALL = "all"
#: The arms read here: the protocol's three and the reference's reruns.
ARMS: tuple[str, ...] = C.ARMS + ("oracle_static",)
TAIL = report.TAIL


# --- inputs ------------------------------------------------------------------------------------

@dataclass
class Inputs:
    """What every analysis reads, loaded once from an out dir; protocol algorithms only.

    ``seeds``: one row per ok run, ``algo, fold, arm, seed, value`` with ``value`` the
    run's tail ARI (``lag_ari_win_tail_mean``); the reference's rows are its reruns of the
    current ``oracle_static.json`` configuration. ``tunable``: (algo, fold) -> the gate of
    ``report.headroom_table``. ``regret``: ``report.regret_windows`` of ``confstream`` on
    the tail. ``runs``: the current ok run rows.
    """

    out_dir: Path
    runs: pd.DataFrame
    seeds: pd.DataFrame
    tunable: dict
    regret: pd.DataFrame


def _protocol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "algo" not in df:
        return df
    return df[~df["algo"].isin(list(C.IMPL_VARIANTS))].reset_index(drop=True)


def load_inputs(out_dir: Path) -> Inputs:
    out_dir = Path(out_dir)
    runs = report.current_runs(out_dir)
    runs = _protocol(runs[(runs["status"] == "ok") & runs["arm"].isin(ARMS)]) if not runs.empty else runs
    seeds = runs[["algo", "fold", "arm", "seed"]].copy() if not runs.empty else pd.DataFrame(
        columns=["algo", "fold", "arm", "seed"])
    seeds["value"] = pd.to_numeric(runs[TAIL], errors="coerce") if TAIL in runs else np.nan
    seeds = seeds.dropna(subset=["value"]).reset_index(drop=True)
    # Under the split seed policy a fixed configuration's runs are not tuner seeds: its mean is read beside
    # every tuner seed, and counts as the runs it rests on (``report.replicate_view``).
    if not runs.empty and report.is_split(runs):
        seeds = report.replicate_view(seeds, split=True)[["algo", "fold", "arm", "seed", "value"]]
    head = report.headroom_table(out_dir)
    tunable = {(r["algo"], r["fold"]): bool(r["tunable"]) for r in head.to_dict("records")}
    regret = _protocol(report.regret_windows(out_dir, arms=("confstream",), region="tail"))
    return Inputs(out_dir=out_dir, runs=runs, seeds=seeds, tunable=tunable, regret=regret)


def cells_table(inputs: Inputs, seeds: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (algo, fold): ``tunable`` and the seed mean of every arm in ``ARMS``,
    from ``inputs.seeds`` or from ``seeds`` in the same shape (other windows, another
    reference); ``tunable`` always comes from the published reference."""
    seeds = inputs.seeds if seeds is None else seeds
    wide = seeds.pivot_table(index=["algo", "fold"], columns="arm", values="value", aggfunc="mean")
    wide = wide.reindex(columns=list(ARMS)).reset_index()
    wide.columns.name = None
    wide.insert(2, "tunable", [inputs.tunable.get((a, f), False) for a, f in zip(wide["algo"], wide["fold"])])
    return wide


def loss_seeds(regret: pd.DataFrame, windows=None) -> pd.DataFrame:
    """Per (algo, fold, seed): ``a_lag`` and ``a_star_lag``, the means over the used
    windows of ``regret`` (``report.regret_windows``), or over the used windows among
    ``windows`` (a callable ``window index -> bool``) when given."""
    used = regret[regret["used"].astype(bool)]
    if windows is not None:
        used = used[[bool(windows(int(w))) for w in used["window"]]]
    g = used.groupby(["algo", "fold", "seed"], sort=True)
    out = g.agg(a_lag=("lag_ari_win", "mean"), a_star_lag=("a_star_lag", "mean")).reset_index()
    return out


def loss_cells(inputs: Inputs, windows=None, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (algo, fold) of ``confstream``: seed means of ``a_lag``, ``a_star_lag`` and
    the reference's ``A_static``. ``reference`` (algo, fold, ``A_static``) replaces the
    published reference; ``windows`` restricts the online readings (see ``loss_seeds``)."""
    per_seed = loss_seeds(inputs.regret, windows)
    out = per_seed.groupby(["algo", "fold"], sort=True)[["a_lag", "a_star_lag"]].mean().reset_index()
    if reference is None:
        ref = inputs.seeds[inputs.seeds["arm"] == "oracle_static"]
        reference = ref.groupby(["algo", "fold"])["value"].mean().rename("A_static").reset_index()
    return out.merge(reference[["algo", "fold", "A_static"]], on=["algo", "fold"], how="left")


def _groups(folds) -> list[str]:
    return [C.stream_group(f) for f in folds]


def capture_by_group(cells: pd.DataFrame, arms=CAPTURE_ARMS) -> pd.DataFrame:
    """Share of the attainable gain each arm realises, per stream group and pooled.

    ``cells``: one row per (algo, fold) with ``tunable`` and the seed means of ``default``,
    ``oracle_static`` and every arm in ``arms``. Over the tunable pairs of a group,
    ``capture = sum(arm - default) / sum(oracle_static - default)``: sums are taken before
    dividing, so a pair with a tiny attainable gain cannot dominate. A pair missing any of
    the three readings is left out; a group with no pair (or a non-positive denominator)
    reads NaN.
    """
    df = cells.copy()
    df["group"] = _groups(df["fold"])
    out = []
    for group in (*C.STREAM_GROUPS, ALL):
        sub = df if group == ALL else df[df["group"] == group]
        sub = sub[sub["tunable"].astype(bool)]
        for arm in arms:
            vals = sub[["default", arm, "oracle_static"]].apply(pd.to_numeric, errors="coerce").dropna()
            num = float((vals[arm] - vals["default"]).sum())
            den = float((vals["oracle_static"] - vals["default"]).sum())
            out.append({"group": group, "arm": arm, "n_pairs": int(len(vals)),
                        "capture": num / den if len(vals) and den > 0 else np.nan})
    return pd.DataFrame(out)


def losses_by_group(loss: pd.DataFrame) -> pd.DataFrame:
    """Mean over the pairs of a group of ``a_lag``, ``a_star_lag``, ``A_static`` and of the
    two losses ``selection_loss = a_star_lag - a_lag`` (the silhouette against labels, both
    picking one window ahead) and ``search_loss = A_static - a_star_lag`` (the best fixed
    configuration against the best the ensemble held; negative when adapting beats it).

    ``loss``: one row per (algo, fold) of the online arm, seed means already taken. A pair
    without one of the three readings is left out of every column, so the two losses of a
    group always sum to its ``A_static - a_lag``.
    """
    df = loss.copy()
    cols = ["a_lag", "a_star_lag", "A_static"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=cols)
    df["group"] = _groups(df["fold"])
    df["selection_loss"] = df["a_star_lag"] - df["a_lag"]
    df["search_loss"] = df["A_static"] - df["a_star_lag"]
    keep = cols + ["selection_loss", "search_loss"]
    out = []
    for group in (*C.STREAM_GROUPS, ALL):
        sub = df if group == ALL else df[df["group"] == group]
        out.append({"group": group, "n_pairs": int(len(sub)),
                    **{c: float(sub[c].mean()) if len(sub) else np.nan for c in keep}})
    return pd.DataFrame(out)


# --- A3: bootstrap ---------------------------------------------------------------------------

#: Readings of the bootstrap's cube: the arms' tail ARIs and the online arm's two loss terms.
CUBE_ARMS: tuple[str, ...] = ARMS
CUBE_LOSS: tuple[str, ...] = ("a_lag", "a_star_lag")
#: Percentiles of the bootstrap interval.
CI = (2.5, 97.5)


@dataclass
class _Cube:
    """Every reading as an (algo, fold, seed slot) array; NaN where a seed is missing.

    ``by_seed=False`` (what ``bootstrap`` has always done): slot i is each arm's i-th seed in
    ascending order, so the reference's reruns on seeds 2-4 take slots 0-2 and one slot drawn
    for a pair picks seed 2 of the reference with seed 1 of the protocol arms. ``by_seed=True``:
    the protocol arms and the online arm's loss terms take the slot of their seed number among
    the protocol seeds (they ran on the same seeds, so a slot is one seed in all of them), and
    the reference keeps slots of its own, which ``bootstrap(pairing="seed")`` draws apart.
    """

    algos: list
    folds: list
    groups: np.ndarray
    arrays: dict
    tunable: np.ndarray


#: The reading whose seeds are not the protocol's: the reference's reruns (2-4 in the full run).
_OWN_SEEDS: tuple[str, ...] = ("oracle_static",)


def _cube(seeds: pd.DataFrame, losses: pd.DataFrame, tunable: dict, by_seed: bool = False) -> _Cube:
    algos = sorted(set(seeds["algo"]))
    folds = sorted(set(seeds["fold"]))
    ai, fi = {a: i for i, a in enumerate(algos)}, {f: i for i, f in enumerate(folds)}

    def fill(frame, value_cols, by_arm):
        grouped = {}
        for col in value_cols:
            sub = frame[frame["arm"] == col] if by_arm else frame
            for (a, f), g in sub.groupby(["algo", "fold"]):
                g = g.sort_values("seed")
                vals = pd.to_numeric(g["value" if by_arm else col], errors="coerce")
                grouped[(col, a, f)] = (g["seed"].astype(int).to_numpy(), vals.to_numpy(dtype=float))
        return grouped

    arm_vals = fill(seeds, CUBE_ARMS, True)
    loss_vals = fill(losses, CUBE_LOSS, False) if not losses.empty else {}
    protocol = sorted({int(s) for s in seeds.loc[~seeds["arm"].isin(list(_OWN_SEEDS)), "seed"]}
                      | ({int(s) for s in losses["seed"]} if not losses.empty else set()))
    slot = {s: i for i, s in enumerate(protocol)}
    n_seeds = max([1] + [len(v) for _, v in (*arm_vals.values(), *loss_vals.values())]
                  + ([len(protocol)] if by_seed else []))
    arrays = {}
    for name in CUBE_ARMS + CUBE_LOSS:
        arr = np.full((len(algos), len(folds), n_seeds), np.nan)
        source = arm_vals if name in CUBE_ARMS else loss_vals
        for (col, a, f), (seed_nums, vals) in source.items():
            if col == name and a in ai and f in fi:
                if by_seed and name not in _OWN_SEEDS:
                    arr[ai[a], fi[f], [slot[int(s)] for s in seed_nums]] = vals
                else:
                    arr[ai[a], fi[f], :len(vals)] = vals
        arrays[name] = arr
    tun = np.array([[bool(tunable.get((a, f), False)) for f in folds] for a in algos], dtype=bool)
    return _Cube(algos, folds, np.array(_groups(folds)), arrays, tun)


#: What a bootstrap resamples: ``fold`` draws the folds of a stream group with replacement (what
#: A3 has always done); ``block`` draws the group's source blocks (``_common.source_block``), each
#: drawn block bringing every one of its folds, so the seven INSECTS recordings of one campaign
#: count as one source and not seven. Model seeds never add units: they are drawn within a pair.
UNITS: tuple[str, ...] = ("fold", "block")
#: How seed slots line up across arms in a bootstrap (see ``_Cube``).
PAIRINGS: tuple[str, ...] = ("index", "seed")


def _family_slots(cube: _Cube, own: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per (algo, fold) of one seed family (the reference's reruns when ``own``, else the
    protocol arms and the loss terms): its slots holding a reading of any of its arrays,
    first, and how many there are. A seed draw picks among those only, so a reference read
    once (a sweep run in slot 0 of three) is never lost to an empty slot."""
    names = [k for k in cube.arrays if (k in _OWN_SEEDS) == own]
    present = np.zeros(next(iter(cube.arrays.values())).shape, dtype=bool)
    for k in names:
        present |= ~np.isnan(cube.arrays[k])
    return np.argsort(~present, axis=2, kind="stable"), present.sum(axis=2)


def _draw_slots(slots: tuple[np.ndarray, np.ndarray], folds: np.ndarray, n_seed: int,
                rng: np.random.Generator) -> np.ndarray:
    """``n_seed`` slots drawn with replacement among each drawn pair's filled slots."""
    order, k = slots
    j = rng.integers(0, np.maximum(k[:, folds], 1)[:, :, None], size=(order.shape[0], len(folds), n_seed))
    return np.take_along_axis(order[:, folds, :], j, axis=2)


def _check_unit(unit: str) -> None:
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")


def unit_index(folds, unit: str = "fold") -> list[tuple[str, str, np.ndarray]]:
    """The resampling units of ``folds``: ``[(stream group, unit name, fold positions)]``,
    groups in ``STREAM_GROUPS`` order and, within a group, units in the order of their first
    fold. With ``unit="fold"`` each fold is its own unit; with ``"block"`` a unit is a source
    block and holds the positions of all its folds."""
    _check_unit(unit)
    folds = list(folds)
    out = []
    for group in C.STREAM_GROUPS:
        names: dict[str, list[int]] = {}
        for i, f in enumerate(folds):
            if C.stream_group(f) == group:
                names.setdefault(f if unit == "fold" else C.source_block(f), []).append(i)
        out += [(group, name, np.array(pos, dtype=int)) for name, pos in names.items()]
    return out


def draw_units(folds, unit: str, rng: np.random.Generator, index=None) -> tuple[np.ndarray, np.ndarray]:
    """One stratified bootstrap draw: within each stream group as many units as it has, with
    replacement. Returns the drawn fold positions and, per position, the id of the drawn unit
    instance it belongs to (a block drawn twice is two instances). With ``unit="fold"`` the
    generator is consumed exactly as ``bootstrap`` always consumed it. ``index``: a precomputed
    ``unit_index(folds, unit)``."""
    index = unit_index(folds, unit) if index is None else index
    pos, ids, n = [], [], 0
    for group in C.STREAM_GROUPS:
        units = [u for u in index if u[0] == group]
        if not units:
            continue
        for k in rng.choice(len(units), size=len(units), replace=True):
            pos.append(units[k][2])
            ids.append(np.full(len(units[k][2]), n))
            n += 1
    if not pos:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(pos), np.concatenate(ids)


def _seed_mean(arr: np.ndarray) -> np.ndarray:
    """Mean over the seed axis, NaN (and no warning) where every seed is missing."""
    n = np.sum(~np.isnan(arr), axis=2)
    total = np.nansum(arr, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def _statistics(arrays: dict, groups: np.ndarray, tunable: np.ndarray) -> dict:
    """``{(statistic, group, arm): value}`` on one (possibly resampled) cube."""
    m = {k: _seed_mean(v) for k, v in arrays.items()}
    d, ref = m["default"], m["oracle_static"]
    out = {}
    for group in (*C.STREAM_GROUPS, ALL):
        in_group = np.ones(len(groups), bool) if group == ALL else groups == group
        for arm in CAPTURE_ARMS:
            mask = tunable & in_group[None, :] & ~np.isnan(d) & ~np.isnan(ref) & ~np.isnan(m[arm])
            den = float(np.sum((ref - d)[mask]))
            out[("capture", group, arm)] = (
                float(np.sum((m[arm] - d)[mask])) / den if mask.any() and den > 0 else np.nan)
        lag, star = m["a_lag"], m["a_star_lag"]
        mask = in_group[None, :] & ~np.isnan(lag) & ~np.isnan(star) & ~np.isnan(ref)
        out[("selection_loss", group, "confstream")] = float(np.mean((star - lag)[mask])) if mask.any() else np.nan
        out[("search_loss", group, "confstream")] = float(np.mean((ref - star)[mask])) if mask.any() else np.nan
    return out


def bootstrap(seeds: pd.DataFrame, losses: pd.DataFrame, tunable: dict,
              n_boot: int = 2000, seed: int = 0, unit: str = "fold", pairing: str = "index") -> pd.DataFrame:
    """Percentile intervals for the capture ratios and the two losses, per group and pooled.

    Two levels: within each stream group the units (``UNITS``: folds, or source blocks that
    bring all their folds) are drawn with replacement, then within each (algo, drawn fold) the
    seed slots are drawn with replacement. ``pairing="index"`` (the default, what A3 has always
    done) draws the same slots for every arm, which pairs the reference's i-th rerun seed with
    the protocol arms' i-th seed; ``pairing="seed"`` keeps the protocol arms and the loss terms
    paired by seed number and draws the reference's reruns apart, since its seeds are other
    seeds, each family among the slots a pair has a reading in (``_family_slots``). Seeds only add within-pair noise and never add units: the statistic's n is the
    pairs, whatever the number of seeds. The point estimate is the same for every unit and
    pairing. Algorithms and the tunable set stay as they are on the full data. A replicate whose
    statistic is undefined (no tunable pair drawn, a non-positive denominator) is left out of
    that statistic's interval; ``n_used`` counts the others.

    ``seeds``: ``Inputs.seeds``; ``losses``: ``loss_seeds`` (algo, fold, seed, a_lag,
    a_star_lag); ``tunable``: (algo, fold) -> bool.
    """
    _check_unit(unit)
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    cube = _cube(seeds, losses, tunable, by_seed=pairing == "seed")
    point = _statistics(cube.arrays, cube.groups, cube.tunable)
    rng = np.random.default_rng(seed)
    index = unit_index(cube.folds, unit)
    n_algo, _, n_seed = next(iter(cube.arrays.values())).shape
    draws = {k: [] for k in point}
    if pairing == "seed":
        slots = (_family_slots(cube, own=False), _family_slots(cube, own=True))
    for _ in range(n_boot):
        folds, _ids = draw_units(cube.folds, unit, rng, index)
        if pairing == "seed":
            picks, own = (_draw_slots(sl, folds, n_seed, rng) for sl in slots)
        else:  # what A3 has always drawn: the same slots, any of n_seed, for every arm
            picks = own = rng.integers(0, n_seed, size=(n_algo, len(folds), n_seed))
        arrays = {k: np.take_along_axis(v[:, folds, :], own if k in _OWN_SEEDS else picks, axis=2)
                  for k, v in cube.arrays.items()}
        for k, v in _statistics(arrays, cube.groups[folds], cube.tunable[:, folds]).items():
            draws[k].append(v)
    rows = []
    for (stat, group, arm), value in point.items():
        vals = np.array(draws[(stat, group, arm)], dtype=float)
        vals = vals[~np.isnan(vals)]
        low, high = (np.percentile(vals, CI) if len(vals) else (np.nan, np.nan))
        rows.append({"statistic": stat, "group": group, "arm": arm, "point": value,
                     "low": float(low), "high": float(high), "n_used": int(len(vals))})
    return pd.DataFrame(rows)


# --- A3: paired tests ------------------------------------------------------------------------

#: Each algorithm's comparisons of the online arm, over the folds.
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("confstream", "default"), ("confstream", "frozen_prefix"), ("confstream", "oracle_static"),
)
_ZERO = 1e-12


def holm(pvalues) -> list:
    """Holm's step-down adjustment; a ``None``/NaN p stays missing and is not counted in m."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and not np.isnan(p)]
    out = [None] * len(pvalues)
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * float(pvalues[i])))
        out[i] = running
    return out


def rank_biserial(diff) -> float:
    """Matched-pairs rank-biserial correlation: ``(W+ - W-) / (W+ + W-)`` over the signed
    ranks of the non-zero differences (ties share their average rank); NaN when all are zero."""
    from scipy.stats import rankdata

    d = np.asarray(diff, dtype=float)
    d = d[np.abs(d) > _ZERO]
    if not len(d):
        return np.nan
    r = rankdata(np.abs(d))
    plus, minus = r[d > 0].sum(), r[d < 0].sum()
    return float((plus - minus) / (plus + minus))


def paired_tests(cells: pd.DataFrame, comparisons=COMPARISONS) -> pd.DataFrame:
    """Per algorithm and comparison, Wilcoxon's signed-rank test over the folds on the seed
    means, two-sided, zero differences dropped (``n_zero`` counts them); Holm over the
    whole family of tests; effect size as the rank-biserial correlation and the median
    difference (over every fold, zeros included). A test with no non-zero difference has
    no p."""
    from scipy.stats import wilcoxon

    rows = []
    for algo, g in cells.groupby("algo", sort=True):
        for a, b in comparisons:
            x = pd.to_numeric(g[a], errors="coerce")
            y = pd.to_numeric(g[b], errors="coerce")
            keep = x.notna() & y.notna()
            d = (x[keep] - y[keep]).to_numpy(dtype=float)
            nonzero = d[np.abs(d) > _ZERO]
            p = stat = np.nan
            if len(nonzero):
                res = wilcoxon(nonzero, alternative="two-sided")
                p, stat = float(res.pvalue), float(res.statistic)
            rows.append({"algo": algo, "comparison": f"{a} vs {b}", "n": int(len(d)),
                         "n_zero": int(len(d) - len(nonzero)), "statistic": stat, "p": p,
                         "rank_biserial": rank_biserial(d),
                         "median_diff": float(np.median(d)) if len(d) else np.nan})
    out = pd.DataFrame(rows)
    adj = holm([None if np.isnan(p) else p for p in out["p"]]) if len(out) else []
    out.insert(out.columns.get_loc("p") + 1, "p_holm", [np.nan if v is None else v for v in adj])
    return out


# --- A2: reference selected on even windows ----------------------------------------------------

#: Window parities: the even tail windows select, the odd ones report.
EVEN, ODD = 0, 1


def _tail_start(tail_from, key) -> int:
    if isinstance(tail_from, dict):
        return int(tail_from.get(key, C.FREEZE_AFTER_WINDOW))
    return int(tail_from)


def interleaved_reference(sweep_win: pd.DataFrame, tail_from=C.FREEZE_AFTER_WINDOW,
                          incumbents: bool = True) -> pd.DataFrame:
    """Per (algo, fold), the sweep candidate with the best mean lagged ARI on the even tail
    windows, and its mean on the odd ones.

    The selection never sees the windows it is reported on, so the reading carries no
    optimism of a maximum, and the drift segments of a stream stay in both halves.
    ``sweep_win``: ``algo, fold, config_id, window, lag_ari_win``, the windows of every seed
    the sweep ran pooled (equal to the mean over seeds when each ran every window).
    ``tail_from``: an int, or ``{(algo, fold): int}``. A candidate needs a reading
    on both parities; ties go to the smaller ``config_id``. ``incumbents=False`` keeps only
    River's default and the Sobol points.
    """
    cols = ["algo", "fold", "config_id", "config_source", "select_value", "report_value", "n_candidates"]
    df = sweep_win.copy()
    df["lag_ari_win"] = pd.to_numeric(df["lag_ari_win"], errors="coerce")
    df["window"] = pd.to_numeric(df["window"], errors="coerce")
    if not incumbents:
        df = df[[report.config_source(c) != "incumbent" for c in df["config_id"]]]
    rows = []
    for (algo, fold), g in df.groupby(["algo", "fold"], sort=True):
        g = g[g["window"] >= _tail_start(tail_from, (algo, fold))]
        sel = g[g["window"] % 2 == EVEN].groupby("config_id")["lag_ari_win"].mean()
        rep = g[g["window"] % 2 == ODD].groupby("config_id")["lag_ari_win"].mean()
        cand = pd.concat([sel.rename("sel"), rep.rename("rep")], axis=1).dropna().sort_index()
        if cand.empty:
            continue
        best = cand["sel"].idxmax()
        rows.append({"algo": algo, "fold": fold, "config_id": int(best),
                     "config_source": report.config_source(best),
                     "select_value": float(cand.loc[best, "sel"]),
                     "report_value": float(cand.loc[best, "rep"]), "n_candidates": int(len(cand))})
    return pd.DataFrame(rows, columns=cols)


def _per_run(windows: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """``values`` (indexed by algo, fold, arm, seed) as an ``Inputs.seeds``-shaped frame. A run without a seed of
    its own (a deterministic fixed configuration under the split seed policy) keeps its row, and under that
    policy every fixed arm's mean stands beside every tuner seed (``report.replicate_view``)."""
    out = values.dropna().reset_index()
    if report.is_split(windows):
        out = report.replicate_view(out, split=True)[["algo", "fold", "arm", "seed", "value"]]
    return out


def parity_values(windows: pd.DataFrame, tail_from: dict, parity: int) -> pd.DataFrame:
    """Per run (algo, fold, arm, seed): the mean ``lag_ari_win`` over its tail windows of
    one parity, in the shape of ``Inputs.seeds``. ``tail_from``: ``{(algo, fold, seed, arm):
    int}``; a run it does not name starts at ``FREEZE_AFTER_WINDOW``."""
    df = windows[["algo", "fold", "arm", "seed", "window", "lag_ari_win"]].copy()
    df["lag_ari_win"] = pd.to_numeric(df["lag_ari_win"], errors="coerce")
    df["window"] = pd.to_numeric(df["window"], errors="coerce")
    start = [_tail_start(tail_from, k) for k in zip(df["algo"], df["fold"], df["seed"], df["arm"])]
    df = df[(df["window"] >= np.array(start)) & (df["window"] % 2 == parity)]
    out = df.groupby(["algo", "fold", "arm", "seed"], sort=True, dropna=False)["lag_ari_win"].mean().rename("value")
    return _per_run(windows, out)


# --- A1: three measures ------------------------------------------------------------------------

#: Measure name -> the window column it is read from (the lagged incumbent's reading).
MEASURES: dict[str, str] = {
    "ari": "lag_ari_win", "nmi": "lag_nmi_win", "purity": "lag_purity_win", "cmm": "lag_cmm_win",
    "cmm_strict": "lag_cmm_strict_win", "n_pred_labels": "lag_n_pred_labels",
}
#: The measures algorithms are ranked by. Purity is 1 when every point is its own
#: cluster, so the number of predicted labels is read beside it, not ranked. CMM is read
#: only by runs made since it was added (2026-10-23); older windows lack the columns and
#: leave them out. MOA's ``cmm`` merges connected classes and can tie every algorithm at 1
#: on a real window; ``cmm_strict`` does neither (``confstream.CMM_VARIANTS``).
RANKED: tuple[str, ...] = ("ari", "nmi", "purity", "cmm", "cmm_strict")


def measure_values(windows: pd.DataFrame, tail_from: dict) -> pd.DataFrame:
    """Per (algo, fold, arm): the seed mean of each run's tail mean of every measure in
    ``MEASURES`` its windows carry. ``tail_from`` as in ``parity_values``. Single-class
    windows read None for the external measures and drop out of their means."""
    present = {m: col for m, col in MEASURES.items() if col in windows}
    df = windows[["algo", "fold", "arm", "seed", "window", *present.values()]].copy()
    for col in present.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["window"] = pd.to_numeric(df["window"], errors="coerce")
    start = [_tail_start(tail_from, k) for k in zip(df["algo"], df["fold"], df["seed"], df["arm"])]
    df = df[df["window"] >= np.array(start)]
    per_run = df.groupby(["algo", "fold", "arm", "seed"], sort=True, dropna=False)[list(present.values())].mean()
    out = per_run.groupby(level=["algo", "fold", "arm"]).mean().reset_index()
    return out.rename(columns={col: m for m, col in present.items()})


def _tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b of two rank vectors, as ``analyze.kendall_matrix`` reads it: 1 for
    identical vectors, NaN when only one of them is constant."""
    from scipy import stats

    if np.array_equal(x, y):
        return 1.0
    if np.ptp(x) > 0 and np.ptp(y) > 0:
        return float(stats.kendalltau(x, y).statistic)
    return np.nan


def measure_agreement(summary: pd.DataFrame, arms=ARMS, measures=RANKED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kendall's tau-b between the rankings of the algorithms under two measures.

    Per (arm, fold), the algorithms with a reading under both measures are ranked among
    themselves (rank 1 = best) and compared; a fold with fewer than two of them has no tau.
    Returns ``(per_fold, by_group)``: one row per (arm, fold, pair), and the mean over the
    folds of each stream group and over all folds.
    """
    from . import analyze

    algos = analyze._algos(summary)
    pairs = [(a, b) for i, a in enumerate(measures) for b in measures[i + 1:]]
    rows = []
    for arm in arms:
        wide = {m: analyze._wide(summary, arm, m, algos) for m in measures if m in summary}
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            for fold in wide[a].index.intersection(wide[b].index):
                both = wide[a].loc[fold].notna() & wide[b].loc[fold].notna()
                if both.sum() < 2:
                    continue
                x = analyze._ranks(wide[a].loc[[fold], both]).iloc[0].to_numpy()
                y = analyze._ranks(wide[b].loc[[fold], both]).iloc[0].to_numpy()
                rows.append({"arm": arm, "fold": fold, "group": C.stream_group(fold),
                             "pair": f"{a}-{b}", "n_algos": int(both.sum()), "tau": _tau(x, y)})
    per_fold = pd.DataFrame(rows, columns=["arm", "fold", "group", "pair", "n_algos", "tau"])
    out = []
    for (arm, pair), g in per_fold.groupby(["arm", "pair"], sort=False):
        for group in (*C.STREAM_GROUPS, ALL):
            sub = g if group == ALL else g[g["group"] == group]
            taus = sub["tau"].dropna()
            out.append({"arm": arm, "group": group, "pair": pair, "n_folds": int(len(taus)),
                        "tau": float(taus.mean()) if len(taus) else np.nan})
    return per_fold, pd.DataFrame(out, columns=["arm", "group", "pair", "n_folds", "tau"])


# --- A4: single-class windows ----------------------------------------------------------------------

def single_class_table(windows: pd.DataFrame, tail_from: dict) -> pd.DataFrame:
    """Per fold: its windows, the single-class ones (``external_defined`` False: no external
    reading there) and the tail windows left to evaluate on. Every job of a fold sees the
    same windows, so one job is read: the ``default`` arm's first when it ran."""
    rows = []
    for fold, g in windows.groupby("fold", sort=True):
        g = g.assign(_d=(g["arm"] != "default")).sort_values(["_d", "algo", "seed"])
        algo, arm, seed = g.iloc[0][["algo", "arm", "seed"]]
        job = g[(g["algo"] == algo) & (g["arm"] == arm) & (g["seed"] == seed)].drop_duplicates("window")
        defined = job["external_defined"].astype(bool)
        tail = pd.to_numeric(job["window"]) >= _tail_start(tail_from, (algo, fold, seed, arm))
        rows.append({"fold": fold, "group": C.stream_group(fold), "n_windows": int(len(job)),
                     "n_single_class": int((~defined).sum()), "n_eval_tail": int((defined & tail).sum())})
    return pd.DataFrame(rows, columns=["fold", "group", "n_windows", "n_single_class", "n_eval_tail"])


def without_folds(summary: pd.DataFrame, arms=ARMS, drop=("kdd99",), metric: str = TAIL):
    """Friedman's test, the Nemenyi CD and the mean Kendall tau between arms, with every
    fold and again without ``drop``. Returns ``(friedman, kendall)``: one row per (arm,
    fold set) with the average rank of every algorithm, and one row per pair of arms."""
    from . import analyze

    label = "without " + ", ".join(drop)
    sets = (("all", summary), (label, summary[~summary["fold"].isin(list(drop))]))
    fried = []
    for name, sub in sets:
        for arm in arms:
            ranks = analyze.rank_matrix(sub, arm, metric)
            if ranks.empty:
                continue
            res = analyze.friedman_nemenyi(ranks)
            fried.append({"arm": arm, "folds": name, "n_folds": res["n_folds"], "k": res["k"],
                          "statistic": res["statistic"], "pvalue": res["pvalue"], "cd": res["cd"],
                          **{f"rank_{a}": float(v) for a, v in res["avg_rank"].items()}})
    k_all = analyze.kendall_matrix(sets[0][1], arms, metric)
    k_wo = analyze.kendall_matrix(sets[1][1], arms, metric)
    kendall = [{"arm_a": a, "arm_b": b, "tau_all": float(k_all.loc[a, b]), "tau_without": float(k_wo.loc[a, b])}
               for i, a in enumerate(arms) for b in list(arms)[i + 1:]]
    return pd.DataFrame(fried), pd.DataFrame(kendall, columns=["arm_a", "arm_b", "tau_all", "tau_without"])


# --- usable streams ----------------------------------------------------------------------------

#: Placeholders, not findings: a fold's labels count as usable with at least this many
#: evaluable tail windows and a best batch ARI (A5, raw features) of at least this much.
#: Both are flags of the CLI.
MIN_EVAL_WINDOWS = 20
MIN_REACH_ARI = 0.3
#: How far a stream clusterer must read above the batch reference before the markdown
#: says so: two readings within it differ by rounding, not by method.
REACH_MARGIN = 0.01
USABLE_COLS = ["fold", "group", "n_windows", "n_single_class", "n_eval_tail", "reach_raw", "reach_std",
               "tau_ari_nmi", "best_tail_ari", "labels_usable"]


def usable_streams(single: pd.DataFrame, reach: pd.DataFrame | None, agreement: pd.DataFrame | None,
                   best: pd.DataFrame | None, min_eval_windows: int = MIN_EVAL_WINDOWS,
                   min_reach_ari: float = MIN_REACH_ARI) -> pd.DataFrame:
    """One row per fold: the evidence on whether its class labels can judge a clustering.

    ``single``: ``single_class_table``; ``reach``: (fold, reach_raw, reach_std), the best
    mean batch ARI of A5 on raw and on window-standardised features; ``agreement``: (fold,
    tau_ari_nmi); ``best``: (fold, best_tail_ari), the best tail ARI any (algo, arm) reached.
    ``labels_usable`` is None when A5 has no reading for the fold.
    """
    out = single.copy()
    if "group" not in out:
        out["group"] = _groups(out["fold"])
    for frame, cols in ((reach, ["reach_raw", "reach_std"]), (agreement, ["tau_ari_nmi"]),
                        (best, ["best_tail_ari"])):
        if frame is None:
            for c in cols:
                out[c] = np.nan
        else:
            out = out.merge(frame[["fold", *cols]], on="fold", how="left")
    usable = []
    for n_eval, raw in zip(out["n_eval_tail"], out["reach_raw"]):
        usable.append(None if pd.isna(raw) else bool(n_eval >= min_eval_windows and raw >= min_reach_ari))
    out["labels_usable"] = pd.Series(usable, index=out.index, dtype=object)
    return out[USABLE_COLS]


# --- A5: how far the labels can be reached -----------------------------------------------------

def reach_table(reach_rows: pd.DataFrame, tail_from: int = C.FREEZE_AFTER_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From ``label_reach.jsonl``: the mean ARI of each (fold, method, representation)
    over the tail windows (the region the arms are compared on), and per fold the best of
    the methods on raw features (``reach_raw``) and on standardised ones (``reach_std``)."""
    df = reach_rows
    if "kind" in df:
        df = df[df["kind"] == "reading"]
    df = df.assign(ari=pd.to_numeric(df["ari"], errors="coerce"), window=pd.to_numeric(df["window"]))
    df = df[df["window"] >= tail_from]
    per_method = (df.groupby(["fold", "method", "representation"], sort=True)["ari"]
                  .agg(ari="mean", n_windows="count").reset_index())
    best = per_method.pivot_table(index="fold", columns="representation", values="ari", aggfunc="max")
    reach = pd.DataFrame({"fold": best.index,
                          "reach_raw": best.get("raw", pd.Series(np.nan, index=best.index)).to_numpy(),
                          "reach_std": best.get("standardised", pd.Series(np.nan, index=best.index)).to_numpy()})
    return per_method, reach.reset_index(drop=True)


# --- P1.2: the result script's checks, as API ---------------------------------------------------
#
# ``result/otb_full/report/make_robustness_extra.py`` printed the numbers the paper drafts cite
# (``robustness_extra.txt``). The functions below compute the same readings; every one that
# bootstraps takes the generator it draws from, and ``write_robustness`` hands one generator
# to them in the script's order, so with ``--boot-seed 0 --n-boot 2000`` they print the
# published intervals (``conf_pipeline.robustness_check`` compares them).

#: The arm pairs the result script read Kendall's tau between, in its order.
TAU_PAIRS: tuple[tuple[str, str], ...] = (
    ("default", "confstream"), ("default", "frozen_prefix"), ("default", "oracle_static"),
    ("confstream", "frozen_prefix"), ("confstream", "oracle_static"), ("frozen_prefix", "oracle_static"),
)
#: Fewest algorithms a fold needs for its tau to be read: the result script's rule. (A1's
#: ``measure_agreement`` reads two, where tau-b can only be +1, -1 or undefined.)
MIN_TAU_ALGOS = 3


def tau_by_fold(xw: pd.DataFrame, yw: pd.DataFrame, keep: pd.DataFrame | None = None,
                min_algos: int = MIN_TAU_ALGOS) -> pd.Series:
    """Kendall's tau-b between two rankings of the algorithms, per row (fold or block).

    ``xw``, ``yw``: rows x algorithms of a reading (higher is better). A row's algorithms with
    a reading in both, and True in ``keep`` when given (a missing cell counts False), are
    ranked among themselves, ties by their average rank; a row with fewer than ``min_algos``
    of them has no tau. Identical rankings read 1, and a ranking constant in only one of the
    two reads no tau (as ``_tau``)."""
    from scipy import stats

    out = {}
    for row in xw.index.intersection(yw.index):
        x, y = xw.loc[row], yw.loc[row]
        m = x.notna() & y.notna()
        if keep is not None:
            m &= keep.loc[row].reindex(m.index).eq(True) if row in keep.index else False
        if m.sum() < min_algos:
            continue
        xr, yr = x[m].rank(ascending=False).to_numpy(), y[m].rank(ascending=False).to_numpy()
        if np.array_equal(xr, yr):
            out[row] = 1.0
        elif np.ptp(xr) > 0 and np.ptp(yr) > 0:
            out[row] = float(stats.kendalltau(xr, yr).statistic)
    return pd.Series(out, dtype=float)


def boot_mean(values: pd.Series, rng: np.random.Generator, n_boot: int = 2000, by_group: bool = True,
              groups=None) -> tuple[float, float, float]:
    """The mean of ``values`` and its 2.5-97.5 percentile bootstrap interval.

    ``by_group``: the entries are drawn with replacement within each stream group, groups in
    alphabetical order (the result script's draw); otherwise over all entries at once.
    ``groups``: the group of each entry, by default ``stream_group`` of the index (pass it when
    the index holds source blocks). An empty series reads NaN and draws nothing. The entries
    are the units: no seed is drawn, the readings being seed means already."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return np.nan, np.nan, np.nan
    g = np.array([C.stream_group(f) for f in values.index] if groups is None else list(groups))
    reps = []
    for _ in range(n_boot):
        if by_group:
            idx = np.concatenate([rng.choice(np.where(g == k)[0], size=(g == k).sum()) for k in np.unique(g)])
        else:
            idx = rng.choice(len(v), size=len(v))
        reps.append(v[idx].mean())
    return float(v.mean()), float(np.percentile(reps, CI[0])), float(np.percentile(reps, CI[1]))


def arm_wides(cells: pd.DataFrame, arms=ARMS) -> dict[str, pd.DataFrame]:
    """``{arm: folds x algorithms}`` of the seed means in ``cells`` (``cells_table``)."""
    algos = sorted(cells["algo"].unique())
    return {a: cells.pivot(index="fold", columns="algo", values=a).reindex(columns=algos)
            for a in arms if a in cells}


def arm_tau_table(wide: dict, rng: np.random.Generator, n_boot: int = 2000, pairs=TAU_PAIRS) -> pd.DataFrame:
    """Per pair of arms: the mean over the folds of ``tau_by_fold`` with its interval, over every
    fold (drawn within stream groups) and over the real ones (drawn among themselves)."""
    rows = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        t = tau_by_fold(wide[a], wide[b])
        real = t[[C.stream_group(f) == "real" for f in t.index]]
        for scope, s, grouped in (("all", t, True), ("real", real, False)):
            m, lo, hi = boot_mean(s, rng, n_boot, by_group=grouped)
            rows.append({"arm_a": a, "arm_b": b, "scope": scope, "n_folds": int(len(s)), "tau": m,
                         "low": lo, "high": hi})
    return pd.DataFrame(rows, columns=["arm_a", "arm_b", "scope", "n_folds", "tau", "low", "high"])


def _block_of(index) -> list[str]:
    return [C.source_block(f) for f in index]


def _block_groups(folds) -> dict[str, str]:
    return {C.source_block(f): C.stream_group(f) for f in folds}


def block_wides(wide: dict) -> dict[str, pd.DataFrame]:
    """``wide`` with the folds of a source block averaged into one row per block (a cell with
    no reading on a fold leaves that fold out of its block's mean)."""
    return {a: w.groupby(_block_of(w.index)).mean() for a, w in wide.items()}


def block_rankings(wide: dict, pairs=TAU_PAIRS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The INSECTS campaign as one block: rankings on the source blocks instead of the folds.

    Returns ``(tau, ranks)``: per pair of arms the mean over the blocks of ``tau_by_fold`` on
    the block means, and per arm the algorithms' average rank (1 = best) over every block and
    over the real ones, with Friedman's p on each set."""
    from scipy import stats

    blocks = block_wides(wide)
    folds = sorted({f for w in wide.values() for f in w.index})
    group = _block_groups(folds)
    tau = []
    for a, b in pairs:
        if a in blocks and b in blocks:
            t = tau_by_fold(blocks[a], blocks[b])
            tau.append({"arm_a": a, "arm_b": b, "n_blocks": int(len(t)),
                        "tau": float(t.mean()) if len(t) else np.nan})
    ranks = []
    for arm, w in blocks.items():
        r = w.rank(axis=1, ascending=False)
        for scope, sub in (("all", r), ("real", r[[group.get(b) == "real" for b in r.index]])):
            p = np.nan
            if len(sub) >= 2 and sub.shape[1] >= 3:
                p = float(stats.friedmanchisquare(*[sub[c] for c in sub.columns]).pvalue)
            ranks.append({"arm": arm, "scope": scope, "n_blocks": int(len(sub)), "friedman_p": p,
                          **{f"rank_{c}": float(sub[c].mean()) for c in sub.columns}})
    return (pd.DataFrame(tau, columns=["arm_a", "arm_b", "n_blocks", "tau"]), pd.DataFrame(ranks))


#: Declared filters of a weak default, as the paper's robustness appendix reads them: name,
#: what is read on the default arm, and the threshold. A broken default is one whose lagged
#: reading is undefined on more than half the windows (``analyze.DEGENERATE_SHARE``); the two
#: ARI floors leave out defaults that read nothing. Sensitivity filters, never the primary table.
WEAK_DEFAULT_FILTERS: tuple[tuple[str, str, float], ...] = (
    ("default not broken (undefined share <= 0.5)", "undefined_share", 0.5),
    ("default ARI >= 0.01", "ari", 0.01),
    ("default ARI >= 0.05", "ari", 0.05),
)


def weak_default_keep(default_ari: pd.DataFrame, undefined: pd.DataFrame,
                      filters=WEAK_DEFAULT_FILTERS) -> dict[str, pd.DataFrame]:
    """``{filter name: folds x algorithms of bool}``: the cells each filter keeps. The default's
    tail ARI and its ``lag_undefined_share`` decide, whatever measure is ranked afterwards, so
    a filter keeps the same cells under every measure. A cell with no reading is not kept."""
    out = {}
    for name, reading, threshold in filters:
        if reading == "undefined_share":
            out[name] = undefined.reindex(index=default_ari.index, columns=default_ari.columns) <= threshold + _ZERO
        else:
            out[name] = default_ari >= threshold
    return out


def weak_default_tau(wide: dict, undefined: pd.DataFrame, rng: np.random.Generator, n_boot: int = 2000,
                     arms=("confstream", "oracle_static"), filters=WEAK_DEFAULT_FILTERS) -> pd.DataFrame:
    """tau(default, arm) on the cells each weak-default filter keeps, mean over the folds with
    its interval (drawn within groups); ``algos_per_fold`` is the mean number of cells kept per
    fold read. ``undefined``: folds x algorithms of the default's ``lag_undefined_share``."""
    keeps = weak_default_keep(wide["default"], undefined, filters)
    rows = []
    for name, *_ in filters:
        keep = keeps[name]
        for arm in arms:
            if arm not in wide:
                continue
            t = tau_by_fold(wide["default"], wide[arm], keep=keep)
            m, lo, hi = boot_mean(t, rng, n_boot)
            rows.append({"filter": name, "arm": arm, "n_folds": int(len(t)),
                         "algos_per_fold": float(keep.sum(axis=1).loc[t.index].mean()) if len(t) else np.nan,
                         "tau": m, "low": lo, "high": hi})
    return pd.DataFrame(rows, columns=["filter", "arm", "n_folds", "algos_per_fold", "tau", "low", "high"])


def sobol_rankings(sweep_rows: pd.DataFrame, wide: dict, rng: np.random.Generator, n_boot: int = 2000,
                   n_points: int = C.SWEEP_N_POINTS, arms=("default", "confstream", "oracle_static")) -> dict:
    """How the arms rank the algorithms against a ranking by one random configuration.

    ``sweep_rows``: ``sweep.jsonl`` (``report.load_sweep``); its ok run rows of the algorithms
    in ``wide`` with ``config_id`` 0 (River's default) to ``n_points`` (the Sobol points, their
    distances already in units of ``s``) are read on their tail ARI, the last row of a
    configuration kept. Returns ``point_zero_tau`` (point 0 against the default arm, a check that
    the sweep reruns the default), ``per_point`` (per Sobol point, the mean tau against each arm),
    ``describe`` (its summary) and ``aggregate`` (the mean and the median over the points,
    ranked, against each arm, with intervals)."""
    algos = list(next(iter(wide.values())).columns)
    df = sweep_rows
    if "kind" in df:
        df = df[df["kind"] == "run"]
    cid = pd.to_numeric(df["config_id"], errors="coerce")
    keep = (df["status"] == "ok") & df["algo"].isin(algos) & cid.notna() & (cid <= n_points)
    sw = pd.DataFrame({"algo": df.loc[keep, "algo"], "fold": df.loc[keep, "fold"],
                       "cid": cid[keep].astype(int), "v": pd.to_numeric(df.loc[keep, TAIL], errors="coerce")})
    sw = sw.drop_duplicates(["algo", "fold", "cid"], keep="last")

    def pivot(sub):
        return sub.pivot(index="fold", columns="algo", values="v").reindex(columns=algos)

    zero = tau_by_fold(pivot(sw[sw["cid"] == 0]), wide["default"]) if "default" in wide else pd.Series(dtype=float)
    per = []
    for c in range(1, n_points + 1):
        wi = pivot(sw[sw["cid"] == c])
        per.append({"config_id": c, **{b: float(tau_by_fold(wi, wide[b]).mean()) for b in arms if b in wide}})
    per = pd.DataFrame(per)
    agg = []
    points = sw[sw["cid"] >= 1]
    for name in ("mean", "median"):
        wi = points.groupby(["fold", "algo"])["v"].agg(name).unstack().reindex(columns=algos)
        for b in arms:
            if b in wide:
                m, lo, hi = boot_mean(tau_by_fold(wi, wide[b]), rng, n_boot)
                agg.append({"aggregate": name, "arm": b, "tau": m, "low": lo, "high": hi})
    return {"point_zero_tau": float(zero.mean()) if len(zero) else np.nan, "per_point": per,
            "describe": per.drop(columns="config_id").describe(),
            "aggregate": pd.DataFrame(agg, columns=["aggregate", "arm", "tau", "low", "high"])}


def capture_parts(cells: pd.DataFrame, arms=CAPTURE_ARMS, min_denominator: float = 0.0) -> pd.DataFrame:
    """``capture_by_group`` with its parts: per group and arm, ``n_pairs``, ``numerator`` =
    sum(arm - default), ``denominator`` = sum(reference - default) and ``capture``. The ratio
    is undefined (NaN) with no pair, or when the denominator per pair is at most
    ``min_denominator``; 0 reproduces ``capture_by_group``. Nothing is clipped."""
    df = cells.copy()
    df["group"] = _groups(df["fold"])
    out = []
    for group in (*C.STREAM_GROUPS, ALL):
        sub = df if group == ALL else df[df["group"] == group]
        sub = sub[sub["tunable"].astype(bool)]
        for arm in arms:
            vals = sub[["default", arm, "oracle_static"]].apply(pd.to_numeric, errors="coerce").dropna()
            n = int(len(vals))
            num = float((vals[arm] - vals["default"]).sum())
            den = float((vals["oracle_static"] - vals["default"]).sum())
            out.append({"group": group, "arm": arm, "n_pairs": n, "numerator": num if n else np.nan,
                        "denominator": den if n else np.nan,
                        "capture": num / den if n and den / n > min_denominator else np.nan})
    return pd.DataFrame(out, columns=["group", "arm", "n_pairs", "numerator", "denominator", "capture"])


#: The published leave-one-out table's two extra rows: every INSECTS variant at once (the
#: block, also a row of the block unit) and the two streams whose defaults are broken or read on
#: few windows. A set none of whose folds is in the data gives no row.
LOO_EXTRA_DROPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("every INSECTS variant", tuple(f for f in C.FOLDS + C.X7_FOLDS if f.startswith("insects_"))),
    ("shuttle + kdd99", ("shuttle", "kdd99")),
)


def leave_one_out(cells: pd.DataFrame, loss: pd.DataFrame, unit: str = "fold", group: str = "real",
                  extra=(), arms=CAPTURE_ARMS, min_denominator: float = 0.0) -> pd.DataFrame:
    """The group's capture ratios and losses with each of its units left out in turn.

    ``unit="fold"`` leaves out one stream at a time, ``"block"`` one source block (every
    INSECTS variant together); ``extra`` (label, folds) adds rows that leave out a set. The
    first row, ``none``, is the full group. Per row: the capture ratio of every arm in ``arms``
    with its numerator and denominator (``capture_parts``), the two losses of
    ``losses_by_group``, their sum ``gap`` (reference minus the online arm) and
    ``selection_share = selection_loss / gap``, undefined when ``gap <= min_denominator``
    and never clipped. ``cells``: ``cells_table``; ``loss``: ``loss_cells``."""
    _check_unit(unit)
    present = sorted(set(cells["fold"]) | set(loss["fold"]))
    in_group = [f for f in present if C.stream_group(f) == group]
    drops = [("none", [])] + [(name, [in_group[i] for i in pos]) for _, name, pos in unit_index(in_group, unit)]
    for label, folds in extra:
        hit = [f for f in folds if f in in_group]
        if hit:
            drops.append((label, hit))
    rows = []
    for label, drop in drops:
        cap = capture_parts(cells[~cells["fold"].isin(drop)], arms, min_denominator)
        cap = cap[cap["group"] == group].set_index("arm")
        lg = losses_by_group(loss[~loss["fold"].isin(drop)]).set_index("group").loc[group]
        sel, sea = float(lg["selection_loss"]), float(lg["search_loss"])
        gap = sel + sea
        row = {"unit": unit, "group": group, "dropped": label, "n_dropped_folds": len(drop),
               "n_tunable_pairs": int(cap["n_pairs"].iloc[0]) if len(cap) else 0, "n_loss_pairs": int(lg["n_pairs"])}
        for arm in arms:
            row[f"capture_{arm}"] = cap.loc[arm, "capture"]
            row[f"capture_num_{arm}"] = cap.loc[arm, "numerator"]
            row[f"capture_den_{arm}"] = cap.loc[arm, "denominator"]
        row.update(selection_loss=sel, search_loss=sea, gap=gap,
                   selection_share=sel / gap if gap > min_denominator else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def per_stream_split(cells: pd.DataFrame, loss: pd.DataFrame, group: str = "real") -> pd.DataFrame:
    """Per stream of a group: its part of the attainable gain and of the selection loss.

    ``numerator`` / ``denominator``: sums over the stream's tunable pairs of online - default
    and reference - default; ``denominator_share`` its part of the group's denominator. Means
    over every pair of the stream of ``a_lag``, ``a_star_lag``, ``A_static`` and the two
    losses; ``selection_share`` its part of the group's summed selection losses."""
    df = cells[[C.stream_group(f) == group for f in cells["fold"]]]
    sub = df[df["tunable"].astype(bool)].copy()
    sub["numerator"] = sub["confstream"] - sub["default"]
    sub["denominator"] = sub["oracle_static"] - sub["default"]
    per = sub.groupby("fold")[["numerator", "denominator"]].sum()
    per["denominator_share"] = per["denominator"] / per["denominator"].sum()
    per["n_tunable"] = sub.groupby("fold").size()
    lr = loss[[C.stream_group(f) == group for f in loss["fold"]]].copy()
    lr["selection"] = lr["a_star_lag"] - lr["a_lag"]
    lr["search"] = lr["A_static"] - lr["a_star_lag"]
    per = per.join(lr.groupby("fold")[["a_lag", "a_star_lag", "A_static", "selection", "search"]].mean(),
                   how="outer")
    per = per.reindex(sorted(set(df["fold"]) | set(per.index)))
    per["n_tunable"] = per["n_tunable"].fillna(0).astype(int)
    per["selection_share"] = per["selection"] / per["selection"].sum()
    per.index.name = "fold"
    return per.reset_index()


def selection_share_interval(loss: pd.DataFrame, rng: np.random.Generator, n_boot: int = 2000,
                             group: str = "real", min_denominator: float = 0.0) -> dict:
    """The selection loss's share of the gap on a group's streams, the streams drawn with
    replacement (the result script's interval: per stream the mean over its algorithms first,
    no seed drawn). A replicate whose gap is at most ``min_denominator`` has no share and is
    counted in ``n_undefined``; ``p_search_le_0`` is the share of replicates whose mean search
    loss is at most 0 (adapting at least matching the best fixed configuration)."""
    lr = loss[[C.stream_group(f) == group for f in loss["fold"]]].copy()
    lr["selection"] = lr["a_star_lag"] - lr["a_lag"]
    lr["search"] = lr["A_static"] - lr["a_star_lag"]
    v = lr.groupby("fold")[["selection", "search"]].mean().to_numpy()
    out = {"group": group, "n_folds": int(len(v))}
    if not len(v):
        return {**out, **dict.fromkeys(("selection", "search", "gap", "share", "low", "high", "p_search_le_0"),
                                       np.nan), "n_used": 0, "n_undefined": 0}
    sel, sea = float(v[:, 0].mean()), float(v[:, 1].mean())
    gap = float(v.sum(axis=1).mean())
    shares, seas = [], []
    for _ in range(n_boot):
        idx = rng.choice(len(v), size=len(v))
        s_, e_ = v[idx, 0].mean(), v[idx, 1].mean()
        shares.append(s_ / (s_ + e_) if s_ + e_ > min_denominator else np.nan)
        seas.append(e_)
    shares = np.array(shares, dtype=float)
    used = shares[~np.isnan(shares)]
    low, high = np.percentile(used, CI) if len(used) else (np.nan, np.nan)
    return {**out, "selection": sel, "search": sea, "gap": gap,
            "share": sel / gap if gap > min_denominator else np.nan, "low": float(low), "high": float(high),
            "n_used": int(len(used)), "n_undefined": int(np.isnan(shares).sum()),
            "p_search_le_0": float(np.mean(np.array(seas) <= 0))}


def seedless_pairs(tunable: dict, summary: pd.DataFrame) -> dict:
    """How many tunable pairs belong to algorithms that take no seed
    (``benchmark.algorithms.registry.accepts_seed``), and whether the reference's reruns read
    the same on every seed for all of their pairs: on those pairs the reruns on seeds 2-4 repeat
    the selection seed's reading, so a reference picked on seed 1 keeps its optimism. Protocol
    algorithms only: ``Inputs.tunable`` also holds the implementation variants' pairs."""
    from benchmark.algorithms.registry import accepts_seed

    tunable = {k: v for k, v in tunable.items() if k[0] not in C.IMPL_VARIANTS}
    algos = sorted({a for a, _ in tunable})
    seedless = [a for a in algos if not accepts_seed(a)]
    ref = summary[(summary["arm"] == "oracle_static") & summary["algo"].isin(seedless)]
    sd = pd.to_numeric(ref.get("lag_ari_win_tail_sd", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return {"seedless_algos": seedless, "n_tunable": int(sum(tunable.values())),
            "n_tunable_seedless": int(sum(v for (a, _), v in tunable.items() if a in seedless)),
            "reference_sd_zero_on_seedless": bool((sd < _ZERO).all())}


# --- P1.2: NMI, the selections kept (a) or redone (b) ---------------------------------------------
#
# Two constructions, never to be confused. (a) "ARI-selected reference, NMI-evaluated": every
# selection stays as it was made (the reference configuration picked by ARI, the online runs,
# the labels' pick within the ensemble made by ARI) and is only read in NMI. (b) "NMI-reselected
# reference": the reference is picked again by NMI over the same sweep candidates, and the
# labels' pick within the ensemble by member NMI. (b) is a reselection over a finite candidate
# set on one sweep seed, not an NMI optimum, and is never called one.

#: Measure -> (the incumbent's same-window column, the lagged column) in ``windows.jsonl``;
#: a member log row carries the first.
MEASURE_COLUMNS: dict[str, tuple[str, str]] = {"ari": ("ari_win", "lag_ari_win"), "nmi": ("nmi_win", "lag_nmi_win")}
#: (pick, read) pairs of the labels' pick within the ensemble: picked by ``pick`` after window
#: t-1, read in ``read`` on window t. ``("ari", "ari")`` is ``report.regret_windows``' a_star_lag.
SELECTION_COMBOS: tuple[tuple[str, str], ...] = (("ari", "ari"), ("ari", "nmi"), ("nmi", "nmi"))
_MEMBER_SLOT = {"ari": 3, "nmi": 4}


def tail_values(windows: pd.DataFrame, tail_from: dict, column: str, parity: int | None = None) -> pd.DataFrame:
    """Per run (algo, fold, arm, seed): the mean of ``column`` over its tail windows, of one
    parity when ``parity`` is given, in the shape of ``Inputs.seeds``. ``tail_from`` as in
    ``parity_values``; a window without a reading (a single-class one) is in no mean."""
    if column not in windows:
        raise KeyError(f"the window log has no {column!r} column")
    df = windows[["algo", "fold", "arm", "seed", "window", column]].copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df["window"] = pd.to_numeric(df["window"], errors="coerce")
    start = [_tail_start(tail_from, k) for k in zip(df["algo"], df["fold"], df["seed"], df["arm"])]
    keep = df["window"] >= np.array(start)
    if parity is not None:
        keep &= df["window"] % 2 == parity
    out = df[keep].groupby(["algo", "fold", "arm", "seed"], sort=True, dropna=False)[column].mean().rename("value")
    return _per_run(windows, out)


def member_picks(by_window: dict, pick: str = "ari", read: str = "ari") -> dict:
    """``report._job_readings``' ``star`` and ``lag_pick`` of one job, with the labels'
    measure chosen: ``star`` the best active member's ``read`` on each window, ``lag_pick`` the
    members a pick by ``pick`` after window t-1 chooses, read in ``read`` on t (the same
    candidates as the report's: active on t, usable on t-1, ties all kept). ``by_window``:
    window -> member id -> ``(is_test, usable, score, ari, nmi)``."""
    p, r = _MEMBER_SLOT[pick], _MEMBER_SLOT[read]
    star, lag_pick = {}, {}
    for w, members in by_window.items():
        active = [m[r] for m in members.values() if not m[0] and m[r] is not None]
        if active:
            star[w] = max(active)
        prev = by_window.get(w - 1)
        if prev:
            cands = {mid: prev[mid][p] for mid, m in members.items()
                     if not m[0] and mid in prev and prev[mid][1] and prev[mid][p] is not None}
            if cands:
                top = max(cands.values())
                lag_pick[w] = {mid: members[mid][r] for mid, a in cands.items() if a == top}
    return {"star": star, "lag_pick": lag_pick}


def load_member_picks(out_dir: Path, arms=("confstream",), combos=SELECTION_COMBOS) -> dict:
    """``{job: {"star": {read: ...}, "lag_pick": {(pick, read): ...}}}`` for every member log of
    ``arms``, one job file at a time (``report.member_readings`` reads ARI only), with ``job =
    (algo, fold, seed, arm, config_id)``. A log without ``nmi_win`` gives NMI picks of nothing."""
    out = {}
    for path in report._member_files(out_dir, arms=set(arms)):
        jobs: dict[tuple, dict] = {}
        for row in C.read_jsonl(path):
            key = (row["algo"], row["fold"], row["seed"], row["arm"], report._cid(row.get("config_id")))
            jobs.setdefault(key, {}).setdefault(int(row["window"]), {})[row["member_id"]] = (
                bool(row.get("is_test")),
                row.get("feasible") is not False and row.get("silhouette_defined") is not False,
                report._num(row.get("score")), report._num(row.get("ari_win")), report._num(row.get("nmi_win")))
        for key, by_window in jobs.items():
            picks = {c: member_picks(by_window, *c) for c in combos}
            out[key] = {"star": {r: picks[(p, r)]["star"] for p, r in combos},
                        "lag_pick": {c: v["lag_pick"] for c, v in picks.items()}}
    return out


def selection_windows(windows: pd.DataFrame, readings: dict, tail_from: dict, arm: str = "confstream",
                      combos=SELECTION_COMBOS) -> pd.DataFrame:
    """Per tail window of every job of ``arm`` and per (pick, read) of ``combos`` whose columns
    the window log has: ``a_lag`` (the incumbent named after t-1, read on t), ``a_star_lag``
    and ``used``, computed by ``report._regret_series`` itself on the ``read`` columns, so the
    windows, the tie-break toward the incumbent and the masks are the report's. ``readings``:
    ``load_member_picks``. Combinations a log cannot give are absent (``selection_loss_seeds``
    says so)."""
    cols = ["algo", "fold", "seed", "arm", "window", "pick", "read", "a_lag", "a_star_lag", "used"]
    win = windows[windows["arm"] == arm] if "arm" in windows else windows.iloc[0:0]
    combos = [(p, r) for p, r in combos if all(c in win for c in MEASURE_COLUMNS[r])]
    frames = []
    for (algo, fold, seed), g in win.groupby(["algo", "fold", "seed"], sort=True):
        start = _tail_start(tail_from, (algo, fold, seed, arm))
        g = g[pd.to_numeric(g["window"], errors="coerce") >= start]
        job = readings.get((algo, fold, seed, arm, None)) or {}
        for p, r in combos:
            same, lag = MEASURE_COLUMNS[r]
            g_r = pd.DataFrame({"window": g["window"], "lag_ari_win": g[lag], "ari_win": g[same],
                                **({"incumbent_id": g["incumbent_id"]} if "incumbent_id" in g else {})})
            ser = report._regret_series(g_r, {"star": job.get("star", {}).get(r),
                                              "lag_pick": job.get("lag_pick", {}).get((p, r))} if job else None)
            frames.append(pd.DataFrame({"algo": algo, "fold": fold, "seed": seed, "arm": arm,
                                        "window": ser["window"], "pick": p, "read": r, "a_lag": ser["lag_ari_win"],
                                        "a_star_lag": ser["a_star_lag"], "used": ser["used"]}))
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def selection_loss_seeds(sel: pd.DataFrame, pick: str, read: str, windows=None) -> pd.DataFrame:
    """``loss_seeds`` of one (pick, read) of ``selection_windows``: per (algo, fold, seed) the
    means of ``a_lag`` and ``a_star_lag`` over the used windows (among ``windows``, a callable
    ``window index -> bool``, when given)."""
    sub = sel[(sel["pick"] == pick) & (sel["read"] == read)]
    if sub.empty:
        raise KeyError(f"no {pick}-picked {read} readings: the window log lacks "
                       f"{' / '.join(MEASURE_COLUMNS[read])} or the run has no {read}")
    used = sub[sub["used"].astype(bool)]
    if windows is not None:
        used = used[[bool(windows(int(w))) for w in used["window"]]]
    g = used.groupby(["algo", "fold", "seed"], sort=True)
    return g.agg(a_lag=("a_lag", "mean"), a_star_lag=("a_star_lag", "mean")).reset_index()


def load_sweep_measures(out_dir: Path, columns=("lag_ari_win", "lag_nmi_win")) -> tuple[pd.DataFrame, dict, set]:
    """``sweep_windows.jsonl`` with the lagged columns of several measures, ``{(algo, fold):
    tail_from}`` and the candidate set the published reference was picked from: the (algo, fold,
    config_id) of the ok sweep rows with a tail ARI reading (``_common.best_static``'s rows)."""
    keys = ("algo", "fold", "seed", "config_id", "window", *columns)
    path = Path(out_dir) / "sweep_windows.jsonl"
    rows = [{k: r.get(k) for k in keys} for r in C.read_jsonl(path)] if path.exists() else []
    sweep_win = _protocol(pd.DataFrame(rows, columns=list(keys)))
    sweep = report.load_sweep(out_dir)
    tail, cands = {}, set()
    if not sweep.empty:
        runs = sweep[sweep["kind"] == "run"] if "kind" in sweep else sweep
        if "tail_from" in runs:
            for a, f, t in zip(runs["algo"], runs["fold"], pd.to_numeric(runs["tail_from"], errors="coerce")):
                if not pd.isna(t):
                    tail[(a, f)] = int(t)
        if TAIL in runs:
            ok = runs[(runs["status"] == "ok") & pd.to_numeric(runs[TAIL], errors="coerce").notna()]
            cands = {(a, f, report._cid(c)) for a, f, c in zip(ok["algo"], ok["fold"], ok["config_id"])}
    return sweep_win, tail, cands


def sweep_tail_means(sweep_win: pd.DataFrame, sweep_tail: dict, column: str, parity: int | None = None) -> pd.DataFrame:
    """Per (algo, fold, config_id): the mean of ``column`` over the sweep's tail windows (of one
    parity when given), every seed the sweep ran pooled, and ``n_windows``."""
    if column not in sweep_win:
        raise KeyError(f"sweep_windows.jsonl was read without {column!r}")
    df = sweep_win[["algo", "fold", "config_id", "window", column]].copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df["window"] = pd.to_numeric(df["window"], errors="coerce")
    start = [_tail_start(sweep_tail, k) for k in zip(df["algo"], df["fold"])]
    keep = df["window"] >= np.array(start)
    if parity is not None:
        keep &= df["window"] % 2 == parity
    g = df[keep].dropna(subset=[column]).groupby(["algo", "fold", "config_id"], sort=True)[column]
    return g.agg(value="mean", n_windows="count").reset_index()


def reselect_reference(sweep_win: pd.DataFrame, sweep_tail: dict, column: str, candidates: set | None = None,
                       parity: int | None = None) -> pd.DataFrame:
    """Per (algo, fold): the candidate configuration with the best tail mean of ``column`` on
    the sweep (``sweep_tail_means``), ties to the smaller ``config_id`` as ``best_static``.
    ``candidates``: (algo, fold, config_id) allowed (``load_sweep_measures``), every swept
    configuration when None. The reading it was picked on is ``select_value``: an in-sample
    maximum over ``n_candidates``, so optimistic by construction."""
    cols = ["algo", "fold", "config_id", "config_source", "select_value", "n_candidates"]
    means = sweep_tail_means(sweep_win, sweep_tail, column, parity)
    if candidates is not None:
        means = means[[(a, f, report._cid(c)) in candidates
                       for a, f, c in zip(means["algo"], means["fold"], means["config_id"])]]
    rows = []
    for (algo, fold), g in means.groupby(["algo", "fold"], sort=True):
        g = g.assign(config_id=g["config_id"].astype(int)).sort_values("config_id")
        best = g.loc[g["value"].idxmax()]
        rows.append({"algo": algo, "fold": fold, "config_id": int(best["config_id"]),
                     "config_source": report.config_source(best["config_id"]),
                     "select_value": float(best["value"]), "n_candidates": int(len(g))})
    return pd.DataFrame(rows, columns=cols)


def tunable_gate(seeds: pd.DataFrame, reference: pd.DataFrame, min_gain: float = C.MIN_CEILING_GAIN,
                 z: float = report.Z_GATE) -> dict:
    """``report.headroom_table``'s gate on any measure: (algo, fold) -> ``gain >= min_gain`` and
    ``gain > z * se``, with ``gain`` = reference - the default's seed mean and ``se`` the
    default's standard error over its seeds (0 for one seed). ``seeds``: rows of the default
    arm (others ignored); ``reference``: (algo, fold, value)."""
    d = seeds[seeds["arm"] == "default"].groupby(["algo", "fold"])["value"].agg(["mean", "std", "count"])
    ref = reference.set_index(["algo", "fold"])["value"]
    out = {}
    for key, row in d.iterrows():
        r = ref.get(key, np.nan)
        se = float(row["std"]) / np.sqrt(row["count"]) if row["count"] > 1 else 0.0
        gain = r - float(row["mean"])
        out[key] = bool(not pd.isna(gain) and gain >= min_gain - report._EPS and gain > z * se)
    return out


# --- P1.2: conclusions by statistical unit and reference construction --------------------------

#: The ``algo`` of a row pooled over the algorithms.
POOLED = "pooled"
#: Placeholder, not a finding: a ratio whose denominator, per pair, is at most this much (in
#: units of the measure) reads undefined rather than a number: a capture ratio over gains of
#: 0.004 ARI, or a selection share of a 0.001 gap, is a ratio of rounding. ``--min-denominator``.
MIN_DENOMINATOR = 0.01
#: The online arm's paired comparisons; the reference's is named ``confstream - reference``.
EFFECTS: tuple[tuple[str, str], ...] = (
    ("confstream", "default"), ("confstream", "frozen_prefix"), ("confstream", "oracle_static"),
)
_RATIOS = ("capture_confstream", "capture_frozen_prefix", "selection_share")


def _effect_name(a: str, b: str) -> str:
    return f"{a} - reference" if b == "oracle_static" else f"{a} - {b}"


@dataclass
class Construction:
    """One way of building the reference and of reading every arm, one column set of the
    conclusions table.

    ``seeds``: ``algo, fold, arm, seed, value`` for the protocol arms and the reference (arm
    ``oracle_static``), all read in ``metric`` on ``basis``; ``losses``: the online arm's
    ``algo, fold, seed, a_lag, a_star_lag`` on the same windows; ``tunable``: (algo, fold) ->
    bool, the pairs the capture ratios sum over, decided by ``gate``. ``effects=False`` when
    another construction already has the same reference readings (a construction that only
    changes the tunable set), so its ``online - reference`` adds no paired-effect test.
    """

    reference: str
    metric: str
    basis: str
    seeds: pd.DataFrame
    losses: pd.DataFrame
    tunable: dict
    gate: str = "ARI gate"
    reference_seeds: str = ""
    note: str = ""
    effects: bool = True


def construction_cells(con: Construction) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(cells, loss)`` of a construction, in the shapes of ``cells_table`` and ``loss_cells``:
    per (algo, fold) the seed mean of every arm with ``tunable``, and the online arm's seed means
    of ``a_lag``, ``a_star_lag`` with the reference's seed mean as ``A_static``."""
    wide = con.seeds.pivot_table(index=["algo", "fold"], columns="arm", values="value", aggfunc="mean")
    wide = wide.reindex(columns=list(ARMS)).reset_index()
    wide.columns.name = None
    wide.insert(2, "tunable", [bool(con.tunable.get((a, f), False)) for a, f in zip(wide["algo"], wide["fold"])])
    if con.losses.empty:
        loss = pd.DataFrame(columns=["algo", "fold", "a_lag", "a_star_lag"])
    else:
        loss = con.losses.groupby(["algo", "fold"], sort=True)[["a_lag", "a_star_lag"]].mean().reset_index()
    ref = wide[["algo", "fold", "oracle_static"]].rename(columns={"oracle_static": "A_static"})
    return wide, loss.merge(ref, on=["algo", "fold"], how="left")


def unit_effects(con: Construction, unit: str = "block") -> pd.DataFrame:
    """The online arm's paired differences per resampling unit: one row per (unit, algo or
    ``pooled``, comparison) with the mean over the unit's (algo, fold) cells of the seed-mean
    difference and ``n_pairs``. Differences are of seed means: the protocol arms ran on the
    same seeds, but the reference's reruns are other seeds and are never matched to them by
    index. Under ``unit="block"`` a block's folds are averaged first, so the seven INSECTS
    variants weigh one source."""
    _check_unit(unit)
    cells, _ = construction_cells(con)
    folds = sorted(cells["fold"].unique())
    rows = []
    for group, name, pos in unit_index(folds, unit):
        sub = cells[cells["fold"].isin([folds[i] for i in pos])]
        for a, b in EFFECTS:
            d = (pd.to_numeric(sub[a], errors="coerce") - pd.to_numeric(sub[b], errors="coerce"))
            ok = d.notna()
            comp = _effect_name(a, b)
            rows.append({"unit": name, "group": group, "algo": POOLED, "comparison": comp,
                         "value": float(d[ok].mean()) if ok.any() else np.nan, "n_pairs": int(ok.sum())})
            for algo, da in d[ok].groupby(sub.loc[ok, "algo"]):
                rows.append({"unit": name, "group": group, "algo": algo, "comparison": comp,
                             "value": float(da.mean()), "n_pairs": int(len(da))})
    return pd.DataFrame(rows, columns=["unit", "group", "algo", "comparison", "value", "n_pairs"])


def _unit_stats(m: dict, groups: np.ndarray, unit_of: np.ndarray, unit_group: np.ndarray, tunable: np.ndarray,
                algos: list, min_denominator: float) -> dict:
    """Every statistic of ``unit_statistics`` on one (possibly resampled) set of seed means:
    ``{(statistic, group, algo): {value, numerator, denominator, n_pairs, n_units}}``."""
    n_units_all = len(unit_group)
    onehot = (unit_of[None, :] == np.arange(n_units_all)[:, None]).astype(float)
    d, ref = m["default"], m["oracle_static"]
    parts = {}
    for a, b in EFFECTS:
        diff = m[a] - m[b]
        valid = ~np.isnan(diff)
        parts[_effect_name(a, b)] = (np.where(valid, diff, 0.0) @ onehot.T, valid.astype(float) @ onehot.T)

    def units_of(mask2d):
        return int(len(np.unique(unit_of[mask2d.any(axis=0)])))

    out = {}
    for group in (*C.STREAM_GROUPS, ALL):
        in_g = np.ones(len(groups), bool) if group == ALL else groups == group
        u_in = np.ones(n_units_all, bool) if group == ALL else unit_group == group
        for arm in CAPTURE_ARMS:
            mask = tunable & in_g[None, :] & ~np.isnan(d) & ~np.isnan(ref) & ~np.isnan(m[arm])
            n = int(mask.sum())
            num, den = float(np.sum((m[arm] - d)[mask])), float(np.sum((ref - d)[mask]))
            out[(f"capture_{arm}", group, POOLED)] = {
                "value": num / den if n and den / n > min_denominator else np.nan,
                "numerator": num if n else np.nan, "denominator": den if n else np.nan,
                "n_pairs": n, "n_units": units_of(mask)}
        lag, star = m["a_lag"], m["a_star_lag"]
        mask = in_g[None, :] & ~np.isnan(lag) & ~np.isnan(star) & ~np.isnan(ref)
        n = int(mask.sum())
        sel = float(np.mean((star - lag)[mask])) if n else np.nan
        sea = float(np.mean((ref - star)[mask])) if n else np.nan
        gap = sel + sea
        base = {"n_pairs": n, "n_units": units_of(mask), "numerator": np.nan, "denominator": np.nan}
        out[("selection_loss", group, POOLED)] = {**base, "value": sel}
        out[("search_loss", group, POOLED)] = {**base, "value": sea}
        out[("gap", group, POOLED)] = {**base, "value": gap}
        out[("selection_share", group, POOLED)] = {
            **base, "value": sel / gap if n and gap > min_denominator else np.nan,
            "numerator": sel, "denominator": gap}
        for name, (s, c) in parts.items():
            su, cu = s[:, u_in], c[:, u_in]
            tot_s, tot_c = su.sum(axis=0), cu.sum(axis=0)
            ok = tot_c > 0
            out[(name, group, POOLED)] = {
                "value": float(np.mean(tot_s[ok] / tot_c[ok])) if ok.any() else np.nan,
                "numerator": np.nan, "denominator": np.nan, "n_pairs": int(tot_c.sum()), "n_units": int(ok.sum())}
            for i, algo in enumerate(algos):
                ok_a = cu[i] > 0
                out[(name, group, algo)] = {
                    "value": float(np.mean(su[i, ok_a] / cu[i, ok_a])) if ok_a.any() else np.nan,
                    "numerator": np.nan, "denominator": np.nan, "n_pairs": int(cu[i].sum()),
                    "n_units": int(ok_a.sum())}
    return out


def unit_statistics(con: Construction, unit: str = "fold", n_boot: int = 2000, seed: int = 0,
                    min_denominator: float = MIN_DENOMINATOR) -> pd.DataFrame:
    """Point and percentile interval of every conclusion of a construction under one unit.

    Statistics, per stream group and pooled: the capture ratios of both deployable arms (sums
    over the tunable pairs, ``numerator`` / ``denominator``), the online arm's selection and
    search losses, their sum ``gap`` and ``selection_share = selection / gap`` (means over the
    pairs); and the paired effects ``confstream - default / frozen_prefix / reference``,
    pooled and per algorithm, averaged within each unit first and then over the units. The
    ratios and the losses keep their published definitions under both units, so only their
    intervals move with the unit; the paired effects average by unit, so with ``"block"`` the
    seven INSECTS variants weigh one source.

    The bootstrap draws the units within each stream group (``draw_units``), then the seeds of
    each drawn pair: the protocol arms and the loss terms by seed number together, the
    reference's reruns apart (``pairing="seed"``); seeds add within-pair noise, never units.
    A ratio is undefined when its denominator per pair is at most ``min_denominator``
    (``status`` "undefined"); a defined ratio some replicates leave undefined is "unstable"
    (``n_used`` < ``n_boot``). Nothing is clipped: a capture above 1 or a share outside [0, 1]
    is reported as it reads."""
    _check_unit(unit)
    cube = _cube(con.seeds, con.losses, con.tunable, by_seed=True)
    index = unit_index(cube.folds, unit)
    unit_of = np.empty(len(cube.folds), dtype=int)
    for k, (_, _, pos) in enumerate(index):
        unit_of[pos] = k
    unit_group = np.array([g for g, _, _ in index])
    means = {k: _seed_mean(v) for k, v in cube.arrays.items()}
    point = _unit_stats(means, cube.groups, unit_of, unit_group, cube.tunable, cube.algos, min_denominator)
    rng = np.random.default_rng(seed)
    n_seed = next(iter(cube.arrays.values())).shape[2]
    slots = (_family_slots(cube, own=False), _family_slots(cube, own=True))
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        folds, ids = draw_units(cube.folds, unit, rng, index)
        picks, own = (_draw_slots(sl, folds, n_seed, rng) for sl in slots)
        m = {k: _seed_mean(np.take_along_axis(v[:, folds, :], own if k in _OWN_SEEDS else picks, axis=2))
             for k, v in cube.arrays.items()}
        groups = cube.groups[folds]
        first = np.unique(ids, return_index=True)[1]
        stats = _unit_stats(m, groups, ids, groups[first], cube.tunable[:, folds], cube.algos, min_denominator)
        for k, v in stats.items():
            draws[k].append(v["value"])
    rows = []
    for (stat, group, algo), v in point.items():
        vals = np.array(draws[(stat, group, algo)], dtype=float)
        vals = vals[~np.isnan(vals)]
        low, high = np.percentile(vals, CI) if len(vals) else (np.nan, np.nan)
        status = ("undefined" if pd.isna(v["value"]) else
                  "unstable" if stat in _RATIOS and len(vals) < n_boot else "ok")
        rows.append({"unit": unit, "statistic": stat, "group": group, "algo": algo, "point": v["value"],
                     "low": float(low), "high": float(high), "n_used": int(len(vals)), "n_boot": int(n_boot),
                     "n_pairs": v["n_pairs"], "n_units": v["n_units"], "numerator": v["numerator"],
                     "denominator": v["denominator"], "status": status})
    return pd.DataFrame(rows)


def _wilcoxon(values) -> dict:
    """``paired_tests``' reading of one set of differences: Wilcoxon two-sided on the non-zero
    ones, the rank-biserial correlation and the median (zeros included)."""
    from scipy.stats import wilcoxon

    d = np.asarray(values, dtype=float)
    d = d[~np.isnan(d)]
    nonzero = d[np.abs(d) > _ZERO]
    p = stat = np.nan
    if len(nonzero):
        res = wilcoxon(nonzero, alternative="two-sided")
        p, stat = float(res.pvalue), float(res.statistic)
    return {"n_units": int(len(d)), "n_zero": int(len(d) - len(nonzero)),
            "mean": float(d.mean()) if len(d) else np.nan,
            "median_diff": float(np.median(d)) if len(d) else np.nan, "rank_biserial": rank_biserial(d),
            "statistic": stat, "p": p}


def paired_unit_tests(effects: pd.DataFrame, unit: str, family=(), by=()) -> pd.DataFrame:
    """Wilcoxon's signed-rank test over the units of ``unit_effects`` (folds or blocks), per
    stream group (``all``: every unit), algorithm (or ``pooled``) and comparison, and per value
    of the ``family`` and ``by`` columns, with Holm's correction within each group and each
    value of the ``family`` columns (``by`` splits tests, not families). With five real blocks
    the smallest two-sided p is 1/16: a block-level test on one group says little."""
    keys = list(family) + list(by)
    rows = []
    for group in (*C.STREAM_GROUPS, ALL):
        sub = effects if group == ALL else effects[effects["group"] == group]
        for key, g in sub.groupby(keys + ["algo", "comparison"], sort=True):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({"unit": unit, "group": group, **dict(zip(keys + ["algo", "comparison"], key)),
                         **_wilcoxon(g["value"])})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for _, g in out.groupby(["group", *family], sort=False):
        adj = holm([None if np.isnan(p) else p for p in g["p"]])
        out.loc[g.index, "p_holm"] = [np.nan if v is None else v for v in adj]
    return out


def tau_units(wide: dict, unit: str, rng: np.random.Generator, n_boot: int = 2000, pairs=TAU_PAIRS,
              keep: pd.DataFrame | None = None) -> pd.DataFrame:
    """Kendall's tau between two arms' rankings of the algorithms, per stream group, on the
    folds (``unit="fold"``, A1's and the result script's reading) or on the source blocks (each
    block's folds averaged first, ``block_rankings``), mean over the units with its interval
    (``boot_mean``: units drawn within groups for ``all``). ``keep``: folds x algorithms of
    the cells to rank (a weak-default filter), applied before any averaging. No seed is drawn:
    rankings are of seed means."""
    _check_unit(unit)
    if keep is not None:
        wide = {a: w.where(keep.reindex(index=w.index, columns=w.columns).eq(True)) for a, w in wide.items()}
    units = wide if unit == "fold" else block_wides(wide)
    folds = sorted({f for w in wide.values() for f in w.index})
    of = {f: C.stream_group(f) for f in folds} if unit == "fold" else _block_groups(folds)
    rows = []
    for a, b in pairs:
        if a not in units or b not in units:
            continue
        t = tau_by_fold(units[a], units[b])
        g = np.array([of[i] for i in t.index])
        for group in (*C.STREAM_GROUPS, ALL):
            s = t if group == ALL else t[g == group]
            gs = g if group == ALL else g[g == group]
            m, lo, hi = boot_mean(s, rng, n_boot, by_group=group == ALL, groups=gs)
            rows.append({"arm_a": a, "arm_b": b, "group": group, "n_units": int(len(s)), "tau": m,
                         "low": lo, "high": hi})
    return pd.DataFrame(rows, columns=["arm_a", "arm_b", "group", "n_units", "tau", "low", "high"])


#: Statistic of ``unit_statistics`` -> its conclusion; the reference-free ones appear once per
#: measure and window basis.
_CONCLUSION_OF = {
    "confstream - default": "online - default", "confstream - frozen_prefix": "online - frozen_prefix",
    "confstream - reference": "online - reference", "capture_confstream": "capture G, online",
    "capture_frozen_prefix": "capture G, frozen_prefix", "selection_loss": "selection loss (online)",
    "search_loss": "search loss (online)", "gap": "gap, reference - online", "selection_share": "selection share of the gap",
}
_REFERENCE_FREE = ("confstream - default", "confstream - frozen_prefix")
_TAU_OF = {("default", "confstream"): "tau(default, online)", ("confstream", "frozen_prefix"): "tau(online, frozen_prefix)",
           ("default", "oracle_static"): "tau(default, reference)", ("confstream", "oracle_static"): "tau(online, reference)"}
CONCLUSION_COLS = ["conclusion", "group", "metric", "unit", "reference", "subset", "value", "ci_low", "ci_high",
                   "n_units", "n_pairs", "numerator", "denominator", "n_used", "n_boot", "status", "note"]
PRIMARY = "all streams"


def _note(stat: str, value, status: str, min_denominator: float, n_pairs) -> str:
    if status == "undefined":
        if stat in _RATIOS:
            return "no pair" if not n_pairs else f"denominator per pair <= {min_denominator}"
        return "no pair" if not n_pairs else "no reading"
    notes = []
    if stat in _RATIOS and not pd.isna(value) and not 0 <= value <= 1:
        notes.append("outside [0, 1], not clipped")
    if status == "unstable":
        notes.append("denominator undefined in some replicates")
    return "; ".join(notes)


def conclusions_table(constructions, n_boot: int = 2000, seed: int = 0, min_denominator: float = MIN_DENOMINATOR,
                      default_ari: pd.DataFrame | None = None, undefined: pd.DataFrame | None = None,
                      filters=WEAK_DEFAULT_FILTERS, stats: dict | None = None) -> pd.DataFrame:
    """One row per conclusion x stream group x measure x unit x reference construction (and,
    for the rankings of the default, x weak-default filter), columns ``CONCLUSION_COLS``.

    The reference-free conclusions (online - default, online - frozen_prefix and their tau)
    are read once per measure and window basis, from the first construction that has them,
    with ``reference`` "none (<basis>)". ``subset`` is "all streams" for every primary row; the
    weak-default filters (``WEAK_DEFAULT_FILTERS``, decided on the default's ARI and
    ``lag_undefined_share`` under every measure) are sensitivity rows of tau(default, ·).
    ``default_ari``: folds x algorithms of the default's tail ARI (else read from the first ARI
    construction); ``undefined``: the same of its ``lag_undefined_share`` (without it the
    broken-default filter is left out); ``stats``: ``{(reference, unit): unit_statistics}``
    already computed with the same arguments."""
    constructions = list(constructions)
    stats = stats or {}
    if default_ari is None:
        ari = next((c for c in constructions if c.metric == "ari"), None)
        default_ari = arm_wides(construction_cells(ari)[0]).get("default") if ari is not None else None
    keeps = {}
    if default_ari is not None:
        active = [f for f in filters if f[1] != "undefined_share" or undefined is not None]
        keeps = weak_default_keep(default_ari, undefined if undefined is not None else default_ari, active)
    rows, seen = [], set()
    for con in constructions:
        first = (con.metric, con.basis) not in seen
        seen.add((con.metric, con.basis))
        free = f"none ({con.basis})"
        wide = arm_wides(construction_cells(con)[0])
        for unit in UNITS:
            st = stats.get((con.reference, unit))
            st = unit_statistics(con, unit, n_boot, seed, min_denominator) if st is None else st
            for r in st[st["algo"] == POOLED].to_dict("records"):
                if r["statistic"] in _REFERENCE_FREE and not first:
                    continue
                rows.append({"conclusion": _CONCLUSION_OF[r["statistic"]], "group": r["group"], "metric": con.metric,
                             "unit": unit, "reference": free if r["statistic"] in _REFERENCE_FREE else con.reference,
                             "subset": PRIMARY, "value": r["point"], "ci_low": r["low"], "ci_high": r["high"],
                             "n_units": r["n_units"], "n_pairs": r["n_pairs"], "numerator": r["numerator"],
                             "denominator": r["denominator"], "n_used": r["n_used"], "n_boot": n_boot,
                             "status": r["status"],
                             "note": _note(r["statistic"], r["point"], r["status"], min_denominator, r["n_pairs"])})
            rng = np.random.default_rng(seed)
            pairs = [p for p in _TAU_OF if first or "oracle_static" in p]
            sets = [(PRIMARY, None, pairs)] + [(name, keep, [p for p in pairs if p[0] == "default"])
                                               for name, keep in keeps.items()]
            for subset, keep, prs in sets:
                for r in tau_units(wide, unit, rng, n_boot, prs, keep).to_dict("records"):
                    pair = (r["arm_a"], r["arm_b"])
                    undefined_row = pd.isna(r["tau"])
                    rows.append({"conclusion": _TAU_OF[pair], "group": r["group"], "metric": con.metric, "unit": unit,
                                 "reference": con.reference if "oracle_static" in pair else free, "subset": subset,
                                 "value": r["tau"], "ci_low": r["low"], "ci_high": r["high"], "n_units": r["n_units"],
                                 "n_pairs": np.nan, "numerator": np.nan, "denominator": np.nan,
                                 "n_used": 0 if undefined_row else n_boot, "n_boot": n_boot,
                                 "status": "undefined" if undefined_row else "ok",
                                 "note": f"no unit with {MIN_TAU_ALGOS} ranked algorithms" if undefined_row else ""})
    return pd.DataFrame(rows, columns=CONCLUSION_COLS)


def constructions_table(constructions) -> pd.DataFrame:
    """What each reference construction is: measure, windows, tunable gate, the seeds its
    reference is read on, the pairs it has and its note."""
    rows = []
    for con in constructions:
        cells, loss = construction_cells(con)
        rows.append({"reference": con.reference, "metric": con.metric, "basis": con.basis, "gate": con.gate,
                     "reference_seeds": con.reference_seeds, "n_cells": int(len(cells)),
                     "n_reference": int(cells["oracle_static"].notna().sum()),
                     "n_tunable": int(cells["tunable"].sum()), "n_loss": int(loss["a_lag"].notna().sum()),
                     "note": con.note})
    return pd.DataFrame(rows)


# --- loading the window logs ---------------------------------------------------------------------

def _tail_from(runs: pd.DataFrame) -> dict:
    """(algo, fold, seed, arm) -> the run's ``tail_from``."""
    if runs.empty or "tail_from" not in runs:
        return {}
    vals = pd.to_numeric(runs["tail_from"], errors="coerce")
    return {(a, f, s, arm): int(t) for a, f, s, arm, t in
            zip(runs["algo"], runs["fold"], runs["seed"], runs["arm"], vals) if not pd.isna(t)}


def current_windows(out_dir: Path, runs: pd.DataFrame) -> pd.DataFrame:
    """``windows.jsonl`` rows of the current ok runs only: protocol algorithms, and for the
    reference only the reruns of the configuration ``oracle_static.json`` names."""
    win = _protocol(report.load_windows(out_dir))
    if win.empty or runs.empty:
        return win
    jobs = {(a, f, s, arm, report._cid(c)) for a, f, s, arm, c in
            zip(runs["algo"], runs["fold"], runs["seed"], runs["arm"], runs.get("config_id", [None] * len(runs)))}
    cid = win["config_id"] if "config_id" in win else pd.Series([None] * len(win), index=win.index)
    keep = [(a, f, s, arm, report._cid(c)) in jobs for a, f, s, arm, c in
            zip(win["algo"], win["fold"], win["seed"], win["arm"], cid)]
    return win[keep].reset_index(drop=True)


def load_sweep_windows(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """``sweep_windows.jsonl`` (only the columns A2 reads) and ``{(algo, fold): tail_from}``
    from ``sweep.jsonl``."""
    keys = ("algo", "fold", "seed", "config_id", "window", "lag_ari_win")
    path = Path(out_dir) / "sweep_windows.jsonl"
    rows = [{k: r.get(k) for k in keys} for r in C.read_jsonl(path)] if path.exists() else []
    sweep_win = pd.DataFrame(rows, columns=list(keys))
    sweep = report.load_sweep(out_dir)
    tail = {}
    if not sweep.empty and "tail_from" in sweep:
        for a, f, t in zip(sweep["algo"], sweep["fold"], pd.to_numeric(sweep["tail_from"], errors="coerce")):
            if not pd.isna(t):
                tail[(a, f)] = int(t)
    return _protocol(sweep_win), tail


# --- P1.2: the reference constructions of an out dir ------------------------------------------------

#: Every reference construction the conclusions table crosses, in its order. The first is the
#: published one; the NMI ones exist only when the logs carry NMI (``build_constructions``).
REFERENCE_CONSTRUCTIONS: tuple[str, ...] = (
    "ARI-selected reference",
    "ARI-selected reference, even/odd",
    "ARI-selected reference, NMI-evaluated",
    "ARI-selected reference, NMI-evaluated, sweep seed",
    "NMI-reselected reference",
    "NMI-reselected reference, NMI gate",
    "NMI-reselected reference, even/odd",
)
BASIS_TAIL, BASIS_ODD = "all tail windows", "odd tail windows"
NMI_REFERENCE_COLS = [
    "algo", "fold", "ari_config_id", "ari_config_source", "ari_repick_config_id", "ari_repick_matches",
    "nmi_config_id", "nmi_config_source", "n_candidates", "same_config", "default_nmi", "ari_ref_nmi_reruns",
    "ari_ref_nmi_sweep", "nmi_ref_nmi_sweep", "reselection_gain", "tunable_ari", "tunable_nmi",
    "nmi_even_config_id", "nmi_even_select", "nmi_odd_report",
]


def _ref_rows(ref: pd.DataFrame, col: str) -> pd.DataFrame:
    """A reference read once (a sweep run, not reruns) as ``oracle_static`` rows of seed 0: a
    label, not a protocol seed, so ``pairing="seed"`` never pairs it with one."""
    return pd.DataFrame({"algo": ref["algo"], "fold": ref["fold"], "arm": "oracle_static", "seed": 0,
                         "value": pd.to_numeric(ref[col], errors="coerce")}).dropna(subset=["value"])


def build_constructions(out_dir: Path, inputs: Inputs | None = None, windows: pd.DataFrame | None = None,
                        sweep: tuple | None = None, readings: dict | None = None
                        ) -> tuple[list, pd.DataFrame, list[str]]:
    """The reference constructions of ``REFERENCE_CONSTRUCTIONS`` an out dir's logs allow, the
    per-pair table of the NMI re-pick (``NMI_REFERENCE_COLS``) and a note for every construction
    left out.

    - ARI-selected reference: the published one (``Inputs``), every arm on all tail windows.
    - ARI-selected reference, even/odd: A2's ``interleaved`` (picked on the even tail windows of
      the sweep, every arm and the losses read on the odd ones).
    - (a) ARI-selected reference, NMI-evaluated: every selection kept as ARI made it (the
      reference configuration and its reruns, the online runs, the labels' pick within the
      ensemble by member ARI) and read in NMI; the tunable set is the published one.
    - (a) on the sweep seed: the same pick read in NMI on the sweep run it was picked on, the
      basis (b) has, so that (b) - (a) is the reselection and not a change of seeds.
    - (b) NMI-reselected reference: re-picked by tail NMI over the published candidate set on
      the sweep (``reselect_reference``), read in-sample, and the labels' pick within the
      ensemble by member NMI; a maximum over a finite set on one seed, not an NMI optimum.
    - (b) with an NMI gate: the same, with the tunable set decided in NMI (``tunable_gate``).
    - (b) even/odd: A2's interleaved construction with NMI in place of ARI.
    """
    out_dir = Path(out_dir)
    inputs = load_inputs(out_dir) if inputs is None else inputs
    windows = current_windows(out_dir, inputs.runs) if windows is None else windows
    tail_from = _tail_from(inputs.runs)
    sweep_all, sweep_tail, cands = load_sweep_measures(out_dir) if sweep is None else sweep
    odd = lambda w: w % 2 == ODD  # noqa: E731
    names = REFERENCE_CONSTRUCTIONS
    ref_seeds = inputs.seeds[inputs.seeds["arm"] == "oracle_static"]
    reruns = f"reruns on seeds {_seed_list(ref_seeds['seed'])}" if len(ref_seeds) else "no rerun"
    swept = _seed_list(sweep_all["seed"].dropna()) if len(sweep_all) else None
    protocol = windows[windows["arm"].isin(list(C.ARMS))] if "arm" in windows else windows
    cons = [Construction(names[0], "ari", BASIS_TAIL, inputs.seeds, loss_seeds(inputs.regret), inputs.tunable,
                         reference_seeds=reruns,
                         note="published: picked by tail ARI over the sweep's candidates, read on its reruns")]
    if len(sweep_all):
        inter = interleaved_reference(sweep_all[["algo", "fold", "seed", "config_id", "window", "lag_ari_win"]],
                                      sweep_tail)
        cons.append(Construction(
            names[1], "ari", BASIS_ODD, pd.concat([parity_values(protocol, tail_from, ODD), _ref_rows(inter, "report_value")],
                                                  ignore_index=True),
            loss_seeds(inputs.regret, windows=odd), inputs.tunable,
            reference_seeds=f"sweep seed {swept}, picked on even, read on odd tail windows",
            note="A2's interleaved reference; every arm and the losses on the odd tail windows"))
    notes = []
    missing = [c for c in MEASURE_COLUMNS["nmi"] if c not in windows or not windows[c].notna().any()]
    if not missing:
        readings = load_member_picks(out_dir) if readings is None else readings
        if not any((r.get("star") or {}).get("nmi") for r in readings.values()):
            missing.append("nmi_win in members/*.jsonl")
    if missing:
        notes.append(f"no NMI construction: the logs lack {', '.join(missing)}")
        return cons, pd.DataFrame(columns=NMI_REFERENCE_COLS), notes
    sel = selection_windows(windows, readings, tail_from)
    nmi = tail_values(windows[windows["arm"].isin(list(ARMS))], tail_from, "lag_nmi_win")
    protocol_nmi = nmi[nmi["arm"].isin(list(C.ARMS))]
    loss_a, loss_b = selection_loss_seeds(sel, "ari", "nmi"), selection_loss_seeds(sel, "nmi", "nmi")
    cons.append(Construction(names[2], "nmi", BASIS_TAIL, nmi, loss_a, inputs.tunable, reference_seeds=reruns,
                             note="(a): every selection made by ARI, read in NMI"))
    oracle = report._read_oracle(out_dir) or {}
    pairs = ref_seeds[["algo", "fold"]].drop_duplicates().sort_values(["algo", "fold"])
    published = pd.DataFrame({"algo": pairs["algo"], "fold": pairs["fold"], "config_id": [
        report._cid((report._oracle_entry(oracle, a, f) or {}).get("config_id")) for a, f in zip(pairs["algo"], pairs["fold"])]})
    table = published.rename(columns={"config_id": "ari_config_id"})
    table["ari_config_source"] = [report.config_source(c) for c in table["ari_config_id"]]
    repick = reselect_reference(sweep_all, sweep_tail, "lag_ari_win", cands) if len(sweep_all) else None
    if repick is not None:
        table = table.merge(repick[["algo", "fold", "config_id"]].rename(columns={"config_id": "ari_repick_config_id"}),
                            on=["algo", "fold"], how="left")
    ref_a = nmi[nmi["arm"] == "oracle_static"].groupby(["algo", "fold"])["value"].mean().rename("ari_ref_nmi_reruns")
    d_nmi = protocol_nmi[protocol_nmi["arm"] == "default"].groupby(["algo", "fold"])["value"].mean().rename("default_nmi")
    table = table.join(ref_a, on=["algo", "fold"]).join(d_nmi, on=["algo", "fold"])
    table["tunable_ari"] = [bool(inputs.tunable.get((a, f), False)) for a, f in zip(table["algo"], table["fold"])]
    if "lag_nmi_win" not in sweep_all or not sweep_all["lag_nmi_win"].notna().any():
        notes.append("no NMI-reselected reference: sweep_windows.jsonl has no lag_nmi_win")
        return cons, table.reindex(columns=NMI_REFERENCE_COLS), notes
    means = sweep_tail_means(sweep_all, sweep_tail, "lag_nmi_win")
    means["config_id"] = means["config_id"].astype(int)
    ari_sweep = published.dropna(subset=["config_id"]).astype({"config_id": int}).merge(
        means, on=["algo", "fold", "config_id"], how="inner")
    cons.append(Construction(names[3], "nmi", BASIS_TAIL, pd.concat([protocol_nmi, _ref_rows(ari_sweep, "value")],
                                                                    ignore_index=True), loss_a, inputs.tunable,
                             reference_seeds=f"sweep seed {swept}, in-sample",
                             note="(a) read on the sweep run it was picked on: the basis (b) is read on"))
    pick = reselect_reference(sweep_all, sweep_tail, "lag_nmi_win", cands)
    seeds_b = pd.concat([protocol_nmi, _ref_rows(pick, "select_value")], ignore_index=True)
    b_note = "(b): re-picked by tail NMI over the same candidates, labels' pick by member NMI"
    cons.append(Construction(names[4], "nmi", BASIS_TAIL, seeds_b, loss_b, inputs.tunable,
                             reference_seeds=f"sweep seed {swept}, in-sample (picked and read on the same windows)",
                             note=b_note))
    gate = tunable_gate(seeds_b, pick.rename(columns={"select_value": "value"}))
    cons.append(Construction(names[5], "nmi", BASIS_TAIL, seeds_b, loss_b, gate,
                             gate=f"NMI gate (gain >= {C.MIN_CEILING_GAIN} and > {report.Z_GATE} se, in NMI)",
                             reference_seeds=f"sweep seed {swept}, in-sample", note=b_note + "; tunable set by NMI",
                             effects=False))
    nmi_sw = sweep_all[["algo", "fold", "seed", "config_id", "window", "lag_nmi_win"]].rename(
        columns={"lag_nmi_win": "lag_ari_win"})
    inter_b = interleaved_reference(nmi_sw, sweep_tail)
    nmi_odd = tail_values(protocol, tail_from, "lag_nmi_win", parity=ODD)
    cons.append(Construction(names[6], "nmi", BASIS_ODD, pd.concat([nmi_odd, _ref_rows(inter_b, "report_value")],
                                                                   ignore_index=True),
                             selection_loss_seeds(sel, "nmi", "nmi", windows=odd), inputs.tunable,
                             reference_seeds=f"sweep seed {swept}, picked on even, read on odd tail windows",
                             note="(b) interleaved: picked by NMI on the even tail windows, read on the odd ones"))
    table = table.merge(ari_sweep[["algo", "fold", "value"]].rename(columns={"value": "ari_ref_nmi_sweep"}),
                        on=["algo", "fold"], how="left")
    table = table.merge(pick.rename(columns={"config_id": "nmi_config_id", "config_source": "nmi_config_source",
                                             "select_value": "nmi_ref_nmi_sweep"}), on=["algo", "fold"], how="left")
    table = table.merge(inter_b[["algo", "fold", "config_id", "select_value", "report_value"]].rename(columns={
        "config_id": "nmi_even_config_id", "select_value": "nmi_even_select", "report_value": "nmi_odd_report"}),
        on=["algo", "fold"], how="left")
    table["same_config"] = table["nmi_config_id"] == table["ari_config_id"]
    table["ari_repick_matches"] = table["ari_repick_config_id"] == table["ari_config_id"]
    table["reselection_gain"] = table["nmi_ref_nmi_sweep"] - table["ari_ref_nmi_sweep"]
    table["tunable_nmi"] = [bool(gate.get((a, f), False)) for a, f in zip(table["algo"], table["fold"])]
    return cons, table.reindex(columns=NMI_REFERENCE_COLS), notes


def paired_effects_table(constructions, stats: dict) -> pd.DataFrame:
    """Per measure and window basis, unit, stream group, algorithm (or ``pooled``) and
    comparison: the paired effect with its interval (``unit_statistics``) and Wilcoxon's test
    over the units with Holm's correction over the family of one (measure, basis, unit,
    group). The reference-free comparisons are read once per (measure, basis); each
    construction with ``effects`` adds its ``confstream - reference``. ``stats``:
    ``{(reference, unit): unit_statistics}``."""
    eff, ci, seen = [], [], set()
    for con in constructions:
        first = (con.metric, con.basis) not in seen
        seen.add((con.metric, con.basis))
        if not (first or con.effects):
            continue
        keep = [_effect_name(a, b) for a, b in EFFECTS if first or b == "oracle_static"]
        for unit in UNITS:
            e = unit_effects(con, unit)
            e = e[e["comparison"].isin(keep)]
            label = np.where(e["comparison"] == "confstream - reference", con.reference, f"none ({con.basis})")
            eff.append(e.assign(metric=con.metric, basis=con.basis, reference=label, unit_kind=unit))
            st = stats[(con.reference, unit)]
            st = st[st["statistic"].isin(keep)]
            label = np.where(st["statistic"] == "confstream - reference", con.reference, f"none ({con.basis})")
            ci.append(st.assign(metric=con.metric, basis=con.basis, reference=label).rename(
                columns={"statistic": "comparison"}))
    if not eff:
        return pd.DataFrame()
    eff = pd.concat(eff, ignore_index=True)
    tests = pd.concat([paired_unit_tests(eff[eff["unit_kind"] == u], u, family=("metric", "basis"), by=("reference",))
                       for u in UNITS], ignore_index=True)
    ci = pd.concat(ci, ignore_index=True)[["metric", "basis", "reference", "unit", "group", "algo", "comparison",
                                           "point", "low", "high", "n_pairs", "n_used"]]
    out = tests.merge(ci, on=["metric", "basis", "reference", "unit", "group", "algo", "comparison"], how="left")
    cols = ["metric", "basis", "reference", "unit", "group", "algo", "comparison", "n_units", "n_pairs", "point",
            "low", "high", "n_used", "median_diff", "rank_biserial", "n_zero", "statistic", "p", "p_holm"]
    return out[cols]


# --- putting it together -------------------------------------------------------------------------

#: The constructions of the reference A2 compares; ``--reference-dir NAME=DIR`` adds one per name.
CONSTRUCTIONS: tuple[str, ...] = ("current", "interleaved", "interleaved_sobol_only")


def _seed_list(seeds) -> str | None:
    seeds = sorted({int(s) for s in seeds})
    return ",".join(map(str, seeds)) if seeds else None


def load_reference(ref_dir: Path) -> pd.DataFrame:
    """A reference built in another out dir (a sweep on several seeds, another search), read
    the way the published one is: per (algo, fold) the configuration its ``oracle_static.json``
    names, ``select_value`` the reading it was picked on, and ``report_value`` the mean tail ARI
    of its current reruns. A rerun on a seed that picked the configuration is left out, so
    the reading never rests on the seed of a maximum; a pair with no rerun left is dropped.
    """
    cols = ["algo", "fold", "config_id", "config_source", "select_value", "report_value", "n_candidates",
            "select_seeds", "report_seeds"]
    ref_dir = Path(ref_dir)
    oracle = report._read_oracle(ref_dir) or {}
    runs = report.current_runs(ref_dir)
    if runs.empty or "arm" not in runs:
        return pd.DataFrame(columns=cols)
    reruns = runs[(runs["arm"] == "oracle_static") & (runs["status"] == "ok")]
    rows = []
    for (algo, fold), g in reruns.groupby(["algo", "fold"], sort=True):
        entry = report._oracle_entry(oracle, algo, fold) or {}
        picked = {int(s) for s in (entry.get("select_seeds") or [])}
        g = g[~g["seed"].astype(int).isin(picked)]
        vals = pd.to_numeric(g[TAIL], errors="coerce").dropna() if TAIL in g else pd.Series(dtype=float)
        if vals.empty:
            continue
        cid = report._cid(entry.get("config_id"))
        rows.append({"algo": algo, "fold": fold, "config_id": cid, "config_source": report.config_source(cid),
                     "select_value": report._num(entry.get(entry.get("select_metric") or TAIL)),
                     "report_value": float(vals.mean()), "n_candidates": entry.get("n_ok"),
                     "select_seeds": _seed_list(picked), "report_seeds": _seed_list(g.loc[vals.index, "seed"])})
    return _protocol(pd.DataFrame(rows, columns=cols))


def _construction_rows(name: str, cap: pd.DataFrame, loss: pd.DataFrame) -> pd.DataFrame:
    out = cap.merge(loss[["group", "selection_loss", "search_loss"]], on="group", how="left")
    off = out["arm"] != "confstream"  # the losses are the online arm's
    out.loc[off, ["selection_loss", "search_loss"]] = np.nan
    return out.assign(construction=name)[["construction", "group", "arm", "n_pairs", "capture",
                                          "selection_loss", "search_loss"]]


def a2_tables(inputs: Inputs, windows: pd.DataFrame, tail_from: dict, sweep_win: pd.DataFrame,
              sweep_tail: dict, references: dict[str, pd.DataFrame] | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A2: the reference per (algo, fold) under each construction, and the capture ratios
    and losses each construction gives. ``current`` reads every arm on all tail windows;
    the interleaved ones read every arm, and the online arm's losses, on the odd ones.
    ``references`` (name -> ``load_reference``) adds one construction per name, read on all
    tail windows like ``current``; a pair it has no reading of keeps the published reference,
    and ``reference_from`` says which one each pair rests on."""
    oracle = report._read_oracle(inputs.out_dir) or {}
    ref_seeds = inputs.seeds[inputs.seeds["arm"] == "oracle_static"]
    current = ref_seeds.groupby(["algo", "fold"])["value"].mean().reset_index()
    entries = [((oracle.get(a) or {}).get(f) or {}) for a, f in zip(current["algo"], current["fold"])]
    refs = [current.assign(
        construction="current",
        config_id=[report._cid(e.get("config_id")) for e in entries],
        report_value=current["value"],
        select_value=[report._num(e.get(e.get("select_metric") or TAIL)) for e in entries],
        n_candidates=[e.get("n_ok") for e in entries],
        select_seeds=[_seed_list(e.get("select_seeds") or [1]) for e in entries],
        report_seeds=[_seed_list(g["seed"]) for _, g in ref_seeds.groupby(["algo", "fold"], sort=True)],
        reference_from="current")]
    refs[0]["config_source"] = [report.config_source(c) for c in refs[0]["config_id"]]
    caps = [_construction_rows("current", capture_by_group(cells_table(inputs)),
                               losses_by_group(loss_cells(inputs)))]
    odd = parity_values(windows[windows["arm"].isin(list(C.ARMS))], tail_from, ODD)
    for name, incumbents in (("interleaved", True), ("interleaved_sobol_only", False)):
        ref = interleaved_reference(sweep_win, sweep_tail, incumbents=incumbents)
        refs.append(ref.assign(construction=name))
        ref_rows = ref.assign(arm="oracle_static", seed=1, value=ref["report_value"])
        seeds = pd.concat([odd, ref_rows[["algo", "fold", "arm", "seed", "value"]]], ignore_index=True)
        loss = loss_cells(inputs, windows=lambda w: w % 2 == ODD,
                          reference=ref.rename(columns={"report_value": "A_static"}))
        caps.append(_construction_rows(name, capture_by_group(cells_table(inputs, seeds)), losses_by_group(loss)))
    for name, other in (references or {}).items():
        ref = refs[0].drop(columns=["construction", "value"]).merge(
            other, on=["algo", "fold"], how="left", suffixes=("_current", ""))
        found = ref["report_value"].notna()
        for col in ("config_id", "config_source", "select_value", "report_value", "n_candidates",
                    "select_seeds", "report_seeds"):
            ref[col] = ref[col].where(found, ref[f"{col}_current"])
        ref["reference_from"] = np.where(found, name, "current")
        ref = ref[[c for c in ref.columns if not c.endswith("_current")]]
        refs.append(ref.assign(construction=name))
        ref_rows = ref.assign(arm="oracle_static", seed=0, value=ref["report_value"])
        seeds = pd.concat([inputs.seeds[inputs.seeds["arm"] != "oracle_static"],
                           ref_rows[["algo", "fold", "arm", "seed", "value"]]], ignore_index=True)
        loss = loss_cells(inputs, reference=ref.rename(columns={"report_value": "A_static"}))
        caps.append(_construction_rows(name, capture_by_group(cells_table(inputs, seeds)), losses_by_group(loss)))
    cols = ["construction", "algo", "fold", "config_id", "config_source", "select_value", "report_value",
            "n_candidates", "select_seeds", "report_seeds", "reference_from"]
    return pd.concat(refs, ignore_index=True).reindex(columns=cols), pd.concat(caps, ignore_index=True)


def p12_tables(out_dir: Path, inputs: Inputs, windows: pd.DataFrame, cells: pd.DataFrame, summary: pd.DataFrame,
               sweep: tuple, n_boot: int = 2000, boot_seed: int = 0,
               min_denominator: float = MIN_DENOMINATOR) -> dict:
    """P1.2 on one out dir: the result script's checks, the reference constructions, the
    conclusions table, the paired effects by unit and the leave-one-out tables.

    The script's checks draw from one generator seeded ``boot_seed``, in the script's order
    (tau between arms, weak-default filters, Sobol points, selection share), so with seed 0 and
    2000 replicates they reproduce ``robustness_extra.txt``. Returns ``{"tables": {csv name:
    frame}, ...}`` with the pieces the markdown reads."""
    rng = np.random.default_rng(boot_seed)
    wide = arm_wides(cells)
    tau_arms = arm_tau_table(wide, rng, n_boot)
    blk_tau, blk_ranks = block_rankings(wide)
    undefined = None
    if "lag_undefined_share" in summary:
        d = summary[summary["arm"] == "default"]
        undefined = d.pivot_table(index="fold", columns="algo", values="lag_undefined_share", aggfunc="mean")
        undefined = undefined.reindex(columns=wide["default"].columns)
    weak = (weak_default_tau(wide, undefined, rng, n_boot) if undefined is not None else
            weak_default_tau(wide, wide["default"], rng, n_boot, filters=WEAK_DEFAULT_FILTERS[1:]))
    sobol = sobol_rankings(report.load_sweep(out_dir), wide, rng, n_boot) if len(wide) else None
    loss = loss_cells(inputs)
    share = selection_share_interval(loss, rng, n_boot)
    real_streams = per_stream_split(cells, loss)
    seedless = seedless_pairs(inputs.tunable, summary)

    cons, nmi_ref, notes = build_constructions(out_dir, inputs, windows, sweep)
    stats = {(c.reference, u): unit_statistics(c, u, n_boot, boot_seed, min_denominator) for c in cons for u in UNITS}
    conclusions = conclusions_table(cons, n_boot, boot_seed, min_denominator, default_ari=wide.get("default"),
                                    undefined=undefined, stats=stats)
    effects = paired_effects_table(cons, stats)
    loo = []
    for con in cons:
        c_cells, c_loss = construction_cells(con)
        for unit in UNITS:
            for group in C.STREAM_GROUPS:
                if not any(C.stream_group(f) == group for f in c_cells["fold"]):
                    continue
                extra = LOO_EXTRA_DROPS if unit == "fold" and group == "real" else ()
                loo.append(leave_one_out(c_cells, c_loss, unit, group, extra, min_denominator=min_denominator)
                           .assign(reference=con.reference, metric=con.metric))
    loo = pd.concat(loo, ignore_index=True) if loo else pd.DataFrame()
    unit_stats = pd.concat([st.assign(reference=r, metric=next(c.metric for c in cons if c.reference == r))
                            for (r, _), st in stats.items()], ignore_index=True)
    tables = {
        "conclusions.csv": conclusions, "constructions.csv": constructions_table(cons),
        "unit_statistics.csv": unit_stats, "paired_effects.csv": effects, "loo.csv": loo,
        "nmi_reference.csv": nmi_ref, "tau_arms.csv": tau_arms, "block_rankings_tau.csv": blk_tau,
        "block_rankings_ranks.csv": blk_ranks, "tau_weak_default.csv": weak, "real_streams.csv": real_streams,
        "selection_share.csv": pd.DataFrame([share]),
    }
    if sobol is not None:
        tables["sobol_rankings.csv"] = sobol["per_point"]
        tables["sobol_aggregate.csv"] = sobol["aggregate"]
    return {"tables": tables, "constructions": cons, "notes": notes, "sobol": sobol, "seedless": seedless,
            "share": share}


def rank_table(summary: pd.DataFrame, arms=ARMS, measures=RANKED) -> pd.DataFrame:
    """A1's ranks in long form: algo, fold, arm, measure, value, rank (1 = best)."""
    from . import analyze

    rows = []
    algos = analyze._algos(summary)
    for arm in arms:
        for m in measures:
            if m not in summary:
                continue
            wide = analyze._wide(summary, arm, m, algos)
            ranks = analyze._ranks(wide)
            for fold in wide.index:
                for algo in algos:
                    rows.append({"algo": algo, "fold": fold, "arm": arm, "measure": m,
                                 "value": wide.loc[fold, algo], "rank": ranks.loc[fold, algo]})
    return pd.DataFrame(rows, columns=["algo", "fold", "arm", "measure", "value", "rank"])


def write_robustness(out_dir: Path, n_boot: int = 2000, boot_seed: int = 0,
                     min_eval_windows: int = MIN_EVAL_WINDOWS, min_reach_ari: float = MIN_REACH_ARI,
                     rep_dir: Path | None = None, reference_dirs: dict[str, Path] | None = None,
                     min_denominator: float = MIN_DENOMINATOR) -> list[Path]:
    """Every analysis of this module on ``out_dir``: CSVs in ``<rep>/robustness/`` and
    ``<rep>/robustness.md``, with ``<rep>`` = ``<out_dir>/report`` unless ``rep_dir``.
    ``reference_dirs`` (name -> out dir) adds A2 constructions (``load_reference``);
    ``min_denominator`` is P1.2's threshold below which a ratio reads undefined."""
    from . import analyze

    out_dir = Path(out_dir)
    rep = Path(rep_dir) if rep_dir is not None else out_dir / C.REPORT_DIR.name
    csv_dir = rep / "robustness"
    csv_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(out_dir)
    windows = current_windows(out_dir, inputs.runs)
    tail_from = _tail_from(inputs.runs)
    cells = cells_table(inputs)

    losses = loss_seeds(inputs.regret)
    boot = bootstrap(inputs.seeds, losses, inputs.tunable, n_boot=n_boot, seed=boot_seed)
    n_pairs = pd.concat([capture_by_group(cells).assign(statistic="capture")[["statistic", "group", "arm", "n_pairs"]]]
                        + [losses_by_group(loss_cells(inputs)).assign(statistic=s, arm="confstream")
                           [["statistic", "group", "arm", "n_pairs"]] for s in ("selection_loss", "search_loss")])
    capture_losses = boot.merge(n_pairs, on=["statistic", "group", "arm"], how="left")
    tests = paired_tests(cells)

    values = measure_values(windows, tail_from)
    ranks = rank_table(values)
    per_fold_tau, tau = measure_agreement(values)

    # one read of sweep_windows.jsonl for A2 and P1.2: A2 reads the ARI columns only, as
    # load_sweep_windows returns them
    sweep_all, sweep_tail, sweep_cands = load_sweep_measures(out_dir)
    sweep_win = sweep_all[["algo", "fold", "seed", "config_id", "window", "lag_ari_win"]]
    references = {name: load_reference(d) for name, d in (reference_dirs or {}).items()}
    a2_ref, a2_cap = a2_tables(inputs, windows, tail_from, sweep_win, sweep_tail, references)

    single = single_class_table(windows, tail_from)
    summary = analyze._protocol(report.summary_table(out_dir))
    fried, kendall = without_folds(summary, arms=ARMS)

    reach_path = out_dir / "label_reach.jsonl"
    per_method = reach = None
    if reach_path.exists():
        per_method, reach = reach_table(pd.DataFrame(list(C.read_jsonl(reach_path))))
    agree = per_fold_tau[(per_fold_tau["arm"] == "confstream") & (per_fold_tau["pair"] == "ari-nmi")]
    best = cells.melt(id_vars=["algo", "fold"], value_vars=list(ARMS)).groupby("fold")["value"].max()
    usable = usable_streams(single, reach, agree.rename(columns={"tau": "tau_ari_nmi"}),
                            best.rename("best_tail_ari").reset_index(),
                            min_eval_windows=min_eval_windows, min_reach_ari=min_reach_ari)

    tables = {
        "capture_losses.csv": capture_losses, "a1_ranks.csv": ranks, "a1_agreement.csv": tau,
        "a1_agreement_per_fold.csv": per_fold_tau, "a1_measures.csv": values,
        "a2_reference.csv": a2_ref, "a2_capture.csv": a2_cap, "a3_tests.csv": tests,
        "a4_single_class.csv": single, "a4_friedman.csv": fried, "a4_kendall.csv": kendall,
        "usable_streams.csv": usable,
    }
    if per_method is not None:
        tables["a5_label_reach.csv"] = per_method
    p12 = p12_tables(out_dir, inputs, windows, cells, summary, (sweep_all, sweep_tail, sweep_cands),
                     n_boot=n_boot, boot_seed=boot_seed, min_denominator=min_denominator)
    tables.update(p12["tables"])
    paths = []
    for name, df in tables.items():
        df.to_csv(csv_dir / name, index=False)
        paths.append(csv_dir / name)
    md = _markdown(out_dir, inputs, n_boot, boot_seed, capture_losses, values, tau, a2_ref, a2_cap, tests,
                   single, fried, kendall, reach_path, reach, usable, min_eval_windows, min_reach_ari)
    md += "\n" + _p12_markdown(p12, n_boot, boot_seed, min_denominator)
    (rep / "robustness.md").write_text(md, encoding="utf-8")
    paths.append(rep / "robustness.md")
    return paths


# --- markdown ----------------------------------------------------------------------------------------

def _table(df: pd.DataFrame, digits: int = 3) -> str:
    from . import analyze

    return analyze._md_table(df, digits) if len(df) else "_(no rows)_"


def _markdown(out_dir, inputs, n_boot, boot_seed, capture_losses, values, tau, a2_ref, a2_cap, tests,
              single, fried, kendall, reach_path, reach, usable, min_eval_windows, min_reach_ari) -> str:
    seeds = inputs.seeds
    n_algos, n_folds = seeds["algo"].nunique(), seeds["fold"].nunique()
    lines = [
        "# Robustness of the full run's conclusions",
        "",
        f"Read from `{out_dir}`: {n_algos} algorithms, {n_folds} folds, arms {', '.join(ARMS)}; "
        "implementation variants are left out. Quality is the lagged ARI on the tail windows "
        "(from each run's `tail_from`); single-class windows have no external reading and are in no mean. "
        "No tuning run was made for this report.",
        "",
        "## Stream groups: capture ratio and losses, with bootstrap intervals (A3)",
        "",
        f"Capture ratio = sum(arm - default) / sum(reference - default) over the tunable pairs of a group. "
        f"Selection loss = label-picked member minus the silhouette's pick, search loss = reference minus the "
        f"label-picked member (online arm, tail). Intervals: {n_boot} bootstrap replicates (seed {boot_seed}), "
        "folds drawn within each group, then seed indices within each pair, the same for every arm; "
        "algorithms and the tunable set fixed; 2.5-97.5 percentiles. `n_used` counts the replicates where "
        "the statistic is defined.",
        "",
        _table(capture_losses[["statistic", "group", "arm", "n_pairs", "point", "low", "high", "n_used"]]),
        "",
        "## A1: rankings under ARI, NMI and purity",
        "",
        "Per arm and fold, the algorithms with a reading under both measures are ranked among themselves "
        "and compared by Kendall's tau-b; the table gives the mean over the folds of each group. Purity is 1 "
        "when every point is its own cluster, so read it beside the mean number of predicted labels "
        "(`a1_measures.csv`, column `n_pred_labels`).",
        "",
        _table(tau.pivot_table(index=["arm", "pair"], columns="group", values="tau").reset_index()
               .reindex(columns=["arm", "pair", *C.STREAM_GROUPS, ALL])),
        "",
        "## A2: a reference selected on the even tail windows",
        "",
        "`interleaved`: per (algo, fold), the sweep candidate with the best mean lagged ARI on the even tail "
        "windows, read on the odd ones; every arm, and the online arm's losses, are read on the odd windows "
        "too. `interleaved_sobol_only`: the same without the tuner's incumbents among the candidates. "
        "`current`: the published reference (selected on seed 1, reported on seeds 2-4), all tail windows. "
        "The interleaved references pool the windows of every seed the sweep ran (seed 1 alone unless it "
        "was swept on more); the tunable set is the published one throughout. A construction named after "
        "`--reference-dir` reads that directory's reference on all tail windows, its reruns on the seeds "
        "that did not pick it (`select_seeds`, `report_seeds` in `a2_reference.csv`); a pair it lacks keeps "
        "the published reference (`reference_from`).",
        "",
        _table(a2_cap),
        "",
    ]
    cur = a2_ref[a2_ref["construction"] == "current"].set_index(["algo", "fold"])["config_id"]
    for name in [c for c in dict.fromkeys(a2_ref["construction"]) if c != "current"]:
        other = a2_ref[a2_ref["construction"] == name].set_index(["algo", "fold"])["config_id"]
        shared = cur.index.intersection(other.index)
        moved = int(sum(cur[k] != other[k] for k in shared))
        line = f"- `{name}` picks another configuration than `current` on {moved} of {len(shared)} pairs"
        froms = a2_ref.loc[a2_ref["construction"] == name, "reference_from"]
        if (froms == name).any():
            line += f" ({int((froms == name).sum())} pairs read from its own directory)"
        lines.append(line + ".")
    lines += [
        "",
        "## A3: paired tests per algorithm",
        "",
        "Wilcoxon's signed-rank test over the folds on the seed means, two-sided, zero differences dropped "
        "(`n_zero`); Holm's correction over all the tests below; effect size as the matched-pairs "
        "rank-biserial correlation and the median difference.",
        "",
        _table(tests[["algo", "comparison", "n", "n_zero", "median_diff", "rank_biserial", "p", "p_holm"]], 4),
        "",
        "## A4: windows removed by the single-class rule",
        "",
        "Per fold: its windows, those holding one true class (no external reading), and the tail windows left "
        "to evaluate on. The logs record no ARI on a single-class window, so a ranking without the rule cannot "
        "be computed from them; the rankings are compared with and without KDD99 instead, where the rule "
        "removes the most windows.",
        "",
        _table(single),
        "",
        _table(fried[[c for c in fried.columns if not c.startswith("rank_")]]) if len(fried) else "_(no rows)_",
        "",
        _table(kendall),
        "",
        "## A5: how far the labels can be reached",
        "",
    ]
    if reach is None:
        lines.append(f"`{reach_path.name}` is not in `{out_dir}`: run `python -m conf_pipeline.label_reach "
                     f"--out-dir {out_dir}` first. No A5 reading is reported.")
    else:
        lines += [
            "Batch k-means and Ward with the true number of classes, and DBSCAN with its radius chosen by ARI, "
            "on raw and on window-standardised features, mean over the tail windows, best method per "
            "representation. These are label-informed references, not upper bounds: they see one window and "
            "(but for DBSCAN) as many clusters as classes, while a stream clusterer may keep more clusters and "
            "the history of earlier windows.",
            "",
            _table(reach),
        ]
    lines += [
        "",
        "## Usable streams",
        "",
        f"`labels_usable`: at least {min_eval_windows} evaluable tail windows and a best raw-feature A5 ARI "
        f"of at least {min_reach_ari}. Both thresholds are placeholders (`--min-eval-windows`, "
        "`--min-reach-ari`), not findings. `tau_ari_nmi` is A1's agreement under online tuning; "
        "`best_tail_ari` is the best tail ARI any (algo, arm) reached.",
        "",
        _table(usable),
        "",
    ]
    above = usable[pd.to_numeric(usable["best_tail_ari"], errors="coerce")
                   > pd.to_numeric(usable["reach_raw"], errors="coerce") + REACH_MARGIN]
    if len(above):
        lines += [f"On {len(above)} folds a stream clusterer reads more than {REACH_MARGIN} ARI above the "
                  f"raw-feature batch reference ({', '.join(above['fold'])}); there the flag judges the labels "
                  "more harshly than the algorithms do.", ""]
    return "\n".join(lines)


def _ci(v, lo, hi, status: str = "ok", digits: int = 3) -> str:
    """``value [low, high]``; "undefined" for no value; a trailing ``*`` marks an unstable ratio."""
    if v is None or pd.isna(v):
        return "undefined"
    out = f"{v:.{digits}f}"
    if lo is not None and not pd.isna(lo):
        out += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    return out + (" *" if status == "unstable" else "")


def _conclusion_pivot(conc: pd.DataFrame, group: str, subset: str = PRIMARY) -> pd.DataFrame:
    sub = conc[(conc["group"] == group) & (conc["subset"] == subset)].copy()
    if sub.empty:
        return sub
    sub["cell"] = [_ci(v, lo, hi, st) for v, lo, hi, st in zip(sub["value"], sub["ci_low"], sub["ci_high"], sub["status"])]
    sub["column"] = sub["metric"].str.upper() + ", " + sub["unit"]
    order = list(dict.fromkeys(sub["conclusion"]))
    wide = sub.pivot_table(index=["conclusion", "reference"], columns="column", values="cell", aggfunc="first")
    wide = wide.reindex(columns=[c for c in ("ARI, fold", "ARI, block", "NMI, fold", "NMI, block") if c in wide.columns])
    wide = wide.reset_index()
    wide["_o"] = [order.index(c) for c in wide["conclusion"]]
    wide["_r"] = [list(REFERENCE_CONSTRUCTIONS).index(r) if r in REFERENCE_CONSTRUCTIONS else -1 for r in wide["reference"]]
    wide = wide.sort_values(["_o", "_r"]).drop(columns=["_o", "_r"]).fillna("")
    wide.columns.name = None
    return wide


def _p12_markdown(p12: dict, n_boot: int, boot_seed: int, min_denominator: float) -> str:
    t = p12["tables"]
    conc = t["conclusions.csv"]
    n_undef = int((conc["status"] == "undefined").sum())
    n_unst = int((conc["status"] == "unstable").sum())
    lines = [
        "## P1.2: statistical unit and metric sensitivity",
        "",
        "Every conclusion is crossed with the measure (ARI, NMI), the statistical unit and the construction of "
        "the reference. **Unit.** `fold`: the streams are the units, drawn within their stream group. "
        "`block`: the source blocks are (`_common.source_block`: the seven INSECTS variants are one recording "
        "campaign), each drawn block bringing all its folds; the paired effects are averaged within a block "
        "first, so INSECTS weighs one source. The capture ratios, the losses and the share keep their "
        "published definitions (sums or means over the pairs) under both units: only their intervals move. "
        "Model seeds never add units: within each drawn pair the seeds are drawn again, the protocol arms "
        "(seeds 1-3) together by seed number, the reference's reruns (seeds 2-4) apart; a reference is never "
        "paired with a protocol seed because it has the same index, and an online - reference difference is "
        f"one of (algo, fold) seed means. {n_boot} replicates, seed {boot_seed}, 2.5-97.5 percentiles; tau "
        "intervals draw the units only (rankings are of seed means).",
        "",
        f"**Ratios.** Each comes with its numerator, denominator and n (`conclusions.csv`). A ratio whose "
        f"denominator per pair is at most {min_denominator} (a placeholder, `--min-denominator`) reads "
        "undefined; one that some replicates leave undefined is marked `*` (unstable, `n_used` < "
        "`n_boot`). Nothing is clipped: a capture above 1 or a share outside [0, 1] is reported as it reads. "
        f"Of {len(conc)} rows, {n_undef} are undefined and {n_unst} unstable.",
        "",
        "**References.** (a) `ARI-selected reference, NMI-evaluated` keeps every selection as ARI made it (the "
        "reference configuration and its reruns, the online runs, the labels' pick within the ensemble) and "
        "reads it in NMI. (b) `NMI-reselected reference` picks the reference again by tail NMI over the same "
        "sweep candidates, and the labels' pick within the ensemble by member NMI. (b) is a maximum over a "
        "finite candidate set read in-sample on the sweep's seed, not an NMI optimum; `ARI-selected "
        "reference, NMI-evaluated, sweep seed` reads (a) on that same basis, so that (b) minus it is the "
        "reselection alone. The tunable set is the published ARI gate except under `NMI gate`.",
        "",
        _table(t["constructions.csv"]),
        "",
    ]
    lines += [f"- {n}" for n in p12["notes"]] + ([""] if p12["notes"] else [])
    for group in ("real", ALL):
        lines += [f"### Conclusions, {'real streams' if group == 'real' else 'all streams'}", "",
                  _table(_conclusion_pivot(conc, group)), ""]
    ratios = conc[(conc["subset"] == PRIMARY) & conc["denominator"].notna()
                  & conc["group"].isin(["real", ALL])]
    if len(ratios):
        lines += ["Ratios with their parts (fold unit; the block unit has the same parts):", "",
                  _table(ratios[ratios["unit"] == "fold"][["conclusion", "group", "metric", "reference", "value",
                                                           "numerator", "denominator", "n_pairs", "status"]]), ""]
    sens = conc[(conc["subset"] != PRIMARY) & (conc["group"] == ALL)]
    if len(sens):
        sens = sens.assign(cell=[_ci(v, lo, hi) for v, lo, hi in zip(sens["value"], sens["ci_low"], sens["ci_high"])],
                           column=sens["metric"].str.upper() + ", " + sens["unit"])
        piv = sens.pivot_table(index=["subset", "conclusion", "reference"], columns="column", values="cell",
                               aggfunc="first").reset_index().fillna("")
        piv.columns.name = None
        lines += ["### Sensitivity: tau(default, ·) on the declared weak-default filters, all streams", "",
                  "The filters are decided on the default's ARI and undefined share under both measures, so "
                  "they keep the same cells; they are sensitivity rows, never the primary table.", "",
                  _table(piv), ""]
    ref = t["nmi_reference.csv"]
    if len(ref) and ref["nmi_config_id"].notna().any():
        n = int(ref["nmi_config_id"].notna().sum())
        lines += ["### NMI: what the reselection changes", "",
                  f"Over {n} pairs, (b) picks the ARI-selected configuration on {int(ref['same_config'].sum())}; "
                  f"re-picking by ARI from the same sweep windows names the published configuration on "
                  f"{int(ref['ari_repick_matches'].sum())} of {int(ref['ari_repick_config_id'].notna().sum())}. "
                  f"The reselection raises the reference's in-sample NMI by {ref['reselection_gain'].mean():.4f} "
                  f"on average (max {ref['reselection_gain'].max():.4f}): a maximum over the same candidates "
                  "cannot read lower. Tunable pairs: "
                  f"{int(ref['tunable_ari'].sum())} under the ARI gate, {int(ref['tunable_nmi'].sum())} under the "
                  "NMI gate (`nmi_reference.csv`).", ""]
    eff = t["paired_effects.csv"]
    if len(eff):
        pooled = eff[(eff["algo"] == POOLED) & eff["group"].isin(["real", ALL])].copy()
        pooled["effect"] = [_ci(v, lo, hi) for v, lo, hi in zip(pooled["point"], pooled["low"], pooled["high"])]
        lines += ["### Paired effects by unit", "",
                  "Online minus another arm, pooled over the algorithms: per unit the mean over its cells, then "
                  "the mean over units with its interval, and Wilcoxon's test over the units with Holm over "
                  "the family of one (measure, basis, unit, group): every algorithm, the pooled row and every "
                  "comparison. Under `block` every source weighs one unit, so in the `all` rows the synthetic "
                  "streams (one block each) weigh more than under `fold` as well as INSECTS weighing less: the "
                  "two points differ by that reweighting, not by noise. Per-algorithm rows are in "
                  "`paired_effects.csv`.", "",
                  _table(pooled[["metric", "basis", "reference", "unit", "group", "comparison", "effect", "n_units",
                                 "n_pairs", "p", "p_holm"]], 4), ""]
    loo = t["loo.csv"]
    if len(loo):
        pub = loo[(loo["reference"] == REFERENCE_CONSTRUCTIONS[0]) & (loo["group"] == "real")]
        lines += ["### Leave one out, real streams, published reference", "",
                  "Each real stream (`fold`) or source block (`block`) left out in turn; every construction "
                  "and group is in `loo.csv`.", "",
                  _table(pub[["unit", "dropped", "n_dropped_folds", "n_tunable_pairs", "capture_confstream",
                              "capture_frozen_prefix", "selection_loss", "search_loss", "gap", "selection_share"]]), ""]
    share, seedless = p12["share"], p12["seedless"]
    lines += [
        "### Checks of make_robustness_extra.py",
        "",
        "The result script's readings, computed here; `python -m conf_pipeline.robustness_check` compares them "
        "with `result/otb_full/report/robustness_extra.txt`.",
        "",
        "Kendall's tau between arms (18 folds drawn within groups; real folds among themselves):",
        "",
        _table(t["tau_arms.csv"]),
        "",
        "Rankings on the source blocks (INSECTS as one block):",
        "",
        _table(t["block_rankings_tau.csv"]),
        "",
        _table(t["block_rankings_ranks.csv"], 2),
        "",
        "tau(default, ·) after the weak-default filters:",
        "",
        _table(t["tau_weak_default.csv"]),
        "",
    ]
    if p12["sobol"] is not None:
        desc = p12["sobol"]["describe"].reset_index().rename(columns={"index": "statistic"})
        lines += [f"Rankings by single Sobol points (tau of the sweep's point 0 against the default arm: "
                  f"{p12['sobol']['point_zero_tau']:.3f}):", "", _table(desc), "",
                  _table(t["sobol_aggregate.csv"]), ""]
    lines += [
        "Per real stream, its part of the attainable gain and of the selection loss:",
        "",
        _table(t["real_streams.csv"]),
        "",
        f"Selection share on the real streams: {_ci(share['share'], share['low'], share['high'])} over "
        f"{share['n_folds']} streams drawn with replacement (per stream the mean over algorithms, no seed "
        f"drawn); P(mean search loss <= 0) = {share['p_search_le_0']:.3f}; {share['n_undefined']} replicates "
        "without a share.",
        "",
        f"Tunable pairs: {seedless['n_tunable']}, of which {seedless['n_tunable_seedless']} belong to algorithms "
        f"that take no seed ({', '.join(seedless['seedless_algos'])}); the reference's reruns read the same on "
        f"every seed for all of them: {seedless['reference_sd_zero_on_seedless']}. On those pairs the reruns on "
        "seeds 2-4 repeat the selection seed, and the reference keeps the optimism of its maximum.",
        "",
        "Not ported (the script's member and drift checks): silhouette on single-class windows, the -1 "
        "sentinel labels, the tie-break replay and the common-threshold recovery after drift; their numbers "
        "stay those of `robustness_extra.txt`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(C.OUT))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument("--min-eval-windows", type=int, default=MIN_EVAL_WINDOWS)
    parser.add_argument("--min-reach-ari", type=float, default=MIN_REACH_ARI)
    parser.add_argument("--min-denominator", type=float, default=MIN_DENOMINATOR,
                        help="P1.2: a ratio whose denominator per pair is at most this reads undefined "
                             "(a placeholder, in units of the measure)")
    parser.add_argument("--reference-dir", action="append", default=[], metavar="NAME=DIR",
                        help="add an A2 construction: the reference of another out dir (e.g. a sweep on "
                             "seeds 1-3 rerun on seeds 4-6); repeatable")
    args = parser.parse_args()
    refs = {}
    for item in args.reference_dir:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--reference-dir wants NAME=DIR, got {item!r}")
        if name in CONSTRUCTIONS or name in refs:
            raise SystemExit(f"construction name {name!r} is taken")
        refs[name] = Path(path)
    for p in write_robustness(Path(args.out_dir), n_boot=args.n_boot, boot_seed=args.boot_seed,
                              min_eval_windows=args.min_eval_windows, min_reach_ari=args.min_reach_ari,
                              reference_dirs=refs, min_denominator=args.min_denominator):
        print(p)


if __name__ == "__main__":
    main()
