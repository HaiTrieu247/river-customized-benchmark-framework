"""Tables and curves from the files the drivers write into ``out/``.

Inputs: ``runs.jsonl`` (one summary per job), ``windows.jsonl`` (one row per
evaluated window), ``members/*.jsonl`` (one file per job, written under
``--log-members``), and the sweep's ``sweep.jsonl`` + ``oracle_static.json``.

Everything reported about quality is the *lagged* reading: the configuration
that was incumbent after window ``t-1``, scored on window ``t``. The same-window
score of the incumbent is the maximum of ``e_size`` noisy values and is kept in
the rows only so the gap between the two can be seen.

Tables (each a function, each written as a CSV and a section of ``report.md``):

- ``summary_table``  mean/sd over seeds per (algo, fold, arm)
- ``gain_table``     every arm against ``default``, tail and whole stream
- ``oracle_table``   the static ceiling per (algo, fold): the oracle's rerun, else the sweep
- ``headroom_table`` share of the ceiling gain each arm realises, behind a noise gate
- ``regret_table``   ``A_static - a_lag`` split into lag, proxy and search regret, whole stream and tail
- ``drift_table``    drop and recovery around the drift points of the synthetic folds

Run from the repo root::

    python -m conf_pipeline.report [--out-dir conf_pipeline/out]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C

#: Per-run fields averaged over seeds, and shown in the summary table. The
#: ``lag_ari_A_*`` / ``lag_ssq_micro_mean`` / ``lag_n_micro_centers_mean`` fields
#: (Table A, the common reclusterer) are optional: runs that predate them read None,
#: as do the cost ratios and the Table A error fields of older runs.
SUMMARY_FIELDS = (
    "lag_ari_win_mean", "lag_score_mean", "lag_ari_cumulative",
    "incumbent_throughput_sps", "ensemble_runtime_s", "wall_s",
    "lag_memory_mb_peak", "lag_n_pred_labels_mean", "lag_n_macro_clusters_mean",
    "n_switches", "n_accepted_total", "n_reset_total",
    "lag_undefined_share", "lag_infeasible_share", "single_class_window_share",
    "lag_n_sentinel_total", "s_online", "lam",
    "lag_ari_win_tail_mean", "lag_score_tail_mean", "lag_nmi_win_mean", "lag_purity_win_mean",
    "lag_ari_A_kmeans_mean", "lag_ari_A_dbscan_mean",
    "lag_ari_A_kmeans_tail_mean", "lag_ari_A_dbscan_tail_mean",
    "lag_ssq_micro_mean", "lag_n_micro_centers_mean", "frozen_at_window",
    "tuning_overhead", "switch_cost", "lag_table_a_error_share", "lag_table_a_first_error",
)
#: Fields of ``SUMMARY_FIELDS`` that are text: the lowest seed's value, no mean and no sd.
SUMMARY_TEXT_FIELDS = ("lag_table_a_first_error",)
GROUP = ["algo", "fold", "arm"]

#: The evaluation region every arm shares (windows >= ``tail_from``) and the whole stream.
TAIL = "lag_ari_win_tail_mean"
WHOLE = "lag_ari_win_mean"
#: Arms that tune online: the regret decomposition and ``evolution_value`` are theirs.
ONLINE_ARMS: tuple[str, ...] = tuple(a for a in C.ARMS if a != "default")
#: Arms given a ``headroom_<arm>`` column even when absent from the runs.
HEADROOM_ARMS: tuple[str, ...] = ONLINE_ARMS + ("oracle_static",)
#: One-sided 95% z: the ceiling gain must clear this many standard errors of the default.
Z_GATE = 1.645
#: Windows in a row the lagged reading must stay within ``delta`` of its reference.
RECOVERY_RUN = 3
#: Folds whose drift multiplies the data's scale: a distance parameter should follow it.
SCALE_DRIFT_FOLDS = ("scale_drift", "mixed_drift")
#: Float slack on the recovery comparison (0.43 - 0.05 is 0.37999999999999995).
_EPS = 1e-12

NO_MEMBERS_NOTE = "members.jsonl not found: R_proxy/R_search unavailable"
NO_ORACLE_NOTE = "no static oracle"

GAIN_COLS = [
    "algo", "fold", "arm", "n_seeds", "arm_tail", "default_tail", "gain_tail",
    "arm_all", "default_all", "gain_all", "arm_sps", "default_sps", "throughput_ratio",
]
ORACLE_COLS = [
    "algo", "fold", "ceiling", "ceiling_sd", "ceiling_metric", "ceiling_all", "ceiling_source",
    "n_seeds", "config_id",
]
REGRET_COLS = [
    "algo", "fold", "arm", "region", "n_seeds", "n_windows", "a_lag", "a_inc", "a_star", "A_static",
    "ceiling_source", "R_lag", "R_proxy", "R_search", "note",
]
#: The regret decomposition's two regions: the whole stream, against ``ceiling_all``, and the
#: tail (windows >= the run's ``tail_from``), against the tail ``ceiling`` -- the primary one.
REGRET_REGIONS: tuple[str, ...] = ("all", "tail")
DRIFT_COLS = [
    "algo", "fold", "arm", "drift_point", "window", "n_seeds", "pre", "drop_depth",
    "recovery_windows", "n_recovered", "ref", "param", "param_ratio",
]
MEMBER_COLS = ["algo", "fold", "seed", "arm", "config_id", "member_id", "window", "is_test", "ari_win"]
SWEEP_COLS = ["algo", "fold", "seed", "arm", "config_id", "unit_override", "status", WHOLE]


def _headroom_cols(arms=HEADROOM_ARMS) -> list[str]:
    return (
        ["algo", "fold", "metric", "default", "default_se", "n_seeds_default", "ceiling",
         "ceiling_source", "gain", "tunable"]
        + [f"headroom_{a}" for a in arms] + ["evolution_value", "note"]
    )


# --- loading -------------------------------------------------------------------------------

def current_runs(out_dir: Path) -> pd.DataFrame:
    """``runs.jsonl`` without the rows a later result made obsolete.

    - ``oracle_static`` rows of a configuration that is no longer the one
      ``oracle_static.json`` names for their (algo, fold). Without that file
      there is nothing to judge staleness by and every row is kept; with it,
      an (algo, fold) it has no entry for keeps only its non-``ok`` rows (the
      skips that record the missing oracle).
    - ``skipped`` rows of a job that also has an ``ok`` row.
    """
    rows = list(C.read_jsonl(Path(out_dir) / "runs.jsonl"))
    oracle = _read_oracle(out_dir)
    rows = [r for r in rows if _is_current(r, oracle)]
    ok = {_job(r) for r in rows if r.get("status") == "ok"}
    rows = [r for r in rows if not (r.get("status") == "skipped" and _job(r) in ok)]
    if not rows:
        return pd.DataFrame(columns=["algo", "fold", "seed", "arm", "status"])
    return pd.DataFrame(rows)


def load_windows(out_dir: Path) -> pd.DataFrame:
    rows = list(C.read_jsonl(Path(out_dir) / "windows.jsonl"))
    if not rows:
        return pd.DataFrame(columns=["algo", "fold", "seed", "arm", "window"])
    return pd.DataFrame(rows)


def load_members(out_dir: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Every member row of ``<out_dir>/members/*.jsonl`` (one file per job).

    ``columns`` keeps only those keys of each row (member logs are large).
    An empty frame when the directory is missing or empty.
    """
    rows = []
    for path in _member_files(out_dir):
        for row in C.read_jsonl(path):
            rows.append({k: row.get(k) for k in columns} if columns else row)
    if not rows:
        return pd.DataFrame(columns=list(columns or MEMBER_COLS))
    return pd.DataFrame(rows)


def load_sweep(out_dir: Path) -> pd.DataFrame:
    rows = list(C.read_jsonl(Path(out_dir) / "sweep.jsonl"))
    if not rows:
        return pd.DataFrame(columns=SWEEP_COLS)
    return pd.DataFrame(rows)


def best_active_members(
    out_dir: Path, folds=None, arms=None,
) -> tuple[dict[tuple, dict[int, float]], set[tuple]]:
    """``a_star`` per job and window, streamed from the member logs.

    Returns ``({job: {window: max ari_win over is_test == False}}, jobs_seen)``
    with ``job = (algo, fold, seed, arm, config_id)``. Streaming, because a full
    benchmark logs millions of member rows and only this maximum is read;
    ``folds`` / ``arms`` skip the other jobs' files by name, unread.
    """
    best: dict[tuple, dict[int, float]] = {}
    seen: set[tuple] = set()
    for path in _member_files(out_dir, folds, arms):
        for row in C.read_jsonl(path):
            key = (row["algo"], row["fold"], row["seed"], row["arm"], _cid(row.get("config_id")))
            seen.add(key)
            ari = _num(row.get("ari_win"))
            if row.get("is_test") or ari is None:
                continue
            per = best.setdefault(key, {})
            w = int(row["window"])
            if w not in per or ari > per[w]:
                per[w] = ari
    return best, seen


def _member_files(out_dir: Path, folds=None, arms=None) -> list[Path]:
    """``members/*.jsonl``; with ``folds``/``arms``, only the jobs named
    ``<algo>__<fold>__seed<seed>__<arm>[__cfg<id>]`` that match (a file whose
    name does not parse is always read)."""
    files = sorted((Path(out_dir) / C.MEMBERS_DIR.name).glob("*.jsonl"))
    keep = []
    for path in files:
        parts = path.stem.split("__")
        if len(parts) >= 4 and (
            (folds is not None and parts[1] not in folds) or (arms is not None and parts[3] not in arms)
        ):
            continue
        keep.append(path)
    return keep


def _read_oracle(out_dir: Path) -> dict | None:
    path = Path(out_dir) / C.ORACLE_JSON.name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _oracle_entry(oracle: dict | None, algo: str, fold: str) -> dict | None:
    return ((oracle or {}).get(algo) or {}).get(fold)


def _is_current(row: dict, oracle: dict | None) -> bool:
    if row.get("arm") != "oracle_static" or oracle is None:
        return True
    entry = _oracle_entry(oracle, row.get("algo"), row.get("fold"))
    if entry is None:
        return row.get("status") != "ok"
    return _cid(row.get("config_id")) == _cid(entry.get("config_id"))


def _job(row: dict) -> tuple:
    return (row.get("algo"), row.get("fold"), row.get("seed"), row.get("arm"))


def _none(v):
    """NaN (a key a row did not have) -> None."""
    return None if isinstance(v, float) and math.isnan(v) else v


def _num(v) -> float | None:
    v = _none(v)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _cid(v) -> int | None:
    f = _num(v)
    return None if f is None else int(f)


def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Rows as an object frame, so a missing reading stays ``None`` (not NaN)."""
    return pd.DataFrame(rows, columns=columns, dtype=object)


def _seed_stats(runs: pd.DataFrame, metric: str) -> dict[tuple, tuple[float, float, int]]:
    """(algo, fold, arm) -> (mean, sd, n) of ``metric`` over the ``ok`` seeds that have it.

    ``sd`` is 0.0 for a single seed.
    """
    out: dict[tuple, tuple[float, float, int]] = {}
    if runs.empty or metric not in runs:
        return out
    ok = runs[runs["status"] == "ok"]
    for key, grp in ok.groupby(GROUP, sort=True):
        vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
        if len(vals):
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            out[key] = (float(vals.mean()), sd, int(len(vals)))
    return out


# --- summary and gain ------------------------------------------------------------------------

def summary_table(out_dir: Path) -> pd.DataFrame:
    """One row per (algo, fold, arm): mean and sd over the seeds that finished.

    ``n_expected`` is the number of distinct seeds the arm was run with over
    all its cells (every row on file counts, failed and skipped ones too) and
    ``all_seeds_ok`` says whether this cell finished every one of them: spec
    5.1 holds a cell valid only when every seed is ``ok``, and ``analyze``
    reads an invalid cell as no reading. Counted per arm, because
    ``oracle_static`` reruns on seeds of its own (2-6 in the full preset,
    against 1-5 for the protocol arms). Text fields (``SUMMARY_TEXT_FIELDS``)
    carry the lowest seed's value instead of a mean.
    """
    runs = current_runs(out_dir)
    if runs.empty:
        return pd.DataFrame(columns=GROUP)
    expected = runs[runs["arm"] != "sweep"].groupby("arm")["seed"].nunique().to_dict()
    out = []
    for key, grp in runs.groupby(GROUP, sort=True):
        good = grp[grp["status"] == "ok"]
        bad = grp[grp["status"] != "ok"]
        row = dict(zip(GROUP, key))
        row["n_seeds"] = int(len(good))
        row["n_expected"] = int(expected.get(key[2], 0))
        row["all_seeds_ok"] = row["n_seeds"] == row["n_expected"]
        row["n_failed"] = int(len(bad))
        row["failed_status"] = ",".join(sorted(set(bad["status"]))) if len(bad) else None
        for f in SUMMARY_FIELDS:
            if f in SUMMARY_TEXT_FIELDS:
                texts = [v for v in good.sort_values("seed")[f] if isinstance(v, str)] if f in good else []
                row[f] = texts[0] if texts else None
                continue
            vals = pd.to_numeric(good[f], errors="coerce").dropna() if f in good else pd.Series(dtype=float)
            row[f] = float(vals.mean()) if len(vals) else None
            # ``lag_ari_win_mean`` -> sd column ``lag_ari_win_sd``, for readability
            row[f"{f.removesuffix('_mean')}_sd" if f.endswith("_mean") else f"{f}_sd"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else None
            )
        out.append(row)
    return pd.DataFrame(out)


def gain_table(out_dir: Path) -> pd.DataFrame:
    """Every non-default arm against ``default``, per (algo, fold, arm).

    ``gain_tail`` compares ``lag_ari_win_tail_mean`` (the region every arm
    shares), ``gain_all`` compares ``lag_ari_win_mean``; ``throughput_ratio`` is
    the arm's incumbent throughput over the default's. NaN without a default.
    """
    runs = current_runs(out_dir)
    if runs.empty:
        return pd.DataFrame(columns=GAIN_COLS)
    tail, whole = _seed_stats(runs, TAIL), _seed_stats(runs, WHOLE)
    sps = _seed_stats(runs, "incumbent_throughput_sps")
    ok = runs[runs["status"] == "ok"]
    n_ok = ok.groupby(GROUP).size().to_dict() if not ok.empty else {}
    keys = sorted({tuple(k) for k in runs[GROUP].itertuples(index=False)})
    out = []
    for algo, fold, arm in keys:
        if arm == "default":
            continue
        key, base = (algo, fold, arm), (algo, fold, "default")
        row = {"algo": algo, "fold": fold, "arm": arm, "n_seeds": int(n_ok.get(key, 0))}
        for name, stats in (("tail", tail), ("all", whole), ("sps", sps)):
            row[f"arm_{name}"] = stats[key][0] if key in stats else np.nan
            row[f"default_{name}"] = stats[base][0] if base in stats else np.nan
        row["gain_tail"] = row["arm_tail"] - row["default_tail"]
        row["gain_all"] = row["arm_all"] - row["default_all"]
        d = row["default_sps"]
        row["throughput_ratio"] = row["arm_sps"] / d if d == d and d != 0 else np.nan
        out.append(row)
    table = pd.DataFrame(out, columns=GAIN_COLS)
    for c in GAIN_COLS[3:]:
        table[c] = pd.to_numeric(table[c], errors="coerce")
    return table


# --- ceiling and headroom --------------------------------------------------------------------

def oracle_table(out_dir: Path) -> pd.DataFrame:
    """The static ceiling per (algo, fold).

    Preferred: the ``oracle_static`` reruns of the configuration
    ``oracle_static.json`` currently names (stale reruns are ignored), averaged
    over seeds, on ``lag_ari_win_tail_mean`` when those rows carry it and
    ``lag_ari_win_mean`` otherwise (``ceiling_metric`` says which). Fallback:
    the sweep's best ``lag_ari_win_mean`` (one seed, whole stream),
    ``ceiling_source = "sweep"``. ``ceiling_all`` is always the whole-stream
    reading, the one the regret decomposition compares against.
    """
    runs = current_runs(out_dir)
    sweep = load_sweep(out_dir)
    oracle = _read_oracle(out_dir)
    keys = set()
    if not runs.empty:
        keys |= {(a, f) for a, f in runs[["algo", "fold"]].itertuples(index=False)}
    if not sweep.empty:
        keys |= {(a, f) for a, f in sweep[["algo", "fold"]].itertuples(index=False)}
    keys |= {(a, f) for a, folds in (oracle or {}).items() for f in folds}
    if not keys:
        return _frame([], ORACLE_COLS)
    rerun = (
        runs[(runs["arm"] == "oracle_static") & (runs["status"] == "ok")]
        if not runs.empty else runs
    )
    out = []
    for algo, fold in sorted(keys):
        row = dict.fromkeys(ORACLE_COLS)
        row.update(algo=algo, fold=fold, n_seeds=0)
        grp = rerun[(rerun["algo"] == algo) & (rerun["fold"] == fold)] if not rerun.empty else rerun
        tail = pd.to_numeric(grp[TAIL], errors="coerce").dropna() if TAIL in grp else pd.Series(dtype=float)
        whole = pd.to_numeric(grp[WHOLE], errors="coerce").dropna() if WHOLE in grp else pd.Series(dtype=float)
        entry = _oracle_entry(oracle, algo, fold)
        if len(tail) or len(whole):
            vals, metric = (tail, TAIL) if len(tail) else (whole, WHOLE)
            if entry is not None:
                config_id = _cid(entry.get("config_id"))
            else:
                config_id = _cid(grp["config_id"].iloc[0]) if "config_id" in grp else None
            row.update(
                ceiling=float(vals.mean()), ceiling_metric=metric,
                ceiling_sd=float(vals.std(ddof=1)) if len(vals) > 1 else None,
                ceiling_all=float(whole.mean()) if len(whole) else None,
                ceiling_source="oracle_static", n_seeds=int(len(vals)), config_id=config_id,
            )
        else:
            best = _sweep_best(sweep, algo, fold)
            if best is None and entry is not None and _num(entry.get(WHOLE)) is not None:
                best = (_num(entry[WHOLE]), _cid(entry.get("config_id")))
            if best is not None:
                row.update(ceiling=best[0], ceiling_all=best[0], ceiling_metric=WHOLE,
                           ceiling_source="sweep", n_seeds=1, config_id=best[1])
        out.append(row)
    return _frame(out, ORACLE_COLS)


def _sweep_best(sweep: pd.DataFrame, algo: str, fold: str) -> tuple[float, int | None] | None:
    """The sweep's best ``lag_ari_win_mean`` for (algo, fold); ties to the smaller config_id."""
    if sweep.empty or WHOLE not in sweep:
        return None
    grp = sweep[(sweep["algo"] == algo) & (sweep["fold"] == fold) & (sweep["status"] == "ok")]
    cands = []
    for row in grp.to_dict("records"):
        value = _num(row.get(WHOLE))
        if value is not None:
            cands.append((value, _cid(row.get("config_id"))))
    if not cands:
        return None
    return max(cands, key=lambda vc: (vc[0], -(vc[1] or 0)))


def headroom_table(out_dir: Path, min_gain: float = C.MIN_CEILING_GAIN) -> pd.DataFrame:
    """How much of the ceiling's gain over ``default`` each arm realises.

    Per (algo, fold), on the ceiling's own metric: ``gain = ceiling - default``,
    ``default_se = sd(default over seeds) / sqrt(n)`` (0.0 for one seed), and
    ``tunable = gain >= min_gain and gain > 1.645 * default_se``. Only a tunable
    fold gets ``headroom_<arm> = (arm - default) / gain``; the others read None
    and say why in ``note`` ("not tunable by ARI"), rather than a ratio of noise.
    ``evolution_value = headroom_confstream - headroom_random_online``.
    """
    runs = current_runs(out_dir)
    oracle = oracle_table(out_dir)
    no_ok_point = _swept_without_an_ok_point(out_dir)
    arms = list(HEADROOM_ARMS)
    if not runs.empty:
        arms += sorted(set(runs["arm"]) - set(arms) - {"default"})
    cols = _headroom_cols(arms)
    keys = set()
    if not runs.empty:
        keys |= {(a, f) for a, f in runs[["algo", "fold"]].itertuples(index=False)}
    ceilings = {(r["algo"], r["fold"]): r for r in oracle.to_dict("records")}
    keys |= set(ceilings)
    if not keys:
        table = _frame([], cols)
        table["tunable"] = table["tunable"].astype(bool)
        return table
    stats_by_metric: dict[str, dict] = {}
    out = []
    for algo, fold in sorted(keys):
        o = ceilings.get((algo, fold)) or {}
        metric = o.get("ceiling_metric") or TAIL
        if metric not in stats_by_metric:
            stats_by_metric[metric] = _seed_stats(runs, metric)
        stats = stats_by_metric[metric]
        ceiling = _num(o.get("ceiling"))
        d = stats.get((algo, fold, "default"))
        default = d[0] if d else None
        default_se = (d[1] / math.sqrt(d[2]) if d[2] > 1 else 0.0) if d else None
        gain = ceiling - default if ceiling is not None and default is not None else None
        tunable = bool(gain is not None and gain >= min_gain - _EPS and gain > Z_GATE * default_se)
        row = {
            "algo": algo, "fold": fold, "metric": metric, "default": default,
            "default_se": default_se, "n_seeds_default": d[2] if d else 0, "ceiling": ceiling,
            "ceiling_source": o.get("ceiling_source"), "gain": gain, "tunable": tunable,
        }
        for arm in arms:
            a = stats.get((algo, fold, arm))
            row[f"headroom_{arm}"] = (a[0] - default) / gain if tunable and a else None
        hc, hr = row.get("headroom_confstream"), row.get("headroom_random_online")
        row["evolution_value"] = hc - hr if hc is not None and hr is not None else None
        if ceiling is None:
            row["note"] = (
                f"no ceiling: sweep produced no ok point for {algo}/{fold}" if (algo, fold) in no_ok_point
                else "no ceiling: no oracle_static rerun and no sweep"
            )
        elif default is None:
            row["note"] = "no finished default run"
        elif gain < min_gain - _EPS:
            row["note"] = f"not tunable by ARI: ceiling gain {gain:.3f} < {min_gain}"
        elif not tunable:
            row["note"] = (f"not tunable by ARI: ceiling gain {gain:.3f} within default noise "
                           f"({Z_GATE}*se = {Z_GATE * default_se:.3f})")
        else:
            row["note"] = ""
        out.append(row)
    table = _frame(out, cols)
    table["tunable"] = table["tunable"].astype(bool)
    return table


def _swept_without_an_ok_point(out_dir: Path) -> set[tuple[str, str]]:
    """(algo, fold) pairs the sweep ran but produced no ``ok`` point with a reading for:
    in ``sweep.jsonl`` with no such row, or in ``oracle_static.json`` with ``config_id`` None."""
    out = {
        (algo, fold) for algo, folds in (_read_oracle(out_dir) or {}).items()
        for fold, entry in folds.items() if (entry or {}).get("config_id") is None
    }
    sweep = load_sweep(out_dir)
    if not sweep.empty:
        for (algo, fold), grp in sweep.groupby(["algo", "fold"], sort=False):
            if _sweep_best(sweep, algo, fold) is None:
                out.add((algo, fold))
    return out


# --- regret decomposition ----------------------------------------------------------------------

def regret_table(out_dir: Path) -> pd.DataFrame:
    """``A_static - a_lag = R_lag + R_proxy + R_search`` per (algo, fold, online arm, region).

    Two rows per (algo, fold, arm): ``region = "all"`` over every window, and
    ``region = "tail"`` over the windows from the run row's ``tail_from`` on
    (``FREEZE_AFTER_WINDOW`` when the row has none), the region every arm
    shares and the paper's primary one. Per job (algo, fold, seed, arm), over
    the region's windows where the lagged reading ``lag_ari_win`` and the
    incumbent's same-window ``ari_win`` both exist (and the member log's
    ``a_star`` too, when there is one):

    - ``a_lag``  mean ``lag_ari_win`` (the incumbent named after t-1, on window t)
    - ``a_inc``  mean ``ari_win`` of the incumbent named on window t itself
    - ``a_star`` mean over windows of the best active member's ``ari_win``
      (``is_test == False`` rows of ``members/*.jsonl``, plus the incumbent:
      a challenger accepted on window t is logged ``is_test=True`` there)

    then means over seeds. ``A_static`` is the oracle's reading of the same
    region: ``oracle_table().ceiling_all`` for ``"all"``, ``.ceiling`` for
    ``"tail"`` (``ceiling_source`` says where it comes from; the note says when
    the ceiling has no tail reading and the whole-stream one stands in), and

    ``R_lag = a_inc - a_lag``, ``R_proxy = a_star - a_inc``, ``R_search = A_static - a_star``

    so the three sum to ``A_static - a_lag`` by construction, row by row.
    Without a member log only ``R_lag`` is defined; without a static oracle
    ``R_search`` is None; a region with no such window (a tail the run never
    reached) reads ``n_windows`` 0 and every term None.
    """
    win = load_windows(out_dir)
    if win.empty or "lag_ari_win" not in win or "ari_win" not in win:
        return _frame([], REGRET_COLS)
    win = win[win["arm"].isin(ONLINE_ARMS)]
    if win.empty:
        return _frame([], REGRET_COLS)
    best, seen = best_active_members(out_dir, arms=set(ONLINE_ARMS))
    ceilings = {(r["algo"], r["fold"]): r for r in oracle_table(out_dir).to_dict("records")}
    tail_from = _tail_from_by_job(current_runs(out_dir))
    out = []
    for (algo, fold, arm), grp in win.groupby(GROUP, sort=True):
        seeds = [(seed, g, (algo, fold, seed, arm, None)) for seed, g in grp.groupby("seed", sort=True)]
        logged = sum(key in seen for _, _, key in seeds)
        ceiling = ceilings.get((algo, fold)) or {}
        for region in REGRET_REGIONS:
            jobs = []
            for seed, g, key in seeds:
                if region == "tail":
                    start = tail_from.get((algo, fold, seed, arm), C.FREEZE_AFTER_WINDOW)
                    g = g[pd.to_numeric(g["window"], errors="coerce") >= start]
                job = _regret_job(g, best.get(key))
                if job is not None:
                    jobs.append(job)
            out.append(_regret_row(algo, fold, arm, region, jobs, ceiling, logged))
    return _frame(out, REGRET_COLS)


def _tail_from_by_job(runs: pd.DataFrame) -> dict[tuple, int]:
    """(algo, fold, seed, arm) -> ``tail_from`` of the ``ok`` run rows that carry one."""
    if runs.empty or "tail_from" not in runs:
        return {}
    out = {}
    for row in runs[runs["status"] == "ok"].to_dict("records"):
        start = _num(row.get("tail_from"))
        if start is not None:
            out[(row["algo"], row["fold"], row["seed"], row["arm"])] = int(start)
    return out


def _regret_row(algo: str, fold: str, arm: str, region: str, jobs: list[dict], ceiling: dict,
                logged: int) -> dict:
    A = _num(ceiling.get("ceiling_all" if region == "all" else "ceiling"))
    row = dict.fromkeys(REGRET_COLS)
    row.update(algo=algo, fold=fold, arm=arm, region=region, n_seeds=len(jobs), n_windows=0,
               A_static=A, ceiling_source=_none(ceiling.get("ceiling_source")))
    if not jobs:
        where = " at or after tail_from" if region == "tail" else ""
        row["note"] = f"no window with both lag_ari_win and ari_win{where}"
        return row
    notes = []
    with_star = [j for j in jobs if j["a_star"] is not None]
    use = with_star or jobs
    a_lag = float(np.mean([j["a_lag"] for j in use]))
    a_inc = float(np.mean([j["a_inc"] for j in use]))
    a_star = float(np.mean([j["a_star"] for j in with_star])) if with_star else None
    row.update(
        n_seeds=len(use), n_windows=int(sum(j["n"] for j in use)),
        a_lag=a_lag, a_inc=a_inc, a_star=a_star, R_lag=a_inc - a_lag,
        R_proxy=a_star - a_inc if a_star is not None else None,
        R_search=A - a_star if A is not None and a_star is not None else None,
    )
    if not with_star:
        notes.append(NO_MEMBERS_NOTE if not logged else "member log has no active-member ari_win")
    elif len(with_star) < len(jobs):
        notes.append(f"member log for {len(with_star)} of {len(jobs)} seeds: every term over those")
    if A is None:
        notes.append(NO_ORACLE_NOTE)
    elif region == "tail" and ceiling.get("ceiling_metric") != TAIL:
        notes.append("A_static is the whole-stream reading: the ceiling has no tail reading")
    row["note"] = "; ".join(notes)
    return row


def _a_star(g: pd.DataFrame, star: dict[int, float] | None) -> pd.Series | None:
    """Per window row of one job: the best active member's ``ari_win``.

    The member log's ``is_test == False`` rows, plus the incumbent itself: a
    challenger accepted on window t is logged with ``is_test=True`` there, yet
    it is active when the incumbent is named and may be the one named. Without
    it ``a_star < a_inc`` on such windows and ``R_proxy`` turns negative.
    NaN on windows the member log does not cover.
    """
    if not star:
        return None
    s = pd.to_numeric(g["window"].map(star), errors="coerce")
    if "ari_win" in g:
        inc = pd.to_numeric(g["ari_win"], errors="coerce")
        s = s.where(s.isna(), np.fmax(s, inc))
    return s


def _regret_job(g: pd.DataFrame, star: dict[int, float] | None) -> dict | None:
    g = g.drop_duplicates("window", keep="last")
    lag = pd.to_numeric(g["lag_ari_win"], errors="coerce")
    inc = pd.to_numeric(g["ari_win"], errors="coerce")
    mask = lag.notna() & inc.notna()
    s = _a_star(g, star)
    if s is not None:
        if (mask & s.notna()).any():
            mask = mask & s.notna()
        else:
            s = None
    if not mask.any():
        return None
    return {
        "n": int(mask.sum()), "a_lag": float(lag[mask].mean()), "a_inc": float(inc[mask].mean()),
        "a_star": float(s[mask].mean()) if s is not None else None,
    }


# --- drift adaptation --------------------------------------------------------------------------

def drift_table(out_dir: Path, delta: float = 0.05, horizon: int = 10) -> pd.DataFrame:
    """Drop and recovery of the lagged ARI around each drift point, per (algo, fold, arm).

    Drift points: the run row's ``drift_points`` (sample indices), else the
    fold's own at the cap the run used (``_common.drift_points``); the drift
    window is ``drift_point // h``. Per seed:

    - ``pre = lag(window - 1)``; ``drop_depth = pre - min lag over [window, window + horizon)``
    - ``recovery_windows``: the first ``t >= window`` from which
      ``lag(t) >= ref(t) - delta`` holds for 3 windows in a row, minus
      ``window``; ``ref`` is the best active member (``a_star``, incumbent
      included) when the job has a member log and more than one active member
      at the drift window, else ``pre`` (with one active member ``a_star`` *is*
      the lagged incumbent and recovery would read 0). The run must finish
      before the next drift window; None if it never happens.
    - ``param_ratio`` (``scale_drift``/``mixed_drift`` only): the first ``scaled``
      parameter of the incumbent at ``window + recovery_windows`` over its value
      at ``window - 1``; a tuner that tracks the scale reads the scale factor.

    Over seeds: means of ``pre`` and ``drop_depth``; ``recovery_windows`` is the
    median seed's (a seed that never recovers counts as never; rounded up;
    None when the median seed never recovered), ``n_recovered`` counts the
    seeds that did; ``param_ratio`` is the median over seeds that have one.
    """
    runs = current_runs(out_dir)
    if runs.empty:
        return _frame([], DRIFT_COLS)
    ok = runs[runs["status"] == "ok"]
    jobs = []
    for row in ok.to_dict("records"):
        points = _drift_points_of(row)
        if points:
            jobs.append((row, points))
    if not jobs:
        return _frame([], DRIFT_COLS)
    win = load_windows(out_dir)
    if win.empty or "lag_ari_win" not in win:
        return _frame([], DRIFT_COLS)
    groups = dict(tuple(win.groupby(["algo", "fold", "seed", "arm"], sort=False)))
    best, _ = best_active_members(out_dir, folds={row["fold"] for row, _ in jobs})
    per: dict[tuple, list[dict]] = {}
    for row, points in jobs:
        algo, fold, seed, arm = row["algo"], row["fold"], row["seed"], row["arm"]
        g = groups.get((algo, fold, seed, arm))
        if g is None:
            continue
        cid = _cid(row.get("config_id"))
        if "config_id" in g:
            g = g[[_cid(v) == cid for v in g["config_id"]]]
        g = g.drop_duplicates("window", keep="last")
        lag = {int(w): _num(v) for w, v in zip(g["window"], g["lag_ari_win"])}
        params = {int(w): p for w, p in zip(g["window"], g["params"])} if "params" in g else {}
        star = best.get((algo, fold, seed, arm, cid))
        s = _a_star(g, star)
        star = {int(w): v for w, v in zip(g["window"], s) if v == v} if s is not None else None
        n_active = (
            {int(w): _num(v) for w, v in zip(g["window"], g["n_active"])} if "n_active" in g else {}
        )
        h = int(_num(row.get("h")) or C.H)
        windows = [p // h for p in points]
        for i, (point, tw) in enumerate(zip(points, windows)):
            nxt = windows[i + 1] if i + 1 < len(windows) else None
            # one active member (default, oracle_static, a frozen ensemble): its best
            # member *is* the lagged incumbent, so recovery is read against ``pre``
            single = (n_active.get(tw) or 0) == 1
            res = _drift_one(lag, params, None if single else star, algo, fold, tw, nxt, delta, horizon)
            per.setdefault((algo, fold, arm, point, tw), []).append(res)
    out = []
    for (algo, fold, arm, point, tw), seeds in sorted(per.items(), key=lambda kv: kv[0]):
        rec = [s["recovery"] for s in seeds]
        med = float(np.median([r if r is not None else math.inf for r in rec]))
        pres = [s["pre"] for s in seeds if s["pre"] is not None]
        drops = [s["drop"] for s in seeds if s["drop"] is not None]
        ratios = [s["ratio"] for s in seeds if s["ratio"] is not None]
        out.append({
            "algo": algo, "fold": fold, "arm": arm, "drift_point": int(point), "window": int(tw),
            "n_seeds": len(seeds),
            "pre": float(np.mean(pres)) if pres else None,
            "drop_depth": float(np.mean(drops)) if drops else None,
            "recovery_windows": None if math.isinf(med) else int(math.ceil(med)),
            "n_recovered": sum(r is not None for r in rec),
            "ref": ",".join(sorted({s["ref"] for s in seeds})),
            "param": next((s["param"] for s in seeds if s["param"]), None),
            "param_ratio": float(np.median(ratios)) if ratios else None,
        })
    return _frame(out, DRIFT_COLS)


def _drift_points_of(row: dict) -> tuple[int, ...]:
    pts = row.get("drift_points")
    if isinstance(pts, (list, tuple)):
        return tuple(int(p) for p in pts)
    cap = _num(row.get("n_samples_cap"))
    try:
        return tuple(int(p) for p in C.drift_points(row["fold"], int(cap) if cap else None))
    except (KeyError, ValueError):
        return ()


def _drift_one(lag: dict, params: dict, star: dict | None, algo: str, fold: str,
               tw: int, nxt: int | None, delta: float, horizon: int) -> dict:
    pre = lag.get(tw - 1)
    span = [lag[t] for t in range(tw, tw + horizon) if lag.get(t) is not None]
    drop = pre - min(span) if pre is not None and span else None

    def ref(t):
        return star.get(t) if star else pre

    def ok(t):
        v, r = lag.get(t), ref(t)
        return v is not None and r is not None and (nxt is None or t < nxt) and v >= r - delta - _EPS

    recovery = None
    last = max(lag) if lag else tw - 1
    for t in range(tw, last - RECOVERY_RUN + 2):
        if all(ok(t + k) for k in range(RECOVERY_RUN)):
            recovery = t - tw
            break
    name = _scaled_param(algo) if fold in SCALE_DRIFT_FOLDS else None
    ratio = None
    if name and recovery is not None:
        after = _num((params.get(tw + recovery) or {}).get(name))
        before = _num((params.get(tw - 1) or {}).get(name))
        ratio = after / before if after is not None and before else None
    return {"pre": pre, "drop": drop, "recovery": recovery, "ratio": ratio, "param": name,
            "ref": "a_star" if star else "pre"}


def _scaled_param(algo: str) -> str | None:
    from .spaces import SPACES

    space = SPACES.get(algo)
    if space is None:
        return None
    return next((p.name for p in space.params if p.kind == "scaled"), None)


# --- curves and the markdown report ----------------------------------------------------------

def curves(out_dir: Path) -> dict[str, pd.DataFrame]:
    """Per fold: lagged ARI and silhouette vs window, averaged over seeds."""
    win = load_windows(out_dir)
    if win.empty:
        return {}
    cols = [c for c in ("lag_ari_win", "lag_score", "score", "lag_ari_cum", "lag_n_pred_labels",
                        "ensemble_runtime_s", "wall_s") if c in win]
    out = {}
    for fold, grp in win.groupby("fold"):
        agg = (
            grp.groupby(["algo", "arm", "window"], sort=True)[cols]
            .agg(["mean", "count"])
        )
        agg.columns = [f"{a}_{b}" if b != "mean" else a for a, b in agg.columns]
        out[fold] = agg.reset_index()
    return out


def _md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n" for r in df.to_dict("records"))
    return head + body


TABLE_A_FIELDS = (
    "lag_ari_A_kmeans_tail_mean", "lag_ari_A_dbscan_tail_mean", "lag_ari_A_kmeans_mean",
    "lag_ari_A_dbscan_mean", "lag_ssq_micro_mean", "lag_n_micro_centers_mean",
)


def write_report(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    rep = out_dir / "report"
    rep.mkdir(parents=True, exist_ok=True)
    written = []

    table = summary_table(out_dir)
    gain = gain_table(out_dir)
    oracle = oracle_table(out_dir)
    headroom = headroom_table(out_dir)
    regret = regret_table(out_dir)
    drift = drift_table(out_dir)
    for name, df in (("summary", table), ("gain", gain), ("oracle", oracle), ("headroom", headroom),
                     ("regret", regret), ("drift", drift)):
        p = rep / f"{name}.csv"
        df.to_csv(p, index=False)
        written.append(p)
    for fold, df in curves(out_dir).items():
        p = rep / f"curves_{fold}.csv"
        df.to_csv(p, index=False)
        written.append(p)

    runs = current_runs(out_dir)
    lines = ["# conf_pipeline report", ""]
    lines.append(
        "Every quality number is the *lagged* reading: the incumbent named after "
        "window t-1, scored on window t. `score` is the batch silhouette in [-1, 1] "
        "(-1 when undefined); `ari_win` is the ARI inside the window; "
        "`lag_ari_cumulative` is the ARI over the whole stream of lagged incumbents. "
        "`*_tail_*` columns cover the windows from `tail_from` (the freeze window) on, "
        "the region every arm shares; they are the headline numbers."
    )
    lines += ["", "## Summary (mean over seeds)", ""]
    if not table.empty:
        cols = ["algo", "fold", "arm", "n_seeds", "lag_ari_win_tail_mean", "lag_ari_win_tail_sd",
                "lag_ari_win_mean", "lag_ari_win_sd", "lag_score_mean", "lag_ari_cumulative",
                "incumbent_throughput_sps", "ensemble_runtime_s", "tuning_overhead", "lag_memory_mb_peak",
                "lag_n_pred_labels_mean", "n_switches", "switch_cost", "lag_undefined_share",
                "single_class_window_share", "frozen_at_window"]
        lines.append(_md_table(table, [c for c in cols if c in table]))

    lines += ["", "## Arms vs default", ""]
    lines.append(
        "`gain_tail` / `gain_all`: the arm's lagged ARI minus the default's, on the tail "
        "and on the whole stream; `throughput_ratio`: incumbent throughput over the default's."
    )
    lines.append("")
    lines.append(_md_table(gain, GAIN_COLS) if not gain.empty else "no non-default arm")

    lines += ["", "## Ceiling and headroom", ""]
    lines.append(
        "The ceiling is the static oracle's lagged ARI (its reruns, else the sweep's best "
        "point). `headroom_<arm> = (arm - default) / (ceiling - default)`, reported only when "
        f"the fold is tunable: `ceiling - default >= {C.MIN_CEILING_GAIN}` and "
        f"`> {Z_GATE} * se(default)`. `evolution_value = headroom_confstream - headroom_random_online`."
    )
    lines.append("")
    if not oracle.empty:
        lines.append(_md_table(oracle, ORACLE_COLS))
        lines.append("")
    if not headroom.empty:
        cols = [c for c in headroom.columns if c not in ("n_seeds_default",)]
        lines.append(_md_table(headroom, cols))
    else:
        lines.append("no (algo, fold) with a ceiling")

    lines += ["", "## Regret decomposition", ""]
    lines.append(
        "`A_static - a_lag = R_lag + R_proxy + R_search`, means over windows then seeds. "
        "`R_lag = a_inc - a_lag` (one window of delay), `R_proxy = a_star - a_inc` "
        "(silhouette picks a worse member than the best active one by ARI), "
        "`R_search = A_static - a_star` (the ensemble never held the static optimum; "
        "negative means adapting beat the best static configuration). Two rows per arm: "
        "`region = tail` covers the windows from the run's `tail_from` on, the region every "
        "arm shares and the primary one, against the oracle's tail reading; `region = all` "
        "covers the whole stream, against its whole-stream reading. `ceiling_source` says "
        "where `A_static` comes from."
    )
    lines.append("")
    lines.append(_md_table(regret, REGRET_COLS) if not regret.empty else "no online arm with lagged windows")

    lines += ["", "## Drift adaptation", ""]
    lines.append(
        "Per drift point: `pre` = lagged ARI on the window before; `drop_depth` = pre minus "
        "the lowest lagged ARI over the next 10 windows; `recovery_windows` = windows until the "
        "lagged ARI stays within 0.05 of `ref` for 3 windows (— = never; `ref` is the best "
        "active member, or `pre` without a member log or with a single active member); "
        "`param_ratio` = the scaled parameter after recovery over before the drift."
    )
    lines.append("")
    lines.append(_md_table(drift, DRIFT_COLS) if not drift.empty else "no drift points in these runs")

    lines += ["", "## Table A (common reclustering)", ""]
    have_a = [f for f in TABLE_A_FIELDS if f in table and table[f].notna().any()] if not table.empty else []
    if have_a:
        lines.append(
            "The same reclusterer (k-means with k = true classes; DBSCAN) on every algorithm's "
            "micro-clusters, so the gap left between algorithms is the online phase's."
        )
        lines.append("")
        cols = ["algo", "fold", "arm", "n_seeds", "lag_ari_win_tail_mean", *TABLE_A_FIELDS]
        rows = table[table[have_a].notna().any(axis=1)]
        lines.append(_md_table(rows, [c for c in cols if c in rows]))
    errors = (
        table[pd.to_numeric(table["lag_table_a_error_share"], errors="coerce") > 0]
        if "lag_table_a_error_share" in table else table.iloc[0:0]
    )
    if not have_a and errors.empty:
        lines.append("no Table A fields in runs.jsonl (runs predate the common reclusterer)")
    if not errors.empty:
        if have_a:
            lines.append("")
        lines.append(
            "Cells where the common reclusterer failed on some lagged windows: the share of lagged "
            "windows with an error, and the first message (lowest seed). Table A never stops a "
            "run, so those windows are simply missing from the means."
        )
        lines.append("")
        first = errors["lag_table_a_first_error"].map(
            lambda v: " ".join(v.split()).replace("|", "/") if isinstance(v, str) else v
        )
        lines.append(_md_table(errors.assign(lag_table_a_first_error=first),
                               ["algo", "fold", "arm", "n_seeds", "lag_table_a_error_share",
                                "lag_table_a_first_error"]))

    failed = runs[runs["status"] != "ok"] if not runs.empty else runs
    lines += ["", "## Runs that did not finish", ""]
    if len(failed):
        lines.append(_md_table(failed, ["algo", "fold", "seed", "arm", "status", "error"]))
    else:
        lines.append("none")
    lines += ["", "## Files", ""]
    lines += [f"- `{p.name}`" for p in written]
    md = rep / "report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(md)

    written += _plots(out_dir, rep)
    return written


def _plots(out_dir: Path, rep: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib is optional (requirements.txt)
        return []
    written = []
    for fold, df in curves(out_dir).items():
        if "lag_ari_win" not in df:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for (algo, arm), grp in df.groupby(["algo", "arm"]):
            style = "-" if arm == "confstream" else "--"
            axes[0].plot(grp["window"], grp["lag_ari_win"], style, label=f"{algo} ({arm})")
            if "lag_score" in grp:
                axes[1].plot(grp["window"], grp["lag_score"], style, label=f"{algo} ({arm})")
        axes[0].set_title(f"{fold}: lagged ARI per window")
        axes[1].set_title(f"{fold}: lagged silhouette per window")
        for ax in axes:
            ax.set_xlabel("window")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        p = rep / f"curves_{fold}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(C.OUT))
    args = parser.parse_args()
    for p in write_report(Path(args.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
