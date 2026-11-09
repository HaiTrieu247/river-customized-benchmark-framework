"""Follow-up readings of a retained run (the review's E2, E3, E6-E10): log-only, no benchmark run.

Every subcommand reads what ``run.py`` and ``sweep.py`` already wrote into an out dir (``runs.jsonl``,
``windows.jsonl``, ``members/``, ``sweep_windows.jsonl``, and ``lifecycle/`` where a newer run wrote it)
and writes its tables to ``<out-dir>/report/followup/<subcommand>/``: CSVs and ``<subcommand>.md``, which
says what was read and how. No other file of the dir is touched, and no number is tuned: every threshold
below is a flag and a placeholder.

- ``post-freeze`` (E2)  the paired contrasts online - every other arm on the tail windows t >= t0,
  t >= t0 + 1 and t >= t0 + 2 (t0 = the freeze window), side by side with their differences; the
  challenger-timing audit (member logs, window rows and, where written, the lifecycle log).
- ``prefix-identity`` (E3)  whether the frozen branches share the online arm's prefix up to the freeze:
  ``pred_sha`` where logged, else the weaker comparison of the logged readings.
- ``survivors`` (E7)  which members a pick by labels made after window t can choose on t+1, and why the
  others drop out.
- ``lagged-validity`` (E7)  the per-window Spearman correlation between a member's silhouette on t and
  its ARI on t+1, beside the same-window one, per stream group and by strata.
- ``recovery`` (E8)  ``make_robustness_extra.py``'s recovery after drift, per event, with censoring, and
  a check against the printed numbers.
- ``equivalence`` (E6)  two one-sided tests of the main paired contrasts against a margin, and per pair
  the seed-noise band beside the equivalence reading.
- ``figure`` (E10)  Figure 1's data regenerated from ``windows.jsonl`` and compared with the paper's.
- ``feasibility`` (E10)  the runtime gate replayed from the member logs, and the counterfactual without
  each window's fastest member.
- ``substitute`` (E9)  Kendall's tau between the default and online rankings and the default pick's
  regret with the patched DenStream in DenStream's place.

Run from the repo root::

    python -m conf_pipeline.followup <subcommand> --out-dir conf_pipeline/out/full [flags]
"""

from __future__ import annotations

import argparse
import math
import os
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from . import analyze, report
from . import robustness as R
from . import robustness_check as RC
from . import selection as S

FOLLOWUP = "followup"
SUBCOMMANDS: tuple[str, ...] = ("post-freeze", "prefix-identity", "survivors", "lagged-validity", "recovery",
                                "equivalence", "figure", "feasibility", "substitute")
#: The online arm every contrast is read against.
ONLINE = "confstream"
N_BOOT, BOOT_SEED = 2000, 0
#: The interval of a two one-sided test at 5% each.
CI90 = (5.0, 95.0)
#: Placeholders, every one a flag: the equivalence margin in tail ARI; the smallest spread of the
#: members' ARIs on a window for its rank correlation to be read (the ceiling gate's minimum gain), and
#: the fewest members it is read over.
MARGIN = 0.05
MIN_SPREAD = C.MIN_CEILING_GAIN
MIN_MEMBERS = 3
#: ``make_robustness_extra.py``'s recovery rule: within this of the reference, ``report.RECOVERY_RUN``
#: windows in a row, with its float slack.
RECOVERY_DELTA = 0.05
_RECOVERY_SLACK = 1e-9
_EPS = 1e-12
PAPER_DAT = C.REPO_ROOT / "result" / "otb_full" / "new_paper_ecml" / "dat"
FIGURE_FOLDS: tuple[str, ...] = ("scale_drift", "k_drift")
FIGURE_ARMS: tuple[str, ...] = ("default", "confstream", "frozen_prefix", "oracle_static")
#: Strata of ``lagged-validity`` (placeholders): model age t - born_window, micro-clusters, and the share
#: of the window's points the member gave its sentinel label. ``(lo, hi)`` is lo <= v <= hi for the age and
#: micro-cluster bins; for the sentinel share the first bin is exactly 0 and the others are lo < v <= hi.
AGE_BINS: tuple[tuple, ...] = ((1, 1), (2, 5), (6, 20), (21, None))
MICRO_BINS: tuple[tuple, ...] = ((0, 1), (2, 10), (11, 100), (101, 1000), (1001, None))
SENTINEL_BINS: tuple[tuple, ...] = ((0.0, 0.0), (0.0, 0.1), (0.1, 0.5), (0.5, 1.0))
#: Origins of the members that fill the ensemble at the start, never challenged (``confstream.BIRTH_ORIGINS``).
START_ORIGINS: tuple[str, ...] = ("start", "initial_offspring")
#: Challenge outcomes that put a challenger in the active set.
ADMITTED: tuple[str, ...] = ("accepted", "filled")
NOT_LOGGED = "not logged"


# --- shared ----------------------------------------------------------------------------------------

def followup_dir(out_dir: Path, sub: str) -> Path:
    path = Path(out_dir) / C.REPORT_DIR.name / FOLLOWUP / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(out_dir: Path, sub: str, tables: dict, md: str, suffix: str = "") -> list[Path]:
    """The tables as ``<name><suffix>.csv`` and ``md`` as ``<sub><suffix>.md`` in the subcommand's directory."""
    rep = followup_dir(out_dir, sub)
    paths = []
    for name, df in tables.items():
        path = rep / f"{name}{suffix}.csv"
        (df if df is not None else pd.DataFrame()).to_csv(path, index=False)
        paths.append(path)
    path = rep / f"{sub}{suffix}.md"
    path.write_text(md, encoding="utf-8")
    return paths + [path]


def _table(df: pd.DataFrame | None, cols=None, digits: int = 3) -> str:
    if df is None or df.empty:
        return "(no rows)\n"
    cols = [c for c in (cols or df.columns) if c in df.columns]
    return report._md_table(df[cols], cols, digits)


def _ok_runs(out_dir: Path, protocol: bool = True) -> pd.DataFrame:
    """The current ``ok`` run rows (``report.current_runs``: deployable arms only), without the
    implementation and label variants unless ``protocol`` is False."""
    runs = report.current_runs(out_dir)
    if runs.empty:
        return runs
    runs = runs[runs["status"] == "ok"]
    if protocol:
        runs = runs[~runs["algo"].isin(list(C.variant_algos()))]
    return runs.reset_index(drop=True)


def _jobkey(algo, fold, seed, arm, config_id) -> str:
    return f"{algo}|{fold}|{report._seed_key(seed)}|{arm}|{report._cid(config_id)}"


def _jobkeys(df: pd.DataFrame) -> list[str]:
    cid = df["config_id"] if "config_id" in df else [None] * len(df)
    return [_jobkey(*k) for k in zip(df["algo"], df["fold"], df["seed"], df["arm"], cid)]


def _run_field(runs: pd.DataFrame, field: str, default=None) -> dict:
    """jobkey -> a field of the run row (``default`` where the row lacks it)."""
    if runs.empty:
        return {}
    vals = runs[field] if field in runs else [None] * len(runs)
    out = {}
    for key, v in zip(_jobkeys(runs), vals):
        v = report._none(v)
        out[key] = default if v is None else v
    return out


def tuned_arms(runs: pd.DataFrame) -> list[str]:
    """Arms whose runs hold an ensemble (``e_size`` > 1): the tuner and its branches."""
    if runs.empty or "e_size" not in runs:
        return []
    e = pd.to_numeric(runs["e_size"], errors="coerce")
    return sorted(set(runs.loc[e > 1, "arm"]))


def frozen_windows(runs: pd.DataFrame) -> dict:
    """jobkey -> the window the run froze after (``frozen_at_window``, else ``freeze_after_window``)."""
    out = {}
    for key, a, b in zip(_jobkeys(runs), runs.get("frozen_at_window", [None] * len(runs)),
                         runs.get("freeze_after_window", [None] * len(runs))):
        v = report._num(a)
        v = report._num(b) if v is None else v
        if v is not None:
            out[key] = int(v)
    return out


def _header(out_dir: Path, sub: str) -> list[str]:
    return [f"# Follow-up `{sub}`", "",
            f"Read from `{out_dir}`. Log-only: no benchmark or tuning run was made for this reading.", ""]


# --- member logs -------------------------------------------------------------------------------------

MEMBER_FIELDS: tuple[str, ...] = (
    "algo", "fold", "seed", "arm", "config_id", "tuner_seed", "model_seed", "member_id", "window", "is_test",
    "born_window", "score", "silhouette_defined", "partitioned", "silhouette_raw", "feasible", "feasible_reason",
    "error", "runtime_s", "memory_mb", "n_micro_clusters", "n_sentinel", "n_pred_labels", "n_true_classes", "ari_win",
    "pred_sha", "params",
)
_MEMBER_FLOATS = ("score", "silhouette_raw", "runtime_s", "memory_mb", "n_micro_clusters", "n_sentinel",
                  "n_pred_labels", "n_true_classes", "ari_win")
MEMBER_COLUMNS: list[str] = [c for c in MEMBER_FIELDS if c not in ("error", "params")] + [
    "params_key", "has_error", "infeasible", "usable", "has_partition", "job"]


def _read_member_file(path: str) -> pd.DataFrame:
    """One member log as a frame: the fields above, plus

    - ``usable``: ``feasible is not False and silhouette_defined is not False`` (``report._job_readings``),
    - ``has_partition``: no error and a partition (``partitioned``, else ``silhouette_defined``), the
      runtime gate's reference set (``confstream.apply_feasibility``),
    - ``infeasible`` (``feasible is False``), ``has_error``, ``params_key`` (the parameters as text) and
      ``job`` (``algo|fold|seed|arm|config_id``)."""
    rows = []
    for r in C.read_jsonl(Path(path)):
        err = r.get("error") is not None
        part = r.get("partitioned")
        sdef = r.get("silhouette_defined")
        params = r.get("params")
        rows.append([r.get(k) for k in MEMBER_FIELDS if k not in ("error", "params")] + [
            params if params is None else str(sorted(params.items())) if isinstance(params, dict) else str(params),
            err, r.get("feasible") is False,
            r.get("feasible") is not False and sdef is not False,
            (not err) and bool(part if part is not None else sdef),
            _jobkey(r.get("algo"), r.get("fold"), r.get("seed"), r.get("arm"), r.get("config_id")),
        ])
    df = pd.DataFrame(rows, columns=MEMBER_COLUMNS)
    for c in _MEMBER_FLOATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["window"] = pd.to_numeric(df["window"], errors="coerce").astype("Int64")
    df["born_window"] = pd.to_numeric(df["born_window"], errors="coerce")
    df["is_test"] = df["is_test"].astype(bool)
    return df


def load_members(out_dir: Path, arms=None, workers: int | None = None) -> pd.DataFrame:
    """Every member-window row of ``<out_dir>/members/*.jsonl`` (``_read_member_file``), files read in
    parallel and concatenated in name order; ``arms`` keeps those arms' files (by name, then by row). An
    empty frame with ``MEMBER_COLUMNS`` when the directory is missing or empty."""
    files = [str(p) for p in report._member_files(out_dir, arms=set(arms) if arms is not None else None)]
    if not files:
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    n = workers if workers is not None else min(8, os.cpu_count() or 1)
    if n > 1 and len(files) > 4:
        with Pool(n) as pool:
            parts = pool.map(_read_member_file, files, chunksize=4)
    else:
        parts = [_read_member_file(f) for f in files]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    if arms is not None:
        df = df[df["arm"].isin(list(arms))].reset_index(drop=True)
    df["window"] = df["window"].astype(int)
    return df


def _window_rows(out_dir: Path, cols=("incumbent_id", "lag_member_id", "pred_sha", "lag_pred_sha")) -> pd.DataFrame:
    """``windows.jsonl`` (``report.load_windows``) reduced to the job, the window and ``cols`` it has."""
    win = report.load_windows(out_dir)
    if win.empty:
        return pd.DataFrame(columns=["job", "algo", "fold", "seed", "arm", "window", *cols])
    keep = ["algo", "fold", "seed", "arm", "window"] + [c for c in cols if c in win]
    out = win[keep].copy()
    out.insert(0, "job", _jobkeys(win))
    out["window"] = pd.to_numeric(out["window"], errors="coerce").astype(int)
    return out


# --- a bootstrap over units for readings that are not robustness' paired effects ---------------------

def unit_bootstrap(cells: pd.DataFrame, value: str = "value", unit: str = "fold", n_boot: int = N_BOOT,
                   seed: int = BOOT_SEED) -> pd.DataFrame:
    """Per stream group (and ``all``), pooled over the algorithms and per algorithm: the mean over the units
    (folds, or source blocks: ``robustness.unit_index``) of each unit's mean over its (algo, fold) cells, as
    ``robustness.unit_statistics`` averages its paired effects, with a percentile interval (``robustness.CI``)
    from resampling the units within each group (``robustness.draw_units``). Cells (seed means already taken)
    are held fixed; ``n_used`` replicates had a reading. ``cells``: ``algo, fold, value``."""
    cols = ["unit", "group", "algo", "point", "low", "high", "n_cells", "n_units", "n_used", "n_boot"]
    df = cells.dropna(subset=[value])
    if df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for algo, sub in [("pooled", df), *sorted(df.groupby("algo"), key=lambda kv: kv[0])]:
        folds = sorted(sub["fold"].unique())
        fsum = sub.groupby("fold")[value].sum().reindex(folds).to_numpy(dtype=float)
        fcnt = sub.groupby("fold")[value].size().reindex(folds).to_numpy(dtype=float)
        groups = np.array([C.stream_group(f) for f in folds])
        index = R.unit_index(folds, unit)
        pos = np.concatenate([p for _, _, p in index])
        ids = np.concatenate([np.full(len(p), k) for k, (_, _, p) in enumerate(index)])

        def stats(pos, ids):
            n_inst = int(ids.max()) + 1 if len(ids) else 0
            s = np.bincount(ids, weights=fsum[pos], minlength=n_inst)
            c = np.bincount(ids, weights=fcnt[pos], minlength=n_inst)
            first = np.unique(ids, return_index=True)[1]
            g = groups[pos][first]
            m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            out = {}
            for grp in (*C.STREAM_GROUPS, R.ALL):
                sel = np.ones(len(m), bool) if grp == R.ALL else g == grp
                out[grp] = (float(np.nanmean(m[sel])) if sel.any() and np.isfinite(m[sel]).any() else np.nan,
                            float(c[sel].sum()), int(sel.sum()))
            return out

        point = stats(pos, ids)
        rng = np.random.default_rng(seed)
        draws = {g: [] for g in point}
        for _ in range(n_boot):
            p, i = R.draw_units(folds, unit, rng, index)
            for g, (v, _, _) in stats(p, i).items():
                draws[g].append(v)
        for g, (v, n_cells, n_units) in point.items():
            d = np.array(draws[g], dtype=float)
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, R.CI) if len(d) else (np.nan, np.nan)
            rows.append({"unit": unit, "group": g, "algo": algo, "point": v, "low": float(lo), "high": float(hi),
                         "n_cells": int(n_cells), "n_units": n_units, "n_used": int(len(d)), "n_boot": n_boot})
    return pd.DataFrame(rows, columns=cols)


@contextmanager
def interval(percentiles):
    """``robustness``' interval percentiles set to ``percentiles`` for the duration (its bootstrap reads the
    module constant ``CI`` when it runs), restored after."""
    old = R.CI
    R.CI = tuple(percentiles)
    try:
        yield
    finally:
        R.CI = old


_NO_LOSS = pd.DataFrame(columns=["algo", "fold", "seed", "a_lag", "a_star_lag"])
CONTRAST_COLS = ["contrast", "unit", "group", "algo", "point", "low", "high", "n_pairs", "n_units", "n_used", "n_boot"]


def contrast(seeds: pd.DataFrame, other: str, unit: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
             online: str = ONLINE) -> pd.DataFrame:
    """``online - other`` by ``robustness.unit_statistics`` itself: seeds paired by seed number within a pair,
    pairs averaged within each unit, units averaged per stream group; per algorithm and pooled. The published
    comparisons (``other`` = ``default``, ``frozen_prefix``, or ``oracle_static``: the reference, its reruns drawn
    apart) are read on the cube of every arm of ``robustness.ARMS`` in ``seeds``, as published: the bootstrap's
    draws depend on which arms the cube holds, so this reproduces the published intervals. Any other arm takes
    ``default``'s place in that cube. ``seeds``: ``Inputs.seeds``-shaped (``algo, fold, arm, seed, value``)."""
    effects = {"default": "confstream - default", "frozen_prefix": "confstream - frozen_prefix",
               "oracle_static": "confstream - reference"}
    if other in effects and online == "confstream":
        sub = seeds[seeds["arm"].isin(list(R.ARMS))].copy()
        effect, read = effects[other], other
    else:
        keep = [a for a in R.ARMS if a not in ("default", "confstream")]
        sub = pd.concat([seeds[seeds["arm"].isin(keep)], seeds[seeds["arm"] == online].assign(arm="confstream"),
                         seeds[seeds["arm"] == other].assign(arm="default")], ignore_index=True)
        effect, read = "confstream - default", "default"
    if sub.empty or not (sub["arm"] == "confstream").any() or not (sub["arm"] == read).any():
        return pd.DataFrame(columns=CONTRAST_COLS)
    con = R.Construction(reference="followup", metric="ari", basis="followup", seeds=sub.reset_index(drop=True),
                         losses=_NO_LOSS, tunable={})
    st = R.unit_statistics(con, unit, n_boot, seed)
    st = st[st["statistic"] == effect].copy()
    st.insert(0, "contrast", f"{online} - {other}")
    return st[CONTRAST_COLS].reset_index(drop=True)


# --- E2: post-freeze ---------------------------------------------------------------------------------

def _freeze_windows(runs: pd.DataFrame) -> dict:
    """(algo, fold, seed, arm) -> the run's freeze window t0: ``tail_freeze_window`` where the run records
    it, else its ``tail_from`` (runs before 2026-11-08 started their tail at the freeze window)."""
    if runs.empty:
        return {}
    if "tail_freeze_window" not in runs:
        return R._tail_from(runs)
    out = R._tail_from(runs)
    vals = pd.to_numeric(runs["tail_freeze_window"], errors="coerce")
    for a, f, s, arm, t in zip(runs["algo"], runs["fold"], runs["seed"], runs["arm"], vals):
        if not pd.isna(t):
            out[(a, f, s, arm)] = int(t)
    return out


def start_specs(runs: pd.DataFrame, starts=None, offsets=(0, 1, 2)) -> list[tuple[str, object]]:
    """``[(label, tail_from)]``: absolute ``starts`` as given, else each run's freeze window t0
    (``_freeze_windows``; ``FREEZE_AFTER_WINDOW`` in every protocol run) plus each of ``offsets``."""
    if starts:
        return [(f"t>={int(s)}", int(s)) for s in starts]
    tail = _freeze_windows(runs)
    values = sorted(set(tail.values()))
    out = []
    for k in offsets:
        label = (f"t>={values[0] + k}" if len(values) == 1 else f"t>=tail_from+{k}") if values else f"t>={C.FREEZE_AFTER_WINDOW + k}"
        spec = {key: v + k for key, v in tail.items()} if tail else C.FREEZE_AFTER_WINDOW + k
        out.append((label, spec))
    return out


def post_freeze_contrasts(windows: pd.DataFrame, runs: pd.DataFrame, specs, n_boot: int = N_BOOT,
                          seed: int = BOOT_SEED, subsets: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per start of ``specs``: the per-run mean lagged ARI over windows >= it (``robustness.tail_values``),
    then ``contrast`` of the online arm against every other arm, units fold and block; the later starts also
    as their paired difference from the first (the contrast of the per-run differences, bootstrapped on the
    same draws). ``subsets``: name -> (arm, set of (algo, fold, seed)) restricting that arm's contrast to those
    pairs of runs. Returns ``(contrasts, per_run)``."""
    others = [a for a in sorted(set(runs["arm"])) if a != ONLINE]
    per_run = []
    for label, spec in specs:
        v = R.tail_values(windows, spec, "lag_ari_win")
        per_run.append(v.assign(start=label))
    per_run = pd.concat(per_run, ignore_index=True) if per_run else pd.DataFrame()
    base = specs[0][0]
    readings = [(label, per_run[per_run["start"] == label].drop(columns="start")) for label, _ in specs]
    first = readings[0][1]
    for label, v in readings[1:]:
        d = v.merge(first, on=["algo", "fold", "arm", "seed"], suffixes=("", "_base"))
        d["value"] = d["value"] - d["value_base"]
        readings.append((f"{label} minus {base}", d[["algo", "fold", "arm", "seed", "value"]]))
    rows = []
    for label, v in readings:
        for other in others:
            for unit in R.UNITS:
                st = contrast(v, other, unit, n_boot, seed)
                rows.append(st.assign(start=label, subset="all runs"))
                for name, (arm, keep) in (subsets or {}).items():
                    if arm != other:
                        continue
                    mask = [(a, f, report._seed_key(s)) in keep or ar not in (ONLINE, other)
                            for a, f, s, ar in zip(v["algo"], v["fold"], v["seed"], v["arm"])]
                    st = contrast(v[np.array(mask, dtype=bool)], other, unit, n_boot, seed)
                    rows.append(st.assign(start=label, subset=name))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=CONTRAST_COLS + ["start", "subset"])
    return out[["subset", "start", *CONTRAST_COLS]], per_run


def lag_sha_divergence(wrows: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per run of an arm that froze and has an online counterpart (same algo, fold, tuner seed, model seed): the first
    window where its ``lag_pred_sha`` differs from the online arm's, against the run's freeze window t0.
    The claim it checks: window t0 + 1 is read by the incumbent named after t0, the same model in every
    branch when the prefix is shared, so the first window that can differ is t0 + 2."""
    cols = ["algo", "fold", "seed_key", "model_seed", "arm", "freeze", "n_windows", "first_divergent_window",
            "offset_from_freeze", "identical_through_freeze_plus_1", "note"]
    if wrows.empty or "lag_pred_sha" not in wrows or not wrows["lag_pred_sha"].notna().any():
        return pd.DataFrame(columns=cols)
    keys = _branch_keys(runs)
    freeze = frozen_windows(runs)
    sha = {k: {int(w): (v if isinstance(v, str) else None) for w, v in zip(g["window"], g["lag_pred_sha"])}
           for k, g in wrows.groupby("job")}
    rows = []
    for job, (bkey, arm) in keys.items():
        if arm == ONLINE or job not in freeze:  # a branch is an arm that froze; the others are other tuners
            continue
        ref = next((j for j, (b, a) in keys.items() if b == bkey and a == ONLINE), None)
        if ref is None or job not in sha:
            continue
        a, b = sha[job], sha.get(ref, {})
        common = sorted(w for w in set(a) & set(b) if a[w] is not None or b[w] is not None)
        first = next((w for w in common if a[w] != b[w]), None)
        t0 = freeze.get(job)
        rows.append({"algo": bkey[0], "fold": bkey[1], "seed_key": bkey[2], "model_seed": bkey[3], "arm": arm,
                     "freeze": t0, "n_windows": len(common), "first_divergent_window": first,
                     "offset_from_freeze": (first - t0) if first is not None and t0 is not None else None,
                     "identical_through_freeze_plus_1": (None if t0 is None else
                                                         first is None or first > t0 + 1),
                     "note": "never differs" if first is None else ""})
    out = pd.DataFrame(rows, columns=cols)
    for c in ("freeze", "n_windows", "first_divergent_window", "offset_from_freeze"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def _branch_keys(runs: pd.DataFrame) -> dict:
    """jobkey -> ((algo, fold, tuner seed or seed, model seed, config_id), arm): runs with the same first part
    are branches of one tuner run."""
    out = {}
    tuner = runs["tuner_seed"] if "tuner_seed" in runs else [None] * len(runs)
    model = runs["model_seed"] if "model_seed" in runs else [None] * len(runs)
    cid = runs["config_id"] if "config_id" in runs else [None] * len(runs)
    for job, a, f, s, t, m, c, arm in zip(_jobkeys(runs), runs["algo"], runs["fold"], runs["seed"], tuner, model,
                                          cid, runs["arm"]):
        t = report._seed_key(report._none(t))
        out[job] = ((a, f, t if t is not None else report._seed_key(s), report._seed_key(report._none(m)),
                     report._cid(c)), arm)
    return out


# --- E2: challenger timing ----------------------------------------------------------------------------

AUDIT_COLS = ["check", "source", "n_checked", "n_violations", "example"]


def _audit(check: str, source: str, n: int, bad: pd.DataFrame | list, show=()) -> dict:
    n_bad = len(bad)
    ex = ""
    if n_bad:
        first = bad.iloc[0] if isinstance(bad, pd.DataFrame) else bad[0]
        ex = "; ".join(f"{k}={first[k]}" for k in show) if show else str(first)
    return {"check": check, "source": source, "n_checked": int(n), "n_violations": int(n_bad), "example": ex}


def member_timing_checks(members: pd.DataFrame, wrows: pd.DataFrame) -> list[dict]:
    """What the member logs and window rows alone say about challenger timing (tuned arms):

    - no member is scored on or before the window it was born on (it was warm-started on that window, so a
      score there would read a window it learned twice);
    - a member on test is scored exactly on its birth window + 1 (a challenger lives one window on test);
    - no member has two rows on one window, nor a gap in its windows (it never leaves and comes back);
    - the incumbent named on window t has a row on t, and if it was on test there it is active on t + 1 (it
      was admitted on t, so it is first read, lagged, on t + 1);
    - the lagged member read on t + 1 is the incumbent named on t."""
    show = ("job", "member_id", "window", "born_window")
    m = members
    out = [
        _audit("no member scored on or before its birth window", "members", len(m),
               m[m["window"] <= m["born_window"]], show),
        _audit("a member on test is scored on its birth window + 1", "members", int(m["is_test"].sum()),
               m[m["is_test"] & (m["window"] != m["born_window"] + 1)], show),
        _audit("no member has two rows on one window", "members", len(m),
               m[m.duplicated(["job", "member_id", "window"], keep=False)], show),
    ]
    span = m.groupby(["job", "member_id"])["window"].agg(["min", "max", "nunique"]).reset_index()
    out.append(_audit("no member leaves and comes back (no gap in its windows)", "members", len(span),
                      span[span["nunique"] != span["max"] - span["min"] + 1], ("job", "member_id", "min", "max")))
    if wrows.empty or "incumbent_id" not in wrows:
        out.append({"check": "incumbent admitted before it is read", "source": "window rows", "n_checked": 0,
                    "n_violations": 0, "example": "windows.jsonl has no incumbent_id: not checked"})
        return out
    w = wrows[wrows["job"].isin(set(m["job"]))][["job", "window", "incumbent_id", *(
        ["lag_member_id"] if "lag_member_id" in wrows else [])]].copy()
    w["incumbent_id"] = pd.to_numeric(w["incumbent_id"], errors="coerce")
    here = m[["job", "member_id", "window", "is_test"]].rename(columns={"member_id": "incumbent_id"})
    here["incumbent_id"] = here["incumbent_id"].astype(float)
    nxt = m[["job", "member_id", "window", "is_test"]].copy()
    nxt["window"] = nxt["window"] - 1
    nxt = nxt.rename(columns={"member_id": "incumbent_id", "is_test": "next_is_test"})
    nxt["incumbent_id"] = nxt["incumbent_id"].astype(float)
    j = w.merge(here, on=["job", "window", "incumbent_id"], how="left", indicator="row")
    j = j.merge(nxt, on=["job", "window", "incumbent_id"], how="left")
    last = m.groupby("job")["window"].max()
    has_next = j["window"] < j["job"].map(last)
    out.append(_audit("the incumbent named on t has a member row on t", "members + window rows", len(j),
                      j[j["row"] != "both"], ("job", "window", "incumbent_id")))
    on_test = j[(j["row"] == "both") & j["is_test"].eq(True) & has_next]
    out.append(_audit("an incumbent on test on t (admitted on t) is active on t + 1", "members + window rows",
                      len(on_test), on_test[on_test["next_is_test"].isna() | on_test["next_is_test"].eq(True)],
                      ("job", "window", "incumbent_id")))
    if "lag_member_id" in w:
        a = w[["job", "window", "incumbent_id"]].copy()
        a["window"] = a["window"] + 1
        b = w[["job", "window", "lag_member_id"]].merge(a, on=["job", "window"])
        b = b[b["lag_member_id"].notna()]
        out.append(_audit("the lagged member read on t + 1 is the incumbent named on t", "window rows", len(b),
                          b[pd.to_numeric(b["lag_member_id"]) != b["incumbent_id"]],
                          ("job", "window", "lag_member_id", "incumbent_id")))
    return out


def load_lifecycle(out_dir: Path) -> pd.DataFrame | None:
    """Every row of ``<out_dir>/lifecycle/*.jsonl`` with its ``job``; None when the directory is missing."""
    d = Path(out_dir) / "lifecycle"
    if not d.is_dir():
        return None
    rows = [r for p in sorted(d.glob("*.jsonl")) for r in C.read_jsonl(p)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "job", _jobkeys(df))
    return df


def lifecycle_checks(events: pd.DataFrame, members: pd.DataFrame, wrows: pd.DataFrame) -> list[dict]:
    """What the lifecycle log records allow on top of the member logs:

    - every member is born once; every challenge names a challenger born on the window before (scored on its
      birth window + 1, never earlier, and challenged once);
    - the incumbent named at a decision on window d is a start member, or a challenger admitted (accepted or
      filled) on a window <= d: it is first read, lagged, on the window after its admission;
    - a member's first scored row is its lifecycle birth window + 1, and the rows' ``born_window`` agree;
    - every decision names the window row's incumbent;
    - a freeze that keeps one member (``reinit_after_freeze`` False, no ``freeze_mode``) has no birth after it
      and names that member at every later decision. Other freeze modes are not checked here."""
    out = []
    ev = events
    births = ev[ev["event"] == "birth"]
    out.append(_audit("every member is born once", "lifecycle", len(births),
                      births[births.duplicated(["job", "member_id"], keep=False)], ("job", "member_id", "window")))
    born = births.drop_duplicates(["job", "member_id"]).set_index(["job", "member_id"])
    ch = ev[ev["event"] == "challenge"].copy()
    if len(ch):
        ch["birth"] = [born["window"].get((j, c)) if (j, c) in born.index else None
                       for j, c in zip(ch["job"], ch["challenger_id"])]
        b = pd.to_numeric(ch["birth"], errors="coerce")
        out.append(_audit("a challenger is challenged on its birth window + 1, never earlier", "lifecycle",
                          len(ch), ch[b.isna() | (ch["window"] < b + 1)], ("job", "challenger_id", "window", "birth")))
        out.append(_audit("a challenger is challenged no later than its birth window + 1", "lifecycle", len(ch),
                          ch[ch["window"] > b + 1], ("job", "challenger_id", "window", "birth")))
        out.append(_audit("a challenger is challenged once", "lifecycle", len(ch),
                          ch[ch.duplicated(["job", "challenger_id"], keep=False)], ("job", "challenger_id", "window")))
    adm = ch[ch["outcome"].isin(ADMITTED)].groupby(["job", "challenger_id"])["window"].min() if len(ch) else pd.Series(dtype=float)
    dec = ev[ev["event"] == "decision"].copy()
    if len(dec):
        def admitted(job, mid):
            key = (job, mid)
            if key in born.index and born.loc[key, "origin"] in START_ORIGINS:
                return born.loc[key, "window"]
            return adm.get(key)
        dec["admitted"] = [admitted(j, m) for j, m in zip(dec["job"], dec["incumbent_id"])]
        a = pd.to_numeric(dec["admitted"], errors="coerce")
        out.append(_audit("no incumbent before the window of its admission (read on the window after it)",
                          "lifecycle", len(dec), dec[a.isna() | (dec["window"] < a)],
                          ("job", "window", "incumbent_id", "admitted")))
        if not wrows.empty and "incumbent_id" in wrows:
            j = dec[["job", "window", "incumbent_id"]].merge(
                wrows[["job", "window", "incumbent_id"]].rename(columns={"incumbent_id": "row_incumbent"}),
                on=["job", "window"], how="left")
            out.append(_audit("every decision names the window row's incumbent", "lifecycle + window rows",
                              len(j), j[pd.to_numeric(j["row_incumbent"], errors="coerce") != j["incumbent_id"]],
                              ("job", "window", "incumbent_id", "row_incumbent")))
    if not members.empty:
        m = members[members["job"].isin(set(ev["job"]))]
        first = m.groupby(["job", "member_id"]).agg(first=("window", "min"), born_row=("born_window", "first"))
        first = first.join(born[["window", "origin"]].rename(columns={"window": "born"}), how="left").reset_index()
        # a start member is scored first on the job's first scored window (after every calibration window)
        start = first["job"].map(m.groupby("job")["window"].min())
        expect = np.where(first["origin"].isin(START_ORIGINS), start, first["born"] + 1)
        out.append(_audit("a member's first scored window is its lifecycle birth window + 1 (a start member: the "
                          "job's first scored window)", "lifecycle + members",
                          len(first), first[first["born"].isna() | (first["first"] != expect)],
                          ("job", "member_id", "first", "born")))
        out.append(_audit("member rows' born_window is the lifecycle birth window", "lifecycle + members",
                          len(first), first[first["born_row"] != first["born"]], ("job", "member_id", "born_row", "born")))
    fr = ev[ev["event"] == "freeze"]
    kept_one = fr[(fr.get("reinit_after_freeze", pd.Series(False, index=fr.index)) != True)  # noqa: E712
                  & (fr["freeze_mode"].isna() if "freeze_mode" in fr else True)]
    bad_birth, bad_dec, n_b, n_d = [], [], 0, 0
    for row in kept_one.to_dict("records"):
        later = ev[(ev["job"] == row["job"]) & (ev["window"] > row["window"])]
        lb = later[later["event"] == "birth"]
        ld = later[later["event"] == "decision"]
        n_b += 1
        n_d += len(ld)
        bad_birth += lb.to_dict("records")
        bad_dec += ld[ld["incumbent_id"] != row["member_id"]].to_dict("records")
    out.append(_audit("a one-member freeze has no birth after it", "lifecycle", n_b, bad_birth,
                      ("job", "member_id", "window")))
    out.append(_audit("a one-member freeze names its member at every later decision", "lifecycle", n_d, bad_dec,
                      ("job", "window", "incumbent_id")))
    return out


UNCHECKABLE = (
    "the learning itself: no log records the learn_one calls, so that each member saw each window's samples "
    "exactly once and in order is implied by its scoring rows and its birth's warm start, not observed;",
    "the warm start's content: a birth records how many samples it replayed (lifecycle `warm_start_samples`), "
    "not which;",
    "the prequential order within a window (predictions made before the model learned each point) is "
    "`run_once`'s and is not logged.",
)


# --- E3: prefix identity --------------------------------------------------------------------------------

READING_FIELDS: tuple[str, ...] = ("is_test", "born_window", "score", "ari_win", "n_micro_clusters", "n_pred_labels",
                                   "infeasible", "feasible_reason", "has_error", "params_key")
PREFIX_COLS = ["algo", "fold", "seed_key", "model_seed", "arm_a", "arm_b", "freeze", "n_windows", "sha_status",
               "sha_identical_windows", "sha_first_divergent", "readings_identical_windows", "readings_first_divergent",
               "readings_cause", "incumbent_identical_windows", "incumbent_first_divergent", "first_divergent", "cause",
               "identical_through_freeze"]
#: The cause given when the readings agree and the incumbents part: members tied on the score, and the tie went
#: another way in each branch (``out/full`` broke ties by measured runtime, which differs between processes).
TIE_CAUSE = "incumbent: a tie broken differently (readings equal)"


def _equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return ((a == b) | (a.isna() & b.isna())).to_numpy(dtype=bool)


def _compare_prefix(ma: pd.DataFrame, mb: pd.DataFrame, upto: int, fields) -> tuple[dict, pd.DataFrame]:
    """Per window <= ``upto`` of two jobs' member rows: identical when both hold the same members and every
    one of ``fields`` agrees. Returns ``({window: bool}, the merged rows)``."""
    a = ma[ma["window"] <= upto][["window", "member_id", *fields]]
    b = mb[mb["window"] <= upto][["window", "member_id", *fields]]
    j = a.merge(b, on=["window", "member_id"], how="outer", suffixes=("_a", "_b"), indicator=True)
    same = j["_merge"].eq("both").to_numpy().copy()
    for f in fields:
        same &= _equal(j[f"{f}_a"], j[f"{f}_b"])
    j["same"] = same
    ok = j.groupby("window")["same"].all().to_dict()
    for w in range(1, upto + 1):
        ok.setdefault(w, False if (w in set(a["window"]) or w in set(b["window"])) else None)
    return {w: v for w, v in ok.items() if v is not None}, j


def _cause(j: pd.DataFrame, w: int) -> str:
    """Why the readings of two branches part on window ``w``: a member in one branch only, the runtime gate
    (a member shed by it in one branch only), its configuration, or its readings."""
    r = j[j["window"] == w]
    if not r["_merge"].eq("both").all():
        return "member set"
    d = r[~r["same"]]
    reasons = set(d["feasible_reason_a"].dropna()) | set(d["feasible_reason_b"].dropna())
    if "runtime" in reasons and (d["feasible_reason_a"].astype(str) != d["feasible_reason_b"].astype(str)).any():
        return "runtime gate"
    if (d["params_key_a"].astype(str) != d["params_key_b"].astype(str)).any():
        return "configuration"
    if (d["feasible_reason_a"].astype(str) != d["feasible_reason_b"].astype(str)).any():
        return "another cap"
    return "model or scoring readings"


def prefix_table(members: pd.DataFrame, wrows: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per pair of branches of one tuner run (every arm that froze, against the online arm, and the frozen arms
    against each other), over the windows up to and including the freeze window F (the windows every branch
    ran as one tuner):

    - ``sha_*``: the windows on which every member's ``pred_sha`` is the same in both (same members), and the
      first that is not; ``sha_status`` "not logged" when either side has no ``pred_sha``;
    - ``readings_*``: the same over the logged readings (``READING_FIELDS``), a weaker test: different readings
      show the branches parted, equal ones do not show equal predictions; ``readings_cause`` says what parted
      them first;
    - ``incumbent_*``: the window rows' incumbent (and its ``pred_sha`` when logged) on each window <= F;
    - ``first_divergent`` / ``cause``: the first window where either the readings or the incumbent part, and why
      (``readings_cause``, or ``TIE_CAUSE`` when only the incumbent does). A runtime is not among the readings
      compared: it always differs between processes.
    ``identical_through_freeze`` is the certification: ``pred_sha`` identical on every window <= F and the same
    incumbent named on each (None when ``pred_sha`` is not logged)."""
    keys = _branch_keys(runs)
    freeze = frozen_windows(runs)
    by_job = {k: g for k, g in members.groupby("job")} if not members.empty else {}
    inc = {k: g.set_index("window") for k, g in wrows.groupby("job")} if not wrows.empty else {}
    branches: dict = {}
    for job, (bkey, arm) in keys.items():
        branches.setdefault(bkey, {})[arm] = job
    rows = []
    for bkey, arms in sorted(branches.items(), key=lambda kv: str(kv[0])):
        # the budget-matched arm tunes with other sizes from window 0: it shares no prefix with the others
        frozen = sorted(a for a in arms if arms[a] in freeze and a not in C.MATCHED_ARMS)
        pairs = [(ONLINE, a) for a in frozen if ONLINE in arms]
        pairs += [(a, b) for i, a in enumerate(frozen) for b in frozen[i + 1:]]
        for a, b in pairs:
            ja, jb = arms[a], arms[b]
            F = freeze.get(jb) if a == ONLINE else min(freeze[ja], freeze[jb])
            ma, mb = by_job.get(ja), by_job.get(jb)
            row = {"algo": bkey[0], "fold": bkey[1], "seed_key": bkey[2], "model_seed": bkey[3], "arm_a": a,
                   "arm_b": b, "freeze": F}
            if ma is None or mb is None:
                row.update(n_windows=0, sha_status="no member log", readings_cause="no member log")
                rows.append(row)
                continue
            logged = ma["pred_sha"].notna().all() and mb["pred_sha"].notna().all()
            rd, jr = _compare_prefix(ma, mb, F, READING_FIELDS)
            first_r = min((w for w, v in rd.items() if not v), default=None)
            row.update(n_windows=len(rd), readings_identical_windows=int(sum(rd.values())),
                       readings_first_divergent=first_r,
                       readings_cause=_cause(jr, first_r) if first_r is not None else "")
            if logged:
                sd, _ = _compare_prefix(ma, mb, F, ("pred_sha",))
                first_s = min((w for w, v in sd.items() if not v), default=None)
                row.update(sha_status="logged", sha_identical_windows=int(sum(sd.values())),
                           sha_first_divergent=first_s, identical_through_freeze=first_s is None and len(sd) > 0)
            else:
                row.update(sha_status=NOT_LOGGED)
            ia, ib = inc.get(ja), inc.get(jb)
            if ia is not None and ib is not None:
                cols = ["incumbent_id"] + (["pred_sha"] if logged and "pred_sha" in ia else [])
                ws = [w for w in range(1, F + 1) if w in ia.index and w in ib.index]
                same = {w: all(report._none(ia.at[w, c]) == report._none(ib.at[w, c]) for c in cols) for w in ws}
                row.update(incumbent_identical_windows=int(sum(same.values())),
                           incumbent_first_divergent=min((w for w, v in same.items() if not v), default=None))
            fi = row.get("incumbent_first_divergent")
            if logged and fi is not None:
                row["identical_through_freeze"] = False  # the same predictions, another decision
            if first_r is not None and (fi is None or first_r <= fi):
                row.update(first_divergent=first_r, cause=row["readings_cause"])
            elif fi is not None:
                row.update(first_divergent=fi, cause=TIE_CAUSE)
            rows.append(row)
    return pd.DataFrame(rows, columns=PREFIX_COLS)


def prefix_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per pair of arms and stream group (and ``all``), and per algorithm: how many branch pairs share their
    readings (and, where logged, their predictions) through the freeze window, and the share of windows."""
    cols = ["arm_a", "arm_b", "level", "unit", "n_pairs", "n_readings_identical", "share_readings_identical",
            "share_windows_readings_identical", "n_readings_and_incumbents_identical", "share_readings_and_incumbents_identical",
            "median_first_divergent", "n_sha_logged", "n_sha_identical", "causes"]
    if table.empty:
        return pd.DataFrame(columns=cols)
    t = table[table["n_windows"].fillna(0) > 0].copy()
    t["group"] = [C.stream_group(f) for f in t["fold"]]
    rows = []
    for (a, b), g in t.groupby(["arm_a", "arm_b"]):
        units = [("group", grp, g if grp == R.ALL else g[g["group"] == grp]) for grp in (*C.STREAM_GROUPS, R.ALL)]
        units += [("algo", al, ga) for al, ga in g.groupby("algo")]
        for level, unit, sub in units:
            if sub.empty:
                continue
            ident = sub["readings_first_divergent"].isna()
            both = sub["first_divergent"].isna()
            logged = sub["sha_status"] == "logged"
            causes = sub.loc[~both, "cause"].value_counts().to_dict()
            rows.append({"arm_a": a, "arm_b": b, "level": level, "unit": unit, "n_pairs": len(sub),
                         "n_readings_identical": int(ident.sum()), "share_readings_identical": float(ident.mean()),
                         "share_windows_readings_identical": float(sub["readings_identical_windows"].sum()
                                                                   / sub["n_windows"].sum()),
                         "n_readings_and_incumbents_identical": int(both.sum()),
                         "share_readings_and_incumbents_identical": float(both.mean()),
                         "median_first_divergent": float(pd.to_numeric(sub.loc[~both, "first_divergent"]).median())
                         if (~both).any() else np.nan,
                         "n_sha_logged": int(logged.sum()),
                         "n_sha_identical": int((sub["identical_through_freeze"] == True).sum()),  # noqa: E712
                         "causes": "; ".join(f"{k}: {v}" for k, v in sorted(causes.items()))})
    return pd.DataFrame(rows, columns=cols)


def identical_prefix_pairs(table: pd.DataFrame) -> dict:
    """arm -> {(algo, fold, seed key)} whose branch shares the online arm's prefix through the freeze: by
    ``pred_sha`` where logged, else by the logged readings and the incumbents."""
    out: dict = {}
    if table.empty:
        return out
    for row in table[table["arm_a"] == ONLINE].to_dict("records"):
        if row.get("sha_status") == "logged":
            ok = bool(row.get("identical_through_freeze"))
        else:
            ok = report._none(row.get("first_divergent")) is None and row.get("n_windows", 0) > 0
        if ok:
            out.setdefault(row["arm_b"], set()).add((row["algo"], row["fold"], row["seed_key"]))
    return out


# --- E7: survivors ---------------------------------------------------------------------------------------

#: Why an eligible member on t has (or lacks) a reading on t + 1, and when that was settled: at the decision
#: after t, which uses nothing of t + 1, or on t + 1 itself.
CATEGORIES: dict[str, str] = {
    "reading": "has a reading on t+1",
    "replaced": "decision after t",
    "challenger_not_admitted": "decision after t",
    "frozen_out": "decision after t",
    "single_class_next": "on t+1",
    "error_next": "on t+1",
    "other_no_reading": "on t+1",
}


def survivor_rows(members: pd.DataFrame, runs: pd.DataFrame, wrows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per eligible member-window (tuned arms): the member scored on window t with a usable reading there
    (feasible, defined silhouette, an ARI: the candidates a pick by labels after t ranks), in a job whose log has
    window t + 1, and its ``category`` (``CATEGORIES``). Also, per (job, t), the pick by labels the report's
    replay makes for t + 1 (``report._a_star_lag``): ``pool`` the eligible members active on t + 1, whether the
    best ARI on t is lost to the pool (``best_lost``, ``best_gap``), and whether the pick, ties going to the
    incumbent named after t then the smallest id, has no reading on t + 1 (``dropped``: the window leaves the
    replay's mean)."""
    m = members
    nxt = m[["job", "member_id", "window", "is_test", "ari_win", "has_error", "infeasible"]].copy()
    nxt["window"] = nxt["window"] - 1
    nxt = nxt.rename(columns={"is_test": "next_is_test", "ari_win": "next_ari", "has_error": "next_error",
                              "infeasible": "next_infeasible"})
    last = m.groupby("job")["window"].max()
    e = m[m["usable"] & m["ari_win"].notna()]
    e = e[e["window"] < e["job"].map(last)]
    e = e.merge(nxt, on=["job", "member_id", "window"], how="left")
    classes = m.groupby(["job", "window"])["n_true_classes"].first()
    any_ari = m["ari_win"].notna().groupby([m["job"], m["window"]]).any()
    nk = pd.MultiIndex.from_arrays([e["job"], e["window"] + 1])
    ncls = classes.reindex(nk).to_numpy(dtype=float)
    nany = any_ari.reindex(nk).eq(True).to_numpy()
    single = np.where(np.isnan(ncls), ~nany, ncls < C.MIN_TRUE_CLASSES)
    freeze = frozen_windows(runs)
    at_freeze = (e["job"].map(freeze) == e["window"]).to_numpy()
    in_next = e["next_is_test"].notna().to_numpy()
    pool = in_next & e["next_is_test"].eq(False).to_numpy()
    reading = pool & e["next_ari"].notna().to_numpy()
    test = e["is_test"].to_numpy(dtype=bool)
    cat = np.select(
        [reading, pool & single, pool & e["next_error"].eq(True).to_numpy(), pool,
         ~in_next & at_freeze, ~in_next & test, ~in_next],
        ["reading", "single_class_next", "error_next", "other_no_reading", "frozen_out", "challenger_not_admitted",
         "replaced"], "other_no_reading")
    e = e.assign(category=cat, in_pool=pool, decided=[CATEGORIES[c] for c in cat],
                 reading_infeasible_next=reading & e["next_infeasible"].eq(True).to_numpy())
    tail = _run_field(runs, "tail_from", C.FREEZE_AFTER_WINDOW)
    e["tail"] = (e["window"] + 1) >= e["job"].map(lambda j: int(tail.get(j, C.FREEZE_AFTER_WINDOW)))
    cols = ["job", "algo", "fold", "seed", "arm", "window", "member_id", "is_test", "ari_win", "next_ari", "category",
            "decided", "in_pool", "reading_infeasible_next", "tail"]
    e = e[cols].reset_index(drop=True)
    return e, replay_windows(e, wrows)


def replay_windows(e: pd.DataFrame, wrows: pd.DataFrame) -> pd.DataFrame:
    """Per (job, t) of ``survivor_rows``: eligible and pool sizes, ``best_lost`` / ``best_gap`` (the best ARI on t
    against the pool's best) and ``dropped`` (the replay's pick, ties to the incumbent named after t then the
    smallest id, has no ARI on t + 1; an empty pool has no pick and counts as dropped)."""
    cols = ["job", "algo", "fold", "arm", "window", "tail", "n_eligible", "n_pool", "best_lost", "best_gap", "dropped"]
    if e.empty:
        return pd.DataFrame(columns=cols)
    keys = ["job", "window"]
    w = e.groupby(keys, sort=False).agg(algo=("algo", "first"), fold=("fold", "first"), arm=("arm", "first"),
                                        tail=("tail", "first"), n_eligible=("member_id", "size"),
                                        n_pool=("in_pool", "sum"), best=("ari_win", "max")).reset_index()
    p = e[e["in_pool"]]
    top = p.groupby(keys)["ari_win"].max().rename("top")
    tied = p.merge(top.reset_index(), on=keys)
    tied = tied[tied["ari_win"] == tied["top"]]
    if not wrows.empty and "incumbent_id" in wrows:
        inc = wrows[["job", "window", "incumbent_id"]].drop_duplicates(keys, keep="last")
        tied = tied.merge(inc, on=keys, how="left")
        tied["is_inc"] = pd.to_numeric(tied["incumbent_id"], errors="coerce") == tied["member_id"]
    else:
        tied["is_inc"] = False
    pick = tied.sort_values(keys + ["is_inc", "member_id"], ascending=[True, True, False, True]).groupby(keys).head(1)
    pick = pick.set_index(keys)
    idx = pd.MultiIndex.from_frame(w[keys])
    w["top"] = top.reindex(idx).to_numpy()
    w["best_lost"] = w["top"].isna() | (w["best"] > w["top"] + _EPS)
    w["best_gap"] = w["best"] - w["top"]
    w["dropped"] = pick["next_ari"].reindex(idx).isna().to_numpy()
    return w[cols]


def survivor_summary(e: pd.DataFrame, wins: pd.DataFrame) -> pd.DataFrame:
    """Shares of the eligible member-windows per category, pooled over member-windows, per (arm, region) and
    stream group (protocol algorithms) or algorithm (every algorithm, variants flagged); and per window, the
    share where the best ARI on t is lost to the pool and where the replay's pick has no reading on t + 1."""
    rows = []
    if e.empty:
        return pd.DataFrame(rows)
    e = e.assign(group=[C.stream_group(f) for f in e["fold"]])
    wins = wins.assign(group=[C.stream_group(f) for f in wins["fold"]])
    for region in ("all", "tail"):
        er = e if region == "all" else e[e["tail"]]
        wr = wins if region == "all" else wins[wins["tail"]]
        for arm in sorted(set(er["arm"])):
            ea, wa = er[er["arm"] == arm], wr[wr["arm"] == arm]
            proto = ~ea["algo"].isin(list(C.variant_algos()))
            wproto = ~wa["algo"].isin(list(C.variant_algos()))
            units = [("group", g, ea[proto] if g == R.ALL else ea[proto & (ea["group"] == g)],
                      wa[wproto] if g == R.ALL else wa[wproto & (wa["group"] == g)]) for g in (*C.STREAM_GROUPS, R.ALL)]
            units += [("algo", a, ga, wa[wa["algo"] == a]) for a, ga in ea.groupby("algo")]
            for level, unit, sub, ws in units:
                if sub.empty:
                    continue
                n = len(sub)
                row = {"region": region, "arm": arm, "level": level, "unit": unit,
                       "variant": unit in C.variant_algos(), "n_windows": len(ws), "n_eligible": n}
                for c in CATEGORIES:
                    row[f"share_{c}"] = float((sub["category"] == c).sum() / n)
                row["share_decided_after_t"] = float(sub["category"].map(CATEGORIES).eq("decision after t").sum() / n)
                row["share_on_next_window"] = float(sub["category"].map(CATEGORIES).eq("on t+1").sum() / n)
                row["share_reading_infeasible_next"] = float(sub["reading_infeasible_next"].sum() / n)
                row["share_windows_best_lost"] = float(ws["best_lost"].mean()) if len(ws) else np.nan
                row["mean_best_gap"] = float(ws["best_gap"].mean()) if len(ws) else np.nan
                row["share_windows_pick_without_reading"] = float(ws["dropped"].mean()) if len(ws) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


# --- E7: lagged validity -----------------------------------------------------------------------------------

RHO_COLS = ["n", "rho", "x_ptp", "y_ptp", "defined"]


def group_spearman(df: pd.DataFrame, keys: list, x: str, y: str, min_spread: float = MIN_SPREAD,
                   min_n: int = MIN_MEMBERS, spread_on: str | None = None) -> pd.DataFrame:
    """Spearman's rho (Pearson on average ranks, as ``scipy.stats.spearmanr``) of ``x`` against ``y`` within each
    group of ``keys``, over the rows with both. ``defined`` needs ``min_n`` rows, ``x`` not constant and the
    range of ``y`` (or of ``spread_on``) at least ``min_spread`` and above 0; elsewhere ``rho`` is NaN."""
    need = [x, y] + ([spread_on] if spread_on and spread_on not in (x, y) else [])
    d = df.dropna(subset=need)
    if d.empty:
        return pd.DataFrame(columns=list(keys) + RHO_COLS)
    g = d.groupby(keys, sort=False)
    t = d[keys].copy()
    t["rx"] = g[x].rank(method="average").to_numpy()
    t["ry"] = g[y].rank(method="average").to_numpy()
    gt = t.groupby(keys, sort=False)
    t["dx"] = t["rx"] - gt["rx"].transform("mean")
    t["dy"] = t["ry"] - gt["ry"].transform("mean")
    t["xy"], t["xx"], t["yy"] = t["dx"] * t["dy"], t["dx"] ** 2, t["dy"] ** 2
    t["x"], t["y"] = d[x].to_numpy(dtype=float), d[spread_on or y].to_numpy(dtype=float)
    a = t.groupby(keys, sort=False).agg(n=("xy", "size"), sxy=("xy", "sum"), sxx=("xx", "sum"), syy=("yy", "sum"),
                                        xmin=("x", "min"), xmax=("x", "max"), ymin=("y", "min"),
                                        ymax=("y", "max")).reset_index()
    a["x_ptp"], a["y_ptp"] = a["xmax"] - a["xmin"], a["ymax"] - a["ymin"]
    a["defined"] = (a["n"] >= min_n) & (a["x_ptp"] > 0) & (a["y_ptp"] > 0) & (a["y_ptp"] >= min_spread - _EPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        a["rho"] = np.where(a["defined"], a["sxy"] / np.sqrt(a["sxx"] * a["syy"]), np.nan)
    return a[list(keys) + RHO_COLS]


def _bin_label(v: float, bins, kind: str) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    for i, (lo, hi) in enumerate(bins):
        if kind == "share":
            if (i == 0 and v == lo) or (i > 0 and lo < v <= hi):
                return f"{lo:g}" if i == 0 else f"({lo:g}, {hi:g}]"
        elif v >= lo and (hi is None or v <= hi):
            return f"{lo}" if hi == lo else (f">={lo}" if hi is None else f"{lo}-{hi}")
    return None


def period_label(t: int, bounds: list) -> str:
    """Where a pair (t, t + 1) lies against a fold's drift boundaries ``[(point, window, straddles)]``:
    ``across drift k`` when t + 1 is the window holding (or starting at) boundary k, or t is a window that
    holds it inside; else ``segment j``, j the boundaries at or before t (0: before the first drift)."""
    for k, (_, tw, straddles) in enumerate(bounds, start=1):
        if t + 1 == tw or (straddles and t == tw):
            return f"across drift {k}"
    return f"segment {sum(tw <= t for _, tw, _ in bounds)}"


def validity_pairs(members: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per member scored on window t with a usable silhouette (``usable``, a score): its ``score`` and ARI on t,
    its ARI on t + 1 (``next_ari``, any row of the member there), its age ``t - born_window``, micro-clusters,
    sentinel share (``n_sentinel / h``), the region (``tail``: t + 1 >= the run's ``tail_from``) and, on a fold
    with drift points, the period of the pair (``period_label``)."""
    m = members[members["usable"] & members["score"].notna()]
    nxt = members[["job", "member_id", "window", "ari_win"]].copy()
    nxt["window"] = nxt["window"] - 1
    p = m.merge(nxt.rename(columns={"ari_win": "next_ari"}), on=["job", "member_id", "window"], how="left")
    h = _run_field(runs, "h", C.H)
    tail = _run_field(runs, "tail_from", C.FREEZE_AFTER_WINDOW)
    p["age"] = p["window"] - p["born_window"]
    p["sentinel_share"] = p["n_sentinel"] / p["job"].map(lambda j: float(h.get(j, C.H)))
    p["tail"] = (p["window"] + 1) >= p["job"].map(lambda j: int(tail.get(j, C.FREEZE_AFTER_WINDOW)))
    bounds = drift_boundaries(runs)
    p["period"] = [period_label(int(t), bounds[f]) if f in bounds else None for t, f in zip(p["window"], p["fold"])]
    return p


def window_rhos(p: pd.DataFrame, min_spread: float = MIN_SPREAD, min_n: int = MIN_MEMBERS,
                stratum: str | None = None) -> pd.DataFrame:
    """Per (job, t[, stratum]): ``rho_lag`` (score on t against ARI on t + 1, over the members with both, the
    spread gate on the ARIs of t + 1), ``rho_same`` (against the ARI on t, every member with it, gate on the ARIs
    of t: ``proxy.csv``'s per-window rho with a per-window gate), and the paired pair ``rho_lag_paired`` /
    ``rho_same_paired`` on the members with both ARIs, defined where both gates hold."""
    keys = ["job", "window"] + ([stratum] if stratum else [])
    q = p if stratum is None else p[p[stratum].notna()]
    lag = group_spearman(q, keys, "score", "next_ari", min_spread, min_n)
    same = group_spearman(q, keys, "score", "ari_win", min_spread, min_n)
    both = q.dropna(subset=["ari_win", "next_ari"])
    lp = group_spearman(both, keys, "score", "next_ari", min_spread, min_n)
    sp = group_spearman(both, keys, "score", "ari_win", min_spread, min_n)
    paired = lp.merge(sp, on=keys, suffixes=("_l", "_s"))
    ok = paired["defined_l"] & paired["defined_s"]
    paired = paired.assign(rho_lag_paired=np.where(ok, paired["rho_l"], np.nan),
                           rho_same_paired=np.where(ok, paired["rho_s"], np.nan))[keys + ["rho_lag_paired", "rho_same_paired"]]
    out = lag[keys + ["rho", "n"]].rename(columns={"rho": "rho_lag", "n": "n_lag"}).merge(
        same[keys + ["rho", "n"]].rename(columns={"rho": "rho_same", "n": "n_same"}), on=keys, how="outer")
    out = out.merge(paired, on=keys, how="outer")
    out["rho_diff_paired"] = out["rho_lag_paired"] - out["rho_same_paired"]
    meta = p.drop_duplicates("job").set_index("job")[["algo", "fold", "seed", "arm"]]
    out = out.join(meta, on="job")
    tails = p.drop_duplicates(["job", "window"]).set_index(["job", "window"])["tail"]
    out["tail"] = tails.reindex(pd.MultiIndex.from_frame(out[["job", "window"]])).to_numpy()
    return out


RHO_STATS = ("rho_lag", "rho_same", "rho_lag_paired", "rho_same_paired", "rho_diff_paired")


def rho_cells(rhos: pd.DataFrame, region: str, by=()) -> pd.DataFrame:
    """Per (arm, algo, fold[, by]) and statistic: the mean over each job's windows of the region, then over seeds;
    ``n_windows`` the defined windows behind it."""
    r = rhos if region == "all" else rhos[rhos["tail"].astype(bool)]
    keys = ["arm", "algo", "fold", *by]
    rows = []
    for stat in RHO_STATS:
        d = r.dropna(subset=[stat])
        if d.empty:
            continue
        per_job = d.groupby(keys + ["job"], dropna=False)[stat].agg(["mean", "size"]).reset_index()
        cell = per_job.groupby(keys, dropna=False).agg(value=("mean", "mean"), n_windows=("size", "sum")).reset_index()
        rows.append(cell.assign(statistic=stat, region=region))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=keys + ["value", "n_windows", "statistic", "region"])


def drift_boundaries(runs: pd.DataFrame) -> dict:
    """fold -> ``[(drift point, window holding it, straddles)]`` from its run rows' ``drift_points`` and ``h``;
    a boundary straddles when it falls strictly inside its window (``point % h != 0``)."""
    out = {}
    for row in runs.to_dict("records"):
        pts = row.get("drift_points")
        if not isinstance(pts, (list, tuple)) or not pts:
            continue
        h = int(report._num(row.get("h")) or C.H)
        bounds = [(int(p), int(p) // h, int(p) % h != 0) for p in pts]
        if out.setdefault(row["fold"], bounds) != bounds:
            raise ValueError(f"fold {row['fold']}: its run rows disagree on the drift points")
    return out


# --- E8: recovery -------------------------------------------------------------------------------------------

def _recovery(lag: dict, tw: int, nxt: int, ref, delta: float = RECOVERY_DELTA, run: int = report.RECOVERY_RUN):
    """``make_robustness_extra.recovery``: the first t in [tw, nxt - run] from which ``run`` windows in a row read
    at least ``ref - delta``, minus tw; None when none does before the next boundary or the stream's end."""
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        return None
    for t in range(tw, nxt - run + 1):
        if all(lag.get(t + k) is not None and lag[t + k] >= ref - delta - _RECOVERY_SLACK for k in range(run)):
            return t - tw
    return None


def recovery_rows(windows: pd.DataFrame, sweep_win: pd.DataFrame, bounds: dict, delta: float = RECOVERY_DELTA,
                  drop_straddling: bool = False) -> pd.DataFrame:
    """Per (algo, fold, arm, seed, boundary) of the folds in ``bounds``: ``make_robustness_extra.recovery_checks``,
    one row per seed. A segment runs from a boundary's window to the next one's (or the fold's last window + 1);
    ``level`` is its best static level: the maximum over the sweep's configurations of their mean lagged ARI over
    the segment's windows. ``rec_static`` / ``rec_pre``: the delay to recovery (0 = the window holding the change)
    against ``level`` and against the run's own lagged ARI on the window before the change (``pre``); None is
    censored (never recovered before the next change or the stream's end). ``shortfall``: the segment's mean lagged
    ARI minus ``level``. ``drop_straddling``: every window that holds a boundary strictly inside it is left out of
    every reading (the run's, the sweep's level, the segment means), so a delay is at least 1 there."""
    cols = ["algo", "fold", "arm", "seed", "boundary", "tw", "next", "straddles", "level", "pre", "rec_static",
            "rec_pre", "shortfall", "n_segment_windows"]
    win = windows[windows["fold"].isin(list(bounds)) & ~windows["algo"].isin(list(C.variant_algos()))]
    if win.empty:
        return pd.DataFrame(columns=cols)
    excluded = {f: {tw for _, tw, s in b if s} if drop_straddling else set() for f, b in bounds.items()}
    sw = sweep_win[sweep_win["fold"].isin(list(bounds))].copy()
    sw["lag_ari_win"] = pd.to_numeric(sw["lag_ari_win"], errors="coerce")
    sw["window"] = pd.to_numeric(sw["window"], errors="coerce")
    last = win.groupby("fold")["window"].max().to_dict()

    def segments(fold):
        b = bounds[fold]
        edges = [tw for _, tw, _ in b] + [int(last[fold]) + 1]
        return [(p, tw, edges[i + 1], s) for i, (p, tw, s) in enumerate(b)]

    level = {}
    for (algo, fold), g in sw.groupby(["algo", "fold"]):
        g = g[~g["window"].isin(excluded[fold])]
        for _, tw, nxt, _ in segments(fold):
            seg = g[(g["window"] >= tw) & (g["window"] < nxt)]
            level[(algo, fold, tw)] = float(seg.groupby("config_id")["lag_ari_win"].mean().max()) if len(seg) else np.nan
    rows = []
    for (algo, fold, arm, seed), g in win.groupby(["algo", "fold", "arm", "seed"], dropna=False):
        g = g.drop_duplicates("window", keep="last")
        lag = {int(w): (None if pd.isna(v) or int(w) in excluded[fold] else float(v))
               for w, v in zip(g["window"], pd.to_numeric(g["lag_ari_win"], errors="coerce"))}
        for p, tw, nxt, s in segments(fold):
            L = level.get((algo, fold, tw), np.nan)
            pre = lag.get(tw - 1)
            seg = [lag[t] for t in range(tw, nxt) if lag.get(t) is not None]
            rows.append({"algo": algo, "fold": fold, "arm": arm, "seed": report._seed_key(seed), "boundary": p,
                         "tw": tw, "next": nxt, "straddles": s, "level": L, "pre": pre,
                         "rec_static": _recovery(lag, tw, nxt, L, delta), "rec_pre": _recovery(lag, tw, nxt, pre, delta),
                         "shortfall": float(np.mean(seg)) - L if seg and not np.isnan(L) else np.nan,
                         "n_segment_windows": len(seg)})
    return pd.DataFrame(rows, columns=cols)


def _median_delay(values) -> float:
    return float(np.median([np.inf if v is None or (isinstance(v, float) and np.isnan(v)) else v for v in values]))


def recovery_events(rows: pd.DataFrame) -> pd.DataFrame:
    """Per event (algo, fold, arm, boundary): the seeds, each seed's delay, the seed median (a censored seed counts
    as never, as the script did; the event is censored when the median is), the seeds recovered, and the mean
    ``pre``, ``level`` and ``shortfall``."""
    out = []
    for (algo, fold, arm, p), g in rows.groupby(["algo", "fold", "arm", "boundary"], sort=True):
        r = {"algo": algo, "fold": fold, "arm": arm, "boundary": p, "tw": int(g["tw"].iat[0]),
             "straddles": bool(g["straddles"].iat[0]), "n_seeds": len(g), "level": float(g["level"].iat[0]),
             "pre": float(pd.to_numeric(g["pre"]).mean()), "shortfall": float(g["shortfall"].mean())}
        for ref in ("static", "pre"):
            vals = list(g[f"rec_{ref}"])
            med = _median_delay(vals)
            r[f"delays_{ref}"] = ",".join("censored" if v is None or v != v else str(int(v)) for v in vals)
            r[f"delay_{ref}"] = None if np.isinf(med) else med
            r[f"censored_{ref}"] = bool(np.isinf(med))
            r[f"n_recovered_{ref}"] = int(sum(v is not None and v == v for v in vals))
        out.append(r)
    return pd.DataFrame(out)


def recovery_summary(events: pd.DataFrame, ref: str) -> pd.DataFrame:
    """The script's per-arm summary: events, censored events, median and largest delay of the others."""
    out = []
    for arm, g in events.groupby("arm"):
        ok = g.loc[~g[f"censored_{ref}"], f"delay_{ref}"].astype(float)
        out.append({"reference": ref, "arm": arm, "events": len(g), "censored": int(g[f"censored_{ref}"].sum()),
                    "median_delay": float(np.median(ok)) if len(ok) else np.nan,
                    "max_delay": float(ok.max()) if len(ok) else np.nan})
    return pd.DataFrame(out)


#: ``robustness_extra.txt``'s recovery sections and what each holds.
RECOVERY_SECTIONS: dict[str, str] = {
    "hồi phục: trong 0,05 của mức best static từng đoạn, 3 cửa sổ liền (seed trung vị)": "static",
    "hồi phục: trong 0,05 của ARI trước drift của chính chế độ đó": "pre",
    "ARI trung bình của đoạn trừ mức best static của đoạn": "shortfall",
}


def recovery_check(text: str, summaries: dict, shortfall: pd.DataFrame) -> pd.DataFrame:
    """Every number of the printed recovery sections against the recomputed one, at the precision it was printed
    with (``robustness_check.matches``). ``summaries``: ref -> ``recovery_summary``; ``shortfall``: arm x fold."""
    cols = ["section", "item", "printed", "computed", "match"]
    secs = RC.sections(text)
    rows = []
    names = {"static": ["events", "censored", "median_delay", "max_delay"]}
    names["pre"] = names["static"]
    for title, ref in RECOVERY_SECTIONS.items():
        lines = secs.get(title)
        if lines is None:
            rows.append({"section": title, "item": "(section)", "printed": None, "computed": None, "match": False})
            continue
        if ref == "shortfall":
            folds = lines[0].split()[1:]
            for line in lines[2:]:
                tok = line.split()
                for fold, printed in zip(folds, tok[1:]):
                    v = shortfall.loc[tok[0], fold] if tok[0] in shortfall.index and fold in shortfall.columns else None
                    rows.append({"section": title, "item": f"{tok[0]} {fold}", "printed": printed, "computed": v,
                                 "match": RC.matches(printed, v)})
            continue
        summ = summaries[ref].set_index("arm")
        for line in lines[1:]:
            tok = line.split()
            for name, printed in zip(names[ref], tok[1:]):
                v = summ.loc[tok[0], name] if tok[0] in summ.index else None
                rows.append({"section": title, "item": f"{tok[0]} {name}", "printed": printed, "computed": v,
                             "match": RC.matches(printed, v)})
    return pd.DataFrame(rows, columns=cols)


# --- E6: equivalence ----------------------------------------------------------------------------------------

def run_seeds(runs: pd.DataFrame) -> pd.DataFrame:
    """``robustness.load_inputs``' ``seeds`` without its other inputs: per ok run of a protocol algorithm under
    ``robustness.ARMS``, the tail ARI, a fixed configuration's mean set beside every tuner seed under the split
    seed policy (``report.replicate_view``)."""
    r = runs[runs["arm"].isin(list(R.ARMS))]
    seeds = r[["algo", "fold", "arm", "seed"]].copy()
    seeds["value"] = pd.to_numeric(r[R.TAIL], errors="coerce") if R.TAIL in r else np.nan
    seeds = seeds.dropna(subset=["value"]).reset_index(drop=True)
    if not r.empty and report.is_split(r):
        seeds = report.replicate_view(seeds, split=True)[["algo", "fold", "arm", "seed", "value"]]
    return seeds


def tost_verdict(low: float, high: float, margin: float) -> str:
    if low is None or high is None or np.isnan(low) or np.isnan(high):
        return "undefined"
    if -margin < low and high < margin:
        return "equivalent"
    if low >= margin or high <= -margin:
        return "different beyond the margin"
    return "not shown equivalent"


def equivalence_table(seeds: pd.DataFrame, margin: float = MARGIN, n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED, comparisons=("default", "frozen_prefix")) -> pd.DataFrame:
    """The main paired contrasts ``confstream - default`` and ``confstream - frozen_prefix`` (``contrast``, i.e.
    ``robustness.unit_statistics``), per stream group, pooled over the algorithms and per algorithm, units fold and
    block, with their 90% interval (the same bootstrap draws, percentiles 5 and 95) beside the published 95% one.
    Two one-sided tests at 5% each against ``margin``: ``equivalent`` when the whole 90% interval lies in
    (-margin, +margin)."""
    rows = []
    for other in comparisons:
        for unit in R.UNITS:
            with interval(CI90):
                st90 = contrast(seeds, other, unit, n_boot, seed)
            st95 = contrast(seeds, other, unit, n_boot, seed)
            st = st90.rename(columns={"low": "low90", "high": "high90"}).merge(
                st95[["contrast", "unit", "group", "algo", "low", "high"]].rename(columns={"low": "low95", "high": "high95"}),
                on=["contrast", "unit", "group", "algo"])
            rows.append(st)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    out["margin"] = margin
    out["verdict"] = [tost_verdict(lo, hi, margin) for lo, hi in zip(out["low90"], out["high90"])]
    return out[["contrast", "unit", "group", "algo", "point", "low90", "high90", "low95", "high95", "margin", "verdict",
                "n_pairs", "n_units", "n_used", "n_boot"]]


def pair_equivalence(summary: pd.DataFrame, seeds: pd.DataFrame, margin: float = MARGIN, other: str = "frozen_prefix",
                     metric: str = report.TAIL) -> pd.DataFrame:
    """Per (algo, fold): ``gap = confstream - other`` on the seed means, the seed-noise band
    ``max(sd_confstream, sd_other)`` of ``analyze.win_tie_loss`` (sample sd over seeds, a missing sd read as 0;
    ``within_band`` is its tie), and an equivalence reading of the pair itself: a paired t interval at 90% over
    the seeds both arms ran (differences seed by seed), ``equivalent`` when it lies in (-margin, +margin);
    undefined with fewer than two seeds or no spread among the differences (``status``). The band says whether
    the gap is distinguishable from seed noise; the interval whether it is shown to be small. They differ."""
    from scipy import stats

    sd = analyze.sd_column(metric)
    valid = analyze._complete(analyze._protocol(summary))
    a, b = analyze._cells(valid, ONLINE, metric).to_dict(), analyze._cells(valid, other, metric).to_dict()
    sa, sb = analyze._cells(valid, ONLINE, sd).to_dict(), analyze._cells(valid, other, sd).to_dict()
    def by_pair(arm):
        sub = seeds[seeds["arm"] == arm]
        return {k: g.groupby("seed")["value"].mean() for k, g in sub.groupby(["fold", "algo"])}

    va, vb = by_pair(ONLINE), by_pair(other)
    empty = pd.Series(dtype=float)
    rows = []
    for key in sorted(k for k in a if k in b):
        fold, algo = key
        gap = a[key] - b[key]
        band = max(sa.get(key, 0.0), sb.get(key, 0.0))
        d = (va.get(key, empty) - vb.get(key, empty)).dropna()
        n = len(d)
        lo = hi = np.nan
        status = "ok"
        if n < 2:
            status = "undefined: fewer than two paired seeds"
        elif float(d.std(ddof=1)) <= _EPS:
            status = "undefined: every seed gives the same difference"
        else:
            half = float(stats.t.ppf(0.95, n - 1)) * float(d.std(ddof=1)) / math.sqrt(n)
            lo, hi = float(d.mean()) - half, float(d.mean()) + half
        rows.append({"algo": algo, "fold": fold, "group": C.stream_group(fold), "contrast": f"{ONLINE} - {other}",
                     "gap": gap, "band": band, "within_band": abs(gap) <= band + _EPS,
                     "n_paired_seeds": n, "low90": lo, "high90": hi, "margin": margin,
                     "equivalent": (None if status != "ok" else bool(-margin < lo and hi < margin)), "status": status})
    return pd.DataFrame(rows)


# --- E10: figure --------------------------------------------------------------------------------------------

def figure_data(curves: dict, fold: str, algos=C.ALGOS, arms=FIGURE_ARMS) -> pd.DataFrame | None:
    """``make_dat.py``'s F3 data: per window, the mean over ``algos`` of each arm's seed-mean lagged ARI
    (``report.curves``)."""
    c = curves.get(fold)
    if c is None or c.empty:
        return None
    c = c[c["algo"].isin(list(algos))]
    p = c.pivot_table(index="window", columns="arm", values="lag_ari_win", aggfunc="mean")
    p = p.reindex(columns=list(arms))
    p.columns.name = None
    return p


def write_dat(p: pd.DataFrame, path: Path) -> None:
    p.to_csv(path, sep=" ", float_format="%.4f", na_rep="nan")


def compare_dat(new_path: Path, old_path: Path) -> list[dict]:
    """Per column of two ``.dat`` files: rows in both, only in one, and the largest absolute difference; and the
    count of identical text lines."""
    if not old_path.exists():
        return [{"file": old_path.name, "column": "(file)", "note": f"{old_path} not found"}]
    new = pd.read_csv(new_path, sep=r"\s+").set_index("window")
    old = pd.read_csv(old_path, sep=r"\s+").set_index("window")
    both = new.index.intersection(old.index)
    rows = []
    for col in dict.fromkeys([*new.columns, *old.columns]):
        if col not in new or col not in old:
            rows.append({"file": old_path.name, "column": col, "note": "column in one file only"})
            continue
        a, b = new.loc[both, col], old.loc[both, col]
        diff = (a - b).abs()
        nan_mismatch = int((a.isna() != b.isna()).sum())
        rows.append({"file": old_path.name, "column": col, "rows_both": len(both),
                     "rows_new_only": len(new.index.difference(old.index)),
                     "rows_published_only": len(old.index.difference(new.index)),
                     "max_abs_diff": float(diff.max()) if diff.notna().any() else np.nan,
                     "n_nan_mismatch": nan_mismatch, "note": ""})
    lines_new = new_path.read_text().splitlines()
    lines_old = old_path.read_text().splitlines()
    rows.append({"file": old_path.name, "column": "(text lines)", "rows_both": len(lines_old),
                 "max_abs_diff": np.nan,
                 "note": f"{sum(x == y for x, y in zip(lines_new, lines_old))} of {len(lines_old)} lines identical"
                         f" (regenerated file has {len(lines_new)})"})
    return rows


# --- E10: feasibility ---------------------------------------------------------------------------------------

def replay_feasibility(members: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """``confstream.apply_feasibility`` replayed on every member-window: the reference is the fastest member with a
    partition (``has_partition``), and a member's reason is the first of error, memory above ``max_memory_mb``,
    micro-clusters above ``max_micro_clusters``, runtime above ``runtime_factor`` times the reference that holds
    (the run row's caps). ``replayed`` beside the logged ``feasible_reason``; ``near_threshold`` where the logged
    runtimes, rounded to 6 decimals, sit within that rounding of the runtime cap. Then the counterfactual without
    the window's fastest member (ties to the smallest id): ``cf_reference`` is the next fastest with a partition
    (none: the runtime cap is not applied) and ``flips`` the remaining members whose runtime shedding changes."""
    m = members.copy()
    # a run without a runtime cap (``--no-runtime-gate``) reads None: it never sheds by runtime
    factor = _run_field(runs, "runtime_factor", None)
    mem = _run_field(runs, "max_memory_mb", C.MAX_MEMORY_MB)
    micro = _run_field(runs, "max_micro_clusters", C.MAX_MICRO_CLUSTERS)
    m["factor"] = pd.to_numeric(m["job"].map(factor), errors="coerce")
    m["mem_cap"] = pd.to_numeric(m["job"].map(mem), errors="coerce").fillna(C.MAX_MEMORY_MB)
    m["micro_cap"] = pd.to_numeric(m["job"].map(micro), errors="coerce").fillna(C.MAX_MICRO_CLUSTERS)
    part = m[m["has_partition"]].sort_values(["job", "window", "runtime_s", "member_id"])
    first = part.groupby(["job", "window"]).head(1).set_index(["job", "window"])
    second = part.groupby(["job", "window"]).nth(1).set_index(["job", "window"])
    idx = pd.MultiIndex.from_frame(m[["job", "window"]])
    m["reference"] = first["runtime_s"].reindex(idx).to_numpy()
    m["fastest_id"] = first["member_id"].reindex(idx).to_numpy()
    m["cf_reference"] = second["runtime_s"].reindex(idx).to_numpy()
    other_cap = np.select([m["has_error"], m["memory_mb"] > m["mem_cap"], m["n_micro_clusters"] > m["micro_cap"]],
                          ["error", "memory", "micro_clusters"], "")
    over = m["runtime_s"].to_numpy() > m["factor"].to_numpy() * m["reference"].to_numpy()
    over_cf = m["runtime_s"].to_numpy() > m["factor"].to_numpy() * m["cf_reference"].to_numpy()
    m["replayed"] = np.where(other_cap != "", other_cap, np.where(over, "runtime", "feasible"))
    m["logged"] = m["feasible_reason"].fillna("feasible")
    m.loc[m["logged"].eq("feasible") & m["infeasible"], "logged"] = "infeasible, no reason"
    m["near_threshold"] = (np.abs(m["runtime_s"] - m["factor"] * m["reference"]) <= 5e-7 * (1 + m["factor"])).to_numpy()
    shed_now = (other_cap == "") & over
    shed_cf = (other_cap == "") & over_cf
    removed = m["member_id"].to_numpy() == m["fastest_id"].to_numpy()
    m["is_fastest"] = removed
    m["flips"] = ~removed & (shed_now != shed_cf)
    return m


def window_feasibility(m: pd.DataFrame) -> pd.DataFrame:
    """Per (job, window) of the replay: members shed by the runtime gate (logged and replayed), flips without the
    fastest member, whether a reference is left after removing it, and the fastest member's silhouette and ARI
    against the window's medians: the silhouette over the members with one (a usable score, or ``silhouette_raw``
    where logged: an infeasible member's score is overwritten), the ARI over the members with one."""
    sil = m["silhouette_raw"].where(m["silhouette_raw"].notna(), m["score"].where(m["usable"]))
    m = m.assign(sil=sil, shed_logged=m["logged"] == "runtime", shed_replayed=m["replayed"] == "runtime")
    g = m.groupby(["job", "window"], sort=False)
    out = g.agg(algo=("algo", "first"), fold=("fold", "first"), arm=("arm", "first"), n_members=("member_id", "size"),
                n_partition=("has_partition", "sum"), n_shed_logged=("shed_logged", "sum"),
                n_shed_replayed=("shed_replayed", "sum"), n_flips=("flips", "sum"),
                med_sil=("sil", "median"), n_sil=("sil", "count"), med_ari=("ari_win", "median"),
                n_ari=("ari_win", "count")).reset_index()
    fast = m[m["is_fastest"]].set_index(["job", "window"])
    idx = pd.MultiIndex.from_frame(out[["job", "window"]])
    out["fastest_sil"] = fast["sil"].reindex(idx).to_numpy()
    out["fastest_ari"] = fast["ari_win"].reindex(idx).to_numpy()
    out["no_reference_left"] = out["n_partition"] == 1
    out["fastest_below_median_sil"] = np.where((out["n_sil"] >= MIN_MEMBERS) & out["fastest_sil"].notna(),
                                               out["fastest_sil"] < out["med_sil"], np.nan)
    out["fastest_below_median_ari"] = np.where((out["n_ari"] >= MIN_MEMBERS) & out["fastest_ari"].notna(),
                                               out["fastest_ari"] < out["med_ari"], np.nan)
    return out


def feasibility_summary(m: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, algo), g in w.groupby(["arm", "algo"]):
        mm = m[(m["arm"] == arm) & (m["algo"] == algo)]
        shed = g["n_shed_logged"] > 0
        agree = mm["replayed"] == mm["logged"].where(mm["logged"] != "infeasible, no reason", "?")
        mism = mm[~agree]
        rows.append({"arm": arm, "algo": algo, "variant": algo in C.variant_algos(), "n_windows": len(g),
                     "n_member_windows": len(mm),
                     "replay_agreement": float(agree.mean()) if len(mm) else np.nan,
                     "n_mismatch": len(mism), "n_mismatch_near_threshold": int(mism["near_threshold"].sum()),
                     "n_shed_runtime_logged": int((mm["logged"] == "runtime").sum()),
                     "n_shed_runtime_replayed": int((mm["replayed"] == "runtime").sum()),
                     "share_windows_with_runtime_shed": float(shed.mean()),
                     "share_windows_with_flip": float((g["n_flips"] > 0).mean()),
                     "share_shed_windows_with_flip": float((g.loc[shed, "n_flips"] > 0).mean()) if shed.any() else np.nan,
                     "flipped_over_shed": float(g["n_flips"].sum() / max(1, g["n_shed_replayed"].sum()))
                     if g["n_shed_replayed"].sum() else np.nan,
                     "n_windows_no_reference_left": int(g["no_reference_left"].sum()),
                     "share_fastest_below_median_sil": float(pd.to_numeric(g["fastest_below_median_sil"]).mean()),
                     "n_windows_sil_read": int(pd.to_numeric(g["fastest_below_median_sil"]).notna().sum()),
                     "share_fastest_below_median_ari": float(pd.to_numeric(g["fastest_below_median_ari"]).mean()),
                     "share_shed_windows_fastest_below_median_sil": float(pd.to_numeric(
                         g.loc[shed, "fastest_below_median_sil"]).mean()) if shed.any() else np.nan})
    return pd.DataFrame(rows)


# --- E9: substitute -------------------------------------------------------------------------------------------

def substitute_frame(df: pd.DataFrame, variant: str, reference: str) -> pd.DataFrame:
    """``df`` (rows with ``algo``) with ``reference``'s rows replaced by ``variant``'s, renamed ``reference``."""
    out = df[df["algo"] != reference].copy()
    out.loc[out["algo"] == variant, "algo"] = reference
    return out.reset_index(drop=True)


def selection_readings(runs: pd.DataFrame, band: float = S.BAND, k: int = S.TOP_K) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``selection``'s same-data reading on ``runs``: ``(groups, rows)``, its per-fold rows (``fold_selection``)
    and their aggregation (``selection.aggregate``)."""
    values = S.seed_values(runs, S.ARMS)
    tables = S.variant_tables(values, None, S.TARGETS, None, None)
    sel, tgts, _ = tables["same_data"]
    if sel is None:
        return pd.DataFrame(), pd.DataFrame(columns=S.ROW_COLS)
    train = {arm: S.wide(values, arm) for arm in (S.SELECTING_ARM, *S.TARGETS)}
    rows = S.fold_selection("same_data", sel, tgts, train, float(band), k)
    return S.aggregate(rows), rows


# --- the subcommands ------------------------------------------------------------------------------------------

def cmd_post_freeze(out_dir: Path, starts=None, n_boot: int = N_BOOT, boot_seed: int = BOOT_SEED,
                    workers: int | None = None) -> list[Path]:
    runs = _ok_runs(out_dir)
    windows = R.current_windows(out_dir, runs)
    specs = start_specs(runs, starts)
    members = load_members(out_dir, workers=workers)
    wrows = _window_rows(out_dir)
    prefix = prefix_table(members, wrows, runs) if not members.empty else pd.DataFrame(columns=PREFIX_COLS)
    subsets = {f"prefix shared with {ONLINE} through the freeze ({arm})": (arm, keep)
               for arm, keep in identical_prefix_pairs(prefix).items()}
    con, per_run = post_freeze_contrasts(windows, runs, specs, n_boot, boot_seed, subsets)
    # reproduction: the first start's per-run means against the run rows, and the contrast against the published one
    check = []
    if len(per_run):
        first = per_run[per_run["start"] == specs[0][0]]
        rr = runs[["algo", "fold", "arm", "seed"]].assign(published=pd.to_numeric(runs.get(R.TAIL), errors="coerce"))
        j = first.merge(rr, on=["algo", "fold", "arm", "seed"], how="inner")
        check.append({"check": f"per-run mean over {specs[0][0]} vs run rows' {R.TAIL}", "n": len(j),
                      "max_abs_diff": float((j["value"] - j["published"]).abs().max()) if len(j) else np.nan})
    pub = Path(out_dir) / C.REPORT_DIR.name / "robustness" / "unit_statistics.csv"
    if pub.exists() and len(con):
        p = pd.read_csv(pub)
        p = p[(p["reference"] == R.REFERENCE_CONSTRUCTIONS[0]) & (p["metric"] == "ari")]
        p = p.assign(contrast=p["statistic"].map({"confstream - default": "confstream - default",
                                                  "confstream - frozen_prefix": "confstream - frozen_prefix",
                                                  "confstream - reference": "confstream - oracle_static"}))
        mine = con[(con["subset"] == "all runs") & (con["start"] == specs[0][0])]
        j = mine.merge(p, on=["contrast", "unit", "group", "algo"], suffixes=("", "_pub"))
        check.append({"check": f"contrasts on {specs[0][0]} vs report/robustness/unit_statistics.csv (point, low, high)",
                      "n": len(j), "max_abs_diff": float(np.nanmax(np.abs(np.concatenate([
                          (j[c] - j[f"{c}_pub"]).to_numpy(dtype=float) for c in ("point", "low", "high")]))))
                      if len(j) else np.nan})
    check = pd.DataFrame(check, columns=["check", "n", "max_abs_diff"])
    sha = lag_sha_divergence(wrows, runs)
    tuned = tuned_arms(runs)
    mt = members[members["arm"].isin(tuned)] if not members.empty else members
    audit = member_timing_checks(mt, wrows) if not mt.empty else []
    events = load_lifecycle(out_dir)
    if events is not None and not events.empty:
        audit += lifecycle_checks(events, mt, wrows)
    audit = pd.DataFrame(audit, columns=AUDIT_COLS)
    tables = {"contrasts": con, "per_run_means": per_run, "reproduction": check, "timing_audit": audit,
              "lag_pred_sha": sha}
    md = "\n".join(_header(out_dir, "post-freeze")) + "\n" + _post_freeze_md(
        con, specs, check, audit, events, sha, members, subsets, n_boot, boot_seed)
    return _save(out_dir, "post-freeze", tables, md)


def _wide_contrasts(con: pd.DataFrame, specs, subset: str = "all runs") -> pd.DataFrame:
    c = con[(con["subset"] == subset) & (con["algo"] == "pooled") & (con["n_pairs"] > 0)]
    if c.empty:
        return c
    c = c.assign(ci=[f"{p:.3f} [{lo:.3f}, {hi:.3f}]" if pd.notna(p) else "—"
                     for p, lo, hi in zip(c["point"], c["low"], c["high"])])
    w = c.pivot_table(index=["contrast", "unit", "group"], columns="start", values="ci", aggfunc="first")
    order = [lbl for lbl, _ in specs] + [f"{lbl} minus {specs[0][0]}" for lbl, _ in specs[1:]]
    w = w.reindex(columns=[o for o in order if o in w.columns]).reset_index()
    w.columns.name = None
    w["group"] = pd.Categorical(w["group"], [*C.STREAM_GROUPS, R.ALL], ordered=True)
    return w.sort_values(["contrast", "unit", "group"]).astype({"group": str})


def _post_freeze_md(con, specs, check, audit, events, sha, members, subsets, n_boot, boot_seed) -> str:
    labels = [lbl for lbl, _ in specs]
    lines = [
        "## What is read",
        "",
        f"Per run, the mean lagged ARI (`lag_ari_win`: the incumbent named after window t-1, scored on t) over the "
        f"windows from each start ({', '.join(labels)}; `--start` overrides), from `windows.jsonl` "
        "(`robustness.current_windows`: protocol algorithms, the reference's reruns of the current configuration). "
        "Contrasts `confstream - X` for every other arm X in the dir, by `robustness.unit_statistics` itself: seeds "
        "paired by seed number within a pair (the reference's reruns drawn apart), pairs averaged within each unit, "
        "units averaged per stream group; unit `fold` (pairs equally weighted, the published mean) and `block` "
        f"(source blocks equally weighted). 95% percentile intervals, {n_boot} replicates, seed {boot_seed}. A later "
        f"start's difference from `{labels[0]}` is the contrast of the per-run differences, bootstrapped on the same "
        "draws.",
        "",
        "Which window first reflects the freeze. The freeze is decided after window t0 (`freeze_after_window`); the "
        "lagged reading of window t0 is the incumbent named after t0 - 1, still the common tuner's. Window t0 + 1 is "
        "read by the incumbent named after t0, which is the member the frozen arm keeps: when the branches share "
        "their prefix it is the same model with the same history in every arm, so t0 + 1 is identical across them "
        "by construction. **t >= t0 + 2 is the strictly post-freeze region**: its first window is read by an "
        "incumbent named after a decision made after the freeze. Whether the branches of this dir do share their "
        "prefix is read by `prefix-identity` (below: the subset where they do).",
        "",
        "## Pooled contrasts (point [95% interval])",
        "",
        _table(_wide_contrasts(con, specs)),
    ]
    for name in subsets:
        lines += [f"### Subset: {name}", "",
                  "The same contrast over the runs whose branch, through the freeze window, logged the same member "
                  "readings and incumbents as the online arm (or the same `pred_sha` and incumbents where logged): a "
                  "supplementary reading on a selected subset, not a random one (which runs keep the prefix depends "
                  "on ties and measured runtimes, unevenly across algorithms: `prefix-identity`); its coverage is "
                  "`n_pairs` in `contrasts.csv`.", "",
                  _table(_wide_contrasts(con, specs, name))]
    lines += ["## Reproduction", "", _table(check, digits=6),
              "## Check of the t0 + 1 claim (`lag_pred_sha`)", ""]
    if sha.empty:
        lines += ["`windows.jsonl` has no `lag_pred_sha` (it predates it): the claim that t0 + 1 is identical across "
                  "branches is not checked on predictions here.", ""]
    else:
        n = len(sha)
        ok = int(sha["identical_through_freeze_plus_1"].fillna(False).sum())
        lines += [f"{ok} of {n} branch runs read the same `lag_pred_sha` as the online arm through t0 + 1. Per run "
                  "(`lag_pred_sha.csv`), the first window where they differ:", "",
                  _table(sha, ["algo", "fold", "seed_key", "arm", "freeze", "first_divergent_window",
                               "offset_from_freeze", "identical_through_freeze_plus_1"])]
    lines += ["## Challenger timing", ""]
    if events is None:
        lines += ["`lifecycle/` not found in this dir (runs made before the lifecycle log): births, challenges and "
                  "decisions are not recorded, so the lifecycle checks are not made. What the member logs and the "
                  "window rows record is checked instead:", ""]
    else:
        lines += ["From `lifecycle/` (births, challenges, decisions, the freeze), the member logs and the window rows:", ""]
    if members.empty:
        lines += ["No member log: nothing below the window rows is checked.", ""]
    lines += [_table(audit), "What no log here records, so is not checked:", ""]
    lines += [f"- {u}" for u in UNCHECKABLE] + [""]
    return "\n".join(lines) + "\n"


def cmd_prefix_identity(out_dir: Path, workers: int | None = None) -> list[Path]:
    runs = _ok_runs(out_dir, protocol=False)
    members = load_members(out_dir, workers=workers)
    wrows = _window_rows(out_dir)
    table = prefix_table(members, wrows, runs) if not members.empty else pd.DataFrame(columns=PREFIX_COLS)
    summ = prefix_summary(table)
    sha = lag_sha_divergence(wrows, runs)
    logged = (table["sha_status"] == "logged").any() if len(table) else False
    lines = _header(out_dir, "prefix-identity") + [
        "## What is read", "",
        "Every pair of branches of one tuner run (same algorithm, fold, tuner seed and model seed: the online arm "
        "and each arm that froze, and the frozen arms with each other), on every window up to and including the "
        "freeze window F, which all of them ran as one tuner. Member logs (`members/`) and window rows "
        "(`windows.jsonl`).", "",
        "- `pred_sha` (a hash of a member's predictions on the window): identical on a window when both branches "
        "hold the same members with the same hash. A pair identical on every window <= F is certified to share "
        "its prefix.",
        "- Logged readings (score, ARI, micro-clusters, labels, feasibility and its reason, error, test status, "
        "birth window, parameters): a weaker test. Readings that differ show the branches parted; readings that "
        "agree do not show identical predictions. `readings_cause` says what parted them first: a member in one "
        "branch only, the runtime gate (a member shed in one branch only: measured runtime differs between "
        "processes), a configuration, another cap, or the readings themselves.",
        "- The incumbent named on each window <= F (and its `pred_sha` where logged). With equal readings the "
        "incumbents can still part where members tie on the score and the tie goes another way in each branch "
        "(`out/full` broke ties by measured runtime: `--tie-break runtime`); runtimes are not among the readings "
        "compared, since they always differ between processes.",
        "- `first_divergent` / `cause` (per pair, `prefix.csv`): the first window where the readings or the "
        "incumbent part, and why. Only a pair with neither through F is read as sharing its prefix (the subset "
        "`post-freeze` reports apart).", "",
        "## Certification by `pred_sha`", "",
    ]
    if not logged:
        lines += [f"**{NOT_LOGGED}**: the member rows of this dir carry no `pred_sha` (runs made before it was "
                  "logged), so no pair can be certified to share its prefix. The readings comparison below is what "
                  "the logs allow.", ""]
    else:
        t = table[table["sha_status"] == "logged"]
        lines += [f"{int(t['identical_through_freeze'].fillna(False).sum())} of {len(t)} pairs logged `pred_sha` "
                  "identical on every window through the freeze.", ""]
    lines += ["## Readings through the freeze, per stream group and algorithm", "",
              "Pairs with the online arm (every pair, the frozen arms with each other too, in `summary.csv`):", "",
              _table(summ[summ["arm_a"] == ONLINE] if len(summ) else summ,
                     ["arm_a", "arm_b", "level", "unit", "n_pairs", "n_readings_identical",
                      "share_readings_identical", "share_windows_readings_identical",
                      "n_readings_and_incumbents_identical", "share_readings_and_incumbents_identical",
                      "median_first_divergent", "n_sha_logged", "n_sha_identical", "causes"]),
              "Per pair: `prefix.csv`.", ""]
    if not sha.empty:
        lines += ["## `lag_pred_sha` after the freeze", "",
                  "Per branch run, the first window whose lagged incumbent's predictions differ from the online "
                  "arm's (`lag_pred_sha.csv`):", "",
                  _table(sha, ["algo", "fold", "seed_key", "arm", "freeze", "first_divergent_window",
                               "offset_from_freeze", "identical_through_freeze_plus_1"])]
    return _save(out_dir, "prefix-identity", {"prefix": table, "summary": summ, "lag_pred_sha": sha},
                 "\n".join(lines) + "\n")


def cmd_survivors(out_dir: Path, workers: int | None = None) -> list[Path]:
    runs = _ok_runs(out_dir, protocol=False)
    tuned = tuned_arms(runs)
    members = load_members(out_dir, arms=tuned, workers=workers) if tuned else pd.DataFrame(columns=MEMBER_COLUMNS)
    lines = _header(out_dir, "survivors")
    if members.empty:
        lines += ["No member log of a tuned arm (`members/` missing or empty): nothing to read.", ""]
        return _save(out_dir, "survivors", {"summary": pd.DataFrame()}, "\n".join(lines) + "\n")
    wrows = _window_rows(out_dir, ("incumbent_id",))
    e, wins = survivor_rows(members, runs, wrows)
    summ = survivor_summary(e, wins)
    cats = pd.DataFrame([{"category": k, "settled": v} for k, v in CATEGORIES.items()])
    show = ["arm", "level", "unit", "n_windows", "n_eligible", "share_reading", "share_replaced",
            "share_challenger_not_admitted", "share_frozen_out", "share_single_class_next", "share_error_next",
            "share_other_no_reading", "share_decided_after_t", "share_on_next_window", "share_reading_infeasible_next",
            "share_windows_best_lost", "share_windows_pick_without_reading"]
    lines += [
        "## What is read", "",
        "The report's replay of a pick by labels (`report.member_readings`' `lag_pick`, read as `a_star_lag` and "
        "`R_proxy_lag`) ranks, for window t+1, the members **active on t+1** that had a usable reading on t "
        "(feasible, a defined silhouette, an ARI), picks the best ARI on t (ties to the incumbent named after t, "
        "then the smallest id), and reads that member's ARI on t+1; a pick with no ARI on t+1 drops the window from "
        "the replay's mean. So it picks among the survivors of the decision after t, not among every member that "
        "had a reading on t.", "",
        "Here, from the member logs of every tuned arm, per window t whose job logged t+1: the **eligible** members "
        "(scored on t, active or on test, with a usable reading and an ARI), and why each one has or lacks a reading "
        "on t+1:", "",
        _table(cats),
        "- `replaced`: active on t, the victim of a challenger accepted at the decision after t; "
        "`challenger_not_admitted`: on test on t and rejected (or discarded); `frozen_out`: dropped by the freeze "
        "after t. These are settled at the decision after t, from readings of t and earlier.",
        "- `single_class_next` (t+1 holds fewer than two true classes, so no ARI), `error_next` (the member raised "
        "on t+1), `other_no_reading`: the member is in the pool on t+1 but has no ARI there. These depend on t+1.",
        "- A member in the pool that is infeasible on t+1 still has an ARI there and the replay reads it "
        "(`share_reading_infeasible_next`, part of `share_reading`).", "",
        "Shares are over eligible member-windows, pooled. Per window: `share_windows_best_lost` is how often the "
        "best ARI on t is not in the pool (the replay could not pick it), `share_windows_pick_without_reading` how "
        "often the replay's pick has no reading on t+1. Region `tail`: t+1 >= the run's `tail_from`, the windows "
        "the published `a_star_lag` reads. Group rows pool the protocol algorithms; algorithm rows include the "
        "implementation variant (`variant`).", "",
    ]
    for region in ("tail", "all"):
        s = summ[summ["region"] == region]
        lines += [f"## Region `{region}`", "", _table(s, show)]
    return _save(out_dir, "survivors", {"summary": summ, "windows": wins, "categories": cats}, "\n".join(lines) + "\n")


def cmd_lagged_validity(out_dir: Path, min_spread: float = MIN_SPREAD, min_members: int = MIN_MEMBERS,
                        n_boot: int = N_BOOT, boot_seed: int = BOOT_SEED, workers: int | None = None) -> list[Path]:
    runs = _ok_runs(out_dir, protocol=False)
    tuned = tuned_arms(runs)
    members = load_members(out_dir, arms=tuned, workers=workers) if tuned else pd.DataFrame(columns=MEMBER_COLUMNS)
    lines = _header(out_dir, "lagged-validity")
    if members.empty:
        lines += ["No member log of a tuned arm: nothing to read.", ""]
        return _save(out_dir, "lagged-validity", {"groups": pd.DataFrame()}, "\n".join(lines) + "\n")
    p = validity_pairs(members, runs)
    p["age_bin"] = [_bin_label(v, AGE_BINS, "int") for v in p["age"]]
    p["micro_bin"] = [_bin_label(v, MICRO_BINS, "int") for v in p["n_micro_clusters"]]
    p["sentinel_bin"] = [_bin_label(v, SENTINEL_BINS, "share") for v in p["sentinel_share"]]
    proto = ~p["algo"].isin(list(C.variant_algos()))
    rhos = window_rhos(p, min_spread, min_members)
    groups, cells_all = [], []
    for region in ("tail", "all"):
        cells = rho_cells(rhos[~rhos["algo"].isin(list(C.variant_algos()))], region)
        cells_all.append(cells)
        for (arm, stat), c in cells.groupby(["arm", "statistic"]):
            for unit in R.UNITS:
                b = unit_bootstrap(c, "value", unit, n_boot, boot_seed)
                groups.append(b.assign(arm=arm, statistic=stat, region=region, stratifier="none", stratum="all"))
    strata = []
    primary = ONLINE if ONLINE in tuned else tuned[0]
    for name, col in (("model age", "age_bin"), ("micro-clusters", "micro_bin"), ("sentinel share", "sentinel_bin"),
                      ("drift period", "period")):
        q = p[proto & (p["arm"] == primary) & p[col].notna()]
        if q.empty:
            continue
        r = window_rhos(q, min_spread, min_members, stratum=col)
        cells = rho_cells(r, "tail", by=(col,))
        for (stat, level), c in cells.groupby(["statistic", col]):
            if stat not in ("rho_lag", "rho_same"):
                continue
            b = unit_bootstrap(c, "value", "fold", n_boot, boot_seed)
            b = b[b["algo"] == "pooled"]
            strata.append(b.assign(arm=primary, statistic=stat, region="tail", stratifier=name, stratum=level,
                                   n_windows=int(c["n_windows"].sum())))
    gcols = ["arm", "region", "stratifier", "stratum", "statistic", "unit", "group", "algo", "point", "low", "high",
             "n_cells", "n_units", "n_used"]
    groups = pd.concat(groups, ignore_index=True)[gcols] if groups else pd.DataFrame(columns=gcols)
    strata = pd.concat(strata, ignore_index=True) if strata else pd.DataFrame(columns=gcols + ["n_windows"])
    cov = rhos.assign(group=[C.stream_group(f) for f in rhos["fold"]]).groupby(["arm", "group"]).agg(
        n_windows=("window", "size"), lag_defined=("rho_lag", "count"), same_defined=("rho_same", "count"),
        paired_defined=("rho_lag_paired", "count")).reset_index()

    def pivot(df, by):
        d = df[(df["algo"] == "pooled") & df["point"].notna()].copy()
        d["ci"] = [f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if pd.notna(v) else "—" for v, lo, hi in zip(d["point"], d["low"], d["high"])]
        w = d.pivot_table(index=by, columns="statistic", values="ci", aggfunc="first").reset_index()
        w.columns.name = None
        return w

    main_tail = groups[(groups["region"] == "tail")]
    lines += [
        "## What is read", "",
        "From the member logs of every tuned arm: per window t, the members scored on t with a usable silhouette "
        "(feasible, defined: the members the tuner ranks), their silhouette on t, their ARI on t and, where the "
        "member has a row on t+1, their ARI on t+1.", "",
        f"- `rho_lag`: Spearman's rho between the silhouette on t and the ARI on t+1, over the members with both. "
        f"Read on a window only with at least {min_members} members (`--min-members`), a silhouette that is not "
        f"constant, and ARIs on t+1 that spread by at least {min_spread} (`--min-spread`, a placeholder: the ceiling "
        "gate's minimum gain; ARI that barely separates the members makes a rank correlation read noise); else "
        "undefined and left out.",
        "- `rho_same`: the same against the ARI on t, every member with it, gate on the ARIs of t (`proxy.csv`'s "
        "per-window rho, gated per window rather than per cell).",
        "- `rho_lag_paired`, `rho_same_paired`, `rho_diff_paired`: both on the same windows and the same members "
        "(those with an ARI on t and on t+1, both gates holding), and their difference: the like-for-like "
        "comparison. The lagged set holds only members that survived the decision after t (see `survivors`).", "",
        "Per job the mean over its windows of the region (`tail`: t+1 >= `tail_from`), then the mean over seeds per "
        "(algorithm, fold), then the mean over the units of a stream group (unit `fold`, or `block`: each source "
        "block's cells averaged first), pooled over the protocol algorithms. 95% percentile intervals from "
        f"resampling the units within each group ({n_boot} replicates, seed {boot_seed}); seeds are not resampled.", "",
        "## Tail, pooled over algorithms", "",
        _table(pivot(main_tail, ["arm", "unit", "group"])),
        "## Coverage (windows with a defined reading)", "", _table(cov),
        f"## Strata (`{primary}`, tail, unit fold, pooled)", "",
        "Each stratum's rho is computed within the window over that stratum's members only (so it needs "
        f"{min_members} of them there); bins are placeholders: model age t - born_window (1 is a challenger's first "
        "scored window), micro-clusters, sentinel share n_sentinel / h. Drift period, on the folds with "
        "`drift_points`: `segment j` holds the pairs after j boundaries, `across drift k` the pairs whose t+1 holds "
        "boundary k or whose t holds it inside.", "",
        _table(pivot(strata, ["stratifier", "stratum", "group"]) if len(strata) else strata),
    ]
    return _save(out_dir, "lagged-validity", {"groups": groups, "strata": strata, "cells": pd.concat(cells_all,
                                                                                                    ignore_index=True),
                                              "coverage": cov}, "\n".join(lines) + "\n")


def cmd_recovery(out_dir: Path, drop_straddling: bool = False, delta: float = RECOVERY_DELTA,
                 published: Path | None = RC.PUBLISHED) -> list[Path]:
    runs = _ok_runs(out_dir)
    bounds = drift_boundaries(runs)
    lines = _header(out_dir, "recovery")
    if not bounds:
        lines += ["No run row has `drift_points`: no designed drift to recover from.", ""]
        return _save(out_dir, "recovery", {"events": pd.DataFrame()}, "\n".join(lines) + "\n")
    windows = R.current_windows(out_dir, runs)
    sweep_win, _ = R.load_sweep_windows(out_dir)

    def build(drop, d):
        rows = recovery_rows(windows, sweep_win, bounds, d, drop)
        ev = recovery_events(rows)
        summ = pd.concat([recovery_summary(ev, ref) for ref in ("static", "pre")], ignore_index=True)
        short = rows.groupby(["arm", "fold"])["shortfall"].mean().unstack()
        return rows, ev, summ, short

    # the published reading: straddling windows kept, delta 0.05
    rows0, ev0, summ0, short0 = build(False, RECOVERY_DELTA)
    check = pd.DataFrame(columns=["section", "item", "printed", "computed", "match"])
    if published is not None and Path(published).exists():
        check = recovery_check(Path(published).read_text(encoding="utf-8"),
                               {r: summ0[summ0["reference"] == r] for r in ("static", "pre")}, short0)
    rows, ev, summ, short = (rows0, ev0, summ0, short0) if not drop_straddling and delta == RECOVERY_DELTA else \
        build(drop_straddling, delta)
    tag = "dropped" if drop_straddling else "kept"
    rows, ev, summ = rows.assign(straddling=tag), ev.assign(straddling=tag), summ.assign(straddling=tag)
    sw = short.reset_index().melt(id_vars="arm", var_name="fold", value_name="shortfall").assign(straddling=tag)
    bnd = pd.DataFrame([{"fold": f, "boundary": p, "window": tw, "straddles": s} for f, b in bounds.items()
                        for p, tw, s in b])
    n_match = int(check["match"].sum()) if len(check) else 0
    lines += [
        "## What is read", "",
        "`result/otb_full/report/make_robustness_extra.py`'s `recovery_checks`, ported: the window rows of the "
        "current ok runs of the protocol algorithms on the folds whose run rows carry `drift_points`, and "
        "`sweep_windows.jsonl`. A boundary's window is `drift_point // h`; a segment runs from it to the next "
        "boundary's window (or the fold's last window + 1). The segment's **best static level** is the maximum over "
        "the sweep's configurations of their mean lagged ARI over the segment's windows (every sweep seed pooled). "
        f"Per run and boundary, the **delay** is the first window t of the segment from which {report.RECOVERY_RUN} "
        f"windows in a row read a lagged ARI of at least the reference - {delta}, minus the boundary's window (0 = the "
        "window holding the change recovers at once), against the best static level (`static`) and against the "
        "run's own lagged ARI on the window before the change (`pre`). **Censored**: no such run of windows before "
        "the next boundary or the stream's end. Per event (algorithm, fold, boundary) the seed median, a censored "
        "seed counting as never (the script's rule): the event is censored when the median is. **Shortfall**: the "
        "segment's mean lagged ARI minus its best static level.", "",
        f"Straddling windows: {tag}. With `--drop-straddling` every window that holds a boundary strictly inside it "
        "is left out of every reading (the run's lagged ARI, the sweep's level, the segment means); a delay is then "
        "at least 1 at such a boundary, counted from the same origin. A boundary that falls on a window's first "
        "sample has no straddling window.", "",
        _table(bnd),
        "## Reproduction of `robustness_extra.txt`", "",
        (f"Without the flag, at delta {RECOVERY_DELTA}: {n_match} of {len(check)} printed numbers of the sections "
         "\"hồi phục\" (both) and \"ARI trung bình của đoạn\" match at their printed precision"
         + (" (mismatches in `reproduction.csv`)." if n_match < len(check) else ".")) if len(check) else
        f"`{published}` not found: not checked.", "",
        "## Per arm", "", _table(summ, ["reference", "arm", "events", "censored", "median_delay", "max_delay"]),
        "## Shortfall (segment mean - best static level), mean over algorithms, seeds and boundaries", "",
        _table(short.reset_index()),
        "Per event: `events{0}.csv` (delays per seed, the seed median, censoring); per seed: `seeds{0}.csv`. A run "
        "with `--drop-straddling` writes its files with the suffix `_drop_straddling`, beside the others.".format(
            "_drop_straddling" if drop_straddling else ""), "",
    ]
    if drop_straddling:
        d = ev.merge(ev0, on=["algo", "fold", "arm", "boundary"], suffixes=("", "_kept"))
        lines += ["## Dropped against kept", "",
                  f"Events whose static-level delay changes: {int((d['delay_static'].astype(float).fillna(-1) != d['delay_static_kept'].astype(float).fillna(-1)).sum())} of {len(d)}; "
                  f"censored events {int(d['censored_static'].sum())} (dropped) against {int(d['censored_static_kept'].sum())} (kept).", ""]
    return _save(out_dir, "recovery", {"events": ev, "seeds": rows, "summary": summ, "shortfall": sw,
                                       "boundaries": bnd, "reproduction": check}, "\n".join(lines) + "\n",
                 suffix="_drop_straddling" if drop_straddling else "")


def cmd_equivalence(out_dir: Path, margin: float = MARGIN, n_boot: int = N_BOOT,
                    boot_seed: int = BOOT_SEED) -> list[Path]:
    runs = _ok_runs(out_dir)
    seeds = run_seeds(runs)
    eq = equivalence_table(seeds, margin, n_boot, boot_seed)
    summary = report.summary_table(out_dir)
    pairs = pd.concat([pair_equivalence(summary, seeds, margin, o) for o in ("frozen_prefix", "default")],
                      ignore_index=True)
    cross = (pairs.assign(equivalent=pairs["equivalent"].map({True: "equivalent", False: "not equivalent",
                                                              None: "undefined"}).fillna("undefined"))
             .groupby(["contrast", "within_band", "equivalent"]).size().rename("n_pairs").reset_index())
    wtl = analyze.win_tie_loss(summary, ONLINE)
    p_def = pairs[pairs["contrast"] == f"{ONLINE} - default"]
    ties_check = {"win_tie_loss ties (confstream vs default)": int(wtl["ties"].sum()),
                  "within_band here": int(p_def["within_band"].sum())}
    pooled = eq[eq["algo"] == "pooled"]
    lines = _header(out_dir, "equivalence") + [
        "## What is read", "",
        f"The run rows' tail ARI (`{R.TAIL}`) of the protocol algorithms, as `robustness` reads them. The contrasts "
        "`confstream - default` and `confstream - frozen_prefix` per stream group, pooled over the algorithms (and "
        "per algorithm in `equivalence.csv`), units fold and source block, by `robustness.unit_statistics` itself; "
        f"the 90% interval is the 5th and 95th percentile of the same bootstrap draws ({n_boot} replicates, seed "
        f"{boot_seed}) that give the published 95% one. Two one-sided tests at 5% each against the margin {margin} "
        "(`--margin`, a placeholder, not a finding): **equivalent** only when the whole 90% interval lies in "
        "(-margin, +margin); **different beyond the margin** when it lies wholly outside; otherwise not shown "
        "equivalent.", "",
        "## Pooled contrasts", "",
        _table(pooled, ["contrast", "unit", "group", "point", "low90", "high90", "low95", "high95", "verdict",
                        "n_pairs", "n_units"]),
        "## Per algorithm-stream pair", "",
        "Two different readings of `confstream - X` on one pair (`pairs.csv`). **Within the seed-noise band**: "
        "|gap| <= max(sd_confstream, sd_X), the sample sd over seeds of each arm's tail ARI (`analyze.win_tie_loss`'s "
        "tie): the gap is not told apart from seed noise. **Equivalent at the margin**: a paired t interval at 90% "
        "over the seeds both arms ran (seed by seed differences; n = 3 gives 2 degrees of freedom) lies in "
        "(-margin, +margin): the gap is shown to be small. A gap inside a wide band need not be equivalent, and an "
        "equivalent gap can lie outside a narrow band. Undefined with fewer than two paired seeds or no spread among "
        "the differences.", "",
        _table(cross),
        f"Check: {ties_check}.", "",
    ]
    return _save(out_dir, "equivalence", {"equivalence": eq, "pairs": pairs, "pairs_crosstab": cross},
                 "\n".join(lines) + "\n")


def cmd_figure(out_dir: Path, published_dir: Path = PAPER_DAT) -> list[Path]:
    rep = followup_dir(out_dir, "figure")
    curves = report.curves(out_dir)
    rows, written = [], []
    for fold in FIGURE_FOLDS:
        p = figure_data(curves, fold)
        if p is None:
            rows.append({"file": f"f3_{fold}.dat", "column": "(file)", "note": f"no window rows of {fold}"})
            continue
        path = rep / f"f3_{fold}.dat"
        write_dat(p, path)
        written.append(path)
        rows += compare_dat(path, Path(published_dir) / f"f3_{fold}.dat")
    cmp = pd.DataFrame(rows)
    lines = _header(out_dir, "figure") + [
        "## What is read", "",
        "`make_dat.py`'s Figure 1 data (F3): `report.curves` over `windows.jsonl` (per algorithm, arm and window, the "
        f"mean lagged ARI over seeds), then per window the mean over the eight protocol algorithms ({', '.join(C.ALGOS)}) "
        f"of each arm ({', '.join(FIGURE_ARMS)}), for {', '.join(FIGURE_FOLDS)}; written as `make_dat.py` wrote it "
        "(space-separated, 4 decimals) into this directory only. Compared with "
        f"`{published_dir}` (never overwritten): per column the rows in both and the largest absolute difference "
        "(the published values are rounded to 4 decimals: a difference up to 5e-5 is that rounding), and identical "
        "text lines.", "",
        _table(cmp, digits=6),
    ]
    return [*written, *_save(out_dir, "figure", {"comparison": cmp}, "\n".join(lines) + "\n")]


def cmd_feasibility(out_dir: Path, workers: int | None = None) -> list[Path]:
    runs = _ok_runs(out_dir, protocol=False)
    members = load_members(out_dir, workers=workers)
    lines = _header(out_dir, "feasibility")
    if members.empty:
        lines += ["No member log (`members/` missing or empty): the gate cannot be replayed.", ""]
        return _save(out_dir, "feasibility", {"summary": pd.DataFrame()}, "\n".join(lines) + "\n")
    m = replay_feasibility(members, runs)
    w = window_feasibility(m)
    summ = feasibility_summary(m, w)
    conf = m.groupby(["logged", "replayed"]).size().rename("n").reset_index()
    has_raw = members["silhouette_raw"].notna().any()
    has_part = members["partitioned"].notna().any()
    factor = sorted({float(v) for v in pd.to_numeric(runs.get("runtime_factor"), errors="coerce").dropna()})
    lines += [
        "## The rule", "",
        "`benchmark/tuning/confstream.apply_feasibility`, as the code reads today: the runtime reference is the "
        "fastest member of the window (active and on test) that produced a partition (no error and "
        "`partitioned`, else a defined silhouette); a member's reason is the first of `error`, memory above "
        "`max_memory_mb`, micro-clusters above `max_micro_clusters`, and runtime above `runtime_factor` "
        f"({', '.join(f'{f:g}' for f in factor) or C.RUNTIME_FACTOR}) times the reference; any member, with a "
        "partition or not, can be shed by the runtime cap; with no reference the runtime cap is not applied. "
        "Replayed on every member-window of the logs with the run row's caps"
        + ("" if has_part else " (these logs have no `partitioned`: the defined silhouette stands for it, as the "
                               "rule does then)") + ". Logged runtimes are rounded to 6 decimals: a disagreement "
        "within that rounding of the cap is counted apart (`n_mismatch_near_threshold`).", "",
        "## Replay against the logged reasons", "", _table(conf),
        "## Without each window's fastest member", "",
        "The counterfactual removes the fastest member with a partition (ties to the smallest id) and takes the next "
        "fastest as the reference (none left: the runtime cap is not applied): `share_windows_with_flip` is the share "
        "of windows where a remaining member's runtime shedding changes, `flipped_over_shed` the flipped members over "
        "the members the replay sheds by runtime. A fast but poor member: `share_fastest_below_median_sil` is how often "
        "the fastest member's silhouette is below the window's median silhouette, over the windows with at least "
        f"{MIN_MEMBERS} silhouettes"
        + ("" if has_raw else " (an infeasible member's score is overwritten in these logs and they have no "
                              "`silhouette_raw`: the median is over the feasible members' silhouettes)")
        + "; `share_fastest_below_median_ari` the same in ARI. Per tuned arm and algorithm; the variants are flagged. "
        "A one-member arm (default, the reference) has its own member as the reference and no counterfactual: its "
        "rows, with the replay's agreement, are in `summary.csv`; a frozen arm's windows after its freeze hold one "
        "member too (`n_windows_no_reference_left`).", "",
        _table(summ[summ["arm"].isin(tuned_arms(runs))], ["arm", "algo", "variant", "n_windows", "replay_agreement", "n_mismatch",
                      "n_mismatch_near_threshold", "n_shed_runtime_logged", "share_windows_with_runtime_shed",
                      "share_windows_with_flip", "share_shed_windows_with_flip", "flipped_over_shed",
                      "n_windows_no_reference_left", "share_fastest_below_median_sil",
                      "share_fastest_below_median_ari"]),
    ]
    return _save(out_dir, "feasibility", {"summary": summ, "confusion": conf, "windows": w},
                 "\n".join(lines) + "\n")


#: The numbers the paper cites for the published reading (``kendall.csv``, ``selection/groups.csv``).
PUBLISHED_SUBSTITUTE = {"tau(default, confstream), 18 folds": 0.073, "regret real": 0.139,
                        "regret param_drift": 0.008, "regret other_synth": 0.007}


def cmd_substitute(out_dir: Path, variant: str = "DenStreamPatchedFast", band: float = S.BAND,
                   top_k: int = S.TOP_K) -> list[Path]:
    reference = C.IMPL_VARIANTS.get(variant)
    lines = _header(out_dir, "substitute")
    runs = report.current_runs(out_dir)
    if reference is None or runs.empty or not (runs["algo"] == variant).any():
        lines += [f"`{variant}` is not an implementation variant of this dir's runs: nothing to substitute.", ""]
        return _save(out_dir, "substitute", {"readings": pd.DataFrame()}, "\n".join(lines) + "\n")
    summary = report.summary_table(out_dir)
    readings, per_fold = [], []
    for name, s, r in (("published (River's DenStream)", summary, runs),
                       (f"{variant} in {reference}'s place", substitute_frame(summary, variant, reference),
                        substitute_frame(runs, variant, reference))):
        k = analyze.kendall_matrix(s, ["default", ONLINE])
        kc = analyze.kendall_matrix(s, ["default", ONLINE], complete_only=True)
        groups, rows = selection_readings(r, band, top_k)
        g = groups[(groups["variant"] == "same_data") & (groups["target"] == ONLINE) & (groups["band"] == 0.0)
                   & (groups["selector"] == S.SELECTING_ARM) & (groups["level"] == "group")].set_index("unit")
        row = {"reading": name, "tau_kendall_csv": float(k.loc["default", ONLINE]),
               "tau_kendall_complete": float(kc.loc["default", ONLINE])}
        for grp in (*C.STREAM_GROUPS, R.ALL):
            if grp in g.index:
                row[f"tau_{grp}"] = g.loc[grp, "tau_fw"]
                row[f"n_tau_folds_{grp}"] = g.loc[grp, "n_tau_folds"]
                row[f"regret_{grp}_fw"] = g.loc[grp, "regret_exp_fw"]
                row[f"regret_{grp}_bw"] = g.loc[grp, "regret_exp_bw"]
        readings.append(row)
        f = rows[(rows["target"] == ONLINE) & (rows["band"] == 0.0) & (rows["selector"] == S.SELECTING_ARM)]
        per_fold.append(f[["fold", "group", "picked", "target_best_algos", "regret_exp", "tau"]].assign(reading=name))
    readings = pd.DataFrame(readings)
    per_fold = pd.concat(per_fold, ignore_index=True)
    pf = per_fold.pivot_table(index=["fold", "group"], columns="reading",
                              values=["picked", "target_best_algos", "regret_exp", "tau"], aggfunc="first")
    pf.columns = [f"{a} | {b}" for a, b in pf.columns]
    pf = pf.reset_index()
    names = list(readings["reading"])
    changed = int(sum(any(str(r[f"{c} | {names[0]}"]) != str(r[f"{c} | {names[1]}"])
                          for c in ("picked", "target_best_algos", "tau")) for r in pf.to_dict("records")))
    pub = pd.DataFrame([{"number": k, "cited": v,
                         "recomputed (published reading)": {"tau(default, confstream), 18 folds": readings.loc[0, "tau_kendall_csv"],
                                                            "regret real": readings.loc[0].get("regret_real_fw"),
                                                            "regret param_drift": readings.loc[0].get("regret_param_drift_fw"),
                                                            "regret other_synth": readings.loc[0].get("regret_other_synth_fw")}[k]}
                        for k, v in PUBLISHED_SUBSTITUTE.items()])
    pub["match"] = [RC.matches(f"{c:.3f}", v) for c, v in zip(pub["cited"], pub["recomputed (published reading)"])]
    show = ["reading", "tau_kendall_csv", "tau_kendall_complete", "tau_real", "n_tau_folds_all", "regret_real_fw",
            "regret_real_bw", "regret_param_drift_fw", "regret_other_synth_fw", "regret_all_fw"]
    lines += [
        "## What is read", "",
        f"`{variant}` (`_common.IMPL_VARIANTS`: {reference} with RIVER_BUGS.md BUG-7/BUG-8 fixed) ran every arm but "
        f"is never ranked. Here its run rows replace {reference}'s, under {reference}'s name, and the published "
        "definitions are applied unchanged: Kendall's tau-b between the default and the online rankings "
        "(`analyze.kendall_matrix`, mean over folds, a missing reading ranked last; `_complete` the variant that "
        "compares only the algorithms both arms read; per group `selection`'s tau on the scores), and the regret of "
        "the default pick under online tuning (`selection`'s same-data reading: per fold, the best tail ARI under "
        "`confstream` minus the default pick's, exact ties a uniform pick, fold-weighted `fw` and block-weighted "
        f"`bw` means). The oracle arm reads {variant}'s own reference; nothing else changes.", "",
        "## Side by side", "", _table(readings, show, digits=4),
        f"Folds whose default pick, online best or per-fold tau changes with the substitution: {changed} of "
        f"{len(pf)} (`per_fold.csv`).", "",
        "## The cited numbers, recomputed", "", _table(pub),
        "Per fold: `per_fold.csv`.", "",
    ]
    return _save(out_dir, "substitute", {"readings": readings, "per_fold": pf, "cited": pub}, "\n".join(lines) + "\n")


# --- CLI -----------------------------------------------------------------------------------------------------

def _ints(text: str) -> list[int]:
    return [int(t) for t in text.split(",") if t.strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m conf_pipeline.followup", description=__doc__.splitlines()[0])
    subs = ap.add_subparsers(dest="command", required=True)

    def sub(name, help_):
        sp = subs.add_parser(name, help=help_)
        sp.add_argument("--out-dir", type=Path, required=True, help="the out dir to read (writes only "
                                                                    "<out-dir>/report/followup/<subcommand>/)")
        return sp

    def boot(sp):
        sp.add_argument("--n-boot", type=int, default=N_BOOT)
        sp.add_argument("--boot-seed", type=int, default=BOOT_SEED)

    def workers(sp):
        sp.add_argument("--workers", type=int, default=None, help="processes reading the member logs")

    sp = sub("post-freeze", "E2: contrasts on t >= t0, t0+1, t0+2 and the challenger-timing audit")
    sp.add_argument("--start", type=_ints, default=None,
                    help="comma list of absolute start windows (default: each run's tail_from + 0, 1, 2)")
    boot(sp), workers(sp)
    workers(sub("prefix-identity", "E3: do the frozen branches share the online arm's prefix?"))
    workers(sub("survivors", "E7: which members the label replay can pick on t+1, and why the others drop"))
    sp = sub("lagged-validity", "E7: silhouette on t against ARI on t+1")
    sp.add_argument("--min-spread", type=float, default=MIN_SPREAD)
    sp.add_argument("--min-members", type=int, default=MIN_MEMBERS)
    boot(sp), workers(sp)
    sp = sub("recovery", "E8: recovery after drift, censored, with the reproduction check")
    sp.add_argument("--drop-straddling", action="store_true",
                    help="leave the window that holds a boundary out of every reading")
    sp.add_argument("--delta", type=float, default=RECOVERY_DELTA)
    sp.add_argument("--published", type=Path, default=RC.PUBLISHED)
    sp = sub("equivalence", "E6: two one-sided tests of the main contrasts")
    sp.add_argument("--margin", type=float, default=MARGIN, help="equivalence margin in tail ARI (a placeholder)")
    boot(sp)
    sp = sub("figure", "E10: Figure 1 data regenerated and compared")
    sp.add_argument("--published-dir", type=Path, default=PAPER_DAT)
    workers(sub("feasibility", "E10: the runtime gate replayed, and without the fastest member"))
    sp = sub("substitute", "E9: tau and regret with the patched DenStream in DenStream's place")
    sp.add_argument("--variant", default="DenStreamPatchedFast")
    sp.add_argument("--band", type=float, default=S.BAND)
    sp.add_argument("--top-k", type=int, default=S.TOP_K)
    a = ap.parse_args(argv)
    out = a.out_dir
    if a.command == "post-freeze":
        paths = cmd_post_freeze(out, a.start, a.n_boot, a.boot_seed, a.workers)
    elif a.command == "prefix-identity":
        paths = cmd_prefix_identity(out, a.workers)
    elif a.command == "survivors":
        paths = cmd_survivors(out, a.workers)
    elif a.command == "lagged-validity":
        paths = cmd_lagged_validity(out, a.min_spread, a.min_members, a.n_boot, a.boot_seed, a.workers)
    elif a.command == "recovery":
        paths = cmd_recovery(out, a.drop_straddling, a.delta, a.published)
    elif a.command == "equivalence":
        paths = cmd_equivalence(out, a.margin, a.n_boot, a.boot_seed)
    elif a.command == "figure":
        paths = cmd_figure(out, a.published_dir)
    elif a.command == "feasibility":
        paths = cmd_feasibility(out, a.workers)
    else:
        paths = cmd_substitute(out, a.variant, a.band, a.top_k)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
