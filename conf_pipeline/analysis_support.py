"""Readers the campaign analyses share: who a job is, which attempt counts, which windows two arms
are compared on, and how a difference is averaged up to a source block.

- **Job identity** (``job_identity``): (algo, fold, arm, seed, model_seed, config_id), the model seed
  included and a missing seed (None or NaN, a deterministic fixed configuration under the split seed
  policy) read as None, never cast. ``latest_attempts`` keeps the last attempt of every job (a retry
  after an error replaces it); a ``skipped`` row stands only for a job with no attempt at all. A
  failed attempt keeps its status (``error``, ``timeout``, ``oom``): it is counted, never scored.
- **Tail windows** (``tail_windows``): the lagged per-window readings of every ok job from its
  ``tail_from`` on, one row per (job, window).
- **Arm readings** (``arm_windows``): per (algo, fold, window), the readings of one arm, one
  replicate per online run (its tuner seed and its model seed) or, for a fixed configuration, the
  mean of its runs (``n_runs``: one for a deterministic algorithm, one per fixed model seed
  otherwise). A fixed run is never read as a tuner-seed replicate.
- **Paired contrasts** (``paired_contrast``): ``a - b`` per replicate on the windows where both
  arms have a reading (the common support), beside each arm's own support. Two online arms pair by
  tuner seed AND model seed (the same tuner seed on another model seed is another replicate, missing
  on the other side); a fixed arm's window mean stands beside every online replicate of the other
  arm (``broadcast``); two fixed arms give one replicate.
- **Aggregation** (``aggregate_contrast``): replicates -> (algo, fold) -> (algo, source block),
  the variants of a block averaged before the blocks -> per block, the mean over algorithms -> per
  stream group, the mean over blocks. Tuner seeds never add a block; uncertainty, when there are
  enough blocks, resamples blocks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import _common as C

#: Statuses of an attempt that ran and failed: counted in coverage, never turned into a score.
FAILED_STATUSES: tuple[str, ...] = ("error", "timeout", "oom")
IDENTITY: tuple[str, ...] = ("algo", "fold", "arm", "seed", "model_seed", "config_id")
#: The practical-effect threshold the campaign declared before running (``_common.MIN_CEILING_GAIN``).
THRESHOLD = C.MIN_CEILING_GAIN
#: Lagged per-window readings a contrast can be read on: ARI primary, NMI a sensitivity.
METRICS: dict[str, str] = {"ari": "lag_ari_win", "nmi": "lag_nmi_win"}
_EPS = 1e-12
#: Per lagged reading, the window-row field that fingerprints the points it was computed on
#: (``window.support_fingerprint`` of ``window.external_support``; ARI and NMI read the same points).
#: Runs written before 2026-11-04 have no such field.
SUPPORT_COLS: dict[str, str] = {"lag_ari_win": "lag_external_support_sha", "lag_nmi_win": "lag_external_support_sha"}
#: Where a window starts and ends in the stream (sample indices): the window's identity.
WINDOW_ID_COLS: tuple[str, ...] = ("start", "end")
#: The extra window-row fields ``tail_windows`` reads for point support.
SUPPORT_EXTRA: tuple[str, ...] = (*sorted(set(SUPPORT_COLS.values())), *WINDOW_ID_COLS)
#: A fixed arm's runs that read a window on different points: no one support stands for them.
MIXED = "<mixed>"
#: How the points of a common window compare between two replicates: ``identical`` (both logged the
#: fingerprint, the same, and the same window identity), ``differs`` (both logged it and it differs, a
#: fixed arm's runs disagree, or the window identities differ), ``undetermined`` (a side has no record).
POINT_SUPPORT_STATUSES: tuple[str, ...] = ("identical", "differs", "undetermined")


# --- identity and attempts ---------------------------------------------------------------------

def seed_key(v):
    """A seed as a key: None for a run without one (None, NaN, a missing value alike), else an int."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return int(v)


def _cid(v):
    return seed_key(v)


def job_identity(row: dict) -> tuple:
    """``IDENTITY`` of a run or window row. A row made before the seed split has no ``model_seed``:
    it is derived as that policy did (``run._model_seed_of``)."""
    from .run import _model_seed_of

    return (row.get("algo"), row.get("fold"), row.get("arm"), seed_key(row.get("seed")),
            seed_key(_model_seed_of(row)), _cid(row.get("config_id")))


def latest_attempts(rows: Iterable[dict]) -> dict[tuple, dict]:
    """Job identity -> the row that stands for it: the last attempt in file order, a ``skipped`` row
    only when the job has no attempt."""
    out: dict[tuple, dict] = {}
    for r in rows:
        if r.get("kind", "run") != "run":
            continue
        key = job_identity(r)
        if r.get("status") == "skipped" and key in out and out[key].get("status") != "skipped":
            continue
        out[key] = r
    return out


def run_table(out_dir: Path, arms=None, diagnostic: bool = False) -> pd.DataFrame:
    """One row per job of ``runs.jsonl`` (its latest attempt), with ``IDENTITY`` columns normalised
    (``seed``/``model_seed``/``config_id`` None when absent). ``arms`` keeps only those arms; the
    diagnostic arms only with ``diagnostic``. Implementation variants are dropped; a label variant
    (``_common.LABEL_VARIANTS``) is a configuration's own algorithm here, read under its own name."""
    rows = []
    for key, r in latest_attempts(C.read_jsonl(Path(out_dir) / "runs.jsonl")).items():
        if arms is not None and r.get("arm") not in arms:
            continue
        if not diagnostic and r.get("arm") in C.DIAGNOSTIC_ARMS:
            continue
        if r.get("algo") in C.IMPL_VARIANTS:
            continue
        rows.append({**r, **dict(zip(IDENTITY, key))})
    if not rows:
        return pd.DataFrame(columns=[*IDENTITY, "status"])
    return _keys_as_objects(pd.DataFrame(rows), [r for r in rows])


#: Identity columns that may hold None: kept as objects so a missing seed stays None (pandas would
#: otherwise read a column of ints and None as floats and NaN).
_NULLABLE_KEYS = ("seed", "model_seed", "config_id")


def _keys_as_objects(df: pd.DataFrame, records) -> pd.DataFrame:
    for k in _NULLABLE_KEYS:
        if k in df:
            df[k] = pd.Series([seed_key(r.get(k) if isinstance(r, dict) else r[IDENTITY.index(k)]) for r in records],
                              index=df.index, dtype=object)
    return df


# --- windows -----------------------------------------------------------------------------------

def tail_windows(out_dir: Path, runs: pd.DataFrame, metrics=tuple(METRICS.values()), extra=()) -> pd.DataFrame:
    """The lagged readings ``metrics`` of every ok job of ``runs`` (``run_table``) on its tail windows
    (window >= the run's ``tail_from``, else ``_common.FREEZE_AFTER_WINDOW``): one row per (job,
    window), ``IDENTITY`` + ``window`` + the metrics + ``extra`` (fields read as written, None when a
    row lacks one, e.g. ``SUPPORT_EXTRA``). The window rows of a job are read only when it has an ok
    latest attempt; a job repeated in the file keeps its last row per window."""
    extra = tuple(e for e in extra if e not in metrics)
    cols = [*IDENTITY, "window", *metrics, *extra]
    ok = runs[runs["status"] == "ok"] if len(runs) else runs
    start = {}
    for r in ok.to_dict("records"):
        key = (r["algo"], r["fold"], r["arm"], seed_key(r["seed"]), seed_key(r["model_seed"]), _cid(r["config_id"]))
        t = r.get("tail_from")
        start[key] = int(t) if t is not None and not (isinstance(t, float) and math.isnan(t)) else C.FREEZE_AFTER_WINDOW
    found: dict[tuple, tuple] = {}
    path = Path(out_dir) / C.WINDOWS_JSONL.name
    if path.exists() and start:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("kind", "window") != "window":
                    continue
                key = job_identity(r)
                if key not in start or int(r["window"]) < start[key]:
                    continue
                found[(*key, int(r["window"]))] = tuple(r.get(m) for m in (*metrics, *extra))
    rows = [(*k, *v) for k, v in found.items()]
    df = pd.DataFrame(rows, columns=cols)
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return _keys_as_objects(df, rows)


def fill_identical_windows(tw: pd.DataFrame, arm: str = "scaled_default", source: str = "default") -> tuple[pd.DataFrame, list[str]]:
    """``tw`` with ``arm``'s rows of an algorithm whose ``arm`` configuration IS ``source``'s (``run.py``
    skips it: ``scaled_default`` of an algorithm without an absolute-distance parameter) copied from
    ``source``, flagged ``filled_from``; and the algorithms copied. An algorithm that ran ``arm`` keeps
    its own rows; nothing is made when no algorithm ran ``arm`` at all."""
    from .selection import _repeats_other

    tw = tw.copy()
    if "filled_from" not in tw:
        tw["filled_from"] = None
    if tw.empty or not (tw["arm"] == arm).any():
        return tw, []
    have = set(tw.loc[tw["arm"] == arm, "algo"])
    src = tw[(tw["arm"] == source) & tw["algo"].map(lambda a: _repeats_other(arm, a) and a not in have)]
    if src.empty:
        return tw, []
    return (pd.concat([tw, src.assign(arm=arm, filled_from=source)], ignore_index=True),
            sorted(map(str, set(src["algo"]))))


def arm_windows(tw: pd.DataFrame, arm: str, metric: str) -> pd.DataFrame:
    """Per (algo, fold, replicate, model_seed, window): ``arm``'s reading of ``metric``. An online arm
    has one replicate per run, named by its tuner seed (``replicate``) and its model seed
    (``model_seed``, None for an algorithm that takes none); a fixed arm (``_common.FIXED_ARMS``) one
    replicate, ``"fixed"`` (model seed None), the mean of its runs on the window, with ``n_runs`` the
    runs that read it. Two online runs on one (tuner seed, model seed) and window raise ``ValueError``:
    a replicate is one run.

    ``support`` and ``window_id`` say which points the reading rests on (``SUPPORT_COLS``) and where
    the window lies (``WINDOW_ID_COLS``), when ``tw`` carries them (else None): a run's own, or for a
    fixed arm the one its runs share (``MIXED`` when they differ, None when one has no record)."""
    cols = ["algo", "fold", "replicate", "model_seed", "window", "value", "n_runs", "support", "window_id"]
    sub = tw[(tw["arm"] == arm)].dropna(subset=[metric])
    if sub.empty:
        return pd.DataFrame(columns=cols)
    sub = sub.assign(support=_column(sub, SUPPORT_COLS.get(metric)), window_id=_window_ids(sub))
    if arm in C.FIXED_ARMS:
        g = sub.groupby(["algo", "fold", "window"], sort=True)
        out = pd.DataFrame({"value": g[metric].mean(), "n_runs": g.size(), "support": g["support"].agg(_agree),
                            "window_id": g["window_id"].agg(_agree)}).reset_index()
        out["replicate"] = "fixed"
        out["model_seed"] = None
    else:
        out = sub[["algo", "fold", "seed", "model_seed", "window", metric, "support", "window_id"]].rename(
            columns={"seed": "replicate", metric: "value"})
        out["n_runs"] = 1
        key = list(zip(out["algo"], out["fold"], out["replicate"].map(seed_key), out["model_seed"].map(seed_key),
                       out["window"]))
        if len(set(key)) != len(key):
            raise ValueError(f"{arm}: two runs read one window under the same tuner seed and model seed")
    out = out[cols].reset_index(drop=True)
    out["model_seed"] = pd.Series([seed_key(v) for v in out["model_seed"]], index=out.index, dtype=object)
    for c in ("support", "window_id"):
        out[c] = pd.Series([_text(v) for v in out[c]], index=out.index, dtype=object)
    return out


def _text(v) -> str | None:
    """A record field as text, None for a missing one (None, NaN)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v)


def _column(df: pd.DataFrame, col: str | None) -> list:
    return [_text(v) for v in df[col]] if col is not None and col in df else [None] * len(df)


def _window_ids(df: pd.DataFrame) -> list:
    """``"start-end"`` of every row, None where a row lacks either."""
    if not all(c in df for c in WINDOW_ID_COLS):
        return [None] * len(df)
    out = []
    for s, e in zip(df["start"], df["end"]):
        s, e = _text(s), _text(e)
        out.append(None if s is None or e is None else f"{int(float(s))}-{int(float(e))}")
    return out


def _agree(values) -> str | None:
    """The record a fixed arm's runs share on a window: None when one lacks it, ``MIXED`` when they differ."""
    vals = [_text(v) for v in values]
    if not vals or any(v is None for v in vals):
        return None
    return vals[0] if len(set(vals)) == 1 else MIXED


def point_support_status(support_a, support_b, window_a, window_b) -> str:
    """``POINT_SUPPORT_STATUSES`` of one common window of two replicates. A count of points is never
    read as a mask: only the fingerprints of the positions are compared."""
    if MIXED in (window_a, window_b):
        return "differs"                       # a fixed arm's runs read other windows: no one window to compare
    if window_a is not None and window_b is not None and window_a != window_b:
        return "differs"
    if support_a is None or support_b is None or window_a is None or window_b is None:
        return "undetermined"
    if MIXED in (support_a, support_b) or support_a != support_b:
        return "differs"
    return "identical"


def own_support(tw: pd.DataFrame, arm: str, metric: str) -> pd.DataFrame:
    """Per (algo, fold): the windows ``arm`` has a reading of ``metric`` on (any replicate), and its runs."""
    sub = tw[tw["arm"] == arm]
    if sub.empty:
        return pd.DataFrame(columns=["algo", "fold", "n_windows", "n_runs"])
    g = sub.groupby(["algo", "fold"], sort=True)
    return pd.DataFrame({
        "n_windows": g.apply(lambda d: int(d.dropna(subset=[metric])["window"].nunique())),
        "n_runs": g.apply(lambda d: int(d[["seed", "model_seed"]].astype(str).drop_duplicates().shape[0])),
    }).reset_index()


# --- paired contrasts --------------------------------------------------------------------------

CONTRAST_COLS = ["algo", "fold", "group", "block", "replicate", "model_seed", "broadcast", "a_n_runs", "b_n_runs",
                 "n_windows_a", "n_windows_b", "n_common", "a_mean", "b_mean", "delta",
                 # the common windows by point support (``point_support_status``), and the difference on
                 # the windows whose points are verified identical only (a supplementary reading)
                 "n_point_identical", "n_point_differs", "n_point_undetermined", "n_window_id_differs",
                 "a_mean_point_identical", "b_mean_point_identical", "delta_point_identical", "note"]


def paired_contrast(tw: pd.DataFrame, arm_a: str, arm_b: str, metric: str) -> pd.DataFrame:
    """``arm_a - arm_b`` per (algo, fold, replicate) on the windows both read (``CONTRAST_COLS``).

    Replicates: two online arms pair by (tuner seed, model seed); a replicate one side lacks, a model
    seed it ran on that the other did not included, is reported, undefined. An online arm against a
    fixed one reads the fixed arm's window means beside every (tuner seed, model seed) of the online
    arm (``broadcast``); two fixed arms are one replicate. A pair without a common window keeps its
    row, undefined, with the reason; an (algo, fold) with one arm only is reported the same."""
    a, b = arm_windows(tw, arm_a, metric), arm_windows(tw, arm_b, metric)
    fixed_a, fixed_b = arm_a in C.FIXED_ARMS, arm_b in C.FIXED_ARMS
    rows = []
    cells = sorted(set(zip(a["algo"], a["fold"])) | set(zip(b["algo"], b["fold"])))
    for algo, fold in cells:
        ga = a[(a["algo"] == algo) & (a["fold"] == fold)]
        gb = b[(b["algo"] == algo) & (b["fold"] == fold)]
        group, block = _group_block(fold)
        base = {"algo": algo, "fold": fold, "group": group, "block": block}
        if ga.empty or gb.empty:
            rows.append({**base, "replicate": None, "note": f"no reading of {arm_a if ga.empty else arm_b}"})
            continue
        fixed = ("fixed", None)
        if fixed_a and fixed_b:
            pairs = [(fixed, fixed, fixed, False)]
        elif fixed_a:
            pairs = [(fixed, r, r, True) for r in _replicates(gb)]
        elif fixed_b:
            pairs = [(r, fixed, r, True) for r in _replicates(ga)]
        else:
            pairs = [(r, r, r, False) for r in sorted(set(_replicates(ga)) | set(_replicates(gb)), key=str)]
        for ra, rb, (name, model), bc in pairs:
            xa = _replicate_rows(ga, ra).set_index("window")
            xb = _replicate_rows(gb, rb).set_index("window")
            row = {**base, "replicate": name, "model_seed": model, "broadcast": bc,
                   "a_n_runs": int(xa["n_runs"].max()) if len(xa) else 0,
                   "b_n_runs": int(xb["n_runs"].max()) if len(xb) else 0,
                   "n_windows_a": int(len(xa)), "n_windows_b": int(len(xb))}
            common = xa.index.intersection(xb.index)
            row["n_common"] = int(len(common))
            if xa.empty or xb.empty:
                row["note"] = (f"replicate {name} (model seed {'none' if model is None else model}) missing under "
                               f"{arm_a if xa.empty else arm_b}")
            elif not len(common):
                row["note"] = "no common window"
            else:
                va, vb = xa.loc[common, "value"].astype(float), xb.loc[common, "value"].astype(float)
                row.update(a_mean=float(va.mean()), b_mean=float(vb.mean()), delta=float((va - vb).mean()))
                status = pd.Series([point_support_status(xa.at[w, "support"], xb.at[w, "support"],
                                                         xa.at[w, "window_id"], xb.at[w, "window_id"])
                                    for w in common], index=common)
                row.update({f"n_point_{s}": int((status == s).sum()) for s in POINT_SUPPORT_STATUSES})
                row["n_window_id_differs"] = int(sum(
                    xa.at[w, "window_id"] is not None and xb.at[w, "window_id"] is not None
                    and xa.at[w, "window_id"] != xb.at[w, "window_id"] for w in common))
                same = status.index[status == "identical"]
                if len(same):
                    pa, pb = va.loc[same], vb.loc[same]
                    row.update(a_mean_point_identical=float(pa.mean()), b_mean_point_identical=float(pb.mean()),
                               delta_point_identical=float((pa - pb).mean()))
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=CONTRAST_COLS)


def _replicates(g: pd.DataFrame) -> list[tuple]:
    """The (replicate, model seed) pairs of ``arm_windows`` rows, sorted."""
    return sorted(set(zip(g["replicate"], (seed_key(v) for v in g["model_seed"]))), key=str)


def _replicate_rows(g: pd.DataFrame, rep: tuple) -> pd.DataFrame:
    name, model = rep
    keep = [r == name and seed_key(m) == model for r, m in zip(g["replicate"], g["model_seed"])]
    return g[keep]


def _group_block(fold: str) -> tuple[str | None, str | None]:
    try:
        return C.stream_group(fold), C.source_block(fold)
    except ValueError:
        return None, None


def block_bootstrap(values: pd.Series, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the mean of ``values`` (one per source block) over resampled blocks;
    NaN with fewer than three blocks (two blocks resample to three distinct means)."""
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def aggregate_contrast(pairs: pd.DataFrame, threshold: float = THRESHOLD, n_boot: int = 2000,
                       boot_seed: int = 0) -> dict[str, pd.DataFrame]:
    """The levels of one contrast (``paired_contrast`` rows):

    - ``cells``: per (algo, fold), the mean ``delta`` over its replicates with a reading, their count,
      and whether it clears ``+threshold`` or ``-threshold``;
    - ``algo_blocks``: per (algo, source block), the mean over the block's folds (variants) of the
      cell means;
    - ``blocks``: per source block, the mean over algorithms of ``algo_blocks``, with ``n_algorithms``;
    - ``groups``: per stream group, the mean over blocks of ``blocks`` (each block once whatever its
      variants or seeds), ``n_blocks``, ``n_algorithms``, a block-bootstrap interval when there are
      three blocks or more, the blocks above zero / below zero, and the share of cells beyond
      ``+-threshold``; plus ``per_algorithm`` rows: per algorithm, the mean over blocks.
    """
    out: dict[str, pd.DataFrame] = {}
    p = pairs.dropna(subset=["delta"]) if len(pairs) else pairs
    if p.empty:
        empty = pd.DataFrame()
        return {"cells": empty, "algo_blocks": empty, "blocks": empty, "groups": empty}
    p = p.copy()
    for c in ("n_point_identical", "n_point_differs", "n_point_undetermined"):
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0) if c in p else 0
    cells = (p.groupby(["algo", "fold", "group", "block"], sort=True)
             .agg(delta=("delta", "mean"), n_replicates=("delta", "size"), n_common=("n_common", "min"),
                  n_pair_windows=("n_common", "sum"), n_point_identical=("n_point_identical", "sum"),
                  n_point_differs=("n_point_differs", "sum"), n_point_undetermined=("n_point_undetermined", "sum"))
             .reset_index())
    cells["above_threshold"] = cells["delta"] > threshold + _EPS
    cells["below_threshold"] = cells["delta"] < -threshold - _EPS
    out["cells"] = cells
    ab = (cells.groupby(["algo", "group", "block"], sort=True)
          .agg(delta=("delta", "mean"), n_variants=("fold", "nunique")).reset_index())
    out["algo_blocks"] = ab
    blocks = (ab.groupby(["group", "block"], sort=True)
              .agg(delta=("delta", "mean"), n_algorithms=("algo", "nunique")).reset_index())
    out["blocks"] = blocks
    rows = []
    for group in [*C.STREAM_GROUPS, "all"]:
        bsub = blocks if group == "all" else blocks[blocks["group"] == group]
        csub = cells if group == "all" else cells[cells["group"] == group]
        if bsub.empty:
            continue
        lo, hi = block_bootstrap(bsub["delta"], n_boot, boot_seed)
        rows.append({
            "level": "group", "unit": group, "delta_block_mean": float(bsub["delta"].mean()),
            "delta_block_median": float(bsub["delta"].median()), "ci_low": lo, "ci_high": hi,
            "n_blocks": int(len(bsub)), "n_blocks_above_0": int((bsub["delta"] > _EPS).sum()),
            "n_blocks_below_0": int((bsub["delta"] < -_EPS).sum()),
            "n_algorithms": int(csub["algo"].nunique()), "n_cells": int(len(csub)),
            "share_cells_above_threshold": float(csub["above_threshold"].mean()),
            "share_cells_below_threshold": float(csub["below_threshold"].mean()),
            # replicate-pair windows behind the cells, by point support (``point_support_status``)
            "n_pair_windows": int(csub["n_pair_windows"].sum()),
            **{f"n_point_{s}": int(csub[f"n_point_{s}"].sum()) for s in POINT_SUPPORT_STATUSES},
        })
        absub = ab if group == "all" else ab[ab["group"] == group]
        for algo, g in absub.groupby("algo", sort=True):
            rows.append({"level": "per_algorithm", "unit": f"{group}/{algo}",
                         "delta_block_mean": float(g["delta"].mean()), "n_blocks": int(len(g)), "n_algorithms": 1,
                         "n_cells": int(len(csub[csub["algo"] == algo]))})
    out["groups"] = pd.DataFrame(rows)
    return out


# --- coverage ----------------------------------------------------------------------------------

def _dict(v) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            d = json.loads(v)
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


#: The sequential partition's reasons for having no silhouette (``window.SILHOUETTE_REASONS`` but ``valid``).
UNDEFINED_REASONS: tuple[str, ...] = ("insufficient_points", "single_label", "one_label_per_point")
#: The caps a member can be shed by (``confstream.apply_feasibility``'s ``feasible_reason``), errors aside.
CAPS: tuple[str, ...] = ("memory", "micro_clusters", "runtime")

COVERAGE_COLS = [
    "arm", "n_jobs_expected", "n_jobs_on_file", "n_skipped", "n_ok", "n_error", "n_timeout", "n_oom", "n_missing",
    # member-windows of the ok runs: every member scored on every window
    "n_member_windows", "n_member_windows_partition_read",
    # partition validity, read from the silhouette's reason whatever the caps did; denominator: the
    # member-windows whose member ran (``n_member_windows_partition_read``)
    "share_partition_undefined", *[f"share_partition_{r}" for r in UNDEFINED_REASONS],
    # feasibility rejection by a cap, whatever the partition; denominator: ``n_member_windows``
    "share_infeasible", *[f"share_infeasible_{c}" for c in CAPS],
    "n_valid_partition_infeasible", "n_undefined_partition_infeasible",
    # the model raised; denominator: ``n_member_windows``
    "share_member_error",
    # scored at the floor for any reason (cap, undefined partition, error, unread signal): what the selection saw
    "share_floor_score",
    "share_member_windows_sentinel", "share_member_windows_fallback", "n_member_windows_snapshot",
    "n_member_windows_stability", "n_member_windows_diag_error",
    # the lagged incumbent's windows: its partition (denominator: those with an incumbent that ran) and its caps
    "n_lag_windows", "n_lag_windows_partition_read", "share_lag_partition_undefined", "share_lag_infeasible",
    "n_births", "n_births_ok", "n_member_evaluations", "n_member_evaluations_ok",
    "n_tail_windows_ari", "n_tail_windows_nmi",
]


def _add(store: dict, counts: dict) -> None:
    for k, v in counts.items():
        store[k] = store.get(k, 0) + int(v)


def coverage(runs: pd.DataFrame, tw: pd.DataFrame | None = None, expected: dict[str, int] | None = None) -> pd.DataFrame:
    """Per arm (``COVERAGE_COLS``): runnable jobs expected (``expected``: arm -> count, e.g. a campaign
    inventory's runnable jobs; else the jobs on file), the ``skipped`` rows on file apart, and their
    latest statuses. Then, over the member-windows of the ok runs, three readings kept apart because a
    member can fail more than one way at once:

    - **partition validity** from ``member_silhouette_reason_counts`` (each member-window's sequential
      partition, read before any cap): the share with no silhouette, per reason, over the member-windows
      whose member ran. A partition shed by a cap still counts here as what it was;
    - **feasibility rejection** from ``member_reason_feasibility_counts``: the share shed by each cap over
      every member-window, and how many of those had a valid or an undefined partition;
    - **member errors** over every member-window.

    ``share_floor_score`` is the share whose selection score is the floor for any of these reasons or an
    unread signal (``member_status_counts``). A run written before these counts existed leaves the shares
    that need them undefined, never derived from the score status. Also: sentinel and fallback predictions,
    snapshot and stability readings, failed diagnostics; the lagged incumbent's partition and caps; births
    and evaluations; tail windows with an ARI and an NMI reading. A job that failed adds to its status
    column only."""
    rows = []
    arms = sorted(set(runs["arm"]) | set(expected or {})) if len(runs) or expected else []
    for arm in arms:
        sub = runs[runs["arm"] == arm] if len(runs) else runs
        st = sub["status"].value_counts().to_dict() if len(sub) else {}
        ok = sub[sub["status"] == "ok"] if len(sub) else sub
        n_skipped = int(st.get("skipped", 0))
        runnable = int(len(sub)) - n_skipped
        n_exp = (expected or {}).get(arm, runnable)
        status, reasons, joint, lag_status, lag_reasons = {}, {}, {}, {}, {}
        have_reasons = have_joint = bool(len(ok))
        for r in ok.to_dict("records"):
            _add(status, _dict(r.get("member_status_counts")))
            _add(lag_status, _dict(r.get("lag_score_status_counts")))
            _add(lag_reasons, _dict(r.get("lag_silhouette_reason_counts")))
            if not isinstance(r.get("member_silhouette_reason_counts"), dict):
                have_reasons = False
            _add(reasons, _dict(r.get("member_silhouette_reason_counts")))
            if not isinstance(r.get("member_reason_feasibility_counts"), dict):
                have_joint = False
            _add(joint, _dict(r.get("member_reason_feasibility_counts")))
        n_mw = sum(status.values())
        n_read = sum(v for k, v in reasons.items() if k != "none")

        def share(num, den, known=True):
            return (num / den) if known and den else np.nan

        def total(col):
            return int(pd.to_numeric(ok[col], errors="coerce").fillna(0).sum()) if col in ok else 0

        def joint_count(pred):
            return sum(v for k, v in joint.items() if pred(*k.split("|", 1)))

        undefined = {r: reasons.get(r, 0) for r in UNDEFINED_REASONS}
        n_joint = sum(joint.values())
        lag_read = sum(v for k, v in lag_reasons.items() if k != "none")
        n_lag = sum(lag_status.values())
        tsub = tw[tw["arm"] == arm] if tw is not None and len(tw) else None
        rows.append({
            "arm": arm, "n_jobs_expected": n_exp, "n_jobs_on_file": runnable, "n_skipped": n_skipped,
            "n_ok": int(st.get("ok", 0)),
            "n_error": int(st.get("error", 0)), "n_timeout": int(st.get("timeout", 0)), "n_oom": int(st.get("oom", 0)),
            "n_missing": max(0, n_exp - runnable),
            "n_member_windows": n_mw, "n_member_windows_partition_read": n_read if have_reasons else np.nan,
            "share_partition_undefined": share(sum(undefined.values()), n_read, have_reasons),
            **{f"share_partition_{r}": share(undefined[r], n_read, have_reasons) for r in UNDEFINED_REASONS},
            "share_infeasible": share(joint_count(lambda _, f: f in CAPS), n_joint, have_joint),
            **{f"share_infeasible_{c}": share(joint_count(lambda _, f, c=c: f == c), n_joint, have_joint) for c in CAPS},
            "n_valid_partition_infeasible": joint_count(lambda r, f: r == "valid" and f in CAPS) if have_joint else np.nan,
            "n_undefined_partition_infeasible": (joint_count(lambda r, f: r in UNDEFINED_REASONS and f in CAPS)
                                                 if have_joint else np.nan),
            "share_member_error": share(status.get("member_error", 0), n_mw),
            "share_floor_score": share(sum(v for k, v in status.items() if k != "valid"), n_mw),
            "share_member_windows_sentinel": share(total("n_member_windows_sentinel"), n_mw),
            "share_member_windows_fallback": share(total("n_member_windows_fallback"), n_mw),
            "n_member_windows_snapshot": total("n_member_windows_snapshot"),
            "n_member_windows_stability": total("n_member_windows_stability"),
            "n_member_windows_diag_error": total("n_member_windows_diag_error"),
            "n_lag_windows": n_lag, "n_lag_windows_partition_read": lag_read,
            "share_lag_partition_undefined": share(sum(v for k, v in lag_reasons.items() if k in UNDEFINED_REASONS),
                                                   lag_read),
            "share_lag_infeasible": share(sum(v for k, v in lag_status.items() if k.startswith("infeasible_")),
                                          sum(v for k, v in lag_status.items() if k != "none")),
            "n_births": total("n_births"), "n_births_ok": total("n_births_ok"),
            "n_member_evaluations": total("n_member_evaluations"),
            "n_member_evaluations_ok": total("n_member_evaluations_ok"),
            "n_tail_windows_ari": int(tsub["lag_ari_win"].notna().sum()) if tsub is not None and "lag_ari_win" in tsub else 0,
            "n_tail_windows_nmi": int(tsub["lag_nmi_win"].notna().sum()) if tsub is not None and "lag_nmi_win" in tsub else 0,
        })
    return pd.DataFrame(rows).reindex(columns=COVERAGE_COLS)
