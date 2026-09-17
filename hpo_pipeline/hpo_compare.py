"""Pilot: four ways of choosing ONE hyperparameter set, on River's own clusterers.

Before the pipeline is scaled to the whole pool it is checked on the two
stream clusterers River itself ships -- ``CluStream`` and ``DenStream`` -- where
no estimator written for this project can muddy the picture.

Each algorithm gets a small grid over the parameters that dominate it: the
resolution axis (``n_macro_clusters`` for CluStream, ``epsilon`` for DenStream)
times a few shape parameters. The value sets come from the two source papers;
``GRIDS`` below documents where every value comes from. Everything not in the
grid stays at River's *constructor* default.

Every arm ends with **one set per algorithm**, and that one set is then run
unchanged on every dataset. The arms differ in the grid and in how the one set
is chosen:

    arm 1  raw grid          the set with the best score over all datasets
    arm 2  raw grid          leave-one-dataset-out: every round picks a set on
                             the other datasets; the final set is the one most
                             rounds picked (ties: arm 1's ordering)
    arm 3  substituted grid  as arm 2. The set is dimensionless (``c``, a
                             micro-ratio) and resolves per dataset through s(D)
    P2R    substituted grid  the shape is chosen as in arm 3 and the rule's phi
                             on all datasets; the resolution level is then read
                             off each dataset's own label-free curve. The
                             product is a *policy*, not one epsilon.
    sf-all substituted grid  arm 1 on the substituted grid, to separate "better
                             grid" from "cost of the LODO choice"

``--aggregate`` sets how scores are pooled across datasets when a set is
chosen: ``mean`` ARI, or ``rank`` (mean per-dataset rank, so one easy dataset
cannot outvote the others).

Scoring the final set on the datasets it was chosen on is in-sample. Arms 2, 3
and P2R therefore also report their **LODO estimate**: in every round the set
chosen without a dataset is scored on that dataset. That estimate, not the
in-sample score, is what the procedure is worth on a dataset it never saw.

Tuning runs every grid point under seed 1. Every chosen set then runs again
under seeds 2 and 3, each with its own permutation -- re-running under the
tuning seed would only repeat the tuning score. River's default is scored the
same way.

Run from the repo root::

    python -m hpo_pipeline.hpo_compare --n-samples 50000 --workers 12 --timeout 7200
    python -m hpo_pipeline.hpo_compare --n-samples 50000 --aggregate rank
    python -m hpo_pipeline.hpo_compare --n-samples 2000            # smoke test
    python -m hpo_pipeline.hpo_compare --report-only
    python -m hpo_pipeline.hpo_compare --grids '{"DenStream": {"levels": {"raw": [0.1, 1, 10]}}}'

    # re-run only what timed out, with a longer limit; check first with --dry-run
    python -m hpo_pipeline.hpo_compare --phase tune --retry-failed --timeout 7200
    python -m hpo_pipeline.hpo_compare --phase score --retry-failed --timeout 7200

P2R readings are added by default; ``--p2r`` chooses which, ``--p2r ""`` none.

Folds must already have a Stage 2 probe (``configs/probe.json``). Missing
permutations for the scoring seeds are materialised on the way.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ._common import (
    FOLD_GROUPS,
    FOLDS,
    HPO_OUT,
    PROBE_JSON,
    SCALER,
    SCALER_PARAMS,
    append_jsonl,
    config_id,
    read_json,
    read_jsonl,
    require,
    shuffle_seed_for,
)
from .build_matrix import execute
from .rules import RULES, orient
from .space import GRAN, resolve
from .stream import materialise

ALGOS = ("CluStream", "DenStream")
TUNE_SEED = 1
FINAL_SEEDS = (2, 3)
OBJ = "ari"

OUT_DIR = HPO_OUT / "compare"
RUNS_JSONL = OUT_DIR / "runs.jsonl"


def set_out_dir(path) -> None:
    """Redirect every artifact of this module (``--out-dir``; smoke tests)."""
    global OUT_DIR, RUNS_JSONL
    from pathlib import Path

    OUT_DIR = Path(path)
    RUNS_JSONL = OUT_DIR / "runs.jsonl"

# --- the grids ---------------------------------------------------------------
#
# Levels of the resolution axis are listed COARSE -> FINE (Stage 3 convention:
# level 1 is the fewest clusters). Counts therefore ascend and distance
# thresholds descend. ``raw`` values are absolute; ``ladder`` values are the
# substitutes that ``resolve_candidate`` turns into absolute values per fold.
#
# CluStream (Aggarwal, Han, Wang, Yu, VLDB 2003; River implementation):
#   n_macro_clusters (k)  the offline k-means input. Log2 ladder 2..64 so that
#                         every real fold's class count (2..51) has a level
#                         within a factor of sqrt(2).
#   max_micro_clusters(q) Sec. 3: "significantly larger than the natural number
#                         of clusters". Sec. 6.4 defines the micro-ratio q/k and
#                         shows SSQ stabilising at a ratio of about 10; the
#                         scalability runs fix q = 10k. Raw: the paper's own
#                         absolute q on KDD (10 x 5 clusters = 50), River's
#                         default 100, and its double. Ladder: ratio {2, 5, 10}
#                         -- the rising part of Fig. 9, just before the plateau,
#                         and the plateau. Beyond 10 the paper reports no gain,
#                         and River's micro-cluster maintenance is O(q^2) per
#                         unabsorbed point: q = 1280 (ratio 20 at k = 64) timed
#                         out at 600 s on a 2000-sample smoke test.
#   micro_cluster_r_factor (t)  the maximal boundary factor. Table 2 sweeps
#                         t in {1, 2, 4, 6, 8}: 14.9% of points fall outside at
#                         t=1, 1.6% at t=2, 0.18% at t=4, then nothing changes.
#                         t=2 is the paper's choice (the 95% rule for Gaussian
#                         deviations); {1, 2, 4} spans the part of the table
#                         where the behaviour still moves.
#   fixed: time_window=1000 (the horizon; River default), time_gap=100 (River's
#          own offline trigger, no paper analogue), halflife of the inner KMeans.
#   constraint: q >= k, or the offline k-means has fewer pseudo-points than
#          seeds (Sec. 4).
#
# DenStream (Cao, Ester, Qian, Zhou, SDM 2006; River implementation):
#   epsilon               the micro-cluster radius bound (Def. 3.3) and the
#                         DBSCAN reach (Def. 4.1: 2*epsilon). The paper sets
#                         epsilon = alpha * d_min with d_min the smallest
#                         inter-cluster gap -- a recipe that needs labels. Raw:
#                         7 geometric levels anchored at the two published
#                         defaults, River's 0.02 and the paper's 16. Ladder:
#                         c in {0.05 .. 3.2}, 7 levels a factor 2 apart, with
#                         epsilon = c * s(D); the top end covers the pilot's
#                         endpoint optimum on shuttle.
#   mu                    the core weight (Def. 3.1). The total stream weight
#                         is W = v / (1 - 2^-lambda) (Sec. 3), so at most W/mu
#                         core micro-clusters can exist; with River's v=100 and
#                         lambda=0.25, W ~ 629 and mu in {2, 5, 10, 20} allows
#                         {315, 126, 63, 31} of them. 2 is River's default, 10
#                         the paper's; the paper says "choose the smallest
#                         acceptable mu".
#   beta                  outlier threshold: a micro-cluster is potential when
#                         its weight exceeds beta*mu (Def. 3.4). Fig. 20: quality
#                         is good for beta in [0.2, 0.6] and collapses at 1.
#                         {0.2, 0.5, 0.75}: paper default, midpoint, River
#                         default.
#   fixed: decaying_factor=0.25 (paper default = River default; Fig. 19 shows a
#          plateau over [0.125, 1], and after shuffling it only sets memory),
#          n_samples_init=1000 (both), stream_speed=100 (River; the paper uses
#          1000, which is why its mu is larger).
#   constraint: beta * mu > 1, or the pruning period T_p (Eq. 4.1) is
#          undefined; River computes log(beta*mu / (beta*mu - 1)).
GRIDS: dict[str, dict] = {
    "CluStream": {
        "levels": {
            "raw": [2, 4, 8, 16, 32, 64],
            "ladder": [2, 4, 8, 16, 32, 64],
        },
        "shape": {
            "raw": {"max_micro_clusters": [50, 100, 200], "micro_cluster_r_factor": [1, 2, 4]},
            "ladder": {"micro_ratio": [2, 5, 10], "micro_cluster_r_factor": [1, 2, 4]},
        },
    },
    "DenStream": {
        "levels": {
            "raw": [16.0, 5.5, 1.8, 0.6, 0.2, 0.06, 0.02],
            "ladder": [3.2, 1.6, 0.8, 0.4, 0.2, 0.1, 0.05],
        },
        "shape": {
            "raw": {"mu": [2, 5, 10, 20], "beta": [0.2, 0.5, 0.75]},
            "ladder": {"mu": [2, 5, 10, 20], "beta": [0.2, 0.5, 0.75]},
        },
    },
}

#: Shape substitutes: dimensionless key -> (absolute parameter it replaces,
#: how to resolve it given the value, the resolved resolution-axis value and
#: the fold's probe).
SUBSTITUTES: dict[str, tuple[str, object]] = {
    "micro_ratio": ("max_micro_clusters", lambda ratio, k, probe: int(round(ratio * k))),
}

#: Validity constraints on the RESOLVED parameters. A grid point that fails is
#: not generated; the report says how many were rejected and why.
CONSTRAINTS: dict[str, list[tuple[str, object]]] = {
    "CluStream": [
        ("max_micro_clusters >= n_macro_clusters",
         lambda p: p["max_micro_clusters"] >= p["n_macro_clusters"]),
    ],
    "DenStream": [
        ("beta * mu > 1", lambda p: p["beta"] * p["mu"] > 1),
    ],
}

#: Arms whose product is one set (or one policy) per algorithm, in report order.
FINAL_ARMS = ("default", "arm1_raw_all", "arm2_raw_lodo", "arm3_sf_lodo", "sf_all")
#: How pooled scores are formed when a set is chosen across datasets.
AGGREGATES = ("mean", "rank")
#: Suffix of the protocol holding a procedure's leave-one-dataset-out estimate.
EST = "_est"
ARM_LABEL = {
    "default": "default",
    "arm1_raw_all": "arm 1",
    "arm2_raw_lodo": "arm 2",
    "arm3_sf_lodo": "arm 3",
    "sf_all": "sf-all",
    "ceiling_raw": "per-dataset ceiling (raw)",
    "ceiling_sf": "per-dataset ceiling (substituted)",
}


def arm_label(protocol: str) -> str:
    base = protocol[: -len(EST)] if protocol.endswith(EST) else protocol
    label = ARM_LABEL.get(base, base[len("p2r_"):] if base.startswith("p2r_") else base)
    return label + (" (LODO estimate)" if protocol.endswith(EST) else "")


#: P2R readings (rule:criterion) computed by default. `matr_trace_win` is defined
#: for every algorithm and is the pipeline's primary criterion. `R_max` on
#: `silhouette` is the contrast pre-registered in rules.py: silhouette is not
#: monotone in the cluster count, so R_max should not degenerate there.
P2R_DEFAULT = (
    "R_max:matr_trace_win,R_parsimony:matr_trace_win,R_knee:matr_trace_win,R_max:silhouette"
)
P2R_CRITERIA = ("matr_trace_win", "silhouette", "ssq")

#: Thresholds the verdicts use. Stated here so the report can quote them.
MIN_SPREAD_TO_NOISE = 3.0
MAX_ENDPOINT_SHARE = 0.20


# --- candidates ------------------------------------------------------------
def river_defaults(algo: str) -> dict:
    """River's constructor defaults, read off the signature.

    ``seed`` is left out because the harness injects the run seed.
    """
    from river import cluster

    skip = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    params = inspect.signature(getattr(cluster, algo).__init__).parameters
    return {
        name: p.default
        for name, p in params.items()
        if name not in ("self", "seed")
        and p.kind not in skip
        and p.default is not inspect.Parameter.empty
    }


def merge_grids(base: dict, override: dict | None) -> dict:
    """``--grids`` JSON merged two levels deep: {algo: {levels|shape: {raw|ladder: ...}}}."""
    out = {a: {k: {g: (dict(v) if isinstance(v, dict) else list(v)) for g, v in sub.items()}
               for k, sub in spec.items()} for a, spec in base.items()}
    for algo, spec in (override or {}).items():
        out.setdefault(algo, {"levels": {}, "shape": {}})
        for key, sub in spec.items():
            out[algo].setdefault(key, {})
            for grid, values in sub.items():
                out[algo][key][grid] = values
    return out


def _candidate(algo, level, g, shape, *, grid, theta, id_level=None) -> dict:
    return {
        "algo": algo,
        "grid": grid,
        "granularity_level": level,
        "theta": theta,
        "g": g,
        "shape": dict(shape),
        "config_id": config_id(algo, level if id_level is None else id_level, g, shape),
        "scaler": SCALER,
        "scaler_params": dict(SCALER_PARAMS),
        "is_default": grid == "default",
        "raw": grid == "raw",
    }


def resolve_candidate(candidate: dict, probe: dict) -> dict:
    """Dimensionless spec -> absolute params for one fold, substitutes included.

    ``space.resolve`` sets the resolution axis (``epsilon = c * s(D)``, or the
    count as is). Substitutes in the shape are then replaced by the absolute
    parameter they stand for, computed from the resolved axis value.
    """
    params = resolve(candidate, probe)
    axis, _ = GRAN[candidate["algo"]]
    for key, (target, fn) in SUBSTITUTES.items():
        if key in params:
            params[target] = fn(params.pop(key), params[axis], probe)
    return params


def _violations(algo: str, params: dict) -> list[str]:
    return [name for name, ok in CONSTRAINTS.get(algo, []) if not ok(params)]


def shape_points(shape_spec: dict) -> list[dict]:
    """Cartesian product of the shape value sets, in a fixed key order."""
    keys = sorted(shape_spec)
    return [dict(zip(keys, values)) for values in itertools.product(*(shape_spec[k] for k in keys))]


def build_grid(algo: str, kind: str, spec: dict) -> tuple[list[dict], list[dict]]:
    """All (level, theta) candidates of one grid, and the points rejected.

    Every parameter outside the grid stays at River's constructor default. A
    substitute's target parameter is excluded from those defaults, otherwise
    the resolved value would be set twice.
    """
    axis, _ = GRAN[algo]
    shape_spec = spec["shape"][kind]
    levels = spec["levels"][kind]
    targets = {SUBSTITUTES[k][0] for k in shape_spec if k in SUBSTITUTES}
    fixed = {
        k: v for k, v in river_defaults(algo).items()
        if k != axis and k not in shape_spec and k not in targets
    }
    accepted, rejected = [], []
    for theta, point in enumerate(shape_points(shape_spec), start=1):
        for level, value in enumerate(levels, start=1):
            if kind == "raw":
                cand = _candidate(algo, level, value, {**fixed, **point, axis: value},
                                  grid="raw", theta=theta, id_level=f"raw{level}")
            else:
                cand = _candidate(algo, level, value, {**fixed, **point},
                                  grid="ladder", theta=theta)
            # Constraints never involve the fold scale, so a unit probe suffices.
            broken = _violations(algo, resolve_candidate(cand, {"s": 1.0}))
            if broken:
                rejected.append({"level": level, "theta": theta, "point": point,
                                 "value": value, "violates": broken})
            else:
                accepted.append(cand)
    return accepted, rejected


def default_candidate(algo: str) -> dict:
    return _candidate(algo, None, None, river_defaults(algo), grid="default", theta=None)


def tuned_params(algo: str, grids: dict) -> list[str]:
    """Names of the absolute parameters any arm tunes for ``algo``."""
    axis, _ = GRAN[algo]
    names = {axis}
    for kind in ("raw", "ladder"):
        for key in grids[algo]["shape"][kind]:
            names.add(SUBSTITUTES[key][0] if key in SUBSTITUTES else key)
    return sorted(names)


# --- running ---------------------------------------------------------------
def _params_blob(params: dict | None) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _key(algo, cid, fold, seed, requested, params) -> tuple:
    # The resolved params are part of the key. `config_id` names a position on
    # the dimensionless ladder, so after s(D) changes the same id resolves to a
    # different epsilon -- and a key without params would silently reuse a run
    # that measured something else.
    return (algo, cid, fold, seed, requested, _params_blob(params))


def _row_key(row: dict) -> tuple:
    return _key(row["algo"], row["config_id"], row["dataset"], row["seed"],
                row.get("requested"), row.get("params_resolved"))


def _job_key(job: dict) -> tuple:
    return _key(job["algo"], job["candidate"]["config_id"], job["fold"], job["seed"],
                job["requested"], job["params"])


def _fmt_value(value) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def describe(job: dict) -> str:
    """Readable label for a run: the tuned values and where they came from."""
    algo, cand = job["algo"], job["candidate"]
    axis, kind = GRAN[algo]
    params = job["params"]
    if cand["grid"] == "default":
        return f"{axis}={_fmt_value(params.get(axis))} [default]"
    tuned = job.get("tuned") or [axis]
    shown = ", ".join(f"{p}={_fmt_value(params.get(p))}" for p in tuned if p in params)
    if cand["grid"] == "raw":
        where = f"raw L{cand['granularity_level']} t{cand['theta']}"
    elif kind == "scaled":
        where = f"ladder L{cand['granularity_level']} t{cand['theta']} c={cand['g']:g}"
    else:
        where = f"ladder L{cand['granularity_level']} t{cand['theta']}"
    return f"{shown} [{where}]"


def pending(jobs: list[dict], rows: list[dict], retry_failed: bool = False) -> list[dict]:
    """Jobs with no usable row on disk, deduplicated.

    A failed run counts as done by default: a timeout is a fact about that
    configuration. ``retry_failed`` re-queues them, for when the limit rather
    than the configuration was at fault. The retry is appended, and the newest
    row for a key is the one every table reads.
    """
    latest = {}
    for row in rows:
        latest[_row_key(row)] = row.get("status")
    done = {k for k, status in latest.items() if status == "ok" or not retry_failed}
    todo, seen = [], set()
    for job in jobs:
        key = _job_key(job)
        if key not in done and key not in seen:
            seen.add(key)
            todo.append(job)
    return todo


def run_jobs(jobs: list[dict], workers: int, timeout: int,
             retry_failed: bool = False, dry_run: bool = False) -> None:
    """Run what is not on disk yet (and, with ``retry_failed``, what failed)."""
    rows = list(read_jsonl(RUNS_JSONL))
    failed = {_row_key(r) for r in rows if r.get("status") != "ok"}
    todo = pending(jobs, rows, retry_failed)
    n_retry = sum(_job_key(j) in failed for j in todo)
    print(f"  {len(todo)} runs to do ({n_retry} retries of failed runs; "
          f"{len(jobs) - len(todo)} already on disk or duplicate)")
    if dry_run:
        for job in todo:
            tag = " (retry)" if _job_key(job) in failed else ""
            print(f"    would run: {job['algo']} {job['fold']} seed={job['seed']} "
                  f"{describe(job)}{tag}")
        return
    if not todo:
        return

    def one(job):
        # The child never sees `tuned`; it is a label-only field.
        row = execute({k: v for k, v in job.items() if k != "tuned"}, timeout)
        row["requested"] = job["requested"]
        # A failed row carries no params from the child; record what was asked.
        row.setdefault("params_resolved", job["params"])
        return row

    started, counts = time.time(), {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, job): job for job in todo}
        # Written in completion order, not submission order: a slow run at the
        # head of the queue must not hold finished rows in memory, where an
        # interrupt would lose them and the log would show no progress.
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            row = future.result()
            append_jsonl(RUNS_JSONL, row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            tag = " (retry)" if _job_key(job) in failed else ""
            print(
                f"  {i}/{len(todo)}  {time.time() - started:.0f}s  "
                f"{row['algo']} {row['dataset']} seed={row['seed']} {describe(job)}{tag} "
                f"{row['status']} ({row.get('runtime_s', '-')}s)  "
                + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            )


def _job(algo, candidate, fold, seed, n_samples, probes, tuned=()) -> dict:
    return {
        "algo": algo,
        "candidate": candidate,
        "fold": fold,
        "seed": seed,
        "budget": n_samples or None,
        "requested": n_samples or "full",
        "params": resolve_candidate(candidate, probes[fold]),
        "tuned": list(tuned),
    }


def load_runs(requested, grid_of: dict, candidates: dict, probes: dict) -> pd.DataFrame:
    """Current runs at this stream length, tagged with their grid.

    A row counts only if its params are what its config resolves to *now*;
    rows measured under an older s(D) stay in the file but out of every table.
    """
    rows = []
    for r in read_jsonl(RUNS_JSONL):
        cand = candidates.get((r["algo"], r["config_id"]))
        if r.get("requested") != requested or cand is None or r["dataset"] not in probes:
            continue
        if _params_blob(r.get("params_resolved")) == _params_blob(
            resolve_candidate(cand, probes[r["dataset"]])
        ):
            rows.append(r)
    if not rows:
        raise SystemExit(f"no current runs for n_samples={requested} in {RUNS_JSONL}")
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["algo", "config_id", "dataset", "seed"], keep="last")
    df["grid"] = [grid_of.get((a, c)) for a, c in zip(df.algo, df.config_id)]
    return df[df.grid.notna()].reset_index(drop=True)


# --- selection (pure: no runs, no files) -----------------------------------
def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.status == "ok") & df[OBJ].notna()]


def _first(frame: pd.DataFrame, score: str, runtime: str) -> str:
    """Deterministic argmax: score desc, runtime asc, coarser level, then id."""
    ranked = frame.sort_values(
        [score, runtime, "granularity_level", "config_id"],
        ascending=[False, True, True, True], kind="mergesort",
    )
    return str(ranked.iloc[0]["config_id"])


def _aggregate(sub: pd.DataFrame, folds_used: list[str], how: str = "mean") -> pd.DataFrame:
    """Pool one algorithm's tuning scores over ``folds_used``; bigger is better.

    Only configs that ran on every one of those folds are eligible: a set that
    cannot run on a dataset is not a set that can be shipped to it.

    ``mean`` averages ARI. ``rank`` averages each config's rank within every
    dataset (negated, so bigger is still better): a dataset where ARI runs to
    0.5 then counts exactly as much as one where it tops out at 0.05.
    """
    if how not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {how!r}")
    sub = _ok(sub)
    sub = sub[sub.dataset.isin(folds_used)]
    n = sub.groupby("config_id").dataset.nunique()
    sub = sub[sub.config_id.isin(n[n == len(folds_used)].index)]
    if sub.empty:
        return pd.DataFrame(columns=["config_id", "score", "runtime", "granularity_level"])
    if how == "rank":
        score = -sub.groupby("dataset")[OBJ].rank(ascending=False, method="average")
    else:
        score = sub[OBJ]
    sub = sub.assign(_score=score)
    return (
        sub.groupby("config_id")
        .agg(score=("_score", "mean"), runtime=("runtime_s", "mean"),
             granularity_level=("granularity_level", "first"))
        .reset_index()
    )


def pick_oracle(tune: pd.DataFrame) -> dict:
    """(algo, fold) -> config chosen on that fold's own labels. A ceiling only."""
    return {
        (algo, fold): _first(grp, OBJ, "runtime_s")
        for (algo, fold), grp in _ok(tune).groupby(["algo", "dataset"])
    }


def pick_all(tune: pd.DataFrame, folds: list[str], how: str = "mean") -> dict:
    """algo -> the one config with the best pooled score over all ``folds``."""
    picks = {}
    for algo in sorted(set(tune.algo)):
        agg = _aggregate(tune[tune.algo == algo], folds, how)
        if not agg.empty:
            picks[algo] = _first(agg, "score", "runtime")
    return picks


def _opt_folds(fold: str, folds: list[str], groups: dict | None, cross: bool) -> list[str]:
    if groups is None:
        return [f for f in folds if f != fold]
    if cross:
        return [f for f in folds if groups[f] != groups[fold]]
    return [f for f in folds if f != fold and groups[f] == groups[fold]]


def pick_lodo(tune: pd.DataFrame, folds: list[str], groups: dict | None = None,
              cross: bool = False, how: str = "mean") -> dict:
    """(algo, held-out fold) -> config chosen on the OTHER folds.

    These per-round picks are what the LODO *estimate* scores; the shipped set
    is their consensus (``consensus``).

    ``groups`` (fold -> group) keeps the optimisation set inside the held-out
    fold's own group, so folds cut from one generator cannot outvote the real
    ones. ``cross=True`` instead takes it from the other groups only. Without
    ``groups`` every other fold is used. The held-out fold is never consulted,
    so a pick that then fails there is a real failure.
    """
    picks = {}
    for algo in sorted(set(tune.algo)):
        sub = tune[tune.algo == algo]
        for fold in folds:
            opt = _opt_folds(fold, folds, groups, cross)
            if not opt:
                continue
            agg = _aggregate(sub, opt, how)
            if not agg.empty:
                picks[(algo, fold)] = _first(agg, "score", "runtime")
    return picks


def consensus(tune: pd.DataFrame, lodo: dict, folds: list[str], how: str = "mean") -> dict:
    """algo -> (config, votes, rounds): the set most LODO rounds picked.

    Ties go to the tied config with the best pooled score over all folds, so
    the result is deterministic and never a config no round chose.
    """
    out = {}
    for algo in sorted({a for a, _ in lodo}):
        votes: dict[str, int] = {}
        for (a, _), cid in sorted(lodo.items()):
            if a == algo:
                votes[cid] = votes.get(cid, 0) + 1
        top = max(votes.values())
        tied = sorted(c for c, v in votes.items() if v == top)
        cid = tied[0]
        if len(tied) > 1:
            agg = _aggregate(tune[tune.algo == algo], folds, how)
            agg = agg[agg.config_id.isin(tied)]
            if not agg.empty:
                cid = _first(agg, "score", "runtime")
        out[algo] = (cid, top, sum(votes.values()))
    return out


def _curve(tune: pd.DataFrame, algo: str, theta, fold: str, criterion: str, n_levels: int):
    """A label-free criterion along the ladder for one shape point on one fold.

    Bigger-is-better. None when any rung is missing or not finite: a timed-out
    rung, or River's streaming silhouette returning inf, breaks the curve
    rather than shortening it -- a rule read off a gappy curve would pick from
    the wrong positions.
    """
    sub = tune[(tune.algo == algo) & (tune.dataset == fold) & (tune.status == "ok")
               & (tune.theta == theta)]
    if criterion not in sub.columns or sub.empty:
        return None
    levels = sub.granularity_level.astype(int)
    series = pd.Series(pd.to_numeric(sub[criterion], errors="coerce").to_numpy(),
                       index=levels.to_numpy()).reindex(range(1, n_levels + 1))
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return orient(values, criterion)


def _ari_at(tune: pd.DataFrame, algo: str, theta, fold: str, level: int) -> float:
    sub = _ok(tune[(tune.algo == algo) & (tune.dataset == fold) & (tune.theta == theta)])
    hit = sub[sub.granularity_level.astype(int) == level][OBJ]
    return float(hit.iloc[0]) if len(hit) else np.nan


def _shape_scores(tune: pd.DataFrame, algo: str, folds_used: list[str],
                  how: str = "mean") -> dict:
    """theta -> pooled best-over-levels score on ``folds_used`` (bigger is better).

    A shape point is judged by the best level it reaches on each dataset, since
    P2R re-picks the level anyway. Datasets where a shape point never ran are
    skipped for that point, not counted as zero.
    """
    ok = _ok(tune[(tune.algo == algo) & tune.dataset.isin(folds_used)])
    if ok.empty:
        return {}
    best = ok.groupby(["theta", "dataset"])[OBJ].max().unstack("dataset")
    if how == "rank":
        best = -best.rank(axis=0, ascending=False, method="average")
    pooled = best.mean(axis=1, skipna=True)
    return {int(t): float(v) for t, v in pooled.items() if np.isfinite(v)}


def _rule_quality(tune, algo, theta, opt, rule, phi, criterion, n_levels) -> float:
    """Mean ARI the rule reaches on the optimisation folds with this phi."""
    vals = []
    for fold in opt:
        curve = _curve(tune, algo, theta, fold, criterion, n_levels)
        if curve is not None:
            vals.append(_ari_at(tune, algo, theta, fold, rule(curve, phi) + 1))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else -np.inf


def pick_p2r(tune: pd.DataFrame, folds: list[str], groups: dict | None,
             rule_name: str, criterion: str, n_levels: dict | int,
             how: str = "mean") -> dict:
    """P2R's LODO rounds: (algo, held-out fold) -> (theta, level).

    In each round the shape point ``theta*`` is chosen on the optimisation
    folds (pooled best-over-level score), and the level is read off the
    held-out fold's own label-free curve at ``theta*``. The rule's phi is
    fitted on the optimisation folds too, so nothing about the held-out fold's
    labels is read. These picks are what P2R's LODO estimate scores.

    The held-out labels are never read, but a rule can only be applied to a
    *complete* curve: a shape point whose held-out curve has a failed rung (a
    timeout is label-free information) yields to the next best one.
    """
    rule, phis = RULES[rule_name]
    picks = {}
    for algo in sorted(set(tune.algo)):
        levels = n_levels[algo] if isinstance(n_levels, dict) else n_levels
        for fold in folds:
            opt = _opt_folds(fold, folds, groups, cross=False)
            if not opt:
                continue
            quality = _shape_scores(tune, algo, opt, how)
            # Best first; ties keep the smaller theta, so the order is fixed.
            ranked = sorted(quality, key=lambda th: (-quality[th], th))
            theta_star, curve = None, None
            for theta in ranked:
                curve = _curve(tune, algo, theta, fold, criterion, levels)
                if curve is not None:
                    theta_star = theta
                    break
            if theta_star is None:
                continue
            phi = max(phis, key=lambda p: _rule_quality(tune, algo, theta_star, opt, rule, p,
                                                        criterion, levels))
            picks[(algo, fold)] = (theta_star, rule(curve, phi) + 1)
    return picks


def p2r_policy(tune: pd.DataFrame, folds: list[str], rule_name: str, criterion: str,
               n_levels: dict | int, est: dict, how: str = "mean") -> dict:
    """algo -> the one P2R policy shipped: a fixed shape, a fixed phi, a rule.

    ``theta`` is the shape most LODO rounds chose (ties: best pooled over all
    folds); ``phi`` is fitted on all folds at that shape. The level is then
    read off each dataset's own label-free curve -- that is the point of P2R,
    so the resolved threshold differs per dataset while the policy does not.
    A dataset whose curve is broken at ``theta`` gets no level.
    """
    rule, phis = RULES[rule_name]
    out = {}
    for algo in sorted(set(tune.algo)):
        votes: dict[int, int] = {}
        for (a, _), (theta, _) in sorted(est.items()):
            if a == algo:
                votes[theta] = votes.get(theta, 0) + 1
        if not votes:
            continue
        levels_n = n_levels[algo] if isinstance(n_levels, dict) else n_levels
        top = max(votes.values())
        tied = sorted(t for t, v in votes.items() if v == top)
        quality = _shape_scores(tune, algo, folds, how)
        theta = max(tied, key=lambda t: (quality.get(t, -np.inf), -t))
        phi = max(phis, key=lambda p: _rule_quality(tune, algo, theta, folds, rule, p,
                                                    criterion, levels_n))
        chosen = {}
        for fold in folds:
            curve = _curve(tune, algo, theta, fold, criterion, levels_n)
            if curve is not None:
                chosen[fold] = rule(curve, phi) + 1
        out[algo] = {"theta": theta, "phi": phi, "votes": top,
                     "rounds": sum(votes.values()), "levels": chosen}
    return out


def parse_p2r(spec: str) -> list[tuple[str, str]]:
    """'R_max:matr_trace_win,R_knee:silhouette' -> [(rule, criterion), ...]."""
    out = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        rule, _, criterion = item.partition(":")
        if rule not in RULES or criterion not in P2R_CRITERIA:
            raise SystemExit(
                f"bad --p2r entry {item!r}; rules: {sorted(RULES)}, criteria: {list(P2R_CRITERIA)}"
            )
        out.append((rule, criterion))
    return out


def all_picks(tune, grids, defaults, algos, folds, groups, p2r=(),
              aggregate: str = "mean") -> tuple[dict, dict]:
    """Every arm's picks, plus the one set each arm ships.

    Returns ``(picks, finals)``:

    * ``picks``: protocol -> {(algo, fold): candidate}. For a final arm the
      candidate is the same set on every fold (for P2R, the same shape with the
      level its rule read on that fold). ``<arm>_est`` protocols hold the LODO
      rounds; ``ceiling_*`` the per-dataset oracles, kept as reference.
    * ``finals``: protocol -> {algo: {"config": candidate, ...}} -- what the
      arm hands over, with how it was reached.
    """
    by_id = {(c["algo"], c["config_id"]): c
             for kind in grids for cands in grids[kind].values() for c in cands}
    by_cell = {(c["algo"], c["theta"], c["granularity_level"]): c
               for a in algos for c in grids["ladder"][a]}
    raw, ladder = tune[tune.grid == "raw"], tune[tune.grid == "ladder"]
    picks: dict[str, dict] = {}
    finals: dict[str, dict] = {}

    def ship(protocol, chosen: dict, extra: dict | None = None):
        picks[protocol] = {(a, f): by_id[(a, cid)] for a, cid in chosen.items() for f in folds}
        finals[protocol] = {a: {"config": by_id[(a, cid)], **((extra or {}).get(a, {}))}
                            for a, cid in chosen.items()}

    picks["default"] = {(a, f): defaults[a] for a in algos for f in folds}
    finals["default"] = {a: {"config": defaults[a]} for a in algos}

    ship("arm1_raw_all", pick_all(raw, folds, aggregate))
    ship("sf_all", pick_all(ladder, folds, aggregate))

    for protocol, sub in (("ceiling_raw", raw), ("ceiling_sf", ladder)):
        picks[protocol] = {k: by_id[(k[0], cid)] for k, cid in pick_oracle(sub).items()}

    for protocol, sub in (("arm2_raw_lodo", raw), ("arm3_sf_lodo", ladder)):
        est = pick_lodo(sub, folds, groups, how=aggregate)
        picks[protocol + EST] = {k: by_id[(k[0], cid)] for k, cid in est.items()}
        agreed = consensus(sub, est, folds, aggregate)
        ship(protocol, {a: v[0] for a, v in agreed.items()},
             {a: {"votes": v[1], "rounds": v[2]} for a, v in agreed.items()})

    if len(set(groups.values())) >= 2:
        cross = pick_lodo(ladder, folds, groups, cross=True, how=aggregate)
        picks["arm3_sf_cross" + EST] = {k: by_id[(k[0], cid)] for k, cid in cross.items()}

    n_levels = {a: len(set(c["granularity_level"] for c in grids["ladder"][a])) for a in algos}
    for rule_name, criterion in p2r:
        protocol = f"p2r_{rule_name}_{criterion}"
        est = pick_p2r(ladder, folds, groups, rule_name, criterion, n_levels, aggregate)
        picks[protocol + EST] = {
            k: by_cell[(k[0], theta, level)] for k, (theta, level) in est.items()
            if (k[0], theta, level) in by_cell
        }
        policy = p2r_policy(ladder, folds, rule_name, criterion, n_levels, est, aggregate)
        picks[protocol] = {
            (a, f): by_cell[(a, pol["theta"], level)]
            for a, pol in policy.items() for f, level in pol["levels"].items()
            if (a, pol["theta"], level) in by_cell
        }
        finals[protocol] = {
            a: {"config": by_cell[(a, pol["theta"], 1)], "rule": rule_name,
                "criterion": criterion, **pol}
            for a, pol in policy.items() if (a, pol["theta"], 1) in by_cell
        }
    return picks, finals


def spec_label(cand: dict, grids_spec: dict, tuned: list[str], *, level: bool = True) -> str:
    """The set as a person would write it down, in the grid's own units.

    A substituted set is written with its substitutes (``epsilon = 1.6 * s(D)``,
    ``micro_ratio=5``) because that, not a per-dataset number, is what it is.
    ``level=False`` leaves the resolution axis out (P2R reads it per dataset).
    """
    algo = cand["algo"]
    axis, kind = GRAN[algo]
    shape = cand["shape"]
    if cand["grid"] == "default":
        return ", ".join(f"{p}={_fmt_value(shape[p])}" for p in tuned if p in shape)
    parts = []
    if level:
        if cand["grid"] == "ladder" and kind == "scaled":
            parts.append(f"{axis} = {cand['g']:g} * s(D)")
        else:
            parts.append(f"{axis}={_fmt_value(cand['g'])}")
    for key in sorted(grids_spec[algo]["shape"][cand["grid"]]):
        parts.append(f"{key}={_fmt_value(shape[key])}")
    return ", ".join(parts)


# --- scoring ---------------------------------------------------------------
def _tuned_label(cand: dict, params: dict, tuned: list[str]) -> str:
    axis, kind = GRAN[cand["algo"]]
    parts = [f"{p}={_fmt_value(params.get(p))}" for p in tuned if p in params]
    extra = []
    if cand["grid"] == "ladder" and kind == "scaled":
        extra.append(f"c={cand['g']:g}")
    for key in cand["shape"]:
        if key in SUBSTITUTES:
            extra.append(f"{key}={_fmt_value(cand['shape'][key])}")
    return ", ".join(parts) + (f" ({'; '.join(extra)})" if extra else "")


def score_table(df, picks, probes, tuned_of: dict) -> pd.DataFrame:
    """One row per (algo, fold, protocol): what was picked and how it scored."""
    rows = []
    for protocol, chosen in picks.items():
        for (algo, fold), cand in sorted(chosen.items()):
            cells = df[
                (df.algo == algo) & (df.config_id == cand["config_id"]) & (df.dataset == fold)
            ]
            final = cells[cells.seed.isin(FINAL_SEEDS)]
            final_ok = _ok(final)
            tune_cell = _ok(cells[cells.seed == TUNE_SEED])
            every_seed = _ok(cells[cells.seed.isin((TUNE_SEED, *FINAL_SEEDS))])
            axis = GRAN[algo][0]
            params = resolve_candidate(cand, probes[fold])
            rows.append(
                {
                    "algo": algo,
                    "dataset": fold,
                    "protocol": protocol,
                    "grid": cand["grid"],
                    "config_id": cand["config_id"],
                    "level": cand["granularity_level"],
                    "theta": cand["theta"],
                    "g": cand["g"],
                    "param": axis,
                    "value": params.get(axis),
                    "picked": _tuned_label(cand, params, tuned_of[algo]),
                    "tune_ari": float(tune_cell[OBJ].iloc[0]) if len(tune_cell) else np.nan,
                    "final_ari": float(final_ok[OBJ].mean()) if len(final_ok) else np.nan,
                    # Two scoring seeds alone understate the noise; the tuning
                    # run is a third draw of the same config and belongs in it.
                    "seed_std": float(every_seed[OBJ].std()) if len(every_seed) > 1 else np.nan,
                    "n_final_ok": len(final_ok),
                    "final_status": ",".join(sorted(set(final.status))) or "missing",
                    "k_hat": float(final_ok["k_hat"].mean()) if len(final_ok) else np.nan,
                    "n_predicted_clusters": (
                        float(final_ok["n_predicted_clusters"].mean()) if len(final_ok) else np.nan
                    ),
                    "runtime_s": float(final_ok["runtime_s"].mean()) if len(final_ok) else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --- report ----------------------------------------------------------------
def _f(x, digits=3) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _nanmean(values) -> float:
    """Mean over the finite entries; NaN (and no warning) when there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def _cv(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.std() / abs(v.mean())) if len(v) > 1 and v.mean() else np.nan


def _grid_section(algos, grids_spec, grids, rejected, tuned_of) -> str:
    out = []
    w = out.append
    w("\n## 0. Grids\n\n")
    w("Every parameter not listed stays at River's constructor default. Levels of the "
      "resolution axis run coarse to fine (level 1 = fewest clusters).\n\n")
    for algo in algos:
        axis, kind = GRAN[algo]
        spec = grids_spec[algo]
        w(f"### {algo}\n\n")
        w("| grid | resolution axis | shape | points | rejected |\n|---|---|---|---|---|\n")
        for g in ("raw", "ladder"):
            unit = (f"`{axis}` = c * s(D), c in" if (g == "ladder" and kind == "scaled")
                    else f"`{axis}` in")
            levels = ", ".join(f"{v:g}" for v in spec["levels"][g])
            shape = "; ".join(
                f"`{k}` in {{{', '.join(f'{v:g}' for v in vals)}}}"
                for k, vals in sorted(spec["shape"][g].items())
            )
            w(f"| {g} | {unit} {{{levels}}} | {shape} | {len(grids[g][algo])} | "
              f"{len(rejected[g][algo])} |\n")
        subs = [k for k in spec["shape"]["ladder"] if k in SUBSTITUTES]
        for key in subs:
            w(f"\n`{key}` resolves to `{SUBSTITUTES[key][0]}` = {key} * `{axis}`.\n")
        rules = "; ".join(name for name, _ in CONSTRAINTS.get(algo, []))
        if rules:
            w(f"\nConstraint: {rules}. Tuned parameters: {', '.join(tuned_of[algo])}.\n")
        w("\n")
    return "".join(out)


def build_report(tune, scores, folds, probes, requested, grids, grids_spec, rejected,
                 tuned_of, finals, aggregate) -> str:
    out = []
    w = out.append
    by = {(r.algo, r.dataset, r.protocol): r._asdict() for r in scores.itertuples(index=False)}

    def cell(algo, fold, protocol, col="final_ari"):
        row = by.get((algo, fold, protocol))
        return np.nan if row is None else row[col]

    def vec(algo, protocol):
        return np.array([cell(algo, f, protocol) for f in folds], dtype=float)

    def mean_of(algo, protocol):
        return _nanmean(vec(algo, protocol))

    def n_scored(algo, protocol):
        return int(np.isfinite(vec(algo, protocol)).sum())

    def mean_cell(algo, protocol):
        # A mean over fewer datasets is not comparable with a full one; say so.
        n = n_scored(algo, protocol)
        text = _f(mean_of(algo, protocol))
        return f"**{text}**" if n == len(folds) else f"{text} ({n}/{len(folds)})"

    def of(algo, protocol):
        return scores[(scores.algo == algo) & (scores.protocol == protocol)]

    algos = sorted(set(scores.algo))
    protocols = set(scores.protocol)
    p2r_names = sorted(p for p in protocols if p.startswith("p2r_") and not p.endswith(EST))
    ships = [p for p in FINAL_ARMS if p in protocols] + p2r_names
    pool = "mean ARI" if aggregate == "mean" else "mean per-dataset rank"

    w("# HPO pilot -- one hyperparameter set per arm\n\n")
    w(f"Stream length: **{requested}** per dataset. Objective: ARI over the whole stream. "
      f"Sets are chosen on tuning seed {TUNE_SEED}; every reported score is the mean over "
      f"fresh runs under seeds {list(FINAL_SEEDS)} (their own permutations). Scores are "
      f"pooled across datasets by **{pool}** (`--aggregate {aggregate}`).\n\n")
    w(f"Datasets ({len(folds)}): {', '.join(folds)}\n\n")
    w("Every arm ends with **one set per algorithm**, run unchanged on every dataset.\n\n"
      "| arm | grid | how the one set is chosen |\n|---|---|---|\n"
      f"| arm 1 | raw | best {pool} over all datasets |\n"
      "| arm 2 | raw | leave-one-dataset-out; the set most rounds picked |\n"
      "| arm 3 | substituted (`c * s(D)`, micro-ratio) | as arm 2; the set is dimensionless "
      "and resolves per dataset |\n"
      "| P2R | substituted | shape as arm 3, rule's phi on all datasets; the resolution level "
      "is read off each dataset's own label-free curve |\n"
      f"| sf-all | substituted | best {pool} over all datasets (arm 1 on the substituted grid) |\n\n"
      "All arms use the labels of all datasets to choose, so their scores on those same "
      "datasets (section 2) are **in-sample**. What a procedure is worth on a dataset it "
      "never saw is its **LODO estimate** (section 3).\n")
    w(_grid_section(algos, grids_spec, grids, rejected, tuned_of))

    # -- the sets
    w("\n## 1. The one set each arm ships\n\n")
    for algo in algos:
        w(f"### {algo}\n\n| arm | set | how it was reached |\n|---|---|---|\n")
        for p in ships:
            info = finals.get(p, {}).get(algo)
            if info is None:
                w(f"| {arm_label(p)} | - | no set: nothing eligible on every dataset |\n")
                continue
            cand = info["config"]
            if p.startswith("p2r_"):
                axis = GRAN[algo][0]
                shape = spec_label(cand, grids_spec, tuned_of[algo], level=False)
                levels = ", ".join(
                    f"{f}: {_level_value(grids_spec, algo, lv)}"
                    for f, lv in sorted(info["levels"].items())
                ) or "-"
                phi = "" if info["phi"] is None else f"phi={_fmt_value(info['phi'])}; "
                how = (f"shape picked by {info['votes']}/{info['rounds']} rounds; "
                       f"{phi}`{axis}` per dataset -> {levels}")
                w(f"| {arm_label(p)} | {shape}; `{axis}` by {info['rule']} on "
                  f"{info['criterion']} | {how} |\n")
                continue
            if p == "default":
                how = "River constructor"
            elif "votes" in info:
                how = f"picked by {info['votes']}/{info['rounds']} LODO rounds"
            else:
                how = f"best {pool} over {len(folds)} datasets"
            w(f"| {arm_label(p)} | {spec_label(cand, grids_spec, tuned_of[algo])} | {how} |\n")
        w("\n")

    # -- the sets on every dataset
    w("\n## 2. The set on every dataset (in-sample)\n\n")
    head = ["algo", "dataset"] + [arm_label(p) for p in ships]
    w("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
    for algo in algos:
        for fold in folds:
            w(f"| {algo} | {fold} | " + " | ".join(_f(cell(algo, fold, p)) for p in ships) + " |\n")
        w(f"| **{algo}** | **mean** | "
          + " | ".join(mean_cell(algo, p) for p in ships) + " |\n")
    w("\nThreshold each set resolves to, per dataset (substituted sets differ by dataset "
      "by design):\n\n")
    head = ["algo", "dataset"] + [arm_label(p) for p in ships if p != "default"]
    w("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
    for algo in algos:
        for fold in folds:
            w(f"| {algo} | {fold} | "
              + " | ".join(_fmt_value(cell(algo, fold, p, "value")) if (algo, fold, p) in by
                           else "-" for p in ships if p != "default") + " |\n")
    finals_rows = scores[scores.protocol.isin(ships)]
    failures = finals_rows[finals_rows.n_final_ok < len(FINAL_SEEDS)]
    if len(failures):
        w("\nRuns that did not complete (counted as failures, not dropped):\n\n")
        for r in failures.itertuples(index=False):
            w(f"- {r.algo} / {r.dataset} / {arm_label(r.protocol)}: status `{r.final_status}`\n")
    missing = [(a, p) for a in algos for p in p2r_names if n_scored(a, p) < len(folds)]
    if missing:
        w("\nA P2R policy scores no value on a dataset where its curve is broken (a failed "
          "rung, or an infinite silhouette); compare the counts before the means: "
          + ", ".join(f"{a} {arm_label(p)} {n_scored(a, p)}/{len(folds)}" for a, p in missing)
          + ".\n")

    w("\nDifferences below are means over the datasets where both sides have a score.\n")
    w("\n| algo | arm 3 - arm 2 | sf-all - arm 1 | arm 2 - arm 1 | arm 3 - sf-all |"
      + "".join(f" {arm_label(p)} - arm 3 |" for p in p2r_names)
      + "\n|" + "---|" * (5 + len(p2r_names)) + "\n")
    with np.errstate(all="ignore"):
        for algo in algos:
            a1, a2, a3 = (vec(algo, p) for p in ("arm1_raw_all", "arm2_raw_lodo", "arm3_sf_lodo"))
            sfa = vec(algo, "sf_all")
            w(f"| {algo} | {_f(_nanmean(a3 - a2))} | {_f(_nanmean(sfa - a1))} | "
              f"{_f(_nanmean(a2 - a1))} | {_f(_nanmean(a3 - sfa))} |"
              + "".join(f" {_f(_nanmean(vec(algo, p) - a3))} |" for p in p2r_names) + "\n")
    w("\n*Arm 3 - arm 2*: same selection, same budget, only the units differ -- the effect "
      "of the substitutes. *sf-all - arm 1*: the same comparison when both simply take the "
      "best set over all datasets. *Arm 2 - arm 1* and *arm 3 - sf-all*: what the LODO "
      "consensus gives up against the plain best-over-all choice, per grid.\n")

    # -- LODO estimates
    w("\n## 3. LODO estimate: the procedure on a dataset it never saw\n\n")
    ests = [p for p in ("arm2_raw_lodo", "arm3_sf_lodo", *p2r_names) if p + EST in protocols]
    if ests:
        head = ["algo", "dataset"] + [arm_label(p) for p in ests]
        w("In each round the set is chosen without the dataset in the row and scored on it.\n\n")
        w("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for algo in algos:
            for fold in folds:
                w(f"| {algo} | {fold} | "
                  + " | ".join(_f(cell(algo, fold, p + EST)) for p in ests) + " |\n")
            w(f"| **{algo}** | **mean** | "
              + " | ".join(mean_cell(algo, p + EST) for p in ests) + " |\n")
        w("\n| algo | arm | in-sample mean | LODO estimate | optimism |\n|---|---|---|---|---|\n")
        for algo in algos:
            for p in ests:
                ins, est = mean_of(algo, p), mean_of(algo, p + EST)
                w(f"| {algo} | {arm_label(p)} | {_f(ins)} | {_f(est)} | {_f(ins - est)} |\n")
        w("\n*Optimism* is how much the in-sample score flatters the procedure. The LODO "
          "estimate of arm 3 minus that of arm 2 is the honest version of the substitutes' "
          "effect.\n")
    if "arm3_sf_cross" + EST in protocols:
        w("\n### Leave-one-group-out\n\n"
          "| algo | arm 3 within group | arm 3 across groups | gap |\n|---|---|---|---|\n")
        for algo in algos:
            within = mean_of(algo, "arm3_sf_lodo" + EST)
            across = mean_of(algo, "arm3_sf_cross" + EST)
            w(f"| {algo} | {_f(within)} | {_f(across)} | {_f(across - within)} |\n")

    # -- checks
    w("\n## 4. Health\n")
    verdicts = {}
    ok_tune = _ok(tune)

    w("\n### 4a. Does the grid separate configurations beyond seed noise?\n\n")
    w("Spread = max - min ARI over the grid on one dataset (tuning seed). Noise = std over "
      "three seeds of that dataset's best config.\n\n"
      "| algo | grid | median spread | median noise | ratio | verdict |\n|---|---|---|---|---|---|\n")
    for algo in algos:
        for grid, protocol in (("raw", "ceiling_raw"), ("ladder", "ceiling_sf")):
            sub = ok_tune[(ok_tune.algo == algo) & (ok_tune.grid == grid)]
            spread = sub.groupby("dataset")[OBJ].agg(lambda s: s.max() - s.min())
            med_s = float(spread.median()) if len(spread) else np.nan
            med_n = float(of(algo, protocol).seed_std.median())
            # No spread at all is a failure however small the noise: 0/0 must
            # not read as an infinite ratio.
            if not np.isfinite(med_s) or med_s <= 0:
                ratio = np.nan
            else:
                ratio = med_s / med_n if med_n > 0 else np.inf
            good = bool(np.isfinite(med_s) and med_s > 0 and ratio >= MIN_SPREAD_TO_NOISE)
            verdicts[f"4a {algo} {grid}"] = good
            w(f"| {algo} | {grid} | {_f(med_s)} | {_f(med_n)} | {_f(ratio, 1)} | "
              f"{'PASS' if good else 'FAIL'} |\n")
    w(f"\nPASS needs ratio >= {MIN_SPREAD_TO_NOISE}. A FAIL means choosing on this grid is "
      "mostly choosing noise.\n")

    w("\n### 4b. Does an optimum sit at an end of the resolution axis?\n\n")
    w("| algo | grid | per-dataset optima at an end | share | verdict | shipped sets at an end |\n"
      "|---|---|---|---|---|---|\n")
    for algo in algos:
        for grid, ceiling, shipped in (("raw", "ceiling_raw", ("arm1_raw_all", "arm2_raw_lodo")),
                                       ("ladder", "ceiling_sf", ("sf_all", "arm3_sf_lodo"))):
            n_levels = len(grids_spec[algo]["levels"][grid])
            levels = of(algo, ceiling).level.dropna()
            hits = int(levels.isin([1, n_levels]).sum())
            share = hits / len(levels) if len(levels) else np.nan
            good = bool(share < MAX_ENDPOINT_SHARE)
            verdicts[f"4b {algo} {grid}"] = good
            at_end = [arm_label(p) for p in shipped
                      if p in finals and algo in finals[p]
                      and finals[p][algo]["config"]["granularity_level"] in (1, n_levels)]
            w(f"| {algo} | {grid} | {hits}/{len(levels)} | {_f(share, 2)} | "
              f"{'PASS' if good else 'FAIL'} | {', '.join(at_end) or 'none'} |\n")
    w(f"\nPASS needs share < {MAX_ENDPOINT_SHARE}. An optimum at an end means the true "
      "optimum may lie outside the axis.\n")

    w("\n### 4c. Winner's curse: tuning score vs. fresh re-run of the shipped set\n\n")
    w("| algo | arm | mean(tune - re-run) | max |\n|---|---|---|---|\n")
    for algo in algos:
        for p in ("arm1_raw_all", "sf_all"):
            gap = of(algo, p).pipe(lambda d: d.tune_ari - d.final_ari)
            w(f"| {algo} | {arm_label(p)} | {_f(float(gap.mean()))} | {_f(float(gap.max()))} |\n")
    w("\nLarge means the set won partly on a lucky permutation, and a grid search reported "
      "without a re-run would publish exactly that number.\n")

    w("\n### 4d. How much do the LODO rounds agree?\n\n")
    w("| algo | arm | distinct sets over the rounds | votes for the shipped set |\n"
      "|---|---|---|---|\n")
    for algo in algos:
        for p in ("arm2_raw_lodo", "arm3_sf_lodo", *p2r_names):
            info = finals.get(p, {}).get(algo)
            rounds = of(algo, p + EST)
            if info is None or rounds.empty:
                continue
            distinct = (rounds.theta.nunique() if p.startswith("p2r_")
                        else rounds.config_id.nunique())
            what = "shapes" if p.startswith("p2r_") else "sets"
            w(f"| {algo} | {arm_label(p)} | {distinct} {what} | "
              f"{info['votes']}/{info['rounds']} |\n")
    w("\nFew votes for the shipped set means no single set transfers; the consensus is "
      "then a weak choice and the LODO estimate is the number to trust.\n")

    scaled = [a for a in algos if GRAN[a][1] == "scaled"]
    if scaled:
        w("\n### 4e. Does s(D) make the threshold transferable?\n\n")
        w("Best threshold per dataset (substituted grid, per-dataset optimum), spread across "
          "datasets.\n\n| algo | CV(c*) | CV(threshold*) | verdict |\n|---|---|---|---|\n")
        for algo in scaled:
            # On a dataset where every rung ties, the tie-break alone picks the
            # level; counting it would fake a perfectly stable optimum.
            sub = ok_tune[(ok_tune.algo == algo) & (ok_tune.grid == "ladder")]
            spread = sub.groupby("dataset")[OBJ].agg(lambda s: s.max() - s.min())
            separating = set(spread[spread > 0].index)
            o = of(algo, "ceiling_sf").dropna(subset=["g"])
            o = o[o.dataset.isin(separating)]
            if len(o) < 2:
                verdicts[f"4e {algo}"] = False
                w(f"| {algo} | - | - | FAIL -- fewer than 2 datasets where the ladder "
                  "separates configs |\n")
                continue
            cv_c, cv_abs = _cv(o.g), _cv(o.value)
            good = bool(np.isfinite(cv_c) and cv_c < cv_abs)
            verdicts[f"4e {algo}"] = good
            w(f"| {algo} | {_f(cv_c)} | {_f(cv_abs)} | {'PASS' if good else 'FAIL'} "
              f"({len(o)} datasets) |\n")
        w("\nIf the optimum does not vary less on `c` than in raw units, one `c` for every "
          "dataset has no more basis than one raw threshold.\n")

    w("\n### 4f. Does any shipped set beat River's default?\n\n")
    tuned_arms = [p for p in ships if p != "default"]
    w("| algo | " + " | ".join(f"{arm_label(p)} - default" for p in tuned_arms)
      + " |\n|---|" + "---|" * len(tuned_arms) + "\n")
    for algo in algos:
        d = vec(algo, "default")
        with np.errstate(all="ignore"):
            w(f"| {algo} | " + " | ".join(
                f"{_f(_nanmean(vec(algo, p) - d))} ({int(np.nansum(vec(algo, p) < d))} "
                f"datasets worse)" for p in tuned_arms) + " |\n")

    w("\n### 4g. Do the arms crown the same algorithm?\n\n")
    if len(algos) < 2:
        w("Needs two algorithms.\n")
    else:
        arms = [p for p in ("arm1_raw_all", "arm2_raw_lodo", "arm3_sf_lodo") if p in protocols]
        w("| dataset | default | " + " | ".join(arm_label(p) for p in arms)
          + " | agree |\n|---|---|" + "---|" * len(arms) + "---|\n")
        agree = total = 0
        for fold in folds:
            winners = {}
            for p in ("default", *arms):
                vals = {a: cell(a, fold, p) for a in algos}
                vals = {a: v for a, v in vals.items() if np.isfinite(v)}
                winners[p] = max(vals, key=vals.get) if len(vals) == len(algos) else None
            named = [winners[p] for p in arms]
            same = all(named) and len(set(named)) == 1
            if all(named):
                total += 1
                agree += int(same)
            w(f"| {fold} | {winners['default'] or '-'} | " + " | ".join(n or "-" for n in named)
              + f" | {'yes' if same else 'no'} |\n")
        w(f"\nAgreement: **{agree}/{total}**. Low agreement means the tuning method changes "
          "which algorithm wins, so a benchmark built on one of them is fragile.\n")

    w("\n## 5. Read these numbers with\n\n")
    dominated = [f for f in folds if probes[f].get("one_d_dominated")]
    w(f"- **1-D dominated datasets** (raw units, top-1 variance share > 0.9): "
      f"{', '.join(dominated) or 'none'}. Comparisons there describe one axis only.\n")
    w("- Dataset scales s(D): " + ", ".join(f"{f}={probes[f]['s']:.3g}" for f in folds)
      + ". One raw threshold is fine-grained on one dataset and coarse on another.\n")
    if aggregate == "mean":
        w("- Pooling by mean ARI lets a dataset with large ARI (kdd99) decide the set for "
          "all of them; `--aggregate rank` removes that.\n")
    by_group: dict[str, list[str]] = {}
    for f in folds:
        by_group.setdefault(FOLD_GROUPS.get(f, "other"), []).append(f)
    w("- LODO rounds tune only on the held-out dataset's own group -- "
      + "; ".join(f"{g}: {', '.join(fs)}" for g, fs in by_group.items()) + ".\n")
    w("- CluStream's resolution axis is a count and has no scale conversion; its substitute "
      "is the micro-ratio only.\n")
    w("- CluStream's offline phase is k-means and DenStream's is DBSCAN-like: an ARI gap "
      "between the two partly compares the offline phases.\n")
    w("- One tuning run per grid point and two scoring seeds: a pilot check of direction, "
      "not a statistical result.\n")

    w("\n## Verdict summary\n\n")
    for name, good in verdicts.items():
        w(f"- {name}: {'PASS' if good else 'FAIL'}\n")
    return "".join(out)


def _level_value(grids_spec: dict, algo: str, level: int) -> str:
    axis, kind = GRAN[algo]
    value = grids_spec[algo]["levels"]["ladder"][level - 1]
    return f"c={value:g}" if kind == "scaled" else f"{value:g}"


def write_final_sets(path, finals, grids_spec, tuned_of, probes, folds) -> None:
    """The shipped sets, machine-readable: the spec and what it resolves to per dataset."""
    payload = {}
    for protocol, per_algo in finals.items():
        for algo, info in per_algo.items():
            cand = info["config"]
            entry = {
                "arm": arm_label(protocol),
                "set": spec_label(cand, grids_spec, tuned_of[algo],
                                  level=not protocol.startswith("p2r_")),
                "config_id": cand["config_id"],
                "grid": cand["grid"],
            }
            for key in ("votes", "rounds", "rule", "criterion", "phi", "theta"):
                if key in info:
                    entry[key] = info[key]
            if protocol.startswith("p2r_"):
                entry["level_per_dataset"] = info["levels"]
                entry["resolved"] = {}
                for fold, level in info["levels"].items():
                    cell = {**cand, "granularity_level": level,
                            "g": grids_spec[algo]["levels"]["ladder"][level - 1]}
                    entry["resolved"][fold] = resolve_candidate(cell, probes[fold])
            else:
                entry["resolved"] = {f: resolve_candidate(cand, probes[f]) for f in folds}
            payload.setdefault(algo, {})[protocol] = entry
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


# --- main ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", default=",".join(ALGOS))
    parser.add_argument("--folds", default="", help="explicit folds; overrides --groups")
    parser.add_argument(
        "--groups",
        default="real",
        help="fold groups to use when --folds is not given. Default is real only: the "
        "synthetic folds shared one generator and seed and are being regenerated",
    )
    parser.add_argument("--grids", default="",
                        help="JSON {algo: {levels|shape: {raw|ladder: ...}}} overriding GRIDS")
    parser.add_argument("--n-samples", type=int, default=0, help="0 = full stream")
    parser.add_argument("--out-dir", default="",
                        help=f"where runs.jsonl and the report go (default {OUT_DIR})")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("all", "tune", "score"),
        default="all",
        help="tune: only the grid runs (no report); score: only the re-runs of picked "
        "configs, then the report; all: both",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-queue runs whose latest row is not ok (e.g. timeouts caused by too low "
        "a --timeout); the retry supersedes the failed row",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="list the runs that would execute, run nothing, write nothing")
    parser.add_argument(
        "--aggregate",
        choices=AGGREGATES,
        default="mean",
        help="how scores are pooled across datasets when one set is chosen: mean ARI, or "
        "mean per-dataset rank (one easy dataset cannot outvote the others)",
    )
    parser.add_argument(
        "--p2r",
        default=P2R_DEFAULT,
        help="comma list of rule:criterion P2R readings; empty string disables",
    )
    args = parser.parse_args()
    p2r = parse_p2r(args.p2r)
    if args.out_dir:
        set_out_dir(args.out_dir)

    # The hash seed every run needs (RIVER_BUGS.md BUG-2) is pinned by
    # build_matrix.execute for each subprocess, so nothing to set here.
    require(PROBE_JSON, "Stage 2 (probe.py)")
    probes = read_json(PROBE_JSON)
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    wanted_groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    folds = (
        [f.strip() for f in args.folds.split(",") if f.strip()]
        if args.folds
        else [f for f in FOLDS if FOLD_GROUPS[f] in wanted_groups]
    )
    missing = [f for f in folds if f not in probes]
    if missing:
        raise SystemExit(f"no probe for {missing}; run stream.py and probe.py for them first")
    groups = {f: FOLD_GROUPS.get(f, "other") for f in folds}
    sizes = {g: list(groups.values()).count(g) for g in set(groups.values())}
    if any(n < 3 for n in sizes.values()):
        raise SystemExit(f"LODO within a group needs at least 3 folds per group; got {sizes}")
    requested = args.n_samples or "full"

    grids_spec = merge_grids(GRIDS, json.loads(args.grids) if args.grids else None)
    unknown = [a for a in algos if a not in grids_spec]
    if unknown:
        raise SystemExit(f"no grid for {unknown}; add it to GRIDS or pass --grids")
    grids: dict[str, dict[str, list[dict]]] = {"raw": {}, "ladder": {}}
    rejected: dict[str, dict[str, list[dict]]] = {"raw": {}, "ladder": {}}
    for a in algos:
        for kind in ("raw", "ladder"):
            grids[kind][a], rejected[kind][a] = build_grid(a, kind, grids_spec[a])
    tuned_of = {a: tuned_params(a, grids_spec) for a in algos}
    defaults = {a: default_candidate(a) for a in algos}
    tuning = [c for grid in grids.values() for cands in grid.values() for c in cands]
    candidates = {(c["algo"], c["config_id"]): c for c in [*tuning, *defaults.values()]}
    grid_of = {key: c["grid"] for key, c in candidates.items()}
    for a in algos:
        print(f"{a}: raw {len(grids['raw'][a])} points ({len(rejected['raw'][a])} rejected), "
              f"ladder {len(grids['ladder'][a])} points ({len(rejected['ladder'][a])} rejected); "
              f"tuned: {', '.join(tuned_of[a])}")

    running = not args.report_only
    if running and not args.dry_run:
        print("materialising permutations")
        for fold in folds:
            for seed in (TUNE_SEED, *FINAL_SEEDS):
                materialise(fold, shuffle_seed_for(seed))

    if running and args.phase in ("all", "tune"):
        print(f"tuning: {len(tuning)} grid points x {len(folds)} folds")
        run_jobs(
            [_job(c["algo"], c, f, TUNE_SEED, args.n_samples, probes, tuned_of[c["algo"]])
             for c in tuning for f in folds],
            args.workers,
            args.timeout,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
    if running and args.phase == "tune":
        print("tuning phase finished; run again with --phase score to re-run the picks "
              "and write the report")
        return

    df = load_runs(requested, grid_of, candidates, probes)
    tune = df[(df.seed == TUNE_SEED) & (df.grid != "default")
              & df.algo.isin(algos) & df.dataset.isin(folds)]
    picks, finals = all_picks(tune, grids, defaults, algos, folds, groups, p2r, args.aggregate)

    if running:
        wanted = {
            (algo, cand["config_id"], fold): cand
            for chosen in picks.values()
            for (algo, fold), cand in chosen.items()
        }
        print(f"scoring: {len(wanted)} picked (algo, config, fold) x seeds {list(FINAL_SEEDS)}")
        if args.dry_run:
            print("  (picks come from the tuning rows on disk now; a tuning retry that "
                  "has not run yet can still change them)")
        run_jobs(
            [
                _job(algo, cand, fold, seed, args.n_samples, probes, tuned_of[algo])
                for (algo, _, fold), cand in wanted.items()
                for seed in FINAL_SEEDS
            ],
            args.workers,
            args.timeout,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            return
        df = load_runs(requested, grid_of, candidates, probes)

    scores = score_table(df, picks, probes, tuned_of)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{requested}" if args.aggregate == "mean" else f"{requested}_{args.aggregate}"
    scores.to_csv(OUT_DIR / f"scores_{tag}.csv", index=False)
    write_final_sets(OUT_DIR / f"final_sets_{tag}.json", finals, grids_spec, tuned_of,
                     probes, folds)
    _ok(tune).pivot_table(
        index=["algo", "grid", "dataset", "theta"], columns="granularity_level", values=OBJ
    ).to_csv(OUT_DIR / f"tune_grid_{requested}.csv")
    report = build_report(tune, scores, folds, probes, requested, grids, grids_spec,
                          rejected, tuned_of, finals, args.aggregate)
    path = OUT_DIR / ("report.md" if tag == "full" else f"report_{tag}.md")
    path.write_text(report, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
