"""Pilot: plain grid search vs. the pipeline's HPO direction, on River's own SOTA.

Before the pipeline is scaled to the whole pool, it is checked against the
simplest thing it has to beat, on the two stream clusterers River itself ships
-- ``CluStream`` and ``DenStream`` -- where no estimator written for this project
can muddy the picture.

Both methods search only the one parameter that dominates, over a full grid.
Every other parameter stays at River's *constructor* default.

* **Metric 1 -- plain grid search.** A hand-given grid in raw units
  (``n_macro_clusters`` in {10, 20, 50, 100}, ``epsilon`` in {0.1, 0.25, 0.5, 1}),
  tuned on the fold itself.
* **Metric 2 -- the pipeline's direction.** Stage 3's scale-free ladder
  (``epsilon = c * s(D)``; CluStream's count ladder, which Stage 3 leaves
  unconverted) chosen by leave-one-dataset-out, so the fold's own labels are
  never read.

The same runs also give the two other cells of the 2 x 2, which is what makes the
gap between the metrics readable:

                   tuned on the fold      tuned on the other folds
    raw grid       metric 1               raw LODO
    scale-free     scale-free oracle      metric 2

Tuning runs every grid point over the full stream under seed 1. Every chosen
configuration then runs over the full stream again under seeds 2 and 3, each
with its own permutation -- re-running under the tuning seed would only repeat
the tuning score. River's default is scored the same way as the baseline.

Run from the repo root::

    python -m hpo_pipeline.hpo_compare                    # full streams
    python -m hpo_pipeline.hpo_compare --n-samples 2000   # smoke test
    python -m hpo_pipeline.hpo_compare --report-only
    python -m hpo_pipeline.hpo_compare --m1-grid '{"DenStream": [0.05, 0.5, 5]}'

    # re-run only what timed out, with a longer limit; check first with --dry-run
    python -m hpo_pipeline.hpo_compare --phase tune --retry-failed --timeout 7200
    python -m hpo_pipeline.hpo_compare --phase score --retry-failed --timeout 7200

P2R readings (a label-free rule re-picks the ladder level on each held-out fold)
are added to the report by default; ``--p2r`` chooses which, ``--p2r ""`` none.

Folds must already have a Stage 2 probe (``configs/probe.json``). Missing
permutations for the scoring seeds are materialised on the way.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ._common import (
    FOLD_GROUPS,
    FOLDS,
    HPO_OUT,
    N_GRANULARITY,
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
from .space import GRAN, granularity_value, resolve
from .stream import materialise

ALGOS = ("CluStream", "DenStream")
#: Metric 1's grid, in raw units. Overridable with --m1-grid.
M1_GRID = {
    "CluStream": [10, 20, 50, 100],
    "DenStream": [0.1, 0.25, 0.5, 1.0],
}
TUNE_SEED = 1
FINAL_SEEDS = (2, 3)
OBJ = "ari"

OUT_DIR = HPO_OUT / "compare"
RUNS_JSONL = OUT_DIR / "runs.jsonl"

#: Readings: name -> (grid, how the config is chosen).
PROTOCOLS = {
    "default": (None, None),
    "m1_grid": ("raw", "oracle"),
    "raw_lodo": ("raw", "lodo"),
    "sf_oracle": ("ladder", "oracle"),
    "m2_pipeline": ("ladder", "lodo"),
    # Leave-one-GROUP-out: tuned on the other groups only. Read only when the
    # run has at least two groups.
    "m2_cross": ("ladder", "cross"),
}

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


def _candidate(algo, level, g, shape, *, grid, id_level=None) -> dict:
    return {
        "algo": algo,
        "grid": grid,
        "granularity_level": level,
        "theta": 1 if level is not None else None,
        "g": g,
        "shape": dict(shape),
        "config_id": config_id(algo, level if id_level is None else id_level, g, shape),
        "scaler": SCALER,
        "scaler_params": dict(SCALER_PARAMS),
        "is_default": grid == "default",
        "raw": grid == "raw",
    }


def ladder_grid(algo: str) -> list[dict]:
    """Metric 2's grid: Stage 3's ladder, resolved per fold by ``resolve``."""
    axis, _ = GRAN[algo]
    # The axis parameter is set by `resolve`, so it stays out of the shape and
    # the config id describes only the ladder position.
    fixed = {k: v for k, v in river_defaults(algo).items() if k != axis}
    return [
        _candidate(algo, level, granularity_value(algo, level), fixed, grid="ladder")
        for level in range(1, N_GRANULARITY + 1)
    ]


def raw_grid(algo: str, values) -> list[dict]:
    """Metric 1's grid: the same absolute value on every fold."""
    axis, _ = GRAN[algo]
    fixed = {k: v for k, v in river_defaults(algo).items() if k != axis}
    out = []
    for level, value in enumerate(values, start=1):
        shape = {**fixed, axis: value}
        # The id is tagged so a raw value can never collide with a ladder rung.
        out.append(_candidate(algo, level, value, shape, grid="raw", id_level=f"raw{level}"))
    return out


def default_candidate(algo: str) -> dict:
    return _candidate(algo, None, None, river_defaults(algo), grid="default")


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


def describe(job: dict) -> str:
    """Readable label for a run: the searched value and where it came from."""
    algo, cand = job["algo"], job["candidate"]
    axis, kind = GRAN[algo]
    value = job["params"].get(axis)
    shown = f"{value:.4g}" if isinstance(value, float) else str(value)
    if cand["grid"] == "default":
        where = "default"
    elif cand["grid"] == "raw":
        where = f"raw #{cand['granularity_level']}"
    elif kind == "scaled":
        where = f"ladder L{cand['granularity_level']} c={cand['g']:g}"
    else:
        where = f"ladder L{cand['granularity_level']}"
    return f"{axis}={shown} [{where}]"


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
        row = execute(job, timeout)
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


def _job(algo, candidate, fold, seed, n_samples, probes) -> dict:
    return {
        "algo": algo,
        "candidate": candidate,
        "fold": fold,
        "seed": seed,
        "budget": n_samples or None,
        "requested": n_samples or "full",
        "params": resolve(candidate, probes[fold]),
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
        if _params_blob(r.get("params_resolved")) == _params_blob(resolve(cand, probes[r["dataset"]])):
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


def _first(frame: pd.DataFrame, score: str, runtime: str) -> int:
    """Deterministic argmax: score desc, then runtime asc, then coarser level."""
    ranked = frame.sort_values(
        [score, runtime, "granularity_level"], ascending=[False, True, True], kind="mergesort"
    )
    return int(ranked.iloc[0]["granularity_level"])


def pick_oracle(tune: pd.DataFrame) -> dict:
    """(algo, fold) -> level chosen on that fold's own labels."""
    return {
        (algo, fold): _first(grp, OBJ, "runtime_s")
        for (algo, fold), grp in _ok(tune).groupby(["algo", "dataset"])
    }


def pick_lodo(tune: pd.DataFrame, folds: list[str], groups: dict | None = None,
              cross: bool = False) -> dict:
    """(algo, fold) -> level chosen on OTHER folds.

    ``groups`` (fold -> group) keeps the optimisation set inside the held-out
    fold's own group, so folds cut from one generator cannot outvote the real
    ones. ``cross=True`` instead takes it from the other groups only -- the
    honest transfer number. Without ``groups`` every other fold is used.

    A level is eligible only if it ran on every optimisation fold; the held-out
    fold is never consulted, so a pick that then fails there is a real failure.
    """
    ok = _ok(tune)
    picks = {}
    for algo in sorted(set(tune.algo)):
        sub = ok[ok.algo == algo]
        for fold in folds:
            if groups is None:
                opt = [f for f in folds if f != fold]
            elif cross:
                opt = [f for f in folds if groups[f] != groups[fold]]
            else:
                opt = [f for f in folds if f != fold and groups[f] == groups[fold]]
            if not opt:
                continue
            agg = (
                sub[sub.dataset.isin(opt)]
                .groupby("granularity_level")
                .agg(score=(OBJ, "mean"), runtime=("runtime_s", "mean"), n=("dataset", "nunique"))
            )
            agg = agg[agg.n == len(opt)]
            if not agg.empty:
                picks[(algo, fold)] = _first(agg.reset_index(), "score", "runtime")
    return picks


def _curve(tune: pd.DataFrame, algo: str, fold: str, criterion: str, n_levels: int):
    """A label-free criterion along the ladder on one fold, bigger-is-better.

    None when any rung is missing or not finite. A timed-out rung, or River's
    streaming silhouette returning inf, breaks the curve rather than shortening
    it -- a rule read off a gappy curve would pick from the wrong positions.
    """
    sub = tune[(tune.algo == algo) & (tune.dataset == fold) & (tune.status == "ok")]
    if criterion not in sub.columns or sub.empty:
        return None
    levels = sub.granularity_level.astype(int)
    series = pd.Series(pd.to_numeric(sub[criterion], errors="coerce").to_numpy(),
                       index=levels.to_numpy()).reindex(range(1, n_levels + 1))
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return orient(values, criterion)


def _ari_at(tune: pd.DataFrame, algo: str, fold: str, level: int) -> float:
    sub = _ok(tune[(tune.algo == algo) & (tune.dataset == fold)])
    hit = sub[sub.granularity_level.astype(int) == level][OBJ]
    return float(hit.iloc[0]) if len(hit) else np.nan


def _rule_quality(tune, algo, opt, rule, phi, criterion, n_levels) -> float:
    """Mean ARI the rule reaches on the optimisation folds with this phi."""
    vals = []
    for fold in opt:
        curve = _curve(tune, algo, fold, criterion, n_levels)
        if curve is not None:
            vals.append(_ari_at(tune, algo, fold, rule(curve, phi) + 1))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else -np.inf


def pick_p2r(tune: pd.DataFrame, folds: list[str], groups: dict | None,
             rule_name: str, criterion: str, n_levels: int = N_GRANULARITY) -> dict:
    """P2R: (algo, fold) -> ladder level re-picked on that fold by a rule.

    Metric 2 transfers a *value* of c. P2R transfers a *rule*: the level is
    chosen again on the held-out fold, from that fold's own label-free curve.
    The rule's free parameter (phi, for R_parsimony) is the only thing fitted,
    and it is fitted on the other folds of the same group using their labels --
    so, like metric 2, nothing about the held-out fold's labels is read.

    A fold whose curve is broken gets no pick rather than a guess.
    """
    rule, phis = RULES[rule_name]
    picks = {}
    for algo in sorted(set(tune.algo)):
        for fold in folds:
            curve = _curve(tune, algo, fold, criterion, n_levels)
            if curve is None:
                continue
            opt = [f for f in folds
                   if f != fold and (groups is None or groups[f] == groups[fold])]
            # Ties keep the first phi, so the choice is deterministic.
            phi = max(phis, key=lambda p: _rule_quality(tune, algo, opt, rule, p,
                                                        criterion, n_levels))
            picks[(algo, fold)] = rule(curve, phi) + 1
    return picks


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


def all_picks(tune, grids, defaults, algos, folds, groups, p2r=()) -> dict:
    """protocol -> {(algo, fold): candidate}."""
    out = {}
    n_groups = len(set(groups.values()))
    for protocol, (grid, how) in PROTOCOLS.items():
        if grid is None:
            out[protocol] = {(a, f): defaults[a] for a in algos for f in folds}
            continue
        sub = tune[tune.grid == grid]
        if how == "oracle":
            levels = pick_oracle(sub)
        elif how == "lodo":
            levels = pick_lodo(sub, folds, groups)
        elif n_groups >= 2:
            levels = pick_lodo(sub, folds, groups, cross=True)
        else:
            continue
        out[protocol] = {k: grids[grid][k[0]][lv - 1] for k, lv in levels.items()}
    ladder = tune[tune.grid == "ladder"]
    for rule_name, criterion in p2r:
        levels = pick_p2r(ladder, folds, groups, rule_name, criterion)
        out[f"p2r_{rule_name}_{criterion}"] = {
            k: grids["ladder"][k[0]][lv - 1] for k, lv in levels.items()
        }
    return out


# --- scoring ---------------------------------------------------------------
def score_table(df, picks, probes) -> pd.DataFrame:
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
            rows.append(
                {
                    "algo": algo,
                    "dataset": fold,
                    "protocol": protocol,
                    "grid": cand["grid"],
                    "level": cand["granularity_level"],
                    "g": cand["g"],
                    "param": axis,
                    "value": resolve(cand, probes[fold]).get(axis),
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


def _pick_label(row) -> str:
    if row is None or row.get("value") is None or pd.isna(row.get("value")):
        return "-"
    value = row["value"]
    shown = f"{value:.4g}" if isinstance(value, float) else str(value)
    if row["grid"] == "ladder" and GRAN[row["algo"]][1] == "scaled":
        return f"{shown} (c={row['g']:g})"
    return shown


def _cv(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.std() / abs(v.mean())) if len(v) > 1 and v.mean() else np.nan


def build_report(tune, scores, folds, probes, requested, grids) -> str:
    out = []
    w = out.append
    by = {(r.algo, r.dataset, r.protocol): r._asdict() for r in scores.itertuples(index=False)}

    def cell(algo, fold, protocol, col="final_ari"):
        row = by.get((algo, fold, protocol))
        return np.nan if row is None else row[col]

    def mean_of(algo, protocol):
        return float(np.nanmean([cell(algo, f, protocol) for f in folds]))

    def of(algo, protocol):
        return scores[(scores.algo == algo) & (scores.protocol == protocol)]

    algos = sorted(set(scores.algo))
    w("# HPO pilot -- plain grid search vs. the pipeline's direction\n\n")
    w(f"Stream length: **{requested}** per fold. Objective: ARI over the whole stream. "
      f"Tuning seed {TUNE_SEED}; every reported score is the mean over fresh runs "
      f"under seeds {list(FINAL_SEEDS)} (their own permutations). Only one axis is "
      "searched; every other parameter is River's constructor default.\n\n")
    w(f"Folds ({len(folds)}): {', '.join(folds)}\n\n")
    w("| algo | metric 1 grid (raw, tuned on the fold) | metric 2 grid (scale-free, LODO) |\n"
      "|---|---|---|\n")
    for algo in algos:
        raw = ", ".join(f"{c['g']:g}" for c in grids["raw"][algo])
        ladder = ", ".join(f"{c['g']:g}" for c in grids["ladder"][algo])
        unit = "c, epsilon = c * s(D)" if GRAN[algo][1] == "scaled" else "count, no conversion"
        w(f"| {algo} | `{GRAN[algo][0]}` in {{{raw}}} | {unit}: {{{ladder}}} |\n")
    w("\nThe grids differ in size, so metric 1 and metric 2 do not spend the same budget; "
      "the scale-free-oracle column below is what isolates that.\n")

    # -- main table
    w("\n## 1. Scores\n\n")
    w("| algo | fold | default | **metric 1** grid | **metric 2** pipeline | raw LODO | "
      "scale-free oracle | M1 pick | M2 pick |\n|---|---|---|---|---|---|---|---|---|\n")
    for algo in algos:
        for fold in folds:
            w(f"| {algo} | {fold} | {_f(cell(algo, fold, 'default'))} | "
              f"**{_f(cell(algo, fold, 'm1_grid'))}** | **{_f(cell(algo, fold, 'm2_pipeline'))}** | "
              f"{_f(cell(algo, fold, 'raw_lodo'))} | {_f(cell(algo, fold, 'sf_oracle'))} | "
              f"{_pick_label(by.get((algo, fold, 'm1_grid')))} | "
              f"{_pick_label(by.get((algo, fold, 'm2_pipeline')))} |\n")
        w(f"| **{algo}** | **mean** | "
          + " | ".join(f"**{_f(mean_of(algo, p))}**"
                       for p in ("default", "m1_grid", "m2_pipeline", "raw_lodo", "sf_oracle"))
          + " | | |\n")
    failures = scores[scores.n_final_ok < len(FINAL_SEEDS)]
    if len(failures):
        w("\nRuns that did not complete (counted as failures, not dropped):\n\n")
        for r in failures.itertuples(index=False):
            w(f"- {r.algo} / {r.dataset} / {r.protocol}: status `{r.final_status}`\n")

    # -- headline
    w("\n## 2. Metric 1 vs. metric 2\n\n")
    w("| algo | M2 - M1 (mean) | folds M2 >= M1 | grid design: sf oracle - M1 | "
      "LODO cost with s(D): M2 - sf oracle | LODO cost raw: raw LODO - M1 |\n"
      "|---|---|---|---|---|---|\n")
    for algo in algos:
        m1 = np.array([cell(algo, f, "m1_grid") for f in folds])
        m2 = np.array([cell(algo, f, "m2_pipeline") for f in folds])
        sfo = np.array([cell(algo, f, "sf_oracle") for f in folds])
        rl = np.array([cell(algo, f, "raw_lodo") for f in folds])
        both = np.isfinite(m1) & np.isfinite(m2)
        w(f"| {algo} | {_f(float(np.nanmean(m2 - m1)))} | {int((m2[both] >= m1[both]).sum())}/"
          f"{int(both.sum())} | {_f(float(np.nanmean(sfo - m1)))} | "
          f"{_f(float(np.nanmean(m2 - sfo)))} | {_f(float(np.nanmean(rl - m1)))} |\n")
    w("\nHow to read it. M2 - M1 is the headline but mixes two things. *Grid design* says "
      "whether the scale-free ladder itself finds better configs when both may look at "
      "the fold. The two *LODO cost* columns say what not looking costs: if it is much "
      "smaller with s(D) than in raw units, s(D) is what makes transfer work -- the "
      "pipeline's central claim. Metric 1 sees the test fold's labels and metric 2 does "
      "not, so M2 matching M1 is already a win for the pipeline.\n")

    if "m2_cross" in set(scores.protocol):
        w("\n### Leave-one-group-out\n\n"
          "| algo | M2 within group | M2 across groups | gap |\n|---|---|---|---|\n")
        for algo in algos:
            within, across = mean_of(algo, "m2_pipeline"), mean_of(algo, "m2_cross")
            w(f"| {algo} | {_f(within)} | {_f(across)} | {_f(across - within)} |\n")
        w("\nAcross groups is the honest transfer estimate: nothing from the held-out "
          "fold's group is seen while tuning.\n")

    p2r_names = sorted(p for p in set(scores.protocol) if p.startswith("p2r_"))
    if p2r_names:
        w("\n### P2R -- re-pick the ladder level on each fold with a label-free rule\n\n")
        w("Metric 2 carries one value of c to the held-out fold. P2R carries a *rule*: "
          "the level is chosen again on the held-out fold from its own label-free curve; "
          "only the rule's phi (R_parsimony) is fitted, on the other folds of the group. "
          "No label of the held-out fold is read, so P2R is as deployable as metric 2.\n\n")
        w("| algo | reading | mean ARI | minus M1 | minus M2 | folds picked | picks |\n"
          "|---|---|---|---|---|---|---|\n")
        with np.errstate(all="ignore"):
            for algo in algos:
                m1 = np.array([cell(algo, f, "m1_grid") for f in folds])
                m2 = np.array([cell(algo, f, "m2_pipeline") for f in folds])
                for p in p2r_names:
                    rows = of(algo, p)
                    vals = np.array([cell(algo, f, p) for f in folds])
                    picked = ", ".join(f"{r.dataset}:L{int(r.level)}" for r in rows.itertuples())
                    w(f"| {algo} | {p[len('p2r_'):]} | {_f(float(np.nanmean(vals)))} | "
                      f"{_f(float(np.nanmean(vals - m1)))} | {_f(float(np.nanmean(vals - m2)))} | "
                      f"{len(rows)}/{len(folds)} | {picked or '-'} |\n")
        w("\nA fold with no pick has a broken curve there (a failed rung, or an infinite "
          "silhouette) and drops out of that reading's mean -- compare the counts before "
          "comparing the means. Pre-registered expectation (rules.py): R_max on "
          "matr_trace_win picks the finest level and loses; R_max on silhouette does not "
          "degenerate.\n")

    # -- checks
    w("\n## 3. Health of each grid\n")
    verdicts = {}
    ok_tune = _ok(tune)

    w("\n### 3a. Does the axis separate configurations beyond seed noise?\n\n")
    w("Spread = max - min ARI over the grid (tuning seed). Noise = std of the oracle "
      "pick over all three seeds.\n\n"
      "| algo | grid | median spread | median noise | ratio | verdict |\n|---|---|---|---|---|---|\n")
    for algo in algos:
        for grid, protocol in (("raw", "m1_grid"), ("ladder", "sf_oracle")):
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
            verdicts[f"3a {algo} {grid}"] = good
            w(f"| {algo} | {grid} | {_f(med_s)} | {_f(med_n)} | {_f(ratio, 1)} | "
              f"{'PASS' if good else 'FAIL'} |\n")
    w(f"\nPASS needs ratio >= {MIN_SPREAD_TO_NOISE}. A FAIL means picking on this grid is "
      "mostly picking noise. It shows whether the axis matters; it cannot show the "
      "untuned parameters matter less.\n")

    w("\n### 3b. Does the oracle optimum sit at an end of the grid?\n\n")
    w("| algo | grid | endpoint picks | share | verdict |\n|---|---|---|---|---|\n")
    for algo in algos:
        for grid, protocol in (("raw", "m1_grid"), ("ladder", "sf_oracle")):
            n_levels = len(grids[grid][algo])
            levels = of(algo, protocol).level.dropna()
            hits = int(levels.isin([1, n_levels]).sum())
            share = hits / len(levels) if len(levels) else np.nan
            good = bool(share < MAX_ENDPOINT_SHARE)
            verdicts[f"3b {algo} {grid}"] = good
            w(f"| {algo} | {grid} | {hits}/{len(levels)} | {_f(share, 2)} | "
              f"{'PASS' if good else 'FAIL'} |\n")
    w(f"\nPASS needs share < {MAX_ENDPOINT_SHARE}. An optimum at an end means the true "
      "optimum may lie outside the grid, so that grid's score is only a floor.\n")

    w("\n### 3c. Winner's curse: tuning score vs. fresh re-run\n\n")
    w("| algo | metric 1: mean(tune - re-run) | max | scale-free oracle: mean | max |\n"
      "|---|---|---|---|---|\n")
    for algo in algos:
        g1 = of(algo, "m1_grid").pipe(lambda d: d.tune_ari - d.final_ari)
        g2 = of(algo, "sf_oracle").pipe(lambda d: d.tune_ari - d.final_ari)
        w(f"| {algo} | {_f(float(g1.mean()))} | {_f(float(g1.max()))} | "
          f"{_f(float(g2.mean()))} | {_f(float(g2.max()))} |\n")
    w("\nLarge means the tuning score was partly a lucky permutation -- and grid search "
      "reported without a re-run would publish exactly that number.\n")

    w("\n### 3d. Does a single value transfer?\n\n")
    w("| algo | distinct raw-LODO picks | distinct M2 picks | M2 pick = sf-oracle pick |\n"
      "|---|---|---|---|\n")
    for algo in algos:
        m2 = of(algo, "m2_pipeline").set_index("dataset")
        sfo = of(algo, "sf_oracle").set_index("dataset")
        common = m2.index.intersection(sfo.index)
        same = int((m2.loc[common, "level"] == sfo.loc[common, "level"]).sum())
        w(f"| {algo} | {of(algo, 'raw_lodo').level.nunique()} | {m2.level.nunique()} | "
          f"{same}/{len(common)} |\n")

    scaled = [a for a in algos if GRAN[a][1] == "scaled"]
    if scaled:
        w("\n### 3e. Does s(D) make the threshold transferable?\n\n")
        w("Best threshold per fold (scale-free oracle), spread across folds.\n\n"
          "| algo | CV(c*) | CV(epsilon*) | verdict |\n|---|---|---|---|\n")
        for algo in scaled:
            # On a fold where every rung ties, the tie-break alone picks the
            # level; counting it would fake a perfectly stable optimum.
            sub = ok_tune[(ok_tune.algo == algo) & (ok_tune.grid == "ladder")]
            spread = sub.groupby("dataset")[OBJ].agg(lambda s: s.max() - s.min())
            separating = set(spread[spread > 0].index)
            o = of(algo, "sf_oracle").dropna(subset=["g"])
            o = o[o.dataset.isin(separating)]
            if len(o) < 2:
                verdicts[f"3e {algo}"] = False
                w(f"| {algo} | - | - | FAIL -- fewer than 2 folds where the ladder "
                  "separates configs |\n")
                continue
            cv_c, cv_abs = _cv(o.g), _cv(o.value)
            good = bool(np.isfinite(cv_c) and cv_c < cv_abs)
            verdicts[f"3e {algo}"] = good
            w(f"| {algo} | {_f(cv_c)} | {_f(cv_abs)} | {'PASS' if good else 'FAIL'} "
              f"({len(o)} folds) |\n")
        w("\nIf the optimum does not vary less on `c` than in raw units, LODO over `c` has "
          "no more basis than LODO over raw epsilon.\n")

    w("\n### 3f. Does any tuning beat River's default?\n\n")
    w("| algo | M1 - default | M2 - default | folds where M2 < default |\n|---|---|---|---|\n")
    for algo in algos:
        d = np.array([cell(algo, f, "default") for f in folds])
        m1 = np.array([cell(algo, f, "m1_grid") for f in folds])
        m2 = np.array([cell(algo, f, "m2_pipeline") for f in folds])
        w(f"| {algo} | {_f(float(np.nanmean(m1 - d)))} | {_f(float(np.nanmean(m2 - d)))} | "
          f"{int(np.nansum(m2 < d))}/{len(folds)} |\n")

    w("\n### 3g. Do both metrics crown the same algorithm?\n\n")
    if len(algos) < 2:
        w("Needs two algorithms.\n")
    else:
        w("| fold | default | metric 1 | metric 2 | agree |\n|---|---|---|---|---|\n")
        agree = total = 0
        for fold in folds:
            winners = {}
            for p in ("default", "m1_grid", "m2_pipeline"):
                vals = {a: cell(a, fold, p) for a in algos}
                vals = {a: v for a, v in vals.items() if np.isfinite(v)}
                winners[p] = max(vals, key=vals.get) if len(vals) == len(algos) else None
            same = winners["m1_grid"] is not None and winners["m1_grid"] == winners["m2_pipeline"]
            if winners["m1_grid"] and winners["m2_pipeline"]:
                total += 1
                agree += int(same)
            w(f"| {fold} | {winners['default'] or '-'} | {winners['m1_grid'] or '-'} | "
              f"{winners['m2_pipeline'] or '-'} | {'yes' if same else 'no'} |\n")
        w(f"\nAgreement: **{agree}/{total}**. Low agreement means the choice of HPO method "
          "changes which algorithm wins, so a benchmark built on one of them is fragile.\n")

    w("\n## 4. Read these numbers with\n\n")
    dominated = [f for f in folds if probes[f].get("one_d_dominated")]
    w(f"- **1-D dominated folds** (raw units, top-1 variance share > 0.9): "
      f"{', '.join(dominated) or 'none'}. Comparisons there describe one axis only.\n")
    w("- Fold scales s(D): " + ", ".join(f"{f}={probes[f]['s']:.3g}" for f in folds)
      + ". A raw epsilon grid is fine-grained on one fold and coarse on another.\n")
    by_group: dict[str, list[str]] = {}
    for f in folds:
        by_group.setdefault(FOLD_GROUPS.get(f, "other"), []).append(f)
    w("- LODO (metric 2 and raw LODO) tunes only on the held-out fold's own group -- "
      + "; ".join(f"{g}: {', '.join(fs)}" for g, fs in by_group.items())
      + ". Folds from one generator must not outvote the real ones.\n")
    if "m2_cross" not in set(scores.protocol):
        w("- Leave-one-group-out needs at least two groups; skipped for this run.\n")
    w("- CluStream's metric 2 has no scale conversion: Stage 3 leaves counts as counts. "
      "For CluStream the comparison tests only the ladder and LODO, not s(D).\n")
    w("- CluStream's offline phase is k-means and DenStream's is DBSCAN-like: an ARI "
      "gap between the two partly compares the offline phases.\n")
    w("- One tuning run per grid point, a handful of folds: a pilot check of direction, "
      "not a statistical result.\n")

    w("\n## Verdict summary\n\n")
    for name, good in verdicts.items():
        w(f"- {name}: {'PASS' if good else 'FAIL'}\n")
    return "".join(out)


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
    parser.add_argument("--m1-grid", default="", help="JSON {algo: [raw values]} overriding M1_GRID")
    parser.add_argument("--n-samples", type=int, default=0, help="0 = full stream")
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
        "--p2r",
        default=P2R_DEFAULT,
        help="comma list of rule:criterion P2R readings; empty string disables",
    )
    args = parser.parse_args()
    p2r = parse_p2r(args.p2r)

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

    m1_values = {**M1_GRID, **(json.loads(args.m1_grid) if args.m1_grid else {})}
    grids = {
        "raw": {a: raw_grid(a, m1_values[a]) for a in algos},
        "ladder": {a: ladder_grid(a) for a in algos},
    }
    defaults = {a: default_candidate(a) for a in algos}
    tuning = [c for grid in grids.values() for cands in grid.values() for c in cands]
    candidates = {(c["algo"], c["config_id"]): c for c in [*tuning, *defaults.values()]}
    grid_of = {key: c["grid"] for key, c in candidates.items()}

    running = not args.report_only
    if running and not args.dry_run:
        print("materialising permutations")
        for fold in folds:
            for seed in (TUNE_SEED, *FINAL_SEEDS):
                materialise(fold, shuffle_seed_for(seed))

    if running and args.phase in ("all", "tune"):
        print(f"tuning: {len(tuning)} grid points x {len(folds)} folds")
        run_jobs(
            [_job(c["algo"], c, f, TUNE_SEED, args.n_samples, probes) for c in tuning for f in folds],
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
    picks = all_picks(tune, grids, defaults, algos, folds, groups, p2r)

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
                _job(algo, cand, fold, seed, args.n_samples, probes)
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

    scores = score_table(df, picks, probes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT_DIR / f"scores_{requested}.csv", index=False)
    _ok(tune).pivot_table(
        index=["algo", "grid", "dataset"], columns="granularity_level", values=OBJ
    ).to_csv(OUT_DIR / f"tune_grid_{requested}.csv")
    report = build_report(tune, scores, folds, probes, requested, grids)
    path = OUT_DIR / ("report.md" if requested == "full" else f"report_{requested}.md")
    path.write_text(report, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
