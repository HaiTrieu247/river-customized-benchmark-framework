"""Driver: one confStream ensemble over one fold; and the CLI around it.

The parent process enumerates (algo, fold, seed, arm) jobs, skips the ones
already in ``out/runs.jsonl`` (resume), and runs each in a child subprocess
with the hash seed pinned, an address-space limit and a timeout -- the same
isolation ``hpo_pipeline.build_matrix`` uses. The child streams its rows back
on stdout, one JSON object per line: ``kind == "window"`` rows for every
scored window and one ``kind == "run"`` summary row at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

from benchmark.algorithms.registry import accepts_seed, get_algorithm
from benchmark.tuning.confstream import ConfStreamEnsemble, EnsembleConfig, WindowOutcome
from benchmark.tuning.recluster import common_recluster, micro_centers
from benchmark.tuning.space import ParamSpace, SpaceContext
from benchmark.tuning.window import WindowBuffer

from . import _common as C
from .spaces import SPACES, clamp_init_buffers, start_params


# --- the driver ---------------------------------------------------------------------

def batch_external(y, pred, metrics: tuple[str, ...]) -> dict:
    """External quality of one stream of (label, prediction) pairs, in one pass.

    River's streaming ARI/NMI keep a dense contingency table and their ``get``
    is quadratic in the number of distinct predicted labels; a clusterer that
    hands out fresh ids on every split (RepStream) reaches thousands within a
    few windows and turns every reading into seconds. scikit-learn's batch
    versions use a sparse table and stay linear, so the lagged stream is scored
    once, at the end.
    """
    from collections import Counter, defaultdict

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    out: dict = {}
    n = len(y)
    if n == 0:
        return {m: None for m in metrics}
    ids = {}
    p = [ids.setdefault(v, len(ids)) for v in pred]
    yy = {}
    t = [yy.setdefault(v, len(yy)) for v in y]
    for m in metrics:
        if m == "ari":
            out[m] = float(adjusted_rand_score(t, p))
        elif m == "nmi":
            out[m] = float(normalized_mutual_info_score(t, p))
        elif m == "purity":
            per_cluster: dict = defaultdict(Counter)
            for a, b in zip(t, p):
                per_cluster[b][a] += 1
            out[m] = float(sum(max(c.values()) for c in per_cluster.values()) / n)
        else:
            out[m] = None
    return out


def _make_adapter_factory(algo: str, h: int):
    """Build one member's adapter; its initialisation buffer never exceeds the
    window ``h`` (``clamp_init_buffers``), whatever path its parameters took.

    The driver clamps the parameters it records the same way (the starts, and
    every child through the ensemble's ``post_resolve``), so this is the
    guarantee, not the place the clamp becomes visible.
    """
    def make(params: dict, seed: int):
        kw = clamp_init_buffers(algo, params, h)
        if accepts_seed(algo):
            kw.setdefault("seed", seed)
        return get_algorithm(algo, scaler=C.SCALER, **kw)
    return make


def external_defined(o: WindowOutcome) -> bool:
    """Whether the window ``o`` was scored on has enough true classes
    (``_common.MIN_TRUE_CLASSES``) for its external metrics to read anything;
    True when the window's classes are unknown."""
    return o.n_true_classes is None or o.n_true_classes >= C.MIN_TRUE_CLASSES


def _external(o: WindowOutcome) -> dict:
    """``o``'s per-window external metrics as ``{name_win: value}``, None on a
    window without enough true classes (``external_defined``)."""
    ok = external_defined(o)
    return {f"{k}_win": (v if ok else None) for k, v in o.window_metrics.items()}


def _incumbent_fields(o: WindowOutcome, prefix: str = "") -> dict:
    return {
        f"{prefix}member_id": o.member_id,
        f"{prefix}score": o.score,
        f"{prefix}defined": o.defined,
        f"{prefix}feasible": o.feasible,
        f"{prefix}feasible_reason": o.feasible_reason,
        f"{prefix}runtime_s": o.runtime_s,
        f"{prefix}memory_mb": o.memory_mb,
        f"{prefix}n_micro_clusters": o.n_micro_clusters,
        f"{prefix}n_macro_clusters": o.n_macro_clusters,
        f"{prefix}n_pred_labels": o.n_pred_labels,
        f"{prefix}n_sentinel": o.n_sentinel,
        f"{prefix}n_fallback": o.n_fallback,
        f"{prefix}error": o.error,
        **{f"{prefix}{k}": v for k, v in _external(o).items()},
    }


def _table_a_fields(lag_member, lag: WindowOutcome | None, buffer: WindowBuffer, eps: float) -> dict:
    """Table A: recluster the lagged member's own micro-cluster centres with a
    common k-means/DBSCAN, over this same window, so the reading is
    comparable across algorithms that use different offline reclusterers.

    ``lag_member`` is the live ``Member`` captured *before* this window's
    tournament (see the call site: the ensemble can evict it from ``active``
    this same window). A failure here -- the member missing entirely, a model
    whose introspection the framework does not yet cover, or a degenerate
    call into sklearn -- is recorded as ``lag_table_a_error`` on the row
    rather than raised, so it never kills the run: Table A is a diagnostic,
    not the protocol's own scoring.
    """
    table_a = {"ari_A_kmeans": None, "ari_A_dbscan": None, "ssq_micro": None, "n_micro_centers": None}
    error = None
    if lag is not None and lag.predictions is not None:
        if lag_member is None or lag_member.id != lag.member_id:
            error = f"lag member {lag.member_id} not found"
        else:
            try:
                mc = (
                    micro_centers(lag_member.adapter.model)
                    if (lag_member.adapter is not None and lag_member.error is None)
                    else None
                )
                table_a = common_recluster(
                    *(mc if mc is not None else (None, None, None)),
                    buffer.xs, buffer.ys, eps=eps,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
    if lag is not None and not external_defined(lag):
        # the centres and their SSQ need no labels; the ARIs of a one-class window read nothing
        table_a.update(ari_A_kmeans=None, ari_A_dbscan=None)
    return {**{f"lag_{k}": v for k, v in table_a.items()}, "lag_table_a_error": error}


def switch_cost(window_rows: list[dict]) -> float | None:
    """Spec 4.5: mean ``lag_ari_win`` on the windows whose ``incumbent_id``
    differs from the previous window's, minus the mean on the other windows.

    The first window has no previous one and counts as not switched; a window
    without a lagged reading (``lag_ari_win`` None, as on the first) is left
    out of both sides. None when either side has no reading -- a run that
    never switches has no switch cost, not a cost of 0.
    """
    switched, kept = [], []
    previous = None
    for i, row in enumerate(window_rows):
        current = row.get("incumbent_id")
        lag = row.get("lag_ari_win")
        if lag is not None:
            (switched if i > 0 and current != previous else kept).append(float(lag))
        previous = current
    if not switched or not kept:
        return None
    return float(np.mean(switched) - np.mean(kept))


def arm_config(arm: str, cfg: EnsembleConfig, freeze_after_window: int | None) -> EnsembleConfig:
    """The ensemble configuration of one arm, derived from the protocol's ``cfg``.

    ``default``, ``oracle_static`` and ``sweep`` run one member that is never
    challenged (River's defaults, the oracle's point, one sweep point);
    ``confstream`` is the tuner as configured; ``random_online`` draws its
    children uniformly from the space; ``frozen_prefix`` tunes until
    ``freeze_after_window`` and keeps its incumbent from then on.
    """
    if arm in ("default", "oracle_static", "sweep"):
        over = dict(e_size=1, e_test=0)
    elif arm == "confstream":
        over = {}
    elif arm == "random_online":
        over = dict(sampler="uniform")
    elif arm == "frozen_prefix":
        over = dict(freeze_after_window=freeze_after_window)
    else:
        raise ValueError(f"unknown arm {arm!r}; arms: {C.ARMS + C.OPTIONAL_ARMS + C.EXTRA_ARMS}")
    return EnsembleConfig(**{**cfg.__dict__, **over})


def run_ensemble(
    algo: str,
    space: ParamSpace,
    start: list[dict],
    dataset,
    fold: str,
    seed: int,
    arm: str,
    cfg: EnsembleConfig,
    h: int,
    n_samples: int | None = None,
    half_life_frac: float = C.HALF_LIFE_FRAC,
    lam: float | None = None,
    member_log: Path | None = None,
    scale_max_windows: int = C.SCALE_MAX_WINDOWS,
    progress=None,
    unit_override: dict | None = None,
    freeze_after_window: int | None = None,
    tail_from: int | None = None,
    config_id: int | None = None,
    params_override: dict | None = None,
) -> list[dict]:
    """Run one ensemble over ``dataset`` and return its window rows + summary row.

    ``unit_override`` replaces the start by a point of the unit cube: a
    dimension it leaves out keeps the start's *unit* value --
    ``space.unit_point(start, ctx)``, i.e. the start clamped into the space --
    not the raw River value the start resolves to. It is resolved once the
    distance unit is known. ``params_override`` replaces the start by exactly
    these constructor parameters instead (a configuration another run settled
    on, see ``sweep.incumbent_candidates``); the two are exclusive.
    ``freeze_after_window`` (default
    ``FREEZE_AFTER_WINDOW``) is where ``frozen_prefix`` stops tuning, and where
    every arm's tail metrics start unless ``tail_from`` says otherwise.

    Every member -- start, ``unit_override`` point or child -- has its
    initialisation buffer (``n_samples_init``, ``chunk_size``) capped at ``h``
    by ``clamp_init_buffers`` before its parameters are recorded and its model
    is built: a member that buffers the whole window predicts one label and
    has no partition to be scored on.
    """
    t_start = time.perf_counter()
    if freeze_after_window is None:
        freeze_after_window = C.FREEZE_AFTER_WINDOW
    if tail_from is None:
        tail_from = freeze_after_window
    cfg = arm_config(arm, cfg, freeze_after_window)
    if unit_override and params_override:
        raise ValueError("give unit_override or params_override, not both")
    if unit_override:
        unknown = sorted(set(unit_override) - set(space.names))
        if unknown:
            raise ValueError(f"unit_override names {unknown}, not dimensions of {space.names}")
    rng = np.random.default_rng(seed)

    try:
        length = len(dataset)
    except TypeError:
        length = None
    total = min(n_samples, length) if n_samples is not None and length is not None else (n_samples or length)
    n_windows_total = (total // h) if total else None
    if lam is None:
        if n_windows_total is None:
            raise ValueError("lam must be given when the stream length is unknown")
        lam = C.lambda_for(n_windows_total, half_life_frac)
    cfg = EnsembleConfig(**{**cfg.__dict__, "lam": lam})

    base = {"algo": algo, "fold": fold, "seed": seed, "arm": arm, "config_id": config_id}
    ensemble: ConfStreamEnsemble | None = None
    calibration: list[WindowBuffer] = []
    s_online = 0.0
    d = None
    buffer = WindowBuffer(h)
    window_index = 0
    n_seen = 0
    window_rows: list[dict] = []
    #: The lagged stream: what a user running the incumbent would have seen.
    lag_y: list = []
    lag_pred: list = []
    last_incumbent_id = None
    n_switches = 0
    n_scored_lag = 0
    frozen_at_window: int | None = None
    frozen_idx: int | None = None
    win_t0 = time.perf_counter()

    def log_members(report):
        if member_log is None:
            return
        for o in report.outcomes:
            C.append_jsonl(member_log, {**base, **o.to_dict(), **_external(o)})

    for x, y in dataset:
        if n_samples is not None and n_seen >= n_samples:
            break
        buffer.push(x, y)
        n_seen += 1
        if not buffer.is_full:
            continue

        if ensemble is None:
            # Calibration: the distance unit, then the ensemble, then a replay.
            calibration.append(buffer)
            xs_all = [x for b in calibration for x in b.xs]
            s_online = C.estimate_scale(xs_all, rng)
            d = len(buffer.xs[0])
            if s_online > 0 or len(calibration) >= scale_max_windows:
                ctx = SpaceContext(s=s_online if s_online > 0 else None, d=d)
                if params_override:
                    start = [dict(params_override)]
                elif unit_override:
                    start = [space.resolve({**space.unit_point(start[0], ctx), **unit_override}, ctx)]
                start = [clamp_init_buffers(algo, p, h) for p in start]
                ensemble = ConfStreamEnsemble(
                    space=space, ctx=ctx, make_adapter=_make_adapter_factory(algo, h),
                    start_params=start, config=cfg, rng=rng, seed=seed,
                    post_resolve=lambda params: clamp_init_buffers(algo, params, h),
                )
                ensemble.initialise(calibration[0], window_index=0)
                for extra in calibration[1:]:
                    ensemble.replay(extra)
                window_index = len(calibration) - 1
            buffer = WindowBuffer(h)
            window_index += 1
            win_t0 = time.perf_counter()
            continue

        # Captured *before* this window's tournament: `process_window` may
        # replace the previous incumbent in `ensemble.active` this same
        # window (a challenger can win against it as a victim), at which
        # point `ensemble.find(lag.member_id)` afterwards would come back
        # empty even though the member trained on this window just fine.
        # `last_incumbent_id` here still holds last window's incumbent id --
        # `ensemble`'s own bookkeeping updates it to the same value at the
        # same points in the loop, so this is the live Member as it stood
        # right after being scored and trained on this window, whether or
        # not it survives into the next one.
        lag_member = ensemble.find(last_incumbent_id) if last_incumbent_id is not None else None
        report = ensemble.process_window(buffer, window_index)
        wall = time.perf_counter() - win_t0
        log_members(report)
        inc = report.incumbent
        if last_incumbent_id is not None and inc.member_id != last_incumbent_id:
            n_switches += 1
        last_incumbent_id = inc.member_id

        lag = report.lag
        if lag is not None and lag.predictions is not None:
            n_scored_lag += 1
            for yy, yp in zip(buffer.ys, lag.predictions):
                if yy is not None and yp is not None:
                    lag_y.append(yy)
                    lag_pred.append(yp)

        table_a_fields = _table_a_fields(lag_member, lag, buffer, C.TABLE_A_EPS_FACTOR * s_online)

        row = {
            "kind": "window", **base, "window": window_index,
            "start": n_seen - h, "end": n_seen - 1, "n_samples_cap": n_samples,
            "incumbent_id": inc.member_id, "params": inc.params,
            **_incumbent_fields(inc),
            **(_incumbent_fields(lag, "lag_") if lag is not None else
               {"lag_member_id": None, "lag_score": None, "lag_defined": None}),
            **table_a_fields,
            "n_true_classes": inc.n_true_classes,
            "majority_class_share": inc.majority_class_share,
            "external_defined": external_defined(inc),
            "n_active": report.n_active, "n_feasible": report.n_feasible,
            "n_tested": report.n_tested, "n_accepted": report.n_accepted,
            "n_discarded": report.n_discarded, "n_spawned": report.n_spawned,
            "n_reset": report.n_reset,
            "ensemble_runtime_s": report.ensemble_runtime_s, "wall_s": wall,
            "frozen": report.frozen,
        }
        if report.frozen and frozen_at_window is None:
            frozen_at_window, frozen_idx = window_index, len(window_rows)
        window_rows.append(row)
        if progress is not None:
            progress(window_index, n_windows_total, row)
        buffer = WindowBuffer(h)
        window_index += 1
        win_t0 = time.perf_counter()

    if ensemble is None:
        return [{
            "kind": "run", **base, "status": "error", "unit_override": unit_override,
            "params_override": params_override, "n_samples_cap": n_samples, "error": f"stream shorter than one window (h={h}, seen {n_seen})",
        }]

    def _mean(key, rows=window_rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _median(key):
        vals = [r[key] for r in window_rows if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    lag_rows = [r for r in window_rows if r.get("lag_score") is not None]
    #: The evaluation region every arm shares: windows from ``tail_from`` on.
    tail = [r for r in lag_rows if r["window"] >= tail_from]
    inc_runtime = sum(r["runtime_s"] for r in window_rows)
    ens_runtime = sum(r["ensemble_runtime_s"] for r in window_rows)
    lag_cum = batch_external(lag_y, lag_pred, tuple(cfg.external_metrics))
    summary = {
        "kind": "run", **base, "status": "ok",
        "n_samples": n_seen, "n_samples_cap": n_samples, "h": h, "n_windows": len(window_rows),
        "n_calibration_windows": len(calibration), "d": d, "s_online": s_online,
        "drift_points": list(C.drift_points(fold, n_samples)),
        "lam": lam, "half_life_frac": half_life_frac,
        "e_size": cfg.e_size, "e_test": cfg.e_test, "p_reset": cfg.p_reset,
        "runtime_factor": cfg.runtime_factor, "max_memory_mb": cfg.max_memory_mb,
        "max_micro_clusters": cfg.max_micro_clusters,
        "start_params": start[0], "final_params": window_rows[-1]["params"] if window_rows else None,
        "wall_s": time.perf_counter() - t_start,
        "ensemble_runtime_s": ens_runtime,
        "incumbent_runtime_s": inc_runtime,
        "incumbent_throughput_sps": (len(window_rows) * h / inc_runtime) if inc_runtime > 0 else None,
        #: Spec 4.5: what tuning costs on top of running the incumbent alone (1.0 for one member).
        "tuning_overhead": (ens_runtime / inc_runtime) if inc_runtime > 0 else None,
        "score_mean": _mean("score"), "score_median": _median("score"),
        "lag_score_mean": _mean("lag_score"), "lag_score_median": _median("lag_score"),
        "lag_ari_win_mean": _mean("lag_ari_win"), "lag_ari_win_median": _median("lag_ari_win"),
        "lag_nmi_win_mean": _mean("lag_nmi_win"), "lag_purity_win_mean": _mean("lag_purity_win"),
        "tail_from": tail_from,
        "lag_ari_win_tail_mean": _mean("lag_ari_win", tail),
        "lag_score_tail_mean": _mean("lag_score", tail),
        "lag_ari_A_kmeans_mean": _mean("lag_ari_A_kmeans"),
        "lag_ari_A_dbscan_mean": _mean("lag_ari_A_dbscan"),
        "lag_ssq_micro_mean": _mean("lag_ssq_micro"),
        "lag_n_micro_centers_mean": _mean("lag_n_micro_centers"),
        "lag_ari_A_kmeans_tail_mean": _mean("lag_ari_A_kmeans", tail),
        "lag_ari_A_dbscan_tail_mean": _mean("lag_ari_A_dbscan", tail),
        #: Table A is a diagnostic that never kills the run; its failures are counted here.
        "lag_table_a_error_share": (
            sum(1 for r in lag_rows if r.get("lag_table_a_error") is not None) / len(lag_rows)
            if lag_rows else None
        ),
        "lag_table_a_first_error": next(
            (r["lag_table_a_error"] for r in lag_rows if r.get("lag_table_a_error") is not None), None
        ),
        **{f"lag_{k}_cumulative": v for k, v in lag_cum.items()},
        "lag_n_scored": len(lag_y),
        #: Windows behind the lagged means: a window with one true class has no reading.
        "lag_ari_n_windows": sum(1 for r in window_rows if r.get("lag_ari_win") is not None),
        "lag_ari_n_windows_tail": sum(1 for r in tail if r.get("lag_ari_win") is not None),
        "n_single_class_windows": sum(1 for r in window_rows if not r["external_defined"]),
        "lag_undefined_share": (
            sum(1 for r in lag_rows if not r["lag_defined"]) / len(lag_rows) if lag_rows else None
        ),
        "lag_infeasible_share": (
            sum(1 for r in lag_rows if not r["lag_feasible"]) / len(lag_rows) if lag_rows else None
        ),
        "lag_n_pred_labels_mean": _mean("lag_n_pred_labels"),
        "lag_n_macro_clusters_mean": _mean("lag_n_macro_clusters"),
        "lag_memory_mb_peak": max((r["lag_memory_mb"] or 0.0) for r in lag_rows) if lag_rows else None,
        "lag_n_sentinel_total": sum(r["lag_n_sentinel"] or 0 for r in lag_rows),
        "n_switches": n_switches,
        "switch_cost": switch_cost(window_rows),
        "n_accepted_total": sum(r["n_accepted"] for r in window_rows),
        "n_reset_total": sum(r["n_reset"] for r in window_rows),
        "single_class_window_share": (
            sum(1 for r in window_rows if (r["n_true_classes"] or 0) <= 1) / len(window_rows)
            if window_rows else None
        ),
        "sampler": cfg.sampler,
        "freeze_after_window": cfg.freeze_after_window,
        "frozen_at_window": frozen_at_window,
        "frozen_params": window_rows[frozen_idx]["params"] if frozen_idx is not None else None,
        "unit_override": unit_override,
        "params_override": params_override,
    }
    return [*window_rows, summary]


# --- CLI: parent + child -------------------------------------------------------------

def _member_log_path(out: Path, job: dict) -> str:
    """Where one job's member outcomes go: its own file under ``<out>/members/``."""
    name = f"{job['algo']}__{job['fold']}__seed{job['seed']}__{job['arm']}"
    if job.get("config_id") is not None:
        name += f"__cfg{job['config_id']}"
    return str(out / C.MEMBERS_DIR.name / f"{name}.jsonl")


def _job_key(job: dict) -> tuple:
    return (job["algo"], job["fold"], job["seed"], job["arm"], job["n_samples"], job.get("config_id"))


def _row_key(row: dict) -> tuple:
    """The job key of a ``runs.jsonl`` row (``config_id`` is None for protocol arms)."""
    return (row["algo"], row["fold"], row["seed"], row["arm"], row.get("n_samples_cap"), row.get("config_id"))


def _done_keys(runs_path: Path) -> set:
    """A job's key is done once ANY non-``skipped`` attempt is on file.

    ``ok``, ``error``, ``timeout`` and ``oom`` all count -- an errored or
    timed-out attempt is still an attempt, and the job stays done unless
    ``--retry-failed`` explicitly asks for it back (see ``main()``). A
    ``skipped`` row (e.g. an ``oracle_static`` job with no oracle entry yet)
    is not an attempt at all, so it never makes a job look done.
    """
    return {_row_key(row) for row in C.read_jsonl(runs_path) if row.get("status") != "skipped"}


def _run_one_child() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (C.MEMORY_LIMIT_BYTES, C.MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass
    job = json.loads(sys.stdin.read())
    algo, fold = job["algo"], job["fold"]
    dataset = C.load_fold(fold, n_samples=job["n_samples"])
    cfg = EnsembleConfig(
        e_size=job["e_size"], e_test=job["e_test"], p_reset=job["p_reset"],
        runtime_factor=job["runtime_factor"], max_memory_mb=job["max_memory_mb"],
        max_micro_clusters=job["max_micro_clusters"], external_metrics=tuple(C.EXTERNAL_METRICS),
    )
    member_log = Path(job["member_log"]) if job.get("member_log") else None
    if member_log is not None:
        # A retried job (--retry-failed) reuses the same path a failed attempt
        # already wrote to; truncate before this attempt's first append so its
        # rows do not pile up behind the failed attempt's partial ones.
        member_log.parent.mkdir(parents=True, exist_ok=True)
        member_log.open("w").close()
    d = getattr(dataset, "n_features", None)
    if d is None:
        first = next(iter(dataset))
        d = len(first[0])
    rows = run_ensemble(
        algo=algo, space=SPACES[algo], start=[start_params(algo, job["h"], d=d)],
        dataset=dataset, fold=fold, seed=job["seed"], arm=job["arm"], cfg=cfg,
        h=job["h"], n_samples=job["n_samples"], half_life_frac=job["half_life_frac"],
        lam=job["lam"], member_log=member_log, unit_override=job.get("unit_override"),
        freeze_after_window=job.get("freeze_after_window"), config_id=job.get("config_id"),
        params_override=job.get("params_override"),
    )
    for row in rows:
        row["n_samples_cap"] = job["n_samples"]
        sys.stdout.write(json.dumps(row, default=C._json_default) + "\n")


def _failed_row(job: dict, status: str, error: str) -> dict:
    return {
        "kind": "run", "algo": job["algo"], "fold": job["fold"], "seed": job["seed"],
        "arm": job["arm"], "config_id": job.get("config_id"), "unit_override": job.get("unit_override"),
        "params_override": job.get("params_override"),
        "n_samples_cap": job["n_samples"], "status": status, "error": (error or "")[-500:],
    }


def execute(job: dict, timeout: int) -> list[dict]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "conf_pipeline.run", "--run-one"],
            input=json.dumps(job), capture_output=True, text=True,
            timeout=timeout, cwd=str(C.REPO_ROOT), env=C.child_env(),
        )
    except subprocess.TimeoutExpired:
        return [_failed_row(job, "timeout", f"exceeded {timeout}s")]
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        status = "oom" if "MemoryError" in err or proc.returncode == -9 else "error"
        return [_failed_row(job, status, err)]
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                return [_failed_row(job, "error", line[-500:])]
    if not rows or rows[-1].get("kind") != "run":
        return [_failed_row(job, "error", "child produced no summary row")]
    return rows


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algos", default=",".join(C.ALGOS))
    parser.add_argument("--folds", default=",".join(C.FOLDS))
    parser.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    parser.add_argument("--arms", default=",".join(C.ARMS),
                        help=f"comma-separated, from {C.ARMS + C.OPTIONAL_ARMS + ('oracle_static',)}")
    parser.add_argument("--n-samples", type=int, default=C.N_SAMPLES_CAP, help="0 = whole stream")
    parser.add_argument("--freeze-after-window", type=int, default=C.FREEZE_AFTER_WINDOW,
                        help="window where frozen_prefix stops tuning; every arm's tail starts there")
    parser.add_argument("--oracle-file", default=None,
                        help="the sweep's oracle, read by the oracle_static arm "
                             f"(default: <out-dir>/{C.ORACLE_JSON.name})")
    parser.add_argument("--h", type=int, default=C.H)
    parser.add_argument("--e-size", type=int, default=C.E_SIZE)
    parser.add_argument("--e-test", type=int, default=C.E_TEST)
    parser.add_argument("--p-reset", type=float, default=C.P_RESET)
    parser.add_argument("--half-life-frac", type=float, default=C.HALF_LIFE_FRAC)
    parser.add_argument("--lam", type=float, default=None, help="override the per-fold lambda")
    parser.add_argument("--runtime-factor", type=float, default=C.RUNTIME_FACTOR)
    parser.add_argument("--max-memory-mb", type=float, default=C.MAX_MEMORY_MB)
    parser.add_argument("--max-micro-clusters", type=int, default=C.MAX_MICRO_CLUSTERS)
    parser.add_argument("--out-dir", default=str(C.OUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=C.TIMEOUT_S)
    parser.add_argument("--log-members", action="store_true",
                        help="also write every member's outcome per window, one file per job under "
                             f"<out-dir>/{C.MEMBERS_DIR.name}/")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.run_one:
        _run_one_child()
        return

    out = Path(args.out_dir)
    runs_path, windows_path = out / "runs.jsonl", out / "windows.jsonl"
    algos = _csv(args.algos)
    unknown = [a for a in algos if a not in SPACES]
    if unknown:
        raise SystemExit(f"no search space for {unknown}; known: {sorted(SPACES)}")
    arms = _csv(args.arms)
    runnable = C.ARMS + C.OPTIONAL_ARMS + ("oracle_static",)
    bad = [a for a in arms if a not in runnable]
    if bad:
        raise SystemExit(
            f"unknown arms {bad}; this driver runs {runnable} (sweep jobs have their own driver)"
        )
    oracle_file = Path(args.oracle_file) if args.oracle_file else out / C.ORACLE_JSON.name
    oracle = (
        json.loads(oracle_file.read_text(encoding="utf-8"))
        if "oracle_static" in arms and oracle_file.exists() else {}
    )
    n_samples = args.n_samples or None
    jobs, skipped = [], []
    for algo, fold, seed, arm in product(algos, _csv(args.folds), map(int, _csv(args.seeds)), arms):
        job = {
            "algo": algo, "fold": fold, "seed": seed, "arm": arm, "n_samples": n_samples,
            "h": args.h, "e_size": args.e_size, "e_test": args.e_test, "p_reset": args.p_reset,
            "half_life_frac": args.half_life_frac, "lam": args.lam,
            "runtime_factor": args.runtime_factor, "max_memory_mb": args.max_memory_mb,
            "max_micro_clusters": args.max_micro_clusters,
            "freeze_after_window": args.freeze_after_window, "member_log": None,
            "config_id": None, "unit_override": None, "params_override": None,
        }
        if arm == "oracle_static":
            # Never fall back to defaults: an (algo, fold) the sweep has not
            # settled is written down as skipped, with the reason.
            entry = (oracle.get(algo) or {}).get(fold)
            if entry is None:
                skipped.append(_failed_row(job, "skipped", f"no oracle for {algo}/{fold} in {oracle_file}"))
                continue
            # The sweep ran but every point failed: there is no configuration to rerun.
            if entry.get("config_id") is None:
                skipped.append(_failed_row(job, "skipped", f"sweep produced no ok point for {algo}/{fold}"))
                continue
            # "unit" is a hard lookup: a well-formed entry always has the key,
            # `None` meaning "River default"; an entry missing it entirely is
            # malformed and should fail loudly rather than silently falling
            # back to the same `None`.
            if "unit" not in entry:
                raise SystemExit(
                    f"oracle entry for {algo}/{fold} in {oracle_file} has no 'unit' key "
                    "(a well-formed sweep output always has one, null meaning River default)"
                )
            # An incumbent candidate carries its exact parameters (sweep.py); a
            # Sobol point or the default carries its unit point.
            job["config_id"], job["unit_override"] = entry.get("config_id"), entry["unit"]
            job["params_override"] = entry.get("params_override")
        if args.log_members:
            job["member_log"] = _member_log_path(out, job)
        jobs.append(job)
    done = _done_keys(runs_path)
    if args.retry_failed:
        # Only a key whose *most recent* (or any) attempt is a real,
        # non-skipped failure is put back in play. A key can carry more than
        # one row across resumes (an old "error" row plus a later "ok" one
        # once the job was retried and succeeded); such a key must stay done,
        # so any row with status "ok" removes it from `failed` regardless of
        # what an earlier attempt recorded. A "skipped" row was never in
        # `done` to begin with (see `_done_keys`).
        all_rows = list(C.read_jsonl(runs_path))
        ok_keys = {_row_key(r) for r in all_rows if r.get("status") == "ok"}
        failed = {_row_key(r) for r in all_rows if r.get("status") not in ("ok", "skipped")} - ok_keys
        done -= failed
    todo = [j for j in jobs if _job_key(j) not in done]
    # A skip already on file is not written again on resume.
    on_file = {_row_key(r) for r in C.read_jsonl(runs_path) if r.get("status") == "skipped"}
    new_skips = [r for r in skipped if _row_key(r) not in on_file]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run"
          + (f", {len(skipped)} skipped (no oracle point)" if skipped else ""))
    if args.dry_run:
        for j in todo:
            print(f"  {j['algo']:10s} {j['fold']:28s} seed={j['seed']} arm={j['arm']}")
        for r in skipped:
            print(f"  {r['algo']:10s} {r['fold']:28s} seed={r['seed']} arm={r['arm']} skipped: {r['error']}")
        return
    if not todo and not new_skips:
        return

    with JsonlLockPair(runs_path, windows_path, force=args.force):
        for row in new_skips:
            C.append_jsonl(runs_path, row)
            print(f"[skipped] {row['algo']} {row['fold']} seed={row['seed']} arm={row['arm']}: {row['error']}",
                  flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(execute, j, args.timeout): j for j in todo}
            for fut in as_completed(futures):
                job = futures[fut]
                rows = fut.result()
                for row in rows:
                    if row["kind"] == "window":
                        C.append_jsonl(windows_path, row)
                summary = rows[-1]
                C.append_jsonl(runs_path, summary)
                status = summary.get("status")
                extra = (
                    f"windows={summary.get('n_windows')} lag_ari={summary.get('lag_ari_win_mean')} "
                    f"wall={summary.get('wall_s', 0):.0f}s"
                    if status == "ok" else summary.get("error")
                )
                print(f"[{status}] {job['algo']} {job['fold']} seed={job['seed']} arm={job['arm']}: {extra}",
                      flush=True)


class JsonlLockPair:
    """One lock covering both output files of a run."""

    def __init__(self, runs_path: Path, windows_path: Path, force: bool = False):
        self.locks = [C.JsonlLock(runs_path, force), C.JsonlLock(windows_path, force)]

    def __enter__(self):
        # A lock that is refused must not leave the ones taken before it behind:
        # the ``with`` statement never calls ``__exit__`` when ``__enter__`` fails.
        held = []
        try:
            for lock in self.locks:
                lock.__enter__()
                held.append(lock)
        except BaseException:
            for lock in reversed(held):
                lock.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for lock in reversed(self.locks):
            lock.__exit__(*exc)
        return False


if __name__ == "__main__":
    main()
