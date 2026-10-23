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
from benchmark.tuning.confstream import TIE_BREAKS, ConfStreamEnsemble, EnsembleConfig, WindowOutcome
from benchmark.tuning.recluster import common_recluster, micro_centers
from benchmark.tuning.signals import NOISE_POLICIES, SIGNALS, check_signals
from benchmark.tuning.space import ParamSpace, SpaceContext
from benchmark.tuning.window import WindowBuffer

from . import _common as C
from . import manifest as M
from .spaces import SPACES, clamp_init_buffers, distance_params, scaled_defaults, start_params


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
        f"{prefix}runtime_ratio": o.runtime_ratio,
        f"{prefix}pred_sha": o.pred_sha,
        f"{prefix}memory_mb": o.memory_mb,
        f"{prefix}n_micro_clusters": o.n_micro_clusters,
        f"{prefix}n_macro_clusters": o.n_macro_clusters,
        f"{prefix}n_pred_labels": o.n_pred_labels,
        f"{prefix}n_sentinel": o.n_sentinel,
        f"{prefix}n_fallback": o.n_fallback,
        f"{prefix}error": o.error,
        **{f"{prefix}{k}": v for k, v in _external(o).items()},
        f"{prefix}cmm_error": o.cmm_error,
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


def job_config(job: dict) -> EnsembleConfig:
    """The protocol's ensemble configuration of one job, before its arm changes it
    (``arm_config``); a field the job leaves out (the sweep's jobs) takes the child's default."""
    return EnsembleConfig(
        e_size=job["e_size"], e_test=job["e_test"], p_reset=job["p_reset"],
        runtime_factor=job["runtime_factor"], max_memory_mb=job["max_memory_mb"],
        max_micro_clusters=job["max_micro_clusters"],
        external_metrics=tuple(m for m in C.EXTERNAL_METRICS if m not in C.CMM_METRICS or job.get("cmm", True)),
        log_signals=tuple(job.get("log_signals") or ()),
        tie_break=job.get("tie_break") or "micro_clusters",
        signal_noise=job.get("signal_noise") or "label",
    )


def arm_config(arm: str, cfg: EnsembleConfig, freeze_after_window: int | None) -> EnsembleConfig:
    """The ensemble configuration of one arm, derived from the protocol's ``cfg``.

    ``default``, ``oracle_static`` and ``sweep`` run one member that is never
    challenged (River's defaults, the oracle's point, one sweep point);
    ``confstream`` is the tuner as configured; ``random_online`` draws its
    children uniformly from the space; ``frozen_prefix`` tunes until
    ``freeze_after_window`` and keeps its incumbent from then on.
    """
    if arm in C.FIXED_ARMS:
        over = dict(e_size=1, e_test=0)
    elif arm == "confstream":
        over = {}
    elif arm == "random_online":
        over = dict(sampler="uniform")
    elif arm == "frozen_reinit":
        over = dict(freeze_after_window=freeze_after_window, reinit_after_freeze=True)
    elif arm in C.SIGNAL_ARMS or arm in C.DIAGNOSTIC_ARMS:
        over = dict(selection_signal=arm.removeprefix("confstream_"))
    elif arm == "frozen_prefix":
        over = dict(freeze_after_window=freeze_after_window)
    else:
        raise ValueError(f"unknown arm {arm!r}; arms: {C.ARMS + C.OPTIONAL_ARMS + C.DIAGNOSTIC_ARMS + C.EXTRA_ARMS}")
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
    model_seed: int | None = None,
    calibration_seed: int | None = None,
    seed_fields: dict | None = None,
) -> list[dict]:
    """Run one ensemble over ``dataset`` and return its window rows + summary row.

    Randomness: ``seed`` seeds the tuner's generator (every draw of the search); ``model_seed``
    is passed to every member's constructor (algorithms that take a seed), and is ``seed``
    itself when None; ``calibration_seed`` seeds the subsample the distance unit is estimated
    on, which draws from the tuner's generator when None. All three None-defaults together are
    the ``shared`` seed policy of the runs made before 2026-10-24, one job seed for everything.
    ``seed_fields`` is merged into every row (and member row), e.g. the policy and each seed.

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
    calibration_rng = rng if calibration_seed is None else np.random.default_rng(calibration_seed)
    if model_seed is None:
        model_seed = seed

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

    base = {"algo": algo, "fold": fold, "seed": seed, "arm": arm, "config_id": config_id, **(seed_fields or {})}
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
    scaled_params: dict | None = None
    calibration_s = 0.0
    win_t0 = time.perf_counter()

    def log_members(report) -> float:
        """Write the window's member rows; the seconds it took."""
        if member_log is None:
            return 0.0
        t0 = time.perf_counter()
        for o in report.outcomes:
            C.append_jsonl(member_log, {**base, **o.to_dict(), **_external(o)})
        return time.perf_counter() - t0

    for x, y in dataset:
        if n_samples is not None and n_seen >= n_samples:
            break
        buffer.push(x, y)
        n_seen += 1
        if not buffer.is_full:
            continue

        if ensemble is None:
            # Calibration: the distance unit, then the ensemble, then a replay.
            t_cal = time.perf_counter()
            calibration.append(buffer)
            xs_all = [x for b in calibration for x in b.xs]
            s_online = C.estimate_scale(xs_all, calibration_rng)
            d = len(buffer.xs[0])
            if s_online > 0 or len(calibration) >= scale_max_windows:
                ctx = SpaceContext(s=s_online if s_online > 0 else None, d=d)
                if params_override:
                    start = [dict(params_override)]
                elif unit_override:
                    start = [space.resolve({**space.unit_point(start[0], ctx), **unit_override}, ctx)]
                elif arm == "scaled_default":
                    scaled, scaled_params = scaled_defaults(algo, start[0], s_online)
                    start = [scaled]
                start = [clamp_init_buffers(algo, p, h) for p in start]
                ensemble = ConfStreamEnsemble(
                    space=space, ctx=ctx, make_adapter=_make_adapter_factory(algo, h),
                    start_params=start, config=cfg, rng=rng, seed=model_seed,
                    post_resolve=lambda params: clamp_init_buffers(algo, params, h),
                )
                ensemble.initialise(calibration[0], window_index=0)
                for extra in calibration[1:]:
                    ensemble.replay(extra)
                window_index = len(calibration) - 1
            calibration_s += time.perf_counter() - t_cal
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
        log_s = log_members(report)
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

        t_a = time.perf_counter()
        table_a_fields = _table_a_fields(lag_member, lag, buffer, C.TABLE_A_EPS_FACTOR * s_online)
        table_a_s = time.perf_counter() - t_a

        row = {
            "kind": "window", **base, "window": window_index,
            "start": n_seen - h, "end": n_seen - 1, "n_samples_cap": n_samples,
            "incumbent_id": inc.member_id, "params": inc.params,
            **_incumbent_fields(inc),
            **(_incumbent_fields(lag, "lag_") if lag is not None else
               {"lag_member_id": None, "lag_score": None, "lag_defined": None, "lag_pred_sha": None}),
            **table_a_fields,
            "n_true_classes": inc.n_true_classes,
            "majority_class_share": inc.majority_class_share,
            "external_defined": external_defined(inc),
            "n_active": report.n_active, "n_feasible": report.n_feasible,
            "n_tested": report.n_tested, "n_accepted": report.n_accepted,
            "n_discarded": report.n_discarded, "n_spawned": report.n_spawned,
            "n_reset": report.n_reset,
            "ensemble_runtime_s": report.ensemble_runtime_s, "wall_s": wall,
            "signals_runtime_s": report.signals_runtime_s,
            "cmm_runtime_s": report.cmm_runtime_s,
            #: The rest of the window's time, apart from the models' (``ensemble_runtime_s``):
            #: the silhouettes, building and warm-starting the next test set, Table A and the
            #: member log. ``wall_s`` ends before the last two.
            "scoring_runtime_s": report.scoring_runtime_s, "spawn_runtime_s": report.spawn_runtime_s,
            "table_a_runtime_s": table_a_s, "log_runtime_s": log_s,
            **{f"cmm_{k}": v for k, v in report.cmm_window.items()},
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
    lag_cum = batch_external(lag_y, lag_pred, tuple(m for m in cfg.external_metrics if m not in C.CMM_METRICS))
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
        "lag_cmm_win_mean": _mean("lag_cmm_win"), "lag_cmm_win_tail_mean": _mean("lag_cmm_win", tail),
        "lag_cmm_strict_win_mean": _mean("lag_cmm_strict_win"),
        "lag_cmm_strict_win_tail_mean": _mean("lag_cmm_strict_win", tail),
        #: MOA's class analysis: merged classes over classes, and the points no other class's sphere holds.
        "cmm_ca_separability_mean": _mean("cmm_ca_separability"), "cmm_ca_model_mean": _mean("cmm_ca_model"),
        #: CMM never stops a run either; the lagged windows it failed on, and the first message.
        "lag_cmm_error_share": (
            sum(1 for r in lag_rows if r.get("lag_cmm_error") is not None) / len(lag_rows) if lag_rows else None
        ),
        "lag_cmm_first_error": next((r["lag_cmm_error"] for r in lag_rows if r.get("lag_cmm_error")), None),
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
        "selection_signal": cfg.selection_signal,
        "log_signals": list(cfg.log_signals),
        #: Seconds spent on the signals beyond the silhouette, the cost of logging or selecting by them.
        "signals_runtime_s": float(sum(r["signals_runtime_s"] for r in window_rows)),
        #: Seconds spent on CMM over every member, part of ``wall_s`` and of no member's runtime.
        "cmm_runtime_s": float(sum(r["cmm_runtime_s"] for r in window_rows)),
        #: Where the rest of ``wall_s`` went: the calibration windows (distance unit, the first
        #: members, their replay), the silhouettes, the test sets' builds and warm starts,
        #: Table A and the member log (``signals_runtime_s`` and ``cmm_runtime_s`` above).
        "calibration_runtime_s": calibration_s,
        "scoring_runtime_s": float(sum(r["scoring_runtime_s"] for r in window_rows)),
        "spawn_runtime_s": float(sum(r["spawn_runtime_s"] for r in window_rows)),
        "table_a_runtime_s": float(sum(r["table_a_runtime_s"] for r in window_rows)),
        "log_runtime_s": float(sum(r["log_runtime_s"] for r in window_rows)),
        "freeze_after_window": cfg.freeze_after_window,
        "reinit_after_freeze": cfg.reinit_after_freeze,
        "tie_break": cfg.tie_break,
        "frozen_at_window": frozen_at_window,
        "frozen_params": window_rows[frozen_idx]["params"] if frozen_idx is not None else None,
        "unit_override": unit_override,
        "params_override": params_override,
        #: ``scaled_default``: each distance parameter's default, the unit and their product.
        "scaled_params": scaled_params,
    }
    return [*window_rows, summary]


# --- CLI: parent + child -------------------------------------------------------------

def _accepts_seed(algo: str) -> bool:
    try:
        return accepts_seed(algo)
    except KeyError:
        return False


def job_seeds(algo: str, arm: str, policy: str, seeds=(), tuner_seeds=(), model_seed: int | None = None,
              fixed_model_seeds=()) -> list[dict]:
    """The runs of (``algo``, ``arm``), one dict per run: ``seed`` (its index within the arm), ``tuner_seed``,
    ``model_seed`` (docs/superpowers/specs/2026-10-24-seed-split-design.md, section 4).

    ``shared``: one run per job seed, which seeds the tuner and, for an algorithm that takes one, the
    model. ``split``: an arm that tunes runs once per tuner seed, its models on ``model_seed``; a fixed
    configuration draws nothing from a tuner, so it runs once per fixed model seed when the algorithm
    takes a seed, and once at all when it is deterministic (another run would repeat it exactly).
    """
    seeded = _accepts_seed(algo)
    if policy == "shared":
        return [{"seed": s, "tuner_seed": s, "model_seed": s if seeded else None} for s in seeds]
    if policy != "split":
        raise ValueError(f"seed policy must be one of {C.SEED_POLICIES}, got {policy!r}")
    if arm in C.FIXED_ARMS:
        if not seeded:
            return [{"seed": None, "tuner_seed": None, "model_seed": None}]
        return [{"seed": m, "tuner_seed": None, "model_seed": m} for m in fixed_model_seeds]
    return [{"seed": t, "tuner_seed": t, "model_seed": model_seed if seeded else None} for t in tuner_seeds]


def seed_fields(job: dict) -> dict:
    """What a job's rows record about its randomness. A job dict that predates the split (the sweep's
    under the shared policy, a hand-made one) reads as the shared policy."""
    policy = job.get("seed_policy") or "shared"
    if policy == "shared" and "tuner_seed" not in job:
        return {"seed": job["seed"], "seed_policy": "shared", "tuner_seed": job["seed"],
                "model_seed": job["seed"] if _accepts_seed(job["algo"]) else None, "calibration_seed": None}
    return {"seed": job["seed"], "seed_policy": policy, "tuner_seed": job.get("tuner_seed"),
            "model_seed": job.get("model_seed"), "calibration_seed": job.get("calibration_seed")}


def _model_seed_of(d: dict):
    """A job's or row's model seed; a row from before the split derives it as the shared policy did."""
    if "model_seed" in d:
        return d["model_seed"]
    return d.get("seed") if _accepts_seed(d.get("algo", "")) else None


def _member_log_path(out: Path, job: dict) -> str:
    """Where one job's member outcomes go: its own file under ``<out>/members/``; a run without a seed
    of its own (a deterministic fixed configuration under the split policy) is ``seedx``."""
    seed = "x" if job["seed"] is None else job["seed"]
    name = f"{job['algo']}__{job['fold']}__seed{seed}__{job['arm']}"
    if job.get("config_id") is not None:
        name += f"__cfg{job['config_id']}"
    return str(out / C.MEMBERS_DIR.name / f"{name}.jsonl")


def _job_key(job: dict) -> tuple:
    return (job["algo"], job["fold"], job["seed"], job["arm"], job["n_samples"], job.get("config_id"),
            _model_seed_of(job))


def _row_key(row: dict) -> tuple:
    """The job key of a ``runs.jsonl`` row (``config_id`` is None for protocol arms)."""
    return (row["algo"], row["fold"], row["seed"], row["arm"], row.get("n_samples_cap"), row.get("config_id"),
            _model_seed_of(row))


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
    # The code this child imported, against which the parent checks the manifest's.
    code_sha = M.code_fingerprint()
    algo, fold = job["algo"], job["fold"]
    data_seed = job.get("data_seed", C.DATA_SEED)
    dataset = C.load_fold(fold, n_samples=job["n_samples"], data_seed=data_seed)
    cfg = job_config(job)
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
    # The samples the run consumes, hashed as it consumes them (the manifest's stream check).
    stream = M.CappedStream(dataset, job["n_samples"])
    fields = seed_fields(job)
    if fields["seed_policy"] == "split":
        # A fixed configuration draws nothing from the tuner, and a deterministic model reads no seed.
        tuner = fields["tuner_seed"] if fields["tuner_seed"] is not None else 0
        model = fields["model_seed"] if fields["model_seed"] is not None else 0
        calibration = fields["calibration_seed"]
    else:
        tuner, model, calibration = job["seed"], None, None
    rows = run_ensemble(
        algo=algo, space=SPACES[algo], start=[start_params(algo, job["h"], d=d)],
        dataset=stream, fold=fold, seed=tuner, arm=job["arm"], cfg=cfg,
        h=job["h"], n_samples=job["n_samples"], half_life_frac=job["half_life_frac"],
        lam=job["lam"], member_log=member_log, unit_override=job.get("unit_override"),
        freeze_after_window=job.get("freeze_after_window"), config_id=job.get("config_id"),
        params_override=job.get("params_override"), model_seed=model, calibration_seed=calibration,
        seed_fields=fields,
    )
    rows[-1].update(stream_sha256=stream.digest.hexdigest(), stream_n_samples=stream.digest.n, code_sha256=code_sha,
                    data_seed=data_seed if C.is_synthetic(fold) else None)
    for row in rows:
        row["n_samples_cap"] = job["n_samples"]
        sys.stdout.write(json.dumps(row, default=C._json_default) + "\n")


def _failed_row(job: dict, status: str, error: str) -> dict:
    return {
        "kind": "run", "algo": job["algo"], "fold": job["fold"], **seed_fields(job),
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
    parser.add_argument("--seed-policy", default="split", choices=C.SEED_POLICIES,
                        help="split (the default): tuner, model and calibration seeds apart; shared: one job "
                             "seed for all three, as in the runs made before 2026-10-24")
    parser.add_argument("--seeds", default=None,
                        help=f"shared policy only: the job seeds (default {','.join(map(str, C.SEEDS))})")
    parser.add_argument("--tuner-seeds", default=None,
                        help=f"split policy: one online run per tuner seed (default {','.join(map(str, C.TUNER_SEEDS))})")
    parser.add_argument("--model-seed", type=int, default=None,
                        help=f"split policy: the online arms' model seed, one per out dir (default {C.MODEL_SEED})")
    parser.add_argument("--fixed-model-seeds", default=None,
                        help="split policy: the runs of a fixed configuration of an algorithm that takes a seed "
                             "(default: the model seed); a deterministic one runs once")
    parser.add_argument("--arms", default=",".join(C.ARMS),
                        help=f"comma-separated, from {C.ARMS + C.OPTIONAL_ARMS + ('oracle_static',)}; "
                             f"diagnostic, selecting by labels: {C.DIAGNOSTIC_ARMS}")
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
    parser.add_argument("--no-runtime-gate", action="store_true",
                        help="never shed a member for its measured runtime (repeatable selection); "
                             "memory, micro-cluster and timeout limits still apply, and every member's "
                             "runtime ratio is still logged")
    parser.add_argument("--max-memory-mb", type=float, default=C.MAX_MEMORY_MB)
    parser.add_argument("--max-micro-clusters", type=int, default=C.MAX_MICRO_CLUSTERS)
    parser.add_argument("--out-dir", default=str(C.OUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=C.TIMEOUT_S)
    parser.add_argument("--log-members", action="store_true",
                        help="also write every member's outcome per window, one file per job under "
                             f"<out-dir>/{C.MEMBERS_DIR.name}/")
    parser.add_argument("--tie-break", default="micro_clusters", choices=TIE_BREAKS,
                        help="how tied members are separated: micro_clusters (deterministic, the default) "
                             "or runtime (the rule of the runs made before 2026-10-22)")
    parser.add_argument("--signals", default="",
                        help=f"comma-separated readings to log for every member, from {SIGNALS} "
                             "(written to the member logs as sig_<name>; needs --log-members)")
    parser.add_argument("--signal-noise", default="label", choices=NOISE_POLICIES,
                        help="how the label-free signals read a model's sentinel label (DenStream's -1): as a "
                             "cluster (label, the default), left out (drop), or one cluster per point (singleton)")
    parser.add_argument("--no-cmm", action="store_true",
                        help="leave CMM (cmm, cmm_strict) out of the per-window external readings")
    parser.add_argument("--data-seed", type=int, default=C.DATA_SEED,
                        help="seed of the synthetic streams' generators, one per out dir (real streams "
                             "have none)")
    parser.add_argument("--adopt-legacy", action="store_true",
                        help="resume into an out dir that holds results but no manifest, recording that "
                             "their protocol is unknown")
    parser.add_argument("--allow-code-change", action="store_true",
                        help="resume into an out dir made by other code (commit, diff or packages), "
                             "recording the change in its manifest")
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
    runnable = C.ARMS + C.OPTIONAL_ARMS + C.DIAGNOSTIC_ARMS + ("oracle_static",)
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
    policy = args.seed_policy
    split_flags = {"--tuner-seeds": args.tuner_seeds, "--model-seed": args.model_seed,
                   "--fixed-model-seeds": args.fixed_model_seeds}
    if policy == "split":
        if args.seeds is not None:
            raise SystemExit("--seeds is the shared policy's job seed; under the split policy give --tuner-seeds, "
                             "--model-seed and --fixed-model-seeds (or pass --seed-policy shared)")
        model_seed = C.MODEL_SEED if args.model_seed is None else args.model_seed
        seed_kw = dict(tuner_seeds=[int(s) for s in _csv(args.tuner_seeds)] if args.tuner_seeds else list(C.TUNER_SEEDS),
                       model_seed=model_seed,
                       fixed_model_seeds=[int(s) for s in _csv(args.fixed_model_seeds)] if args.fixed_model_seeds
                       else [model_seed])
    else:
        given = [flag for flag, v in split_flags.items() if v is not None]
        if given:
            raise SystemExit(f"{', '.join(given)} belong to the split policy; the shared policy takes --seeds")
        model_seed = None
        seed_kw = dict(seeds=[int(s) for s in _csv(args.seeds)] if args.seeds else list(C.SEEDS))
    signals = list(check_signals(_csv(args.signals)))
    if signals and not args.log_members:
        raise SystemExit("--signals needs --log-members: the readings are written only to the member logs")
    jobs, skipped = [], []
    picked = []
    planned = [(algo, fold, arm, seeds_of) for algo, fold, arm in product(algos, _csv(args.folds), arms)
               for seeds_of in job_seeds(algo, arm, policy, **seed_kw)]
    for algo, fold, arm, seeds_of in planned:
        job = {
            "algo": algo, "fold": fold, **seeds_of, "arm": arm, "n_samples": n_samples,
            "seed_policy": policy, "online_model_seed": model_seed,
            "calibration_seed": C.CALIBRATION_SEED if policy == "split" else None,
            "h": args.h, "e_size": args.e_size, "e_test": args.e_test, "p_reset": args.p_reset,
            "half_life_frac": args.half_life_frac, "lam": args.lam,
            "runtime_factor": None if args.no_runtime_gate else args.runtime_factor,
            "max_memory_mb": args.max_memory_mb, "max_micro_clusters": args.max_micro_clusters,
            "freeze_after_window": args.freeze_after_window, "member_log": None,
            "config_id": None, "unit_override": None, "params_override": None,
            "log_signals": signals, "tie_break": args.tie_break, "cmm": not args.no_cmm,
            "data_seed": args.data_seed, "signal_noise": args.signal_noise,
        }
        if arm == "scaled_default" and not distance_params(algo):
            # Nothing to scale: the run would be the default arm's again, not a new reading.
            skipped.append((job, _failed_row(
                job, "skipped", f"no absolute-distance parameter: {algo}'s scaled default is its default")))
            continue
        if arm == "oracle_static":
            # Never fall back to defaults: an (algo, fold) the sweep has not
            # settled is written down as skipped, with the reason.
            entry = (oracle.get(algo) or {}).get(fold)
            if entry is None:
                skipped.append((job, _failed_row(job, "skipped", f"no oracle for {algo}/{fold} in {oracle_file}")))
                continue
            # The sweep ran but every point failed: there is no configuration to rerun.
            if entry.get("config_id") is None:
                skipped.append((job, _failed_row(job, "skipped", f"sweep produced no ok point for {algo}/{fold}")))
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
            # The reference is never read on a model seed that picked it (docs/SEED_RULES.md, rule 4).
            if policy == "split" and job["model_seed"] is not None and job["model_seed"] in (entry.get("select_seeds") or []):
                picked.append(f"{algo}/{fold} model seed {job['model_seed']}")
        if args.log_members:
            job["member_log"] = _member_log_path(out, job)
        jobs.append(job)
    if picked:
        raise SystemExit("the reference would be rerun on a model seed that picked it:\n  " + "\n  ".join(picked)
                         + "\nGive --fixed-model-seeds other than the sweep's (oracle_static.json select_seeds).")
    described = jobs + [j for j, _ in skipped]
    if not described:
        print("0 jobs")
        return
    # Before anything is read as done: a dir made under another protocol is never resumed.
    history = {"workers": args.workers, "timeout": args.timeout, "memory_limit_bytes": C.MEMORY_LIMIT_BYTES}
    if "oracle_static" in arms and oracle_file.exists():
        history.update(oracle_file=str(oracle_file), oracle_sha256=M.fingerprint(oracle))
    new_manifest = M.build(described)
    guard = dict(adopt_legacy=args.adopt_legacy, allow_code_change=args.allow_code_change, **history)
    session_manifest, event = M.check(out, new_manifest, "conf_pipeline.run", **guard)
    done = _done_keys(runs_path)
    # A rerun of the reference is keyed by its config_id, which names a point only within one
    # sweep: an oracle file from another design can name another point under the same id.
    ran = {_row_key(r): r for r in C.read_jsonl(runs_path) if r.get("arm") == "oracle_static" and r.get("status") == "ok"}
    clashes = [
        f"{j['algo']}/{j['fold']} seed {j['seed']} config_id {j['config_id']}: on file "
        f"{M.canonical([ran[_job_key(j)].get('unit_override'), ran[_job_key(j)].get('params_override')])}, "
        f"now {M.canonical([j['unit_override'], j['params_override']])}"
        for j in jobs if j["arm"] == "oracle_static" and _job_key(j) in ran
        and M.canonical([ran[_job_key(j)].get("unit_override"), ran[_job_key(j)].get("params_override")])
        != M.canonical([j["unit_override"], j["params_override"]])
    ]
    if clashes:
        raise SystemExit(f"{oracle_file} names another configuration than the one already rerun here:\n  "
                         + "\n  ".join(clashes) + "\nWrite into a new --out-dir.")
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
    new_skips = [(j, r) for j, r in skipped if _row_key(r) not in on_file]
    print(M.describe(event, session_manifest))
    reasons: dict[str, int] = {}
    for _, r in skipped:
        why = ("no absolute-distance parameter" if r["error"].startswith("no absolute-distance") else "no oracle point")
        reasons[why] = reasons.get(why, 0) + 1
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run"
          + (f", {len(skipped)} skipped ({', '.join(f'{n} {w}' for w, n in reasons.items())})" if skipped else ""))
    if args.dry_run:
        for j in todo:
            print(f"  {j['algo']:10s} {j['fold']:28s} seed={j['seed']} arm={j['arm']}")
        for _, r in skipped:
            print(f"  {r['algo']:10s} {r['fold']:28s} seed={r['seed']} arm={r['arm']} skipped: {r['error']}")
        return
    if not todo and not new_skips:
        return

    with M.locked_session(out, new_manifest, "conf_pipeline.run", force=args.force, **guard) as session, \
            JsonlLockPair(runs_path, windows_path, force=args.force):
        if todo:
            C.record_environment(out, "conf_pipeline.run")
        for job, row in new_skips:
            C.append_jsonl(runs_path, session.stamp(job, row))
            print(f"[skipped] {row['algo']} {row['fold']} seed={row['seed']} arm={row['arm']}: {row['error']}",
                  flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(execute, j, args.timeout): j for j in todo}
            for fut in as_completed(futures):
                job = futures[fut]
                rows = fut.result()
                try:
                    session.accept(job, rows)
                except SystemExit:
                    pool.shutdown(wait=False, cancel_futures=True)
                    print("stopping: the jobs still running finish, nothing more is written", file=sys.stderr)
                    raise
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
