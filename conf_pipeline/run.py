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
from benchmark.tuning.space import ParamSpace, SpaceContext
from benchmark.tuning.window import WindowBuffer

from . import _common as C
from .spaces import SPACES, start_params


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


def _make_adapter_factory(algo: str):
    def make(params: dict, seed: int):
        kw = dict(params)
        if accepts_seed(algo):
            kw.setdefault("seed", seed)
        return get_algorithm(algo, scaler=C.SCALER, **kw)
    return make


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
        **{f"{prefix}{k}_win": v for k, v in o.window_metrics.items()},
    }


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
) -> list[dict]:
    """Run one ensemble over ``dataset`` and return its window rows + summary row."""
    t_start = time.perf_counter()
    if arm == "default":
        cfg = EnsembleConfig(**{**cfg.__dict__, "e_size": 1, "e_test": 0})
    rng = np.random.default_rng(seed)

    try:
        length = len(dataset)
    except TypeError:
        length = None
    total = n_samples if n_samples is not None else length
    n_windows_total = (total // h) if total else None
    if lam is None:
        if n_windows_total is None:
            raise ValueError("lam must be given when the stream length is unknown")
        lam = C.lambda_for(n_windows_total, half_life_frac)
    cfg = EnsembleConfig(**{**cfg.__dict__, "lam": lam})

    base = {"algo": algo, "fold": fold, "seed": seed, "arm": arm}
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
    win_t0 = time.perf_counter()

    def log_members(report):
        if member_log is None:
            return
        for o in report.outcomes:
            C.append_jsonl(member_log, {**base, **o.to_dict()})

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
                ensemble = ConfStreamEnsemble(
                    space=space, ctx=ctx, make_adapter=_make_adapter_factory(algo),
                    start_params=start, config=cfg, rng=rng, seed=seed,
                )
                ensemble.initialise(calibration[0], window_index=0)
                for extra in calibration[1:]:
                    ensemble.replay(extra)
                window_index = len(calibration) - 1
            buffer = WindowBuffer(h)
            window_index += 1 if ensemble is None else 0
            if ensemble is not None:
                window_index += 1
            win_t0 = time.perf_counter()
            continue

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

        row = {
            "kind": "window", **base, "window": window_index,
            "start": n_seen - h, "end": n_seen - 1,
            "incumbent_id": inc.member_id, "params": inc.params,
            **_incumbent_fields(inc),
            **(_incumbent_fields(lag, "lag_") if lag is not None else
               {"lag_member_id": None, "lag_score": None, "lag_defined": None}),
            "n_true_classes": inc.n_true_classes,
            "majority_class_share": inc.majority_class_share,
            "n_active": report.n_active, "n_feasible": report.n_feasible,
            "n_tested": report.n_tested, "n_accepted": report.n_accepted,
            "n_discarded": report.n_discarded, "n_spawned": report.n_spawned,
            "n_reset": report.n_reset,
            "ensemble_runtime_s": report.ensemble_runtime_s, "wall_s": wall,
        }
        window_rows.append(row)
        if progress is not None:
            progress(window_index, n_windows_total, row)
        buffer = WindowBuffer(h)
        window_index += 1
        win_t0 = time.perf_counter()

    if ensemble is None:
        return [{
            "kind": "run", **base, "status": "error",
            "error": f"stream shorter than one window (h={h}, seen {n_seen})",
        }]

    def _mean(key):
        vals = [r[key] for r in window_rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def _median(key):
        vals = [r[key] for r in window_rows if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    lag_rows = [r for r in window_rows if r.get("lag_score") is not None]
    inc_runtime = sum(r["runtime_s"] for r in window_rows)
    lag_cum = batch_external(lag_y, lag_pred, tuple(cfg.external_metrics))
    summary = {
        "kind": "run", **base, "status": "ok",
        "n_samples": n_seen, "h": h, "n_windows": len(window_rows),
        "n_calibration_windows": len(calibration), "d": d, "s_online": s_online,
        "lam": lam, "half_life_frac": half_life_frac,
        "e_size": cfg.e_size, "e_test": cfg.e_test, "p_reset": cfg.p_reset,
        "runtime_factor": cfg.runtime_factor, "max_memory_mb": cfg.max_memory_mb,
        "max_micro_clusters": cfg.max_micro_clusters,
        "start_params": start[0], "final_params": window_rows[-1]["params"] if window_rows else None,
        "wall_s": time.perf_counter() - t_start,
        "ensemble_runtime_s": sum(r["ensemble_runtime_s"] for r in window_rows),
        "incumbent_runtime_s": inc_runtime,
        "incumbent_throughput_sps": (len(window_rows) * h / inc_runtime) if inc_runtime > 0 else None,
        "score_mean": _mean("score"), "score_median": _median("score"),
        "lag_score_mean": _mean("lag_score"), "lag_score_median": _median("lag_score"),
        "lag_ari_win_mean": _mean("lag_ari_win"), "lag_ari_win_median": _median("lag_ari_win"),
        **{f"lag_{k}_cumulative": v for k, v in lag_cum.items()},
        "lag_n_scored": len(lag_y),
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
        "n_accepted_total": sum(r["n_accepted"] for r in window_rows),
        "n_reset_total": sum(r["n_reset"] for r in window_rows),
        "single_class_window_share": (
            sum(1 for r in window_rows if (r["n_true_classes"] or 0) <= 1) / len(window_rows)
            if window_rows else None
        ),
    }
    return [*window_rows, summary]


# --- CLI: parent + child -------------------------------------------------------------

def _job_key(job: dict) -> tuple:
    return (job["algo"], job["fold"], job["seed"], job["arm"], job["n_samples"])


def _done_keys(runs_path: Path) -> set:
    keys = set()
    for row in C.read_jsonl(runs_path):
        if row.get("status") == "ok":
            keys.add((row["algo"], row["fold"], row["seed"], row["arm"], row.get("n_samples_cap")))
    return keys


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
    d = getattr(dataset, "n_features", None)
    if d is None:
        first = next(iter(dataset))
        d = len(first[0])
    rows = run_ensemble(
        algo=algo, space=SPACES[algo], start=[start_params(algo, job["h"], d=d)],
        dataset=dataset, fold=fold, seed=job["seed"], arm=job["arm"], cfg=cfg,
        h=job["h"], n_samples=job["n_samples"], half_life_frac=job["half_life_frac"],
        lam=job["lam"], member_log=member_log,
    )
    for row in rows:
        if row["kind"] == "run":
            row["n_samples_cap"] = job["n_samples"]
        sys.stdout.write(json.dumps(row, default=C._json_default) + "\n")


def _failed_row(job: dict, status: str, error: str) -> dict:
    return {
        "kind": "run", "algo": job["algo"], "fold": job["fold"], "seed": job["seed"],
        "arm": job["arm"], "n_samples_cap": job["n_samples"], "status": status,
        "error": (error or "")[-500:],
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
    parser.add_argument("--arms", default=",".join(C.ARMS))
    parser.add_argument("--n-samples", type=int, default=0, help="0 = whole stream")
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
                        help="also write every member's outcome per window under out/members/")
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
    n_samples = args.n_samples or None
    jobs = []
    for algo, fold, seed, arm in product(algos, _csv(args.folds), map(int, _csv(args.seeds)), _csv(args.arms)):
        jobs.append({
            "algo": algo, "fold": fold, "seed": seed, "arm": arm, "n_samples": n_samples,
            "h": args.h, "e_size": args.e_size, "e_test": args.e_test, "p_reset": args.p_reset,
            "half_life_frac": args.half_life_frac, "lam": args.lam,
            "runtime_factor": args.runtime_factor, "max_memory_mb": args.max_memory_mb,
            "max_micro_clusters": args.max_micro_clusters,
            "member_log": str(out / "members" / f"{algo}_{fold}_s{seed}_{arm}.jsonl")
            if args.log_members else None,
        })
    done = _done_keys(runs_path)
    if args.retry_failed:
        failed = {(r["algo"], r["fold"], r["seed"], r["arm"], r.get("n_samples_cap"))
                  for r in C.read_jsonl(runs_path) if r.get("status") != "ok"}
        done -= failed
    todo = [j for j in jobs if _job_key(j) not in done]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run")
    if args.dry_run:
        for j in todo:
            print(f"  {j['algo']:10s} {j['fold']:28s} seed={j['seed']} arm={j['arm']}")
        return
    if not todo:
        return

    with JsonlLockPair(runs_path, windows_path, force=args.force):
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
        for lock in self.locks:
            lock.__enter__()
        return self

    def __exit__(self, *exc):
        for lock in reversed(self.locks):
            lock.__exit__(*exc)
        return False


if __name__ == "__main__":
    main()
