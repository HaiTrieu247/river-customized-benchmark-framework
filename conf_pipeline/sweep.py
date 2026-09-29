"""The a-posteriori oracle: a Sobol sweep of fixed configurations, and their winner.

Every ``sweep`` job runs a single, never-challenged member (``e_size=1``,
``e_test=0``) at one point of the search space -- either River's own default
(``config_id=0``, ``unit_override=None``) or one of a low-discrepancy Sobol
design over the unit cube. Because the ensemble has one member, that member's
lagged score *is* the run's ``lag_ari_win_mean``: the summary row already
carries the number ``select_oracle`` needs, so no extra plumbing is required.

The sweep writes its own files (``sweep.jsonl`` for summaries,
``sweep_windows.jsonl`` for window rows) rather than sharing ``runs.jsonl`` /
``windows.jsonl`` with ``run.py``: a sweep job is not one of the four protocol
arms, and mixing them would make the resume keys in ``run.py`` ambiguous.

``select_oracle`` then reduces the sweep to one fixed configuration per
(algo, fold) -- the one with the best lagged score -- and writes it to
``oracle_static.json``, which ``run.py --arms oracle_static`` reads.
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from scipy.stats import qmc

from . import _common as C
from . import run
from .spaces import SPACES


# --- the design ----------------------------------------------------------------------

def sobol_units(space, n_points: int, seed: int) -> list[dict | None]:
    """``n_points`` Sobol points of ``space``'s unit cube, plus the River default.

    Index 0 is always ``None`` -- "no override", i.e. River's own default,
    resolved the same way every other arm's default member is. The remaining
    ``n_points`` entries are unit-interval dicts keyed by ``space.names``, from
    a scrambled Sobol sequence seeded for reproducibility. ``n_points`` need
    not be a power of two; scipy's warning about that is expected and silenced
    so test output (and sweep logs) stay pristine.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sampler = qmc.Sobol(d=space.dim, scramble=True, seed=seed)
        points = sampler.random(n_points)
    return [None] + [dict(zip(space.names, map(float, row))) for row in points]


# --- jobs ------------------------------------------------------------------------------

def build_sweep_jobs(
    algos: list[str],
    folds: list[str],
    n_points: int,
    seed: int,
    n_samples: int | None,
    h: int,
    run_seed: int | None = None,
    **protocol,
) -> list[dict]:
    """One ``arm="sweep"`` job per (algo, fold, config), config 0 being the default.

    ``seed`` drives the Sobol design; ``run_seed`` (default: ``seed``) is the
    run's own seed, matching the job dict ``run.execute`` / the child expect
    (see ``run.main``). ``**protocol`` overrides the protocol defaults taken
    from ``conf_pipeline._common``; unrecognised keys are a caller error,
    caught the same way an unknown constructor kwarg would be.
    """
    if run_seed is None:
        run_seed = seed
    defaults = dict(
        p_reset=C.P_RESET, half_life_frac=C.HALF_LIFE_FRAC, runtime_factor=C.RUNTIME_FACTOR,
        max_memory_mb=C.MAX_MEMORY_MB, max_micro_clusters=C.MAX_MICRO_CLUSTERS,
        freeze_after_window=C.FREEZE_AFTER_WINDOW,
    )
    unknown = sorted(set(protocol) - set(defaults))
    if unknown:
        raise TypeError(f"build_sweep_jobs got unknown protocol overrides {unknown}")
    proto = {**defaults, **protocol}

    jobs: list[dict] = []
    for algo, fold in product(algos, folds):
        units = sobol_units(SPACES[algo], n_points, seed)
        for config_id, unit_override in enumerate(units):
            jobs.append({
                "algo": algo, "fold": fold, "seed": run_seed, "arm": "sweep",
                "n_samples": n_samples, "h": h, "e_size": 1, "e_test": 0,
                "p_reset": proto["p_reset"], "half_life_frac": proto["half_life_frac"], "lam": None,
                "runtime_factor": proto["runtime_factor"], "max_memory_mb": proto["max_memory_mb"],
                "max_micro_clusters": proto["max_micro_clusters"],
                "freeze_after_window": proto["freeze_after_window"], "member_log": None,
                "config_id": config_id, "unit_override": unit_override,
            })
    return jobs


# --- running the sweep -----------------------------------------------------------------

def _job_key(job: dict) -> tuple:
    return (job["algo"], job["fold"], job["seed"], job.get("config_id"), job["n_samples"])


def _row_key(row: dict) -> tuple:
    return (row["algo"], row["fold"], row["seed"], row.get("config_id"), row.get("n_samples_cap"))


def run_sweep(
    jobs: list[dict],
    out_dir,
    workers: int = 1,
    timeout: int = C.TIMEOUT_S,
    retry_failed: bool = False,
    force: bool = False,
) -> None:
    """Run every job in ``jobs`` not already on file, resuming by its key.

    A job counts as done -- and is skipped -- when a row with its
    ``(algo, fold, seed, config_id, n_samples_cap)`` key is already in
    ``sweep.jsonl``, whatever its status: a sweep point that errored once
    (infeasible configuration, OOM, timeout) is not worth paying for again on
    every resume. ``retry_failed=True`` excludes those non-``ok`` rows from
    "done", so they are retried.
    """
    out = Path(out_dir)
    sweep_path = out / "sweep.jsonl"
    windows_path = out / "sweep_windows.jsonl"

    done = {_row_key(r) for r in C.read_jsonl(sweep_path)}
    if retry_failed:
        done -= {_row_key(r) for r in C.read_jsonl(sweep_path) if r.get("status") != "ok"}
    todo = [j for j in jobs if _job_key(j) not in done]
    if not todo:
        return

    with C.JsonlLock(sweep_path, force=force):
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(run.execute, j, timeout): j for j in todo}
            for fut in as_completed(futures):
                job = futures[fut]
                rows = fut.result()
                for row in rows:
                    if row.get("kind") == "window":
                        C.append_jsonl(windows_path, row)
                summary = rows[-1]
                C.append_jsonl(sweep_path, summary)
                status = summary.get("status")
                extra = (
                    f"lag_ari={summary.get('lag_ari_win_mean')} wall={summary.get('wall_s', 0):.0f}s"
                    if status == "ok" else summary.get("error")
                )
                print(f"[{status}] {job['algo']} {job['fold']} seed={job['seed']} "
                      f"config_id={job.get('config_id')}: {extra}", flush=True)


# --- the oracle --------------------------------------------------------------------------

def select_oracle(out_dir, metric: str = "lag_ari_win_mean") -> dict:
    """Per (algo, fold), the ``ok`` sweep row with the best ``metric``.

    Rows whose ``metric`` is missing or ``None`` never win, even if their
    ``status`` says ``"ok"``. Ties go to the smaller ``config_id`` (the
    earlier, cheaper point). An (algo, fold) the sweep ran but produced no
    such point for still gets an entry, with ``config_id``, ``unit`` and
    ``params`` None and ``n_ok`` 0: ``run.py`` skips it with that reason and
    the report tells it apart from an (algo, fold) never swept. Writes
    ``<out_dir>/oracle_static.json`` and returns the same dict.
    """
    out = Path(out_dir)
    groups: dict[tuple, list[dict]] = {}
    for row in C.read_jsonl(out / "sweep.jsonl"):
        groups.setdefault((row["algo"], row["fold"]), []).append(row)

    oracle: dict = {}
    for (algo, fold), rows in groups.items():
        ok_rows = [r for r in rows if r.get("status") == "ok" and r.get(metric) is not None]
        if not ok_rows:
            oracle.setdefault(algo, {})[fold] = {
                "config_id": None, "unit": None, "params": None, "n_ok": 0, "n_error": len(rows),
            }
            continue
        best = max(ok_rows, key=lambda r: (r[metric], -r["config_id"]))
        oracle.setdefault(algo, {})[fold] = {
            "config_id": best["config_id"],
            "unit": best.get("unit_override"),
            "params": best.get("final_params"),
            metric: best[metric],
            "n_ok": len(ok_rows),
            "n_error": len(rows) - len(ok_rows),
        }
    C.write_json(out / "oracle_static.json", oracle)
    return oracle


# --- CLI -----------------------------------------------------------------------------------

def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", default=",".join(C.ALGOS))
    parser.add_argument("--folds", default=",".join(C.FOLDS))
    parser.add_argument("--n-points", type=int, default=C.SWEEP_N_POINTS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=C.N_SAMPLES_CAP, help="0 = whole stream")
    parser.add_argument("--h", type=int, default=C.H)
    parser.add_argument("--out-dir", default=str(C.OUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=C.TIMEOUT_S)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--select-only", action="store_true",
                        help="skip the sweep itself; just (re)select the oracle from sweep.jsonl")
    args = parser.parse_args()

    out = Path(args.out_dir)
    if not args.select_only:
        algos = _csv(args.algos)
        unknown = [a for a in algos if a not in SPACES]
        if unknown:
            raise SystemExit(f"no search space for {unknown}; known: {sorted(SPACES)}")
        n_samples = args.n_samples or None
        jobs = build_sweep_jobs(
            algos, _csv(args.folds), n_points=args.n_points, seed=args.seed,
            n_samples=n_samples, h=args.h,
        )
        print(f"{len(jobs)} sweep jobs")
        if args.dry_run:
            for j in jobs:
                print(f"  {j['algo']:10s} {j['fold']:28s} config_id={j['config_id']}")
            return
        run_sweep(
            jobs, out, workers=args.workers, timeout=args.timeout,
            retry_failed=args.retry_failed, force=args.force,
        )

    oracle_path = out / C.ORACLE_JSON.name
    select_oracle(out)
    print(f"wrote {oracle_path}")


if __name__ == "__main__":
    main()
