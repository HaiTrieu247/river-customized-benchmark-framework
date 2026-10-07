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

A design of 33 points is thin in five dimensions, and an online arm that keeps
one configuration for most of the stream can beat every one of them, which
reads as headroom above 1 without any adaptation behind it. ``--incumbents``
therefore adds, per (algo, fold), the configurations the online arms settled on
(``incumbent_candidates``: the incumbent held longest and the last one, per
run), each rerun as a fixed configuration from the start of the stream with its
exact parameters (``params_override``) and a ``config_id`` derived from them.
Labels play no part in picking the candidates; they only pick the winner, as
for every other point.

``select_oracle`` then reduces the sweep to one fixed configuration per
(algo, fold) -- the one with the best lagged score on the tail, the region
every arm is compared on (the whole stream when the stream has no tail) --
and writes it to ``oracle_static.json``, which ``run.py --arms oracle_static``
reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    proto = _protocol(protocol)

    jobs: list[dict] = []
    for algo, fold in product(algos, folds):
        units = sobol_units(SPACES[algo], n_points, seed)
        for config_id, unit_override in enumerate(units):
            jobs.append(_sweep_job(algo, fold, run_seed, n_samples, h, proto, config_id,
                                   unit_override=unit_override))
    return jobs


def _protocol(protocol: dict) -> dict:
    defaults = dict(
        p_reset=C.P_RESET, half_life_frac=C.HALF_LIFE_FRAC, runtime_factor=C.RUNTIME_FACTOR,
        max_memory_mb=C.MAX_MEMORY_MB, max_micro_clusters=C.MAX_MICRO_CLUSTERS,
        freeze_after_window=C.FREEZE_AFTER_WINDOW,
    )
    unknown = sorted(set(protocol) - set(defaults))
    if unknown:
        raise TypeError(f"unknown protocol overrides {unknown}")
    return {**defaults, **protocol}


def _sweep_job(algo, fold, run_seed, n_samples, h, proto, config_id,
               unit_override=None, params_override=None) -> dict:
    """The job dict ``run.execute`` / the child expect, for one fixed configuration."""
    return {
        "algo": algo, "fold": fold, "seed": run_seed, "arm": "sweep",
        "n_samples": n_samples, "h": h, "e_size": 1, "e_test": 0,
        "p_reset": proto["p_reset"], "half_life_frac": proto["half_life_frac"], "lam": None,
        "runtime_factor": proto["runtime_factor"], "max_memory_mb": proto["max_memory_mb"],
        "max_micro_clusters": proto["max_micro_clusters"],
        "freeze_after_window": proto["freeze_after_window"], "member_log": None,
        "config_id": config_id, "unit_override": unit_override, "params_override": params_override,
    }


# --- incumbents of the online arms as candidates ----------------------------------------

#: Arms whose incumbents are candidates: the ones that pick configurations online.
ONLINE_ARMS = ("confstream", "random_online", "frozen_prefix")
#: An incumbent candidate's ``config_id`` is above every Sobol index.
INCUMBENT_ID_BASE = C.INCUMBENT_ID_BASE


def params_config_id(params: dict) -> int:
    """A ``config_id`` for exact parameters: stable across resumes and key orders."""
    blob = json.dumps(params, sort_keys=True, default=C._json_default)
    return INCUMBENT_ID_BASE + int(hashlib.sha1(blob.encode("utf-8")).hexdigest()[:7], 16)


def incumbent_candidates(out_dir, algos=None, folds=None, n_samples=None) -> dict[tuple, list[dict]]:
    """Per (algo, fold), the configurations the online arms settled on, deduplicated.

    From ``<out_dir>/windows.jsonl``, for every run of an ``ONLINE_ARMS`` arm
    (algo, fold, arm, seed): the parameters of the incumbent that held the most
    windows (ties: the earlier one), then those of the last window's incumbent.
    Each entry is ``{"config_id", "params", "sources"}``, ``sources`` naming
    every run and pick that produced it (``"confstream/seed1/longest"``), in
    arm order, then seed. Rows of another sample cap than ``n_samples`` are
    ignored when both are known.
    """
    runs: dict[tuple, list[tuple]] = {}
    for row in C.read_jsonl(Path(out_dir) / "windows.jsonl"):
        if row.get("kind") != "window" or row.get("arm") not in ONLINE_ARMS or row.get("params") is None:
            continue
        if (algos is not None and row["algo"] not in algos) or (folds is not None and row["fold"] not in folds):
            continue
        cap = row.get("n_samples_cap")
        if n_samples is not None and cap is not None and cap != n_samples:
            continue
        key = (row["algo"], row["fold"], ONLINE_ARMS.index(row["arm"]), row["seed"])
        runs.setdefault(key, []).append((row["window"], row.get("incumbent_id"), row["params"]))

    out: dict[tuple, list[dict]] = {}
    for (algo, fold, arm_i, seed) in sorted(runs):
        rows = sorted(runs[(algo, fold, arm_i, seed)], key=lambda r: r[0])
        held, first, params_of = {}, {}, {}
        for window, inc, params in rows:
            held[inc] = held.get(inc, 0) + 1
            first.setdefault(inc, window)
            params_of[inc] = params
        longest = max(held, key=lambda i: (held[i], -first[i]))
        cands = out.setdefault((algo, fold), [])
        for pick, params in (("longest", params_of[longest]), ("last", rows[-1][2])):
            cid = params_config_id(params)
            source = f"{ONLINE_ARMS[arm_i]}/seed{seed}/{pick}"
            hit = next((c for c in cands if c["config_id"] == cid), None)
            if hit is None:
                cands.append({"config_id": cid, "params": params, "sources": [source]})
            else:
                hit["sources"].append(source)
    return out


def build_incumbent_jobs(out_dir, algos, folds, seed: int, n_samples: int | None, h: int,
                         **protocol) -> list[dict]:
    """One ``arm="sweep"`` job per incumbent candidate, run with its exact parameters."""
    proto = _protocol(protocol)
    cands = incumbent_candidates(out_dir, algos=set(algos), folds=set(folds), n_samples=n_samples)
    return [
        _sweep_job(algo, fold, seed, n_samples, h, proto, c["config_id"], params_override=c["params"])
        for (algo, fold), group in sorted(cands.items()) for c in group
    ]


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

def select_oracle(out_dir, metrics: tuple[str, ...] = C.ORACLE_METRICS) -> dict:
    """Per (algo, fold), the ``ok`` sweep row with the best reading (``_common.best_static``).

    The tail decides when the sweep's rows have one, the whole stream when the
    stream is too short to have a tail; ``select_metric`` names the one used,
    and the winner's tail and whole-stream readings are both kept. Rows
    without the deciding reading never win, even if their ``status`` says
    ``"ok"``; ties go to the smaller ``config_id``. An (algo, fold) the sweep
    ran but produced no such point for still gets an entry, with
    ``config_id``, ``unit`` and ``params`` None and ``n_ok`` 0: ``run.py``
    skips it with that reason and the report tells it apart from an
    (algo, fold) never swept. Writes ``<out_dir>/oracle_static.json`` and
    returns the same dict.
    """
    out = Path(out_dir)
    groups: dict[tuple, list[dict]] = {}
    for row in C.read_jsonl(out / "sweep.jsonl"):
        groups.setdefault((row["algo"], row["fold"]), []).append(row)

    oracle: dict = {}
    for (algo, fold), rows in groups.items():
        picked = C.best_static(rows, metrics)
        if picked is None:
            oracle.setdefault(algo, {})[fold] = {
                "config_id": None, "unit": None, "params": None, "n_ok": 0, "n_error": len(rows),
            }
            continue
        best, metric = picked
        n_ok = sum(1 for r in rows if r.get("status") == "ok" and C._reading(r.get(metric)) is not None)
        oracle.setdefault(algo, {})[fold] = {
            "config_id": best["config_id"],
            "source": ("incumbent" if best.get("params_override")
                       else "default" if best["config_id"] == 0 else "sobol"),
            "unit": best.get("unit_override"),
            "params_override": best.get("params_override"),
            "params": best.get("final_params"),
            "select_metric": metric,
            **{m: best.get(m) for m in metrics},
            "n_ok": n_ok,
            "n_error": len(rows) - n_ok,
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
    parser.add_argument("--incumbents", action="store_true",
                        help="also rerun, as fixed configurations, the incumbents the online arms "
                             "settled on (read from <out-dir>/windows.jsonl)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    if not args.select_only:
        algos = _csv(args.algos)
        unknown = [a for a in algos if a not in SPACES]
        if unknown:
            raise SystemExit(f"no search space for {unknown}; known: {sorted(SPACES)}")
        n_samples = args.n_samples or None
        folds = _csv(args.folds)
        jobs = build_sweep_jobs(
            algos, folds, n_points=args.n_points, seed=args.seed, n_samples=n_samples, h=args.h,
        )
        if args.incumbents:
            incumbents = build_incumbent_jobs(out, algos, folds, seed=args.seed,
                                              n_samples=n_samples, h=args.h)
            print(f"{len(incumbents)} incumbent jobs from {out / 'windows.jsonl'}")
            jobs += incumbents
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
