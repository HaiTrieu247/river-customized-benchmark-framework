"""Stage 4 -- MATRIX: run the grid once, read it many times.

``A algorithms x |Lambda| configs x D folds x S seeds`` runs at a fixed fidelity
rung, appended to one JSONL file. Everything downstream is a *reading* of this
file, which is why it is worth running carefully once.

Three operational requirements, all of which matter at this scale:

* **Resumable.** The key ``(algo, config_id, dataset, seed, budget)`` is checked
  against what is already on disk before anything runs. Fifteen thousand runs
  will be interrupted.
* **Isolated.** Each run happens in a subprocess under an address-space limit
  and a hard timeout. A configuration that exhausts memory is a fact about that
  configuration; it must not take the batch with it.
* **Honest about failure.** ``status`` in {ok, error, oom, timeout} is data. A
  row that failed is never silently dropped later -- an algorithm whose search
  space is 60% unusable is a hard-to-use algorithm, and a table of scores alone
  would hide that.

The label-free criteria are recorded in **every** cell, not only in winning
ones, because Stage 6 reads whole 8-point curves and one missing cell is a
broken curve.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from ._common import (
    BUDGET,
    CACHE,
    EXTERNAL_METRICS,
    FOLDS,
    INTERNAL_METRICS,
    MATRIX_JSONL,
    PROBE_JSON,
    REPO_ROOT,
    SEEDS,
    SPACE_JSON,
    WINDOW_SIZE,
    append_jsonl,
    read_json,
    read_jsonl,
    require,
    shuffle_seed_for,
)

#: Address-space ceiling per run.
MEMORY_LIMIT_BYTES = 4 * 1024**3
#: Wall-clock ceiling per run.
TIMEOUT_S = 900


def run_key(row: dict) -> tuple:
    return (row["algo"], row["config_id"], row["dataset"], row["seed"], row["budget"])


def done_keys(path=MATRIX_JSONL, retry_failed: bool = False) -> set:
    """Keys already on disk.

    A failed run counts as done by default: ``status`` is data, and a config
    that cannot run is a finding about that config rather than a gap to paper
    over. ``retry_failed`` is for the other case -- when the failure was the
    harness's fault (a missing cache, a machine that ran out of disk) and the
    rows say nothing about the algorithm.
    """
    return {
        run_key(row)
        for row in read_jsonl(path)
        if not (retry_failed and row.get("status") != "ok")
    }


def _row_for(algo, candidate, fold, seed, budget, probe_entry):
    """Execute one run in-process and return its matrix row."""
    from benchmark.experiments import BenchmarkConfig, Setting, run_setting

    from .space import resolve
    from .stream import shuffled_stream

    params = resolve(candidate, probe_entry)
    shuffle_seed = shuffle_seed_for(seed)
    dataset = shuffled_stream(fold, shuffle_seed)

    setting = Setting(
        algorithm=algo,
        params=params,
        scaler=candidate["scaler"],
        scaler_params=candidate["scaler_params"],
        label=f'{algo}@{candidate["config_id"]}',
    )
    config = BenchmarkConfig(
        settings=[setting],
        seeds=(seed,),
        n_samples=budget,
        n_checkpoints=20,
        window_size=WINDOW_SIZE,
        external_metrics=EXTERNAL_METRICS,
        internal_metrics=INTERNAL_METRICS,
    )
    result = run_setting(setting, dataset, config, seed)
    eff = result.efficiency.to_dict()
    stream_stat = result.quality_stream
    sizes = result.cluster_size_diagnostics

    def win(metric, stat="mean"):
        return (stream_stat.get(metric) or {}).get(stat)

    return {
        "algo": algo,
        "config_id": candidate["config_id"],
        "granularity_level": candidate["granularity_level"],
        "theta": candidate["theta"],
        "g": candidate["g"],
        "is_default": bool(candidate.get("is_default")),
        "s_fold": probe_entry["s"],
        "params_resolved": params,
        "scaler": candidate["scaler"],
        "scaler_params": candidate["scaler_params"],
        "dataset": fold,
        "seed": seed,
        "shuffle_seed": shuffle_seed,
        "budget": result.n_samples,
        # raw-units mode: this column travels with every score
        "top1_variance_share": probe_entry["top1_variance_share"],
        "effective_dim": probe_entry["effective_dim"],
        "one_d_dominated": probe_entry.get("one_d_dominated"),
        **{m: result.quality.get(m) for m in EXTERNAL_METRICS},
        "ari_win_mean": win("ari"),
        "ari_win_median": win("ari", "median"),
        "silhouette": result.quality.get("silhouette"),
        "ssq": result.quality.get("ssq"),
        "matr_trace": result.quality.get("matr_trace"),
        # The per-window trace is the criterion P2R and P4a read: cluster ids
        # are recycled by the online-merging algorithms, so accumulating by id
        # over a whole stream can fuse two different clusters into one label.
        "matr_trace_win": win("matr_trace"),
        "silhouette_win": win("silhouette"),
        "ssq_win": win("ssq"),
        "k_hat": eff["final_macro_clusters"],
        "n_classes_total": probe_entry.get("n_classes_total"),
        "dust_cluster_share": sizes.get("dust_cluster_share"),
        "largest_cluster_share": sizes.get("largest_cluster_share"),
        "n_predicted_clusters": sizes.get("n_predicted_clusters"),
        "runtime_s": eff["total_runtime_s"],
        "throughput_sps": eff["throughput_sps"],
        "peak_memory_mb": eff["peak_memory_mb"],
        "mean_micro_clusters": eff["mean_micro_clusters"],
        "max_micro_clusters": eff["max_micro_clusters"],
        "final_macro_clusters": eff["final_macro_clusters"],
        "n_fallback_predictions": result.n_fallback_predictions,
        "rho_deg": result.window_label_diagnostics.get("single_class_window_share"),
        "status": "ok",
        "error": None,
    }


def _run_one_child() -> None:
    """Child entry point: read one job from stdin, write one row to stdout."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        # Not every platform enforces an address-space limit; the parent's
        # timeout still applies, and the row records what actually happened.
        pass

    job = json.loads(sys.stdin.read())
    probes = read_json(PROBE_JSON)
    row = _row_for(
        job["algo"],
        job["candidate"],
        job["fold"],
        job["seed"],
        job["budget"],
        probes[job["fold"]],
    )
    sys.stdout.write(json.dumps(row, default=str))


def _failed_row(job, status, error):
    candidate = job["candidate"]
    return {
        "algo": job["algo"],
        "config_id": candidate["config_id"],
        "granularity_level": candidate["granularity_level"],
        "theta": candidate["theta"],
        "g": candidate["g"],
        "is_default": bool(candidate.get("is_default")),
        "dataset": job["fold"],
        "seed": job["seed"],
        "shuffle_seed": shuffle_seed_for(job["seed"]),
        "budget": job["budget"],
        "status": status,
        "error": error[-500:] if error else None,
    }


def execute(job: dict, timeout: int) -> dict:
    """Run one job in a subprocess and classify whatever comes back."""
    payload = json.dumps(job)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "hpo_pipeline.build_matrix", "--run-one"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return _failed_row(job, "timeout", f"exceeded {timeout}s")

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        status = "oom" if "MemoryError" in err or proc.returncode == -9 else "error"
        return _failed_row(job, status, err)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _failed_row(job, "error", (proc.stdout or "")[-500:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algos", default="")
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="cap configs per algorithm (smoke tests only)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-run rows whose status is not ok (use when the harness, not the "
        "configuration, was at fault)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.run_one:
        _run_one_child()
        return

    require(SPACE_JSON, "Stage 3 (space.py)")
    require(PROBE_JSON, "Stage 2 (probe.py)")
    space = read_json(SPACE_JSON)
    probes = read_json(PROBE_JSON)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()] or list(space)
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # A missing permutation is a precondition failure, not a per-run failure:
    # without this check every single run fails identically and the matrix
    # fills up with rows that say nothing about any algorithm.
    missing = [
        f"{fold}_{shuffle_seed_for(seed)}_X.npy"
        for fold in folds
        for seed in seeds
        if not (CACHE / f"{fold}_{shuffle_seed_for(seed)}_X.npy").exists()
    ]
    if missing:
        raise SystemExit(
            "missing cached permutations -- run Stage 1 (stream.py) for these first:\n  "
            + "\n  ".join(sorted(missing))
        )

    already = done_keys(retry_failed=args.retry_failed)
    jobs = []
    for algo in algos:
        candidates = space[algo]
        if args.max_configs:
            candidates = candidates[: args.max_configs]
        for candidate in candidates:
            for fold in folds:
                if fold not in probes:
                    raise SystemExit(f"no probe for fold {fold!r}; rerun Stage 2")
                for seed in seeds:
                    key = (algo, candidate["config_id"], fold, seed, args.budget)
                    if key in already:
                        continue
                    jobs.append(
                        {
                            "algo": algo,
                            "candidate": candidate,
                            "fold": fold,
                            "seed": seed,
                            "budget": args.budget,
                        }
                    )

    print(f"{len(jobs)} runs to do ({len(already)} already on disk)")
    if args.dry_run or not jobs:
        return

    started = time.time()
    counts: dict[str, int] = {}
    done = 0

    def submit(job):
        return execute(job, args.timeout)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for row in pool.map(submit, jobs):
            append_jsonl(MATRIX_JSONL, row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            done += 1
            if done % 25 == 0 or done == len(jobs):
                rate = done / max(time.time() - started, 1e-9)
                print(
                    f"  {done}/{len(jobs)}  {rate:.2f} run/s  "
                    + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                )

    print(f"wrote {MATRIX_JSONL}")
    print("status counts:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
