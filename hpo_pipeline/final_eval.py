"""Run the hyperparameter sets the pilot shipped, and collect everything to plot.

``hpo_compare`` ends with one set per (algorithm, arm), written to
``out/compare/final_sets_<n>[_rank].json`` together with the absolute
parameters each set resolves to on every dataset. This module runs those sets
again -- on the datasets and at the stream length the caller chooses -- and
keeps the full run record (whole-stream quality, per-window quality, resource
checkpoints), so a notebook can plot whether tuning helped and what it cost.

Each run is its own subprocess, for the same reasons as ``build_matrix``:
River's KMeans (inside CluStream) is reproducible only under a pinned
``PYTHONHASHSEED``, which a running notebook kernel cannot change; an address
space limit and a timeout keep one bad run from taking the batch down; and a
failed run is recorded as data. Finished runs are appended to a JSONL file and
skipped when the same run is asked for again.

The data path is the pipeline's: raw -> shuffle -> stream, no scaler, with the
permutation paired to the seed (``shuffle_seed = 1000 + seed``). Seed 1 tuned
the sets, so the default seeds here are 2 and 3.

A P2R entry is the policy as the pilot applied it: the resolution level its
rule read on each dataset's label-free curve during tuning. Only the datasets
the pilot tuned on have such a level; others are reported as missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ._common import (
    HPO_OUT,
    REPO_ROOT,
    SCALER,
    SCALER_PARAMS,
    append_jsonl,
    read_jsonl,
    shuffle_seed_for,
)
from .build_matrix import MEMORY_LIMIT_BYTES, child_env
from .stream import materialise

#: The arms a final comparison usually shows, in plotting order.
ARMS = ("default", "arm1_raw_all", "arm2_raw_lodo", "arm3_sf_lodo", "p2r_R_knee_matr_trace_win")
ARM_NAMES = {
    "default": "default",
    "arm1_raw_all": "arm 1 (raw, all datasets)",
    "arm2_raw_lodo": "arm 2 (raw, LODO)",
    "arm3_sf_lodo": "arm 3 (s(D), LODO)",
    "sf_all": "sf-all",
    "p2r_R_knee_matr_trace_win": "P2R R_knee",
    "p2r_R_max_matr_trace_win": "P2R R_max",
    "p2r_R_parsimony_matr_trace_win": "P2R R_parsimony",
    "p2r_R_max_silhouette": "P2R R_max silhouette",
}
FINAL_SETS = HPO_OUT / "compare" / "final_sets_50000.json"
RESULTS_JSONL = HPO_OUT / "final_eval" / "runs.jsonl"
SEEDS = (2, 3)
WINDOW_SIZE = 1000
N_CHECKPOINTS = 40
EXTERNAL_METRICS = ("ari", "nmi", "ami", "purity", "homogeneity", "completeness")


def load_final_sets(path: Path | str = FINAL_SETS) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run hpo_compare first")
    return json.loads(path.read_text(encoding="utf-8"))


def _blob(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def run_key(job: dict) -> tuple:
    # The params are part of the key: an arm re-chosen after new tuning runs
    # must not reuse a run of the set it used to be.
    return (job["algo"], job["dataset"], job["seed"], job["n_samples"], _blob(job["params"]))


def build_jobs(final_sets: dict, algos, datasets, arms=ARMS, seeds=SEEDS,
               n_samples: int | None = None) -> tuple[list[dict], list[dict]]:
    """Jobs to run, and the (algo, arm, dataset) cells that have no set.

    Arms that resolve to identical params on a dataset share one run; the job
    lists every arm it stands for.
    """
    by_key: dict[tuple, dict] = {}
    missing = []
    for algo in algos:
        for arm in arms:
            entry = final_sets.get(algo, {}).get(arm)
            for dataset in datasets:
                params = (entry or {}).get("resolved", {}).get(dataset)
                if params is None:
                    missing.append({"algo": algo, "arm": arm, "dataset": dataset,
                                    "reason": "no set" if entry is None
                                    else "no resolved params for this dataset"})
                    continue
                for seed in seeds:
                    job = {"algo": algo, "dataset": dataset, "seed": seed,
                           "n_samples": n_samples, "params": params, "arms": [arm]}
                    key = run_key(job)
                    if key in by_key:
                        by_key[key]["arms"].append(arm)
                    else:
                        by_key[key] = job
    return list(by_key.values()), missing


def _run_child() -> None:
    """Child entry point: one job on stdin, one full run record on stdout."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass

    from benchmark.experiments import BenchmarkConfig, Setting, run_setting

    from .stream import shuffled_stream

    job = json.loads(sys.stdin.read())
    setting = Setting(algorithm=job["algo"], params=job["params"],
                      scaler=SCALER, scaler_params=dict(SCALER_PARAMS), label=job["algo"])
    config = BenchmarkConfig(
        settings=[setting], seeds=(job["seed"],), n_samples=job["n_samples"],
        n_checkpoints=N_CHECKPOINTS, window_size=WINDOW_SIZE,
        external_metrics=EXTERNAL_METRICS,
    )
    dataset = shuffled_stream(job["dataset"], shuffle_seed_for(job["seed"]))
    result = run_setting(setting, dataset, config, job["seed"])
    sys.stdout.write(json.dumps(result.to_dict(), default=str))


def execute(job: dict, timeout: int) -> dict:
    """Run one job in a subprocess; the record says what happened."""
    record = {k: job[k] for k in ("algo", "dataset", "seed", "n_samples", "params")}
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "hpo_pipeline.final_eval", "--run-one"],
            input=json.dumps(job), capture_output=True, text=True,
            timeout=timeout, cwd=str(REPO_ROOT), env=child_env(),
        )
    except subprocess.TimeoutExpired:
        return {**record, "status": "timeout", "error": f"exceeded {timeout}s",
                "wall_s": round(time.time() - started, 1)}
    record["wall_s"] = round(time.time() - started, 1)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        status = "oom" if "MemoryError" in err or proc.returncode == -9 else "error"
        return {**record, "status": status, "error": err[-500:]}
    try:
        return {**record, "status": "ok", "result": json.loads(proc.stdout)}
    except json.JSONDecodeError:
        return {**record, "status": "error", "error": (proc.stdout or "")[-500:]}


def run_jobs(jobs: list[dict], out: Path | str = RESULTS_JSONL, workers: int = 4,
             timeout: int = 7200, retry_failed: bool = False, log=print) -> None:
    """Run the jobs not yet on disk, appending one record per run as it ends."""
    out = Path(out)
    latest = {}
    for row in read_jsonl(out):
        latest[run_key(row)] = row.get("status")
    todo = [j for j in jobs
            if run_key(j) not in latest or (retry_failed and latest[run_key(j)] != "ok")]
    log(f"{len(todo)} runs to do, {len(jobs) - len(todo)} already in {out}")
    if not todo:
        return
    for job in todo:
        materialise(job["dataset"], shuffle_seed_for(job["seed"]))
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(execute, job, timeout): job for job in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            append_jsonl(out, record)
            job = futures[future]
            log(f"  {i}/{len(todo)} {time.time() - started:6.0f}s  {job['algo']:9s} "
                f"{job['dataset']:15s} seed={job['seed']} {'/'.join(job['arms'])}: "
                f"{record['status']} ({record.get('wall_s')}s)")


def load_records(jobs: list[dict], out: Path | str = RESULTS_JSONL) -> list[dict]:
    """The newest record of every job, one per (job, arm) it stands for."""
    latest = {}
    for row in read_jsonl(Path(out)):
        latest[run_key(row)] = row
    records = []
    for job in jobs:
        row = latest.get(run_key(job))
        if row is None:
            row = {**{k: job[k] for k in ("algo", "dataset", "seed", "n_samples", "params")},
                   "status": "missing"}
        for arm in job["arms"]:
            records.append({**row, "arm": arm})
    return records


# --- tidy frames for plotting ------------------------------------------------
def summary_frame(records: list[dict]):
    """One row per (algo, arm, dataset, seed): quality and resources."""
    import pandas as pd

    rows = []
    for r in records:
        res = r.get("result") or {}
        eff = res.get("efficiency") or {}
        q = res.get("quality") or {}
        qs = res.get("quality_stream") or {}
        rows.append({
            "algo": r["algo"], "arm": r["arm"], "dataset": r["dataset"], "seed": r["seed"],
            "status": r["status"],
            **{m: q.get(m) for m in EXTERNAL_METRICS},
            "ari_window_mean": (qs.get("ari") or {}).get("mean"),
            "runtime_s": eff.get("total_runtime_s"),
            "throughput_sps": eff.get("throughput_sps"),
            "ms_per_sample": eff.get("ms_per_sample"),
            "peak_memory_mb": eff.get("peak_memory_mb"),
            "peak_process_rss_mb": eff.get("peak_process_rss_mb"),
            "mean_micro_clusters": eff.get("mean_micro_clusters"),
            "max_micro_clusters": eff.get("max_micro_clusters"),
            "final_macro_clusters": eff.get("final_macro_clusters"),
            "n_predicted_clusters": (res.get("cluster_size_diagnostics") or {})
            .get("n_predicted_clusters"),
            "n_samples": res.get("n_samples"),
        })
    return pd.DataFrame(rows)


def window_frame(records: list[dict], metric: str = "ari"):
    """Per-window and cumulative ``metric`` along the stream."""
    import pandas as pd

    rows = []
    for r in records:
        for w in (r.get("result") or {}).get("windows", []):
            rows.append({
                "algo": r["algo"], "arm": r["arm"], "dataset": r["dataset"], "seed": r["seed"],
                "position": (w["start"] + w["end"]) // 2,
                "window": (w.get("window_metrics") or {}).get(metric),
                "cumulative": w.get(metric),
            })
    return pd.DataFrame(rows)


def checkpoint_frame(records: list[dict]):
    """Resource checkpoints along the stream."""
    import pandas as pd

    rows = []
    for r in records:
        for c in (r.get("result") or {}).get("checkpoints", []):
            rows.append({
                "algo": r["algo"], "arm": r["arm"], "dataset": r["dataset"], "seed": r["seed"],
                "n_seen": c["n_seen"], "runtime_s": c["runtime_s"],
                "memory_mb": (c["memory_bytes"] / 2**20) if c.get("memory_bytes") else None,
                "rss_mb": (c["process_rss_bytes"] / 2**20) if c.get("process_rss_bytes") else None,
                "n_micro_clusters": c.get("n_micro_clusters"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--final-sets", default=str(FINAL_SETS))
    parser.add_argument("--algos", default="CluStream,DenStream")
    parser.add_argument("--datasets", default="kdd99,covtype,insects_ir_bal")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--n-samples", type=int, default=0, help="0 = full stream")
    parser.add_argument("--out", default=str(RESULTS_JSONL))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.run_one:
        _run_child()
        return
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    jobs, missing = build_jobs(
        load_final_sets(args.final_sets), split(args.algos), split(args.datasets),
        split(args.arms), [int(s) for s in split(args.seeds)], args.n_samples or None,
    )
    for m in missing:
        print(f"no set: {m['algo']} / {m['arm']} / {m['dataset']} ({m['reason']})")
    run_jobs(jobs, args.out, args.workers, args.timeout, args.retry_failed)


if __name__ == "__main__":
    main()
