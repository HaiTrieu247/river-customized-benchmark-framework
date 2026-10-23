"""What a planned run would cost, read off what measured runs cost.

A job's cost is the mean ``wall_s`` of the same (algo, fold, arm) over the seeds of a measured
out dir (``--from``), scaled linearly from that run's sample cap to the planned one and by
``--factor`` (an overhead the measured runs did not pay: a gate-free run keeps slow members, a
loaded machine...). An arm never run there is read through ``--proxy`` (``frozen_reinit=confstream``):
which arm stands in for which is a stated assumption, printed beside the projection. A cell with
no measurement reads unknown, and is counted, never guessed. The wall time schedules the jobs on
``--workers``, longest first (``makespan``), beside the bound no schedule beats
(``max(total / workers, longest job)``). Wall times of the measured run are process times on its
machine, not a hardware-independent CPU measure.

    python -m conf_pipeline.cost --from conf_pipeline/out/full --algos ... --folds ... \\
        --arms default,frozen_reinit --seeds 1,2,3 --workers 8 --proxy frozen_reinit=confstream
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import numpy as np

from . import _common as C


def cell_costs(src: Path) -> dict:
    """(algo, fold, arm) -> (mean wall_s over the ok runs, their sample cap)."""
    cells: dict = {}
    for r in C.read_jsonl(Path(src) / "runs.jsonl"):
        if r.get("status") != "ok" or r.get("wall_s") is None:
            continue
        cells.setdefault((r["algo"], r["fold"], r["arm"]), []).append((float(r["wall_s"]), r.get("n_samples_cap")))
    return {k: (float(np.mean([w for w, _ in v])), v[0][1]) for k, v in cells.items()}


def plan(src: Path, algos, folds, arms, seeds, n_samples: int | None = C.N_SAMPLES_CAP,
         proxy: dict | None = None, factor: float = 1.0) -> list[dict]:
    """One row per planned job with its projected ``cost_s`` (None when unmeasured)."""
    cells, proxy = cell_costs(src), dict(proxy or {})
    jobs = []
    for algo in algos:
        for fold in folds:
            for arm in arms:
                source = proxy.get(arm, arm)
                hit = cells.get((algo, fold, source))
                cost = None
                if hit is not None:
                    wall, cap = hit
                    scale = (n_samples / cap) if (n_samples and cap) else 1.0
                    cost = wall * scale * factor
                for seed in seeds:
                    jobs.append({"algo": algo, "fold": fold, "arm": arm, "seed": seed, "source": source,
                                 "cost_s": cost})
    return jobs


def summarise(jobs: list[dict], workers: int) -> dict:
    known = sorted((j["cost_s"] for j in jobs if j["cost_s"] is not None), reverse=True)
    loads = [0.0] * max(1, workers)
    heapq.heapify(loads)
    for c in known:
        heapq.heappush(loads, heapq.heappop(loads) + c)
    total = float(sum(known))
    return {
        "n_jobs": len(jobs), "n_unknown": sum(1 for j in jobs if j["cost_s"] is None),
        "cpu_h": total / 3600, "makespan_h": max(loads) / 3600,
        "lower_bound_h": max(total / max(1, workers), known[0] if known else 0.0) / 3600,
    }


def _csv(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="src", required=True, help="a measured out dir (its runs.jsonl)")
    parser.add_argument("--algos", default=",".join(C.ALGOS))
    parser.add_argument("--folds", default=",".join(C.FOLDS))
    parser.add_argument("--arms", required=True)
    parser.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    parser.add_argument("--n-samples", type=int, default=C.N_SAMPLES_CAP, help="0 = the measured run's cap")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--proxy", default="", help="comma-separated arm=measured_arm")
    args = parser.parse_args()
    proxy = dict(p.split("=", 1) for p in _csv(args.proxy))
    jobs = plan(Path(args.src), _csv(args.algos), _csv(args.folds), _csv(args.arms),
                [int(s) for s in _csv(args.seeds)], args.n_samples or None, proxy, args.factor)
    s = summarise(jobs, args.workers)
    print(f"{s['n_jobs']} jobs ({s['n_unknown']} unmeasured, left out), {s['cpu_h']:.2f} CPU-h, "
          f"wall on {args.workers} workers: {s['makespan_h']:.2f} h longest-first "
          f"(no schedule under {s['lower_bound_h']:.2f} h)")
    if proxy:
        print("proxies (assumed): " + ", ".join(f"{a} read as {b}" for a, b in proxy.items()))
    if args.factor != 1.0:
        print(f"factor {args.factor:g} applied to every measured cost")


if __name__ == "__main__":
    main()
