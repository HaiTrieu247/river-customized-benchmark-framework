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

``--campaign <config.json>`` projects the campaign's exact inventory instead (``campaign.inventory``:
split seeds, a fixed configuration once per fixed model seed or once at all, identical scaled defaults
skipped), never the Cartesian product of arms and seeds. ``--arm-factor arm=x`` multiplies one arm's
measured cost (e.g. the gate-free overhead of the online arms, stated as an assumption),
``--algo-factor algo=x`` one algorithm's in every arm (e.g. a logging overhead measured per algorithm),
``--algo-proxy algo=measured`` reads an algorithm the measured dir never ran through another (printed as an
assumption, e.g. ``HPStreamStableLabels=HPStream``),
``--stat max`` reads a cell's slowest measured run instead of the mean (what a timeout is checked against),
``--timeout`` lists every job projected past it, and ``--out`` writes the per-job projection as CSV.
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import numpy as np

from . import _common as C


STATS = {"mean": np.mean, "max": np.max}


def cell_costs(src: Path, stat: str = "mean") -> dict:
    """(algo, fold, arm) -> (the ``stat`` of wall_s over the ok runs, mean or max; their sample cap)."""
    cells: dict = {}
    for r in C.read_jsonl(Path(src) / "runs.jsonl"):
        if r.get("status") != "ok" or r.get("wall_s") is None:
            continue
        cells.setdefault((r["algo"], r["fold"], r["arm"]), []).append((float(r["wall_s"]), r.get("n_samples_cap")))
    return {k: (float(STATS[stat]([w for w, _ in v])), v[0][1]) for k, v in cells.items()}


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


def project(src: Path, jobs: list[dict], proxy: dict | None = None, arm_factor: dict | None = None,
            factor: float = 1.0, algo_factor: dict | None = None, stat: str = "mean",
            algo_proxy: dict | None = None) -> list[dict]:
    """One row per job of an exact inventory (dicts with algo, fold, arm, seed, model_seed, n_samples): the
    measured cell its cost is read from (``proxy`` arm -> measured arm, ``algo_proxy`` algo -> measured
    algo, ``stat`` of its runs), the scaling and the factors applied (``factor`` everywhere,
    ``arm_factor[arm]`` on that arm, ``algo_factor[algo]`` on that algorithm), and ``cost_s`` (None when
    the cell was never measured)."""
    cells, proxy, arm_factor = cell_costs(src, stat), dict(proxy or {}), dict(arm_factor or {})
    algo_factor, algo_proxy = dict(algo_factor or {}), dict(algo_proxy or {})
    out = []
    for j in jobs:
        source = proxy.get(j["arm"], j["arm"])
        algo_source = algo_proxy.get(j["algo"], j["algo"])
        hit = cells.get((algo_source, j["fold"], source))
        f = factor * float(arm_factor.get(j["arm"], 1.0)) * float(algo_factor.get(j["algo"], 1.0))
        row = {k: j.get(k) for k in ("group", "algo", "fold", "arm", "seed", "model_seed")}
        row.update(source=source, proxied=source != j["arm"], algo_source=algo_source,
                   algo_proxied=algo_source != j["algo"], factor=f, measured_s=None, cap_scale=None, cost_s=None)
        if hit is not None:
            wall, cap = hit
            n = j.get("n_samples")
            scale = (n / cap) if (n and cap) else 1.0
            row.update(measured_s=wall, cap_scale=scale, cost_s=wall * scale * f)
        out.append(row)
    return out


def over_timeout(jobs: list[dict], timeout_s: float) -> list[dict]:
    """The jobs projected past ``timeout_s``: each would hold a worker until the timeout and be written
    ``timeout``. A projection, read off another run's machine and protocol."""
    return sorted((j for j in jobs if j["cost_s"] is not None and j["cost_s"] > timeout_s),
                  key=lambda j: -j["cost_s"])


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="src", required=True, help="a measured out dir (its runs.jsonl)")
    parser.add_argument("--algos", default=",".join(C.ALGOS))
    parser.add_argument("--folds", default=",".join(C.FOLDS))
    parser.add_argument("--arms", default=None)
    parser.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
    parser.add_argument("--n-samples", type=int, default=C.N_SAMPLES_CAP, help="0 = the measured run's cap")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--proxy", default="", help="comma-separated arm=measured_arm")
    parser.add_argument("--campaign", default=None,
                        help="a campaign config: project its exact inventory (--arms/--seeds/--algos/--folds unused)")
    parser.add_argument("--arm-factor", default="", help="comma-separated arm=factor, on top of --factor")
    parser.add_argument("--algo-factor", default="", help="comma-separated algo=factor, on top of the others")
    parser.add_argument("--algo-proxy", default="",
                        help="with --campaign: comma-separated algo=measured_algo, for an algorithm the measured dir "
                             "never ran (an assumption, printed)")
    parser.add_argument("--stat", default="mean", choices=sorted(STATS),
                        help="what a measured cell costs: the mean of its runs, or the slowest (max)")
    parser.add_argument("--timeout", type=float, default=None, help="list the jobs projected past this many seconds")
    parser.add_argument("--out", default=None, help="write the per-job projection here (CSV)")
    args = parser.parse_args(argv)
    proxy = dict(p.split("=", 1) for p in _csv(args.proxy))
    arm_factor = {a: float(v) for a, v in (p.split("=", 1) for p in _csv(args.arm_factor))}
    algo_factor = {a: float(v) for a, v in (p.split("=", 1) for p in _csv(args.algo_factor))}
    algo_proxy = dict(p.split("=", 1) for p in _csv(args.algo_proxy))
    if args.campaign:
        from . import campaign as K

        cfg = K.load_config(Path(args.campaign))
        inv = K.inventory(cfg, Path(cfg["out_dir"]))
        jobs = project(Path(args.src), inv["jobs"], proxy, arm_factor, args.factor, algo_factor, args.stat, algo_proxy)
        if args.timeout is None:
            args.timeout = cfg["operational"]["timeout_s"]
        print(f"campaign {cfg['name']}: {inv['n_runnable']} runnable jobs ({inv['n_skipped']} skipped as identical)")
    else:
        if not args.arms:
            raise SystemExit("give --arms (or --campaign)")
        if args.stat != "mean" or algo_factor or algo_proxy:
            raise SystemExit("--stat, --algo-factor and --algo-proxy apply to --campaign projections")
        jobs = plan(Path(args.src), _csv(args.algos), _csv(args.folds), _csv(args.arms),
                    [int(s) for s in _csv(args.seeds)], args.n_samples or None, proxy, args.factor)
        for a, f in arm_factor.items():
            for j in jobs:
                if j["arm"] == a and j["cost_s"] is not None:
                    j["cost_s"] *= f
    s = summarise(jobs, args.workers)
    print(f"{s['n_jobs']} jobs ({s['n_unknown']} unmeasured, left out), {s['cpu_h']:.2f} CPU-h, "
          f"wall on {args.workers} workers: {s['makespan_h']:.2f} h longest-first "
          f"(no schedule under {s['lower_bound_h']:.2f} h)")
    if proxy:
        print("proxies (assumed): " + ", ".join(f"{a} read as {b}" for a, b in proxy.items()))
    if algo_proxy:
        print("algorithm proxies (assumed): " + ", ".join(f"{a} read as {b}" for a, b in algo_proxy.items()))
    if args.factor != 1.0:
        print(f"factor {args.factor:g} applied to every measured cost")
    if arm_factor:
        print("arm factors (assumed): " + ", ".join(f"{a} x{f:g}" for a, f in arm_factor.items()))
    if algo_factor:
        print("algorithm factors: " + ", ".join(f"{a} x{f:g}" for a, f in algo_factor.items()))
    if args.stat != "mean":
        print(f"each measured cell costs the {args.stat} of its runs")
    unknown = sorted({(j["algo"], j["fold"], j["arm"]) for j in jobs if j["cost_s"] is None})
    if unknown:
        print(f"unmeasured cells ({len(unknown)}): " + ", ".join("/".join(u) for u in unknown[:20])
              + (" ..." if len(unknown) > 20 else ""))
    per_arm: dict = {}
    for j in jobs:
        if j["cost_s"] is not None:
            per_arm[j["arm"]] = per_arm.get(j["arm"], 0.0) + j["cost_s"] / 3600
    print("process-hours per arm: " + ", ".join(f"{a} {h:.1f}" for a, h in sorted(per_arm.items())))
    if args.timeout is not None:
        over = over_timeout(jobs, args.timeout)
        print(f"jobs projected past the timeout {args.timeout:g} s: {len(over)}")
        for j in over:
            print(f"  {j['algo']} {j['fold']} {j['arm']} seed={j['seed']}: {j['cost_s'] / 3600:.2f} h")
    if args.out:
        import csv

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=sorted({k for j in jobs for k in j}))
            w.writeheader()
            w.writerows(jobs)
        print(f"per-job projection: {args.out}")


if __name__ == "__main__":
    main()
