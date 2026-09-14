"""Stage 7a -- P3: prefix-holdout, and the learning curve of *choosing*.

On a shuffled stream this is no longer a drift measurement: the prefix and the
tail come from one distribution, so there is nothing to degrade. The three
origins still carry content, but different content:

    for each n0 in {10%, 20%, 40%} of the stream:
        tune: run every candidate on [0, n0), score on [0, n0)  -> lambda*(n0)
        eval: run lambda*(n0) over the whole stream, scoring ONLY from n0 on

The model still learns from sample 0 in the eval phase -- it is an online model
and there is no reason to discard the head. Only the scoring starts at n0, which
is what ``run_once(score_from=...)`` provides.

The question it answers is: **how much data is needed before a configuration can
be chosen?** A curve that flattens early means a short warm-up suffices; one
still climbing at 40% means choosing from a small sample is choosing noise.
``P1 - P3(40%)`` is the oracle gap that remains.

This is the winner's curse along the *data* axis, complementing
``P1 - P1_holdout`` along the seed axis. Two axes, two numbers, one phenomenon.
"""

from __future__ import annotations

import argparse

import numpy as np

from ._common import (
    EXTERNAL_METRICS,
    FOLDS,
    INTERNAL_METRICS,
    PREFIX_JSONL,
    PROBE_JSON,
    SEEDS,
    SPACE_JSON,
    WINDOW_SIZE,
    append_jsonl,
    read_json,
    read_jsonl,
    require,
    shuffle_seed_for,
)

ORIGINS = (0.10, 0.20, 0.40)
#: Floor on n0. Below this a prefix cannot support a choice at all.
MIN_N0 = 5000


def _run(algo, candidate, fold, seed, probe_entry, n_samples, score_from=0):
    from benchmark.experiments import BenchmarkConfig, Setting, run_setting

    from .space import resolve
    from .stream import shuffled_stream

    params = resolve(candidate, probe_entry)
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
        n_samples=n_samples,
        n_checkpoints=5,
        window_size=WINDOW_SIZE,
        external_metrics=EXTERNAL_METRICS,
        internal_metrics=INTERNAL_METRICS,
        score_from=score_from,
    )
    dataset = shuffled_stream(fold, shuffle_seed_for(seed))
    return run_setting(setting, dataset, config, seed)


def _floor_for(candidate) -> int:
    """n0 must clear the model's own initialisation window."""
    init = candidate["shape"].get("n_samples_init", 0) or 0
    return max(MIN_N0, 10 * int(init))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", default="")
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--seeds", default=str(SEEDS[0]))
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--origins", default=",".join(str(o) for o in ORIGINS))
    args = parser.parse_args()

    require(SPACE_JSON, "Stage 3 (space.py)")
    require(PROBE_JSON, "Stage 2 (probe.py)")
    space = read_json(SPACE_JSON)
    probes = read_json(PROBE_JSON)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()] or list(space)
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    origins = [float(o) for o in args.origins.split(",") if o.strip()]

    done = {
        (r["algo"], r["dataset"], r["seed"], r["origin"]) for r in read_jsonl(PREFIX_JSONL)
    }

    from .stream import stats_of

    for algo in algos:
        candidates = space[algo]
        if args.max_configs:
            candidates = candidates[: args.max_configs]
        for fold in folds:
            length = stats_of(fold)["n_samples"]
            for seed in seeds:
                for origin in origins:
                    if (algo, fold, seed, origin) in done:
                        continue
                    n0 = max(int(origin * length), MIN_N0)
                    if n0 >= length:
                        continue

                    # tune on the prefix
                    best, best_score = None, -np.inf
                    for candidate in candidates:
                        if n0 < _floor_for(candidate):
                            continue
                        try:
                            res = _run(algo, candidate, fold, seed, probes[fold], n0)
                        except Exception:
                            continue
                        score = res.quality.get("ari")
                        runtime = res.efficiency.total_runtime_s
                        if score is None or not np.isfinite(score):
                            continue
                        key = (score, -runtime, candidate["config_id"])
                        if best is None or key > best_score:
                            best, best_score = candidate, key

                    if best is None:
                        continue

                    # evaluate on the tail, learning from 0
                    try:
                        res = _run(
                            algo, best, fold, seed, probes[fold], length, score_from=n0
                        )
                        row = {
                            "algo": algo,
                            "dataset": fold,
                            "seed": seed,
                            "origin": origin,
                            "n0": n0,
                            "length": length,
                            "config_id": best["config_id"],
                            "granularity_level": best["granularity_level"],
                            "theta": best["theta"],
                            "ari_tail": res.quality.get("ari"),
                            "ari_prefix": best_score[0],
                            "k_hat": res.efficiency.final_macro_clusters,
                            "status": "ok",
                        }
                    except Exception as exc:
                        row = {
                            "algo": algo,
                            "dataset": fold,
                            "seed": seed,
                            "origin": origin,
                            "n0": n0,
                            "config_id": best["config_id"],
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    append_jsonl(PREFIX_JSONL, row)
                    print(
                        f"{algo:14s} {fold:16s} seed={seed} n0={n0:>7} "
                        f"prefix={row.get('ari_prefix')} tail={row.get('ari_tail')}"
                    )

    print(f"wrote {PREFIX_JSONL}")


if __name__ == "__main__":
    main()
