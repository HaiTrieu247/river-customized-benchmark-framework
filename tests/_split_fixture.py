"""A toy out dir as the split seed policy writes it, for the campaign's readers.

Real fold names (so stream groups and source blocks are the protocol's), a synthetic fold that a
real-only reading must ignore, deterministic fixed runs without a seed (``seed`` None), HPStream's
fixed runs on model seed 201, online runs on tuner seeds 101 and 102, and no ``scaled_default`` run
of HPStream and RepStream (``run.py`` skips them as identical to ``default``).
"""

from __future__ import annotations

import json
from pathlib import Path

ALGOS = ("DenStream", "HPStream", "RepStream", "SOStream")
SEEDED = {"HPStream"}
REAL = ("covtype", "kdd99", "electricity", "shuttle", "insects_abrupt_balanced", "insects_gradual_balanced")
SYNTH = ("static_overlap",)
TUNER_SEEDS = (101, 102)
MODEL_SEED = 201
TAIL = (20, 21, 22, 23)


#: 32 member-windows per run, failing in more than one way at once: 27 valid and feasible, 2 with one
#: label and feasible, 1 valid partition shed by the memory cap, 1 one-label partition shed by it too
#: (its score status says ``infeasible_memory``: the cap takes precedence), 1 member that raised.
MEMBER_COUNTS = {
    "member_status_counts": {"valid": 27, "undefined_single_label": 2, "infeasible_memory": 2, "member_error": 1},
    "member_silhouette_reason_counts": {"valid": 28, "single_label": 3, "none": 1},
    "member_reason_feasibility_counts": {"valid|feasible": 27, "single_label|feasible": 2, "valid|memory": 1,
                                         "single_label|memory": 1, "none|error": 1},
}


def base_value(algo: str, fold: str, arm: str) -> float:
    """A reading with structure: the regimes disagree, the sources disagree."""
    a = ALGOS.index(algo)
    f = (REAL + SYNTH).index(fold)
    arm_shift = {"default": 0.0, "scaled_default": 0.05, "confstream": 0.1, "frozen_prefix": 0.08,
                 "frozen_reinit": 0.09, "random_online": 0.07}[arm]
    return round(0.1 + 0.07 * ((a * 3 + f * (1 + (arm == "confstream"))) % 5) + arm_shift, 6)


def job_rows(algo: str, fold: str, arm: str, values: dict | None = None):
    """The run and window rows of every run of (algo, fold, arm) under the split policy."""
    fixed = arm in ("default", "scaled_default")
    if fixed:
        seeds = [(MODEL_SEED, None, MODEL_SEED)] if algo in SEEDED else [(None, None, None)]
    else:
        seeds = [(t, t, MODEL_SEED if algo in SEEDED else None) for t in TUNER_SEEDS]
    runs, windows = [], []
    for seed, tuner, model in seeds:
        ident = {"algo": algo, "fold": fold, "arm": arm, "seed": seed, "seed_policy": "split", "tuner_seed": tuner,
                 "model_seed": model, "calibration_seed": 0, "config_id": None}
        vals = []
        for w in TAIL:
            v = (values or {}).get(w, base_value(algo, fold, arm) + 0.001 * (w - 20) + 0.0001 * ((tuner or 0) % 7))
            vals.append(v)
            windows.append({"kind": "window", **ident, "window": w, "lag_ari_win": v,
                            "lag_nmi_win": None if v is None else v / 2})
        defined = [v for v in vals if v is not None]
        runs.append({"kind": "run", **ident, "status": "ok", "tail_from": TAIL[0], "n_samples_cap": 100000,
                     "lag_ari_win_tail_mean": sum(defined) / len(defined) if defined else None,
                     **MEMBER_COUNTS,
                     "lag_score_status_counts": {"valid": 4}, "lag_silhouette_reason_counts": {"valid": 4},
                     "n_births": 12, "n_births_ok": 12,
                     "n_member_evaluations": 32, "n_member_evaluations_ok": 31, "n_member_windows_sentinel": 1,
                     "n_member_windows_snapshot": 32, "wall_s": 10.0})
    return runs, windows


def make_out_dir(root: Path, arms=("default", "scaled_default", "confstream"), algos=ALGOS, folds=REAL + SYNTH,
                 override=None, drop=()) -> Path:
    """Write ``runs.jsonl`` + ``windows.jsonl``. ``override``: {(algo, fold, arm): {window: value}};
    ``drop``: (algo, fold, arm) cells written as an ``error`` attempt with no window."""
    from conf_pipeline.spaces import distance_params

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    runs, windows = [], []
    for algo in algos:
        for fold in folds:
            for arm in arms:
                if arm == "scaled_default" and not distance_params(algo):
                    runs.append({"kind": "run", "algo": algo, "fold": fold, "arm": arm, "seed": None,
                                 "seed_policy": "split", "tuner_seed": None, "model_seed": None,
                                 "status": "skipped", "error": "identical"})
                    continue
                r, w = job_rows(algo, fold, arm, (override or {}).get((algo, fold, arm)))
                if (algo, fold, arm) in drop:
                    r = [{**x, "status": "error", "error": "boom"} for x in r]
                    w = []
                runs += r
                windows += w
    (root / "runs.jsonl").write_text("\n".join(json.dumps(x) for x in runs) + "\n")
    (root / "windows.jsonl").write_text("\n".join(json.dumps(x) for x in windows) + "\n")
    return root
