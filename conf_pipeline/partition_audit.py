"""The sequential-partition audit (``benchmark.tuning.partition_audit``) on the protocol's own starts.

For every (algorithm, fold, arm) this builds one member as ``run.py`` builds it -- River's defaults
(``spaces.start_params``) or, for ``scaled_default``, its distance parameters in units of the
calibration window's ``s`` -- warm-starts it on the first window as the ensemble does, then runs it
over the next windows predict-before-learn and audits each of them: how often the labels of the
window were renamed (the same partition under other raw labels), how often the partition itself
changed, how often a prediction changed the model's state. Nothing is tuned and no result file of
the benchmark is written; the audit is evidence for ``docs/SEQUENTIAL_PARTITION_AUDIT.md``.

    python -m conf_pipeline.partition_audit --algos DenStream,CluStream --folds static_overlap \\
        --h 300 --windows 3 --out result/partition_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.tuning.partition_audit import audit_window

from . import _common as C
from .spaces import scaled_defaults, start_params

ROW_COLS = ["algo", "fold", "arm", "window", "n_points", "n_probes", "n_rename_events", "n_rename_reuse_events",
            "n_partition_change_events", "n_renumbering_events", "renumbered_points", "n_label_handoff_events",
            "label_handoff_points", "n_predict_mutations",
            "n_state_writes", "first_rename_at", "first_partition_change_at",
            "share_after_first_rename", "ari_sequential_vs_snapshot", "raw_agreement_sequential_vs_snapshot",
            "n_labels", "error"]


def _windows(fold: str, h: int, n_windows: int, data_seed: int, stream_scaling: str = C.STREAM_SCALING,
             scaling_window: int | None = C.SCALING_WINDOW):
    from benchmark.datasets.scaling import scale_stream

    data = C.load_fold(fold, n_samples=h * (n_windows + 1), data_seed=data_seed)
    scaling = C.stream_scaling_of(fold, stream_scaling or "none")
    if scaling != "none":
        data = scale_stream(data, scaling, C.scaling_window(scaling_window, h))
    buf, out = [], []
    for x, _ in data:
        buf.append(dict(x))
        if len(buf) == h:
            out.append(buf)
            buf = []
            if len(out) == n_windows + 1:
                break
    return out


def audit_cell(algo: str, fold: str, arm: str, h: int, n_windows: int, every: int = 1,
               data_seed: int = C.DATA_SEED, model_seed: int = C.MODEL_SEED,
               stream_scaling: str = C.STREAM_SCALING, scaling_window: int | None = C.SCALING_WINDOW) -> list[dict]:
    from .run import _make_adapter_factory

    wins = _windows(fold, h, n_windows, data_seed, stream_scaling, scaling_window)
    if len(wins) < 2:
        return [{"algo": algo, "fold": fold, "arm": arm, "error": "stream shorter than two windows"}]
    s = C.estimate_scale(wins[0], np.random.default_rng(C.CALIBRATION_SEED))
    params = start_params(algo, h, d=len(wins[0][0]))
    if arm == "scaled_default":
        params, _ = scaled_defaults(algo, params, s)
    elif arm != "default":
        raise ValueError(f"the audit reads the fixed starts: default or scaled_default, not {arm!r}")
    adapter = _make_adapter_factory(algo, h)(params, model_seed)
    for x in wins[0]:
        adapter.learn_only(dict(x))
    rows = []
    for w, xs in enumerate(wins[1:], start=1):
        row = {"algo": algo, "fold": fold, "arm": arm, "window": w}
        try:
            a = audit_window(adapter, xs, every=every)
            row.update(a.to_dict(), n_labels=len({lab for lab in a.sequential if lab is not None}))
        except Exception as exc:  # an audit failure is a finding, never a stop
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        for x in xs:  # the member itself moves on, as the tuner's would
            adapter.process_one(dict(x))
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algos", default=",".join(C.ALGOS))
    parser.add_argument("--folds", default="static_overlap")
    parser.add_argument("--arms", default="default,scaled_default")
    parser.add_argument("--h", type=int, default=300)
    parser.add_argument("--windows", type=int, default=3, help="audited windows after the warm-start window")
    parser.add_argument("--every", type=int, default=1, help="probe the window's labels every this many points")
    parser.add_argument("--data-seed", type=int, default=C.DATA_SEED)
    parser.add_argument("--stream-scaling", default=C.STREAM_SCALING, choices=C.STREAM_SCALING_RULES)
    parser.add_argument("--scaling-window", type=int, default=C.SCALING_WINDOW, help="default: one window h")
    parser.add_argument("--out", required=True, help="a new directory for audit.csv and audit.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; the audit never overwrites a result")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    csv = lambda v: [x.strip() for x in v.split(",") if x.strip()]  # noqa: E731
    for algo in csv(args.algos):
        for fold in csv(args.folds):
            for arm in csv(args.arms):
                rows += audit_cell(algo, fold, arm, args.h, args.windows, args.every, args.data_seed,
                                   stream_scaling=args.stream_scaling, scaling_window=args.scaling_window)
                print(json.dumps(rows[-1], default=str), flush=True)
    df = pd.DataFrame(rows).reindex(columns=ROW_COLS)
    df.to_csv(out / "audit.csv", index=False)
    (out / "audit.json").write_text(json.dumps({"argv": argv, "h": args.h, "windows": args.windows,
                                                "every": args.every, "data_seed": args.data_seed,
                                                "stream_scaling": args.stream_scaling,
                                                "scaling_window": args.scaling_window,
                                                "model_seed": C.MODEL_SEED}, indent=2), encoding="utf-8")
    print(out / "audit.csv")


if __name__ == "__main__":
    main()
