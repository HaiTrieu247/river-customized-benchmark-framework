"""Cluster identity audit: does the sequential partition of a window mean what the window scores read?

For every (algorithm, stream, configuration, seed) cell, one member is built as ``run.py`` builds it (River's
defaults, ``spaces.start_params``, or ``scaled_default``), warm-started on window 0 like the ensemble, then
run predict-before-learn over the later windows (``benchmark.tuning.cluster_identity.replay_window``). The
streams are the fidelity scenarios, where the events are known (``fidelity.make_scenario``: ``stationary``,
``birth``, ``deletion``, ``merge``, ``split``, ``identity_stress``), and any protocol fold (``--folds``).
Per window it reads four partitions of the same points:

- ``seq``      the sequential labels the tuner scores (point i labelled after learning 0..i-1);
- ``tracked``  the same predictions under identities carried by ``IdentityTracker`` (no evaluation label
               is read; one tracker per cell, so identities also carry from window to window);
- ``pre``      the snapshot of the model as it stood when the window began (one state, out of sample);
- ``post``     the snapshot of the model once it has learned the window (one state, in sample).

and writes, under ``<out>/``:

- ``intervals.csv``  every ``probe_every`` points, the transition of the window's snapshot since the last
  probe (survived, absorbed, split, disappeared, emerged; labels kept, renamed, reused), the micro-cluster
  births and deaths in between, its ``cause`` (``none``, ``rename_only``: the macro clusters were rebuilt
  under other names; ``structural_micro``: the partition changed with micro-level objects born or gone;
  ``structural_macro_only``: it changed with the same micro-level objects; ``structural_unattributed``: the
  model exposes none) and the scenario's true events in the interval;
- ``windows.csv``    ARI, NMI against the true classes and the classic silhouette of each partition, their
  agreements (``ari_seq_vs_post``...), and the window's interval counts;
- ``pairs.csv``      pair consistency by lag (``pair_consistency``): seq and tracked against post and against
  the truth, pre and post against the truth;
- ``cells.csv``      the cell, its parameters, its tracker's counts and the full-run readings: the mean of the
  window ARIs of each partition beside the ARI of the concatenated sequential and tracked labels over all
  windows (``cum_ari_seq``, ``cum_ari_tracked``), the reading a whole-run score makes;
- ``validation.csv`` where HPStream and HPStreamStableLabels ran on the same cell: the stable variant's
  labels are the exact identity-safe partition of the same model, against which the raw and the tracked
  HPStream labels are read; and the tracker on the stable variant, which should change nothing;
- ``identity_audit.md`` the tables, and ``run.json`` the command, the settings and the machine.

``--read-run DIR`` instead reads a benchmark run made with ``run --snapshot-eval`` (its ``windows.jsonl``)
and writes ``DIR/report/snapshot_eval/``: per algorithm, stream and arm, the lagged incumbent's sequential
partition beside the snapshots of one model state, at the start of the window and at its end.

Every threshold is a runnable placeholder, not a recommendation (``--tau``, ``--tau-split``,
``--anchors-per-id``, ``--max-anchors``, ``--lag-edges``). Nothing is tuned, no benchmark result file is
written, and no label is ever corrected in a campaign: this is evidence beside it.

    PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.identity_audit --out result/identity_audit_smoke \\
        --algos SOStream,HPStream,HPStreamStableLabels --scenarios birth,merge --seeds 1 --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.tuning.cluster_identity import (
    LAG_EDGES, TAU, TAU_SPLIT, IdentityTracker, pair_consistency, replay_window,
)
from benchmark.tuning.window import silhouette_status

from . import _common as C
from . import fidelity as F

ALGOS = F.ALGOS
CONFIGS = F.CONFIGS
SCENARIOS = F.SCENARIOS
PARTITIONS: tuple[str, ...] = ("seq", "tracked", "pre", "post")
#: (labels, reference) pairs ``pairs.csv`` reads.
PAIR_READINGS: tuple[tuple[str, str], ...] = (("seq", "post"), ("tracked", "post"), ("seq", "truth"),
                                              ("tracked", "truth"), ("pre", "truth"), ("post", "truth"))
TRANSITION_COUNTS: tuple[str, ...] = ("survived", "absorbed", "absorb_events", "split", "disappeared", "emerged",
                                      "offspring", "kept", "renamed", "reused_labels", "reused_on_emergence",
                                      "reused_on_offspring", "fresh_labels", "retired_labels")
CAUSES: tuple[str, ...] = ("none", "rename_only", "structural_micro", "structural_macro_only",
                           "structural_unattributed")
#: Seconds per cell measured on the 08/11 smoke (fidelity scenarios, h 500, default configuration, one
#: Apple M3 Pro core): only the order in which cells are handed to the workers, the costliest first, so a
#: long cell does not start last. Results never depend on it.
COST_HINT: dict[str, float] = {"BIRCH": 130.0, "DenStream": 75.0, "DenStreamPatchedFast": 13.0, "CluStream": 8.0,
                               "RepStream": 7.0, "SOStream": 4.0, "DBSTREAM": 2.0, "HPStream": 1.0,
                               "HPStreamStableLabels": 1.0, "STREAMKMeans": 1.0}
#: Placeholders, every one a flag.
DEFAULTS = dict(h=500, windows=10, fold_h=1000, fold_windows=20, probe_every=10, refresh_every=1,
                anchors_per_id=5, max_anchors=200)


# --- readings ---------------------------------------------------------------------------------------

def _truth_metrics(truth, labels) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    both = [(t, p) for t, p in zip(truth, labels) if t is not None and p is not None]
    if len({t for t, _ in both}) < C.MIN_TRUE_CLASSES:
        return {"ari": None, "nmi": None}
    ids: dict = {}
    t = [a for a, _ in both]
    p = [ids.setdefault(b, len(ids)) for _, b in both]
    return {"ari": float(adjusted_rand_score(t, p)), "nmi": float(normalized_mutual_info_score(t, p))}


def _agreement(a, b) -> float | None:
    from sklearn.metrics import adjusted_rand_score

    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(both) < 2:
        return None
    ia, ib = {}, {}
    return float(adjusted_rand_score([ia.setdefault(x, len(ia)) for x, _ in both],
                                     [ib.setdefault(y, len(ib)) for _, y in both]))


def _silhouette(xs, labels) -> tuple[float | None, str]:
    try:
        st = silhouette_status(xs, labels)
    except Exception as exc:  # a reading that fails is recorded, never a stop
        return None, f"error: {type(exc).__name__}"
    return (float(st.score) if st.defined else None), st.reason


def cause_of(row: dict) -> str:
    """The ``cause`` of one interval (see the module docstring)."""
    if row["kind"] == "none":
        return "none"
    if row["kind"] == "rename_only":
        return "rename_only"
    born, died = row.get("micro_born"), row.get("micro_died")
    if born is not None and died is not None:
        return "structural_micro" if (born or died) else "structural_macro_only"
    a, b = row.get("n_micro_before"), row.get("n_micro_after")
    if a is not None and b is not None and a != b:
        return "structural_micro"
    return "structural_unattributed"


# --- streams ----------------------------------------------------------------------------------------

def _scenario_stream(name: str, seed: int, shape: F.StreamShape, stream_scaling: str = "none",
                     scaling_window: int | None = None):
    scen = F.make_scenario(name, seed, shape)
    xs, ys = scen.points(), scen.y.tolist()
    h = shape.h
    xs = F.scaled_scenario(xs, ys, [], h, C.stream_scaling_of(name, stream_scaling), scaling_window)[0]
    windows = [(xs[w * h:(w + 1) * h], ys[w * h:(w + 1) * h]) for w in range(shape.n_windows)]
    events = [dict(e) for e in scen.events]
    return windows, events, scen.phases


def _fold_stream(fold: str, h: int, n_windows: int, data_seed: int, stream_scaling: str = "none",
                 scaling_window: int | None = None):
    from benchmark.datasets.scaling import scale_stream

    data = C.load_fold(fold, n_samples=h * (n_windows + 1), data_seed=data_seed)
    scaling = C.stream_scaling_of(fold, stream_scaling or "none")
    if scaling != "none":
        data = scale_stream(data, scaling, C.scaling_window(scaling_window, h))
    xs, ys, out = [], [], []
    for x, y in data:
        xs.append(dict(x))
        ys.append(y)
        if len(xs) == h:
            out.append((xs, ys))
            xs, ys = [], []
            if len(out) == n_windows + 1:
                break
    return out, [], ["warmup"] + [""] * (len(out) - 1)


# --- one cell ---------------------------------------------------------------------------------------

def audit_cell(task: dict) -> dict:
    """One (algorithm, stream, configuration, seed) cell; never raises (an error is recorded on the cell)."""
    from .run import _make_adapter_factory

    t0 = time.perf_counter()
    algo, config, source, stream, seed = task["algo"], task["config"], task["source"], task["stream"], task["seed"]
    base = {"source": source, "stream": stream, "seed": seed, "config": config, "algo": algo}
    out = {"key": (source, stream, seed, config, algo), "cell": dict(base), "windows": [], "intervals": [],
           "pairs": [], "labels": {}}
    try:
        if source == "scenario":
            shape = F.StreamShape(**task["shape"])
            windows, events, phases = _scenario_stream(stream, seed, shape, task["stream_scaling"],
                                                       task["scaling_window"])
            h = shape.h
        else:
            h = task["fold_h"]
            windows, events, phases = _fold_stream(stream, h, task["fold_windows"], task["data_seed"],
                                                   task["stream_scaling"], task["scaling_window"])
        if len(windows) < 2:
            raise ValueError(f"stream shorter than two windows of {h}")
        d = len(windows[0][0][0])
        params, s, record = F.cell_params(algo, config, h, d, windows[0][0])
        out["cell"].update(params=params, s=s, scaled=record, h=h, n_windows=len(windows) - 1)
        adapter = _make_adapter_factory(algo, h)(params, task["model_seed"])
        for x in windows[0][0]:
            adapter.learn_only(dict(x))
        tracker = IdentityTracker(anchors_per_id=task["anchors_per_id"], max_anchors=task["max_anchors"],
                                  tau=task["tau"], tau_split=task["tau_split"]) if task["refresh_every"] else None
        keep_labels = algo in ("HPStream", "HPStreamStableLabels")
        cum = {"truth": [], "seq": [], "tracked": []}
        for w in range(1, len(windows)):
            xs, ys = windows[w]
            lo = w * h
            tw = time.perf_counter()
            r = replay_window(adapter, xs, tracker=tracker, probe_every=task["probe_every"] or None,
                              tau=task["tau"], tau_split=task["tau_split"],
                              refresh_every=max(1, task["refresh_every"] or 1))
            replay_s = time.perf_counter() - tw
            parts = {"seq": r.sequential, "tracked": r.tracked, "pre": r.pre, "post": r.post}
            wbase = {**base, "window": w, "phase": phases[w] if w < len(phases) else "", "start": lo,
                     "end": lo + len(xs) - 1}
            row = {**wbase, "n_points": len(xs), "n_true_classes": len({y for y in ys if y is not None}),
                   "events": " ".join(e["event"] for e in events if lo <= e["t"] < lo + len(xs)),
                   "replay_s": round(replay_s, 4)}
            for name, labels in parts.items():
                if labels is None:
                    continue
                m = _truth_metrics(ys, labels)
                sil, reason = _silhouette(xs, labels)
                row.update({f"ari_{name}": m["ari"], f"nmi_{name}": m["nmi"], f"silhouette_{name}": sil,
                            f"silhouette_reason_{name}": reason,
                            f"n_labels_{name}": len({p for p in labels if p is not None})})
            row["ari_seq_vs_post"] = _agreement(r.sequential, r.post)
            row["ari_seq_vs_pre"] = _agreement(r.sequential, r.pre)
            row["ari_pre_vs_post"] = _agreement(r.pre, r.post)
            if r.tracked is not None:
                row["ari_tracked_vs_post"] = _agreement(r.tracked, r.post)
                row["ari_seq_vs_tracked"] = _agreement(r.sequential, r.tracked)
            for iv in r.intervals:
                a, b = lo + iv["t_prev"] + 1, lo + iv["t"]
                iv_row = {**wbase, **iv, "cause": cause_of(iv),
                          "events": " ".join(e["event"] for e in events if a <= e["t"] <= b)}
                out["intervals"].append(iv_row)
            for key in TRANSITION_COUNTS:
                row[f"iv_{key}"] = sum(iv[key] for iv in r.intervals)
            for cause in CAUSES:
                row[f"iv_cause_{cause}"] = sum(1 for iv in out["intervals"][-len(r.intervals):]
                                               if iv["cause"] == cause) if r.intervals else 0
            row["n_intervals"] = len(r.intervals)
            out["windows"].append(row)
            refs = {"post": r.post, "truth": ys, "seq": r.sequential, "tracked": r.tracked, "pre": r.pre}
            for lab, ref in task["pair_readings"]:
                if refs.get(lab) is None or refs.get(ref) is None:
                    continue
                for pr in pair_consistency(refs[lab], refs[ref], task["lag_edges"]):
                    out["pairs"].append({**wbase, "labels": lab, "reference": ref, **pr})
            cum["truth"] += ys
            cum["seq"] += r.sequential
            if r.tracked is not None:
                cum["tracked"] += r.tracked
            if keep_labels:
                out["labels"][w] = {"seq": r.sequential, "tracked": r.tracked}
        cells = out["cell"]
        for name in PARTITIONS:
            vals = [wr.get(f"ari_{name}") for wr in out["windows"] if wr.get(f"ari_{name}") is not None]
            cells[f"mean_window_ari_{name}"] = float(np.mean(vals)) if vals else None
        cells["cum_ari_seq"] = _truth_metrics(cum["truth"], cum["seq"])["ari"]
        cells["cum_ari_tracked"] = _truth_metrics(cum["truth"], cum["tracked"])["ari"] if cum["tracked"] else None
        if tracker is not None:
            cells.update({f"tracker_{k}": v for k, v in tracker.stats.items()})
        cells["n_sentinel"] = adapter.n_sentinel_predictions
        cells["n_fallback"] = adapter.n_fallback_predictions
        cells["error"] = None
    except Exception as exc:  # a cell that cannot run is a finding
        out["cell"]["error"] = f"{type(exc).__name__}: {exc}"
    out["cell"]["seconds"] = round(time.perf_counter() - t0, 3)
    return out


# --- the whole audit --------------------------------------------------------------------------------

def validation_rows(cells: dict) -> list[dict]:
    """HPStream's raw and tracked labels against HPStreamStableLabels' on the same cells, and the tracker
    on the stable variant, window by window."""
    rows = []
    for key, res in cells.items():
        source, stream, seed, config, algo = key
        if algo != "HPStream":
            continue
        twin = cells.get((source, stream, seed, config, "HPStreamStableLabels"))
        if twin is None:
            continue
        for w, lab in res["labels"].items():
            st = twin["labels"].get(w)
            if st is None:
                continue
            rows.append({"source": source, "stream": stream, "seed": seed, "config": config, "window": w,
                         "ari_raw_vs_stable": _agreement(lab["seq"], st["seq"]),
                         "ari_tracked_vs_stable": (_agreement(lab["tracked"], st["seq"])
                                                   if lab["tracked"] is not None else None),
                         "ari_stable_tracked_vs_stable": (_agreement(st["tracked"], st["seq"])
                                                          if st["tracked"] is not None else None),
                         "raw_equals_stable_partition": _agreement(lab["seq"], st["seq"]) == 1.0})
    return rows


def _tasks(args) -> tuple[list[dict], list[dict]]:
    shape = F.StreamShape(h=args.h, n_windows=args.windows)
    shape.validate()
    common = {"model_seed": args.model_seed, "probe_every": args.probe_every, "refresh_every": args.refresh_every,
              "anchors_per_id": args.anchors_per_id, "max_anchors": args.max_anchors, "tau": args.tau,
              "tau_split": args.tau_split, "lag_edges": tuple(args.lag_edges), "pair_readings": PAIR_READINGS,
              "shape": dataclasses.asdict(shape), "fold_h": args.fold_h, "fold_windows": args.fold_windows,
              "data_seed": args.data_seed, "stream_scaling": args.stream_scaling,
              "scaling_window": args.scaling_window}
    tasks, skipped = [], []
    streams = [("scenario", sc, seed) for sc in args.scenarios for seed in args.seeds]
    streams += [("fold", fold, args.data_seed) for fold in args.folds]
    for source, stream, seed in streams:
        w0, d, h = None, None, args.h if source == "scenario" else args.fold_h
        if "scaled_default" in args.configs and "default" in args.configs:
            if source == "scenario":
                w0 = _scenario_stream(stream, seed, shape, args.stream_scaling, args.scaling_window)[0][0][0]
            else:
                w0 = _fold_stream(stream, h, 0, args.data_seed, args.stream_scaling, args.scaling_window)[0][0][0]
            d = len(w0[0])
        for algo in args.algos:
            for cfg in args.configs:
                if w0 is not None and cfg == "scaled_default" and (
                        F.cell_params(algo, cfg, h, d, w0)[0] == F.cell_params(algo, "default", h, d, w0)[0]):
                    skipped.append({"source": source, "stream": stream, "seed": seed, "config": cfg, "algo": algo,
                                    "reason": "scaled_default leaves every parameter as in default"})
                    continue
                tasks.append({**common, "source": source, "stream": stream, "seed": seed, "config": cfg,
                              "algo": algo})
    weight = lambda t: COST_HINT.get(t["algo"], 10.0) * (  # noqa: E731
        args.fold_h * args.fold_windows if t["source"] == "fold" else args.h * (args.windows - 1))
    tasks.sort(key=lambda t: -weight(t))
    return tasks, skipped


def _map(fn, tasks, workers: int):
    if workers <= 1:
        for t in tasks:
            yield fn(t)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, t) for t in tasks]
        for f in as_completed(futs):
            yield f.result()


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    if df is None or df.empty:
        return "(no rows)\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("–")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def summarise(intervals: pd.DataFrame, windows: pd.DataFrame, pairs: pd.DataFrame, cells: pd.DataFrame,
              validation: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The report's tables (also written as CSVs under ``summary/``)."""
    out: dict[str, pd.DataFrame] = {}
    if not intervals.empty:
        g = intervals.groupby(["algo", "cause"])
        t = g.size().rename("intervals").reset_index()
        sums = g[["kept", "renamed", "reused_labels", "survived", "absorbed", "split", "disappeared", "emerged",
                  "offspring"]].sum().reset_index()
        t = t.merge(sums, on=["algo", "cause"])
        t["renamed_share"] = t["renamed"] / (t["kept"] + t["renamed"]).replace(0, np.nan)
        out["by_cause"] = t
        ev = intervals.copy()
        ev["event"] = ev["events"].fillna("").map(lambda s: s.split()[0] if s else "")
        ev = ev[ev["source"] == "scenario"]
        if not ev.empty:
            g = ev.groupby(["algo", "event"])
            t = g.size().rename("intervals").reset_index()
            t = t.merge(g[["survived", "absorbed", "split", "disappeared", "emerged", "kept", "renamed",
                           "reused_labels"]].sum().reset_index(), on=["algo", "event"])
            t["structural_share"] = g["kind"].apply(lambda k: float((k == "structural").mean())).values
            t["rename_only_share"] = g["kind"].apply(lambda k: float((k == "rename_only").mean())).values
            t["renamed_share"] = t["renamed"] / (t["kept"] + t["renamed"]).replace(0, np.nan)
            out["by_event"] = t
    if not pairs.empty:
        g = pairs.groupby(["algo", "labels", "reference", "lag_lo"])
        t = g[["n_pairs", "n_same_labels", "n_same_reference", "n_both"]].sum().reset_index()
        t["precision"] = t["n_both"] / t["n_same_labels"].replace(0, np.nan)
        t["recall"] = t["n_both"] / t["n_same_reference"].replace(0, np.nan)
        out["pairs"] = t
    if not windows.empty:
        cols = [f"ari_{p}" for p in PARTITIONS if f"ari_{p}" in windows] + \
               [f"silhouette_{p}" for p in PARTITIONS if f"silhouette_{p}" in windows]
        out["windows"] = windows.groupby(["algo", "source", "config"])[cols].mean().reset_index()
    if not cells.empty and "cum_ari_seq" in cells:
        keep = ["algo", "source", "stream", "seed", "config"] + [c for c in (
            "mean_window_ari_seq", "cum_ari_seq", "mean_window_ari_tracked", "cum_ari_tracked",
            "mean_window_ari_pre", "mean_window_ari_post", "seconds") if c in cells]
        out["full_run"] = cells[keep]
    if not validation.empty:
        out["validation"] = validation.groupby(["source", "config"])[[
            "ari_raw_vs_stable", "ari_tracked_vs_stable", "ari_stable_tracked_vs_stable"]].mean().reset_index()
    return out


def write_report(path: Path, meta: dict, tables: dict, cells: pd.DataFrame) -> Path:
    errors = cells[cells["error"].notna()] if "error" in cells else pd.DataFrame()
    lines = [
        "# Cluster identity audit", "",
        f"Command: `{meta['command']}`", "",
        f"{meta['n_cells']} cells ({meta['n_skipped']} scaled_default cells identical to default, not run), "
        f"{meta['wall_s']:.0f} s wall on {meta['workers']} workers, {meta['cpu_s']:.0f} s summed over cells. "
        f"Probe every {meta['probe_every']} points, tracker refreshed every {meta['refresh_every']} "
        f"(0: off), {meta['anchors_per_id']} anchors per identity, at most {meta['max_anchors']}; "
        f"tau {meta['tau']}, tau_split {meta['tau_split']} (placeholders).", "",
        "Partitions of one window: `seq` the sequential labels the tuner scores; `tracked` the same predictions "
        "under label-free identities; `pre` the model at the start of the window, one state, out of sample; "
        "`post` the model once it has learned the window, one state, in sample.", "",
    ]
    if len(errors):
        lines += ["## Cells that failed", "", _md(errors[["algo", "source", "stream", "seed", "config", "error"]]), ""]
    sections = [
        ("by_cause", "1. What happens to labels at each kind of change (every probe interval)",
         "`rename_only`: the snapshot's partition is unchanged and its labels are not (a macro recomputation "
         "that only renamed); `structural_micro`: the partition changed with micro-level objects born or gone; "
         "`structural_macro_only`: it changed with the same micro-level objects. `kept`/`renamed` count the "
         "after-clusters that continue a cluster under one of its old labels or under another; `reused_labels` "
         "the labels that name, after, a cluster that does not descend from the one they named before."),
        ("by_event", "1b. The same around the scenarios' true events",
         "Intervals of the fidelity scenarios that hold a true event (birth, death, deletion, merge_start, "
         "merge_done, split_start, split_done), and those that hold none (empty event)."),
        ("pairs", "2. Are historical predictions within a window comparable? (pairs by lag)",
         "Over the pairs i < j of a window, by lag j - i: `precision` is the share of pairs under one label that "
         "the reference also puts together, `recall` the share of pairs the reference puts together that share "
         "one label. A label that keeps its meaning keeps both flat in the lag."),
        ("windows", "3. One window, four partitions (window means)",
         "ARI against the true classes and the classic silhouette of each partition, averaged over windows."),
        ("full_run", "3b. Window means against the whole-run reading",
         "`cum_ari_*` is the ARI of the concatenated labels of every window, the reading a whole-run score "
         "makes; with raw labels it joins whatever a label named in any window, with tracked identities it "
         "joins what one identity named."),
        ("validation", "4. The tracker where the exact answer is known",
         "HPStreamStableLabels is HPStream with labels never reused: the same clusters, the exact identity-safe "
         "partition. `ari_tracked_vs_stable` near 1 where `ari_raw_vs_stable` is not says the tracker undid the "
         "reuse; `ari_stable_tracked_vs_stable` below 1 measures the tracker's own splits on a model with no "
         "reuse at all."),
    ]
    for key, title, text in sections:
        lines += [f"## {title}", "", text, "", _md(tables.get(key)), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


SNAPSHOT_COLS: tuple[str, ...] = (
    "lag_snap_seq_ari_win", "lag_snap_pre_ari_win", "lag_snap_post_ari_win",
    "lag_snap_seq_nmi_win", "lag_snap_pre_nmi_win", "lag_snap_post_nmi_win",
    "lag_snap_pre_vs_seq_ari", "lag_snap_post_vs_seq_ari", "lag_snap_pre_silhouette", "lag_snap_post_silhouette",
    "lag_score")


def read_run(out_dir: Path) -> Path:
    """``--read-run``: the snapshot evaluation of a run made with ``run --snapshot-eval``."""
    out_dir = Path(out_dir)
    rows = [r for r in C.read_jsonl(out_dir / "windows.jsonl")
            if r.get("kind") == "window" and "lag_snap_seq_ari_win" in r]
    rep = out_dir / C.REPORT_DIR.name / "snapshot_eval"
    rep.mkdir(parents=True, exist_ok=True)
    md = rep / "snapshot_eval.md"
    if not rows:
        md.write_text("# Snapshot evaluation\n\nNo window row holds `lag_snap_*`: run with `--snapshot-eval`.\n",
                      encoding="utf-8")
        return md
    df = pd.DataFrame(rows)
    df["group"] = df["fold"].map(lambda f: _group(f))
    cols = [c for c in SNAPSHOT_COLS if c in df]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    per_stream = df.groupby(["algo", "fold", "group", "arm"])[cols].mean().reset_index()
    per_group = df.groupby(["algo", "group", "arm"])[cols].mean().reset_index()
    per_stream.to_csv(rep / "per_stream.csv", index=False)
    per_group.to_csv(rep / "per_group.csv", index=False)
    show = ["algo", "group", "arm", "lag_snap_seq_ari_win", "lag_snap_pre_ari_win", "lag_snap_post_ari_win",
            "lag_snap_pre_vs_seq_ari", "lag_score", "lag_snap_pre_silhouette"]
    lines = ["# Snapshot evaluation", "",
             f"Read from `{out_dir / 'windows.jsonl'}` ({len(df)} windows). For the lagged incumbent of every window: "
             "`seq` the sequential partition the lagged ARI reads; `pre` every point labelled by the model as it "
             "stood when the window began (one state, out of sample); `post` the same once it has learned the "
             "window (in sample). All three are scored by one batch computation; `lag_score` is the sequential "
             "silhouette the tuner selected by, `lag_snap_pre_silhouette` the silhouette of `pre`.", "",
             "## Window means per algorithm, stream group and arm", "",
             _md(per_group[[c for c in show if c in per_group]]), ""]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def _group(fold: str) -> str:
    try:
        return C.stream_group(fold)
    except ValueError:
        return "other"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    csv = lambda v: [x.strip() for x in v.split(",") if x.strip()]  # noqa: E731
    p.add_argument("--out", help="a new directory for the audit's files")
    p.add_argument("--read-run", metavar="DIR", help="read a run made with run --snapshot-eval instead")
    p.add_argument("--algos", type=csv, default=list(ALGOS))
    p.add_argument("--configs", type=csv, default=list(CONFIGS))
    p.add_argument("--scenarios", type=csv, default=list(SCENARIOS), help="fidelity scenarios ('' for none)")
    p.add_argument("--seeds", type=lambda v: [int(s) for s in csv(v)], default=[1, 2, 3],
                   help="scenario draws (the folds read --data-seed)")
    p.add_argument("--folds", type=csv, default=[], help="protocol folds (none by default)")
    p.add_argument("--h", type=int, default=DEFAULTS["h"], help="scenario window")
    p.add_argument("--windows", type=int, default=DEFAULTS["windows"], help="scenario windows, warm start included")
    p.add_argument("--fold-h", type=int, default=DEFAULTS["fold_h"])
    p.add_argument("--fold-windows", type=int, default=DEFAULTS["fold_windows"], help="fold windows after the warm start")
    p.add_argument("--data-seed", type=int, default=C.DATA_SEED)
    p.add_argument("--model-seed", type=int, default=C.MODEL_SEED)
    p.add_argument("--stream-scaling", default=C.STREAM_SCALING, choices=C.STREAM_SCALING_RULES,
                   help="the benchmark's stream scaling rule (default protocol since 2026-11-09)")
    p.add_argument("--scaling-window", type=int, default=C.SCALING_WINDOW, help="default: one window h")
    p.add_argument("--probe-every", type=int, default=DEFAULTS["probe_every"], help="0: no transitions")
    p.add_argument("--refresh-every", type=int, default=DEFAULTS["refresh_every"], help="0: no tracker")
    p.add_argument("--anchors-per-id", type=int, default=DEFAULTS["anchors_per_id"])
    p.add_argument("--max-anchors", type=int, default=DEFAULTS["max_anchors"])
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--tau-split", type=float, default=TAU_SPLIT)
    p.add_argument("--lag-edges", type=lambda v: [int(s) for s in csv(v)], default=list(LAG_EDGES))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="list the cells and stop")
    p.add_argument("--force", action="store_true", help="write into a non-empty --out")
    return p


def main(argv=None) -> Path | None:
    args = build_parser().parse_args(argv)
    if args.read_run:
        path = read_run(Path(args.read_run))
        print(path, flush=True)
        return path
    if not args.out:
        raise SystemExit("give --out (an audit) or --read-run DIR (a run made with --snapshot-eval)")
    from .spaces import SPACES

    for bad, what in ((set(args.algos) - set(SPACES), "algorithm (one with a protocol space)"),
                      (set(args.configs) - set(CONFIGS), "config"), (set(args.scenarios) - set(SCENARIOS), "scenario"),
                      (set(args.folds) - set(C.FOLDS) - set(getattr(C, "X7_FOLDS", ())), "fold")):
        if bad:
            raise SystemExit(f"unknown {what}: {sorted(bad)}")
    os.environ.update(C.THREAD_PINS)
    t_start = time.perf_counter()
    tasks, skipped = _tasks(args)
    print(f"identity_audit: {len(tasks)} cells ({len(skipped)} scaled_default cells identical to default, not run)",
          flush=True)
    if args.dry_run:
        for t in tasks:
            print(f"  {t['algo']:<22} {t['source']:<8} {t['stream']:<34} seed {t['seed']} {t['config']}")
        return None
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; the audit never overwrites a result (--force to write into it)")
    out.mkdir(parents=True, exist_ok=True)
    cells: dict = {}
    windows, intervals, pairs, cell_rows = [], [], [], []
    for i, res in enumerate(_map(audit_cell, tasks, args.workers), start=1):
        cells[res["key"]] = res
        windows += res["windows"]
        intervals += res["intervals"]
        pairs += res["pairs"]
        cell_rows.append(res["cell"])
        c = res["cell"]
        print(f"[{i}/{len(tasks)}] {c['algo']:<22} {c['source']:<8} {c['stream']:<28} seed {c['seed']} "
              f"{c['config']:<15} {c['seconds']:8.1f}s {'ERROR ' + c['error'] if c.get('error') else ''}", flush=True)
    validation = validation_rows(cells)
    order = ["source", "stream", "seed", "config", "algo"]
    windows = sorted(windows, key=lambda r: tuple(str(r[k]) for k in order) + (r["window"],))
    intervals = sorted(intervals, key=lambda r: tuple(str(r[k]) for k in order) + (r["window"], r["t"]))
    pairs = sorted(pairs, key=lambda r: tuple(str(r[k]) for k in order) + (r["window"], r["labels"], r["reference"],
                                                                           r["lag_lo"]))
    cell_rows = sorted(cell_rows, key=lambda r: tuple(str(r[k]) for k in order))
    dfs = {"windows": pd.DataFrame(windows), "intervals": pd.DataFrame(intervals), "pairs": pd.DataFrame(pairs),
           "cells": pd.DataFrame([{**c, "params": json.dumps(c.get("params"), default=str),
                                   "scaled": json.dumps(c.get("scaled"), default=str)} for c in cell_rows] + skipped),
           "validation": pd.DataFrame(validation)}
    for name, df in dfs.items():
        df.to_csv(out / f"{name}.csv", index=False)
    tables = summarise(dfs["intervals"], dfs["windows"], dfs["pairs"], dfs["cells"], dfs["validation"])
    (out / "summary").mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / "summary" / f"{name}.csv", index=False)
    import sklearn

    from benchmark.reports.environment import capture_environment

    meta = {"command": "python -m conf_pipeline.identity_audit " + " ".join(sys.argv[1:] if argv is None else argv),
            "args": {k: v for k, v in vars(args).items()}, "n_cells": len(tasks), "n_skipped": len(skipped),
            "workers": args.workers, "probe_every": args.probe_every, "refresh_every": args.refresh_every,
            "anchors_per_id": args.anchors_per_id, "max_anchors": args.max_anchors, "tau": args.tau,
            "tau_split": args.tau_split, "wall_s": time.perf_counter() - t_start,
            "cpu_s": float(sum(c.get("seconds") or 0.0 for c in cell_rows)), "river_commit": F._river_commit(),
            "sklearn": sklearn.__version__, "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    C.write_json(out / "run.json", {**meta, "environment": capture_environment()})
    report = write_report(out / "identity_audit.md", meta, tables, dfs["cells"])
    print(report, flush=True)
    return report


if __name__ == "__main__":
    main()
