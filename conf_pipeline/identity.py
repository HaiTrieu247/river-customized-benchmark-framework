"""E1: does within-window label identity change the sequential readings?

A sequential partition is read as if a label named one cluster from the first point of the window to
the last. In five of the eight implementations a label can be handed to another cluster inside a window
(``docs/SEQUENTIAL_PARTITION_AUDIT.md``, RIVER_BUGS.md GAP-6), and then the silhouette the tuner selects
by and the lagged ARI it is evaluated with both read a partition the model never held.

This module measures that, from the per-point predictions a run writes with ``--log-predictions``
(``predictions/<job>.jsonl``), without using any evaluation label to correct anything:

1. **Detection, label-free.** Within a window, every label keeps a running centroid of the last
   ``memory`` points it was given. A point given label ``L`` farther than ``kappa * s`` from ``L``'s
   centroid (``s`` the run's distance unit, the median pairwise distance of the first window), once
   ``L`` holds at least ``min_points`` points, starts a new segment of ``L``: from there on the points
   are read as another cluster (``L#1``, ``L#2``, ...). ``kappa``, ``min_points`` and ``memory`` are
   placeholders, read at several values (``--kappa``); a genuine cluster wider than ``kappa * s`` is
   split too, which is why the detection is validated below.
2. **Corrected readings.** For every member-window: the raw and the corrected sequential ARI (on the
   points with a label and a prediction) and silhouette (the tuner's own ``silhouette_status``), and
   the number of detected changes.
3. **Selection replay.** Per window, the member the corrected silhouette would have picked among the
   members that were feasible, read one window later on its corrected ARI, beside the raw replay (the
   raw silhouette's pick read on its raw ARI) and the logged lagged ARI.
4. **Validation.** Where a dir holds both ``HPStream`` and ``HPStreamStableLabels`` on the same fixed
   configuration (the ``default`` arm: one member, no selection, so both run the same model), the
   stable variant's labels are the exact identity-safe partition: every point where an HPStream label
   changes its stable label is a true reuse. The detector's precision and recall against those events,
   and the ARI between the corrected and the stable partitions, say how far the detection can be
   trusted. Nothing validates it for DenStream, DBSTREAM, BIRCH or CluStream, where no exact rule exists.

Run ``python -m conf_pipeline.identity --out-dir DIR`` (after a run with ``--log-predictions``; members
logs make the replay use the members' feasibility). Writes ``DIR/report/identity/``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from benchmark.tuning.window import silhouette_status

from . import _common as C

#: Placeholders (AGENTS.md rule 2): a change is a point farther than ``kappa * s`` from its label's
#: running centroid of the last ``memory`` points, once the label holds ``min_points`` of them.
DEFAULT_KAPPAS: tuple[float, ...] = (0.5, 1.0, 2.0)
DEFAULT_MIN_POINTS = 5
DEFAULT_MEMORY = 50
#: A detected change counts as a hit when it lies at most this many points after a true one.
VALIDATION_TOLERANCE = 5
PREDICTIONS_DIR = "predictions"


def detect_identity_changes(X: np.ndarray, labels, s: float, kappa: float = 1.0,
                            min_points: int = DEFAULT_MIN_POINTS, memory: int = DEFAULT_MEMORY):
    """``(corrected_labels, change_positions)`` for one sequential partition (see the module docstring).
    A ``None`` label stays ``None``; with ``s <= 0`` nothing is detected."""
    recent: dict = {}
    segment: dict = defaultdict(int)
    out, events = [], []
    threshold = kappa * s if s and s > 0 else np.inf
    for i, lab in enumerate(labels):
        if lab is None:
            out.append(None)
            continue
        q = recent.get(lab)
        if q is not None and len(q) >= min_points:
            if float(np.linalg.norm(X[i] - np.mean(q, axis=0))) > threshold:
                segment[lab] += 1
                events.append(i)
                q = None
        if q is None:
            q = recent[lab] = deque(maxlen=memory)
        q.append(X[i])
        out.append(f"{lab}#{segment[lab]}")
    return out, events


def exact_reuse_events(raw, stable) -> list[int]:
    """Positions where a raw label starts naming another cluster of the identity-safe partition
    ``stable`` (same points, same model): the label's stable counterpart changes."""
    last: dict = {}
    events = []
    for i, (a, b) in enumerate(zip(raw, stable)):
        if a is None or b is None:
            continue
        if a in last and last[a] != b:
            events.append(i)
        last[a] = b
    return events


def match_events(detected: list[int], true: list[int], tol: int = VALIDATION_TOLERANCE) -> dict:
    """Precision and recall of ``detected`` against ``true``: a detection hits when it lies in
    ``[e, e + tol]`` of an unmatched true event ``e``."""
    true_left = sorted(true)
    hits = 0
    for d in sorted(detected):
        for j, e in enumerate(true_left):
            if e <= d <= e + tol:
                hits += 1
                true_left.pop(j)
                break
    return {"n_detected": len(detected), "n_true": len(true), "n_hits": hits,
            "precision": hits / len(detected) if detected else np.nan,
            "recall": hits / len(true) if true else np.nan}


def _ari(ys, preds) -> float | None:
    keep = [i for i, (y, p) in enumerate(zip(ys, preds)) if y is not None and p is not None]
    if len(keep) < 2 or len({ys[i] for i in keep}) < C.MIN_TRUE_CLASSES:
        return None
    return float(adjusted_rand_score([ys[i] for i in keep], [str(preds[i]) for i in keep]))


def _silhouette(xs, preds) -> float | None:
    st = silhouette_status(xs, [None if p is None else str(p) for p in preds])
    return float(st.score) if st.defined else None


def _job_key(row: dict) -> tuple:
    return (row["algo"], row["fold"], row.get("seed"), row["arm"], row.get("config_id"), row.get("model_seed"))


class _Streams:
    """Each (fold, cap, data seed, prefix scaling) stream read once, as dicts and as an array."""

    def __init__(self):
        self.cache: dict = {}

    def get(self, run: dict):
        key = (run["fold"], run.get("n_samples_cap"), run.get("data_seed"), run.get("prefix_scaling"), run.get("h"),
               run.get("stream_scaling"), run.get("scaling_window"))
        if key not in self.cache:
            from .run import PrefixStandardised

            data = C.load_fold(run["fold"], n_samples=run.get("n_samples_cap"),
                               data_seed=run.get("data_seed") if run.get("data_seed") is not None else C.DATA_SEED)
            if run.get("prefix_scaling") not in (None, "none"):
                data = PrefixStandardised(data, run["h"])
            from benchmark.datasets.scaling import scale_stream

            # the coordinates the run's models saw: a run records its stream scaling (none before 2026-11-09)
            data = scale_stream(data, run.get("stream_scaling"), run.get("scaling_window"))
            xs, ys = [], []
            for x, y in data:
                if run.get("n_samples_cap") and len(xs) >= run["n_samples_cap"]:
                    break
                xs.append(dict(x))
                ys.append(y)
            keys = sorted(xs[0]) if xs else []
            X = np.asarray([[float(x.get(k, 0.0)) for k in keys] for x in xs], dtype=float)
            self.cache[key] = (xs, ys, X)
        return self.cache[key]


def analyse_job(run: dict, windows: list[dict], predictions: list[dict], members: list[dict], streams: _Streams,
                kappas=DEFAULT_KAPPAS, min_points=DEFAULT_MIN_POINTS, memory=DEFAULT_MEMORY):
    """Member-window readings and the replay of one job."""
    xs, ys, X = streams.get(run)
    span = {w["window"]: (w["start"], w["end"]) for w in windows}
    feasible = {(m["window"], m["member_id"]): bool(m.get("feasible", True)) for m in members}
    s = float(run.get("s_online") or 0.0)
    base = {k: run.get(k) for k in ("algo", "fold", "seed", "arm", "config_id", "model_seed")}
    rows = []
    for p in predictions:
        w = p["window"]
        if w not in span or p.get("predictions") is None or p.get("is_shadow"):
            continue
        a, b = span[w]
        preds = p["predictions"]
        wx, wy, wX = xs[a:b + 1], ys[a:b + 1], X[a:b + 1]
        if len(preds) != len(wx):
            continue
        row = {**base, "window": w, "member_id": p["member_id"], "is_test": p.get("is_test"),
               "feasible": feasible.get((w, p["member_id"]), True),
               "raw_ari": _ari(wy, preds), "raw_silhouette": _silhouette(wx, preds)}
        for k in kappas:
            fixed, events = detect_identity_changes(wX, preds, s, k, min_points, memory)
            row[f"n_changes_k{k:g}"] = len(events)
            row[f"ari_k{k:g}"] = _ari(wy, fixed)
            row[f"silhouette_k{k:g}"] = _silhouette(wx, fixed)
        rows.append(row)
    mw = pd.DataFrame(rows)
    replay = _replay(mw, windows, kappas, base)
    return mw, replay


def _replay(mw: pd.DataFrame, windows: list[dict], kappas, base: dict) -> pd.DataFrame:
    """Per window t: the raw and every corrected silhouette's pick among feasible members, read on t+1."""
    if mw.empty:
        return pd.DataFrame()
    # the sequential lagged ARI: under the snapshot rule ``lag_ari_win`` reads the snapshot instead
    logged = {w["window"]: w.get("lag_seq_ari_win", w.get("lag_ari_win")) for w in windows}
    by_w = {w: g for w, g in mw.groupby("window")}
    out = []
    for w in sorted(by_w):
        nxt = by_w.get(w + 1)
        if nxt is None:
            continue
        g = by_w[w][by_w[w]["feasible"]]
        row = {**base, "window": w + 1, "logged_lag_ari": logged.get(w + 1)}
        for name, sil, ari in [("raw", "raw_silhouette", "raw_ari")] + [
                (f"k{k:g}", f"silhouette_k{k:g}", f"ari_k{k:g}") for k in kappas]:
            cand = g.dropna(subset=[sil])
            if cand.empty:
                row[f"pick_{name}"], row[f"lag_ari_{name}"] = None, None
                continue
            pick = cand.sort_values([sil, "member_id"], ascending=[False, True]).iloc[0]["member_id"]
            later = nxt[nxt["member_id"] == pick]
            row[f"pick_{name}"] = int(pick)
            row[f"lag_ari_{name}"] = (None if later.empty or pd.isna(later.iloc[0][ari]) else float(later.iloc[0][ari]))
        out.append(row)
    return pd.DataFrame(out)


def validate(mw_hp: pd.DataFrame, preds_hp: list[dict], preds_st: list[dict], run: dict, windows: list[dict],
             streams: _Streams, kappas=DEFAULT_KAPPAS, min_points=DEFAULT_MIN_POINTS, memory=DEFAULT_MEMORY):
    """Detection against the exact reuse events of HPStream vs HPStreamStableLabels on one fixed configuration."""
    xs, ys, X = streams.get(run)
    span = {w["window"]: (w["start"], w["end"]) for w in windows}
    st = {(p["window"], p["member_id"]): p["predictions"] for p in preds_st if p.get("predictions") is not None}
    s = float(run.get("s_online") or 0.0)
    rows = []
    for p in preds_hp:
        key = (p["window"], p["member_id"])
        if p.get("predictions") is None or key not in st or p["window"] not in span:
            continue
        a, b = span[p["window"]]
        raw, stable = p["predictions"], st[key]
        true = exact_reuse_events(raw, stable)
        row = {"fold": run["fold"], "arm": run["arm"], "window": p["window"], "member_id": p["member_id"],
               "n_true_reuses": len(true),
               "ari_raw_vs_stable": float(adjusted_rand_score([str(v) for v in stable], [str(v) for v in raw]))}
        for k in kappas:
            fixed, events = detect_identity_changes(X[a:b + 1], raw, s, k, min_points, memory)
            m = match_events(events, true)
            row.update({f"{name}_k{k:g}": v for name, v in m.items() if name != "n_true"})
            row[f"ari_corrected_vs_stable_k{k:g}"] = float(adjusted_rand_score([str(v) for v in stable],
                                                                                [str(v) for v in fixed]))
        rows.append(row)
    return pd.DataFrame(rows)


def _group(fold: str) -> str:
    """The fold's stream group, ``other`` for a fold outside the protocol's groups (a smoke stream)."""
    try:
        return C.stream_group(fold)
    except ValueError:
        return "other"


def _summaries(mw: pd.DataFrame, replay: pd.DataFrame, kappas) -> pd.DataFrame:
    rows = []
    for (algo, fold, arm), g in mw.groupby(["algo", "fold", "arm"]):
        r = replay[(replay["algo"] == algo) & (replay["fold"] == fold) & (replay["arm"] == arm)] if len(replay) else replay
        row = {"algo": algo, "fold": fold, "arm": arm, "group": _group(fold), "n_member_windows": len(g),
               "raw_ari_mean": g["raw_ari"].mean(), "raw_silhouette_mean": g["raw_silhouette"].mean()}
        if len(r):
            row["replay_raw_lag_ari"] = pd.to_numeric(r["lag_ari_raw"], errors="coerce").mean()
            row["logged_lag_ari"] = pd.to_numeric(r["logged_lag_ari"], errors="coerce").mean()
        for k in kappas:
            n = g[f"n_changes_k{k:g}"]
            row[f"share_with_change_k{k:g}"] = float((n > 0).mean())
            row[f"ari_change_k{k:g}"] = float((g[f"ari_k{k:g}"] - g["raw_ari"]).mean())
            row[f"silhouette_change_k{k:g}"] = float((g[f"silhouette_k{k:g}"] - g["raw_silhouette"]).mean())
            if len(r):
                row[f"replay_lag_ari_k{k:g}"] = pd.to_numeric(r[f"lag_ari_k{k:g}"], errors="coerce").mean()
                row[f"pick_agrees_k{k:g}"] = float((r[f"pick_k{k:g}"] == r["pick_raw"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def analyse(out_dir: Path, kappas=DEFAULT_KAPPAS, min_points=DEFAULT_MIN_POINTS, memory=DEFAULT_MEMORY):
    out_dir = Path(out_dir)
    runs = [r for r in C.read_jsonl(out_dir / "runs.jsonl") if r.get("status") == "ok" and r.get("prediction_log")]
    windows = defaultdict(list)
    for w in C.read_jsonl(out_dir / "windows.jsonl"):
        windows[_job_key(w)].append(w)
    streams = _Streams()
    mws, replays, logs = [], [], {}
    from .run import job_name

    for run in runs:
        name = job_name(run)
        preds = list(C.read_jsonl(out_dir / PREDICTIONS_DIR / f"{name}.jsonl"))
        members = list(C.read_jsonl(out_dir / C.MEMBERS_DIR.name / f"{name}.jsonl")) \
            if (out_dir / C.MEMBERS_DIR.name / f"{name}.jsonl").exists() else []
        mw, rp = analyse_job(run, windows[_job_key(run)], preds, members, streams, kappas, min_points, memory)
        mws.append(mw)
        replays.append(rp)
        logs[_job_key(run)] = (run, preds)
    mw = pd.concat(mws, ignore_index=True) if mws else pd.DataFrame()
    replay = pd.concat(replays, ignore_index=True) if replays else pd.DataFrame()
    val = []
    for key, (run, preds) in logs.items():
        if run["algo"] != "HPStream" or run["arm"] not in C.FIXED_ARMS:
            continue
        twin = logs.get(("HPStreamStableLabels",) + key[1:])
        if twin is not None:
            val.append(validate(mw, preds, twin[1], run, windows[key], streams, kappas, min_points, memory))
    validation = pd.concat(val, ignore_index=True) if val else pd.DataFrame()
    summary = _summaries(mw, replay, kappas) if len(mw) else pd.DataFrame()
    return {"member_windows": mw, "replay": replay, "summary": summary, "validation": validation}


def write(out_dir: Path, **kw) -> list[Path]:
    res = analyse(out_dir, **kw)
    rep = Path(out_dir) / C.REPORT_DIR.name / "identity"
    rep.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in res.items():
        path = rep / f"{name}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    kappas = kw.get("kappas", DEFAULT_KAPPAS)
    lines = ["# Partition identity within a window (E1)", "",
             "Read from the per-point predictions (`--log-predictions`). A change is detected without labels: a point "
             f"farther than kappa * s from its label's running centroid (kappa in {list(kappas)}, placeholders); the "
             "corrected partition reads the label as a new cluster from there on. Raw and corrected readings are both "
             "reported; neither replaces the other.", ""]
    if res["member_windows"].empty:
        lines.append("No run in this dir wrote predictions: run with `--log-predictions`.")
    else:
        from .analyze import _md_table

        lines += ["## Per algorithm, stream and arm", "", _md_table(res["summary"], digits=4), ""]
    if len(res["validation"]):
        v = res["validation"]
        lines += ["## Validation on HPStream against HPStreamStableLabels (fixed configuration)", ""]
        for k in kappas:
            hits, det, true = (v[f"n_hits_k{k:g}"].sum(), v[f"n_detected_k{k:g}"].sum(), v["n_true_reuses"].sum())
            lines.append(f"- kappa {k:g}: {int(true)} true reuses, {int(det)} detections, {int(hits)} hits "
                         f"(precision {hits / det if det else float('nan'):.3f}, recall {hits / true if true else float('nan'):.3f}); "
                         f"mean ARI raw vs stable {v['ari_raw_vs_stable'].mean():.4f}, corrected vs stable "
                         f"{v[f'ari_corrected_vs_stable_k{k:g}'].mean():.4f}")
    else:
        lines.append("No validation: the dir does not hold HPStream and HPStreamStableLabels on the same fixed "
                     "configuration with predictions logged; the detection is then unvalidated.")
    md = rep / "identity.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths + [md]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--kappa", default=",".join(f"{k:g}" for k in DEFAULT_KAPPAS),
                   help="comma-separated multiples of s at which a change is detected (placeholders)")
    p.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS)
    p.add_argument("--memory", type=int, default=DEFAULT_MEMORY)
    args = p.parse_args(argv)
    kappas = tuple(float(k) for k in args.kappa.split(",") if k.strip())
    for path in write(Path(args.out_dir), kappas=kappas, min_points=args.min_points, memory=args.memory):
        print(path)


if __name__ == "__main__":
    main()
