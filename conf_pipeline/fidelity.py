"""Implementation fidelity: each clusterer on small controlled streams whose events are known.

The OTB study ranks eight stream-clustering implementations. This module asks a different question of
each of them: on a stream where the right answer is known (well-separated Gaussian clusters, the units
known, events at known sample indices), does the implementation behave the way its source algorithm
says it should -- and where it does not, is that its design, River's code, or the benchmark's
conventions? It writes evidence, never a verdict: every check is reported as ``pass``, ``fail``,
``not-applicable`` (the algorithm's design does not promise the behaviour, or the configuration cannot
represent it) or ``undetermined`` (the check's precondition did not hold, or nothing could be read),
per (implementation, scenario, seed, configuration). There is no "valid" column.

What runs, and how
------------------
Every implementation is built exactly as ``run.py`` builds an ensemble member
(``run._make_adapter_factory``: the registry's adapter, the protocol's scaler, the start
``spaces.start_params`` with its pinned values and its initialisation buffers capped at the window
``h``; ``scaled_default`` multiplies the distance parameters by the calibration window's distance unit
``s``, ``spaces.scaled_defaults``). As in the ensemble, it is warm-started on window 0
(``ClustererAdapter.learn_only``), then every later window is run predict-before-learn through
``ClustererAdapter.process_one``, so the adapter's conventions (fallback label 0 and its count,
DenStream's sentinel -1 kept as a label and counted, RIVER_BUGS.md BUG-6) are the benchmark's.
After each window, the window is predicted again by a deep copy of the adapter (the end-of-window
snapshot): predictions can change a River model (BUG-7), so the live model is never asked twice.

Scenarios (``SCENARIOS``; ``make_scenario``), each ``n_windows`` windows of ``h`` points, low dimension,
isotropic Gaussian clusters of standard deviation ``sigma`` whose centres differ by at least
``separation * sigma`` in every coordinate (a Latin-hypercube layout, drawn per seed), points drawn
i.i.d. from the active clusters so every window holds several classes:

- ``stationary``: three clusters throughout.
- ``birth``: two clusters; a third starts emitting at the first point of window ``event_window``.
- ``deletion``: three clusters; one stops emitting at the first point of window ``event_window``.
- ``merge``: two clusters move linearly to their midpoint over the transition windows and coincide
  after it (one class, a new label, from then on); a third stays put.
- ``split``: one cluster becomes two children (new labels) that move apart over the transition
  windows; two others stay put.
- ``identity_stress``: three clusters at a time; every ``stress_period`` points the oldest stops and a
  new one (a new label, never reused) starts at the slot freed longest ago, so clusters are created
  and retired inside windows.

Truth is the generating component, except that two components that coincide after a merge are one
class. Windows inside a merge or split transition are marked ``transition`` and no check reads them.

Label identity
--------------
Within a window, five implementations can hand one label to another cluster
(``docs/SEQUENTIAL_PARTITION_AUDIT.md``, RIVER_BUGS.md GAP-6). The model-agnostic proxy recorded for
every implementation (``label_class_changes``): cut the window into ``segments`` consecutive parts;
in each, a label's majority true class counts when the label has at least ``min_support`` points
there and the majority holds at least ``min_share`` of them; a label is flagged when two parts give
it different majority classes. It reads truth, not the model, so it also flags a cluster that really
moved onto another class. It is validated where reuse is known exactly: ``HPStreamStableLabels`` is
HPStream with labels never reused (same clusters, same predicted cluster at every point), so within a
window an HPStream label covering two of the variant's labels names two cluster objects
(``exact_reuse``). The same proxy on the snapshot labels, which come from one model state and cannot
reuse anything, is a per-implementation false-positive control.

Cross-library agreement
-----------------------
Only scikit-learn is installed in the repo's ``.venv`` among independent stream-clustering code
(``independent_implementations``): its ``Birch`` is run against the River-fork BIRCH on the same
stream, in the same order, with the same threshold and branching factor (``birch_crosslib``), with
River's tree set as close to scikit-learn's as its parameters allow (radius threshold, centroid
distance, leaf capacity = branching factor, no memory budget), with the protocol's pins, and with each
library's global phase on. Agreement is expected behaviour, not numerical identity.

Every threshold and size here is a runnable placeholder (``THRESHOLDS``, ``PROXY``, ``StreamShape``),
not a recommendation: override any of them on the command line (``--threshold key=value``).

    PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.fidelity --out-dir conf_pipeline/out/fidelity_pilot \\
        --seeds 1,2 --config default,scaled_default --workers 6
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _common as C  # first: it puts the river tree on sys.path
from .spaces import scaled_defaults, start_params

# --- what is run -------------------------------------------------------------------------------
#: The eight implementations the paper ranks, then the two named variants read against their reference:
#: HPStream with labels never reused (``_common.LABEL_VARIANTS``) and DenStream with BUG-7/BUG-8 fixed
#: (``_common.IMPL_VARIANTS``).
ALGOS: tuple[str, ...] = (
    "BIRCH", "CluStream", "DBSTREAM", "DenStream", "HPStream", "RepStream", "SOStream", "STREAMKMeans",
    "HPStreamStableLabels", "DenStreamPatchedFast",
)
#: The implementations whose code was written for this study (in the River fork), the others are
#: River upstream (``benchmark.algorithms.registry`` provenance).
STUDY_WRITTEN = ("BIRCH", "HPStream", "RepStream", "SOStream")
CONFIGS: tuple[str, ...] = ("default", "scaled_default")
SCENARIOS: tuple[str, ...] = ("stationary", "birth", "deletion", "merge", "split", "identity_stress")
STATUSES = ("pass", "fail", "not-applicable", "undetermined")

#: Check thresholds and windows: runnable placeholders, not recommendations (every one a CLI override).
#: Delays are counted in windows from the event's window (0: by the end of that window).
THRESHOLDS: dict[str, float] = {
    "stationary_seq_ari_min": 0.8,
    "stationary_snap_ari_min": 0.8,
    "warmup_skip_windows": 1,          # scored windows left out after the warm-start window
    "birth_max_delay_windows": 1,
    "deletion_max_delay_windows": 2,
    "merge_max_delay_windows": 1,      # counted from the first window after the transition
    "split_max_delay_windows": 1,      # likewise
    "stress_max_delay_windows": 1,
    "stress_detect_share_min": 0.8,
    "cover_share": 0.9,                # labels needed to cover this share of a class's points
    "min_class_points": 5,             # a class needs this many points in a window to have a plurality label
}
#: The label-identity proxy (``label_class_changes``); placeholders.
PROXY: dict[str, float] = {"segments": 4, "min_support": 5, "min_share": 0.6}

#: Parameters that bound how many clusters a configuration can output; a check needing more classes
#: than this is not applicable to that configuration.
CAPACITY_PARAM = {"CluStream": "n_macro_clusters", "STREAMKMeans": "n_clusters",
                  "HPStream": "max_clusters", "HPStreamStableLabels": "max_clusters"}

#: Checks the design of an implementation does not promise, with the reason (a reading of the code,
#: references in ``AUDIT``). Everything else applies, subject to capacity (``CHECK_CAPACITY``).
NOT_PROMISED: dict[tuple[str, str], str] = {}
for _a in ("BIRCH",):
    NOT_PROMISED[(_a, "deletion_absorbed")] = (
        "no forgetting: leaf entries are never decayed or deleted, only re-inserted by a rebuild (birch.py:324-341)")
    NOT_PROMISED[(_a, "merge_label_count_down")] = (
        "leaf entries are not merged online (only node refinement and rebuilds under the memory budget); "
        "Phase 3 is off in the protocol")
for _a in ("STREAMKMeans",):
    NOT_PROMISED[(_a, "deletion_absorbed")] = (
        "no forgetting: the global KMeans moves only the centre nearest to a chunk centre "
        "(streamkmeans.py:104-105, k_means.py:115)")
for _a in ("CluStream", "STREAMKMeans"):
    NOT_PROMISED[(_a, "merge_label_count_down")] = (
        "the number of output clusters is a parameter (n_macro_clusters / n_clusters), not a reading of the data")
for _a in ("HPStream", "HPStreamStableLabels"):
    NOT_PROMISED[(_a, "deletion_absorbed")] = (
        "clusters fade but are deleted only by LRU eviction when a new cluster needs a slot, or by starvation "
        "(hpstream.py:618-627)")
    NOT_PROMISED[(_a, "merge_label_count_down")] = "no online merge of clusters (hpstream.py:592-629)"
NOT_PROMISED[("RepStream", "deletion_absorbed")] = (
    "representatives are kept in a repository by usefulness on purpose (repstream.py:287-334); retirement is "
    "FIFO over the vertex budget (repstream.py:739-754)")
#: Classes a check needs the configuration to be able to output at once.
CHECK_CAPACITY = {"stationary_seq_ari": 3, "stationary_snap_ari": 3, "birth_new_label": 3,
                  "split_label_count_up": 4, "split_children_distinct": 4, "stress_births_told_apart": 3}
SCENARIO_CHECKS: dict[str, tuple[str, ...]] = {
    "stationary": ("stationary_seq_ari", "stationary_snap_ari"),
    "birth": ("birth_new_label",),
    "deletion": ("deletion_absorbed",),
    "merge": ("merge_label_count_down",),
    "split": ("split_label_count_up", "split_children_distinct"),
    "identity_stress": ("stress_births_told_apart",),
}


# --- scenarios -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamShape:
    """Size and geometry of every scenario: runnable placeholders, not recommendations."""

    h: int = 500
    n_windows: int = 10                 # window 0 is the warm start
    d: int = 2
    sigma: float = 1.0
    separation: float = 8.0             # in sigma, per coordinate
    stress_period: int | None = None    # None: h // 2
    stress_active: int = 3
    probe_points: int = 100             # deletion: points drawn from the deleted cluster, predicted per window

    @property
    def n(self) -> int:
        return self.h * self.n_windows

    @property
    def period(self) -> int:
        return self.stress_period if self.stress_period is not None else max(1, self.h // 2)

    def validate(self) -> None:
        if self.n_windows < 5:
            raise ValueError(f"n_windows must be at least 5 (warm start, pre, event, post), got {self.n_windows}")
        if self.h < 20 or self.d < 1 or self.sigma <= 0 or self.separation <= 0:
            raise ValueError(f"bad stream shape {self}")
        if self.stress_active < 2:
            raise ValueError("stress_active must be at least 2")

    def event_window(self) -> int:
        """Window whose first point is a birth or a deletion."""
        return min(self.n_windows - 2, max(2, round(0.4 * self.n_windows)))

    def transition(self) -> tuple[int, int]:
        """``(first, end)`` windows of a merge or split: it moves over ``[first, end)``, done at ``end``."""
        first = max(2, round(0.3 * self.n_windows))
        return first, min(self.n_windows - 1, max(first + 1, round(0.6 * self.n_windows)))


@dataclass
class Component:
    """One generating Gaussian: ``label`` emits over ``[start, end)``, its centre following ``path``
    (``((t0, c0), (t1, c1))``: ``c0`` until ``t0``, linear to ``c1`` at ``t1``, ``c1`` after)."""

    label: int
    start: int
    end: int
    path: tuple
    weight: float = 1.0

    def centre(self, t: int) -> np.ndarray:
        (t0, c0), (t1, c1) = self.path
        if t1 <= t0:
            return c1
        if t <= t0:
            return c0
        if t >= t1:
            return c1
        return c0 + (c1 - c0) * ((t - t0) / (t1 - t0))


@dataclass
class Scenario:
    name: str
    seed: int
    shape: StreamShape
    X: np.ndarray
    y: np.ndarray
    events: list[dict]
    meta: dict
    phases: list[str]
    probe: np.ndarray | None = None

    def points(self) -> list[dict]:
        keys = [f"x{j}" for j in range(self.X.shape[1])]
        return [dict(zip(keys, map(float, row))) for row in self.X]

    def probe_points(self) -> list[dict]:
        if self.probe is None:
            return []
        keys = [f"x{j}" for j in range(self.probe.shape[1])]
        return [dict(zip(keys, map(float, row))) for row in self.probe]


def _static(c: np.ndarray) -> tuple:
    return ((0, c), (0, c))


def _latin(n_slots: int, d: int, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Slot centres whose coordinates are, per dimension, a permutation of ``0..n_slots-1`` times ``spacing``:
    any two slots differ by at least ``spacing`` in every coordinate, so in every projection too."""
    return np.stack([rng.permutation(n_slots) for _ in range(d)], axis=1).astype(float) * spacing


def _draw(shape: StreamShape, comps: list[Component], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n, d = shape.n, shape.d
    X = np.empty((n, d))
    y = np.empty(n, dtype=int)
    noise = rng.standard_normal((n, d)) * shape.sigma
    u = rng.random(n)
    for t in range(n):
        active = [c for c in comps if c.start <= t < c.end]
        if not active:
            raise ValueError(f"no component emits at t={t}")
        w = np.cumsum([c.weight for c in active])
        c = active[int(np.searchsorted(w, u[t] * w[-1], side="right"))]
        X[t] = c.centre(t) + noise[t]
        y[t] = c.label
    return X, y


PHASE_ABBR = {"warmup": "W", "stationary": "S", "stress": "S", "pre": "-", "transition": "T", "post": "+"}


def _window_phases(shape: StreamShape, kind: str) -> list[str]:
    out = ["warmup"]
    for w in range(1, shape.n_windows):
        if kind in ("stationary", "identity_stress"):
            out.append(kind if kind == "stationary" else "stress")
        elif kind in ("birth", "deletion"):
            out.append("pre" if w < shape.event_window() else "post")
        else:
            first, end = shape.transition()
            out.append("pre" if w < first else "transition" if w < end else "post")
    return out


def make_scenario(name: str, seed: int, shape: StreamShape | None = None) -> Scenario:
    """The scenario ``name`` drawn from ``seed``: deterministic per (name, seed, shape)."""
    shape = shape or StreamShape()
    shape.validate()
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {SCENARIOS}")
    rng = np.random.default_rng([seed, SCENARIOS.index(name)])
    h, n, d, sep = shape.h, shape.n, shape.d, shape.separation * shape.sigma
    events: list[dict] = []
    meta: dict = {}
    probe = None
    if name in ("stationary", "birth", "deletion"):
        L = _latin(3, d, sep, rng)
        te = shape.event_window() * h
        comps = [Component(0, 0, n, _static(L[0])), Component(1, 0, n, _static(L[1]))]
        if name == "stationary":
            comps.append(Component(2, 0, n, _static(L[2])))
            meta = {"classes": [0, 1, 2]}
        elif name == "birth":
            comps.append(Component(2, te, n, _static(L[2])))
            events.append({"event": "birth", "t": te, "window": te // h, "labels": [2]})
            meta = {"new": 2, "old": [0, 1], "window": te // h}
        else:
            comps.append(Component(2, 0, te, _static(L[2])))
            events.append({"event": "deletion", "t": te, "window": te // h, "labels": [2]})
            meta = {"deleted": 2, "remaining": [0, 1], "window": te // h}
            probe = L[2] + np.random.default_rng([seed, 99]).standard_normal((shape.probe_points, d)) * shape.sigma
    elif name == "merge":
        # Three positions per coordinate; the static cluster takes an end one in every coordinate, so the
        # midpoint of the other two stays at least 1.5 separations away from it.
        L = np.empty((3, d))
        for j in range(d):
            c_end = 2 * int(rng.integers(2))
            ab = [v for v in (0, 1, 2) if v != c_end]
            rng.shuffle(ab)
            L[0, j], L[1, j], L[2, j] = ab[0], ab[1], c_end
        L *= sep
        first, end = shape.transition()
        ts, tm = first * h, end * h
        mid = (L[0] + L[1]) / 2
        comps = [Component(0, 0, tm, ((ts, L[0]), (tm, mid))), Component(1, 0, tm, ((ts, L[1]), (tm, mid))),
                 Component(3, tm, n, _static(mid), weight=2.0), Component(2, 0, n, _static(L[2]))]
        events += [{"event": "merge_start", "t": ts, "window": first, "labels": [0, 1]},
                   {"event": "merge_done", "t": tm, "window": end, "labels": [3]}]
        meta = {"pair": [0, 1], "merged": 3, "other": [2], "first": first, "end": end}
    elif name == "split":
        L = _latin(3, d, 2 * sep, rng)
        first, end = shape.transition()
        ts, tm = first * h, end * h
        v = rng.choice([-1.0, 1.0], size=d) * (sep / 2)
        comps = [Component(0, 0, ts, _static(L[0])),
                 Component(3, ts, n, ((ts, L[0]), (tm, L[0] + v))), Component(4, ts, n, ((ts, L[0]), (tm, L[0] - v))),
                 Component(1, 0, n, _static(L[1])), Component(2, 0, n, _static(L[2]))]
        events += [{"event": "split_start", "t": ts, "window": first, "labels": [3, 4]},
                   {"event": "split_done", "t": tm, "window": end, "labels": [3, 4]}]
        meta = {"parent": 0, "children": [3, 4], "other": [1, 2], "first": first, "end": end}
    else:  # identity_stress
        k = shape.stress_active
        L = _latin(2 * k, d, sep, rng)
        comps = [Component(i, 0, n, _static(L[i])) for i in range(k)]
        alive = list(range(k))           # indices into comps, oldest first
        slot_of = {i: i for i in range(k)}
        free = list(range(k, 2 * k))     # slots, freed longest ago first
        births, label = [], k
        t = h
        while t < n:
            dead = alive.pop(0)
            comps[dead].end = t
            free.append(slot_of[dead])
            slot = free.pop(0)
            comps.append(Component(label, t, n, _static(L[slot])))
            slot_of[len(comps) - 1] = slot
            alive.append(len(comps) - 1)
            events += [{"event": "death", "t": t, "window": t // h, "labels": [comps[dead].label]},
                       {"event": "birth", "t": t, "window": t // h, "labels": [label]}]
            births.append({"t": t, "label": label, "window": t // h})
            label += 1
            t += shape.period
        meta = {"births": births, "active": k}
    X, y = _draw(shape, comps, rng)
    return Scenario(name, seed, shape, X, y, events, meta, _window_phases(shape, name), probe)


# --- readings of one window ------------------------------------------------------------------------
def _ari(truth, pred) -> float | None:
    """ARI over the points both label; None when fewer than ``MIN_TRUE_CLASSES`` true classes."""
    from sklearn.metrics import adjusted_rand_score

    both = [(t, p) for t, p in zip(truth, pred) if t is not None and p is not None]
    if len({t for t, _ in both}) < C.MIN_TRUE_CLASSES:
        return None
    ids: dict = {}
    return float(adjusted_rand_score([t for t, _ in both], [ids.setdefault(p, len(ids)) for _, p in both]))


def _majority(counter: Counter, min_support: float, min_share: float):
    total = sum(counter.values())
    if total < min_support or total == 0:
        return None
    top = max(counter.items(), key=lambda kv: (kv[1], repr(kv[0])))
    return top[0] if top[1] / total >= min_share else None


def label_class_changes(pred, truth, segments: int = int(PROXY["segments"]), min_support: float = PROXY["min_support"],
                        min_share: float = PROXY["min_share"]) -> dict:
    """The label-identity proxy on one window: the labels whose majority true class differs between two
    of ``segments`` consecutive parts of the window (see the module docstring), the points they carry,
    and the labels the proxy could read at all (a majority in at least two parts)."""
    n = len(pred)
    bounds = np.linspace(0, n, int(segments) + 1).astype(int)
    majorities: dict = defaultdict(list)
    for s in range(int(segments)):
        per: dict = defaultdict(Counter)
        for i in range(bounds[s], bounds[s + 1]):
            if pred[i] is not None and truth[i] is not None:
                per[pred[i]][truth[i]] += 1
        for lab, cnt in per.items():
            m = _majority(cnt, min_support, min_share)
            if m is not None:
                majorities[lab].append(m)
    flagged = {lab for lab, ms in majorities.items() if len(set(ms)) > 1}
    readable = {lab for lab, ms in majorities.items() if len(ms) >= 2}
    return {"flagged": flagged, "n_flagged": len(flagged), "readable": readable, "n_readable": len(readable),
            "points": sum(1 for p in pred if p in flagged)}


def exact_reuse(river_pred, stable_pred, truth, min_support: float = PROXY["min_support"],
                min_share: float = PROXY["min_share"]) -> dict:
    """HPStream's labels within one window read against HPStreamStableLabels' (same clusters, labels
    never reused): an HPStream label covering two of the variant's labels named two cluster objects.

    ``reused``: such labels. ``cross_any``: among them, those whose objects have different plurality
    true classes (one point is enough). ``cross_supported``: the same with each object's majority
    read as the proxy reads one (``min_support`` points, ``min_share``). ``function``: whether every
    variant label maps to one HPStream label, which holds when the two models hold the same clusters."""
    groups: dict = defaultdict(set)
    back: dict = defaultdict(set)
    per_obj: dict = defaultdict(Counter)
    for a, b, t in zip(river_pred, stable_pred, truth):
        groups[a].add(b)
        back[b].add(a)
        if t is not None:
            per_obj[b][t] += 1
    reused = {a for a, bs in groups.items() if len(bs) >= 2}
    any_major = {b: _majority(c, 1, 0.0) for b, c in per_obj.items()}
    sup_major = {b: _majority(c, min_support, min_share) for b, c in per_obj.items()}
    cross_any = {a for a in reused if len({any_major.get(b) for b in groups[a]} - {None}) >= 2}
    cross_sup = {a for a in reused if len({sup_major.get(b) for b in groups[a]} - {None}) >= 2}
    return {"labels": set(groups), "reused": reused, "cross_any": cross_any, "cross_supported": cross_sup,
            "function": all(len(v) == 1 for v in back.values())}


def _plurality(labels: list, min_points: int):
    if len(labels) < min_points:
        return None
    cnt = Counter(lab for lab in labels if lab is not None)
    if not cnt:
        return None
    return max(cnt.items(), key=lambda kv: (kv[1], repr(kv[0])))[0]


def _cover(labels: list, share: float) -> int | None:
    labels = [lab for lab in labels if lab is not None]
    if not labels:
        return None
    need, got = share * len(labels), 0
    for i, (_, c) in enumerate(Counter(labels).most_common(), start=1):
        got += c
        if got >= need:
            return i
    return len(set(labels))


def _of(pred, truth, classes) -> list:
    cs = set(classes)
    return [p for p, t in zip(pred, truth) if t in cs]


# --- one cell: one implementation on one scenario --------------------------------------------------
def capacity(algo: str, params: dict) -> int | None:
    name = CAPACITY_PARAM.get(algo)
    return int(params[name]) if name and params.get(name) is not None else None


def cell_params(algo: str, config: str, h: int, d: int, xs_window0: list[dict]) -> tuple[dict, float, dict | None]:
    """The parameters ``run.py`` gives this implementation's start under ``config``, the calibration
    distance unit ``s`` read off window 0 (as ``partition_audit`` reads it), and what scaling changed."""
    params = start_params(algo, h, d=d)
    s = C.estimate_scale(xs_window0, np.random.default_rng(C.CALIBRATION_SEED))
    record = None
    if config == "scaled_default":
        params, record = scaled_defaults(algo, params, s)
    elif config != "default":
        raise ValueError(f"config must be one of {CONFIGS}, got {config!r}")
    return params, s, record


def _predict_copy(probe, x: dict):
    x = dict(x)
    return probe.predict_one(probe.scaler.transform_one(x) if probe.scaler is not None else x)


def scaled_scenario(xs: list[dict], ys: list, probe: list[dict], h: int, scaling: str | None,
                    window: int | None) -> tuple[list[dict], dict | None]:
    """A scenario's points under the benchmark's stream scaling (``benchmark.datasets.scaling``), and the
    probe set as the scaler stands at the end of each window ``w`` (``{w: points}``; None unscaled)."""
    from benchmark.datasets.scaling import scale_stream, transform_side

    if scaling in (None, "none"):
        return xs, None
    stream = scale_stream(list(zip(xs, ys)), scaling, C.scaling_window(window, h))
    out, probes = [], {}
    for i, (x, _) in enumerate(stream):
        out.append(x)
        if probe and (i + 1) % h == 0:
            probes[(i + 1) // h - 1] = [transform_side(stream, p) for p in probe]
    return out, probes


def run_cell(task: dict) -> dict:
    """Run one (scenario, seed, config, algo) cell; see the module docstring. Never raises: an error is
    recorded on the cell and on the windows it stopped."""
    from .run import _make_adapter_factory

    t0 = time.perf_counter()
    shape = StreamShape(**task["shape"])
    th = {**THRESHOLDS, **task.get("thresholds", {})}
    px = {**PROXY, **task.get("proxy", {})}
    scen = make_scenario(task["scenario"], task["seed"], shape)
    algo, config = task["algo"], task["config"]
    h, W = shape.h, shape.n_windows
    xs, ys = scen.points(), scen.y.tolist()
    probe_xs = scen.probe_points()
    xs, probe_by_window = scaled_scenario(xs, ys, probe_xs, h,
                                          C.stream_scaling_of(scen.name, task.get("stream_scaling", "none")),
                                          task.get("scaling_window"))
    base = {"scenario": scen.name, "seed": scen.seed, "config": config, "algo": algo}
    out = {"key": (scen.name, scen.seed, config, algo), "windows": [], "checks": [], "seq": {}, "error": None,
           "cell": dict(base)}
    snaps: dict[int, dict] = {}
    params, s, record = cell_params(algo, config, h, shape.d, xs[:h])
    out["cell"].update(params=params, s=s, scaled=record, capacity=capacity(algo, params))
    keep_seq = algo in ("HPStream", "HPStreamStableLabels")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            adapter = _make_adapter_factory(algo, h)(params, task["model_seed"])
            for x in xs[:h]:
                adapter.learn_only(dict(x))
            for w in range(1, W):
                lo, hi = w * h, (w + 1) * h
                wy = ys[lo:hi]
                s0, f0 = adapter.n_sentinel_predictions, adapter.n_fallback_predictions
                tw = time.perf_counter()
                seq = [adapter.process_one(dict(x)) for x in xs[lo:hi]]
                seq_s = time.perf_counter() - tw
                snap_adapter = copy.deepcopy(adapter)
                ss0, sf0 = snap_adapter.n_sentinel_predictions, snap_adapter.n_fallback_predictions
                snap = [_predict_copy(snap_adapter, x) for x in xs[lo:hi]]
                side = probe_by_window.get(w, probe_xs) if probe_by_window is not None else probe_xs
                probe = ([_predict_copy(snap_adapter, p) for p in side]
                         if probe_xs and w >= scen.meta.get("window", W) - 1 else None)
                try:
                    n_model = snap_adapter.n_macro_clusters()
                    n_micro = snap_adapter.n_micro_clusters()
                except Exception:  # a count the model cannot give is not a stop
                    n_model = n_micro = None
                pseq = label_class_changes(seq, wy, px["segments"], px["min_support"], px["min_share"])
                psnap = label_class_changes(snap, wy, px["segments"], px["min_support"], px["min_share"])
                snaps[w] = {"snap": snap, "truth": wy, "probe": probe}
                if keep_seq:
                    out["seq"][w] = seq
                out["windows"].append({
                    **base, "window": w, "phase": scen.phases[w], "start": lo, "end": hi - 1,
                    "n_points": len(seq), "n_true_classes": len(set(wy)),
                    "ari_seq": _ari(wy, seq), "ari_snap": _ari(wy, snap),
                    "n_labels_seq": len({p for p in seq if p is not None}),
                    "n_labels_snap": len({p for p in snap if p is not None}),
                    "n_none_seq": sum(p is None for p in seq),
                    "n_sentinel_seq": adapter.n_sentinel_predictions - s0,
                    "n_fallback_seq": adapter.n_fallback_predictions - f0,
                    "n_sentinel_snap": snap_adapter.n_sentinel_predictions - ss0,
                    "n_fallback_snap": snap_adapter.n_fallback_predictions - sf0,
                    "n_model_clusters": n_model, "n_micro_clusters": n_micro,
                    "proxy_flags_seq": pseq["n_flagged"], "proxy_points_seq": pseq["points"],
                    "proxy_readable_seq": pseq["n_readable"],
                    "proxy_flags_snap": psnap["n_flagged"], "proxy_points_snap": psnap["points"],
                    "seconds_seq": round(seq_s, 4),
                })
        except Exception as exc:  # an implementation that cannot run a scenario is a finding
            out["error"] = f"{type(exc).__name__}: {exc}"
    out["cell"].update(n_warnings=len(caught), error=out["error"], seconds=round(time.perf_counter() - t0, 3),
                       warning_kinds=sorted({type(w.message).__name__ for w in caught}))
    out["checks"] = evaluate_checks(scen, snaps, out["windows"], algo, config, params, th, out["error"])
    return out


# --- expected-behaviour checks ------------------------------------------------------------------------
def _check(base: dict, check: str, status: str, value=None, threshold=None, reason: str = "", **extra) -> dict:
    return {**base, "check": check, "status": status, "value": value, "threshold": threshold, "reason": reason,
            **extra}


def _applicability(algo: str, check: str, cap: int | None) -> str | None:
    """None when the check applies, else the not-applicable reason."""
    if (algo, check) in NOT_PROMISED:
        return NOT_PROMISED[(algo, check)]
    need = CHECK_CAPACITY.get(check)
    if need is not None and cap is not None and cap < need:
        return f"capacity: {CAPACITY_PARAM[algo]}={cap} < {need} classes the check needs"
    return None


def evaluate_checks(scen: Scenario, snaps: dict, windows: list[dict], algo: str, config: str, params: dict,
                    th: dict, error: str | None = None) -> list[dict]:
    base = {"scenario": scen.name, "seed": scen.seed, "algo": algo, "config": config}
    cap = capacity(algo, params)
    mp = int(th["min_class_points"])
    W = scen.shape.n_windows
    out = []
    for check in SCENARIO_CHECKS[scen.name]:
        na = _applicability(algo, check, cap)
        if na is not None:
            out.append(_check(base, check, "not-applicable", reason=na))
            continue
        if error is not None and not snaps:
            out.append(_check(base, check, "undetermined", reason=f"the cell raised: {error}"))
            continue
        out.append(_evaluate_one(scen, check, snaps, windows, th, mp, W, base))
    return out


def _first_delay(windows_range, ok) -> int | None:
    for w in windows_range:
        if ok(w):
            return w
    return None


def _evaluate_one(scen, check, snaps, windows, th, mp, W, base) -> dict:
    meta = scen.meta
    if check in ("stationary_seq_ari", "stationary_snap_ari"):
        key = "ari_seq" if check == "stationary_seq_ari" else "ari_snap"
        thr = th["stationary_seq_ari_min" if key == "ari_seq" else "stationary_snap_ari_min"]
        vals = [r[key] for r in windows if r["window"] >= 1 + int(th["warmup_skip_windows"]) and r[key] is not None]
        if not vals:
            return _check(base, check, "undetermined", threshold=thr, reason="no scored window with ARI defined")
        m = float(np.mean(vals))
        return _check(base, check, "pass" if m >= thr else "fail", round(m, 4), thr,
                      f"mean {key} over {len(vals)} windows")

    def plur(w, classes):
        sn = snaps.get(w)
        return None if sn is None else _plurality(_of(sn["snap"], sn["truth"], classes), mp)

    def plurals_present(w, exclude=()):
        sn = snaps.get(w)
        if sn is None:
            return set()
        return {p for c in set(sn["truth"]) - set(exclude) if (p := plur(w, [c])) is not None}

    if check == "birth_new_label":
        w0, new = meta["window"], meta["new"]
        hit = _first_delay(range(w0, W), lambda w: (p := plur(w, [new])) is not None
                           and p not in plurals_present(w, exclude=[new]))
        thr = th["birth_max_delay_windows"]
        if hit is None:
            return _check(base, check, "fail", None, thr, "the new class never got a plurality label of its own")
        d = hit - w0
        return _check(base, check, "pass" if d <= thr else "fail", d, thr, f"told apart {d} window(s) after the event")

    if check == "deletion_absorbed":
        w0, gone, rest = meta["window"], meta["deleted"], meta["remaining"]
        thr = th["deletion_max_delay_windows"]
        def carried(w, label) -> bool:
            # at least min_class_points points of the remaining classes carry ``label`` in snapshot w
            sn = snaps.get(w)
            return sn is not None and sum(lab == label for lab in _of(sn["snap"], sn["truth"], rest)) >= mp

        pre = plur(w0 - 1, [gone])
        if pre is None or carried(w0 - 1, pre):
            return _check(base, check, "undetermined", threshold=thr,
                          reason="before the deletion the class had no plurality label of its own")

        def absorbed(w):
            sn = snaps.get(w)
            if sn is None or not sn["probe"]:
                return False
            p = _plurality(sn["probe"], mp)
            return p is not None and carried(w, p)

        hit = _first_delay(range(w0, W), absorbed)
        if hit is None:
            return _check(base, check, "fail", None, thr,
                          "probes of the deleted cluster kept a label of their own to the end of the stream")
        d = hit - w0
        return _check(base, check, "pass" if d <= thr else "fail", d, thr,
                      f"probes absorbed by a remaining cluster's label {d} window(s) after the event")

    if check == "merge_label_count_down":
        first, end, pair, merged = meta["first"], meta["end"], meta["pair"], meta["merged"]
        thr = th["merge_max_delay_windows"]
        pa, pb = plur(first - 1, [pair[0]]), plur(first - 1, [pair[1]])
        sn = snaps.get(first - 1)
        if sn is None or pa is None or pb is None or pa == pb:
            return _check(base, check, "undetermined", threshold=thr,
                          reason="before the merge the two clusters were not under different plurality labels")
        pre = _cover(_of(sn["snap"], sn["truth"], pair), th["cover_share"])
        posts = {w: _cover(_of(snaps[w]["snap"], snaps[w]["truth"], [merged]), th["cover_share"])
                 for w in range(end, W) if w in snaps}
        hit = _first_delay(range(end, W), lambda w: posts.get(w) is not None and posts[w] < pre)
        extra = {"pre_cover": pre, "post_cover": posts.get(end)}
        if hit is None:
            return _check(base, check, "fail", None, thr,
                          f"labels covering the pair: {pre} before, never fewer after", **extra)
        d = hit - end
        return _check(base, check, "pass" if d <= thr else "fail", d, thr,
                      f"labels covering the pair: {pre} before, {posts[hit]} after", **extra)

    if check in ("split_label_count_up", "split_children_distinct"):
        first, end, parent, kids = meta["first"], meta["end"], meta["parent"], meta["children"]
        thr = th["split_max_delay_windows"]
        if check == "split_label_count_up":
            sn = snaps.get(first - 1)
            pre = None if sn is None else _cover(_of(sn["snap"], sn["truth"], [parent]), th["cover_share"])
            if pre is None:
                return _check(base, check, "undetermined", threshold=thr, reason="no parent points before the split")
            posts = {w: _cover(_of(snaps[w]["snap"], snaps[w]["truth"], kids), th["cover_share"])
                     for w in range(end, W) if w in snaps}
            hit = _first_delay(range(end, W), lambda w: posts.get(w) is not None and posts[w] > pre)
            extra = {"pre_cover": pre, "post_cover": posts.get(end)}
            if hit is None:
                return _check(base, check, "fail", None, thr,
                              f"labels covering the parent: {pre}; the children never needed more", **extra)
            d = hit - end
            return _check(base, check, "pass" if d <= thr else "fail", d, thr,
                          f"labels covering the parent {pre}, the children {posts[hit]}", **extra)
        hit = _first_delay(range(end, W), lambda w: (a := plur(w, [kids[0]])) is not None
                           and (b := plur(w, [kids[1]])) is not None and a != b)
        if hit is None:
            return _check(base, check, "fail", None, thr, "the children never got different plurality labels")
        d = hit - end
        return _check(base, check, "pass" if d <= thr else "fail", d, thr, f"children told apart after {d} window(s)")

    if check == "stress_births_told_apart":
        thr = th["stress_max_delay_windows"]
        delays = []
        for b in meta["births"]:
            c, w0 = b["label"], b["window"]
            seen = [w for w in range(w0, W) if w in snaps and sum(t == c for t in snaps[w]["truth"]) >= mp]
            if not seen:
                continue
            hit = _first_delay(seen, lambda w: (p := plur(w, [c])) is not None and p not in plurals_present(w, [c]))
            delays.append(None if hit is None else hit - w0)
        if not delays:
            return _check(base, check, "undetermined", threshold=th["stress_detect_share_min"],
                          reason="no birth with enough points to read")
        share = sum(d is not None and d <= thr for d in delays) / len(delays)
        ok = share >= th["stress_detect_share_min"]
        return _check(base, check, "pass" if ok else "fail", round(share, 4), th["stress_detect_share_min"],
                      f"{sum(d is not None and d <= thr for d in delays)}/{len(delays)} births told apart within "
                      f"{int(thr)} window(s)", n_births=len(delays))
    raise KeyError(check)


# --- identity proxy validation on HPStream vs HPStreamStableLabels -----------------------------------
def identity_validation(cells: dict, shape: StreamShape, px: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Per (scenario, seed, config): HPStream's exact label reuse, read against HPStreamStableLabels,
    against the proxy on HPStream's sequential labels; and the per-window detail. ``cells`` maps
    ``(scenario, seed, config, algo)`` to ``run_cell``'s result; a pair with a missing or failed cell
    is left out."""
    px = {**PROXY, **(px or {})}
    summary, detail = [], []
    keys = {(k[0], k[1], k[2]) for k in cells}
    for sc, seed, cfg in sorted(keys):
        river = cells.get((sc, seed, cfg, "HPStream"))
        stable = cells.get((sc, seed, cfg, "HPStreamStableLabels"))
        if not river or not stable or river["error"] or stable["error"]:
            continue
        scen = make_scenario(sc, seed, shape)
        h = scen.shape.h
        tot = Counter()
        function_ok = True
        for w in sorted(set(river["seq"]) & set(stable["seq"])):
            truth = scen.y[w * h:(w + 1) * h].tolist()
            a, b = river["seq"][w], stable["seq"][w]
            ex = exact_reuse(a, b, truth, px["min_support"], px["min_share"])
            pr = label_class_changes(a, truth, px["segments"], px["min_support"], px["min_share"])
            prs = label_class_changes(b, truth, px["segments"], px["min_support"], px["min_share"])
            function_ok &= ex["function"]
            ex["cross_readable"] = ex["cross_any"] & pr["readable"]
            row = {"scenario": sc, "seed": seed, "config": cfg, "window": w, "n_labels": len(ex["labels"]),
                   "n_reused": len(ex["reused"]), "n_cross_any": len(ex["cross_any"]),
                   "n_cross_supported": len(ex["cross_supported"]), "n_cross_readable": len(ex["cross_readable"]),
                   "n_proxy": pr["n_flagged"], "n_proxy_readable": pr["n_readable"],
                   "n_proxy_stable": prs["n_flagged"], "n_proxy_stable_readable": prs["n_readable"],
                   "n_stable_labels": len(set(b)), "function": ex["function"]}
            for ref in ("reused", "cross_any", "cross_supported", "cross_readable"):
                pos = ex[ref]
                row[f"tp_{ref}"] = len(pr["flagged"] & pos)
                row[f"fp_{ref}"] = len(pr["flagged"] - pos)
                row[f"fn_{ref}"] = len(pos - pr["flagged"])
                row[f"tn_{ref}"] = len(ex["labels"] - pos - pr["flagged"])
            detail.append(row)
            tot.update({k: v for k, v in row.items() if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                        and k not in ("seed", "window")})
        s = {"scenario": sc, "seed": seed, "config": cfg, "n_windows": len(set(river["seq"]) & set(stable["seq"])),
             "stable_to_river_is_function": function_ok,
             **{k: tot.get(k, 0) for k in ("n_labels", "n_reused", "n_cross_any", "n_cross_supported",
                                           "n_cross_readable", "n_proxy", "n_proxy_readable", "n_proxy_stable",
                                           "n_proxy_stable_readable", "n_stable_labels")}}
        for ref in ("reused", "cross_any", "cross_supported", "cross_readable"):
            tp, fp, fn, tn = (tot.get(f"{q}_{ref}", 0) for q in ("tp", "fp", "fn", "tn"))
            s.update({f"tp_{ref}": tp, f"fp_{ref}": fp, f"fn_{ref}": fn, f"tn_{ref}": tn,
                      f"precision_{ref}": (tp / (tp + fp)) if tp + fp else None,
                      f"recall_{ref}": (tp / (tp + fn)) if tp + fn else None})
        summary.append(s)
    return summary, detail


# --- cross-library agreement: BIRCH -----------------------------------------------------------------
def independent_implementations() -> list[dict]:
    """What independent stream-clustering code this environment has, and what it was used for."""
    out = []
    try:
        import sklearn

        out.append({"library": f"scikit-learn {sklearn.__version__}", "estimator": "sklearn.cluster.Birch",
                    "compared_with": "BIRCH", "status": "run",
                    "note": "online CF-tree insertion in stream order (fit iterates the rows), radius threshold, "
                            "centroid-distance descent, optional AgglomerativeClustering global phase"})
        out.append({"library": f"scikit-learn {sklearn.__version__}", "estimator": "sklearn.cluster.MiniBatchKMeans",
                    "compared_with": "STREAMKMeans", "status": "not run",
                    "note": "a different algorithm (mini-batch Lloyd updates, k-means++ seeding), not an independent "
                            "implementation of STREAM/STREAMKMeans"})
        out.append({"library": f"scikit-learn {sklearn.__version__}", "estimator": "sklearn.cluster.DBSCAN",
                    "compared_with": "DenStream/DBSTREAM offline phase", "status": "not run",
                    "note": "batch DBSCAN only; no online micro-cluster maintenance to compare"})
    except ImportError:
        out.append({"library": "scikit-learn", "estimator": "Birch", "compared_with": "BIRCH",
                    "status": "not available", "note": "not importable in this environment"})
    try:
        java_ok = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        java_ok = False
    out.append({"library": "MOA (Java)", "estimator": "CluStream, DenStream, ClusTree, StreamKM++, BICO, D-Stream",
                "compared_with": "CluStream, DenStream, STREAMKMeans", "status": "not run",
                "note": "no MOA jar in the repo" + ("" if java_ok else "; no Java runtime on this machine")})
    r_ok = any((Path(p) / "Rscript").exists() for p in os.environ.get("PATH", "").split(os.pathsep))
    out.append({"library": "R stream / streamMOA", "estimator": "DSC_DBSTREAM, DSC_DenStream, DSC_CluStream, "
                "DSC_BIRCH, DSC_BICO", "compared_with": "DBSTREAM, DenStream, CluStream, BIRCH", "status": "not run",
                "note": "R not installed" if not r_ok else "R found but the stream package was not checked"})
    for mod, name in (("clusopt_core", "clusopt-core (CluStream, StreamKM)"), ("pyclustering", "pyclustering (BIRCH)"),
                      ("capymoa", "CapyMOA (MOA bindings)")):
        try:
            __import__(mod)
            out.append({"library": name, "estimator": "", "compared_with": "", "status": "installed, not run",
                        "note": "found in this environment; no comparison written"})
        except ImportError:
            out.append({"library": name, "estimator": "", "compared_with": "", "status": "not available",
                        "note": "not installed in .venv"})
    return out


BIRCH_VARIANTS = ("aligned_leaf", "protocol_leaf", "aligned_global", "protocol_global")


def _birch_river(variant: str, T: float, B: int, k: int, h: int, d: int):
    from river import cluster

    if variant.startswith("aligned"):
        kw = dict(threshold=T, branching_factor=B, leaf_capacity=B, max_entries=10**9,
                  distance_measure="d0", threshold_measure="radius")
    else:
        kw = {**start_params("BIRCH", h, d=d), "threshold": T, "branching_factor": B}
    if variant.endswith("global"):
        kw["n_macro_clusters"] = k
    return cluster.BIRCH(**kw), kw


def run_birch_crosslib(task: dict) -> list[dict]:
    """River-fork BIRCH against scikit-learn's ``Birch`` on one scenario's whole stream (see the module
    docstring): both learn every point in stream order, then partition every point and the last window."""
    from sklearn.cluster import Birch

    shape = StreamShape(**task["shape"])
    scen = make_scenario(task["scenario"], task["seed"], shape)
    X, y = scen.X, scen.y
    xs = scen.points()
    T = task["factor"] * shape.sigma
    B = task["branching_factor"]
    rows = []
    for variant in task["variants"]:
        part = {"all": slice(0, len(xs)), "last_window": slice(len(xs) - shape.h, len(xs))}
        k = len(set(y.tolist()))
        k_last = len(set(y[part["last_window"]].tolist()))
        base = {"scenario": scen.name, "seed": scen.seed, "variant": variant, "threshold_factor": task["factor"],
                "threshold": T, "branching_factor": B}
        t0 = time.perf_counter()
        try:
            for name, sl in part.items():
                kk = k if name == "all" else k_last
                model, kw = _birch_river(variant, T, B, kk, shape.h, shape.d)
                for x in xs:
                    model.learn_one(dict(x))
                r = [model.predict_one(dict(x)) for x in xs[sl]]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    sk = Birch(threshold=T, branching_factor=B,
                               n_clusters=kk if variant.endswith("global") else None).fit(X)
                    s = sk.predict(X[sl]).tolist()
                truth = y[sl].tolist()
                rows.append({**base, "partition": name, "n_true_classes": len(set(truth)),
                             "ari_river_sklearn": _ari_pair(r, s), "ari_river_truth": _ari(truth, r),
                             "ari_sklearn_truth": _ari(truth, s),
                             "n_labels_river": len(set(r)), "n_labels_sklearn": len(set(s)),
                             "n_entries_river": len(model.micro_clusters),
                             "n_subclusters_sklearn": int(len(sk.subcluster_centers_)),
                             "n_rebuilds_river": model.n_rebuilds, "final_threshold_river": model.current_threshold,
                             "sklearn_warnings": sorted({str(w.message)[:80] for w in caught}),
                             "river_params": {k_: v for k_, v in kw.items()},
                             "seconds": round(time.perf_counter() - t0, 3), "error": None})
        except Exception as exc:
            rows.append({**base, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def _ari_pair(a, b) -> float | None:
    from sklearn.metrics import adjusted_rand_score

    if len(a) < 2:
        return None
    ia, ib = {}, {}
    return float(adjusted_rand_score([ia.setdefault(v, len(ia)) for v in a], [ib.setdefault(v, len(ib)) for v in b]))


# --- the audit -----------------------------------------------------------------------------------------
#: ``(where, line, snippet)``: ``where`` is ``river:<path in ../river>`` or ``fw:<path in this repo>``; a
#: reference is verified when ``snippet`` is in that line of the current tree (``verify_refs``).
Ref = tuple


def _R(path, line, snippet):
    return ("river:river/cluster/" + path, line, snippet)


def _F(path, line, snippet):
    return ("fw:" + path, line, snippet)


#: Per implementation and aspect: what was read, where, and on what basis (``code``: read in the code
#: at the references; ``notes``: taken from RIVER_BUGS.md / docs / the registry; ``inferred``: a reading
#: of the source algorithm or of absent code that this module did not verify).
AUDIT: list[dict] = [
    # BIRCH ----------------------------------------------------------------------------------------
    {"algo": "BIRCH", "aspect": "initialisation", "basis": "code",
     "text": "No initialisation phase: every point is inserted into the CF tree from the first one. The threshold "
             "starts at `threshold`, River's default 0.0, which the protocol's start keeps (and 0 x s stays 0 under "
             "scaled_default), so until the first rebuild every distinct point is its own leaf entry.",
     "refs": [_R("birch.py", 11, "threshold: float = 0.0"), _R("birch.py", 43, "self._threshold = float(threshold)"),
              _R("birch.py", 398, "self._insert_entry(BIRCHClusteringFeature.from_point(values, w))")]},
    {"algo": "BIRCH", "aspect": "prediction", "basis": "code",
     "text": "Rebuilds the clustering when stale (after any learn_one) and returns the nearest cluster feature under "
             "`distance_measure` (d2, the protocol's pin); 0 before any point.",
     "refs": [_R("birch.py", 405, "self._recluster()"), _R("birch.py", 407, "return 0"),
              _R("birch.py", 412, "for i, feature in self._clusters.items():")]},
    {"algo": "BIRCH", "aspect": "output", "basis": "code",
     "text": "Micro: with Phase 3 off (the protocol pins n_macro_clusters and macro_threshold to None) the labels are "
             "leaf CF entries, numbered afresh by enumerate at every reclustering (GAP-6).",
     "refs": [_R("birch.py", 390, "if self.n_macro_clusters is not None or self.macro_threshold is not None:"),
              _R("birch.py", 392, "self._clusters = dict(enumerate(features))"),
              _F("conf_pipeline/spaces.py", 119, '"n_macro_clusters": None, "macro_threshold": None}')]},
    {"algo": "BIRCH", "aspect": "forgetting/pruning/merging/noise", "basis": "code",
     "text": "No decay, no deletion. Over `max_entries` the tree is rebuilt with a larger threshold; leaf entries "
             "lighter than outlier_fraction x the mean weight are set aside as outliers and re-absorbed when they fit, "
             "and are not labels; predict_one never returns noise. After a split, node refinement merges the closest "
             "pair of children.",
     "refs": [_R("birch.py", 400, "if self._n_entries > self._budget:"),
              _R("birch.py", 334, "self._threshold = self._next_threshold()"),
              _R("birch.py", 335, "kept, outliers = self._partition_outliers(self._leaf_features())"),
              _R("birch.py", 159, "def _refine(self, node, split_index, new_index):")]},
    {"algo": "BIRCH", "aspect": "deviations", "basis": "code+notes",
     "text": "Absorption is tested on the merged diameter by default and the descent uses d2 (average inter-cluster "
             "distance); scikit-learn's Birch tests the merged radius and descends by centroid distance. No class "
             "docstring (river's check_doc fails). Registry: verified by branch tests against the paper pseudocode, "
             "no reference implementation compared; the cross-library section is a first comparison with one. "
             "Phases 2 (condensing) and 4 (refinement passes over the data) of Zhang et al. have no code here "
             "(inferred from their absence).",
     "refs": [_R("birch.py", 16, 'threshold_measure: str = "diameter"'), _R("birch.py", 15, 'distance_measure: str = "d2"'),
              _F("benchmark/algorithms/registry.py", 328, "no reference implementation compared")]},
    # CluStream ------------------------------------------------------------------------------------
    {"algo": "CluStream", "aspect": "initialisation", "basis": "code+inferred",
     "text": "The first max_micro_clusters points each become a one-point micro-cluster, all stamped "
             "max_micro_clusters - 1. Aggarwal et al. (2003) initialise by an offline k-means over InitNumber points "
             "(inferred from the paper, not re-read here).",
     "refs": [_R("clustream.py", 212, "if not self._initialized:"), _R("clustream.py", 219, "timestamp=self.max_micro_clusters - 1"),
              _R("clustream.py", 222, "if len(self.micro_clusters) == self.max_micro_clusters:")]},
    {"algo": "CluStream", "aspect": "prediction", "basis": "code",
     "text": "The label the last offline k-means gives the nearest micro-cluster's centre; 0 until the first k-means "
             "(caught inside River, so the adapter does not count it as a fallback).",
     "refs": [_R("clustream.py", 266, "index, _ = self._get_closest_mc(x)"),
              _R("clustream.py", 268, "return self._kmeans_mc.predict_one(self.micro_clusters[index].center)"),
              _R("clustream.py", 269, "except (KeyError, AttributeError):")]},
    {"algo": "CluStream", "aspect": "output", "basis": "code+notes",
     "text": "Macro: a fresh KMeans every time_gap points, one unweighted pass over the micro-cluster centres; its keys "
             "are the labels and carry no identity across refits (GAP-6). Its centres start at N(mu, sigma) in absolute "
             "units (GAP-5).",
     "refs": [_R("clustream.py", 252, "if self._timestamp % self.time_gap == self.time_gap - 1:"),
              _R("clustream.py", 257, "self._kmeans_mc = cluster.KMeans("),
              _R("clustream.py", 261, "self._kmeans_mc.learn_one(center)")]},
    {"algo": "CluStream", "aspect": "forgetting/pruning/merging/noise", "basis": "code",
     "text": "A point outside its nearest micro-cluster's boundary replaces the first micro-cluster whose relevance "
             "stamp is older than time_window, else the two closest micro-clusters are merged and the point opens one; "
             "no noise label.",
     "refs": [_R("clustream.py", 151, "threshold = self._timestamp - self.time_window"),
              _R("clustream.py", 156, "if mc.relevance_stamp(self.max_micro_clusters) < threshold:"),
              _R("clustream.py", 187, "self.micro_clusters[closest_a] += self.micro_clusters[closest_b]")]},
    {"algo": "CluStream", "aspect": "deviations", "basis": "notes",
     "text": "The docstring's pyramidal time frame is not implemented (GAP-4); horizon queries exist only in the "
             "framework's CluStreamPyramidal, which the runner never queries. The paper's weighted offline k-means is "
             "not implemented (RIVER_BUGS.md GAP-4, 'Not implemented').",
     "refs": [_R("clustream.py", 16, "following a pyramidal pattern")]},
    # DBSTREAM -------------------------------------------------------------------------------------
    {"algo": "DBSTREAM", "aspect": "initialisation", "basis": "code",
     "text": "No initialisation: a point with no micro-cluster within clustering_threshold opens one.",
     "refs": [_R("dbstream.py", 184, "if not neighbor_clusters:"),
              _R("dbstream.py", 187, "self._micro_clusters[new_id] = DBSTREAMMicroCluster(x=x, last_update=t, weight=1)")]},
    {"algo": "DBSTREAM", "aspect": "prediction", "basis": "code+inferred",
     "text": "Reclusters when stale, then the macro cluster whose merged centre is nearest; 0 when there is none. "
             "Hahsler & Bolanos (and R's stream) assign through the nearest micro-cluster within its radius, else noise "
             "(inferred, not re-read): River's nearest merged centre can differ for elongated macro clusters and never "
             "says noise.",
     "refs": [_R("dbstream.py", 402, "self._recluster()"), _R("dbstream.py", 408, "closest_cluster_index = 0"),
              _R("dbstream.py", 410, "for i, center_i in self._centers.items():")]},
    {"algo": "DBSTREAM", "aspect": "output", "basis": "code",
     "text": "Macro: connected components of the shared-density graph over micro-clusters heavier than minimum_weight "
             "(lighter ones belong to no macro cluster), numbered afresh at each reclustering (GAP-6).",
     "refs": [_R("dbstream.py", 320, "if self._micro_clusters[index].weight < self.minimum_weight:"),
              _R("dbstream.py", 388, "self._n_clusters, self._clusters = self._generate_clusters_from_labels(labels)")]},
    {"algo": "DBSTREAM", "aspect": "forgetting/pruning/merging/noise", "basis": "code",
     "text": "Weights fade; every cleanup_interval points weak micro-clusters and weak shared-density entries are "
             "removed; centres that come within the threshold of each other are moved back; no noise label.",
     "refs": [_R("dbstream.py", 198, "mc.weight = mc.weight * 2 ** (-fading * (t - mc.last_update)) + 1"),
              _R("dbstream.py", 252, "if mc.weight * value < weight_weak:"),
              _R("dbstream.py", 396, "if self._time_stamp % self.cleanup_interval == 0:"),
              _R("dbstream.py", 233, "if self._distance(mc_i.center, mc_j.center) < self.clustering_threshold:")]},
    {"algo": "DBSTREAM", "aspect": "deviations", "basis": "notes",
     "text": "Labels carry no identity across reclusterings (GAP-6; counterexample in tests/test_partition_audit.py). "
             "learn_one keeps the caller's x (BUG-3, check_no_state_aliasing_with_input).",
     "refs": [_R("dbstream.py", 360, "for label in sorted(by_label):")]},
    # DenStream ------------------------------------------------------------------------------------
    {"algo": "DenStream", "aspect": "initialisation", "basis": "code",
     "text": "Buffers n_samples_init points (the protocol's start caps it at h), then a DBSCAN over them seeds the "
             "p-micro-clusters; predict_one returns 0 until then (inside River, not counted as a fallback).",
     "refs": [_R("denstream.py", 320, "if not self.initialized:"), _R("denstream.py", 323, "self._initial_dbscan()"),
              _R("denstream.py", 354, "if not self.initialized:")]},
    {"algo": "DenStream", "aspect": "prediction", "basis": "code+notes",
     "text": "Every call re-runs the offline DBSCAN over the p-micro-clusters (GAP-3) and returns the nearest merged "
             "centre; -1 when there is no p-micro-cluster (BUG-6; the adapter keeps it as a label and counts it).",
     "refs": [_R("denstream.py", 361, "labels = {pmc: None for pmc in self.p_micro_clusters.values()}"),
              _R("denstream.py", 389, "return self._get_closest_cluster_key(x, self.clusters)"),
              _R("denstream.py", 193, "key = -1")]},
    {"algo": "DenStream", "aspect": "output", "basis": "code+notes",
     "text": "Macro by design, but the seed queue only takes already-labelled neighbours (BUG-8), so macro clusters are "
             "in practice the p-micro-clusters; the shallow copy before merging lets predict_one rewrite a live "
             "p-micro-cluster (BUG-7); labels renumbered on every call (GAP-6).",
     "refs": [_R("denstream.py", 384, "if labels[neighbor_neighbor] is not None:"),
              _R("denstream.py", 274, "cluster = copy.copy(micro_clusters[0])")]},
    {"algo": "DenStream", "aspect": "forgetting/pruning/merging/noise", "basis": "code+notes",
     "text": "Weights fade; every _time_period, p-micro-clusters below beta*mu and o-micro-clusters below xi are "
             "deleted; promotion o->p is keyed by len(dict), which can overwrite a live key (GAP-3 notes). "
             "o-micro-clusters are the outlier buffer, but predict_one never returns a noise label.",
     "refs": [_R("denstream.py", 332, "if self.timestamp > 0 and self.timestamp % self._time_period == 0:"),
              _R("denstream.py", 334, "if p_micro_cluster_i.calc_weight(self.timestamp) < self.mu * self.beta:"),
              _R("denstream.py", 347, "if o_micro_cluster_j.calc_weight(self.timestamp) < xi:"),
              _R("denstream.py", 223, "self.p_micro_clusters[len(self.p_micro_clusters)] = closest_omc")]},
    {"algo": "DenStream", "aspect": "deviations", "basis": "notes",
     "text": "BUG-6, BUG-7, BUG-8, GAP-3 (RIVER_BUGS.md); calc_radius's norm-based formula returns 0 for micro-clusters "
             "away from the origin (BUG-8 notes), so few pairs are ever density-reachable.",
     "refs": [_R("denstream.py", 246, "if distance < 2 * self.epsilon and distance <= c_p.calc_radius(")]},
    # DenStreamPatchedFast -------------------------------------------------------------------------
    {"algo": "DenStreamPatchedFast", "aspect": "all", "basis": "code+notes",
     "text": "DenStream's learning unchanged; predict_one deep-copies before merging (BUG-7 fixed), expands the seed "
             "queue with unlabelled neighbours (BUG-8 fixed) and caches the offline phase (GAP-3). Exact with "
             "DenStreamPatched (tests/test_denstream_fast.py), not with DenStream. Sentinel -1 inherited (BUG-6).",
     "refs": [_F("benchmark/algorithms/denstream_patched.py", 57, "merged = copy.deepcopy(micro_clusters[0])"),
              _F("benchmark/algorithms/denstream_patched.py", 86, "if labels[neighbor] is None:"),
              _F("benchmark/algorithms/denstream_fast.py", 188, "def predict_one(self, x, w=None):")]},
    # HPStream -------------------------------------------------------------------------------------
    {"algo": "HPStream", "aspect": "initialisation", "basis": "code",
     "text": "Buffers n_samples_init points (capped at h by the protocol), then an offline k-means whose assignment "
             "and projected dimensions are refined against each other; the stream is normalised by running standard "
             "deviations, refreshed every normalization_interval points; 0 while no cluster exists.",
     "refs": [_R("hpstream.py", 416, "if not self._initialized:"), _R("hpstream.py", 419, "self._initialize()"),
              _R("hpstream.py", 462, "def _renormalize(self):"),
              _R("hpstream.py", 432, "return 0 if label is None else label")]},
    {"algo": "HPStream", "aspect": "prediction", "basis": "code",
     "text": "Nearest cluster by the Manhattan segmental distance over that cluster's own dimensions; the limiting "
             "radius is not tested at prediction, so every point gets a cluster.",
     "refs": [_R("hpstream.py", 431, "label, _ = self._closest(self._normalize(x), self._clusters)"),
              _R("hpstream.py", 553, "distance += abs(point.get(feature, 0.0) - sum1[feature] / total)")]},
    {"algo": "HPStream", "aspect": "output", "basis": "code+notes",
     "text": "The fading clusters (at most max_clusters) are the output, with no macro phase; labels are their keys, "
             "and a new cluster takes the smallest free key, so an evicted cluster's label passes to the newcomer "
             "inside a window (GAP-6).",
     "refs": [_R("hpstream.py", 628, "clusters[self._free_label()] = fresh"),
              _R("hpstream.py", 648, "The smallest label no cluster is holding")]},
    {"algo": "HPStream", "aspect": "forgetting/pruning/merging/noise", "basis": "code",
     "text": "Decay applied lazily on write; a cluster is deleted only by LRU eviction when a new one needs a slot, or "
             "when the global dimension ranking leaves it none; a point outside every limiting radius opens a cluster "
             "(noise is not a label); no online merge or split.",
     "refs": [_R("hpstream.py", 672, "factor = self._decay ** (self._timestamp - cluster.last_update)"),
              _R("hpstream.py", 627, "del clusters[min(clusters, key=lambda key: clusters[key].last_seen)]"),
              _R("hpstream.py", 618, "for starved in [key for key, cluster in clusters.items() if not cluster.dimensions]:")]},
    {"algo": "HPStream", "aspect": "deviations", "basis": "code+notes",
     "text": "Room is made before the newcomer is filed, unlike Figure 1 (documented in the code; it is what lets the "
             "evicted label pass on). Degenerate regime (one-point clusters with limiting radius 0) documented and "
             "warned about. Registry: 'verification state not recorded'.",
     "refs": [_R("hpstream.py", 622, "Room is made before the newcomer is filed"),
              _R("hpstream.py", 115, "When the algorithm breaks down"),
              _F("benchmark/algorithms/registry.py", 395, "verification state not")]},
    {"algo": "HPStreamStableLabels", "aspect": "all", "basis": "code+notes",
     "text": "HPStream with `_free_label` drawing from a counter that only grows (from max_clusters): the same clusters "
             "and predicted cluster at every point, labels never reused (tests/test_hpstream_stable_labels.py).",
     "refs": [_F("benchmark/algorithms/hpstream_stable_labels.py", 41, "def _free_label(self) -> int:")]},
    # RepStream ------------------------------------------------------------------------------------
    {"algo": "RepStream", "aspect": "initialisation", "basis": "code",
     "text": "No initialisation: every point becomes a vertex of the k-NN sparse graph from the first one; a point "
             "equal to a singularity only updates it.",
     "refs": [_R("repstream.py", 769, "neighbours = self._nearest(values, self.k)"),
              _R("repstream.py", 759, "singularity = self._matching_singularity(values)")]},
    {"algo": "RepStream", "aspect": "prediction", "basis": "code",
     "text": "The cluster of the nearest representative point; 0 while there is none.",
     "refs": [_R("repstream.py", 787, "for i in self._representatives:"), _R("repstream.py", 793, "return vertices[best[1]].cluster")]},
    {"algo": "RepStream", "aspect": "output", "basis": "code",
     "text": "Macro with no offline phase: connected components of density-related links between representatives; a "
             "merge keeps the larger cluster's label, a split and a new representative take fresh labels from a "
             "monotone counter, so no label is reused.",
     "refs": [_R("repstream.py", 448, "related = distance <= vertex.density * alpha and distance <= other.density * alpha"),
              _R("repstream.py", 392, "target = self._next_cluster"), _R("repstream.py", 469, "cluster = self._next_cluster")]},
    {"algo": "RepStream", "aspect": "forgetting/pruning/merging/noise", "basis": "code",
     "text": "A vertex budget (max_vertices): vertices outside the repository are retired FIFO; the repository keeps "
             "representatives by usefulness (decay_rate); no noise label.",
     "refs": [_R("repstream.py", 747, "def _enforce_budget(self):"),
              _R("repstream.py", 743, "if index in self._vertices and index not in self._repository:")]},
    {"algo": "RepStream", "aspect": "deviations", "basis": "notes",
     "text": "The registry says six pseudocode/prose conflicts and one extra maintenance step (Definition 7) are "
             "'recorded in the estimator docstring'; the docstring in the vendored tree (repstream.py:12-164) "
             "contains no such record (searched for pseudocode, prose, Definition, paper), so they could not be read "
             "here. Registry: no reference implementation compared.",
     "refs": [_F("benchmark/algorithms/registry.py", 474, "Six places"),
              _R("repstream.py", 84, "Lühr S, Lazarescu M (2009)")]},
    # SOStream -------------------------------------------------------------------------------------
    {"algo": "SOStream", "aspect": "initialisation", "basis": "code",
     "text": "The first min_pts points each open a micro-cluster.",
     "refs": [_R("sostream.py", 266, "if len(self._micro_clusters) >= self.min_pts:"),
              _R("sostream.py", 277, "self._new_cluster(values, w)")]},
    {"algo": "SOStream", "aspect": "prediction", "basis": "code",
     "text": "The nearest micro-cluster centre; 0 when there is none.",
     "refs": [_R("sostream.py", 287, "label = 0"), _R("sostream.py", 290, "distance = euclidean_distance_dict(micro_cluster.center, values)")]},
    {"algo": "SOStream", "aspect": "output", "basis": "code",
     "text": "Micro = macro: the micro-clusters are the clusters (no offline phase), named from a monotone counter, so "
             "no label is reused; a merge drops the absorbed label.",
     "refs": [_R("sostream.py", 298, "return len(self._micro_clusters)"), _R("sostream.py", 194, "self._next_id = new_id + 1")]},
    {"algo": "SOStream", "aspect": "forgetting/pruning/merging/noise", "basis": "code",
     "text": "Every fading_interval points every micro-cluster fades and those below fade_threshold are deleted; "
             "overlapping neighbours closer than merge_threshold are merged into the winner; a point outside the "
             "winner's k-distance radius opens a micro-cluster; no noise label.",
     "refs": [_R("sostream.py", 282, "if self.fading_interval and self._n_samples_seen % self.fading_interval == 0:"),
              _R("sostream.py", 260, "if micro_cluster.weight < fade_threshold:"),
              _R("sostream.py", 246, "if euclidean_distance_dict(winner.center, other.center) < merge_threshold:")]},
    {"algo": "SOStream", "aspect": "deviations", "basis": "notes",
     "text": "Registry: verified by branch tests against the paper pseudocode, no reference implementation compared; "
             "none is available in this environment.",
     "refs": [_F("benchmark/algorithms/registry.py", 485, "no reference implementation compared")]},
    # STREAMKMeans ---------------------------------------------------------------------------------
    {"algo": "STREAMKMeans", "aspect": "initialisation", "basis": "code+notes",
     "text": "Centres of a KMeans whose coordinates are drawn lazily from N(mu, sigma) in absolute units (GAP-5); "
             "before any learn_one there are no centres and predict_one raises (BUG-5), which the adapter turns into "
             "label 0 and counts as a fallback.",
     "refs": [_R("streamkmeans.py", 84, "self._kmeans = cluster.KMeans(n_clusters=self.n_clusters, **self.kwargs)"),
              _R("streamkmeans.py", 86, "self.centers = {}"), _F("benchmark/algorithms/adapter.py", 190, "self.n_fallback_predictions += 1")]},
    {"algo": "STREAMKMeans", "aspect": "prediction", "basis": "code",
     "text": "The nearest centre of the global incremental KMeans; the centres exist from the first learn_one, still "
             "at their random start until the first chunk completes.",
     "refs": [_R("streamkmeans.py", 107, "self.centers = self._kmeans.centers"),
              _R("streamkmeans.py", 113, "return min(self.centers, key=get_distance)")]},
    {"algo": "STREAMKMeans", "aspect": "output", "basis": "code",
     "text": "Macro with k fixed (n_clusters; River's default 2, which the protocol's start keeps); centre keys persist "
             "for the whole run, so labels are never renamed.",
     "refs": [_R("streamkmeans.py", 77, "def __init__(self, chunk_size=10, n_clusters=2, **kwargs):")]},
    {"algo": "STREAMKMeans", "aspect": "forgetting/pruning/merging/noise", "basis": "code",
     "text": "No explicit forgetting: each full chunk is clustered by a fresh KMeans whose centres each move the "
             "nearest global centre by halflife; a global centre nearest to no chunk centre never moves.",
     "refs": [_R("streamkmeans.py", 101, "kmeans_i = cluster.KMeans(n_clusters=self.n_clusters, **self.kwargs)"),
              _R("streamkmeans.py", 105, "self._kmeans.learn_one(center_j)"),
              _R("k_means.py", 115, "self.centers[closest][i] += self.halflife * (xi - self.centers[closest][i])")]},
    {"algo": "STREAMKMeans", "aspect": "deviations", "basis": "code",
     "text": "STREAMLSEARCH's k-median LSEARCH replaced by incremental k-means (the docstring says so); chunk centres "
             "enter the global KMeans unweighted.",
     "refs": [_R("streamkmeans.py", 11, "by replacing the k-medians using `LSEARCH` by the k-means algorithm")]},
]


def _ref_root(where: str) -> Path:
    kind, rel = where.split(":", 1)
    root = C.REPO_ROOT if kind == "fw" else C.REPO_ROOT.parent / "river"
    return root / rel


def verify_refs(audit: list[dict] = AUDIT) -> list[dict]:
    """Every audit reference checked against the current tree: ``ok`` when its snippet is in that line."""
    out = []
    cache: dict[Path, list[str]] = {}
    for entry in audit:
        for where, line, snippet in entry["refs"]:
            path = _ref_root(where)
            try:
                lines = cache.setdefault(path, path.read_text(encoding="utf-8").splitlines())
                ok = 0 < line <= len(lines) and snippet in lines[line - 1]
                status = "ok" if ok else "mismatch"
            except OSError:
                status = "missing file"
            out.append({"algo": entry["algo"], "aspect": entry["aspect"], "ref": _show_ref(where, line),
                        "snippet": snippet, "status": status})
    return out


def _show_ref(where: str, line: int) -> str:
    kind, rel = where.split(":", 1)
    return f"{'../river/' if kind == 'river' else ''}{rel}:{line}"


# --- river's own behavioural checks (reused, benchmark.algorithms.checks) ------------------------------
def river_checks(algos) -> list[dict]:
    """``benchmark.algorithms.checks.run_clusterer_checks`` on each implementation as the registry builds it
    (its own defaults, not the protocol's start): the existing behavioural matrix, reused here rather than
    duplicated; ``tests/test_clusterer_checks.py`` holds the expected cells and their RIVER_BUGS.md entries."""
    import random

    from benchmark.algorithms import checks
    from benchmark.algorithms.registry import get_spec

    rows = []
    for algo in algos:
        random.seed(0)  # as tests/test_clusterer_checks.py pins it
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = checks.run_clusterer_checks(get_spec(algo).factory()[0])
        except Exception as exc:
            rows.append({"algo": algo, "check": "<harness>", "outcome": f"error: {type(exc).__name__}: {exc}"})
            continue
        rows += [{"algo": algo, "check": k, "outcome": v} for k, v in sorted(report.items())]
    return rows


# --- the report ------------------------------------------------------------------------------------------
def _md_table(rows: list[dict], cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v).replace("|", "/").replace("\n", " ")


def _status_cell(statuses: list[str]) -> str:
    c = Counter(statuses)
    if not c:
        return "-"
    if set(c) == {"not-applicable"}:
        return "n/a"
    parts = [f"{c[s]}{s[0].upper()}" for s in ("pass", "fail", "undetermined") if c.get(s)]
    if c.get("not-applicable"):
        parts.append(f"{c['not-applicable']}n/a")
    return " ".join(parts)


def write_report(fid: Path, meta: dict, windows: list[dict], checks: list[dict], cells: list[dict],
                 ident: list[dict], birch: list[dict], indep: list[dict], refs: list[dict],
                 rchecks: list[dict]) -> Path:
    import pandas as pd

    L: list[str] = []
    A = L.append
    shape, th, px = meta["shape"], meta["thresholds"], meta["proxy"]
    A("# Implementation fidelity on controlled streams\n")
    A(f"Generated {meta['finished']} by `{meta['command']}` in {meta['wall_s']:.0f} s "
      f"({meta['workers']} workers). Python {platform.python_version()}, river tree `{meta['river_commit']}`, "
      f"scikit-learn {meta['sklearn']}, PYTHONHASHSEED={meta['pythonhashseed']}. Evidence, not a verdict: each check "
      "reads pass, fail, not-applicable or undetermined per cell; nothing here says an implementation is valid.\n")
    if meta["pythonhashseed"] != "0":
        A("**Warning: PYTHONHASHSEED was not 0; River's KMeans results depend on it (RIVER_BUGS.md BUG-2).**\n")
    A("## 1. What was run\n")
    A(f"Implementations: {', '.join(meta['algos'])}. Study-written (River fork): {', '.join(STUDY_WRITTEN)}; "
      "CluStream, DBSTREAM, DenStream and STREAMKMeans are River upstream; HPStreamStableLabels and "
      "DenStreamPatchedFast are framework variants read against HPStream and DenStream.\n")
    A(f"Configurations: {', '.join(meta['configs'])} (the protocol's start, `spaces.start_params`, and its distance "
      "parameters times the calibration window's `s`, `spaces.scaled_defaults`), built by `run._make_adapter_factory` "
      "with scaler `none`, model seed "
      f"{meta['model_seed']}. A `scaled_default` cell whose parameters equal the `default` ones is not run again "
      "(HPStream, HPStreamStableLabels and RepStream have no distance parameter; BIRCH's threshold is 0). Data seeds: "
      f"{', '.join(map(str, meta['seeds']))}.\n")
    A(f"Stream shape (placeholders): {shape['n_windows']} windows of h = {shape['h']} points "
      f"({shape['h'] * shape['n_windows']} per scenario), d = {shape['d']}, sigma = {shape['sigma']}, centres at least "
      f"{shape['separation']} sigma apart in every coordinate. Window 0 is the warm start (learn only, as the ensemble "
      "warms a member); windows 1.. are predict-before-learn through the adapter, then predicted again by a deep "
      "copy (the snapshot).\n")
    A("### Scenarios and event times\n")
    srows = []
    sh = StreamShape(**shape)
    for name in meta["scenarios"]:
        sc = make_scenario(name, meta["seeds"][0], sh)
        ev = "; ".join(f"{e['event']} {e['labels']} at t={e['t']} (window {e['window']})" for e in sc.events[:6])
        if len(sc.events) > 6:
            ev += f"; ... {len(sc.events)} events in all"
        srows.append({"scenario": name, "classes": len(set(sc.y.tolist())), "events": ev or "none",
                      "phases": " ".join(PHASE_ABBR[p] for p in sc.phases)})
    A(_md_table(srows, ["scenario", "classes", "events", "phases"]))
    A("\nPhases per window: W warm start, S stationary or stress, `-` before the event, T transition (no check reads "
      "it), `+` after the event. Truth is the generating component; the two coinciding components after a merge are "
      f"one new class. Classes counted for seed {meta['seeds'][0]}; full event lists per seed: `scenarios.csv`.\n")
    A("### Checks (thresholds are placeholders)\n")
    defs = [
        ("stationary_seq_ari", f"mean sequential ARI over windows >= {1 + int(th['warmup_skip_windows'])} is at least "
                               f"{th['stationary_seq_ari_min']}"),
        ("stationary_snap_ari", f"the same on the snapshot, at least {th['stationary_snap_ari_min']}"),
        ("birth_new_label", "the new class gets a plurality snapshot label that no other class has as its plurality, "
                            f"within {int(th['birth_max_delay_windows'])} window(s) of the event window"),
        ("deletion_absorbed", f"{shape['probe_points']} probes drawn from the deleted cluster, predicted on each "
                              "snapshot, take (in plurality) a label that at least "
                              f"{int(th['min_class_points'])} points of the remaining classes carry, within "
                              f"{int(th['deletion_max_delay_windows'])} window(s); undetermined when, before the "
                              "deletion, the class's plurality label was already carried by that many remaining "
                              "points"),
        ("merge_label_count_down", f"labels needed to cover {th['cover_share']} of the merged points after the merge "
                                   "are fewer than those covering the pair before it, within "
                                   f"{int(th['merge_max_delay_windows'])} window(s); undetermined when the pair shared "
                                   "a plurality label before"),
        ("split_label_count_up", "labels covering the children after the split exceed those covering the parent "
                                 f"before, within {int(th['split_max_delay_windows'])} window(s)"),
        ("split_children_distinct", "the two children get different plurality labels within "
                                    f"{int(th['split_max_delay_windows'])} window(s)"),
        ("stress_births_told_apart", f"a share of at least {th['stress_detect_share_min']} of the births get a "
                                     "plurality label of their own within "
                                     f"{int(th['stress_max_delay_windows'])} window(s)"),
    ]
    A(_md_table([{"check": a, "passes when": b} for a, b in defs], ["check", "passes when"]))
    A(f"\nA class has a plurality label in a window when it has at least {int(th['min_class_points'])} points there. "
      "Not-applicable: the design does not promise the behaviour (below, with the code it rests on) or the "
      "configuration's cluster capacity (CluStream n_macro_clusters, STREAMKMeans n_clusters, HPStream max_clusters) "
      "is below the classes the check needs. The checks read the snapshot (the model's partition at the end of each "
      "window); only `stationary_seq_ari` reads the sequential labels. They are weak for implementations that cover "
      "every class with many labels (BIRCH's leaf entries, RepStream, SOStream at their defaults): a new class or a "
      "child then gets a label of its own trivially, and a count can move by sampling alone; `checks.csv` keeps the "
      "counts and delays.\n")
    A(_md_table([{"implementation": a, "check": c, "reason": r} for (a, c), r in sorted(NOT_PROMISED.items())],
                ["implementation", "check", "reason"]))
    A("\n## 2. Expected-behaviour checks\n")
    A("Each cell counts seeds: P pass, F fail, U undetermined, n/a not-applicable. An implementation's error "
      "(the cell raised) is listed below the table.\n")
    ck = pd.DataFrame(checks)
    order = [c for _, cs in SCENARIO_CHECKS.items() for c in cs]
    for cfg in meta["configs"]:
        sub = ck[ck.config == cfg] if not ck.empty else ck
        if sub.empty:
            continue
        A(f"### {cfg}\n")
        rows = []
        for algo in meta["algos"]:
            r = {"implementation": algo}
            for c in order:
                r[c] = _status_cell(sub[(sub.algo == algo) & (sub.check == c)]["status"].tolist())
            rows.append(r)
        A(_md_table(rows, ["implementation"] + order, ["implementation"] + [c.replace("_", " ") for c in order]))
        A("")
    errs = [c for c in cells if c.get("error")]
    if errs:
        A("Cells that raised:\n")
        for c in errs:
            A(f"- {c['algo']} / {c['scenario']} / seed {c['seed']} / {c['config']}: {c['error']}")
        A("")
    A("### Readings behind the checks (means over seeds and over the windows a check reads)\n")
    A("No warm-start or transition window. Per scenario: sequential / snapshot ARI; predicted labels per window "
      "(snapshot); the identity proxy's flags per window on sequential / snapshot labels. A snapshot comes from one "
      "model state and cannot reuse a label, so a flag there is the proxy firing without any reuse (a label covering "
      "several classes while one of them dies or appears): read the sequential count against it. Last table: the "
      "share of sequential predictions that were DenStream's sentinel or the adapter's fallback, over every "
      "scenario, and the cluster count the adapter reads.\n")
    wd = pd.DataFrame(windows)
    if not wd.empty:
        wd = wd[~wd.phase.isin(["warmup", "transition"])]
        for cfg in meta["configs"]:
            sub = wd[wd.config == cfg]
            if sub.empty:
                continue
            A(f"#### {cfg}\n")
            g = {k: v for k, v in sub.groupby(["algo", "scenario"])}
            scs = [s for s in meta["scenarios"] if s in set(sub.scenario)]

            def pivot(fn):
                rows = []
                for algo in meta["algos"]:
                    r = {"implementation": algo}
                    for s in scs:
                        r[s] = fn(g[(algo, s)]) if (algo, s) in g else "-"
                    rows.append(r)
                return rows

            A("ARI, sequential / snapshot:\n")
            A(_md_table(pivot(lambda x: f"{_fmt(x.ari_seq.mean())} / {_fmt(x.ari_snap.mean())}"), ["implementation"] + scs))
            A("\nPredicted labels per window (snapshot):\n")
            A(_md_table(pivot(lambda x: f"{x.n_labels_snap.mean():.1f}"), ["implementation"] + scs))
            A("\nIdentity proxy, flagged labels per window, sequential / snapshot:\n")
            A(_md_table(pivot(lambda x: f"{x.proxy_flags_seq.mean():.2f} / {x.proxy_flags_snap.mean():.2f}"),
                        ["implementation"] + scs))
            rows = []
            for algo in meta["algos"]:
                a = sub[sub.algo == algo]
                n = a.n_points.sum()
                rows.append({"implementation": algo, "sentinel share": (a.n_sentinel_seq.sum() / n) if n else None,
                             "fallback share": (a.n_fallback_seq.sum() / n) if n else None,
                             "model clusters (adapter, mean)": a.n_model_clusters.mean() if n else None})
            A("\n" + _md_table(rows, list(rows[0])))
            A("")
    A("## 3. Label identity: the proxy, validated on HPStream\n")
    A(f"Proxy: {int(px['segments'])} segments per window; a label's majority class counts in a segment with at least "
      f"{px['min_support']} points and a share of at least {px['min_share']}; a label is flagged when two segments "
      "disagree, and is readable when two segments give it a majority at all. Exact reuse: an HPStream label covering "
      "two HPStreamStableLabels labels in one window. Cross-class: reuse whose cluster objects have different "
      "plurality true classes in the window (the harm the proxy is meant to see; reuse within one class does not "
      "change an ARI). Readable: the cross-class reuse the proxy could read at all. Counts are (window, HPStream label) "
      "pairs, summed over seeds. Variant: the proxy on HPStreamStableLabels' own sequential labels, which are never "
      "reused, so every flag there is a false positive by construction (flags / readable / all label-windows).\n")
    if ident:
        idf = pd.DataFrame(ident)
        rows = []
        for (cfg, sc), g in idf.groupby(["config", "scenario"], sort=False):
            tot = g.sum(numeric_only=True)

            def ratio(a, b):
                return (a / b) if b else None

            rows.append({"config": cfg, "scenario": sc, "label-windows": int(tot.n_labels),
                         "reused": int(tot.n_reused), "cross-class": int(tot.n_cross_any),
                         "readable cross-class": int(tot.n_cross_readable), "flagged": int(tot.n_proxy),
                         "precision vs reuse": ratio(tot.tp_reused, tot.tp_reused + tot.fp_reused),
                         "recall vs reuse": ratio(tot.tp_reused, tot.n_reused),
                         "recall vs cross-class": ratio(tot.tp_cross_any, tot.n_cross_any),
                         "recall vs readable": ratio(tot.tp_cross_readable, tot.n_cross_readable),
                         "variant": f"{int(tot.n_proxy_stable)} / {int(tot.n_proxy_stable_readable)} / "
                                    f"{int(tot.n_stable_labels)}",
                         "same clusters": bool(g.stable_to_river_is_function.all())})
        A(_md_table(rows, list(rows[0])))
        A("\n`same clusters`: every variant label maps to one HPStream label in every window, which holds when both "
          "models hold the same clusters (the premise of the validation); detail per window in "
          "`identity_windows.csv`.\n")
    else:
        A("Not run: HPStream and HPStreamStableLabels were not both run without error.\n")
    A("## 4. Cross-library agreement: BIRCH against scikit-learn\n")
    A(_md_table(indep, ["library", "estimator", "compared_with", "status", "note"]))
    A("\nBoth learn every point of a scenario in stream order with threshold T = factor x sigma and branching factor "
      f"{meta['birch_branching_factor']}; then every point (`all`) and the last window are partitioned. Variants: "
      "`aligned` sets River's tree as close to scikit-learn's as its parameters allow (merged-radius threshold, "
      "centroid-distance descent d0, leaf capacity = branching factor, no memory budget); `protocol` keeps the "
      "benchmark's pins (merged diameter, d2, leaf capacity 50, max_entries 500) with the same T; `global` turns on "
      "each library's global phase with the true number of classes (River: agglomerative merging of leaf entries by "
      "the same measure; scikit-learn: Ward AgglomerativeClustering on subcluster centroids). Remaining differences by "
      "design: River's node refinement after splits and capacity-balanced splits; scikit-learn's unweighted global "
      "phase. Means over seeds and scenarios:\n")
    if birch:
        bdf = pd.DataFrame([r for r in birch if not r.get("error")])
        if not bdf.empty:
            rows = []
            for (v, f, p), g in bdf.groupby(["variant", "threshold_factor", "partition"], sort=True):
                rows.append({"variant": v, "T/sigma": f, "partition": p, "ARI river-sklearn": g.ari_river_sklearn.mean(),
                             "min": g.ari_river_sklearn.min(), "labels river": g.n_labels_river.mean(),
                             "labels sklearn": g.n_labels_sklearn.mean(), "ARI river-truth": g.ari_river_truth.mean(),
                             "ARI sklearn-truth": g.ari_sklearn_truth.mean(), "rebuilds": g.n_rebuilds_river.mean()})
            A(_md_table(rows, list(rows[0])))
            A("\nPer scenario and seed: `birch_crosslib.csv`.\n")
        berr = [r for r in birch if r.get("error")]
        for r in berr:
            A(f"- error: {r['variant']} / {r['scenario']} / seed {r['seed']}: {r['error']}")
    A("\n## 5. Audit per implementation\n")
    A("Read at the references (river tree paths are `../river/...`). Basis: `code` read in the code at the "
      "references; `notes` taken from RIVER_BUGS.md, docs/ or the registry; `inferred` a reading of the source paper "
      "or of absent code that was not verified here. Every reference's snippet is checked against the current tree "
      f"on each run: {sum(r['status'] == 'ok' for r in refs)}/{len(refs)} found at the stated line"
      + ("" if all(r["status"] == "ok" for r in refs) else " (mismatches in `audit_refs.csv`)") + ".\n")
    A(_md_table([{"implementation": e["algo"], "aspect": e["aspect"], "basis": e["basis"], "reading": e["text"],
                  "refs": ", ".join(_show_ref(w, ln) for w, ln, _ in e["refs"])} for e in AUDIT],
                ["implementation", "aspect", "basis", "reading", "refs"]))
    A("\nConventions the benchmark adds (`benchmark/algorithms/adapter.py`): predict before learn on every point; a "
      "ValueError, ZeroDivisionError or KeyError from predict_one becomes label 0, counted (`n_fallback`); DenStream's "
      "-1 is kept as a label and counted (`n_sentinel`, BUG-6); no implementation is given a noise label, and none "
      "returns one.\n")
    A("### River's behavioural checks (existing harness, reused)\n")
    A("`benchmark.algorithms.checks.run_clusterer_checks` on each implementation as the registry builds it (registry "
      "defaults, not the protocol's start). Failing checks only; the expected cells and their RIVER_BUGS.md entries are "
      "pinned in `tests/test_clusterer_checks.py`.\n")
    if rchecks:
        fails = defaultdict(list)
        for r in rchecks:
            if not r["outcome"].startswith(("pass", "skip")):
                body = r["outcome"].removeprefix("fail: ")
                msg, sep, where = body.rpartition(" @ ")
                if not sep:
                    msg, where = body, ""
                short = (msg[:60] + "...") if len(msg) > 60 else msg
                fails[r["algo"]].append(f"{r['check']} ({short}{' @ ' + where if where else ''})")
        A(_md_table([{"implementation": a, "failing checks": "; ".join(fails.get(a, [])) or "none",
                      "checks run": sum(r["algo"] == a for r in rchecks)} for a in meta["algos"]],
                    ["implementation", "failing checks", "checks run"]))
    else:
        A("Not run (`--no-river-checks`).")
    A("\n## 6. What this does not show\n")
    A("- Every threshold, window count and the geometry are placeholders; a pass or fail moves with them. The "
      "configurations are the protocol's fixed starts, never tuned ones; how a tuned configuration behaves on these "
      "cases is not measured here.\n"
      "- The scenarios are low-dimensional isotropic Gaussians: nothing here tests HPStream's projections on "
      "high-dimensional data, arbitrary shapes, or noise points (there is no noise).\n"
      "- The identity proxy needs truth and confounds reuse with clusters that really change what they cover; its "
      "agreement is measured only on HPStream, the one implementation whose reuse is known exactly.\n"
      "- Cross-library agreement covers BIRCH only; no MOA, R `stream` or other independent implementation was run "
      "(table in section 4).\n"
      "- `n_model_clusters` in `windows.csv` is what the adapter reads; for CluStream and STREAMKMeans it is the "
      "parameter k.\n")
    path = fid / "fidelity.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# --- the command -------------------------------------------------------------------------------------------
def _csv(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def _kv(items: list[str], known: dict) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if k not in known:
            raise SystemExit(f"unknown key {k!r}; known: {sorted(known)}")
        out[k] = float(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="results go to <out-dir>/fidelity/")
    p.add_argument("--algos", default=",".join(ALGOS))
    p.add_argument("--seeds", default="1,2,3", help="data seeds of the scenarios (placeholder)")
    p.add_argument("--config", default="default", help="comma list of default, scaled_default")
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--h", type=int, default=StreamShape.h)
    p.add_argument("--windows", type=int, default=StreamShape.n_windows, help="windows per scenario, warm start included")
    p.add_argument("--d", type=int, default=StreamShape.d)
    p.add_argument("--sigma", type=float, default=StreamShape.sigma)
    p.add_argument("--separation", type=float, default=StreamShape.separation, help="in sigma, per coordinate")
    p.add_argument("--stress-period", type=int, default=None, help="points between births in identity_stress (h/2)")
    p.add_argument("--model-seed", type=int, default=C.MODEL_SEED)
    p.add_argument("--stream-scaling", default=C.STREAM_SCALING, choices=C.STREAM_SCALING_RULES,
                   help="the benchmark's stream scaling (BIRCH's cross-library check reads the raw scenario)")
    p.add_argument("--scaling-window", type=int, default=C.SCALING_WINDOW, help="default: one window h")
    p.add_argument("--threshold", action="append", default=[], metavar="KEY=VALUE",
                   help=f"override a check threshold; keys: {', '.join(THRESHOLDS)}")
    p.add_argument("--proxy", action="append", default=[], metavar="KEY=VALUE",
                   help=f"override a proxy setting; keys: {', '.join(PROXY)}")
    p.add_argument("--birch-factors", default="0.5,1,2", help="BIRCH threshold in sigma (placeholders)")
    p.add_argument("--birch-branching-factor", type=int, default=50)
    p.add_argument("--no-birch", action="store_true")
    p.add_argument("--no-river-checks", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true", help="write into a non-empty <out-dir>/fidelity")
    return p


def _river_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(C.REPO_ROOT.parent / "river"), "rev-parse", "--short=8", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _write_csv(path: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    if cols is not None:
        df = df.reindex(columns=cols)
    df.to_csv(path, index=False)


def _map(fn, tasks, workers: int):
    if workers <= 1:
        for t in tasks:
            yield fn(t)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, t) for t in tasks]
        for f in as_completed(futs):
            yield f.result()


def main(argv=None) -> Path:
    args = build_parser().parse_args(argv)
    t_start = time.perf_counter()
    shape = StreamShape(h=args.h, n_windows=args.windows, d=args.d, sigma=args.sigma, separation=args.separation,
                        stress_period=args.stress_period)
    shape.validate()
    algos, seeds, scenarios = _csv(args.algos), [int(s) for s in _csv(args.seeds)], _csv(args.scenarios)
    configs = _csv(args.config)
    from .spaces import SPACES

    for bad, what in ((set(algos) - set(SPACES), "algorithm (one with a protocol space)"),
                      (set(configs) - set(CONFIGS), "config"), (set(scenarios) - set(SCENARIOS), "scenario")):
        if bad:
            raise SystemExit(f"unknown {what}: {sorted(bad)}")
    th, px = {**THRESHOLDS, **_kv(args.threshold, THRESHOLDS)}, {**PROXY, **_kv(args.proxy, PROXY)}
    fid = Path(args.out_dir) / "fidelity"
    if fid.exists() and any(fid.iterdir()) and not args.force:
        raise SystemExit(f"{fid} is not empty; pass --force to write into it")
    fid.mkdir(parents=True, exist_ok=True)
    os.environ.update(C.THREAD_PINS)
    shape_d = dataclasses.asdict(shape)

    # cells: skip a scaled_default cell whose parameters are the default's
    tasks, skipped = [], []
    for sc in scenarios:
        for seed in seeds:
            scen = make_scenario(sc, seed, shape)
            w0 = scaled_scenario(scen.points(), scen.y.tolist(), [], shape.h,
                                 C.stream_scaling_of(sc, args.stream_scaling), args.scaling_window)[0][:shape.h]
            for algo in algos:
                base_params = cell_params(algo, "default", shape.h, shape.d, w0)[0]
                for cfg in configs:
                    if (cfg == "scaled_default" and "default" in configs
                            and cell_params(algo, cfg, shape.h, shape.d, w0)[0] == base_params):
                        skipped.append({"scenario": sc, "seed": seed, "config": cfg, "algo": algo,
                                        "reason": "scaled_default leaves every parameter as in default"})
                        continue
                    tasks.append({"scenario": sc, "seed": seed, "config": cfg, "algo": algo, "shape": shape_d,
                                  "model_seed": args.model_seed, "thresholds": th, "proxy": px,
                                  "stream_scaling": args.stream_scaling, "scaling_window": args.scaling_window})
    print(f"fidelity: {len(tasks)} cells ({len(skipped)} scaled_default cells identical to default, not run)", flush=True)
    cells: dict = {}
    windows, checks, cell_rows = [], [], []
    done = 0
    for res in _map(run_cell, tasks, args.workers):
        done += 1
        res["shape"] = shape_d
        cells[res["key"]] = res
        windows += res["windows"]
        checks += res["checks"]
        cell_rows.append(res["cell"])
        c = res["cell"]
        print(f"[{done}/{len(tasks)}] {c['algo']:<22} {c['scenario']:<16} seed {c['seed']} {c['config']:<15} "
              f"{c['seconds']:7.1f}s {'ERROR ' + c['error'] if c['error'] else ''}", flush=True)
    # a skipped scaled_default cell reads as its default cell, under its own config name and marked so; the
    # identity validation is not repeated on such copies
    for sk in skipped:
        src = cells.get((sk["scenario"], sk["seed"], "default", sk["algo"]))
        if src is None:
            continue
        checks += [{**r, "config": sk["config"], "identical_to": "default"} for r in src["checks"]]
        windows += [{**r, "config": sk["config"], "identical_to": "default"} for r in src["windows"]]
    ident, ident_detail = identity_validation(cells, shape, px)

    birch: list[dict] = []
    if not args.no_birch and "BIRCH" in algos:
        btasks = [{"scenario": sc, "seed": seed, "shape": shape_d, "factor": float(f),
                   "branching_factor": args.birch_branching_factor, "variants": BIRCH_VARIANTS}
                  for sc in scenarios for seed in seeds for f in _csv(args.birch_factors)]
        for i, rows in enumerate(_map(run_birch_crosslib, btasks, args.workers), start=1):
            birch += rows
            print(f"[birch {i}/{len(btasks)}] {rows[0]['scenario']} seed {rows[0]['seed']} T/sigma "
                  f"{rows[0]['threshold_factor']}", flush=True)
    indep = independent_implementations()
    refs = verify_refs()
    rchecks = [] if args.no_river_checks else river_checks(algos)

    # files
    srows = []
    for sc in scenarios:
        for seed in seeds:
            scen = make_scenario(sc, seed, shape)
            srows += [{"scenario": sc, "seed": seed, **e, "labels": " ".join(map(str, e["labels"]))} for e in scen.events]
    _write_csv(fid / "scenarios.csv", srows, ["scenario", "seed", "event", "t", "window", "labels"])
    _write_csv(fid / "windows.csv", windows)
    _write_csv(fid / "checks.csv", checks)
    _write_csv(fid / "cells.csv", [{**c, "params": json.dumps(c.get("params"), default=str),
                                    "scaled": json.dumps(c.get("scaled"), default=str)} for c in cell_rows] + skipped)
    _write_csv(fid / "identity_proxy.csv", ident)
    _write_csv(fid / "identity_windows.csv", ident_detail)
    _write_csv(fid / "birch_crosslib.csv", [{**r, "river_params": json.dumps(r.get("river_params"), default=str)}
                                            for r in birch])
    _write_csv(fid / "independent_implementations.csv", indep)
    _write_csv(fid / "audit.csv", [{"algo": e["algo"], "aspect": e["aspect"], "basis": e["basis"], "reading": e["text"],
                                    "refs": " ".join(_show_ref(w, ln) for w, ln, _ in e["refs"])} for e in AUDIT])
    _write_csv(fid / "audit_refs.csv", refs)
    _write_csv(fid / "river_checks.csv", rchecks, ["algo", "check", "outcome"])
    import sklearn

    meta = {"command": "python -m conf_pipeline.fidelity " + " ".join(sys.argv[1:] if argv is None else argv),
            "argv": list(sys.argv[1:] if argv is None else argv), "algos": algos, "seeds": seeds,
            "configs": configs, "scenarios": scenarios, "shape": shape_d, "thresholds": th, "proxy": px,
            "model_seed": args.model_seed, "workers": args.workers, "birch_branching_factor": args.birch_branching_factor,
            "stream_scaling": args.stream_scaling, "scaling_window": args.scaling_window,
            "birch_factors": _csv(args.birch_factors), "river_commit": _river_commit(), "sklearn": sklearn.__version__,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"), "n_cells": len(tasks),
            "n_skipped_identical": len(skipped), "wall_s": time.perf_counter() - t_start,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    from benchmark.reports.environment import capture_environment

    C.write_json(fid / "run.json", {**meta, "environment": capture_environment()})
    report = write_report(fid, meta, windows, checks, cell_rows, ident, birch, indep, refs, rchecks)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    main()
