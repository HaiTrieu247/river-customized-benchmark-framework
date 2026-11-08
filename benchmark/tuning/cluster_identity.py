"""Cluster identity within a window: what a label means across the points it was given, read without
evaluation labels.

The tuner and the lagged ARI read the *sequential* partition of a window: point ``i`` carries the label
the model gave it after learning points ``0..i-1``. That partition joins two points under one label only
if the label named the same cluster at both times. ``partition_audit`` counts renames and label handoffs
between probes; this module adds four readings beside it.

**Transitions** (``transitions``). Two labellings of the same points (the model at two times) are read as
the cluster-transition taxonomy of MONIC (Spiliopoulou et al., KDD 2006): with ``N[p, q]`` the points of
before-cluster ``p`` that after-cluster ``q`` holds, ``p`` *splits* when at least two after-clusters each
hold ``tau_split`` of ``p`` and are made at least ``tau`` of ``p`` (its children), together holding ``tau``
of it; otherwise it *survives* into the after-cluster holding ``tau`` of it, or is *absorbed* into it when
another before-cluster also has ``tau`` of its points there (a merge); otherwise it *disappears*. An
after-cluster no before-cluster reaches *emerges*. Every before-cluster passes its identity to one heir
(where it survives or is absorbed; of its children, the one under its label, else the largest), and the
label outcome is read on the
after-clusters: *kept* (its label was the label of one of its heirs), *renamed* (it has heirs, none under
its label), *offspring* (a split child that is no heir) and *emerged*; a label present on both sides
whose after-cluster does not descend from its before-cluster is *reused*: under it the sequential
partition joins clusters the model kept apart at both times. ``kind`` is ``none`` (same labels),
``rename_only`` (same partition, other labels: a recomputation that only renamed) or ``structural``.
``tau`` and ``tau_split`` are runnable placeholders (MONIC leaves them to the user), not recommendations.

**Pair consistency** (``pair_consistency``). Are historical predictions within a window comparable? For
the pairs ``i < j`` of a window, binned by lag ``j - i``: the share of pairs under one sequential label
that one reference labelling (the end-of-window snapshot, or the true classes) also puts together
(``precision``), and the share of pairs the reference puts together that carry one sequential label
(``recall``). A label that names one cluster over the window keeps both flat in the lag; a reused label
drops precision at long lags, a renamed cluster drops recall there.

**Snapshot** (``snapshot_labels``). A coherent partition: every point of a window labelled by one model
state, on a copy of the adapter. River's own DenStream changes its state when it predicts (RIVER_BUGS.md
BUG-7, ``PREDICT_MUTATES``); for such a model ``isolate`` restores the state before every point, so the
snapshot is the partition of one state and not of a chain of predictions.

**Identity tracking** (``IdentityTracker``). A way to keep the sequential reading with cluster identity
held across the model's renamings, without evaluation labels. Every identity keeps a few *anchors*, the
last points it was given; after every learned point the anchors are predicted again by the model and the
transition between the identities and the model's new raw labels is read as above. A raw label then
carries the identity of its heir: the heir that held this raw label before when there is one (no evidence
says the model's name stopped naming that cluster), else the heir with the most anchors there, then the
older identity; an emerged cluster or a split child that is no heir gets a new
identity, and an identity is never given to a second cluster. The tracked label of a point is the
identity its raw label carried when it was predicted. What two points under one tracked label mean: the
cluster of the earlier one passed its identity, survival by survival or absorption, to the cluster of
the later one. The tracker reads only the model's predictions on points of the stream.

``replay_window`` replays a window predict-before-learn on the adapter it is given, as the tuner does
(``ClustererAdapter.process_one``), and returns the sequential labels, the tracked labels, the snapshots
of the model at the start (``pre``: the state the window is predicted from, out of sample) and at the
end (``post``: once it has learned the window) and, every ``probe_every`` points, the transition of the
window's snapshot since the previous probe with the model's micro-cluster births and deaths
(``micro_objects``: the objects by identity, where the model holds them). The tracker
predicts on the live model only when a prediction cannot change it, otherwise on a copy, so it never
changes what the model predicts.
"""

from __future__ import annotations

import copy
import pickle
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

#: Placeholders (MONIC's tau and tau_split), not recommendations.
TAU = 0.5
TAU_SPLIT = 0.25
#: Lag bins of ``pair_consistency``: ``[1, 10), [10, 100), [100, 1000), [1000, inf)``.
LAG_EDGES: tuple[int, ...] = (1, 10, 100, 1000)
#: Whether a model's ``predict_one`` can change its state, by the nearest class of its MRO.
PREDICT_MUTATES: dict[str, bool] = {"DenStreamPatched": False, "DenStream": True}
#: Containers of a model's micro-level objects, read in this order (``micro_objects``).
MICRO_ATTRS: tuple[str, ...] = ("micro_clusters", "_micro_clusters", "_vertices", "p_micro_clusters",
                                "o_micro_clusters")
#: Models whose clusters are their only objects, by the nearest class of the MRO.
CLUSTER_ATTRS: dict[str, tuple[str, ...]] = {"HPStream": ("_clusters",)}


def _key(v) -> tuple:
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return (0, float(v), "")
    return (1, 0.0, repr(v))


def _nearest(model, table: dict):
    for cls in type(model).__mro__:
        if cls.__name__ in table:
            return table[cls.__name__]
    return None


def predict_mutates(model) -> bool:
    """Whether ``model``'s ``predict_one`` is known to change its state (``PREDICT_MUTATES``)."""
    return bool(_nearest(model, PREDICT_MUTATES))


# --- transitions between two labellings -----------------------------------------------------------

@dataclass
class Lineage:
    """How the clusters of one labelling became those of another (see the module docstring)."""

    N: dict
    size_before: Counter
    size_after: Counter
    status: dict          # before-cluster -> survived | absorbed | split | disappeared
    descendants: dict     # before-cluster -> set of after-clusters
    heir: dict            # before-cluster -> after-cluster (absent for a disappeared one)
    heirs: dict           # after-cluster -> list of before-clusters

    @property
    def before(self) -> list:
        return sorted(self.size_before, key=_key)

    @property
    def after(self) -> list:
        return sorted(self.size_after, key=_key)


def lineage(before: Sequence, after: Sequence, tau: float = TAU, tau_split: float = TAU_SPLIT,
            prefer: Callable | None = None) -> Lineage:
    """The lineage of ``after``'s clusters from ``before``'s over the points both label. Of a split's
    children, the heir is the one ``prefer(p, q)`` names (the child that kept the parent's name), else the
    largest."""
    N: dict = defaultdict(Counter)
    sb, sa = Counter(), Counter()
    for p, q in zip(before, after):
        if p is None or q is None:
            continue
        N[p][q] += 1
        sb[p] += 1
        sa[q] += 1
    share = {p: {q: n / sb[p] for q, n in row.items()} for p, row in N.items()}
    status, desc, heir = {}, {}, {}
    order = sorted(N, key=_key)
    for p in order:
        row = N[p]
        main = min(row, key=lambda q: (-row[q], _key(q)))
        kids = [q for q in row if share[p][q] >= tau_split and row[q] / sa[q] >= tau]
        if len(kids) >= 2 and sum(row[q] for q in kids) / sb[p] >= tau:
            status[p], desc[p] = "split", set(kids)
            heir[p] = min(kids, key=lambda q: (0 if (prefer and prefer(p, q)) else 1, -row[q], _key(q)))
        elif share[p][main] >= tau:
            shared = any(p2 != p and share[p2].get(main, 0.0) >= tau for p2 in order)
            status[p], desc[p], heir[p] = ("absorbed" if shared else "survived"), {main}, main
        else:
            status[p], desc[p] = "disappeared", set()
    heirs: dict = defaultdict(list)
    for p, q in heir.items():
        heirs[q].append(p)
    return Lineage(dict(N), sb, sa, status, desc, heir, dict(heirs))


@dataclass
class Transition:
    """``transitions`` on one pair of labellings: counts over the points both label."""

    n_points: int
    kind: str
    survived: int = 0
    absorbed: int = 0
    absorb_events: int = 0
    split: int = 0
    disappeared: int = 0
    emerged: int = 0
    offspring: int = 0
    kept: int = 0
    renamed: int = 0
    reused_labels: int = 0
    reused_on_emergence: int = 0
    reused_on_offspring: int = 0
    fresh_labels: int = 0
    retired_labels: int = 0
    n_labels_before: int = 0
    n_labels_after: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def transitions(before: Sequence, after: Sequence, tau: float = TAU, tau_split: float = TAU_SPLIT) -> Transition:
    """The transition from ``before`` to ``after`` (two labellings of the same points, raw labels as
    cluster names): see the module docstring."""
    both = [(p, q) for p, q in zip(before, after) if p is not None and q is not None]
    b = [p for p, _ in both]
    a = [q for _, q in both]
    lin = lineage(b, a, tau, tau_split, prefer=lambda p, q: p == q)
    if b == a:
        kind = "none"
    else:
        fwd, bwd = {}, {}
        bijective = all(fwd.setdefault(p, q) == q and bwd.setdefault(q, p) == p for p, q in both)
        kind = "rename_only" if bijective else "structural"
    t = Transition(n_points=len(both), kind=kind, n_labels_before=len(lin.size_before),
                   n_labels_after=len(lin.size_after))
    for st in lin.status.values():
        setattr(t, st, getattr(t, st) + 1)
    t.absorb_events = sum(1 for hs in lin.heirs.values() if len(hs) >= 2)
    old, new = set(lin.size_before), set(lin.size_after)
    sources = defaultdict(set)
    for p, ds in lin.descendants.items():
        for q in ds:
            sources[q].add(p)
    for q in lin.after:
        hs = lin.heirs.get(q)
        if hs:
            if q in hs:
                t.kept += 1
            else:
                t.renamed += 1
        elif sources.get(q):
            t.offspring += 1
            t.reused_on_offspring += q in old
        else:
            t.emerged += 1
            t.reused_on_emergence += q in old
    t.reused_labels = sum(1 for lab in old & new if lab not in lin.descendants.get(lab, set()))
    t.fresh_labels = len(new - old)
    t.retired_labels = len(old - new)
    return t


# --- pair consistency by lag ----------------------------------------------------------------------

def _codes(labels: Sequence, keep: np.ndarray) -> np.ndarray:
    ids: dict = {}
    return np.asarray([ids.setdefault(labels[i], len(ids)) for i in np.flatnonzero(keep)], dtype=np.int64)


def pair_consistency(labels: Sequence, reference: Sequence, edges: Sequence[int] = LAG_EDGES) -> list[dict]:
    """Per lag bin ``[edges[k], edges[k+1])`` (the last open), over the pairs ``i < j`` both labellings
    label: ``n_pairs``, ``n_same_labels`` (one label in ``labels``), ``n_same_reference``, ``n_both``,
    ``precision = n_both / n_same_labels`` and ``recall = n_both / n_same_reference`` (None on an empty
    denominator). Positions are the points' order in the window."""
    n = len(labels)
    if len(reference) != n:
        raise ValueError(f"labels and reference differ in length: {n} vs {len(reference)}")
    keep = np.asarray([a is not None and b is not None for a, b in zip(labels, reference)], dtype=bool)
    pos = np.flatnonzero(keep)
    a, b = _codes(labels, keep), _codes(reference, keep)
    bounds = list(edges) + [np.inf]
    rows = []
    if len(pos) < 2:
        same_a = same_b = lag = np.empty(0)
    else:
        iu, ju = np.triu_indices(len(pos), k=1)
        lag = pos[ju] - pos[iu]
        same_a = a[iu] == a[ju]
        same_b = b[iu] == b[ju]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (lag >= lo) & (lag < hi) if len(lag) else np.zeros(0, dtype=bool)
        n_pairs = int(m.sum())
        sa_, sb_ = same_a[m] if n_pairs else np.zeros(0, bool), same_b[m] if n_pairs else np.zeros(0, bool)
        n_a, n_b, n_ab = int(sa_.sum()), int(sb_.sum()), int((sa_ & sb_).sum())
        rows.append({"lag_lo": int(lo), "lag_hi": None if hi == np.inf else int(hi), "n_pairs": n_pairs,
                     "n_same_labels": n_a, "n_same_reference": n_b, "n_both": n_ab,
                     "precision": (n_ab / n_a) if n_a else None, "recall": (n_ab / n_b) if n_b else None})
    return rows


# --- predicting without touching the adapter --------------------------------------------------------

def quiet_predict(adapter, x: dict):
    """``adapter.predict_one`` on a point not yet preprocessed, with the adapter's conventions (fallback
    label 0, the sentinel kept or dropped, None when the model has no per-sample partition) but none of
    its counters, and without teaching the preprocessor anything."""
    from ..algorithms.adapter import sentinel_labels

    if not adapter.partitions_samples:
        return None
    x = dict(x)
    if adapter.scaler is not None:
        x = adapter.scaler.transform_one(x)
    try:
        label = adapter.model.predict_one(x)
    except NotImplementedError:
        return None
    except (ValueError, ZeroDivisionError, KeyError):
        return 0
    if label in sentinel_labels(adapter.model) and adapter.drop_sentinel_predictions:
        return None
    return label


def _frozen(adapter):
    """A function restoring a pristine copy of ``adapter``: from a pickle when it can be pickled."""
    try:
        blob = pickle.dumps(adapter, protocol=pickle.HIGHEST_PROTOCOL)
        return lambda: pickle.loads(blob)
    except Exception:
        frozen = copy.deepcopy(adapter)
        return lambda: copy.deepcopy(frozen)


def snapshot_labels(adapter, xs: Sequence[dict], isolate: bool | None = None) -> list:
    """Every point of ``xs`` labelled by ``adapter``'s model as it stands, on a copy: the adapter is not
    touched. ``isolate`` (None: when ``predict_mutates``) restores the state before every point."""
    if isolate is None:
        isolate = predict_mutates(adapter.model)
    if isolate:
        restore = _frozen(adapter)
        return [quiet_predict(restore(), x) for x in xs]
    probe = copy.deepcopy(adapter)
    return [quiet_predict(probe, x) for x in xs]


def micro_objects(model) -> list | None:
    """The model's micro-level objects (``MICRO_ATTRS``, ``CLUSTER_ATTRS``), or None when the model exposes
    none or builds them anew on every read. Holding the list keeps their ``id`` from being reused."""
    names = _nearest(model, CLUSTER_ATTRS) or MICRO_ATTRS

    def read():
        found, objs = False, []
        for name in names:
            try:
                obj = getattr(model, name, None)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                found = True
                objs.extend(obj.values())
            elif isinstance(obj, (list, tuple)):
                found = True
                objs.extend(obj)
        return objs if found else None

    first = read()
    if first is None:
        return None
    second = read()
    return first if {id(o) for o in second} == {id(o) for o in first} else None


# --- identity tracking ------------------------------------------------------------------------------

class IdentityTracker:
    """Persistent cluster identities over a model's raw labels, without evaluation labels (see the
    module docstring). ``anchors_per_id`` and ``max_anchors`` (runnable placeholders) bound the points
    each refresh predicts: the identities given a point most recently are anchored first."""

    def __init__(self, anchors_per_id: int = 5, max_anchors: int = 400, tau: float = TAU,
                 tau_split: float = TAU_SPLIT):
        if anchors_per_id < 1 or max_anchors < 1:
            raise ValueError("anchors_per_id and max_anchors must be at least 1")
        self.anchors_per_id, self.max_anchors = anchors_per_id, max_anchors
        self.tau, self.tau_split = tau, tau_split
        self.raw_to_id: dict = {}
        self.anchors: dict[int, deque] = {}
        self.last_used: dict[int, int] = {}
        self.retired: set[int] = set()
        self._next = 0
        self._t = 0
        self.stats = Counter()

    def _new_id(self) -> int:
        i = self._next
        self._next += 1
        self.stats["ids_created"] += 1
        return i

    def assign(self, raw, x: dict):
        """The identity of a point the model just labelled ``raw``; the point becomes an anchor of it."""
        if raw is None:
            return None
        ident = self.raw_to_id.get(raw)
        if ident is None:
            ident = self.raw_to_id[raw] = self._new_id()
        self.anchors.setdefault(ident, deque(maxlen=self.anchors_per_id)).append((self._t, dict(x)))
        self.last_used[ident] = self._t
        self._t += 1
        return ident

    def _anchor_set(self) -> list[tuple[int, int, dict]]:
        out = []
        for ident in sorted(self.anchors, key=lambda i: (-self.last_used.get(i, -1), i)):
            q = self.anchors[ident]
            if len(out) + len(q) > self.max_anchors:
                break
            out.extend((ident, t, x) for t, x in q)
        return out

    def refresh(self, predict: Callable[[list[dict]], list]) -> None:
        """Read the model again on the anchors (``predict`` labels a list of points with the model as it
        stands) and carry every identity to its heir."""
        anchored = self._anchor_set()
        if not anchored:
            return
        self.stats["refreshes"] += 1
        self.stats["anchor_predictions"] += len(anchored)
        raws = predict([x for _, _, x in anchored])
        idents = [i for i, _, _ in anchored]
        held = defaultdict(set)
        for raw, i in self.raw_to_id.items():
            held[i].add(raw)
        lin = lineage(idents, raws, self.tau, self.tau_split, prefer=lambda i, q: q in held.get(i, ()))
        new_map: dict = {}
        for q in lin.after:
            hs = lin.heirs.get(q)
            if hs:
                new_map[q] = min(hs, key=lambda i: (0 if q in held.get(i, ()) else 1, -lin.N[i][q], i))
            else:
                new_map[q] = self._new_id()
        alive = set(new_map.values())
        gone = {i for i in lin.size_before if i not in alive}
        for raw, ident in self.raw_to_id.items():
            if raw in new_map:
                if new_map[raw] != ident:
                    self.stats["remaps"] += 1
            elif ident not in alive and ident not in gone and ident not in self.retired:
                new_map[raw] = ident  # not observed on the anchors: no evidence it changed
        for i in gone:
            self.retired.add(i)
            self.anchors.pop(i, None)
            self.last_used.pop(i, None)
            self.stats["ids_retired"] += 1
        moved: dict = defaultdict(list)
        for (i, t, x), q in zip(anchored, raws):
            if q is not None:
                moved[new_map[q]].append((t, x))
        for i in set(lin.size_before):
            self.anchors.pop(i, None)
        for ident, pts in moved.items():
            keep = deque(sorted(pts, key=lambda tx: tx[0]), maxlen=self.anchors_per_id)
            old = self.anchors.get(ident)
            if old:
                keep = deque(sorted(list(old) + list(keep), key=lambda tx: tx[0]), maxlen=self.anchors_per_id)
            self.anchors[ident] = keep
            self.last_used[ident] = max(self.last_used.get(ident, -1), keep[-1][0])
        self.raw_to_id = new_map


def tracker_predict(adapter) -> Callable[[list[dict]], list]:
    """The ``predict`` a tracker refreshes with: on the live model when a prediction cannot change it,
    otherwise on one copy per refresh, thrown away after it. On that copy the anchors are predicted in
    turn, so for River's DenStream the later anchors of a refresh read a state the earlier predictions
    wrote to (BUG-7); the snapshots, which are scored, restore the state before every point instead."""
    if predict_mutates(adapter.model):
        def on_copy(xs):
            probe = copy.deepcopy(adapter)
            return [quiet_predict(probe, x) for x in xs]
        return on_copy
    return lambda xs: [quiet_predict(adapter, x) for x in xs]


# --- one window ---------------------------------------------------------------------------------------

@dataclass
class WindowReplay:
    sequential: list = field(default_factory=list)
    tracked: list | None = None
    pre: list = field(default_factory=list)
    post: list = field(default_factory=list)
    intervals: list = field(default_factory=list)


def replay_window(adapter, xs: Sequence[dict], tracker: IdentityTracker | None = None,
                  probe_every: int | None = None, isolate: bool | None = None,
                  tau: float = TAU, tau_split: float = TAU_SPLIT, refresh_every: int = 1) -> WindowReplay:
    """Run ``xs`` through ``adapter`` predict-before-learn (it learns them, as the tuner's member does)
    and read the window (see the module docstring). The tracker is refreshed after every
    ``refresh_every`` learned points (1: after every one, the exact reading)."""
    if probe_every is not None and probe_every < 1:
        raise ValueError(f"probe_every must be >= 1, got {probe_every}")
    if refresh_every < 1:
        raise ValueError(f"refresh_every must be >= 1, got {refresh_every}")
    out = WindowReplay(tracked=[] if tracker is not None else None)
    out.pre = snapshot_labels(adapter, xs, isolate)
    prev, prev_t = out.pre, -1
    micro_prev = micro_objects(adapter.model)
    n_micro_prev, n_macro_prev = _counts(adapter)
    refresh = tracker_predict(adapter) if tracker is not None else None
    for i, x in enumerate(xs):
        xs_i = adapter._scale(dict(x))
        adapter.last_x_scaled = xs_i
        raw = adapter.predict_one(xs_i)
        out.sequential.append(raw)
        if tracker is not None:
            out.tracked.append(tracker.assign(raw, x))
        adapter.learn_one(xs_i)
        adapter._n_seen += 1
        if tracker is not None and ((i + 1) % refresh_every == 0 or i + 1 == len(xs)):
            tracker.refresh(refresh)
        if probe_every is not None and ((i + 1) % probe_every == 0 or i + 1 == len(xs)):
            snap = snapshot_labels(adapter, xs, isolate)
            micro = micro_objects(adapter.model)
            n_micro, n_macro = _counts(adapter)
            row = {"t": i, "t_prev": prev_t, **transitions(prev, snap, tau, tau_split).to_dict(),
                   "n_micro_before": n_micro_prev, "n_micro_after": n_micro,
                   "n_macro_before": n_macro_prev, "n_macro_after": n_macro,
                   **_born_died(micro_prev, micro)}
            out.intervals.append(row)
            prev, prev_t, micro_prev, n_micro_prev, n_macro_prev = snap, i, micro, n_micro, n_macro
    out.post = prev if probe_every is not None and xs else snapshot_labels(adapter, xs, isolate)
    return out


def _born_died(before: list | None, after: list | None) -> dict:
    if before is None or after is None:
        return {"micro_born": None, "micro_died": None}
    a, b = {id(o) for o in before}, {id(o) for o in after}
    return {"micro_born": len(b - a), "micro_died": len(a - b)}


def _counts(adapter) -> tuple:
    try:
        return adapter.n_micro_clusters(), adapter.n_macro_clusters()
    except Exception:
        return None, None
