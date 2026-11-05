"""Does a label keep its meaning within one window? An audit of the sequential partition.

The tuner scores a member on the partition its predict-before-learn labels make over a window: point
``i`` is labelled by the model as it stood after learning points ``0..i-1``. That partition means
what it seems to only if a label names the same cluster from the first point of the window to the
last. A model that recomputes its macro clusters and numbers them afresh (DenStream on every
``predict_one``, CluStream's offline k-means every ``time_gap`` points, DBSTREAM's and BIRCH's
reclustering after every insert) can hand label 0 to one region early in the window and to another
later, and the sequential partition then joins points the model never grouped.

``audit_window`` replays a window on a copy of a member's adapter, exactly as the tuner does
(``ClustererAdapter.process_one``: preprocess, predict, learn), and after every ``every`` points
predicts a probe set (the window itself by default) with a further copy, so neither the replay nor
the caller's adapter is touched by the probes. Two successive probe labellings that differ are one
event, read without relabelling anything:

- ``rename``: the same partition of the probe set (ARI 1 between the two labellings over the points
  both label) under other raw labels: the model's clusters are unchanged and their names moved;
- ``partition_change``: another partition (ARI below 1): the model's clusters themselves changed,
  which the sequential partition is meant to record.

A rename is the case a sequential silhouette or ARI cannot see: it splits one cluster into two labels
or, worse, gives one label to two clusters. ``rename_reuse_events`` counts the renames under which some
raw label passed from one cluster to another (a label present before and after that names a different
set of probe points), the case that merges clusters in the sequential partition. A partition change can
carry a renumbering too (DenStream drops a p-micro-cluster and every later macro cluster's number moves
down): for every event the audit reads the share of probe points keeping their raw label and the share
keeping it under the best one-to-one matching of the two labellings (``matched``, Hungarian on the
contingency table). The matching is read, never applied: ``renumbered_points`` sums, over the events,
the probe points whose cluster survived under another name. Low ``stability`` (``signals.stability``)
alone says neither: it mixes renames with real change, and is read over the whole window at once.

``predict_one`` may also change the model (RIVER_BUGS.md BUG-7). ``n_predict_mutations`` counts the
predictions after which the model labels the probe set otherwise than just before them (read on copies,
before the point is learned): a change of behaviour, not of memoised state. ``n_state_writes`` counts
the predictions after which the model's pickled state differs (``state_digest``, without the
attributes ``CACHE_ATTRS`` names, which only memoise the last reclustering): lazily cached centres
count there too, so it is an upper bound. A model that cannot be pickled reads None.

The audit is a diagnostic run beside a campaign, never inside one: it costs a probe pass per step.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Sequence

#: Attributes that only cache a model's last reclustering: rewriting them on ``predict_one`` is not a
#: change of the model's state. Keyed by River class name (the nearest class of the model's MRO).
CACHE_ATTRS: dict[str, tuple[str, ...]] = {
    "DenStream": ("clusters", "n_clusters"),
    "DBSTREAM": ("_n_clusters", "_clusters", "_centers", "clustering_is_up_to_date"),
    "BIRCH": ("_clusters", "_n_clusters", "_clustering_is_up_to_date"),
}
_EPS = 1e-12


def _cache_attrs(model) -> tuple[str, ...]:
    for cls in type(model).__mro__:
        if cls.__name__ in CACHE_ATTRS:
            return CACHE_ATTRS[cls.__name__]
    return ()


def state_digest(model) -> str | None:
    """A hash of ``model``'s state without its reclustering caches; None when it cannot be pickled."""
    skip = set(_cache_attrs(model))
    state = {k: v for k, v in vars(model).items() if k not in skip}
    try:
        blob = pickle.dumps(state, protocol=4)
    except Exception:
        return None
    return hashlib.sha256(blob).hexdigest()


def _ari(a: Sequence, b: Sequence) -> float | None:
    from sklearn.metrics import adjusted_rand_score

    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(both) < 2:
        return None
    ids_a, ids_b = {}, {}
    return float(adjusted_rand_score([ids_a.setdefault(x, len(ids_a)) for x, _ in both],
                                     [ids_b.setdefault(y, len(ids_b)) for _, y in both]))


def _members(labels: Sequence) -> dict:
    out: dict = {}
    for i, lab in enumerate(labels):
        if lab is not None:
            out.setdefault(lab, set()).add(i)
    return out


def _reuses_a_label(before: Sequence, after: Sequence) -> bool:
    """Some raw label present on both sides names another set of points after than before."""
    a, b = _members(before), _members(after)
    return any(a[lab] != b[lab] for lab in set(a) & set(b))


def label_handoffs(before: Sequence, after: Sequence) -> list[tuple]:
    """The labels that pass, between two labellings of one point set, to other points while the points
    they named stay together under another label: ``(label, old points, new points, the old points' new
    label)`` per such label. Read off the raw labels; nothing is matched.

    A label ``L`` hands off when the points it names after share none with those it named before, and
    the points it named before all carry one other label after. A merge (``L`` keeps its points and gains
    others), a split (``L`` keeps some of them) or a fresh label is never one. Joined over the two
    times, as a sequential partition joins them, ``L`` then puts together points the model kept apart
    at both times: that is what a renaming or a reused label does to it. A cluster that moves whole
    onto other points between two probes would read the same, so for a model whose labels name a
    persistent cluster (SOStream, RepStream, STREAMKMeans) a handoff is not, by itself, a renaming."""
    a, b = _members(before), _members(after)
    out = []
    for lab in sorted(set(a) & set(b), key=repr):
        if a[lab] & b[lab]:
            continue
        now = {after[i] for i in a[lab]}
        if len(now) == 1 and None not in now:
            out.append((lab, sorted(a[lab]), sorted(b[lab]), next(iter(now))))
    return out


def _agreements(a: Sequence, b: Sequence) -> tuple[float | None, float | None]:
    """``(raw, matched)``: over the points both label, the share with the same raw label, and the share
    with the same label once the labels of ``b`` are matched one-to-one to those of ``a`` (Hungarian, on
    the contingency table) -- read only, nothing is relabelled."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not both:
        return None, None
    raw = sum(x == y for x, y in both) / len(both)
    ia, ib = {}, {}
    table = {}
    for x, y in both:
        key = (ia.setdefault(x, len(ia)), ib.setdefault(y, len(ib)))
        table[key] = table.get(key, 0) + 1
    m = np.zeros((len(ia), len(ib)))
    for (i, j), n in table.items():
        m[i, j] = n
    rows, cols = linear_sum_assignment(-m)
    return raw, float(m[rows, cols].sum() / len(both))


def _predict_all(adapter, points: Sequence[dict]) -> list:
    probe = copy.deepcopy(adapter)
    out = []
    for x in points:
        x = dict(x)
        out.append(probe.predict_one(probe.scaler.transform_one(x) if probe.scaler is not None else x))
    return out


@dataclass
class WindowAudit:
    """What ``audit_window`` found on one window (see the module docstring)."""

    n_points: int
    n_probes: int
    n_rename_events: int = 0
    n_rename_reuse_events: int = 0
    n_partition_change_events: int = 0
    #: Events (renames and partition changes) under which some points kept their cluster under another
    #: name, and how many probe points that was over all events (see the module docstring).
    n_renumbering_events: int = 0
    renumbered_points: int = 0
    #: Events (of either kind) with a label handoff (``label_handoffs``: read off the raw labels, no
    #: matching), and over all of them the probe points a handed-off label took.
    n_label_handoff_events: int = 0
    label_handoff_points: int = 0
    #: Predictions after which the model labels the probe set otherwise (a change of behaviour).
    n_predict_mutations: int = 0
    #: Predictions after which the model's pickled state (caches aside) differs; None when not hashable.
    n_state_writes: int | None = 0
    #: Window positions (0-based, the point just learned) of every event, by kind.
    events: dict = field(default_factory=lambda: {"rename": [], "partition_change": []})
    #: Share of the window's points predicted after the first rename (their labels may name other
    #: clusters than the points before it): None when there was no rename.
    share_after_first_rename: float | None = None
    #: The sequential labels against the end-of-window snapshot of the same points.
    ari_sequential_vs_snapshot: float | None = None
    raw_agreement_sequential_vs_snapshot: float | None = None
    sequential: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("sequential", "events")}
        d["first_rename_at"] = self.events["rename"][0] if self.events["rename"] else None
        d["first_partition_change_at"] = (self.events["partition_change"][0]
                                          if self.events["partition_change"] else None)
        return d


def audit_window(adapter, xs: Sequence[dict], probe: Sequence[dict] | None = None, every: int = 1,
                 check_mutation: bool = True) -> WindowAudit:
    """Replay ``xs`` on a copy of ``adapter`` (predict then learn, as the tuner does) and read, every
    ``every`` points, how the labels of ``probe`` (``xs`` when None) moved; the caller's adapter is
    left as it was. See the module docstring for what each count means."""
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    probe = list(xs if probe is None else probe)
    live = copy.deepcopy(adapter)
    out = WindowAudit(n_points=len(xs), n_probes=0)
    prev = _predict_all(live, probe)
    first_rename = None
    for i, x in enumerate(xs):
        xs_i = live._scale(dict(x))
        live.last_x_scaled = xs_i
        probing = check_mutation and (i % every == 0)
        before = state_digest(live.model) if probing else None
        label = live.predict_one(xs_i)
        if probing:
            after = state_digest(live.model)
            if before is None or after is None:
                out.n_state_writes = None
            elif out.n_state_writes is not None and before != after:
                out.n_state_writes += 1
            mid = _predict_all(live, probe)
            if mid != prev:
                out.n_predict_mutations += 1
                prev = mid  # the change the prediction made is not the learning step's
        out.sequential.append(label)
        live.learn_one(xs_i)
        live._n_seen += 1
        if (i + 1) % every and i + 1 != len(xs):
            continue
        snap = _predict_all(live, probe)
        out.n_probes += 1
        if snap != prev:
            ari = _ari(prev, snap)
            raw, matched = _agreements(prev, snap)
            if raw is not None and matched - raw > _EPS:
                out.n_renumbering_events += 1
                out.renumbered_points += int(round((matched - raw) * sum(
                    1 for a, b in zip(prev, snap) if a is not None and b is not None)))
            handoffs = label_handoffs(prev, snap)
            if handoffs:
                out.n_label_handoff_events += 1
                out.label_handoff_points += sum(len(new) for _, _, new, _ in handoffs)
            if ari is not None and ari >= 1.0 - _EPS:
                out.n_rename_events += 1
                out.events["rename"].append(i)
                out.n_rename_reuse_events += _reuses_a_label(prev, snap)
                if first_rename is None:
                    first_rename = i
            else:
                out.n_partition_change_events += 1
                out.events["partition_change"].append(i)
        prev = snap
    if first_rename is not None and len(xs):
        out.share_after_first_rename = (len(xs) - 1 - first_rename) / len(xs)
    # the last probe pass ran after the last point was learned: the end-of-window snapshot
    final = prev if probe == list(xs) else _predict_all(live, xs)
    out.ari_sequential_vs_snapshot = _ari(out.sequential, final)
    both = [(a, b) for a, b in zip(out.sequential, final) if a is not None and b is not None]
    out.raw_agreement_sequential_vs_snapshot = (sum(a == b for a, b in both) / len(both)) if both else None
    return out
