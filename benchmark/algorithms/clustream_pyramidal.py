"""``CluStream`` with the pyramidal time frame it documents, closing RIVER_BUGS.md GAP-4.

River's ``CluStream`` docstring says the micro-clusters "are stored at
snapshots in time following a pyramidal pattern" so that statistics can be
recalled "from different time horizons". The class does neither (GAP-4): it
keeps the current micro-cluster set only, has no ``alpha``/``l`` parameters,
and ``predict_one`` takes no horizon. Per AGENTS.md rule 1 the vendored tree is
not changed, and per its step 3 (rule 1.3) the frame lives here under its own
name, so river's class stays benchmarkable exactly as shipped.

The method is Aggarwal, Han, Wang and Yu, "A Framework for Clustering Evolving
Data Streams", VLDB 2003: the pyramidal time frame, the horizon lemma, and the
subtraction of snapshot micro-clusters through id lists. Details the paper
leaves open, and how they were settled here:

* **Clock.** One tick per ``snapshot_interval`` calls to ``learn_one``
  (sample weights do not move the clock). Tick
  ``t`` is taken right after the ``t * snapshot_interval``-th learned sample,
  and every time below is a count of learned samples, so ``t_c`` is the number
  of samples learned so far. The stream start (time 0) acts as an implicit,
  never-stored empty snapshot: 0 is divisible by every power of alpha, so it
  has no order.
* **Retention.** A tick's order is the largest ``i`` with ``alpha**i`` dividing
  it, and the snapshot is stored once under that order. The paper keeps, for
  every order ``i``, the last ``alpha**l + 1`` ticks divisible by
  ``alpha**i``; the stored set is the union of those lists. A tick of order
  ``o`` lies in the lists of orders ``0..o``, and the order-``o`` list reaches
  furthest back, so the tick is kept exactly while it is among the last
  ``alpha**l + 1`` multiples of ``alpha**o``. Per exact order at most
  ``alpha**l + 1`` snapshots are kept, at most
  ``(alpha**l + 1) * (floor(log_alpha T) + 1)`` in total after ``T`` ticks.
* **Horizon lemma and its preconditions.** For a horizon ``h`` the snapshot
  used is the last stored one at or before ``t_c - h`` (time 0 if there is
  none). Let ``eps = 1 / alpha**(l - 1)``. If ``h >= snapshot_interval *
  alpha**(l - 1)``, take the largest ``i`` with ``snapshot_interval *
  alpha**(i + l - 1) <= h``: the order-``i`` list reaches back
  ``alpha**l`` steps of ``P = snapshot_interval * alpha**i`` samples, which is
  more than ``h`` (or to the stream start), so it holds a multiple of ``P`` in
  ``(t_c - h - P, t_c - h]`` (or time 0 is), and ``t_c - t_s < h + P <= (1 +
  eps) * h``. On the paper's own clock (queried at a tick, ``h`` a whole
  number of ticks) the lemma then holds for every ``h`` of at least one tick:
  below ``alpha**(l - 1)`` ticks the target tick is still in the order-0 list
  and the horizon is covered exactly. Off that grid and below
  ``snapshot_interval * alpha**(l - 1)`` samples, the order-0 list only gives
  ``t_c - t_s < h + snapshot_interval``, and that is all that can be promised.
* **Id lists.** River reuses dict keys, so ids are tracked per key. A new id
  is issued whenever river creates a micro-cluster: in the initialisation
  branch of ``learn_one``, when the deletion path replaces an expired
  micro-cluster at the same key, and for the new micro-cluster the merge path
  puts at the key it has just absorbed. A merge ``a += b`` sets ``ids[a] =
  ids[a] | ids[b]``. Nothing ever splits, so the live id lists partition the
  live ids.
* **Subtraction.** The snapshot micro-clusters subtracted from a current one
  are those whose id list it contains. By the partition property a snapshot
  micro-cluster is either contained in exactly one current id list or shares
  no id with any (it, or what absorbed it, was deleted). Subtraction uses
  river's own ``stats.Var.__isub__``, the exact inverse of the Chan merge that
  ``CluStreamMicroCluster.__iadd__`` uses, on the shared feature keys only (as
  ``__iadd__`` merges only shared keys). Micro-clusters whose remaining weight
  is ``<= 0`` (relative tolerance 1e-9 against the current weight; unit
  weights make it exact) are dropped, and the tiny negative second moments
  float error can leave are clamped to 0, in ``var_time`` as in ``var_x``. They
  are common when one point remains: ``Var.get()`` then returns 0 whatever
  the second moment is, but a later merge of the horizon micro-cluster would
  carry the negative value along. What remains is every point learned
  after ``t_s`` that a live micro-cluster still holds. Points that went into a
  micro-cluster deleted since are gone from both views: CluStream forgets them.
* **Horizon macro-clusters.** ``predict_one(x, horizon=h)`` maps ``x`` to the
  nearest horizon micro-cluster and labels it with a k-means built exactly
  like river's offline phase: a fresh ``cluster.KMeans(n_clusters=
  n_macro_clusters, seed=seed, **kwargs)`` that learns each horizon centre
  once, in key order, unweighted. The paper's offline phase is a weighted
  k-means over pseudo-points; river's is not. Keeping river's leaves two
  differences between ``predict_one(x)`` and ``predict_one(x, horizon=h)``:
  the micro-cluster set, and the time of the fit. River fits its k-means
  only at ``time_gap`` boundaries, and not before the first boundary after
  initialisation, so until then it answers 0. The horizon k-means is fitted
  on demand, on the micro-clusters of the current timestamp. A horizon
  reaching past the stream start therefore reproduces river's labels exactly
  at every ``time_gap`` boundary once river has fitted, and may differ in
  between. The horizon set and its k-means are cached per (snapshot time,
  timestamp) and the cache is cleared by every ``learn_one``;
  ``micro_clusters_for_horizon`` returns copies, so callers cannot change it.

Not implemented: horizons that end before ``t_c`` (the paper's evolution
analysis compares two such windows), the paper's weighted offline k-means,
its disk-resident snapshot store, and anything about the online phase. The
online phase is river's: ``learn_one`` and ``_maintain_micro_clusters`` call
river's code and only observe what it did, so micro-cluster statistics, the
offline k-means every ``time_gap`` and ``predict_one(x)`` are river's own.
"""

from __future__ import annotations

import copy
import numbers
from typing import NamedTuple

from .adapter import build_river_path

# Ensure the sibling river tree is importable before we touch river.*, exactly
# as benchmark/algorithms/registry.py does.
build_river_path()

from river import cluster, stats  # noqa: E402
from river.cluster import CluStream  # noqa: E402
from river.cluster.clustream import CluStreamMicroCluster  # noqa: E402

#: Relative weight below which a subtracted micro-cluster counts as empty.
_EMPTY = 1e-9


class SnapshotMicroCluster(NamedTuple):
    """A micro-cluster as it was at a snapshot: its id list and copied statistics."""

    ids: frozenset
    var_x: dict
    var_time: stats.Var


class Snapshot(NamedTuple):
    tick: int
    order: int
    micro_clusters: dict  # key -> SnapshotMicroCluster


class HorizonMicroClusters(NamedTuple):
    """What a horizon query returns.

    ``covered`` is ``t_c - snapshot_time``, the horizon actually summarised,
    in learned samples; it is at least ``requested`` unless the stream is
    shorter. ``whole_stream`` is True when no stored snapshot lies at or before
    ``t_c - requested``, so nothing was subtracted.
    """

    micro_clusters: dict  # key -> CluStreamMicroCluster, the caller's own copies
    requested: float
    covered: int
    snapshot_time: int
    whole_stream: bool


def _copy_var(var):
    # An exact, independent copy: a river Var's state is (ddof, mean, _S) and
    # only `mean` is mutable. Same result as copy.deepcopy, about 3x faster.
    out = copy.copy(var)
    out.mean = copy.copy(var.mean)
    return out


def _micro_cluster(var_x, var_time):
    # The constructor's own statistics are replaced before `center` is first
    # read, so only var_x and var_time carry state.
    mc = CluStreamMicroCluster(x={}, w=1.0, timestamp=0)
    mc.var_x = var_x
    mc.var_time = var_time
    return mc


def _copy_micro_cluster(mc):
    return _micro_cluster({f: _copy_var(v) for f, v in mc.var_x.items()}, _copy_var(mc.var_time))


def _check_int(name, value, low):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < low:
        raise ValueError(f"{name} must be an int >= {low}, got {value!r}")


class CluStreamPyramidal(CluStream):
    """River ``CluStream`` plus the paper's pyramidal time frame and horizon queries.

    The online phase is river's, unchanged. On top of it, a snapshot of every
    micro-cluster (id list and copies of ``var_x`` and ``var_time``) is taken
    every ``snapshot_interval`` learned samples and kept or discarded by the
    pyramidal rule; ``micro_clusters_for_horizon(h)`` and
    ``predict_one(x, horizon=h)`` answer for the last ``h`` learned samples.
    Not implemented: horizons that end before the current time, the paper's
    weighted offline k-means (the horizon k-means is river's, unweighted) and
    its disk-resident snapshot store. The module docstring states the
    interpretation of every point the paper leaves open.

    A horizon only differs from the current set when the current set still
    holds points from before it, so what it buys depends on ``time_window``.
    On the tests' two-regime stream (2,000 samples after a switch) with
    river's default ``time_window=1000``, river had already deleted every
    micro-cluster of the old regime on 5 of 5 seeds: the horizon
    micro-clusters were the current ones and both predictions agreed.

    Parameters
    ----------
    n_macro_clusters, max_micro_clusters, micro_cluster_r_factor, time_window, time_gap
        As in ``river.cluster.CluStream``.
    seed, kwargs
        As in ``river.cluster.CluStream``; also used for the horizon k-means.
    alpha
        Snapshot base: order-``i`` snapshots are taken at ticks divisible by
        ``alpha**i``. Integer >= 2.
    l
        ``alpha**l + 1`` snapshots are kept per order, which bounds the
        horizon error by a factor ``1 + 1/alpha**(l-1)``. Integer >= 1.
    snapshot_interval
        Learned samples per clock tick. Integer >= 1.

    The defaults ``alpha=2, l=1, snapshot_interval=100`` are runnable
    placeholders, not tuned values. They were picked for a small store: three
    snapshots per order, at most ``3 * (floor(log2(n / 100)) + 1)`` after ``n``
    samples, each copying ``max_micro_clusters * (n_features + 1)`` Var objects
    of a few hundred bytes. ``l=1`` is also the loosest setting: a horizon of
    at least ``snapshot_interval`` samples may be covered up to twice over,
    and a shorter one off the tick grid by up to ``h + snapshot_interval``
    (module docstring, "Horizon lemma").

    Cost, as orders of magnitude rather than a benchmark: three runs over two
    synthetic Gaussian streams, CPython 3.13, a shared machine, these defaults
    and this framework's registry defaults for CluStream (100 micro-clusters,
    time_gap 100). With 30,000 samples of 54 features there were 12 snapshots
    and 14.7-15.7 MB against 1.8 MB for river's CluStream, and learning was
    6-16% slower. Id lists follow the paper and only deletions shrink them.
    With a ``time_window`` longer than the stream nothing is deleted and they
    hold every id ever issued, growing with every merge: about 24,800 ids
    after 30,000 samples of 10 features, in a 14.2 MB model against 0.3 MB.

    Attributes
    ----------
    snapshots : dict
        tick -> ``Snapshot(tick, order, micro_clusters)``, oldest first.
    micro_cluster_ids : dict
        key -> frozenset of the ids merged into the live micro-cluster at that key.
    n_deletions, n_merges : int
        How often river's maintenance took its deletion path (an expired
        micro-cluster replaced) and its merge path (two merged; with a single
        micro-cluster, river adds it to itself and replaces it).
    """

    def __init__(
        self,
        n_macro_clusters: int = 5,
        max_micro_clusters: int = 100,
        micro_cluster_r_factor: int = 2,
        time_window: int = 1000,
        time_gap: int = 100,
        seed: int | None = None,
        alpha: int = 2,
        l: int = 1,  # noqa: E741 -- the paper's name
        snapshot_interval: int = 100,
        **kwargs,
    ):
        _check_int("alpha", alpha, 2)
        _check_int("l", l, 1)
        _check_int("snapshot_interval", snapshot_interval, 1)
        super().__init__(
            n_macro_clusters=n_macro_clusters,
            max_micro_clusters=max_micro_clusters,
            micro_cluster_r_factor=micro_cluster_r_factor,
            time_window=time_window,
            time_gap=time_gap,
            seed=seed,
            **kwargs,
        )
        self.alpha = alpha
        self.l = l
        self.snapshot_interval = snapshot_interval

        self.snapshots: dict[int, Snapshot] = {}
        self.micro_cluster_ids: dict[int, frozenset] = {}
        self.n_deletions = 0
        self.n_merges = 0
        self._last_id = 0
        self._horizon_cache: dict = {}

    # -- online phase: river's code, observed ---------------------------------
    def _new_ids(self):
        self._last_id += 1
        return frozenset((self._last_id,))

    def learn_one(self, x, w=1.0):
        self._horizon_cache.clear()
        initialized = self._initialized
        super().learn_one(x, w)
        if not initialized:
            # River's initialisation branch adds a micro-cluster under a new key.
            for k in self.micro_clusters.keys() - self.micro_cluster_ids.keys():
                self.micro_cluster_ids[k] = self._new_ids()
        n = self._timestamp + 1
        if n % self.snapshot_interval == 0:
            self._take_snapshot(n // self.snapshot_interval)

    def _maintain_micro_clusters(self, x, w):
        before = {k: (mc, mc.var_time.n) for k, mc in self.micro_clusters.items()}
        super()._maintain_micro_clusters(x=x, w=w)
        # River stores a new micro-cluster under exactly one existing key: the
        # expired one (deletion) or the one just absorbed by `a += b` (merge),
        # in which case `a` is the one key whose object stayed but grew. (An
        # absorbed micro-cluster of zero weight would read as a deletion; the
        # framework learns with unit weights.)
        if self.micro_clusters.keys() != before.keys():
            raise RuntimeError("river CluStream changed its keys during maintenance")
        replaced = [k for k, mc in self.micro_clusters.items() if mc is not before[k][0]]
        grown = [
            k for k, mc in self.micro_clusters.items()
            if mc is before[k][0] and mc.var_time.n != before[k][1]
        ]
        if len(replaced) != 1 or len(grown) > 1:
            raise RuntimeError("river CluStream maintenance no longer has the shape tracked here")
        (new,) = replaced
        if grown:
            (into,) = grown
            ids = self.micro_cluster_ids
            ids[into] = ids[into] | ids[new]
            self.n_merges += 1
        elif before[new][0].var_time.n != before[new][1]:
            # One micro-cluster only: river's merge path finds no pair, adds the
            # micro-cluster to itself and replaces it, so its points are dropped
            # as on deletion although nothing expired.
            self.n_merges += 1
        else:
            self.n_deletions += 1
        self.micro_cluster_ids[new] = self._new_ids()

    # -- pyramidal time frame ------------------------------------------------
    def _take_snapshot(self, tick):
        alpha = int(self.alpha)
        order, rest = 0, tick
        while rest % alpha == 0:
            rest //= alpha
            order += 1
        self.snapshots[tick] = Snapshot(
            tick,
            order,
            {
                k: SnapshotMicroCluster(
                    self.micro_cluster_ids[k],
                    {f: _copy_var(v) for f, v in mc.var_x.items()},
                    _copy_var(mc.var_time),
                )
                for k, mc in self.micro_clusters.items()
            },
        )
        # Keep a tick of order o while it is among the last alpha**l + 1
        # multiples of alpha**o (module docstring, "Retention").
        reach = alpha ** int(self.l)
        expired = [
            t for t, s in self.snapshots.items()
            if tick // alpha**s.order - t // alpha**s.order > reach
        ]
        for t in expired:
            del self.snapshots[t]

    def _snapshot_at_or_before(self, time):
        si = self.snapshot_interval
        for tick in reversed(self.snapshots):
            if tick * si <= time:
                return tick * si, self.snapshots[tick]
        return 0, None

    def _subtract(self, snap):
        out = {}
        for k, mc in self.micro_clusters.items():
            ids = self.micro_cluster_ids[k]
            olds = [] if snap is None else [s for s in snap.micro_clusters.values() if s.ids <= ids]
            if mc.var_time.n - sum(s.var_time.n for s in olds) <= _EMPTY * mc.var_time.n:
                continue
            var_time = _copy_var(mc.var_time)
            var_x = {f: _copy_var(v) for f, v in mc.var_x.items()}
            for s in olds:
                var_time -= s.var_time
                for f, v in var_x.items():
                    old = s.var_x.get(f)
                    if old is not None:
                        v -= old
            for v in (var_time, *var_x.values()):  # module docstring, "Subtraction"
                if v._S < 0.0:
                    v._S = 0.0
            out[k] = _micro_cluster(var_x, var_time)
        return out

    def _horizon(self, h):
        if isinstance(h, bool) or not isinstance(h, numbers.Real) or not h > 0:
            raise ValueError(f"horizon must be a positive number of learned samples, got {h!r}")
        t_c = self._timestamp + 1
        t_s, snap = self._snapshot_at_or_before(t_c - h)
        key = (t_s, self._timestamp)
        if key not in self._horizon_cache:
            mcs = self._subtract(snap)
            kmeans = None
            if mcs:
                kmeans = cluster.KMeans(
                    n_clusters=self.n_macro_clusters, seed=self.seed, **self.kwargs
                )
                for mc in mcs.values():
                    kmeans.learn_one(mc.center)
            self._horizon_cache[key] = (mcs, kmeans)
        mcs, kmeans = self._horizon_cache[key]
        return HorizonMicroClusters(mcs, h, t_c - t_s, t_s, snap is None), kmeans

    def micro_clusters_for_horizon(self, h):
        """Micro-clusters summarising the last ``h`` learned samples, by subtraction.

        Returns a :class:`HorizonMicroClusters`; ``covered`` says how far back
        the summary really reaches, ``whole_stream`` that nothing could be
        subtracted. Its micro-clusters are fresh copies: changing them changes
        neither the model nor later queries.
        """
        hz = self._horizon(h)[0]
        return hz._replace(
            micro_clusters={k: _copy_micro_cluster(mc) for k, mc in hz.micro_clusters.items()}
        )

    def predict_one(self, x, horizon=None):
        if horizon is None:
            return super().predict_one(x)
        hz, kmeans = self._horizon(horizon)
        if kmeans is None:
            return 0  # no micro-cluster yet; river's predict_one answers 0 then too
        mcs = hz.micro_clusters
        nearest = min(mcs, key=lambda k: self._distance(mcs[k].center, x))
        return kmeans.predict_one(mcs[nearest].center)
