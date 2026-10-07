"""``DenStreamPatched`` with an exact, incremental offline phase (RIVER_BUGS.md GAP-3).

Per AGENTS.md rule 1 the vendored ``river`` tree is not modified; per rule 3
this is registered under its own name. It addresses **GAP-3**:
``DenStream.predict_one`` re-labels every p-micro-cluster from scratch on every
call. Each p-micro-cluster runs ``_query_neighbor``, an O(P) scan, so one
prediction costs P*(P-1) reachability tests even when nothing changed since
the previous one, and test-then-train evaluation pays that once per sample.

``DenStreamPatchedFast`` returns what ``DenStreamPatched.predict_one`` returns,
leaves ``n_clusters`` and ``clusters`` equal to what it leaves, at every call
and under any interleaving of ``learn_one`` and ``predict_one``. It is exact
with ``DenStreamPatched`` (BUG-7 and BUG-8 fixed), **not** with upstream
``DenStream``: there BUG-7 mutates live micro-clusters inside the prediction
and BUG-8 makes the labelling something other than connected components, so
caching it would change its results. For the same reason this class says
nothing about the speed of upstream ``DenStream``.

Why caching is exact, for the vendored river (``denstream.py:242-259``):

* ``_is_directly_density_reachable(p, q)`` is ``w(p) > mu and w(q) > mu and
  d < 2*epsilon and d <= r(p) + r(q)`` with ``d`` the distance between the
  centres. Weights decay with ``self.timestamp`` and are recomputed on every
  call. Centre and ``calc_radius`` ignore the timestamp, so the two distance
  conditions -- the geometric edge -- change only when an endpoint's sums do.
* Sums change only through ``insert`` and ``merge``, both of which increase
  ``N``; a row is recomputed when its object is new or its ``(N,
  last_edit_time)`` moved. Objects are tracked by identity, never by dict key:
  ``denstream.py:223`` keys a promoted micro-cluster by
  ``len(p_micro_clusters)``, which after a deletion can be the key of a live
  one, and that one is replaced.
* ``euclidean_distance_dict`` sums in its first argument's key order, so the
  two directions can differ in the last bit when the centres order their keys
  differently. Each direction is evaluated in the argument order upstream uses
  and stored separately.
* Seed expansion with BUG-8 fixed labels, from each still-unlabelled
  p-micro-cluster in iteration order, everything reachable from it over edges
  between heavy p-micro-clusters, however many hops away. That set does not
  depend on the order the traversal visits it in.
* The reference reads ``.center``, which memoises into ``_center``, of every
  heavy p-micro-cluster when at least two are heavy, and of no light one.
  This class fills the same memos and reads every other centre through a
  copy, so after each call the live micro-clusters equal the reference's,
  memo slot included.
* Merged clusters come from the inherited ``_generate_clusters_for_labels``
  with members in iteration order. One is reused only while its member tuple,
  every member's ``(N, last_edit_time)`` and its own ``N`` are unchanged and
  its ``x`` still equals the first member's, key order included; otherwise it
  is rebuilt. The ``x`` check is there because a micro-cluster's ``x`` is the
  caller's input dict by reference (BUG-3), which the caller can edit without
  ``N`` moving. A reused one has its centre memo dropped, because ``centers``
  hands that dict out and the reference would build a fresh one. A caller
  that edits the sums or times of ``clusters[k]`` in place without changing
  its ``N`` is not detected: the reference hands out fresh objects on every
  call, this class reuses them.

Cost of one ``predict_one``: O(P) for weights and bookkeeping, 2*(P-1)
geometric tests per p-micro-cluster changed or added since the previous call
(``learn_one`` changes or adds at most one), O(P + E) for the traversal over the
E cached edges, a centre per component, and a deep copy plus merges for each
component whose members changed. The cache lives in ``_offline_cache``; it
pickles with the model, and memory probes count it.
"""

from __future__ import annotations

import copy
from collections import defaultdict

from .adapter import build_river_path
from .denstream_patched import DenStreamPatched

build_river_path()

from river.utils.vectordict import euclidean_distance_dict  # noqa: E402


class _OfflineCache:
    """Per-object state of the offline phase, valid for one ``epsilon``."""

    __slots__ = ("epsilon", "state", "out", "merged")

    def __init__(self, epsilon):
        self.epsilon = epsilon
        # micro-cluster -> (N, last_edit_time, radius) the row was built from
        self.state = {}
        # micro-cluster -> {q: None} for every q it is geometrically linked to,
        # in the direction upstream tests it (`self` as the first argument)
        self.out = {}
        # tuple of members -> (their (N, last_edit_time), merged cluster, its N
        # when built)
        self.merged = {}


def _peek_center(mc):
    # `center` memoises into `_center`; read it through a copy when the memo is
    # empty, so this leaves the live object as the reference would.
    return mc._center if mc._center is not None else copy.copy(mc).center


class DenStreamPatchedFast(DenStreamPatched):
    """``DenStreamPatched`` whose offline DBSCAN reuses unchanged neighbourhoods (GAP-3)."""

    def _is_geometrically_reachable(self, center_p, center_q, r_p, r_q):
        # The timestamp-free half of `_is_directly_density_reachable`, with the
        # same argument order and the same expressions.
        distance = euclidean_distance_dict(center_p, center_q)
        return distance < 2 * self.epsilon and distance <= r_p + r_q

    def _sync_offline_cache(self, live):
        cache = self.__dict__.get("_offline_cache")
        if cache is None or cache.epsilon != self.epsilon:
            cache = self._offline_cache = _OfflineCache(self.epsilon)
        state, out = cache.state, cache.out

        live_set = set(live)
        removed = [mc for mc in state if mc not in live_set]
        for mc in removed:
            del state[mc]
            del out[mc]
        if removed:
            for row in out.values():
                for mc in removed:
                    row.pop(mc, None)

        dirty = []
        for mc in live:
            s = state.get(mc)
            if s is None or s[0] != mc.N or s[1] != mc.last_edit_time:
                state[mc] = (mc.N, mc.last_edit_time, mc.calc_radius(self.timestamp))
                out[mc] = {}
                dirty.append(mc)

        geometric = self._is_geometrically_reachable
        centers = {mc: _peek_center(mc) for mc in live} if dirty else {}
        done = set()
        for p in dirty:
            c_p, r_p = centers[p], state[p][2]
            row = out[p]
            for q in live:
                if q is p or q in done:
                    continue
                c_q, r_q = centers[q], state[q][2]
                if geometric(c_p, c_q, r_p, r_q):
                    row[q] = None
                if geometric(c_q, c_p, r_q, r_p):
                    out[q][p] = None
                else:
                    out[q].pop(p, None)
            done.add(p)
        return cache

    def _merged_clusters(self, cache, labels):
        mcs_per_label = defaultdict(list)
        for mc, label in labels.items():
            mcs_per_label[label].append(mc)

        merged_cache = {}
        clusters = {}
        for label, members in mcs_per_label.items():
            key = tuple(members)
            stamp = tuple((mc.N, mc.last_edit_time) for mc in members)
            hit = cache.merged.get(key)
            # The merged cluster is a deep copy of the first member, so its `x`
            # is too; that member's `x` aliases the caller's dict (BUG-3) and
            # can change without N moving, so compare contents, then key
            # order. N of the cached object guards against a caller inserting
            # into `clusters[k]`.
            x0 = members[0].x
            if (
                hit is None
                or hit[0] != stamp
                or hit[1].N != hit[2]
                or hit[1].x != x0
                or (len(x0) > 1 and list(hit[1].x) != list(x0))
            ):
                _, built = self._generate_clusters_for_labels(dict.fromkeys(members, label))
                hit = (stamp, built[label], built[label].N)
            else:
                # `centers` hands out this memo itself; drop whatever a caller
                # did to it, as the reference's fresh object would.
                hit[1]._center = None
            merged_cache[key] = hit
            clusters[label] = hit[1]
        cache.merged = merged_cache
        return len(clusters), clusters

    def predict_one(self, x, w=None):
        if not self.initialized:
            return 0

        live = list(self.p_micro_clusters.values())
        heavy = {mc for mc in live if mc.calc_weight(self.timestamp) > self.mu}
        # The reference's scan reads `.center` -- filling its memo -- of every
        # heavy p-micro-cluster when at least two are heavy, and of no other.
        if len(heavy) > 1:
            for mc in heavy:
                mc.center
        cache = self._sync_offline_cache(live)
        out = cache.out

        c = -1
        labels = dict.fromkeys(live)
        for pmc in live:
            if labels[pmc] is not None:
                continue
            c += 1
            labels[pmc] = c
            if pmc not in heavy:
                continue
            stack = [pmc]
            while stack:
                for neighbor in out[stack.pop()]:
                    if labels[neighbor] is None and neighbor in heavy:
                        labels[neighbor] = c
                        stack.append(neighbor)

        self.n_clusters, self.clusters = self._merged_clusters(cache, labels)

        return self._get_closest_cluster_key(x, self.clusters)
