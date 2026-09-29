"""``DenStream`` with RIVER_BUGS.md BUG-7 and BUG-8 fixed in a subclass.

Per AGENTS.md rule 1 the vendored ``river`` tree is never modified to fix a
bug in an existing algorithm -- see ``RIVER_BUGS.md`` for the write-up and the
revert history that motivates it. Per rule 3, the workaround is instead
registered under its own name so the untouched upstream behaviour stays
available for comparison. This module is that workaround for two bugs in
``river.cluster.denstream.DenStream``:

* **BUG-7** -- ``_generate_clusters_for_labels`` does
  ``copy.copy(micro_clusters[0])`` and then ``merge``s the rest into it. A
  ``copy.copy`` of a ``__slots__`` object shares its dict-valued slots by
  reference, so ``linear_sum``/``squared_sum`` on the "merged" cluster are the
  *same dicts* as the first member's, and ``merge`` mutates them in place.
  Every ``predict_one`` call that puts two live p-micro-clusters under one
  DBSCAN label therefore corrupts one of them permanently.
* **BUG-8** -- the seed-expansion step inside ``predict_one``'s DBSCAN reads
  ``if labels[neighbor_neighbor] is not None: seed_queue.append(...)``, the
  opposite of what expansion needs. Only neighbours that are *already*
  labelled get queued, so a label can never travel past the direct neighbours
  of the point DBSCAN started from -- a density-reachable chain of
  p-micro-clusters never becomes one cluster.

``DenStreamPatched`` overrides exactly those two methods and inherits
everything else -- ``learn_one``, the online merge/prune logic, the buffer
initialisation -- unchanged from ``river.cluster.DenStream``.
"""

from __future__ import annotations

import copy
from collections import defaultdict, deque

from .adapter import build_river_path

# Ensure the sibling river tree is importable before we touch river.*, exactly
# as benchmark/algorithms/registry.py does.
build_river_path()

from river.cluster import DenStream  # noqa: E402


class DenStreamPatched(DenStream):
    """``DenStream`` with RIVER_BUGS.md BUG-7 and BUG-8 fixed; everything else inherited."""

    @staticmethod
    def _generate_clusters_for_labels(cluster_labels):
        # BUG-7 fix: deep-copy the first micro-cluster before merging into it,
        # so the merged object never shares `linear_sum`/`squared_sum` with a
        # live p-micro-cluster that `learn_one` keeps updating.
        clusters = {}
        mcs_per_label = defaultdict(deque)
        for mc, label in cluster_labels.items():
            mcs_per_label[label].append(mc)

        for label, micro_clusters in mcs_per_label.items():
            merged = copy.deepcopy(micro_clusters[0])
            for mc in list(micro_clusters)[1:]:
                merged.merge(mc)
            clusters[label] = merged

        return len(clusters), clusters

    def predict_one(self, x, w=None):
        if not self.initialized:
            return 0

        c = -1
        labels = {pmc: None for pmc in self.p_micro_clusters.values()}

        for pmc in self.p_micro_clusters.values():
            if labels[pmc] is not None:
                continue
            c += 1
            labels[pmc] = c
            queue = self._query_neighbor(pmc)
            while queue:
                cur = queue.popleft()
                if labels[cur] is not None:
                    continue
                labels[cur] = c
                # BUG-8 fix: expand into the *unlabelled* neighbours, not the
                # already-labelled ones -- otherwise a label never reaches
                # past the direct neighbours of the point DBSCAN started from.
                for neighbor in self._query_neighbor(cur):
                    if labels[neighbor] is None:
                        queue.append(neighbor)

        self.n_clusters, self.clusters = self._generate_clusters_for_labels(labels)

        return self._get_closest_cluster_key(x, self.clusters)
