"""``HPStream`` whose cluster labels are never reused (RIVER_BUGS.md GAP-6).

Per AGENTS.md rule 1 the vendored ``river`` tree is not modified; per rule 3 this is registered under
its own name, beside the untouched ``HPStream``.

River's ``HPStream`` labels a cluster by its key in ``_clusters``. A cluster opened online takes
``_free_label()``, the smallest key no cluster holds, and the eviction that makes room for it runs
first (``hpstream.py:620-628``), so the newcomer takes the evicted cluster's key. Within one window of
predict-before-learn labels, that key then names the evicted cluster for the points before the
eviction and the newcomer for the points after it, and a sequential partition joins points the model
never held together.

Here ``_free_label`` hands out keys from a counter that only grows, starting above every key the
offline initialisation can make (``max_clusters``). Nothing else changes. Why the model is the same
model, read off the vendored source at commit ``873fa6e0``:

* a key is only ever a name: ``_closest``, ``_compute_dimensions`` (its ``sort`` is on the radius only,
  stable, so ties go by iteration order), the eviction's ``min`` over ``last_seen`` and the starvation
  sweep all walk ``_clusters`` in insertion order, and no code compares two keys outside
  ``_initialize``/``_group``, which run before the first ``_free_label`` call;
* the newcomer is inserted last under both rules: River's reused key was deleted the line before, so
  it goes to the end of the dict, as a fresh key does.

So at every point both models hold the same clusters, with the same sums, weights, times and
dimensions, in the same order, and predict the same cluster; only its name may differ
(``tests/test_hpstream_stable_labels.py`` checks this point by point, with evictions and starved
clusters). ``predict_one`` still returns 0 when no cluster exists, as River's does.
"""

from __future__ import annotations

from river import cluster


class HPStreamStableLabels(cluster.HPStream):
    """River's ``HPStream`` with every new cluster labelled from a counter that only grows, so a label is
    never handed from an evicted cluster to the one opened in its place (RIVER_BUGS.md GAP-6). The
    parameters, the clusters and the predicted cluster are HPStream's; see the module docstring.
    """

    def _free_label(self) -> int:
        nxt = getattr(self, "_next_stable_label", None)
        if nxt is None:
            nxt = max(self.max_clusters, max(self._clusters, default=-1) + 1)
        while nxt in self._clusters:
            nxt += 1
        self._next_stable_label = nxt + 1
        return nxt
