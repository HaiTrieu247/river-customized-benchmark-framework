"""Cluster-quality metrics.

Two families, both needed by the evaluation protocol:

* **streaming metrics** (``build_external_metrics`` / ``build_internal_metrics``)
  — river metric objects updated one sample at a time. These are what the runner
  uses along the stream.
* **batch helpers** (``ari``, ``nmi``, ``ami``, ``purity``, ``silhouette``,
  ``davies_bouldin``, ``calinski_harabasz``) — one-shot functions over a whole
  list of labels/points, used by tests and for offline cross-checks against
  scikit-learn's reference implementations.

External metrics need ground-truth labels; internal ones do not. Samples whose
ground truth is ``None`` (e.g. injected background noise) are dropped from the
external computations rather than treated as a class of their own.
"""

from __future__ import annotations

import collections

from river import metrics


class StreamingPurity(metrics.base.Metric):
    """Purity: fraction of samples that fall in their cluster's majority class.

    Maintains a ``cluster -> class -> weight`` matrix and returns
    ``sum_over_clusters(max_class_weight) / n_samples``. Bounded in [0, 1] but
    *not* chance-corrected: it grows mechanically with the number of clusters,
    so read it alongside ARI/AMI rather than on its own.
    """

    def __init__(self):
        super().__init__()
        self.matrix = collections.defaultdict(lambda: collections.defaultdict(float))
        self.n_samples = 0.0

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred, sample_weight=1.0):
        # river metrics conventionally take y_true, y_pred in update
        self.matrix[y_pred][y_true] += sample_weight
        self.n_samples += sample_weight
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        self.matrix[y_pred][y_true] -= sample_weight
        self.n_samples -= sample_weight
        return self

    def get(self):
        if self.n_samples == 0:
            return 0.0
        majority_sum = 0
        for cluster_classes in self.matrix.values():
            if cluster_classes:
                majority_sum += max(cluster_classes.values())
        return majority_sum / self.n_samples

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model):
        return True


#: External metrics (need ground truth) exposed by name.
EXTERNAL_METRICS = ("ari", "nmi", "ami", "purity")
#: Internal metrics (no labels needed) exposed by name.
INTERNAL_METRICS = ("silhouette",)


def build_external_metrics() -> dict:
    """Fresh river metric objects, keyed by name. Updated as ``(y_true, y_pred)``."""
    return {
        "ari": metrics.AdjustedRand(),
        "nmi": metrics.NormalizedMutualInfo(),
        "ami": metrics.AdjustedMutualInfo(),
        "purity": StreamingPurity(),
    }


def build_internal_metrics() -> dict:
    """Fresh label-free metric objects, keyed by name.

    river's streaming ``Silhouette`` is updated as ``(x, y_pred, centers)`` and,
    unlike the batch silhouette, is a *cohesion / separation* ratio where
    **lower is better** (``bigger_is_better is False``). The runner passes the
    model's centers; clusterers that do not expose centers simply never update
    it and it reports no value.
    """
    return {"silhouette": metrics.Silhouette()}


# --- batch helpers (offline cross-checks) ----------------------------------
def _paired(y_true, y_pred):
    """Drop pairs whose ground truth is None; return ([...], [...]) or None."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None]
    if not pairs:
        return None
    return [t for t, _ in pairs], [p for _, p in pairs]


def ari(y_true, y_pred) -> float | None:
    """Adjusted Rand Index over a whole label sequence (scikit-learn)."""
    from sklearn.metrics import adjusted_rand_score

    pair = _paired(y_true, y_pred)
    return None if pair is None else float(adjusted_rand_score(*pair))


def nmi(y_true, y_pred) -> float | None:
    """Normalized Mutual Information over a whole label sequence."""
    from sklearn.metrics import normalized_mutual_info_score

    pair = _paired(y_true, y_pred)
    return None if pair is None else float(normalized_mutual_info_score(*pair))


def ami(y_true, y_pred) -> float | None:
    """Adjusted Mutual Information over a whole label sequence."""
    from sklearn.metrics import adjusted_mutual_info_score

    pair = _paired(y_true, y_pred)
    return None if pair is None else float(adjusted_mutual_info_score(*pair))


def purity(y_true, y_pred) -> float | None:
    """Batch counterpart of :class:`StreamingPurity`."""
    pair = _paired(y_true, y_pred)
    if pair is None:
        return None
    metric = StreamingPurity()
    for t, p in zip(*pair):
        metric.update(t, p)
    return metric.get()


def _n_clusters(labels) -> int:
    return len(set(labels))


def silhouette(X, labels) -> float | None:
    """Batch silhouette (higher is better). ``None`` when fewer than 2 clusters."""
    from sklearn.metrics import silhouette_score

    if _n_clusters(labels) < 2 or len(labels) <= _n_clusters(labels):
        return None
    return float(silhouette_score(X, labels))


def davies_bouldin(X, labels) -> float | None:
    """Batch Davies-Bouldin index (lower is better)."""
    from sklearn.metrics import davies_bouldin_score

    if _n_clusters(labels) < 2:
        return None
    return float(davies_bouldin_score(X, labels))


def calinski_harabasz(X, labels) -> float | None:
    """Batch Calinski-Harabasz index (higher is better)."""
    from sklearn.metrics import calinski_harabasz_score

    if _n_clusters(labels) < 2:
        return None
    return float(calinski_harabasz_score(X, labels))
