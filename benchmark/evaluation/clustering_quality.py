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


class StreamingMacroPurity(metrics.base.Metric):
    """Purity averaged over clusters, unweighted -- the papers' definition.

    ``mean_over_clusters(max_class_weight / cluster_weight)``, i.e. every cluster
    counts once however many samples it holds. This is the formula the stream
    clustering papers report, and it is **not** what :class:`StreamingPurity`
    computes: that one is the sample-weighted (micro) version,
    ``sum_over_clusters(max_class_weight) / n_samples``.

    DenStream (Cao et al., SDM 2006, Sec. 5)::

        pur = (sum_i |C_i^d| / |C_i|) / K

    RepStream (Lühr & Lazarescu, DKE 2009, Eq. 18) defines the same quantity as
    ``TCP(C) = mean_i CP(c_i)``; SOStream reports purity the same way.

    Why both are carried: the micro version answers "what fraction of samples
    sit in a cluster dominated by their own class", the macro version answers
    "how pure is the average cluster". They diverge exactly when cluster sizes
    are uneven -- a swarm of tiny single-class clusters pushes the macro value
    to 1 while barely moving the micro one. So the macro number is the one that
    compares against the numbers printed in those papers, and it is only
    readable next to the cluster count; on its own it rewards fragmentation
    harder than the micro version does.
    """

    def __init__(self):
        super().__init__()
        self.matrix = collections.defaultdict(lambda: collections.defaultdict(float))

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred, sample_weight=1.0):
        self.matrix[y_pred][y_true] += sample_weight
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        self.matrix[y_pred][y_true] -= sample_weight
        return self

    def get(self):
        shares = []
        for cluster_classes in self.matrix.values():
            total = sum(cluster_classes.values())
            if total > 0:
                shares.append(max(cluster_classes.values()) / total)
        if not shares:
            return 0.0
        return sum(shares) / len(shares)

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model):
        return True


class StreamingSSQ(metrics.base.Metric):
    """Sum of Squared Distance (SSQ) between samples and their cluster centers.

    Lower values indicate tighter/more compact clusters (better cohesion).
    """

    def __init__(self, average: bool = True):
        super().__init__()
        self.average = average
        self.total_ssq = 0.0
        self.n_samples = 0.0

    @property
    def requires_labels(self):
        return False

    def update(self, x: dict, y_pred: int | None = None, centers: dict | None = None, sample_weight: float = 1.0):
        if not centers:
            return self
        center = None
        if y_pred is not None and isinstance(centers, dict) and y_pred in centers:
            center = centers[y_pred]
        else:
            shortest = float("inf")
            candidate_centers = centers.values() if isinstance(centers, dict) else centers
            for c in candidate_centers:
                if not isinstance(c, dict):
                    continue
                d2 = sum((x.get(f, 0.0) - c.get(f, 0.0)) ** 2 for f in set(x) | set(c))
                if d2 < shortest:
                    shortest = d2
                    center = c
        if not center:
            return self
        dist_sq = sum((x.get(f, 0.0) - center.get(f, 0.0)) ** 2 for f in set(x) | set(center))
        self.total_ssq += dist_sq * sample_weight
        self.n_samples += sample_weight
        return self

    def revert(self, x: dict, y_pred: int | None = None, centers: dict | None = None, sample_weight: float = 1.0):
        if not centers or self.n_samples <= 0:
            return self
        center = None
        if y_pred is not None and isinstance(centers, dict) and y_pred in centers:
            center = centers[y_pred]
        if center:
            dist_sq = sum((x.get(f, 0.0) - center.get(f, 0.0)) ** 2 for f in set(x) | set(center))
            self.total_ssq = max(0.0, self.total_ssq - dist_sq * sample_weight)
            self.n_samples = max(0.0, self.n_samples - sample_weight)
        return self

    def get(self) -> float:
        if self.n_samples == 0:
            return 0.0
        return (self.total_ssq / self.n_samples) if self.average else self.total_ssq

    @property
    def bigger_is_better(self):
        return False

    def works_with(self, model):
        return True


class StreamingMATRTrace(metrics.base.Metric):
    """Trace criterion of Fan et al. (2020), accumulated from cluster features.

    Maintains ``(n_k, LS_k, SS_k)`` per predicted cluster label and returns

    ``-2 * sum_k ( SS_k - ||LS_k||^2 / n_k )``

    which is the negated within-cluster sum of squares, so **bigger is better**.

    Two properties distinguish it from :class:`StreamingSSQ`, and both are why it
    exists. It needs only the predicted label, never the model's ``centers``, so
    it is defined for every clusterer in the pool rather than only the ones that
    expose centres; and it is not divided by the sample count.

    The similarity measure is fixed to the raw squared distance and never
    depends on the hyperparameter being tuned -- only the *partition* varies
    with the configuration. Fan et al. require this: letting the criterion
    depend on the tuned parameter makes it self-referential and voids the
    theoretical guarantees.

    Accumulating by label id over a whole stream is only meaningful while ids
    are stable. Algorithms that merge clusters online (SOStream, DBSTREAM)
    recycle ids, so one label can cover two different clusters at two points in
    the stream and inflate the trace. Read the **per-window** value for those.
    """

    #: The trace needs the predicted label only, so the runner updates it even
    #: when the model exposes no centres.
    requires_centers = False

    def __init__(self):
        super().__init__()
        self.n = collections.defaultdict(float)
        self.ls = collections.defaultdict(lambda: collections.defaultdict(float))
        self.ss = collections.defaultdict(float)

    @property
    def requires_labels(self):
        return False

    def update(self, x: dict, y_pred=None, centers: dict | None = None, sample_weight: float = 1.0):
        if y_pred is None:
            return self
        self.n[y_pred] += sample_weight
        row = self.ls[y_pred]
        sq = 0.0
        for feature, value in x.items():
            value = float(value)
            row[feature] += value * sample_weight
            sq += value * value
        self.ss[y_pred] += sq * sample_weight
        return self

    def revert(self, x: dict, y_pred=None, centers: dict | None = None, sample_weight: float = 1.0):
        if y_pred is None or y_pred not in self.n:
            return self
        self.n[y_pred] -= sample_weight
        row = self.ls[y_pred]
        sq = 0.0
        for feature, value in x.items():
            value = float(value)
            row[feature] -= value * sample_weight
            sq += value * value
        self.ss[y_pred] -= sq * sample_weight
        return self

    def get(self) -> float:
        total = 0.0
        for label, n_k in self.n.items():
            if n_k <= 0:
                continue
            ls_norm_sq = sum(v * v for v in self.ls[label].values())
            total += self.ss[label] - ls_norm_sq / n_k
        return -2.0 * total

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model):
        return True


#: External metrics (need ground truth) exposed by name.
#:
#: ``homogeneity`` and ``completeness`` are carried alongside ARI because ARI
#: folds two different mistakes into one number: splitting a true class across
#: several clusters, and merging several classes into one cluster. High
#: homogeneity with low completeness means over-segmentation; the reverse means
#: under-segmentation. The pair separates them, ARI alone cannot.
EXTERNAL_METRICS = (
    "ari", "nmi", "ami", "purity", "purity_macro", "homogeneity", "completeness",
)
#: Internal metrics (no labels needed) exposed by name.
INTERNAL_METRICS = ("silhouette", "ssq", "matr_trace")


def build_external_metrics() -> dict:
    """Fresh river metric objects, keyed by name. Updated as ``(y_true, y_pred)``."""
    return {
        "ari": metrics.AdjustedRand(),
        "nmi": metrics.NormalizedMutualInfo(),
        "ami": metrics.AdjustedMutualInfo(),
        "purity": StreamingPurity(),
        "purity_macro": StreamingMacroPurity(),
        "homogeneity": metrics.Homogeneity(),
        "completeness": metrics.Completeness(),
    }


def build_internal_metrics() -> dict:
    """Fresh label-free metric objects, keyed by name.

    river's streaming ``Silhouette`` is updated as ``(x, y_pred, centers)`` and,
    unlike the batch silhouette, is a *cohesion / separation* ratio where
    **lower is better** (``bigger_is_better is False``). The runner passes the
    model's centers; clusterers that do not expose centers simply never update
    it and it reports no value.

    ``matr_trace`` is the exception to that: it accumulates cluster features from
    the predicted label alone, so it is defined for every clusterer including the
    ones with no centres. The runner updates it regardless of ``centers``.
    """
    return {
        "silhouette": metrics.Silhouette(),
        "ssq": StreamingSSQ(),
        "matr_trace": StreamingMATRTrace(),
    }


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

def compute_ssq(X, centers: dict | list, labels: list | None = None) -> float:
    """Compute Sum of Squared Distance (SSQ) over points and cluster centers.

    Lower is better (tighter cluster cohesion).
    """
    if not centers:
        return 0.0

    total_ssq = 0.0
    candidate_centers = list(centers.values()) if isinstance(centers, dict) else list(centers)

    for i, x in enumerate(X):
        x_dict = x if isinstance(x, dict) else {j: float(v) for j, v in enumerate(x)}
        center = None
        if labels is not None and i < len(labels) and isinstance(centers, dict) and labels[i] in centers:
            center = centers[labels[i]]
        else:
            shortest = float("inf")
            for c in candidate_centers:
                if not isinstance(c, dict):
                    continue
                d2 = sum((x_dict.get(f, 0.0) - c.get(f, 0.0)) ** 2 for f in set(x_dict) | set(c))
                if d2 < shortest:
                    shortest = d2
                    center = c

        if center:
            dist_sq = sum((x_dict.get(f, 0.0) - center.get(f, 0.0)) ** 2 for f in set(x_dict) | set(center))
            total_ssq += dist_sq

    return total_ssq

