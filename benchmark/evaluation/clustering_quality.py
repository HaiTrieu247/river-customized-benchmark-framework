"""Cluster-quality metrics.

External metrics (need ground-truth labels): ARI, NMI, AMI, macro-average
purity. Internal metrics (no labels needed): silhouette, Davies-Bouldin,
Calinski-Harabasz. All external metrics operate on the labels seen in a set of
samples, so the runner can compute them per time-window and build a
quality-over-time picture rather than a single final score.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as _skm

# External metrics keyed by name (all take (y_true, y_pred) -> float | None).
EXTERNAL_METRICS = ("ari", "nmi", "ami", "purity")
# Internal metrics keyed by name (all take (x_rows, y_pred) -> float | None).
INTERNAL_METRICS = ("silhouette", "davies_bouldin", "calinski_harabasz")


def _clean(y_true, y_pred):
    """Drop pairs where the ground-truth label is missing (noise samples)."""
    yt, yp = [], []
    for a, b in zip(y_true, y_pred):
        if a is None or b is None:
            continue
        yt.append(a)
        yp.append(b)
    return yt, yp


def ari(y_true, y_pred) -> float | None:
    yt, yp = _clean(y_true, y_pred)
    if len(yt) < 2 or len(set(yt)) < 1:
        return None
    return float(_skm.adjusted_rand_score(yt, yp))


def nmi(y_true, y_pred) -> float | None:
    yt, yp = _clean(y_true, y_pred)
    if len(yt) < 2:
        return None
    return float(_skm.normalized_mutual_info_score(yt, yp))


def ami(y_true, y_pred) -> float | None:
    """Adjusted Mutual Information — chance-corrected variant of NMI."""
    yt, yp = _clean(y_true, y_pred)
    if len(yt) < 2:
        return None
    return float(_skm.adjusted_mutual_info_score(yt, yp))


def purity(y_true, y_pred) -> float | None:
    """Macro-average purity: per-cluster dominant-class ratio, averaged over K.

    Matches the README convention (macro-average, not the sample-weighted
    variant), so results line up with the existing notebooks.
    """
    yt, yp = _clean(y_true, y_pred)
    if not yt:
        return None
    clusters: dict[object, dict[object, int]] = {}
    for t, p in zip(yt, yp):
        clusters.setdefault(p, {}).setdefault(t, 0)
        clusters[p][t] += 1
    ratios = []
    for counts in clusters.values():
        total = sum(counts.values())
        if total:
            ratios.append(max(counts.values()) / total)
    return float(np.mean(ratios)) if ratios else None


def _valid_internal(x_rows, y_pred):
    """Shared guard for internal metrics: enough points and >=2 distinct labels."""
    if len(x_rows) < 3:
        return None
    labels = list(y_pred)
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return None
    return np.asarray(x_rows, dtype=float), labels


def silhouette(x_rows, y_pred) -> float | None:
    """Silhouette score in [-1, 1]; higher is better."""
    guard = _valid_internal(x_rows, y_pred)
    if guard is None:
        return None
    X, labels = guard
    try:
        return float(_skm.silhouette_score(X, labels))
    except Exception:
        return None


def davies_bouldin(x_rows, y_pred) -> float | None:
    """Davies-Bouldin index; lower is better (>= 0)."""
    guard = _valid_internal(x_rows, y_pred)
    if guard is None:
        return None
    X, labels = guard
    try:
        return float(_skm.davies_bouldin_score(X, labels))
    except Exception:
        return None


def calinski_harabasz(x_rows, y_pred) -> float | None:
    """Calinski-Harabasz index; higher is better."""
    guard = _valid_internal(x_rows, y_pred)
    if guard is None:
        return None
    X, labels = guard
    try:
        return float(_skm.calinski_harabasz_score(X, labels))
    except Exception:
        return None


EXTERNAL_FNS = {"ari": ari, "nmi": nmi, "ami": ami, "purity": purity}
INTERNAL_FNS = {
    "silhouette": silhouette,
    "davies_bouldin": davies_bouldin,
    "calinski_harabasz": calinski_harabasz,
}
