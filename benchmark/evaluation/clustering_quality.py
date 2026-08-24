"""Cluster-quality metrics (§11).

Wraps scikit-learn's external metrics (ARI / NMI) and adds macro-average purity
(the formula this repo uses, README). All operate on the *labels seen in a time
window*, so the runner can build quality-over-time curves rather than a single
final score (§11, Principle 2).
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as _skm


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


def silhouette(x_rows, y_pred) -> float | None:
    """Internal metric; needs >=2 distinct predicted labels and enough points."""
    if len(x_rows) < 3:
        return None
    labels = list(y_pred)
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return None
    X = np.asarray(x_rows, dtype=float)
    try:
        return float(_skm.silhouette_score(X, labels))
    except Exception:
        return None
