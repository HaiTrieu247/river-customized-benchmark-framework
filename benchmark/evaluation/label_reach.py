"""How far the class labels of one window can be reached by batch clustering.

A stream clusterer scored by ARI against class labels can only do as well as the
labels are reachable from the features at all. On one window of points this module
runs batch methods that are *given* what a stream clusterer never has -- the true
number of classes, or the labels to choose a radius with -- and reports the ARI they
reach. The readings are label-informed references, a diagnostic of the labels and of
the representation, not upper bounds: they see one window at a time and (but for
DBSCAN) exactly as many clusters as classes, while a stream clusterer may keep more
clusters and the history of earlier windows, and so can read higher.

- ``kmeans``  k-means with k = the window's number of true classes (k-means++, 10 starts)
- ``ward``    Ward agglomerative clustering with the same k
- ``dbscan``  DBSCAN over a grid of ``eps`` values, multiples of the window's median
              pairwise distance; the best ARI over the grid and the ``eps`` that gave it
              (the labels choose ``eps``).
              DBSCAN's noise points share the label -1.

Each method runs on the raw features and on the features standardised with the mean and
standard deviation of the window itself (a constant feature becomes 0), which separates
"the labels are not clusters" from "a few wide features dominate the distances".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

METHODS: tuple[str, ...] = ("kmeans", "ward", "dbscan")
REPRESENTATIONS: tuple[str, ...] = ("raw", "standardised")
#: DBSCAN's radius as multiples of the window's median pairwise distance. A runnable
#: placeholder, overridable per call (and by the driver's ``--eps-grid``).
EPS_GRID: tuple[float, ...] = tuple(float(v) for v in np.geomspace(0.02, 2.0, 12))
#: DBSCAN's ``min_samples``; a runnable placeholder.
MIN_SAMPLES = 5


def _matrix(xs: Sequence[dict]) -> np.ndarray:
    keys = sorted({k for x in xs for k in x})
    return np.asarray([[float(x.get(k, 0.0)) for k in keys] for x in xs], dtype=float)


def _standardise(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    return np.where(sd > 0, (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)


def _median_distance(X: np.ndarray, rng: np.random.Generator, max_points: int = 2000) -> float:
    from scipy.spatial.distance import pdist

    if len(X) > max_points:
        X = X[rng.choice(len(X), size=max_points, replace=False)]
    d = pdist(X)
    return float(np.median(d)) if len(d) else 0.0


def window_readings(
    xs: Sequence[dict], ys: Sequence, eps_grid: Sequence[float] = EPS_GRID,
    min_samples: int = MIN_SAMPLES, seed: int = 0,
) -> list[dict]:
    """One reading per (method, representation) on this window; ``[]`` when its labelled
    points hold fewer than two classes (ARI against one class rewards one label).

    A point whose label is ``None`` is left out. Each reading: ``method``,
    ``representation``, ``ari`` (None when the method could not run, e.g. fewer distinct
    points than classes), ``n_points``, ``n_true_classes``, ``n_pred_labels``, and for
    DBSCAN ``eps`` (the chosen multiple) and ``eps_abs`` (the radius itself).
    """
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score

    keep = [i for i, y in enumerate(ys) if y is not None]
    y = [ys[i] for i in keep]
    k = len(set(y))
    if k < 2:
        return []
    raw = _matrix([xs[i] for i in keep])
    rng = np.random.default_rng(seed)
    rows = []
    for rep, X in (("raw", raw), ("standardised", _standardise(raw))):
        n_distinct = len(np.unique(X, axis=0))
        base = {"representation": rep, "n_points": len(y), "n_true_classes": k}
        for method in ("kmeans", "ward"):
            row = {"method": method, **base, "ari": None, "n_pred_labels": None, "eps": None, "eps_abs": None}
            if n_distinct >= k:
                model = (KMeans(n_clusters=k, n_init=10, random_state=seed) if method == "kmeans"
                         else AgglomerativeClustering(n_clusters=k, linkage="ward"))
                pred = model.fit_predict(X)
                row.update(ari=float(adjusted_rand_score(y, pred)), n_pred_labels=int(len(set(pred))))
            rows.append(row)
        row = {"method": "dbscan", **base, "ari": None, "n_pred_labels": None, "eps": None, "eps_abs": None}
        unit = _median_distance(X, rng)
        if unit > 0:
            for mult in eps_grid:
                pred = DBSCAN(eps=float(mult) * unit, min_samples=min_samples).fit_predict(X)
                ari = float(adjusted_rand_score(y, pred))
                if row["ari"] is None or ari > row["ari"]:
                    row.update(ari=ari, n_pred_labels=int(len(set(pred))), eps=float(mult),
                               eps_abs=float(mult) * unit)
        rows.append(row)
    return rows
