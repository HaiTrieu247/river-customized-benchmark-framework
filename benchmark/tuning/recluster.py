"""Table A: common reclustering of a protocol algorithm's micro-clusters.

Comparing the ARI of two stream clusterers that use different *offline*
reclusterers -- k-means (CluStream, HPStream, STREAMKMeans) vs. DBSCAN
(DenStream) vs. whatever an online-merge algorithm does instead (DBSTREAM,
SOStream, BIRCH, RepStream) -- compares the reclusterers, not the online
phase that actually built the micro-clusters. Table A removes that
confound: pull every algorithm's current micro-cluster centres (and their
weights) through :func:`micro_centers`, then run the *same* k-means and the
*same* DBSCAN over those centres with :func:`common_recluster`, scoring the
result against the window's own labels.

``micro_centers`` has to speak every protocol algorithm's dialect for where
its micro-cluster level lives, because none of it is standardised across
``river.cluster``:

======================  ===========================================  ================================
Algorithm               Micro-cluster container                       Centre / weight source
======================  ===========================================  ================================
DenStream               ``model.p_micro_clusters`` (dict)             ``mc.center`` (dict) / ``mc.N``
CluStream               ``model.micro_clusters`` (dict)                ``mc.center`` (dict) / ``mc.weight``
DBSTREAM                ``model.micro_clusters`` (dict, property)      ``mc.center`` (dict) / ``mc.weight``
SOStream                ``model.micro_clusters`` (dict, property)      ``mc.center`` (dict) / ``mc.weight``
BIRCH                   ``model.micro_clusters`` (dict, property);     ``mc.center`` (LIST, needs
                        leaf ``BIRCHClusteringFeature`` entries        ``model._as_dict``) / ``mc.n``
RepStream               ``model.micro_clusters`` (dict, property);     ``mc.values`` (LIST, needs
                        the representative vertices                   ``model._as_dict``) / ``mc.count``
HPStream                ``model.centers`` (dict, property)             centre dict directly / weight 1.0
STREAMKMeans            ``model.centers`` (plain dict attribute)       centre dict directly / weight 1.0
======================  ===========================================  ================================

BIRCH's and RepStream's leaf/vertex objects store their coordinates as a
plain ``list`` positioned against the *model's* ``_keys``/``_as_dict``, not
as a self-describing dict the way every other algorithm's micro-cluster
does -- so those two need the extra indirection through the owning model
that the generic ``dict``-shaped branch below does not.

The centre *weight* means something different per algorithm -- a raw point
count (DenStream, BIRCH), a decayed weight (DBSTREAM, SOStream), a usage
counter (RepStream) or a fixed 1.0 (HPStream, STREAMKMeans) -- so a weighted
reading is not comparable across algorithms on that axis alone. k-means uses
the weights (a centre it moves still has to reflect how much mass sits on
it); DBSCAN in :func:`common_recluster` deliberately does not (see there).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin_min

__all__ = ["micro_centers", "common_recluster"]


def _rows(items):
    """``(center dict, weight)`` pairs -> ``(centers, weights, sorted keys)``.

    Any falsy/empty center dict is dropped. A missing or non-positive weight
    becomes 1.0, so a downstream ``(w > 0).all()`` check never trips on an
    algorithm whose "weight" bookkeeping happens to read zero for a young
    micro-cluster. Returns ``None`` when nothing usable is left.
    """
    items = [(c, 1.0 if w is None else float(w)) for c, w in items if c]
    if not items:
        return None
    keys = sorted(items[0][0].keys())
    C = np.asarray([[float(c.get(k, 0.0)) for k in keys] for c, _ in items])
    w = np.asarray([w for _, w in items])
    return C, np.where(w > 0, w, 1.0), keys


def _center_to_dict(model, c):
    """Coerce a centre to a ``{feature: value}`` dict.

    Most algorithms' micro-clusters already keep their centre as a dict.
    BIRCH's ``BIRCHClusteringFeature.center`` and RepStream's
    ``RepStreamVertex.values`` are plain coordinate lists instead, positional
    against the *model's* feature order -- ``model._as_dict`` (or, failing
    that, ``model._keys``) is what turns one back into a dict.
    """
    if c is None or isinstance(c, dict):
        return c
    if isinstance(c, (list, tuple)):
        as_dict = getattr(model, "_as_dict", None)
        if callable(as_dict):
            return as_dict(list(c))
        keys = getattr(model, "_keys", None)
        if keys:
            return dict(zip(keys, c))
        return None
    return None


def micro_centers(model) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Best-effort ``(centres, weights, feature keys)`` for a clusterer's
    micro-cluster level, or ``None`` when it has not produced any yet.

    Returns a tuple ``(C, w, keys)`` where ``C`` is ``(m, d)`` float, ``w`` is
    ``(m,)`` float and strictly positive, and ``keys`` is the sorted list of
    feature names ``C``'s columns line up with.
    """
    p = getattr(model, "p_micro_clusters", None)  # DenStream
    if isinstance(p, dict):
        return _rows((mc.center, mc.N) for mc in p.values())

    # CluStream, DBSTREAM, SOStream, BIRCH, RepStream.
    mcs = getattr(model, "micro_clusters", None)
    if isinstance(mcs, dict) and mcs:
        out = []
        for mc in mcs.values():
            c = getattr(mc, "center", None)
            if c is None:
                c = getattr(mc, "x", None)
            if c is None:
                c = getattr(mc, "values", None)  # RepStream vertex
            if c is None and hasattr(mc, "linear_sum") and getattr(mc, "n", 0):  # BIRCH CF, defensive
                ls = mc.linear_sum
                c = {k: v / mc.n for k, v in ls.items()} if isinstance(ls, dict) else [v / mc.n for v in ls]
            c = _center_to_dict(model, c)
            w = getattr(mc, "weight", None)
            if w is None:
                w = getattr(mc, "n", None)  # BIRCH CF count
            if w is None:
                w = getattr(mc, "count", None)  # RepStream vertex usage count
            if c:
                out.append((c, w))
        return _rows(out)

    # HPStream, STREAMKMeans, and a generic fallback for anything else that
    # exposes centres the same way.
    centers = getattr(model, "centers", None)
    if callable(centers):
        centers = centers()
    if isinstance(centers, dict) and centers:
        return _rows((c, 1.0) for c in centers.values())
    return None


def common_recluster(
    centers: np.ndarray | None,
    weights: np.ndarray | None,
    keys: list[str] | None,
    xs: list[dict],
    ys: list,
    eps: float,
) -> dict:
    """Recluster ``centers`` with a shared k-means and DBSCAN, score both.

    ``centers``/``weights``/``keys`` are one :func:`micro_centers` reading.
    ``xs``/``ys`` are the window's own points and labels. A label of ``None``
    means "unlabelled" and that point is skipped; every other label --
    including a string (KDD99, INSECTS) or an int such as ``-1`` -- is a real
    class, exactly as :func:`conf_pipeline.run.batch_external` treats it, so
    labels are factorised rather than compared as numbers.

    Nearest-centre assignment is O(h*m) (``m`` centres, ``h`` window points)
    via :func:`sklearn.metrics.pairwise_distances_argmin_min`, not the naive
    ``(h, m, d)`` broadcast: BIRCH/CluStream can carry thousands of centres,
    and an (h, m, d) cube at ``h=1000``, ``m=10_000``, ``d=41`` is gigabytes.

    k-means uses ``k`` = number of distinct labelled classes in the window,
    weighted by ``weights``; when there are no more centres than classes,
    k-means is skipped and every centre is its own macro cluster. DBSCAN uses
    ``eps`` and ``min_samples=2`` and is **unweighted**: a centre's weight
    means a different thing per algorithm (a raw count for DenStream/BIRCH, a
    decayed weight for DBSTREAM/SOStream, a usage counter for RepStream, a
    fixed 1.0 for HPStream/STREAMKMeans), so a weighted "core point" reading
    would not mean the same threshold across algorithms -- unweighted, every
    centre counts once and ``min_samples=2`` uniformly means "needs a
    neighbour". A noise centre keeps its own singleton label rather than
    being dropped, so no window point is left unassigned.

    Returns a dict with keys ``ari_A_kmeans``, ``ari_A_dbscan``,
    ``ssq_micro``, ``n_micro_centers``. All four are ``None`` when ``centers``
    is ``None``/empty; ``ssq_micro``/``n_micro_centers`` are otherwise always
    filled, while the two ARIs additionally require at least two distinct
    labelled classes in the window.
    """
    empty = {"ari_A_kmeans": None, "ari_A_dbscan": None, "ssq_micro": None, "n_micro_centers": None}
    if centers is None or len(centers) == 0:
        return empty

    centers = np.asarray(centers, dtype=float)
    weights = np.asarray(weights, dtype=float)
    X = np.asarray([[float(x.get(k, 0.0)) for k in keys] for x in xs])
    nearest, dist = pairwise_distances_argmin_min(X, centers)
    ssq = float(np.sum(dist**2))
    out = dict(empty, ssq_micro=ssq, n_micro_centers=int(len(centers)))

    keep = np.array([v is not None for v in ys])
    labelled = [v for v, k in zip(ys, keep) if k]
    k = len(set(labelled))
    if keep.sum() < 2 or k < 2:
        return out
    # Factorise: labels can be strings (KDD99, INSECTS) or ints including -1
    # (a real class here, not a sentinel) -- never compared as numbers.
    _, y_ids = np.unique(np.asarray(labelled, dtype=object), return_inverse=True)
    nearest_kept = nearest[keep]

    with warnings.catch_warnings():
        # A handful of near-duplicate micro-centres (or k close to the
        # number of centres) routinely triggers sklearn's "fewer distinct
        # clusters than n_clusters" ConvergenceWarning; it is noise here,
        # not a signal -- Table A's own metrics (ssq, n_micro_centers)
        # already say when the centres are degenerate.
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        if len(centers) > k:
            km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(centers, sample_weight=weights)
            macro = km.labels_
        else:
            macro = np.arange(len(centers))
        out["ari_A_kmeans"] = float(adjusted_rand_score(y_ids, macro[nearest_kept]))

        db = DBSCAN(eps=eps, min_samples=2).fit(centers).labels_  # unweighted, see docstring
    db = np.where(db < 0, db.max() + 1 + np.arange(len(db)), db)  # noise centres keep their own label
    out["ari_A_dbscan"] = float(adjusted_rand_score(y_ids, db[nearest_kept]))
    return out
