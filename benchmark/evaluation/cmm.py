"""CMM, the Cluster Mapping Measure (Kremer et al., KDD 2011), ported from MOA.

ARI, NMI and purity treat every misassigned point alike. CMM was designed for
evolving streams and weighs a fault by how clearly the point belongs to its class:
a point far from the rest of its class costs little when it lands elsewhere, one in
the core of its class costs a lot, and older points of the evaluation horizon count
less than recent ones. It also forgives errors the class structure forces on any
clusterer: classes whose points are connected are merged before scoring, and a point
that lies inside the hull of the class its cluster is mapped to is not charged.

This is a port of MOA's implementation, not a re-derivation from the paper, so that
the numbers can be set beside MOA's. Files ported (read from Waikato/moa, master; the
tests check every reading against MOA 2024.07.0 itself, tests/fixtures/cmm_moa_cases.json):

- ``moa/evaluation/CMM_GTAnalysis.java``: the ground-truth analysis -- k-nearest-
  neighbour distances within every class, point connectivity, the class-to-class
  connections and the merging of connected classes (``CMMWindow.build``);
- ``moa/evaluation/CMM.java``: the matching of found clusters to (merged) classes by
  frequency profile, the three faults -- missed, misplaced, noise inclusion -- and the
  final ratio (``CMMWindow.measures``);
- ``moa/cluster/Clustering.java`` (the constructor from labelled points that
  ``EvaluateClustering`` uses for the ground truth: one ``SphereCluster`` per class),
  ``moa/cluster/SphereCluster.java`` and ``moa/cluster/Miniball.java`` (Gaertner's
  smallest enclosing ball, which is that sphere): ``_miniball``. The ball is unique, so
  it is computed by the Fischer-Gaertner-Kutz algorithm (``_meb``), which needs a few
  dozen steps where MOA's needs tens of thousands in 30 dimensions; the line-for-line
  port (``_miniball_moa``) is kept and the tests hold the two to 1e-9;
- ``moa/gui/BatchCmd.java`` and ``moa/gui/visualization/DataPoint.java``: the weight
  of a point (``fading_weights``).

Where this differs from MOA, and why:

- MOA scores a ``Clustering`` of geometric clusters (mostly spheres) and assigns a
  point to every cluster whose inclusion probability is at least 0.5. River assigns
  each point one label, so here a found cluster *is* the set of points given its
  label, and a point is covered by exactly that cluster. A missed point's penalty is
  scaled by its distance to the hull of the clusters mapped to its class; MOA computes
  that distance only for a ``SphereCluster`` and uses 1 for any other shape, which is
  the branch a label-defined cluster takes: the weight is ``1 - exp(-lambda_missed)``.
- MOA returns 1 when the denominator of the ratio is zero (no point with a positive
  connectivity and weight); ``score`` returns None there instead, because 1 would read
  as a perfect clustering of a window CMM cannot score. It also returns None for an
  empty window and for any NaN.
- Points carry no class attribute, so "noise" is ``y_true is None`` rather than MOA's
  nominal value named ``noise``.

Faithful to MOA on purpose, including two quirks that change numbers: after a merge,
the merged class's mean and deviation of the kNN distance are accumulated onto the
values it had before instead of recomputed from zero (``GTCluster.calculateKnn``), and
the class-to-class connections are computed before the points' own connectivities,
so every point enters them with connectivity 1. A single-class window is scored like
any other (CMM is then about 1 for any partition that leaves no point out); whether
such a window carries information is the caller's rule, not this module's.

What to expect on real data: the classes of a real window are often connected, and the
analysis then merges them -- on an INSECTS window all six classes become one and every
partition scores 1 -- or lie inside each other's enclosing sphere, which excuses the
misplaced points (a KDD99 window scores 0.99999 for k-means). ``ca_separability`` (merged
classes over classes) and ``ca_model`` (share of points outside the other classes'
spheres) say when this happened; ``enable_class_merge`` and ``enable_model_error`` turn
the two off, at the price of no longer being MOA's number.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: ``CMM_GTAnalysis.knnNeighbourhood``: neighbours in the kNN distance.
KNN_NEIGHBOURHOOD = 2
#: ``CMM_GTAnalysis.tauConnection``: two classes connected above this are merged (1 disables).
TAU_CONNECTION = 0.5
#: ``CMM_GTAnalysis.clusterConnectionMaxPoints``: best-connected points per class-to-class link.
CLUSTER_CONNECTION_MAX_POINTS = 2
#: ``CMM.lamdaMissed``: how steeply a missed point's penalty grows with its hull distance.
LAMBDA_MISSED = 1.0
#: ``ClusteringStream.decayThresholdOption``: the weight the horizon decays towards.
DECAY_THRESHOLD = 0.01
#: The two fading rules MOA has. ``batch`` is ``BatchCmd`` (``EvaluateClustering``, the
#: command-line task, which scores disjoint windows of ``decayHorizon`` points as this
#: framework does): rate ``ln(1/threshold) / horizon`` used as a power of 2, so the
#: oldest point weighs about 0.04, not 0.01. ``visualizer`` is ``RunVisualizer`` (the
#: GUI, a sliding horizon): rate ``log2(1/threshold) / horizon``, oldest weight 0.01.
DECAY_RULES: tuple[str, ...] = ("batch", "visualizer")
#: ``CMM``'s MIN_ERROR: a fault the model is excused for still counts in "CMM Basic".
MIN_ERROR = 0.00001
#: ``Miniball.radius()`` pads the ball by this relative margin.
MINIBALL_SLACK = 0.00001

#: The readings ``CMMWindow.measures`` returns, MOA's names in snake case.
MEASURES: tuple[str, ...] = (
    "cmm", "cmm_basic", "cmm_missed", "cmm_misplaced", "cmm_noise",
    "ca_separability", "ca_noise", "ca_model",
)


# --- weights ---------------------------------------------------------------------------

def fading_weights(n: int, horizon: int | None = None, threshold: float = DECAY_THRESHOLD,
                   rule: str = "batch") -> np.ndarray:
    """MOA's weight of each of ``n`` points in arrival order: ``2 ** (-rate * age)`` with
    ``age = n - 1 - i`` (``DataPoint.updateWeight`` at the last point's timestamp) and the
    rate of ``rule`` over ``horizon`` points (default: ``n``, the window)."""
    if rule not in DECAY_RULES:
        raise ValueError(f"rule must be one of {DECAY_RULES}, got {rule!r}")
    horizon = n if horizon is None else horizon
    if n == 0:
        return np.zeros(0)
    if rule == "batch":
        rate = -1 * math.log(threshold) / horizon
    else:
        rate = math.log(1.0 / threshold) / math.log(2) / horizon
    age = np.arange(n - 1, -1, -1, dtype=float)
    return np.power(2.0, (-1.0) * rate * age)


# --- the smallest enclosing ball (MOA's Miniball, Gaertner 1999) ------------------------

def _seqsum(v: np.ndarray, start: float = 0.0) -> float:
    """Left-to-right sum, the order of a Java loop (numpy's pairwise sum rounds differently)."""
    if v.size == 0:
        return float(start)
    return float(np.cumsum(np.concatenate(([start], v)))[-1])


class _MiniballB:
    """``Miniball.Miniball_b``: the ball through the points pushed so far."""

    def __init__(self, d: int):
        self.d = d
        self.q0 = np.zeros(d)
        self.z = np.zeros(d + 1)
        self.f = np.zeros(d + 1)
        self.v = np.zeros((d + 1, d))
        self.a = np.zeros((d + 1, d + 1))
        self.c = np.zeros((d + 1, d))
        self.sqr_r = np.zeros(d + 1)
        self.m = 0
        self.s = 0
        self.current_c = self.c[0]
        self.current_sqr_r = -1.0

    def reset(self) -> None:
        self.m = self.s = 0
        self.c[0, :] = 0.0
        self.current_c = self.c[0]  # a view, as Java's array reference
        self.current_sqr_r = -1.0

    def excess(self, P: np.ndarray) -> np.ndarray:
        """``excess`` of every row of ``P``, summed over coordinates in Java's order."""
        sq = (P - self.current_c) ** 2
        lead = np.full((len(P), 1), -self.current_sqr_r)
        return np.cumsum(np.hstack([lead, sq]), axis=1)[:, -1]

    def push(self, p: np.ndarray) -> bool:
        eps = 1e-32
        m = self.m
        if m == 0:
            self.q0[:] = p
            self.c[0, :] = self.q0
            self.sqr_r[0] = 0.0
        else:
            self.v[m] = p - self.q0
            for i in range(1, m):
                self.a[m, i] = _seqsum(self.v[i] * self.v[m]) * (2 / self.z[i])
            for i in range(1, m):
                self.v[m] = self.v[m] - self.a[m, i] * self.v[i]
            self.z[m] = _seqsum(self.v[m] ** 2) * 2
            if self.z[m] < eps * self.current_sqr_r:
                return False
            e = _seqsum((p - self.c[m - 1]) ** 2, -self.sqr_r[m - 1])
            self.f[m] = e / self.z[m]
            self.c[m] = self.c[m - 1] + self.f[m] * self.v[m]
            self.sqr_r[m] = self.sqr_r[m - 1] + e * self.f[m] / 2
        self.current_c = self.c[m]
        self.current_sqr_r = float(self.sqr_r[m])
        self.m = m + 1
        self.s = self.m
        return True

    def pop(self) -> None:
        self.m -= 1


class _Miniball:
    """``moa.cluster.Miniball``, line for line but for the scans, which are vectorised:
    between two violations the ball does not change, so the first point with a
    positive excess is found in one pass."""

    def __init__(self, points: np.ndarray):
        self.P = np.asarray(points, dtype=float)
        self.d = self.P.shape[1]
        self.L = list(range(len(self.P)))
        self.B = _MiniballB(self.d)
        self.support_end = 0

    def build(self) -> tuple[np.ndarray, float]:
        self.B.reset()
        self.support_end = 0
        self._pivot_mb(len(self.L))
        return np.array(self.B.current_c), self.B.current_sqr_r

    def _mtf_mb(self, i: int) -> None:
        self.support_end = 0
        if self.B.m == self.d + 1:
            return
        k = 0
        while k < i:
            ex = self.B.excess(self.P[self.L[k:i]])
            hits = np.flatnonzero(ex > 0)
            if hits.size == 0:
                return
            j = k + int(hits[0])
            k = j + 1
            if self.B.push(self.P[self.L[j]]):
                self._mtf_mb(j)
                self.B.pop()
                self._move_to_front(j)

    def _move_to_front(self, j: int) -> None:
        if self.support_end <= j:
            self.support_end += 1
        self.L.insert(0, self.L.pop(j))

    def _pivot_mb(self, i: int) -> None:
        t = 1
        self._mtf_mb(t)
        old_sqr_r = -1.0
        while True:
            max_e, pivot = 0.0, 0
            if t < i:
                ex = self.B.excess(self.P[self.L[t:i]])
                k = int(np.argmax(ex))
                if ex[k] > 0:
                    max_e, pivot = float(ex[k]), t + k
            if max_e > 0:
                t = self.support_end
                if t == pivot:
                    t += 1
                old_sqr_r = self.B.current_sqr_r
                self.B.push(self.P[self.L[pivot]])
                self._mtf_mb(self.support_end)
                self.B.pop()
                self._move_to_front(pivot)
            if not (max_e > 0 and self.B.current_sqr_r > old_sqr_r):
                return


def _miniball_moa(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and squared radius by MOA's own algorithm, ported line for line. Kept to
    check ``_meb`` against: it pushes tens of thousands of times on a few hundred points
    in 30 dimensions, seconds per window in Python."""
    return _Miniball(points).build()


def _meb(points: np.ndarray, max_iter: int | None = None) -> tuple[np.ndarray, float]:
    """Centre and squared radius of the smallest enclosing ball, by the algorithm of
    Fischer, Gaertner and Kutz (ESA 2003), made for high dimensions.

    The ball is unique, so this returns MOA's ball up to rounding; it walks the centre
    towards the affine hull of a support set, adding the point that stops the walk and
    dropping a support point whose barycentric coefficient turns negative, in a few dozen
    vectorised steps. Falls back on ``_miniball_moa`` if it does not settle."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n == 1:
        return P[0].copy(), 0.0
    c = P[0].copy()
    d2 = ((P - c) ** 2).sum(axis=1)
    far = int(np.argmax(d2))
    if d2[far] == 0:
        return c, 0.0
    scale = math.sqrt(d2[far])
    support = [far]
    max_iter = max_iter or 20 * (n + P.shape[1])

    def affine(cc):
        # projection of cc onto aff(support), and its barycentric coefficients there
        s0 = P[support[0]]
        if len(support) == 1:
            return s0.copy(), np.ones(1)
        A = (P[support[1:]] - s0).T
        x = np.linalg.lstsq(A, cc - s0, rcond=None)[0]
        return s0 + A @ x, np.concatenate(([1.0 - x.sum()], x))

    for _ in range(max_iter):
        proj, lam = affine(c)
        v = proj - c
        dv = math.sqrt(float(v @ v))
        if dv <= 1e-13 * scale:
            # the centre is the circumcentre of the support: optimal unless a coefficient is negative
            k = int(np.argmin(lam))
            if lam[k] >= -1e-12 or len(support) == 1:
                r2 = float(((P - c) ** 2).sum(axis=1).max())
                return c, r2
            del support[k]
            continue
        r2 = float(((P[support[0]] - c) ** 2).sum())
        diff = P - c
        den = dv * dv - diff @ v
        free = np.ones(n, dtype=bool)
        free[support] = False
        ok = free & (den > 1e-12 * dv * math.sqrt(r2))
        t, stopper = 1.0, None
        if ok.any():
            bound = (r2 - (diff[ok] ** 2).sum(axis=1)) / (2 * den[ok])
            j = int(np.argmin(bound))
            if bound[j] < 1:
                t, stopper = max(float(bound[j]), 0.0), int(np.flatnonzero(ok)[j])
        c = c + t * v
        if stopper is not None:
            support.append(stopper)
    return _miniball_moa(P)


def _miniball(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of ``SphereCluster(points)``: the smallest enclosing ball, padded
    by ``MINIBALL_SLACK`` as ``Miniball.radius()`` does."""
    center, sqr_r = _meb(points)
    return center, (1 + MINIBALL_SLACK) * math.sqrt(sqr_r)


# --- the ground-truth analysis ------------------------------------------------------------

def _matrix(X) -> np.ndarray:
    """``X`` as a float matrix; a sequence of feature dicts is read in sorted key order."""
    if len(X) and isinstance(X[0], dict):
        keys = sorted({k for x in X for k in x})
        return np.asarray([[float(x.get(k, 0.0)) for k in keys] for x in X], dtype=float)
    A = np.asarray(X, dtype=float)
    return A.reshape(len(A), -1) if A.size else A.reshape(len(A), 0)


def _knn_avg(dist: np.ndarray, k: int) -> np.ndarray:
    """Per row, the mean of the ``k`` smallest finite entries (``getKnnInCluster`` then the
    average in ``getConnectionValue``); NaN where a row has none."""
    if dist.shape[1] == 0:
        return np.full(dist.shape[0], np.nan)
    kk = min(k, dist.shape[1])
    part = np.sort(np.partition(dist, kk - 1, axis=1)[:, :kk], axis=1)
    finite = np.isfinite(part)
    total = np.where(finite, part, 0.0)
    acc = total[:, 0].copy()
    for c in range(1, kk):  # ascending order, as Java's sorted list is summed
        acc = acc + total[:, c]
    cnt = finite.sum(axis=1)
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


class CMMWindow:
    """Ground-truth analysis of one window, built once and shared by every member scored on it.

    ``build`` does everything that depends only on the points and their classes -- the
    pairwise distances, one enclosing sphere per class, the merging of connected
    classes, every point's connectivity to its own class and its connection to every
    other one -- so scoring a partition (``score``, ``measures``) is a matching and a
    weighted sum.
    """

    def __init__(self):
        raise TypeError("use CMMWindow.build(X, y_true, ...)")

    @classmethod
    def build(cls, X, y_true, weights=None, *, knn: int = KNN_NEIGHBOURHOOD,
              tau_connection: float = TAU_CONNECTION,
              cluster_connection_max_points: int = CLUSTER_CONNECTION_MAX_POINTS,
              enable_class_merge: bool = True, enable_model_error: bool = True,
              lambda_missed: float = LAMBDA_MISSED,
              use_exp_connectivity: bool = False, lambda_conn_ref_x_value: float = 0.01,
              lambda_conn_x: float = 4.0,
              horizon: int | None = None, decay_threshold: float = DECAY_THRESHOLD,
              decay_rule: str = "batch") -> "CMMWindow":
        """``X``: (n, d) raw features in arrival order, oldest first (or a sequence of
        feature dicts). ``y_true``: class labels, None for a ground-truth noise point.
        ``weights``: one per point; None gives MOA's fading (``fading_weights`` over a
        horizon of ``horizon`` points, default the window). The keyword options are
        MOA's, with MOA's defaults (``use_exp_connectivity`` and its two lambdas are
        MOA's experimental connectivity, off by default)."""
        from scipy.spatial.distance import cdist

        self = object.__new__(cls)
        Xm = _matrix(X)
        n = len(Xm)
        self.n = n
        y = list(y_true)
        if len(y) != n:
            raise ValueError(f"X has {n} rows but y_true has {len(y)} labels")
        self.enable_model_error = enable_model_error
        self.lambda_missed = lambda_missed
        if weights is None:
            w = fading_weights(n, horizon, decay_threshold, decay_rule)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if len(w) != n:
                raise ValueError(f"{len(w)} weights for {n} points")
        self.weights = w
        self.noise = np.array([v is None for v in y], dtype=bool)
        if n == 0:
            self.n_classes = 0
            return self
        tau = tau_connection if enable_class_merge else 1.0
        lam_conn = -math.log(lambda_conn_ref_x_value) / math.log(2) / lambda_conn_x

        # classes in order of first appearance, as Clustering.classValues numbers them
        order: dict = {}
        for v in y:
            if v is not None and v not in order:
                order[v] = len(order)
        K = len(order)
        self.n_classes = K
        cls_of = np.array([order[v] if v is not None else -1 for v in y], dtype=int)
        members = [np.flatnonzero(cls_of == j) for j in range(K)]

        D = cdist(Xm, Xm) if n > 1 else np.zeros((n, n))
        # per (point, original class): its k smallest distances to that class, self excluded
        self_mask = np.eye(n, dtype=bool)
        knn_lists = np.full((n, K, knn), np.inf)
        for j, idx in enumerate(members):
            sub = D[:, idx].copy()
            sub[self_mask[:, idx]] = np.inf
            kk = min(knn, len(idx))
            if kk:
                knn_lists[:, j, :kk] = np.sort(np.partition(sub, kk - 1, axis=1)[:, :kk], axis=1)

        # one SphereCluster per class (Clustering(List<Instance>)), and who lies inside
        centers, radii = [], []
        for idx in members:
            c, r = _miniball(Xm[idx])
            centers.append(c)
            radii.append(r)
        inside = (cdist(Xm, np.asarray(centers)) <= np.asarray(radii)) if K else np.zeros((n, 0), bool)

        # GT0 clusters: original classes, then merged ones. ``kc[p, g]`` holds the k smallest
        # distances from point p to cluster g (self excluded), ``avg`` their mean (NaN: no
        # neighbour) and ``con`` the connection value getConnectionValue(p, g); merging two
        # clusters only changes the merged one's column.
        groups = [[j] for j in range(K)]          # original classes behind each cluster
        points = [list(m) for m in members]       # point ids, in MOA's order
        mean_avg = np.zeros(K)
        dev_avg = np.zeros(K)
        kc = knn_lists.copy()
        avg = np.stack([_knn_avg(kc[:, g, :], knn) for g in range(K)], axis=1) if K else np.zeros((n, 0))
        con = np.zeros((n, K))

        def calculate_knn(g):
            # GTCluster.calculateKnn; for a merged cluster the sums start from the values the
            # cluster had (MOA's accumulation, kept)
            pids = np.asarray(points[g], dtype=int)
            a = np.nan_to_num(avg[pids, g], nan=0.0)  # no neighbour: avgKnn stays 0
            m = _seqsum(a, mean_avg[g]) / len(pids)
            s2 = _seqsum(a * a, dev_avg[g]) / len(pids)
            var = s2 - m * m
            if var <= 0.0:
                var = 1e-50
            mean_avg[g], dev_avg[g] = m, math.sqrt(var)

        def connection_column(g):
            # getConnectionValue of every point to cluster g
            a = avg[:, g]
            upper = mean_avg[g] + dev_avg[g]
            with np.errstate(divide="ignore", invalid="ignore"):
                if use_exp_connectivity:
                    far = np.power(2.0, -lam_conn * (a - upper) / upper)
                else:
                    far = upper / a
            con[:, g] = np.where(np.isnan(a), 0.0, np.where(a < upper, 1.0, far))

        def top_mean(block):
            # calculateClusterConnection over the rows of ``block`` (one source cluster's points,
            # connectivity still 1.0): the mean of its best-connected points, per column
            if block.shape[0] == 0:
                return np.full(block.shape[1], np.nan)
            top = -np.sort(-block, axis=0)[:cluster_connection_max_points]
            acc = top[0].copy()
            for r in range(1, top.shape[0]):
                acc = acc + top[r]
            return acc / top.shape[0]

        for g in range(K):
            calculate_knn(g)
            connection_column(g)
        conn = np.zeros((K, K))
        for g in range(K):
            conn[g] = top_mean(1.0 * con[np.asarray(points[g], dtype=int)])
        np.fill_diagonal(conn, 1.0)

        while True:
            best, bi, bj = 0.0, -1, -1
            G = len(points)
            for c0 in range(G):
                for c1 in range(c0 + 1, G):
                    a, b = conn[c0, c1], conn[c1, c0]
                    v = float("nan") if a != a or b != b else min(a, b)  # Java's Math.min
                    if v > best:  # NaN never wins, as in Java
                        best, bi, bj = v, c0, c1
            if not (bi != -1 and best > tau):
                break
            # mergeCluster(bj) on cluster bi (bi < bj)
            points[bi] = points[bi] + points[bj]
            groups[bi] = groups[bi] + groups[bj]
            del points[bj], groups[bj]
            merged = np.sort(np.concatenate([kc[:, bi, :], kc[:, bj, :]], axis=1), axis=1)[:, :knn]
            kc = np.delete(kc, bj, axis=1)
            kc[:, bi, :] = merged
            avg = np.delete(avg, bj, axis=1)
            avg[:, bi] = _knn_avg(merged, knn)
            con = np.delete(con, bj, axis=1)
            mean_avg = np.delete(mean_avg, bj)
            dev_avg = np.delete(dev_avg, bj)
            calculate_knn(bi)
            connection_column(bi)
            conn = np.delete(np.delete(conn, bj, axis=0), bj, axis=1)
            for c in range(len(points)):
                conn[c, bi] = top_mean(1.0 * con[np.asarray(points[c], dtype=int)][:, [bi]])[0]
            conn[bi] = top_mean(1.0 * con[np.asarray(points[bi], dtype=int)])
            conn[bi, bi] = 1.0

        G = len(points)
        self.n_gt0 = G
        work = np.full(n, -1, dtype=int)
        for g, pids in enumerate(points):
            work[np.asarray(pids, dtype=int)] = g
        self.work = work
        gt0_inside = np.zeros((n, G), dtype=bool)
        for g in range(G):
            gt0_inside[:, g] = inside[:, groups[g]].any(axis=1)
        self.gt0_inside = gt0_inside

        # calculateGTPointQualities: connectivity to the own (merged) class; noise keeps 1.0
        connectivity = np.ones(n)
        nn = ~self.noise
        connectivity[nn] = con[nn, work[nn]]
        self.connectivity = connectivity
        self.con_to = con

        # mapGT: the class sizes, and how many points of each class the other's hull covers
        map_gt = np.zeros((G, G), dtype=int)
        for g in range(G):
            map_gt[g] = np.bincount(work[nn & gt0_inside[:, g]], minlength=G)
            map_gt[g, g] = int((work == g).sum())
        self.map_gt = map_gt

        # the class-analysis readings (CA Seperability, CA Noise, CA Model)
        self.ca_separability = G / K if K else float("nan")
        noise_ids = np.flatnonzero(self.noise)
        if noise_ids.size == 0:
            self.ca_noise = 1.0
        else:
            best_con = (np.maximum(con[noise_ids].max(axis=1), 0.0) if G
                        else np.zeros(noise_ids.size))
            self.ca_noise = 1 - _seqsum(best_con) / noise_ids.size
        other_ball = inside.copy()
        if K:
            other_ball[nn, cls_of[nn]] = False  # a ball of the point's own class is not an error
        self.ca_model = 1 - int(other_ball.any(axis=1).sum()) / n
        return self

    # --- scoring a partition ---------------------------------------------------------------

    def _assignment(self, y_pred, noise_labels) -> np.ndarray:
        if len(y_pred) != self.n:
            raise ValueError(f"{len(y_pred)} predictions for a window of {self.n} points")
        ids: dict = {}
        out = np.full(self.n, -1, dtype=int)
        for i, v in enumerate(y_pred):
            if v is None or v in noise_labels:
                continue
            out[i] = ids.setdefault(v, len(ids))
        return out

    def measures(self, y_pred: Sequence, noise_labels=frozenset()) -> dict:
        """Every reading of MOA's CMM collection for the partition ``y_pred`` (None, or a
        label in ``noise_labels``, leaves a point in no cluster). Values follow MOA,
        including its 1 for an empty denominator; ``score`` is the guarded one."""
        if self.n == 0:
            return dict.fromkeys(MEASURES)
        fc = self._assignment(y_pred, noise_labels)
        F = int(fc.max()) + 1 if (fc >= 0).any() else 0
        G = self.n_gt0
        noise, work, w, conn = self.noise, self.work, self.weights, self.connectivity

        # calculateMatching
        covered = fc >= 0
        map_fc = np.zeros((F, G), dtype=int)
        np.add.at(map_fc, (fc[covered & ~noise], work[covered & ~noise]), 1)
        sums_fc = map_fc.sum(axis=1)
        match = np.full(F, -1, dtype=int)
        for c in range(F):
            nz = np.flatnonzero(map_fc[c])
            if nz.size == 1:
                match[c] = nz[0]
                continue
            if sums_fc[c] == 0:
                continue
            err = np.maximum(map_fc[c][None, :] - self.map_gt, 0).sum(axis=1)
            idx = int(np.argmin(err))  # the first smallest, as the strict < in Java
            fits = np.flatnonzero(err == 0)
            if fits.size:
                idx = int(fits[np.argmax(map_fc[c, fits])])  # the first of the largest
            match[c] = idx

        # calculateError
        err = np.zeros(self.n)
        g_of = np.where(covered, match[np.maximum(fc, 0)], -1) if F else np.full(self.n, -1)
        missed_w = 1 - math.exp(-self.lambda_missed * 1.0)  # hull distance 1: not a sphere
        for i in range(self.n):
            if not covered[i]:
                if not noise[i]:
                    err[i] = missed_w * conn[i]
                continue
            g = g_of[i]
            if noise[i]:
                if g == -1:
                    err[i] = 1.0
                elif self.enable_model_error and self.gt0_inside[i, g]:
                    err[i] = MIN_ERROR
                else:
                    err[i] = 1 - self.con_to[i, g]
            elif g != work[i]:
                if g == -1:
                    err[i] = 1.0
                    continue
                if self.enable_model_error and self.gt0_inside[i, g]:
                    weight = 0.0
                else:
                    weight = 1 - self.con_to[i, g]
                err[i] = MIN_ERROR if weight == 0 else weight * conn[i]

        cw = conn * w
        total_max = _seqsum(cw)
        missed_max = _seqsum(np.where(noise, 0.0, cw))
        noise_max = _seqsum(np.where(noise, cw, 0.0))
        ew = err * w
        total = _seqsum(ew)
        missed = _seqsum(np.where(~covered & ~noise, w * err, 0.0))
        misplaced = _seqsum(np.where(covered & ~noise, ew, 0.0))
        noise_err = _seqsum(np.where(covered & noise, ew, 0.0))
        return {
            "cmm": 1 - total / total_max if total_max != 0 else 1.0,
            "cmm_basic": 1 - int((err != 0).sum()) / self.n,
            "cmm_missed": 1 - missed / missed_max if missed_max != 0 else 1.0,
            "cmm_misplaced": 1 - misplaced / missed_max if missed_max != 0 else 1.0,
            "cmm_noise": 1 - noise_err / noise_max if noise_max != 0 else 1.0,
            "ca_separability": self.ca_separability,
            "ca_noise": self.ca_noise,
            "ca_model": self.ca_model,
        }

    def score(self, y_pred: Sequence, noise_labels=frozenset()) -> float | None:
        """CMM of the partition ``y_pred``, in [0, 1] with 1 for no fault; None when the
        window is empty, when no point has a positive connectivity times weight (MOA
        would report 1), or when the computation gives NaN."""
        if self.n == 0:
            return None
        if _seqsum(self.connectivity * self.weights) == 0:
            return None
        value = self.measures(y_pred, noise_labels)["cmm"]
        return None if value is None or value != value else float(value)


def cmm(X, y_true, y_pred, weights=None, noise_labels=frozenset(), **options) -> float | None:
    """CMM of one partition of one window: ``CMMWindow.build(X, y_true, weights, **options)
    .score(y_pred, noise_labels)``. Build the window once when several partitions of it are
    scored."""
    return CMMWindow.build(X, y_true, weights, **options).score(y_pred, noise_labels)
