"""Label-free readings of a member's partition of one window, to select configurations by.

The ensemble ranks its members by the silhouette of their own predictions on the window
they have just processed (:func:`benchmark.tuning.window.batch_silhouette`). On real streams
that signal often fails to pick the configuration the labels would pick, so this module
offers candidates to log beside it, or to select by instead:

- ``silhouette``             the classic silhouette on raw features: what the tuner uses
- ``silhouette_std``         the same after standardising the window with its own mean and
                             standard deviation, so a few wide-ranged features cannot decide it
- ``dbcv`` / ``dbcv_std``    Density-Based Clustering Validation (Moulavi et al., SDM 2014) on
                             raw / standardised features: rewards dense clusters separated by
                             sparse regions, whatever their shape
- ``calinski_harabasz_std``, ``davies_bouldin_std``  on standardised features, the components
                             of ``rank_agg`` (Davies-Bouldin: lower is better)
- ``rank_agg``               per window, each member's mean rank among the members under
                             ``silhouette_std``, ``calinski_harabasz_std`` and
                             ``davies_bouldin_std``, rescaled to [0, 1] (1 = best on all three);
                             no single index is reliable on all data (Arbelaitz et al., 2013)
- ``stability``              the ARI between the member's predictions of the window while it
                             processed it (test-then-train) and its predictions of the same
                             points once it has learned them all: how much one window moves its
                             partition. It needs a second prediction pass, run by the ensemble on
                             a copy of the model, so it costs about one more prediction per point
                             and is computed only when asked for

Standardising uses only the window being scored, so no reading looks ahead. A member has no
reading (``None``) when it has no partition among its labelled points -- fewer than two
labels, or one label per point -- exactly where the silhouette is undefined. A point whose
predicted label is ``None`` is left out, as in ``batch_silhouette``.

Noise and sentinel labels. A clusterer may hand out a label that names no cluster: DenStream's
``-1`` when it holds no p-micro-cluster (``benchmark.algorithms.adapter.SENTINEL_LABELS``,
RIVER_BUGS.md BUG-6), or a noise label. ``member_signals`` and ``stability`` take the policy
for such points as ``noise`` (``NOISE_POLICIES``) and the labels it applies to as
``noise_labels`` (``(-1,)`` unless given; the caller knows its model's sentinels):

- ``"label"``     (the default, and what ``batch_silhouette``, the tuner's own silhouette, does)
                  an ordinary label: the noise points of a member form one cluster together, so
                  one cluster plus a few sentinel points is a two-cluster partition with a reading
- ``"drop"``      left out exactly as a ``None`` prediction is: every reading is taken on the
                  clustered points only, and nothing charges a member for the points it left
                  unclustered (one that clusters only its easiest points can read well)
- ``"singleton"`` each noise point is a cluster of its own, and each reading then applies its
                  own rule for a one-point cluster: the silhouette gives the point 0 (Rousseeuw,
                  1987) and keeps it as a neighbouring cluster of the others; DBCV counts it in
                  the total but in no cluster, which is how Moulavi et al. (Def. 7) and hdbscan's
                  ``validity_index`` treat noise; Calinski-Harabasz and Davies-Bouldin count it as
                  a cluster without spread

The default leaves every reading as it was before the policy existed. The policy only moves
readings of members whose predictions hold a noise label, and ``stability`` applies it to both
of its predictions. The silhouette the tuner selects by (``window.batch_silhouette``) is not
read here and keeps the ``"label"`` reading whatever ``noise`` says: comparing a signal under
another policy with it compares two policies, not two signals.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Readings computed for one member alone.
PER_MEMBER: tuple[str, ...] = (
    "silhouette", "silhouette_std", "dbcv", "dbcv_std", "calinski_harabasz_std", "davies_bouldin_std",
)
#: Readings that rank a member among the others of its window.
CROSS_MEMBER: tuple[str, ...] = ("rank_agg",)
#: Readings that need the member's model, not only its labels (the ensemble computes them).
MODEL_SIGNALS: tuple[str, ...] = ("stability",)
SIGNALS: tuple[str, ...] = PER_MEMBER + CROSS_MEMBER + MODEL_SIGNALS
#: Signals a tuner can select by: higher is better and a member without a partition has none.
SELECTABLE: tuple[str, ...] = ("silhouette", "silhouette_std", "dbcv", "dbcv_std", "rank_agg", "stability")
#: Readings where lower is better; every other reading is higher-is-better.
LOWER_IS_BETTER: tuple[str, ...] = ("davies_bouldin_std",)
#: ``rank_agg``'s components and whether higher is better for each.
RANK_COMPONENTS: tuple[tuple[str, bool], ...] = (
    ("silhouette_std", True), ("calinski_harabasz_std", True), ("davies_bouldin_std", False),
)
#: How a noise or sentinel label is read (module docstring); ``"label"`` is the default.
NOISE_POLICIES: tuple[str, ...] = ("label", "drop", "singleton")
#: The labels ``noise`` applies to unless the caller names others: DenStream's sentinel.
DEFAULT_NOISE_LABELS: tuple = (-1,)


def check_signals(names) -> tuple[str, ...]:
    names = tuple(names)
    unknown = [n for n in names if n not in SIGNALS]
    if unknown:
        raise ValueError(f"unknown signal(s) {unknown}; known: {SIGNALS}")
    return names


def check_noise(policy: str) -> str:
    if policy not in NOISE_POLICIES:
        raise ValueError(f"unknown noise policy {policy!r}; known: {NOISE_POLICIES}")
    return policy


class _NoisePoint:
    """The label of one noise point under ``noise="singleton"``: equal only to itself, so no two
    noise points, and no noise point and a real cluster, ever share a label."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<noise>"


def apply_noise_policy(labels: Sequence, noise: str = "label",
                       noise_labels: Sequence = DEFAULT_NOISE_LABELS) -> list:
    """``labels`` as the readings see them under ``noise``: unchanged for ``"label"``, a noise
    label replaced by ``None`` for ``"drop"`` and by a label of its own for ``"singleton"``.
    ``None`` stays ``None``. Public so that a caller can hand the same labels to a reading
    computed elsewhere (``window.batch_silhouette``) and keep one policy across signals."""
    check_noise(noise)
    if noise == "label":
        return list(labels)
    is_noise = frozenset(noise_labels)
    if noise == "drop":
        return [None if lab is not None and lab in is_noise else lab for lab in labels]
    return [_NoisePoint() if lab is not None and lab in is_noise else lab for lab in labels]


def per_member_needed(names) -> tuple[str, ...]:
    """The per-member readings ``names`` needs, ``rank_agg`` expanded into its components."""
    out = [n for n in check_signals(names) if n in PER_MEMBER]
    if "rank_agg" in names:
        out += [c for c, _ in RANK_COMPONENTS]
    return tuple(dict.fromkeys(out))


class WindowData:
    """The points of one window as matrices, built once and shared by every member.

    ``raw`` / ``std`` and their pairwise Euclidean distances are computed on first use.
    """

    def __init__(self, xs: Sequence[dict]):
        keys = sorted({k for x in xs for k in x})
        self.raw = np.asarray([[float(x.get(k, 0.0)) for k in keys] for x in xs], dtype=float)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def n_features(self) -> int:
        return self.raw.shape[1]

    def matrix(self, rep: str) -> np.ndarray:
        if rep == "raw":
            return self.raw
        if "std" not in self._cache:
            X = self.raw
            sd = X.std(axis=0)
            self._cache["std"] = np.where(sd > 0, (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
        return self._cache["std"]

    def distances(self, rep: str) -> np.ndarray:
        key = f"dist_{rep}"
        if key not in self._cache:
            from scipy.spatial.distance import pdist, squareform

            self._cache[key] = squareform(pdist(self.matrix(rep)))
        return self._cache[key]


def _partition(labels: Sequence) -> tuple[np.ndarray, np.ndarray] | None:
    """(kept point indices, dense label ids), or None when there is no partition."""
    keep = np.array([i for i, lab in enumerate(labels) if lab is not None], dtype=int)
    if len(keep) < 2:
        return None
    kept = [labels[i] for i in keep]
    ids = {lab: j for j, lab in enumerate(dict.fromkeys(kept))}
    if not 2 <= len(ids) < len(keep):
        return None
    return keep, np.asarray([ids[lab] for lab in kept])


def member_signals(window: WindowData, labels: Sequence, names, *, noise: str = "label",
                   noise_labels: Sequence = DEFAULT_NOISE_LABELS) -> dict[str, float | None]:
    """The per-member readings in ``names`` of one member's ``labels`` on ``window``, with the
    points labelled ``noise_labels`` read as ``noise`` says (module docstring)."""
    names = [n for n in check_signals(names) if n in PER_MEMBER]
    if check_noise(noise) != "label":
        labels = apply_noise_policy(labels, noise, noise_labels)
    part = _partition(labels)
    if part is None:
        return dict.fromkeys(names)
    keep, y = part
    out: dict[str, float | None] = {}
    for name in names:
        rep = "std" if name.endswith("_std") else "raw"
        if name.startswith("silhouette"):
            from sklearn.metrics import silhouette_score

            D = window.distances(rep)[np.ix_(keep, keep)]
            out[name] = float(silhouette_score(D, y, metric="precomputed"))
        elif name.startswith("dbcv"):
            D = window.distances(rep)[np.ix_(keep, keep)]
            out[name] = dbcv(D, y, window.n_features)
        elif name == "calinski_harabasz_std":
            from sklearn.metrics import calinski_harabasz_score

            out[name] = float(calinski_harabasz_score(window.matrix("std")[keep], y))
        elif name == "davies_bouldin_std":
            from sklearn.metrics import davies_bouldin_score

            out[name] = float(davies_bouldin_score(window.matrix("std")[keep], y))
    return out


def stability(before: Sequence, after: Sequence, *, noise: str = "label",
              noise_labels: Sequence = DEFAULT_NOISE_LABELS) -> float | None:
    """ARI between two predictions of the same points, over the points both labelled;
    None when ``before`` has no partition (as for every other reading). ``noise`` applies to
    both predictions: under ``"drop"`` a point that is noise in either is left out."""
    from sklearn.metrics import adjusted_rand_score

    if check_noise(noise) != "label":
        before = apply_noise_policy(before, noise, noise_labels)
        after = apply_noise_policy(after, noise, noise_labels)
    if _partition(before) is None:
        return None
    both = [i for i, (a, b) in enumerate(zip(before, after)) if a is not None and b is not None]
    if len(both) < 2:
        return None
    return float(adjusted_rand_score([before[i] for i in both], [after[i] for i in both]))


def rank_aggregate(readings: Sequence[dict]) -> list[float | None]:
    """``rank_agg`` of every member of a window, from their component readings.

    Only members with every component take part; each component ranks them (1 = best, ties
    averaged), and a member's score is ``(n - mean rank) / (n - 1)``: 1 when it is best on
    all three, 0 when it is worst on all three, 1 for a member alone. The others get None.
    """
    from scipy.stats import rankdata

    full = [i for i, r in enumerate(readings)
            if all(r.get(c) is not None and np.isfinite(r[c]) for c, _ in RANK_COMPONENTS)]
    out: list[float | None] = [None] * len(readings)
    n = len(full)
    if n == 0:
        return out
    if n == 1:
        out[full[0]] = 1.0
        return out
    ranks = []
    for comp, higher_better in RANK_COMPONENTS:
        v = np.array([readings[i][comp] for i in full], dtype=float)
        ranks.append(rankdata(-v if higher_better else v, method="average"))
    mean_rank = np.mean(ranks, axis=0)
    for i, m in zip(full, mean_rank):
        out[i] = float((n - m) / (n - 1))
    return out


# --- DBCV ------------------------------------------------------------------------------------

def _core_distances(D: np.ndarray, d: int) -> np.ndarray:
    """All-points core distance of every point of one cluster (Moulavi et al., Def. 1):
    ``(mean over the other points of dist ** -d) ** (-1/d)``, in log space so that a large
    ``d`` neither overflows nor underflows. Zero distances (duplicates, the point itself)
    are left out, as in the reference implementation; a point with only duplicates gets 0."""
    from scipy.special import logsumexp

    n = len(D)
    positive = D > 0
    terms = np.where(positive, -d * np.log(np.where(positive, D, 1.0)), -np.inf)
    has = positive.any(axis=1)
    out = np.zeros(n)
    lse = logsumexp(terms[has], axis=1) - np.log(n - 1)
    out[has] = np.exp(-lse / d)
    return out


def dbcv(D: np.ndarray, y: np.ndarray, d: int) -> float | None:
    """Density-Based Clustering Validation of labels ``y`` given pairwise distances ``D``.

    Per cluster: the minimum spanning tree of its mutual reachability distances; its
    density sparseness is the largest edge between internal nodes (nodes of degree > 1).
    Between clusters: the density separation is the smallest mutual reachability distance
    between their internal nodes. A cluster's validity is ``(sep - sparse) / max(sep,
    sparse)`` with ``sep`` its smallest separation, and DBCV is the size-weighted sum over
    clusters, in [-1, 1]. A cluster of one point cannot have a core distance and counts as
    noise (in the total, in no cluster); with fewer than two clusters left there is no value.

    Fidelity. No independent DBCV implementation is installed here (hdbscan, the ``dbcv``
    package and the authors' MATLAB code are all absent), so what is checked is against the
    paper's definitions (tests/test_tuning_signals.py): fixtures derived by hand, and a plain
    transcription of Defs. 1-7 evaluated on every minimum spanning tree of small clusters.
    Choices the paper leaves open, and how this function makes them:

    - zero distances (duplicates) are left out of the core distance, whose mean still divides
      by ``n_i - 1``, as hdbscan does; read literally, Def. 1 gives a point with a duplicate a
      core distance of 0. A point with only duplicates gets 0 here (hdbscan: infinite).
    - a minimum spanning tree without an internal edge has the sparseness of its largest edge
      (the authors' MATLAB code and hdbscan); one without an internal node (a two-point cluster)
      is read on all its points for the separation, where hdbscan fails.
    - fewer than two clusters: ``None`` (hdbscan returns nan).
    - noise is what the labels make of it: the module's ``noise`` policy decides; only
      ``"singleton"`` gives the paper's (and hdbscan's) handling, noise in the total and in no
      cluster. Under the default ``"label"`` a noise label is a cluster like any other.
    - ties. Mutual reachability distances tie by construction: every pair within the larger of
      their core distances weighs that core distance. The minimum spanning tree is then not
      unique, and which nodes are internal -- so the sparseness and the separation -- depends
      on how the tree algorithm breaks ties. The value here is the paper's for the tree scipy's
      ``minimum_spanning_tree`` returns; another tree of the same weight (Prim's, as hdbscan
      builds it) can give a very different value for the same partition: by more than 0.1 on
      the small test fixtures and, in a one-off check on 999-point windows of three Gaussian
      blobs (five tie rules), by up to 1.0 in 2 dimensions, 0.5 in 9 and 0.13 in 33 to 54.
      Members of one window are all read with the same rule, so their ranking may move less;
      that is not measured. Agreement with another implementation is to be expected only where
      the tree is unique; elsewhere both are one reading of a definition that does not fix
      the value.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    n_total = len(y)
    clusters = [np.flatnonzero(y == c) for c in np.unique(y)]
    clusters = [idx for idx in clusters if len(idx) >= 2]
    if len(clusters) < 2:
        return None
    core = np.zeros(n_total)
    for idx in clusters:
        core[idx] = _core_distances(D[np.ix_(idx, idx)], d)
    mreach = np.maximum(np.maximum(core[:, None], core[None, :]), D)
    scale = float(mreach.max()) or 1.0
    internal, sparse = [], []
    for idx in clusters:
        M = mreach[np.ix_(idx, idx)]
        # csgraph reads a zero as "no edge": shift every pair by a negligible amount
        tree = minimum_spanning_tree(M + 1e-12 * scale).tocoo()
        degree = np.bincount(np.concatenate([tree.row, tree.col]), minlength=len(idx))
        inner = np.flatnonzero(degree > 1)
        if len(inner) == 0:
            inner = np.arange(len(idx))
        inner_set = set(inner.tolist())
        edges = [w for r, c, w in zip(tree.row, tree.col, tree.data) if r in inner_set and c in inner_set]
        sparse.append(float(max(edges) if edges else tree.data.max()) - 1e-12 * scale)
        internal.append(idx[inner])
    # density separation: per cluster, the smallest mutual reachability distance from one of
    # its internal nodes to an internal node of any other cluster
    nodes = np.concatenate(internal)
    owner = np.concatenate([np.full(len(nodes_i), i) for i, nodes_i in enumerate(internal)])
    between = mreach[np.ix_(nodes, nodes)].copy()
    between[owner[:, None] == owner[None, :]] = np.inf
    nearest = between.min(axis=1)
    sep = np.full(len(clusters), np.inf)
    np.minimum.at(sep, owner, nearest)
    validity = 0.0
    for i, idx in enumerate(clusters):
        denom = max(sep[i], sparse[i])
        v = (sep[i] - sparse[i]) / denom if denom > 0 else 0.0
        validity += len(idx) / n_total * v
    return float(validity)
