"""benchmark.tuning.signals: the label-free readings a tuner may rank configurations by."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_circles
from sklearn.metrics import silhouette_score

from conf_pipeline import _common  # noqa: F401  (puts the river tree on sys.path)
from benchmark.tuning import signals as S
from benchmark.tuning.window import UNDEFINED_SCORE, batch_silhouette


def _points(X):
    return [{f"f{j}": float(v) for j, v in enumerate(row)} for row in X]


def _blobs(seed=0, n=60, wide=False):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([rng.normal(c, 0.5, size=(n, 2)) for c in centers])
    y = np.repeat([0, 1, 2], n)
    if wide:
        X = np.column_stack([X, rng.normal(0, 1e4, size=len(X))])
    return X, y


def test_silhouette_reading_is_the_batch_silhouette_the_tuner_already_uses():
    X, y = _blobs()
    xs = _points(X)
    w = S.WindowData(xs)
    got = S.member_signals(w, list(y), ("silhouette",))["silhouette"]
    assert got == pytest.approx(batch_silhouette(xs, list(y))[0])
    assert got == pytest.approx(silhouette_score(X, y))


def test_standardised_silhouette_sees_through_one_wide_feature():
    X, y = _blobs(wide=True)
    w = S.WindowData(_points(X))
    by_wide = list((X[:, 2] > np.median(X[:, 2])).astype(int))   # a split along the wide feature
    right = S.member_signals(w, list(y), ("silhouette", "silhouette_std"))
    wrong = S.member_signals(w, by_wide, ("silhouette", "silhouette_std"))
    assert wrong["silhouette"] > right["silhouette"]             # raw distances reward the wrong split
    assert right["silhouette_std"] > wrong["silhouette_std"]


def test_dbcv_rewards_density_connected_clusters_that_the_silhouette_misses():
    X, y = make_circles(n_samples=300, factor=0.3, noise=0.03, random_state=0)
    w = S.WindowData(_points(X))
    halves = list((X[:, 0] > 0).astype(int))                     # a straight cut through both rings
    right = S.member_signals(w, list(y), ("silhouette", "dbcv"))
    cut = S.member_signals(w, halves, ("silhouette", "dbcv"))
    assert cut["silhouette"] > right["silhouette"]
    assert right["dbcv"] > 0.5 > cut["dbcv"]


def test_dbcv_is_near_one_for_separated_blobs_and_below_zero_for_random_labels():
    X, y = _blobs()
    w = S.WindowData(_points(X))
    assert S.member_signals(w, list(y), ("dbcv",))["dbcv"] > 0.8
    rnd = list(np.random.default_rng(1).integers(0, 3, len(y)))
    assert S.member_signals(w, rnd, ("dbcv",))["dbcv"] < 0.0


def test_dbcv_does_not_change_when_every_coordinate_is_rescaled():
    X, y = _blobs()
    a = S.member_signals(S.WindowData(_points(X)), list(y), ("dbcv",))["dbcv"]
    b = S.member_signals(S.WindowData(_points(X * 1e3)), list(y), ("dbcv",))["dbcv"]
    assert a == pytest.approx(b, rel=1e-6)


def test_dbcv_stays_finite_in_many_dimensions():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0, 1, (50, 60)), rng.normal(20, 1, (50, 60))]) * 3000.0
    y = [0] * 50 + [1] * 50
    v = S.member_signals(S.WindowData(_points(X)), y, ("dbcv",))["dbcv"]
    assert np.isfinite(v) and v > 0.5


def test_a_reading_is_none_without_a_partition_and_unlabelled_points_are_left_out():
    X, y = _blobs()
    w = S.WindowData(_points(X))
    names = ("silhouette", "silhouette_std", "dbcv", "dbcv_std")
    assert all(v is None for v in S.member_signals(w, [0] * len(y), names).values())
    assert all(v is None for v in S.member_signals(w, list(range(len(y))), names).values())
    some = [None if i % 7 == 0 else lab for i, lab in enumerate(y)]
    assert S.member_signals(w, some, ("dbcv",))["dbcv"] > 0.8


def test_rank_aggregate_averages_the_ranks_of_three_internal_indices():
    # member 0 best on every index, member 2 worst on every index; member 1 has no partition
    readings = [
        {"silhouette_std": 0.8, "calinski_harabasz_std": 900.0, "davies_bouldin_std": 0.3},
        {"silhouette_std": None, "calinski_harabasz_std": None, "davies_bouldin_std": None},
        {"silhouette_std": 0.1, "calinski_harabasz_std": 10.0, "davies_bouldin_std": 2.0},
        {"silhouette_std": 0.5, "calinski_harabasz_std": 50.0, "davies_bouldin_std": 0.2},
    ]
    scores = S.rank_aggregate(readings)
    assert scores[1] is None
    assert scores[0] > scores[3] > scores[2]
    # member 0 ranks 1, 1 and 2 (Davies-Bouldin: lower is better, member 3 has 0.2) among 3
    assert scores[0] == pytest.approx((3 - 4 / 3) / 2)
    assert scores[3] == pytest.approx((3 - 5 / 3) / 2) and scores[2] == pytest.approx(0.0)
    assert 0.0 <= min(s for s in scores if s is not None) and max(s for s in scores if s is not None) <= 1.0


def test_unknown_signal_names_are_refused():
    with pytest.raises(ValueError, match="unknown signal"):
        S.check_signals(("silhouette", "entropy"))
    assert UNDEFINED_SCORE == -1.0


def test_stability_is_the_agreement_of_two_predictions_of_the_same_points():
    before = [0, 0, 1, 1, 2, 2, None]
    assert S.stability(before, [5, 5, 7, 7, 9, 9, 9]) == pytest.approx(1.0)   # same partition, other ids
    assert S.stability(before, [0, 1, 0, 1, 0, 1, 0]) < 0.2
    assert S.stability([0] * 6, [0, 1, 0, 1, 0, 1]) is None                  # no partition to keep
    assert "stability" in S.SELECTABLE and "stability" not in S.per_member_needed(("stability",))


# --- noise and sentinel labels ----------------------------------------------------------------

ALL_PER_MEMBER = ("silhouette", "silhouette_std", "dbcv", "dbcv_std", "calinski_harabasz_std",
                  "davies_bouldin_std")


def _blobs_with_far_noise(n_noise=6):
    """Three blobs and ``n_noise`` points far from them and from each other, labelled -1."""
    X, y = _blobs()
    far = np.column_stack([1e4 + 1e3 * np.arange(n_noise), np.full(n_noise, -1e4)])
    return np.vstack([X, far]), list(y) + [-1] * n_noise


def test_by_default_a_noise_label_is_an_ordinary_cluster_label():
    X, labels = _blobs_with_far_noise()
    w = S.WindowData(_points(X))
    assert S.member_signals(w, labels, ALL_PER_MEMBER) == S.member_signals(w, labels, ALL_PER_MEMBER, noise="label")
    one_cluster_and_noise = [0 if lab != -1 else -1 for lab in labels]
    assert S.member_signals(w, one_cluster_and_noise, ("silhouette",))["silhouette"] is not None
    assert S.member_signals(w, one_cluster_and_noise, ("silhouette",), noise="drop")["silhouette"] is None


def test_dropping_noise_reads_the_noise_points_as_unlabelled():
    X, labels = _blobs_with_far_noise()
    w = S.WindowData(_points(X))
    as_none = [None if lab == -1 else lab for lab in labels]
    assert S.member_signals(w, labels, ALL_PER_MEMBER, noise="drop") == S.member_signals(w, as_none, ALL_PER_MEMBER)


def test_singleton_noise_scores_each_noise_point_as_a_cluster_of_its_own():
    """Far noise points are nobody's nearest cluster, so the silhouette of the blob points is
    unchanged and each noise point adds a 0; DBCV counts them in the total and in no cluster
    (Moulavi et al., 2014, Def. 7). Both readings are the dropped ones scaled by the share of
    clustered points."""
    X, labels = _blobs_with_far_noise()
    w = S.WindowData(_points(X))
    share = sum(lab != -1 for lab in labels) / len(labels)
    drop = S.member_signals(w, labels, ("silhouette", "dbcv"), noise="drop")
    single = S.member_signals(w, labels, ("silhouette", "dbcv"), noise="singleton")
    assert single["silhouette"] == pytest.approx(share * drop["silhouette"], rel=1e-9)
    assert single["dbcv"] == pytest.approx(share * drop["dbcv"], rel=1e-9)


def test_which_labels_are_noise_is_the_callers_to_say():
    labels = [0, 0, "noise", 1, None, -1]
    assert S.apply_noise_policy(labels, "label") == labels
    assert S.apply_noise_policy(labels, "drop", noise_labels=("noise",)) == [0, 0, None, 1, None, -1]
    single = S.apply_noise_policy(labels, "singleton")
    assert single[:5] == labels[:5] and single[5] not in labels
    two = S.apply_noise_policy([-1, -1, 0], "singleton")
    assert two[0] != two[1] and two[2] == 0
    with pytest.raises(ValueError, match="noise policy"):
        S.check_noise("ignore")
    with pytest.raises(ValueError, match="noise policy"):
        S.member_signals(S.WindowData(_points(_blobs()[0])), list(_blobs()[1]), ("dbcv",), noise="ignore")


def test_stability_follows_the_same_noise_policy():
    before = [0, 0, 1, 1, -1, -1]
    after = [0, 0, 1, 1, 0, 1]
    assert S.stability(before, after) == S.stability(before, after, noise="label") < 1.0
    assert S.stability(before, after, noise="drop") == pytest.approx(1.0)


# --- DBCV against the paper's definitions -----------------------------------------------------

def _paper_dbcv_values(X, y, noise=None):
    """DBCV transcribed from Moulavi et al. (SDM 2014), Defs. 1-7, in plain Python, for EVERY
    minimum spanning tree of every cluster (by enumeration, so only for clusters of a few
    points). Returns the set of values. A second reading of the paper by the same hands, not an
    independent implementation: it checks the module's arithmetic and conventions, and that the
    definition leaves the value open when spanning trees tie."""
    import itertools
    import math

    n, d = X.shape
    dist = lambda a, b: math.dist(X[a], X[b])  # noqa: E731
    groups: dict = {}
    for i, c in enumerate(y):
        if c != noise:
            groups.setdefault(c, []).append(i)
    groups = [m for m in groups.values() if len(m) >= 2]
    core = {}
    for m in groups:  # Def. 1, zero distances (duplicates) left out as the reference code does
        for p in m:
            terms = [dist(p, q) ** -d for q in m if q != p and dist(p, q) > 0]
            core[p] = (sum(terms) / (len(m) - 1)) ** (-1.0 / d) if terms else 0.0
    mr = lambda a, b: max(core[a], core[b], dist(a, b))  # noqa: E731  (Def. 2)
    options = []
    for m in groups:  # Def. 3: every spanning tree of minimum weight
        trees = []
        for combo in itertools.combinations(itertools.combinations(m, 2), len(m) - 1):
            root = {p: p for p in m}

            def find(p):
                while root[p] != p:
                    p = root[p]
                return p

            spanning = True
            for a, b in combo:
                ra, rb = find(a), find(b)
                if ra == rb:
                    spanning = False
                    break
                root[ra] = rb
            if spanning:
                trees.append((sum(mr(a, b) for a, b in combo), combo))
        best = min(wt for wt, _ in trees)
        per_tree = []
        for wt, combo in trees:
            if not math.isclose(wt, best, rel_tol=1e-12):
                continue
            degree = {p: 0 for p in m}
            for a, b in combo:
                degree[a] += 1
                degree[b] += 1
            inner = tuple(p for p in m if degree[p] > 1) or tuple(m)
            internal_edges = [mr(a, b) for a, b in combo if degree[a] > 1 and degree[b] > 1]
            # Def. 4; without an internal edge, the largest edge (the authors' MATLAB code)
            per_tree.append((inner, max(internal_edges) if internal_edges else max(mr(a, b) for a, b in combo)))
        options.append(per_tree)
    values = set()
    for choice in itertools.product(*options):
        total = 0.0
        for i, m in enumerate(groups):
            inner, sparse = choice[i]
            sep = min(mr(a, b) for j in range(len(groups)) if j != i for a in inner for b in choice[j][0])  # Def. 5
            total += len(m) / n * (sep - sparse) / max(sep, sparse)  # Defs. 6 and 7
        values.add(round(total, 12))
    return values


def _dbcv_of(X, labels, noise="label"):
    return S.member_signals(S.WindowData(_points(X)), list(labels), ("dbcv",), noise=noise)["dbcv"]


def test_dbcv_matches_a_fixture_derived_by_hand():
    """Two rows of four points, x = 0..3 and x = 5..8 (d = 2). Core distances (Def. 1):
    sqrt(108/49) at the ends, 2/sqrt(3) inside; the only minimum spanning tree is the chain, its
    internal edge (1, 2) weighs 2/sqrt(3) (Def. 4); the separation is read between internal nodes
    only (Def. 5): points 2 and 6, at 4. DBCV = (4 - 2/sqrt(3)) / 4. Reading the separation
    between all points instead (3 and 5, at 2) would give (2 - 2/sqrt(3)) / 2 = 0.42."""
    X = np.column_stack([[0, 1, 2, 3, 5, 6, 7, 8], np.zeros(8)]).astype(float)
    y = [0] * 4 + [1] * 4
    expected = (4 - 2 / np.sqrt(3)) / 4
    assert _paper_dbcv_values(X, y) == {round(expected, 12)}
    assert _dbcv_of(X, y) == pytest.approx(expected, abs=1e-9)


def test_dbcv_without_an_internal_edge_or_node_follows_the_reference_code():
    """Three points per cluster (d = 1): the middle point is the only internal node, so there
    is no internal edge and the sparseness is the largest edge, 4/3 (the authors' MATLAB code,
    and hdbscan). Two points per cluster: no internal node at all, so every point is read for
    the separation (hdbscan fails there). Neither case is defined by the paper."""
    X3 = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    assert _dbcv_of(X3, [0, 0, 0, 1, 1, 1]) == pytest.approx((10 - 4 / 3) / 10, abs=1e-9)
    X2 = np.array([[0.0], [1.0], [10.0], [11.0]])
    assert _dbcv_of(X2, [0, 0, 1, 1]) == pytest.approx(8 / 9, abs=1e-9)


def test_dbcv_is_the_papers_value_for_one_of_the_minimum_spanning_trees():
    """On small random partitions the module's DBCV is always one of the values the paper's
    definitions give over the tied minimum spanning trees; which one depends on how scipy's
    ``minimum_spanning_tree`` breaks ties. Mutual reachability distances tie by construction
    (every pair inside a core distance weighs that core distance), and on this fixture the
    tied trees disagree by more than 0.1: the definition does not fix the value."""
    rng = np.random.default_rng(0)
    spreads = []
    for trial in range(12):
        k = 2 + trial % 2
        X = np.vstack([rng.normal(rng.uniform(-3, 3, 2), 1.0, (5, 2)) for _ in range(k)])
        y = np.repeat(np.arange(k), 5)
        if trial % 3 == 2:
            y = rng.permutation(y)
        values = _paper_dbcv_values(X, list(y))
        ours = _dbcv_of(X, y)
        assert min(abs(ours - v) for v in values) < 1e-9, (trial, ours, sorted(values))
        spreads.append(max(values) - min(values))
    assert max(spreads) > 0.1


def test_dbcv_with_singleton_noise_is_the_papers_handling_of_noise():
    X = np.column_stack([[0, 1, 2, 3, 5, 6, 7, 8, 30], np.zeros(9)]).astype(float)
    y = [0] * 4 + [1] * 4 + [-1]
    expected = _paper_dbcv_values(X, y, noise=-1)
    assert len(expected) == 1
    assert _dbcv_of(X, y, noise="singleton") == pytest.approx(expected.pop(), abs=1e-9)


def test_dbcv_agrees_with_hdbscan_where_the_tree_is_unique():
    """Runs only where ``hdbscan`` is installed (it is not in this repo's environment): its
    ``validity_index`` follows the authors' MATLAB code and reads label -1 as noise. Compared on
    fixtures whose minimum spanning trees are unique, where the definitions leave no choice."""
    validity = pytest.importorskip("hdbscan.validity")
    cases = [
        (np.column_stack([[0, 1, 2, 3, 5, 6, 7, 8], np.zeros(8)]).astype(float), [0] * 4 + [1] * 4),
        (np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]]), [0, 0, 0, 1, 1, 1]),
        (np.column_stack([[0, 1, 2, 3, 5, 6, 7, 8, 30], np.zeros(9)]).astype(float), [0] * 4 + [1] * 4 + [-1]),
    ]
    for X, y in cases:
        assert len(_paper_dbcv_values(X, y, noise=-1)) == 1
        ref = validity.validity_index(X, np.asarray(y, dtype=np.intp))
        assert _dbcv_of(X, y, noise="singleton") == pytest.approx(ref, abs=1e-9)
