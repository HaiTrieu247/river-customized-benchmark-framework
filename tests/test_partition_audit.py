"""The sequential-partition audit: renames, real partition changes and mutating predictions told apart."""

from __future__ import annotations

from benchmark.algorithms.adapter import make_adapter
from benchmark.tuning.partition_audit import audit_window, state_digest


class _Swapper:
    """Two fixed centres whose names swap after ``swap_after`` points are learned: the partition never
    changes, only its labels (a pure rename that reuses both labels)."""

    def __init__(self, swap_after=5):
        self.n = 0
        self.swap_after = swap_after

    def learn_one(self, x):
        self.n += 1

    def predict_one(self, x):
        side = int(x["a"] > 2.5)
        return side if self.n < self.swap_after else 1 - side


class _Grower:
    """One cluster until ``split_after`` points are learned, then two: a real change of partition."""

    def __init__(self, split_after=5):
        self.n = 0
        self.split_after = split_after

    def learn_one(self, x):
        self.n += 1

    def predict_one(self, x):
        return 0 if self.n < self.split_after else int(x["a"] > 2.5)


class _Toucher(_Grower):
    """A prediction that moves the model: every call counts as a learned point."""

    def predict_one(self, x):
        self.n += 1
        return super().predict_one(x)


def _xs(n=10):
    return [{"a": 5.0 * (i % 2)} for i in range(n)]


def test_a_swap_of_names_is_a_rename_that_reuses_labels_not_a_partition_change():
    adapter = make_adapter("swap", _Swapper(), {})
    a = audit_window(adapter, _xs())
    assert (a.n_rename_events, a.n_rename_reuse_events, a.n_partition_change_events) == (1, 1, 0)
    assert a.n_renumbering_events == 1 and a.renumbered_points == 10
    assert a.events["rename"] == [4] and a.share_after_first_rename == 0.5
    # the sequential labels give one name to both clusters across the swap: that partition mixes them
    assert a.sequential == [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    assert a.ari_sequential_vs_snapshot < 0 and a.raw_agreement_sequential_vs_snapshot == 0.5
    assert a.n_predict_mutations == 0
    assert adapter.model.n == 0  # the caller's model is never touched


def test_a_new_cluster_is_a_partition_change_and_no_rename():
    a = audit_window(make_adapter("grow", _Grower(), {}), _xs())
    assert (a.n_rename_events, a.n_partition_change_events, a.n_renumbering_events) == (0, 1, 0)
    assert a.share_after_first_rename is None


def test_a_prediction_that_moves_the_model_is_counted_and_kept_apart_from_learning():
    a = audit_window(make_adapter("touch", _Toucher(split_after=3), {}), _xs())
    assert a.n_predict_mutations == 1 and a.n_partition_change_events == 0
    assert state_digest(_Toucher()) is not None


def test_denstream_renumbers_its_macro_clusters_when_an_earlier_p_micro_cluster_goes():
    """Structural repro (docs/SEQUENTIAL_PARTITION_AUDIT.md): ``predict_one`` numbers the macro clusters
    from 0 in ``p_micro_clusters`` order on every call, so removing the first p-micro-cluster (what the
    periodic removal of ``learn_one`` does) renames every later cluster, though they did not move."""
    from river import cluster

    model = cluster.DenStream(decaying_factor=0.01, beta=0.75, mu=2, epsilon=0.5, n_samples_init=30)
    pts = [{"a": float(c) + 0.01 * (i % 5)} for c in (0, 10, 20) for i in range(10)]
    for x in pts:
        model.learn_one(dict(x))
    far = {"a": 20.02}
    before = model.predict_one(far)
    first = next(iter(model.p_micro_clusters))
    first_center = model.p_micro_clusters[first].calc_center(model.timestamp)["a"]
    del model.p_micro_clusters[first]          # the cluster around 0 is gone, the one around 20 is not
    after = model.predict_one(far)
    assert first_center < 5.0 and before != after and after == before - 1


# --- label handoffs: a label passing to other points while its own stay together -------------------

from benchmark.tuning.partition_audit import label_handoffs  # noqa: E402


def test_a_handoff_is_a_label_that_moves_off_points_that_stay_together_and_never_a_merge_split_or_new_label():
    assert label_handoffs([0, 0, 1, 1], [1, 1, 0, 0]) == [(0, [0, 1], [2, 3], 1), (1, [2, 3], [0, 1], 0)]
    assert label_handoffs([0, 0, 1, 1], [0, 0, 0, 0]) == []        # a merge: 0 keeps its points
    assert label_handoffs([0, 0, 0, 0], [0, 0, 1, 1]) == []        # a split: 0 keeps some
    assert label_handoffs([0, 0, 1, 1], [0, 0, 2, 2]) == []        # a fresh label
    # a cluster gone, its points absorbed, its label reused by others: the reuse is a handoff only when
    # the points it named stay together (here they join label 1, so label 0's move is one)
    assert label_handoffs([0, 1, 2], [1, 1, 0]) == [(0, [0], [2], 1)]


def _probe(model, probes):
    import copy

    m = copy.deepcopy(model)                       # a prediction can move a River model (BUG-7)
    return [m.predict_one(dict(p)) for p in probes]


def _blobs(rng, centres, n, sd=0.05):
    return [{"x0": c + rng.gauss(0, sd), "x1": c + rng.gauss(0, sd)} for c in (centres[i % len(centres)] for i in range(n))]


def _first_handoff(model, stream, probes):
    """The first step of ``stream`` after which the probes' labels hand a label off: (step, before, after)."""
    prev = _probe(model, probes)
    for t, x in enumerate(stream):
        model.learn_one(dict(x))
        cur = _probe(model, probes)
        if label_handoffs(prev, cur):
            return t, prev, cur
        prev = cur
    return None


PROBES4 = [{"x0": c, "x1": c} for c in (0, 10, 20, 30)]


def test_denstream_prunes_an_earlier_p_micro_cluster_and_the_clusters_that_stay_change_label():
    """River's DenStream, within what one window of 1 000 points spans: the p-micro-cluster around 0
    fades and is pruned by ``learn_one``; the ones around 10 and 20 are the same objects before and
    after, and each takes a lower number, so label 1 moves from the cluster at 10 to the one at 20."""
    import random

    from river import cluster

    model = cluster.DenStream(decaying_factor=0.05, beta=0.75, mu=2, epsilon=1.0, n_samples_init=90, stream_speed=1)
    rng = random.Random(0)
    for x in _blobs(rng, [0, 10, 20], 90):
        model.learn_one(dict(x))
    before = dict(model.p_micro_clusters)
    near = lambda pmc, c: abs(pmc.calc_center(model.timestamp)["x0"] - c) < 1   # noqa: E731
    found = _first_handoff(model, _blobs(rng, [10, 20], 200), PROBES4)
    assert found is not None
    t, prev, cur = found
    assert (prev, cur) == ([0, 1, 2, 2], [0, 0, 1, 1])
    kept = [obj for obj in model.p_micro_clusters.values()]
    assert all(any(obj is old for old in before.values()) for obj in kept)       # nothing new, only a removal
    assert not any(near(obj, 0) for obj in kept) and len(kept) == len(before) - 1
    assert label_handoffs(prev, cur) == [(1, [1], [2, 3], 0)]


def test_dbstream_cleans_up_a_micro_cluster_and_the_macro_clusters_that_stay_change_label():
    import random

    from river import cluster

    model = cluster.DBSTREAM(clustering_threshold=1.0, fading_factor=0.05, cleanup_interval=10,
                             intersection_factor=0.3, minimum_weight=1.0)
    rng = random.Random(0)
    for x in _blobs(rng, [0, 10, 20], 90):
        model.learn_one(dict(x))
    stream = _blobs(rng, [10, 20], 60) + _blobs(rng, [30], 10) + _blobs(rng, [10, 20, 30], 30)
    before = None
    prev = _probe(model, PROBES4)
    for t, x in enumerate(stream):
        before = dict(model._micro_clusters)
        model.learn_one(dict(x))
        cur = _probe(model, PROBES4)
        if label_handoffs(prev, cur):
            break
        prev = cur
    assert (prev, cur) == ([0, 1, 2, 3], [0, 0, 1, 2])
    survivors = [obj for obj in model._micro_clusters.values() if any(obj is old for old in before.values())]
    assert len(survivors) >= 3                                    # the clusters at 10, 20, 30 stayed
    assert [h[0] for h in label_handoffs(prev, cur)] == [1, 2]


def test_birch_under_rivers_defaults_renames_its_clusters_with_the_partition_unchanged():
    """``threshold = 0`` (River's default) and the protocol's ``leaf_capacity = 50``: every point is a
    leaf entry of its own and labels are positions in the leaves, so an entry filed earlier in the tree
    shifts every later label. The partition of the probes does not move; a label passes to other points."""
    import random

    from river import cluster

    from benchmark.tuning.partition_audit import _ari

    probes = [{"x0": c, "x1": c} for c in (-10, 0, 5, 10, 20, 30)]
    model = cluster.BIRCH(threshold=0.0, leaf_capacity=50, distance_measure="d2", threshold_measure="diameter")
    rng = random.Random(0)
    for x in _blobs(rng, [0, 10, 20], 90):
        model.learn_one(dict(x))
    found = _first_handoff(model, _blobs(rng, [10, 20], 40), probes)
    assert found is not None
    t, prev, cur = found
    assert _ari(prev, cur) == 1.0                                  # the same partition, other names
    assert label_handoffs(prev, cur)


def test_clustream_names_its_macro_clusters_afresh_at_every_offline_k_means():
    import random

    from river import cluster

    model = cluster.CluStream(max_micro_clusters=10, time_gap=20, n_macro_clusters=3, seed=0, halflife=0.5)
    rng = random.Random(0)
    for x in _blobs(rng, [0, 10, 20], 90):
        model.learn_one(dict(x))
    stream = _blobs(rng, [10, 20], 60) + _blobs(rng, [30], 10) + _blobs(rng, [10, 20, 30], 30)
    found = _first_handoff(model, stream, PROBES4)
    assert found is not None
    t, prev, cur = found
    assert (prev, cur) == ([2, 1, 1, 1], [0, 2, 1, 1])
    assert (90 + t + 1) % 20 == 0                                  # at a time_gap boundary: a new k-means
    assert label_handoffs(prev, cur) == [(2, [0], [1], 0)]         # the cluster at 0 stays, label 2 leaves it


def test_sostream_never_gives_one_label_to_two_micro_clusters():
    """The persistent-identity side: SOStream draws labels from ``_next_id`` and merges keep a label on
    the surviving micro-cluster, so a label never names a second micro-cluster object."""
    import random

    from river import cluster

    model = cluster.SOStream(merge_threshold=2.0, min_pts=2)
    rng = random.Random(0)
    owner: dict = {}
    for i, x in enumerate(_blobs(rng, [0, 3, 10, 20], 400, sd=1.0)):
        model.learn_one(dict(x))
        for k, obj in model._micro_clusters.items():
            assert owner.setdefault(k, obj) is obj, f"point {i}: label {k} named another micro-cluster"
    assert len(owner) > len(model._micro_clusters)                 # micro-clusters were merged away
