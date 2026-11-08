"""Cluster identity within a window: transitions between two labellings, pair consistency by lag,
the label-free identity tracker and the one-state snapshot (``benchmark.tuning.cluster_identity``)."""

from __future__ import annotations

import random

import pytest

from benchmark.algorithms.adapter import make_adapter
from benchmark.tuning.cluster_identity import (
    IdentityTracker,
    pair_consistency,
    predict_mutates,
    replay_window,
    snapshot_labels,
    transitions,
)


# --- transitions ---------------------------------------------------------------------------------

def test_an_unchanged_labelling_is_no_event():
    t = transitions([0, 0, 1, 1], [0, 0, 1, 1])
    assert t.kind == "none"
    assert (t.survived, t.kept, t.renamed, t.reused_labels) == (2, 2, 0, 0)


def test_a_swap_of_names_is_a_rename_that_reuses_both_labels():
    t = transitions([0, 0, 1, 1], [1, 1, 0, 0])
    assert t.kind == "rename_only"
    assert (t.survived, t.kept, t.renamed, t.reused_labels) == (2, 0, 2, 2)


def test_a_split_keeps_the_parent_label_on_one_child_and_names_the_other_afresh():
    t = transitions([0, 0, 0, 0, 1, 1], [0, 0, 2, 2, 1, 1])
    assert t.kind == "structural"
    assert (t.split, t.survived, t.emerged) == (1, 1, 0)
    assert (t.kept, t.renamed, t.reused_labels, t.fresh_labels) == (2, 0, 0, 1)


def test_a_merge_keeps_one_parent_label_and_retires_the_other():
    t = transitions([0, 0, 1, 1, 2, 2], [0, 0, 0, 0, 2, 2])
    assert (t.absorbed, t.absorb_events, t.survived) == (2, 1, 1)
    assert (t.kept, t.renamed, t.reused_labels) == (2, 0, 0)
    assert t.retired_labels == 1


def test_a_cluster_whose_points_scatter_disappears_and_its_label_is_retired():
    before = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    after = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2]
    t = transitions(before, after)
    assert (t.disappeared, t.survived, t.reused_labels, t.retired_labels) == (1, 3, 0, 1)


def test_a_new_cluster_under_an_old_label_is_an_emergence_that_reuses_the_label():
    # A (label 0) survives renamed to 2; B (label 1) survives; one point of each opens a cluster named 0,
    # too small a slice of either to be a child of it
    before = [0] * 5 + [1] * 5
    after = [2, 2, 2, 2, 0, 1, 1, 1, 1, 0]
    t = transitions(before, after)
    assert (t.survived, t.emerged, t.split) == (2, 1, 0)
    assert (t.kept, t.renamed) == (1, 1)
    assert t.reused_labels == 1 and t.reused_on_emergence == 1 and t.fresh_labels == 1


def test_a_slice_large_enough_is_a_split_child_and_an_old_name_on_it_is_a_reuse():
    t = transitions([0, 0, 0, 1, 1, 1], [2, 2, 2, 1, 1, 0])
    assert (t.survived, t.split, t.emerged, t.offspring) == (1, 1, 0, 1)
    assert (t.kept, t.renamed, t.reused_labels) == (1, 1, 1)


def test_points_without_a_label_on_either_side_are_left_out():
    t = transitions([0, None, 1, 1], [0, 5, None, 1])
    assert t.n_points == 2 and t.kind == "none"


# --- pair consistency ----------------------------------------------------------------------------

def test_identical_labellings_are_consistent_at_every_lag():
    rows = pair_consistency([0, 0, 1, 1, 0], [3, 3, 4, 4, 3], edges=(1, 2, 4))
    assert [r["precision"] for r in rows] == [1.0, 1.0, 1.0]
    assert [r["recall"] for r in rows] == [1.0, 1.0, 1.0]


def test_one_label_naming_two_clusters_loses_precision_at_long_lags_only():
    rows = pair_consistency([0, 0, 0, 0], [0, 0, 1, 1], edges=(1, 2, 3))
    by_lo = {r["lag_lo"]: r for r in rows}
    assert by_lo[1]["n_same_labels"] == 3 and by_lo[1]["precision"] == pytest.approx(2 / 3)
    assert by_lo[2]["precision"] == 0.0 and by_lo[3]["precision"] == 0.0
    assert by_lo[1]["recall"] == 1.0 and by_lo[2]["recall"] is None


def test_pair_consistency_skips_unlabelled_points():
    rows = pair_consistency([0, None, 0], [1, 1, 1], edges=(1,))
    assert rows[0]["n_pairs"] == 1 and rows[0]["precision"] == 1.0


# --- stubs ---------------------------------------------------------------------------------------

class _Swapper:
    """Two fixed sides whose names swap after ``swap_after`` learned points."""

    def __init__(self, swap_after=5):
        self.n = 0
        self.swap_after = swap_after

    def learn_one(self, x):
        self.n += 1

    def predict_one(self, x):
        side = int(x["a"] > 2.5)
        return side if self.n < self.swap_after else 1 - side


class _Reuser:
    """Clusters at 0 and 5 named 0 and 1; once a point at 30 is learned, the cluster at 0 is evicted and
    its name 0 goes to the new cluster at 30 (what River's HPStream does), and the points at 0 fall to the
    cluster at 5."""

    def __init__(self):
        self.far = False

    def learn_one(self, x):
        self.far = self.far or x["a"] > 20

    def predict_one(self, x):
        if not self.far:
            return 0 if x["a"] < 2.5 else 1
        return 0 if x["a"] > 20 else 1


class _Toucher:
    """A prediction that moves the model: the label is the number of predictions made so far."""

    def __init__(self):
        self.calls = 0

    def learn_one(self, x):
        pass

    def predict_one(self, x):
        self.calls += 1
        return self.calls


def _sides(n):
    return [{"a": 5.0 * (i % 2)} for i in range(n)]


# --- the tracker ---------------------------------------------------------------------------------

def test_the_tracker_carries_identity_across_a_swap_of_names():
    r = replay_window(make_adapter("swap", _Swapper(), {}), _sides(10), tracker=IdentityTracker())
    assert r.sequential == [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    side0 = {r.tracked[i] for i in range(0, 10, 2)}
    side1 = {r.tracked[i] for i in range(1, 10, 2)}
    assert len(side0) == 1 and len(side1) == 1 and side0 != side1


def test_the_tracker_gives_a_reused_label_a_new_identity():
    xs = [{"a": 0.0}, {"a": 5.0}] * 3 + [{"a": 30.0}, {"a": 30.0}, {"a": 5.0}]
    r = replay_window(make_adapter("reuse", _Reuser(), {}), xs, tracker=IdentityTracker())
    # raw: the points at 0 and the cluster at 30 share label 0 within the window
    assert {r.sequential[i] for i in (0, 2, 4)} == {0} and r.sequential[7] == 0
    assert r.tracked[7] not in {r.tracked[i] for i in (0, 2, 4)}
    assert r.tracked[8] == r.tracked[1]                     # the cluster at 5 kept its identity


def test_the_tracker_never_changes_what_the_model_predicts():
    from river import cluster

    rng = random.Random(1)
    xs = [{"x0": rng.gauss(c, 0.3), "x1": rng.gauss(c, 0.3)} for c in [0, 5, 10] * 60]
    make = lambda: make_adapter("db", cluster.DBSTREAM(clustering_threshold=1.0, fading_factor=0.01,  # noqa: E731
                                                       cleanup_interval=10, intersection_factor=0.3,
                                                       minimum_weight=1.0), {})
    plain = make()
    raw = [plain.process_one(dict(x)) for x in xs]
    r = replay_window(make(), xs, tracker=IdentityTracker(), probe_every=7)
    assert r.sequential == raw


def test_river_hpstream_reuse_is_undone_by_the_tracker_as_the_stable_variant_undoes_it():
    from river import cluster
    from sklearn.metrics import adjusted_rand_score

    from benchmark.algorithms.hpstream_stable_labels import HPStreamStableLabels

    kw = dict(max_clusters=3, n_samples_init=60, stream_speed=10, spread_radius_factor=6.0,
              n_projected_dimensions=4, seed=0)

    def blob(rng, c):
        return {f"x{j}": c + rng.gauss(0, 0.3) for j in range(4)}

    def run(model):
        rng = random.Random(0)
        for i in range(60):
            model.learn_one(blob(rng, [0, 5, 10][i % 3]))
        xs = [blob(rng, c) for c in [0] * 3 + [5, 10] * 4 + [30] * 3]
        return replay_window(make_adapter("hp", model, {}), xs, tracker=IdentityTracker())

    river, stable = run(cluster.HPStream(**kw)), run(HPStreamStableLabels(**kw))
    assert adjusted_rand_score(stable.sequential, river.sequential) < 1.0
    assert adjusted_rand_score(stable.sequential, river.tracked) == 1.0


# --- the snapshot --------------------------------------------------------------------------------

def test_an_isolated_snapshot_labels_every_point_from_one_state_and_leaves_the_adapter_alone():
    adapter = make_adapter("touch", _Toucher(), {})
    xs = _sides(4)
    assert snapshot_labels(adapter, xs, isolate=True) == [1, 1, 1, 1]
    assert snapshot_labels(adapter, xs, isolate=False) == [1, 2, 3, 4]
    assert adapter.model.calls == 0 and adapter.n_fallback_predictions == 0


def test_only_rivers_own_denstream_is_known_to_move_on_predict():
    from river import cluster

    from benchmark.algorithms.denstream_fast import DenStreamPatchedFast

    assert predict_mutates(cluster.DenStream())
    assert not predict_mutates(DenStreamPatchedFast())
    assert not predict_mutates(cluster.DBSTREAM())


def test_replay_reads_an_interval_per_probe_and_a_post_snapshot():
    r = replay_window(make_adapter("swap", _Swapper(), {}), _sides(10), probe_every=5)
    assert [iv["t"] for iv in r.intervals] == [4, 9]
    assert r.intervals[0]["kind"] == "rename_only" and r.intervals[1]["kind"] == "none"
    assert r.post == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert r.pre == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


ALL_ALGOS = ("BIRCH", "CluStream", "DBSTREAM", "DenStream", "HPStream", "RepStream", "SOStream", "STREAMKMeans",
             "HPStreamStableLabels", "DenStreamPatchedFast")


@pytest.mark.parametrize("algo", ALL_ALGOS)
def test_tracking_and_probing_leave_every_protocol_model_predicting_as_it_would(algo):
    from conf_pipeline.run import _make_adapter_factory
    from conf_pipeline.spaces import start_params

    rng = random.Random(3)
    xs = [{"x0": rng.gauss(c, 0.4), "x1": rng.gauss(-c, 0.4)} for c in [0, 4, 8] * 70]
    make = lambda: _make_adapter_factory(algo, 60)(start_params(algo, 60, d=2), 5)  # noqa: E731
    plain, audited = make(), make()
    for a in (plain, audited):
        for x in xs[:60]:
            a.learn_only(dict(x))
    raw = [plain.process_one(dict(x)) for x in xs[60:]]
    r = replay_window(audited, xs[60:], tracker=IdentityTracker(), probe_every=25)
    assert r.sequential == raw
    assert len(r.tracked) == len(raw) and len(r.post) == len(raw)
