"""CluStreamPyramidal closes RIVER_BUGS.md GAP-4, and changes nothing online.

GAP-4: river's ``CluStream`` docstring says the micro-clusters "are stored at
snapshots in time following a pyramidal pattern" so that statistics can be
recalled "from different time horizons", but the class keeps only the current
micro-cluster set and ``predict_one`` takes no horizon.

The tests that assert the *upstream* state are deliberate: they prove the gap
exists in the vendored tree, so the pyramidal-side assertions are measuring
something new rather than re-describing river.

Feature keys are ints throughout. River's KMeans iterates a set of feature keys
(RIVER_BUGS.md BUG-2) and int hashes do not depend on PYTHONHASHSEED, so these
tests are reproducible even without the pinned hash seed the suite runs under.
"""

from __future__ import annotations

import ast
import inspect
import math
import pickle
import random

import pytest
from sklearn.metrics import adjusted_rand_score

# Import order matters: this module puts the sibling river tree on sys.path.
from benchmark.algorithms.clustream_pyramidal import CluStreamPyramidal
from benchmark.algorithms.registry import ALGORITHMS, accepts_seed
from river import cluster
from river.cluster import clustream as river_clustream
from river.cluster.clustream import CluStreamMicroCluster


def _blobs(centers, n, spread, seed):
    """``n`` points drawn uniformly from isotropic Gaussian blobs, with labels."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        c = rng.randrange(len(centers))
        out.append(({f: v + rng.gauss(0.0, spread) for f, v in enumerate(centers[c])}, c))
    return out


def _var_state(v):
    return (v.n, v.mean.get(), v.get())


def _mc_state(mc):
    return _var_state(mc.var_time), tuple((f, _var_state(v)) for f, v in mc.var_x.items())


def _state(model):
    """Every online statistic river CluStream keeps, as comparable values."""
    mcs = {k: _mc_state(mc) for k, mc in model.micro_clusters.items()}
    centers = {c: dict(center) for c, center in model.centers.items()}
    return model._timestamp, model._initialized, mcs, centers


def _code_without_docstrings(module):
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree).lower()


# --- upstream state ---------------------------------------------------------
def test_gap4_upstream_docstring_promises_a_pyramid_the_code_does_not_keep():
    doc = cluster.CluStream.__doc__.lower()
    assert "snapshots in time" in doc and "pyramidal pattern" in doc
    assert "different time horizons" in doc

    code = _code_without_docstrings(river_clustream)
    assert "snapshot" not in code and "pyramid" not in code and "horizon" not in code

    m = cluster.CluStream(max_micro_clusters=5, time_gap=10, seed=0)
    for x, _ in _blobs([(0.0, 0.0), (1.0, 1.0)], 300, 0.2, seed=0):
        m.learn_one(x)
    holders = [
        name for name, value in vars(m).items()
        if isinstance(value, dict)
        and any(isinstance(v, CluStreamMicroCluster) for v in value.values())
    ]
    assert holders == ["micro_clusters"]  # the current set, and no store of past ones

    assert list(inspect.signature(cluster.CluStream.predict_one).parameters) == ["self", "x"]
    init = inspect.signature(cluster.CluStream.__init__).parameters
    assert "alpha" not in init and "l" not in init


def test_the_subclass_adds_the_paper_parameters_and_a_horizon():
    init = inspect.signature(CluStreamPyramidal.__init__).parameters
    assert {"alpha", "l", "snapshot_interval"} <= set(init)
    assert list(inspect.signature(CluStreamPyramidal.predict_one).parameters) == [
        "self", "x", "horizon",
    ]
    assert inspect.signature(CluStreamPyramidal.predict_one).parameters["horizon"].default is None


@pytest.mark.parametrize(
    "bad",
    [dict(alpha=1), dict(alpha=2.0), dict(alpha=True), dict(l=0), dict(l=1.5),
     dict(snapshot_interval=0), dict(snapshot_interval=2.5)],
)
def test_invalid_pyramid_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        CluStreamPyramidal(**bad)


# --- online phase -----------------------------------------------------------
_LOCKSTEP = dict(
    n_macro_clusters=3, max_micro_clusters=12, time_window=300, time_gap=50, seed=7,
    halflife=0.4,
)


def _lockstep_stream():
    a = _blobs([(-1.0, -1.0), (-1.0, 1.0), (0.0, 0.0)], 1500, 0.25, seed=1)
    b = _blobs([(1.0, -1.0), (1.0, 1.0)], 1500, 0.25, seed=2)
    return a + b


def _learn_and_classify(plain, x, w=1.0):
    """Learn ``x`` into river's CluStream; return the maintenance path it took.

    Judged independently from river's own rule: when a point fits nowhere, a
    key gets a new object; that is the deletion path if some micro-cluster's
    relevance stamp is below the threshold at that moment, the merge path
    otherwise. None if maintenance did not run.
    """
    before = dict(plain.micro_clusters)
    threshold = plain._timestamp + 1 - plain.time_window
    stale = any(mc.relevance_stamp(plain.max_micro_clusters) < threshold for mc in before.values())
    plain.learn_one(x, w)
    if any(k in before and mc is not before[k] for k, mc in plain.micro_clusters.items()):
        return "deletion" if stale else "merge"
    return None


def _weights(n, weighted, seed=5):
    """Unit weights, or weights river accepts that are not 1 (none is 0)."""
    rng = random.Random(seed)
    return [rng.choice([0.5, 1.0, 2.0, 3.0]) if weighted else 1.0 for _ in range(n)]


@pytest.mark.parametrize("weighted", [False, True])
def test_online_phase_is_identical_to_river_clustream(weighted):
    # The framework learns with unit weights (adapter.py); river accepts any w,
    # and the wrappers must pass it through on both of river's paths.
    plain = cluster.CluStream(**_LOCKSTEP)
    pyr = CluStreamPyramidal(**_LOCKSTEP, alpha=2, l=2, snapshot_interval=10)
    probes = [x for x, _ in _blobs([(-1.0, 0.0), (1.0, 0.0)], 40, 0.8, seed=3)]
    stream = _lockstep_stream()

    deletions = merges = 0
    for j, ((x, _), w) in enumerate(zip(stream, _weights(len(stream), weighted))):
        path = _learn_and_classify(plain, x, w)
        pyr.learn_one(x, w)
        deletions += path == "deletion"
        merges += path == "merge"
        if j % 100 == 0:
            assert [pyr.predict_one(p) for p in probes] == [plain.predict_one(p) for p in probes]
            # Horizon queries are read-only: the state must not move.
            for h in (25, 400):
                pyr.micro_clusters_for_horizon(h)
                pyr.predict_one(probes[0], horizon=h)
        if j % 250 == 0:
            assert _state(pyr) == _state(plain)

    assert deletions > 0 and merges > 0
    assert (pyr.n_deletions, pyr.n_merges) == (deletions, merges)
    assert _state(pyr) == _state(plain)
    assert [pyr.predict_one(p, horizon=None) for p in probes] == [
        plain.predict_one(p) for p in probes
    ]


@pytest.mark.parametrize("time_window", [10**9, 30])
def test_a_single_micro_cluster_merged_into_itself_counts_as_a_merge(time_window):
    # With one micro-cluster river's merge path finds no pair, adds the
    # micro-cluster to itself and replaces it (clustream.py, closest_a ==
    # closest_b). Nothing expires when time_window is longer than the stream.
    params = dict(max_micro_clusters=1, time_window=time_window, time_gap=50, seed=0)
    plain = cluster.CluStream(**params)
    pyr = CluStreamPyramidal(**params, snapshot_interval=5)
    paths = {"deletion": 0, "merge": 0, None: 0}
    for x, _ in _blobs([(0.0, 0.0), (1.0, 1.0)], 500, 0.3, seed=8):
        paths[_learn_and_classify(plain, x)] += 1
        pyr.learn_one(x)
    assert (pyr.n_deletions, pyr.n_merges) == (paths["deletion"], paths["merge"])
    assert _state(pyr) == _state(plain)
    assert paths["merge"] > 0
    assert (paths["deletion"] > 0) == (time_window == 30)
    (ids,) = pyr.micro_cluster_ids.values()
    assert ids == {pyr._last_id}  # each replacement starts a fresh id list


def test_id_lists_partition_the_live_micro_clusters():
    pyr = CluStreamPyramidal(**_LOCKSTEP, alpha=2, l=2, snapshot_interval=10)
    for x, _ in _lockstep_stream():
        pyr.learn_one(x)
    assert set(pyr.micro_cluster_ids) == set(pyr.micro_clusters)
    seen = set()
    for ids in pyr.micro_cluster_ids.values():
        assert ids and not (ids & seen)
        seen |= ids
    assert any(len(ids) > 1 for ids in pyr.micro_cluster_ids.values())  # merges unioned


# --- pyramid ----------------------------------------------------------------
def _top_order(t, alpha):
    """floor(log_alpha t) in exact integer arithmetic."""
    i = 0
    while alpha ** (i + 1) <= t:
        i += 1
    return i


def _paper_store(t, alpha, keep):
    """Ticks the paper keeps at clock t: the union over orders i of the last
    ``keep`` ticks divisible by alpha**i (Aggarwal et al. 2003, pyramidal time frame)."""
    ticks = set()
    for i in range(_top_order(t, alpha) + 1):
        step = alpha ** i
        last = t // step
        ticks |= {m * step for m in range(max(1, last - keep + 1), last + 1)}
    return ticks


def _exact_order(tick, alpha):
    o = 0
    while tick % alpha == 0:
        tick //= alpha
        o += 1
    return o


@pytest.mark.parametrize("alpha,l", [(2, 1), (2, 3), (3, 2), (5, 1)])
def test_pyramid_invariants_hold_at_every_tick(alpha, l):  # noqa: E741
    keep = alpha ** l + 1
    m = CluStreamPyramidal(
        max_micro_clusters=3, time_gap=50, seed=0, alpha=alpha, l=l, snapshot_interval=1
    )
    rng = random.Random(0)
    for t in range(1, 1201):
        m.learn_one({0: rng.gauss(0.0, 1.0)})
        stored = m.snapshots
        per_order = {}
        for tick, snap in stored.items():
            assert snap.tick == tick and 1 <= tick <= t
            assert tick % alpha ** snap.order == 0 and tick % alpha ** (snap.order + 1) != 0
            assert snap.order == _exact_order(tick, alpha)
            per_order[snap.order] = per_order.get(snap.order, 0) + 1
        assert max(per_order.values()) <= keep
        assert len(stored) <= keep * (_top_order(t, alpha) + 1)
        assert set(stored) == _paper_store(t, alpha, keep)
        assert list(stored) == sorted(stored)


def test_ticks_count_learned_samples_and_snapshots_are_frozen_copies():
    si = 7
    m = CluStreamPyramidal(max_micro_clusters=4, time_gap=50, seed=0, snapshot_interval=si)
    rng = random.Random(1)
    for n in range(1, 400):
        m.learn_one({0: rng.gauss(0.0, 1.0), 1: rng.gauss(0.0, 1.0)})
        if n < si:
            assert not m.snapshots
            continue
        assert max(m.snapshots) == n // si
        if n % si == 0:
            snap = m.snapshots[n // si]
            assert snap.micro_clusters.keys() == m.micro_clusters.keys()
            for k, smc in snap.micro_clusters.items():
                live = m.micro_clusters[k]
                assert smc.ids == m.micro_cluster_ids[k]
                assert _var_state(smc.var_time) == _var_state(live.var_time)
                assert smc.var_time is not live.var_time
                for f, v in smc.var_x.items():
                    assert _var_state(v) == _var_state(live.var_x[f])
                    assert v is not live.var_x[f] and v.mean is not live.var_x[f].mean

    frozen = {
        tick: {k: _var_state(s.var_time) for k, s in snap.micro_clusters.items()}
        for tick, snap in m.snapshots.items()
    }
    for _ in range(3 * si - 1):  # learn without reaching a tick that could prune these
        m.learn_one({0: rng.gauss(0.0, 1.0), 1: rng.gauss(0.0, 1.0)})
    for tick, stats in frozen.items():
        if tick in m.snapshots:
            snap = m.snapshots[tick].micro_clusters
            assert {k: _var_state(s.var_time) for k, s in snap.items()} == stats


@pytest.mark.parametrize("alpha,l,si", [(2, 1, 1), (2, 2, 3), (3, 2, 2), (2, 3, 1)])
def test_lemma_bound_on_the_covered_horizon(alpha, l, si):  # noqa: E741
    """Paper Lemma: with t_s the last stored snapshot at or before t_c - h,
    t_c - t_s <= (1 + 1/alpha**(l-1)) * h.

    Preconditions, worked out in the module docstring: on the paper's own clock
    (query at a tick, horizon a whole number of ticks) the bound holds for every
    horizon of at least one tick. Off that grid it holds, strictly, for
    h >= snapshot_interval * alpha**(l-1); below that the covered horizon is
    still short of h + snapshot_interval. Checked exhaustively over every
    (t_c, h) pair up to t_c = 240 samples.
    """
    eps = 1.0 / alpha ** (l - 1)
    m = CluStreamPyramidal(
        max_micro_clusters=3, time_gap=50, seed=0, alpha=alpha, l=l, snapshot_interval=si
    )
    rng = random.Random(2)
    for n in range(1, 241):
        m.learn_one({0: rng.gauss(0.0, 1.0)})
        stored = {tick * si for tick in m.snapshots}
        for h in range(1, n + 1):
            hz = m.micro_clusters_for_horizon(h)
            t_s = hz.snapshot_time
            assert hz.requested == h and hz.covered == n - t_s
            assert t_s <= n - h
            assert hz.whole_stream == (t_s == 0)
            assert t_s == max([s for s in stored if s <= n - h], default=0)
            assert hz.covered < h + max(si, h * eps)
            if h >= si * alpha ** (l - 1):
                assert hz.covered < (1 + eps) * h
            if n % si == 0 and h % si == 0:
                assert hz.covered <= (1 + eps) * h


def test_horizon_past_the_stream_start_is_the_whole_stream():
    # sigma is a KMeans option (**kwargs) that moves the labels here, so the
    # comparison with river's offline phase below also pins that the horizon
    # k-means is built with the model's kwargs.
    m = CluStreamPyramidal(
        max_micro_clusters=5, time_gap=20, seed=0, snapshot_interval=10, sigma=0.3
    )
    probes = [x for x, _ in _blobs([(0.0, 0.0), (1.0, 1.0)], 30, 0.5, seed=5)]
    # h past the stream start: the micro-cluster set of river's offline phase.
    # Its k-means is refitted on demand, river's only at time_gap boundaries
    # (none before the first boundary after initialisation, so river answers
    # 0 until then). The labels must agree at every boundary once river has
    # fitted; in between they may differ.
    boundaries = before_first_fit = off_boundary_diffs = 0
    for n, (x, _) in enumerate(_blobs([(0.0, 0.0), (1.0, 1.0)], 200, 0.2, seed=4), start=1):
        m.learn_one(x)
        if n < m.max_micro_clusters:
            continue
        river_labels = [m.predict_one(p) for p in probes]
        horizon_labels = [m.predict_one(p, horizon=10_000) for p in probes]
        if m._kmeans_mc is None:
            assert set(river_labels) == {0} and len(set(horizon_labels)) > 1
            before_first_fit += 1
        elif n % m.time_gap == 0:
            assert horizon_labels == river_labels and len(set(river_labels)) > 1
            boundaries += 1
        else:
            off_boundary_diffs += horizon_labels != river_labels
    assert before_first_fit == 15 and boundaries == 10 and off_boundary_diffs > 0

    hz = m.micro_clusters_for_horizon(10_000)
    assert hz.whole_stream and hz.snapshot_time == 0 and hz.covered == 200
    assert hz.micro_clusters.keys() == m.micro_clusters.keys()
    for k, mc in hz.micro_clusters.items():
        assert _var_state(mc.var_time) == _var_state(m.micro_clusters[k].var_time)
        assert mc is not m.micro_clusters[k]
    river_labels = [m.predict_one(p) for p in probes]

    without_kwargs = cluster.KMeans(n_clusters=m.n_macro_clusters, seed=m.seed)
    for mc in m.micro_clusters.values():
        without_kwargs.learn_one(mc.center)
    nearest = [m.micro_clusters[m._get_closest_mc(p)[0]].center for p in probes]
    assert [without_kwargs.predict_one(c) for c in nearest] != river_labels  # sigma matters


def test_horizon_labels_are_river_offline_kmeans_over_the_horizon_centres():
    m = CluStreamPyramidal(**_LOCKSTEP, alpha=2, l=2, snapshot_interval=10)
    probes = [x for x, _ in _blobs([(-1.0, 0.0), (1.0, 0.0)], 100, 0.8, seed=6)]
    differs_from_current = subtracted = 0
    for j, (x, _) in enumerate(_lockstep_stream(), start=1):
        m.learn_one(x)
        if j % 500:
            continue
        for h in (60, 700):
            hz = m.micro_clusters_for_horizon(h)
            subtracted += not hz.whole_stream
            mcs = list(hz.micro_clusters.values())
            km = cluster.KMeans(n_clusters=m.n_macro_clusters, seed=m.seed, **m.kwargs)
            for mc in mcs:
                km.learn_one(mc.center)
            nearest = [
                min(mcs, key=lambda mc: sum((mc.center[f] - p[f]) ** 2 for f in p)) for p in probes
            ]
            got = [m.predict_one(p, horizon=h) for p in probes]
            assert got == [km.predict_one(mc.center) for mc in nearest]
            assert len(set(got)) > 1
            differs_from_current += got != [m.predict_one(p) for p in probes]
    assert subtracted >= 6 and differs_from_current > 0


def test_horizon_queries_validate_and_degrade_like_river():
    m = CluStreamPyramidal(seed=0)
    assert m.predict_one({0: 1.0}, horizon=10) == 0  # river returns 0 before any clustering
    hz = m.micro_clusters_for_horizon(10)
    assert hz.micro_clusters == {} and hz.whole_stream
    for bad in (0, -5):
        with pytest.raises(ValueError):
            m.micro_clusters_for_horizon(bad)


def test_returned_horizon_micro_clusters_are_the_callers_own_copies():
    # The horizon set is cached for predict_one(x, horizon=h) at the same
    # timestamp; changing what a query returned must not reach that cache.
    m = CluStreamPyramidal(**_LOCKSTEP, alpha=2, l=2, snapshot_interval=10)
    for x, _ in _lockstep_stream()[:800]:
        m.learn_one(x)
    probes = [x for x, _ in _blobs([(-1.0, 0.0), (1.0, 0.0)], 40, 0.8, seed=3)]
    h = 150
    labels = [m.predict_one(p, horizon=h) for p in probes]
    first = m.micro_clusters_for_horizon(h)
    assert not first.whole_stream
    expected = {k: _mc_state(mc) for k, mc in first.micro_clusters.items()}
    for mc in first.micro_clusters.values():
        mc.insert({0: 50.0, 1: 50.0}, 5.0, 0)
        mc += mc
    again = m.micro_clusters_for_horizon(h)
    assert {k: _mc_state(mc) for k, mc in again.micro_clusters.items()} == expected
    assert [m.predict_one(p, horizon=h) for p in probes] == labels


# --- subtraction ------------------------------------------------------------
class _Ledger(CluStreamPyramidal):
    """Test-only: tracks which learned points each live micro-cluster holds.

    It reads river's dict after every ``learn_one``: an insert grows one key by
    exactly ``w``; river's initialisation adds a new key; a deletion puts a new
    object at one key; a merge does that and grows one other key by exactly the
    weight the replaced key held. Each point is kept with its weight and the
    timestamp river put into ``var_time``: ``max_micro_clusters - 1`` for every
    point of the initialisation branch (river's own convention), the sample's
    ``_timestamp`` after it.
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.held = {}  # key -> [(sample index, x, w, river timestamp)]

    def learn_one(self, x, w=1.0):
        before = {k: (mc, mc.var_time.n) for k, mc in self.micro_clusters.items()}
        initialising = not self._initialized
        super().learn_one(x, w)
        stamp = self.max_micro_clusters - 1 if initialising else self._timestamp
        point = (self._timestamp + 1, dict(x), w, stamp)
        replaced = [
            k for k, mc in self.micro_clusters.items() if k not in before or mc is not before[k][0]
        ]
        grown = [
            k for k, mc in self.micro_clusters.items()
            if k in before and mc is before[k][0] and mc.var_time.n != before[k][1]
        ]
        if replaced:
            (k,) = replaced
            if grown:
                (a,) = grown
                assert self.micro_clusters[a].var_time.n == before[a][1] + before[k][1]
                self.held[a] = self.held[a] + self.held[k]
            self.held[k] = [point]
        else:
            (k,) = grown
            assert self.micro_clusters[k].var_time.n == before[k][1] + w
            self.held[k].append(point)


def _weighted_stats(values, weights):
    """Weight, mean and variance as river's ``stats.Var`` (ddof=1) defines them."""
    total = sum(weights)
    mean = sum(v * w for v, w in zip(values, weights)) / total
    s = sum(w * (v - mean) ** 2 for v, w in zip(values, weights))
    return total, mean, s / (total - 1) if total > 1 else 0.0


def _check_against_ledger(m, hz):
    """Every live micro-cluster's horizon part holds exactly its points learned
    after the snapshot: same weight; the same mean (rel 1e-9) and variance per
    feature and of the timestamps (``var_time``, which ``relevance_stamp``
    reads). Variances get rel 1e-6, as the second moment is a difference of
    large sums (measured worst 2.5e-9 on features, 1.5e-9 on timestamps), plus
    an absolute floor for a true variance of 0, where float residue of about
    1e-16 remains. Returns the keys left with no point after the snapshot."""
    emptied = set()
    for k in m.micro_clusters:
        points = [p for p in m.held[k] if p[0] > hz.snapshot_time]
        if not points:
            assert k not in hz.micro_clusters
            emptied.add(k)
            continue
        got = hz.micro_clusters[k]
        weights = [w for _, _, w, _ in points]
        total, t_mean, t_var = _weighted_stats([t for *_, t in points], weights)
        assert got.weight == total
        assert got.var_time.mean.get() == pytest.approx(t_mean, rel=1e-9)
        assert got.var_time.get() == pytest.approx(t_var, rel=1e-6, abs=1e-6)
        for f in points[0][1]:
            _, mean, var = _weighted_stats([x[f] for _, x, _, _ in points], weights)
            assert got.center[f] == pytest.approx(mean, rel=1e-9, abs=1e-12)
            assert got.var_x[f].get() == pytest.approx(var, rel=1e-6, abs=1e-12)
    return emptied


@pytest.mark.parametrize("weighted", [False, True])
def test_subtraction_leaves_exactly_the_points_learned_after_the_snapshot(weighted):
    """Checked by ``_check_against_ledger``. A live micro-cluster's id list is
    unchanged since the snapshot, new since, or grown by merges; all three
    occur, and so do micro-clusters that hold nothing after the snapshot."""
    m = _Ledger(**_LOCKSTEP, alpha=2, l=2, snapshot_interval=10)
    stream = _lockstep_stream()
    kinds = {"unchanged": 0, "new": 0, "merged": 0, "emptied": 0}
    for j, ((x, _), w) in enumerate(zip(stream, _weights(len(stream), weighted)), start=1):
        m.learn_one(x, w)
        if j % 100 or j < 300:
            continue
        for h in (15, 40, 150, 450, 1100):
            hz = m.micro_clusters_for_horizon(h)
            if hz.whole_stream:
                continue
            snap = m.snapshots[hz.snapshot_time // m.snapshot_interval].micro_clusters
            snap_ids = set().union(*(s.ids for s in snap.values()))
            for k, ids in m.micro_cluster_ids.items():
                if k in snap and snap[k].ids == ids:
                    kinds["unchanged"] += 1
                elif not ids & snap_ids:
                    kinds["new"] += 1
                else:
                    kinds["merged"] += 1
            kinds["emptied"] += len(_check_against_ledger(m, hz))
    assert min(kinds.values()) > 10, kinds


def test_subtraction_is_exact_for_snapshots_taken_while_river_initialises():
    """River creates micro-clusters in three places, and its initialisation
    branch is one: each of the first ``max_micro_clusters`` points gets a new
    key and must get a new id, without touching the ids of the keys before it.
    Snapshots every 3 samples with 10 micro-clusters fall inside that branch;
    every horizon at every sample is checked against the ledger, past the end
    of initialisation and through river's merges."""
    m = _Ledger(
        n_macro_clusters=3, max_micro_clusters=10, time_window=40, time_gap=5, seed=7,
        alpha=2, l=2, snapshot_interval=3,
    )
    from_initialisation = 0
    for n, (x, _) in enumerate(_blobs([(-1.0, -1.0), (1.0, 1.0), (1.0, -1.0)], 60, 0.3, 11), 1):
        m.learn_one(x)
        for h in range(1, n + 1):
            hz = m.micro_clusters_for_horizon(h)
            if not hz.whole_stream:
                _check_against_ledger(m, hz)
                from_initialisation += hz.snapshot_time < m.max_micro_clusters <= n
    assert from_initialisation > 50 and m.n_merges > 0


def test_subtracted_variances_are_never_negative():
    """Checked on the stored second moment ``Var._S`` as well as on ``Var.get()``:
    when one point remains, ``get()`` returns 0 whatever ``_S`` holds, and
    river's subtraction often leaves ``_S`` slightly negative there, in
    ``var_time`` as in ``var_x``."""
    m = CluStreamPyramidal(**_LOCKSTEP, alpha=2, l=2, snapshot_interval=10)
    single = 0
    for j, (x, _) in enumerate(_lockstep_stream(), start=1):
        m.learn_one(x)
        if j % 500 == 0:
            for h in (30, 200, 900):
                for mc in m.micro_clusters_for_horizon(h).micro_clusters.values():
                    assert mc.weight > 0
                    single += mc.weight == 1
                    for v in (mc.var_time, *mc.var_x.values()):
                        assert v._S >= 0.0 and v.get() >= 0.0
                    assert math.isfinite(mc.radius(2))
    assert single > 0


def test_float_error_in_var_subtraction_is_clamped():
    # One micro-cluster that every point fits; three points before the
    # snapshot, two identical ones after it.
    m = CluStreamPyramidal(
        max_micro_clusters=1, micro_cluster_r_factor=1000, time_gap=1000, seed=0,
        snapshot_interval=3,
    )
    for v in (0.0, 0.1, 0.2, 0.6, 0.6):
        m.learn_one({0: v})
    raw = m.micro_clusters[0].var_x[0] - m.snapshots[1].micro_clusters[0].var_x[0]
    assert raw.n == 2 and raw.get() < 0.0  # river's Var.__isub__ on its own
    hz = m.micro_clusters_for_horizon(2)
    assert hz.snapshot_time == 3 and hz.micro_clusters[0].weight == 2
    assert hz.micro_clusters[0].var_x[0].get() == 0.0
    assert hz.micro_clusters[0].radius(2) == 0.0

    # Unclamped, river's own radius takes the square root of that variance.
    unclamped = CluStreamMicroCluster(x={0: 0.6}, w=1.0, timestamp=3)
    unclamped.insert({0: 0.6}, 1.0, 4)
    unclamped.var_x[0] = raw
    with pytest.raises(ValueError):
        unclamped.radius(2)


# --- behaviour --------------------------------------------------------------
_REGIME_A = [(-1.5, -1.0), (-1.5, 1.0), (-0.5, 0.0)]
_REGIME_B = [(1.0, -1.0), (1.0, 1.0), (2.0, 0.0)]
_SPREAD = 0.15
# Why these CluStream parameters, at this (unit) scale:
# * n_macro_clusters=3 -- the number of blobs in regime B, which is what the
#   horizon query should recover.
# * time_window=100_000, longer than the stream -- no micro-cluster ever
#   expires, so regime A survives in the *current* set (merged, not deleted).
#   That is the situation a horizon exists for; with deletions on, river
#   forgets A by itself and both queries see nearly the same set.
# * max_micro_clusters=30 -- room for every blob, small enough that regime B
#   forces merges of A's micro-clusters.
# * snapshot_interval=100 and l=3 -- tick 60 (sample 6000) is still in the
#   order-2 list at tick 80, so the horizon of 2000 is covered exactly; with
#   l=1 the last stored tick at or before 60 is 48, so 3200 samples are covered.
_BEHAVIOUR = dict(
    n_macro_clusters=3, max_micro_clusters=30, time_window=100_000, time_gap=100,
    alpha=2, snapshot_interval=100,
)


def _closer_to(center, blobs, others):
    point = (center[0], center[1])
    return min(math.dist(b, point) for b in blobs) < min(math.dist(b, point) for b in others)


def _two_regimes(l, seed, **overrides):  # noqa: E741
    m = CluStreamPyramidal(**{**_BEHAVIOUR, **overrides}, l=l, seed=seed)
    for x, _ in _blobs(_REGIME_A, 6000, _SPREAD, seed=10 + seed):
        m.learn_one(x)
    for x, _ in _blobs(_REGIME_B, 2000, _SPREAD, seed=20 + seed):
        m.learn_one(x)
    return m


@pytest.mark.parametrize("l,covered", [(3, 2000), (1, 3200)])
def test_horizon_micro_clusters_summarise_the_covered_horizon(l, covered):  # noqa: E741
    h = 2000
    for seed in range(3):
        m = _two_regimes(l, seed)
        hz = m.micro_clusters_for_horizon(h)
        assert not hz.whole_stream and hz.covered == covered
        assert hz.covered <= (1 + 1 / m.alpha ** (m.l - 1)) * h
        # No micro-cluster expired, so every point learned after t_s is still held.
        assert m.n_deletions == 0
        assert sum(mc.weight for mc in hz.micro_clusters.values()) == hz.covered
        # Regime A enters only through the part of the covered horizon before
        # t_c - h; with l=3 that part is empty and every centre lies nearer B.
        weight_a = sum(
            mc.weight for mc in hz.micro_clusters.values()
            if _closer_to(mc.center, _REGIME_A, _REGIME_B)
        )
        assert weight_a == hz.covered - h
        # Regime A is still in the current set, which is what makes this bite.
        assert any(_closer_to(mc.center, _REGIME_A, _REGIME_B) for mc in m.micro_clusters.values())


def test_horizon_predictions_recover_the_recent_regime():
    h = 2000
    gains = []
    for seed in range(5):
        m = _two_regimes(3, seed)
        held_out = _blobs(_REGIME_B, 600, _SPREAD, seed=30 + seed)
        y = [label for _, label in held_out]
        ari_h = adjusted_rand_score(y, [m.predict_one(x, horizon=h) for x, _ in held_out])
        ari_0 = adjusted_rand_score(y, [m.predict_one(x) for x, _ in held_out])
        assert ari_h >= ari_0, (seed, ari_h, ari_0)
        gains.append(ari_h - ari_0)
    # Measured: equal on seeds 3 and 4, +0.57 / +0.57 / +0.39 on seeds 0-2.
    # Only with time_window longer than the stream; see the next test.
    assert sum(gains) > 1.0


def test_at_river_default_time_window_the_horizon_is_the_current_set_here():
    """The condition the gain above depends on. With river's default
    time_window=1000, river's own deletions have removed every regime-A
    micro-cluster by the end of this stream: nothing is subtracted, the horizon
    micro-clusters are the current ones, and both predictions agree. (Measured
    with time_window=3000: gains 0 / 0 / +0.46 / +0.12 / 0 on seeds 0-4.)"""
    h = 2000
    for seed in range(5):
        m = _two_regimes(3, seed, time_window=1000)
        hz = m.micro_clusters_for_horizon(h)
        assert m.n_deletions > 0 and hz.covered == h
        live = m.micro_clusters.values()
        assert not any(_closer_to(mc.center, _REGIME_A, _REGIME_B) for mc in live)
        assert {k: mc.weight for k, mc in hz.micro_clusters.items()} == {
            k: mc.weight for k, mc in m.micro_clusters.items()
        }
        # Points that went into micro-clusters deleted since are gone from both views.
        assert sum(mc.weight for mc in hz.micro_clusters.values()) < h
        held_out = [x for x, _ in _blobs(_REGIME_B, 600, _SPREAD, seed=30 + seed)]
        plain = [m.predict_one(x) for x in held_out]
        assert [m.predict_one(x, horizon=h) for x in held_out] == plain


# --- persistence ------------------------------------------------------------
def test_pickle_roundtrip_keeps_the_pyramid_and_the_online_state():
    m = CluStreamPyramidal(**_LOCKSTEP, alpha=3, l=2, snapshot_interval=10)
    stream = _lockstep_stream()
    for x, _ in stream[:1700]:
        m.learn_one(x)
    m.predict_one(stream[0][0], horizon=300)  # populate the horizon cache too
    r = pickle.loads(pickle.dumps(m))

    assert list(r.snapshots) == list(m.snapshots)
    assert r.micro_cluster_ids == m.micro_cluster_ids
    assert _state(r) == _state(m)
    probes = [x for x, _ in stream[1700:1760]]
    for h in (None, 50, 300, 1200):
        assert [r.predict_one(p, horizon=h) for p in probes] == [
            m.predict_one(p, horizon=h) for p in probes
        ]
    for x, _ in stream[1700:]:
        m.learn_one(x)
        r.learn_one(x)
    assert _state(r) == _state(m) and list(r.snapshots) == list(m.snapshots)
    assert r.micro_cluster_ids == m.micro_cluster_ids


# --- registry ---------------------------------------------------------------
# Skipped until benchmark/algorithms/registry.py gains the proposed entry; it
# then checks that entry the way test_denstream_patched.py checks its own.
@pytest.mark.skipif(
    "CluStreamPyramidal" not in ALGORITHMS,
    reason="not registered yet: the registry entry is proposed with RIVER_BUGS.md GAP-4",
)
def test_clustream_pyramidal_is_registered_next_to_clustream():
    spec = ALGORITHMS["CluStreamPyramidal"]
    assert spec.default_preprocessor == ALGORITHMS["CluStream"].default_preprocessor == "standard"
    assert "GAP-4" in spec.note and "GAP-4" in spec.provenance
    assert "upstream untouched" in spec.provenance
    assert accepts_seed("CluStreamPyramidal")
    model, params = spec.factory()
    assert type(model) is CluStreamPyramidal
    knobs = {"alpha", "l", "snapshot_interval"}
    assert knobs <= set(params)
    # Same defaults as CluStream's entry, plus the pyramid's own knobs.
    _, river_params = ALGORITHMS["CluStream"].factory()
    assert {k: v for k, v in params.items() if k not in knobs} == river_params


def test_clone_is_a_fresh_model_with_the_same_parameters():
    m = CluStreamPyramidal(**_LOCKSTEP, alpha=3, l=2, snapshot_interval=10)
    for x, _ in _lockstep_stream()[:500]:
        m.learn_one(x)
    c = m.clone()
    assert type(c) is CluStreamPyramidal
    assert c._get_params() == m._get_params()
    assert (c.alpha, c.l, c.snapshot_interval, c.kwargs) == (3, 2, 10, {"halflife": 0.4})
    assert not c.snapshots and not c.micro_clusters and not c.micro_cluster_ids
    fresh = CluStreamPyramidal(**_LOCKSTEP, alpha=3, l=2, snapshot_interval=10)
    for x, _ in _lockstep_stream()[:500]:
        c.learn_one(x)
        fresh.learn_one(x)
    assert _state(c) == _state(fresh) and list(c.snapshots) == list(fresh.snapshots)
