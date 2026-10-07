"""DenStreamPatchedFast closes RIVER_BUGS.md GAP-3 for DenStreamPatched, exactly.

GAP-3: ``DenStream.predict_one`` re-runs the whole offline DBSCAN on every call.
Every p-micro-cluster runs ``_query_neighbor``, an O(P) scan, so one prediction
costs P*(P-1) reachability tests even when nothing changed since the last one.
``DenStreamPatchedFast`` keeps the timestamp-free half of that test per
micro-cluster object and recomputes only what changed.

What is proved here, in this order:

* the gap is real, upstream (``river.cluster.DenStream``) and on the reference
  (``DenStreamPatched``): exactly P*(P-1) tests per ``predict_one``, also back
  to back with no ``learn_one`` between;
* the premises the cache rests on hold in the vendored river: centre and radius
  ignore the timestamp, and the distance is *not* bit-symmetric when two
  centres order their keys differently;
* lockstep exactness against ``DenStreamPatched`` on every step of several
  streams -- return value, ``n_clusters``, every merged cluster and every live
  p-/o-micro-cluster compared with ``==``, never approx, centre memos included
  -- with each stream asserting it actually hit the path it is there for
  (merged groups, pruning, o->p promotion, the ``len(dict)`` key overwrite,
  light p-micro-clusters, cached edges on the real-data projections);
* crafted boundaries: chains that need several hops (BUG-8's path, which no
  natural stream here reaches), one cut and rejoined at a link that turns
  light, members swapped or reordered under unchanged stamps, edge direction
  under asymmetric distances, identical centres with radius 0, a caller
  editing an input after ``learn_one`` (BUG-3), inserting into
  ``clusters[k]`` or editing a centre ``centers`` returned;
* the cost: at most 2*(P-1) geometric tests per prequential prediction;
* pickle, ``clone`` and ``deepcopy`` keep exactness.

Exactness is against ``DenStreamPatched`` (BUG-7 and BUG-8 fixed), not against
upstream ``DenStream``; the upstream labelling is a different function.
"""

from __future__ import annotations

import copy
import pickle
import random
import time
from dataclasses import dataclass, field

import pytest

# Import order matters: importing benchmark.algorithms puts the sibling river
# tree on sys.path before `river` is imported below.
from benchmark.algorithms.adapter import make_adapter
from benchmark.algorithms.denstream_fast import DenStreamPatchedFast
from benchmark.algorithms.denstream_patched import DenStreamPatched
from benchmark.algorithms.registry import make_scaler
from benchmark.evaluation.runner import run_once
from river import cluster, datasets
from river.cluster.denstream import DenStreamMicroCluster
from river.utils.vectordict import euclidean_distance_dict


# -- counting subclasses (module level so they pickle) -----------------------
class CountingUpstream(cluster.DenStream):
    """Upstream river DenStream, counting its pairwise reachability tests."""

    n_tests = 0

    def _is_directly_density_reachable(self, c_p, c_q):
        self.n_tests += 1
        return super()._is_directly_density_reachable(c_p, c_q)


class CountingPatched(DenStreamPatched):
    """The reference, counting its pairwise reachability tests."""

    n_tests = 0

    def _is_directly_density_reachable(self, c_p, c_q):
        self.n_tests += 1
        return super()._is_directly_density_reachable(c_p, c_q)


class CountingFast(DenStreamPatchedFast):
    """The fast class, counting both pairwise tests it could run."""

    n_tests = 0
    n_full_tests = 0

    def _is_geometrically_reachable(self, center_p, center_q, r_p, r_q):
        self.n_tests += 1
        return super()._is_geometrically_reachable(center_p, center_q, r_p, r_q)

    def _is_directly_density_reachable(self, c_p, c_q):
        self.n_full_tests += 1
        return super()._is_directly_density_reachable(c_p, c_q)

    @property
    def pair_tests(self):
        return self.n_tests + self.n_full_tests


# -- streams -----------------------------------------------------------------
def _jumping(n, seed, eps=0.1, k=5, sd=0.8, jump=1.0, every=60, noise=0.03, box=1.0):
    """1-D blobs that jump by about ``eps`` every ``every`` samples, plus noise.

    In 1-D the norm-based ``calc_radius`` is the plain standard deviation, so
    a blob jumping by ~eps lands next to a saturated p-micro-cluster and new
    ones form overlapping it: this is what produces directly-density-reachable
    pairs, while the abandoned positions decay, turn light and get pruned.
    """
    rng = random.Random(seed)
    centers = [rng.uniform(-box, box) for _ in range(k)]
    for i in range(n):
        if i and i % every == 0:
            j = rng.randrange(k)
            centers[j] += rng.choice((-1, 1)) * jump * eps
            if abs(centers[j]) > box:
                centers[j] *= -0.5
        if rng.random() < noise:
            yield {"a": rng.uniform(-box - 0.3, box + 0.3)}
        else:
            yield {"a": rng.gauss(rng.choice(centers), sd * eps)}


def _drifting_blobs(n, seed, k=6, spread=0.15, drift=0.0005, noise=0.1, box=1.0, dim=2):
    rng = random.Random(seed)
    centers = [[rng.uniform(-box, box) for _ in range(dim)] for _ in range(k)]
    vel = [[rng.uniform(-1, 1) * drift for _ in range(dim)] for _ in range(k)]
    for _ in range(n):
        if rng.random() < noise:
            yield {f"x{d}": rng.uniform(-1.5 * box, 1.5 * box) for d in range(dim)}
            continue
        j = rng.randrange(k)
        for d in range(dim):
            centers[j][d] += vel[j][d]
        yield {f"x{d}": rng.gauss(centers[j][d], spread) for d in range(dim)}


def _sites(n, seed, eps=0.1, n_sites=100, gap=1.5, sd=0.9, noise=0.05):
    """Many 1-D Gaussian sites ``gap*eps`` apart: P settles around 50."""
    rng = random.Random(seed)
    base = [(i - n_sites / 2) * gap * eps for i in range(n_sites)]
    for _ in range(n):
        if rng.random() < noise:
            yield {"a": rng.uniform(base[0] - 3 * eps, base[-1] + 3 * eps)}
        else:
            yield {"a": rng.gauss(rng.choice(base), sd * eps)}


def _shuffled_sparse(n, seed, eps=0.1):
    """3-D points whose key order is shuffled and which sometimes miss a feature.

    Micro-cluster sums inherit the key order of the point that created them,
    so centres end up ordering their keys differently and
    ``euclidean_distance_dict`` stops being bit-symmetric between them.
    """
    rng = random.Random(seed)
    for x in _jumping(n, seed, eps=eps):
        point = {"a": x["a"], "b": rng.gauss(0, 0.3 * eps), "c": rng.gauss(0, 0.3 * eps)}
        if rng.random() < 0.1:
            del point[rng.choice(("b", "c"))]
        keys = list(point)
        rng.shuffle(keys)
        yield {k: point[k] for k in keys}


def _prequential(xs):
    for x in xs:
        yield "predict", x
        yield "learn", x


JUMPING = dict(decaying_factor=0.25, beta=0.75, mu=2, epsilon=0.1, n_samples_init=200,
               stream_speed=5)
SITES = dict(decaying_factor=0.05, beta=0.75, mu=2, epsilon=0.1, n_samples_init=300,
             stream_speed=10)


# -- state comparison ----------------------------------------------------------
def _mc_state(mc):
    # Ordered item lists, not dicts: key order of the sums is state too -- it
    # decides the summation order of every distance computed from the centre.
    return (
        mc.N,
        list(mc.linear_sum.items()),
        list(mc.squared_sum.items()),
        mc.last_edit_time,
        mc.creation_time,
        list(mc.x.items()),
        mc.decaying_factor,
    )


def _model_state(m):
    return {
        "timestamp": m.timestamp,
        "initialized": m.initialized,
        "n_seen": m._n_samples_seen,
        "buffer": len(getattr(m, "_init_buffer", ())),
        "p": [(k, _mc_state(mc)) for k, mc in m.p_micro_clusters.items()],
        "o": [(k, _mc_state(mc)) for k, mc in m.o_micro_clusters.items()],
    }


def _memos(m):
    # `center` memoises into `_center`. Which memos are filled is live state
    # too (it pickles), and `predict_one` is allowed to fill exactly the ones
    # the reference fills, so it is compared against the reference instead of
    # being frozen.
    return [(kind, k, None if mc._center is None else list(mc._center.items()))
            for kind, mcs in (("p", m.p_micro_clusters), ("o", m.o_micro_clusters))
            for k, mc in mcs.items()]


def _clusters_state(m):
    return (
        m.n_clusters,
        [(k, _mc_state(c), list(c.center.items())) for k, c in m.clusters.items()],
    )


def _live_objects(m):
    return {**{("p", k): mc for k, mc in m.p_micro_clusters.items()},
            **{("o", k): mc for k, mc in m.o_micro_clusters.items()}}


@dataclass
class _Trace:
    """What a lockstep run exercised, read off the reference model."""

    predicts: int = 0
    removals: int = 0
    additions: int = 0
    overwrites: int = 0
    merged_groups: int = 0
    light: int = 0
    heavy_to_light: int = 0
    #: predictions that met a light p-micro-cluster whose centre memo was empty
    bare_light: int = 0
    timestamps: set = field(default_factory=set)
    max_p: int = 0
    #: most directed geometric edges the fast model's cache held after a predict
    max_edges: int = 0
    #: (weight, centre) of each live p-micro-cluster replaced by a promotion
    victims: list = field(default_factory=list)
    _heavy: set = field(default_factory=set)

    def before_learn(self, ref):
        return dict(ref.p_micro_clusters)

    def after_learn(self, ref, before):
        after = ref.p_micro_clusters
        b, a = {id(v) for v in before.values()}, {id(v) for v in after.values()}
        self.removals += bool(b - a)
        self.additions += bool(a - b)
        # A key present before and after that maps to another object can only
        # be denstream.py:223 replacing a live p-micro-cluster.
        replaced = [before[k] for k, v in after.items() if k in before and before[k] is not v]
        self.overwrites += bool(replaced)
        self.victims += [(mc.calc_weight(ref.timestamp), dict(mc.center)) for mc in replaced]
        self.timestamps.add(ref.timestamp)
        self.max_p = max(self.max_p, len(after))

    def seen_by_predict(self, ref):
        """The part of the reference state a ``predict_one`` call reads."""
        live = list(ref.p_micro_clusters.values())
        bare = {mc for mc in live if mc._center is None}
        return ref.initialized, live, ref.timestamp, bare

    def after_predict(self, ref, seen):
        self.predicts += 1
        initialized, live, timestamp, bare = seen
        if not initialized:
            return
        self.merged_groups += ref.n_clusters < len(live)
        # Objects, not id()s: an id can be reused once its object is gone.
        heavy = {mc for mc in live if mc.calc_weight(timestamp) > ref.mu}
        self.light += len(heavy) < len(live)
        self.heavy_to_light += bool(self._heavy & (set(live) - heavy))
        self.bare_light += bool(bare - heavy)
        self._heavy = heavy

    def edges_of(self, fast):
        cache = fast.__dict__.get("_offline_cache")
        if cache is not None:
            self.max_edges = max(self.max_edges, sum(map(len, cache.out.values())))


def _assert_same(ref, fast, step):
    assert _model_state(fast) == _model_state(ref), f"model state differs at step {step}"
    assert _memos(fast) == _memos(ref), f"centre memos differ at step {step}"


def _lockstep(ref, fast, ops, tr=None):
    """Drive both models through ``ops`` and compare everything after every step."""
    tr = _Trace() if tr is None else tr
    for step, (op, x) in enumerate(ops):
        if op == "learn":
            before = tr.before_learn(ref)
            ref.learn_one(x)
            fast.learn_one(x)
            tr.after_learn(ref, before)
        else:
            frozen, objects = _model_state(fast), _live_objects(fast)
            seen = tr.seen_by_predict(ref)
            got_ref, got_fast = ref.predict_one(x), fast.predict_one(x)
            assert got_fast == got_ref, f"prediction differs at step {step}"
            assert _clusters_state(fast) == _clusters_state(ref), f"clusters differ at step {step}"
            # predict_one is a query: no live micro-cluster changes or moves
            # (centre memos aside, which _assert_same holds to the reference's).
            assert _model_state(fast) == frozen, f"predict_one mutated state at step {step}"
            after = _live_objects(fast)
            assert after.keys() == objects.keys()
            assert all(after[k] is objects[k] for k in objects)
            tr.after_predict(ref, seen)
            tr.edges_of(fast)
        _assert_same(ref, fast, step)
    return tr


def _pair(params, ref_cls=DenStreamPatched, fast_cls=DenStreamPatchedFast):
    return ref_cls(**params), fast_cls(**params)


# -- premises ----------------------------------------------------------------
def test_geometry_ignores_the_timestamp_in_the_vendored_river():
    # The cache keeps the distance half of the reachability test across calls.
    # That is only valid while centre and radius ignore the timestamp; if a
    # river update makes them depend on it, this fails before anything else.
    mc = DenStreamMicroCluster({"a": 0.3, "b": -0.1}, timestamp=0, decaying_factor=0.25)
    mc.insert({"a": 0.1, "b": 0.2}, 3)
    mc.insert({"b": 0.05, "a": -0.2}, 7)
    assert mc.calc_radius(0) == mc.calc_radius(7) == mc.calc_radius(10**6)
    assert mc.calc_center(0) == mc.calc_center(10**6) == mc.center
    assert mc.calc_weight(7) != mc.calc_weight(10**6)


def test_distance_is_bit_symmetric_only_for_matching_key_order():
    rng = random.Random(0)
    asymmetric = 0
    for _ in range(2000):
        a = {k: rng.gauss(0, 3) for k in ("p", "q", "r", "s")}
        b = {k: rng.gauss(0, 3) for k in ("p", "q", "r", "s")}
        assert euclidean_distance_dict(a, b) == euclidean_distance_dict(b, a)
        b_rev = {k: b[k] for k in reversed(list(b))}
        asymmetric += euclidean_distance_dict(a, b_rev) != euclidean_distance_dict(b_rev, a)
    # This is why the fast class stores each direction separately.
    assert asymmetric > 0


# -- the gap, upstream and on the reference --------------------------------------
@pytest.mark.parametrize("cls", [CountingUpstream, CountingPatched],
                         ids=["DenStream", "DenStreamPatched"])
def test_gap_every_pair_on_every_predict(cls):
    # GAP-3 is a claim about river.cluster.DenStream; DenStreamPatched, the
    # model the fast class is exact with, pays the same. If a river update
    # caches the offline phase upstream, the DenStream case fails and
    # RIVER_BUGS.md GAP-3 is stale.
    m = cls(**SITES)
    per_call = []
    for i, x in enumerate(_sites(1500, seed=1)):
        before = m.n_tests
        m.predict_one(x)
        if m.initialized:
            p = len(m.p_micro_clusters)
            per_call.append((m.n_tests - before, p))
            if i % 100 == 0:
                # Back to back, nothing learned in between: the same full scan.
                before = m.n_tests
                m.predict_one(x)
                assert m.n_tests - before == p * (p - 1)
        m.learn_one(x)
    assert per_call and all(n == p * (p - 1) for n, p in per_call)
    assert max(p for _, p in per_call) >= 40


# -- the fix: cost -------------------------------------------------------------
def test_fast_tests_only_the_rows_that_changed():
    m = CountingFast(**SITES)
    first, steady = None, []
    for i, x in enumerate(_sites(1500, seed=1)):
        before = m.pair_tests
        m.predict_one(x)
        if m.initialized:
            p = len(m.p_micro_clusters)
            if first is None:
                first = (m.pair_tests - before, p)
            else:
                steady.append((m.pair_tests - before, p))
            before = m.pair_tests
            m.predict_one(x)
            assert m.pair_tests == before, "a repeated prediction must not test any pair"
        m.learn_one(x)
    # The first prediction after the initial DBSCAN builds every row once: the
    # same P*(P-1) directed tests the reference pays on every call.
    assert first[0] == first[1] * (first[1] - 1)
    # Afterwards one learn_one changes or adds at most one p-micro-cluster,
    # whose row costs 2*(P-1) tests (both directions of every pair).
    assert all(n <= 2 * max(p - 1, 0) for n, p in steady)
    mean_n = sum(n for n, _ in steady) / len(steady)
    mean_p = sum(p for _, p in steady) / len(steady)
    assert mean_n <= 2 * mean_p
    assert mean_p >= 40
    assert m.n_full_tests == 0, "the full reachability test must never run"


def test_wall_clock_against_the_reference_at_p_50(capsys):
    ref, fast = _pair(SITES)
    t_ref = t_fast = 0.0
    ps = []
    for x in _sites(2500, seed=1):
        if ref.initialized and len(ref.p_micro_clusters) >= 45:
            t0 = time.perf_counter()
            a = ref.predict_one(x)
            t1 = time.perf_counter()
            b = fast.predict_one(x)
            t2 = time.perf_counter()
            t_ref += t1 - t0
            t_fast += t2 - t1
            ps.append(len(ref.p_micro_clusters))
            assert a == b
        else:
            assert ref.predict_one(x) == fast.predict_one(x)
        ref.learn_one(x)
        fast.learn_one(x)
    n = len(ps)
    with capsys.disabled():
        print(
            f"\n[GAP-3] {n} predictions at P {min(ps)}-{max(ps)} (mean {sum(ps) / n:.1f}): "
            f"DenStreamPatched {1e3 * t_ref / n:.3f} ms/predict, "
            f"DenStreamPatchedFast {1e3 * t_fast / n:.3f} ms/predict, "
            f"speed-up {t_ref / t_fast:.1f}x"
        )
    assert n >= 1000
    # Loose on purpose: the measured ratio is far larger, this only has to
    # survive a loaded machine.
    assert t_fast * 2 < t_ref


# -- the fix: exactness --------------------------------------------------------
def test_lockstep_jumping_blobs_prequential():
    ref, fast = _pair(JUMPING)
    tr = _lockstep(ref, fast, _prequential(_jumping(3000, seed=2)))
    assert tr.merged_groups > 100
    assert tr.removals > 10 and tr.additions > 10 and tr.overwrites > 5
    assert tr.light > 0 and tr.heavy_to_light > 0
    assert len(tr.timestamps) > 100


def test_lockstep_drifting_blobs_2d():
    params = dict(decaying_factor=0.25, beta=0.75, mu=2, epsilon=0.08, n_samples_init=200,
                  stream_speed=10)
    ref, fast = _pair(params)
    tr = _lockstep(ref, fast, _prequential(_drifting_blobs(3000, seed=1)))
    assert tr.removals > 10 and tr.additions > 10 and tr.overwrites > 5
    assert tr.max_p >= 10


def test_lockstep_stream_speed_one_prunes_often():
    # Timestamp advances on every sample, so weights move on every call and
    # the prune runs every `_time_period` samples.
    params = dict(JUMPING, stream_speed=1, n_samples_init=50)
    ref, fast = _pair(params)
    tr = _lockstep(ref, fast, _prequential(_jumping(2000, seed=3)))
    assert len(tr.timestamps) > 1900
    assert tr.removals > 10 and tr.additions > 10
    assert tr.light > 0


def test_lockstep_predict_and_learn_bursts():
    rng = random.Random(5)
    stream = iter(_jumping(6000, seed=2))

    def ops():
        for _ in range(120):
            for _ in range(rng.randint(1, 80)):
                x = next(stream, None)
                if x is None:
                    return
                yield "learn", x
            for _ in range(rng.randint(1, 30)):
                yield "predict", {"a": rng.uniform(-1.5, 1.5)}
            x = next(stream, None)
            if x is None:
                return
            yield "predict", x
            yield "learn", x

    ref, fast = _pair(JUMPING)
    tr = _lockstep(ref, fast, ops())
    assert tr.predicts > 1000
    assert tr.merged_groups > 0 and tr.removals > 0 and tr.additions > 0


def _overwrite_stream():
    """The minimal repro of the ``len(p_micro_clusters)`` key overwrite.

    Three p-micro-clusters at 0, 10, 20 (keys 0, 1, 2). 0 stops receiving
    points and is pruned, leaving keys {1, 2}; a new blob at 30 then grows an
    o-micro-cluster that is promoted under key ``len(p) == 2`` -- the key of the
    live, heavy cluster at 20, which is silently dropped (denstream.py:223).
    """
    for _ in range(10):
        for v in (0.0, 10.0, 20.0):
            yield {"a": v}
    for _ in range(25):
        for v in (10.0, 20.0):
            yield {"a": v}
    for i in range(10):
        yield {"a": 30.0 if i % 2 else 10.0}


def test_lockstep_promotion_after_a_deletion_overwrites_a_live_key():
    params = dict(decaying_factor=0.25, beta=0.75, mu=2, epsilon=0.5, n_samples_init=30,
                  stream_speed=1)
    ref, fast = _pair(params)
    tr = _lockstep(ref, fast, _prequential(_overwrite_stream()))
    assert tr.overwrites >= 1, "the fixture must hit the key-overwrite path"
    # The cluster at 20 was replaced while far above the prune threshold:
    # river dropped a live micro-cluster, and both models did so identically.
    assert any(c == {"a": 20.0} for _, c in tr.victims)
    assert all(w >= params["mu"] * params["beta"] for w, _ in tr.victims)
    assert all(mc.center != {"a": 20.0} for mc in fast.p_micro_clusters.values())
    # The cache tracks the live objects only: the replaced one was released.
    assert set(map(id, fast._offline_cache.state)) == set(map(id, fast.p_micro_clusters.values()))


@pytest.mark.parametrize("n_samples_init", [1, 5, 50])
def test_lockstep_through_the_init_phase(n_samples_init):
    params = dict(JUMPING, n_samples_init=n_samples_init)
    ref, fast = _pair(params)
    ops = list(_prequential(_jumping(400, seed=6)))
    # Extra predictions before anything is learned and right at the boundary.
    ops = [("predict", {"a": 0.0})] * 3 + ops[: 2 * n_samples_init] + \
        [("predict", {"a": 0.1})] * 3 + ops[2 * n_samples_init:]
    _lockstep(ref, fast, ops)
    assert fast.initialized


def test_lockstep_init_without_any_p_micro_cluster():
    # Every buffered point is isolated, so the initial DBSCAN yields nothing:
    # both models must answer the -1 sentinel (RIVER_BUGS.md BUG-6) identically.
    params = dict(JUMPING, n_samples_init=20)
    ref, fast = _pair(params)
    xs = [{"a": 10.0 * i} for i in range(20)]
    _lockstep(ref, fast, _prequential(xs))
    assert not fast.p_micro_clusters
    assert fast.predict_one({"a": 0.0}) == ref.predict_one({"a": 0.0}) == -1
    assert fast.n_clusters == 0


def _asymmetric_pair():
    """Two heavy micro-clusters whose distance differs by direction, both within reach."""
    rng = random.Random(7)
    for _ in range(20_000):
        p = DenStreamMicroCluster({"a": 0.0, "b": 0.0, "c": 0.0}, 0, 0.25)
        q = DenStreamMicroCluster({"c": 0.0, "b": 0.0, "a": 0.0}, 0, 0.25)
        for _ in range(3):
            p.insert({k: rng.gauss(0, 1) for k in ("a", "b", "c")}, 0)
            q.insert({k: rng.gauss(0, 1) for k in ("c", "b", "a")}, 0)
        d_pq = euclidean_distance_dict(p.center, q.center)
        d_qp = euclidean_distance_dict(q.center, p.center)
        if d_pq != d_qp and p.calc_radius(0) + q.calc_radius(0) >= max(d_pq, d_qp):
            return p, q, d_pq, d_qp
    raise AssertionError("no asymmetric pair found")


def test_edge_direction_follows_the_reference_argument_order():
    p, q, d_pq, d_qp = _asymmetric_pair()
    # 2*epsilon equals the larger distance: `distance < 2*epsilon` then holds
    # in one direction only, so one ordering of the dict links the pair and
    # the other does not.
    eps = max(d_pq, d_qp) / 2
    query = {"a": 0.0, "b": 0.0, "c": 0.0}
    seen = set()
    for order in ((p, q), (q, p)):
        # "together": both rows are built in one call. "staggered": the first
        # is cached alone, then the second arrives, so the first's edge to it
        # is filled in from the second's row -- the case where computing one
        # direction and mirroring it would go wrong.
        for staggered in (False, True):
            (ref, ref_mcs), (fast, fast_mcs) = _crafted(eps, order, n_first=1 if staggered else 2)
            _lockstep(ref, fast, [("predict", query)])
            if staggered:
                ref.p_micro_clusters[1] = ref_mcs[1]
                fast.p_micro_clusters[1] = fast_mcs[1]
                _lockstep(ref, fast, [("predict", query)])
            seen.add(ref.n_clusters)
    assert seen == {1, 2}, "the fixture must make the edge direction matter"


def _crafted(eps, mcs, n_first=None, fast_cls=DenStreamPatchedFast):
    """A reference and a fast model at timestamp 0 holding copies of ``mcs``.

    Only the first ``n_first`` copies go into ``p_micro_clusters``; all of them
    are returned so a test can add the rest later.
    """
    models = []
    for cls in (DenStreamPatched, fast_cls):
        m = cls(epsilon=eps, mu=2, beta=0.75, decaying_factor=0.25, n_samples_init=0)
        m.initialized = True
        m.timestamp = 0
        copies = [copy.deepcopy(mc) for mc in mcs]
        m.p_micro_clusters = dict(enumerate(copies[:n_first]))
        models.append((m, copies))
    return models


def test_identical_centres_with_zero_radius_are_linked():
    # The boundary of `distance <= r_p + r_q`: 0 <= 0 + 0. In more than one
    # dimension the norm-based calc_radius clamps to 0 for most micro-clusters
    # (RIVER_BUGS.md BUG-8), so there this equality is the only way two of
    # them are ever linked; a strict `<` would link nothing.
    def mc():
        m = DenStreamMicroCluster({"a": 1.0, "b": 1.0}, 0, 0.25)
        m.insert({"a": 1.0, "b": 1.0}, 0)
        m.insert({"a": 1.0, "b": 1.0}, 0)
        return m

    p, q = mc(), mc()
    assert p.calc_radius(0) == q.calc_radius(0) == 0.0
    assert euclidean_distance_dict(p.center, q.center) == 0.0
    assert p.calc_weight(0) > 2
    (ref, _), (fast, _) = _crafted(0.1, [p, q])
    _lockstep(ref, fast, [("predict", {"a": 1.0, "b": 1.0})] * 2)
    assert ref.n_clusters == 1, "the reference must link the pair"


def _line_mc(x, spread=0.8, copies=1):
    """A 1-D micro-cluster of ``copies`` times (x - spread, x, x + spread) at time 0.

    In 1-D the norm-based radius is the standard deviation, ~0.653 at spread
    0.8 whatever ``copies`` is, while the weight is 3 * ``copies``.
    """
    mc = DenStreamMicroCluster({"a": x - spread}, 0, 0.25)
    for v in [x, x + spread] + [x - spread, x, x + spread] * (copies - 1):
        mc.insert({"a": v}, 0)
    return mc


#: centres one apart: with epsilon 1.1 and radius ~0.653 a pair at distance 1
#: is directly density-reachable and a pair at distance 2 is not.
CHAIN = (0.0, 1.0, 2.0, 3.0, 4.0)
CHAIN_EPS = 1.1


def _max_hops(model):
    """Most hops the reference labelling needs from a component's first member."""
    m = copy.deepcopy(model)  # the reachability test fills centre memos
    live = list(m.p_micro_clusters.values())
    labelled, deepest = set(), 0
    for start in live:
        if start in labelled:
            continue
        labelled.add(start)
        frontier, hops = [start], 0
        while frontier:
            nxt = [q for cur in frontier for q in live
                   if q not in labelled and m._is_directly_density_reachable(cur, q)]
            nxt = list(dict.fromkeys(nxt))
            labelled.update(nxt)
            hops += bool(nxt)
            frontier = nxt
        deepest = max(deepest, hops)
    return deepest


@pytest.mark.parametrize(
    "order", [(0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (2, 0, 4, 1, 3), (0, 4, 2, 1, 3)],
    ids=["end_first", "other_end_first", "middle_first", "ends_first"],
)
def test_lockstep_on_a_chain_that_needs_several_hops(order):
    # BUG-8's path. Only neighbours along the line are directly
    # density-reachable, so the reference labels the chain as one cluster only
    # by expanding past the direct neighbours of the first member; a traversal
    # that stopped after one hop would split it. No natural stream here builds
    # such a chain, so it is crafted.
    mcs = [_line_mc(CHAIN[i]) for i in order]
    (ref, _), (fast, _) = _crafted(CHAIN_EPS, mcs)
    _lockstep(ref, fast, [("predict", {"a": 4.0}), ("predict", {"a": 0.0})])
    assert ref.n_clusters == 1
    assert _max_hops(ref) >= 2


@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4), (2, 0, 1, 3, 4)],
                         ids=["end_first", "light_link_first"])
def test_lockstep_on_a_chain_cut_and_rejoined_at_a_light_link(order):
    # The middle micro-cluster has half the weight of the others, so as time
    # passes it turns light while they stay heavy. Its geometric edges stay
    # cached; only the per-call weight check can cut the chain. A point
    # learned at its centre makes it heavy again and rejoins the chain.
    mcs = [_line_mc(CHAIN[i], copies=1 if i == 2 else 2) for i in order]
    (ref, _), (fast, _) = _crafted(CHAIN_EPS, mcs, fast_cls=CountingFast)
    _lockstep(ref, fast, [("predict", {"a": 0.0})])
    assert ref.n_clusters == 1 and _max_hops(ref) >= 2

    ref.timestamp = fast.timestamp = 3
    before = fast.pair_tests
    _lockstep(ref, fast, [("predict", {"a": 0.0})])
    assert fast.pair_tests == before, "only weights moved: no row may be recomputed"
    heavy = [mc.calc_weight(ref.timestamp) > ref.mu for mc in ref.p_micro_clusters.values()]
    assert heavy == [i != 2 for i in order]
    assert ref.n_clusters == 3

    before = fast.pair_tests
    _lockstep(ref, fast, [("learn", {"a": 2.0}), ("predict", {"a": 0.0})])
    assert fast.pair_tests - before == 2 * (len(mcs) - 1)
    assert ref.n_clusters == 1 and _max_hops(ref) >= 2


@pytest.mark.parametrize("change", ["swap", "reorder"])
def test_merged_cluster_is_rebuilt_when_its_members_change_but_not_their_stamps(change):
    # One component before and after, with the same first member and every
    # member at N 3 and last_edit_time 0, so only the member objects tell the
    # merged clusters apart. "swap": (A, B) becomes (A, C) because C takes B's
    # key, as denstream.py:223 does when it overwrites a live p-micro-cluster.
    # "reorder": (A, B, C) becomes (A, C, B). River never reorders live
    # p-micro-clusters, but `merge` adds in member order, so the merged sums
    # differ in the last bit.
    a, b, c = _line_mc(1.1), _line_mc(0.1), _line_mc(2.1)
    swap = change == "swap"
    (ref, ref_mcs), (fast, fast_mcs) = _crafted(CHAIN_EPS, [a, b, c], n_first=2 if swap else 3)
    _lockstep(ref, fast, [("predict", {"a": 1.0})])
    first = _mc_state(ref.clusters[0])
    for m, (a, b, c) in ((ref, ref_mcs), (fast, fast_mcs)):
        m.p_micro_clusters = {0: a, 1: c} if swap else {0: a, 2: c, 1: b}
    _lockstep(ref, fast, [("predict", {"a": 1.0})])
    assert ref.n_clusters == 1
    assert _mc_state(ref.clusters[0]) != first, "the fixture must change the merged cluster"


def test_lockstep_shuffled_keys_and_missing_features():
    ref, fast = _pair(JUMPING)
    ops = list(_prequential(_shuffled_sparse(2000, seed=8)))
    tr = _lockstep(ref, fast, ops)
    centers = [mc.center for mc in ref.p_micro_clusters.values()]
    asym = sum(
        euclidean_distance_dict(a, b) != euclidean_distance_dict(b, a)
        for i, a in enumerate(centers) for b in centers[i + 1:]
    )
    assert asym > 0, "the stream must produce asymmetric centre pairs"
    assert tr.merged_groups > 0


def _adapters(params, scaler="standard"):
    def build(cls):
        return make_adapter(cls.__name__, cls(**params), dict(params),
                            scaler=make_scaler(scaler), scaler_name=scaler)
    return build(DenStreamPatched), build(DenStreamPatchedFast)


def _lockstep_adapters(ad_ref, ad_fast, xs):
    """The runner's per-sample step (``process_one``), compared after every sample."""
    tr = _Trace()
    for step, x in enumerate(xs):
        before = tr.before_learn(ad_ref.model)
        seen = tr.seen_by_predict(ad_ref.model)
        y_ref, y_fast = ad_ref.process_one(dict(x)), ad_fast.process_one(dict(x))
        assert y_fast == y_ref, f"prediction differs at step {step}"
        assert _clusters_state(ad_fast.model) == _clusters_state(ad_ref.model)
        _assert_same(ad_ref.model, ad_fast.model, step)
        tr.after_learn(ad_ref.model, before)
        tr.after_predict(ad_ref.model, seen)
        tr.edges_of(ad_fast.model)
    assert ad_fast.n_fallback_predictions == ad_ref.n_fallback_predictions
    return tr


def test_lockstep_through_the_adapter_process_one():
    ad_ref, ad_fast = _adapters(dict(JUMPING, epsilon=0.05))
    tr = _lockstep_adapters(ad_ref, ad_fast, _jumping(2500, seed=9))
    assert tr.max_p >= 5


BANANAS = dict(decaying_factor=0.25, beta=0.5, mu=2.5, epsilon=0.1, n_samples_init=300,
               stream_speed=100)


def test_lockstep_on_bananas_through_the_adapter():
    # Bundled with river, so this non-synthetic slice always runs.
    ad_ref, ad_fast = _adapters(BANANAS)
    xs = (x for x, _ in datasets.Bananas())
    tr = _lockstep_adapters(ad_ref, ad_fast, (x for _, x in zip(range(3000), xs)))
    assert tr.merged_groups > 0
    assert tr.max_p >= 20


def test_run_once_gives_identical_predictions_and_scores_on_bananas():
    from benchmark.datasets import from_river_dataset

    ds = from_river_dataset(datasets.Bananas(), "bananas", n_samples=2000)
    results, preds = [], []
    for ad in _adapters(BANANAS):
        p = []
        res = run_once(ad, ds, n_checkpoints=4, window_size=500,
                       internal_metrics=("silhouette",), predictions=p)
        results.append(res)
        preds.append(p)
    assert preds[0] == preds[1]
    assert results[0].quality == results[1].quality
    assert [w.window_metrics for w in results[0].windows] == \
        [w.window_metrics for w in results[1].windows]


# With every feature, the norm-based calc_radius of a multi-dimensional
# micro-cluster is 0 almost always (RIVER_BUGS.md BUG-8), so on these slices
# no pair is ever linked: every p-micro-cluster is its own component, and the
# "all" cases are only smoke tests of exactness on real data. The projections
# onto one or two features do link pairs and merge components, and assert it.
def _real_slice(load, features, epsilon):
    params = dict(BANANAS, epsilon=epsilon, n_samples_init=500, stream_speed=20)
    ad_ref, ad_fast = _adapters(params)
    xs = (x if features is None else {f: x[f] for f in features} for x, _ in load())
    tr = _lockstep_adapters(ad_ref, ad_fast, xs)
    assert tr.max_p >= 2
    if features is not None:
        assert tr.max_edges > 0 and tr.merged_groups > 0, "the projection must link pairs"


@pytest.mark.parametrize(("features", "epsilon"), [(None, 0.3), (("f3", "f4"), 0.2)],
                         ids=["all", "f3_f4"])
def test_lockstep_on_an_insects_slice(features, epsilon):
    variant = datasets.Insects(variant="abrupt_balanced")
    if not variant.is_downloaded:
        pytest.skip("INSECTS abrupt_balanced is not in the local river cache")
    from benchmark.datasets import load_insects

    _real_slice(lambda: load_insects("abrupt_balanced", n_samples=2500), features, epsilon)


@pytest.mark.parametrize(("features", "epsilon"),
                         [(None, 0.5), (("vertical_distance_to_hydrology",), 0.05)],
                         ids=["all", "vertical_distance_to_hydrology"])
def test_lockstep_on_a_covtype_slice(features, epsilon):
    sklearn_datasets = pytest.importorskip("sklearn.datasets")
    try:
        sklearn_datasets.fetch_covtype(download_if_missing=False)
    except OSError:
        pytest.skip("covtype is not in the local scikit-learn cache")
    from benchmark.datasets import load_covtype

    _real_slice(lambda: load_covtype(n_samples=2500), features, epsilon)


def test_lockstep_after_epsilon_is_changed_in_place():
    # Nothing in the framework does this today, but the cached geometry
    # depends on epsilon, so an in-place change must not go unnoticed.
    # Lowering it is the direction that bites: raising epsilon cannot link an
    # unchanged pair, because `distance <= r_p + r_q` already binds there.
    ref, fast = _pair(JUMPING)
    xs = list(_jumping(3000, seed=2))
    _lockstep(ref, fast, _prequential(xs[:1000]))
    i = 1000
    while True:  # stop at a prediction that merges p-micro-clusters
        _lockstep(ref, fast, [("predict", xs[i])])
        if ref.n_clusters < len(ref.p_micro_clusters):
            break
        _lockstep(ref, fast, [("learn", xs[i])])
        i += 1
    unchanged = copy.deepcopy(ref)
    ref.epsilon = fast.epsilon = 0.04
    _lockstep(ref, fast, [("predict", xs[i])])
    unchanged.predict_one(xs[i])
    assert ref.n_clusters != unchanged.n_clusters, "the change must alter the labelling"
    _lockstep(ref, fast, [("learn", xs[i])] + list(_prequential(xs[i + 1:])))


def test_caller_mutating_an_input_after_learn_one_shows_in_clusters():
    # RIVER_BUGS.md BUG-3: a micro-cluster keeps the dict it was created from
    # as `x`, by reference. The reference deep-copies it into `clusters[k].x`
    # on every call, so an input edited after learn_one shows up there; a
    # reused merged cluster must not keep serving the old copy.
    params = dict(decaying_factor=0.25, beta=0.75, mu=2, epsilon=0.08, n_samples_init=200,
                  stream_speed=10)
    ref, fast = _pair(params)
    xs = list(_drifting_blobs(1500, seed=1))
    _lockstep(ref, fast, _prequential(xs[:1000]))
    _lockstep(ref, fast, [("predict", xs[1000])])
    first = next(iter(fast.p_micro_clusters.values()))
    assert first.x is next(iter(ref.p_micro_clusters.values())).x, "both alias the input"
    assert list(first.x) == ["x0", "x1"]

    first.x["x0"] = 99.0  # a value
    _lockstep(ref, fast, [("predict", xs[1000])])
    assert fast.clusters[0].x["x0"] == 99.0
    first.x["x0"] = first.x.pop("x0")  # only the key order
    _lockstep(ref, fast, [("predict", xs[1000])])
    assert list(fast.clusters[0].x) == ["x1", "x0"]
    _lockstep(ref, fast, _prequential(xs[1001:]))


def test_caller_inserting_into_a_cluster_does_not_get_it_back():
    # The reference builds fresh merged clusters on every call; this class
    # reuses unchanged ones, so a caller's insert into `clusters[k]` must not
    # survive into the next prediction.
    ref, fast = _pair(JUMPING)
    xs = list(_jumping(1500, seed=2))
    _lockstep(ref, fast, _prequential(xs[:1000]))
    _lockstep(ref, fast, [("predict", xs[1000])])
    for m in (ref, fast):
        m.clusters[0].insert({"a": 5.0}, m.timestamp)
    _lockstep(ref, fast, [("predict", xs[1000])])
    _lockstep(ref, fast, _prequential(xs[1001:]))


def test_caller_editing_a_returned_centre_does_not_get_it_back():
    # `centers` hands out each merged cluster's centre memo itself, and the
    # runner calls it on every sample. The reference builds fresh clusters on
    # every call, so an edit is gone by the next prediction; a reused merged
    # cluster must not keep it.
    ref, fast = _pair(JUMPING)
    xs = list(_jumping(1500, seed=2))
    _lockstep(ref, fast, _prequential(xs[:1000]))
    q = xs[1000]
    _lockstep(ref, fast, [("predict", q)])
    k = ref._get_closest_cluster_key(q, ref.clusters)
    assert ref.n_clusters > 1, "the edit must be able to move the answer"
    for m in (ref, fast):
        m.centers[k]["a"] += 100.0
    _lockstep(ref, fast, [("predict", q)])
    assert ref._get_closest_cluster_key(q, ref.clusters) == k
    _lockstep(ref, fast, _prequential(xs[1001:]))


def test_predict_one_fills_centre_memos_where_the_reference_does():
    # The reference fills the `_center` memo of heavy p-micro-clusters only.
    # Every lockstep step compares memos; this stream (stream_speed 1, a high
    # mu) brings many light p-micro-clusters with an empty memo to
    # predict_one, and with memos equal the pickled live state is identical.
    params = dict(decaying_factor=0.5, beta=0.3, mu=6, epsilon=0.1, n_samples_init=200,
                  stream_speed=1)
    rng = random.Random(3)
    xs = [{"a": rng.gauss(rng.choice((0.0, 0.12, 0.6, 1.5)), 0.06)} for _ in range(3000)]
    ref, fast = _pair(params)
    tr = _Trace()
    for op in _prequential(xs):
        _lockstep(ref, fast, [op], tr)
        if op[0] == "predict":
            assert pickle.dumps(fast.p_micro_clusters) == pickle.dumps(ref.p_micro_clusters)
    assert tr.bare_light > 100, "light p-micro-clusters with an empty memo must reach predict_one"


# -- persistence ---------------------------------------------------------------
def test_pickle_roundtrip_mid_stream_keeps_the_cache_and_exactness():
    ref, fast = _pair(SITES, fast_cls=CountingFast)
    xs = list(_sites(1600, seed=1))
    _lockstep(ref, fast, _prequential(xs[:800]))
    _lockstep(ref, fast, [("predict", xs[800])])  # cache now current
    restored = pickle.loads(pickle.dumps(fast))
    assert restored.pair_tests == fast.pair_tests

    # The cache came back attached to the restored micro-clusters, so a
    # repeated prediction tests nothing, and one learn costs one row, not a
    # rebuild.
    p = len(restored.p_micro_clusters)
    before = restored.pair_tests
    _lockstep(ref, restored, [("predict", xs[800])])
    assert restored.pair_tests == before
    before = restored.pair_tests
    _lockstep(ref, restored, [("learn", xs[800]), ("predict", xs[801])])
    assert restored.pair_tests - before <= 2 * p

    _lockstep(ref, restored, [("learn", xs[801])] + list(_prequential(xs[802:])))
    assert p >= 40


def test_clone_starts_fresh_and_stays_exact():
    ref, fast = _pair(JUMPING)
    xs = list(_jumping(2400, seed=11))
    _lockstep(ref, fast, _prequential(xs[:1200]))
    fresh = fast.clone()
    assert type(fresh) is DenStreamPatchedFast
    assert fresh._get_params() == fast._get_params()
    assert "_offline_cache" not in fresh.__dict__
    assert not fresh.p_micro_clusters
    _lockstep(ref.clone(), fresh, _prequential(xs[1200:]))


@pytest.mark.parametrize("how", ["clone_with_attributes", "deepcopy"])
def test_copies_mid_stream_stay_exact(how):
    # clone(include_attributes=True) deep-copies every attribute on its own,
    # so the copied cache refers to micro-cluster copies that are not the ones
    # in the clone's p_micro_clusters: it has to be detected as stale, not used.
    ref, fast = _pair(JUMPING)
    xs = list(_jumping(2400, seed=12))
    _lockstep(ref, fast, _prequential(xs[:1200]))
    if how == "deepcopy":
        ref2, fast2 = copy.deepcopy(ref), copy.deepcopy(fast)
    else:
        ref2, fast2 = ref.clone(include_attributes=True), fast.clone(include_attributes=True)
    _lockstep(ref2, fast2, _prequential(xs[1200:]))
    # The originals are untouched by the copies and still agree.
    _lockstep(ref, fast, _prequential(xs[1200:1500]))


def test_signature_and_lineage():
    assert issubclass(DenStreamPatchedFast, DenStreamPatched)
    assert issubclass(DenStreamPatchedFast, cluster.DenStream)
    # No constructor of its own: clone() and accepts_seed() read DenStream's.
    assert DenStreamPatchedFast.__init__ is cluster.DenStream.__init__
    assert DenStreamPatchedFast()._get_params() == cluster.DenStream()._get_params()
