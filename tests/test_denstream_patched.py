"""DenStreamPatched fixes RIVER_BUGS.md BUG-7 and BUG-8, and only those two.

BUG-7: ``DenStream._generate_clusters_for_labels`` does
``copy.copy(micro_clusters[0])`` then ``merge``s the rest into it. A shallow
copy of a ``__slots__`` object shares its dict-valued slots by reference, so
the "merged" cluster's ``linear_sum``/``squared_sum`` are the *same dicts* as
the first member's, and ``merge`` mutates them in place -- corrupting a live
p-micro-cluster from what is documented as a read-only query.

BUG-8: the seed-expansion step in ``predict_one``'s DBSCAN reads
``if labels[neighbor_neighbor] is not None: seed_queue.append(...)`` --
inverted. Only already-labelled neighbours get queued, so a label never
travels past the direct neighbours of the point DBSCAN started from, and a
density-reachable chain of p-micro-clusters never becomes one cluster.

The tests that assert the *upstream* (buggy) behaviour are deliberate: they
prove the fixture actually exercises the bug, so the patched-side assertions
are measuring something rather than passing vacuously.
"""

from __future__ import annotations

# Import order matters: benchmark.algorithms.denstream_patched puts the
# sibling river source tree on sys.path (build_river_path()) as a side effect
# of import, and nothing else in this test file does that before `river` is
# needed -- unlike tests/test_algorithms.py, which never imports river
# directly and always goes through benchmark.* first.
from benchmark.algorithms.denstream_patched import DenStreamPatched
from benchmark.algorithms.registry import AlgorithmSpec, accepts_seed, get_algorithm
from river import cluster
from river.cluster.denstream import DenStreamMicroCluster


def _chain_mc(x, spread, timestamp=0, decaying_factor=0.25):
    """A p-micro-cluster with positive radius, built around ``x``.

    A micro-cluster built from one point repeated has radius 0 (`calc_radius`
    is norm-based: `sqrt(|CF2|/N - (|CF1|/N)**2)`, which is exactly 0 when
    every inserted point is identical). A radius of 0 makes the
    directly-density-reachable test (`distance <= r_p + r_q`) fail for every
    pair identically, which would not exercise BUG-8 at all. Giving each
    cluster a small spread of its own gives it a real, positive radius.
    """
    mc = DenStreamMicroCluster(
        {"a": x - spread}, timestamp=timestamp, decaying_factor=decaying_factor
    )
    mc.insert({"a": x}, timestamp)
    mc.insert({"a": x + spread}, timestamp)
    return mc


def _chain_model(cls):
    """Three p-micro-clusters on a line at a=0, 1, 2, each built with spread 0.8.

    ``epsilon=1.1``, ``mu=2``, ``beta=0.75`` (``mu * beta == 1.5 > 1``, which
    ``DenStream.__init__`` requires -- it takes ``math.log(mu*beta/(mu*beta-1))``,
    so ``mu*beta <= 1`` raises ``ZeroDivisionError``/``ValueError`` before a
    model even exists).

    At spread 0.8 each cluster's radius is ~0.653, so adjacent pairs (distance
    1.0, radius sum ~1.306) are directly density reachable and the two end
    clusters (distance 2.0) are not: a genuine chain, not a clique. Both facts
    were checked numerically against the real `DenStreamMicroCluster` before
    writing this fixture.
    """
    m = cls(epsilon=1.1, mu=2, beta=0.75, decaying_factor=0.25, n_samples_init=0)
    m.initialized = True
    for i, x in enumerate((0.0, 1.0, 2.0)):
        m.p_micro_clusters[i] = _chain_mc(x, spread=0.8)
    return m


def _linear_sums(model):
    return {k: dict(mc.linear_sum) for k, mc in model.p_micro_clusters.items()}


def test_bug7_predict_one_does_not_mutate_the_parent_micro_clusters():
    m = _chain_model(DenStreamPatched)
    before = _linear_sums(m)
    m.predict_one({"a": 0.5})
    assert _linear_sums(m) == before


def test_bug7_is_present_upstream_so_the_patch_is_measuring_something():
    m = _chain_model(cluster.DenStream)
    before = _linear_sums(m)
    m.predict_one({"a": 0.5})
    assert _linear_sums(m) != before


def test_bug8_density_reachable_chain_becomes_one_cluster():
    m = _chain_model(DenStreamPatched)
    m.predict_one({"a": 0.5})
    assert m.n_clusters == 1

    u = _chain_model(cluster.DenStream)
    u.predict_one({"a": 0.5})
    assert u.n_clusters >= 2


def test_registered_and_tunable_like_denstream():
    # mu=3 (not the brief's mu=2): with the default beta=0.5, mu*beta must be
    # > 1 for DenStream's own __init__ or it raises ZeroDivisionError -- this
    # is upstream's constraint, not something the patch introduces. `seed` is
    # also dropped: DenStream's constructor does not accept it (verified this
    # fails identically for plain "DenStream"), so passing it would not be
    # "tunable like DenStream", it would be a call DenStream itself rejects.
    ad = get_algorithm("DenStreamPatched", epsilon=0.5, mu=3)
    assert type(ad.model).__name__ == "DenStreamPatched"
    assert ad.model.epsilon == 0.5
    assert ad.model.mu == 3


def test_accepts_seed_inspects_the_actual_class_not_just_river_cluster():
    # DenStreamPatched's class lives outside `river.cluster` on purpose
    # (RIVER_BUGS.md BUG-7/BUG-8, AGENTS.md rule 1), so `getattr(cluster,
    # "DenStreamPatched", None)` misses it. `accepts_seed` must fall back to
    # the class the registered factory actually builds rather than reading
    # that miss as "no seed" -- DenStream's own __init__ has neither a `seed`
    # parameter nor `**kwargs`, so the correct answer here is still False, but
    # it has to be false for the right reason.
    assert accepts_seed("DenStreamPatched") is False


def test_accepts_seed_fallback_really_inspects_the_constructor(monkeypatch):
    # Prove the fallback isn't just "always False for a non-river class": a
    # subclass that does declare `seed` must report True through the same
    # non-river code path.
    from benchmark.algorithms import registry

    class SeededPatched(DenStreamPatched):
        def __init__(self, seed=None, **kw):
            super().__init__(**kw)
            self.seed = seed

    def _seeded_patched_factory(**kw):
        return SeededPatched(**kw), kw

    monkeypatch.setitem(
        registry.ALGORITHMS, "SeededPatched", AlgorithmSpec(_seeded_patched_factory)
    )
    assert accepts_seed("SeededPatched") is True


def test_denstream_patched_is_registered_next_to_denstream():
    from benchmark.algorithms.registry import ALGORITHMS

    assert "DenStreamPatched" in ALGORITHMS
    spec = ALGORITHMS["DenStreamPatched"]
    assert spec.default_preprocessor == "standard"
    assert "BUG-7" in spec.note and "BUG-8" in spec.note
    assert "BUG-7" in spec.provenance and "BUG-8" in spec.provenance
    assert "upstream untouched" in spec.provenance
