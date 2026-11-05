"""HPStreamStableLabels: River's HPStream with labels that are never reused (RIVER_BUGS.md GAP-6).

River's HPStream hands an evicted cluster's label to the cluster opened in its place, so within one
window of predict-before-learn labels a label can name two clusters. The variant names new clusters
from a growing counter; these tests hold it to the same model as River's point by point, and show
that River's own class does reuse a label inside one window."""

from __future__ import annotations

import random
import warnings

import pytest

from benchmark.algorithms.registry import get_algorithm
from benchmark.algorithms.hpstream_stable_labels import HPStreamStableLabels


def _river_hpstream(**kw):
    from river import cluster

    return cluster.HPStream(**kw)


def _blob(rng, c, d=4, sd=0.3):
    return {f"x{j}": c + rng.gauss(0, sd) for j in range(d)}


def _warm(model, rng, d=4):
    for i in range(60):
        model.learn_one(_blob(rng, [0, 5, 10][i % 3], d))


HEALTHY = dict(max_clusters=3, n_samples_init=60, stream_speed=10, spread_radius_factor=6.0,
               n_projected_dimensions=4, seed=0)


def _window(model, rng):
    """Three points of the blob at 0, then the blobs at 5 and 10 (the one at 0 goes stale), then three
    points far away: the first opens a cluster, which evicts the stalest one, the blob at 0's."""
    out = []
    for c in [0] * 3 + [5, 10] * 4 + [30] * 3:
        x = _blob(rng, c)
        out.append((c, model.predict_one(x)))
        model.learn_one(x)
    return out


def _counterexample(model):
    rng = random.Random(0)
    _warm(model, rng)
    return _window(model, rng)


def test_river_hpstream_hands_the_evicted_label_to_the_newcomer_within_one_window():
    seq = _counterexample(_river_hpstream(**HEALTHY))
    blob0 = {p for c, p in seq if c == 0}
    far_after_birth = {p for c, p in seq[-2:]}
    # the sequential labels give the blob at 0 and the far cluster one label, though the far cluster
    # was opened only after the blob at 0's cluster had been evicted
    assert blob0 == far_after_birth == {0}


def test_the_stable_variant_names_the_newcomer_afresh_and_keeps_everything_else():
    a, b = _counterexample(_river_hpstream(**HEALTHY)), _counterexample(HPStreamStableLabels(**HEALTHY))
    assert [c for c, _ in a] == [c for c, _ in b]
    assert {p for c, p in b if c == 0} == {0} and {p for c, p in b[-2:]} == {HEALTHY["max_clusters"]}
    assert [p for _, p in a[:-2]] == [p for _, p in b[:-2]]       # before the birth: the same names


def _state(cluster):
    return (dict(cluster.sum1), dict(cluster.sum2), cluster.weight, cluster.last_update, cluster.last_seen,
            list(cluster.dimensions))


CONFIGS = [
    HEALTHY,
    # River's defaults scaled down: clusters of one point have no spread, so nearly every point opens
    # a cluster and evicts one (the degenerate regime the audit measured)
    dict(max_clusters=4, n_samples_init=40, stream_speed=10, seed=3),
    # a radius threshold that leaves some clusters without any dimension: they are starved and removed
    dict(max_clusters=5, n_samples_init=40, stream_speed=5, spread_radius_factor=4.0, radius_threshold=0.2,
         n_projected_dimensions=2, seed=7),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_both_models_hold_the_same_clusters_in_the_same_order_and_predict_the_same_cluster(cfg):
    warnings.simplefilter("ignore")
    river_model, stable = _river_hpstream(**cfg), HPStreamStableLabels(**cfg)
    rng = random.Random(11)
    centres = [0, 5, 10, 30, 60]
    stream = [_blob(rng, centres[min(int(rng.random() * 6), 4)] + (40 * (i > 400)), sd=0.5) for i in range(800)]
    owner_stable: dict = {}            # label -> the cluster object it has named (kept alive: `is` is exact)
    n_vanished = n_river_reuse = 0
    for i, x in enumerate(stream):
        pa, pb = river_model.predict_one(dict(x)), stable.predict_one(dict(x))
        ka, kb = list(river_model._clusters), list(stable._clusters)
        assert len(ka) == len(kb)
        assert [_state(river_model._clusters[k]) for k in ka] == [_state(stable._clusters[k]) for k in kb]
        name = dict(zip(ka, kb))
        assert pb == (name[pa] if ka else pa)                      # the same cluster under its own name
        before_a = dict(river_model._clusters)
        before_b = dict(stable._clusters)
        river_model.learn_one(dict(x))
        stable.learn_one(dict(x))
        n_vanished += sum(1 for c in before_b.values() if all(c is not d for d in stable._clusters.values()))
        n_river_reuse += sum(1 for k, c in river_model._clusters.items() if k in before_a and c is not before_a[k])
        for k, c in stable._clusters.items():
            assert owner_stable.setdefault(k, c) is c, f"point {i}: the stable variant reused label {k}"
    assert n_vanished > 0                                          # clusters were evicted or starved
    assert n_river_reuse > 0                                       # ... and River handed their labels on


def test_the_registry_builds_the_variant_with_hpstreams_defaults_and_hpstream_stays_rivers():
    from river import cluster

    v, h = get_algorithm("HPStreamStableLabels"), get_algorithm("HPStream")
    assert type(v.model) is HPStreamStableLabels and type(h.model) is cluster.HPStream
    assert v.params == h.params


def test_under_the_protocols_start_the_variants_window_partition_refines_hpstreams():
    """What the fix moves in a run: within a window the variant's predict-before-learn labels split
    what River's reused labels merged, and nothing else (each variant label maps to one River label)."""
    from benchmark.datasets import make_scenario
    from conf_pipeline import _common as C
    from conf_pipeline.run import _make_adapter_factory
    from conf_pipeline.spaces import SPACES, start_params

    warnings.simplefilter("ignore")
    h, d = 300, 4
    assert start_params("HPStreamStableLabels", h, d) == start_params("HPStream", h, d)
    assert SPACES["HPStreamStableLabels"].params == SPACES["HPStream"].params
    assert C.LABEL_VARIANTS["HPStreamStableLabels"] == "HPStream" and "HPStreamStableLabels" not in C.IMPL_VARIANTS
    assert C.variant_algos()["HPStreamStableLabels"] == "HPStream"
    ads = [_make_adapter_factory(a, h)(start_params(a, h, d), 201) for a in ("HPStream", "HPStreamStableLabels")]
    ds = list(make_scenario("static_overlap", n_samples=6 * h, n_features=d, n_clusters=4, seed=5))
    n_split = 0
    for w in range(6):
        pa, pb = [], []
        for x, _ in ds[w * h:(w + 1) * h]:
            pa.append(ads[0].process_one(dict(x)))
            pb.append(ads[1].process_one(dict(x)))
        to_river: dict = {}
        for a, b in zip(pa, pb):
            assert to_river.setdefault(b, a) == a          # a variant label is one River label
        n_split += len(set(pb)) > len(set(pa))
    assert n_split > 0                                   # River's labels did merge clusters here
