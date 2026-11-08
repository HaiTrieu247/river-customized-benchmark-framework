import numpy as np
import pytest

from benchmark.datasets import PARAM_DRIFT_SCENARIOS, make_param_drift


@pytest.mark.parametrize("name", sorted(PARAM_DRIFT_SCENARIOS))
def test_every_scenario_has_the_requested_length_and_known_drift_points(name):
    ds = make_param_drift(name, n_samples=3000)
    rows = list(ds)
    assert len(rows) == 3000 == len(ds) and ds.has_labels and ds.n_features == 5 + len(ds.irrelevant_features)
    assert all(len(x) == ds.n_features for x, _ in rows)
    assert all(0 < p < 3000 for p in ds.drift_points) and list(ds.drift_points) == sorted(ds.drift_points)
    # a label is None only for the generator's background noise, which only E8's noise level has
    assert all(y is not None for _, y in rows) == all(s.noise == 0 for s in ds.segments)


def test_scale_drift_multiplies_the_features_by_the_schedule():
    ds = make_param_drift("scale_drift", n_samples=3000)      # boundaries 1000, 2000; factors 1, 4, 0.25
    rows = list(ds)
    seg = lambda a, b: np.array([[v for v in x.values()] for x, _ in rows[a:b]])
    assert np.allclose(seg(1000, 2000), 4.0 * seg(0, 1000))
    assert np.allclose(seg(2000, 3000), 0.25 * seg(0, 1000))
    assert [y for _, y in rows[:1000]] == [y for _, y in rows[1000:2000]]


def test_density_drift_keeps_the_same_centres_and_changes_the_radius():
    ds = make_param_drift("density_drift", n_samples=3000)
    rows = list(ds)
    def radial(a, b):
        X = np.array([[v for v in x.values()] for x, _ in rows[a:b]]); y = np.array([y for _, y in rows[a:b]])
        return np.mean([np.linalg.norm(X[y == c] - X[y == c].mean(0), axis=1).mean() for c in np.unique(y)])
    assert radial(1000, 2000) > 2.5 * radial(0, 1000)
    assert set(y for _, y in rows[:1000]) == set(y for _, y in rows[1000:2000])


def test_k_drift_changes_the_number_of_live_clusters():
    ds = make_param_drift("k_drift", n_samples=3000)
    rows = list(ds)
    assert len(set(y for _, y in rows[:1000])) == 4
    assert len(set(y for _, y in rows[1000:2000])) == 10
    assert len(set(y for _, y in rows[2000:])) == 3


def test_stream_is_reiterable_and_deterministic():
    ds = make_param_drift("mixed_drift", n_samples=600)
    assert list(ds) == list(ds)


def test_too_short_stream_is_rejected():
    with pytest.raises(ValueError):
        make_param_drift("scale_drift", n_samples=2)


# --- the X7 variants -------------------------------------------------------------------------

from itertools import islice  # noqa: E402

from benchmark.datasets.synthetic.generator import StreamConfig, SyntheticStream  # noqa: E402
from benchmark.datasets.synthetic.param_drift import (  # noqa: E402
    DRIFT_VARIANT_SCENARIOS, ParamDriftStream, Segment, random_segments,
)


def _X(rows):
    return np.array([[v for v in x.values()] for x, _ in rows])


def test_the_abrupt_scenarios_still_concatenate_the_segments_point_for_point():
    ds = make_param_drift("k_drift", n_samples=3000)
    expected = []
    for seg, n in zip(ds.segments, (1000, 1000, 1000)):
        cfg = StreamConfig(n_samples=n, n_features=5, n_clusters=seg.n_clusters,
                           cluster_radius=seg.cluster_radius, separation=None, seed=42)
        expected += [(x, y + seg.label_offset) for x, y in islice(SyntheticStream(cfg), n)]
    assert list(ds) == expected


@pytest.mark.parametrize("name,factor", [("scale_drift_x1_5", 1.5), ("scale_drift_x8", 8.0)])
def test_the_magnitude_variants_scale_the_middle_and_last_segments(name, factor):
    rows = list(make_param_drift(name, n_samples=3000))
    assert np.allclose(_X(rows[1000:2000]), factor * _X(rows[:1000]))
    assert np.allclose(_X(rows[2000:]), _X(rows[:1000]) / factor)


def test_random_drift_changes_kind_at_every_point_and_off_the_window_grid():
    segs = DRIFT_VARIANT_SCENARIOS["random_drift"]["segments"]
    kinds = [(s.n_clusters, s.cluster_radius, s.scale) for s in segs]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    assert all(s.frac >= 0.08 for s in segs) and sum(s.frac for s in segs) == pytest.approx(1.0)
    assert random_segments(7, 5, 0.08) == segs                                  # deterministic
    ds = make_param_drift("random_drift", n_samples=100_000)
    assert len(ds.drift_points) == 4 and all(p % 1000 for p in ds.drift_points)
    assert all(s.label_offset == 100 * s.n_clusters for s in segs)


def test_a_gradual_change_mixes_the_two_segments_along_moas_sigmoid():
    segs = (Segment(0.5), Segment(0.5, label_offset=100))
    ds = ParamDriftStream("g", segs, n_samples=20_000, n_features=3, seed=1, width=0.2)   # w = 4000 around 10 000
    rows = list(ds)
    new = np.array([y >= 100 for _, y in rows])
    # P(new) = 1 / (1 + exp(-4 (t - p) / w)): 3e-4 at p - 2w, 0.018 at p - w
    assert not new[:2000].any() and new[:6000].mean() < 0.01 and new[14_500:].mean() > 0.97
    assert abs(new[9500:10_500].mean() - 0.5) < 0.06
    assert new[8000:9000].mean() < new[9000:10_000].mean() < new[10_000:11_000].mean() < new[11_000:12_000].mean()
    # each segment's points come in their own order: the old ones are the abrupt stream's prefix
    abrupt = list(ParamDriftStream("a", segs, n_samples=20_000, n_features=3, seed=1))
    old = [r for r, n in zip(rows, new) if not n]
    assert old == abrupt[:len(old)]
    assert list(ds) == rows                                                    # re-iterable


def test_scale_drift_gradual_has_the_abrupt_drift_points_and_a_width():
    ds = make_param_drift("scale_drift_gradual", n_samples=3000)
    assert ds.drift_points == make_param_drift("scale_drift", n_samples=3000).drift_points and ds.width == 0.1
    assert len(list(ds)) == 3000


# --- E8: class identities, stationary controls, the factorial family ------------------------

import hashlib  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import replace  # noqa: E402
from itertools import combinations  # noqa: E402

from benchmark.datasets.synthetic.param_drift import (  # noqa: E402
    E8_FACTORIAL_SCENARIOS, E8_SAME_ID_SCENARIOS, E8_SCENARIOS, E8_STATIONARY_SCENARIOS,
    FACTORIAL_DIFFICULTIES, FACTORIAL_MECHANISMS, FACTORIAL_TIMINGS, FACTORIAL_TYPES, FACTORIAL_WIDTH,
    FACTORIAL_RECURRENCE_SPAN, factorial_scenarios,
)


def _digest(ds) -> str:
    h = hashlib.sha256()
    for x, y in ds:
        h.update(repr((list(x.items()), y)).encode())
    return h.hexdigest()


#: Hashes of every scenario that existed before E8, recorded from the code before E8 was added
#: (PYTHONHASHSEED=0): (name, n_samples, seed) -> (sha256 of the whole stream, drift points).
_BEFORE_E8 = {
    ("density_drift", 3000, 42): ("7799d84a5d068f5e0c254f63f79300b8a200ee5aaf41d8b65bbd6799fc2f8088", (1000, 2000)),
    ("k_drift", 3000, 42): ("fe4cad4ccb8a6cdac42608c1b55b42c231338649a6722bd71b43883fda3577c0", (1000, 2000)),
    ("mixed_drift", 3000, 42): ("46679a2f6940ed08195751c0daab5ceaefea26af46e86060859f4d6bacc4bd2b", (1500,)),
    ("random_drift", 3000, 42): ("a423ec858de6c37b3e3f2681f902fabbae10140cb9eaf391f84d5f103a4d1eb4", (614, 1396, 1937, 2650)),
    ("scale_drift", 3000, 42): ("4d7f671e406575280574644bcc520a8742bc8d763b65381709d5dcef37946f3b", (1000, 2000)),
    ("scale_drift_gradual", 3000, 42): ("f11c897a7b083e9434df340303f7bedd5aca54ec14f14a5cf29a2272f48fd8ba", (1000, 2000)),
    ("scale_drift_x1_5", 3000, 42): ("8ff081fe4240f73ceb72f9bc9e0e0ce122af073ad75a4d763cffe47aa1fb5148", (1000, 2000)),
    ("scale_drift_x8", 3000, 42): ("95a0c76567a7cf57d930705669d9d31ea49bee7e93652c1c6a92e14c763360ea", (1000, 2000)),
    ("density_drift", 2000, 3): ("d56e559d6f5ef5d1b458164cd031de83bc1ee908037a19b0e9abaeb3a627aa1f", (667, 1334)),
    ("k_drift", 2000, 3): ("db27978507f393aea5b79f5df128d56f9eb4bd24635ec8cb3072b11932e6508e", (667, 1334)),
    ("mixed_drift", 2000, 3): ("e3f23e810f6808f993e0b6653692c343ae82de3d5dba946d436af82c19d17222", (1000,)),
    ("random_drift", 2000, 3): ("68d766ca92f68ddf90a9c4f169b30296b437b23c9cdbd98b0eea827c42ecd777", (410, 932, 1292, 1768)),
    ("scale_drift", 2000, 3): ("65bd7af527dbb39af9c723cc9e8380071d77449a6a964ddae6ae5acf01414c70", (667, 1334)),
    ("scale_drift_gradual", 2000, 3): ("be802e24daaa88abcebb471bbaf0b30b3642916e4cdf44627fe1b6050e9dd01d", (667, 1334)),
    ("scale_drift_x1_5", 2000, 3): ("b3caae47b65dce1da351e5d501e457072c4f05c656c423804d2a7ad35d1000cb", (667, 1334)),
    ("scale_drift_x8", 2000, 3): ("d2f6f774861c3b4d8d549ccb095e69a316af5c80476e9b7033121fe767738d0d", (667, 1334)),
}


@pytest.mark.parametrize("key", sorted(_BEFORE_E8))
def test_the_scenarios_before_e8_produce_byte_identical_streams(key):
    name, n, seed = key
    ds = make_param_drift(name, n_samples=n, seed=seed)
    assert (_digest(ds), ds.drift_points) == _BEFORE_E8[key]


def test_the_pre_e8_scenarios_are_all_guarded():
    assert {k[0] for k in _BEFORE_E8} == set(PARAM_DRIFT_SCENARIOS) - set(E8_SCENARIOS)


def _centres(n_clusters: int, seed: int, n_features: int = 5) -> np.ndarray:
    s = SyntheticStream(StreamConfig(n_features=n_features, n_clusters=n_clusters, cluster_radius=0.05,
                                     separation=None, seed=seed))
    return np.array([c.centre for c in s._initial_clusters(np.random.default_rng(seed))])


@pytest.mark.parametrize("seed", [1, 7, 42, 123])
def test_generator_ids_align_with_the_centres_that_persist_across_cluster_counts(seed):
    c3, c4, c10 = _centres(3, seed), _centres(4, seed), _centres(10, seed)
    assert np.array_equal(c10[:4], c4) and np.array_equal(c4[:3], c3)


@pytest.mark.parametrize("name", ["k_drift", "mixed_drift"])
def test_same_ids_variants_have_the_original_points_and_only_drop_the_label_offsets(name):
    orig = make_param_drift(name, n_samples=3000, seed=9)
    same = make_param_drift(f"{name}_same_ids", n_samples=3000, seed=9)
    a, b = list(orig), list(same)
    assert same.drift_points == orig.drift_points and [x for x, _ in a] == [x for x, _ in b]
    offsets = [orig.segments[j].label_offset for j in orig.active_segments()]
    assert [y for _, y in b] == [y - off for (_, y), off in zip(a, offsets)]
    assert any(off for off in offsets) and all(s.label_offset == 0 for s in same.segments)


def _label_means(rows):
    X, y = _X(rows), np.array([lab for _, lab in rows])
    return {int(c): X[y == c].mean(0) for c in np.unique(y)}


def test_k_drift_same_ids_gives_a_persisting_component_its_id_in_every_segment():
    seed = 11
    ds = make_param_drift("k_drift_same_ids", n_samples=9000, seed=seed)
    rows = list(ds)
    c4, c10, c3 = _centres(4, seed), _centres(10, seed), _centres(3, seed)
    for (a, b), centres in zip([(0, 3000), (3000, 6000), (6000, 9000)], (c4, c10, c3)):
        means = _label_means(rows[a:b])
        assert sorted(means) == list(range(len(centres)))
        for c, m in means.items():   # each label sits on its own centre, nowhere near another one
            d = np.linalg.norm(centres - m, axis=1)
            assert d[c] < 0.02 and np.argmin(d) == c
    # labels 0-3 of segment 2 and 0-2 of segment 3 are the components of segment 1
    first = _label_means(rows[:3000])
    assert all(np.linalg.norm(_label_means(rows[3000:6000])[c] - first[c]) < 0.03 for c in range(4))
    assert all(np.linalg.norm(_label_means(rows[6000:])[c] - first[c]) < 0.03 for c in range(3))


def test_mixed_drift_same_ids_keeps_the_ids_of_the_four_centres_it_scales():
    seed = 11
    rows = list(make_param_drift("mixed_drift_same_ids", n_samples=8000, seed=seed))
    first, second = _label_means(rows[:4000]), _label_means(rows[4000:])
    assert sorted(first) == [0, 1, 2, 3] and sorted(second) == list(range(10))
    assert all(np.linalg.norm(second[c] - 4.0 * first[c]) < 0.1 for c in range(4))


def test_stationary_controls_never_change_and_replay_the_base_layouts():
    k4, k10 = make_param_drift("stationary_k4", n_samples=3000), make_param_drift("stationary_k10", n_samples=3000)
    assert k4.drift_points == () == k10.drift_points
    a, b = list(k4), list(k10)
    assert len({y for _, y in a}) == 4 and len({y for _, y in b}) == 10
    k_drift = list(make_param_drift("k_drift", n_samples=3000))
    assert a[:1000] == k_drift[:1000]                                            # segment 1 of k_drift
    assert b[:1000] == [(x, y - 100) for x, y in k_drift[1000:2000]]             # segment 2, without the offset


def test_e8_names_count_and_every_one_is_registered():
    assert len(E8_SAME_ID_SCENARIOS) == 2 and len(E8_STATIONARY_SCENARIOS) == 6
    assert len(E8_FACTORIAL_SCENARIOS) == 3 * 4 * 4 * 2 == 96 and len(E8_SCENARIOS) == 104 == len(set(E8_SCENARIOS))
    assert set(E8_SCENARIOS) <= set(PARAM_DRIFT_SCENARIOS)
    names = {f"fd_{t}_{m}_{d}_{s}" for t in FACTORIAL_TYPES for m in FACTORIAL_MECHANISMS
             for d in FACTORIAL_DIFFICULTIES for s in FACTORIAL_TIMINGS}
    assert set(E8_FACTORIAL_SCENARIOS) == names
    assert set(E8_STATIONARY_SCENARIOS) == {"stationary_k4", "stationary_k10"} | {
        f"fd_stationary_{d}" for d in FACTORIAL_DIFFICULTIES}


def test_the_factorial_subset_is_selectable():
    sub = factorial_scenarios(types=("recurring",), mechanisms=("count", "scale"), difficulties=("noise",),
                              timings=("late",))
    assert list(sub) == ["fd_recurring_count_noise_late", "fd_recurring_scale_noise_late", "fd_stationary_noise"]
    assert list(factorial_scenarios(types=("abrupt",), mechanisms=("scale",), difficulties=("noise",),
                                    timings=("late",), controls=False)) == ["fd_abrupt_scale_noise_late"]
    assert sub["fd_recurring_count_noise_late"] == PARAM_DRIFT_SCENARIOS["fd_recurring_count_noise_late"]
    with pytest.raises(ValueError):
        factorial_scenarios(types=("sudden",))
    with pytest.raises(ValueError):
        factorial_scenarios(fraction="third")


def test_the_half_fraction_is_balanced_in_every_pair_of_factors():
    half = [n for n in factorial_scenarios(fraction="half", controls=False)]
    assert len(half) == 48 and set(half) <= set(E8_FACTORIAL_SCENARIOS)
    cells = [tuple(n.split("_")[1:]) for n in half]
    sizes = (len(FACTORIAL_TYPES), len(FACTORIAL_MECHANISMS), len(FACTORIAL_DIFFICULTIES), len(FACTORIAL_TIMINGS))
    for i, j in combinations(range(4), 2):
        counts = Counter((c[i], c[j]) for c in cells)
        assert len(counts) == sizes[i] * sizes[j] and len(set(counts.values())) == 1
    with pytest.raises(ValueError):            # one timing: no half fraction
        factorial_scenarios(fraction="half", timings=("late",))
    with pytest.raises(ValueError):            # 1 x 1 x 1 x 2 cannot be halved and keep both timings
        factorial_scenarios(fraction="half", types=("abrupt",), mechanisms=("scale",), difficulties=("noise",))


@pytest.mark.parametrize("name", E8_FACTORIAL_SCENARIOS)
def test_a_factorial_stream_has_its_drift_points_and_type(name):
    kind, mech, diff, timing = name.split("_")[1:]
    ds = make_param_drift(name, n_samples=2000, seed=3)
    p = round(FACTORIAL_TIMINGS[timing] * 2000)
    if kind == "recurring":
        assert ds.drift_points == (p, p + round(FACTORIAL_RECURRENCE_SPAN * 2000)) and ds.width == 0
        assert replace(ds.segments[2], frac=ds.segments[0].frac) == ds.segments[0]        # A -> B -> A
    else:
        assert ds.drift_points == (p,) and ds.width == (FACTORIAL_WIDTH if kind == "gradual" else 0)
    assert ds.lockstep and ds.drift_points == make_param_drift(name, n_samples=2000, seed=99).drift_points


@pytest.mark.parametrize("name", E8_FACTORIAL_SCENARIOS)
def test_a_factorial_stream_is_its_stationary_control_wherever_it_is_in_concept_a(name):
    kind, mech, diff, timing = name.split("_")[1:]
    n, seed = 1500, 5
    ds = make_param_drift(name, n_samples=n, seed=seed)
    rows, ctrl = list(ds), list(make_param_drift(f"fd_stationary_{diff}", n_samples=n, seed=seed))
    in_a = [j % 2 == 0 for j in ds.active_segments()]                 # segments go A, B (, A)
    assert sum(in_a) < n and all(r == c for r, c, a in zip(rows, ctrl, in_a) if a)
    b_rows = [(r, c) for r, c, a in zip(rows, ctrl, in_a) if not a]
    if kind != "gradual":
        assert len(b_rows) == (round(FACTORIAL_RECURRENCE_SPAN * n) if kind == "recurring"
                               else n - round(FACTORIAL_TIMINGS[timing] * n))
    if mech == "scale":                          # the whole vector x 4, the same components
        assert all(_X([r]).tolist() == (4.0 * _X([c])).tolist() and r[1] == c[1] for r, c in b_rows)
        return
    # cluster points differ; background noise (None) is uniform on the cube in both concepts and,
    # under density, the two generators share their seed, so a noise point may coincide
    assert all(r[0] != c[0] for r, c in b_rows if r[1] is not None)
    labels = {r[1] for r, _ in b_rows} - {None}
    if mech == "location":                       # new components, new ids
        assert labels and min(labels) >= 100
    elif mech == "count":                        # 4 persisting ids and new ones, 10 at most
        assert labels <= set(range(10)) and len(labels) > 4
    elif mech == "density" and diff != "noise":  # same centres, same draws, 4 x the spread
        assert all(r[1] == c[1] for r, c in b_rows)
        d = 5
        centre = {}
        for r, c in b_rows:
            implied = (4.0 * _X([c])[0, :d] - _X([r])[0, :d]) / 3.0
            assert np.allclose(centre.setdefault(r[1], implied), implied, atol=1e-9)
            assert _X([r])[0, d:].tolist() == _X([c])[0, d:].tolist()          # irrelevant features unchanged


def test_location_moves_every_centre_to_a_layout_of_its_own():
    seed = 4
    ds = make_param_drift("fd_abrupt_location_imbalance_early", n_samples=8000, seed=seed)
    rows = list(ds)
    a, b = _label_means(rows[:2000]), _label_means(rows[2000:])
    assert sorted(a) == [0, 1, 2, 3] and sorted(b) == [100, 101, 102, 103]
    A, B = np.array([a[c] for c in sorted(a)]), np.array([b[c] for c in sorted(b)])
    assert np.linalg.norm(A[:, None] - B[None], axis=2).min() > 0.1          # no centre stays put
    # layouts of one data seed are never layouts of another (no seed + j collisions)
    layout_seeds = {ParamDriftStream("x", (Segment(1.0),), 10, 5, seed=s)._layout_seed(1) for s in range(100)}
    assert len(layout_seeds) == 100 and not layout_seeds & set(range(100))


def test_irrelevant_features_are_uniform_extras_that_leave_the_informative_ones_alone():
    ds = make_param_drift("fd_stationary_irrelevant", n_samples=4000)
    assert ds.n_features == 10 and ds.irrelevant_features == ("x5", "x6", "x7", "x8", "x9")
    rows = list(ds)
    U = _X(rows)[:, 5:]
    assert U.min() >= 0.0 and U.max() < 1.0 and np.allclose(U.mean(0), 0.5, atol=0.03)
    plain = list(ParamDriftStream("p", ds.segments, 4000, 5, 42, lockstep=True))
    assert [{k: v for k, v in x.items() if k not in ds.irrelevant_features} for x, _ in rows] == [x for x, _ in plain]
    assert [y for _, y in rows] == [y for _, y in plain]


def test_noise_level_labels_the_generators_background_noise_none():
    rows = list(make_param_drift("fd_stationary_noise", n_samples=5000))
    share = np.mean([y is None for _, y in rows])
    assert abs(share - FACTORIAL_DIFFICULTIES["noise"]["noise"]) < 0.02


def test_overlap_and_imbalance_reach_the_generator():
    over = make_param_drift("fd_stationary_overlap", n_samples=10)
    assert over.segments[0].cluster_radius == pytest.approx(0.15)
    dens = make_param_drift("fd_abrupt_density_overlap_early", n_samples=10)
    assert dens.segments[1].cluster_radius == pytest.approx(0.6)
    rows = list(make_param_drift("fd_stationary_imbalance", n_samples=6000))
    counts = Counter(y for _, y in rows)
    assert len(counts) == 4 and max(counts.values()) > 1.5 * min(counts.values())


@pytest.mark.parametrize("name", ["k_drift_same_ids", "stationary_k10", "fd_stationary_noise",
                                  "fd_gradual_location_overlap_late", "fd_recurring_density_irrelevant_early"])
def test_a_data_seed_is_an_independent_realisation_with_the_same_drift_points(name):
    a, b = make_param_drift(name, n_samples=1200, seed=1), make_param_drift(name, n_samples=1200, seed=2)
    ra, rb = list(a), list(b)
    assert a.drift_points == b.drift_points and ra[0][0] != rb[0][0]
    assert list(a) == ra                                                       # re-iterable
