import numpy as np
import pytest

from benchmark.datasets import PARAM_DRIFT_SCENARIOS, make_param_drift


@pytest.mark.parametrize("name", sorted(PARAM_DRIFT_SCENARIOS))
def test_every_scenario_has_the_requested_length_and_known_drift_points(name):
    ds = make_param_drift(name, n_samples=3000)
    rows = list(ds)
    assert len(rows) == 3000 == len(ds) and ds.has_labels and ds.n_features == 5
    assert all(0 < p < 3000 for p in ds.drift_points) and list(ds.drift_points) == sorted(ds.drift_points)
    assert all(y is not None for _, y in rows)


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
