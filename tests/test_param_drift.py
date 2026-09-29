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
