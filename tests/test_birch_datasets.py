"""Tests for the BIRCH (SIGMOD '96) synthetic generator and its base workload.

Each test names the clause of Sec. 6.2 / Table 1 / Table 3 it pins down, so a failure
says which part of the paper the generator stopped matching.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmark.datasets import (
    BIRCH_BASE_WORKLOAD,
    BIRCH_PAPER_REFERENCE,
    BirchGeneratorConfig,
    make_birch_dataset,
    make_ds1,
    make_ds2,
    make_ds3,
    weighted_average_diameter,
)
from benchmark.datasets.synthetic.birch import (
    SQRT2,
    cluster_diameter,
    cluster_radius,
)


def points_of(dataset):
    return np.array([[x["x0"], x["x1"]] for x, _ in dataset])


def groups_of(dataset):
    groups: dict[int, list] = {}
    for x, y in dataset:
        if y is None:
            continue
        groups.setdefault(int(y), []).append([x["x0"], x["x1"]])
    return {k: np.asarray(v) for k, v in groups.items()}


# --- generator geometry -----------------------------------------------------
def test_grid_spacing_is_kg_times_the_mean_radius():
    """Sec. 6.2: neighbouring centres on a row/column are k_g * (r_l + r_h) / 2 apart."""
    cfg = BirchGeneratorConfig(pattern="grid", n_clusters=16, r_low=1.0, r_high=3.0,
                               distance_multiplier=4.0, n_low=1, n_high=1)
    dataset = make_birch_dataset(cfg, seed=0)
    xs = sorted({round(c[0], 9) for c in dataset.actual_centers})

    expected = 4.0 * (1.0 + 3.0) / 2.0
    assert len(xs) == 4
    assert np.allclose(np.diff(xs), expected)


def test_grid_overview_matches_the_stated_range():
    """Sec. 6.2: overview is [0, sqrt(K) * k_g * (r_l + r_h) / 2] on both dimensions."""
    dataset = make_ds1()
    side = int(math.sqrt(100))
    expected = side * 4.0 * SQRT2

    assert dataset.overview == pytest.approx((0.0, expected, 0.0, expected))


def test_grid_pattern_rejects_a_non_square_number_of_clusters():
    with pytest.raises(ValueError):
        make_birch_dataset(BirchGeneratorConfig(pattern="grid", n_clusters=10), seed=0)


def test_sine_centers_follow_the_stated_formula():
    """Sec. 6.2: centre i sits at x = 2*pi*i, y = (K/n_c) * sin(2*pi*i / (K/n_c))."""
    cfg = BirchGeneratorConfig(pattern="sine", n_clusters=20, n_cycles=4, n_low=1, n_high=1)
    centers = make_birch_dataset(cfg, seed=0).actual_centers
    period = 20 / 4

    for index, (x, y) in enumerate(centers, start=1):
        assert x == pytest.approx(2.0 * math.pi * index)
        assert y == pytest.approx(period * math.sin(2.0 * math.pi * index / period))


def test_sine_overview_matches_the_stated_range():
    dataset = make_ds2()

    assert dataset.overview == pytest.approx((0.0, 2.0 * math.pi * 100, -25.0, 25.0))


def test_random_centers_stay_inside_zero_to_k():
    dataset = make_ds3()
    centers = dataset.actual_centers

    assert centers.min() >= 0.0
    assert centers.max() <= 100.0
    assert dataset.overview == pytest.approx((0.0, 100.0, 0.0, 100.0))


# --- point generation -------------------------------------------------------
def test_variance_per_dimension_is_half_the_squared_radius():
    """Sec. 6.2 says variance r^2/2, which makes the BIRCH radius of a cluster equal r."""
    cfg = BirchGeneratorConfig(pattern="random", n_clusters=1, n_low=200_000,
                               n_high=200_000, r_low=3.0, r_high=3.0)
    points = points_of(make_birch_dataset(cfg, seed=7))

    assert np.var(points, axis=0) == pytest.approx([4.5, 4.5], rel=0.02)
    assert cluster_radius(points) == pytest.approx(3.0, rel=0.01)
    assert cluster_diameter(points) == pytest.approx(3.0 * SQRT2, rel=0.01)


def test_cluster_sizes_are_drawn_from_the_stated_range():
    cfg = BirchGeneratorConfig(pattern="random", n_clusters=200, n_low=5, n_high=9,
                               r_low=1.0, r_high=1.0)
    sizes = make_birch_dataset(cfg, seed=3).actual_sizes

    assert sizes.min() >= 5
    assert sizes.max() <= 9
    assert set(np.unique(sizes)) == {5, 6, 7, 8, 9}


def test_a_fixed_radius_range_gives_every_cluster_the_same_radius():
    """Footnote 4: when r_l = r_h the radius is fixed, when n_l = n_h the size is fixed."""
    dataset = make_ds1()

    assert np.allclose(dataset.actual_radii, SQRT2)
    assert np.all(dataset.actual_sizes == 1000)


# --- noise and order --------------------------------------------------------
def test_noise_rate_is_a_share_of_the_whole_dataset():
    """Sec. 6.2: r_n is the percentage of points in the dataset that are noise."""
    dataset = make_ds1(noise_rate=10.0)
    rows = list(dataset)
    n_noise = sum(1 for _, y in rows if y is None)

    assert n_noise / len(rows) == pytest.approx(0.10, abs=1e-4)


def test_noise_lands_inside_the_overview_and_carries_no_label():
    dataset = make_ds1(noise_rate=5.0)
    noise = np.array([[x["x0"], x["x1"]] for x, y in dataset if y is None])
    x_lo, x_hi, y_lo, y_hi = dataset.overview

    assert len(noise) > 0
    assert noise[:, 0].min() >= x_lo and noise[:, 0].max() <= x_hi
    assert noise[:, 1].min() >= y_lo and noise[:, 1].max() <= y_hi


def test_zero_noise_rate_adds_nothing():
    dataset = make_ds1()

    assert dataset.n_noise == 0
    assert all(y is not None for _, y in dataset)


def test_ordered_keeps_each_cluster_together_and_randomized_does_not():
    """Sec. 6.2: the order parameter only permutes, it never changes the points."""
    randomized = list(make_ds1(seed=42))
    ordered = list(make_ds1(seed=42, ordered=True))

    assert sorted(map(repr, randomized)) == sorted(map(repr, ordered))
    assert randomized != ordered
    assert [y for _, y in ordered] == sorted(y for _, y in ordered)
    assert len({y for _, y in randomized[:200]}) > 1


def test_ordered_puts_the_noise_at_the_end():
    ordered = list(make_ds1(noise_rate=5.0, ordered=True))
    labels = [y for _, y in ordered]
    first_noise = labels.index(None)

    assert all(y is None for y in labels[first_noise:])


# --- Dataset contract -------------------------------------------------------
def test_the_stream_replays_identically():
    """base.Dataset promises re-iterability: every algorithm sees the same stream."""
    dataset = make_ds3(seed=11)

    assert list(dataset) == list(dataset)


def test_a_different_seed_gives_a_different_stream():
    assert list(make_ds3(seed=1)) != list(make_ds3(seed=2))


def test_length_and_feature_count_are_reported():
    dataset = make_ds1()

    assert len(dataset) == 100_000
    assert dataset.n_features == 2
    assert dataset.has_labels


# --- the base workload of Table 3 -------------------------------------------
def test_ds1_and_ds2_hold_exactly_one_hundred_thousand_points():
    """Sec. 6.4: 100,000 data points for each dataset."""
    assert len(list(make_ds1())) == 100_000
    assert len(list(make_ds2())) == 100_000


def test_ds3_holds_about_one_hundred_thousand_points():
    """DS3 draws n from U[0, 2000] per cluster, so N is 100,000 only in expectation."""
    assert len(list(make_ds3())) == pytest.approx(100_000, rel=0.05)


def test_ds1_and_ds2_reproduce_the_published_actual_diameter():
    """Table 3: D_act = 2.00 for both."""
    for make in (make_ds1, make_ds2):
        assert weighted_average_diameter(make()) == pytest.approx(2.00, abs=0.02)


def test_ds1_reproduces_the_published_actual_radius():
    """Sec. 6.4 quotes the actual-cluster radius as 1.41."""
    radii = [cluster_radius(points) for points in groups_of(make_ds1()).values()]

    assert np.mean(radii) == pytest.approx(1.41, abs=0.02)


def test_ds3_does_not_reproduce_the_published_actual_diameter():
    """Table 3 says D_act = 4.18; the stated parameters give about 2.78.

    Pinned deliberately. sqrt(2) * E[r] with r ~ U[0, 4] is 2.83, and over twelve seeds the
    measured value stays inside [2.44, 3.07], so this is not seed noise. Reaching 4.18 needs
    r_h near 6. If this test starts failing, the generator changed -- not the paper.
    """
    measured = weighted_average_diameter(make_ds3())

    assert measured == pytest.approx(2.78, abs=0.05)
    assert measured < BIRCH_PAPER_REFERENCE["birch_ds3"]["d_act"]


def test_ds3_reaches_the_published_diameter_with_a_larger_radius_bound():
    assert weighted_average_diameter(make_ds3(r_high=6.0)) == pytest.approx(4.18, abs=0.1)


def test_every_base_workload_entry_builds_and_is_registered():
    for name, factory in BIRCH_BASE_WORKLOAD.items():
        dataset = factory(seed=42)
        assert dataset.name == name
        assert dataset.n_features == 2
        assert name in BIRCH_PAPER_REFERENCE


def test_the_ordered_variants_carry_the_same_points_as_the_randomized_ones():
    """Sec. 6.4: DS1o, DS2o, DS3o differ from DS1..DS3 only in the order parameter."""
    for base, ordered in (("birch_ds1", "birch_ds1o"),
                          ("birch_ds2", "birch_ds2o"),
                          ("birch_ds3", "birch_ds3o")):
        left = sorted(map(repr, BIRCH_BASE_WORKLOAD[base](seed=42)))
        right = sorted(map(repr, BIRCH_BASE_WORKLOAD[ordered](seed=42)))
        assert left == right


# --- the D measure ----------------------------------------------------------
def test_diameter_and_radius_match_their_definitions():
    """Eq. (2) and eq. (3) of the paper, checked against the raw pairwise sums."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    center = points.mean(axis=0)
    n = len(points)

    radius = math.sqrt(sum(float(np.sum((p - center) ** 2)) for p in points) / n)
    pairs = sum(float(np.sum((p - q) ** 2)) for p in points for q in points)
    diameter = math.sqrt(pairs / (n * (n - 1)))

    assert cluster_radius(points) == pytest.approx(radius)
    assert cluster_diameter(points) == pytest.approx(diameter)


def test_singleton_clusters_have_no_diameter():
    assert cluster_diameter(np.array([[1.0, 2.0]])) is None
    assert cluster_diameter(np.empty((0, 2))) is None
    assert cluster_radius(np.array([[1.0, 2.0]])) == pytest.approx(0.0)


# --- validation of the config ----------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"pattern": "spiral"},
        {"order": "sorted"},
        {"n_clusters": 0},
        {"n_low": 10, "n_high": 5},
        {"r_low": 3.0, "r_high": 1.0},
        {"noise_rate": 100.0},
        {"pattern": "sine", "n_cycles": 0},
    ],
)
def test_invalid_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        BirchGeneratorConfig(**kwargs)
