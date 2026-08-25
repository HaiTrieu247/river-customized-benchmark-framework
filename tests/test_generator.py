"""Tests for the synthetic static-cluster generator and dataset interface."""

from __future__ import annotations

from benchmark.datasets.synthetic import StreamConfig, SyntheticStream


def test_stream_reproducible_with_seed():
    cfg = StreamConfig(n_samples=500, n_clusters=3, seed=7)
    s1 = list(SyntheticStream(cfg))
    s2 = list(SyntheticStream(cfg))
    assert [x for x, _ in s1] == [x for x, _ in s2]


def test_stream_length_and_labels():
    cfg = StreamConfig(n_samples=600, n_clusters=4, seed=1)
    stream = SyntheticStream(cfg)
    items = list(stream)
    assert len(items) == 600
    assert len(stream) == 600
    labels = {y for _, y in items}
    assert labels == {0, 1, 2, 3}
    assert stream.has_labels


def test_stream_is_reiterable():
    cfg = StreamConfig(n_samples=100, seed=3)
    stream = SyntheticStream(cfg)
    first = list(stream)
    second = list(stream)  # a fresh pass must reproduce the same stream
    assert [x for x, _ in first] == [x for x, _ in second]


def test_noise_produces_unlabeled_samples():
    cfg = StreamConfig(n_samples=2000, n_clusters=3, noise=0.1, seed=1)
    labels = [y for _, y in SyntheticStream(cfg)]
    assert any(y is None for y in labels)


def test_feature_dimensionality():
    cfg = StreamConfig(n_samples=50, n_features=5, seed=1)
    stream = SyntheticStream(cfg)
    assert stream.n_features == 5
    x, _ = next(iter(stream))
    assert len(x) == 5
