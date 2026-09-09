"""Synthetic dataset generator of BIRCH (Zhang, Ramakrishnan & Livny, SIGMOD '96).

Implements the generator described in Sec. 6.2 and its Table 1 parameters, and the
three base-workload datasets DS1, DS2, DS3 of Table 3 plus their ordered variants
DS1o, DS2o, DS3o (Sec. 6.4).

Generator, verbatim from Sec. 6.2
---------------------------------
Each dataset consists of ``K`` clusters of 2-d data points. A cluster is characterised
by its number of points ``n`` (drawn from ``[n_l, n_h]``), its radius ``r`` (drawn from
``[r_l, r_h]``) and its centre ``c``. Centre placement follows the ``pattern`` parameter:

``grid``
    Centres on a ``sqrt(K) x sqrt(K)`` grid. The distance between neighbouring centres
    on the same row/column is ``k_g * (r_l + r_h) / 2``. Overview:
    ``[0, sqrt(K) * k_g * (r_l + r_h) / 2]`` on both dimensions.
``sine``
    Centres on a sine curve, the ``K`` clusters split into ``n_c`` groups each placed on
    a different cycle. Centre of cluster ``i`` is at ``x = 2*pi*i`` and
    ``y = (K/n_c) * sin(2*pi*i / (K/n_c))``. Overview: ``[0, 2*pi*K]`` on x and
    ``[-K/n_c, +K/n_c]`` on y.
``random``
    Centres uniform in ``[0, K]`` on both dimensions.

Points of a cluster are drawn from a 2-d independent normal distribution with mean ``c``
and **variance ``r^2 / 2`` in each dimension**. That choice makes the BIRCH radius of a
cluster equal to ``r`` and its BIRCH diameter equal to ``sqrt(2) * r``.

Noise, at rate ``r_n`` percent of the whole dataset, is uniform over the overview and is
emitted with label ``None``. ``order`` is ``"randomized"`` (all cluster points and noise
shuffled across the whole dataset) or ``"ordered"`` (points of a cluster kept together,
clusters in generation order, noise appended at the end).

What reproduces and what does not
---------------------------------
Measured on this implementation (seed 42), against the paper's Table 3 ``D_act`` column
and the actual-cluster radius quoted in Sec. 6.4:

===== ================= ================== =================
set   D_act (paper)     D_act (measured)   mean radius
===== ================= ================== =================
DS1   2.00              2.0052             1.4172 (paper: 1.41)
DS2   2.00              2.0052             1.4172
DS3   4.18              2.7827             1.9248
===== ================= ================== =================

DS1 and DS2 reproduce. DS3 does not, and it is not seed noise: over 12 seeds the measured
``D_act`` stays in ``[2.44, 3.07]``. No reading of the stated parameters gets to 4.18 --
``sqrt(2) * E[r]`` with ``r ~ U[0, 4]`` is ``2.83``, and the alternatives (weighting by
pairs, taking a root-mean-square over clusters, or reading the variance as ``r^2`` instead
of ``r^2 / 2``) land between 2.73 and 3.94 while breaking the DS1/DS2 agreement. Table 3's
DS3 row also carries a visible typo in the paper (``r_n = r_n = 0%``). Reaching 4.18 needs
``r_h`` near 6 rather than 4, so ``r_h`` is left as a parameter: pass ``r_h=6.0`` to
``make_ds3`` to match the published number instead of the published setting. Which of the
two to trust is not decidable from the paper.

Running them
------------
``run_benchmark.py`` exposes the six as ``--dataset birch_ds1 .. birch_ds3o``. The runner
truncates every dataset to ``--n-samples``, so the paper's workload needs
``--n-samples 100000`` explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterator, Literal

import numpy as np

from ..base import IterableDataset

Pattern = Literal["grid", "sine", "random"]
Order = Literal["randomized", "ordered"]

SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class BirchGeneratorConfig:
    """Table 1 of the paper. Ranges in the comments are those the paper experimented with."""

    pattern: Pattern = "grid"
    n_clusters: int = 100          # K, 4 .. 256
    n_low: int = 1000              # n_l, 0 .. 2500
    n_high: int = 1000             # n_h, 50 .. 2500
    r_low: float = SQRT2           # r_l, 0 .. sqrt(2)
    r_high: float = SQRT2          # r_h, sqrt(2) .. sqrt(32)
    distance_multiplier: float = 4.0   # k_g, grid only
    n_cycles: int = 4                  # n_c, sine only
    noise_rate: float = 0.0            # r_n in percent, 0 .. 10
    order: Order = "randomized"

    def __post_init__(self) -> None:
        if self.pattern not in ("grid", "sine", "random"):
            raise ValueError(f"unknown pattern {self.pattern!r}")
        if self.order not in ("randomized", "ordered"):
            raise ValueError(f"unknown order {self.order!r}")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if self.n_low > self.n_high:
            raise ValueError("n_low must be <= n_high")
        if self.r_low > self.r_high:
            raise ValueError("r_low must be <= r_high")
        if not 0.0 <= self.noise_rate < 100.0:
            raise ValueError("noise_rate is a percentage in [0, 100)")
        if self.pattern == "sine" and self.n_cycles < 1:
            raise ValueError("n_cycles must be >= 1 for the sine pattern")


def _grid_centers(cfg: BirchGeneratorConfig) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    side = int(round(math.sqrt(cfg.n_clusters)))
    if side * side != cfg.n_clusters:
        raise ValueError(
            f"the grid pattern needs a square n_clusters, got {cfg.n_clusters}"
        )
    spacing = cfg.distance_multiplier * (cfg.r_low + cfg.r_high) / 2.0
    coords = [(i + 1) * spacing for i in range(side)]
    centers = np.array([(x, y) for x in coords for y in coords], dtype=float)
    extent = side * spacing
    return centers, (0.0, extent, 0.0, extent)


def _sine_centers(cfg: BirchGeneratorConfig) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    period = cfg.n_clusters / cfg.n_cycles
    centers = np.array(
        [
            (2.0 * math.pi * i, period * math.sin(2.0 * math.pi * i / period))
            for i in range(1, cfg.n_clusters + 1)
        ],
        dtype=float,
    )
    return centers, (0.0, 2.0 * math.pi * cfg.n_clusters, -period, period)


def _random_centers(
    cfg: BirchGeneratorConfig, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    span = float(cfg.n_clusters)
    centers = rng.uniform(0.0, span, size=(cfg.n_clusters, 2))
    return centers, (0.0, span, 0.0, span)


def _place_centers(
    cfg: BirchGeneratorConfig, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if cfg.pattern == "grid":
        return _grid_centers(cfg)
    if cfg.pattern == "sine":
        return _sine_centers(cfg)
    return _random_centers(cfg, rng)


def _build(cfg: BirchGeneratorConfig, seed: int):
    rng = np.random.default_rng(seed)
    centers, overview = _place_centers(cfg, rng)
    sizes = rng.integers(cfg.n_low, cfg.n_high + 1, size=cfg.n_clusters)
    if cfg.r_low == cfg.r_high:
        radii = np.full(cfg.n_clusters, cfg.r_low, dtype=float)
    else:
        radii = rng.uniform(cfg.r_low, cfg.r_high, size=cfg.n_clusters)

    blocks = []
    labels = []
    for i in range(cfg.n_clusters):
        n = int(sizes[i])
        if n == 0:
            continue
        scale = math.sqrt(radii[i] * radii[i] / 2.0)
        blocks.append(rng.normal(centers[i], scale, size=(n, 2)))
        labels.append(np.full(n, i, dtype=np.int64))

    points = np.vstack(blocks) if blocks else np.empty((0, 2))
    label = np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)

    n_clustered = len(points)
    n_noise = 0
    if cfg.noise_rate > 0.0 and n_clustered > 0:
        fraction = cfg.noise_rate / 100.0
        n_noise = int(round(n_clustered * fraction / (1.0 - fraction)))
    if n_noise > 0:
        x_lo, x_hi, y_lo, y_hi = overview
        noise = np.column_stack(
            [rng.uniform(x_lo, x_hi, n_noise), rng.uniform(y_lo, y_hi, n_noise)]
        )
        points = np.vstack([points, noise])
        label = np.concatenate([label, np.full(n_noise, -1, dtype=np.int64)])

    if cfg.order == "randomized":
        permutation = rng.permutation(len(points))
        points = points[permutation]
        label = label[permutation]

    return points, label, centers, radii, sizes, overview, n_noise


def make_birch_dataset(
    cfg: BirchGeneratorConfig,
    seed: int = 42,
    name: str | None = None,
) -> IterableDataset:
    """Materialise one dataset from the paper's generator.

    Labels are the generating cluster index; noise points carry ``None``. The stream is
    materialised once and replayed identically on every ``iter()``, which is what the
    framework's ``Dataset`` contract asks for.
    """
    points, label, centers, radii, sizes, overview, n_noise = _build(cfg, seed)
    rows = [
        ({"x0": float(p[0]), "x1": float(p[1])}, None if c < 0 else int(c))
        for p, c in zip(points, label)
    ]

    def factory() -> Iterator[tuple[dict[str, float], int | None]]:
        return iter(rows)

    dataset = IterableDataset(
        factory,
        name=name or f"birch_{cfg.pattern}_{cfg.n_clusters}",
        n_features=2,
        has_labels=True,
        length=len(rows),
    )
    dataset.config = cfg
    dataset.seed = seed
    dataset.actual_centers = centers
    dataset.actual_radii = radii
    dataset.actual_sizes = sizes
    dataset.overview = overview
    dataset.n_noise = n_noise
    return dataset


DS1 = BirchGeneratorConfig(
    pattern="grid",
    n_clusters=100,
    n_low=1000,
    n_high=1000,
    r_low=SQRT2,
    r_high=SQRT2,
    distance_multiplier=4.0,
    noise_rate=0.0,
    order="randomized",
)

DS2 = BirchGeneratorConfig(
    pattern="sine",
    n_clusters=100,
    n_low=1000,
    n_high=1000,
    r_low=SQRT2,
    r_high=SQRT2,
    n_cycles=4,
    noise_rate=0.0,
    order="randomized",
)

DS3 = BirchGeneratorConfig(
    pattern="random",
    n_clusters=100,
    n_low=0,
    n_high=2000,
    r_low=0.0,
    r_high=4.0,
    noise_rate=0.0,
    order="randomized",
)


def make_ds1(seed: int = 42, ordered: bool = False, **overrides) -> IterableDataset:
    cfg = replace(DS1, order="ordered" if ordered else "randomized", **overrides)
    return make_birch_dataset(cfg, seed, name="birch_ds1o" if ordered else "birch_ds1")


def make_ds2(seed: int = 42, ordered: bool = False, **overrides) -> IterableDataset:
    cfg = replace(DS2, order="ordered" if ordered else "randomized", **overrides)
    return make_birch_dataset(cfg, seed, name="birch_ds2o" if ordered else "birch_ds2")


def make_ds3(seed: int = 42, ordered: bool = False, **overrides) -> IterableDataset:
    cfg = replace(DS3, order="ordered" if ordered else "randomized", **overrides)
    return make_birch_dataset(cfg, seed, name="birch_ds3o" if ordered else "birch_ds3")


BIRCH_BASE_WORKLOAD = {
    "birch_ds1": lambda seed=42, **kw: make_ds1(seed, **kw),
    "birch_ds2": lambda seed=42, **kw: make_ds2(seed, **kw),
    "birch_ds3": lambda seed=42, **kw: make_ds3(seed, **kw),
    "birch_ds1o": lambda seed=42, **kw: make_ds1(seed, ordered=True, **kw),
    "birch_ds2o": lambda seed=42, **kw: make_ds2(seed, ordered=True, **kw),
    "birch_ds3o": lambda seed=42, **kw: make_ds3(seed, ordered=True, **kw),
}

#: Table 3 ``D_act`` and Table 4 ``D`` of the paper, for comparison against a run.
BIRCH_PAPER_REFERENCE = {
    "birch_ds1": {"d_act": 2.00, "birch_d": 1.87, "time_s": 47.1},
    "birch_ds2": {"d_act": 2.00, "birch_d": 1.99, "time_s": 47.5},
    "birch_ds3": {"d_act": 4.18, "birch_d": 3.39, "time_s": 49.5},
    "birch_ds1o": {"d_act": 2.00, "birch_d": 1.87, "time_s": 47.4},
    "birch_ds2o": {"d_act": 2.00, "birch_d": 1.99, "time_s": 46.4},
    "birch_ds3o": {"d_act": 4.18, "birch_d": 3.26, "time_s": 48.4},
}


def cluster_diameter(points: np.ndarray) -> float | None:
    """BIRCH diameter, eq. (3): root mean square distance over the ordered pairs."""
    n = len(points)
    if n <= 1:
        return None
    squared_sum = float((points**2).sum())
    linear_sum = points.sum(axis=0)
    return math.sqrt(
        max(0.0, 2.0 * (n * squared_sum - float((linear_sum**2).sum())) / (n * (n - 1.0)))
    )


def cluster_radius(points: np.ndarray) -> float | None:
    """BIRCH radius, eq. (2): root mean square distance of the points to the centroid."""
    n = len(points)
    if n == 0:
        return None
    squared_sum = float((points**2).sum())
    linear_sum = points.sum(axis=0)
    return math.sqrt(max(0.0, squared_sum / n - float((linear_sum**2).sum()) / (n * n)))


def weighted_average_diameter(dataset) -> float:
    """The paper's quality measure ``D``, over whatever grouping the labels give.

    Clusters of fewer than two points have no diameter and drop out of the average.
    Noise, which carries label ``None``, is excluded.
    """
    groups: dict[int, list[list[float]]] = {}
    for x, y in dataset:
        if y is None:
            continue
        groups.setdefault(int(y), []).append([x["x0"], x["x1"]])

    total_weight = 0.0
    total = 0.0
    for members in groups.values():
        points = np.asarray(members, dtype=float)
        diameter = cluster_diameter(points)
        if diameter is None:
            continue
        total += len(points) * diameter
        total_weight += len(points)
    return total / total_weight if total_weight else 0.0
