"""Synthetic benchmark datasets following D-Stream (Chen & Tu, KDD '07)
and CluStream (Aggarwal et al., VLDB '03).

Datasets provided:
1. `make_dstream_30k_nonconvex`:
   - 4 complex non-convex intertwined clusters (interlaced crescents and concentric rings).
   - 5,000 uniform random noise points in [0, 1]^2.
   - Evaluates ability to identify arbitrary non-convex clusters without splitting them.

2. `make_dstream_85k_dynamic`:
   - 85,000 points total, with 10,000 uniform random noise points.
   - Sequential cluster emergence, movement, and fading:
     * Phase 1 (0 to 25K, t1): Clusters 1, 2, 3 active and stable.
     * Phase 2 (25K to 55K, t2): Cluster 1 moves, Cluster 2 fades away, Cluster 4 emerges.
     * Phase 3 (55K to 85K, t3): Cluster 2 is completely gone, Cluster 1 stabilizes at new position,
       Cluster 4 is dense, and noise points are removed by sporadic grid cleanup.

3. `make_scalability_dataset`:
   - Scalable stream generator with n_samples in [5K, 85K] and n_features in [2, 40].
"""

from __future__ import annotations

from typing import Iterator
import numpy as np

from ..base import IterableDataset


def _normalize_points(points: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Linearly scale points to [margin, 1 - margin] along each coordinate."""
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    span = np.where(p_max > p_min, p_max - p_min, 1.0)
    scaled = (points - p_min) / span
    return margin + scaled * (1.0 - 2 * margin)


def make_dstream_30k_nonconvex(
    n_samples: int = 30_000,
    n_noise: int = 5_000,
    seed: int = 42,
) -> IterableDataset:
    """Generate 30K 2D stream matching Figure 5 of Chen & Tu (2007).

    4 clusters (non-convex shapes, some interwoven) + 5K random outlier data:
    - Class 1: Top-left inverted arch / crescent
    - Class 2: Bottom-left circular ring
    - Class 3: Top-right disk / solid circle
    - Class 4: Bottom-right figure-8 / double ring
    - Outliers: 5K points uniformly distributed in [0, 1]^2 (labeled None)
    Data points from different clusters and outliers alternately appear in the stream.
    """
    if n_samples < n_noise:
        n_noise = int(n_samples * (5 / 30))
    rng = np.random.default_rng(seed)
    n_cluster_pts = max(4, n_samples - n_noise)
    pts_per_c = max(1, n_cluster_pts // 4)

    # 1. Class 1 (top-left arch / crescent)
    th1 = rng.uniform(np.pi * 0.15, np.pi * 0.85, pts_per_c)
    r1 = 0.18 + rng.normal(0, 0.015, pts_per_c)
    x1 = np.column_stack([0.28 + r1 * np.cos(th1), 0.65 + r1 * np.sin(th1)])
    y1 = np.ones(pts_per_c, dtype=int)

    # 2. Class 2 (bottom-left ring)
    th2 = rng.uniform(0, 2 * np.pi, pts_per_c)
    r2 = 0.14 + rng.normal(0, 0.015, pts_per_c)
    x2 = np.column_stack([0.25 + r2 * np.cos(th2), 0.28 + r2 * np.sin(th2)])
    y2 = np.full(pts_per_c, 2, dtype=int)

    # 3. Class 3 (top-right disk / circle)
    th3 = rng.uniform(0, 2 * np.pi, pts_per_c)
    r3 = np.sqrt(rng.uniform(0, 0.13**2, pts_per_c))
    x3 = np.column_stack([0.75 + r3 * np.cos(th3), 0.72 + r3 * np.sin(th3)])
    y3 = np.full(pts_per_c, 3, dtype=int)

    # 4. Class 4 (bottom-right figure-8 / double ring)
    half = pts_per_c // 2
    th4_1 = rng.uniform(0, 2 * np.pi, half)
    r4_1 = 0.12 + rng.normal(0, 0.012, half)
    ring1 = np.column_stack([0.62 + r4_1 * np.cos(th4_1), 0.28 + r4_1 * np.sin(th4_1)])

    rem = pts_per_c - half
    th4_2 = rng.uniform(0, 2 * np.pi, rem)
    r4_2 = 0.12 + rng.normal(0, 0.012, rem)
    ring2 = np.column_stack([0.83 + r4_2 * np.cos(th4_2), 0.28 + r4_2 * np.sin(th4_2)])

    x4 = np.vstack([ring1, ring2])
    y4 = np.full(pts_per_c, 4, dtype=int)

    cluster_points = np.vstack([x1, x2, x3, x4])
    cluster_labels = np.concatenate([y1, y2, y3, y4])

    noise_points = rng.uniform(0.0, 1.0, (n_noise, 2))
    noise_labels = [None] * n_noise

    all_points = np.vstack([cluster_points, noise_points])
    all_labels = list(cluster_labels) + noise_labels

    perm = rng.permutation(len(all_labels))
    shuffled_X = all_points[perm]
    shuffled_y = [all_labels[i] for i in perm]

    def factory() -> Iterator[tuple[dict[str, float], int | None]]:
        for row, label in zip(shuffled_X, shuffled_y):
            yield {"x0": float(np.clip(row[0], 0.0, 1.0)), "x1": float(np.clip(row[1], 0.0, 1.0))}, label

    return IterableDataset(
        factory,
        name="dstream_30k_nonconvex",
        n_features=2,
        has_labels=True,
    )


def make_dstream_85k_dynamic(
    n_samples: int = 85_000,
    n_noise: int = 10_000,
    seed: int = 42,
) -> IterableDataset:
    """Generate 85K dynamic stream matching Section 5.1 & Figures 7-10 of Chen & Tu (2007).

    Four classes generated sequentially one by one + 10K random outliers uniformly spread:
    - Phase 1 (0 to 25K, t1=25): Class 1 (top-left arch) generated.
    - Phase 2 (25K to 55K, t2=55):
        * Class 1 receives no data -> decays/disappears
        * Class 2 (bottom-left ring) generated early in phase, then disappears little by little
        * Class 3 (top-right disk) generated
    - Phase 3 (55K to 85K, t3=85):
        * Class 2 completely disappeared
        * Class 3 disappears slowly
        * Class 4 (bottom-right figure-8) generated and dense
        * Outliers removed by sporadic grid cleanup.
    """
    if n_samples < n_noise:
        n_noise = int(n_samples * (10 / 85))
    rng = np.random.default_rng(seed)

    t1_end = int(n_samples * (25 / 85))  # 25K
    t2_end = int(n_samples * (55 / 85))  # 55K

    # Noise indices distributed uniformly across the stream
    noise_indices = set(rng.choice(n_samples, size=n_noise, replace=False))

    def _sample_class(cid: int) -> np.ndarray:
        if cid == 1:
            # Top-left arch
            th = rng.uniform(np.pi * 0.15, np.pi * 0.85)
            r = 0.18 + rng.normal(0, 0.015)
            return np.array([0.28 + r * np.cos(th), 0.65 + r * np.sin(th)])
        elif cid == 2:
            # Bottom-left ring
            th = rng.uniform(0, 2 * np.pi)
            r = 0.14 + rng.normal(0, 0.015)
            return np.array([0.25 + r * np.cos(th), 0.28 + r * np.sin(th)])
        elif cid == 3:
            # Top-right circle/disk
            th = rng.uniform(0, 2 * np.pi)
            r = np.sqrt(rng.uniform(0, 0.13**2))
            return np.array([0.75 + r * np.cos(th), 0.72 + r * np.sin(th)])
        else:
            # Bottom-right double ring
            if rng.random() < 0.5:
                th = rng.uniform(0, 2 * np.pi)
                r = 0.12 + rng.normal(0, 0.012)
                return np.array([0.62 + r * np.cos(th), 0.28 + r * np.sin(th)])
            else:
                th = rng.uniform(0, 2 * np.pi)
                r = 0.12 + rng.normal(0, 0.012)
                return np.array([0.83 + r * np.cos(th), 0.28 + r * np.sin(th)])

    def factory() -> Iterator[tuple[dict[str, float], int | None]]:
        for t in range(n_samples):
            if t in noise_indices:
                pt = rng.uniform(0.0, 1.0, 2)
                yield {"x0": float(pt[0]), "x1": float(pt[1])}, None
                continue

            if t <= t1_end:
                # Phase 1: Class 1 generated
                pt = _sample_class(1)
                lbl = 1
            elif t <= t2_end:
                # Phase 2: Class 2 generated first half, Class 3 generated second half
                progress = (t - t1_end) / (t2_end - t1_end)
                if progress < 0.5:
                    pt = _sample_class(2)
                    lbl = 2
                else:
                    pt = _sample_class(3)
                    lbl = 3
            else:
                # Phase 3: Class 4 generated
                pt = _sample_class(4)
                lbl = 4

            pt = np.clip(pt, 0.0, 1.0)
            yield {"x0": float(pt[0]), "x1": float(pt[1])}, lbl

    return IterableDataset(
        factory,
        name="dstream_85k_dynamic",
        n_features=2,
        has_labels=True,
    )


def make_scalability_dataset(
    n_samples: int = 30_000,
    n_features: int = 2,
    n_clusters: int = 5,
    seed: int = 42,
) -> IterableDataset:
    """Generate high-dimensional multi-size stream for scalability benchmarks.

    n_samples in [5K, 85K]
    n_features in [2, 40]
    """
    rng = np.random.default_rng(seed)
    # Centers distributed within [0.2, 0.8]^d
    centers = rng.uniform(0.2, 0.8, (n_clusters, n_features))
    std = 0.05

    def factory() -> Iterator[tuple[dict[str, float], int]]:
        for _ in range(n_samples):
            cluster_id = rng.integers(0, n_clusters)
            center = centers[cluster_id]
            point = center + rng.normal(0, std, n_features)
            point = np.clip(point, 0.0, 1.0)
            x = {f"f{i}": float(val) for i, val in enumerate(point)}
            yield x, int(cluster_id)

    return IterableDataset(
        factory,
        name=f"scalability_{n_samples}_{n_features}d",
        n_features=n_features,
        has_labels=True,
    )
