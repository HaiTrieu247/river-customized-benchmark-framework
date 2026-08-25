"""Synthetic static-cluster stream generator.

Generates a labeled stream of Gaussian blobs with a fixed, configurable
structure (no drift). Useful as a controlled dataset where ground-truth labels
are known, so clustering quality can be measured exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import Dataset


@dataclass
class StreamConfig:
    """Configuration for a synthetic blob stream."""

    n_samples: int = 20_000
    n_features: int = 2
    n_clusters: int = 5
    cluster_std: float = 1.0
    center_box: float = 20.0  # centers drawn uniformly in [-box, box]^d
    noise: float = 0.0  # fraction of uniform background-noise samples
    seed: int = 42


class SyntheticStream(Dataset):
    """Labeled stream of static Gaussian clusters.

    Yields ``(x, y)`` where ``x`` is a feature dict and ``y`` is the cluster id
    (``None`` for background-noise samples). Re-iterable and deterministic given
    the seed, so every algorithm sees the identical sequence.
    """

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()
        self.n_features = self.config.n_features
        self.has_labels = True
        self.name = (
            f"synthetic_k{self.config.n_clusters}"
            f"_d{self.config.n_features}_n{self.config.n_samples}"
        )
        self._feature_names = [f"x{i}" for i in range(self.config.n_features)]
        # fixed centers + weights (independent of the sampling seed so the layout
        # is stable, while per-run sampling uses the run seed)
        rng = np.random.default_rng(self.config.seed)
        box = self.config.center_box
        self._centers = rng.uniform(-box, box, size=(self.config.n_clusters, self.config.n_features))
        self._weights = np.full(self.config.n_clusters, 1.0 / self.config.n_clusters)

    def __len__(self) -> int:
        return self.config.n_samples

    def __iter__(self):
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        lo, hi = -cfg.center_box, cfg.center_box
        for _ in range(cfg.n_samples):
            if cfg.noise > 0 and rng.random() < cfg.noise:
                vals = rng.uniform(lo, hi, size=cfg.n_features)
                yield {n: float(v) for n, v in zip(self._feature_names, vals)}, None
                continue
            k = int(rng.choice(cfg.n_clusters, p=self._weights))
            point = self._centers[k] + rng.normal(0.0, cfg.cluster_std, size=cfg.n_features)
            x = {n: float(v) for n, v in zip(self._feature_names, point)}
            yield x, k
