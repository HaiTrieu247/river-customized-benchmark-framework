"""Streams whose *optimal hyperparameters* change at known points.

Each stream is a concatenation of segments; every segment is a SyntheticStream
sharing the layout seed, plus a feature scale factor and a label offset. The
boundaries are recorded in ``drift_points`` so the report can measure how long
a tuner takes to recover. Density segments share their centres because the
generator draws all cluster centres before any per-cluster radius or weight
(``_initial_clusters`` places every centre first, then draws radius/weight/
velocity), so two segments with the same ``seed``, ``n_clusters`` and
``n_features`` get identical centres regardless of ``cluster_radius`` —
verified directly against ``SyntheticStream._initial_clusters`` before relying
on it here.

``scale_drift``'s segments all share the same ``seed``, ``n_clusters`` and
``cluster_radius``, so the generator replays the identical point sequence in
every segment and only the applied ``scale`` differs — segment 2 is exactly
``4.0 ×`` segment 1 and segment 3 is exactly ``0.25 ×`` segment 1, point for
point. No clipping is applied anywhere in this module: the generator's own
output is used as-is, unbounded.

The four scenarios above change the best parameters abruptly, at round shares
of the stream, by a factor of 4. Four more vary that design, so a result on
them can be told apart from one that only holds for it: a smaller and a larger
scale jump (1.5 and 8), changes at points drawn at random (off the window grid,
of random kind), and a gradual change. A gradual change follows MOA's
``ConceptDriftStream``: around a drift point ``p`` each sample comes from the
next segment with probability ``1 / (1 + exp(-4 (t - p) / w))``, ``w`` the
width, and several drift points nest the way MOA nests the streams. These
variants are runnable placeholders for the study's drift-robustness check
(X7), not a recommended set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import islice

import numpy as np

from ..base import Dataset
from .generator import StreamConfig, SyntheticStream


@dataclass(frozen=True)
class Segment:
    frac: float                 # share of n_samples
    n_clusters: int = 4
    cluster_radius: float = 0.05
    scale: float = 1.0
    label_offset: int = 0


class ParamDriftStream(Dataset):
    """Concatenation of :class:`Segment` s with known drift points.

    Re-iterable and deterministic: each ``__iter__`` call rebuilds a fresh
    ``SyntheticStream`` per segment from ``seed``, so repeated iteration
    yields identical output.

    ``width`` (a share of ``n_samples``; 0 = abrupt) makes every change gradual,
    centred on its drift point (module docstring); the draw of which segment a
    sample comes from uses its own generator, seeded from ``seed``, so the
    segments' points are those of the abrupt stream, in the same order.
    """

    def __init__(self, name: str, segments: tuple[Segment, ...], n_samples: int, n_features: int, seed: int,
                 width: float = 0.0):
        self.name, self.segments = name, segments
        self.n_features, self.has_labels = n_features, True
        self._n, self._seed = n_samples, seed
        if width < 0:
            raise ValueError(f"width must be >= 0, got {width}")
        self.width = width
        lengths = [int(round(s.frac * n_samples)) for s in segments]
        lengths[-1] = n_samples - sum(lengths[:-1])
        if any(L < 1 for L in lengths):
            raise ValueError(
                f"n_samples={n_samples} is too small for {len(segments)} segments "
                f"(scenario {name!r}); every segment needs at least 1 sample"
            )
        self._lengths = lengths
        cut, pts = 0, []
        for L in lengths[:-1]:
            cut += L
            pts.append(cut)
        self.drift_points = tuple(pts)

    def __len__(self):
        return self._n

    def _segment_stream(self, seg: Segment, n: int):
        cfg = StreamConfig(
            n_samples=n,
            n_features=self.n_features,
            n_clusters=seg.n_clusters,
            cluster_radius=seg.cluster_radius,
            separation=None,
            seed=self._seed,
        )
        return SyntheticStream(cfg)

    @staticmethod
    def _emit(seg: Segment, x, y):
        if seg.scale != 1.0:
            x = {k: v * seg.scale for k, v in x.items()}
        return x, (None if y is None else y + seg.label_offset)

    def __iter__(self):
        if self.width > 0:
            yield from self._iter_gradual()
            return
        for seg, L in zip(self.segments, self._lengths):
            for x, y in islice(self._segment_stream(seg, L), L):
                yield self._emit(seg, x, y)

    def _iter_gradual(self):
        """MOA's nested ``ConceptDriftStream``: past drift point k with probability
        sigmoid((t - p_k) / w), then on to the next one; each segment's own stream is read
        in order, as far as it is drawn from."""
        w = self.width * self._n
        streams = [iter(self._segment_stream(seg, self._n)) for seg in self.segments]
        rng = np.random.default_rng([self._seed, 1])
        for t in range(self._n):
            j = 0
            for p in self.drift_points:
                z = -4.0 * (t - p) / w
                if z > 700 or rng.random() >= 1.0 / (1.0 + math.exp(z)):
                    break
                j += 1
            x, y = next(streams[j])
            yield self._emit(self.segments[j], x, y)


#: Named scenarios: feature-scale jump, cluster-radius jump, cluster-count
#: jump, and one combining a count jump with a scale jump.
#:
#: ``density_drift``'s middle radius (0.20, a 4x jump from the 0.05
#: baseline) changes the mean per-cluster radial spread (each point's
#: distance to its own cluster's centroid) by exactly the same 4x factor —
#: verified empirically for this seed, well clear of the 2.5x threshold the
#: test checks for. Points are never clipped in this module: the generator's
#: raw, unbounded output is used as-is.
PARAM_DRIFT_SCENARIOS: dict[str, dict] = {
    "scale_drift": dict(segments=(
        Segment(1 / 3),
        Segment(1 / 3, scale=4.0),
        Segment(1 / 3, scale=0.25),
    )),
    "density_drift": dict(segments=(
        Segment(1 / 3, cluster_radius=0.05),
        Segment(1 / 3, cluster_radius=0.20),
        Segment(1 / 3, cluster_radius=0.05),
    )),
    "k_drift": dict(segments=(
        Segment(1 / 3, n_clusters=4),
        Segment(1 / 3, n_clusters=10, label_offset=100),
        Segment(1 / 3, n_clusters=3, label_offset=200),
    )),
    "mixed_drift": dict(segments=(
        Segment(0.5, n_clusters=4),
        Segment(0.5, n_clusters=10, scale=4.0, label_offset=100),
    )),
}


def random_segments(seed: int, n_segments: int, min_frac: float,
                    n_clusters=(3, 4, 10), cluster_radius=(0.05, 0.20), scale=(0.25, 1.0, 4.0)) -> tuple[Segment, ...]:
    """``n_segments`` segments of random shares (none below ``min_frac``) and random kind.

    Each segment draws its number of clusters, radius and scale from the values the
    abrupt scenarios use, never repeating the previous segment's; the shares are a
    uniform split conditioned on the minimum, so the drift points fall off the window
    grid. Segments with the same number of clusters share their centres (same seed),
    so they share their labels too: the label offset is ``100 * n_clusters``.
    """
    rng = np.random.default_rng(seed)
    if n_segments * min_frac >= 1:
        raise ValueError(f"{n_segments} segments of at least {min_frac} do not fit in 1")
    fracs = min_frac + (1 - n_segments * min_frac) * rng.dirichlet(np.ones(n_segments))
    kinds = [(k, r, sc) for k in n_clusters for r in cluster_radius for sc in scale]
    out, prev = [], None
    for f in fracs:
        choices = [c for c in kinds if c != prev]
        prev = choices[int(rng.integers(len(choices)))]
        k, r, sc = prev
        out.append(Segment(float(f), n_clusters=k, cluster_radius=r, scale=sc, label_offset=100 * k))
    return tuple(out)


#: Variants of the design above (module docstring), for checking that a result on the
#: four abrupt scenarios is not specific to them. Placeholders: change them freely.
#: ``width`` is the share of the stream a gradual change spreads over.
DRIFT_VARIANT_SCENARIOS: dict[str, dict] = {
    "scale_drift_x1_5": dict(segments=(
        Segment(1 / 3),
        Segment(1 / 3, scale=1.5),
        Segment(1 / 3, scale=1 / 1.5),
    )),
    "scale_drift_x8": dict(segments=(
        Segment(1 / 3),
        Segment(1 / 3, scale=8.0),
        Segment(1 / 3, scale=1 / 8.0),
    )),
    "random_drift": dict(segments=random_segments(seed=7, n_segments=5, min_frac=0.08)),
    "scale_drift_gradual": dict(segments=PARAM_DRIFT_SCENARIOS["scale_drift"]["segments"], width=0.1),
}
PARAM_DRIFT_SCENARIOS.update(DRIFT_VARIANT_SCENARIOS)


def make_param_drift(name: str, n_samples: int = 120_000, n_features: int = 5, seed: int = 42) -> ParamDriftStream:
    """Build one of the named :data:`PARAM_DRIFT_SCENARIOS`."""
    if name not in PARAM_DRIFT_SCENARIOS:
        raise KeyError(f"unknown param-drift scenario {name!r}; available: {sorted(PARAM_DRIFT_SCENARIOS)}")
    spec = PARAM_DRIFT_SCENARIOS[name]
    return ParamDriftStream(name, spec["segments"], n_samples, n_features, seed, width=spec.get("width", 0.0))
