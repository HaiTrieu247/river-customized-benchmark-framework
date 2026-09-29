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
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

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
    """

    def __init__(self, name: str, segments: tuple[Segment, ...], n_samples: int, n_features: int, seed: int):
        self.name, self.segments = name, segments
        self.n_features, self.has_labels = n_features, True
        self._n, self._seed = n_samples, seed
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

    def __iter__(self):
        for seg, L in zip(self.segments, self._lengths):
            cfg = StreamConfig(
                n_samples=L,
                n_features=self.n_features,
                n_clusters=seg.n_clusters,
                cluster_radius=seg.cluster_radius,
                separation=None,
                seed=self._seed,
            )
            for x, y in islice(SyntheticStream(cfg), L):
                if seg.scale != 1.0:
                    x = {k: v * seg.scale for k, v in x.items()}
                yield x, (None if y is None else y + seg.label_offset)


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


def make_param_drift(name: str, n_samples: int = 120_000, n_features: int = 5, seed: int = 42) -> ParamDriftStream:
    """Build one of the named :data:`PARAM_DRIFT_SCENARIOS`."""
    if name not in PARAM_DRIFT_SCENARIOS:
        raise KeyError(f"unknown param-drift scenario {name!r}; available: {sorted(PARAM_DRIFT_SCENARIOS)}")
    return ParamDriftStream(name, PARAM_DRIFT_SCENARIOS[name]["segments"], n_samples, n_features, seed)
