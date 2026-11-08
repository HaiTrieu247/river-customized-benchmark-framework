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

**E8: class identities, stationary controls, a factorial design.** Every name
below is a runnable placeholder for the study's synthetic-design check (E8),
not a recommended set; change the levels freely.

* *Class identities.* Maximin placement draws centres one at a time from the
  seed, so layouts of one seed with 3, 4 and 10 clusters share their first
  centres, and the generator's label is the centre's index: in ``k_drift`` the
  4 centres of segment 1 are centres 0-3 of segment 2's 10, and 3 of them
  continue into segment 3, yet the label offsets (+100, +200) give a persisting
  component a new class id. ``k_drift_same_ids`` and ``mixed_drift_same_ids``
  are the same points with offset 0, so a persisting component keeps its id (in
  ``mixed_drift`` the persisting centres are multiplied by 4). Reading a result
  on both versions says how much of it hangs on that labelling.
* *Stationary controls.* ``stationary_k4`` and ``stationary_k10`` never change:
  the layouts of the base scenarios' 4- and 10-cluster segments (a prefix of
  ``stationary_k4`` is segment 1 of ``k_drift``, point for point), for
  measuring switching and degradation where no change happened.
* *Factorial family* (:func:`factorial_scenarios`), named
  ``fd_<type>_<mechanism>_<difficulty>_<timing>``. Every stream goes from
  concept A (:data:`FACTORIAL_BASE` under the difficulty) to concept B (A under
  the mechanism), at the timing's share of the stream:

  - *type*: ``abrupt`` (A → B), ``gradual`` (A → B along MOA's sigmoid, width
    :data:`FACTORIAL_WIDTH`), ``recurring`` (A → B → A, B lasting
    :data:`FACTORIAL_RECURRENCE_SPAN`);
  - *mechanism* (:data:`FACTORIAL_MECHANISMS`): ``location`` (every centre
    moves: B is an independent layout of the same number of clusters and
    radius, drawn from the data seed and a layout index; its components are
    new, so their labels are offset by 100), ``scale`` (every feature × 4),
    ``density`` (cluster radius × 4, same centres), ``count`` (4 → 10 clusters;
    the 4 persisting components keep their ids);
  - *difficulty* (:data:`FACTORIAL_DIFFICULTIES`), applied to both concepts:
    ``overlap`` (cluster radius × 3), ``imbalance`` (the generator's
    ``weight_range`` 0.8), ``noise`` (the generator's uniform background noise,
    10 % of points, labelled ``None``), ``irrelevant`` (5 extra features,
    uniform on [0, 1), appended to ``x``, see :class:`ParamDriftStream`);
  - *timing* (:data:`FACTORIAL_TIMINGS`): the change at 25 % (``early``) or
    80 % (``late``) of the stream, the latter meant to fall after a tuner's
    exploration would have narrowed (a placeholder share, not derived from
    the tuner's schedule).

  The default levels give 3 × 4 × 4 × 2 = 96 streams, plus one stationary
  control per difficulty (``fd_stationary_<difficulty>``: concept A
  throughout). ``fraction="half"`` gives a 48-stream half fraction in which
  every pair of levels of any two factors appears equally often. These
  streams run in *lockstep* (:class:`ParamDriftStream`), so each one equals its
  stationary control at every sample drawn from concept A, and a ``scale``
  stream's concept-B samples are its control's × 4.

Drift points depend on the shares and ``n_samples`` only, never on the seed, so
every data seed (an independent realisation of the generators) has the same
change points.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations, islice, product

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
    #: The generator's imbalance knob: cluster weights drawn from ``1 +/- weight_range``.
    weight_range: float = 0.0
    #: The generator's background noise: this share of points uniform on the cube, labelled None.
    noise: float = 0.0
    #: Which layout the segment's generator draws: 0 is the stream's seed itself (every scenario
    #: before E8); ``j > 0`` an independent layout seeded from ``(seed, j)``.
    layout: int = 0


class ParamDriftStream(Dataset):
    """Concatenation of :class:`Segment` s with known drift points.

    Re-iterable and deterministic: each ``__iter__`` call rebuilds a fresh
    ``SyntheticStream`` per segment from ``seed``, so repeated iteration
    yields identical output.

    ``width`` (a share of ``n_samples``; 0 = abrupt) makes every change gradual,
    centred on its drift point (module docstring); the draw of which segment a
    sample comes from uses its own generator, seeded from ``seed``, so the
    segments' points are those of the abrupt stream, in the same order.

    ``lockstep`` (default ``False``, every scenario before E8): segments with the same
    generator settings (number of clusters, radius, weight range, noise, layout) share one
    generator, and every generator advances one point per sample of the stream, so sample
    ``t`` is its segment's generator's point ``t``. A stream then equals its stationary
    control (one segment, concept A) at every sample drawn from concept A, before and after
    a change. Without it each segment gets a fresh generator, which replays its layout's
    sequence from the start.

    ``n_irrelevant`` features, named after the informative ones (``x<n_features>`` on; listed
    in ``irrelevant_features``), are appended to every sample: uniform on [0, 1), the cube the
    cluster centres are drawn in, from a generator of their own seeded from ``seed`` (so the
    informative features are those of the same stream without them), then multiplied by the
    segment's ``scale`` like every other feature, so a scale change rescales the whole vector.
    ``n_features`` counts both kinds.
    """

    def __init__(self, name: str, segments: tuple[Segment, ...], n_samples: int, n_features: int, seed: int,
                 width: float = 0.0, lockstep: bool = False, n_irrelevant: int = 0):
        self.name, self.segments = name, segments
        if n_irrelevant < 0:
            raise ValueError(f"n_irrelevant must be >= 0, got {n_irrelevant}")
        self._d = n_features
        self.irrelevant_features = tuple(f"x{n_features + i}" for i in range(n_irrelevant))
        self.n_features, self.has_labels = n_features + n_irrelevant, True
        self._n, self._seed = n_samples, seed
        if width < 0:
            raise ValueError(f"width must be >= 0, got {width}")
        self.width = width
        self.lockstep = lockstep
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

    def _layout_seed(self, layout: int) -> int:
        """The generator seed of a layout: the stream's seed for 0, else one drawn from
        ``(seed, layout)``, so no layout of one data seed is a layout of another."""
        if layout == 0:
            return self._seed
        return int(np.random.SeedSequence([self._seed, 3, layout]).generate_state(1)[0])

    @staticmethod
    def _generator_key(seg: Segment) -> tuple:
        return seg.n_clusters, seg.cluster_radius, seg.weight_range, seg.noise, seg.layout

    def _segment_stream(self, seg: Segment, n: int):
        cfg = StreamConfig(
            n_samples=n,
            n_features=self._d,
            n_clusters=seg.n_clusters,
            cluster_radius=seg.cluster_radius,
            separation=None,
            weight_range=seg.weight_range,
            noise=seg.noise,
            seed=self._layout_seed(seg.layout),
        )
        return SyntheticStream(cfg)

    @staticmethod
    def _emit(seg: Segment, x, y):
        if seg.scale != 1.0:
            x = {k: v * seg.scale for k, v in x.items()}
        return x, (None if y is None else y + seg.label_offset)

    def active_segments(self):
        """The index of the segment each sample comes from, in order (``n_samples`` of them)."""
        if self.width > 0:
            yield from self._gradual_choices()
            return
        for j, L in enumerate(self._lengths):
            for _ in range(L):
                yield j

    def __iter__(self):
        if self.lockstep:
            rows = self._lockstep_rows()
        elif self.width > 0:
            rows = self._gradual_rows()
        else:
            rows = self._abrupt_rows()
        extra = self._irrelevant_rows()
        for seg, x, y in rows:
            if extra is not None:
                x = {**x, **next(extra)}
            yield self._emit(seg, x, y)

    def _abrupt_rows(self):
        for seg, L in zip(self.segments, self._lengths):
            for x, y in islice(self._segment_stream(seg, L), L):
                yield seg, x, y

    def _gradual_choices(self):
        """MOA's nested ``ConceptDriftStream``: past drift point k with probability
        sigmoid((t - p_k) / w), then on to the next one."""
        w = self.width * self._n
        rng = np.random.default_rng([self._seed, 1])
        for t in range(self._n):
            j = 0
            for p in self.drift_points:
                z = -4.0 * (t - p) / w
                if z > 700 or rng.random() >= 1.0 / (1.0 + math.exp(z)):
                    break
                j += 1
            yield j

    def _gradual_rows(self):
        """Each segment's own stream is read in order, as far as it is drawn from."""
        streams = [iter(self._segment_stream(seg, self._n)) for seg in self.segments]
        for j in self._gradual_choices():
            x, y = next(streams[j])
            yield self.segments[j], x, y

    def _lockstep_rows(self):
        keys = [self._generator_key(seg) for seg in self.segments]
        generators: dict[tuple, object] = {}
        for seg, key in zip(self.segments, keys):
            if key not in generators:
                generators[key] = iter(self._segment_stream(seg, self._n))
        for j in self.active_segments():
            drawn = {key: next(g) for key, g in generators.items()}   # every generator advances
            x, y = drawn[keys[j]]
            yield self.segments[j], x, y

    def _irrelevant_rows(self):
        if not self.irrelevant_features:
            return None
        names = self.irrelevant_features

        def rows():
            rng = np.random.default_rng([self._seed, 2])
            while True:
                yield dict(zip(names, map(float, rng.uniform(0.0, 1.0, size=len(names)))))

        return rows()


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


# --- E8 (module docstring) --------------------------------------------------------------------

#: ``k_drift`` and ``mixed_drift`` with every label offset 0: the same points, and a component
#: that persists through a change (same centre, up to the segment's scale) keeps its class id.
SAME_ID_SCENARIOS: dict[str, dict] = {
    f"{base}_same_ids": dict(segments=tuple(
        replace(s, label_offset=0) for s in PARAM_DRIFT_SCENARIOS[base]["segments"]))
    for base in ("k_drift", "mixed_drift")
}

#: No change at all, on the base scenarios' 4- and 10-cluster layouts (radius 0.05, scale 1).
STATIONARY_SCENARIOS: dict[str, dict] = {
    "stationary_k4": dict(segments=(Segment(1.0, n_clusters=4),)),
    "stationary_k10": dict(segments=(Segment(1.0, n_clusters=10),)),
}

#: The factorial's levels. Every value is a runnable placeholder, not a recommendation.
#: Concept A before any difficulty: the base scenarios' first segment.
FACTORIAL_BASE: dict = dict(n_clusters=4, cluster_radius=0.05)
FACTORIAL_TYPES: tuple[str, ...] = ("abrupt", "gradual", "recurring")
#: Concept B as a change of concept A: ``radius_factor`` multiplies A's radius, ``layout``
#: picks an independent layout (see ``Segment.layout``), other keys are ``Segment`` fields.
FACTORIAL_MECHANISMS: dict[str, dict] = {
    "location": dict(layout=1, label_offset=100),
    "scale": dict(scale=4.0),
    "density": dict(radius_factor=4.0),
    "count": dict(n_clusters=10),
}
#: Applied to both concepts: ``radius_factor`` multiplies the base radius, ``n_irrelevant`` is
#: the number of uniform features appended (``ParamDriftStream``), other keys are ``Segment``
#: fields. At the base radius the closest pair of centres sits about 10.8 radii apart with 4
#: clusters and 8.1 with 10 (5 features, seeds 1-30); ``overlap``'s 0.15 brings that to about
#: 3.6 and 2.7, and ``density`` under ``overlap`` to about 0.9.
FACTORIAL_DIFFICULTIES: dict[str, dict] = {
    "overlap": dict(radius_factor=3.0),
    "imbalance": dict(weight_range=0.8),
    "noise": dict(noise=0.1),
    "irrelevant": dict(n_irrelevant=5),
}
#: Share of the stream at which the (first) change happens.
FACTORIAL_TIMINGS: dict[str, float] = {"early": 0.25, "late": 0.8}
#: A gradual change's width, as a share of the stream (as ``scale_drift_gradual``'s).
FACTORIAL_WIDTH = 0.1
#: How long concept B lasts in a recurring stream, as a share of the stream.
FACTORIAL_RECURRENCE_SPAN = 0.1


def _concept_a(difficulty: dict) -> tuple[Segment, int]:
    change = dict(difficulty)
    n_irrelevant = change.pop("n_irrelevant", 0)
    radius = FACTORIAL_BASE["cluster_radius"] * change.pop("radius_factor", 1.0)
    return Segment(1.0, **{**FACTORIAL_BASE, "cluster_radius": radius, **change}), n_irrelevant


def _concept_b(a: Segment, mechanism: dict) -> Segment:
    change = dict(mechanism)
    if "radius_factor" in change:
        change["cluster_radius"] = a.cluster_radius * change.pop("radius_factor")
    return replace(a, **change)


def _balanced_pairs(cells: list[tuple[int, ...]], sizes: list[int]) -> bool:
    """Whether every pair of levels of any two factors (``sizes[i]`` levels for factor ``i``)
    appears, and equally often, among ``cells``."""
    for i, j in combinations(range(len(sizes)), 2):
        counts = Counter((c[i], c[j]) for c in cells)
        if len(counts) != sizes[i] * sizes[j] or len(set(counts.values())) != 1:
            return False
    return True


def factorial_scenarios(types=None, mechanisms=None, difficulties=None, timings=None, *,
                        fraction: str = "full", controls: bool = True,
                        width: float = FACTORIAL_WIDTH,
                        recurrence_span: float = FACTORIAL_RECURRENCE_SPAN) -> dict[str, dict]:
    """The factorial family over the levels given (default: all of each factor's levels), as a
    name → scenario mapping in factor order: ``fd_<type>_<mechanism>_<difficulty>_<timing>``,
    then ``fd_stationary_<difficulty>`` for each difficulty when ``controls``.

    ``fraction="full"`` is the cross product (96 streams at the default levels); ``"half"``
    keeps the cells whose level indices (in the order given) sum to an even number at the
    first timing and an odd one at the second, two timings required: every pair of levels of
    any two factors then appears equally often (48 streams at the default levels), and a
    selection for which that fails raises. Levels are names in :data:`FACTORIAL_TYPES`,
    :data:`FACTORIAL_MECHANISMS`, :data:`FACTORIAL_DIFFICULTIES` and
    :data:`FACTORIAL_TIMINGS`. A mapping built from levels other than the registered ones is
    loaded by ``PARAM_DRIFT_SCENARIOS.update(...)`` and ``make_param_drift``.
    """
    levels = {
        "type": tuple(FACTORIAL_TYPES if types is None else types),
        "mechanism": tuple(FACTORIAL_MECHANISMS if mechanisms is None else mechanisms),
        "difficulty": tuple(FACTORIAL_DIFFICULTIES if difficulties is None else difficulties),
        "timing": tuple(FACTORIAL_TIMINGS if timings is None else timings),
    }
    known = {"type": FACTORIAL_TYPES, "mechanism": FACTORIAL_MECHANISMS,
             "difficulty": FACTORIAL_DIFFICULTIES, "timing": FACTORIAL_TIMINGS}
    for factor, chosen in levels.items():
        unknown = [v for v in chosen if v not in known[factor]]
        if unknown or not chosen or len(set(chosen)) != len(chosen):
            raise ValueError(f"{factor} levels must be distinct names from {list(known[factor])}, got {list(chosen)}")
    if fraction not in ("full", "half"):
        raise ValueError(f"fraction must be 'full' or 'half', got {fraction!r}")

    sizes = [len(v) for v in levels.values()]
    cells = list(product(*(range(n) for n in sizes)))
    if fraction == "half":
        if sizes[3] != 2:
            raise ValueError(f"a half fraction needs exactly two timings, got {list(levels['timing'])}")
        cells = [c for c in cells if (c[0] + c[1] + c[2]) % 2 == c[3]]
        if not _balanced_pairs(cells, sizes):
            raise ValueError(f"the half fraction of {sizes} levels is not balanced in every pair of factors")

    out: dict[str, dict] = {}
    for it, im, idf, iti in cells:
        kind, mech = levels["type"][it], levels["mechanism"][im]
        diff, timing = levels["difficulty"][idf], levels["timing"][iti]
        a, n_irrelevant = _concept_a(FACTORIAL_DIFFICULTIES[diff])
        b = _concept_b(a, FACTORIAL_MECHANISMS[mech])
        p = FACTORIAL_TIMINGS[timing]
        if kind == "recurring":
            if not 0 < p < p + recurrence_span < 1:
                raise ValueError(f"a recurring change at {p} lasting {recurrence_span} does not fit in the stream")
            segments = (replace(a, frac=p), replace(b, frac=recurrence_span), replace(a, frac=1 - p - recurrence_span))
        else:
            if not 0 < p < 1:
                raise ValueError(f"timing {timing!r} at {p} is not inside the stream")
            segments = (replace(a, frac=p), replace(b, frac=1 - p))
        out[f"fd_{kind}_{mech}_{diff}_{timing}"] = dict(
            segments=segments, width=width if kind == "gradual" else 0.0, lockstep=True, n_irrelevant=n_irrelevant)
    if controls:
        for diff in levels["difficulty"]:
            a, n_irrelevant = _concept_a(FACTORIAL_DIFFICULTIES[diff])
            out[f"fd_stationary_{diff}"] = dict(segments=(a,), lockstep=True, n_irrelevant=n_irrelevant)
    return out


#: The full factorial at the default levels and its controls (96 + 4 streams).
FACTORIAL_SCENARIOS: dict[str, dict] = factorial_scenarios()

E8_SAME_ID_SCENARIOS: tuple[str, ...] = tuple(SAME_ID_SCENARIOS)
E8_STATIONARY_SCENARIOS: tuple[str, ...] = tuple(STATIONARY_SCENARIOS) + tuple(
    n for n in FACTORIAL_SCENARIOS if n.startswith("fd_stationary_"))
E8_FACTORIAL_SCENARIOS: tuple[str, ...] = tuple(n for n in FACTORIAL_SCENARIOS if not n.startswith("fd_stationary_"))
#: Every E8 name (2 + 6 + 96 = 104), each loadable by ``make_param_drift``.
E8_SCENARIOS: tuple[str, ...] = E8_SAME_ID_SCENARIOS + E8_STATIONARY_SCENARIOS + E8_FACTORIAL_SCENARIOS

PARAM_DRIFT_SCENARIOS.update(SAME_ID_SCENARIOS)
PARAM_DRIFT_SCENARIOS.update(STATIONARY_SCENARIOS)
PARAM_DRIFT_SCENARIOS.update(FACTORIAL_SCENARIOS)


def make_param_drift(name: str, n_samples: int = 120_000, n_features: int = 5, seed: int = 42) -> ParamDriftStream:
    """Build one of the named :data:`PARAM_DRIFT_SCENARIOS`.

    ``n_features`` is the number of informative features (a scenario with irrelevant ones
    appends them); ``seed`` is the data seed, one independent realisation of the generators.
    """
    if name not in PARAM_DRIFT_SCENARIOS:
        raise KeyError(f"unknown param-drift scenario {name!r}; available: {sorted(PARAM_DRIFT_SCENARIOS)}")
    spec = PARAM_DRIFT_SCENARIOS[name]
    return ParamDriftStream(name, spec["segments"], n_samples, n_features, seed, width=spec.get("width", 0.0),
                            lockstep=spec.get("lockstep", False), n_irrelevant=spec.get("n_irrelevant", 0))
