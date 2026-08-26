"""Synthetic stream generator for clustering benchmarks.

The design follows what the stream-clustering literature actually evaluates on.
The de-facto standard is MOA's ``RandomRBFGeneratorEvents`` (Kranen et al.,
2010), used to evaluate DenStream, DBSTREAM, evoStream and most later work; the
generator here reproduces its structure and adds two knobs the benchmark needs.

**What is modelled**

* *Geometry* — clusters live in the unit hypercube :math:`[0,1]^d` and a point
  is drawn as ``centre + magnitude * direction``, with ``direction`` uniform on
  the unit sphere and ``magnitude ~ N(0, radius)``. This is the MOA/River radial
  form rather than an isotropic Gaussian, and the difference matters: it keeps a
  cluster's physical size **independent of** :math:`d`, so a "quality vs
  dimensionality" sweep varies dimensionality alone instead of silently
  inflating every cluster by :math:`\\sqrt{d}`.
* *Separation* — centres are placed under a minimum-distance constraint
  expressed in cluster radii. Difficulty becomes an explicit, comparable
  parameter instead of an accident of the bounding box: drawing centres
  uniformly (the naive approach) makes clusters trivially separable as
  :math:`d` grows, which flatters every algorithm.
* *Imbalance* — per-cluster weights and radii can vary, because equal-sized
  equal-density clusters are the easy case and rarely the realistic one.
* *Drift* — centres move a fixed distance every ``drift_every`` points and
  bounce off the unit-cube walls (the MOA/River convention).
* *Events* — clusters merge, split, appear and disappear at ``event_frequency``.
  These are what separate a stream clusterer from a batch one, so a benchmark
  that omits them cannot distinguish the two.
* *Noise* — a fraction of points drawn uniformly, labelled ``None`` so the
  scoring layer drops them rather than treating noise as a class.

**Ground truth.** ``y`` is the id of the *generating cluster*, not a class label
mapped onto several clusters. For a clustering benchmark that is the correct
target: recovering the generating mixture component is exactly the task.

**Ordering.** Points are drawn i.i.d. from the mixture, so consecutive samples
come from different clusters and every evaluation window sees several classes.
This is deliberate. Real streams such as KDDCup99 and Forest CoverType arrive in
long same-class blocks, which makes per-window external metrics degenerate (see
``benchmark.evaluation.runner.WindowQuality.is_degenerate``). Set
``block_ordered=True`` to reproduce that pathology on purpose as a control.

References
----------
Kranen, Kremer, Jansen, Seidl, Bifet, Holmes, Pfahringer (2010). *Clustering
Performance on Evolving Data Streams: Assessing Algorithms and Evaluation
Measures within MOA.*
Carnein & Trautmann (2019). *Optimizing Data Stream Representation: An
Extensive Survey on Stream Clustering Algorithms.*
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..base import Dataset

#: Events the generator can inject at ``event_frequency`` intervals.
EVENTS = ("merge", "split", "create", "delete")


@dataclass
class StreamConfig:
    """Configuration for a synthetic clustering stream.

    Attributes
    ----------
    n_samples, n_features, n_clusters:
        Stream length, dimensionality, and initial number of clusters.
    cluster_radius:
        Standard deviation of the radial offset, in unit-cube units. MOA's
        default is ``0.07``. Used directly only when ``separation is None``.
    radius_range:
        Per-cluster radius is drawn from
        ``cluster_radius * (1 +/- radius_range)``. ``0`` => every cluster the
        same size.
    separation:
        Distance between the closest pair of centres, measured in cluster
        radii, and the main difficulty knob. ``1`` means the two closest
        clusters sit one standard deviation apart and overlap heavily; ``3``
        separates them clearly; ``6`` makes the problem trivial.

        When set, the radius is *derived* from the drawn layout so the
        constraint holds exactly: ``radius = d_min / (2 * separation)``, where
        ``d_min`` is the smallest pairwise centre distance. Deriving it rather
        than treating it as a lower bound is what makes the knob work at all —
        the unit cube gets roomier as ``d`` grows, so a minimum-distance
        constraint stops binding and every layout comes out equally easy. It
        also makes difficulty comparable across ``d`` and ``n_clusters``, which
        is what a dimensionality sweep needs.

        Set to ``None`` to use ``cluster_radius`` literally (MOA-compatible).
    weight_range:
        Cluster sampling weights are drawn from ``1 +/- weight_range`` and
        normalised. ``0`` => balanced clusters.
    noise:
        Fraction of points drawn uniformly over the cube. Their label is
        ``None`` so external metrics skip them.
    noise_in_cluster:
        When ``False`` (default) a noise point that lands inside an existing
        cluster is redrawn, so noise really is background.
    drift_speed:
        Distance each drifting centre travels every ``drift_every`` points.
        ``0`` => static clusters.
    drift_every:
        Interval, in points, between centre movements.
    n_drift_clusters:
        How many clusters drift. ``None`` => all of them.
    event_frequency:
        Interval, in points, between cluster events. ``0`` => no events.
    events:
        Which events may fire. Subset of :data:`EVENTS`.
    n_clusters_range:
        How far the live cluster count may wander from ``n_clusters`` through
        create/delete and merge/split events.
    block_ordered:
        Emit the stream as one contiguous block per cluster instead of i.i.d.
        draws. A control setting for studying degenerate evaluation windows;
        not a realistic clustering stream.
    seed:
        Controls layout and sampling; the stream is fully reproducible.
    """

    n_samples: int = 20_000
    n_features: int = 10
    n_clusters: int = 5

    # geometry
    cluster_radius: float = 0.07
    radius_range: float = 0.0
    separation: float | None = 3.0

    # composition
    weight_range: float = 0.0
    noise: float = 0.0
    noise_in_cluster: bool = False

    # dynamics
    drift_speed: float = 0.0
    drift_every: int = 100
    n_drift_clusters: int | None = None
    event_frequency: int = 0
    events: tuple[str, ...] = ("merge", "split", "create", "delete")
    n_clusters_range: int = 2

    # ordering
    block_ordered: bool = False

    seed: int = 42

    def __post_init__(self):
        bad = set(self.events) - set(EVENTS)
        if bad:
            raise ValueError(f"unknown events {sorted(bad)}; available: {list(EVENTS)}")
        if not 0.0 <= self.noise < 1.0:
            raise ValueError("noise must be in [0, 1)")
        if self.separation is not None and self.separation <= 0:
            raise ValueError("separation must be > 0, or None to use cluster_radius directly")


@dataclass
class _Cluster:
    """One live mixture component."""

    cid: int
    centre: np.ndarray
    radius: float
    weight: float
    velocity: np.ndarray
    #: When set, the centre is being pulled towards this point (a merge in
    #: progress); the pair collapses into one cluster on arrival.
    merge_target: int | None = None


class SyntheticStream(Dataset):
    """Labeled stream of Gaussian clusters with optional drift and events.

    Yields ``(x, y)`` where ``x`` maps feature names to floats and ``y`` is the
    generating cluster id (``None`` for background noise). Re-iterable and
    deterministic given the seed, so every algorithm sees the identical
    sequence.
    """

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()
        cfg = self.config
        self.n_features = cfg.n_features
        self.has_labels = True
        self._feature_names = [f"x{i}" for i in range(cfg.n_features)]
        self.name = self._build_name()

    def _build_name(self) -> str:
        cfg = self.config
        parts = [
            f"synthetic_k{cfg.n_clusters}",
            f"d{cfg.n_features}",
            f"n{cfg.n_samples}",
            f"sep{cfg.separation:g}" if cfg.separation is not None
            else f"r{cfg.cluster_radius:g}",
        ]
        if cfg.drift_speed:
            parts.append(f"drift{cfg.drift_speed:g}")
        if cfg.event_frequency:
            parts.append("events")
        if cfg.noise:
            parts.append(f"noise{cfg.noise:g}")
        if cfg.block_ordered:
            parts.append("blocked")
        return "_".join(parts)

    def __len__(self) -> int:
        return self.config.n_samples

    # -- layout --------------------------------------------------------------
    def _draw_radius(self, rng) -> float:
        """Per-cluster radius, jittered around the layout's base radius."""
        cfg = self.config
        base = getattr(self, "_base_r", cfg.cluster_radius)
        if not cfg.radius_range:
            return base
        return float(base * (1.0 + rng.uniform(-cfg.radius_range, cfg.radius_range)))

    def _draw_weight(self, rng) -> float:
        cfg = self.config
        if not cfg.weight_range:
            return 1.0
        return float(1.0 + rng.uniform(-cfg.weight_range, cfg.weight_range))

    def _draw_velocity(self, rng) -> np.ndarray:
        v = rng.normal(size=self.config.n_features)
        norm = np.linalg.norm(v)
        return v / norm if norm else np.zeros(self.config.n_features)

    _N_CANDIDATES = 64

    def _place_centre(self, rng, existing) -> np.ndarray:
        """Maximin placement: the candidate farthest from what is already placed.

        Purely uniform draws leave the closest pair to chance, so ``d_min`` --
        and with it the derived radius and the whole difficulty of the stream --
        swings wildly between seeds. Picking the best of a few candidates gives a
        well-spread layout whose ``d_min`` is stable, which is what makes results
        comparable across seeds.
        """
        cfg = self.config
        candidates = rng.uniform(0.0, 1.0, size=(self._N_CANDIDATES, cfg.n_features))
        if not existing:
            return candidates[0]
        centres = np.array([c.centre if isinstance(c, _Cluster) else c for c in existing])
        # distance from every candidate to its nearest existing centre
        nearest = np.min(
            np.linalg.norm(candidates[:, None, :] - centres[None, :, :], axis=2), axis=1
        )
        return candidates[int(np.argmax(nearest))]

    @staticmethod
    def _min_pairwise(centres: np.ndarray) -> float:
        if len(centres) < 2:
            return math.inf
        d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
        np.fill_diagonal(d, math.inf)
        return float(d.min())

    def _base_radius(self, centres: np.ndarray) -> float:
        """Radius implied by the layout and the requested separation."""
        cfg = self.config
        if cfg.separation is None:
            return cfg.cluster_radius
        d_min = self._min_pairwise(centres)
        if not math.isfinite(d_min):  # a single cluster has nothing to separate from
            return cfg.cluster_radius
        return d_min / (2.0 * cfg.separation)

    def _initial_clusters(self, rng) -> list[_Cluster]:
        cfg = self.config
        placed: list[np.ndarray] = []
        for _ in range(cfg.n_clusters):
            placed.append(self._place_centre(rng, placed))
        centres = np.array(placed)
        self._base_r = self._base_radius(centres)

        clusters: list[_Cluster] = []
        for cid, centre in enumerate(centres):
            clusters.append(
                _Cluster(
                    cid=cid,
                    centre=centre.copy(),
                    radius=self._draw_radius(rng),
                    weight=self._draw_weight(rng),
                    velocity=self._draw_velocity(rng),
                )
            )
        return clusters

    # -- sampling ------------------------------------------------------------
    def _sample_point(self, rng, cluster: _Cluster) -> np.ndarray:
        """MOA/River radial form: unit direction times a Gaussian magnitude.

        Cluster size stays independent of ``n_features`` this way, which is what
        makes a dimensionality sweep interpretable.
        """
        direction = rng.normal(size=self.config.n_features)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return cluster.centre.copy()
        magnitude = rng.normal(0.0, cluster.radius)
        return cluster.centre + direction / norm * magnitude

    def _sample_noise(self, rng, clusters: list[_Cluster]) -> np.ndarray:
        cfg = self.config
        for _ in range(20):
            point = rng.uniform(0.0, 1.0, size=cfg.n_features)
            if cfg.noise_in_cluster:
                return point
            inside = any(
                float(np.linalg.norm(point - c.centre)) <= c.radius for c in clusters
            )
            if not inside:
                return point
        return rng.uniform(0.0, 1.0, size=cfg.n_features)

    # -- dynamics ------------------------------------------------------------
    def _move(self, clusters: list[_Cluster], n_drift: int) -> None:
        """Advance drifting centres, bouncing off the unit-cube walls."""
        cfg = self.config
        for cluster in clusters[:n_drift]:
            if cluster.merge_target is not None:
                continue  # merge movement is handled separately
            cluster.centre += cluster.velocity * cfg.drift_speed
            for j in range(cfg.n_features):
                if cluster.centre[j] > 1.0 or cluster.centre[j] < 0.0:
                    cluster.centre[j] = 1.0 if cluster.centre[j] > 1.0 else 0.0
                    cluster.velocity[j] = -cluster.velocity[j]

    def _advance_merges(self, clusters: list[_Cluster]) -> list[_Cluster]:
        """Pull merging pairs together; collapse them once they touch."""
        cfg = self.config
        by_id = {c.cid: c for c in clusters}
        absorbed: set[int] = set()
        for cluster in clusters:
            if cluster.merge_target is None:
                continue
            target = by_id.get(cluster.merge_target)
            if target is None or target.cid in absorbed:
                cluster.merge_target = None
                continue
            delta = target.centre - cluster.centre
            distance = float(np.linalg.norm(delta))
            step = max(cfg.drift_speed, cfg.cluster_radius * 0.05)
            if distance <= step:
                # arrival: the moving cluster is absorbed into its target
                target.weight += cluster.weight
                absorbed.add(cluster.cid)
            else:
                cluster.centre += delta / distance * step
        return [c for c in clusters if c.cid not in absorbed]

    def _fire_event(self, rng, clusters: list[_Cluster], next_id: int) -> tuple[list[_Cluster], int]:
        """Apply one randomly chosen enabled event."""
        cfg = self.config
        lo = max(1, cfg.n_clusters - cfg.n_clusters_range)
        hi = cfg.n_clusters + cfg.n_clusters_range

        allowed = []
        for event in cfg.events:
            if event in ("merge", "delete") and len(clusters) <= lo:
                continue
            if event in ("split", "create") and len(clusters) >= hi:
                continue
            if event == "merge" and len(clusters) < 2:
                continue
            allowed.append(event)
        if not allowed:
            return clusters, next_id

        event = allowed[rng.integers(len(allowed))]

        if event == "delete":
            victim = int(rng.integers(len(clusters)))
            clusters.pop(victim)

        elif event == "create":
            clusters.append(
                _Cluster(
                    cid=next_id,
                    centre=self._place_centre(rng, clusters),
                    radius=self._draw_radius(rng),
                    weight=self._draw_weight(rng),
                    velocity=self._draw_velocity(rng),
                )
            )
            next_id += 1

        elif event == "merge":
            # pick the closest pair, so the merge is a gradual approach rather
            # than a teleport across the cube
            pairs = [
                (float(np.linalg.norm(a.centre - b.centre)), i, j)
                for i, a in enumerate(clusters)
                for j, b in enumerate(clusters)
                if i < j and a.merge_target is None and b.merge_target is None
            ]
            if pairs:
                _, i, j = min(pairs)
                clusters[i].merge_target = clusters[j].cid

        elif event == "split":
            parent = clusters[int(rng.integers(len(clusters)))]
            direction = self._draw_velocity(rng)
            offset = direction * parent.radius * 1.5
            parent.weight /= 2.0
            child = _Cluster(
                cid=next_id,
                centre=np.clip(parent.centre - offset, 0.0, 1.0),
                radius=parent.radius,
                weight=parent.weight,
                velocity=-direction,
            )
            parent.centre = np.clip(parent.centre + offset, 0.0, 1.0)
            parent.velocity = direction
            clusters.append(child)
            next_id += 1

        return clusters, next_id

    # -- iteration -----------------------------------------------------------
    def __iter__(self):
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        clusters = self._initial_clusters(rng)
        next_id = len(clusters)
        n_drift = len(clusters) if cfg.n_drift_clusters is None else cfg.n_drift_clusters

        if cfg.block_ordered:
            yield from self._iter_blocked(rng, clusters)
            return

        for t in range(cfg.n_samples):
            if cfg.drift_speed and cfg.drift_every and t and t % cfg.drift_every == 0:
                self._move(clusters, min(n_drift, len(clusters)))
                clusters = self._advance_merges(clusters)
            if cfg.event_frequency and t and t % cfg.event_frequency == 0:
                clusters, next_id = self._fire_event(rng, clusters, next_id)
                if not clusters:  # every cluster deleted: restart the layout
                    clusters = self._initial_clusters(rng)
                    next_id = max(next_id, len(clusters))

            if cfg.noise and rng.random() < cfg.noise:
                point = self._sample_noise(rng, clusters)
                yield self._as_dict(point), None
                continue

            weights = np.array([c.weight for c in clusters], dtype=float)
            weights /= weights.sum()
            chosen = clusters[int(rng.choice(len(clusters), p=weights))]
            yield self._as_dict(self._sample_point(rng, chosen)), chosen.cid

    def _iter_blocked(self, rng, clusters: list[_Cluster]):
        """Control ordering: one contiguous block per cluster, no drift/events.

        Exists so the degenerate-window effect that real block-ordered streams
        exhibit can be reproduced and measured under known conditions.
        """
        cfg = self.config
        per_cluster = cfg.n_samples // len(clusters)
        emitted = 0
        for cluster in clusters:
            for _ in range(per_cluster):
                if cfg.noise and rng.random() < cfg.noise:
                    yield self._as_dict(self._sample_noise(rng, clusters)), None
                else:
                    yield self._as_dict(self._sample_point(rng, cluster)), cluster.cid
                emitted += 1
        while emitted < cfg.n_samples:  # remainder goes to the last cluster
            yield self._as_dict(self._sample_point(rng, clusters[-1])), clusters[-1].cid
            emitted += 1

    def _as_dict(self, point: np.ndarray) -> dict:
        return {name: float(v) for name, v in zip(self._feature_names, point)}


# --- named scenarios -------------------------------------------------------
#: Preset scenarios covering the situations the stream-clustering literature
#: evaluates on. Using a named set keeps results comparable between runs and
#: stops each experiment from inventing its own difficulty.
SCENARIOS: dict[str, dict] = {
    # Baseline: static, well separated, balanced. Every algorithm should score
    # highly here; a low score means the configuration is wrong, not the data.
    "static_easy": dict(separation=4.0),
    # Overlapping clusters. The closest pair sits one radius apart, so the
    # boundary is genuinely ambiguous — this is where algorithms separate.
    "static_overlap": dict(separation=1.0),
    # Unequal sizes and densities: the common real-world shape, and the case
    # where methods with a single global radius struggle.
    "unbalanced": dict(separation=2.5, weight_range=0.8, radius_range=0.6),
    # 10% uniform background noise, kept out of the clusters. Tests whether a
    # method absorbs noise into clusters or isolates it.
    "noisy": dict(separation=3.0, noise=0.1),
    # Centres drift steadily: tests forgetting. A method with no decay keeps
    # stale micro-clusters and degrades along the stream.
    "incremental_drift": dict(separation=3.0, drift_speed=0.002, drift_every=100),
    # Clusters merge, split, appear and disappear — the behaviour that actually
    # distinguishes a stream clusterer from a batch one applied repeatedly.
    "evolving": dict(separation=3.0, drift_speed=0.001, event_frequency=5_000),
    # Control, not a realistic stream: the same clusters emitted in contiguous
    # same-class blocks, the ordering that makes per-window external metrics
    # degenerate. Use it to verify the diagnostic fires.
    "block_ordered": dict(separation=4.0, block_ordered=True),
}


def make_scenario(name: str, **overrides) -> SyntheticStream:
    """Build one of the named :data:`SCENARIOS`.

    ``make_scenario("static_overlap", n_features=50, seed=7)`` keeps the
    scenario's difficulty settings and overrides the rest.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
    params = dict(SCENARIOS[name])
    params.update(overrides)
    stream = SyntheticStream(StreamConfig(**params))
    stream.name = f"{name}_d{stream.config.n_features}_n{stream.config.n_samples}"
    return stream
