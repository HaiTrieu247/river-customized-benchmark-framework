"""Registry of clusterers + paper-reference presets (README, §22).

Presets are *starting points*; the hyperparameter protocol (§19, Phase 5) is
responsible for selecting/freezing the actual params used on test streams.
"""

from __future__ import annotations

from typing import Any, Callable

from .adapter import ClustererAdapter, build_river_path, make_adapter

# Ensure the sibling river tree is importable before we touch river.cluster.
build_river_path()

from river import base, cluster  # noqa: E402


def _clustream(**kw) -> Any:
    params = dict(n_macro_clusters=3, max_micro_clusters=50, time_gap=100, seed=42)
    params.update(kw)
    return cluster.CluStream(**params), params


def _denstream(**kw) -> Any:
    # epsilon must match the data scale: the synthetic blobs have spread ~1 and
    # separation ~10, so epsilon must be large enough to form (but not merge)
    # the ground-truth clusters (README caveat on epsilon vs scaler).
    params = dict(
        decaying_factor=0.25, beta=0.5, mu=5.0, epsilon=4.0, n_samples_init=500,
        stream_speed=100,
    )
    params.update(kw)
    return cluster.DenStream(**params), params


def _dbstream(**kw) -> Any:
    params = dict(
        clustering_threshold=2.0, fading_factor=0.05, cleanup_interval=100,
        intersection_factor=0.3, minimum_weight=1.0,
    )
    params.update(kw)
    return cluster.DBSTREAM(**params), params


def _streamkmeans(**kw) -> Any:
    params = dict(chunk_size=50, n_clusters=3, seed=42)
    params.update(kw)
    return cluster.STREAMKMeans(**params), params


#: name -> factory returning (model, resolved_params)
ALGORITHMS: dict[str, Callable[..., Any]] = {
    "CluStream": _clustream,
    "DenStream": _denstream,
    "DBSTREAM": _dbstream,
    "STREAMKMeans": _streamkmeans,
}


def get_algorithm(name: str, display_name: str | None = None, **overrides) -> ClustererAdapter:
    """Instantiate a registered algorithm as a :class:`ClustererAdapter`."""
    if name not in ALGORITHMS:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {sorted(ALGORITHMS)}"
        )
    model, params = ALGORITHMS[name](**overrides)
    return make_adapter(display_name or name, model, params)


def discover_clusterers() -> dict[str, type]:
    """All ``river.cluster`` classes deriving from ``base.Clusterer`` (README)."""
    out: dict[str, type] = {}
    for attr in dir(cluster):
        obj = getattr(cluster, attr)
        if isinstance(obj, type) and issubclass(obj, base.Clusterer) and obj is not base.Clusterer:
            out[attr] = obj
    return out
