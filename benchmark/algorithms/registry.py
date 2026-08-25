"""Registry of clusterers + reference presets, and online-scaler factory.

Presets are sensible *starting points* (some drawn from the original papers, see
README). You override any param per benchmark setting. Scalers are looked up by
name because the README shows the scaler strongly affects the resulting number
of clusters and quality.
"""

from __future__ import annotations

from typing import Any, Callable

from .adapter import ClustererAdapter, build_river_path, make_adapter

# Ensure the sibling river tree is importable before we touch river.*.
build_river_path()

from river import base, cluster, preprocessing  # noqa: E402


# --- clusterer factories: name -> (model, resolved_params) -----------------
def _clustream(**kw) -> Any:
    params = dict(n_macro_clusters=5, max_micro_clusters=100, time_gap=100, seed=42)
    params.update(kw)
    return cluster.CluStream(**params), params


def _denstream(**kw) -> Any:
    params = dict(
        decaying_factor=0.25, beta=0.5, mu=2.5, epsilon=0.5, n_samples_init=1000,
        stream_speed=100,
    )
    params.update(kw)
    return cluster.DenStream(**params), params


def _dbstream(**kw) -> Any:
    params = dict(
        clustering_threshold=1.0, fading_factor=0.01, cleanup_interval=2,
        intersection_factor=0.3, minimum_weight=1.0,
    )
    params.update(kw)
    return cluster.DBSTREAM(**params), params


def _streamkmeans(**kw) -> Any:
    params = dict(chunk_size=50, n_clusters=5, seed=42)
    params.update(kw)
    return cluster.STREAMKMeans(**params), params


def _kmeans(**kw) -> Any:
    params = dict(n_clusters=5, halflife=0.5, seed=42)
    params.update(kw)
    return cluster.KMeans(**params), params


#: name -> factory returning (model, resolved_params)
ALGORITHMS: dict[str, Callable[..., Any]] = {
    "CluStream": _clustream,
    "DenStream": _denstream,
    "DBSTREAM": _dbstream,
    "STREAMKMeans": _streamkmeans,
    "KMeans": _kmeans,
}


# --- online scalers: name -> zero/short-arg factory ------------------------
SCALERS: dict[str, Callable[[], Any]] = {
    "none": lambda: None,
    "standard": preprocessing.StandardScaler,
    "minmax": preprocessing.MinMaxScaler,
    "maxabs": preprocessing.MaxAbsScaler,
    "robust": preprocessing.RobustScaler,
    "adaptive_standard": preprocessing.AdaptiveStandardScaler,
}


def make_scaler(name: str):
    if name not in SCALERS:
        raise KeyError(f"unknown scaler {name!r}; available: {sorted(SCALERS)}")
    return SCALERS[name]()


def get_algorithm(
    name: str,
    display_name: str | None = None,
    scaler: str = "none",
    **overrides,
) -> ClustererAdapter:
    """Instantiate a registered algorithm (optionally with a scaler) as an adapter."""
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(ALGORITHMS)}")
    model, params = ALGORITHMS[name](**overrides)
    scaler_obj = make_scaler(scaler)
    return make_adapter(
        display_name or name, model, params,
        scaler=scaler_obj, scaler_name=scaler,
    )


def discover_clusterers() -> dict[str, type]:
    """All ``river.cluster`` classes deriving from ``base.Clusterer`` (README)."""
    out: dict[str, type] = {}
    for attr in dir(cluster):
        obj = getattr(cluster, attr)
        if isinstance(obj, type) and issubclass(obj, base.Clusterer) and obj is not base.Clusterer:
            out[attr] = obj
    return out
