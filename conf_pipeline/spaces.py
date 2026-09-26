"""The search space of every algorithm in the protocol.

The ranges of DenStream, CluStream and RepStream are the ones
``hpo_pipeline/space.py`` searches, so the two pipelines can be read against
each other; HPStream is new here. Every parameter is explored on its own scale
(see ``benchmark.tuning.space``) and the only data-dependent unit is ``s``, the
median pairwise distance of the calibration window, which DenStream's
``epsilon`` alone consumes. The other three carry no distance in their
parameters: CluStream tunes a count and a radius factor, HPStream normalises
internally, RepStream's ``alpha`` scales a representative's own local density.

``n_samples_init`` is capped at the window size: a child spends its whole test
window buffering otherwise, has no partition to be scored on, and can never be
accepted -- whatever the rest of its configuration is worth.
"""

from __future__ import annotations

from benchmark.tuning.space import Param, ParamSpace

from ._common import river_defaults

SPACES: dict[str, ParamSpace] = {
    "DenStream": ParamSpace(
        "DenStream",
        (
            Param("epsilon", "scaled", 0.05, 2.0),
            Param("mu", "log", 1.5, 12.0),
            Param("decaying_factor", "log", 0.05, 1.0),
            Param("n_samples_init", "choice", values=(200, 500, 1000)),
        ),
        fixed={"beta": 0.75, "stream_speed": 100},
    ),
    "CluStream": ParamSpace(
        "CluStream",
        (
            Param("n_macro_clusters", "log_int", 2, 60),
            Param("max_micro_clusters", "log_int", 25, 800),
            Param("micro_cluster_r_factor", "choice", values=(1, 2, 4, 8)),
        ),
        fixed={"time_gap": 100},
    ),
    "HPStream": ParamSpace(
        "HPStream",
        (
            Param("max_clusters", "log_int", 2, 60),
            Param("n_projected_dimensions", "frac_of_d", 0.2, 1.0),
            Param("spread_radius_factor", "lin", 1.5, 4.0),
            Param("decaying_factor", "log", 0.05, 2.0),
            Param("n_samples_init", "choice", values=(0, 200, 500, 1000)),
        ),
        fixed={"radius_threshold": None, "stream_speed": 200, "normalization_interval": 1000},
    ),
    "RepStream": ParamSpace(
        "RepStream",
        (
            Param("alpha", "lin", 1.05, 3.0),
            Param("k", "log_int", 3, 20),
            Param("decay_rate", "log", 0.9, 0.9999),
            Param("max_vertices", "log_int", 50, 1000),
            Param("repository_fraction", "lin", 0.2, 0.8),
        ),
        fixed={"distance_measure": "euclidean"},
    ),
}


#: River's HPStream projects on this share of the dimensions when
#: ``n_projected_dimensions`` is left at ``None`` (`hpstream._DEFAULT_PROJECTION`).
HPSTREAM_DEFAULT_PROJECTION = 0.6


def start_params(algo: str, h: int, d: int | None = None) -> dict:
    """River's defaults for ``algo``, with the pinned values applied and the
    initialisation buffer capped at the window size.

    ``n_projected_dimensions`` defaults to ``None`` in River (60% of the
    dimensions seen); the space needs a number and River needs an integer, so
    the start is that same 60% of ``d``, the feature count of the fold.
    """
    space = SPACES[algo]
    params = river_defaults(algo)
    params.update(space.fixed)
    if "n_samples_init" in params:
        params["n_samples_init"] = _snap_choice(space, "n_samples_init", min(params["n_samples_init"], h))
    if algo == "HPStream" and params.get("n_projected_dimensions") is None:
        if d is None:
            raise ValueError("HPStream's start needs the feature count d to write out River's 60%")
        params["n_projected_dimensions"] = max(1, int(round(HPSTREAM_DEFAULT_PROJECTION * d)))
    return {k: v for k, v in params.items() if k in space.names or k in space.fixed}


def _snap_choice(space: ParamSpace, name: str, value):
    """The largest declared choice not above ``value`` (so it stays in range)."""
    for p in space.params:
        if p.name == name and p.kind == "choice":
            below = [v for v in p.values if v <= value]
            return max(below) if below else min(p.values)
    return value
