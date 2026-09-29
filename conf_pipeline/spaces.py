"""The search space of every algorithm in the protocol.

The ranges of DenStream, CluStream and RepStream are the ones
``hpo_pipeline/space.py`` searches, so the two pipelines can be read against
each other; the other five are new here. Every parameter is explored on its own
scale (see ``benchmark.tuning.space``) and the only data-dependent unit is
``s``, the median pairwise distance of the calibration window. Four algorithms
consume it, each through one distance parameter of kind ``scaled``: DenStream's
``epsilon``, DBSTREAM's ``clustering_threshold``, SOStream's ``merge_threshold``
and BIRCH's ``threshold``. The rest carry no distance in their parameters:
CluStream tunes a count and a radius factor, HPStream normalises internally,
RepStream's ``alpha`` scales a representative's own local density, and
STREAMKMeans tunes a count, a chunk size and a half-life.

``n_samples_init`` (and STREAMKMeans' ``chunk_size``) is capped at the window
size, for the start (``start_params``) and for every child the ensemble draws
(``clamp_init_buffers``, applied by the driver): a member whose buffer exceeds
the window spends its whole test window buffering, has no partition to be
scored on, and can never be accepted -- whatever the rest of its
configuration is worth.
"""

from __future__ import annotations

import dataclasses

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
    "DBSTREAM": ParamSpace(
        "DBSTREAM",
        (
            Param("clustering_threshold", "scaled", 0.05, 2.0),
            Param("fading_factor", "log", 0.001, 0.1),
            Param("cleanup_interval", "log_int", 2, 500),
            Param("intersection_factor", "lin", 0.1, 0.5),
            Param("minimum_weight", "lin", 0.5, 3.0),
        ),
    ),
    "STREAMKMeans": ParamSpace(
        "STREAMKMeans",
        (
            Param("n_clusters", "log_int", 2, 60),
            Param("chunk_size", "choice", values=(10, 100, 500, 1000)),
            Param("halflife", "lin", 0.1, 0.9),
        ),
    ),
    "SOStream": ParamSpace(
        "SOStream",
        (
            Param("alpha", "lin", 0.01, 0.5),
            Param("min_pts", "log_int", 2, 10),
            Param("merge_threshold", "scaled", 0.05, 2.0),
        ),
        fixed={"fading_factor": 0.1, "fade_threshold": 0.1, "fading_interval": 100, "stream_speed": 1},
    ),
    # Phase 3 stays OFF: with n_macro_clusters set River re-runs agglomerative
    # clustering on every predict_one (learn_one invalidates the cache), which is
    # infeasible prequentially. Leaf entries are the partition; Table A supplies
    # the macro reading.
    "BIRCH": ParamSpace(
        "BIRCH",
        (
            Param("threshold", "scaled", 0.05, 2.0),
            Param("branching_factor", "log_int", 10, 100),
            Param("max_entries", "log_int", 50, 2000),
        ),
        fixed={"leaf_capacity": 50, "distance_measure": "d2", "threshold_measure": "diameter",
               "n_macro_clusters": None, "macro_threshold": None},
    ),
}

#: DenStreamPatched (Task 7) is a River-side workaround registered under its own
#: name; the search space it tunes is identical to DenStream's.
SPACES["DenStreamPatched"] = dataclasses.replace(SPACES["DenStream"], algo="DenStreamPatched")


#: River's HPStream projects on this share of the dimensions when
#: ``n_projected_dimensions`` is left at ``None`` (`hpstream._DEFAULT_PROJECTION`).
HPSTREAM_DEFAULT_PROJECTION = 0.6

#: Tunable names River's signature does not default (STREAMKMeans forwards
#: halflife through **kwargs to KMeans, whose default is 0.5).
START_OVERRIDES = {"STREAMKMeans": {"halflife": 0.5}}
#: Buffers a member must fill before it partitions anything; capped at h.
INIT_BUFFER_PARAMS = ("n_samples_init", "chunk_size")


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
    for k, v in START_OVERRIDES.get(algo, {}).items():
        params.setdefault(k, v)
    for name in INIT_BUFFER_PARAMS:
        if name in params:
            params[name] = _snap_choice(space, name, min(params[name], h))
    if algo == "HPStream" and params.get("n_projected_dimensions") is None:
        if d is None:
            raise ValueError("HPStream's start needs the feature count d to write out River's 60%")
        params["n_projected_dimensions"] = max(1, int(round(HPSTREAM_DEFAULT_PROJECTION * d)))
    return {k: v for k, v in params.items() if k in space.names or k in space.fixed}


def clamp_init_buffers(algo: str, params: dict, h: int) -> dict:
    """``params`` with every initialisation buffer (``INIT_BUFFER_PARAMS``)
    that exceeds the window ``h`` brought back within it; a new dict.

    A buffer above ``h`` becomes the largest declared choice not above ``h``
    (the smallest choice when every one is above it, so the value stays a point
    of the space -- the same rule as ``start_params``); a value already within
    ``h`` is left alone. An algorithm without a protocol space has no choices
    to snap to and is capped at ``h`` itself.
    """
    space = SPACES.get(algo)
    out = dict(params)
    for name in INIT_BUFFER_PARAMS:
        value = out.get(name)
        if value is not None and value > h:
            out[name] = _snap_choice(space, name, h) if space is not None else h
    return out


def _snap_choice(space: ParamSpace, name: str, value):
    """The largest declared choice not above ``value`` (so it stays in range)."""
    for p in space.params:
        if p.name == name and p.kind == "choice":
            below = [v for v in p.values if v <= value]
            return max(below) if below else min(p.values)
    return value
