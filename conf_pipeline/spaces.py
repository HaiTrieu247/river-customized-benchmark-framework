"""The search space of every algorithm in the protocol.

The ranges of DenStream, CluStream and RepStream are the ones
``hpo_pipeline/space.py`` searches, so the two pipelines can be read against
each other; the other five are new here. Every parameter is explored on its own
scale (see ``benchmark.tuning.space``) and the only data-dependent unit is
``s``, the median pairwise distance of the calibration window. Six algorithms
consume it, each through one distance parameter of kind ``scaled``: DenStream's
``epsilon``, DBSTREAM's ``clustering_threshold``, SOStream's ``merge_threshold``,
BIRCH's ``threshold``, and the ``sigma`` that CluStream and STREAMKMeans forward
to their inner ``KMeans``. That ``KMeans`` draws every centre coordinate from
N(mu=0, sigma=1) in absolute units (RIVER_BUGS.md GAP-5), so on data whose
spread is far from 1 all points fall to one centre; ``sigma`` in units of ``s``
follows the data's spread. It cannot fix a stream whose features sit on
different scales or far from the origin (covtype, INSECTS raw): no single
N(mu, sigma) covers them, and the protocol does not rescale. The rest carry no
distance in their parameters: HPStream normalises internally and RepStream's
``alpha`` scales a representative's own local density.

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
            Param("sigma", "scaled", 0.01, 1.0),
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
            Param("sigma", "scaled", 0.01, 1.0),
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

#: DenStreamPatched (BUG-7/BUG-8) and DenStreamPatchedFast (the same, with GAP-3
#: closed) are River-side workarounds registered under their own names; the
#: search space they tune is identical to DenStream's. The protocol runs the
#: fast one as DenStream's implementation variant (``_common.IMPL_VARIANTS``).
SPACES["DenStreamPatched"] = dataclasses.replace(SPACES["DenStream"], algo="DenStreamPatched")
SPACES["DenStreamPatchedFast"] = dataclasses.replace(SPACES["DenStream"], algo="DenStreamPatchedFast")


#: River's HPStream projects on this share of the dimensions when
#: ``n_projected_dimensions`` is left at ``None`` (`hpstream._DEFAULT_PROJECTION`).
HPSTREAM_DEFAULT_PROJECTION = 0.6

#: Tunable names River's signature does not default: CluStream and
#: STREAMKMeans forward them through **kwargs to KMeans, whose defaults these are.
START_OVERRIDES = {
    "CluStream": {"sigma": 1.0},
    "STREAMKMeans": {"halflife": 0.5, "sigma": 1.0},
}
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


def distance_params(algo: str) -> tuple[str, ...]:
    """The parameters of ``algo`` read as an absolute distance: the space's ``scaled`` dimensions
    (see the module docstring for the six; HPStream and RepStream have none)."""
    return tuple(p.name for p in SPACES[algo].params if p.kind == "scaled")


def scaled_defaults(algo: str, params: dict, s: float) -> tuple[dict, dict]:
    """``params`` (a start, see ``start_params``) with every distance parameter multiplied by the
    distance unit ``s``, everything else as given; and what was changed, per parameter, as
    ``{"default", "s", "value"}``.

    The control of E1 (``run.py --arms scaled_default``): River's defaults are absolute, so on
    data whose spread is far from 1 they can fail for their units alone; read in units of ``s``
    they ask whether the defaults' failures are units or the rest of the configuration. A zero
    default stays zero (BIRCH's ``threshold``): nothing is added to make the control better.
    A single ``s`` changes neither a feature's scale relative to another nor a translation
    (KMeans' ``mu``).
    """
    out, record = dict(params), {}
    for name in distance_params(algo):
        if name in out and out[name] is not None:
            out[name] = out[name] * s
            record[name] = {"default": params[name], "s": s, "value": out[name]}
    return out, record


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
