"""Pilot: four ways of choosing ONE hyperparameter set, on River's own clusterers.

Before the pipeline is scaled to the whole pool it is checked on the two
stream clusterers River itself ships -- ``CluStream`` and ``DenStream`` -- where
no estimator written for this project can muddy the picture.

Each algorithm gets a small grid over the parameters that dominate it: the
resolution axis (``n_macro_clusters`` for CluStream, ``epsilon`` for DenStream)
times a few shape parameters. The value sets come from the two source papers;
``GRIDS`` below documents where every value comes from. Everything not in the
grid stays at River's *constructor* default.

Every arm ends with **one set per algorithm**, and that one set is then run
unchanged on every dataset. The arms differ in the grid and in how the one set
is chosen:

    arm 1  raw grid          the set with the best score over all datasets
    arm 2  raw grid          leave-one-dataset-out: every round picks a set on
                             the other datasets; the final set is the one most
                             rounds picked (ties: arm 1's ordering)
    arm 3  substituted grid  as arm 2. The set is dimensionless (``c``, a
                             micro-ratio) and resolves per dataset through s(D)
    P2R    substituted grid  the shape is chosen as in arm 3 and the rule's phi
                             on all datasets; the resolution level is then read
                             off each dataset's own label-free curve. The
                             product is a *policy*, not one epsilon.
    sf-all substituted grid  arm 1 on the substituted grid, to separate "better
                             grid" from "cost of the LODO choice"

``--aggregate`` sets how scores are pooled across datasets when a set is
chosen: ``headroom`` (the default -- each dataset's share of its own
default-to-ceiling range, over the datasets that pass the tunability gate),
``mean`` ARI, or ``rank`` (mean per-dataset rank, so one easy dataset cannot
outvote the others).

``--irace raw,ladder,p2r`` adds irace arms (``irace_tune``): the same one-set
choice, made by racing the candidates over (dataset, seed) instances with
cost = -headroom instead of taking an argmax over one tuning draw. They are
reported beside the grid arm that chooses from the same space (section 3c).

Scoring the final set on the datasets it was chosen on is in-sample. Arms 2, 3
and P2R therefore also report their **LODO estimate**: in every round the set
chosen without a dataset is scored on that dataset. That estimate, not the
in-sample score, is what the procedure is worth on a dataset it never saw.

Tuning runs every grid point under every seed in ``TUNE_SEEDS`` and selects on
their mean (one seed at present -- see that constant for what it costs on
KDDCup99). Every chosen set then runs again under each seed in ``FINAL_SEEDS``,
which is disjoint from the tuning seeds -- re-running under a tuning seed would
only repeat the tuning score. Each seed carries its own stream permutation, so
one seed moves both the model and the arrival order. River's default is scored
the same way.

Run from the repo root::

    python -m hpo_pipeline.hpo_compare --n-samples 50000 --workers 12 --timeout 7200
    python -m hpo_pipeline.hpo_compare --n-samples 50000 --aggregate rank
    python -m hpo_pipeline.hpo_compare --n-samples 2000            # smoke test
    python -m hpo_pipeline.hpo_compare --report-only
    python -m hpo_pipeline.hpo_compare --grids '{"DenStream": {"levels": {"raw": [0.1, 1, 10]}}}'

    # re-run only what timed out, with a longer limit; check first with --dry-run
    python -m hpo_pipeline.hpo_compare --phase tune --retry-failed --timeout 7200
    python -m hpo_pipeline.hpo_compare --phase score --retry-failed --timeout 7200

    # irace arms (R + the irace package): tune, then race, then score
    python -m hpo_pipeline.hpo_compare --n-samples 50000 --workers 12 --irace raw,ladder,p2r

P2R readings are added by default; ``--p2r`` chooses which, ``--p2r ""`` none.

Folds must already have a Stage 2 probe (``configs/probe.json``). Missing
permutations for the scoring seeds are materialised on the way.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ._common import (
    FOLD_GROUPS,
    FOLDS,
    HPO_OUT,
    PROBE_JSON,
    SCALER,
    SCALER_PARAMS,
    append_jsonl,
    config_id,
    read_json,
    river_defaults,
    read_jsonl,
    require,
    shuffle_seed_for,
)
from .build_matrix import execute
from .rules import RULES, orient
from .space import GRAN, resolve
from .stream import materialise

ALGOS = ("CluStream", "DenStream")
#: Tuning seeds. Every grid point runs under all of them and the cell is their
#: MEAN (`seed_mean`), so the number of them is a budget dial, not a structural
#: choice: the selection code never sees a seed either way.
#:
#: Currently ONE, which is a deliberate cost decision and not a claim that one
#: draw is enough. Measured on the runs already on disk, |ARI(seed 1) - mean of
#: two fresh seeds| over the re-run configs is:
#:
#:     covtype   0.0004      keystroke  0.0061      CluStream, all folds  0.0013
#:     shuttle   0.0030      insects    0.0033      kdd99 (DenStream)     0.1112
#:
#: Four of the five folds are far below the gaps between the arms, so one seed
#: costs nothing there. **KDDCup99 is not.** Its per-config spread over three
#: seeds reaches sd = 0.18 (one configuration scored 0.071 / 0.086 / 0.392), and
#: the arm-to-arm differences this pilot reports on that fold are 0.05 - 0.08 --
#: smaller than the seed noise. A kdd99 pick is therefore a single draw of a
#: quantity that moves more than the effect, and any reading of the kdd99 column
#: has to say so. Raising this tuple to (1, 2, 3, 4, 5) cuts that by sqrt(5) and
#: costs 480 extra tuning runs; nothing else in the module changes.
#:
#: Disjoint from `FINAL_SEEDS` by construction, and that is the whole point of
#: having two sets: a chosen set re-run under a seed it was chosen on would just
#: repeat its own selection luck, which is the number the split exists to avoid.
TUNE_SEEDS = (1,)
#: Scoring seeds. Each pairs a model seed with its own stream permutation
#: (`shuffle_seed_for`), so one seed moves both and the spread over them is the
#: total run-to-run variance, not half of it.
#:
#: Five, not two. With two draws the sample standard deviation has one degree of
#: freedom: it is itself so noisy that "beats the default by 0.03 +/- 0.02" can
#: mean anything, and the noise floor that health check 4a divides by is
#: estimated from almost nothing. Five gives four degrees of freedom -- still
#: small, but enough that a gap of the size this pilot reports stops being
#: indistinguishable from permutation luck. The cost is linear in this number
#: and falls on the SCORING phase only; tuning stays at one run per grid point.
FINAL_SEEDS = (2, 3, 4, 5, 6)
OBJ = "ari"

OUT_DIR = HPO_OUT / "compare"
RUNS_JSONL = OUT_DIR / "runs.jsonl"


def set_out_dir(path) -> None:
    """Redirect every artifact of this module (``--out-dir``; smoke tests)."""
    global OUT_DIR, RUNS_JSONL
    from pathlib import Path

    OUT_DIR = Path(path)
    RUNS_JSONL = OUT_DIR / "runs.jsonl"

# --- the grids ---------------------------------------------------------------
#
# Levels of the resolution axis are listed COARSE -> FINE (Stage 3 convention:
# level 1 is the fewest clusters). Counts therefore ascend and distance
# thresholds descend. ``raw`` values are absolute; ``ladder`` values are the
# substitutes that ``resolve_candidate`` turns into absolute values per fold.
#
# CluStream (Aggarwal, Han, Wang, Yu, VLDB 2003; River implementation):
#   n_macro_clusters (k)  the offline k-means input. Log2 ladder 2..64 so that
#                         every real fold's class count (2..51) has a level
#                         within a factor of sqrt(2).
#   max_micro_clusters(q) Sec. 3: "significantly larger than the natural number
#                         of clusters". Sec. 6.4 defines the micro-ratio q/k and
#                         shows SSQ stabilising at a ratio of about 10 -- quoted
#                         verbatim and verified 2026-09-22 against the paper
#                         text: "The average SSQ for each real dataset becomes
#                         stable when the micro-ratio is about 10." (Two details
#                         worth knowing when reading Fig. 9: its x-axis is
#                         labelled micro/MACRO-clusters while the text defines
#                         the ratio as micro/NATURAL-clusters, and the sweep was
#                         run at stream speed 200 with horizon 16.) The
#                         scalability runs fix q = 10k. Raw: the paper's own
#                         absolute q on KDD (10 x 5 clusters = 50), River's
#                         default 100, and its double. Ladder: ratio {2, 5, 10}
#                         -- the rising part of Fig. 9, just before the plateau,
#                         and the plateau. Beyond 10 the paper reports no gain,
#                         and River's micro-cluster maintenance is O(q^2) per
#                         unabsorbed point: q = 1280 (ratio 20 at k = 64) timed
#                         out at 600 s on a 2000-sample smoke test.
#   micro_cluster_r_factor (t)  the maximal boundary factor. Table 2 sweeps
#                         t in {1, 2, 4, 6, 8}: 14.9% of points fall outside at
#                         t=1, 1.6% at t=2, 0.18% at t=4, then nothing changes.
#                         t=2 is the paper's choice (the 95% rule for Gaussian
#                         deviations); {1, 2, 4} spans the part of the table
#                         where the behaviour still moves.
#   fixed: time_window=1000 (the horizon; River default), time_gap=100 (River's
#          own offline trigger, no paper analogue), halflife of the inner KMeans.
#   constraint: q >= k, or the offline k-means has fewer pseudo-points than
#          seeds (Sec. 4).
#
# DenStream (Cao, Ester, Qian, Zhou, SDM 2006; River implementation):
#   epsilon               the micro-cluster radius bound (Def. 3.3) and the
#                         DBSCAN reach (Def. 4.1: 2*epsilon). The paper sets
#                         epsilon = alpha * d_min with d_min the smallest
#                         inter-cluster gap -- a recipe that needs labels. Raw:
#                         7 geometric levels anchored at the two published
#                         defaults, River's 0.02 and the paper's 16. Ladder:
#                         c in {0.05 .. 3.2}, 7 levels a factor 2 apart, with
#                         epsilon = c * s(D); the top end covers the pilot's
#                         endpoint optimum on shuttle.
#   mu                    the core weight (Def. 3.1). The total stream weight
#                         is W = v / (1 - 2^-lambda) (Sec. 3), so at most W/mu
#                         core micro-clusters can exist; with River's v=100 and
#                         lambda=0.25, W ~ 629 and mu in {2, 5, 10, 20} allows
#                         {315, 126, 63, 31} of them. 2 is River's default, 10
#                         the paper's; the paper says "choose the smallest
#                         acceptable mu".
#   decaying_factor (lambda)  the fading rate. It replaced `beta` on this axis:
#                         see FIXED_NOTE for why beta had to go, and the short
#                         version for why lambda took the slot is that it is the
#                         only remaining parameter that moves cost without
#                         touching epsilon. It sets the total stream weight
#                         W = v / (1 - 2^-lambda) and hence the ceiling W/(beta*mu)
#                         on p-micro-clusters, and it sets the pruning period
#                         T_p (Eq. 4.1). With v=100 and beta*mu=3.75:
#                             lambda  half-life (samples)   W     ceiling P
#                             0.125         800           1205       321
#                             0.25          400            629       168
#                             1.0           100            200        53
#                         Fig. 19 sweeps lambda from 0.00625 to 4 and finds a
#                         plateau over [0.125, 1]. The grid {0.25, 1, 2} was NOT
#                         read off that figure -- it was measured here, on a
#                         5-fold pilot at 20k samples (c=0.4, mu=5), because the
#                         figure's plateau is about quality only and says nothing
#                         about where the cost moves:
#                             lambda   insects ARI / samples-per-s   shuttle
#                             0.0625        0.061 /   561           0.016 / 2011
#                             0.25          0.061 /   580           0.014 / 2075
#                             1.0           0.062 /  1297           0.017 / 3351
#                             2.0           0.078 /  2793           0.032 / 10050
#                             4.0           0.005 / 31799           0.045 / 45997
#                         Two things fall out. Below 0.25 nothing moves -- ARI is
#                         flat to three decimals and throughput within 40% -- so
#                         spending a grid point there buys a duplicate. And the
#                         interesting region is lambda >= 1, where insects and
#                         shuttle get *more* ARI and several times the
#                         throughput: a strict Pareto improvement, not a trade.
#                         4.0 is excluded because it collapses covtype, insects
#                         and keystroke. Caveat on the record: one tuning seed at
#                         20k, and the insects gain (+0.017) is about 1.2x that
#                         fold's measured seed noise -- suggestive, not settled.
#                         Note what this does and
#                         does not claim: the earlier decision to fix lambda was
#                         argued on quality ("after shuffling it only sets
#                         memory"), and that argument still holds. It is tuned
#                         now because memory and throughput are themselves an
#                         objective (report 3b-iii), not because it was wrong.
#                         Measured: at lambda=0.25 the observed micro-cluster
#                         count reaches only 7.6-23.4% of its ceiling, so lambda
#                         is not currently the binding constraint -- which is
#                         precisely why it is worth a slot.
#   fixed: beta=0.75 (pinned, see FIXED_NOTE), n_samples_init=1000 (paper and
#          River agree), stream_speed=100 (River; the paper uses 1000, which is
#          why its mu is larger). `stream_speed` is deliberately NOT on the shape
#          axis even though it also moves W: half-life in samples is v/lambda, so
#          v and lambda are very nearly the same knob, and lambda is the one the
#          paper actually swept.
#   constraint: beta * mu > 1, or the pruning period T_p (Eq. 4.1) is
#          undefined; River computes log(beta*mu / (beta*mu - 1)). With beta
#          pinned at 0.75 and mu >= 2 this can no longer reject a grid point --
#          it stays to guard --grids overrides.
GRIDS: dict[str, dict] = {
    "CluStream": {
        "levels": {
            "raw": [2, 4, 8, 16, 32, 64],
            "ladder": [2, 4, 8, 16, 32, 64],
        },
        "shape": {
            "raw": {"max_micro_clusters": [50, 100, 200], "micro_cluster_r_factor": [1, 2, 4]},
            "ladder": {"micro_ratio": [2, 5, 10], "micro_cluster_r_factor": [1, 2, 4]},
        },
    },
    "DenStream": {
        "levels": {
            "raw": [16.0, 5.5, 1.8, 0.6, 0.2, 0.06, 0.02],
            "ladder": [3.2, 1.6, 0.8, 0.4, 0.2, 0.1, 0.05],
        },
        "shape": {
            "raw": {"mu": [2, 5, 10, 20], "decaying_factor": [0.25, 1.0, 2.0]},
            "ladder": {"mu": [2, 5, 10, 20], "decaying_factor": [0.25, 1.0, 2.0]},
        },
        # Pinned on purpose, not inherited -- see FIXED below.
        "fixed": {"beta": 0.75},
    },
}

#: Parameters a grid pins **deliberately**, as opposed to the ones it simply
#: leaves at River's constructor default.
#:
#: The distinction is not bookkeeping. Dropping a dimension out of ``shape``
#: silently sends every run to River's default for it, and that default can be
#: the value the source paper warns about -- so "pinned at a value we can
#: defend" and "whatever the library happened to ship" have to be separable in
#: the report and in ``analyze``'s sensitivity table. A value listed here is
#: named in the grid section, travels into ``config_id`` and into every result
#: row, and is checked against the grid so a parameter cannot be pinned and
#: swept at the same time.
FIXED_NOTE = {
    ("DenStream", "beta"):
        "beta reaches River's DenStream in exactly three places -- denstream.py:172 "
        "(the pruning period), :220 (promoting an o-micro-cluster) and :334 (pruning a "
        "p-micro-cluster) -- and in all three only as the product `mu * beta`. `mu` also "
        "acts alone, at :243 and :302. So beta is a sub-channel of mu: there is nothing "
        "it can do that mu cannot, while mu does two things it cannot. Measured on the "
        "50k pilot, pinning it costs 0.0004 ARI of per-dataset ceiling summed over five "
        "datasets, and it explains 0.0% of the ARI variance on four of them. 0.75 rather "
        "than 0.5 because it is River's own default, it leaves the mu ladder untouched, "
        "and it makes `beta*mu > 1` hold for every mu in the ladder -- under beta=0.5 the "
        "constraint rejects mu=2, which is where keystroke's ceiling sits. Fig. 20 of the "
        "DenStream paper sweeps beta at a fixed mu=10, so its warning is about the "
        "PRODUCT; with beta pinned the mu ladder still sweeps that product over "
        "{1.5, 3.75, 7.5, 15}.",
}

#: Shape substitutes: dimensionless key -> (absolute parameter it replaces,
#: how to resolve it given the value, the resolved resolution-axis value and
#: the fold's probe).
SUBSTITUTES: dict[str, tuple[str, object]] = {
    "micro_ratio": ("max_micro_clusters", lambda ratio, k, probe: int(round(ratio * k))),
}

#: Validity constraints on the RESOLVED parameters. A grid point that fails is
#: not generated; the report says how many were rejected and why.
CONSTRAINTS: dict[str, list[tuple[str, object]]] = {
    "CluStream": [
        ("max_micro_clusters >= n_macro_clusters",
         lambda p: p["max_micro_clusters"] >= p["n_macro_clusters"]),
    ],
    # DenStream's `beta * mu > 1` (River computes log(beta*mu / (beta*mu - 1)) for
    # the pruning period, Eq. 4.1) is still a real constraint -- it is simply no
    # longer capable of rejecting anything, because beta is pinned at 0.75 and
    # the smallest mu in the ladder is 2, giving beta*mu = 1.5. It stays listed
    # so that overriding the grid via --grids cannot silently produce a
    # configuration River will divide by zero on.
    "DenStream": [
        ("beta * mu > 1", lambda p: p["beta"] * p["mu"] > 1),
    ],
}

#: Arms whose product is one set (or one policy) per algorithm, in report order.
FINAL_ARMS = ("default", "arm1_raw_all", "arm2_raw_lodo", "arm3_sf_lodo", "sf_all")
#: How pooled scores are formed when a set is chosen across datasets.
#:
#: ``mean`` averages raw ARI, and that is the weakness this pilot ran into: the
#: two folds with the largest ARI (kdd99, shuttle) are also the two with the
#: largest seed noise, so the pooled number is decided by the folds whose
#: measurements are least trustworthy. ``rank`` removes the scale but keeps
#: every dataset, including the ones where nothing in the grid works.
#: ``headroom`` normalises each dataset by its own reachable range before
#: pooling -- (arm - default) / (ceiling - default) -- and, because that
#: quantity is undefined where the ceiling is not above the default, it carries
#: the tunability gate with it.
#:
#: Nothing here is borrowed: Cerqueira et al. (KDD'26) pool by average rank but
#: only after designing F1/NDT to be comparable across streams in the first
#: place, and van Rijn & Hutter (KDD'18) refuse to pool at all, showing the
#: per-dataset distribution because "performance scales become a confounding
#: factor". ``headroom`` is this pipeline's version of the first idea.
AGGREGATES = ("mean", "rank", "headroom")

#: Tunability gate. A dataset is *tunable* for an algorithm when the grid can
#: reach something the default cannot: its ceiling clears `MIN_CEILING` **and**
#: beats the default by `MIN_CEILING_GAIN`. Both numbers are this project's
#: choice, not a published threshold, and the report quotes them wherever it
#: uses them. A dataset that fails carries no evidence for any arm, so leaving
#: it in a pooled score only adds noise -- measured here: DenStream/covtype has
#: a ceiling of 0.019, and CluStream clears neither bar on any fold.
#:
#: Runnable placeholders: `--min-ceiling` / `--min-ceiling-gain` override them.
#: The two answer different questions -- MIN_CEILING is a smallest effect of
#: practical interest, MIN_CEILING_GAIN stands in for "is the gain real", which
#: the noise columns of report 3b-i measure per dataset instead of assuming.
MIN_CEILING = 0.05
MIN_CEILING_GAIN = 0.02

#: Throughput floor for the constrained ceiling. Multi-objective HPO in the
#: Franceschi et al. monograph comes in two forms (Sec. 2.2.2, 8.1): optimise a
#: vector, or optimise one measure subject to a constraint on another. The
#: second is what this pipeline needs, because on DenStream the only knob that
#: raises ARI (epsilon down) also multiplies the micro-cluster count, and cost
#: per sample grows as P^1.2..P^1.6 -- so an unconstrained argmax over ARI walks
#: straight up the cost curve every time. A runnable placeholder, not a
#: recommendation: override with ``--min-throughput``.
MIN_THROUGHPUT_SPS = 1000.0

#: Regret above which a label-free criterion is called unusable on a dataset.
#: Regret is measured in ARI against what the same curve could have reached, so
#: the bar is in the same units as the arm-to-arm gaps the report prints. Also
#: this project's choice.
MAX_CRITERION_REGRET = 0.05
#: Suffix of the protocol holding a procedure's leave-one-dataset-out estimate.
EST = "_est"
ARM_LABEL = {
    "default": "default",
    "arm1_raw_all": "arm 1",
    "arm2_raw_lodo": "arm 2",
    "arm3_sf_lodo": "arm 3",
    "sf_all": "sf-all",
    "ceiling_raw": "per-dataset ceiling (raw)",
    "ceiling_sf": "per-dataset ceiling (substituted)",
    "irace_raw": "irace raw",
    "irace_ladder": "irace ladder",
}
#: How each `--aggregate` is named in the report.
POOL_LABEL = {
    "mean": "mean ARI",
    "rank": "mean per-dataset rank",
    "headroom": "mean headroom",
}


def arm_label(protocol: str) -> str:
    base = protocol[: -len(EST)] if protocol.endswith(EST) else protocol
    if base.startswith("p2r_irace_"):
        label = "irace P2R " + base[len("p2r_irace_"):]
    else:
        label = ARM_LABEL.get(base, base[len("p2r_"):] if base.startswith("p2r_") else base)
    return label + (" (LODO estimate)" if protocol.endswith(EST) else "")


#: P2R readings (rule:criterion) computed by default. `matr_trace_win` is defined
#: for every algorithm and is the pipeline's primary criterion. `R_max` on
#: `silhouette` is the contrast pre-registered in rules.py: silhouette is not
#: monotone in the cluster count, so R_max should not degenerate there.
P2R_DEFAULT = (
    "R_max:matr_trace_win,R_parsimony:matr_trace_win,R_knee:matr_trace_win,R_max:silhouette"
)
P2R_CRITERIA = ("matr_trace_win", "silhouette", "ssq")

#: Thresholds the verdicts use. Stated here so the report can quote them.
MIN_SPREAD_TO_NOISE = 3.0
MAX_ENDPOINT_SHARE = 0.20


# --- candidates ------------------------------------------------------------
#: Grid-spec keys that are {raw|ladder: ...} maps. ``fixed`` is not one of them:
#: it is a flat {param: value} dict shared by both grids, so it merges one level
#: shallower than the rest.
_PER_GRID_KEYS = ("levels", "shape")


def merge_grids(base: dict, override: dict | None) -> dict:
    """``--grids`` JSON merged over `GRIDS`.

    ``{algo: {levels|shape: {raw|ladder: [...]}}}`` merges two levels deep;
    ``{algo: {fixed: {param: value}}}`` one level, since pins are not per-grid.
    An override REPLACES the list or dict it names rather than extending it, so
    ``--grids '{"DenStream": {"shape": {"ladder": {"mu": [2, 5]}}}}'`` leaves the
    ladder sweeping ``mu`` alone.
    """
    def _copy(value):
        return dict(value) if isinstance(value, dict) else list(value)

    out = {}
    for algo, spec in base.items():
        out[algo] = {}
        for key, sub in spec.items():
            out[algo][key] = (
                {g: _copy(v) for g, v in sub.items()} if key in _PER_GRID_KEYS
                else _copy(sub)
            )
    for algo, spec in (override or {}).items():
        out.setdefault(algo, {"levels": {}, "shape": {}})
        for key, sub in spec.items():
            if key in _PER_GRID_KEYS:
                out[algo].setdefault(key, {})
                for grid, values in sub.items():
                    out[algo][key][grid] = values
            else:
                out[algo][key] = sub
    return out


def _candidate(algo, level, g, shape, *, grid, theta, id_level=None) -> dict:
    return {
        "algo": algo,
        "grid": grid,
        "granularity_level": level,
        "theta": theta,
        "g": g,
        "shape": dict(shape),
        "config_id": config_id(algo, level if id_level is None else id_level, g, shape),
        "scaler": SCALER,
        "scaler_params": dict(SCALER_PARAMS),
        "is_default": grid == "default",
        "raw": grid == "raw",
    }


def resolve_candidate(candidate: dict, probe: dict) -> dict:
    """Dimensionless spec -> absolute params for one fold, substitutes included.

    ``space.resolve`` sets the resolution axis (``epsilon = c * s(D)``, or the
    count as is). Substitutes in the shape are then replaced by the absolute
    parameter they stand for, computed from the resolved axis value.
    """
    params = resolve(candidate, probe)
    axis, _ = GRAN[candidate["algo"]]
    for key, (target, fn) in SUBSTITUTES.items():
        if key in params:
            params[target] = fn(params.pop(key), params[axis], probe)
    return params


def _violations(algo: str, params: dict) -> list[str]:
    return [name for name, ok in CONSTRAINTS.get(algo, []) if not ok(params)]


def shape_points(shape_spec: dict) -> list[dict]:
    """Cartesian product of the shape value sets, in a fixed key order."""
    keys = sorted(shape_spec)
    return [dict(zip(keys, values)) for values in itertools.product(*(shape_spec[k] for k in keys))]


def fixed_params(algo: str, kind: str, spec: dict) -> dict:
    """Every parameter of one grid that is not swept: River defaults plus pins.

    Every parameter outside the grid stays at River's constructor default. A
    substitute's target parameter is excluded from those defaults, otherwise
    the resolved value would be set twice. Shared with `irace_tune`, which must
    rebuild a grid point with exactly this shape so that it hashes to the same
    `config_id` and reuses the grid's runs.
    """
    axis, _ = GRAN[algo]
    shape_spec = spec["shape"][kind]
    pinned = dict(spec.get("fixed", {}))
    clash = sorted(set(pinned) & set(shape_spec))
    if clash:
        raise ValueError(f"{algo}: {clash} is both pinned and swept in the {kind} grid")
    targets = {SUBSTITUTES[k][0] for k in shape_spec if k in SUBSTITUTES}
    fixed = {
        k: v for k, v in river_defaults(algo).items()
        if k != axis and k not in shape_spec and k not in targets
    }
    # A pinned value overrides the inherited default and, unlike it, is named in
    # the report. Both land in the shape and therefore in `config_id`.
    fixed.update(pinned)
    return fixed


def build_grid(algo: str, kind: str, spec: dict) -> tuple[list[dict], list[dict]]:
    """All (level, theta) candidates of one grid, and the points rejected."""
    axis, _ = GRAN[algo]
    shape_spec = spec["shape"][kind]
    levels = spec["levels"][kind]
    fixed = fixed_params(algo, kind, spec)
    accepted, rejected = [], []
    for theta, point in enumerate(shape_points(shape_spec), start=1):
        for level, value in enumerate(levels, start=1):
            if kind == "raw":
                cand = _candidate(algo, level, value, {**fixed, **point, axis: value},
                                  grid="raw", theta=theta, id_level=f"raw{level}")
            else:
                cand = _candidate(algo, level, value, {**fixed, **point},
                                  grid="ladder", theta=theta)
            # Constraints never involve the fold scale, so a unit probe suffices.
            broken = _violations(algo, resolve_candidate(cand, {"s": 1.0}))
            if broken:
                rejected.append({"level": level, "theta": theta, "point": point,
                                 "value": value, "violates": broken})
            else:
                accepted.append(cand)
    return accepted, rejected


def default_candidate(algo: str) -> dict:
    return _candidate(algo, None, None, river_defaults(algo), grid="default", theta=None)


def tuned_params(algo: str, grids: dict) -> list[str]:
    """Names of the absolute parameters any arm tunes for ``algo``."""
    axis, _ = GRAN[algo]
    names = {axis}
    for kind in ("raw", "ladder"):
        for key in grids[algo]["shape"][kind]:
            names.add(SUBSTITUTES[key][0] if key in SUBSTITUTES else key)
    return sorted(names)


# --- running ---------------------------------------------------------------
def _params_blob(params: dict | None) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _key(algo, cid, fold, seed, requested, params) -> tuple:
    # The resolved params are part of the key. `config_id` names a position on
    # the dimensionless ladder, so after s(D) changes the same id resolves to a
    # different epsilon -- and a key without params would silently reuse a run
    # that measured something else.
    return (algo, cid, fold, seed, requested, _params_blob(params))


def _row_key(row: dict) -> tuple:
    return _key(row["algo"], row["config_id"], row["dataset"], row["seed"],
                row.get("requested"), row.get("params_resolved"))


def _job_key(job: dict) -> tuple:
    return _key(job["algo"], job["candidate"]["config_id"], job["fold"], job["seed"],
                job["requested"], job["params"])


def _fmt_value(value) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def describe(job: dict) -> str:
    """Readable label for a run: the tuned values and where they came from."""
    algo, cand = job["algo"], job["candidate"]
    axis, kind = GRAN[algo]
    params = job["params"]
    if cand["grid"] == "default":
        return f"{axis}={_fmt_value(params.get(axis))} [default]"
    tuned = job.get("tuned") or [axis]
    shown = ", ".join(f"{p}={_fmt_value(params.get(p))}" for p in tuned if p in params)
    if cand.get("irace"):
        # Off the grid (irace --irace-space continuous): no level or theta index.
        where = f"irace {cand['grid']}" + (
            f" c={cand['g']:.4g}" if cand["grid"] == "ladder" and kind == "scaled" else "")
    elif cand["grid"] == "raw":
        where = f"raw L{cand['granularity_level']} t{cand['theta']}"
    elif kind == "scaled":
        where = f"ladder L{cand['granularity_level']} t{cand['theta']} c={cand['g']:g}"
    else:
        where = f"ladder L{cand['granularity_level']} t{cand['theta']}"
    return f"{shown} [{where}]"


def pending(jobs: list[dict], rows: list[dict], retry_failed: bool = False) -> list[dict]:
    """Jobs with no usable row on disk, deduplicated.

    A failed run counts as done by default: a timeout is a fact about that
    configuration. ``retry_failed`` re-queues them, for when the limit rather
    than the configuration was at fault. The retry is appended, and the newest
    row for a key is the one every table reads.
    """
    latest = {}
    for row in rows:
        latest[_row_key(row)] = row.get("status")
    done = {k for k, status in latest.items() if status == "ok" or not retry_failed}
    todo, seen = [], set()
    for job in jobs:
        key = _job_key(job)
        if key not in done and key not in seen:
            seen.add(key)
            todo.append(job)
    return todo


def run_one(job: dict, timeout: int) -> dict:
    """Execute one job in its own subprocess and return the row to append.

    Module-level so that `irace_tune`'s target runner produces rows through the
    same door as `run_jobs` -- same hash seed, same keys, same failure rows.
    """
    # The child never sees `tuned`; it is a label-only field.
    row = execute({k: v for k, v in job.items() if k != "tuned"}, timeout)
    row["requested"] = job["requested"]
    # A failed row carries no params from the child; record what was asked.
    row.setdefault("params_resolved", job["params"])
    return row


def run_jobs(jobs: list[dict], workers: int, timeout: int,
             retry_failed: bool = False, dry_run: bool = False) -> None:
    """Run what is not on disk yet (and, with ``retry_failed``, what failed)."""
    rows = list(read_jsonl(RUNS_JSONL))
    failed = {_row_key(r) for r in rows if r.get("status") != "ok"}
    todo = pending(jobs, rows, retry_failed)
    n_retry = sum(_job_key(j) in failed for j in todo)
    print(f"  {len(todo)} runs to do ({n_retry} retries of failed runs; "
          f"{len(jobs) - len(todo)} already on disk or duplicate)")
    if dry_run:
        for job in todo:
            tag = " (retry)" if _job_key(job) in failed else ""
            print(f"    would run: {job['algo']} {job['fold']} seed={job['seed']} "
                  f"{describe(job)}{tag}")
        return
    if not todo:
        return

    def one(job):
        return run_one(job, timeout)

    started, counts = time.time(), {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, job): job for job in todo}
        # Written in completion order, not submission order: a slow run at the
        # head of the queue must not hold finished rows in memory, where an
        # interrupt would lose them and the log would show no progress.
        for i, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            row = future.result()
            append_jsonl(RUNS_JSONL, row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            tag = " (retry)" if _job_key(job) in failed else ""
            print(
                f"  {i}/{len(todo)}  {time.time() - started:.0f}s  "
                f"{row['algo']} {row['dataset']} seed={row['seed']} {describe(job)}{tag} "
                f"{row['status']} ({row.get('runtime_s', '-')}s)  "
                + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            )


def _job(algo, candidate, fold, seed, n_samples, probes, tuned=()) -> dict:
    return {
        "algo": algo,
        "candidate": candidate,
        "fold": fold,
        "seed": seed,
        "budget": n_samples or None,
        "requested": n_samples or "full",
        "params": resolve_candidate(candidate, probes[fold]),
        "tuned": list(tuned),
    }


def load_runs(requested, grid_of: dict, candidates: dict, probes: dict) -> pd.DataFrame:
    """Current runs at this stream length, tagged with their grid.

    A row counts only if its params are what its config resolves to *now*;
    rows measured under an older s(D) stay in the file but out of every table.
    """
    rows = []
    for r in read_jsonl(RUNS_JSONL):
        cand = candidates.get((r["algo"], r["config_id"]))
        if r.get("requested") != requested or cand is None or r["dataset"] not in probes:
            continue
        if _params_blob(r.get("params_resolved")) == _params_blob(
            resolve_candidate(cand, probes[r["dataset"]])
        ):
            rows.append(r)
    if not rows:
        raise SystemExit(f"no current runs for n_samples={requested} in {RUNS_JSONL}")
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["algo", "config_id", "dataset", "seed"], keep="last")
    df["grid"] = [grid_of.get((a, c)) for a, c in zip(df.algo, df.config_id)]
    return df[df.grid.notna()].reset_index(drop=True)


# --- selection (pure: no runs, no files) -----------------------------------
def seed_mean(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the tuning seeds of one cell into a single averaged row.

    Every selection helper below -- ``_curve``, ``_ari_at``, ``_shape_scores``,
    the argmax in ``_first`` -- is written against ONE row per
    (algo, config_id, dataset). Handing them several seeds per cell does not
    average anything: ``_curve`` would reindex on duplicate levels and raise,
    ``_ari_at`` would silently return whichever seed happened to land first, and
    ``_shape_scores`` would take its top-k across seeds as well as across rungs.

    So the averaging happens once, here, at the boundary between "runs on disk"
    and "selection", and the rest of the module keeps its one-row invariant.
    This mirrors ``views._mbar`` in the main pipeline.

    A cell is ``ok`` only if *every* tuning seed of it is. A config that fails on
    one seed out of five is not a config that scored slightly lower; it is one
    that cannot be relied on, and letting the surviving seeds average it into
    contention would hide exactly that.
    """
    if frame.empty:
        return frame
    key = ["algo", "config_id", "dataset"]
    measured = [
        c for c in frame.columns
        if c not in key and c != "seed" and pd.api.types.is_numeric_dtype(frame[c])
    ]
    grouped = frame.groupby(key, dropna=False)
    out = grouped[measured].mean().reset_index()

    # Identity travels from the first row: these describe the cell, and are the
    # same on every seed of it by construction.
    identity = [c for c in frame.columns if c not in measured and c not in ("seed", "status")]
    out = out.merge(frame.drop_duplicates(subset=key, keep="first")[identity],
                    on=key, how="left")

    status = grouped["status"].agg(
        lambda col: "ok" if set(col) == {"ok"} else ",".join(sorted(set(col) - {"ok"}))
    ).rename("status").reset_index()
    out = out.merge(status, on=key, how="left")
    out = out.merge(grouped.size().rename("n_tune_seeds").reset_index(), on=key, how="left")
    return out


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.status == "ok") & df[OBJ].notna()]


def _first(frame: pd.DataFrame, score: str, runtime: str) -> str:
    """Deterministic argmax: score desc, runtime asc, coarser level, then id."""
    ranked = frame.sort_values(
        [score, runtime, "granularity_level", "config_id"],
        ascending=[False, True, True, True], kind="mergesort",
    )
    return str(ranked.iloc[0]["config_id"])


def fold_reference(sub: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: what the default scores, what the whole grid can reach, and
    whether the gap between them is big enough to tune on.

    ``sub`` is one algorithm's tuning rows across **every** grid *and* its
    default row -- not a grid-restricted slice. The ceiling has to be the best
    the whole search space holds, and the default has to be in the same frame,
    or an arm on the raw grid would be normalised against a different yardstick
    from one on the substituted grid. ``main`` runs the default under the tuning
    seeds for exactly this reason. The ceiling is the
    per-dataset oracle -- the best the search space holds when it is allowed to
    look at that dataset's labels -- so ``headroom`` measures an arm against
    what was actually reachable rather than against zero.

    Index: dataset. Columns: ``default``, ``ceiling``, ``gain``, ``tunable``.
    """
    ok = _ok(sub)
    cols = ["default", "ceiling", "gain", "tunable"]
    if ok.empty:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="dataset"))
    ceiling = ok.groupby("dataset")[OBJ].max()
    # `is_default` is missing from a frame assembled without the default row --
    # an older runs.jsonl, or a caller that only has the grid. The reference is
    # then ceiling-only and nothing is tunable, which is the correct answer:
    # without a baseline there is no gain to measure.
    flag = (ok["is_default"].fillna(False).astype(bool) if "is_default" in ok
            else pd.Series(False, index=ok.index))
    defaults = ok[flag].groupby("dataset")[OBJ].max()
    out = pd.DataFrame({"ceiling": ceiling})
    out["default"] = defaults.reindex(out.index)
    out["gain"] = out["ceiling"] - out["default"]
    out["tunable"] = (out["ceiling"] >= MIN_CEILING) & (out["gain"] >= MIN_CEILING_GAIN)
    out["tunable"] = out["tunable"].fillna(False)
    out.index.name = "dataset"
    return out[cols]


def tunable_folds(sub: pd.DataFrame, folds: list[str]) -> list[str]:
    """``folds``, minus the ones whose grid holds no reachable gain."""
    ref = fold_reference(sub)
    return [f for f in folds if bool(ref["tunable"].get(f, False))]


def _headroom(sub: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """(ari - default) / (ceiling - default), per row, on tunable datasets."""
    base = sub.dataset.map(ref["default"])
    span = sub.dataset.map(ref["gain"])
    keep = sub.dataset.map(ref["tunable"]).fillna(False).astype(bool)
    with np.errstate(all="ignore"):
        value = (sub[OBJ] - base) / span
    return value.where(keep & (span > 0))


def _aggregate(sub: pd.DataFrame, folds_used: list[str], how: str = "mean",
               ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pool one algorithm's tuning scores over ``folds_used``; bigger is better.

    Only configs that ran on every one of those folds are eligible: a set that
    cannot run on a dataset is not a set that can be shipped to it.

    ``mean`` averages ARI. ``rank`` averages each config's rank within every
    dataset (negated, so bigger is still better): a dataset where ARI runs to
    0.5 then counts exactly as much as one where it tops out at 0.05.
    ``headroom`` scores each dataset as the share of its own reachable range the
    config captured, and drops the datasets where that range is empty -- so an
    arm reaching 80% of what insects allows counts as much as one reaching 80%
    of what kdd99 allows, and covtype stops voting at all.

    The reference values ``headroom`` needs (each dataset's default and ceiling)
    come from ``sub`` *before* it is narrowed to ``folds_used``, so a LODO round
    measures the held-out-from folds against the same yardstick every other
    round uses.
    """
    if how not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {how!r}")
    if how == "headroom" and ref is None:
        # Fallback for callers holding only this grid's slice; the report and
        # `all_picks` always pass the whole-algorithm reference instead.
        ref = fold_reference(sub)
    sub = _ok(sub)
    if how == "headroom":
        folds_used = [f for f in folds_used if bool(ref["tunable"].get(f, False))]
        if not folds_used:
            return pd.DataFrame(columns=["config_id", "score", "runtime", "granularity_level"])
    sub = sub[sub.dataset.isin(folds_used)]
    n = sub.groupby("config_id").dataset.nunique()
    sub = sub[sub.config_id.isin(n[n == len(folds_used)].index)]
    if sub.empty:
        return pd.DataFrame(columns=["config_id", "score", "runtime", "granularity_level"])
    if how == "rank":
        score = -sub.groupby("dataset")[OBJ].rank(ascending=False, method="average")
    elif how == "headroom":
        score = _headroom(sub, ref)
    else:
        score = sub[OBJ]
    sub = sub.assign(_score=score)
    return (
        sub.groupby("config_id")
        .agg(score=("_score", "mean"), runtime=("runtime_s", "mean"),
             granularity_level=("granularity_level", "first"))
        .reset_index()
    )


def pick_oracle(tune: pd.DataFrame) -> dict:
    """(algo, fold) -> config chosen on that fold's own labels. A ceiling only."""
    return {
        (algo, fold): _first(grp, OBJ, "runtime_s")
        for (algo, fold), grp in _ok(tune).groupby(["algo", "dataset"])
    }


def pick_all(tune: pd.DataFrame, folds: list[str], how: str = "mean",
             refs: dict | None = None) -> dict:
    """algo -> the one config with the best pooled score over all ``folds``."""
    picks = {}
    for algo in sorted(set(tune.algo)):
        agg = _aggregate(tune[tune.algo == algo], folds, how, (refs or {}).get(algo))
        if not agg.empty:
            picks[algo] = _first(agg, "score", "runtime")
    return picks


def _opt_folds(fold: str, folds: list[str], groups: dict | None, cross: bool) -> list[str]:
    if groups is None:
        return [f for f in folds if f != fold]
    if cross:
        return [f for f in folds if groups[f] != groups[fold]]
    return [f for f in folds if f != fold and groups[f] == groups[fold]]


def pick_lodo(tune: pd.DataFrame, folds: list[str], groups: dict | None = None,
              cross: bool = False, how: str = "mean", refs: dict | None = None) -> dict:
    """(algo, held-out fold) -> config chosen on the OTHER folds.

    These per-round picks are what the LODO *estimate* scores; the shipped set
    is their consensus (``consensus``).

    ``groups`` (fold -> group) keeps the optimisation set inside the held-out
    fold's own group, so folds cut from one generator cannot outvote the real
    ones. ``cross=True`` instead takes it from the other groups only. Without
    ``groups`` every other fold is used. The held-out fold is never consulted,
    so a pick that then fails there is a real failure.
    """
    picks = {}
    for algo in sorted(set(tune.algo)):
        sub = tune[tune.algo == algo]
        for fold in folds:
            opt = _opt_folds(fold, folds, groups, cross)
            if not opt:
                continue
            agg = _aggregate(sub, opt, how, (refs or {}).get(algo))
            if not agg.empty:
                picks[(algo, fold)] = _first(agg, "score", "runtime")
    return picks


def consensus(tune: pd.DataFrame, lodo: dict, folds: list[str], how: str = "mean",
              refs: dict | None = None) -> dict:
    """algo -> (config, votes, rounds): the set most LODO rounds picked.

    Ties go to the tied config with the best pooled score over all folds, so
    the result is deterministic and never a config no round chose.
    """
    out = {}
    for algo in sorted({a for a, _ in lodo}):
        votes: dict[str, int] = {}
        for (a, _), cid in sorted(lodo.items()):
            if a == algo:
                votes[cid] = votes.get(cid, 0) + 1
        top = max(votes.values())
        tied = sorted(c for c, v in votes.items() if v == top)
        cid = tied[0]
        if len(tied) > 1:
            agg = _aggregate(tune[tune.algo == algo], folds, how, (refs or {}).get(algo))
            agg = agg[agg.config_id.isin(tied)]
            if not agg.empty:
                cid = _first(agg, "score", "runtime")
        out[algo] = (cid, top, sum(votes.values()))
    return out


def _curve(tune: pd.DataFrame, algo: str, theta, fold: str, criterion: str, n_levels: int):
    """A label-free criterion along the ladder for one shape point on one fold.

    Bigger-is-better. None when any rung is missing or not finite: a timed-out
    rung, or River's streaming silhouette returning inf, breaks the curve
    rather than shortening it -- a rule read off a gappy curve would pick from
    the wrong positions.
    """
    sub = tune[(tune.algo == algo) & (tune.dataset == fold) & (tune.status == "ok")
               & (tune.theta == theta)]
    if criterion not in sub.columns or sub.empty:
        return None
    levels = sub.granularity_level.astype(int)
    series = pd.Series(pd.to_numeric(sub[criterion], errors="coerce").to_numpy(),
                       index=levels.to_numpy()).reindex(range(1, n_levels + 1))
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    return orient(values, criterion)


def criterion_regret(tune: pd.DataFrame, algo: str, fold: str, criterion: str,
                     n_levels: int) -> dict:
    """How much ARI a label-free criterion gives away on one (algo, dataset).

    P2R does not use a criterion to rank configurations in general; it uses it
    to read **one curve** -- the resolution axis at a fixed shape -- and pick a
    rung. So the question that matters is not "does this criterion correlate
    with ARI over the grid", it is "on each curve the rule will actually be
    handed, how far from that curve's best is the rung the criterion points
    at". That is the regret measured here, in ARI, at every shape point:

        regret(theta) = max_level ARI(theta, level)
                        - ARI(theta, argmax_level criterion(theta, level))

    A criterion can correlate positively overall and still be useless, because
    the curve it has to read is the one place the correlation has to hold.

    Returns ``{median, max, n_curves, n_negative}``, where ``n_negative`` counts
    the curves on which the criterion ranks *against* ARI (Spearman < 0).
    """
    regrets, negative = [], 0
    thetas = sorted(set(tune[(tune.algo == algo) & (tune.dataset == fold)].theta.dropna()))
    for theta in thetas:
        curve = _curve(tune, algo, theta, fold, criterion, n_levels)
        if curve is None:
            continue
        aris = [_ari_at(tune, algo, theta, fold, lvl + 1) for lvl in range(n_levels)]
        if not np.all(np.isfinite(aris)):
            continue
        pick = int(np.argmax(curve))
        regrets.append(float(np.max(aris) - aris[pick]))
        if len(set(curve)) > 1 and len(set(aris)) > 1:
            rho = pd.Series(curve).corr(pd.Series(aris), method="spearman")
            negative += int(np.isfinite(rho) and rho < 0)
    if not regrets:
        return {"median": np.nan, "max": np.nan, "n_curves": 0, "n_negative": 0}
    return {"median": float(np.median(regrets)), "max": float(np.max(regrets)),
            "n_curves": len(regrets), "n_negative": negative}


def _ari_at(tune: pd.DataFrame, algo: str, theta, fold: str, level: int) -> float:
    sub = _ok(tune[(tune.algo == algo) & (tune.dataset == fold) & (tune.theta == theta)])
    hit = sub[sub.granularity_level.astype(int) == level][OBJ]
    return float(hit.iloc[0]) if len(hit) else np.nan


#: How many of a shape point's levels its profile score averages. A plain max
#: over the ladder is the maximum of 8 estimates, so it carries a winner's curse
#: even once each estimate is a mean over `TUNE_SEEDS`: averaging the seeds
#: shrinks each rung's error bar, it does not stop the argmax from preferring
#: whichever rung's residual error points up. Health check 4c measures what that
#: cost at one seed -- DenStream's mean(tune - re-run) was 0.015 with a max of
#: 0.061, the same size as the gaps between the arms. Averaging the best two
#: rungs keeps the profile (each shape point is still judged at the resolution
#: that suits it) while halving what is left of the bias. Two rather than three
#: because the ladder is only 8 rungs long and the peak is often narrow.
PROFILE_TOP_K = 2


def _shape_scores(tune: pd.DataFrame, algo: str, folds_used: list[str],
                  how: str = "mean", ref: pd.DataFrame | None = None) -> dict:
    """theta -> pooled top-``PROFILE_TOP_K``-levels score (bigger is better).

    A shape point is judged by the levels it reaches on each dataset, since P2R
    re-picks the level anyway. Datasets where a shape point never ran are
    skipped for that point, not counted as zero.

    The profile is the *mean of the best ``PROFILE_TOP_K`` rungs*, not the max;
    see that constant for why. With fewer rungs available the mean is taken over
    what there is, so a short curve is not silently penalised.
    """
    ok = _ok(tune[(tune.algo == algo) & tune.dataset.isin(folds_used)])
    if ok.empty:
        return {}
    best = (
        ok.groupby(["theta", "dataset"])[OBJ]
        .apply(lambda s: s.nlargest(min(PROFILE_TOP_K, len(s))).mean())
        .unstack("dataset")
    )
    if how == "rank":
        best = -best.rank(axis=0, ascending=False, method="average")
    elif how == "headroom" and ref is not None:
        keep = [d for d in best.columns if bool(ref["tunable"].get(d, False))]
        best = best[keep]
        if best.empty:
            return {}
        with np.errstate(all="ignore"):
            best = (best - ref["default"].reindex(best.columns)) / \
                ref["gain"].reindex(best.columns)
    pooled = best.mean(axis=1, skipna=True)
    return {int(t): float(v) for t, v in pooled.items() if np.isfinite(v)}


def _rule_quality(tune, algo, theta, opt, rule, phi, criterion, n_levels) -> float:
    """Mean ARI the rule reaches on the optimisation folds with this phi."""
    vals = []
    for fold in opt:
        curve = _curve(tune, algo, theta, fold, criterion, n_levels)
        if curve is not None:
            vals.append(_ari_at(tune, algo, theta, fold, rule(curve, phi) + 1))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else -np.inf


def pick_p2r(tune: pd.DataFrame, folds: list[str], groups: dict | None,
             rule_name: str, criterion: str, n_levels: dict | int,
             how: str = "mean", refs: dict | None = None) -> dict:
    """P2R's LODO rounds: (algo, held-out fold) -> (theta, level).

    In each round the shape point ``theta*`` is chosen on the optimisation
    folds (pooled best-over-level score), and the level is read off the
    held-out fold's own label-free curve at ``theta*``. The rule's phi is
    fitted on the optimisation folds too, so nothing about the held-out fold's
    labels is read. These picks are what P2R's LODO estimate scores.

    The held-out labels are never read, but a rule can only be applied to a
    *complete* curve: a shape point whose held-out curve has a failed rung (a
    timeout is label-free information) yields to the next best one.
    """
    rule, phis = RULES[rule_name]
    picks = {}
    for algo in sorted(set(tune.algo)):
        levels = n_levels[algo] if isinstance(n_levels, dict) else n_levels
        for fold in folds:
            opt = _opt_folds(fold, folds, groups, cross=False)
            if not opt:
                continue
            quality = _shape_scores(tune, algo, opt, how, (refs or {}).get(algo))
            # Best first; ties keep the smaller theta, so the order is fixed.
            ranked = sorted(quality, key=lambda th: (-quality[th], th))
            theta_star, curve = None, None
            for theta in ranked:
                curve = _curve(tune, algo, theta, fold, criterion, levels)
                if curve is not None:
                    theta_star = theta
                    break
            if theta_star is None:
                continue
            phi = max(phis, key=lambda p: _rule_quality(tune, algo, theta_star, opt, rule, p,
                                                        criterion, levels))
            picks[(algo, fold)] = (theta_star, rule(curve, phi) + 1)
    return picks


def p2r_policy(tune: pd.DataFrame, folds: list[str], rule_name: str, criterion: str,
               n_levels: dict | int, est: dict, how: str = "mean",
               refs: dict | None = None) -> dict:
    """algo -> the one P2R policy shipped: a fixed shape, a fixed phi, a rule.

    ``theta`` is the shape most LODO rounds chose (ties: best pooled over all
    folds); ``phi`` is fitted on all folds at that shape. The level is then
    read off each dataset's own label-free curve -- that is the point of P2R,
    so the resolved threshold differs per dataset while the policy does not.
    A dataset whose curve is broken at ``theta`` gets no level.
    """
    rule, phis = RULES[rule_name]
    out = {}
    for algo in sorted(set(tune.algo)):
        votes: dict[int, int] = {}
        for (a, _), (theta, _) in sorted(est.items()):
            if a == algo:
                votes[theta] = votes.get(theta, 0) + 1
        if not votes:
            continue
        levels_n = n_levels[algo] if isinstance(n_levels, dict) else n_levels
        top = max(votes.values())
        tied = sorted(t for t, v in votes.items() if v == top)
        quality = _shape_scores(tune, algo, folds, how, (refs or {}).get(algo))
        theta = max(tied, key=lambda t: (quality.get(t, -np.inf), -t))
        phi = max(phis, key=lambda p: _rule_quality(tune, algo, theta, folds, rule, p,
                                                    criterion, levels_n))
        chosen = {}
        for fold in folds:
            curve = _curve(tune, algo, theta, fold, criterion, levels_n)
            if curve is not None:
                chosen[fold] = rule(curve, phi) + 1
        out[algo] = {"theta": theta, "phi": phi, "votes": top,
                     "rounds": sum(votes.values()), "levels": chosen}
    return out


def parse_p2r(spec: str) -> list[tuple[str, str]]:
    """'R_max:matr_trace_win,R_knee:silhouette' -> [(rule, criterion), ...]."""
    out = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        rule, _, criterion = item.partition(":")
        if rule not in RULES or criterion not in P2R_CRITERIA:
            raise SystemExit(
                f"bad --p2r entry {item!r}; rules: {sorted(RULES)}, criteria: {list(P2R_CRITERIA)}"
            )
        out.append((rule, criterion))
    return out


def all_picks(tune, grids, defaults, algos, folds, groups, p2r=(),
              aggregate: str = "mean", refs: dict | None = None) -> tuple[dict, dict]:
    """Every arm's picks, plus the one set each arm ships.

    Returns ``(picks, finals)``:

    * ``picks``: protocol -> {(algo, fold): candidate}. For a final arm the
      candidate is the same set on every fold (for P2R, the same shape with the
      level its rule read on that fold). ``<arm>_est`` protocols hold the LODO
      rounds; ``ceiling_*`` the per-dataset oracles, kept as reference.
    * ``finals``: protocol -> {algo: {"config": candidate, ...}} -- what the
      arm hands over, with how it was reached.
    """
    # One reference per algorithm, built from the WHOLE tuning frame (both
    # grids plus the default row) before anything is sliced, so every arm and
    # every LODO round is normalised against the same per-dataset yardstick.
    if refs is None:
        refs = {a: fold_reference(tune[tune.algo == a]) for a in algos}
    by_id = {(c["algo"], c["config_id"]): c
             for kind in grids for cands in grids[kind].values() for c in cands}
    by_cell = {(c["algo"], c["theta"], c["granularity_level"]): c
               for a in algos for c in grids["ladder"][a]}
    raw, ladder = tune[tune.grid == "raw"], tune[tune.grid == "ladder"]
    picks: dict[str, dict] = {}
    finals: dict[str, dict] = {}

    def ship(protocol, chosen: dict, extra: dict | None = None):
        picks[protocol] = {(a, f): by_id[(a, cid)] for a, cid in chosen.items() for f in folds}
        finals[protocol] = {a: {"config": by_id[(a, cid)], **((extra or {}).get(a, {}))}
                            for a, cid in chosen.items()}

    picks["default"] = {(a, f): defaults[a] for a in algos for f in folds}
    finals["default"] = {a: {"config": defaults[a]} for a in algos}

    ship("arm1_raw_all", pick_all(raw, folds, aggregate, refs))
    ship("sf_all", pick_all(ladder, folds, aggregate, refs))

    for protocol, sub in (("ceiling_raw", raw), ("ceiling_sf", ladder)):
        picks[protocol] = {k: by_id[(k[0], cid)] for k, cid in pick_oracle(sub).items()}

    for protocol, sub in (("arm2_raw_lodo", raw), ("arm3_sf_lodo", ladder)):
        est = pick_lodo(sub, folds, groups, how=aggregate, refs=refs)
        picks[protocol + EST] = {k: by_id[(k[0], cid)] for k, cid in est.items()}
        agreed = consensus(sub, est, folds, aggregate, refs)
        ship(protocol, {a: v[0] for a, v in agreed.items()},
             {a: {"votes": v[1], "rounds": v[2]} for a, v in agreed.items()})

    if len(set(groups.values())) >= 2:
        cross = pick_lodo(ladder, folds, groups, cross=True, how=aggregate, refs=refs)
        picks["arm3_sf_cross" + EST] = {k: by_id[(k[0], cid)] for k, cid in cross.items()}

    n_levels = {a: len(set(c["granularity_level"] for c in grids["ladder"][a])) for a in algos}
    for rule_name, criterion in p2r:
        protocol = f"p2r_{rule_name}_{criterion}"
        est = pick_p2r(ladder, folds, groups, rule_name, criterion, n_levels, aggregate, refs)
        picks[protocol + EST] = {
            k: by_cell[(k[0], theta, level)] for k, (theta, level) in est.items()
            if (k[0], theta, level) in by_cell
        }
        policy = p2r_policy(ladder, folds, rule_name, criterion, n_levels, est, aggregate, refs)
        picks[protocol] = {
            (a, f): by_cell[(a, pol["theta"], level)]
            for a, pol in policy.items() for f, level in pol["levels"].items()
            if (a, pol["theta"], level) in by_cell
        }
        finals[protocol] = {
            a: {"config": by_cell[(a, pol["theta"], 1)], "rule": rule_name,
                "criterion": criterion, **pol}
            for a, pol in policy.items() if (a, pol["theta"], 1) in by_cell
        }
    return picks, finals


# --- irace arms (irace_tune) ---------------------------------------------------
#: The irace arms that ship a plain set; the P2R one is named ``p2r_irace_*`` so
#: that every P2R-specific table picks it up with the grid P2R readings.
IRACE_SHIPS = ("irace_raw", "irace_ladder")


def irace_protocol(arm: str, rule: str | None = None, criterion: str | None = None) -> str:
    return f"p2r_irace_{rule}_{criterion}" if arm == "p2r" else f"irace_{arm}"


def irace_rounds(folds: list[str], groups: dict, tunable) -> dict[str, list[str]]:
    """Round -> the datasets its race uses as instances.

    ``all`` is the race whose elite is shipped. ``lodo-<f>`` is the race whose
    elite is scored on ``f`` for the LODO estimate; it never sees ``f``. Only
    tunable datasets are instances, so a held-out dataset that failed the gate
    races on the same datasets as ``all`` -- and reuses that race.
    """
    keep = set(tunable)
    rounds = {"all": [f for f in folds if f in keep]}
    for fold in folds:
        rounds[f"lodo-{fold}"] = [f for f in _opt_folds(fold, folds, groups, False) if f in keep]
    return rounds


def irace_results(*, algos, arms, space, p2r_reading, folds, groups, refs, probes,
                  grids_spec, tuned_of, n_samples, timeout, workers, rscript, settings,
                  run: bool) -> dict:
    """(algo, protocol) -> {round: race result, or None where nothing could race}.

    One race per distinct optimisation set. ``run=False`` only reads finished
    races whose inputs still match (the score and report phases).
    """
    from . import irace_tune

    requested = n_samples or "full"
    out = {}
    for algo in algos:
        ref = refs[algo]
        # gain > 0 as well: a gate lowered to zero must not hand irace a
        # dataset whose headroom is 0/0.
        tunable = [f for f in folds if bool(ref["tunable"].get(f, False))
                   and float(ref["gain"].get(f, 0.0)) > 0]
        if not tunable:
            print(f"irace: {algo} passes the tunability gate on no dataset -- nothing to race")
            continue
        reference = {f: {"default": float(ref.loc[f, "default"]),
                         "gain": float(ref.loc[f, "gain"])} for f in tunable}
        rounds = irace_rounds(folds, groups, tunable)
        for arm in arms:
            rule, criterion = p2r_reading if arm == "p2r" else (None, None)
            protocol = irace_protocol(arm, rule, criterion)
            races: dict[tuple, dict | None] = {}
            per_round = {}
            for name, opt in rounds.items():
                if not opt:
                    per_round[name] = None
                    continue
                key = tuple(opt)
                if key not in races:
                    tag = "all" if key == tuple(rounds["all"]) else "opt-" + "+".join(opt)
                    races[key] = irace_tune.race(
                        algo=algo, arm=arm, space=space, rule=rule, criterion=criterion,
                        opt_folds=list(opt), grids_spec=grids_spec, reference=reference,
                        probes=probes, tuned=tuned_of[algo], n_samples=n_samples,
                        timeout=timeout, workers=workers, out_dir=OUT_DIR, rscript=rscript,
                        run_dir=OUT_DIR / "irace" / str(requested) / space / algo / protocol / tag,
                        run=run, **settings)
                    if races[key] is None:
                        print(f"  irace {algo} {arm_label(protocol)} on {', '.join(opt)}: no "
                              "finished race with the current inputs -- run --phase irace")
                per_round[name] = races[key]
            out[(algo, protocol)] = per_round
    return out


def irace_cells(results: dict, grids_spec: dict, space: str) -> dict:
    """(algo, protocol) -> the candidates each race's elite stands for.

    ``{"ship": [cand, ...] | None, "ship_values": {...}, "est": {fold: [...]},
    "est_values": {fold: {...}}}`` -- one candidate for a plain set, the whole
    ladder at the elite's shape for P2R.
    """
    from . import irace_tune

    out = {}
    for (algo, protocol), per_round in results.items():
        arm = "p2r" if protocol.startswith("p2r_") else protocol[len("irace_"):]
        entry = {"ship": None, "ship_values": None, "est": {}, "est_values": {}}
        ship = per_round.get("all")
        if ship and ship.get("best"):
            entry["ship"] = irace_tune.candidates_for(algo, arm, space, grids_spec, ship["best"])
            entry["ship_values"] = ship["best"]
        for name, res in sorted(per_round.items()):
            if name.startswith("lodo-") and res and res.get("best"):
                fold = name[len("lodo-"):]
                entry["est"][fold] = irace_tune.candidates_for(algo, arm, space, grids_spec,
                                                               res["best"])
                entry["est_values"][fold] = res["best"]
        out[(algo, protocol)] = entry
    return out


def irace_ladder_jobs(cells: dict, folds: list[str], n_samples, probes, tuned_of) -> list[dict]:
    """The tuning-seed ladder runs P2R needs to read a level on every dataset.

    On the grid they are already on disk; off the grid (continuous space) the
    shipped shape has never run on a dataset that was not an instance, and a
    LODO shape never on its held-out dataset.
    """
    jobs = []
    for (algo, protocol), entry in cells.items():
        if not protocol.startswith("p2r_"):
            continue
        wanted = []
        if entry["ship"]:
            wanted += [(entry["ship"], f) for f in folds]
        wanted += [(ladder, f) for f, ladder in entry["est"].items()]
        for ladder, fold in wanted:
            for cand in ladder:
                for seed in TUNE_SEEDS:
                    jobs.append(_job(algo, cand, fold, seed, n_samples, probes, tuned_of[algo]))
    return jobs


def _ladder_level(df: pd.DataFrame, ladder: list[dict], fold: str, criterion: str,
                  rule_name: str, phi) -> int | None:
    """The level a P2R rule reads on ``fold``'s own label-free curve, or None.

    The curve is the criterion averaged over `TUNE_SEEDS`, as `seed_mean` does
    for the grid P2R; a rung that failed on any tuning seed breaks the curve.
    """
    rule, _ = RULES[rule_name]
    values = []
    for cand in ladder:
        hit = df[(df.algo == cand["algo"]) & (df.config_id == cand["config_id"])
                 & (df.dataset == fold) & df.seed.isin(TUNE_SEEDS)]
        if (hit.seed.nunique() != len(TUNE_SEEDS) or criterion not in hit
                or not (hit.status == "ok").all()):
            return None
        values.append(float(pd.to_numeric(hit[criterion], errors="coerce").mean()))
    curve = np.asarray(values, dtype=float)
    if not np.isfinite(curve).all():
        return None
    return int(rule(orient(curve, criterion), phi)) + 1


def _irace_summary(result: dict | None) -> dict:
    if not result or not result.get("elites"):
        return {}
    top = result["elites"][0]
    return {
        "elite_instances": top["n_instances"],
        "elite_mean_headroom": top["mean_headroom"],
        "target_calls": result.get("target_calls"),
        "runs_executed": result.get("runs_executed"),
        "n_experiments": result.get("n_experiments"),
        "n_configurations": result.get("n_configurations"),
        "budget": result.get("budget"),
        "opt_folds": result.get("opt_folds"),
        "seeds": result.get("seeds"),
        "test": result.get("test"),
    }


def irace_picks(cells: dict, results: dict, df: pd.DataFrame, folds: list[str],
                p2r_reading) -> tuple[dict, dict]:
    """Picks and finals for the irace arms, in the shape `all_picks` returns.

    The shipped set is the elite of the ``all`` race; the LODO estimate scores
    each round's elite on the dataset that round left out. A P2R elite is a
    shape (and phi): its level is read per dataset off the tuning-seed curve,
    exactly as the grid P2R does, and a dataset with a broken curve gets none.
    """
    picks: dict[str, dict] = {}
    finals: dict[str, dict] = {}
    for (algo, protocol), entry in cells.items():
        summary = _irace_summary(results[(algo, protocol)].get("all"))
        if protocol.startswith("p2r_"):
            rule, criterion = p2r_reading
            if entry["ship"]:
                phi = entry["ship_values"].get("phi")
                levels = {}
                for fold in folds:
                    level = _ladder_level(df, entry["ship"], fold, criterion, rule, phi)
                    if level is not None:
                        levels[fold] = level
                picks.setdefault(protocol, {}).update(
                    {(algo, f): entry["ship"][lv - 1] for f, lv in levels.items()})
                finals.setdefault(protocol, {})[algo] = {
                    "config": entry["ship"][0], "rule": rule, "criterion": criterion,
                    "theta": entry["ship"][0]["theta"], "phi": phi, "votes": None,
                    "rounds": None, "levels": levels, "irace": summary,
                }
            for fold, ladder in entry["est"].items():
                phi = entry["est_values"][fold].get("phi")
                level = _ladder_level(df, ladder, fold, criterion, rule, phi)
                if level is not None:
                    picks.setdefault(protocol + EST, {})[(algo, fold)] = ladder[level - 1]
            continue
        if entry["ship"]:
            cand = entry["ship"][0]
            picks.setdefault(protocol, {}).update({(algo, f): cand for f in folds})
            finals.setdefault(protocol, {})[algo] = {"config": cand, "irace": summary}
        for fold, cands in entry["est"].items():
            picks.setdefault(protocol + EST, {})[(algo, fold)] = cands[0]
    return picks, finals


def spec_label(cand: dict, grids_spec: dict, tuned: list[str], *, level: bool = True) -> str:
    """The set as a person would write it down, in the grid's own units.

    A substituted set is written with its substitutes (``epsilon = 1.6 * s(D)``,
    ``micro_ratio=5``) because that, not a per-dataset number, is what it is.
    ``level=False`` leaves the resolution axis out (P2R reads it per dataset).
    """
    algo = cand["algo"]
    axis, kind = GRAN[algo]
    shape = cand["shape"]
    if cand["grid"] == "default":
        return ", ".join(f"{p}={_fmt_value(shape[p])}" for p in tuned if p in shape)
    parts = []
    if level:
        if cand["grid"] == "ladder" and kind == "scaled":
            parts.append(f"{axis} = {cand['g']:g} * s(D)")
        else:
            parts.append(f"{axis}={_fmt_value(cand['g'])}")
    for key in sorted(grids_spec[algo]["shape"][cand["grid"]]):
        parts.append(f"{key}={_fmt_value(shape[key])}")
    return ", ".join(parts)


# --- scoring ---------------------------------------------------------------
def _tuned_label(cand: dict, params: dict, tuned: list[str]) -> str:
    axis, kind = GRAN[cand["algo"]]
    parts = [f"{p}={_fmt_value(params.get(p))}" for p in tuned if p in params]
    extra = []
    if cand["grid"] == "ladder" and kind == "scaled":
        extra.append(f"c={cand['g']:g}")
    for key in cand["shape"]:
        if key in SUBSTITUTES:
            extra.append(f"{key}={_fmt_value(cand['shape'][key])}")
    return ", ".join(parts) + (f" ({'; '.join(extra)})" if extra else "")


def score_table(df, picks, probes, tuned_of: dict) -> pd.DataFrame:
    """One row per (algo, fold, protocol): what was picked and how it scored."""
    rows = []
    for protocol, chosen in picks.items():
        for (algo, fold), cand in sorted(chosen.items()):
            cells = df[
                (df.algo == algo) & (df.config_id == cand["config_id"]) & (df.dataset == fold)
            ]
            final = cells[cells.seed.isin(FINAL_SEEDS)]
            final_ok = _ok(final)
            tune_cell = _ok(cells[cells.seed.isin(TUNE_SEEDS)])
            every_seed = _ok(cells[cells.seed.isin((*TUNE_SEEDS, *FINAL_SEEDS))])
            axis = GRAN[algo][0]
            params = resolve_candidate(cand, probes[fold])
            rows.append(
                {
                    "algo": algo,
                    "dataset": fold,
                    "protocol": protocol,
                    "grid": cand["grid"],
                    "config_id": cand["config_id"],
                    "level": cand["granularity_level"],
                    "theta": cand["theta"],
                    "g": cand["g"],
                    "param": axis,
                    "value": params.get(axis),
                    "picked": _tuned_label(cand, params, tuned_of[algo]),
                    "tune_ari": float(tune_cell[OBJ].iloc[0]) if len(tune_cell) else np.nan,
                    "final_ari": float(final_ok[OBJ].mean()) if len(final_ok) else np.nan,
                    # The tuning run is one more draw of the same config on its
                    # own permutation, so it belongs in the spread too -- it is
                    # only the *score* that must not come from the tuning seed.
                    "seed_std": float(every_seed[OBJ].std()) if len(every_seed) > 1 else np.nan,
                    "n_final_ok": len(final_ok),
                    "final_status": ",".join(sorted(set(final.status))) or "missing",
                    "k_hat": float(final_ok["k_hat"].mean()) if len(final_ok) else np.nan,
                    "n_predicted_clusters": (
                        float(final_ok["n_predicted_clusters"].mean()) if len(final_ok) else np.nan
                    ),
                    "runtime_s": float(final_ok["runtime_s"].mean()) if len(final_ok) else np.nan,
                    "throughput_sps": (
                        float(final_ok["throughput_sps"].mean()) if len(final_ok) else np.nan
                    ),
                    # Degeneracy evidence. A run that predicts one label for the
                    # whole stream scores 0 and costs nothing, which reads as a
                    # fast baseline unless these columns are beside it.
                    "largest_cluster_share": (
                        float(final_ok["largest_cluster_share"].mean())
                        if len(final_ok) and "largest_cluster_share" in final_ok else np.nan
                    ),
                    "max_micro_clusters": (
                        float(final_ok["max_micro_clusters"].mean())
                        if len(final_ok) and "max_micro_clusters" in final_ok else np.nan
                    ),
                    "homogeneity": (
                        float(final_ok["homogeneity"].mean())
                        if len(final_ok) and "homogeneity" in final_ok else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


# --- report ----------------------------------------------------------------
def _f(x, digits=3) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def _nanmean(values) -> float:
    """Mean over the finite entries; NaN (and no warning) when there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


#: A run is *degenerate* when it never produced a partition to score: one
#: predicted label over the whole stream. DenStream reaches this state at small
#: epsilon -- `_get_closest_cluster_key` returns the sentinel -1 when there are
#: no p-micro-clusters (RIVER_BUGS.md BUG-6) and that -1 is then scored as if it
#: were a cluster id. Such a run scores ~0 ARI *and* costs almost nothing, so in
#: a throughput table it outranks every configuration that actually clustered.
#: Flagging it is not cosmetic: it is the difference between "merged everything
#: into one cluster" and "produced no clusters at all".
MAX_NON_DEGENERATE_SHARE = 0.999


def _degenerate(row: dict) -> bool:
    share = row.get("largest_cluster_share")
    homog = row.get("homogeneity")
    n_pred = row.get("n_predicted_clusters")
    if share is not None and np.isfinite(share) and share >= MAX_NON_DEGENERATE_SHARE:
        return True
    if (n_pred is not None and np.isfinite(n_pred) and n_pred <= 2
            and homog is not None and np.isfinite(homog) and homog <= 1e-9):
        return True
    return False


def _cv(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.std() / abs(v.mean())) if len(v) > 1 and v.mean() else np.nan


#: Error rates behind the noise columns of report 3b-i: a one-sided test at
#: NOISE_ALPHA, and the gain it detects with probability NOISE_POWER.
#: Conventional values; they change what is printed, never what is picked.
NOISE_ALPHA = 0.05
NOISE_POWER = 0.80


def _noise_section(algos, folds, cell, refs) -> str:
    """Report 3b-i, second table: the gate's thresholds against the measured noise.

    Everything here is read off the scoring seeds, so it can describe the gate
    but must never feed it: a gate computed from FINAL_SEEDS would let the
    scoring draws decide which datasets get a vote.
    """
    from statistics import NormalDist

    z_a = NormalDist().inv_cdf(1 - NOISE_ALPHA)
    z_b = NormalDist().inv_cdf(NOISE_POWER)
    n = len(FINAL_SEEDS)
    out = []
    w = out.append
    w("\n**What the seed noise says about the two thresholds.** The ceiling is the "
      "argmax over the whole grid at one tuning draw, so it is biased upwards -- the "
      "selection bias of Cawley & Talbot (JMLR 2010). *Re-run* is that same configuration "
      f"scored on the {n} fresh seeds and *curse* the difference. *SE* is the standard error "
      "of the re-run gain over the default and *z* = gain / SE. *Detectable gain* is what a "
      f"one-sided test at alpha {NOISE_ALPHA} finds with power {NOISE_POWER} "
      f"(({z_a:.3f} + {z_b:.3f}) x SE): a `--min-ceiling-gain` below it asks for a gain the "
      "measurement cannot see. *Resolution* is one SE of an arm-vs-default difference in "
      f"headroom units (sd x sqrt(2/{n}) / gain); headroom differences smaller than it are "
      "seed noise on that dataset. Diagnostics only: they use the scoring seeds, so they "
      "never feed a selection.\n\n")
    w("| algo | dataset | ceiling | re-run | curse | sd | gain (re-run) | SE | z | "
      "detectable gain | resolution |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for algo in algos:
        pooled = []
        for fold in folds:
            tunes = {p: cell(algo, fold, p, "tune_ari") for p in ("ceiling_raw", "ceiling_sf")}
            tunes = {p: v for p, v in tunes.items() if np.isfinite(v)}
            if not tunes:
                continue
            top = max(tunes, key=tunes.get)
            ceil, again = tunes[top], cell(algo, fold, top, "final_ari")
            sd_c = cell(algo, fold, top, "seed_std")
            dflt = cell(algo, fold, "default", "final_ari")
            sd_d = cell(algo, fold, "default", "seed_std")
            sd_d = sd_d if np.isfinite(sd_d) else 0.0
            gain = again - dflt
            se = float(np.sqrt(sd_c ** 2 / n + sd_d ** 2 / n)) if np.isfinite(sd_c) else np.nan
            z = gain / se if np.isfinite(se) and se > 0 else np.nan
            res = sd_c * np.sqrt(2 / n) / gain if np.isfinite(sd_c) and gain > 0 else np.nan
            if bool(refs[algo]["tunable"].get(fold, False)) and np.isfinite(res):
                pooled.append(res)
            w(f"| {algo} | {fold} | {_f(ceil)} | {_f(again)} | {_f(ceil - again)} | "
              f"{_f(sd_c)} | {_f(gain)} | {_f(se, 4)} | {_f(z, 1)} | "
              f"{_f((z_a + z_b) * se)} | {_f(res)} |\n")
        if pooled:
            se_pool = float(np.sqrt(np.sum(np.square(pooled)))) / len(pooled)
            w(f"\n- **{algo}**: over its {len(pooled)} tunable datasets, one SE of a difference "
              f"in mean headroom is about {_f(se_pool)}, so a gap between two arms smaller than "
              f"{z_a:.2f} x that = {_f(z_a * se_pool)} is inside the seed noise at the one-sided "
              f"{NOISE_ALPHA:g} level (approximate: it takes the ceiling configuration's sd as "
              "every arm's and the datasets as independent).\n")
    return "".join(out)


def _grid_section(algos, grids_spec, grids, rejected, tuned_of) -> str:
    out = []
    w = out.append
    w("\n## 0. Grids\n\n")
    w("Every parameter not listed stays at River's constructor default. Levels of the "
      "resolution axis run coarse to fine (level 1 = fewest clusters).\n\n")
    for algo in algos:
        axis, kind = GRAN[algo]
        spec = grids_spec[algo]
        w(f"### {algo}\n\n")
        w("| grid | resolution axis | shape | points | rejected |\n|---|---|---|---|---|\n")
        for g in ("raw", "ladder"):
            unit = (f"`{axis}` = c * s(D), c in" if (g == "ladder" and kind == "scaled")
                    else f"`{axis}` in")
            levels = ", ".join(f"{v:g}" for v in spec["levels"][g])
            shape = "; ".join(
                f"`{k}` in {{{', '.join(f'{v:g}' for v in vals)}}}"
                for k, vals in sorted(spec["shape"][g].items())
            )
            w(f"| {g} | {unit} {{{levels}}} | {shape} | {len(grids[g][algo])} | "
              f"{len(rejected[g][algo])} |\n")
        subs = [k for k in spec["shape"]["ladder"] if k in SUBSTITUTES]
        for key in subs:
            w(f"\n`{key}` resolves to `{SUBSTITUTES[key][0]}` = {key} * `{axis}`.\n")
        pinned = spec.get("fixed", {})
        if pinned:
            w("\nPinned on purpose (named here so it is not confused with a value that "
              "merely fell through to River's default): "
              + ", ".join(f"`{k}` = {v:g}" for k, v in sorted(pinned.items())) + ".\n")
            for key in sorted(pinned):
                note = FIXED_NOTE.get((algo, key))
                if note:
                    w(f"\n> **Why `{key}` is pinned.** {note}\n")
        rules = "; ".join(name for name, _ in CONSTRAINTS.get(algo, []))
        if rules:
            w(f"\nConstraint: {rules}. Tuned parameters: {', '.join(tuned_of[algo])}.\n")
        w("\n")
    return "".join(out)


def build_report(tune, scores, folds, probes, requested, grids, grids_spec, rejected,
                 tuned_of, finals, aggregate, irace: dict | None = None) -> str:
    """The pilot report. ``irace`` (optional) is ``{"results": irace_results(...),
    "settings": {...}, "space": str}`` when irace arms were raced."""
    out = []
    w = out.append
    by = {(r.algo, r.dataset, r.protocol): r._asdict() for r in scores.itertuples(index=False)}
    # One reference per algorithm, from the tuning frame -- the yardstick every
    # selection used, so a headroom printed here means what it meant there.
    refs = {a: fold_reference(tune[tune.algo == a]) for a in sorted(set(scores.algo))}

    def cell(algo, fold, protocol, col="final_ari"):
        row = by.get((algo, fold, protocol))
        return np.nan if row is None else row[col]

    def headroom_of(algo, fold, value):
        ref = refs.get(algo)
        if ref is None or not bool(ref["tunable"].get(fold, False)) or not np.isfinite(value):
            return np.nan
        gain = ref["gain"].get(fold, np.nan)
        return (value - ref["default"][fold]) / gain if np.isfinite(gain) and gain > 0 else np.nan

    def head_mean(algo, protocol):
        return _nanmean([headroom_of(algo, f, cell(algo, f, protocol)) for f in folds])

    def head_cell(algo, protocol):
        # Pooled over the tunable datasets only, like `--aggregate headroom`.
        k = int(refs[algo]["tunable"].reindex(folds).fillna(False).astype(bool).sum()) \
            if algo in refs else 0
        vals = np.array([headroom_of(algo, f, cell(algo, f, protocol)) for f in folds])
        n = int(np.isfinite(vals).sum())
        if not k:
            return "-"
        text = _f(_nanmean(vals))
        return text if n == k else f"{text} ({n}/{k})"

    def vec(algo, protocol):
        return np.array([cell(algo, f, protocol) for f in folds], dtype=float)

    def mean_of(algo, protocol):
        return _nanmean(vec(algo, protocol))

    def n_scored(algo, protocol):
        return int(np.isfinite(vec(algo, protocol)).sum())

    def mean_cell(algo, protocol):
        # A mean over fewer datasets is not comparable with a full one; say so.
        n = n_scored(algo, protocol)
        text = _f(mean_of(algo, protocol))
        return f"**{text}**" if n == len(folds) else f"{text} ({n}/{len(folds)})"

    def of(algo, protocol):
        return scores[(scores.algo == algo) & (scores.protocol == protocol)]

    algos = sorted(set(scores.algo))
    protocols = set(scores.protocol)
    p2r_names = sorted(p for p in protocols if p.startswith("p2r_") and not p.endswith(EST))
    irace_names = [p for p in IRACE_SHIPS if p in protocols]
    ships = [p for p in FINAL_ARMS if p in protocols] + p2r_names + irace_names
    raced = bool(irace_names) or any(p.startswith("p2r_irace_") for p in p2r_names)
    pool = POOL_LABEL[aggregate]

    w("# HPO pilot -- one hyperparameter set per arm\n\n")
    w(f"Stream length: **{requested}** per dataset. Objective: ARI over the whole stream. "
      f"Sets are chosen on the mean over tuning seeds {list(TUNE_SEEDS)}; every reported "
      f"score is the mean over "
      f"fresh runs under seeds {list(FINAL_SEEDS)} (their own permutations). Scores are "
      f"pooled across datasets by **{pool}** (`--aggregate {aggregate}`"
      + (", over the datasets that pass the 3b-i gate" if aggregate == "headroom" else "")
      + ").\n\n")
    w(f"Datasets ({len(folds)}): {', '.join(folds)}\n\n")
    if aggregate == "headroom":
        silent = [a for a in algos if a not in {r.algo for r in scores.itertuples()
                                                if r.protocol in tuple(FINAL_ARMS)[1:]}]
        w("> Under `--aggregate headroom` an algorithm with **no** tunable dataset ships "
          "no set at all: there is nothing to normalise against, and inventing a pick "
          "from datasets that hold no signal would be reporting noise as a result. "
          + (f"Here that is: **{', '.join(silent)}** -- absent from sections 2 and 3 by "
             "design; see 3b-i and its cost columns in 3b-iii.\n\n"
             if silent else "No algorithm is in that position here.\n\n"))
    w("Every arm ends with **one set per algorithm**, run unchanged on every dataset.\n\n"
      "| arm | grid | how the one set is chosen |\n|---|---|---|\n"
      f"| arm 1 | raw | best {pool} over all datasets |\n"
      "| arm 2 | raw | leave-one-dataset-out; the set most rounds picked |\n"
      "| arm 3 | substituted (`c * s(D)`, micro-ratio) | as arm 2; the set is dimensionless "
      "and resolves per dataset |\n"
      "| P2R | substituted | shape as arm 3, rule's phi on all datasets; the resolution level "
      "is read off each dataset's own label-free curve |\n"
      f"| sf-all | substituted | best {pool} over all datasets (arm 1 on the substituted grid) |\n"
      + ("| irace raw / ladder / P2R | as arm 1 / sf-all / P2R | the elite of an irace race over "
         "(dataset, seed) instances of the tunable datasets, cost = -headroom (section 3c); "
         "the LODO estimate races without the dataset it is scored on |\n" if raced else "")
      + "\nAll arms use the labels of all datasets to choose, so their scores on those same "
      "datasets (section 2) are **in-sample**. What a procedure is worth on a dataset it "
      "never saw is its **LODO estimate** (section 3).\n")
    w(_grid_section(algos, grids_spec, grids, rejected, tuned_of))

    # -- the sets
    w("\n## 1. The one set each arm ships\n\n")
    for algo in algos:
        w(f"### {algo}\n\n| arm | set | how it was reached |\n|---|---|---|\n")
        for p in ships:
            info = finals.get(p, {}).get(algo)
            if info is None:
                w(f"| {arm_label(p)} | - | no set: nothing eligible on every dataset |\n")
                continue
            cand = info["config"]
            if p.startswith("p2r_"):
                axis = GRAN[algo][0]
                shape = spec_label(cand, grids_spec, tuned_of[algo], level=False)
                levels = ", ".join(
                    f"{f}: {_level_value(grids_spec, algo, lv)}"
                    for f, lv in sorted(info["levels"].items())
                ) or "-"
                phi = "" if info["phi"] is None else f"phi={_fmt_value(info['phi'])}; "
                if info.get("votes") is not None:
                    chosen = f"shape picked by {info['votes']}/{info['rounds']} rounds; "
                else:
                    chosen = f"shape = {_irace_how(info.get('irace') or {})}; "
                how = f"{chosen}{phi}`{axis}` per dataset -> {levels}"
                w(f"| {arm_label(p)} | {shape}; `{axis}` by {info['rule']} on "
                  f"{info['criterion']} | {how} |\n")
                continue
            if p == "default":
                how = "River constructor"
            elif "irace" in info:
                how = _irace_how(info["irace"])
            elif "votes" in info:
                how = f"picked by {info['votes']}/{info['rounds']} LODO rounds"
            else:
                how = f"best {pool} over {len(folds)} datasets"
            w(f"| {arm_label(p)} | {spec_label(cand, grids_spec, tuned_of[algo])} | {how} |\n")
        w("\n")

    # -- the sets on every dataset
    w("\n## 2. The set on every dataset (in-sample)\n\n")
    head = ["algo", "dataset"] + [arm_label(p) for p in ships]
    w("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
    for algo in algos:
        for fold in folds:
            w(f"| {algo} | {fold} | " + " | ".join(_f(cell(algo, fold, p)) for p in ships) + " |\n")
        w(f"| **{algo}** | **mean** | "
          + " | ".join(mean_cell(algo, p) for p in ships) + " |\n")
        w(f"| **{algo}** | **mean headroom** | "
          + " | ".join(head_cell(algo, p) for p in ships) + " |\n")
    w("\n*Mean headroom* is over the datasets that passed the 3b-i gate, against the "
      "tuning-seed reference every selection used: 1 = that dataset's grid ceiling, "
      "0 = River's default. It is the quantity `--aggregate headroom` and the irace "
      "arms choose on.\n")
    w("\nThreshold each set resolves to, per dataset (substituted sets differ by dataset "
      "by design):\n\n")
    head = ["algo", "dataset"] + [arm_label(p) for p in ships if p != "default"]
    w("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
    for algo in algos:
        for fold in folds:
            w(f"| {algo} | {fold} | "
              + " | ".join(_fmt_value(cell(algo, fold, p, "value")) if (algo, fold, p) in by
                           else "-" for p in ships if p != "default") + " |\n")
    finals_rows = scores[scores.protocol.isin(ships)]
    failures = finals_rows[finals_rows.n_final_ok < len(FINAL_SEEDS)]
    if len(failures):
        w("\nRuns that did not complete (counted as failures, not dropped):\n\n")
        for r in failures.itertuples(index=False):
            w(f"- {r.algo} / {r.dataset} / {arm_label(r.protocol)}: status `{r.final_status}`\n")
    missing = [(a, p) for a in algos for p in p2r_names if n_scored(a, p) < len(folds)]
    if missing:
        w("\nA P2R policy scores no value on a dataset where its curve is broken (a failed "
          "rung, or an infinite silhouette); compare the counts before the means: "
          + ", ".join(f"{a} {arm_label(p)} {n_scored(a, p)}/{len(folds)}" for a, p in missing)
          + ".\n")

    w("\nDifferences below are means over the datasets where both sides have a score.\n")
    w("\n| algo | arm 3 - arm 2 | sf-all - arm 1 | arm 2 - arm 1 | arm 3 - sf-all |"
      + "".join(f" {arm_label(p)} - arm 3 |" for p in p2r_names)
      + "\n|" + "---|" * (5 + len(p2r_names)) + "\n")
    with np.errstate(all="ignore"):
        for algo in algos:
            a1, a2, a3 = (vec(algo, p) for p in ("arm1_raw_all", "arm2_raw_lodo", "arm3_sf_lodo"))
            sfa = vec(algo, "sf_all")
            w(f"| {algo} | {_f(_nanmean(a3 - a2))} | {_f(_nanmean(sfa - a1))} | "
              f"{_f(_nanmean(a2 - a1))} | {_f(_nanmean(a3 - sfa))} |"
              + "".join(f" {_f(_nanmean(vec(algo, p) - a3))} |" for p in p2r_names) + "\n")
    w("\n*Arm 3 - arm 2*: same selection, same budget, only the units differ -- the effect "
      "of the substitutes. *sf-all - arm 1*: the same comparison when both simply take the "
      "best set over all datasets. *Arm 2 - arm 1* and *arm 3 - sf-all*: what the LODO "
      "consensus gives up against the plain best-over-all choice, per grid.\n")

    # -- LODO estimates
    w("\n## 3. LODO estimate: the procedure on a dataset it never saw\n\n")
    ests = [p for p in ("arm2_raw_lodo", "arm3_sf_lodo", *p2r_names, *irace_names)
            if p + EST in protocols]
    if ests:
        head = ["algo", "dataset"] + [arm_label(p) for p in ests]
        w("In each round the set is chosen without the dataset in the row and scored on it.\n\n")
        w("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for algo in algos:
            for fold in folds:
                w(f"| {algo} | {fold} | "
                  + " | ".join(_f(cell(algo, fold, p + EST)) for p in ests) + " |\n")
            w(f"| **{algo}** | **mean** | "
              + " | ".join(mean_cell(algo, p + EST) for p in ests) + " |\n")
            w(f"| **{algo}** | **mean headroom** | "
              + " | ".join(head_cell(algo, p + EST) for p in ests) + " |\n")
        w("\n| algo | arm | in-sample mean | LODO estimate | optimism | in-sample headroom | "
          "LODO headroom |\n|---|---|---|---|---|---|---|\n")
        for algo in algos:
            for p in ests:
                ins, est = mean_of(algo, p), mean_of(algo, p + EST)
                w(f"| {algo} | {arm_label(p)} | {_f(ins)} | {_f(est)} | {_f(ins - est)} | "
                  f"{_f(head_mean(algo, p))} | {_f(head_mean(algo, p + EST))} |\n")
        w("\n*Optimism* is how much the in-sample score flatters the procedure. The LODO "
          "estimate of arm 3 minus that of arm 2 is the honest version of the substitutes' "
          "effect.\n")
    if "arm3_sf_cross" + EST in protocols:
        w("\n### Leave-one-group-out\n\n"
          "| algo | arm 3 within group | arm 3 across groups | gap |\n|---|---|---|---|\n")
        for algo in algos:
            within = mean_of(algo, "arm3_sf_lodo" + EST)
            across = mean_of(algo, "arm3_sf_cross" + EST)
            w(f"| {algo} | {_f(within)} | {_f(across)} | {_f(across - within)} |\n")

    # The default row lives in `tune` as the headroom baseline; it is not part
    # of any grid, so every grid statistic below excludes it.
    ok_tune = _ok(tune[tune.grid != "default"])
    # -- per-dataset reading (B1/B2/B3): the tables that replace a single mean
    tuned_arms_all = [p for p in ships if p != "default"]

    w("\n## 3b. Per dataset, not pooled\n\n")
    w("A single pooled number is decided by whichever dataset has the largest ARI, "
      "and here those are also the datasets with the largest seed noise. The three "
      "tables below are the pooled score taken apart.\n")

    w("\n### 3b-i. Is there anything to tune? (the gate)\n\n")
    w(f"A dataset is **tunable** for an algorithm when its ceiling clears "
      f"`{MIN_CEILING}` **and** beats the default by `{MIN_CEILING_GAIN}`. Both "
      "numbers are this project's choice, not a published threshold. A dataset "
      "that fails holds no evidence for any arm; pooling it in only adds noise.\n\n")
    w("| algo | dataset | default | ceiling | gain | seed noise | tunable |\n"
      "|---|---|---|---|---|---|---|\n")
    noise = {}
    for algo in algos:
        ref = refs[algo]
        for fold in folds:
            row = ref.reindex([fold]).iloc[0] if fold in ref.index else None
            sd = float(np.nanmax([
                cell(algo, fold, p, "seed_std") for p in ships
            ])) if ships else np.nan
            noise[(algo, fold)] = sd
            if row is None:
                w(f"| {algo} | {fold} | - | - | - | {_f(sd)} | no |\n")
                continue
            w(f"| {algo} | {fold} | {_f(row['default'])} | {_f(row['ceiling'])} | "
              f"{_f(row['gain'])} | {_f(sd)} | {'YES' if row['tunable'] else 'no'} |\n")
    for algo in algos:
        keep = [f for f in folds if bool(refs[algo]["tunable"].get(f, False))]
        w(f"\n- **{algo}**: tunable on {len(keep)}/{len(folds)} "
          f"({', '.join(keep) or 'none'}).")
        if not keep:
            w(f" Nothing in the grid reaches past River's default on any dataset, so "
              f"no ARI-based statement about {algo} is supported here; read its cost "
              "columns instead.")
    w("\n")
    w(_noise_section(algos, folds, cell, refs))

    w("\n### 3b-ii. Win / tie / loss against the default\n\n")
    w("The tie band is that dataset's own measured seed noise (max `seed_std` over "
      "the arms), not a fixed number: a gap smaller than the spread of repeated runs "
      "of the same configuration is not a result. `worst` is the largest loss against "
      "the default on any dataset -- the column an average hides.\n\n")
    w("| algo | arm | win | tie | loss | worst | mean headroom (tunable only) |\n"
      "|---|---|---|---|---|---|---|\n")
    for algo in algos:
        ref = refs[algo]
        for prot in tuned_arms_all:
            win = tie = loss = 0
            worst = np.nan
            heads = []
            for fold in folds:
                v, d = cell(algo, fold, prot), cell(algo, fold, "default")
                if not (np.isfinite(v) and np.isfinite(d)):
                    continue
                band = noise.get((algo, fold), np.nan)
                band = band if np.isfinite(band) and band > 0 else 0.0
                diff = v - d
                worst = diff if not np.isfinite(worst) else min(worst, diff)
                if diff > band:
                    win += 1
                elif -diff > band:
                    loss += 1
                else:
                    tie += 1
                if bool(ref["tunable"].get(fold, False)) and ref["gain"].get(fold, 0) > 0:
                    heads.append((v - ref["default"][fold]) / ref["gain"][fold])
            w(f"| {algo} | {arm_label(prot)} | {win} | {tie} | {loss} | {_f(worst)} | "
              f"{_f(_nanmean(heads)) if heads else '-'} |\n")
    w("\nHeadroom 1.00 means the arm reached the per-dataset ceiling, 0.00 means it "
      "matched the default, negative means it did worse. It is blank where no dataset "
      "passed the gate.\n")

    w("\n### 3b-iii. Quality at a cost floor\n\n")
    w(f"Best ARI the grid reaches **subject to** `throughput >= {MIN_THROUGHPUT_SPS:g}` "
      "samples/s, beside the unconstrained ceiling. This is constrained HPO in the "
      "sense of Franceschi et al. Sec. 2.2.2 / 8.1, and it is here because the only "
      "knob that raises ARI also multiplies the micro-cluster count: an unconstrained "
      "argmax picks the slowest configuration on the curve every time. The floor is a "
      "runnable placeholder (`--min-throughput`), not a recommendation.\n\n")
    w("| algo | dataset | ceiling | ceiling at the floor | cost of the floor | "
      "throughput of the unconstrained pick |\n|---|---|---|---|---|---|\n")
    for algo in algos:
        sub_all = ok_tune[ok_tune.algo == algo]
        for fold in folds:
            cells_f = sub_all[sub_all.dataset == fold]
            if cells_f.empty:
                continue
            top = cells_f.loc[cells_f[OBJ].idxmax()]
            fast = cells_f[cells_f.throughput_sps >= MIN_THROUGHPUT_SPS] \
                if "throughput_sps" in cells_f else cells_f.iloc[0:0]
            best_fast = float(fast[OBJ].max()) if len(fast) else np.nan
            w(f"| {algo} | {fold} | {_f(float(top[OBJ]))} | {_f(best_fast)} | "
              f"{_f(float(top[OBJ]) - best_fast)} | "
              f"{_f(float(top.get('throughput_sps', np.nan)), 0)} |\n")
    w("\nA large *cost of the floor* means the arm's quality was bought with compute "
      "and would not survive a deployment budget.\n")

    if raced:
        w(_irace_section(algos, protocols, head_mean, mean_of, irace or {}))

    # -- checks
    w("\n## 4. Health\n")
    verdicts = {}

    w("\n### 4a. Does the grid separate configurations beyond seed noise?\n\n")
    n_seeds = len(TUNE_SEEDS) + len(FINAL_SEEDS)
    w("Spread = max - min ARI over the grid on one dataset (mean over the tuning seeds). "
      f"Noise = std over all {n_seeds} seeds of that dataset's best config.\n\n"
      "| algo | grid | median spread | median noise | ratio | verdict |\n|---|---|---|---|---|---|\n")
    for algo in algos:
        for grid, protocol in (("raw", "ceiling_raw"), ("ladder", "ceiling_sf")):
            sub = ok_tune[(ok_tune.algo == algo) & (ok_tune.grid == grid)]
            spread = sub.groupby("dataset")[OBJ].agg(lambda s: s.max() - s.min())
            med_s = float(spread.median()) if len(spread) else np.nan
            med_n = float(of(algo, protocol).seed_std.median())
            # No spread at all is a failure however small the noise: 0/0 must
            # not read as an infinite ratio.
            if not np.isfinite(med_s) or med_s <= 0:
                ratio = np.nan
            else:
                ratio = med_s / med_n if med_n > 0 else np.inf
            good = bool(np.isfinite(med_s) and med_s > 0 and ratio >= MIN_SPREAD_TO_NOISE)
            verdicts[f"4a {algo} {grid}"] = good
            w(f"| {algo} | {grid} | {_f(med_s)} | {_f(med_n)} | {_f(ratio, 1)} | "
              f"{'PASS' if good else 'FAIL'} |\n")
    w(f"\nPASS needs ratio >= {MIN_SPREAD_TO_NOISE}. A FAIL means choosing on this grid is "
      "mostly choosing noise.\n")

    w("\n### 4b. Does an optimum sit at an end of the resolution axis?\n\n")
    w("| algo | grid | per-dataset optima at an end | share | verdict | shipped sets at an end |\n"
      "|---|---|---|---|---|---|\n")
    for algo in algos:
        for grid, ceiling, shipped in (("raw", "ceiling_raw", ("arm1_raw_all", "arm2_raw_lodo")),
                                       ("ladder", "ceiling_sf", ("sf_all", "arm3_sf_lodo"))):
            n_levels = len(grids_spec[algo]["levels"][grid])
            levels = of(algo, ceiling).level.dropna()
            hits = int(levels.isin([1, n_levels]).sum())
            share = hits / len(levels) if len(levels) else np.nan
            good = bool(share < MAX_ENDPOINT_SHARE)
            verdicts[f"4b {algo} {grid}"] = good
            at_end = [arm_label(p) for p in shipped
                      if p in finals and algo in finals[p]
                      and finals[p][algo]["config"]["granularity_level"] in (1, n_levels)]
            w(f"| {algo} | {grid} | {hits}/{len(levels)} | {_f(share, 2)} | "
              f"{'PASS' if good else 'FAIL'} | {', '.join(at_end) or 'none'} |\n")
    w(f"\nPASS needs share < {MAX_ENDPOINT_SHARE}. An optimum at an end means the true "
      "optimum may lie outside the axis.\n")

    w("\n### 4c. Winner's curse: tuning score vs. fresh re-run of the shipped set\n\n")
    w("| algo | arm | mean(tune - re-run) | max |\n|---|---|---|---|\n")
    for algo in algos:
        for p in ("arm1_raw_all", "sf_all"):
            gap = of(algo, p).pipe(lambda d: d.tune_ari - d.final_ari)
            w(f"| {algo} | {arm_label(p)} | {_f(float(gap.mean()))} | {_f(float(gap.max()))} |\n")
    w("\nLarge means the set won partly on a lucky permutation, and a grid search reported "
      "without a re-run would publish exactly that number.\n")
    for algo in algos:
        curse = max(
            (float(of(algo, prot).pipe(lambda d: d.tune_ari - d.final_ari).mean())
             for prot in ("arm1_raw_all", "sf_all") if len(of(algo, prot))),
            default=np.nan,
        )
        spread = [abs(mean_of(algo, a) - mean_of(algo, b))
                  for a in tuned_arms_all for b in tuned_arms_all if a < b]
        spread = [x for x in spread if np.isfinite(x)]
        gap = max(spread) if spread else np.nan
        if not (np.isfinite(curse) and np.isfinite(gap)):
            continue
        # Always stated, not only when it dominates: the number is meaningless
        # on its own and decisive next to the gaps it has to be compared with.
        verdict = ("**as large as** the biggest gap between any two arms"
                   if curse >= gap else
                   f"{gap / curse:.1f}x smaller than the biggest gap between any two arms"
                   if curse > 0 else "not measurable against the arm gaps")
        w(f"\n> **{algo}:** winner's curse {_f(curse)} vs. largest arm-to-arm gap "
          f"{_f(gap)} -- {verdict}. ")
        if curse >= gap:
            w("No arm-to-arm comparison for this algorithm survives that; read "
              "sections 2 and 3 as upper bounds on the differences, not as the "
              "differences.\n")
        else:
            w("Any single arm-to-arm difference below the first number is inside "
              "the selection luck and should not be read as an effect.\n")

    w("\n### 4d. How much do the LODO rounds agree?\n\n")
    w("| algo | arm | distinct sets over the rounds | votes for the shipped set |\n"
      "|---|---|---|---|\n")
    irace_res = (irace or {}).get("results") or {}
    for algo in algos:
        for p in ("arm2_raw_lodo", "arm3_sf_lodo", *p2r_names, *irace_names):
            info = finals.get(p, {}).get(algo)
            rounds = of(algo, p + EST)
            if info is None or rounds.empty:
                continue
            if "irace" in info:
                # An irace round is a whole race; its elite is the round's pick.
                per_round = irace_res.get((algo, p)) or {}
                distinct = len({json.dumps(res["best"], sort_keys=True, default=str)
                                for name, res in per_round.items()
                                if name.startswith("lodo-") and res and res.get("best")})
                what, votes = "elites", "- (irace ships the elite of the all-dataset race)"
            else:
                distinct = (rounds.theta.nunique() if p.startswith("p2r_")
                            else rounds.config_id.nunique())
                what = "shapes" if p.startswith("p2r_") else "sets"
                votes = f"{info['votes']}/{info['rounds']}"
            w(f"| {algo} | {arm_label(p)} | {distinct} {what} | {votes} |\n")
    w("\nFew votes for the shipped set means no single set transfers; the consensus is "
      "then a weak choice and the LODO estimate is the number to trust.\n")

    scaled = [a for a in algos if GRAN[a][1] == "scaled"]
    if scaled:
        w("\n### 4e. Does s(D) make the threshold transferable?\n\n")
        w("Best threshold per dataset (substituted grid, per-dataset optimum), spread across "
          "datasets.\n\n| algo | CV(c*) | CV(threshold*) | verdict |\n|---|---|---|---|\n")
        for algo in scaled:
            # On a dataset where every rung ties, the tie-break alone picks the
            # level; counting it would fake a perfectly stable optimum.
            sub = ok_tune[(ok_tune.algo == algo) & (ok_tune.grid == "ladder")]
            spread = sub.groupby("dataset")[OBJ].agg(lambda s: s.max() - s.min())
            separating = set(spread[spread > 0].index)
            o = of(algo, "ceiling_sf").dropna(subset=["g"])
            o = o[o.dataset.isin(separating)]
            if len(o) < 2:
                verdicts[f"4e {algo}"] = False
                w(f"| {algo} | - | - | FAIL -- fewer than 2 datasets where the ladder "
                  "separates configs |\n")
                continue
            cv_c, cv_abs = _cv(o.g), _cv(o.value)
            good = bool(np.isfinite(cv_c) and cv_c < cv_abs)
            verdicts[f"4e {algo}"] = good
            w(f"| {algo} | {_f(cv_c)} | {_f(cv_abs)} | {'PASS' if good else 'FAIL'} "
              f"({len(o)} datasets) |\n")
        w("\nIf the optimum does not vary less on `c` than in raw units, one `c` for every "
          "dataset has no more basis than one raw threshold.\n")

    w("\n### 4f. Does any shipped set beat River's default?\n\n")
    tuned_arms = [p for p in ships if p != "default"]
    w("| algo | " + " | ".join(f"{arm_label(p)} - default" for p in tuned_arms)
      + " |\n|---|" + "---|" * len(tuned_arms) + "\n")
    for algo in algos:
        d = vec(algo, "default")
        with np.errstate(all="ignore"):
            w(f"| {algo} | " + " | ".join(
                f"{_f(_nanmean(vec(algo, p) - d))} ({int(np.nansum(vec(algo, p) < d))} "
                f"datasets worse)" for p in tuned_arms) + " |\n")

    w("\n### 4g. Do the arms crown the same algorithm?\n\n")
    if len(algos) < 2:
        w("Needs two algorithms.\n")
    else:
        arms = [p for p in ("arm1_raw_all", "arm2_raw_lodo", "arm3_sf_lodo") if p in protocols]
        w("| dataset | default | " + " | ".join(arm_label(p) for p in arms)
          + " | agree |\n|---|---|" + "---|" * len(arms) + "---|\n")
        agree = total = 0
        for fold in folds:
            winners = {}
            for p in ("default", *arms):
                vals = {a: cell(a, fold, p) for a in algos}
                vals = {a: v for a, v in vals.items() if np.isfinite(v)}
                winners[p] = max(vals, key=vals.get) if len(vals) == len(algos) else None
            named = [winners[p] for p in arms]
            same = all(named) and len(set(named)) == 1
            if all(named):
                total += 1
                agree += int(same)
            w(f"| {fold} | {winners['default'] or '-'} | " + " | ".join(n or "-" for n in named)
              + f" | {'yes' if same else 'no'} |\n")
        w(f"\nAgreement: **{agree}/{total}**. Low agreement means the tuning method changes "
          "which algorithm wins, so a benchmark built on one of them is fragile.\n")

    if p2r_names:
        w("\n### 4h. Is the label-free criterion usable on this dataset?\n\n")
        w("**Gate for every P2R reading below and for P4.** A rule can only be as good "
          "as the curve it reads. Regret is measured in ARI, per shape point: what the "
          "curve's best rung scores, minus what the rung the criterion points at "
          f"scores. A criterion whose median regret exceeds `{MAX_CRITERION_REGRET}` on "
          "a dataset is **not a usable proxy there**, and the P2R numbers for that "
          "dataset are reporting the criterion's failure, not the rule's.\n\n")
        criteria = sorted({c for c in P2R_CRITERIA
                           if any(p.endswith(c) for p in p2r_names)})
        w("| algo | dataset | criterion | curves | median regret | max regret | "
          "curves ranking against ARI | verdict |\n|---|---|---|---|---|---|---|---|\n")
        for algo in algos:
            n_lv = len(grids_spec[algo]["levels"]["ladder"])
            for fold in folds:
                for crit in criteria:
                    r = criterion_regret(ok_tune[ok_tune.grid == "ladder"], algo, fold,
                                         crit, n_lv)
                    if not r["n_curves"]:
                        continue
                    good = bool(r["median"] <= MAX_CRITERION_REGRET)
                    verdicts[f"4h {algo} {fold} {crit}"] = good
                    w(f"| {algo} | {fold} | {crit} | {r['n_curves']} | "
                      f"{_f(r['median'])} | {_f(r['max'])} | "
                      f"{r['n_negative']}/{r['n_curves']} | "
                      f"{'PASS' if good else 'UNUSABLE'} |\n")
        w("\nAn UNUSABLE row is a result in its own right, not a missing number: it is "
          "direct evidence for this search space that no label-free criterion here "
          "substitutes for labels on that dataset. Note also that it cannot be detected "
          "without labels -- which is why the label-free arm has to be a *protocol* run "
          "on labelled data (P4), and not a separate pool of unlabelled datasets.\n")

    w("\n### 4i. Which shipped runs produced no partition at all?\n\n")
    w("A run listed here did not merge everything into one cluster -- it never "
      "produced a cluster. DenStream returns the sentinel -1 from "
      "`_get_closest_cluster_key` when it holds no p-micro-clusters "
      "(RIVER_BUGS.md BUG-6) and that -1 is then scored as a cluster id. Such a run "
      "scores ~0 ARI and is nearly free, so **its throughput is the speed of doing "
      "nothing** and must not be read as an efficiency win, and its 0.000 must not be "
      "read as better than a negative ARI from a run that did cluster.\n\n")
    degen = [(a, f, p) for a in algos for f in folds for p in ships
             if (by.get((a, f, p)) is not None and _degenerate(by[(a, f, p)]))]
    if not degen:
        w("None.\n")
    else:
        w("| algo | dataset | arm | ARI | predicted clusters | largest cluster share | "
          "throughput | homogeneity |\n|---|---|---|---|---|---|---|---|\n")
        for a, f, prot in degen:
            r = by[(a, f, prot)]
            w(f"| {a} | {f} | {arm_label(prot)} | {_f(r.get('final_ari'))} | "
              f"{_f(r.get('n_predicted_clusters'), 1)} | "
              f"{_f(r.get('largest_cluster_share'))} | "
              f"{_f(r.get('throughput_sps'), 0)} | {_f(r.get('homogeneity'))} |\n")
        w(f"\n{len(degen)} degenerate (algo, dataset, arm) cells.\n")
    verdicts["4i no degenerate shipped run"] = not degen

    w("\n## 5. Read these numbers with\n\n")
    dominated = [f for f in folds if probes[f].get("one_d_dominated")]
    w(f"- **1-D dominated datasets** (raw units, top-1 variance share > 0.9): "
      f"{', '.join(dominated) or 'none'}. Comparisons there describe one axis only.\n")
    w("- Dataset scales s(D): " + ", ".join(f"{f}={probes[f]['s']:.3g}" for f in folds)
      + ". One raw threshold is fine-grained on one dataset and coarse on another.\n")
    if aggregate == "mean":
        w("- **Pooling by mean ARI is the weakest choice available here.** It lets the "
          "dataset with the largest ARI decide the set for all of them, and on this pool "
          "the datasets with the largest ARI are also the ones with the largest seed "
          "noise -- so the pooled number is driven by the least trustworthy "
          "measurements. `--aggregate rank` removes the scale; `--aggregate headroom` "
          "normalises each dataset by its own reachable range and drops the ones that "
          "failed the 3b-i gate. Read section 3b before this section's means.\n")
    for algo in algos:
        bad = [g for g in ("raw", "ladder") if verdicts.get(f"4a {algo} {g}") is False]
        untunable = not any(bool(refs[algo]["tunable"].get(f, False)) for f in folds)
        if untunable:
            w(f"- **{algo} is not tunable by ARI on this pool.** No dataset passed the "
              f"3b-i gate, and check 4a rates its {' and '.join(bad) or 'grid'} "
              "separation at or below the seed noise. Its ARI differences between arms "
              "are noise; only its cost columns (throughput, memory, micro-cluster "
              "count) carry a signal, and those are deterministic in the grid.\n")
        elif bad:
            w(f"- **{algo}**: the {', '.join(bad)} grid failed check 4a -- choosing on it "
              "is mostly choosing noise. Per-dataset spread-to-noise is in 3b-i.\n")
    by_group: dict[str, list[str]] = {}
    for f in folds:
        by_group.setdefault(FOLD_GROUPS.get(f, "other"), []).append(f)
    w("- LODO rounds tune only on the held-out dataset's own group -- "
      + "; ".join(f"{g}: {', '.join(fs)}" for g, fs in by_group.items()) + ".\n")
    w("- **Ground truth is not the same thing as cluster structure.** Every dataset here "
      "carries labels, but a label is a supervised target and need not be a geometric "
      "cluster: a dataset can have complete ground truth and still hold no clustering "
      "signal at all, which is what the 3b-i gate detects. Three labels in this pool are "
      "weak for a further, separate reason, and none of them should carry a headline: "
      "`shuttle` ships in River's outlier-detection framing, so its label is **binary** "
      "(nominal vs anomalous) and not the 7 original classes; `electricity` has 2 classes "
      "whose autocorrelation is such that predicting \"same as last\" already scores 85%; "
      "`dstream_30k_nonconvex` labels 5,000 of its 30,000 points `None` (injected noise), "
      "and the external metrics skip those, so its ARI is computed on 5/6 of the "
      "stream.\n")
    w("- CluStream's resolution axis is a count and has no scale conversion; its substitute "
      "is the micro-ratio only.\n")
    w("- CluStream's offline phase is k-means and DenStream's is DBSCAN-like: an ARI gap "
      "between the two partly compares the offline phases.\n")
    w(f"- {len(TUNE_SEEDS)} tuning runs per grid point and {len(FINAL_SEEDS)} scoring seeds "
      "on a handful of datasets: a pilot check of direction, not a statistical result.\n")
    if len(TUNE_SEEDS) == 1:
        loud = sorted(
            ((f, max((noise.get((a, f), np.nan) for a in algos), default=np.nan))
             for f in folds),
            key=lambda t: (-t[1] if np.isfinite(t[1]) else 0),
        )
        worst = [f"{f} (sd {sd:.3f})" for f, sd in loud[:2] if np.isfinite(sd) and sd > 0]
        w("- **The set was chosen from a single tuning draw.** On "
          f"{', '.join(worst) or 'some folds'} that draw moves by more than the gaps "
          "between the arms, so a difference of that size on those folds is not an "
          "effect. The remaining folds are an order of magnitude quieter. Raising "
          "`TUNE_SEEDS` is the fix and nothing else in the module changes.\n")

    if raced:
        s = (irace or {}).get("settings", {})
        w(f"- **irace arms** raced on seeds {s.get('seeds')} (disjoint from the scoring seeds) "
          f"with a budget of {s.get('budget')} target calls per race and a {s.get('test')} at "
          f"confidence {s.get('confidence')}. Their elites were seen on several draws per "
          "dataset, so their in-sample scores carry less selection luck than a grid pick's, "
          "but they are still in-sample: section 3's LODO estimate is the number that "
          "transfers.\n")

    w("\n## Verdict summary\n\n")
    for name, good in verdicts.items():
        w(f"- {name}: {'PASS' if good else 'FAIL'}\n")
    return "".join(out)


def _values_label(values: dict | None) -> str:
    if not values:
        return "-"
    return ", ".join(f"{k}={_fmt_value(v)}" for k, v in values.items())


def _irace_section(algos, protocols, head_mean, mean_of, irace: dict) -> str:
    """Report 3c: each irace arm beside the grid arm that chooses from the same space."""
    out = []
    w = out.append
    settings = irace.get("settings", {})
    space = irace.get("space", "grid")
    w("\n## 3c. irace against the grid\n\n")
    w("Same search space, same per-dataset reference, same scoring seeds; only the "
      "selector differs. A grid arm takes the argmax (or the LODO consensus) of **one** "
      "tuning draw per point. An irace arm races the points over (dataset, seed) "
      f"instances -- seeds {settings.get('seeds')}, one block per seed across the tunable "
      f"datasets -- with a {settings.get('test')} at confidence "
      f"{settings.get('confidence')}, cost = -headroom, and ships the elite of the race "
      "over every tunable dataset; its LODO estimate races again without the dataset it "
      "is scored on. ")
    if space == "grid":
        w("Space `grid`: one race over every grid point (irace FAQ 12.11), so both "
          "columns choose from exactly the same candidates.\n\n")
    else:
        w("Space `continuous`: iterated racing over the grid's bounding box, so irace "
          "can pick between grid points; a headroom above 1 on a dataset means past "
          "that dataset's grid ceiling.\n\n")
    pairs = []
    for p in IRACE_SHIPS:
        if p in protocols:
            pairs.append((p, *(("arm1_raw_all", "arm2_raw_lodo") if p == "irace_raw"
                               else ("sf_all", "arm3_sf_lodo"))))
    for p in sorted(protocols):
        if p.startswith("p2r_irace_") and not p.endswith(EST):
            grid_p2r = "p2r_" + p[len("p2r_irace_"):]
            pairs.append((p, grid_p2r, grid_p2r))
    w("| algo | irace arm | grid arm | in-sample headroom: grid | irace | "
      "LODO headroom: grid | irace | in-sample ARI: grid | irace |\n"
      "|---|---|---|---|---|---|---|---|---|\n")
    for algo in algos:
        for p, grid_ship, grid_est in pairs:
            w(f"| {algo} | {arm_label(p)} | {arm_label(grid_ship)} | "
              f"{_f(head_mean(algo, grid_ship))} | {_f(head_mean(algo, p))} | "
              f"{_f(head_mean(algo, grid_est + EST))} | {_f(head_mean(algo, p + EST))} | "
              f"{_f(mean_of(algo, grid_ship))} | {_f(mean_of(algo, p))} |\n")
    w("\nThe LODO grid column is arm 2's rounds for irace raw and arm 3's for irace "
      "ladder: the same selection rule per round, with an argmax in place of a race. "
      "Read every gap against the pooled resolution under 3b-i before calling it an "
      "effect.\n")

    results = irace.get("results") or {}
    if results:
        w("\n### The races\n\n"
          "| algo | arm | instances | rounds it serves | configurations | target calls "
          "(budget) | new runs | rejected | elite | elite seen on | elite headroom there |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|\n")
        for (algo, protocol), per_round in sorted(results.items()):
            seen: dict[int, tuple[dict, list[str]]] = {}
            for name, res in per_round.items():
                if res is not None:
                    seen.setdefault(id(res), (res, []))[1].append(name)
            for res, names in seen.values():
                opt = res.get("opt_folds") or []
                top = (res.get("elites") or [{}])[0]
                head = top.get("mean_headroom")
                w(f"| {algo} | {arm_label(protocol)} | {len(opt)} dataset"
                  f"{'' if len(opt) == 1 else 's'} x "
                  f"{len(res.get('seeds') or [])} seeds ({', '.join(opt)}) | "
                  f"{', '.join(names)} | {res.get('n_configurations')} | "
                  f"{res.get('target_calls')} ({res.get('budget')}) | "
                  f"{res.get('runs_executed')} | {res.get('n_rejected')} | "
                  f"{_values_label(top.get('values'))} | {top.get('n_instances', '-')} | "
                  f"{_f(head) if isinstance(head, (int, float)) else '-'} |\n")
        w("\n*Target calls* are irace's experiments; a call answered from `runs.jsonl` "
          "costs nothing, so *new runs* is what the race actually spent (a P2R call runs "
          "a whole ladder). *Elite seen on* is how many instances the shipped elite was "
          "evaluated on -- a grid pick is one draw per dataset.\n")
    return "".join(out)


def _irace_how(summary: dict) -> str:
    """How an irace arm reached its set, for report section 1."""
    if not summary:
        return "irace elite"
    head = summary.get("elite_mean_headroom")
    head = _f(head) if isinstance(head, (int, float)) else "-"
    return (f"irace elite over {', '.join(summary.get('opt_folds') or [])}: seen on "
            f"{summary.get('elite_instances')} instances, mean headroom {head} there; "
            f"{summary.get('target_calls')} target calls, {summary.get('runs_executed')} new runs")


def _level_value(grids_spec: dict, algo: str, level: int) -> str:
    axis, kind = GRAN[algo]
    value = grids_spec[algo]["levels"]["ladder"][level - 1]
    return f"c={value:g}" if kind == "scaled" else f"{value:g}"


def write_final_sets(path, finals, grids_spec, tuned_of, probes, folds) -> None:
    """The shipped sets, machine-readable: the spec and what it resolves to per dataset."""
    payload = {}
    for protocol, per_algo in finals.items():
        for algo, info in per_algo.items():
            cand = info["config"]
            entry = {
                "arm": arm_label(protocol),
                "set": spec_label(cand, grids_spec, tuned_of[algo],
                                  level=not protocol.startswith("p2r_")),
                "config_id": cand["config_id"],
                "grid": cand["grid"],
            }
            for key in ("votes", "rounds", "rule", "criterion", "phi", "theta", "irace"):
                if key in info:
                    entry[key] = info[key]
            if protocol.startswith("p2r_"):
                entry["level_per_dataset"] = info["levels"]
                entry["resolved"] = {}
                for fold, level in info["levels"].items():
                    cell = {**cand, "granularity_level": level,
                            "g": grids_spec[algo]["levels"]["ladder"][level - 1]}
                    entry["resolved"][fold] = resolve_candidate(cell, probes[fold])
            else:
                entry["resolved"] = {f: resolve_candidate(cand, probes[f]) for f in folds}
            payload.setdefault(algo, {})[protocol] = entry
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


# --- main ------------------------------------------------------------------
def set_min_throughput(value: float) -> None:
    """Override the cost floor used by report section 3b-iii."""
    global MIN_THROUGHPUT_SPS
    MIN_THROUGHPUT_SPS = float(value)


def set_gate(min_ceiling: float, min_ceiling_gain: float) -> None:
    """Override the tunability gate (`--min-ceiling`, `--min-ceiling-gain`).

    The two numbers are a research decision, not a measurement; the report
    prints whichever values were in force next to the noise quantities that
    could inform them (section 3b-i).
    """
    global MIN_CEILING, MIN_CEILING_GAIN
    MIN_CEILING = float(min_ceiling)
    MIN_CEILING_GAIN = float(min_ceiling_gain)


def _irace_args(args) -> tuple[list[str], tuple | None, dict]:
    """Validate the --irace* flags: (arms, the P2R reading, settings for `race`)."""
    from . import irace_tune

    arms = [a.strip() for a in args.irace.split(",") if a.strip()]
    bad = [a for a in arms if a not in irace_tune.IRACE_ARMS]
    if bad:
        raise SystemExit(f"--irace: unknown arm(s) {bad}; choose from {irace_tune.IRACE_ARMS}")
    reading = None
    if "p2r" in arms:
        readings = parse_p2r(args.irace_p2r or irace_tune.IRACE_P2R_DEFAULT)
        if len(readings) != 1:
            raise SystemExit("--irace-p2r takes exactly one rule:criterion")
        reading = readings[0]
    seeds = (tuple(int(s) for s in args.irace_seeds.split(",") if s.strip())
             if args.irace_seeds else irace_tune.IRACE_SEEDS)
    clash = sorted(set(seeds) & set(FINAL_SEEDS))
    if clash:
        raise SystemExit(f"--irace-seeds {clash} are scoring seeds {list(FINAL_SEEDS)}: a set "
                         "would be scored on the draws that chose it")
    test = args.irace_test or irace_tune.IRACE_TEST
    if test not in irace_tune.IRACE_TESTS:
        raise SystemExit(f"--irace-test must be one of {irace_tune.IRACE_TESTS}")
    settings = {
        "seeds": seeds,
        "budget": args.irace_budget or irace_tune.IRACE_BUDGET,
        "test": test,
        "confidence": args.irace_confidence or irace_tune.IRACE_CONFIDENCE,
        "first_test": args.irace_first_test or irace_tune.IRACE_FIRST_TEST,
        "each_test": irace_tune.IRACE_EACH_TEST,
        "rng_seed": irace_tune.IRACE_RNG_SEED,
        "force": bool(args.irace_force),
    }
    return arms, reading, settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", default=",".join(ALGOS))
    parser.add_argument("--folds", default="", help="explicit folds; overrides --groups")
    parser.add_argument(
        "--groups",
        default="real",
        help="fold groups to use when --folds is not given. Default is real only: the "
        "synthetic folds shared one generator and seed and are being regenerated",
    )
    parser.add_argument("--grids", default="",
                        help="JSON {algo: {levels|shape: {raw|ladder: ...}}} overriding GRIDS")
    parser.add_argument("--n-samples", type=int, default=0, help="0 = full stream")
    parser.add_argument("--out-dir", default="",
                        help=f"where runs.jsonl and the report go (default {OUT_DIR})")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("all", "tune", "irace", "score"),
        default="all",
        help="tune: only the grid runs (no report); irace: only the irace races of --irace "
        "(needs the tune rows); score: only the re-runs of picked configs, then the "
        "report; all: every phase in that order",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-queue runs whose latest row is not ok (e.g. timeouts caused by too low "
        "a --timeout); the retry supersedes the failed row",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="list the runs that would execute, run nothing, write nothing")
    parser.add_argument(
        "--aggregate",
        choices=AGGREGATES,
        default="headroom",
        help="how scores are pooled across datasets when one set is chosen. "
        "headroom (default): share of each dataset's own default-to-ceiling range, pooled "
        "only over the datasets that passed the tunability gate. "
        "mean: mean ARI (the largest-ARI dataset decides). "
        "rank: mean per-dataset rank (removes the scale, keeps every dataset)",
    )
    parser.add_argument(
        "--min-ceiling", type=float, default=MIN_CEILING,
        help=f"tunability gate: smallest per-dataset ceiling ARI (default {MIN_CEILING:g}); "
        "a runnable placeholder -- see report 3b-i for what could justify a value",
    )
    parser.add_argument(
        "--min-ceiling-gain", type=float, default=MIN_CEILING_GAIN,
        help=f"tunability gate: smallest ceiling-over-default gain (default "
        f"{MIN_CEILING_GAIN:g}); a runnable placeholder -- report 3b-i prints the gain the "
        "seed noise lets each dataset detect",
    )
    parser.add_argument(
        "--irace", default="",
        help="comma list of arms to also choose with irace instead of an argmax: "
        "raw, ladder, p2r. Needs R with the irace package. Empty (default) disables",
    )
    parser.add_argument("--irace-space", choices=("grid", "continuous"), default="grid",
                        help="grid: one race over every grid point; continuous: iterated "
                        "racing over the grid's bounding box")
    parser.add_argument("--irace-p2r", default="",
                        help="the one rule:criterion P2R reading irace races "
                        "(default R_knee:matr_trace_win, a placeholder)")
    parser.add_argument("--irace-budget", type=int, default=0,
                        help="maxExperiments per race (0 = irace_tune.IRACE_BUDGET)")
    parser.add_argument("--irace-seeds", default="",
                        help="instance seeds, one block each, disjoint from the scoring "
                        "seeds (default irace_tune.IRACE_SEEDS)")
    parser.add_argument("--irace-test", default="",
                        help="irace testType (default t-test; F-test ignores headroom)")
    parser.add_argument("--irace-first-test", type=int, default=0,
                        help="blocks before the first elimination test "
                        "(0 = irace_tune.IRACE_FIRST_TEST)")
    parser.add_argument("--irace-confidence", type=float, default=0.0,
                        help="confidence of irace's elimination test (0 = irace's 0.95)")
    parser.add_argument("--irace-force", action="store_true",
                        help="re-race even when a finished race with the same inputs exists")
    parser.add_argument("--rscript", default="",
                        help="path to Rscript (default $RSCRIPT, then PATH)")
    parser.add_argument(
        "--min-throughput",
        type=float,
        default=MIN_THROUGHPUT_SPS,
        help="samples/s floor for the constrained ceiling in report section 3b-iii "
        f"(default {MIN_THROUGHPUT_SPS:g}); a runnable placeholder, not a recommendation",
    )
    parser.add_argument(
        "--p2r",
        default=P2R_DEFAULT,
        help="comma list of rule:criterion P2R readings; empty string disables",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_min_throughput(args.min_throughput)
    set_gate(args.min_ceiling, args.min_ceiling_gain)
    p2r = parse_p2r(args.p2r)
    if args.out_dir:
        set_out_dir(args.out_dir)
    irace_arms, irace_reading, irace_settings = _irace_args(args)
    race_now = bool(irace_arms) and not args.report_only and not args.dry_run \
        and args.phase in ("all", "irace")
    rscript = None
    if race_now:
        # Fail before the grid runs, not after them, when R or irace is missing.
        from . import irace_tune

        rscript = irace_tune.find_rscript(args.rscript or None)
        print(f"irace {irace_tune.irace_version(rscript)} via {rscript}: arms {irace_arms}, "
              f"space {args.irace_space}")

    # The hash seed every run needs (RIVER_BUGS.md BUG-2) is pinned by
    # build_matrix.execute for each subprocess, so nothing to set here.
    require(PROBE_JSON, "Stage 2 (probe.py)")
    probes = read_json(PROBE_JSON)
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    wanted_groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    folds = (
        [f.strip() for f in args.folds.split(",") if f.strip()]
        if args.folds
        else [f for f in FOLDS if FOLD_GROUPS[f] in wanted_groups]
    )
    missing = [f for f in folds if f not in probes]
    if missing:
        raise SystemExit(f"no probe for {missing}; run stream.py and probe.py for them first")
    groups = {f: FOLD_GROUPS.get(f, "other") for f in folds}
    sizes = {g: list(groups.values()).count(g) for g in set(groups.values())}
    if any(n < 3 for n in sizes.values()):
        raise SystemExit(f"LODO within a group needs at least 3 folds per group; got {sizes}")
    requested = args.n_samples or "full"

    grids_spec = merge_grids(GRIDS, json.loads(args.grids) if args.grids else None)
    unknown = [a for a in algos if a not in grids_spec]
    if unknown:
        raise SystemExit(f"no grid for {unknown}; add it to GRIDS or pass --grids")
    grids: dict[str, dict[str, list[dict]]] = {"raw": {}, "ladder": {}}
    rejected: dict[str, dict[str, list[dict]]] = {"raw": {}, "ladder": {}}
    for a in algos:
        for kind in ("raw", "ladder"):
            grids[kind][a], rejected[kind][a] = build_grid(a, kind, grids_spec[a])
    tuned_of = {a: tuned_params(a, grids_spec) for a in algos}
    defaults = {a: default_candidate(a) for a in algos}
    tuning = [c for grid in grids.values() for cands in grid.values() for c in cands]
    candidates = {(c["algo"], c["config_id"]): c for c in [*tuning, *defaults.values()]}
    grid_of = {key: c["grid"] for key, c in candidates.items()}
    for a in algos:
        print(f"{a}: raw {len(grids['raw'][a])} points ({len(rejected['raw'][a])} rejected), "
              f"ladder {len(grids['ladder'][a])} points ({len(rejected['ladder'][a])} rejected); "
              f"tuned: {', '.join(tuned_of[a])}")

    running = not args.report_only
    if running and not args.dry_run:
        print("materialising permutations")
        extra = irace_settings["seeds"] if irace_arms else ()
        for fold in folds:
            for seed in dict.fromkeys((*TUNE_SEEDS, *FINAL_SEEDS, *extra)):
                materialise(fold, shuffle_seed_for(seed))

    if running and args.phase in ("all", "tune"):
        # The default runs at the tuning seeds too. It is not a candidate -- it
        # never enters a grid and cannot be picked -- but `fold_reference` needs
        # its score on the same seed as everything else in the selection frame,
        # or `--aggregate headroom` has no baseline to measure against and the
        # tunability gate cannot be evaluated. Cost: one run per (algo, fold).
        tune_points = [*tuning, *defaults.values()]
        print(f"tuning: {len(tuning)} grid points (+{len(defaults)} defaults, as the "
              f"headroom baseline) x {len(folds)} folds "
              f"x {len(TUNE_SEEDS)} seed(s) {list(TUNE_SEEDS)}")
        run_jobs(
            [_job(c["algo"], c, f, seed, args.n_samples, probes, tuned_of[c["algo"]])
             for c in tune_points for f in folds for seed in TUNE_SEEDS],
            args.workers,
            args.timeout,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
    if running and args.phase == "tune":
        print("tuning phase finished; run again with --phase score to re-run the picks "
              "and write the report")
        return

    df = load_runs(requested, grid_of, candidates, probes)
    # One averaged row per cell before any selection touches it; see `seed_mean`.
    # The default row stays in: `fold_reference` needs it. It cannot be picked,
    # because every selector is handed the `raw` or `ladder` slice and the
    # default's grid is neither.
    tune = seed_mean(df[df.seed.isin(TUNE_SEEDS)
                        & df.algo.isin(algos) & df.dataset.isin(folds)])
    # An older runs.jsonl holds scoring rows at seeds that are now tuning seeds,
    # and it holds them only for the configs that were picked last time. Those
    # cells would then be averaged over more draws than their competitors and
    # would look steadier for a reason that is not about the configuration. The
    # count is on every row, so say it rather than let it pass.
    missing_default = [
        (a, f) for a in algos for f in folds
        if tune[(tune.algo == a) & (tune.dataset == f) & (tune.grid == "default")].empty
    ]
    if missing_default:
        print(f"WARNING: no tuning-seed run of the default for {len(missing_default)} "
              "(algo, fold) pairs. `--aggregate headroom` and the tunability gate need "
              "it; run `--phase tune` to fill them in.")
    if "n_tune_seeds" in tune.columns and tune.n_tune_seeds.nunique() > 1:
        counts = tune.n_tune_seeds.value_counts().sort_index()
        print("WARNING: tuning cells are averaged over different numbers of seeds "
              f"({dict(counts)}); expected {len(TUNE_SEEDS)} everywhere. Run "
              "`--phase tune` to fill the gaps before trusting the picks.")
    refs = {a: fold_reference(tune[tune.algo == a]) for a in algos}
    picks, finals = all_picks(tune, grids, defaults, algos, folds, groups, p2r, args.aggregate,
                              refs)

    irace_info = None
    if irace_arms:
        results = irace_results(
            algos=algos, arms=irace_arms, space=args.irace_space, p2r_reading=irace_reading,
            folds=folds, groups=groups, refs=refs, probes=probes, grids_spec=grids_spec,
            tuned_of=tuned_of, n_samples=args.n_samples, timeout=args.timeout,
            workers=args.workers, rscript=rscript, settings=irace_settings, run=race_now)
        cells = irace_cells(results, grids_spec, args.irace_space)
        # Off-grid candidates (continuous space) join the lookup under their own
        # grid label, so their rows load for scoring but never reach a grid table.
        for entry in cells.values():
            for cands in [entry["ship"] or [], *entry["est"].values()]:
                for cand in cands:
                    key = (cand["algo"], cand["config_id"])
                    if cand.get("irace") and key not in candidates:
                        candidates[key] = cand
                        grid_of[key] = "irace"
        ladder_jobs = irace_ladder_jobs(cells, folds, args.n_samples, probes, tuned_of)
        if ladder_jobs and running:
            print("irace P2R: tuning-seed ladders needed to read the level on every dataset")
            run_jobs(ladder_jobs, args.workers, args.timeout,
                     retry_failed=args.retry_failed, dry_run=args.dry_run)
        df = load_runs(requested, grid_of, candidates, probes)
        irace_pk, irace_fn = irace_picks(cells, results, df, folds, irace_reading)
        picks.update(irace_pk)
        finals.update(irace_fn)
        irace_info = {"results": results, "space": args.irace_space,
                      "settings": {k: v for k, v in irace_settings.items() if k != "force"}}
    if running and args.phase == "irace":
        print("irace phase finished; run again with --phase score to re-run the picks and "
              "write the report")
        return

    if running:
        wanted = {
            (algo, cand["config_id"], fold): cand
            for chosen in picks.values()
            for (algo, fold), cand in chosen.items()
        }
        print(f"scoring: {len(wanted)} picked (algo, config, fold) x seeds {list(FINAL_SEEDS)}")
        if args.dry_run:
            print("  (picks come from the tuning rows on disk now; a tuning retry that "
                  "has not run yet can still change them)")
        run_jobs(
            [
                _job(algo, cand, fold, seed, args.n_samples, probes, tuned_of[algo])
                for (algo, _, fold), cand in wanted.items()
                for seed in FINAL_SEEDS
            ],
            args.workers,
            args.timeout,
            retry_failed=args.retry_failed,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            return
        df = load_runs(requested, grid_of, candidates, probes)

    scores = score_table(df, picks, probes, tuned_of)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{requested}" if args.aggregate == "mean" else f"{requested}_{args.aggregate}"
    scores.to_csv(OUT_DIR / f"scores_{tag}.csv", index=False)
    write_final_sets(OUT_DIR / f"final_sets_{tag}.json", finals, grids_spec, tuned_of,
                     probes, folds)
    _ok(tune[tune.grid != "default"]).pivot_table(
        index=["algo", "grid", "dataset", "theta"], columns="granularity_level", values=OBJ
    ).to_csv(OUT_DIR / f"tune_grid_{requested}.csv")
    report = build_report(tune, scores, folds, probes, requested, grids, grids_spec,
                          rejected, tuned_of, finals, args.aggregate, irace=irace_info)
    path = OUT_DIR / ("report.md" if tag == "full" else f"report_{tag}.md")
    path.write_text(report, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
