"""Paths, protocol constants and helpers shared by the conf_pipeline modules.

Importing this module puts the repo root and the sibling river tree on
``sys.path``, exactly as ``hpo_pipeline._common`` does; the two pipelines do not
import each other.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.algorithms.adapter import build_river_path  # noqa: E402

build_river_path()

from benchmark.tuning.signals import SELECTABLE  # noqa: E402

# --- artifacts -----------------------------------------------------------------
OUT = PIPELINE_ROOT / "out"
WINDOWS_JSONL = OUT / "windows.jsonl"
RUNS_JSONL = OUT / "runs.jsonl"
#: ``--log-members``: one file per job, ``<algo>__<fold>__seed<seed>__<arm>[__cfg<id>].jsonl``,
#: so concurrent children never append to the same file.
MEMBERS_DIR = OUT / "members"
REPORT_DIR = OUT / "report"
#: The a-posteriori best static configuration per (algo, fold), written by the sweep
#: into its out dir; ``run.py`` reads ``<out-dir>/oracle_static.json`` unless told otherwise.
ORACLE_JSON = OUT / "oracle_static.json"

# --- protocol constants (every one of them is a CLI flag) -------------------------
#: Window size in samples: the unit of evaluation, replacement and warm start.
H = 1000
#: Ensemble size and challengers per window (confStream's e_size, e_test).
E_SIZE = 20
E_TEST = 10
#: Probability that a child restarts at the initial exploration width; with
#: e_size = 20 that is 0.2 resets per window, the paper's rate.
P_RESET = 0.01
#: The partition the tuner scores and every ``*_win`` reading reads (``confstream.SCORE_PARTITIONS``):
#: ``snapshot`` since 2026-11-08 (docs/SEQUENTIAL_PARTITION_AUDIT.md section 7.7); the runs before it,
#: ``out/full`` among them, are ``sequential``.
SCORE_PARTITION = "snapshot"
#: The fading factor is set per fold from the number of windows: sigma halves
#: after ``HALF_LIFE_FRAC`` of the stream's windows. A fixed lambda means
#: "converged in the first 5% of the stream" on a 2M-sample stream and "never
#: converged" on a 50k one.
HALF_LIFE_FRAC = 0.10
#: Feasibility caps per window, the offline ones of hpo_pipeline made local.
RUNTIME_FACTOR = 30.0
MAX_MEMORY_MB = 2048.0
MAX_MICRO_CLUSTERS = 10_000
#: Distance unit: median pairwise distance over a subsample of the calibration
#: window(s). Grown one window at a time while it is 0 (a prefix of identical
#: rows), up to this many windows.
SCALE_SUBSAMPLE = 2000
SCALE_MAX_WINDOWS = 20

ALGOS: tuple[str, ...] = (
    "DenStream", "CluStream", "HPStream", "RepStream", "DBSTREAM", "STREAMKMeans", "SOStream", "BIRCH",
)
REAL_FOLDS: tuple[str, ...] = (
    "covtype", "kdd99", "insects_abrupt_balanced", "insects_abrupt_imbalanced",
    "insects_gradual_balanced", "insects_gradual_imbalanced", "insects_incremental_balanced",
    "insects_incremental_abrupt_balanced", "insects_incremental_reoccurring_balanced",
    "electricity", "shuttle",
)
SYNTH_FOLDS: tuple[str, ...] = (
    "scale_drift", "density_drift", "k_drift", "mixed_drift",
    "static_overlap", "incremental_drift", "evolving",
)
FOLDS: tuple[str, ...] = REAL_FOLDS + SYNTH_FOLDS
#: Synthetic folds whose best parameters change at known points: the scale of every
#: coordinate, the cluster radius, the number of clusters, or the last two together.
PARAM_DRIFT_FOLDS: tuple[str, ...] = ("scale_drift", "density_drift", "k_drift", "mixed_drift")
#: The three groups results are reported by: their attainable gains differ by an order of
#: magnitude, so a figure pooled over all folds is dominated by the synthetic ones.
STREAM_GROUPS: tuple[str, ...] = ("real", "param_drift", "other_synth")
#: Streams for checking that the findings do not hang on drift the study designed (X7), run
#: with ``--folds`` into an out dir of their own; ``FOLDS``, what the full run ran, leaves
#: them out. Real streams with natural drift (``benchmark.datasets.real.drift_streams``)...
NATURAL_DRIFT_FOLDS: tuple[str, ...] = (
    "rialto", "sensor", "insects_incremental_imbalanced", "insects_incremental_reoccurring_imbalanced",
)
#: ... and variants of the parameter-drift design: a smaller and a larger scale jump, changes
#: at random points, a gradual change (``param_drift.DRIFT_VARIANT_SCENARIOS``).
DRIFT_VARIANT_FOLDS: tuple[str, ...] = (
    "scale_drift_x1_5", "scale_drift_x8", "random_drift", "scale_drift_gradual",
)
X7_FOLDS: tuple[str, ...] = NATURAL_DRIFT_FOLDS + DRIFT_VARIANT_FOLDS
#: The scaling-sensitivity twins of the parameter-drift folds: the same streams, standardised as the real
#: ones are (``stream_scaling_of``). Their group is ``SENSITIVITY_GROUP``; the reports read them apart
#: (``report --scaling-sensitivity``), never in a main table, ranking or pool.
SCALED_TWIN_SUFFIX = "_std"
SCALED_TWIN_FOLDS: tuple[str, ...] = tuple(f"{f}{SCALED_TWIN_SUFFIX}" for f in PARAM_DRIFT_FOLDS)
SENSITIVITY_GROUP = "param_drift_scaled"


def twin_base(fold: str) -> str:
    """The parameter-drift fold a scaled twin copies; any other fold itself."""
    return fold[: -len(SCALED_TWIN_SUFFIX)] if fold in SCALED_TWIN_FOLDS else fold


def _e8_folds() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """E8's drift streams (the same-id variants of ``k_drift``/``mixed_drift`` and the factorial
    design) and its stationary controls (``param_drift`` module docstring), run with ``--folds`` into
    an out dir of their own; ``FOLDS`` leaves them out. Imported lazily, as ``load_fold`` does."""
    from benchmark.datasets.synthetic.param_drift import (
        E8_FACTORIAL_SCENARIOS, E8_SAME_ID_SCENARIOS, E8_STATIONARY_SCENARIOS,
    )
    return E8_SAME_ID_SCENARIOS + E8_FACTORIAL_SCENARIOS, E8_STATIONARY_SCENARIOS


def stream_group(fold: str) -> str:
    """``real``, ``param_drift`` or ``other_synth``; a fold in none of the lists raises,
    so a new stream is assigned its group explicitly rather than by a guess. The natural-drift
    streams are real ones; the drift variants change their best parameters as the
    parameter-drift folds do. E8's drift streams join ``param_drift`` and its stationary controls
    ``other_synth`` (a grouping for the reports, the researchers' to change)."""
    if fold in SCALED_TWIN_FOLDS:
        return SENSITIVITY_GROUP
    if fold in REAL_FOLDS or fold in NATURAL_DRIFT_FOLDS:
        return "real"
    if fold in PARAM_DRIFT_FOLDS or fold in DRIFT_VARIANT_FOLDS:
        return "param_drift"
    if fold in SYNTH_FOLDS:
        return "other_synth"
    e8_drift, e8_controls = _e8_folds()
    if fold in e8_drift:
        return "param_drift"
    if fold in e8_controls:
        return "other_synth"
    raise ValueError(f"fold {fold!r} has no stream group: add it to REAL_FOLDS or SYNTH_FOLDS")


#: Drift variants that only change a parameter of one design keep that design's block; the same-id
#: variants are the same points as their design, so they are the same block too.
_VARIANT_OF: dict[str, str] = {
    "scale_drift_x1_5": "scale_drift", "scale_drift_x8": "scale_drift", "scale_drift_gradual": "scale_drift",
    "k_drift_same_ids": "k_drift", "mixed_drift_same_ids": "mixed_drift",
}


def source_block(fold: str) -> str:
    """The independent source a fold comes from, the unit a bootstrap or a leave-one-out
    resamples: every INSECTS variant is one recording campaign (``"insects"``), a drift
    variant that changes one parameter of a design is that design, any other fold is its
    own. A fold without a stream group raises, as in ``stream_group``."""
    stream_group(fold)
    fold = twin_base(fold)
    if fold.startswith("insects_"):
        return "insects"
    return _VARIANT_OF.get(fold, fold)

#: Seeds of every arm. Three, the same for every arm, so every comparison
#: between arms sees the same seed noise (decided 2026-10-07 after the pilot,
#: for budget: five seeds cost 900-1 600 CPU-h).
SEEDS: tuple[int, ...] = (1, 2, 3)
#: How a job's randomness is seeded (docs/SEED_RULES.md). ``split``, the default since 2026-10-24:
#: the tuner, the models and the calibration of the distance unit each have a seed of their own.
#: ``shared``: one job seed drives all three, as in every run made before; kept to run beside them.
SEED_POLICIES: tuple[str, ...] = ("split", "shared")
#: The split policy's seeds (placeholders, every one a flag): the tuner's, repeated per online run;
#: the models' of the online arms, one per out dir; the calibration's, a constant, so every arm and
#: every tuner seed reads the stream in the same distance unit. Another range than the shared
#: policy's 1-3, so a row says which policy it comes from at a glance.
TUNER_SEEDS: tuple[int, ...] = (101, 102, 103)
MODEL_SEED = 201
CALIBRATION_SEED = 0
#: The tuning regimes every (algo, fold, seed) is run under.
ARMS: tuple[str, ...] = ("default", "confstream", "frozen_prefix")
#: The online tuner selecting by another label-free signal than the silhouette
#: (``benchmark.tuning.signals``): one arm per signal, e.g. ``confstream_dbcv``.
SIGNAL_ARMS: tuple[str, ...] = tuple(f"confstream_{s}" for s in SELECTABLE if s != "silhouette")
#: Arms the driver still runs on request (``--arms``) and the report still reads, but that
#: the protocol does not include. ``random_online`` left it on 2026-10-12: it cost about
#: half the full budget, and CluStream's runs of it risked the 12 h timeout.
#: ``frozen_reinit`` freezes the parameters at ``FREEZE_AFTER_WINDOW`` like ``frozen_prefix``
#: but keeps renewing the model as the tuner does (fresh warm-started copies competing by
#: silhouette), which separates new models from new parameters after a drift. The signal
#: arms test candidate replacements for the silhouette.
#: ``scaled_default`` (E1) runs River's defaults with every absolute-distance parameter
#: multiplied by the distance unit ``s`` (``spaces.scaled_defaults``): one fixed configuration,
#: never tuned, which tells failures of units from failures of the configuration.
#: The mechanism study's branches (E3, ``confstream.FREEZE_MODES``): each runs the tuner up to
#: ``FREEZE_AFTER_WINDOW`` exactly as ``confstream`` and ``frozen_prefix`` do, then ``frozen_ensemble``
#: keeps the whole portfolio and only goes on selecting among its existing models,
#: ``portfolio_renewal`` keeps the portfolio's configurations and renews their models (each fresh model
#: meets only the model of its own configuration), and ``random_after_prefix`` goes on tuning with
#: children drawn uniformly. Under a deterministic protocol (``--no-runtime-gate``, the default
#: tie-break) and one tuner and model seed, every branch and ``confstream``/``frozen_prefix``/
#: ``frozen_reinit`` share the same ensemble up to the boundary (``followup prefix-identity`` checks it).
BRANCH_ARMS: dict[str, str] = {"frozen_ensemble": "ensemble", "portfolio_renewal": "portfolio",
                               "random_after_prefix": "random"}
#: ``frozen_prefix_matched`` (E3, equal budget): one configuration frozen at ``FREEZE_AFTER_WINDOW`` like
#: ``frozen_prefix``, but tuned before the freeze with ``e_size`` and ``e_test`` scaled by ``budget_factor`` so
#: that its models evaluated over the whole stream equal ``confstream``'s: the same budget spent before the
#: freeze instead of spread over the stream. It parts from the other branches on window 0, so it shares no
#: prefix with them and ``followup prefix-identity`` leaves it out.
MATCHED_ARMS: tuple[str, ...] = ("frozen_prefix_matched",)
OPTIONAL_ARMS: tuple[str, ...] = (("random_online", "frozen_reinit", "scaled_default") + tuple(BRANCH_ARMS)
                                  + MATCHED_ARMS + SIGNAL_ARMS)
#: Single-configuration arms driven by a file: the sweep's points and its winner.
EXTRA_ARMS: tuple[str, ...] = ("oracle_static", "sweep")
#: Diagnostic arms (E4), never deployable and in no report's default list: the tuner with its
#: selection score replaced by the ARI of the window just processed
#: (``benchmark.tuning.confstream.ORACLE_SIGNALS``), with and without the silhouette's gate.
DIAGNOSTIC_ARMS: tuple[str, ...] = ("confstream_ari_oracle", "confstream_ari_oracle_constrained")
#: Arms that run one configuration, never challenged: no tuning happens in them.
FIXED_ARMS: tuple[str, ...] = ("default", "scaled_default") + EXTRA_ARMS
#: Every fold is cut at this many samples (the driver caps; loaders never see it).
N_SAMPLES_CAP = 100_000
#: ``frozen_prefix`` stops tuning after this window.
FREEZE_AFTER_WINDOW = 20
#: Every arm's tail metrics start this many windows after the freeze window, so all arms are compared on
#: the same region and every online--frozen contrast is strictly post-freeze: under the lagged reading,
#: window ``t0`` is read from the incumbent named after ``t0 - 1`` and window ``t0 + 1`` from the one named
#: after ``t0``, the same model under both regimes; ``t0 + 2`` is the first window a decision taken after
#: the freeze can reach. Runs before 2026-11-08 started their tail at the freeze window itself.
POST_FREEZE_LAG = 2


def tail_start(freeze_after_window: int) -> int:
    """The first window of the evaluation region of a run frozen after ``freeze_after_window``."""
    return int(freeze_after_window) + POST_FREEZE_LAG
SWEEP_N_POINTS = 32
#: The readings the static oracle is picked on, in order of preference: the tail, the
#: region every arm is compared on, then the whole stream for a stream too short to
#: have a tail. Picking on the whole stream would let a strong start win a ceiling
#: that is read on the tail.
ORACLE_METRICS: tuple[str, ...] = ("lag_ari_win_tail_mean", "lag_ari_win_mean")
MIN_CEILING_GAIN = 0.05
#: Length of a synthetic fold when no cap is given.
SYNTH_N_SAMPLES = 120_000
#: Seed of every synthetic generator, apart from the tuner's and the models' seeds: what
#: every synthetic stream of the full run was drawn with. One per out dir (``--data-seed``),
#: so the reports, which key runs by (algo, fold, seed, arm), never mix two draws of a stream.
DATA_SEED = 42

#: No scaler inside a model's adapter: a stream is scaled once, before every model and every reading
#: (``STREAM_SCALING``), so all models of a run share their coordinates.
SCALER = "none"
#: The scaling rule of the streams (``benchmark.datasets.scaling``), chosen on 2026-11-09 before any run
#: under it: ``protocol`` standardises every stream with River's ``StandardScaler(window_size=W)`` (as
#: ``standard_safe``, RIVER_BUGS.md BUG-1), each point learned then transformed, except the
#: parameter-drift group, which keeps its supplied coordinates: those streams are built to change the best
#: parameters (the coordinate scale, the radius, the cluster count), and a rolling standardisation would
#: undo the change they are built to produce. Their scaled twins (``SCALED_TWIN_FOLDS``) are the
#: sensitivity check. ``rolling_standard`` and ``none`` apply one scaling to every stream (``none`` as every
#: run before 2026-11-09). ``SCALING_WINDOW`` None means one window ``h``: a placeholder.
STREAM_SCALING = "protocol"
STREAM_SCALING_RULES: tuple[str, ...] = ("protocol", "rolling_standard", "none")
SCALING_WINDOW: int | None = None


def stream_scaling_of(fold: str, rule: str = STREAM_SCALING) -> str:
    """The scaling ``rule`` gives ``fold`` (``none`` or ``rolling_standard``): under ``protocol``, the
    parameter-drift group reads its supplied coordinates, its scaled twins and every other stream,
    a stream outside the protocol's lists included, are standardised."""
    if rule not in STREAM_SCALING_RULES:
        raise ValueError(f"scaling rule must be one of {STREAM_SCALING_RULES}, got {rule!r}")
    if rule != "protocol":
        return rule
    if fold in SCALED_TWIN_FOLDS:
        return "rolling_standard"
    try:
        group = stream_group(fold)
    except ValueError:
        return "rolling_standard"
    return "none" if group == "param_drift" else "rolling_standard"


def scaling_window(window: int | None, h: int) -> int:
    """The rolling window a stream is standardised over: ``window``, else one window ``h``."""
    return int(window) if window is not None else int(h)
#: ``cmm`` is MOA's CMM (``benchmark.evaluation.cmm``) and ``cmm_strict`` the same without
#: MOA's class merging and sphere excuse (``confstream.CMM_VARIANTS``); both read the points of
#: a window, not its labels alone, so they have no cumulative form. ``--no-cmm`` leaves them out.
EXTERNAL_METRICS = ("ari", "nmi", "purity", "cmm", "cmm_strict")
CMM_METRICS = ("cmm", "cmm_strict")
#: A window needs this many true classes for an external metric to say anything. With one,
#: ARI is 1 for a one-label prediction and 0 for any other, NMI likewise, and purity is 1
#: whatever the prediction: the reading measures "did it output one label", not quality,
#: and rewards a broken configuration. Such a window's external readings are recorded as
#: undefined (None) and so drop out of every mean. A definition, not a tunable.
MIN_TRUE_CLASSES = 2
#: Table A's DBSCAN eps over the lagged member's micro-cluster centres, as a
#: fraction of the calibration window's distance unit ``s_online`` -- the same
#: unit DenStream's own ``epsilon`` is read in, so the two DBSCANs are
#: comparable across algorithms and folds.
TABLE_A_EPS_FACTOR = 0.5

PYTHONHASHSEED = "0"
#: One thread per native library in every child: 8 workers each running
#: sklearn's KMeans/DBSCAN/silhouette would otherwise start a thread per core
#: apiece, and the incumbent's throughput would measure the contention.
THREAD_PINS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
MEMORY_LIMIT_BYTES = 4 * 1024**3
TIMEOUT_S = 6 * 3600


def evaluations(n_scored: int, n_prefix: int, e_size: int, e_test: int, frozen: bool) -> int:
    """Models evaluated over a run of ``n_scored`` scored windows: ``e_size`` on the first (the ensemble has
    no test set yet), ``e_size + e_test`` on each later one up to and including the freeze window (the
    ``n_prefix``-th scored window), then ``e_size + e_test`` again for an arm that goes on tuning, one for a
    frozen single model. ``frozen=False`` reads every window as tuning (``confstream``)."""
    if n_scored <= 0:
        return 0
    if not frozen:
        n_prefix = n_scored
    p = max(1, min(n_scored, n_prefix))
    return e_size + (p - 1) * (e_size + e_test) + (n_scored - p)


def matched_sizes(n_scored: int, n_prefix: int, e_size: int, e_test: int) -> tuple[int, int, float]:
    """``(e_size, e_test, factor)`` before the freeze for ``frozen_prefix_matched``: both sizes scaled by the
    factor ``k`` at which a frozen single model evaluates, over the run, as many models as ``confstream``
    (``evaluations``), ``k (E + (P - 1)(E + C)) + (n - P) = E + (n - 1)(E + C)``, each size rounded. At
    least the protocol's sizes."""
    target = evaluations(n_scored, n_scored, e_size, e_test, frozen=False)
    p = max(1, min(n_scored, n_prefix))
    prefix = e_size + (p - 1) * (e_size + e_test)
    k = max(1.0, (target - (n_scored - p)) / prefix) if prefix > 0 else 1.0
    return max(1, int(round(k * e_size))), int(round(k * e_test)), k


def lambda_for(n_windows: int, half_life_frac: float = HALF_LIFE_FRAC) -> float:
    """Fading factor whose half-life is ``half_life_frac`` of the windows."""
    half_life = max(1, int(round(half_life_frac * n_windows)))
    return 1.0 / half_life


def estimate_scale(xs, rng: np.random.Generator, subsample: int = SCALE_SUBSAMPLE) -> float:
    """Median pairwise Euclidean distance over (a subsample of) ``xs``."""
    if len(xs) < 2:
        return 0.0
    keys = sorted(xs[0].keys())
    X = np.asarray([[float(x.get(k, 0.0)) for k in keys] for x in xs], dtype=float)
    if len(X) > subsample:
        X = X[rng.choice(len(X), size=subsample, replace=False)]
    sq = (X**2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
    iu = np.triu_indices(len(X), k=1)
    return float(np.median(D[iu]))


#: Algorithms registered under a project name whose constructor is a River class.
RIVER_CLASS = {"DenStreamPatched": "DenStream", "DenStreamPatchedFast": "DenStream",
               "HPStreamStableLabels": "HPStream"}
#: Implementation variants run beside the River class they re-implement (spec
#: 4.6, table T7): never ranked with the protocol algorithms, read against
#: their reference instead. DenStreamPatchedFast fixes BUG-7/BUG-8 and returns
#: what DenStreamPatched returns at a fraction of the cost (GAP-3). The notebook's presets run these.
IMPL_VARIANTS = {"DenStreamPatchedFast": "DenStream"}
#: Variants that change only how a River class names its clusters (RIVER_BUGS.md GAP-6): never ranked in the
#: OTB tables, read against their reference there (``analyze.impl_table``), and in no preset or campaign
#: inventory unless named; a campaign that names one reads it under its own name, and a LOBO plan holding it
#: beside its reference is not eligible. HPStreamStableLabels is HPStream with labels that are never reused.
LABEL_VARIANTS = {"HPStreamStableLabels": "HPStream"}


def variant_algos() -> dict[str, str]:
    """Every variant the OTB readers never rank (``analyze``, ``robustness``, ``variants``, P1.1
    ``selection``): ``IMPL_VARIANTS`` and ``LABEL_VARIANTS``, read when called. The campaign readers
    (``analysis_support.run_table``, LOBO) drop the implementation variants only: a label variant a
    campaign names is read there under its own name."""
    return {**IMPL_VARIANTS, **LABEL_VARIANTS}
#: ``config_id`` of a sweep point that reruns an online arm's incumbent with its
#: exact parameters (``sweep.incumbent_candidates``): above every Sobol index.
INCUMBENT_ID_BASE = 1_000_000


def river_defaults(algo: str) -> dict:
    """River's constructor defaults, read off the signature (``seed`` left out)."""
    import inspect

    from river import cluster

    skip = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    params = inspect.signature(getattr(cluster, RIVER_CLASS.get(algo, algo)).__init__).parameters
    return {
        name: p.default
        for name, p in params.items()
        if name not in ("self", "seed")
        and p.kind not in skip
        and p.default is not inspect.Parameter.empty
    }


def load_fold(name: str, n_samples: int | None = None, data_seed: int = DATA_SEED):
    """The protocol folds in their natural order, plus the synthetic scenarios.

    Real loaders get the whole stream (the driver cuts it at ``n_samples``);
    synthetic streams are generated at ``n_samples``, or ``SYNTH_N_SAMPLES``, from
    ``data_seed``. ``fold_source`` describes the same branches for the manifest.
    """
    from benchmark.datasets import (
        PARAM_DRIFT_SCENARIOS, SCENARIOS, load_covtype, load_electricity, load_insects,
        load_kddcup99, load_shuttle, make_param_drift, make_scenario,
    )

    if name in SCALED_TWIN_FOLDS:
        # the same points as the fold it copies; the driver scales it (``stream_scaling_of``)
        return load_fold(twin_base(name), n_samples=n_samples, data_seed=data_seed)
    if name == "covtype":
        return load_covtype(quantitative_only=True)
    if name == "kdd99":
        return load_kddcup99(n_samples=None)
    if name == "shuttle":
        return load_shuttle()
    if name == "electricity":
        return load_electricity()
    if name in ("rialto", "sensor") or name.removeprefix("insects_") in _insects_extra():
        from benchmark.datasets.real.drift_streams import load_insects_extra, load_rialto, load_sensor

        if name == "rialto":
            return load_rialto()
        if name == "sensor":
            return load_sensor()
        return load_insects_extra(variant=name.removeprefix("insects_"))
    if name.startswith("insects_"):
        return load_insects(variant=name[len("insects_"):])
    if name in PARAM_DRIFT_SCENARIOS:
        return make_param_drift(name, n_samples=n_samples or SYNTH_N_SAMPLES, seed=data_seed)
    if name in SCENARIOS:
        return make_scenario(name, seed=data_seed, n_samples=n_samples or SYNTH_N_SAMPLES)
    raise KeyError(
        f"unknown fold {name!r}; protocol folds: {FOLDS}; "
        f"synthetic: {sorted(SCENARIOS) + sorted(PARAM_DRIFT_SCENARIOS)}"
    )


def is_synthetic(name: str) -> bool:
    """Whether ``name`` is drawn by a generator, so that a data seed reaches it."""
    from benchmark.datasets import PARAM_DRIFT_SCENARIOS, SCENARIOS

    name = twin_base(name)
    return name in PARAM_DRIFT_SCENARIOS or name in SCENARIOS


def fold_source(name: str) -> dict:
    """What ``load_fold(name)`` reads: the loader and its arguments, for the manifest.

    Kept beside ``load_fold``, branch for branch; the manifest also records a hash of the
    content each run consumed, which catches what a description cannot (a data file
    replaced under the same loader).
    """
    if name in SCALED_TWIN_FOLDS:
        return {**fold_source(twin_base(name)), "scaled_twin_of": twin_base(name)}
    real = {
        "covtype": ("load_covtype", {"quantitative_only": True}),
        "kdd99": ("load_kddcup99", {"n_samples": None}),
        "shuttle": ("load_shuttle", {}),
        "electricity": ("load_electricity", {}),
        "rialto": ("load_rialto", {}),
        "sensor": ("load_sensor", {}),
    }
    if name in real:
        loader, kwargs = real[name]
        return {"loader": loader, "kwargs": kwargs}
    if name.removeprefix("insects_") in _insects_extra():
        return {"loader": "load_insects_extra", "kwargs": {"variant": name.removeprefix("insects_")}}
    if name.startswith("insects_"):
        return {"loader": "load_insects", "kwargs": {"variant": name[len("insects_"):]}}
    from benchmark.datasets import PARAM_DRIFT_SCENARIOS, SCENARIOS

    if name in PARAM_DRIFT_SCENARIOS:
        return {"loader": "make_param_drift", "kwargs": {"name": name}}
    if name in SCENARIOS:
        return {"loader": "make_scenario", "kwargs": {"name": name}}
    raise KeyError(f"unknown fold {name!r}")


def _insects_extra() -> dict:
    from benchmark.datasets.real.drift_streams import INSECTS_EXTRA_VARIANTS

    return INSECTS_EXTRA_VARIANTS


def drift_points(fold: str, n_samples: int | None) -> tuple[int, ...]:
    """Sample indices where a parameter-drift fold changes regime; ``()`` otherwise."""
    from benchmark.datasets import PARAM_DRIFT_SCENARIOS, make_param_drift

    fold = twin_base(fold)
    if fold in PARAM_DRIFT_SCENARIOS:
        return make_param_drift(fold, n_samples=n_samples or SYNTH_N_SAMPLES).drift_points
    return ()


def best_static(rows, metrics: tuple[str, ...] = ORACLE_METRICS) -> tuple[dict, str] | None:
    """The winning fixed configuration among one (algo, fold)'s sweep rows, and the metric it won on.

    A configuration swept on several seeds competes with its mean over them
    (``seed_means``). The first of ``metrics`` that some configuration has a
    reading of decides; one without that reading never wins. Ties go to the
    smaller ``config_id`` (the earlier, cheaper point). None when no configuration
    has any of ``metrics``. With one seed per configuration this is the plain
    maximum over the ``ok`` rows.
    """
    for metric in metrics:
        cands = seed_means(rows, metric, also=metrics)
        if cands:
            best = max(cands, key=lambda r: (r[metric], -(r.get("config_id") or 0)))
            return best, metric
    return None


def seed_means(rows, metric: str, also: tuple[str, ...] = ()) -> list[dict]:
    """One row per configuration of one (algo, fold) that has ``metric`` on every seed swept.

    The seeds swept are those any row of the (algo, fold) ran on. A configuration
    that failed, or read nothing, on one of them is left out rather than judged on
    its luckier seeds: a maximum over configurations averaged on different numbers
    of seeds favours the noisiest. Each row is the configuration's row of its first
    seed with ``metric`` replaced by the mean, plus ``n_seeds``, ``seeds`` and
    ``<metric>_sd`` (None on one seed); every reading in ``also`` is replaced by its
    mean too, or None unless every seed has it.
    """
    seeds = sorted({r.get("seed") for r in rows})
    by_config: dict = {}
    for r in rows:
        if r.get("status") == "ok" and _reading(r.get(metric)) is not None:
            by_config.setdefault(r.get("config_id"), {}).setdefault(r.get("seed"), r)
    out = []
    for per_seed in by_config.values():
        if sorted(per_seed) != seeds:
            continue
        row = {**per_seed[seeds[0]], "n_seeds": len(seeds), "seeds": seeds}
        for m in dict.fromkeys((*also, metric)):
            vals = [_reading(per_seed[s].get(m)) for s in seeds]
            row[m] = float(np.mean(vals)) if None not in vals else None
        vals = [_reading(per_seed[s][metric]) for s in seeds]
        row[f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        out.append(row)
    return out


def _reading(v) -> float | None:
    """A finite number, or None (a missing key, None, NaN)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def child_env() -> dict:
    """The parent's environment with the hash seed and the thread counts pinned."""
    return {**os.environ, "PYTHONHASHSEED": PYTHONHASHSEED, **THREAD_PINS}


# --- where a run was made ------------------------------------------------------------
#: One record per command that ran jobs into an out dir: when, which command, and the machine
#: and library versions (``benchmark.reports.capture_environment``), so the cost a report
#: prints can be told apart by the machine it was measured on.
ENVIRONMENT_JSONL = OUT / "environment.jsonl"


def record_environment(out_dir: Path, command: str, argv=None) -> dict:
    """Append this process's environment to ``<out_dir>/environment.jsonl`` and return it."""
    from benchmark.reports.environment import capture_environment

    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "command": command,
           "argv": list(sys.argv[1:] if argv is None else argv), **capture_environment()}
    append_jsonl(Path(out_dir) / ENVIRONMENT_JSONL.name, row)
    return row


# --- files -------------------------------------------------------------------------
def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
                    encoding="utf-8")


class JsonlLock:
    """Advisory lock against a second writer resuming the same JSONL.

    Same rationale as ``hpo_pipeline._common.JsonlLock``: two resumes read the
    same "done" set and would append every job twice.
    """

    def __init__(self, path: Path, force: bool = False):
        self.path = Path(str(path) + ".lock")
        self.force = force
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = self.path.read_text(encoding="utf-8").strip() or "unknown"
            if not self.force:
                raise SystemExit(
                    f"{self.path} exists: another run holds these results ({owner}).\n"
                    "If that process is gone, delete the lock or pass --force."
                ) from None
            print(f"WARNING: overriding lock held by {owner}")
            self.path.write_text(f"pid={os.getpid()} (forced)\n", encoding="utf-8")
            self.acquired = True
            return self
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        self.acquired = True
        return self

    def __exit__(self, *exc):
        if self.acquired:
            self.path.unlink(missing_ok=True)
        return False
