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
#: Seeds of every arm. Three, the same for every arm, so every comparison
#: between arms sees the same seed noise (decided 2026-10-07 after the pilot,
#: for budget: five seeds cost 900-1 600 CPU-h).
SEEDS: tuple[int, ...] = (1, 2, 3)
#: The tuning regimes every (algo, fold, seed) is run under.
ARMS: tuple[str, ...] = ("default", "confstream", "frozen_prefix")
#: Arms the driver still runs on request (``--arms``) and the report still reads, but that
#: the protocol no longer includes. ``random_online`` left it on 2026-10-12: it cost about
#: half the full budget, and CluStream's runs of it risked the 12 h timeout.
OPTIONAL_ARMS: tuple[str, ...] = ("random_online",)
#: Single-configuration arms driven by a file: the sweep's points and its winner.
EXTRA_ARMS: tuple[str, ...] = ("oracle_static", "sweep")
#: Every fold is cut at this many samples (the driver caps; loaders never see it).
N_SAMPLES_CAP = 100_000
#: ``frozen_prefix`` stops tuning after this window; every arm's tail metrics
#: start here, so all arms are compared on the same evaluation region.
FREEZE_AFTER_WINDOW = 20
SWEEP_N_POINTS = 32
#: The readings the static oracle is picked on, in order of preference: the tail, the
#: region every arm is compared on, then the whole stream for a stream too short to
#: have a tail. Picking on the whole stream would let a strong start win a ceiling
#: that is read on the tail.
ORACLE_METRICS: tuple[str, ...] = ("lag_ari_win_tail_mean", "lag_ari_win_mean")
MIN_CEILING_GAIN = 0.05
#: Length of a synthetic fold when no cap is given.
SYNTH_N_SAMPLES = 120_000

#: No preprocessing: the algorithms see the data's raw units, and ``s`` is the
#: only unit in play (HPStream normalises internally; that is the algorithm's).
SCALER = "none"
EXTERNAL_METRICS = ("ari", "nmi", "purity")
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
RIVER_CLASS = {"DenStreamPatched": "DenStream", "DenStreamPatchedFast": "DenStream"}
#: Implementation variants run beside the River class they re-implement (spec
#: 4.6, table T7): never ranked with the protocol algorithms, read against
#: their reference instead. DenStreamPatchedFast fixes BUG-7/BUG-8 and returns
#: what DenStreamPatched returns at a fraction of the cost (GAP-3).
IMPL_VARIANTS = {"DenStreamPatchedFast": "DenStream"}
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


def load_fold(name: str, n_samples: int | None = None):
    """The protocol folds in their natural order, plus the synthetic scenarios.

    Real loaders get the whole stream (the driver cuts it at ``n_samples``);
    synthetic streams are generated at ``n_samples``, or ``SYNTH_N_SAMPLES``.
    """
    from benchmark.datasets import (
        PARAM_DRIFT_SCENARIOS, SCENARIOS, load_covtype, load_electricity, load_insects,
        load_kddcup99, load_shuttle, make_param_drift, make_scenario,
    )

    if name == "covtype":
        return load_covtype(quantitative_only=True)
    if name == "kdd99":
        return load_kddcup99(n_samples=None)
    if name == "shuttle":
        return load_shuttle()
    if name == "electricity":
        return load_electricity()
    if name.startswith("insects_"):
        return load_insects(variant=name[len("insects_"):])
    if name in PARAM_DRIFT_SCENARIOS:
        return make_param_drift(name, n_samples=n_samples or SYNTH_N_SAMPLES)
    if name in SCENARIOS:
        return make_scenario(name, seed=42, n_samples=n_samples or SYNTH_N_SAMPLES)
    raise KeyError(
        f"unknown fold {name!r}; protocol folds: {FOLDS}; "
        f"synthetic: {sorted(SCENARIOS) + sorted(PARAM_DRIFT_SCENARIOS)}"
    )


def drift_points(fold: str, n_samples: int | None) -> tuple[int, ...]:
    """Sample indices where a parameter-drift fold changes regime; ``()`` otherwise."""
    from benchmark.datasets import PARAM_DRIFT_SCENARIOS, make_param_drift

    if fold in PARAM_DRIFT_SCENARIOS:
        return make_param_drift(fold, n_samples=n_samples or SYNTH_N_SAMPLES).drift_points
    return ()


def best_static(rows, metrics: tuple[str, ...] = ORACLE_METRICS) -> tuple[dict, str] | None:
    """The winning fixed configuration among one (algo, fold)'s sweep rows, and the metric it won on.

    The first of ``metrics`` that some ``ok`` row has a reading of decides; a row
    without that reading never wins. Ties go to the smaller ``config_id`` (the
    earlier, cheaper point). None when no ``ok`` row has any of ``metrics``.
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    for metric in metrics:
        cands = [r for r in ok if _reading(r.get(metric)) is not None]
        if cands:
            best = max(cands, key=lambda r: (_reading(r[metric]), -(r.get("config_id") or 0)))
            return best, metric
    return None


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
