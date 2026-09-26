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
MEMBERS_DIR = OUT / "members"
REPORT_DIR = OUT / "report"

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

ALGOS: tuple[str, ...] = ("DenStream", "CluStream", "HPStream", "RepStream")
FOLDS: tuple[str, ...] = ("covtype", "insects_abrupt_imbalanced")
SEEDS: tuple[int, ...] = (1, 2, 3)
ARMS: tuple[str, ...] = ("confstream", "default")

#: No preprocessing: the algorithms see the data's raw units, and ``s`` is the
#: only unit in play (HPStream normalises internally; that is the algorithm's).
SCALER = "none"
EXTERNAL_METRICS = ("ari", "nmi", "purity")

PYTHONHASHSEED = "0"
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


def river_defaults(algo: str) -> dict:
    """River's constructor defaults, read off the signature (``seed`` left out)."""
    import inspect

    from river import cluster

    skip = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    params = inspect.signature(getattr(cluster, algo).__init__).parameters
    return {
        name: p.default
        for name, p in params.items()
        if name not in ("self", "seed")
        and p.kind not in skip
        and p.default is not inspect.Parameter.empty
    }


def load_fold(name: str, n_samples: int | None = None):
    """The protocol folds in their natural order, plus the synthetic scenarios."""
    from benchmark.datasets import load_covtype, load_insects, make_scenario
    from benchmark.datasets.synthetic.generator import SCENARIOS

    if name == "covtype":
        return load_covtype(quantitative_only=True)
    if name.startswith("insects_"):
        return load_insects(variant=name[len("insects_"):])
    if name in SCENARIOS:
        kw = {"seed": 42}
        if n_samples is not None:
            kw["n_samples"] = n_samples
        return make_scenario(name, **kw)
    raise KeyError(f"unknown fold {name!r}; protocol folds: {FOLDS}; synthetic: {sorted(SCENARIOS)}")


def child_env() -> dict:
    return {**os.environ, "PYTHONHASHSEED": PYTHONHASHSEED}


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
