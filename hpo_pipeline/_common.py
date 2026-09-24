"""Paths, constants and helpers shared by every stage.

Importing this module also puts the repo root and the sibling river source tree
on ``sys.path``. Both are needed: the stages live under ``hpo_pipeline/`` so the
repo root is not importable on its own, and the river datasets are imported from
the sibling checkout rather than from site-packages.

Everything this pipeline reads or writes stays inside ``hpo_pipeline/`` -- the
frozen configs, the materialised permutations and every run artifact. Nothing
is scattered across the repo root.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

#: This package. Every artifact below hangs off it.
PIPELINE_ROOT = Path(__file__).resolve().parent
#: The benchmark framework, which has to be importable for `benchmark.*`.
REPO_ROOT = PIPELINE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.algorithms.adapter import build_river_path  # noqa: E402

build_river_path()

# --- artifacts -------------------------------------------------------------
CONFIGS = PIPELINE_ROOT / "configs"
CACHE = PIPELINE_ROOT / "cache"
HPO_OUT = PIPELINE_ROOT / "out"
VIEWS_DIR = HPO_OUT / "views"
REPORT_DIR = HPO_OUT / "report"

POOL_JSON = CONFIGS / "pool.json"
PROBE_JSON = CONFIGS / "probe.json"
SPACE_JSON = CONFIGS / "search_space.json"
MATRIX_JSONL = HPO_OUT / "matrix.jsonl"
PREFIX_JSONL = HPO_OUT / "prefix.jsonl"
MATRCV_JSONL = HPO_OUT / "matrcv.jsonl"
MANIFEST_JSON = HPO_OUT / "manifest.json"
HEALTH_MD = HPO_OUT / "health.md"

# --- protocol constants ----------------------------------------------------
#: Model-init seeds. Paired with the shuffle seeds below, so one repetition
#: changes both the stream order and the model init, and the spread over the
#: three is the total variance rather than half of it.
SEEDS = (1, 2, 3)
#: Generator seed for the synthetic folds. Fixed, so `seed` only ever moves the
#: model -- otherwise a "seed" would silently mean two different things.
DATA_SEED = 42
#: Stream length every synthetic fold is generated at.
SYNTH_N = 50_000
#: Fidelity rung for the HPO matrix.
BUDGET = 50_000
WINDOW_SIZE = 1000
#: Fixed forever: the candidate set is part of the protocol, not an
#: implementation detail.
SPACE_SEED = 20260101
N_GRANULARITY = 8
N_SHAPE = 8

#: No preprocessing anywhere. The algorithms work in the data's raw units, and
#: `s(D)` from Stage 2 is then the only normalisation left in the pipeline.
SCALER = "none"
SCALER_PARAMS: dict = {}

EXTERNAL_METRICS = (
    "ari", "nmi", "ami", "purity", "purity_macro", "homogeneity", "completeness",
)
#: ``purity`` is sample-weighted (micro); ``purity_macro`` averages over clusters
#: unweighted, which is the formula DenStream/RepStream/SOStream print. Both are
#: carried so the tables can be read against those papers, and both only mean
#: something next to the cluster count.
INTERNAL_METRICS = ("silhouette", "ssq", "matr_trace")

#: Feasibility thresholds. A configuration that trips one of these is not
#: "poor", it is unusable, and it must never be picked as an argmax -- but it
#: stays in the matrix and stays counted.
MAX_MEMORY_MB = 2048.0
MAX_MICRO_CLUSTERS = 10_000
RUNTIME_FACTOR = 30.0

#: Label-free criteria P2R and P4 may read. Only `matr_trace` is defined for
#: every algorithm; the other two need the model to expose centres.
CRITERIA = ("matr_trace_win", "silhouette", "ssq")
#: River's streaming silhouette is a cohesion/separation ratio and its SSQ is a
#: distance, so for both of these a smaller value is the better one.
SMALLER_IS_BETTER = {"silhouette", "ssq"}


#: Hash seed pinned for every run subprocess. River's KMeans iterates a set of
#: feature names, whose order follows the per-process hash seed, and that order
#: decides which random draw initialises which coordinate (RIVER_BUGS.md BUG-2).
#: Everything built on KMeans -- KMeans, STREAMKMeans, CluStream's offline phase
#: -- is reproducible only under a fixed hash seed. The river tree stays
#: untouched; the seed is pinned instead.
PYTHONHASHSEED = "0"


def shuffle_seed_for(seed: int) -> int:
    """Stream permutation paired with a model seed."""
    return 1000 + seed


def river_defaults(algo: str) -> dict:
    """River's constructor defaults for one clusterer, read off the signature.

    This is deliberately **not** `benchmark.algorithms.registry.ALGORITHMS[algo]
    .factory()`. That returns this framework's preset, which for several
    algorithms differs from what River ships -- DenStream's preset is
    epsilon=0.5, mu=2.5, beta=0.5 against River's 0.02, 2, 0.75, and CluStream's
    pins seed=42 while omitting micro_cluster_r_factor. Anything that means "what
    a River user gets today" (P0, the headroom baseline, the `default` arm) has
    to come from here; a preset there answers a different question.

    ``seed`` is left out because the harness injects the run seed.
    """
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


# --- folds -----------------------------------------------------------------
#: fold name -> group. `group` is what Stage 6 uses to build the
#: leave-one-group-out reading, the honest number for transfer: the synthetic
#: folds share one generator and are not independent of each other, so a
#: plain LODO over all nine is optimistic.
FOLD_GROUPS: dict[str, str] = {
    "kdd99": "real",
    "covtype": "real",
    "insects_ir_bal": "real",
    "keystroke": "real",
    "shuttle": "real",
    "static_easy": "synth",
    "static_overlap": "synth",
    "unbalanced": "synth",
    "noisy": "synth",
}

FOLDS = tuple(FOLD_GROUPS)

#: Folds deliberately kept OUT of `FOLD_GROUPS`, and therefore out of every
#: tuning and LODO round: the out-of-sample check. A fold listed here is loadable
#: by name but can never be selected on, which is the only property that makes
#: the final comparison a test rather than a second reading of the training set.
#:
#: What each one is for, and what it cannot support:
#:
#:   `electricity` -- 2 classes, and Souza et al. (DMKD 2020) show its labels are
#:     autocorrelated enough that "same as last" scores 85% where independence
#:     would give 51%. A weak clustering target; a secondary check, never a
#:     headline.
#:   `dstream_30k_nonconvex` -- non-convex clusters, which every centroid method
#:     here is expected to fail. 5,000 of its 30,000 points carry the label
#:     `None` (injected noise) and the external metrics skip them, so its ARI is
#:     computed on 5/6 of the stream.
#:   `insects_abrupt_imbalanced`, `insects_gradual_imbalanced` -- the evidence
#:     the pool was missing. Every fold in `FOLD_GROUPS` is either
#:     one-dimensionally dominated (kdd99 at 99.8% of the variance, shuttle at
#:     95.4%), has a label that is not a cluster structure (covtype's ceiling is
#:     0.019), or is class-balanced. These two are multi-dimensional, real, and
#:     class-imbalanced, which is the combination nothing else here covers. They
#:     share their source data with `insects_ir_bal`, so they are out-of-sample
#:     for the *tuning* but NOT independent of it -- say so wherever they are
#:     averaged with the others.
HOLDOUT_FOLDS: tuple[str, ...] = (
    "electricity",
    "dstream_30k_nonconvex",
    "insects_abrupt_imbalanced",
    "insects_gradual_imbalanced",
)


def load_fold(name: str):
    """Build one fold as its **raw** dataset, before shuffling.

    Nominal columns stay off: one-hot 0/1 columns sitting next to a feature of
    order 1e9 add dimensions without adding distance.
    """
    from benchmark.datasets import (
        load_covtype,
        load_electricity,
        load_insects,
        load_kddcup99,
        load_keystroke,
        load_shuttle,
        make_dstream_30k_nonconvex,
        make_scenario,
    )

    if name == "kdd99":
        return load_kddcup99(include_nominal=False)
    if name == "covtype":
        return load_covtype(quantitative_only=True)
    if name == "insects_ir_bal":
        return load_insects(variant="incremental_reoccurring_balanced")
    if name == "keystroke":
        return load_keystroke()
    if name == "shuttle":
        return load_shuttle()
    if name in ("static_easy", "static_overlap", "unbalanced", "noisy"):
        return make_scenario(name, n_samples=SYNTH_N, seed=DATA_SEED)
    # Out-of-sample folds: loadable, never selectable (see HOLDOUT_FOLDS).
    if name == "electricity":
        return load_electricity()
    if name == "dstream_30k_nonconvex":
        return make_dstream_30k_nonconvex(seed=DATA_SEED)
    if name == "insects_abrupt_imbalanced":
        return load_insects(variant="abrupt_imbalanced")
    if name == "insects_gradual_imbalanced":
        return load_insects(variant="gradual_imbalanced")
    raise KeyError(
        f"unknown fold {name!r}; tuning folds: {sorted(FOLD_GROUPS)}; "
        f"out-of-sample folds: {sorted(HOLDOUT_FOLDS)}"
    )


# --- identity --------------------------------------------------------------
def config_id(algo: str, granularity_level, g, shape: dict) -> str:
    """Hash of the **dimensionless** spec.

    Deliberately not a hash of the resolved params. A distance threshold is
    resolved per fold (``epsilon = g * s_fold``), so hashing resolved params
    would give every fold its own ids, no config would be shared across folds,
    and every protocol built on "the same config across folds" would collapse.
    """
    blob = json.dumps(
        {"a": algo, "gl": granularity_level, "g": g, "shape": shape},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


# --- jsonl -----------------------------------------------------------------
def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


class JsonlLock:
    """Advisory lock guarding one append-only JSONL against a second writer.

    `build_matrix` decides what to run by reading the whole file once, at
    startup, and then appends as results land. Two invocations therefore both
    see the same "already done" set, both queue the same jobs, and both append
    them: the file ends up with duplicate rows for the same
    (algo, config, fold, seed, budget) key, silently doubling the weight of
    those cells in every mean downstream. Resuming an interrupted run is the
    normal way to use the stage, so this is easy to hit by accident -- start a
    resume before noticing the first run is still going.

    Created with ``O_CREAT | O_EXCL`` so the check and the claim are one atomic
    operation, and carrying the owning pid so a stale file can be identified.
    A crashed run leaves the lock behind; ``--force`` is the deliberate way past
    it, and it says what it is overriding.
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
                    f"{self.path} exists: another run holds this matrix ({owner}).\n"
                    "Two writers resuming the same file duplicate rows and silently\n"
                    "double-count those cells. If that process is gone, delete the lock\n"
                    "or pass --force."
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


def read_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# --- matrix ----------------------------------------------------------------
def load_matrix(path: Path = MATRIX_JSONL):
    """Read the matrix and attach the feasibility column.

    Feasibility is computed on read rather than stored, because one of its three
    thresholds is relative to the fastest configuration of the same
    (algorithm, fold) pair and so is only knowable once the whole matrix exists.
    The file itself stays append-only.
    """
    import pandas as pd

    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"{path} is empty or missing; run Stage 4 first")
    df = pd.DataFrame(rows)
    # The file is append-only, so a re-run leaves the superseded attempt behind
    # rather than overwriting it. The last row for a key is the current one;
    # counting the earlier attempts too would double-count the cell and would
    # understate coverage in Stage 5.
    key = ["algo", "config_id", "dataset", "seed", "budget"]
    if all(column in df.columns for column in key):
        df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    return add_feasibility(df)


def add_feasibility(df):
    """Mark rows that are usable. Unusable is a fact about the config, not a gap."""
    import numpy as np
    import pandas as pd

    df = df.copy()
    ok = df["status"] == "ok"

    runtime = pd.to_numeric(df.get("runtime_s"), errors="coerce")
    memory = pd.to_numeric(df.get("peak_memory_mb"), errors="coerce")
    micro = pd.to_numeric(df.get("max_micro_clusters"), errors="coerce")

    fastest = runtime.where(ok).groupby([df["algo"], df["dataset"]]).transform("min")
    runtime_cap = fastest * RUNTIME_FACTOR

    # A missing measurement is not a violation: `max_micro_clusters` is None for
    # models that keep no micro-clusters at all.
    within_memory = ~(memory > MAX_MEMORY_MB)
    within_micro = ~(micro > MAX_MICRO_CLUSTERS)
    within_runtime = ~(runtime > runtime_cap)

    df["feasible"] = np.asarray(ok & within_memory & within_micro & within_runtime)
    return df


def tie_break_frame(frame, score_col: str = "ari"):
    """Deterministic argmax order: score desc, then runtime asc, then id.

    Without a fixed rule the oracle protocols stop being reproducible, because
    ties are common once a score saturates.
    """
    by = [score_col]
    ascending = [False]
    if "runtime_s" in frame.columns:
        by.append("runtime_s")
        ascending.append(True)
    if "config_id" in frame.columns:
        by.append("config_id")
        ascending.append(True)
    return frame.sort_values(by=by, ascending=ascending, kind="mergesort")


def require(path: Path, stage: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {path} -- run {stage} first")
