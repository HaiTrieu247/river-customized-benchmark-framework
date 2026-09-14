"""Write ``hpo_out/manifest.json``: everything needed to reproduce the matrix.

The matrix is append-only and means nothing without the conditions it was
produced under, so the manifest records both checkouts, the interpreter, every
seed, the fold list and the full candidate set.

``PYTHONHASHSEED`` is in here because river's KMeans is sensitive to it: without
a pinned value the same seed gives a different clustering from one process to
the next, and a result is only interpretable next to the hash seed it was
produced under.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys

from ._common import (
    BUDGET,
    DATA_SEED,
    FOLD_GROUPS,
    MANIFEST_JSON,
    REPO_ROOT,
    SEEDS,
    SPACE_JSON,
    SPACE_SEED,
    SYNTH_N,
    WINDOW_SIZE,
    read_json,
    shuffle_seed_for,
    write_json,
)


def _git_head(repo) -> str | None:
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if head.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        commit = head.stdout.strip()
        return f"{commit} (dirty)" if dirty.stdout.strip() else commit
    except (OSError, subprocess.SubprocessError):
        return None


def build() -> dict:
    from benchmark.reports import capture_environment

    space = read_json(SPACE_JSON) if SPACE_JSON.exists() else {}
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": capture_environment(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random (unpinned)"),
        "interpreter": sys.executable,
        "git": {
            "benchmark_framework": _git_head(REPO_ROOT),
            "river": _git_head(REPO_ROOT.parent / "river"),
        },
        "seeds": {
            "model": list(SEEDS),
            "shuffle": [shuffle_seed_for(s) for s in SEEDS],
            "data": DATA_SEED,
            "space": SPACE_SEED,
        },
        "protocol": {
            "scaler": "none",
            "shuffle": True,
            "budget": BUDGET,
            "window_size": WINDOW_SIZE,
            "synthetic_length": SYNTH_N,
            "objective": "ari (whole stream)",
        },
        "folds": FOLD_GROUPS,
        "search_space": {
            algo: {
                "n_candidates": len(cands),
                "config_ids": [c["config_id"] for c in cands],
            }
            for algo, cands in space.items()
        },
    }


def main() -> None:
    payload = build()
    write_json(MANIFEST_JSON, payload)
    print(f"wrote {MANIFEST_JSON}")
    print(f"  river:     {payload['git']['river']}")
    print(f"  framework: {payload['git']['benchmark_framework']}")
    print(f"  hash seed: {payload['python_hash_seed']}")


if __name__ == "__main__":
    main()
