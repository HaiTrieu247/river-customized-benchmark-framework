"""Stage 4 runs must not depend on the hash seed of whoever launched the batch.

River's KMeans is hash-order sensitive (RIVER_BUGS.md BUG-2), so this spawns
real subprocesses the way ``build_matrix.execute`` does, from parents with
different hash seeds, and requires identical output.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest import mock

from hpo_pipeline import build_matrix
from hpo_pipeline._common import PYTHONHASHSEED, REPO_ROOT

# String feature names are what make the set iteration order hash-dependent.
_KMEANS = """
import hpo_pipeline._common  # puts the sibling river tree on sys.path
import random
from river import cluster
rng = random.Random(0)
model = cluster.KMeans(n_clusters=4, seed=1)
out = []
for _ in range(300):
    x = {f"feature_{j}": rng.random() for j in range(8)}
    out.append(model.predict_one(x))
    model.learn_one(x)
print(out)
"""


def _child_output(parent_hash_seed: str) -> str:
    with mock.patch.dict(os.environ, {"PYTHONHASHSEED": parent_hash_seed}):
        env = build_matrix.child_env()
    proc = subprocess.run(
        [sys.executable, "-c", _KMEANS], env=env, cwd=str(REPO_ROOT),
        capture_output=True, text=True, check=True,
    )
    return proc.stdout


def test_child_env_pins_the_hash_seed():
    with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "123"}):
        assert build_matrix.child_env()["PYTHONHASHSEED"] == PYTHONHASHSEED


def test_kmeans_is_reproducible_across_differently_seeded_parents():
    assert _child_output("1") == _child_output("7")
