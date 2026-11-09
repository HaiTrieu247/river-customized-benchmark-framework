"""frozen_prefix_matched: the frozen single model with confstream's evaluation budget spent before the freeze."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conf_pipeline import _common as C
from conf_pipeline import run as R
from benchmark.tuning.confstream import EnsembleConfig

REPO = Path(__file__).resolve().parent.parent


def test_evaluations_counts_the_first_window_then_e_plus_c_then_one_after_a_freeze():
    # E = 3, C = 2, 14 scored windows, freeze on the 3rd: 3 + 5 + 5 then 11 x 1 (as logged by run)
    assert C.evaluations(14, 3, 3, 2, frozen=True) == 24
    assert C.evaluations(14, 3, 3, 2, frozen=False) == 3 + 13 * 5 == 68


@pytest.mark.parametrize("n_scored,n_prefix,e,c", [(14, 3, 3, 2), (99, 20, 20, 10), (49, 20, 20, 10),
                                                    (199, 20, 20, 10), (30, 20, 20, 10)])
def test_matched_sizes_spend_confstreams_budget_within_rounding(n_scored, n_prefix, e, c):
    e_m, c_m, k = C.matched_sizes(n_scored, n_prefix, e, c)
    target = C.evaluations(n_scored, n_scored, e, c, frozen=False)
    planned = C.evaluations(n_scored, n_prefix, e_m, c_m, frozen=True)
    assert k >= 1 and e_m >= e and c_m >= c
    assert abs(planned - target) <= n_prefix          # one model of rounding per prefix window at most
    assert abs(planned - target) / target < 0.02


def test_the_protocols_sizes_are_scaled_about_4_9_times_on_a_full_stream():
    assert C.matched_sizes(99, 20, 20, 10)[:2] == (98, 49)


def test_the_arm_is_runnable_and_freezes_like_frozen_prefix():
    assert "frozen_prefix_matched" in C.OPTIONAL_ARMS and "frozen_prefix_matched" in C.MATCHED_ARMS
    cfg = R.arm_config("frozen_prefix_matched", EnsembleConfig(), 20)
    ref = R.arm_config("frozen_prefix", EnsembleConfig(), 20)
    assert cfg == ref                                   # the sizes are scaled in run_ensemble, from the stream


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("matched") / "out"
    cmd = [sys.executable, "-m", "conf_pipeline.run", "--algos", "STREAMKMeans", "--folds", "static_overlap",
           "--arms", "frozen_prefix,frozen_prefix_matched,confstream", "--seed-policy", "split",
           "--tuner-seeds", "101", "--model-seed", "201", "--fixed-model-seeds", "201", "--data-seed", "42",
           "--n-samples", "3000", "--freeze-after-window", "3", "--h", "200", "--e-size", "3", "--e-test", "2",
           "--tie-break", "micro_clusters", "--score-partition", "snapshot", "--log-members",
           "--no-runtime-gate", "--workers", "1", "--out-dir", str(out)]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, env={**os.environ, "PYTHONHASHSEED": "0"})
    runs = {r["arm"]: r for r in map(json.loads, (out / "runs.jsonl").read_text().splitlines())
            if r.get("kind") == "run"}
    return out, runs


def test_a_run_spends_exactly_confstreams_evaluations_and_records_its_budget(smoke):
    _, runs = smoke
    m, cs, fp = runs["frozen_prefix_matched"], runs["confstream"], runs["frozen_prefix"]
    assert m["status"] == cs["status"] == "ok"
    assert m["n_evaluated_total"] == m["budget_planned_evaluations"] == m["budget_target_evaluations"] \
        == cs["n_evaluated_total"] == 68
    assert fp["n_evaluated_total"] == 24
    assert (m["budget_e_size"], m["budget_e_test"]) == (13, 9)
    assert (m["e_size"], m["e_test"]) == (3, 2)          # the protocol's sizes stay in e_size / e_test
    assert m["tail_freeze_window"] == fp["tail_freeze_window"]


def test_prefix_identity_leaves_the_matched_arm_out(smoke):
    from conf_pipeline import followup as F

    out, _ = smoke
    F.cmd_prefix_identity(out)
    table = pd.read_csv(out / "report" / "followup" / "prefix-identity" / "prefix.csv")
    arms = set(table["arm_a"]) | set(table["arm_b"])
    assert "frozen_prefix_matched" not in arms and {"confstream", "frozen_prefix"} <= arms
