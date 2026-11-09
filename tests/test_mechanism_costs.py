"""The mechanism campaign under the snapshot rule: its arms, its status, the cost beside every contrast and the
notebook's gate-free default."""

from __future__ import annotations

import json

import pytest

from conf_pipeline import campaign as CP
from conf_pipeline import report

CONFIG = CP.C.REPO_ROOT / "conf_pipeline" / "campaigns" / "mechanism_pilot.json"


def test_the_campaign_runs_every_branch_under_the_snapshot_rule_and_gate_free():
    cfg = CP.load_config(CONFIG)
    assert {"frozen_prefix", "frozen_ensemble", "portfolio_renewal", "random_after_prefix", "confstream"} <= set(cfg["arms"])
    argv = CP.run_argv(cfg, cfg["groups"][0], CP.C.REPO_ROOT / "x", 1, dry_run=True)
    assert argv[argv.index("--score-partition") + 1] == "snapshot" and "--no-runtime-gate" in argv


def test_within_window_labelling_issues_do_not_limit_a_snapshot_campaign_but_still_limit_a_sequential_one():
    cfg = CP.load_config(CONFIG)
    snap = CP.measurement_status(cfg)
    assert all(v["status"] in ("resolved", "no known issue") for v in snap["algorithms"].values())
    assert not any("GAP-6" in r for r in snap["reasons"])
    seq = CP.measurement_status({**cfg, "protocol": {**cfg["protocol"], "score_partition": "sequential"}})
    assert seq["paper_confirmation"] == "not eligible" and any("GAP-6" in r for r in seq["reasons"])


def test_every_branch_is_contrasted_with_the_frozen_single_model():
    pairs = {(a, b) for a, b, _ in report.CONTRASTS}
    for arm in ("frozen_ensemble", "portfolio_renewal", "random_after_prefix"):
        assert (arm, "frozen_prefix") in pairs
    assert {("portfolio_renewal", "frozen_ensemble"), ("confstream", "random_after_prefix"),
            ("confstream", "portfolio_renewal")} <= pairs


def test_the_cost_of_each_side_is_read_after_the_freeze_only(tmp_path):
    runs, wins = [], []
    for arm, per_window, evals in (("portfolio_renewal", 3.0, 30), ("frozen_prefix", 1.0, 1)):
        runs.append({"kind": "run", "algo": "DBSTREAM", "fold": "covtype", "seed": 1, "arm": arm, "status": "ok",
                     "ari_basis": "snapshot_pre", "tail_freeze_window": 2})
        for w in range(1, 6):
            # before the freeze both arms run the same ensemble: 30 models, 3 s
            pre = w <= 2
            wins.append({"kind": "window", "algo": "DBSTREAM", "fold": "covtype", "seed": 1, "arm": arm, "window": w,
                         "ensemble_runtime_s": 3.0 if pre else per_window, "n_evaluated": 30 if pre else evals})
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    (tmp_path / "windows.jsonl").write_text("\n".join(json.dumps(r) for r in wins) + "\n")
    costs = report.contrast_costs(tmp_path)
    row = costs[(costs["contrast"] == "portfolio_renewal-frozen_prefix") & (costs["group"] == "all")].iloc[0]
    assert row["post_freeze_s_a"] == pytest.approx(9.0) and row["post_freeze_s_b"] == pytest.approx(3.0)
    assert row["cost_ratio_median"] == pytest.approx(3.0)
    assert (row["evaluations_per_window_a"], row["evaluations_per_window_b"]) == (30, 1)


def test_the_notebook_runs_gate_free_unless_asked():
    nb = json.loads((CP.C.REPO_ROOT / "otb_benchmark.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'RUNTIME_GATE = os.environ.get("OTB_RUNTIME_GATE", "0") == "1"' in code
    assert 'COMMON_ARGS += [] if RUNTIME_GATE else ["--no-runtime-gate"]' in code
