"""conf_pipeline.campaign: the config, the exact inventory, the manifest and its strict resume."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conf_pipeline import _common as C
from conf_pipeline import campaign as K
from conf_pipeline import run as R

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "conf_pipeline" / "campaigns" / "mechanism_pilot.json"
PILOT_FOLDS = ["covtype", "kdd99", "electricity", "shuttle", "insects_abrupt_imbalanced", "scale_drift", "density_drift",
               "k_drift", "mixed_drift", "static_overlap", "incremental_drift", "evolving"]


@pytest.fixture(scope="module")
def cfg():
    return K.load_config(CONFIG)


@pytest.fixture(scope="module")
def inv(cfg, tmp_path_factory):
    return K.inventory(cfg, tmp_path_factory.mktemp("camp") / "out")


def test_the_pilot_config_runs_hpstream_as_its_label_variant_by_the_users_decision(cfg):
    algos = [a for g in cfg["groups"] for a in g["algos"]]
    assert "HPStreamStableLabels" in algos and "HPStream" not in algos      # one model, never both
    assert "HPStreamStableLabels" in cfg["decisions"]["HPStream"]


def test_the_pilot_config_holds_the_locked_protocol_streams_arms_and_seeds(cfg):
    p = cfg["protocol"]
    assert cfg["folds"] == PILOT_FOLDS
    assert cfg["arms"] == ["default", "scaled_default", "confstream", "frozen_prefix", "frozen_reinit", "random_online"]
    assert (p["h"], p["e_size"], p["e_test"], p["freeze_after_window"], p["n_samples"]) == (1000, 20, 10, 20, 100000)
    assert (p["seed_policy"], p["model_seed"], p["fixed_model_seeds"], p["data_seed"]) == ("split", 201, [201], 42)
    assert p["runtime_gate"] is False and p["tie_break"] == "micro_clusters" and p["signals"] == ["stability"]
    assert p["log_members"] and p["p_reset"] == C.P_RESET and p["half_life_frac"] == C.HALF_LIFE_FRAC
    assert cfg["operational"]["timeout_s"] == 43200
    groups = {g["name"]: g for g in cfg["groups"]}
    assert groups["six_algorithms_three_tuner_seeds"]["tuner_seeds"] == [101, 102, 103]
    assert groups["clustream_repstream_one_tuner_seed"]["algos"] == ["CluStream", "RepStream"]
    assert groups["clustream_repstream_one_tuner_seed"]["tuner_seeds"] == [101]
    assert not set(cfg["arms"]) & set(C.DIAGNOSTIC_ARMS) and set(cfg["excluded_arms"]) == set(C.DIAGNOSTIC_ARMS)


def test_the_inventory_is_996_plus_132_runnable_jobs_and_24_identical_skips(inv):
    assert inv["n_runnable"] == 1128 and inv["n_skipped"] == 24
    assert inv["by_group"] == {"clustream_repstream_one_tuner_seed": 132, "six_algorithms_three_tuner_seeds": 996}
    assert inv["by_arm"] == {"confstream": 240, "default": 96, "frozen_prefix": 240, "frozen_reinit": 240,
                             "random_online": 240, "scaled_default": 72}
    assert {(s["algo"], s["arm"]) for s in inv["skipped"]} == {("HPStreamStableLabels", "scaled_default"),
                                                               ("RepStream", "scaled_default")}


def test_fixed_runs_are_deduplicated_by_the_seed_policy_and_online_runs_follow_the_tuner_seeds(inv):
    jobs = inv["jobs"]
    fixed = [j for j in jobs if j["arm"] in ("default", "scaled_default")]
    assert all(j["tuner_seed"] is None for j in fixed)
    for j in fixed:
        if j["algo"] in ("CluStream", "HPStreamStableLabels", "STREAMKMeans"):
            assert j["seed"] == j["model_seed"] == 201
        else:
            assert j["seed"] is None and j["model_seed"] is None and "__seedx__" in j["member_log"]
    online = [j for j in jobs if j["arm"] not in ("default", "scaled_default")]
    seeds = {(j["algo"], j["fold"], j["arm"]): set() for j in online}
    for j in online:
        seeds[(j["algo"], j["fold"], j["arm"])].add(j["tuner_seed"])
        assert j["model_seed"] == (201 if j["algo"] in ("CluStream", "HPStreamStableLabels", "STREAMKMeans") else None)
    for (algo, _, _), s in seeds.items():
        assert s == ({101} if algo in ("CluStream", "RepStream") else {101, 102, 103})
    assert all("/lifecycle/" in j["lifecycle_log"] and "/members/" in j["member_log"] for j in jobs)


def test_the_commands_pass_the_protocol_and_never_a_bypass_flag(cfg, tmp_path):
    for g in cfg["groups"]:
        argv = K.run_argv(cfg, g, tmp_path, 8)
        assert not set(argv) & {"--force", "--allow-code-change", "--adopt-legacy", "--retry-failed"}
        a = R.build_parser().parse_args(argv)
        assert a.no_runtime_gate and a.log_members and a.signals == "stability" and a.tie_break == "micro_clusters"
        assert (a.h, a.e_size, a.e_test, a.freeze_after_window, a.n_samples, a.timeout) == (1000, 20, 10, 20, 100000, 43200)
        assert a.seed_policy == "split" and a.model_seed == 201 and a.fixed_model_seeds == "201" and a.data_seed == 42
        assert a.workers == 8


def test_a_config_that_retunes_a_locked_value_or_runs_an_unknown_arm_is_refused(cfg):
    bad = copy.deepcopy(cfg)
    bad["protocol"]["e_size"] = 10
    with pytest.raises(SystemExit, match="locked value"):
        K.check_config(bad)
    bad = copy.deepcopy(cfg)
    bad["arms"].append("nope")
    with pytest.raises(SystemExit, match="unknown arms"):
        K.check_config(bad)
    bad = copy.deepcopy(cfg)
    bad["groups"][1]["algos"].append("DenStream")
    with pytest.raises(SystemExit, match="one group per algorithm"):
        K.check_config(bad)


def test_e1_configs_mark_what_scaling_changes(cfg):
    by = {r["algo"]: r for r in K.e1_configs(cfg)}
    assert by["HPStreamStableLabels"]["scaled_default"].startswith("not run")
    assert by["RepStream"]["scaled_default"].startswith("not run")
    assert "equals default" in by["BIRCH"]["scaled_default"] and by["BIRCH"]["start_values"] == {"threshold": 0.0}
    assert by["DenStream"]["scaled_default"] == "runs; scaled: epsilon"


def _manifest(cfg, inv, out, **change):
    m = {"name": cfg["name"], "config_sha256": K.config_sha256(cfg), "protocol": cfg["protocol"],
         "operational": K.operational(cfg), "code": {"code_sha256": "c1", "packages_sha256": "p1"},
         "inventory": {"jobs": inv["jobs"], "sha256": inv["sha256"]}}
    m.update(change)
    return json.loads(json.dumps(m, default=str))


def test_resume_is_refused_on_another_protocol_timeout_code_package_lock_or_a_dropped_job(cfg, inv, tmp_path):
    old, _, _ = K.check_resume(None, _manifest(cfg, inv, tmp_path), workers=8, argv=[])
    same, kind, added = K.check_resume(old, _manifest(cfg, inv, tmp_path), workers=4, argv=[])
    assert kind == "resumed" and not added and [h["event"] for h in same["history"]] == ["created", "resumed"]
    for change in ({"protocol": {**cfg["protocol"], "tie_break": "runtime"}},
                   {"operational": {**K.operational(cfg), "timeout_s": 21600}},
                   {"code": {"code_sha256": "c2", "packages_sha256": "p1"}},
                   {"code": {"code_sha256": "c1", "packages_sha256": "p2"}},
                   {"inventory": {"jobs": inv["jobs"][1:], "sha256": "x"}}):
        with pytest.raises(SystemExit, match="no flag bypasses"):
            K.check_resume(old, _manifest(cfg, inv, tmp_path, **change), workers=8, argv=[])


def test_more_tuner_seeds_extend_the_campaign_and_are_recorded(cfg, inv, tmp_path):
    old, _, _ = K.check_resume(None, _manifest(cfg, inv, tmp_path), workers=8, argv=[])
    more = copy.deepcopy(cfg)
    more["groups"][1]["tuner_seeds"] = [101, 102, 103]
    inv2 = K.inventory(more, tmp_path / "out")
    assert inv2["n_runnable"] == 1128 + 2 * 12 * 4 * 2
    new = _manifest(more, inv2, tmp_path)
    merged, kind, added = K.check_resume(old, new, workers=8, argv=[])
    assert kind == "extended" and len(added) == 2 * 12 * 4 * 2 and merged["history"][-1]["n_jobs_added"] == len(added)


def test_the_full_run_and_a_dir_no_campaign_made_are_never_written(tmp_path):
    with pytest.raises(SystemExit, match="published full run"):
        K.guard_out_dir(C.OUT / "full")
    with pytest.raises(SystemExit, match="published full run"):
        K.guard_out_dir(C.OUT / "full" / "sub")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "runs.jsonl").write_text('{"kind": "run"}\n')
    with pytest.raises(SystemExit, match="no campaign made"):
        K.guard_out_dir(legacy)
    K.guard_out_dir(tmp_path / "fresh")


def test_run_refuses_a_command_whose_timeout_differs_from_the_campaign_in_its_dir(cfg, tmp_path):
    K.write_manifest(tmp_path, {"operational": K.operational(cfg)})
    ok = R.build_parser().parse_args(K.run_argv(cfg, cfg["groups"][0], tmp_path, 8))
    R.check_campaign_limits(tmp_path, ok)
    # a later --timeout overrides the campaign's: the run CLI default (6 h) or any other value is refused here
    short = R.build_parser().parse_args(K.run_argv(cfg, cfg["groups"][0], tmp_path, 8) + ["--timeout", "21600"])
    with pytest.raises(SystemExit, match="timeout_s"):
        R.check_campaign_limits(tmp_path, short)


def test_the_dry_run_and_help_work_and_write_nothing_into_the_out_dir(tmp_path):
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    out = tmp_path / "camp"
    inv_path = tmp_path / "inventory.json"
    proc = subprocess.run([sys.executable, "-m", "conf_pipeline.campaign", "--config", str(CONFIG), "--out-dir", str(out),
                           "--write-inventory", str(inv_path)], cwd=REPO, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "1128 runnable jobs, 24 written down as skipped" in proc.stdout and "dry run" in proc.stdout
    assert "paper confirmation NOT ELIGIBLE" in proc.stdout and "DenStream: open, GAP-6" in proc.stdout
    assert "HPStreamStableLabels: resolved" in proc.stdout
    assert not out.exists()
    assert json.loads(inv_path.read_text())["inventory"]["n_runnable"] == 1128
    help_ = subprocess.run([sys.executable, "-m", "conf_pipeline.campaign", "--help"], cwd=REPO, capture_output=True,
                           text=True, env=env)
    assert help_.returncode == 0 and "--execute" in help_.stdout
    refused = subprocess.run([sys.executable, "-m", "conf_pipeline.campaign", "--config", str(CONFIG), "--out-dir",
                              str(C.OUT / "full")], cwd=REPO, capture_output=True, text=True, env=env)
    assert refused.returncode != 0 and "published full run" in refused.stderr


def test_execute_refuses_a_parent_without_the_pinned_hash_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with pytest.raises(SystemExit, match="PYTHONHASHSEED=0"):
        K.main(["--config", str(CONFIG), "--out-dir", str(tmp_path / "c"), "--execute"])
    assert not (tmp_path / "c").exists()


def test_the_measurement_status_keeps_every_algorithm_and_never_calls_the_campaign_ready_for_the_paper(cfg, inv, tmp_path):
    ms = K.measurement_status(cfg)
    algos = [a for g in cfg["groups"] for a in g["algos"]]
    assert set(ms["algorithms"]) == set(algos) == set(inv["by_algo"])       # nobody left out of the design
    open_ = sorted(a for a, v in ms["algorithms"].items() if v["status"] == "open")
    assert open_ == ["BIRCH", "CluStream", "DBSTREAM", "DenStream"]
    assert ms["algorithms"]["HPStreamStableLabels"]["status"] == "resolved"
    assert {a for a, v in ms["algorithms"].items() if v["status"] == "no known issue"} == {"RepStream", "SOStream",
                                                                                          "STREAMKMeans"}
    for a in open_:
        v = ms["algorithms"][a]
        assert v["mechanism"] and v["evidence"] and v["scope"] and v["fix"] and v["unblock"]
    assert ms["paper_confirmation"] == "not eligible" and len(ms["reasons"]) == 4 + len(K.OPEN_CONDITIONS)
    assert "HPStreamStableLabels" in K._algo_status("HPStream")["fix"]       # River's HPStream, were it run
    # an inventory of algorithms without an issue is still not eligible while a campaign condition is open
    only = {**cfg, "groups": [{"name": "g", "algos": ["SOStream"], "tuner_seeds": [101]}]}
    assert K.measurement_status(only)["paper_confirmation"] == "not eligible"
    m = K.build_manifest(cfg, CONFIG, tmp_path / "out", inv)
    assert m["measurement_status"] == ms


def test_a_subclass_inherits_its_river_class_issue_and_an_algorithm_never_audited_blocks(cfg):
    groups = [{"name": "g", "algos": ["DenStreamPatched", "DenStreamPatchedFast", "HPStreamStableLabels",
                                      "CluStreamPyramidal", "KMeans", "SOStream"], "tuner_seeds": [101]}]
    ms = K.measurement_status({**cfg, "groups": groups})["algorithms"]
    for a, base in (("DenStreamPatched", "DenStream"), ("DenStreamPatchedFast", "DenStream"),
                    ("CluStreamPyramidal", "CluStream")):
        assert ms[a]["status"] == "open" and ms[a]["inherited_from"] == base
    assert ms["HPStreamStableLabels"]["status"] == "resolved"
    assert ms["KMeans"]["status"] == "not audited" and ms["SOStream"]["status"] == "no known issue"
    # with the campaign conditions cleared, a not-audited algorithm still keeps the campaign not eligible
    saved = K.OPEN_CONDITIONS
    try:
        K.OPEN_CONDITIONS = ()
        only = {**cfg, "groups": [{"name": "g", "algos": ["SOStream", "KMeans"], "tuner_seeds": [101]}]}
        assert K.measurement_status(only)["paper_confirmation"] == "not eligible"
        clean = {**cfg, "groups": [{"name": "g", "algos": ["SOStream", "HPStreamStableLabels"], "tuner_seeds": [101]}]}
        assert K.measurement_status(clean)["paper_confirmation"] == "eligible on the known measurement issues"
    finally:
        K.OPEN_CONDITIONS = saved
