"""Tests for the irace arms: scenario files, the target runner, result parsing,
and how `hpo_compare` turns races into picks. No R, no datasets, no runs --
the runner is faked and the irace output is the CSV the dump script writes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from hpo_pipeline import hpo_compare as H
from hpo_pipeline import irace_tune as I

GRIDS = H.GRIDS


# --- seeds and flags -----------------------------------------------------------
def test_irace_seeds_never_touch_the_scoring_seeds():
    assert not set(I.IRACE_SEEDS) & set(H.FINAL_SEEDS)


def test_irace_starts_with_the_tuning_seed_so_the_grid_runs_are_reused():
    assert I.IRACE_SEEDS[0] == H.TUNE_SEEDS[0]


def test_headroom_is_now_the_default_pooling():
    assert H.build_parser().parse_args([]).aggregate == "headroom"


def test_irace_flags_reject_a_scoring_seed():
    args = H.build_parser().parse_args(["--irace", "ladder", "--irace-seeds", "1,2"])
    with pytest.raises(SystemExit, match="scoring seeds"):
        H._irace_args(args)


def test_irace_flags_reject_an_unknown_arm():
    with pytest.raises(SystemExit, match="unknown arm"):
        H._irace_args(H.build_parser().parse_args(["--irace", "ladder,nope"]))


def test_irace_flags_default_to_the_documented_placeholders():
    arms, reading, settings = H._irace_args(
        H.build_parser().parse_args(["--irace", "raw,ladder,p2r"]))
    assert arms == ["raw", "ladder", "p2r"]
    assert reading == ("R_knee", "matr_trace_win")
    assert settings["seeds"] == I.IRACE_SEEDS
    assert settings["test"] == "t-test"
    assert settings["budget"] == I.IRACE_BUDGET


def test_gate_knobs_override_and_the_defaults_stay_documented():
    try:
        H.set_gate(0.1, 0.03)
        assert (H.MIN_CEILING, H.MIN_CEILING_GAIN) == (0.1, 0.03)
    finally:
        H.set_gate(0.05, 0.02)
    args = H.build_parser().parse_args([])
    assert (args.min_ceiling, args.min_ceiling_gain) == (0.05, 0.02)


# --- parameters ----------------------------------------------------------------
def test_axes_use_c_on_a_scaled_ladder_and_the_threshold_on_raw():
    ladder = [a["name"] for a in I.axes_of("DenStream", "ladder", GRIDS)]
    raw = [a["name"] for a in I.axes_of("DenStream", "raw", GRIDS)]
    assert ladder == ["c", "decaying_factor", "mu"]
    assert raw == ["epsilon", "decaying_factor", "mu"]


def test_p2r_axes_are_the_shape_only_plus_phi_when_the_rule_fits_one():
    knee = [a["name"] for a in I.axes_of("DenStream", "p2r", GRIDS, "R_knee")]
    pars = [a["name"] for a in I.axes_of("DenStream", "p2r", GRIDS, "R_parsimony")]
    assert knee == ["decaying_factor", "mu"]
    assert pars == ["decaying_factor", "mu", "phi"]


def test_grid_parameters_are_ordinal_with_exact_tokens_in_increasing_order():
    """irace 4.5 stops on a numeric-looking ordinal domain that is not increasing
    (found on the first real race: the grid lists thresholds coarse -> fine)."""
    text = I.parameters_text(I.axes_of("DenStream", "ladder", GRIDS), "grid")
    assert 'c "--c=" o ("0.05", "0.1", "0.2", "0.4", "0.8", "1.6", "3.2")' in text
    assert 'mu "--mu=" o ("2", "5", "10", "20")' in text
    raw = I.parameters_text(I.axes_of("DenStream", "raw", GRIDS), "grid")
    assert 'epsilon "--epsilon=" o ("0.02", "0.06", "0.2", "0.6", "1.8", "5.5", "16.0")' in raw


def test_continuous_parameters_span_the_grid_on_a_log_scale():
    text = I.parameters_text(I.axes_of("DenStream", "ladder", GRIDS), "continuous")
    assert 'c "--c=" r,log (0.05, 3.2)' in text
    assert 'mu "--mu=" i,log (2, 20)' in text
    assert 'decaying_factor "--decaying_factor=" r,log (0.25, 2.0)' in text


def test_grid_points_are_every_accepted_grid_candidate():
    for arm, kind in (("raw", "raw"), ("ladder", "ladder")):
        accepted, _ = H.build_grid("CluStream", kind, GRIDS["CluStream"])
        assert len(I.grid_points("CluStream", arm, GRIDS)) == len(accepted)


def test_p2r_grid_points_are_whole_ladders_only():
    """CluStream raw rejects k=64 with q=50, but the P2R ladder is the substituted
    one and loses nothing: every shape keeps its six rungs."""
    points = I.grid_points("CluStream", "p2r", GRIDS, "R_knee")
    assert len(points) == 3 * 3
    points = I.grid_points("DenStream", "p2r", GRIDS, "R_parsimony")
    assert len(points) == 4 * 3 * 5  # mu x lambda x phi


def test_values_round_trip_from_switches_and_from_csv_numbers():
    axes = I.axes_of("DenStream", "ladder", GRIDS)
    from_switches = I.values_of(
        I.parse_switches(["--c=1.6", "--decaying_factor=0.25", "--mu=5"]), axes, "grid")
    from_csv = I.values_of({"c": 1.6, "decaying_factor": 0.25, "mu": 5.0}, axes, "grid")
    assert from_switches == from_csv == {"c": 1.6, "decaying_factor": 0.25, "mu": 5}
    with pytest.raises(ValueError):
        I.values_of({"c": 1.7, "decaying_factor": 0.25, "mu": 5}, axes, "grid")


def test_instances_are_seed_blocks_with_every_dataset_once():
    assert I.instances_text(["a", "b"], (1, 7)) == "a@1\nb@1\na@7\nb@7\n"
    assert I.parse_instance("insects_ir_bal@7") == ("insects_ir_bal", 7)


def test_scenario_counts_tests_in_blocks_and_keeps_mu_a_multiple_of_the_block(tmp_path):
    text = I.scenario_text(tmp_path, n_blocks_folds=4, budget=1000, workers=12,
                           test="t-test", confidence=0.95, first_test=2, each_test=1,
                           rng_seed=1, n_grid_points=84)
    assert "blockSize = 4" in text and "firstTest = 2" in text and "mu = 8" in text
    assert "deterministic = 1" in text and "sampleInstances = 0" in text
    assert "nbIterations = 1" in text and "nbConfigurations = 84" in text
    assert 'testType = "t-test"' in text and "parallel = 12" in text


def test_race_refuses_a_budget_that_cannot_reach_the_first_test(tmp_path):
    with pytest.raises(SystemExit, match="budget"):
        I.race(algo="DenStream", arm="ladder", space="grid", opt_folds=["a", "b"],
               grids_spec=GRIDS, reference={}, probes={}, tuned=[], n_samples=2000,
               timeout=10, workers=1, run_dir=tmp_path, out_dir=tmp_path, rscript="Rscript",
               budget=10)


def test_budget_floor_matches_what_irace_4_5_enforces(tmp_path, monkeypatch):
    """84 DenStream points over 4 datasets, firstTest 2, eachTest 1: irace needs
    84 x (2 + 1) x 4 = 1008 experiments before it starts a race. 1000 was
    refused by irace itself with "not enough budget to enforce the value of
    nbConfigurations" (2026-09-27); the check has to refuse it first, and 1008
    has to get through to irace."""
    monkeypatch.setattr(I, "_run_irace", lambda cmd, **kw: 1)
    folds = ["kdd99", "insects_ir_bal", "keystroke", "shuttle"]
    common = dict(algo="DenStream", arm="raw", space="grid", opt_folds=folds, grids_spec=GRIDS,
                  reference={f: {"default": 0, "gain": 1} for f in folds},
                  probes={f: {"s": 1.0} for f in folds}, tuned=[], n_samples=50000,
                  timeout=10, workers=1, run_dir=tmp_path / "r", out_dir=tmp_path / "o",
                  rscript="Rscript")
    assert len(I.grid_points("DenStream", "raw", GRIDS)) == 84
    with pytest.raises(SystemExit, match="1008"):
        I.race(budget=1007, **common)
    with pytest.raises(SystemExit, match="irace failed"):
        I.race(budget=1008, **common)
    assert I.IRACE_BUDGET >= 84 * 4 * len(I.IRACE_SEEDS), "default must never cap a grid race"


def test_out_dir_is_made_absolute_because_irace_changes_directory(tmp_path, monkeypatch):
    """`irace_tune` starts irace with cwd=run_dir and the target runner from
    irace's execDir; a relative --out-dir was resolved against those and irace
    reported a doubled scenario path (seen 2026-09-27 on the Linux box)."""
    monkeypatch.chdir(tmp_path)
    before = (H.OUT_DIR, H.RUNS_JSONL)
    try:
        H.set_out_dir("hpo_pipeline/out/compare_smoke")
        assert H.OUT_DIR.is_absolute()
        assert H.OUT_DIR == (tmp_path / "hpo_pipeline/out/compare_smoke").resolve()
        assert H.RUNS_JSONL == H.OUT_DIR / "runs.jsonl"
    finally:
        H.OUT_DIR, H.RUNS_JSONL = before


def test_race_hands_irace_absolute_paths_even_for_a_relative_run_dir(tmp_path, monkeypatch):
    """Every path irace or the target runner sees must survive a change of cwd."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_irace(cmd, *, cwd, log_path, echo=True):
        calls.append((cmd, {"cwd": cwd, "log_path": log_path}))
        return 1  # a failed race: `race` stops right after writing its files

    monkeypatch.setattr(I, "_run_irace", fake_irace)
    with pytest.raises(SystemExit, match="irace failed"):
        I.race(algo="DenStream", arm="ladder", space="grid", opt_folds=["a", "b"],
               grids_spec=GRIDS, reference={"a": {"default": 0, "gain": 1},
                                            "b": {"default": 0, "gain": 1}},
               probes={"a": {"s": 1.0}, "b": {"s": 1.0}}, tuned=[], n_samples=2000,
               timeout=10, workers=1, run_dir="rel/run", out_dir="rel/out", rscript="Rscript",
               budget=1000)
    cmd, kwargs = calls[0]
    scenario = cmd[cmd.index("--scenario") + 1]
    assert os.path.isabs(scenario) and os.path.isabs(kwargs["cwd"])
    run_dir = (tmp_path / "rel/run").resolve()
    text = (run_dir / "scenario.txt").read_text()
    for key in ("parameterFile", "trainInstancesFile", "targetRunner", "execDir", "logFile"):
        line = next(l for l in text.splitlines() if l.startswith(key))
        assert os.path.isabs(line.split('"')[1]), line
    ctx = json.loads((run_dir / "context.json").read_text())
    assert os.path.isabs(ctx["out_dir"]) and os.path.isabs(ctx["run_dir"])
    assert str(run_dir / "context.json") in (run_dir / "target-runner").read_text()


def test_irace_output_is_logged_whole_and_echoed_after_the_banner(tmp_path, capsys):
    """A race can run for hours; the terminal must show irace's table rows as
    they arrive, while the log keeps everything including the banner."""
    import sys

    script = ("import sys; print('# irace: banner'); print('# Copyright'); "
              "print('# called with: --scenario x'); print('# 2026 Iteration 1 of 1'); "
              "print('|x|  1|  84|'); sys.exit(3)")
    code = I._run_irace([sys.executable, "-c", script], cwd=str(tmp_path),
                        log_path=tmp_path / "irace.out")
    assert code == 3
    log = (tmp_path / "irace.out").read_text()
    assert log.startswith("# irace: banner") and "|x|  1|  84|" in log
    out = capsys.readouterr().out
    assert "banner" not in out and "called with" not in out
    assert "    # 2026 Iteration 1 of 1" in out and "    |x|  1|  84|" in out


# --- candidates ----------------------------------------------------------------
@pytest.mark.parametrize("algo", ["DenStream", "CluStream"])
@pytest.mark.parametrize("arm,kind", [("raw", "raw"), ("ladder", "ladder")])
def test_grid_space_rebuilds_exactly_the_grid_config_ids(algo, arm, kind):
    """The whole point of the grid space: every run the grid made is reused."""
    ids = {c["config_id"]: c for c in H.build_grid(algo, kind, GRIDS[algo])[0]}
    for point in I.grid_points(algo, arm, GRIDS):
        (cand,) = I.candidates_for(algo, arm, "grid", GRIDS, point)
        assert cand["config_id"] in ids
        grid = ids[cand["config_id"]]
        assert (cand["granularity_level"], cand["theta"], cand["g"]) == \
            (grid["granularity_level"], grid["theta"], grid["g"])
        assert not cand.get("irace")


def test_a_p2r_configuration_is_the_grid_ladder_at_its_shape():
    ids = {c["config_id"] for c in H.build_grid("DenStream", "ladder", GRIDS["DenStream"])[0]}
    ladder = I.candidates_for("DenStream", "p2r", "grid", GRIDS,
                              {"decaying_factor": 1.0, "mu": 10})
    assert [c["granularity_level"] for c in ladder] == list(range(1, 8))
    assert {c["config_id"] for c in ladder} <= ids


def test_continuous_candidates_resolve_but_never_take_a_grid_id():
    grid_ids = {c["config_id"] for kind in ("raw", "ladder")
                for c in H.build_grid("DenStream", kind, GRIDS["DenStream"])[0]}
    (cand,) = I.candidates_for("DenStream", "ladder", "continuous", GRIDS,
                               {"c": 1.6, "decaying_factor": 1.0, "mu": 5})
    assert cand["irace"] and cand["config_id"] not in grid_ids
    assert H.resolve_candidate(cand, {"s": 10.0})["epsilon"] == pytest.approx(16.0)
    (raw,) = I.candidates_for("DenStream", "raw", "continuous", GRIDS,
                              {"epsilon": 0.7, "decaying_factor": 0.3, "mu": 7})
    assert H.resolve_candidate(raw, {"s": 10.0})["epsilon"] == pytest.approx(0.7)


def test_a_continuous_p2r_shape_equal_to_a_grid_shape_reuses_the_grid_ladder():
    grid = I.candidates_for("DenStream", "p2r", "grid", GRIDS, {"decaying_factor": 1.0, "mu": 5})
    cont = I.candidates_for("DenStream", "p2r", "continuous", GRIDS,
                            {"decaying_factor": 1.0, "mu": 5})
    assert [c["config_id"] for c in grid] == [c["config_id"] for c in cont]
    assert not any(c.get("irace") for c in cont)
    off = I.candidates_for("DenStream", "p2r", "continuous", GRIDS,
                           {"decaying_factor": 0.7, "mu": 6})
    assert all(c.get("irace") for c in off)


def test_the_substitute_resolves_off_the_grid_too():
    (cand,) = I.candidates_for("CluStream", "ladder", "continuous", GRIDS,
                               {"n_macro_clusters": 10, "micro_ratio": 3.0,
                                "micro_cluster_r_factor": 2})
    params = H.resolve_candidate(cand, {"s": 1.0})
    assert params["n_macro_clusters"] == 10 and params["max_micro_clusters"] == 30


# --- the target runner ---------------------------------------------------------
def _ctx(tmp_path, arm="ladder", rule=None, criterion=None):
    return {
        "algo": "DenStream", "arm": arm, "space": "grid", "rule": rule, "criterion": criterion,
        "axes": I.axes_of("DenStream", arm, GRIDS, rule), "grids_spec": GRIDS,
        "reference": {"d1": {"default": 0.1, "gain": 0.4}},
        "probes": {"d1": {"s": 10.0}}, "tuned": H.tuned_params("DenStream", GRIDS),
        "n_samples": 2000, "timeout": 10, "out_dir": str(tmp_path), "run_dir": str(tmp_path),
    }


def _fake_row(job, ari=0.3, status="ok", **extra):
    return {"algo": job["algo"], "config_id": job["candidate"]["config_id"],
            "dataset": job["fold"], "seed": job["seed"], "requested": job["requested"],
            "params_resolved": job["params"], "status": status, "ari": ari, **extra}


def test_cost_is_minus_headroom():
    ctx = {"arm": "ladder", "reference": {"d1": {"default": 0.1, "gain": 0.4}}}
    cost, detail = I.cost_of(ctx, {}, "d1", [{"status": "ok", "ari": 0.3}])
    assert cost == pytest.approx(-0.5)
    assert detail["headroom"] == pytest.approx(0.5)


def test_a_failed_run_is_rejected_immediately():
    ctx = {"arm": "ladder", "reference": {"d1": {"default": 0.1, "gain": 0.4}}}
    assert I.cost_of(ctx, {}, "d1", [{"status": "timeout", "ari": None}])[0] == math.inf
    assert I._format_cost(math.inf) == "Inf"


def test_p2r_scores_the_level_its_rule_reads_off_the_label_free_curve():
    """matr_trace peaks at rung 3 (R_max), where ARI is 0.5 -> headroom 1.0."""
    ctx = {"arm": "p2r", "rule": "R_max", "criterion": "matr_trace_win",
           "reference": {"d1": {"default": 0.1, "gain": 0.4}}}
    rows = [{"status": "ok", "ari": a, "matr_trace_win": m}
            for a, m in [(0.1, -9.0), (0.2, -5.0), (0.5, -1.0), (0.4, -3.0)]]
    cost, detail = I.cost_of(ctx, {}, "d1", rows)
    assert detail["level"] == 3
    assert cost == pytest.approx(-1.0)


def test_p2r_rejects_a_broken_curve():
    ctx = {"arm": "p2r", "rule": "R_max", "criterion": "matr_trace_win",
           "reference": {"d1": {"default": 0.1, "gain": 0.4}}}
    rows = [{"status": "ok", "ari": 0.1, "matr_trace_win": -1.0},
            {"status": "ok", "ari": 0.2, "matr_trace_win": float("inf")}]
    assert I.cost_of(ctx, {}, "d1", rows)[0] == math.inf


def test_runs_on_disk_are_reused_and_new_runs_are_appended(tmp_path):
    ctx = _ctx(tmp_path)
    a, b = (I.candidates_for("DenStream", "ladder", "grid", GRIDS,
                             {"c": c, "decaying_factor": 0.25, "mu": 5})[0] for c in (1.6, 0.8))
    job_a = H._job("DenStream", a, "d1", 7, 2000, ctx["probes"], ctx["tuned"])
    I.locked_append(tmp_path / "runs.jsonl", _fake_row(job_a, ari=0.42))
    calls = []

    def runner(job):
        calls.append(job["candidate"]["config_id"])
        return _fake_row(job, ari=0.11)

    rows, executed = I.fetch_or_run(ctx, [a, b], "d1", 7, runner)
    assert [r["ari"] for r in rows] == [0.42, 0.11]
    assert executed == 1 and calls == [b["config_id"]]
    # and b is now on disk: a second call runs nothing
    _, executed = I.fetch_or_run(ctx, [a, b], "d1", 7, runner)
    assert executed == 0


def test_target_main_prints_the_cost_and_logs_the_call(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path)
    path = tmp_path / "context.json"
    path.write_text(json.dumps(ctx, default=str))
    monkeypatch.setattr(H, "run_one", lambda job, timeout: _fake_row(job, ari=0.3))
    assert I.target_main([str(path), "5", "2", "999", "d1@7",
                          "--c=1.6", "--decaying_factor=0.25", "--mu=5"]) == 0
    assert float(capsys.readouterr().out.strip()) == pytest.approx(-0.5)
    (log,) = [json.loads(line) for line in (tmp_path / "target.jsonl").read_text().splitlines()]
    assert log["instance"] == "d1@7" and log["executed"] == 1


def test_a_constraint_violation_is_rejected_without_running(tmp_path):
    ctx = _ctx(tmp_path)
    ctx["grids_spec"] = H.merge_grids(GRIDS, {"DenStream": {"fixed": {"beta": 0.2}}})
    cost, detail = I.evaluate(ctx, {"c": 1.6, "decaying_factor": 0.25, "mu": 2}, "d1", 1,
                              runner=lambda job: pytest.fail("must not run"))
    assert cost == math.inf and "violates" in detail["why"]


# --- reading irace's output ----------------------------------------------------
def test_read_result_ranks_elites_and_counts_their_instances(tmp_path):
    (tmp_path / "configurations.csv").write_text(
        '".ID.","c","decaying_factor","mu",".PARENT."\n'
        '1,"1.6","0.25","5",NA\n2,"0.8","1","10",NA\n3,"0.4","2","20",NA\n')
    (tmp_path / "elites.csv").write_text('"rank","id"\n1,2\n2,1\n')
    (tmp_path / "experiments.csv").write_text(
        '"row","id","cost","instance"\n'
        '1,"1",-0.4,"d1@1"\n2,"1",-0.6,"d2@1"\n'
        '1,"2",-0.9,"d1@1"\n2,"2",-0.7,"d2@1"\n3,"2",-0.8,"d1@7"\n'
        '1,"3",Inf,"d1@1"\n')
    (tmp_path / "rejected.txt").write_text("3\n")
    res = I.read_result(tmp_path, I.axes_of("DenStream", "ladder", GRIDS), "grid")
    assert res["best"] == {"c": 0.8, "decaying_factor": 1.0, "mu": 10}
    assert res["elites"][0]["n_instances"] == 3
    assert res["elites"][0]["mean_headroom"] == pytest.approx(0.8)
    assert res["n_rejected"] == 1 and res["n_configurations"] == 3


# --- hpo_compare: races into picks ----------------------------------------------
def test_only_tunable_datasets_are_raced_and_an_untunable_hold_out_reuses_all():
    rounds = H.irace_rounds(["a", "b", "c", "d"], None, ["a", "b", "c"])
    assert rounds["all"] == ["a", "b", "c"]
    assert rounds["lodo-a"] == ["b", "c"]
    assert rounds["lodo-d"] == rounds["all"]


def _race(best, n=5):
    return {"best": best, "elites": [{"id": "1", "values": best, "n_instances": n,
                                      "mean_headroom": 0.6}],
            "n_configurations": 84, "n_experiments": 400, "target_calls": 400,
            "runs_executed": 250, "budget": 1000, "opt_folds": ["a", "b"], "seeds": [1, 7]}


def test_irace_ships_the_all_race_elite_and_scores_each_lodo_elite_on_its_hold_out():
    best = {"c": 1.6, "decaying_factor": 0.25, "mu": 5}
    other = {"c": 0.4, "decaying_factor": 1.0, "mu": 10}
    results = {("DenStream", "irace_ladder"): {
        "all": _race(best), "lodo-a": _race(other), "lodo-b": _race(best), "lodo-c": None}}
    cells = H.irace_cells(results, GRIDS, "grid")
    picks, finals = H.irace_picks(cells, results, pd.DataFrame(), ["a", "b", "c"], None)
    ship = finals["irace_ladder"]["DenStream"]["config"]
    assert {c["config_id"] for c in picks["irace_ladder"].values()} == {ship["config_id"]}
    assert ship["g"] == 1.6 and ship["shape"]["mu"] == 5
    est = picks["irace_ladder" + H.EST]
    assert est[("DenStream", "a")]["g"] == 0.4
    assert ("DenStream", "c") not in est
    assert finals["irace_ladder"]["DenStream"]["irace"]["elite_instances"] == 5


def _ladder_frame(ladder, fold, trace, ari, seed=1):
    return pd.DataFrame([
        {"algo": c["algo"], "config_id": c["config_id"], "dataset": fold, "seed": seed,
         "status": "ok", "ari": a, "matr_trace_win": m}
        for c, m, a in zip(ladder, trace, ari)])


def test_irace_p2r_reads_the_level_on_each_dataset_off_its_own_curve():
    shape = {"decaying_factor": 0.25, "mu": 5}
    ladder = I.candidates_for("DenStream", "p2r", "grid", GRIDS, shape)
    df = pd.concat([
        _ladder_frame(ladder, "a", [-9, -8, -1, -2, -3, -4, -5], [0.1] * 7),
        _ladder_frame(ladder, "b", [-9, -8, -7, -6, -5, -4, -1], [0.1] * 7),
    ])
    results = {("DenStream", "p2r_irace_R_max_matr_trace_win"): {"all": _race(shape)}}
    cells = H.irace_cells(results, GRIDS, "grid")
    picks, finals = H.irace_picks(cells, results, df, ["a", "b", "c"],
                                  ("R_max", "matr_trace_win"))
    policy = finals["p2r_irace_R_max_matr_trace_win"]["DenStream"]
    assert policy["levels"] == {"a": 3, "b": 7}  # c has no curve: no level, no pick
    chosen = picks["p2r_irace_R_max_matr_trace_win"]
    assert chosen[("DenStream", "a")]["granularity_level"] == 3
    assert ("DenStream", "c") not in chosen


def test_irace_protocol_names_and_labels():
    assert H.irace_protocol("raw") == "irace_raw"
    assert H.irace_protocol("p2r", "R_knee", "matr_trace_win") == \
        "p2r_irace_R_knee_matr_trace_win"
    assert H.arm_label("p2r_irace_R_knee_matr_trace_win" + H.EST) == \
        "irace P2R R_knee_matr_trace_win (LODO estimate)"
    assert H.arm_label("irace_ladder") == "irace ladder"


# --- the report, end to end on a synthetic pool --------------------------------
def _ari(cid, fold, seed):
    h = int(hashlib.sha256(f"{cid}|{fold}|{seed}".encode()).hexdigest()[:8], 16)
    return round(0.05 + 0.5 * (h % 1000) / 1000, 4)


def _runs_for(cands, folds, seeds, probes, tuned_of):
    rows = []
    for cand in cands:
        for fold in folds:
            for seed in seeds:
                job = H._job(cand["algo"], cand, fold, seed, 2000, probes, tuned_of[cand["algo"]])
                ari = 0.0 if cand.get("is_default") else _ari(cand["config_id"], fold, seed)
                rows.append({
                    "algo": cand["algo"], "config_id": cand["config_id"],
                    "granularity_level": cand["granularity_level"], "theta": cand["theta"],
                    "g": cand["g"], "is_default": bool(cand.get("is_default")),
                    "dataset": fold, "seed": seed, "requested": 2000,
                    "params_resolved": job["params"], "status": "ok", "ari": ari,
                    "runtime_s": 1.0, "throughput_sps": 2000.0, "k_hat": 3.0,
                    "n_predicted_clusters": 3.0, "largest_cluster_share": 0.5,
                    "max_micro_clusters": 10.0, "homogeneity": 0.3,
                    "matr_trace_win": -float(cand["granularity_level"] or 1),
                })
    return rows


def test_the_report_carries_the_irace_arms_beside_the_grid(tmp_path):
    spec = H.merge_grids(GRIDS, {"DenStream": {
        "levels": {"raw": [5.5, 0.6], "ladder": [1.6, 0.4]},
        "shape": {g: {"mu": [2, 5], "decaying_factor": [0.25]} for g in ("raw", "ladder")}}})
    folds = ["a", "b", "c"]
    probes = {f: {"s": 10.0} for f in folds}
    algos = ["DenStream"]
    grids = {k: {"DenStream": H.build_grid("DenStream", k, spec["DenStream"])[0]}
             for k in ("raw", "ladder")}
    rejected = {k: {"DenStream": []} for k in ("raw", "ladder")}
    tuned_of = {"DenStream": H.tuned_params("DenStream", spec)}
    defaults = {"DenStream": H.default_candidate("DenStream")}
    cands = [*grids["raw"]["DenStream"], *grids["ladder"]["DenStream"], defaults["DenStream"]]
    seeds = (*H.TUNE_SEEDS, *H.FINAL_SEEDS, 7)
    df = pd.DataFrame(_runs_for(cands, folds, seeds, probes, tuned_of))
    df["grid"] = df.config_id.map({c["config_id"]: c["grid"] for c in cands})
    tune = H.seed_mean(df[df.seed.isin(H.TUNE_SEEDS)])
    refs = {"DenStream": H.fold_reference(tune[tune.algo == "DenStream"])}
    groups = {f: "real" for f in folds}
    picks, finals = H.all_picks(tune, grids, defaults, algos, folds, groups, (), "headroom", refs)
    best = {"c": 1.6, "decaying_factor": 0.25, "mu": 5}
    results = {("DenStream", "irace_ladder"): {
        "all": _race(best), **{f"lodo-{f}": _race(best) for f in folds}}}
    cells = H.irace_cells(results, spec, "grid")
    ipk, ifn = H.irace_picks(cells, results, df, folds, None)
    picks.update(ipk)
    finals.update(ifn)
    scores = H.score_table(df, picks, probes, tuned_of)
    report = H.build_report(tune, scores, folds, probes, 2000, grids, spec, rejected,
                            tuned_of, finals, "headroom",
                            irace={"results": results, "space": "grid",
                                   "settings": {"seeds": [1, 7], "test": "t-test",
                                                "confidence": 0.95, "budget": 1000}})
    assert "## 3c. irace against the grid" in report
    assert "| DenStream | irace ladder | sf-all |" in report
    assert ("by **mean headroom** (`--aggregate headroom`, over the datasets that pass "
            "the 3b-i gate)") in report
    assert "What the seed noise says about the two thresholds" in report
    lodo = report.split("## 3. LODO estimate")[1].split("## 3b.")[0]
    assert "irace ladder" in lodo
    assert np.isfinite(scores[scores.protocol == "irace_ladder"].final_ari).all()
    assert np.isfinite(scores[scores.protocol == "irace_ladder" + H.EST].final_ari).all()
