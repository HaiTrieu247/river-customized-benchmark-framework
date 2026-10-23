"""conf_pipeline.variants: protocol constants varied one at a time (X6)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conf_pipeline import _common as C
from conf_pipeline import variants as V
from tests.test_conf_robustness import TOY_QUALITY, _jsonl, _toy_out_dir

TAIL = "lag_ari_win_tail_mean"


def test_every_variant_changes_known_constants_and_keeps_challengers_at_most_half_the_ensemble():
    for name in V.VARIANTS:
        c = V.constants(name)
        assert set(c) == set(V.BASE) and c != V.BASE
        assert c["e_test"] <= c["e_size"] / 2
    assert V.constants("e10") == {**V.BASE, "e_size": 10, "e_test": 5}
    assert V.changed("c20") == "e_size=40, e_test=20" and V.changed("base") == ""
    with pytest.raises(KeyError):
        V.constants("h300")


def test_only_variants_that_keep_h_and_t0_can_borrow_the_base_reference():
    assert {n for n in V.VARIANTS if V.keeps_reference(n)} == {"e10", "e40", "c5", "c20"}


def test_the_commands_of_a_variant_carry_its_constants_and_its_own_out_dir(tmp_path):
    (arms,) = V.commands("h2000", tmp_path, algos=["DenStream"], folds=["covtype"], seeds=[1],
                         arms=["default", "confstream"], workers=4, tie_break="runtime")
    line = " ".join(arms)
    assert "conf_pipeline.run" in line and f"--out-dir {tmp_path / 'h2000'}" in line
    assert "--h 2000" in line and "--e-size 20" in line and "--e-test 10" in line
    assert "--freeze-after-window 20" in line and "--tie-break runtime" in line
    cmds = V.commands("t40", tmp_path, algos=["DenStream"], folds=["covtype"], seeds=[1],
                      arms=["default"], workers=1, tie_break="micro_clusters", own_reference=True,
                      oracle_seeds=[2])
    assert [c[2] for c in cmds] == ["conf_pipeline.run", "conf_pipeline.sweep", "conf_pipeline.run"]
    assert "--freeze-after-window 40" in " ".join(cmds[1]) and "--incumbents" in cmds[1]
    assert "--arms oracle_static" in " ".join(cmds[2]) and "--seeds 2" in " ".join(cmds[2])


def test_the_base_tie_break_is_what_its_online_rows_recorded(tmp_path):
    base = _toy_out_dir(tmp_path / "base")
    assert V.base_tie_break(base) == "runtime"            # rows from before the field existed
    rows = [json.loads(l) for l in (base / "runs.jsonl").read_text().splitlines()]
    _jsonl(base / "runs.jsonl", [{**r, "tie_break": "micro_clusters"} for r in rows])
    assert V.base_tie_break(base) == "micro_clusters"


def _variant_dir(root, shift):
    """Seed-1 runs of the three protocol arms, online tuning ``shift`` above the toy base."""
    rows = []
    for algo, q in TOY_QUALITY.items():
        for fold in ("covtype", "kdd99", "k_drift", "evolving"):
            for arm, off in (("default", -0.21), ("confstream", -0.01 + shift), ("frozen_prefix", -0.11)):
                rows.append({"kind": "run", "algo": algo, "fold": fold, "seed": 1, "arm": arm,
                             "config_id": None, "status": "ok", "tail_from": 20, TAIL: q + off})
    _jsonl(root / "runs.jsonl", rows)
    return root


def test_compare_reads_every_setting_on_the_same_seeds_and_borrows_the_reference_only_when_valid(tmp_path):
    base = _toy_out_dir(tmp_path / "base")
    out = tmp_path / "x6"
    _variant_dir(out / "e40", shift=0.05)
    _variant_dir(out / "h500", shift=0.05)
    table = V.compare(out, base, names=["e40", "h500", "t10"], seeds=[1], algos=list(TOY_QUALITY))
    t = table.set_index(["setting", "group"])
    assert set(table["setting"]) == {"base", "e40", "h500"}                  # t10 never ran: no row
    # base, seed 1: default q - 0.21, online q - 0.01, reference q + 0.1 (its reruns on seeds 2-4)
    assert t.loc[("base", "real"), "online_minus_default"] == pytest.approx(0.2)
    assert t.loc[("base", "real"), "capture_confstream"] == pytest.approx(0.2 / 0.31)
    assert t.loc[("e40", "real"), "reference"] == "base"
    assert t.loc[("e40", "real"), "capture_confstream"] == pytest.approx(0.25 / 0.31)
    assert pd.isna(t.loc[("h500", "real"), "reference"]) and pd.isna(t.loc[("h500", "real"), "capture_confstream"])
    assert t.loc[("h500", "all"), "online_minus_default"] == pytest.approx(0.25)
    assert t.loc[("e40", "all"), "tau_default_confstream"] == pytest.approx(1.0)
    assert t.loc[("base", "real"), "n_folds"] == 2 and t.loc[("base", "all"), "n_pairs"] == 8


def test_the_cli_dry_run_lists_the_commands_without_running_them(tmp_path):
    base = _toy_out_dir(tmp_path / "base")
    repo = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "conf_pipeline.variants", "run", "--out-dir", str(tmp_path / "x6"),
         "--base-dir", str(base), "--variants", "h500,c5", "--dry-run"],
        cwd=repo, capture_output=True, text=True, env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert "tie-break: runtime" in proc.stdout and proc.stdout.count("conf_pipeline.run") == 2
    assert not (tmp_path / "x6").exists()


# --- E/C grid: can the ensemble and its challengers be smaller? --------------------------------------

def test_the_ec_grid_starts_at_the_protocol_and_keeps_challengers_at_most_half_the_ensemble():
    names = list(V.EC_GRID)
    assert names[0] == V.EC_BASE == "ec20_10"
    assert V.EC_GRID[V.EC_BASE] == {"e_size": C.E_SIZE, "e_test": C.E_TEST}
    for name, c in V.EC_GRID.items():
        assert name == f"ec{c['e_size']}_{c['e_test']}" and c["e_test"] <= c["e_size"] / 2
    assert {"ec10_5", "ec5_2", "ec20_5", "ec20_2", "ec15_7"} <= set(names)


def test_an_ec_command_runs_the_online_arm_gate_free_into_its_own_dir(tmp_path):
    cmd = " ".join(V.ec_command("ec10_5", tmp_path, algos=["DenStream"], folds=["covtype"], tuner_seeds=[101],
                                workers=8, timeout=86400))
    assert "conf_pipeline.run" in cmd and f"--out-dir {tmp_path / 'ec10_5'}" in cmd
    assert "--e-size 10" in cmd and "--e-test 5" in cmd and "--arms confstream" in cmd
    assert "--no-runtime-gate" in cmd and "--timeout 86400" in cmd and "--workers 8" in cmd
    assert "--seed-policy split" in cmd and "--tuner-seeds 101" in cmd and f"--model-seed {C.MODEL_SEED}" in cmd
    with pytest.raises(KeyError):
        V.ec_command("ec7_7", tmp_path, algos=["DenStream"], folds=["covtype"], tuner_seeds=[101], workers=1)


def _ec_row(algo, fold, tail, wall, model, status="ok", seed=1):
    return {"kind": "run", "algo": algo, "fold": fold, "seed": seed, "arm": "confstream", "status": status,
            "config_id": None, "n_samples_cap": 1000, "n_samples": 1000, "lag_ari_win_tail_mean": tail,
            "wall_s": wall, "ensemble_runtime_s": model, "incumbent_throughput_sps": 500.0}


def _ec_dirs(root):
    base = [_ec_row("A", "covtype", 0.30, 100.0, 80.0), _ec_row("B", "covtype", 0.20, 50.0, 40.0),
            _ec_row("A", "scale_drift", 0.90, 100.0, 80.0), _ec_row("B", "scale_drift", 0.80, 50.0, 40.0)]
    small = [_ec_row("A", "covtype", 0.29, 40.0, 30.0), _ec_row("B", "covtype", 0.10, 25.0, 20.0),
             _ec_row("A", "scale_drift", 0.90, 50.0, 40.0),
             {**_ec_row("B", "scale_drift", None, None, None, status="timeout")}]
    for name, rows in (("ec20_10", base), ("ec10_5", small)):
        (root / name).mkdir(parents=True)
        _jsonl(root / name / "runs.jsonl", rows)
    return root


def test_ec_pairs_read_each_setting_against_the_base_on_the_same_algo_and_stream(tmp_path):
    pairs = V.ec_pairs(V.ec_runs(_ec_dirs(tmp_path)))
    row = pairs[(pairs.setting == "ec10_5") & (pairs.algo == "A") & (pairs.fold == "covtype")].iloc[0]
    assert row.d_tail == pytest.approx(-0.01) and row.speedup_wall == pytest.approx(2.5)
    assert row.speedup_model == pytest.approx(80 / 30) and row.sps == pytest.approx(1000 / 40)
    unfinished = pairs[(pairs.setting == "ec10_5") & (pairs.algo == "B") & (pairs.fold == "scale_drift")].iloc[0]
    assert unfinished.status == "timeout" and pd.isna(unfinished.d_tail)
    assert set(pairs.setting) == {"ec10_5"}                       # the base is not compared with itself


def test_the_ec_summary_says_per_setting_whether_it_keeps_the_quality_and_runs_faster(tmp_path):
    summary = V.ec_summary(V.ec_pairs(V.ec_runs(_ec_dirs(tmp_path))), band=0.05, max_mean_loss=0.01,
                           max_share_over_band=0.1, min_sps=15)
    alls = summary[(summary.setting == "ec10_5") & (summary.group == "all")].iloc[0]
    assert alls.n_pairs == 3 and alls.n_unfinished == 1
    assert alls.mean_d_tail == pytest.approx((-0.01 - 0.10 + 0.0) / 3)
    assert alls.share_loss_over_band == pytest.approx(1 / 3) and not alls.keeps_quality
    assert alls.median_speedup_wall == pytest.approx(2.0) and alls.verdict == "quality drops"
    real = summary[(summary.setting == "ec10_5") & (summary.group == "real")].iloc[0]
    assert real.n_pairs == 2 and real.share_sps_at_least_min == pytest.approx(1.0)


def test_a_setting_that_keeps_quality_and_runs_faster_reads_reducible():
    pairs = pd.DataFrame([{"setting": "ec10_5", "algo": a, "fold": "covtype", "group": "real", "status": "ok",
                           "tail": 0.3, "base_tail": 0.3, "d_tail": 0.0, "speedup_wall": 2.0, "speedup_model": 2.0,
                           "sps": 50.0, "base_sps": 25.0} for a in "ABC"])
    s = V.ec_summary(pairs)
    assert set(s.verdict) == {"reducible"}


def test_the_ec_compare_cli_writes_its_tables(tmp_path):
    _ec_dirs(tmp_path)
    proc = subprocess.run([sys.executable, "-m", "conf_pipeline.variants", "ec-compare", "--out-dir", str(tmp_path)],
                          cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    rep = tmp_path / "report" / "ec"
    assert {p.name for p in rep.iterdir()} >= {"runs.csv", "pairs.csv", "summary.csv", "ec.md"}


def test_an_ec_command_carries_the_window_and_t0_it_is_given(tmp_path):
    cmd = " ".join(V.ec_command("ec5_2", tmp_path, algos=["DenStream"], folds=["static_easy"], tuner_seeds=[101],
                                workers=1, h=300, freeze_after_window=1))
    assert "--h 300" in cmd and "--freeze-after-window 1" in cmd
    default = " ".join(V.ec_command("ec5_2", tmp_path, algos=["DenStream"], folds=["static_easy"], tuner_seeds=[101],
                                    workers=1))
    assert f"--h {C.H}" in default and f"--freeze-after-window {C.FREEZE_AFTER_WINDOW}" in default


def test_the_ec_run_cli_dry_run_lists_one_command_per_setting(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "conf_pipeline.variants", "ec-run", "--out-dir", str(tmp_path),
                           "--algos", "DenStream", "--folds", "static_easy", "--h", "300", "--grid", "ec20_10,ec5_2",
                           "--dry-run"], cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lines = [l for l in proc.stdout.splitlines() if l.startswith("$")]
    assert len(lines) == 2 and all("--h 300" in l for l in lines) and "ec5_2" in lines[1]
