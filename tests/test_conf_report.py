"""conf_pipeline.report: tables from runs.jsonl, windows.jsonl, members/ and the sweep."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conf_pipeline import report


def _run(algo, arm, seed, **over):
    row = {
        "kind": "run", "algo": algo, "fold": "covtype", "seed": seed, "arm": arm, "status": "ok",
        "config_id": None,
        "n_samples": 20000, "h": 1000, "n_windows": 19, "d": 10, "s_online": 2200.0, "lam": 0.5,
        "e_size": 4 if arm == "confstream" else 1, "e_test": 2 if arm == "confstream" else 0,
        "wall_s": 100.0, "ensemble_runtime_s": 80.0, "incumbent_runtime_s": 20.0,
        "incumbent_throughput_sps": 950.0,
        "lag_score_mean": 0.4, "lag_ari_win_mean": 0.3, "lag_ari_win_median": 0.31,
        "lag_ari_cumulative": 0.28, "lag_undefined_share": 0.1, "lag_infeasible_share": 0.0,
        "lag_n_pred_labels_mean": 5.0, "lag_n_macro_clusters_mean": 4.0,
        "lag_memory_mb_peak": 0.5, "lag_n_sentinel_total": 0, "n_switches": 3,
        "n_accepted_total": 6, "n_reset_total": 0, "single_class_window_share": 0.2,
        "final_params": {"epsilon": 100.0}, "start_params": {"epsilon": 0.02},
    }
    row.update(over)
    return row


def _window(algo, arm, seed, w, **over):
    row = {
        "kind": "window", "algo": algo, "fold": "covtype", "seed": seed, "arm": arm, "window": w,
        "config_id": None,
        "start": w * 1000, "end": w * 1000 + 999, "incumbent_id": 0, "params": {"epsilon": 100.0},
        "score": 0.5, "ari_win": 0.35,
        "lag_score": 0.4 if w > 1 else None, "lag_ari_win": 0.3 if w > 1 else None,
        "lag_ari_cum": 0.28 if w > 1 else None, "lag_defined": w > 1, "lag_feasible": True,
        "lag_n_pred_labels": 5, "runtime_s": 1.0, "ensemble_runtime_s": 4.0, "wall_s": 5.0,
        "n_true_classes": 2, "n_active": 4, "n_accepted": 1, "n_reset": 0,
    }
    row.update(over)
    return row


def _member(algo, arm, seed, w, member_id, ari, is_test=False, fold="covtype", config_id=None):
    return {
        "algo": algo, "fold": fold, "seed": seed, "arm": arm, "config_id": config_id,
        "member_id": member_id, "window": w, "is_test": is_test, "parent_id": None,
        "born_window": 0, "params": {"epsilon": 1.0}, "score": 0.1, "silhouette_defined": True,
        "feasible": True, "feasible_reason": None, "error": None, "runtime_s": 0.1,
        "memory_mb": 0.1, "n_micro_clusters": 5, "n_macro_clusters": 3, "n_pred_labels": 3,
        "ari_win": ari, "nmi_win": 0.5, "purity_win": 0.5,
    }


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _regret_row(out_dir, algo, fold, arm, region="all"):
    r = report.regret_table(out_dir)
    return r[(r.algo == algo) & (r.fold == fold) & (r.arm == arm) & (r.region == region)].iloc[0]


#: scale_drift: h = 1000, drift at sample 3000 -> window 3.
DRIFT_LAG = {1: 0.4, 2: 0.4, 3: 0.1, 4: 0.2, 5: 0.38, 6: 0.39, 7: 0.4, 8: 0.4}


def _drift_windows(arm, seed=1):
    return [
        _window("DenStream", arm, seed, w, fold="scale_drift", lag_ari_win=lag, ari_win=lag,
                params={"epsilon": 1.0 if w <= 4 else 4.0, "mu": 2.0})
        for w, lag in DRIFT_LAG.items()
    ]


@pytest.fixture
def out_dir(tmp_path):
    tail = "lag_ari_win_tail_mean"
    runs = [
        _run("DenStream", "confstream", 1, **{tail: 0.4}),
        _run("DenStream", "confstream", 2, lag_ari_win_mean=0.5, **{tail: 0.4}),
        _run("DenStream", "default", 1, lag_ari_win_mean=0.1, incumbent_throughput_sps=1200.0, **{tail: 0.1}),
        _run("DenStream", "default", 2, lag_ari_win_mean=0.1, incumbent_throughput_sps=1200.0, **{tail: 0.1}),
        _run("DenStream", "random_online", 1, **{tail: 0.3}),
        _run("DenStream", "frozen_prefix", 1, frozen_at_window=20, **{tail: 0.2}),
        _run("DenStream", "oracle_static", 1, config_id=7, lag_ari_win_mean=0.6, **{tail: 0.6}),
        _run("DenStream", "oracle_static", 2, config_id=7, lag_ari_win_mean=0.6, **{tail: 0.6}),
        # a row from an earlier oracle (config 3): the current oracle_static.json names config 7
        _run("DenStream", "oracle_static", 3, config_id=3, lag_ari_win_mean=0.9, **{tail: 0.9}),
        _run("HPStream", "confstream", 1, lag_ari_win_mean=0.2, **{tail: 0.2}),
        _run("HPStream", "default", 1, status="timeout", error="exceeded 10s"),
        _run("HPStream", "default", 2, lag_ari_win_mean=0.1, **{tail: 0.1}),
        # HPStream has no oracle_static run: its ceiling falls back to the sweep
        _run("HPStream", "oracle_static", 1, status="skipped", error="no oracle for HPStream/covtype"),
        _run("DenStream", "confstream", 1, fold="scale_drift", n_samples=8000, n_windows=8,
             drift_points=[3000], **{tail: None}),
    ]
    windows = [_window("DenStream", "confstream", s, w) for s in (1, 2) for w in range(1, 4)]
    windows += [_window("DenStream", "default", 1, w) for w in range(1, 4)]
    windows += [_window("HPStream", "confstream", 1, w) for w in range(1, 4)]
    windows += _drift_windows("confstream")
    _write_jsonl(tmp_path / "runs.jsonl", runs)
    _write_jsonl(tmp_path / "windows.jsonl", windows)

    # member logs, one file per job: two active members (incumbent 0.35, best 0.42) and a
    # challenger with 0.9 that the best-active reading must ignore
    for s in (1, 2):
        members = [
            m for w in range(1, 4) for m in (
                _member("DenStream", "confstream", s, w, 0, 0.35),
                _member("DenStream", "confstream", s, w, 1, 0.42),
                _member("DenStream", "confstream", s, w, 2, 0.9, is_test=True),
            )
        ]
        _write_jsonl(tmp_path / "members" / f"DenStream__covtype__seed{s}__confstream.jsonl", members)
    drift_members = [
        m for w in DRIFT_LAG for m in (
            _member("DenStream", "confstream", 1, w, 0, 0.45 if w <= 2 else 0.43, fold="scale_drift"),
            _member("DenStream", "confstream", 1, w, 1, 0.05, fold="scale_drift"),
            _member("DenStream", "confstream", 1, w, 2, 0.99, is_test=True, fold="scale_drift"),
        )
    ]
    _write_jsonl(tmp_path / "members" / "DenStream__scale_drift__seed1__confstream.jsonl", drift_members)

    sweep = [
        {"kind": "run", "algo": "HPStream", "fold": "covtype", "seed": 1, "arm": "sweep",
         "config_id": 0, "unit_override": None, "status": "ok", "lag_ari_win_mean": 0.13,
         "final_params": {"max_clusters": 5}},
        {"kind": "run", "algo": "HPStream", "fold": "covtype", "seed": 1, "arm": "sweep",
         "config_id": 1, "unit_override": {"max_clusters": 0.5}, "status": "ok",
         "lag_ari_win_mean": 0.12, "final_params": {"max_clusters": 10}},
        {"kind": "run", "algo": "HPStream", "fold": "covtype", "seed": 1, "arm": "sweep",
         "config_id": 2, "unit_override": {"max_clusters": 0.9}, "status": "error",
         "lag_ari_win_mean": None, "final_params": None},
    ]
    _write_jsonl(tmp_path / "sweep.jsonl", sweep)
    oracle = {
        "DenStream": {"covtype": {"config_id": 7, "unit": {"epsilon": 0.3}, "params": {"epsilon": 1.2},
                                  "lag_ari_win_mean": 0.61, "n_ok": 33, "n_error": 0}},
        "HPStream": {"covtype": {"config_id": 0, "unit": None, "params": {"max_clusters": 5},
                                 "lag_ari_win_mean": 0.13, "n_ok": 2, "n_error": 1}},
    }
    (tmp_path / "oracle_static.json").write_text(json.dumps(oracle))
    return tmp_path


# --- summary ---------------------------------------------------------------------------

def test_summary_table_aggregates_over_seeds_per_algo_fold_arm(out_dir):
    table = report.summary_table(out_dir)
    assert set(table.columns) >= {
        "algo", "fold", "arm", "n_seeds", "lag_ari_win_mean", "lag_ari_win_sd",
        "lag_score_mean", "lag_ari_cumulative", "incumbent_throughput_sps",
        "ensemble_runtime_s", "lag_memory_mb_peak", "lag_n_pred_labels_mean",
        "n_switches", "lag_undefined_share", "single_class_window_share", "n_failed",
        "lag_ari_win_tail_mean", "lag_ari_win_tail_sd", "lag_score_tail_mean",
        "lag_nmi_win_mean", "lag_purity_win_mean", "lag_ari_A_kmeans_mean",
        "lag_ari_A_dbscan_mean", "lag_ssq_micro_mean", "lag_n_micro_centers_mean",
        "frozen_at_window",
    }
    den = table[(table.algo == "DenStream") & (table.fold == "covtype") & (table.arm == "confstream")].iloc[0]
    assert den.n_seeds == 2
    assert den.lag_ari_win_mean == pytest.approx(0.4)
    assert den.lag_ari_win_sd == pytest.approx(0.1414, abs=1e-3)
    assert den.lag_ari_win_tail_mean == pytest.approx(0.4)
    hp_default = table[(table.algo == "HPStream") & (table.arm == "default")].iloc[0]
    assert hp_default.n_seeds == 1 and hp_default.n_failed == 1
    frozen = table[(table.algo == "DenStream") & (table.arm == "frozen_prefix")].iloc[0]
    assert frozen.frozen_at_window == 20


def test_summary_table_averages_the_cost_ratios_and_the_table_a_errors(out_dir):
    rows = _read_jsonl(out_dir / "runs.jsonl")
    for r in rows:
        if (r["algo"], r["fold"], r["arm"]) == ("DenStream", "covtype", "confstream"):
            r.update(tuning_overhead=3.0 + r["seed"], switch_cost=-0.1 * r["seed"],
                     lag_table_a_error_share=0.5 * (r["seed"] - 1),
                     lag_table_a_first_error=None if r["seed"] == 1 else "ValueError: bad centres")
    _write_jsonl(out_dir / "runs.jsonl", rows)
    table = report.summary_table(out_dir)
    assert {"tuning_overhead", "switch_cost", "lag_table_a_error_share",
            "lag_table_a_first_error"} <= set(report.SUMMARY_FIELDS)
    den = table.set_index(["algo", "fold", "arm"]).loc[("DenStream", "covtype", "confstream")]
    assert den["tuning_overhead"] == pytest.approx(4.5)
    assert den["tuning_overhead_sd"] == pytest.approx(0.7071, abs=1e-3)
    assert den["switch_cost"] == pytest.approx(-0.15)
    assert den["lag_table_a_error_share"] == pytest.approx(0.25)
    assert den["lag_table_a_first_error"] == "ValueError: bad centres"   # a message, not a mean
    assert "lag_table_a_first_error_sd" not in table


def test_summary_table_flags_a_cell_missing_a_seed_of_its_arm(out_dir):
    t = report.summary_table(out_dir).set_index(["algo", "fold", "arm"])
    den = t.loc[("DenStream", "covtype", "confstream")]
    assert den["n_expected"] == 2 and den["all_seeds_ok"]           # seeds 1 and 2 both ok
    hp = t.loc[("HPStream", "covtype", "confstream")]
    assert hp["n_expected"] == 2 and not hp["all_seeds_ok"]         # seed 2 never ran
    hp_default = t.loc[("HPStream", "covtype", "default")]
    assert hp_default["n_seeds"] == 1 and not hp_default["all_seeds_ok"]   # seed 1 timed out


def test_seed_gate_counts_each_arms_own_seeds(tmp_path):
    # The full preset reruns the oracle on seeds 2-6 while the protocol arms use 1-5:
    # a count over every row would expect six seeds of every cell and pass none.
    runs = [_run("DenStream", arm, seed) for arm in ("default", "confstream") for seed in (1, 2)]
    runs += [_run("DenStream", "oracle_static", seed, config_id=0) for seed in (2, 3)]
    runs += [_run("DenStream", "confstream", 1, fold="kdd99")]
    _write_jsonl(tmp_path / "runs.jsonl", runs)
    t = report.summary_table(tmp_path).set_index(["algo", "fold", "arm"])
    assert t.loc[("DenStream", "covtype", "oracle_static"), "all_seeds_ok"]
    assert t.loc[("DenStream", "covtype", "default"), "all_seeds_ok"]
    kdd = t.loc[("DenStream", "kdd99", "confstream")]
    assert kdd["n_expected"] == 2 and not kdd["all_seeds_ok"]


def test_summary_table_drops_rows_of_a_stale_oracle(out_dir):
    table = report.summary_table(out_dir)
    oracle = table[(table.algo == "DenStream") & (table.arm == "oracle_static")].iloc[0]
    assert oracle.n_seeds == 2 and oracle.lag_ari_win_tail_mean == pytest.approx(0.6)


# --- arms vs default ---------------------------------------------------------------------

def test_gain_table_has_one_row_per_non_default_arm(out_dir):
    g = report.gain_table(out_dir)
    assert set(g["arm"]) == {"confstream", "random_online", "frozen_prefix", "oracle_static"}
    assert {"gain_tail", "gain_all", "throughput_ratio"} <= set(g.columns)
    r = g[(g.algo == "DenStream") & (g.fold == "covtype") & (g.arm == "confstream")].iloc[0]
    assert r["gain_tail"] == pytest.approx(0.4 - 0.1)
    assert r["gain_all"] == pytest.approx(0.4 - 0.1)
    assert r["throughput_ratio"] == pytest.approx(950.0 / 1200.0)
    o = g[(g.algo == "DenStream") & (g.fold == "covtype") & (g.arm == "oracle_static")].iloc[0]
    assert o["gain_tail"] == pytest.approx(0.6 - 0.1)


def test_gain_table_is_nan_without_a_default_run(out_dir):
    g = report.gain_table(out_dir)
    r = g[(g.fold == "scale_drift") & (g.arm == "confstream")].iloc[0]
    assert r["gain_all"] != r["gain_all"]  # NaN: scale_drift has no default arm


# --- ceiling and headroom -------------------------------------------------------------------

def test_oracle_table_prefers_the_oracle_rerun_and_falls_back_to_the_sweep(out_dir):
    o = report.oracle_table(out_dir).set_index(["algo", "fold"])
    den = o.loc[("DenStream", "covtype")]
    assert den["ceiling"] == pytest.approx(0.6)  # the stale config-3 row (0.9) is ignored
    assert den["ceiling_source"] == "oracle_static"
    hp = o.loc[("HPStream", "covtype")]
    assert hp["ceiling"] == pytest.approx(0.13) and hp["ceiling_source"] == "sweep"


def test_oracle_table_sweep_fallback_reads_the_best_tail_point(out_dir):
    # HPStream has no oracle rerun. Point 0 wins the whole stream, point 1 the tail: the
    # ceiling is point 1's tail, and its whole-stream reading goes with it
    sweep = _read_jsonl(out_dir / "sweep.jsonl")
    for r in sweep:
        r["lag_ari_win_tail_mean"] = {0: 0.10, 1: 0.20, 2: None}[r["config_id"]]
    _write_jsonl(out_dir / "sweep.jsonl", sweep)
    hp = report.oracle_table(out_dir).set_index(["algo", "fold"]).loc[("HPStream", "covtype")]
    assert hp["ceiling"] == pytest.approx(0.20) and hp["ceiling_metric"] == "lag_ari_win_tail_mean"
    assert hp["ceiling_all"] == pytest.approx(0.12) and hp["config_id"] == 1
    assert hp["ceiling_source"] == "sweep"


def test_headroom_gates_on_gain_and_noise(out_dir):
    h = report.headroom_table(out_dir).set_index(["algo", "fold"])
    d = h.loc[("DenStream", "covtype")]
    assert d["ceiling"] == pytest.approx(0.6) and d["tunable"] and d["headroom_confstream"] == pytest.approx(0.3 / 0.5)
    assert d["default_se"] == pytest.approx(0.0)
    assert d["evolution_value"] == pytest.approx(d["headroom_confstream"] - d["headroom_random_online"])
    assert d["headroom_frozen_prefix"] == pytest.approx(0.1 / 0.5)
    hp = h.loc[("HPStream", "covtype")]
    assert not hp["tunable"]   # gain below 0.05
    assert hp["headroom_confstream"] is None and hp["evolution_value"] is None


def test_headroom_gate_uses_the_default_standard_error(out_dir):
    # a noisy default: seeds 0.1 and 0.9 -> mean 0.5, se 0.4; gain 0.1 >= 0.05 but < 1.645 * 0.4
    rows = [json.loads(line) for line in (out_dir / "runs.jsonl").read_text().splitlines()]
    for r in rows:
        if r["algo"] == "DenStream" and r["arm"] == "default" and r["seed"] == 2:
            r["lag_ari_win_tail_mean"] = 0.9
    _write_jsonl(out_dir / "runs.jsonl", rows)
    d = report.headroom_table(out_dir).set_index(["algo", "fold"]).loc[("DenStream", "covtype")]
    assert d["gain"] == pytest.approx(0.1) and d["default_se"] == pytest.approx(0.4)
    assert not d["tunable"] and d["headroom_confstream"] is None and "noise" in d["note"]


def test_headroom_gate_admits_a_gain_equal_to_the_minimum_despite_float_error(out_dir):
    # 0.15 - 0.1 is 0.04999999999999999 in floating point: the gate is gain >= 0.05
    rows = _read_jsonl(out_dir / "runs.jsonl")
    for r in rows:
        if r["algo"] == "DenStream" and r["arm"] == "oracle_static" and r["config_id"] == 7:
            r["lag_ari_win_tail_mean"] = 0.15
    _write_jsonl(out_dir / "runs.jsonl", rows)
    d = report.headroom_table(out_dir).set_index(["algo", "fold"]).loc[("DenStream", "covtype")]
    assert d["gain"] == pytest.approx(0.05) and d["tunable"] and d["note"] == ""


def test_headroom_note_tells_a_sweep_without_an_ok_point_from_no_sweep(out_dir):
    rows = _read_jsonl(out_dir / "runs.jsonl")
    rows += [_run("CluStream", "default", s, lag_ari_win_tail_mean=0.2) for s in (1, 2)]
    _write_jsonl(out_dir / "runs.jsonl", rows)
    sweep = _read_jsonl(out_dir / "sweep.jsonl")
    sweep.append({"kind": "run", "algo": "CluStream", "fold": "covtype", "seed": 1, "arm": "sweep",
                  "config_id": 0, "unit_override": None, "status": "error", "lag_ari_win_mean": None})
    _write_jsonl(out_dir / "sweep.jsonl", sweep)
    oracle = json.loads((out_dir / "oracle_static.json").read_text())
    oracle["CluStream"] = {"covtype": {"config_id": None, "unit": None, "params": None,
                                       "n_ok": 0, "n_error": 1}}
    (out_dir / "oracle_static.json").write_text(json.dumps(oracle))
    h = report.headroom_table(out_dir).set_index(["algo", "fold"])
    clu = h.loc[("CluStream", "covtype")]
    assert clu["ceiling"] is None and not clu["tunable"]
    assert clu["note"] == "no ceiling: sweep produced no ok point for CluStream/covtype"
    assert h.loc[("DenStream", "scale_drift"), "note"] == "no ceiling: no oracle_static rerun and no sweep"


# --- regret ------------------------------------------------------------------------------------

def test_regret_decomposition_sums_to_the_gap_to_the_static_oracle(out_dir):
    r = _regret_row(out_dir, "DenStream", "covtype", "confstream")
    assert r["R_lag"] + r["R_proxy"] + r["R_search"] == pytest.approx(r["A_static"] - r["a_lag"])
    assert r["R_proxy"] >= 0 and r["note"] == "" and r["ceiling_source"] == "oracle_static"
    assert r["a_lag"] == pytest.approx(0.3) and r["a_inc"] == pytest.approx(0.35)
    assert r["a_star"] == pytest.approx(0.42)  # the 0.9 challenger is not an active member
    assert r["A_static"] == pytest.approx(0.6)


def test_regret_without_member_log_reports_only_the_lag_term(out_dir):
    shutil.rmtree(out_dir / "members")
    r = _regret_row(out_dir, "DenStream", "covtype", "confstream")
    assert r["R_lag"] is not None and r["R_proxy"] is None and "members" in r["note"]
    assert r["R_search"] is None
    assert r["a_star_lag"] is None and r["R_proxy_lag"] is None and r["R_search_lag"] is None


def _rewrite_members(out_dir, seed, rows):
    _write_jsonl(out_dir / "members" / f"DenStream__covtype__seed{seed}__confstream.jsonl", rows)


def test_regret_delayed_label_pick_is_not_the_same_window_maximum(out_dir):
    # two members swap places every window: whichever led on t-1 trails on t. The best
    # member of each window reads 0.5, but a pick made by labels one window earlier,
    # the delay the silhouette pick has, reads 0.2
    swap = {1: (0.2, 0.5), 2: (0.5, 0.2), 3: (0.2, 0.5)}
    for s in (1, 2):
        _rewrite_members(out_dir, s, [_member("DenStream", "confstream", s, w, m, swap[w][m])
                                      for w in swap for m in (0, 1)])
    r = _regret_row(out_dir, "DenStream", "covtype", "confstream")
    assert r["a_star"] == pytest.approx(0.5) and r["a_star_lag"] == pytest.approx(0.2)
    assert r["R_proxy_lag"] == pytest.approx(0.2 - 0.3)      # a_star_lag - a_lag
    assert r["R_search_lag"] == pytest.approx(0.6 - 0.2)     # A_static - a_star_lag
    assert r["R_proxy_lag"] + r["R_search_lag"] == pytest.approx(r["A_static"] - r["a_lag"])


def test_regret_delayed_label_pick_skips_infeasible_members_and_ties_to_the_incumbent(out_dir):
    # windows 1 and 3: members 0 and 1 tie at 0.4, member 3 reads 0.9 but is infeasible;
    # window 2: 0.3, 0.6 and 0.95. The incumbent is member 1 on every window.
    ari = {1: (0.4, 0.4, 0.9), 2: (0.3, 0.6, 0.95), 3: (0.4, 0.4, 0.9)}
    rows = _read_jsonl(out_dir / "windows.jsonl")
    for r in rows:
        if (r["algo"], r["fold"], r["arm"]) == ("DenStream", "covtype", "confstream"):
            r["incumbent_id"] = 1
    _write_jsonl(out_dir / "windows.jsonl", rows)
    for s in (1, 2):
        members = []
        for w, (a0, a1, a3) in ari.items():
            infeasible = _member("DenStream", "confstream", s, w, 3, a3)
            infeasible.update(feasible=False, score=-1.0)
            members += [_member("DenStream", "confstream", s, w, 0, a0),
                        _member("DenStream", "confstream", s, w, 1, a1), infeasible]
        _rewrite_members(out_dir, s, members)
    # window 2: the tie on window 1 goes to the incumbent, member 1 (0.6 on window 2);
    # window 3: member 1 led window 2 among the feasible ones (0.4 on window 3)
    r = _regret_row(out_dir, "DenStream", "covtype", "confstream")
    assert r["a_star_lag"] == pytest.approx((0.6 + 0.4) / 2)


def test_regret_delayed_label_pick_counts_a_challenger_accepted_on_the_previous_window(out_dir):
    # challenger 2 leads window 1 on test (0.9) and is accepted: it is an active member from
    # window 2 on, where it reads 0.1. The label pick for window 2 is it; for window 3, member 1.
    for s in (1, 2):
        members = []
        for w in (1, 2, 3):
            members += [_member("DenStream", "confstream", s, w, 0, 0.35),
                        _member("DenStream", "confstream", s, w, 1, 0.42),
                        _member("DenStream", "confstream", s, w, 2, 0.9 if w == 1 else 0.1, is_test=w == 1)]
        _rewrite_members(out_dir, s, members)
    r = _regret_row(out_dir, "DenStream", "covtype", "confstream")
    assert r["a_star_lag"] == pytest.approx((0.1 + 0.42) / 2)


def test_regret_delayed_label_pick_ignores_a_challenger_that_was_not_accepted(out_dir):
    # the fixture's challenger 2 reads 0.9 on test every window and never joins the active set
    r = _regret_row(out_dir, "DenStream", "covtype", "confstream")
    assert r["a_star_lag"] == pytest.approx(0.42)


def test_regret_without_a_static_oracle_leaves_the_search_term_out(out_dir):
    r = _regret_row(out_dir, "DenStream", "scale_drift", "confstream")
    assert r["R_search"] is None and r["R_proxy"] is not None and r["note"] == "no static oracle"


def test_regret_counts_an_incumbent_accepted_on_the_same_window_as_active(out_dir):
    # window 3 of seed 1: the incumbent is challenger 2, accepted on this window and so
    # logged with is_test=True; its ari_win (0.9) is what a_star must be on that window
    rows = [json.loads(line) for line in (out_dir / "windows.jsonl").read_text().splitlines()]
    for r in rows:
        if (r["algo"], r["fold"], r["arm"], r["seed"], r["window"]) == ("DenStream", "covtype", "confstream", 1, 3):
            r.update(incumbent_id=2, ari_win=0.9)
    _write_jsonl(out_dir / "windows.jsonl", rows)
    r = _regret_row(out_dir, "DenStream", "covtype", "confstream")
    # seed 1: a_star (0.42 + 0.9) / 2, a_inc (0.35 + 0.9) / 2; seed 2: 0.42, 0.35
    assert r["a_star"] == pytest.approx(((0.42 + 0.9) / 2 + 0.42) / 2)
    assert r["a_inc"] == pytest.approx(((0.35 + 0.9) / 2 + 0.35) / 2)
    assert r["R_proxy"] >= 0
    assert r["R_lag"] + r["R_proxy"] + r["R_search"] == pytest.approx(r["A_static"] - r["a_lag"])


def test_regret_table_has_a_whole_stream_and_a_tail_row_per_online_arm(out_dir):
    r = report.regret_table(out_dir)
    assert {"region", "ceiling_source"} <= set(report.REGRET_COLS) <= set(r.columns)
    regions = [sorted(g["region"]) for _, g in r.groupby(["algo", "fold", "arm"])]
    assert len(regions) == 3 and all(x == ["all", "tail"] for x in regions)


def test_regret_tail_row_reads_only_the_windows_from_tail_from(out_dir):
    rows = _read_jsonl(out_dir / "runs.jsonl")
    for r in rows:
        if (r["algo"], r["fold"]) == ("DenStream", "covtype"):
            r["tail_from"] = 3
            if r["arm"] == "oracle_static":
                r["lag_ari_win_tail_mean"] = 0.7     # the tail ceiling, unlike the whole stream's 0.6
    _write_jsonl(out_dir / "runs.jsonl", rows)
    wins = _read_jsonl(out_dir / "windows.jsonl")
    for w in wins:
        if (w["algo"], w["fold"], w["arm"], w["window"]) == ("DenStream", "covtype", "confstream", 2):
            w.update(lag_ari_win=0.1, ari_win=0.2)   # an early window unlike the tail
    _write_jsonl(out_dir / "windows.jsonl", wins)

    t = _regret_row(out_dir, "DenStream", "covtype", "confstream", "tail")
    assert t["n_seeds"] == 2 and t["n_windows"] == 2            # window 3 of each seed
    assert t["a_lag"] == pytest.approx(0.3) and t["a_inc"] == pytest.approx(0.35)
    assert t["a_star"] == pytest.approx(0.42)
    assert t["A_static"] == pytest.approx(0.7) and t["ceiling_source"] == "oracle_static"
    assert t["R_lag"] + t["R_proxy"] + t["R_search"] == pytest.approx(t["A_static"] - t["a_lag"])

    a = _regret_row(out_dir, "DenStream", "covtype", "confstream", "all")
    assert a["n_windows"] == 4 and a["a_lag"] == pytest.approx(0.2) and a["A_static"] == pytest.approx(0.6)
    assert a["R_lag"] + a["R_proxy"] + a["R_search"] == pytest.approx(a["A_static"] - a["a_lag"])


def test_regret_tail_row_is_empty_when_no_window_reaches_tail_from(out_dir):
    # no tail_from on the rows: the protocol's freeze window (20) is past these 3 windows
    t = _regret_row(out_dir, "DenStream", "covtype", "confstream", "tail")
    assert t["n_windows"] == 0 and t["n_seeds"] == 0
    assert all(t[c] is None for c in ("a_lag", "a_inc", "a_star", "R_lag", "R_proxy", "R_search"))
    assert "tail_from" in t["note"]


# --- silhouette against ARI --------------------------------------------------------------------

def _scored(seed, w, member_id, score, ari, **over):
    m = _member("DenStream", "confstream", seed, w, member_id, ari)
    m.update(score=score, **over)
    return m


def _proxy_row(out_dir, algo="DenStream", arm="confstream", region="all"):
    p = report.proxy_table(out_dir)
    return p[(p.algo == algo) & (p.fold == "covtype") & (p.arm == arm) & (p.region == region)].iloc[0]


def test_proxy_table_is_the_mean_spearman_over_windows_then_seeds(out_dir):
    # seed 1: silhouette and ARI agree on every window (rho 1); an infeasible member that
    # would break the agreement is left out. Seed 2: one swap among four members, the
    # fourth a challenger on test, which counts: rho = 1 - 6 * 2 / (4 * 15) = 0.8
    seed1 = [m for w in (1, 2, 3) for m in (
        _scored(1, w, 0, 0.1, 0.1), _scored(1, w, 1, 0.2, 0.2), _scored(1, w, 2, 0.3, 0.3),
        _scored(1, w, 3, -1.0, 0.99, feasible=False))]
    seed2 = [m for w in (1, 2, 3) for m in (
        _scored(2, w, 0, 0.1, 0.2), _scored(2, w, 1, 0.2, 0.1), _scored(2, w, 2, 0.3, 0.3),
        _scored(2, w, 3, 0.4, 0.4, is_test=True))]
    _rewrite_members(out_dir, 1, seed1)
    _rewrite_members(out_dir, 2, seed2)
    r = _proxy_row(out_dir)
    assert r["n_seeds"] == 2 and r["n_windows"] == 6
    assert r["spearman"] == pytest.approx((1.0 + 0.8) / 2)
    assert r["ari_spread"] == pytest.approx((0.2 + 0.3) / 2)   # max - min ARI of the members kept
    assert r["note"] == ""


def test_proxy_table_skips_windows_with_too_few_members_or_a_constant_reading(out_dir):
    # the fixture's members all score 0.1: the silhouette is constant, no window has a rho
    r = _proxy_row(out_dir)
    assert r["n_windows"] == 0 and r["spearman"] is None and "no window" in r["note"]


def test_proxy_table_gives_no_number_where_ari_barely_differs_between_members(out_dir):
    # ARIs 0.01-0.03 on every window: a spread under 0.05, so a rank correlation with ARI
    # would read noise as a result
    for s in (1, 2):
        _rewrite_members(out_dir, s, [_scored(s, w, m, 0.1 * (m + 1), 0.01 * (m + 1))
                                      for w in (1, 2, 3) for m in (0, 1, 2)])
    r = _proxy_row(out_dir)
    assert r["spearman"] is None and r["n_windows"] == 6
    assert r["ari_spread"] == pytest.approx(0.02) and r["note"].startswith("ARI spread")


def test_proxy_table_reads_a_fold_whose_static_ceiling_is_flat(out_dir):
    # HPStream's static ceiling clears the default by 0.03 only, yet its members differ on
    # every window: the static gate says nothing about how members compare within a window
    _write_jsonl(out_dir / "members" / "HPStream__covtype__seed1__confstream.jsonl", [
        {**_member("HPStream", "confstream", 1, w, m, a), "score": s}
        for w in (1, 2, 3) for m, s, a in ((0, 0.1, 0.1), (1, 0.2, 0.2), (2, 0.3, 0.3))
    ])
    r = _proxy_row(out_dir, algo="HPStream")
    assert r["spearman"] == pytest.approx(1.0) and r["n_windows"] == 3


# --- drift -------------------------------------------------------------------------------------

def test_drift_table_measures_drop_and_recovery(out_dir):
    d = report.drift_table(out_dir)
    row = d[(d.algo == "DenStream") & (d.fold == "scale_drift") & (d.arm == "confstream")].iloc[0]
    assert row["window"] == 3 and row["drop_depth"] == pytest.approx(0.4 - 0.1) and row["recovery_windows"] == 2
    assert row["param_ratio"] == pytest.approx(4.0)
    assert row["drift_point"] == 3000 and row["pre"] == pytest.approx(0.4)


def test_drift_reference_is_pre_for_a_single_member_ensemble(out_dir):
    # default runs one member: its best active member is the lagged incumbent itself, so
    # "recovered to a_star" would read 0 at every drift; it is measured against pre instead
    rows = [json.loads(line) for line in (out_dir / "runs.jsonl").read_text().splitlines()]
    rows.append(_run("DenStream", "default", 1, fold="scale_drift", drift_points=[3000]))
    _write_jsonl(out_dir / "runs.jsonl", rows)
    with open(out_dir / "windows.jsonl", "a") as handle:
        for w in _drift_windows("default"):
            w["n_active"] = 1
            handle.write(json.dumps(w) + "\n")
    _write_jsonl(out_dir / "members" / "DenStream__scale_drift__seed1__default.jsonl",
                 [_member("DenStream", "default", 1, w, 0, lag, fold="scale_drift") for w, lag in DRIFT_LAG.items()])
    d = report.drift_table(out_dir)
    row = d[(d.fold == "scale_drift") & (d.arm == "default")].iloc[0]
    # pre = 0.4: lag >= 0.35 from window 5 (0.38, 0.39, 0.4) -> 2 windows
    assert row["ref"] == "pre" and row["recovery_windows"] == 2


def test_drift_points_fall_back_to_the_fold_at_the_run_cap(out_dir):
    # no drift_points on the row: scale_drift generated at 9000 samples cuts at 3000 and 6000
    rows = [json.loads(line) for line in (out_dir / "runs.jsonl").read_text().splitlines()]
    rows.append(_run("DenStream", "random_online", 1, fold="scale_drift", n_samples_cap=9000))
    _write_jsonl(out_dir / "runs.jsonl", rows)
    with open(out_dir / "windows.jsonl", "a") as handle:
        for w in _drift_windows("random_online"):
            handle.write(json.dumps(w) + "\n")
    d = report.drift_table(out_dir)
    ro = d[(d.fold == "scale_drift") & (d.arm == "random_online")].sort_values("drift_point")
    assert list(ro["drift_point"]) == [3000, 6000] and list(ro["window"]) == [3, 6]
    assert ro.iloc[0]["pre"] == pytest.approx(0.4)


def test_new_tables_are_empty_frames_on_an_empty_out_dir(tmp_path):
    for fn, cols in (
        (report.gain_table, {"algo", "fold", "arm", "gain_tail", "gain_all", "throughput_ratio"}),
        (report.oracle_table, {"algo", "fold", "ceiling", "ceiling_source"}),
        (report.headroom_table, {"algo", "fold", "default", "ceiling", "gain", "default_se", "tunable",
                                 "evolution_value"}),
        (report.regret_table, {"algo", "fold", "arm", "R_lag", "R_proxy", "R_search", "a_lag", "a_inc",
                               "a_star", "A_static", "note"}),
        (report.drift_table, {"algo", "fold", "arm", "drift_point", "window", "pre", "drop_depth",
                              "recovery_windows", "param_ratio"}),
    ):
        df = fn(tmp_path)
        assert df.empty and cols <= set(df.columns), fn.__name__
    assert report.load_members(tmp_path).empty
    assert report.load_sweep(tmp_path).empty


# --- the report --------------------------------------------------------------------------------

def test_write_report_shows_table_a_when_runs_carry_it(out_dir):
    rows = [json.loads(line) for line in (out_dir / "runs.jsonl").read_text().splitlines()]
    for r in rows:
        if r["algo"] == "DenStream" and r["fold"] == "covtype" and r["arm"] == "default":
            r.update(lag_ari_A_kmeans_tail_mean=0.25, lag_ari_A_dbscan_tail_mean=0.15,
                     lag_ari_A_kmeans_mean=0.24, lag_ari_A_dbscan_mean=0.14,
                     lag_ssq_micro_mean=12.5, lag_n_micro_centers_mean=40.0)
    _write_jsonl(out_dir / "runs.jsonl", rows)
    report.write_report(out_dir)
    md = (out_dir / "report" / "report.md").read_text()
    section = md.split("## Table A (common reclustering)")[1].split("## ")[0]
    assert "lag_ari_A_kmeans_tail_mean" in section and "0.250" in section and "12.500" in section
    assert "HPStream" not in section  # only rows that carry Table A



def test_write_report_summary_shows_the_cost_ratios(out_dir):
    report.write_report(out_dir)
    md = (out_dir / "report" / "report.md").read_text()
    section = md.split("## Summary (mean over seeds)")[1].split("\n## ")[0]
    assert "tuning_overhead" in section and "switch_cost" in section


def test_write_report_lists_table_a_errors_under_table_a(out_dir):
    rows = _read_jsonl(out_dir / "runs.jsonl")
    for r in rows:
        if (r["algo"], r["fold"], r["arm"]) == ("DenStream", "covtype", "default"):
            r.update(lag_table_a_error_share=1.0, lag_table_a_first_error="RuntimeError: boom")
    _write_jsonl(out_dir / "runs.jsonl", rows)
    report.write_report(out_dir)
    md = (out_dir / "report" / "report.md").read_text()
    section = md.split("## Table A (common reclustering)")[1].split("\n## ")[0]
    assert "RuntimeError: boom" in section and "1.000" in section
    assert "predate" not in section     # errors, not old runs, left Table A empty
    assert "HPStream" not in section    # only the cells with an error are listed


def test_write_report_produces_csv_markdown_and_curves(out_dir):
    paths = report.write_report(out_dir)
    rep = out_dir / "report"
    for name in ("summary.csv", "gain.csv", "oracle.csv", "headroom.csv", "regret.csv", "proxy.csv",
                 "drift.csv"):
        assert (rep / name).exists(), name
    md = (rep / "report.md").read_text()
    assert "DenStream" in md and "confstream" in md and "default" in md
    assert "timeout" in md  # failed runs are listed, not hidden
    assert "lag" in md.lower()
    for section in ("Arms vs default", "Ceiling and headroom", "Regret decomposition",
                    "Silhouette against ARI", "Drift adaptation", "Table A (common reclustering)"):
        assert f"## {section}" in md, section
    curves = rep / "curves_covtype.csv"
    assert curves.exists()
    assert all(p.exists() for p in paths)


def test_oracle_table_names_where_the_ceiling_configuration_came_from(out_dir):
    import pandas as pd

    from conf_pipeline import _common as C

    o = report.oracle_table(out_dir)
    expected = [None if pd.isna(c) else "default" if c == 0
                else "incumbent" if c >= C.INCUMBENT_ID_BASE else "sobol" for c in o["config_id"]]
    assert o["config_source"].tolist() == expected
    assert report.config_source(C.INCUMBENT_ID_BASE + 7) == "incumbent"
