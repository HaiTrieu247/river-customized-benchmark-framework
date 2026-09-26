"""conf_pipeline.report: tables from runs.jsonl and windows.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conf_pipeline import report


def _run(algo, arm, seed, **over):
    row = {
        "kind": "run", "algo": algo, "fold": "covtype", "seed": seed, "arm": arm, "status": "ok",
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
        "start": w * 1000, "end": w * 1000 + 999, "incumbent_id": 0, "params": {"epsilon": 100.0},
        "score": 0.5, "lag_score": 0.4 if w > 1 else None, "lag_ari_win": 0.3 if w > 1 else None,
        "lag_ari_cum": 0.28 if w > 1 else None, "lag_defined": w > 1, "lag_feasible": True,
        "lag_n_pred_labels": 5, "runtime_s": 1.0, "ensemble_runtime_s": 4.0, "wall_s": 5.0,
        "n_true_classes": 2, "n_active": 4, "n_accepted": 1, "n_reset": 0,
    }
    row.update(over)
    return row


@pytest.fixture
def out_dir(tmp_path):
    runs = [
        _run("DenStream", "confstream", 1), _run("DenStream", "confstream", 2, lag_ari_win_mean=0.5),
        _run("DenStream", "default", 1, lag_ari_win_mean=0.1, incumbent_throughput_sps=1200.0),
        _run("HPStream", "confstream", 1, lag_ari_win_mean=0.2),
        _run("HPStream", "default", 1, status="timeout", error="exceeded 10s"),
    ]
    windows = [_window("DenStream", "confstream", s, w) for s in (1, 2) for w in range(1, 4)]
    windows += [_window("DenStream", "default", 1, w) for w in range(1, 4)]
    windows += [_window("HPStream", "confstream", 1, w) for w in range(1, 4)]
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    (tmp_path / "windows.jsonl").write_text("\n".join(json.dumps(w) for w in windows) + "\n")
    return tmp_path


def test_summary_table_aggregates_over_seeds_per_algo_fold_arm(out_dir):
    table = report.summary_table(out_dir)
    assert set(table.columns) >= {
        "algo", "fold", "arm", "n_seeds", "lag_ari_win_mean", "lag_ari_win_sd",
        "lag_score_mean", "lag_ari_cumulative", "incumbent_throughput_sps",
        "ensemble_runtime_s", "lag_memory_mb_peak", "lag_n_pred_labels_mean",
        "n_switches", "lag_undefined_share", "single_class_window_share", "n_failed",
    }
    den = table[(table.algo == "DenStream") & (table.arm == "confstream")].iloc[0]
    assert den.n_seeds == 2
    assert den.lag_ari_win_mean == pytest.approx(0.4)
    assert den.lag_ari_win_sd == pytest.approx(0.1414, abs=1e-3)
    hp_default = table[(table.algo == "HPStream") & (table.arm == "default")].iloc[0]
    assert hp_default.n_seeds == 0 and hp_default.n_failed == 1


def test_gain_table_reads_the_tuner_against_the_default_arm(out_dir):
    gain = report.gain_table(out_dir)
    den = gain[gain.algo == "DenStream"].iloc[0]
    assert den.lag_ari_win_mean_confstream == pytest.approx(0.4)
    assert den.lag_ari_win_mean_default == pytest.approx(0.1)
    assert den.gain_lag_ari_win == pytest.approx(0.3)
    hp = gain[gain.algo == "HPStream"].iloc[0]
    assert hp.gain_lag_ari_win is None or hp.gain_lag_ari_win != hp.gain_lag_ari_win  # NaN: no default


def test_write_report_produces_csv_markdown_and_curves(out_dir):
    paths = report.write_report(out_dir)
    assert (out_dir / "report" / "summary.csv").exists()
    assert (out_dir / "report" / "gain.csv").exists()
    md = (out_dir / "report" / "report.md").read_text()
    assert "DenStream" in md and "confstream" in md and "default" in md
    assert "timeout" in md  # failed runs are listed, not hidden
    assert "lag" in md.lower()
    curves = out_dir / "report" / "curves_covtype.csv"
    assert curves.exists()
    assert all(p.exists() for p in paths)
