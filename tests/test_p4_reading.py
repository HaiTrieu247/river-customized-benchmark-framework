"""P4 -- the label-free protocol -- is now read by `analyze`.

Stage 7b (`matr_cv.py`) has always written the cell MATR/MATR-CV chose, but
nothing downstream opened the file, so `P1 - P4` never appeared. These tests
pin the reading, not the choosing.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from hpo_pipeline import views


def _panel(rows):
    """rows: (algo, theta, level, dataset, ari)."""
    df = pd.DataFrame(
        [{"algo": a, "theta": th, "granularity_level": float(lv), "dataset": d,
          "ari": ari, "status": "ok", "feasible": True, "config_id": f"{a}{th}{lv}",
          "seed": 1}
         for a, th, lv, d, ari in rows]
    )
    return views.panel(df)


def test_p4_scores_the_cell_stage_7b_named():
    pan = _panel([
        ("A", 1, 1, "d1", 0.10), ("A", 1, 2, "d1", 0.70), ("A", 1, 3, "d1", 0.30),
        ("A", 2, 2, "d1", 0.99),
    ])
    rows = [{"algo": "A", "dataset": "d1", "theta_star": 1, "g_hat": 2}]
    out = views.p4(pan, rows)
    assert out[("A", "d1")] == pytest.approx(0.70)
    # not the best cell in the panel -- P4 is blind, so it need not find it
    assert out[("A", "d1")] < 0.99


def test_p4_reads_the_delta_sweep_column_it_is_asked_for():
    pan = _panel([("A", 1, 1, "d1", 0.10), ("A", 1, 3, "d1", 0.55)])
    rows = [{"algo": "A", "dataset": "d1", "theta_star": 1,
             "g_hat": 1, "g_hat_argmax": 3}]
    assert views.p4(pan, rows)[("A", "d1")] == pytest.approx(0.10)
    assert views.p4(pan, rows, "g_hat_argmax")[("A", "d1")] == pytest.approx(0.55)


def test_p4_skips_a_row_whose_level_is_none():
    """MATR-CV returns None when every split's curve was unusable."""
    pan = _panel([("A", 1, 1, "d1", 0.10), ("A", 1, 1, "d2", 0.20)])
    rows = [{"algo": "A", "dataset": "d1", "theta_star": 1, "g_hat": None},
            {"algo": "A", "dataset": "d2", "theta_star": 1, "g_hat": 1}]
    out = views.p4(pan, rows)
    assert list(out.index) == [("A", "d2")]


def test_p4_is_nan_when_the_named_cell_is_not_in_the_matrix():
    pan = _panel([("A", 1, 1, "d1", 0.10)])
    rows = [{"algo": "A", "dataset": "d1", "theta_star": 9, "g_hat": 9}]
    assert np.isnan(views.p4(pan, rows)[("A", "d1")])


def test_p4_is_empty_without_a_stage_7b_file():
    pan = _panel([("A", 1, 1, "d1", 0.10)])
    assert views.p4(pan, []).empty


def test_analyze_emits_the_p1_minus_p4_gap(tmp_path, monkeypatch):
    """End to end through `analyze.main`, on a hand-built matrix."""
    from hpo_pipeline import analyze

    matrix = tmp_path / "matrix.jsonl"
    matrcv = tmp_path / "matrcv.jsonl"
    report = tmp_path / "report"
    rows = []
    for theta in (1, 2):
        for level in (1, 2, 3):
            for fold in ("d1", "d2", "d3"):
                rows.append({
                    "algo": "A", "config_id": f"t{theta}g{level}", "dataset": fold,
                    "seed": 1, "theta": theta, "granularity_level": level,
                    "ari": 0.1 * level + 0.05 * theta, "status": "ok", "feasible": True,
                    "is_default": False, "runtime_s": 1.0, "peak_memory_mb": 1.0,
                    "matr_trace_win": -1.0 * level, "silhouette": 1.0, "ssq": 1.0,
                    "top1_variance_share": 0.3, "effective_dim": 4.0,
                    "k_hat": level, "n_classes_total": 3, "budget": 1000,
                })
    for fold in ("d1", "d2", "d3"):
        rows.append({"algo": "A", "config_id": "dflt", "dataset": fold, "seed": 1,
                     "theta": None, "granularity_level": None, "ari": 0.02,
                     "status": "ok", "feasible": True, "is_default": True,
                     "runtime_s": 1.0, "peak_memory_mb": 1.0,
                     "top1_variance_share": 0.3, "effective_dim": 4.0,
                     "k_hat": 1, "n_classes_total": 3, "budget": 1000})
    import json
    matrix.write_text("".join(json.dumps(r) + "\n" for r in rows))
    matrcv.write_text("".join(
        json.dumps({"algo": "A", "dataset": f, "theta_star": 2, "g_hat": 2,
                    "g_hat_argmax": 3}) + "\n" for f in ("d1", "d2", "d3")))

    monkeypatch.setattr(analyze, "MATRIX_JSONL", matrix)
    monkeypatch.setattr(analyze, "MATRCV_JSONL", matrcv)
    monkeypatch.setattr(analyze, "REPORT_DIR", report)
    monkeypatch.setattr(analyze, "FOLDS", ("d1", "d2", "d3"))
    monkeypatch.setattr(analyze, "FOLD_GROUPS", {f: "real" for f in ("d1", "d2", "d3")})
    monkeypatch.setattr(sys, "argv", ["analyze"])
    analyze.main()

    gaps = pd.read_csv(report / "gaps.csv")
    labels = set(gaps["comparison"])
    assert "P1 - P4" in labels, sorted(labels)
    # ari = 0.1*level + 0.05*theta, so the ceiling is (theta=2, g=3) = 0.40 and
    # P4's cell (theta*=2, g_hat=2) is 0.30 -- a gap of 0.10 on every fold.
    row = gaps[gaps["comparison"] == "P1 - P4"].iloc[0]
    assert row["mean_absolute_gap"] == pytest.approx(0.10, abs=1e-9)
    assert int(row["n_cells"]) == 3
    # argmax lands on the ceiling here, so its gap is zero
    assert "P1 - P4(argmax)" in labels
    argmax = gaps[gaps["comparison"] == "P1 - P4(argmax)"].iloc[0]
    assert argmax["mean_absolute_gap"] == pytest.approx(0.0, abs=1e-9)
