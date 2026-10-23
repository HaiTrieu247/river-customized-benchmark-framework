"""A5: how far the class labels of a window can be reached by batch clustering."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from conf_pipeline import label_reach as driver  # first: conf_pipeline puts river on sys.path
from conf_pipeline import robustness as R

LR = driver.LR


def _blobs(rng, centers, n_per, sd):
    xs, ys = [], []
    for label, c in enumerate(centers):
        pts = rng.normal(c, sd, size=(n_per, len(c)))
        xs += [{f"f{i}": float(v) for i, v in enumerate(p)} for p in pts]
        ys += [label] * n_per
    return xs, ys


def _by(rows):
    return {(r["method"], r["representation"]): r for r in rows}


def test_every_method_reaches_well_separated_blobs():
    xs, ys = _blobs(np.random.default_rng(0), [(0, 0), (10, 0), (0, 10)], 100, 0.5)
    rows = _by(LR.window_readings(xs, ys))
    assert set(rows) == {(m, r) for m in LR.METHODS for r in LR.REPRESENTATIONS}
    for key in [("kmeans", "raw"), ("ward", "raw"), ("dbscan", "raw")]:
        assert rows[key]["ari"] == pytest.approx(1.0, abs=0.01), key
    assert rows[("kmeans", "raw")]["n_true_classes"] == 3
    assert any(rows[("dbscan", "raw")]["eps"] == pytest.approx(v) for v in LR.EPS_GRID)
    assert rows[("kmeans", "raw")]["eps"] is None


def test_standardising_the_window_recovers_a_partition_hidden_by_one_wide_feature():
    rng = np.random.default_rng(1)
    xs, ys = _blobs(rng, [(0, 0), (10, 0), (0, 10)], 100, 0.5)
    for x in xs:
        x["wide"] = float(rng.normal(0, 1e4))
    rows = _by(LR.window_readings(xs, ys))
    assert rows[("kmeans", "raw")]["ari"] < 0.2
    assert rows[("kmeans", "standardised")]["ari"] > 0.9


def test_a_window_with_one_true_class_has_no_reading():
    xs, ys = _blobs(np.random.default_rng(2), [(0, 0), (5, 5)], 20, 0.5)
    assert LR.window_readings(xs, [0] * len(xs)) == []
    assert LR.window_readings(xs, [None] * (len(xs) - 1) + [1]) == []


def test_points_without_a_label_are_left_out():
    xs, ys = _blobs(np.random.default_rng(3), [(0, 0), (10, 10)], 50, 0.5)
    ys = [None if i % 10 == 0 else y for i, y in enumerate(ys)]
    rows = _by(LR.window_readings(xs, ys))
    assert rows[("kmeans", "raw")]["n_points"] == 90 and rows[("kmeans", "raw")]["ari"] == pytest.approx(1.0)


# --- driver ------------------------------------------------------------------------------------

def test_driver_writes_one_row_per_window_method_and_representation_then_resumes(tmp_path, monkeypatch):
    argv = ["label_reach", "--folds", "k_drift", "--n-samples", "3000", "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    driver.main()
    lines = [json.loads(line) for line in (tmp_path / "label_reach.jsonl").read_text().splitlines()]
    rows = [r for r in lines if r["kind"] == "reading"]
    assert {r["window"] for r in rows} == {0, 1, 2}
    assert len(rows) == 3 * len(LR.METHODS) * len(LR.REPRESENTATIONS)
    assert all(r["fold"] == "k_drift" and r["h"] == 1000 for r in rows)
    done = [r for r in lines if r["kind"] == "fold"]
    assert len(done) == 1 and done[0]["n_windows"] == 3 and done[0]["n_single_class"] == 0
    driver.main()  # the fold is on file: nothing is appended
    assert len((tmp_path / "label_reach.jsonl").read_text().splitlines()) == len(lines)
    assert not (tmp_path / "label_reach.jsonl.lock").exists()


def test_reach_table_takes_the_best_method_per_representation_on_the_tail():
    rows = []
    for w in (5, 20, 21):
        for method, raw, std in (("kmeans", 0.2, 0.6), ("ward", 0.4, 0.5)):
            early = w < 20
            rows += [{"fold": "covtype", "window": w, "method": method, "representation": "raw",
                      "ari": 1.0 if early else raw},
                     {"fold": "covtype", "window": w, "method": method, "representation": "standardised",
                      "ari": 1.0 if early else std}]
    per_method, reach = R.reach_table(pd.DataFrame(rows), tail_from=20)
    r = reach.set_index("fold").loc["covtype"]
    assert r["reach_raw"] == pytest.approx(0.4) and r["reach_std"] == pytest.approx(0.6)
    assert len(per_method) == 4
