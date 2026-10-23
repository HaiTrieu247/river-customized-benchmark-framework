"""conf_pipeline.cost: what a planned run would cost, from what measured runs cost."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conf_pipeline import cost as K

REPO = Path(__file__).resolve().parent.parent


def _runs(tmp_path, rows):
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps({"kind": "run", "status": "ok",
                                                               "n_samples_cap": 1000, **r}) for r in rows) + "\n")
    return tmp_path


def test_a_cell_costs_its_mean_measured_wall_time_over_seeds(tmp_path):
    src = _runs(tmp_path, [{"algo": "A", "fold": "f", "arm": "confstream", "seed": s, "wall_s": w}
                           for s, w in ((1, 10.0), (2, 20.0))])
    table = K.cell_costs(src)
    assert table[("A", "f", "confstream")] == (pytest.approx(15.0), 1000)


def test_a_planned_job_reads_its_proxy_arm_scaled_by_the_cap_and_a_factor(tmp_path):
    src = _runs(tmp_path, [{"algo": "A", "fold": "f", "arm": "confstream", "seed": 1, "wall_s": 10.0}])
    jobs = K.plan(src, algos=["A"], folds=["f"], arms=["frozen_reinit"], seeds=[1, 2], n_samples=2000,
                  proxy={"frozen_reinit": "confstream"}, factor=1.5)
    assert [j["cost_s"] for j in jobs] == [pytest.approx(30.0)] * 2
    assert all(j["source"] == "confstream" for j in jobs)


def test_a_cell_never_measured_is_unknown_not_guessed(tmp_path):
    src = _runs(tmp_path, [{"algo": "A", "fold": "f", "arm": "default", "seed": 1, "wall_s": 1.0}])
    (job,) = K.plan(src, algos=["B"], folds=["f"], arms=["default"], seeds=[1], n_samples=1000)
    assert job["cost_s"] is None
    summary = K.summarise(K.plan(src, algos=["A", "B"], folds=["f"], arms=["default"], seeds=[1],
                                 n_samples=1000), workers=2)
    assert summary["n_jobs"] == 2 and summary["n_unknown"] == 1 and summary["cpu_h"] == pytest.approx(1 / 3600)


def test_the_wall_time_schedules_the_longest_jobs_first_on_the_workers():
    jobs = [{"cost_s": c} for c in (10.0, 9.0, 8.0, 3.0, 3.0, 3.0)]
    s = K.summarise(jobs, workers=2)
    assert s["makespan_h"] == pytest.approx(19.0 / 3600)          # LPT: {10, 3, 3, 3} and {9, 8}
    assert s["lower_bound_h"] == pytest.approx(18.0 / 3600)       # max(sum / workers, longest job)


def test_the_cli_prints_a_projection(tmp_path):
    _runs(tmp_path, [{"algo": "DenStream", "fold": "covtype", "arm": "default", "seed": 1, "wall_s": 36.0}])
    proc = subprocess.run([sys.executable, "-m", "conf_pipeline.cost", "--from", str(tmp_path),
                           "--algos", "DenStream", "--folds", "covtype", "--arms", "default", "--seeds", "1,2",
                           "--workers", "1", "--n-samples", "1000"], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "PYTHONHASHSEED": "0"})
    assert proc.returncode == 0, proc.stderr
    assert "2 jobs" in proc.stdout and "0.02 CPU-h" in proc.stdout
