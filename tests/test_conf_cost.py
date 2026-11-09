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


def test_an_exact_inventory_is_projected_job_by_job_with_its_proxy_and_arm_factor(tmp_path):
    src = _runs(tmp_path, [{"algo": "A", "fold": "f", "arm": "confstream", "seed": 1, "wall_s": 10.0},
                           {"algo": "A", "fold": "f", "arm": "default", "seed": 1, "wall_s": 2.0}])
    inventory = [{"algo": "A", "fold": "f", "arm": "random_online", "seed": s, "model_seed": None, "n_samples": 2000}
                 for s in (101, 102)]
    inventory += [{"algo": "A", "fold": "f", "arm": "default", "seed": None, "model_seed": None, "n_samples": 1000},
                  {"algo": "B", "fold": "f", "arm": "default", "seed": None, "model_seed": None, "n_samples": 1000}]
    jobs = K.project(src, inventory, proxy={"random_online": "confstream"}, arm_factor={"random_online": 1.5})
    assert [j["cost_s"] for j in jobs] == [pytest.approx(30.0), pytest.approx(30.0), pytest.approx(2.0), None]
    assert jobs[0]["proxied"] and jobs[0]["source"] == "confstream" and not jobs[2]["proxied"]
    assert [j["seed"] for j in K.over_timeout(jobs, 25.0)] == [101, 102]
    s = K.summarise(jobs, workers=2)
    assert s["n_jobs"] == 4 and s["n_unknown"] == 1


def test_an_algorithm_factor_and_the_slowest_run_change_the_projection_of_that_algorithm_only(tmp_path):
    src = _runs(tmp_path, [{"algo": a, "fold": "f", "arm": "confstream", "seed": s, "wall_s": w}
                           for a, s, w in (("A", 1, 10.0), ("A", 2, 30.0), ("B", 1, 4.0))])
    inventory = [{"algo": a, "fold": "f", "arm": "confstream", "seed": 101, "model_seed": None, "n_samples": 1000}
                 for a in ("A", "B")]
    mean = K.project(src, inventory, arm_factor={"confstream": 2.0}, algo_factor={"A": 1.5})
    assert [j["cost_s"] for j in mean] == [pytest.approx(20.0 * 2.0 * 1.5), pytest.approx(4.0 * 2.0)]
    assert mean[0]["factor"] == pytest.approx(3.0) and mean[1]["factor"] == pytest.approx(2.0)
    slowest = K.project(src, inventory, stat="max")
    assert [j["cost_s"] for j in slowest] == [pytest.approx(30.0), pytest.approx(4.0)]


def test_the_cli_projects_a_campaign_inventory(tmp_path):
    # measured at a cap of 1 000: 500 s x 100 (to the campaign's 100 000) x 1.2 = 60 000 s, past the 12 h timeout
    _runs(tmp_path, [{"algo": "DenStream", "fold": "covtype", "arm": "confstream", "seed": 1, "wall_s": 500.0}])
    out = tmp_path / "proj.csv"
    proc = subprocess.run([sys.executable, "-m", "conf_pipeline.cost", "--from", str(tmp_path), "--campaign",
                           str(REPO / "conf_pipeline" / "campaigns" / "mechanism_pilot.json"), "--workers", "8",
                           "--arm-factor", "confstream=1.2", "--out", str(out)],
                          cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "0"})
    assert proc.returncode == 0, proc.stderr
    # the campaign's inventory with the three E3 branches (2026-11-09): 1 848 jobs
    assert "1848 runnable jobs" in proc.stdout and "1848 jobs (" in proc.stdout and "unmeasured cells" in proc.stdout
    assert "jobs projected past the timeout 43200 s: 3" in proc.stdout       # the three tuner seeds
    assert "DenStream covtype confstream seed=101: 16.67 h" in proc.stdout
    assert len(out.read_text().splitlines()) == 1849


def test_an_algorithm_the_measured_dir_never_ran_is_read_through_a_stated_proxy(tmp_path):
    from conf_pipeline import cost

    _runs(tmp_path, [{"algo": "HPStream", "fold": "covtype", "arm": "confstream", "seed": 1, "wall_s": 100.0}])
    job = {"algo": "HPStreamStableLabels", "fold": "covtype", "arm": "confstream", "seed": 101, "n_samples": None}
    (bare,) = cost.project(tmp_path, [job])
    assert bare["cost_s"] is None                                    # never measured: unknown, not guessed
    (row,) = cost.project(tmp_path, [job], algo_proxy={"HPStreamStableLabels": "HPStream"})
    assert row["cost_s"] == 100.0 and row["algo_source"] == "HPStream" and row["algo_proxied"] is True
