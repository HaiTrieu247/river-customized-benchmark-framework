"""conf_pipeline.sweep: the Sobol sweep and the a-posteriori oracle."""

from __future__ import annotations

import json

from conf_pipeline import sweep
from conf_pipeline.spaces import SPACES


def test_sobol_units_are_deterministic_and_start_with_the_river_default():
    a = sweep.sobol_units(SPACES["DenStream"], 4, seed=1)
    b = sweep.sobol_units(SPACES["DenStream"], 4, seed=1)
    assert a == b and a[0] is None and len(a) == 5
    assert all(
        set(u) == set(SPACES["DenStream"].names) and all(0 <= v <= 1 for v in u.values())
        for u in a[1:]
    )


def test_sweep_runs_resume_and_select_the_best_config(tmp_path):
    jobs = sweep.build_sweep_jobs(["DenStream"], ["static_easy"], n_points=2, seed=1, n_samples=900, h=300)
    assert [j["config_id"] for j in jobs] == [0, 1, 2] and jobs[0]["unit_override"] is None
    sweep.run_sweep(jobs, tmp_path, workers=1, timeout=300)
    rows = [json.loads(l) for l in (tmp_path / "sweep.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and {r["status"] for r in rows} <= {"ok", "error"}
    sweep.run_sweep(jobs, tmp_path, workers=1, timeout=300)          # resume: nothing appended
    assert len((tmp_path / "sweep.jsonl").read_text().splitlines()) == 3
    oracle = sweep.select_oracle(tmp_path)
    entry = oracle["DenStream"]["static_easy"]
    best = max((r for r in rows if r["status"] == "ok"), key=lambda r: r["lag_ari_win_mean"])
    assert entry["config_id"] == best["config_id"] and entry["params"] == best["final_params"]
    assert json.loads((tmp_path / "oracle_static.json").read_text()) == oracle


def test_select_oracle_skips_errored_configs_and_reports_them(tmp_path):
    rows = [{"algo": "A", "fold": "f", "seed": 1, "config_id": 0, "status": "error", "error": "boom"},
            {"algo": "A", "fold": "f", "seed": 1, "config_id": 1, "status": "ok", "lag_ari_win_mean": 0.2,
             "unit_override": {"x": 0.5}, "final_params": {"x": 1}}]
    (tmp_path / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    o = sweep.select_oracle(tmp_path)
    assert o["A"]["f"]["config_id"] == 1 and o["A"]["f"]["n_error"] == 1


def test_select_oracle_records_an_algo_fold_whose_sweep_produced_no_ok_point(tmp_path):
    rows = [{"algo": "A", "fold": "g", "seed": 1, "config_id": c, "status": "error", "error": "boom"}
            for c in (0, 1)]
    rows.append({"algo": "A", "fold": "f", "seed": 1, "config_id": 0, "status": "ok",
                 "lag_ari_win_mean": 0.2, "unit_override": None, "final_params": {"x": 1}})
    (tmp_path / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    o = sweep.select_oracle(tmp_path)
    assert o["A"]["g"] == {"config_id": None, "unit": None, "params": None, "n_ok": 0, "n_error": 2}
    assert o["A"]["f"]["config_id"] == 0
    assert json.loads((tmp_path / "oracle_static.json").read_text()) == o
