"""Tests for running the shipped HPO sets. No datasets, no runs."""

from __future__ import annotations

from hpo_pipeline import final_eval as fe

SETS = {
    "DenStream": {
        "default": {"resolved": {"a": {"epsilon": 0.02}, "b": {"epsilon": 0.02}}},
        "arm1_raw_all": {"resolved": {"a": {"epsilon": 5.5}, "b": {"epsilon": 5.5}}},
        "arm2_raw_lodo": {"resolved": {"a": {"epsilon": 5.5}, "b": {"epsilon": 5.5}}},
        "p2r_R_knee_matr_trace_win": {"resolved": {"a": {"epsilon": 3.0}}},
    }
}
ARMS = ["default", "arm1_raw_all", "arm2_raw_lodo", "p2r_R_knee_matr_trace_win", "arm3_sf_lodo"]


def test_identical_sets_share_one_run_per_seed():
    jobs, _ = fe.build_jobs(SETS, ["DenStream"], ["a", "b"], ARMS, seeds=[2, 3])
    shared = [j for j in jobs if j["params"] == {"epsilon": 5.5}]
    assert len(shared) == 4  # 2 datasets x 2 seeds
    assert all(j["arms"] == ["arm1_raw_all", "arm2_raw_lodo"] for j in shared)
    assert len({fe.run_key(j) for j in jobs}) == len(jobs)


def test_missing_sets_are_reported_not_invented():
    jobs, missing = fe.build_jobs(SETS, ["DenStream"], ["a", "b"], ARMS, seeds=[2])
    reasons = {(m["arm"], m["dataset"]): m["reason"] for m in missing}
    assert reasons[("p2r_R_knee_matr_trace_win", "b")] == "no resolved params for this dataset"
    assert reasons[("arm3_sf_lodo", "a")] == "no set"
    assert not any(j["dataset"] == "b" and j["params"] == {"epsilon": 3.0} for j in jobs)


def test_records_fan_out_to_every_arm_and_mark_missing_runs(tmp_path):
    jobs, _ = fe.build_jobs(SETS, ["DenStream"], ["a"], ARMS, seeds=[2])
    out = tmp_path / "runs.jsonl"
    done = next(j for j in jobs if j["params"] == {"epsilon": 5.5})
    row = {k: done[k] for k in ("algo", "dataset", "seed", "n_samples", "params")}
    fe.append_jsonl(out, {**row, "status": "ok", "result": {
        "quality": {"ari": 0.3}, "efficiency": {"throughput_sps": 10.0},
        "windows": [{"start": 0, "end": 999, "ari": 0.2, "window_metrics": {"ari": 0.25}}],
        "checkpoints": [{"n_seen": 100, "runtime_s": 1.0, "memory_bytes": 2**20,
                         "process_rss_bytes": None, "n_micro_clusters": 7}],
    }})
    records = fe.load_records(jobs, out)
    by_arm = {r["arm"]: r["status"] for r in records}
    assert by_arm == {"default": "missing", "arm1_raw_all": "ok", "arm2_raw_lodo": "ok",
                      "p2r_R_knee_matr_trace_win": "missing"}

    summary = fe.summary_frame(records)
    assert set(summary[summary.status == "ok"].ari) == {0.3}
    windows = fe.window_frame(records)
    assert list(windows.window) == [0.25, 0.25] and list(windows.position) == [499, 499]
    checkpoints = fe.checkpoint_frame(records)
    assert list(checkpoints.memory_mb) == [1.0, 1.0]


def test_a_changed_set_is_not_served_by_an_old_run(tmp_path):
    jobs, _ = fe.build_jobs(SETS, ["DenStream"], ["a"], ["arm1_raw_all"], seeds=[2])
    out = tmp_path / "runs.jsonl"
    old = {**{k: jobs[0][k] for k in ("algo", "dataset", "seed", "n_samples")},
           "params": {"epsilon": 1.8}, "status": "ok"}
    fe.append_jsonl(out, old)
    assert fe.load_records(jobs, out)[0]["status"] == "missing"
