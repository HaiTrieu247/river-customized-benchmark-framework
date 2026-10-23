"""conf_pipeline.repeat: two runs of one protocol compared window by window."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conf_pipeline import repeat as RP

REPO = Path(__file__).resolve().parent.parent
JOB = {"algo": "A", "fold": "f", "seed": 1, "arm": "confstream", "config_id": None}


def _window(w, inc, sha, ari, score=0.5, reason=None, params=None):
    return {"kind": "window", **JOB, "window": w, "incumbent_id": inc, "params": params or {"x": inc},
            "pred_sha": sha, "score": score, "feasible_reason": reason, "lag_member_id": None,
            "lag_pred_sha": None, "lag_ari_win": ari, "wall_s": 1.0}


def _member(w, mid, sha, score, reason=None, ratio=1.0):
    return {**JOB, "window": w, "member_id": mid, "params": {"x": mid}, "pred_sha": sha, "score": score,
            "feasible": reason is None, "feasible_reason": reason, "runtime_ratio": ratio}


def _dir(path: Path, windows, members=None, wall=10.0):
    path.mkdir(parents=True)
    summary = {"kind": "run", **JOB, "status": "ok", "n_samples_cap": 900, "wall_s": wall,
               "lag_ari_win_tail_mean": sum(w["lag_ari_win"] for w in windows) / len(windows)}
    (path / "runs.jsonl").write_text(json.dumps(summary) + "\n")
    (path / "windows.jsonl").write_text("\n".join(json.dumps(w) for w in windows) + "\n")
    if members is not None:
        (path / "members").mkdir()
        (path / "members" / "A__f__seed1__confstream.jsonl").write_text(
            "\n".join(json.dumps(m) for m in members) + "\n")
    return path


def test_two_identical_runs_have_no_divergence_whatever_their_runtimes(tmp_path):
    ws = [_window(1, 0, "aa", 0.1), _window(2, 1, "bb", 0.2)]
    a = _dir(tmp_path / "a", ws, wall=10.0)
    b = _dir(tmp_path / "b", ws, wall=13.0)
    jobs, _ = RP.compare(a, b)
    (row,) = jobs.to_dict("records")
    assert row["identical"] and row["first_divergence_window"] is None
    assert row["wall_ratio"] == pytest.approx(1.3) and row["tail_diff"] == 0.0


def test_the_first_divergence_names_its_window_and_field_and_counts_the_rest(tmp_path):
    a = _dir(tmp_path / "a", [_window(1, 0, "aa", 0.1), _window(2, 1, "bb", 0.2), _window(3, 1, "cc", 0.3)])
    b = _dir(tmp_path / "b", [_window(1, 0, "aa", 0.1), _window(2, 2, "dd", 0.25), _window(3, 1, "cc", 0.3)])
    jobs, windows = RP.compare(a, b)
    (row,) = jobs.to_dict("records")
    assert not row["identical"] and row["first_divergence_window"] == 2
    assert row["first_divergence_field"] == "incumbent_id"
    assert row["n_windows_differing"] == 1 and row["max_abs_lag_ari_diff"] == pytest.approx(0.05)
    assert list(windows["window"]) == [2]


def test_a_divergence_after_a_runtime_shed_is_attributed_to_the_runtime_gate(tmp_path):
    ws_a = [_window(1, 0, "aa", 0.1), _window(2, 1, "bb", 0.2)]
    ws_b = [_window(1, 0, "aa", 0.1), _window(2, 0, "aa", 0.1)]
    ma = [_member(1, 0, "aa", 0.4), _member(1, 1, "bb", 0.6)]
    mb = [_member(1, 0, "aa", 0.4), _member(1, 1, "bb", -1.0, reason="runtime", ratio=31.0)]
    jobs, _ = RP.compare(_dir(tmp_path / "a", ws_a, ma), _dir(tmp_path / "b", ws_b, mb))
    (row,) = jobs.to_dict("records")
    assert row["member_first_divergence_window"] == 1 and row["cause"] == "runtime gate"


def test_a_divergence_with_equal_member_readings_is_a_tie_break(tmp_path):
    ws_a = [_window(1, 0, "aa", 0.1)]
    ws_b = [_window(1, 1, "bb", 0.3)]
    members = [_member(1, 0, "aa", 0.5), _member(1, 1, "bb", 0.5)]
    jobs, _ = RP.compare(_dir(tmp_path / "a", ws_a, members), _dir(tmp_path / "b", ws_b, members))
    assert jobs.iloc[0]["cause"] == "tie-break"


def test_members_that_predicted_differently_are_a_nondeterministic_model(tmp_path):
    ws = [_window(1, 0, "aa", 0.1)]
    jobs, _ = RP.compare(_dir(tmp_path / "a", ws, [_member(1, 0, "aa", 0.5)]),
                         _dir(tmp_path / "b", ws, [_member(1, 0, "zz", 0.5)]))
    row = jobs.iloc[0]
    assert row["member_first_divergence_field"] == "pred_sha" and row["cause"] == "model or scoring"


def test_a_job_on_one_side_only_is_listed_as_missing(tmp_path):
    a = _dir(tmp_path / "a", [_window(1, 0, "aa", 0.1)])
    b = tmp_path / "b"
    b.mkdir()
    (b / "runs.jsonl").write_text("")
    jobs, _ = RP.compare(a, b)
    assert jobs.iloc[0]["status"] == "only in a"


def test_the_gate_replay_counts_members_a_gate_would_have_shed(tmp_path):
    ws = [_window(1, 0, "aa", 0.1)]
    members = [_member(1, 0, "aa", 0.5, ratio=1.0), _member(1, 1, "bb", 0.4, ratio=45.0),
               _member(1, 2, "cc", 0.3, ratio=None)]
    jobs, _ = RP.compare(_dir(tmp_path / "a", ws, members), _dir(tmp_path / "b", ws, members))
    assert jobs.iloc[0]["b_share_over_factor"] == pytest.approx(0.5)     # 1 of 2 members with a ratio


def test_the_cli_writes_the_tables_and_a_report(tmp_path):
    ws = [_window(1, 0, "aa", 0.1)]
    a, b = _dir(tmp_path / "a", ws), _dir(tmp_path / "b", ws)
    proc = subprocess.run([sys.executable, "-m", "conf_pipeline.repeat", "--a", str(a), "--b", str(b),
                           "--out", str(tmp_path / "rep")], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "PYTHONHASHSEED": "0"})
    assert proc.returncode == 0, proc.stderr
    assert {p.name for p in (tmp_path / "rep").iterdir()} >= {"jobs.csv", "windows.csv", "repeat.md"}
