"""conf_pipeline.oracle_compare: the silhouette's tuner, the labels' pick within it, and the labels' tuner."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conf_pipeline import oracle_compare as OC

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def two_arms(tmp_path_factory):
    out = tmp_path_factory.mktemp("e4")
    proc = subprocess.run(
        [sys.executable, "-m", "conf_pipeline.run", "--algos", "DenStream", "--folds", "static_easy",
         "--seeds", "1", "--arms", "confstream,confstream_ari_oracle,confstream_ari_oracle_constrained",
         "--n-samples", "2400", "--h", "300",
         "--e-size", "3", "--e-test", "1", "--freeze-after-window", "2", "--log-members",
         "--out-dir", str(out), "--workers", "2", "--seed-policy", "shared"],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "0"})
    assert proc.returncode == 0, proc.stderr
    return out


def test_the_three_readings_are_paired_on_the_same_tail_windows(two_arms):
    jobs, windows = OC.compare(two_arms)
    jobs = jobs[jobs["oracle_arm"] == "confstream_ari_oracle"]
    windows = windows[windows["oracle_arm"] == "confstream_ari_oracle"]
    (row,) = jobs.to_dict("records")
    assert row["n_windows"] == len(windows) > 0 and windows["window"].min() >= 2
    assert row["a_silhouette"] == pytest.approx(windows["a_silhouette"].mean())
    assert row["a_label_replay"] == pytest.approx(windows["a_label_replay"].mean())
    assert row["a_oracle"] == pytest.approx(windows["a_oracle"].mean())
    assert row["d_replay"] == pytest.approx(row["a_label_replay"] - row["a_silhouette"])
    assert row["d_search"] == pytest.approx(row["a_oracle"] - row["a_label_replay"])
    assert row["d_trajectory"] == pytest.approx(row["a_oracle"] - row["a_silhouette"])


def test_both_oracles_are_read_and_the_constrained_one_is_the_like_for_like_search_comparison(two_arms):
    jobs, _ = OC.compare(two_arms)
    assert set(jobs["oracle_arm"]) == {"confstream_ari_oracle", "confstream_ari_oracle_constrained"}
    assert dict(zip(jobs["oracle_arm"], jobs["like_for_like"])) == {
        "confstream_ari_oracle": False, "confstream_ari_oracle_constrained": True}
    OC.write_compare(two_arms)
    md = (two_arms / "report" / "oracle_compare" / "oracle_compare.md").read_text()
    assert "silhouette's gate" in md and "tie" in md


def test_differences_are_signed_and_never_clipped():
    w = pd.DataFrame({"algo": "A", "fold": "f", "seed": 1, "oracle_arm": "confstream_ari_oracle", "window": [5, 6],
                      "a_silhouette": [0.5, 0.5], "a_label_replay": [0.6, 0.6], "a_oracle": [0.2, 0.3]})
    jobs = OC.job_means(w)
    assert jobs.iloc[0]["d_search"] == pytest.approx(-0.35)


def test_a_dir_without_the_oracle_arm_says_so(tmp_path):
    (tmp_path / "runs.jsonl").write_text("")
    jobs, _ = OC.compare(tmp_path)
    assert jobs.empty


def test_the_cli_writes_its_tables_and_report(two_arms):
    proc = subprocess.run([sys.executable, "-m", "conf_pipeline.oracle_compare", "--out-dir", str(two_arms)],
                          cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "0"})
    assert proc.returncode == 0, proc.stderr
    rep = two_arms / "report" / "oracle_compare"
    assert {p.name for p in rep.iterdir()} >= {"jobs.csv", "pairs.csv", "groups.csv", "windows.csv", "oracle_compare.md"}
