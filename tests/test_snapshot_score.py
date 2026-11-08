"""The tuner scores the clustering a member holds once it has learned the window (the paper's rule,
``score_partition="snapshot"``), not its predict-before-learn labels, unless asked to."""

from __future__ import annotations

import random

import numpy as np
import pytest

from benchmark.algorithms.adapter import make_adapter
from benchmark.tuning.confstream import ConfStreamEnsemble, EnsembleConfig
from benchmark.tuning.space import Param, ParamSpace, SpaceContext
from benchmark.tuning.window import WindowBuffer, silhouette_status

SPACE = ParamSpace("KMeans", (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)))
CTX = SpaceContext(s=1.0, d=2)
H = 60


class _Renamer:
    """Two well-separated sides whose names swap every ``H // 2`` learned points: the model tells the
    sides apart at every moment, and its names change halfway through every window."""

    def __init__(self):
        self.n = 0

    def learn_one(self, x):
        self.n += 1

    def predict_one(self, x):
        side = int(x["a"] > 2.5)
        return side if (self.n // (H // 2)) % 2 == 0 else 1 - side


def _window(seed: int) -> WindowBuffer:
    rng = random.Random(seed)
    buf = WindowBuffer(H)
    for i in range(H):
        c = i % 2
        buf.push({"a": 5.0 * c + rng.gauss(0, 0.1), "b": rng.gauss(0, 0.1)}, c)
    return buf


def _run(score_partition: str):
    cfg = EnsembleConfig(e_size=1, e_test=0, lam=0.05, p_reset=0.0, score_partition=score_partition,
                         runtime_factor=None)
    ens = ConfStreamEnsemble(space=SPACE, ctx=CTX, make_adapter=lambda params, seed: make_adapter("r", _Renamer(), {}),
                             start_params=[{"n_clusters": 2, "halflife": 0.5}], config=cfg,
                             rng=np.random.default_rng(0), seed=0)
    ens.initialise(_window(0), window_index=0)
    buf = _window(1)
    return ens.process_window(buf, window_index=1).outcomes[0], buf


def test_a_model_that_only_renames_its_clusters_is_scored_on_its_clustering_not_on_its_names():
    snap, buf = _run("snapshot")
    seq, _ = _run("sequential")
    truth = [i % 2 for i in range(H)]
    best = silhouette_status(buf.xs, truth).score
    assert snap.score == pytest.approx(best)            # the state after the window separates the sides
    assert seq.score < 0.0 < best                       # the sequential labels join the sides
    assert snap.seq_silhouette == pytest.approx(seq.score)
    assert snap.score_partition == "snapshot" and seq.score_partition == "sequential"
    row = snap.to_dict()
    assert row["score_partition"] == "snapshot" and row["seq_silhouette"] == pytest.approx(seq.score, abs=1e-6)
    assert row["snap_silhouette"] == pytest.approx(best, abs=1e-6) and row["snap_ari_win"] == 1.0
    assert "seq_silhouette" not in seq.to_dict()


def test_the_score_partition_must_be_known():
    with pytest.raises(ValueError, match="score_partition"):
        EnsembleConfig(score_partition="halfway")


def test_new_jobs_score_the_snapshot_and_old_jobs_keep_the_sequential_rule(tmp_path):
    from conf_pipeline.run import build_parser, job_config, plan_jobs

    base = ["--algos", "DBSTREAM", "--folds", "static_overlap", "--arms", "confstream", "--out-dir",
            str(tmp_path / "o")]
    new = plan_jobs(build_parser().parse_args(base))[0]
    old_rule = plan_jobs(build_parser().parse_args(base + ["--score-partition", "sequential"]))[0]
    assert {j["score_partition"] for j in new} == {"snapshot"}
    assert {j["score_partition"] for j in old_rule} == {"sequential"}
    assert all(j.get("snapshot_eval") is True for j in new)
    legacy = {k: v for k, v in new[0].items() if k != "score_partition"}
    assert job_config(legacy).score_partition == "sequential"
    assert job_config(new[0]).score_partition == "snapshot"


def test_the_manifest_tells_the_two_rules_apart_and_keeps_old_dirs_readable(tmp_path):
    from conf_pipeline import manifest as M
    from conf_pipeline.run import build_parser, plan_jobs

    base = ["--algos", "DBSTREAM", "--folds", "static_overlap", "--arms", "confstream", "--out-dir",
            str(tmp_path / "o")]
    job = plan_jobs(build_parser().parse_args(base))[0][0]
    legacy = {k: v for k, v in job.items() if k != "score_partition"}
    seq = {**job, "score_partition": "sequential"}
    assert M.arm_spec(job)["score_partition"] == "snapshot"
    assert "score_partition" not in M.arm_spec(legacy)
    assert M.arm_spec(seq) == M.arm_spec(legacy)


def test_the_primary_ari_reads_the_start_of_window_snapshot_and_the_sequential_one_is_kept():
    from benchmark.datasets import make_scenario
    from conf_pipeline.run import run_ensemble

    ds = make_scenario("static_easy", n_samples=1200, n_features=2, n_clusters=3, seed=3)
    cfg = EnsembleConfig(e_size=2, e_test=1, lam=0.1, p_reset=0.0, runtime_factor=None)
    space = ParamSpace("KMeans", (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)))
    rows = run_ensemble(algo="KMeans", space=space, start=[{"n_clusters": 2, "halflife": 0.5}], dataset=ds,
                        fold="toy", seed=1, arm="confstream", cfg=cfg, h=300, n_samples=1200, snapshot_eval=True)
    wins = [r for r in rows if r["kind"] == "window" and r.get("lag_member_id") is not None]
    assert wins and all(r["ari_basis"] == "snapshot_pre" for r in rows)
    for r in wins:
        assert abs(r["lag_ari_win"] - r["lag_snap_pre_ari_win"]) < 1e-9
        assert r["lag_seq_ari_win"] is not None


def test_a_dir_that_mixes_the_two_bases_cannot_be_reported(tmp_path):
    import json

    from conf_pipeline import report

    rows = [{"kind": "run", "algo": "A", "fold": "f", "seed": 1, "arm": "default", "status": "ok",
             "ari_basis": "snapshot_pre"},
            {"kind": "run", "algo": "B", "fold": "f", "seed": 1, "arm": "default", "status": "ok"}]
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(ValueError, match="mixes ARI bases"):
        report.current_runs(tmp_path)
    (tmp_path / "runs.jsonl").write_text(json.dumps(rows[1]) + "\n")
    assert len(report.current_runs(tmp_path)) == 1            # a legacy dir reads as sequential
