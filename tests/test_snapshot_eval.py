"""``run --snapshot-eval``: the lagged incumbent's coherent partitions of one model state, log-only."""

from __future__ import annotations

from benchmark.datasets import make_scenario
from benchmark.tuning.confstream import EnsembleConfig
from benchmark.tuning.space import Param, ParamSpace
from conf_pipeline.run import build_parser, plan_jobs, run_ensemble, snapshot_fields
from benchmark.tuning.window import WindowBuffer

TOY = ParamSpace("KMeans", (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)))
KEEP = ("incumbent_id", "params", "pred_sha", "score", "lag_member_id", "lag_pred_sha", "lag_ari_win",
        "n_spawned", "n_accepted", "n_reset")


def _run(snapshot_eval: bool):
    ds = make_scenario("static_easy", n_samples=1500, n_features=2, n_clusters=3, seed=3)
    cfg = EnsembleConfig(e_size=3, e_test=1, lam=0.1, p_reset=0.1, runtime_factor=None)
    return run_ensemble(algo="KMeans", space=TOY, start=[{"n_clusters": 2, "halflife": 0.5}], dataset=ds,
                        fold="toy", seed=1, arm="confstream", cfg=cfg, h=300, n_samples=1500,
                        snapshot_eval=snapshot_eval)


def test_the_snapshot_evaluation_changes_no_decision_and_reads_every_lagged_window():
    off, on = _run(False), _run(True)
    key = lambda rows: [tuple(str(r.get(k)) for k in KEEP) for r in rows if r["kind"] == "window"]  # noqa: E731
    assert key(off) == key(on)
    assert not any(k.startswith("lag_snap_") for r in off for k in r)
    wins = [r for r in on if r["kind"] == "window" and r.get("lag_member_id") is not None]
    assert len(wins) == 3
    for r in wins:
        # the sequential labels scored by the snapshot block are the ones lag_ari_win reads
        assert abs(r["lag_snap_seq_ari_win"] - r["lag_ari_win"]) < 1e-9
        assert r["lag_snap_pre_ari_win"] is not None and r["lag_snap_post_ari_win"] is not None
        assert r["lag_snap_pre_n_labels"] >= 1 and r["snapshot_runtime_s"] >= 0
    summary = on[-1]
    assert summary["snapshot_eval"] is True
    assert summary["lag_snap_seq_ari_win_mean"] is not None and summary["lag_snap_pre_ari_win_mean"] is not None


def test_a_window_with_one_true_class_has_no_external_snapshot_reading():
    buf = WindowBuffer(4)
    for i in range(4):
        buf.push({"a": float(i)}, 0)
    out = snapshot_fields({"seq": [0, 0, 1, 1], "pre": [0, 0, 1, 1], "post": None}, buf, ("ari", "nmi"))
    assert out["lag_snap_pre_ari_win"] is None and out["lag_snap_seq_ari_win"] is None
    assert out["lag_snap_pre_vs_seq_ari"] == 1.0 and out["lag_snap_post_silhouette"] is None


def test_the_flag_reaches_the_jobs(tmp_path):
    args = build_parser().parse_args(["--algos", "DBSTREAM", "--folds", "static_overlap", "--arms", "default",
                                      "--out-dir", str(tmp_path / "o"), "--snapshot-eval"])
    jobs = plan_jobs(args)[0]
    assert jobs and all(j.get("snapshot_eval") is True for j in jobs)


def test_the_identity_audit_reads_a_run_made_with_the_flag(tmp_path):
    import json

    from conf_pipeline.identity_audit import main

    rows = [r for r in _run(True) if r["kind"] == "window"]
    for r in rows:
        r.update(algo="KMeans", fold="static_overlap", arm="confstream")
    (tmp_path / "windows.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    md = main(["--read-run", str(tmp_path)])
    text = md.read_text()
    assert "lag_snap_pre_ari_win" in text and (tmp_path / "report" / "snapshot_eval" / "per_group.csv").exists()
