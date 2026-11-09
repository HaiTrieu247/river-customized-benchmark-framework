"""Stream-level scaling (``benchmark.datasets.scaling``) and its wiring into every driver of the benchmark."""

from __future__ import annotations

import cmath

import numpy as np
import pytest

from benchmark.datasets.scaling import RollingStandardised, scale_stream, transform_side


def _river_reference(xs, w):
    from river import preprocessing

    sc = preprocessing.StandardScaler(window_size=w)
    out = []
    for x in xs:
        sc.learn_one(dict(x))
        out.append(sc.transform_one(dict(x)))
    return out


def test_the_stream_is_rivers_rolling_standard_scaler_learned_then_transformed():
    rng = np.random.default_rng(0)
    xs = [{"a": float(v), "b": float(u)} for v, u in rng.normal([5.0, -2.0], [3.0, 0.5], size=(60, 2))]
    got = [x for x, _ in RollingStandardised([(x, i) for i, x in enumerate(xs)], 10)]
    want = _river_reference(xs, 10)
    for g, w in zip(got, want):
        assert g == pytest.approx(w)


def test_labels_pass_untouched_and_non_numeric_features_pass_through():
    stream = RollingStandardised([({"a": 1.0, "tag": "x"}, 7), ({"a": 3.0, "tag": "y"}, 8)], 5)
    rows = list(stream)
    assert [y for _, y in rows] == [7, 8] and [x["tag"] for x, _ in rows] == ["x", "y"]
    assert stream.summary()["n_passthrough"] == 2


def test_a_constant_feature_never_becomes_complex():
    xs = [({"c": 0.1, "a": float(i)}, 0) for i in range(200)]
    for x, _ in RollingStandardised(xs, 7):
        assert not isinstance(x["c"], complex) and cmath.isfinite(x["c"])


def test_a_side_point_is_read_with_the_statistics_so_far_and_teaches_nothing():
    xs = [({"a": float(i)}, 0) for i in range(5)]
    stream = RollingStandardised(xs, 5)
    it = iter(stream)
    for _ in range(5):
        next(it)
    before = stream.scaler.transform_one({"a": 2.0})
    assert transform_side(stream, {"a": 2.0}) == before == stream.scaler.transform_one({"a": 2.0})
    assert scale_stream(xs, "none", None) is xs and transform_side(xs, {"a": 1.0}) == {"a": 1.0}


def test_new_jobs_scale_their_stream_with_a_window_of_h_and_old_jobs_read_raw_coordinates(tmp_path):
    from conf_pipeline.run import build_parser, job_config, plan_jobs

    base = ["--algos", "DBSTREAM", "--folds", "static_overlap", "--arms", "default", "--out-dir", str(tmp_path / "o"),
            "--h", "500"]
    jobs = plan_jobs(build_parser().parse_args(base))[0]
    assert {(j["scaling_rule"], j["stream_scaling"], j["scaling_window"]) for j in jobs} == {("protocol", "rolling_standard", 500)}
    drift = plan_jobs(build_parser().parse_args(base[:-2] + ["--h", "500", "--folds", "scale_drift,scale_drift_std,covtype"]))[0]
    assert {j["fold"]: j["stream_scaling"] for j in drift} == {
        "scale_drift": "none", "scale_drift_std": "rolling_standard", "covtype": "rolling_standard"}
    assert {j["scaling_window"] for j in drift} == {500}
    raw = plan_jobs(build_parser().parse_args(base + ["--stream-scaling", "none"]))[0]
    assert {(j["stream_scaling"], j["scaling_window"]) for j in raw} == {("none", None)}
    with pytest.raises(SystemExit, match="prefix-scaling"):
        plan_jobs(build_parser().parse_args(base + ["--prefix-scaling", "standard"]))


def test_a_run_reads_the_scaled_stream_everywhere_and_records_it():
    from benchmark.datasets import make_scenario
    from benchmark.tuning.confstream import EnsembleConfig
    from benchmark.tuning.space import Param, ParamSpace
    from conf_pipeline.run import run_ensemble

    space = ParamSpace("KMeans", (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)))
    cfg = EnsembleConfig(e_size=1, e_test=0, runtime_factor=None)

    def run(scaling):
        ds = make_scenario("static_easy", n_samples=900, n_features=2, n_clusters=3, seed=3)
        return run_ensemble(algo="KMeans", space=space, start=[{"n_clusters": 3, "halflife": 0.5}], dataset=ds,
                            fold="toy", seed=1, arm="default", cfg=cfg, h=300, n_samples=900,
                            stream_scaling=scaling, scaling_window=300)[-1]

    raw, scaled = run("none"), run("rolling_standard")
    assert raw["stream_scaling"] == "none" and "scaling_window" not in raw
    assert scaled["stream_scaling"] == "rolling_standard" and scaled["scaling_window"] == 300
    assert scaled["stream_scaling_stats"]["n_seen"] == 900
    assert scaled["s_online"] != raw["s_online"]                   # the distance unit is read on the scaled stream


def test_the_manifest_tells_a_scaled_protocol_from_a_raw_one(tmp_path):
    from conf_pipeline import manifest as M
    from conf_pipeline.run import build_parser, plan_jobs

    base = ["--algos", "DBSTREAM", "--folds", "static_overlap", "--arms", "default", "--out-dir", str(tmp_path / "o")]
    scaled = M.build(plan_jobs(build_parser().parse_args(base))[0])
    raw = M.build(plan_jobs(build_parser().parse_args(base + ["--stream-scaling", "none"]))[0])
    assert scaled["protocol"]["scaling_rule"] == "protocol" and "scaling_rule" not in raw["protocol"]
    assert all(v.get("stream_scaling") == "rolling_standard" for v in scaled["streams"].values())
    assert scaled["protocol_fingerprint"] != raw["protocol_fingerprint"]


def test_the_sweep_and_the_campaign_pass_the_same_scaling():
    from conf_pipeline import campaign as CP
    from conf_pipeline.sweep import build_sweep_jobs

    jobs = build_sweep_jobs(["DBSTREAM"], ["static_overlap", "k_drift", "k_drift_std"], n_points=0, seed=1,
                            n_samples=None, h=400)
    assert {j["fold"]: (j["stream_scaling"], j["scaling_window"]) for j in jobs} == {
        "static_overlap": ("rolling_standard", 400), "k_drift": ("none", 400), "k_drift_std": ("rolling_standard", 400)}
    cfg = CP.load_config(CP.C.REPO_ROOT / "conf_pipeline" / "campaigns" / "mechanism_pilot.json")
    argv = CP.run_argv(cfg, cfg["groups"][0], CP.C.REPO_ROOT / "x", 1, dry_run=True)
    assert argv[argv.index("--stream-scaling") + 1] == "protocol" and "--scaling-window" not in argv


def test_the_audits_read_the_same_scaled_stream():
    from conf_pipeline import fidelity as F
    from conf_pipeline import identity_audit as IA
    from conf_pipeline import partition_audit as PA

    shape = F.StreamShape(h=50, n_windows=5)
    raw = IA._scenario_stream("birth", 1, shape)[0]
    scaled = IA._scenario_stream("birth", 1, shape, "rolling_standard", None)[0]
    assert raw[1][1] == scaled[1][1] and raw[1][0] != scaled[1][0]       # labels kept, points scaled
    xs, probes = F.scaled_scenario([{"a": float(i)} for i in range(100)], [0] * 100, [{"a": 1.0}], 50,
                                   "rolling_standard", None)
    assert sorted(probes) == [0, 1] and len(xs) == 100
    wins = PA._windows("static_overlap", 100, 1, 42, "rolling_standard", None)
    assert len(wins) == 2 and abs(np.mean([x[next(iter(x))] for x in wins[1]])) < 1.0


def test_the_protocol_keeps_the_parameter_drift_group_raw_and_scales_its_twins_and_everything_else():
    from conf_pipeline import _common as C

    for f in C.PARAM_DRIFT_FOLDS + C.DRIFT_VARIANT_FOLDS:
        assert C.stream_scaling_of(f) == "none"
    for f in C.REAL_FOLDS + ("static_overlap", "incremental_drift", "evolving") + C.SCALED_TWIN_FOLDS:
        assert C.stream_scaling_of(f) == "rolling_standard"
    assert C.stream_scaling_of("scale_drift", "rolling_standard") == "rolling_standard"
    assert C.stream_scaling_of("covtype", "none") == "none"


def test_a_twin_is_its_drift_stream_point_for_point_in_its_own_group_and_block():
    from conf_pipeline import _common as C

    a = [x for x, _ in C.load_fold("k_drift", n_samples=50)][:50]
    b = [x for x, _ in C.load_fold("k_drift_std", n_samples=50)][:50]
    assert a == b
    assert C.stream_group("k_drift_std") == C.SENSITIVITY_GROUP and C.source_block("k_drift_std") == "k_drift"
    assert C.drift_points("scale_drift_std", None) == C.drift_points("scale_drift", None)
    assert C.is_synthetic("mixed_drift_std") and C.fold_source("k_drift_std")["scaled_twin_of"] == "k_drift"


def test_the_twins_stay_out_of_every_main_table_and_are_read_apart(tmp_path):
    import json

    from conf_pipeline import report

    rows = []
    for fold, arm, v in (("covtype", "default", 0.1), ("covtype", "confstream", 0.2),
                         ("k_drift", "default", 0.3), ("k_drift", "confstream", 0.9),
                         ("k_drift_std", "default", 0.6), ("k_drift_std", "confstream", 0.8)):
        rows.append({"kind": "run", "algo": "DBSTREAM", "fold": fold, "seed": 1, "arm": arm, "status": "ok",
                     "ari_basis": "snapshot_pre", "lag_ari_win_tail_mean": v,
                     "stream_scaling": "none" if fold == "k_drift" else "rolling_standard"})
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert "k_drift_std" not in set(report.current_runs(tmp_path)["fold"])
    assert "k_drift_std" in set(report.current_runs(tmp_path, twins=True)["fold"])
    text = report.scaling_text(report.current_runs(tmp_path))
    assert "mixes two preprocessings" in text and "`param_drift`: none" in text
    t = report.scaling_sensitivity(tmp_path)
    arms = t["arms"].set_index("arm")
    assert arms.loc["confstream", "raw"] == pytest.approx(0.9) and arms.loc["confstream", "scaled"] == pytest.approx(0.8)
    c = t["contrasts"].set_index("contrast").loc["confstream-default"]
    assert c["real_mean"] == pytest.approx(0.1) and c["param_drift_raw_mean"] == pytest.approx(0.6)
    assert c["param_drift_scaled_mean"] == pytest.approx(0.2) and c["drift_minus_real_scaled"] == pytest.approx(0.1)
    paths = report.write_scaling_sensitivity(tmp_path)
    assert (tmp_path / "report" / "scaling_sensitivity" / "scaling_sensitivity.md") in paths
