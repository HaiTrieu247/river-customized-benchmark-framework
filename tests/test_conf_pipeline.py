"""conf_pipeline: protocol constants, the per-fold driver and the CLI."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark.algorithms.registry import accepts_seed, get_algorithm
from benchmark.datasets import make_scenario
from benchmark.tuning.confstream import EnsembleConfig
from benchmark.tuning.space import Param, ParamSpace, SpaceContext
from conf_pipeline import _common, run, spaces
from conf_pipeline.run import run_ensemble

REPO = Path(__file__).resolve().parents[1]


# --- protocol helpers -------------------------------------------------------------


def test_lambda_half_life_is_a_fraction_of_the_windows():
    assert _common.lambda_for(581) == pytest.approx(1 / 58)
    assert _common.lambda_for(80) == pytest.approx(1 / 8)
    assert _common.lambda_for(5) == 1.0  # half-life floors at one window
    assert _common.lambda_for(200, half_life_frac=0.5) == pytest.approx(1 / 100)


def test_child_env_pins_the_hash_seed_and_one_thread_per_native_library(monkeypatch):
    # 8 worker children each running sklearn KMeans/DBSCAN/silhouette would
    # otherwise each start a thread per core, and the incumbent's throughput
    # would measure the contention rather than the configuration.
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    env = _common.child_env()
    assert env["PYTHONHASHSEED"] == "0"
    assert {k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")} == {
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }


def test_estimate_scale_is_the_median_pairwise_distance():
    xs = [{"a": 0.0}, {"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    # pairwise distances: 1,2,3,1,2,1 -> median 1.5
    assert _common.estimate_scale(xs, np.random.default_rng(0)) == pytest.approx(1.5)


def test_estimate_scale_is_zero_on_identical_points():
    xs = [{"a": 1.0, "b": 2.0}] * 50
    assert _common.estimate_scale(xs, np.random.default_rng(0)) == 0.0


def test_load_fold_knows_the_protocol_folds_and_the_synthetic_ones():
    ds = _common.load_fold("static_easy", n_samples=500)
    assert len(ds) == 500 and ds.has_labels
    with pytest.raises(KeyError):
        _common.load_fold("no_such_fold")


def test_protocol_lists_eight_algorithms_eighteen_folds_four_arms():
    assert len(_common.ALGOS) == 8 and len(_common.FOLDS) == 18 and _common.ARMS == ("default","confstream","random_online","frozen_prefix")
    assert _common.SEEDS == (1,2,3,4,5) and _common.N_SAMPLES_CAP == 100_000


@pytest.mark.parametrize("name", ["scale_drift", "k_drift", "static_overlap"])
def test_load_fold_knows_synthetic_and_param_drift_folds(name):
    ds = _common.load_fold(name, n_samples=600)
    assert len(ds) == 600
    pts = _common.drift_points(name, 600)
    assert (pts == (200, 400)) if name in ("scale_drift", "k_drift") else (pts == ())


# --- search spaces ------------------------------------------------------------------


@pytest.mark.parametrize("algo", _common.ALGOS)
def test_every_protocol_algorithm_has_a_space_whose_points_build(algo):
    space = spaces.SPACES[algo]
    ctx = SpaceContext(s=100.0, d=10)
    for u_val in (0.0, 0.5, 1.0):
        params = space.resolve({n: u_val for n in space.names}, ctx)
        if accepts_seed(algo):
            params["seed"] = 1
        adapter = get_algorithm(algo, **params)
        assert adapter is not None


@pytest.mark.parametrize("algo", _common.ALGOS)
def test_start_params_are_river_defaults_clamped_to_the_window(algo):
    start = spaces.start_params(algo, h=1000, d=10)
    space = spaces.SPACES[algo]
    ctx = SpaceContext(s=100.0, d=10)
    u = space.unit_point(start, ctx)  # every tunable dimension is present
    assert set(u) == set(space.names)
    for k, v in space.fixed.items():
        assert start[k] == v
    if "n_samples_init" in start:
        assert start["n_samples_init"] <= 1000
    # and River accepts it as-is: the default arm passes it straight in
    kw = dict(start)
    if accepts_seed(algo):
        kw["seed"] = 1
    assert get_algorithm(algo, **kw) is not None


@pytest.mark.parametrize(
    "algo", ["DBSTREAM", "STREAMKMeans", "SOStream", "BIRCH", "DenStreamPatched"]
)
def test_new_spaces_build_and_start_from_river_defaults(algo):
    space = spaces.SPACES[algo]
    ctx = SpaceContext(s=100.0, d=10)
    for u_val in (0.0, 0.5, 1.0):
        params = space.resolve({n: u_val for n in space.names}, ctx)
        if accepts_seed(algo):
            params["seed"] = 1
        assert get_algorithm(algo, **params) is not None
    start = spaces.start_params(algo, h=1000, d=10)
    assert set(space.unit_point(start, ctx)) == set(space.names)


def test_streamkmeans_start_has_halflife_and_chunk_within_window():
    start = spaces.start_params("STREAMKMeans", h=300, d=4)
    assert start["halflife"] == 0.5 and start["chunk_size"] <= 300 and start["n_clusters"] == 2


def test_birch_start_is_river_default_threshold_zero_with_phase_three_off():
    start = spaces.start_params("BIRCH", h=1000, d=4)
    assert start["threshold"] == 0.0 and start["n_macro_clusters"] is None


def test_denstreampatched_reuses_denstreams_space_and_river_defaults_via_the_alias():
    assert spaces.SPACES["DenStreamPatched"].names == spaces.SPACES["DenStream"].names
    assert spaces.SPACES["DenStreamPatched"].fixed == spaces.SPACES["DenStream"].fixed
    assert _common.RIVER_CLASS["DenStreamPatched"] == "DenStream"
    assert _common.river_defaults("DenStreamPatched") == _common.river_defaults("DenStream")


def test_hpstream_start_snaps_its_2000_sample_initialisation_into_the_space():
    # River's 2000 is above h=1000, and above the largest declared choice: the
    # start must be a point of the space, so it lands on 1000 either way.
    assert spaces.start_params("HPStream", h=1000, d=10)["n_samples_init"] == 1000
    assert spaces.start_params("HPStream", h=5000, d=10)["n_samples_init"] == 1000
    assert spaces.start_params("HPStream", h=300, d=10)["n_samples_init"] == 200


def test_hpstream_start_projects_on_sixty_percent_of_the_dimensions():
    # River's None means "60% of the dimensions seen"; written out for the space.
    assert spaces.start_params("HPStream", h=1000, d=10)["n_projected_dimensions"] == 6
    assert spaces.start_params("HPStream", h=1000, d=33)["n_projected_dimensions"] == 20
    with pytest.raises(ValueError):
        spaces.start_params("HPStream", h=1000, d=None)


def test_clamp_init_buffers_caps_at_the_largest_declared_choice_within_the_window():
    c = spaces.clamp_init_buffers
    assert c("DenStream", {"n_samples_init": 1000, "mu": 2.0}, 300) == {"n_samples_init": 200, "mu": 2.0}
    assert c("DenStream", {"n_samples_init": 500}, 300)["n_samples_init"] == 200
    assert c("DenStream", {"n_samples_init": 1000}, 600)["n_samples_init"] == 500
    assert c("DenStream", {"n_samples_init": 200}, 300)["n_samples_init"] == 200     # within h: kept
    assert c("DenStream", {"n_samples_init": 1000}, 1000)["n_samples_init"] == 1000
    assert c("HPStream", {"n_samples_init": 0}, 300)["n_samples_init"] == 0
    assert c("STREAMKMeans", {"chunk_size": 1000, "n_clusters": 5}, 300) == {"chunk_size": 100, "n_clusters": 5}
    # every declared choice above h: the smallest one, as start_params does
    assert c("DenStream", {"n_samples_init": 1000}, 100)["n_samples_init"] == 200
    # nothing to clamp, or no declared choices to snap to (a plain cap at h)
    assert c("DBSTREAM", {"clustering_threshold": 1.0}, 300) == {"clustering_threshold": 1.0}
    assert c("KMeans", {"n_clusters": 3}, 300) == {"n_clusters": 3}
    assert c("NoProtocolSpace", {"chunk_size": 1000}, 300) == {"chunk_size": 300}
    params = {"n_samples_init": 1000}
    assert c("DenStream", params, 300) is not params and params == {"n_samples_init": 1000}


def test_adapter_factory_builds_every_model_with_its_init_buffer_within_the_window():
    make = run._make_adapter_factory("DenStream", 300)
    assert make({"n_samples_init": 1000}, 1).model.n_samples_init == 200
    assert make({"n_samples_init": 200}, 1).model.n_samples_init == 200


def test_batch_external_scores_the_lagged_stream_in_one_pass():
    y = [0, 0, 0, 1, 1, 1, 2, 2]
    pred = [10, 10, 10, 20, 20, 30, 30, 30]
    out = run.batch_external(y, pred, ("ari", "nmi", "purity"))
    assert set(out) == {"ari", "nmi", "purity"}
    assert 0.0 < out["ari"] < 1.0 and 0.0 < out["nmi"] < 1.0
    # purity: cluster 10 -> 3 of class 0, 20 -> 2 of class 1, 30 -> 2 of class 2
    assert out["purity"] == pytest.approx(7 / 8)
    assert run.batch_external([], [], ("ari",)) == {"ari": None}
    # thousands of fresh labels must not cost more than a linear pass
    import time
    n = 200_000
    yy = [i % 7 for i in range(n)]
    pp = list(range(n))
    t0 = time.perf_counter()
    run.batch_external(yy, pp, ("ari", "nmi", "purity"))
    assert time.perf_counter() - t0 < 5.0


# --- the driver ---------------------------------------------------------------------

TOY = ParamSpace(
    "KMeans", (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)),
)


def _toy_run(arm="confstream", n_samples=1500, h=300, seed=1, log=None, **kw):
    ds = make_scenario("static_easy", n_samples=n_samples, n_features=2, n_clusters=3, seed=3)
    cfg = EnsembleConfig(e_size=3, e_test=1, lam=0.1, p_reset=0.0)
    return run_ensemble(
        algo="KMeans", space=TOY, start=[{"n_clusters": 2, "halflife": 0.5}],
        dataset=ds, fold="toy", seed=seed, arm=arm, cfg=cfg, h=h, n_samples=n_samples,
        member_log=log, **kw,
    )


def test_run_ensemble_scores_every_window_after_the_calibration_one():
    rows = _toy_run()
    windows = [r for r in rows if r["kind"] == "window"]
    runs = [r for r in rows if r["kind"] == "run"]
    assert len(windows) == 4 and [w["window"] for w in windows] == [1, 2, 3, 4]
    assert windows[0]["start"] == 300 and windows[0]["end"] == 599
    assert windows[0]["lag_score"] is None  # nothing to lag behind yet
    assert windows[1]["lag_member_id"] == windows[0]["incumbent_id"]
    assert all(-1.0 <= w["score"] <= 1.0 for w in windows)
    assert all(w["n_active"] == 3 and w["n_spawned"] == 1 for w in windows)
    (summary,) = runs
    assert summary["status"] == "ok" and summary["n_windows"] == 4
    assert summary["s_online"] > 0 and summary["n_calibration_windows"] == 1
    assert summary["lam"] == pytest.approx(1 / max(1, round(0.1 * 5)))
    assert 0.0 <= summary["lag_ari_cumulative"] <= 1.0
    assert summary["arm"] == "confstream" and summary["algo"] == "KMeans"
    assert summary["start_params"] == {"n_clusters": 2, "halflife": 0.5}


def test_window_rows_carry_table_a_readings_for_the_lagged_member():
    rows = _toy_run()
    wins = [r for r in rows if r["kind"] == "window"]
    (summary,) = [r for r in rows if r["kind"] == "run"]
    assert wins[0]["lag_ari_A_kmeans"] is None          # no lag yet
    assert all(w["lag_n_micro_centers"] is not None for w in wins[1:])
    assert summary["lag_n_micro_centers_mean"] is not None
    assert summary["lag_ari_A_kmeans_mean"] is not None and summary["lag_ari_A_dbscan_mean"] is not None
    assert summary["lag_table_a_error_share"] == 0.0 and summary["lag_table_a_first_error"] is None


def test_table_a_errors_are_counted_on_the_summary_with_the_first_message(monkeypatch):
    def no_centres(model):
        raise RuntimeError("no centres here")

    monkeypatch.setattr(run, "micro_centers", no_centres)
    rows = _toy_run()
    wins = [r for r in rows if r["kind"] == "window"]
    (summary,) = [r for r in rows if r["kind"] == "run"]
    assert all(w["lag_table_a_error"] == "RuntimeError: no centres here" for w in wins[1:])
    assert summary["lag_table_a_error_share"] == 1.0
    assert summary["lag_table_a_first_error"] == "RuntimeError: no centres here"


def test_every_window_row_carries_the_run_cap():
    rows = _toy_run(n_samples=1500)
    wins = [r for r in rows if r["kind"] == "window"]
    (summary,) = [r for r in rows if r["kind"] == "run"]
    assert [w["n_samples_cap"] for w in wins] == [1500] * 4 and summary["n_samples_cap"] == 1500


def test_switch_cost_is_the_lagged_ari_after_a_switch_minus_the_rest():
    rows = [
        {"incumbent_id": 0, "lag_ari_win": None},
        {"incumbent_id": 0, "lag_ari_win": 0.5},
        {"incumbent_id": 1, "lag_ari_win": 0.2},   # switch
        {"incumbent_id": 1, "lag_ari_win": 0.6},
        {"incumbent_id": 2, "lag_ari_win": 0.1},   # switch
        {"incumbent_id": 3, "lag_ari_win": None},  # switch without a lagged reading: skipped
    ]
    assert run.switch_cost(rows) == pytest.approx((0.2 + 0.1) / 2 - (0.5 + 0.6) / 2)
    assert run.switch_cost(rows[:2]) is None                 # no switch
    assert run.switch_cost([rows[1], rows[5]]) is None       # the switch side has no reading
    assert run.switch_cost([rows[0], rows[2]]) is None       # the other side has no reading


def test_summary_reports_tuning_overhead_and_switch_cost():
    (s,) = [r for r in _toy_run() if r["kind"] == "run"]
    assert s["tuning_overhead"] >= 1.0
    assert s["tuning_overhead"] == pytest.approx(s["ensemble_runtime_s"] / s["incumbent_runtime_s"])
    assert s["switch_cost"] is None or isinstance(s["switch_cost"], float)
    (d,) = [r for r in _toy_run(arm="default") if r["kind"] == "run"]
    assert d["tuning_overhead"] == pytest.approx(1.0)
    assert d["switch_cost"] is None                  # one member never switches


def test_drift_points_are_reported_on_the_summary_row():
    (summary,) = [r for r in _toy_run() if r["kind"] == "run"]
    assert summary["drift_points"] == []  # "toy" is not a param-drift scenario

    ds = _common.load_fold("scale_drift", n_samples=900)
    cfg = EnsembleConfig(e_size=3, e_test=1, lam=0.1, p_reset=0.0)
    rows = run_ensemble(
        algo="KMeans", space=TOY, start=[{"n_clusters": 2, "halflife": 0.5}],
        dataset=ds, fold="scale_drift", seed=1, arm="default", cfg=cfg, h=300, n_samples=900,
    )
    (summary,) = [r for r in rows if r["kind"] == "run"]
    assert summary["drift_points"] == [300, 600]


def test_default_arm_runs_a_single_untuned_member():
    rows = _toy_run(arm="default")
    windows = [r for r in rows if r["kind"] == "window"]
    assert all(w["n_active"] == 1 and w["n_spawned"] == 0 for w in windows)
    assert all(w["params"] == {"n_clusters": 2, "halflife": 0.5} for w in windows)
    (summary,) = [r for r in rows if r["kind"] == "run"]
    assert summary["e_size"] == 1 and summary["e_test"] == 0


def test_member_log_receives_one_line_per_member_per_window(tmp_path):
    log = tmp_path / "members.jsonl"
    _toy_run(log=log)
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    # window 1: 3 active; windows 2-4: 3 active + 1 on test
    assert len(lines) == 3 + 3 * 4
    assert {l["window"] for l in lines} == {1, 2, 3, 4}
    assert all("member_id" in l and "score" in l for l in lines)


def test_arm_config_maps_each_arm_to_its_regime():
    base = EnsembleConfig(e_size=20, e_test=10)
    assert run.arm_config("default", base, 20).e_size == 1
    assert run.arm_config("random_online", base, 20).sampler == "uniform"
    assert run.arm_config("frozen_prefix", base, 20).freeze_after_window == 20
    assert run.arm_config("oracle_static", base, 20).e_size == 1
    assert run.arm_config("confstream", base, 20) == EnsembleConfig(e_size=20, e_test=10)
    with pytest.raises(ValueError): run.arm_config("nope", base, 20)


def test_random_online_and_frozen_prefix_arms_run_end_to_end():
    rows = _toy_run(arm="random_online")
    (s,) = [r for r in rows if r["kind"] == "run"]
    assert s["status"] == "ok" and s["sampler"] == "uniform"
    rows = _toy_run(arm="frozen_prefix", freeze_after_window=2)
    wins = [r for r in rows if r["kind"] == "window"]; (s,) = [r for r in rows if r["kind"] == "run"]
    assert [w["frozen"] for w in wins] == [False, True, True, True]
    assert s["frozen_at_window"] == 2 and s["frozen_params"] == wins[-1]["params"]
    assert s["lag_ari_win_tail_mean"] is not None


def test_frozen_prefix_on_a_stream_shorter_than_the_prefix_never_freezes():
    rows = _toy_run(arm="frozen_prefix", freeze_after_window=50)
    (s,) = [r for r in rows if r["kind"] == "run"]
    assert s["status"] == "ok" and s["frozen_at_window"] is None and s["lag_ari_win_tail_mean"] is None


def test_a_cap_longer_than_the_stream_sets_lambda_from_the_real_length():
    # The 100k cap is the default; a 1500-sample fold has 5 windows, not 333.
    ds = make_scenario("static_easy", n_samples=1500, n_features=2, n_clusters=3, seed=3)
    rows = run_ensemble(
        algo="KMeans", space=TOY, start=[{"n_clusters": 2, "halflife": 0.5}], dataset=ds,
        fold="toy", seed=1, arm="default", cfg=EnsembleConfig(e_size=3, e_test=1, p_reset=0.0),
        h=300, n_samples=100_000,
    )
    (s,) = [r for r in rows if r["kind"] == "run"]
    assert s["status"] == "ok" and s["n_windows"] == 4
    assert s["lam"] == pytest.approx(1 / max(1, round(0.1 * 5)))


def test_a_do_nothing_member_does_not_poison_the_ensemble(tmp_path):
    # The verification probe of Task 11a: an initial child resolved to
    # n_samples_init=1000 (> h) buffered its whole window, predicted a single
    # label and was the fastest member; the runtime cap measured River's default
    # against it, every score fell to -1 and the runtime tie-break crowned the
    # do-nothing member (lagged ARI 0.0 against the default arm's 0.77).
    log = tmp_path / "members.jsonl"
    ds = make_scenario("static_easy", n_samples=900, n_features=2, n_clusters=3, seed=3)
    rows = run_ensemble(
        algo="DenStream", space=spaces.SPACES["DenStream"],
        start=[spaces.start_params("DenStream", h=300, d=2)], dataset=ds, fold="static_easy",
        seed=1, arm="confstream", cfg=EnsembleConfig(e_size=2, e_test=1, lam=0.1, p_reset=0.0),
        h=300, n_samples=900, member_log=log, freeze_after_window=1,
    )
    wins = [r for r in rows if r["kind"] == "window"]
    (summary,) = [r for r in rows if r["kind"] == "run"]
    assert summary["lag_ari_win_mean"] >= 0.5
    for w in wins:
        for prefix in ("", "lag_"):
            if w.get(f"{prefix}defined"):
                assert not (w[f"{prefix}feasible"] is False
                            and w[f"{prefix}feasible_reason"] == "runtime"), w
    members = [json.loads(l) for l in log.read_text().splitlines()]
    assert members
    for m in members:
        assert not (m["silhouette_defined"] and m["feasible"] is False
                    and m["feasible_reason"] == "runtime"), m
        assert m["params"]["n_samples_init"] <= 300, m


def test_a_child_drawn_above_the_window_is_recorded_and_built_at_the_largest_choice_below_it(
    tmp_path, monkeypatch
):
    from benchmark.tuning import space as space_mod

    space = spaces.SPACES["DenStream"]
    start = spaces.start_params("DenStream", h=300, d=2)
    ctx = SpaceContext(s=1.0, d=2)
    assert space.resolve({**space.unit_point(start, ctx), "n_samples_init": 1.0}, ctx)["n_samples_init"] == 1000

    real_offspring = space_mod.SamplerState.offspring

    def offspring_at_1000(self, parent, rng, lam):
        child, state = real_offspring(self, parent, rng, lam)
        return {**child, "n_samples_init": 1.0}, state       # unit 1.0 -> 1000 samples

    monkeypatch.setattr(space_mod.SamplerState, "offspring", offspring_at_1000)
    built = []
    real_get = run.get_algorithm

    def recording_get(algo, **kw):
        built.append(real_get(algo, **kw))
        return built[-1]

    monkeypatch.setattr(run, "get_algorithm", recording_get)
    log = tmp_path / "members.jsonl"
    ds = make_scenario("static_easy", n_samples=900, n_features=2, n_clusters=3, seed=3)
    run_ensemble(
        algo="DenStream", space=space, start=[start], dataset=ds, fold="static_easy", seed=1,
        arm="confstream", cfg=EnsembleConfig(e_size=2, e_test=1, lam=0.1, p_reset=0.0),
        h=300, n_samples=900, member_log=log,
    )
    members = [json.loads(l) for l in log.read_text().splitlines()]
    children = [m for m in members if m["parent_id"] is not None]
    assert {m["member_id"] for m in children} == {1, 2}
    assert all(m["params"]["n_samples_init"] == 200 for m in children)
    # start + initial child + one test child per window, every one built with 200
    assert len(built) == 4 and all(a.model.n_samples_init == 200 for a in built)


def test_a_unit_override_start_is_recorded_with_its_init_buffer_within_the_window():
    ds = make_scenario("static_easy", n_samples=900, n_features=2, n_clusters=3, seed=3)
    rows = run_ensemble(
        algo="DenStream", space=spaces.SPACES["DenStream"],
        start=[spaces.start_params("DenStream", h=300, d=2)], dataset=ds, fold="static_easy",
        seed=1, arm="sweep", cfg=EnsembleConfig(e_size=2, e_test=1, lam=0.1, p_reset=0.0),
        h=300, n_samples=900, unit_override={"n_samples_init": 1.0},
    )
    wins = [r for r in rows if r["kind"] == "window"]
    (s,) = [r for r in rows if r["kind"] == "run"]
    assert s["start_params"]["n_samples_init"] == 200
    assert all(w["params"]["n_samples_init"] == 200 for w in wins)


def test_unit_override_replaces_the_start_configuration():
    rows = _toy_run(arm="sweep", unit_override={"n_clusters": 1.0, "halflife": 0.0})
    wins = [r for r in rows if r["kind"] == "window"]
    assert all(w["params"]["n_clusters"] == 8 for w in wins)   # log_int hi
    (s,) = [r for r in rows if r["kind"] == "run"]
    assert s["e_size"] == 1 and s["unit_override"] == {"n_clusters": 1.0, "halflife": 0.0}


# --- CLI ------------------------------------------------------------------------------


def _cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "conf_pipeline.run", *args],
        capture_output=True, text=True, cwd=str(REPO),
        env={**os.environ, "PYTHONHASHSEED": "0", **(env or {})},
    )


def test_cli_dry_run_lists_the_full_protocol_by_default(tmp_path):
    proc = _cli("--dry-run", "--out-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert "2880 jobs" in proc.stdout   # 8 algos x 18 folds x 5 seeds x 4 arms


def test_cli_oracle_arm_without_an_oracle_entry_is_skipped_not_run(tmp_path):
    (tmp_path / "oracle_static.json").write_text(json.dumps({"DenStream": {}}))
    proc = _cli("--algos", "DenStream", "--folds", "static_easy", "--seeds", "1", "--arms", "oracle_static",
                "--n-samples", "900", "--h", "300", "--out-dir", str(tmp_path), "--workers", "1",
                "--oracle-file", str(tmp_path / "oracle_static.json"))
    assert proc.returncode == 0, proc.stderr
    runs = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert runs[0]["status"] == "skipped" and "no oracle" in runs[0]["error"]


def test_cli_oracle_entry_without_an_ok_sweep_point_is_skipped_with_that_reason(tmp_path):
    entry = {"config_id": None, "unit": None, "params": None, "n_ok": 0, "n_error": 3}
    (tmp_path / "oracle_static.json").write_text(json.dumps({"DenStream": {"static_easy": entry}}))
    proc = _cli("--algos", "DenStream", "--folds", "static_easy", "--seeds", "1", "--arms", "oracle_static",
                "--n-samples", "900", "--h", "300", "--out-dir", str(tmp_path), "--workers", "1")
    assert proc.returncode == 0, proc.stderr
    runs = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert [r["status"] for r in runs] == ["skipped"]
    assert runs[0]["error"] == "sweep produced no ok point for DenStream/static_easy"
    assert not (tmp_path / "windows.jsonl").exists()


def test_cli_child_and_parent_round_trip_with_resume(tmp_path):
    common = ["--algos", "DenStream", "--folds", "static_easy", "--seeds", "1",
              "--arms", "default", "--n-samples", "900", "--h", "300",
              "--e-size", "2", "--e-test", "1", "--out-dir", str(tmp_path), "--workers", "1"]
    first = _cli(*common)
    assert first.returncode == 0, first.stderr
    runs = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    windows = [json.loads(l) for l in (tmp_path / "windows.jsonl").read_text().splitlines()]
    assert len(runs) == 1 and runs[0]["status"] == "ok"
    assert runs[0]["algo"] == "DenStream" and runs[0]["fold"] == "static_easy"
    assert len(windows) == 2 and windows[-1]["window"] == 2
    assert runs[0]["n_samples_cap"] == 900 and all(w["n_samples_cap"] == 900 for w in windows)

    second = _cli(*common)
    assert second.returncode == 0, second.stderr
    assert "0 to run" in second.stdout
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1


def test_cli_oracle_arm_runs_the_oracle_point_and_resumes_by_config_id(tmp_path):
    # No --oracle-file: the oracle is read from <out-dir>/oracle_static.json.
    (tmp_path / "oracle_static.json").write_text(json.dumps(
        {"DenStream": {"static_easy": {"config_id": 3, "unit": {"epsilon": 0.9}, "params": {}}}}
    ))
    args = ("--algos", "DenStream", "--folds", "static_easy,static_overlap", "--seeds", "1",
            "--arms", "oracle_static", "--n-samples", "900", "--h", "300", "--out-dir", str(tmp_path),
            "--workers", "1", "--log-members")
    first = _cli(*args)
    assert first.returncode == 0, first.stderr
    # one member log per job, never a file shared by concurrent children
    assert [p.name for p in (tmp_path / "members").iterdir()] == ["DenStream__static_easy__seed1__oracle_static__cfg3.jsonl"]
    second = _cli(*args)
    assert second.returncode == 0 and "0 to run" in second.stdout
    runs = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert sorted(r["status"] for r in runs) == ["ok", "skipped"]   # the skip is not written twice
    (ok,) = [r for r in runs if r["status"] == "ok"]
    assert ok["config_id"] == 3 and ok["unit_override"] == {"epsilon": 0.9} and ok["e_size"] == 1


def test_oracle_entry_missing_the_unit_key_fails_loudly(tmp_path):
    # A malformed oracle entry (no "unit" key at all, not even null) must not
    # be silently treated as "no override" -- it is a broken sweep output, not
    # a River-default request.
    (tmp_path / "oracle_static.json").write_text(json.dumps(
        {"DenStream": {"static_easy": {"config_id": 3}}}
    ))
    proc = _cli("--algos", "DenStream", "--folds", "static_easy", "--seeds", "1",
                "--arms", "oracle_static", "--n-samples", "900", "--h", "300",
                "--out-dir", str(tmp_path), "--oracle-file", str(tmp_path / "oracle_static.json"),
                "--dry-run")
    assert proc.returncode != 0
    assert "unit" in proc.stderr


def test_resume_counts_an_errored_attempt_as_done_until_retry_failed(tmp_path):
    # Controller ruling: any non-skipped status (ok/error/timeout/oom) makes a
    # job look done; --retry-failed puts the non-ok ones back in play.
    row = {
        "kind": "run", "algo": "DenStream", "fold": "static_easy", "seed": 1,
        "arm": "default", "config_id": None, "n_samples_cap": 900, "status": "error",
        "error": "boom",
    }
    (tmp_path / "runs.jsonl").write_text(json.dumps(row) + "\n")
    common = ["--algos", "DenStream", "--folds", "static_easy", "--seeds", "1",
              "--arms", "default", "--n-samples", "900", "--h", "300",
              "--out-dir", str(tmp_path), "--dry-run"]

    proc = _cli(*common)
    assert proc.returncode == 0, proc.stderr
    assert "0 to run" in proc.stdout

    proc2 = _cli(*common, "--retry-failed")
    assert proc2.returncode == 0, proc2.stderr
    assert "1 to run" in proc2.stdout
    # --dry-run: nothing is executed, the fake row is untouched
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1


def test_retry_failed_does_not_rerun_a_key_that_later_succeeded(tmp_path):
    # A key can carry more than one row across resumes: an old failure, then
    # a later "ok" once the job was retried and actually succeeded. That key
    # must stay done under --retry-failed too -- it is not still failing.
    key = dict(algo="DenStream", fold="static_easy", seed=1, arm="default", config_id=None, n_samples_cap=900)
    error_row = {"kind": "run", "status": "error", "error": "boom", **key}
    ok_row = {"kind": "run", "status": "ok", **key}
    (tmp_path / "runs.jsonl").write_text(json.dumps(error_row) + "\n" + json.dumps(ok_row) + "\n")
    proc = _cli("--algos", "DenStream", "--folds", "static_easy", "--seeds", "1",
                "--arms", "default", "--n-samples", "900", "--h", "300",
                "--out-dir", str(tmp_path), "--dry-run", "--retry-failed")
    assert proc.returncode == 0, proc.stderr
    assert "0 to run" in proc.stdout


def test_skipped_row_never_counts_as_done_once_an_oracle_entry_exists(tmp_path):
    # A stale "skipped" row (written before the sweep produced an oracle
    # entry) must not make the job look done once a real entry for it shows
    # up -- the job has to actually run, with or without --retry-failed.
    key = dict(algo="DenStream", fold="static_easy", seed=1, arm="oracle_static",
               config_id=None, n_samples_cap=900)
    stale_skip = {"kind": "run", "status": "skipped", "error": "no oracle for DenStream/static_easy", **key}
    (tmp_path / "runs.jsonl").write_text(json.dumps(stale_skip) + "\n")
    (tmp_path / "oracle_static.json").write_text(json.dumps(
        {"DenStream": {"static_easy": {"config_id": 0, "unit": None}}}   # River's default point
    ))
    common = ["--algos", "DenStream", "--folds", "static_easy", "--seeds", "1",
              "--arms", "oracle_static", "--n-samples", "900", "--h", "300",
              "--out-dir", str(tmp_path), "--oracle-file", str(tmp_path / "oracle_static.json"),
              "--dry-run"]
    proc = _cli(*common)
    assert proc.returncode == 0, proc.stderr
    assert "1 to run" in proc.stdout

    proc2 = _cli(*common, "--retry-failed")
    assert proc2.returncode == 0, proc2.stderr
    assert "1 to run" in proc2.stdout


def test_member_log_is_truncated_before_a_child_writes_so_a_retry_does_not_append(tmp_path):
    log_path = tmp_path / "members" / "fake.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("stale line from a previous failed attempt\n")
    job = {
        "algo": "DenStream", "fold": "static_easy", "seed": 1, "arm": "default",
        "n_samples": 900, "h": 300, "e_size": 2, "e_test": 1, "p_reset": 0.01,
        "half_life_frac": _common.HALF_LIFE_FRAC, "lam": None,
        "runtime_factor": _common.RUNTIME_FACTOR, "max_memory_mb": _common.MAX_MEMORY_MB,
        "max_micro_clusters": _common.MAX_MICRO_CLUSTERS,
        "freeze_after_window": _common.FREEZE_AFTER_WINDOW,
        "member_log": str(log_path), "config_id": None, "unit_override": None,
    }
    proc = subprocess.run(
        [sys.executable, "-m", "conf_pipeline.run", "--run-one"],
        input=json.dumps(job), capture_output=True, text=True,
        cwd=str(REPO), env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    lines = log_path.read_text().splitlines()
    assert "stale line from a previous failed attempt" not in lines
    assert len(lines) > 0
