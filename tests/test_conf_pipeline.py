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


def _toy_run(arm="confstream", n_samples=1500, h=300, seed=1, log=None):
    ds = make_scenario("static_easy", n_samples=n_samples, n_features=2, n_clusters=3, seed=3)
    cfg = EnsembleConfig(e_size=3, e_test=1, lam=0.1, p_reset=0.0)
    return run_ensemble(
        algo="KMeans", space=TOY, start=[{"n_clusters": 2, "halflife": 0.5}],
        dataset=ds, fold="toy", seed=seed, arm=arm, cfg=cfg, h=h, n_samples=n_samples,
        member_log=log,
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
    assert "48 jobs" in proc.stdout  # 4 algos x 2 folds x 3 seeds x 2 arms


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

    second = _cli(*common)
    assert second.returncode == 0, second.stderr
    assert "0 to run" in second.stdout
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1
