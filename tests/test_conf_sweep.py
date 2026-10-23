"""conf_pipeline.sweep: the Sobol sweep and the a-posteriori oracle."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conf_pipeline import _common as C
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


def test_select_oracle_picks_the_best_tail_not_the_best_whole_stream(tmp_path):
    # config 1 wins the whole stream on a strong start; config 2 wins the tail, the region
    # every arm is compared on
    rows = [{"algo": "A", "fold": "f", "seed": 1, "config_id": 1, "status": "ok",
             "lag_ari_win_mean": 0.6, "lag_ari_win_tail_mean": 0.3,
             "unit_override": {"x": 0.1}, "final_params": {"x": 1}},
            {"algo": "A", "fold": "f", "seed": 1, "config_id": 2, "status": "ok",
             "lag_ari_win_mean": 0.5, "lag_ari_win_tail_mean": 0.4,
             "unit_override": {"x": 0.9}, "final_params": {"x": 2}}]
    (tmp_path / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    entry = sweep.select_oracle(tmp_path)["A"]["f"]
    assert entry["config_id"] == 2 and entry["select_metric"] == "lag_ari_win_tail_mean"
    assert entry["lag_ari_win_tail_mean"] == 0.4 and entry["lag_ari_win_mean"] == 0.5


def test_select_oracle_falls_back_to_the_whole_stream_without_a_tail(tmp_path):
    # a stream shorter than tail_from has no tail reading on any point
    rows = [{"algo": "A", "fold": "f", "seed": 1, "config_id": c, "status": "ok",
             "lag_ari_win_mean": whole, "lag_ari_win_tail_mean": None,
             "unit_override": {"x": 0.5}, "final_params": {"x": c}} for c, whole in ((1, 0.2), (2, 0.3))]
    (tmp_path / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    entry = sweep.select_oracle(tmp_path)["A"]["f"]
    assert entry["config_id"] == 2 and entry["select_metric"] == "lag_ari_win_mean"


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


# --- incumbents of the online arms as ceiling candidates ---------------------------------

def _win(algo, fold, arm, seed, w, inc, params):
    return {"kind": "window", "algo": algo, "fold": fold, "arm": arm, "seed": seed, "window": w,
            "config_id": None, "incumbent_id": inc, "params": params}


P1, P2, P3 = {"x": 1, "y": 0.5}, {"x": 2, "y": 0.5}, {"x": 3, "y": 0.1}


def _write_windows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_incumbent_candidates_are_the_longest_held_and_the_last_incumbent(tmp_path):
    rows = [_win("A", "f", "confstream", 1, w, inc, p) for w, (inc, p) in
            enumerate([(7, P1), (7, P1), (9, P2), (9, P2), (9, P2), (4, P3)])]
    rows += [_win("A", "f", "frozen_prefix", 1, w, 2, P2) for w in range(4)]          # same as P2
    rows += [_win("A", "f", "default", 1, w, 0, {"x": 9, "y": 9}) for w in range(4)]   # not online
    rows += [_win("A", "f", "oracle_static", 1, w, 0, {"x": 8, "y": 8}) for w in range(4)]
    _write_windows(tmp_path / "windows.jsonl", rows)
    cands = sweep.incumbent_candidates(tmp_path)
    got = cands[("A", "f")]
    assert [c["params"] for c in got] == [P2, P3]            # longest held, then last; deduplicated
    assert {c["config_id"] for c in got} == {sweep.params_config_id(P2), sweep.params_config_id(P3)}
    assert all(c["config_id"] >= sweep.INCUMBENT_ID_BASE for c in got)
    assert got[0]["sources"] == ["confstream/seed1/longest", "frozen_prefix/seed1/longest",
                                 "frozen_prefix/seed1/last"]


def test_params_config_id_is_stable_and_ignores_key_order():
    assert sweep.params_config_id({"x": 1, "y": 0.5}) == sweep.params_config_id({"y": 0.5, "x": 1})
    assert sweep.params_config_id(P1) != sweep.params_config_id(P2)


def test_incumbent_jobs_run_their_parameters_and_can_become_the_oracle(tmp_path):
    rows = [_win("A", "f", "confstream", 1, w, 3, P1) for w in range(3)]
    _write_windows(tmp_path / "windows.jsonl", rows)
    jobs = sweep.build_incumbent_jobs(tmp_path, ["A"], ["f"], seed=1, n_samples=900, h=300)
    (job,) = jobs
    assert job["params_override"] == P1 and job["unit_override"] is None and job["arm"] == "sweep"
    assert job["config_id"] == sweep.params_config_id(P1)
    (tmp_path / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"algo": "A", "fold": "f", "seed": 1, "config_id": 0, "status": "ok", "lag_ari_win_mean": 0.2,
         "unit_override": None, "params_override": None, "final_params": {"x": 0}},
        {"algo": "A", "fold": "f", "seed": 1, "config_id": job["config_id"], "status": "ok",
         "lag_ari_win_mean": 0.6, "unit_override": None, "params_override": P1, "final_params": P1},
    ]) + "\n")
    entry = sweep.select_oracle(tmp_path)["A"]["f"]
    assert entry["source"] == "incumbent" and entry["params_override"] == P1 and entry["unit"] is None


def test_the_cli_reruns_an_incumbent_oracle_with_its_parameters(tmp_path):
    """End to end: arms, sweep with incumbents, oracle rerun on the incumbent's exact parameters."""
    repo = Path(__file__).resolve().parent.parent
    common = ["--algos", "DenStream", "--folds", "static_easy", "--n-samples", "900", "--h", "300",
              "--out-dir", str(tmp_path), "--workers", "1"]
    env = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}

    def cli(module, *args):
        # the shared policy's job seeds (--seeds, --seed), as before the seed split
        proc = subprocess.run([sys.executable, "-m", module, *common, *args, "--seed-policy", "shared"], cwd=repo,
                              env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    cli("conf_pipeline.run", "--seeds", "1", "--arms", "confstream", "--e-size", "2", "--e-test", "1",
        "--freeze-after-window", "1")
    # the sweep and the reference's rerun read the same tail as the arms (the manifest refuses another t0)
    out = cli("conf_pipeline.sweep", "--n-points", "1", "--incumbents", "--freeze-after-window", "1")
    assert "incumbent" in out
    sweep_rows = [json.loads(l) for l in (tmp_path / "sweep.jsonl").read_text().splitlines()]
    inc = [r for r in sweep_rows if r["config_id"] >= sweep.INCUMBENT_ID_BASE]
    assert inc and all(r["status"] == "ok" and r["params_override"] for r in inc)
    # Make an incumbent the oracle whatever won, so the rerun path is always exercised.
    pick = inc[0]
    path = tmp_path / "oracle_static.json"
    oracle = json.loads(path.read_text())
    oracle["DenStream"]["static_easy"].update(
        config_id=pick["config_id"], source="incumbent", unit=None,
        params_override=pick["params_override"], params=pick["final_params"],
    )
    path.write_text(json.dumps(oracle))
    cli("conf_pipeline.run", "--seeds", "1", "--arms", "oracle_static", "--freeze-after-window", "1")
    runs = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    (o,) = [r for r in runs if r["arm"] == "oracle_static"]
    assert o["status"] == "ok" and o["config_id"] == pick["config_id"]
    assert o["start_params"] == pick["params_override"]
    assert o["lag_ari_win_mean"] == pick["lag_ari_win_mean"]      # same seed, same configuration


# --- a reference picked on several seeds -----------------------------------------------------

def _point(config_id, seed, tail, status="ok"):
    return {"algo": "A", "fold": "f", "seed": seed, "config_id": config_id, "status": status,
            "lag_ari_win_mean": tail, "lag_ari_win_tail_mean": tail,
            "unit_override": {"x": config_id / 10}, "final_params": {"x": config_id}}


def test_a_config_swept_on_several_seeds_competes_with_its_mean(tmp_path):
    # config 1 has the best single run (0.9) but the worse mean; config 2 wins on the mean
    rows = [_point(1, 1, 0.9), _point(1, 2, 0.1), _point(2, 1, 0.6), _point(2, 2, 0.5)]
    (tmp_path / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    entry = sweep.select_oracle(tmp_path)["A"]["f"]
    assert entry["config_id"] == 2 and entry["select_seeds"] == [1, 2]
    assert entry["lag_ari_win_tail_mean"] == pytest.approx(0.55)
    assert entry["lag_ari_win_mean"] == pytest.approx(0.55)
    assert entry["select_sd"] == pytest.approx(np.std([0.6, 0.5], ddof=1))
    assert entry["n_ok"] == 2 and entry["n_error"] == 0


def test_a_config_missing_a_seed_is_not_judged_on_its_luckier_seeds(tmp_path):
    rows = [_point(1, 1, 0.9), _point(1, 2, None, status="error"),     # failed on seed 2
            _point(2, 1, 0.4), _point(2, 2, 0.4),
            _point(3, 1, 0.8)]                                         # never run on seed 2
    (tmp_path / "sweep.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    entry = sweep.select_oracle(tmp_path)["A"]["f"]
    assert entry["config_id"] == 2 and entry["n_ok"] == 1 and entry["n_error"] == 2


def test_one_seed_reads_exactly_as_before():
    rows = [_point(1, 1, 0.3), _point(2, 1, 0.4)]
    best, metric = C.best_static(rows)
    assert best["config_id"] == 2 and best["lag_ari_win_tail_mean"] == 0.4 and best["n_seeds"] == 1
    assert best["lag_ari_win_tail_mean_sd"] is None and metric == "lag_ari_win_tail_mean"


def test_run_seeds_expand_only_the_algorithms_that_take_a_seed():
    jobs = sweep.build_sweep_jobs(["CluStream", "DenStream"], ["static_easy"], n_points=2, seed=1,
                                  n_samples=900, h=300, run_seeds=(1, 2, 3))
    seeds = {a: sorted({j["seed"] for j in jobs if j["algo"] == a}) for a in ("CluStream", "DenStream")}
    assert seeds == {"CluStream": [1, 2, 3], "DenStream": [1]}
    clu = [j for j in jobs if j["algo"] == "CluStream"]
    by_seed = {s: [j["unit_override"] for j in clu if j["seed"] == s] for s in (1, 2, 3)}
    assert by_seed[1] == by_seed[2] == by_seed[3]                     # one design for every seed
    assert len(jobs) == 3 * 3 + 3
    assert sweep.build_sweep_jobs(["DenStream"], ["f"], 1, seed=5, n_samples=None, h=300)[0]["seed"] == 5


def test_incumbent_jobs_follow_the_run_seeds(tmp_path):
    rows = [_win("CluStream", "f", "confstream", 1, w, 3, P1) for w in range(3)]
    rows += [_win("DenStream", "f", "confstream", 1, w, 3, P2) for w in range(3)]
    _write_windows(tmp_path / "windows.jsonl", rows)
    jobs = sweep.build_incumbent_jobs(tmp_path, ["CluStream", "DenStream"], ["f"], seed=1, n_samples=900,
                                      h=300, run_seeds=(1, 2))
    assert sorted((j["algo"], j["seed"]) for j in jobs) == [("CluStream", 1), ("CluStream", 2), ("DenStream", 1)]


def test_the_cli_sweeps_on_several_seeds_and_reads_incumbents_from_another_dir(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    arms_dir, ref_dir = tmp_path / "arms", tmp_path / "ref"
    common = ["--folds", "static_easy", "--n-samples", "900", "--h", "300", "--workers", "2"]
    env = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"}

    def cli(module, *args):
        # the shared policy's job seeds (--seeds, --seed), as before the seed split
        proc = subprocess.run([sys.executable, "-m", module, *common, *args, "--seed-policy", "shared"], cwd=repo,
                              env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    cli("conf_pipeline.run", "--algos", "STREAMKMeans", "--seeds", "1", "--arms", "confstream",
        "--e-size", "2", "--e-test", "1", "--freeze-after-window", "1", "--out-dir", str(arms_dir))
    out = cli("conf_pipeline.sweep", "--algos", "STREAMKMeans,DenStream", "--n-points", "1", "--seeds", "1,2",
              "--incumbents", "--incumbents-from", str(arms_dir), "--out-dir", str(ref_dir))
    assert str(arms_dir / "windows.jsonl") in out
    rows = [json.loads(l) for l in (ref_dir / "sweep.jsonl").read_text().splitlines()]
    seeds = {a: sorted({r["seed"] for r in rows if r["algo"] == a}) for a in ("STREAMKMeans", "DenStream")}
    assert seeds == {"STREAMKMeans": [1, 2], "DenStream": [1]}
    assert any(r["config_id"] >= sweep.INCUMBENT_ID_BASE for r in rows if r["algo"] == "STREAMKMeans")
    oracle = json.loads((ref_dir / "oracle_static.json").read_text())
    assert oracle["STREAMKMeans"]["static_easy"]["select_seeds"] == [1, 2]
    assert oracle["DenStream"]["static_easy"]["select_seeds"] == [1]
