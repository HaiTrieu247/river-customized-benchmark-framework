"""Seeds of the tuner, of the models and of the calibration: separate, recorded, and the old policy intact.

Spec: docs/superpowers/specs/2026-10-24-seed-split-design.md.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.datasets import make_scenario
from benchmark.tuning.confstream import EnsembleConfig
from conf_pipeline import _common as C
from conf_pipeline import run
from conf_pipeline.spaces import SPACES, start_params

REPO = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((REPO / "tests" / "fixtures" / "seed_shared_golden.json").read_text())


def _run(algo, arm, seed=7, log=None, dataset=None, **kw):
    ds = dataset if dataset is not None else make_scenario("static_easy", n_samples=1500, n_features=2,
                                                           n_clusters=3, seed=3)
    cfg = EnsembleConfig(e_size=4, e_test=2, lam=0.1, p_reset=0.0, runtime_factor=None)
    return run.run_ensemble(algo=algo, space=SPACES[algo], start=[start_params(algo, 300)], dataset=ds,
                            fold="toy", seed=seed, arm=arm, cfg=cfg, h=300, n_samples=kw.pop("n_samples", 1500),
                            member_log=log, **kw)


# --- the shared policy is the code as it was -------------------------------------------------------

@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_one_job_seed_still_produces_exactly_what_it_produced_before_the_split(key):
    algo, arm = key.split("/")
    rows = _run(algo, arm)
    wins = [{k: w[k] for k in ("window", "incumbent_id", "params", "pred_sha", "lag_pred_sha", "score")}
            for w in rows if w["kind"] == "window"]
    (s,) = [r for r in rows if r["kind"] == "run"]
    assert json.loads(json.dumps(wins, default=str)) == GOLDEN[key]["windows"]
    assert json.loads(json.dumps({k: s[k] for k in GOLDEN[key]["summary"]}, default=str)) == GOLDEN[key]["summary"]


# --- the split policy: three independent sources ---------------------------------------------------

def _members(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def test_the_same_tuner_seed_with_another_model_seed_draws_the_same_configurations(tmp_path):
    a = _run("STREAMKMeans", "confstream", seed=101, model_seed=201, calibration_seed=0, log=tmp_path / "a.jsonl")
    b = _run("STREAMKMeans", "confstream", seed=101, model_seed=202, calibration_seed=0, log=tmp_path / "b.jsonl")
    ma, mb = _members(tmp_path / "a.jsonl"), _members(tmp_path / "b.jsonl")
    first = lambda ms: [(m["member_id"], m["params"]) for m in ms if m["window"] == 1]
    assert first(ma) == first(mb)                                   # the tuner drew the same population
    assert [m["pred_sha"] for m in ma if m["window"] == 1] != [m["pred_sha"] for m in mb if m["window"] == 1]
    assert a[-1]["s_online"] == b[-1]["s_online"]


def test_a_deterministic_algorithm_ignores_the_model_seed(tmp_path):
    _run("DenStream", "confstream", seed=101, model_seed=201, calibration_seed=0, log=tmp_path / "a.jsonl")
    _run("DenStream", "confstream", seed=101, model_seed=202, calibration_seed=0, log=tmp_path / "b.jsonl")
    strip = lambda ms: [{k: v for k, v in m.items() if k not in ("runtime_s", "runtime_ratio", "scoring_runtime_s",
                                                                  "memory_mb")} for m in ms]
    assert strip(_members(tmp_path / "a.jsonl")) == strip(_members(tmp_path / "b.jsonl"))


def test_another_tuner_seed_with_the_same_model_seed_draws_other_configurations(tmp_path):
    _run("STREAMKMeans", "confstream", seed=101, model_seed=201, calibration_seed=0, log=tmp_path / "a.jsonl")
    _run("STREAMKMeans", "confstream", seed=102, model_seed=201, calibration_seed=0, log=tmp_path / "b.jsonl")
    first = lambda ms: [m["params"] for m in ms if m["window"] == 1 and m["member_id"] > 0]
    assert first(_members(tmp_path / "a.jsonl")) != first(_members(tmp_path / "b.jsonl"))


def _flat_then_spread(n_flat=1800, n_spread=2400, seed=0):
    """Identical rows first, so the distance unit needs a subsample over several calibration windows."""
    rng = random.Random(seed)
    rows = [({"a": 0.0, "b": 0.0}, 0)] * n_flat
    rows += [({"a": rng.gauss(0, 1), "b": rng.gauss(0, 1)}, i % 2) for i in range(n_spread)]
    return rows


def test_the_distance_unit_does_not_depend_on_the_tuner_seed_once_calibration_has_its_own_seed():
    s = {t: _run("DenStream", "default", seed=t, model_seed=201, calibration_seed=0,
                 dataset=_flat_then_spread(), n_samples=4200)[-1]["s_online"] for t in (101, 102)}
    assert s[101] == s[102] > 0
    shared = {t: _run("DenStream", "default", seed=t, dataset=_flat_then_spread(), n_samples=4200)[-1]["s_online"]
              for t in (101, 102)}
    assert shared[101] != shared[102]                               # the old coupling, which the split removes


def test_every_row_records_the_seeds_it_ran_with(tmp_path):
    fields = {"seed": None, "tuner_seed": None, "model_seed": None, "calibration_seed": 0, "seed_policy": "split"}
    rows = _run("DenStream", "default", seed=0, model_seed=0, calibration_seed=0, seed_fields=fields,
                log=tmp_path / "m.jsonl")
    for r in [*rows, *_members(tmp_path / "m.jsonl")]:
        assert {k: r[k] for k in fields} == fields


# --- which jobs run, and what they record ----------------------------------------------------------

def test_job_seeds_follow_the_table_of_the_spec():
    split = dict(policy="split", tuner_seeds=(101, 102), model_seed=201, fixed_model_seeds=(201, 202))
    assert run.job_seeds("DenStream", "default", **split) == [{"seed": None, "tuner_seed": None, "model_seed": None}]
    assert run.job_seeds("STREAMKMeans", "default", **split) == [
        {"seed": 201, "tuner_seed": None, "model_seed": 201}, {"seed": 202, "tuner_seed": None, "model_seed": 202}]
    assert run.job_seeds("DenStream", "confstream", **split) == [
        {"seed": 101, "tuner_seed": 101, "model_seed": None}, {"seed": 102, "tuner_seed": 102, "model_seed": None}]
    assert run.job_seeds("CluStream", "frozen_prefix", **split) == [
        {"seed": 101, "tuner_seed": 101, "model_seed": 201}, {"seed": 102, "tuner_seed": 102, "model_seed": 201}]
    shared = dict(policy="shared", seeds=(1, 2))
    assert run.job_seeds("DenStream", "default", **shared) == [
        {"seed": 1, "tuner_seed": 1, "model_seed": None}, {"seed": 2, "tuner_seed": 2, "model_seed": None}]
    assert run.job_seeds("HPStream", "confstream", **shared)[1] == {"seed": 2, "tuner_seed": 2, "model_seed": 2}


def _cli(*args, module="conf_pipeline.run"):
    return subprocess.run([sys.executable, "-m", module, *args], capture_output=True, text=True, cwd=str(REPO),
                          env={**os.environ, "PYTHONHASHSEED": "0"})


def test_the_full_protocol_under_the_split_policy_runs_each_deterministic_fixed_configuration_once(tmp_path):
    proc = _cli("--dry-run", "--out-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    # default: 5 deterministic x 18 once + 3 seeded x 18 on one model seed; two online arms: 8 x 18 x 3 tuner seeds
    assert f"{5 * 18 + 3 * 18 + 8 * 18 * 3 * 2} jobs" in proc.stdout
    shared = _cli("--dry-run", "--out-dir", str(tmp_path), "--seed-policy", "shared")
    assert shared.returncode == 0 and "1296 jobs" in shared.stdout


def test_the_job_seeds_flag_belongs_to_the_shared_policy_only(tmp_path):
    proc = _cli("--dry-run", "--out-dir", str(tmp_path), "--seeds", "1")
    assert proc.returncode != 0 and "--tuner-seeds" in proc.stderr
    proc = _cli("--dry-run", "--out-dir", str(tmp_path), "--seed-policy", "shared", "--tuner-seeds", "101")
    assert proc.returncode != 0 and "split" in proc.stderr


TINY = ["--algos", "DenStream,STREAMKMeans", "--folds", "static_easy", "--arms", "default,confstream",
        "--n-samples", "900", "--h", "300", "--e-size", "2", "--e-test", "1", "--workers", "2", "--tuner-seeds", "101,102"]


def test_a_split_run_records_every_seed_and_its_manifest_the_policy(tmp_path):
    proc = _cli(*TINY, "--out-dir", str(tmp_path), "--log-members")
    assert proc.returncode == 0, proc.stderr
    rows = {(r["algo"], r["arm"], r["seed"]): r for r in map(json.loads, (tmp_path / "runs.jsonl").read_text().splitlines())}
    assert set(rows) == {("DenStream", "default", None), ("STREAMKMeans", "default", 201),
                         ("DenStream", "confstream", 101), ("DenStream", "confstream", 102),
                         ("STREAMKMeans", "confstream", 101), ("STREAMKMeans", "confstream", 102)}
    r = rows[("STREAMKMeans", "confstream", 102)]
    assert (r["seed_policy"], r["tuner_seed"], r["model_seed"], r["calibration_seed"]) == ("split", 102, 201, 0)
    d = rows[("DenStream", "default", None)]
    assert (d["tuner_seed"], d["model_seed"]) == (None, None)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema"] == 2
    assert {k: m["protocol"][k] for k in ("seed_policy", "online_model_seed", "calibration_seed")} == {
        "seed_policy": "split", "online_model_seed": 201, "calibration_seed": C.CALIBRATION_SEED}
    assert m["seeds"]["tuner"] == [101, 102] and m["seeds"]["model"] == [201]
    assert (tmp_path / "members" / "DenStream__static_easy__seedx__default.jsonl").exists()
    again = _cli(*TINY, "--out-dir", str(tmp_path), "--log-members", "--allow-code-change")
    assert again.returncode == 0 and "0 to run" in again.stdout


def test_another_model_seed_in_the_same_dir_is_another_protocol(tmp_path):
    assert _cli(*TINY, "--out-dir", str(tmp_path)).returncode == 0
    proc = _cli(*TINY, "--out-dir", str(tmp_path), "--model-seed", "202", "--allow-code-change", "--dry-run")
    assert proc.returncode != 0 and "protocol.online_model_seed" in proc.stderr


def test_fixed_model_seeds_repeat_only_the_seeded_algorithms(tmp_path):
    proc = _cli(*TINY, "--out-dir", str(tmp_path), "--arms", "default", "--fixed-model-seeds", "201,202", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "3 jobs" in proc.stdout                                    # DenStream once, STREAMKMeans twice


def test_the_reference_is_never_rerun_on_a_model_seed_that_picked_it(tmp_path):
    oracle = {"STREAMKMeans": {"static_easy": {"config_id": 0, "unit": None, "select_seeds": [201]}},
              "DenStream": {"static_easy": {"config_id": 0, "unit": None, "select_seeds": [None]}}}
    (tmp_path / "oracle_static.json").write_text(json.dumps(oracle))
    base = [*TINY[:4], "--arms", "oracle_static", *TINY[6:], "--out-dir", str(tmp_path), "--dry-run"]
    proc = _cli(*base)
    assert proc.returncode != 0 and "picked" in proc.stderr and "STREAMKMeans" in proc.stderr
    ok = _cli(*base, "--fixed-model-seeds", "202,203")
    assert ok.returncode == 0, ok.stderr
    assert "3 jobs" in ok.stdout


def test_old_rows_keep_their_keys_under_the_shared_policy():
    old = {"algo": "STREAMKMeans", "fold": "f", "seed": 2, "arm": "default", "n_samples_cap": 900, "config_id": None}
    job = {"algo": "STREAMKMeans", "fold": "f", "seed": 2, "arm": "default", "n_samples": 900, "config_id": None,
           **run.job_seeds("STREAMKMeans", "default", policy="shared", seeds=(2,))[0]}
    assert run._row_key(old) == run._job_key(job)


# --- the sweep: a Sobol seed, and model seeds only where a model reads one -------------------------

def test_a_split_sweep_runs_a_deterministic_point_once_and_a_seeded_one_per_model_seed():
    from conf_pipeline import sweep

    jobs = sweep.build_sweep_jobs(["DenStream", "STREAMKMeans"], ["static_easy"], n_points=2, seed=5, n_samples=900,
                                  h=300, seed_policy="split", model_seed=201, model_seeds=(201, 202))
    den = [j for j in jobs if j["algo"] == "DenStream"]
    skm = [j for j in jobs if j["algo"] == "STREAMKMeans"]
    assert len(den) == 3 and {j["seed"] for j in den} == {None} and {j["model_seed"] for j in den} == {None}
    assert len(skm) == 6 and sorted({j["seed"] for j in skm}) == [201, 202]
    assert all(j["seed_policy"] == "split" and j["online_model_seed"] == 201 and j["tuner_seed"] is None for j in jobs)
    values = lambda js: [None if j["unit_override"] is None else list(j["unit_override"].values()) for j in js]
    assert values(den) == values(skm[:3])                          # one design (seed 5), both 4-dimensional


def test_the_sweep_seed_flags_say_what_they_seed(tmp_path):
    base = ["--algos", "DenStream", "--folds", "static_easy", "--n-points", "1", "--out-dir", str(tmp_path), "--dry-run"]
    old = _cli(*base, "--seed", "3", module="conf_pipeline.sweep")
    assert old.returncode != 0 and "--sobol-seed" in old.stderr
    ok = _cli(*base, "--sobol-seed", "3", module="conf_pipeline.sweep")
    assert ok.returncode == 0, ok.stderr
    shared = _cli(*base, "--seed-policy", "shared", "--seed", "3", module="conf_pipeline.sweep")
    assert shared.returncode == 0, shared.stderr
    mixed = _cli(*base, "--seed-policy", "shared", "--model-seeds", "201", module="conf_pipeline.sweep")
    assert mixed.returncode != 0 and "split" in mixed.stderr


def test_a_split_sweep_joins_a_split_run_in_one_dir(tmp_path):
    assert _cli(*TINY, "--out-dir", str(tmp_path), "--freeze-after-window", "1").returncode == 0
    proc = _cli("--algos", "DenStream,STREAMKMeans", "--folds", "static_easy", "--n-samples", "900", "--h", "300",
                "--n-points", "1", "--freeze-after-window", "1", "--out-dir", str(tmp_path), "--allow-code-change",
                module="conf_pipeline.sweep")
    assert proc.returncode == 0, proc.stderr
    oracle = json.loads((tmp_path / "oracle_static.json").read_text())
    assert oracle["STREAMKMeans"]["static_easy"]["select_seeds"] == [201]
    assert oracle["DenStream"]["static_easy"]["select_seeds"] == [None]


# --- reports: a run made once is read once --------------------------------------------------------

def _split_dir(root):
    import pandas as pd  # noqa: F401

    rows = []
    def row(algo, arm, seed, tail, tuner=None, model=None):
        rows.append({"kind": "run", "algo": algo, "fold": "covtype", "arm": arm, "seed": seed, "status": "ok",
                     "config_id": None, "n_samples_cap": 1000, "seed_policy": "split", "tuner_seed": tuner,
                     "model_seed": model, "calibration_seed": 0, "lag_ari_win_tail_mean": tail,
                     "lag_ari_win_mean": tail, "tail_from": 20})
    row("DenStream", "default", None, 0.1)
    row("DenStream", "confstream", 101, 0.3, tuner=101)
    row("DenStream", "confstream", 102, 0.4, tuner=102)
    row("STREAMKMeans", "default", 201, 0.2, model=201)
    row("STREAMKMeans", "default", 202, 0.4, model=202)
    row("STREAMKMeans", "confstream", 101, 0.5, tuner=101, model=201)
    row("STREAMKMeans", "confstream", 102, 0.6, tuner=102, model=201)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return root


def test_a_fixed_arm_is_read_beside_every_tuner_seed_at_the_mean_of_its_own_runs(tmp_path):
    import pandas as pd

    from conf_pipeline import report

    runs = report.current_runs(_split_dir(tmp_path))
    values = runs[["algo", "fold", "arm", "seed"]].assign(value=runs["lag_ari_win_tail_mean"])
    view = report.replicate_view(values, split=report.is_split(runs))
    den = view[(view.algo == "DenStream") & (view.arm == "default")].sort_values("seed")
    assert list(den.seed) == [101, 102] and list(den.value) == [0.1, 0.1]
    assert den.broadcast.all() and list(den.n_runs) == [1, 1]
    skm = view[(view.algo == "STREAMKMeans") & (view.arm == "default")]
    assert list(skm.value) == pytest.approx([0.3, 0.3]) and set(skm.n_runs) == {2}
    online = view[view.arm == "confstream"]
    assert not online.broadcast.any() and set(online.n_runs) == {1} and len(online) == 4
    shared = report.replicate_view(values, split=False)
    assert len(shared) == len(values) and not shared.broadcast.any()


def test_the_summary_says_what_varies_and_how_many_runs_there_were(tmp_path):
    from conf_pipeline import report

    s = report.summary_table(_split_dir(tmp_path)).set_index(["algo", "arm"])
    assert s.loc[("DenStream", "default"), "replicates"] == "none: deterministic, run once"
    assert s.loc[("STREAMKMeans", "default"), "replicates"] == "model seeds"
    assert s.loc[("STREAMKMeans", "confstream"), "replicates"] == "tuner seeds"
    assert s.loc[("DenStream", "default"), "n_runs"] == 1 and s.loc[("STREAMKMeans", "default"), "n_runs"] == 2
    assert s["all_seeds_ok"].all()


def test_robustness_selection_and_variants_read_a_split_dir_without_pairing_by_seed_number(tmp_path):
    from conf_pipeline import report, robustness, selection
    from conf_pipeline import variants as V

    d = _split_dir(tmp_path)
    seeds = robustness.load_inputs(d).seeds
    assert set(seeds[seeds.arm == "default"].seed) == {101, 102}
    cells = robustness.cells_table(robustness.load_inputs(d)).set_index("algo")
    assert cells.loc["STREAMKMeans", "default"] == pytest.approx(0.3)
    sv = selection.seed_values(report.current_runs(d), arms=("default", "confstream"))
    assert set(sv[sv.arm == "default"].seed) == {101, 102}
    means = V.arm_means(d).set_index(["algo", "arm"])
    assert means.loc[("DenStream", "default"), "all_seeds_ok"] and means.loc[("DenStream", "default"), "n_runs"] == 1


def test_window_readers_keep_a_run_made_once_and_set_it_beside_every_tuner_seed():
    import pandas as pd

    from conf_pipeline import robustness

    rows = []
    for arm, seed, base in (("default", None, 0.1), ("confstream", 101, 0.3), ("confstream", 102, 0.5)):
        for w in range(20, 24):
            rows.append({"algo": "DenStream", "fold": "covtype", "arm": arm, "seed": seed, "window": w,
                         "lag_ari_win": base + 0.01 * (w % 2), "lag_nmi_win": base, "seed_policy": "split"})
    win = pd.DataFrame(rows)
    even = robustness.parity_values(win, {}, 0)
    assert set(even[even.arm == "default"].seed) == {101, 102}
    assert even[even.arm == "default"].value.tolist() == pytest.approx([0.1, 0.1])
    tail = robustness.tail_values(win, {}, "lag_nmi_win")
    assert set(tail[tail.arm == "default"].seed) == {101, 102}
    measures = robustness.measure_values(win, {}).set_index("arm")
    assert measures.loc["default", "ari"] == pytest.approx(0.105)


def test_arms_on_one_tuner_seed_start_alike_and_frozen_prefix_follows_the_tuner_up_to_its_freeze():
    kw = dict(seed=101, model_seed=201, calibration_seed=0, freeze_after_window=3,
              dataset=make_scenario("static_easy", n_samples=2400, n_features=2, n_clusters=3, seed=3), n_samples=2400)
    online = [r for r in _run("STREAMKMeans", "confstream", **kw) if r["kind"] == "window"]
    kw["dataset"] = make_scenario("static_easy", n_samples=2400, n_features=2, n_clusters=3, seed=3)
    frozen = [r for r in _run("STREAMKMeans", "frozen_prefix", **kw) if r["kind"] == "window"]
    trail = lambda rows: [(r["window"], r["incumbent_id"], r["pred_sha"]) for r in rows if r["window"] <= 3]
    assert trail(online) == trail(frozen) and len(trail(online)) == 3
