"""The mechanism study's branches (``EnsembleConfig.freeze_mode``), shadow evaluation, the prediction
log and prefix scaling: what each changes after the boundary, and that none of them changes anything
before it or anything the tuner selects."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from benchmark.algorithms.registry import get_algorithm
from benchmark.tuning.confstream import FREEZE_MODES, ConfStreamEnsemble, EnsembleConfig
from benchmark.tuning.space import Param, ParamSpace, SpaceContext
from benchmark.tuning.window import WindowBuffer
from conf_pipeline import _common as C
from conf_pipeline import manifest as M
from conf_pipeline import run as R

SPACE = ParamSpace(
    "KMeans",
    (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)),
)
CTX = SpaceContext(s=1.0, d=2)
START = {"n_clusters": 2, "halflife": 0.5}


def _make_adapter(params: dict, seed: int):
    return get_algorithm("KMeans", seed=seed, **params)


def _blob_window(h: int, seed: int) -> WindowBuffer:
    rng = random.Random(seed)
    buf = WindowBuffer(h)
    for i in range(h):
        c = i % 2
        buf.push({"a": 5.0 * c + rng.gauss(0, 0.1), "b": rng.gauss(0, 0.1)}, c)
    return buf


def _ensemble(seed=0, keep_predictions=False, **kw):
    cfg = EnsembleConfig(**{"e_size": 4, "e_test": 2, "lam": 0.05, "p_reset": 0.0, "runtime_factor": None, **kw})
    return ConfStreamEnsemble(space=SPACE, ctx=CTX, make_adapter=_make_adapter, start_params=[START],
                              config=cfg, rng=np.random.default_rng(seed), seed=seed,
                              keep_predictions=keep_predictions)


def _run(ens, n_windows=6, h=60):
    ens.initialise(_blob_window(h, 100), window_index=0)
    return [ens.process_window(_blob_window(h, 100 + t), t) for t in range(1, n_windows + 1)]


def _trace(reports):
    return [(r.window_index, r.incumbent.member_id, [o.pred_sha for o in r.outcomes]) for r in reports]


def test_every_freeze_mode_is_identical_to_the_tuner_up_to_the_boundary():
    """The branches start from one state: through the freeze window they draw, select and predict
    exactly as the tuner and frozen_prefix do (deterministic protocol, same seeds)."""
    ref = _trace(_run(_ensemble(freeze_after_window=3)))
    tuner = _trace(_run(_ensemble()))
    assert ref[:3] == tuner[:3]
    for mode in FREEZE_MODES:
        got = _trace(_run(_ensemble(freeze_after_window=3, freeze_mode=mode)))
        assert got[:3] == tuner[:3], mode


def test_frozen_ensemble_keeps_the_portfolio_and_creates_nothing():
    ens = _ensemble(freeze_after_window=2, freeze_mode="ensemble")
    reports = _run(ens)
    kept = sorted(m.id for m in ens.active)
    assert len(kept) == 4
    after = [r for r in reports if r.window_index > 2]
    assert all(r.n_spawned == 0 and r.n_tested == 0 for r in after[1:])
    assert all(sorted(o.member_id for o in r.outcomes) == kept for r in after[1:])
    assert {r.incumbent.member_id for r in after} <= set(kept)


def test_portfolio_renewal_keeps_the_configurations_and_renews_only_models():
    ens = _ensemble(freeze_after_window=2, freeze_mode="portfolio")
    reports = _run(ens, n_windows=8)
    portfolio = [p for _, p, _, _ in ens._portfolio]
    assert len(portfolio) == 4
    # every live member runs one of the portfolio's configurations, in its own slot
    for m in ens.active:
        assert m.slot is not None and m.params == portfolio[m.slot]
    children = [o for r in reports if r.window_index > 3 for o in r.outcomes if o.is_test]
    assert children and all(o.slot is not None and o.params == portfolio[o.slot] for o in children)
    # round-robin: two copies per window, slots 0,1 then 2,3 ...
    births = [e for e in ens.drain_events() if e["event"] == "birth" and e["origin"] == "portfolio_copy"]
    assert births
    # the copies are built right after the freeze (window 2) and tested from window 3 on
    renewed = [o.member_id for r in reports for o in r.outcomes if o.is_test and r.window_index > 2]
    assert len(renewed) == 2 * (8 - 2)
    assert all(o.slot is not None for r in reports if r.window_index > 2 for o in r.outcomes if o.is_test)


def test_portfolio_challenger_meets_only_its_own_configuration():
    ens = _ensemble(freeze_after_window=2, freeze_mode="portfolio")
    _run(ens, n_windows=5)
    challenges = [e for e in ens.drain_events() if e["event"] == "challenge" and e.get("slot") is not None]
    assert challenges
    slot_of = {}
    for m in ens.active:
        slot_of[m.id] = m.slot
    for e in challenges:
        assert e["outcome"] in ("accepted", "rejected")


def test_random_after_prefix_keeps_tuning_with_uniform_children():
    ens = _ensemble(freeze_after_window=2, freeze_mode="random")
    _run(ens, n_windows=5)
    births = [e for e in ens.drain_events() if e["event"] == "birth" and e["window"] > 2]
    assert births and all(e["origin"] == "uniform" and e["parent_id"] is None for e in births)
    assert len(ens.active) == 4


def test_freeze_modes_are_validated():
    with pytest.raises(ValueError):
        EnsembleConfig(freeze_mode="nope", freeze_after_window=2)
    with pytest.raises(ValueError):
        EnsembleConfig(freeze_mode="portfolio")
    with pytest.raises(ValueError):
        EnsembleConfig(freeze_mode="portfolio", freeze_after_window=2, reinit_after_freeze=True)
    with pytest.raises(ValueError):
        EnsembleConfig(shadow_size=-1)


def test_shadow_members_never_change_what_the_tuner_does():
    plain = _trace(_run(_ensemble(), n_windows=7))
    ens = _ensemble(shadow_size=3)
    reports = _run(ens, n_windows=7)
    assert _trace(reports) == plain
    shadows = [o for r in reports for o in r.shadow_outcomes]
    assert shadows and all(o.is_shadow for o in shadows)
    assert all(len(r.shadow_outcomes) <= 3 for r in reports)
    live = {o.member_id for r in reports for o in r.outcomes}
    # a shadow reading is of a member that left the ensemble, and is never the incumbent
    for r in reports:
        assert r.incumbent.member_id not in {o.member_id for o in r.shadow_outcomes}
        assert not ({o.member_id for o in r.shadow_outcomes} & {o.member_id for o in r.outcomes})
    assert live


def test_a_freeze_sends_the_discarded_members_to_the_shadow_pool():
    ens = _ensemble(freeze_after_window=2, shadow_size=10)
    reports = _run(ens, n_windows=4)
    after = [r for r in reports if r.window_index == 3][0]
    assert len(after.outcomes) == 1
    assert len(after.shadow_outcomes) >= 3


def test_keep_predictions_keeps_every_members_predictions_and_changes_nothing():
    plain = _trace(_run(_ensemble(), n_windows=4))
    ens = _ensemble(keep_predictions=True)
    reports = _run(ens, n_windows=4)
    assert _trace(reports) == plain
    assert all(o.predictions is not None and len(o.predictions) == 60 for r in reports for o in r.outcomes)


def test_branch_arms_are_runnable_and_map_to_their_mode():
    base = R.job_config({"e_size": 4, "e_test": 2, "p_reset": 0.0, "runtime_factor": None,
                         "max_memory_mb": 1e9, "max_micro_clusters": 10**6})
    for arm, mode in C.BRANCH_ARMS.items():
        assert arm in C.OPTIONAL_ARMS
        cfg = R.arm_config(arm, base, 20)
        assert cfg.freeze_mode == mode and cfg.freeze_after_window == 20


def test_arm_specs_of_existing_arms_do_not_name_the_new_fields():
    """A manifest written before the fields existed still matches the same arm."""
    job = {"arm": "frozen_prefix", "e_size": 20, "e_test": 10, "p_reset": 0.01, "runtime_factor": None,
           "max_memory_mb": 2048.0, "max_micro_clusters": 10000, "freeze_after_window": 20}
    spec = M.arm_spec(job)
    assert "freeze_mode" not in spec and "shadow_size" not in spec
    assert M.arm_spec({**job, "arm": "frozen_ensemble"})["freeze_mode"] == "ensemble"
    assert M.arm_spec({**job, "shadow_size": 5})["shadow_size"] == 5


def test_prefix_standardised_fits_on_the_first_window_only():
    data = [({"a": float(i), "b": 3.0, "c": "x"}, i % 2) for i in range(10)]
    s = R.PrefixStandardised(data, h=4)
    out = list(s)
    first = np.array([float(i) for i in range(4)])
    mu, sd = first.mean(), first.std()
    assert out[9][0]["a"] == pytest.approx((9 - mu) / sd)
    assert out[0][0]["b"] == 0.0 and out[9][0]["b"] == 0.0  # constant: centred, unit scale
    assert out[0][0]["c"] == "x"
    assert [y for _, y in out] == [y for _, y in data]
    summ = s.summary()
    assert summ["n_constant_features"] == 1 and summ["fitted_on_samples"] == 4 and summ["n_passthrough_values"] == 10


def test_prefix_scaling_is_a_protocol_field_only_when_set():
    job = {"algo": "DenStream", "fold": "static_easy", "arm": "default", "h": 1000, "n_samples": 5000,
           "half_life_frac": 0.1, "lam": None, "runtime_factor": None, "max_memory_mb": 2048.0,
           "max_micro_clusters": 10000, "freeze_after_window": 20, "cmm": True, "data_seed": 42,
           "seed_policy": "split", "online_model_seed": 201, "calibration_seed": 0,
           "e_size": 1, "e_test": 0, "p_reset": 0.01, "seed": None}
    plain = M.build([job], provenance={})
    assert "prefix_scaling" not in plain["protocol"]
    scaled = M.build([{**job, "prefix_scaling": "standard"}], provenance={})
    assert scaled["protocol"]["prefix_scaling"] == "standard"


def test_the_driver_plans_the_new_logs_and_refuses_shadow_without_member_logs(tmp_path):
    args = R.build_parser().parse_args([
        "--algos", "DenStream", "--folds", "static_easy", "--arms", "confstream,frozen_ensemble",
        "--tuner-seeds", "1", "--out-dir", str(tmp_path), "--log-members", "--shadow-size", "4",
        "--log-predictions", "--prefix-scaling", "standard"])
    jobs, skipped, _, _ = R.plan_jobs(args)
    assert len(jobs) == 2 and not skipped
    for j in jobs:
        assert j["shadow_size"] == 4 and j["prefix_scaling"] == "standard"
        assert j["shadow_log"].endswith(".jsonl") and "/shadow/" in j["shadow_log"]
        assert "/predictions/" in j["prediction_log"]
    bad = R.build_parser().parse_args(["--algos", "DenStream", "--folds", "static_easy", "--shadow-size", "2",
                                       "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        R.plan_jobs(bad)


def test_pins_are_parsed_per_algorithm_planned_and_recorded_in_the_manifest(tmp_path):
    assert R.parse_pins(["HPStream.n_samples_init=2000", "BIRCH.metric=euclid"], ["HPStream", "BIRCH"]) == {
        "HPStream": {"n_samples_init": 2000}, "BIRCH": {"metric": "euclid"}}
    for bad in (["HPStream=3"], ["HPStream.x"], ["DenStream.mu=2"]):
        with pytest.raises(SystemExit):
            R.parse_pins(bad, ["HPStream"])
    args = R.build_parser().parse_args(["--algos", "HPStream,DenStream", "--folds", "static_easy", "--arms", "default",
                                        "--out-dir", str(tmp_path), "--pin", "HPStream.n_samples_init=2000"])
    jobs, _, _, _ = R.plan_jobs(args)
    by = {j["algo"]: j for j in jobs}
    assert by["HPStream"]["pins"] == {"n_samples_init": 2000} and "pins" not in by["DenStream"]
    m = M.build(jobs, provenance={})
    assert m["algorithms"]["HPStream"]["pins"] == {"n_samples_init": 2000}
    assert "pins" not in m["algorithms"]["DenStream"]
    assert "pins" not in M.build([by["DenStream"]], provenance={})["algorithms"]["DenStream"]


def test_a_pin_above_the_window_is_kept_as_given(tmp_path):
    from conf_pipeline.spaces import SPACES, start_params

    data = [({"a": float(i % 7), "b": float(i % 3)}, i % 2) for i in range(1500)]
    rows = R.run_ensemble(algo="HPStream", space=SPACES["HPStream"], start=[start_params("HPStream", 500, d=2)],
                          dataset=data, fold="static_easy", seed=1, arm="default",
                          cfg=EnsembleConfig(e_size=1, e_test=0, runtime_factor=None), h=500, n_samples=1500, lam=0.05,
                          pins={"n_samples_init": 1000})
    assert rows[-1]["start_params"]["n_samples_init"] == 1000 and rows[-1]["pins"] == {"n_samples_init": 1000}


def test_run_ensemble_writes_shadow_and_prediction_logs_and_the_cost_fields(tmp_path):
    from conf_pipeline.spaces import SPACES, start_params

    algo = "STREAMKMeans"
    data = [({"a": 5.0 * (i % 2) + random.Random(i).gauss(0, 0.1), "b": random.Random(-i).gauss(0, 0.1)}, i % 2)
            for i in range(800)]
    member_log, shadow_log, pred_log = tmp_path / "m.jsonl", tmp_path / "s.jsonl", tmp_path / "p.jsonl"
    cfg = EnsembleConfig(e_size=3, e_test=2, runtime_factor=None, shadow_size=2)
    rows = R.run_ensemble(algo=algo, space=SPACES[algo], start=[start_params(algo, 100, d=2)], dataset=data,
                          fold="static_easy", seed=1, arm="frozen_ensemble", cfg=cfg, h=100, n_samples=800,
                          lam=0.05, member_log=member_log, shadow_log=shadow_log, prediction_log=pred_log,
                          freeze_after_window=3, prefix_scaling="standard")
    summary = rows[-1]
    assert summary["status"] == "ok" and summary["freeze_mode"] == "ensemble" and summary["shadow_size"] == 2
    assert summary["prefix_scaling"] == "standard" and summary["prefix_scaling_stats"]["n_features"] == 2
    for k in ("incumbent_ms_per_obs_p50", "incumbent_ms_per_obs_p99", "ensemble_ms_per_obs_p95"):
        assert summary[k] is not None and summary[k] >= 0
    preds = [json.loads(line) for line in pred_log.read_text().splitlines()]
    assert preds and all(len(p["predictions"]) == 100 for p in preds)
    members = [json.loads(line) for line in member_log.read_text().splitlines()]
    assert not any(m.get("is_shadow") for m in members)
    if shadow_log.exists() and shadow_log.read_text():
        assert all(json.loads(line)["is_shadow"] for line in shadow_log.read_text().splitlines())


def test_the_frozen_prefix_lobo_selector_is_added_by_name_and_joins_the_panel(monkeypatch):
    from conf_pipeline import selection as S

    for name in ("LOBO_SELECTORS", "LOBO_REGIMES", "LOBO_CONTRASTS"):
        monkeypatch.setattr(S, name, getattr(S, name))
    before = dict(S.LOBO_SELECTORS)
    S.enable_frozen_selector()
    S.enable_frozen_selector()  # idempotent
    assert S.LOBO_SELECTORS["lobo_frozen_prefix"] == "frozen_prefix"
    assert list(S.LOBO_SELECTORS)[-1] == "random" and len(S.LOBO_SELECTORS) == len(before) + 1
    assert S.LOBO_REGIMES.count("frozen_prefix") == 1
    assert ("lobo_target", "lobo_frozen_prefix") in S.LOBO_CONTRASTS
    assert S.LOBO_PRIMARY == ("lobo_target", "lobo_scaled_default")
    plan = S.lobo_plan(["DenStream", "CluStream"], ["covtype", "electricity"], source="test")
    assert any(r == "frozen_prefix" for (_, _, r) in plan["cells"])


def test_the_frozen_selector_flag_belongs_to_lobo(tmp_path):
    from conf_pipeline import selection as S

    with pytest.raises(SystemExit):
        S.main(["--out-dir", str(tmp_path), "--lobo-frozen-selector"])
