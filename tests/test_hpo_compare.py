"""Tests for the HPO pilot's grids and selection logic. No datasets, no runs."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from hpo_pipeline import hpo_compare
from hpo_pipeline.space import GRAN


# --- helpers -----------------------------------------------------------------
def _tune(rows):
    """rows: (algo, dataset, config_id, level, theta, ari, status[, criterion])."""
    out = []
    for r in rows:
        algo, d, cid, lv, th, s, st = r[:7]
        row = {"algo": algo, "dataset": d, "config_id": cid, "granularity_level": lv,
               "theta": th, "ari": s, "runtime_s": 1.0, "status": st, "grid": "ladder"}
        if len(r) > 7:
            row["matr_trace_win"] = r[7]
        out.append(row)
    return pd.DataFrame(out)


# --- grids -------------------------------------------------------------------
def test_defaults_are_river_constructor_not_registry_preset():
    default = hpo_compare.default_candidate("DenStream")
    # River's constructor says beta=0.75, mu=2, epsilon=0.02; the registry preset differs.
    assert default["shape"]["epsilon"] == 0.02
    assert default["shape"]["beta"] == 0.75
    assert "seed" not in default["shape"]


def test_levels_run_coarse_to_fine_in_both_grids():
    for algo, spec in hpo_compare.GRIDS.items():
        kind = GRAN[algo][1]
        for grid in ("raw", "ladder"):
            levels = spec["levels"][grid]
            expected = sorted(levels, reverse=(kind != "count"))
            assert levels == expected, (algo, grid)


def test_grid_is_the_full_factorial_minus_rejections():
    for algo, spec in hpo_compare.GRIDS.items():
        for kind in ("raw", "ladder"):
            accepted, rejected = hpo_compare.build_grid(algo, kind, spec)
            n_shape = len(list(itertools.product(*spec["shape"][kind].values())))
            assert len(accepted) + len(rejected) == len(spec["levels"][kind]) * n_shape
            assert len({c["config_id"] for c in accepted}) == len(accepted)
            assert all(c["grid"] == kind for c in accepted)


def test_denstream_rejects_beta_mu_at_or_below_one():
    accepted, rejected = hpo_compare.build_grid("DenStream", "raw", hpo_compare.GRIDS["DenStream"])
    for c in accepted:
        assert c["shape"]["beta"] * c["shape"]["mu"] > 1
    assert rejected and all(r["violates"] == ["beta * mu > 1"] for r in rejected)
    # (0.2, 2), (0.2, 5), (0.5, 2) at every one of the 7 levels
    assert len(rejected) == 3 * 7


def test_clustream_ladder_never_needs_rejection_but_raw_does():
    spec = hpo_compare.GRIDS["CluStream"]
    _, rejected_ladder = hpo_compare.build_grid("CluStream", "ladder", spec)
    assert rejected_ladder == []
    # q = 50 cannot host k = 64 macro-clusters: one rejection per r-factor.
    _, rejected_raw = hpo_compare.build_grid("CluStream", "raw", spec)
    assert [(r["value"], r["point"]["max_micro_clusters"]) for r in rejected_raw] == [(64, 50)] * 3
    assert all(r["violates"] == ["max_micro_clusters >= n_macro_clusters"] for r in rejected_raw)


def test_micro_ratio_resolves_to_a_multiple_of_k():
    ladder, _ = hpo_compare.build_grid("CluStream", "ladder", hpo_compare.GRIDS["CluStream"])
    cand = next(c for c in ladder if c["shape"]["micro_ratio"] == 10 and c["g"] == 8)
    params = hpo_compare.resolve_candidate(cand, {"s": 123.0})
    assert params["n_macro_clusters"] == 8
    assert params["max_micro_clusters"] == 80
    assert "micro_ratio" not in params
    # every other parameter is River's constructor default
    assert params["time_gap"] == 100 and params["time_window"] == 1000


def test_raw_grid_is_not_rescaled_but_ladder_is():
    probe = {"s": 10.0}
    spec = hpo_compare.GRIDS["DenStream"]
    raw, _ = hpo_compare.build_grid("DenStream", "raw", spec)
    ladder, _ = hpo_compare.build_grid("DenStream", "ladder", spec)
    assert {hpo_compare.resolve_candidate(c, probe)["epsilon"] for c in raw} == set(spec["levels"]["raw"])
    for c in ladder:
        assert hpo_compare.resolve_candidate(c, probe)["epsilon"] == pytest.approx(c["g"] * 10.0)
        assert "epsilon" not in c["shape"]
    # the untuned parameters are River's, not the registry preset's
    assert all(c["shape"]["decaying_factor"] == 0.25 for c in ladder)
    assert all(c["shape"]["n_samples_init"] == 1000 for c in ladder)


def test_grids_never_share_a_config_id():
    for algo in hpo_compare.ALGOS:
        spec = hpo_compare.GRIDS[algo]
        cands = (
            hpo_compare.build_grid(algo, "raw", spec)[0]
            + hpo_compare.build_grid(algo, "ladder", spec)[0]
            + [hpo_compare.default_candidate(algo)]
        )
        assert len({c["config_id"] for c in cands}) == len(cands)


def test_config_id_is_fold_independent():
    from hpo_pipeline._common import config_id

    c = hpo_compare.build_grid("DenStream", "ladder", hpo_compare.GRIDS["DenStream"])[0][0]
    assert c["config_id"] == config_id("DenStream", c["granularity_level"], c["g"], c["shape"])
    a = hpo_compare.resolve_candidate(c, {"s": 1.0})["epsilon"]
    b = hpo_compare.resolve_candidate(c, {"s": 100.0})["epsilon"]
    assert b == pytest.approx(100 * a)


def test_merge_grids_overrides_one_leaf_only():
    merged = hpo_compare.merge_grids(
        hpo_compare.GRIDS, {"DenStream": {"levels": {"raw": [1.0, 0.1]}}}
    )
    assert merged["DenStream"]["levels"]["raw"] == [1.0, 0.1]
    assert merged["DenStream"]["levels"]["ladder"] == hpo_compare.GRIDS["DenStream"]["levels"]["ladder"]
    assert merged["DenStream"]["shape"] == hpo_compare.GRIDS["DenStream"]["shape"]
    assert merged["CluStream"] == hpo_compare.GRIDS["CluStream"]
    # the module constant is untouched
    assert hpo_compare.GRIDS["DenStream"]["levels"]["raw"][0] == 16.0


def test_tuned_params_name_absolute_parameters():
    assert hpo_compare.tuned_params("CluStream", hpo_compare.GRIDS) == [
        "max_micro_clusters", "micro_cluster_r_factor", "n_macro_clusters"]
    assert hpo_compare.tuned_params("DenStream", hpo_compare.GRIDS) == ["beta", "epsilon", "mu"]


# --- selection ---------------------------------------------------------------
def test_oracle_picks_best_config_on_the_fold_itself():
    tune = _tune([
        ("A", "f1", "c1", 1, 1, 0.2, "ok"), ("A", "f1", "c2", 2, 1, 0.9, "ok"),
        ("A", "f2", "c1", 1, 1, 0.7, "ok"), ("A", "f2", "c2", 2, 1, 0.1, "ok"),
    ])
    assert hpo_compare.pick_oracle(tune) == {("A", "f1"): "c2", ("A", "f2"): "c1"}


def test_oracle_breaks_ties_towards_the_coarser_level_then_the_id():
    tune = _tune([("A", "f1", "c3", 3, 1, 0.5, "ok"), ("A", "f1", "c2", 2, 1, 0.5, "ok"),
                  ("A", "f1", "b2", 2, 2, 0.5, "ok")])
    assert hpo_compare.pick_oracle(tune) == {("A", "f1"): "b2"}


def test_lodo_never_reads_the_held_out_fold():
    # c2 is superb on f3 only; holding out f3 must not pick it.
    tune = _tune([
        ("A", "f1", "c1", 1, 1, 0.8, "ok"), ("A", "f1", "c2", 2, 1, 0.1, "ok"),
        ("A", "f2", "c1", 1, 1, 0.8, "ok"), ("A", "f2", "c2", 2, 1, 0.1, "ok"),
        ("A", "f3", "c1", 1, 1, 0.0, "ok"), ("A", "f3", "c2", 2, 1, 1.0, "ok"),
    ])
    picks = hpo_compare.pick_lodo(tune, ["f1", "f2", "f3"])
    assert picks[("A", "f3")] == "c1"


def test_lodo_skips_configs_that_failed_on_an_optimisation_fold():
    tune = _tune([
        ("A", "f1", "c1", 1, 1, 0.3, "ok"), ("A", "f1", "c2", 2, 1, 0.9, "timeout"),
        ("A", "f2", "c1", 1, 1, 0.3, "ok"), ("A", "f2", "c2", 2, 1, 0.9, "ok"),
        ("A", "f3", "c1", 1, 1, 0.3, "ok"), ("A", "f3", "c2", 2, 1, 0.9, "ok"),
    ])
    picks = hpo_compare.pick_lodo(tune, ["f1", "f2", "f3"])
    assert picks[("A", "f3")] == "c1"   # c2 did not run on f1
    assert picks[("A", "f1")] == "c2"   # f1 is held out, so its failure is irrelevant


def test_group_lodo_keeps_a_synthetic_majority_from_outvoting_real_folds():
    rows = []
    for d in ("s1", "s2", "s3"):
        rows += [("A", d, "c1", 1, 1, 0.1, "ok"), ("A", d, "c2", 2, 1, 0.9, "ok")]
    for d in ("r1", "r2", "r3"):
        rows += [("A", d, "c1", 1, 1, 0.5, "ok"), ("A", d, "c2", 2, 1, 0.0, "ok")]
    tune = _tune(rows)
    folds = ["s1", "s2", "s3", "r1", "r2", "r3"]
    groups = {f: ("synth" if f.startswith("s") else "real") for f in folds}

    assert hpo_compare.pick_lodo(tune, folds)[("A", "r1")] == "c2"           # plain: outvoted
    assert hpo_compare.pick_lodo(tune, folds, groups)[("A", "r1")] == "c1"   # within group
    cross = hpo_compare.pick_lodo(tune, folds, groups, cross=True)
    assert cross[("A", "r1")] == "c2" and cross[("A", "s1")] == "c1"         # other group only


def test_p2r_transfers_the_shape_and_re_picks_the_level_on_the_held_out_curve():
    # Two shape points. theta=1 is the better one on the optimisation folds.
    # On f3 the criterion at theta=1 peaks at level 2 while ARI peaks at level 1:
    # the rule must follow the criterion, never the held-out labels.
    rows = []
    for d in ("f1", "f2"):
        rows += [(d, "a1", 1, 1, 0.2, 1.0), (d, "a2", 2, 1, 0.8, 2.0),
                 (d, "b1", 1, 2, 0.1, 5.0), (d, "b2", 2, 2, 0.1, 6.0)]
    rows += [("f3", "a1", 1, 1, 0.9, 1.0), ("f3", "a2", 2, 1, 0.1, 2.0),
             ("f3", "b1", 1, 2, 0.0, 9.0), ("f3", "b2", 2, 2, 0.0, 1.0)]
    tune = _tune([("A", d, cid, lv, th, ari, "ok", crit) for d, cid, lv, th, ari, crit in rows])
    picks = hpo_compare.pick_p2r(tune, ["f1", "f2", "f3"], None, "R_max", "matr_trace_win", 2)
    assert picks[("A", "f3")] == (1, 2)


def test_p2r_falls_back_to_the_best_shape_with_a_complete_held_out_curve():
    # theta=1 wins on the optimisation folds but its curve is broken on f3
    # (a timed-out rung); theta=2 is complete there and must be used instead.
    rows = []
    for d in ("f1", "f2"):
        rows += [(d, "a1", 1, 1, 0.9, 1.0, "ok"), (d, "a2", 2, 1, 0.9, 2.0, "ok"),
                 (d, "b1", 1, 2, 0.5, 1.0, "ok"), (d, "b2", 2, 2, 0.5, 2.0, "ok")]
    rows += [("f3", "a1", 1, 1, 0.9, 1.0, "ok"), ("f3", "a2", 2, 1, None, None, "timeout"),
             ("f3", "b1", 1, 2, 0.3, 1.0, "ok"), ("f3", "b2", 2, 2, 0.3, 2.0, "ok")]
    tune = _tune([("A", d, cid, lv, th, ari, st, crit) for d, cid, lv, th, ari, crit, st in rows])
    picks = hpo_compare.pick_p2r(tune, ["f1", "f2", "f3"], None, "R_max", "matr_trace_win", 2)
    assert picks[("A", "f3")] == (2, 2)


def test_p2r_skips_a_fold_whose_curve_is_broken():
    rows = [("f1", "a1", 1, 1, 0.2, 1.0, "ok"), ("f1", "a2", 2, 1, 0.8, 2.0, "ok"),
            ("f2", "a1", 1, 1, 0.2, 1.0, "ok"), ("f2", "a2", 2, 1, 0.8, 2.0, "ok"),
            ("f3", "a1", 1, 1, 0.2, 1.0, "ok"), ("f3", "a2", 2, 1, 0.8, None, "timeout")]
    tune = _tune([("A", d, cid, lv, th, ari, st, crit) for d, cid, lv, th, ari, crit, st in rows])
    picks = hpo_compare.pick_p2r(tune, ["f1", "f2", "f3"], None, "R_max", "matr_trace_win",
                                 {"A": 2})
    assert ("A", "f3") not in picks and ("A", "f1") in picks


def test_parse_p2r_rejects_unknown_entries():
    assert hpo_compare.parse_p2r("R_knee:ssq, R_max:silhouette") == [
        ("R_knee", "ssq"), ("R_max", "silhouette")]
    assert hpo_compare.parse_p2r("") == []
    with pytest.raises(SystemExit):
        hpo_compare.parse_p2r("R_best:ari")


# --- bookkeeping -------------------------------------------------------------
def _job(cid, status_row=None):
    cand = {"config_id": cid, "grid": "ladder", "granularity_level": 3, "theta": 1, "g": 0.8,
            "algo": "DenStream", "shape": {"mu": 2, "beta": 0.75}}
    return {"algo": "DenStream", "candidate": cand, "fold": "kdd99", "seed": 1,
            "requested": "full", "params": {"epsilon": 680.2, "mu": 2, "beta": 0.75},
            "tuned": ["beta", "epsilon", "mu"]}


def _row(cid, status):
    return {"algo": "DenStream", "config_id": cid, "dataset": "kdd99", "seed": 1,
            "requested": "full", "params_resolved": {"epsilon": 680.2, "mu": 2, "beta": 0.75},
            "status": status}


def test_pending_keeps_failed_runs_done_unless_retried():
    jobs = [_job("ok1"), _job("to1"), _job("new")]
    rows = [_row("ok1", "ok"), _row("to1", "timeout")]
    assert [j["candidate"]["config_id"] for j in hpo_compare.pending(jobs, rows)] == ["new"]
    retried = hpo_compare.pending(jobs, rows, retry_failed=True)
    assert [j["candidate"]["config_id"] for j in retried] == ["to1", "new"]


def test_pending_does_not_retry_a_key_whose_retry_already_succeeded():
    rows = [_row("to1", "timeout"), _row("to1", "ok")]
    assert hpo_compare.pending([_job("to1")], rows, retry_failed=True) == []


def test_describe_names_the_tuned_values_and_their_origin():
    assert hpo_compare.describe(_job("x")) == (
        "beta=0.75, epsilon=680.2, mu=2 [ladder L3 t1 c=0.8]")
