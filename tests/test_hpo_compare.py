"""Tests for the HPO pilot's selection logic and grids. No datasets, no runs."""

from __future__ import annotations

import pandas as pd
import pytest

from hpo_pipeline import hpo_compare
from hpo_pipeline.space import GRAN, resolve


def _tune(rows):
    return pd.DataFrame(
        [
            {"algo": a, "dataset": d, "granularity_level": lv, "ari": s,
             "runtime_s": 1.0, "status": st}
            for a, d, lv, s, st in rows
        ]
    )


def test_oracle_picks_best_level_on_the_fold_itself():
    tune = _tune([
        ("A", "f1", 1, 0.2, "ok"), ("A", "f1", 2, 0.9, "ok"),
        ("A", "f2", 1, 0.7, "ok"), ("A", "f2", 2, 0.1, "ok"),
    ])
    assert hpo_compare.pick_oracle(tune) == {("A", "f1"): 2, ("A", "f2"): 1}


def test_oracle_breaks_ties_towards_the_coarser_level():
    tune = _tune([("A", "f1", 3, 0.5, "ok"), ("A", "f1", 2, 0.5, "ok")])
    assert hpo_compare.pick_oracle(tune) == {("A", "f1"): 2}


def test_lodo_never_reads_the_held_out_fold():
    # Level 2 is superb on f3 only; holding out f3 must not pick it.
    tune = _tune([
        ("A", "f1", 1, 0.8, "ok"), ("A", "f1", 2, 0.1, "ok"),
        ("A", "f2", 1, 0.8, "ok"), ("A", "f2", 2, 0.1, "ok"),
        ("A", "f3", 1, 0.0, "ok"), ("A", "f3", 2, 1.0, "ok"),
    ])
    picks = hpo_compare.pick_lodo(tune, ["f1", "f2", "f3"])
    assert picks[("A", "f3")] == 1


def test_lodo_skips_levels_that_failed_on_an_optimisation_fold():
    tune = _tune([
        ("A", "f1", 1, 0.3, "ok"), ("A", "f1", 2, 0.9, "timeout"),
        ("A", "f2", 1, 0.3, "ok"), ("A", "f2", 2, 0.9, "ok"),
        ("A", "f3", 1, 0.3, "ok"), ("A", "f3", 2, 0.9, "ok"),
    ])
    picks = hpo_compare.pick_lodo(tune, ["f1", "f2", "f3"])
    assert picks[("A", "f3")] == 1   # level 2 did not run on f1
    assert picks[("A", "f1")] == 2   # f1 is held out, so its failure is irrelevant


def test_group_lodo_keeps_a_synthetic_majority_from_outvoting_real_folds():
    # Three synthetic folds love level 2, the real folds love level 1.
    rows = []
    for d in ("s1", "s2", "s3"):
        rows += [("A", d, 1, 0.1, "ok"), ("A", d, 2, 0.9, "ok")]
    for d in ("r1", "r2", "r3"):
        rows += [("A", d, 1, 0.5, "ok"), ("A", d, 2, 0.0, "ok")]
    tune = _tune(rows)
    folds = ["s1", "s2", "s3", "r1", "r2", "r3"]
    groups = {f: ("synth" if f.startswith("s") else "real") for f in folds}

    assert hpo_compare.pick_lodo(tune, folds)[("A", "r1")] == 2           # plain: outvoted
    assert hpo_compare.pick_lodo(tune, folds, groups)[("A", "r1")] == 1   # within group
    cross = hpo_compare.pick_lodo(tune, folds, groups, cross=True)
    assert cross[("A", "r1")] == 2 and cross[("A", "s1")] == 1            # other group only


def _p2r_tune(rows):
    return pd.DataFrame(
        [
            {"algo": "A", "dataset": d, "granularity_level": lv, "ari": ari,
             "runtime_s": 1.0, "status": st, "matr_trace_win": crit}
            for d, lv, ari, crit, st in rows
        ]
    )


def test_p2r_picks_from_the_held_out_curve_never_its_labels():
    # On f3 the criterion peaks at level 2 while ARI peaks at level 1.
    rows = []
    for d in ("f1", "f2"):
        rows += [(d, 1, 0.2, 1.0, "ok"), (d, 2, 0.8, 2.0, "ok")]
    rows += [("f3", 1, 0.9, 1.0, "ok"), ("f3", 2, 0.1, 2.0, "ok")]
    picks = hpo_compare.pick_p2r(_p2r_tune(rows), ["f1", "f2", "f3"], None,
                                 "R_max", "matr_trace_win", n_levels=2)
    assert picks[("A", "f3")] == 2


def test_p2r_skips_a_fold_whose_curve_is_broken():
    rows = [("f1", 1, 0.2, 1.0, "ok"), ("f1", 2, 0.8, 2.0, "ok"),
            ("f2", 1, 0.2, 1.0, "ok"), ("f2", 2, 0.8, 2.0, "ok"),
            ("f3", 1, 0.2, 1.0, "ok"), ("f3", 2, 0.8, None, "timeout")]
    picks = hpo_compare.pick_p2r(_p2r_tune(rows), ["f1", "f2", "f3"], None,
                                 "R_max", "matr_trace_win", n_levels=2)
    assert ("A", "f3") not in picks and ("A", "f1") in picks


def test_parse_p2r_rejects_unknown_entries():
    assert hpo_compare.parse_p2r("R_knee:ssq, R_max:silhouette") == [
        ("R_knee", "ssq"), ("R_max", "silhouette")]
    assert hpo_compare.parse_p2r("") == []
    with pytest.raises(SystemExit):
        hpo_compare.parse_p2r("R_best:ari")


def _job(cid, status_row=None):
    cand = {"config_id": cid, "grid": "ladder", "granularity_level": 3, "g": 0.7}
    return {"algo": "DenStream", "candidate": cand, "fold": "kdd99", "seed": 1,
            "requested": "full", "params": {"epsilon": 595.2}}


def _row(cid, status):
    return {"algo": "DenStream", "config_id": cid, "dataset": "kdd99", "seed": 1,
            "requested": "full", "params_resolved": {"epsilon": 595.2}, "status": status}


def test_pending_keeps_failed_runs_done_unless_retried():
    jobs = [_job("ok1"), _job("to1"), _job("new")]
    rows = [_row("ok1", "ok"), _row("to1", "timeout")]
    assert [j["candidate"]["config_id"] for j in hpo_compare.pending(jobs, rows)] == ["new"]
    retried = hpo_compare.pending(jobs, rows, retry_failed=True)
    assert [j["candidate"]["config_id"] for j in retried] == ["to1", "new"]


def test_pending_does_not_retry_a_key_whose_retry_already_succeeded():
    rows = [_row("to1", "timeout"), _row("to1", "ok")]
    assert hpo_compare.pending([_job("to1")], rows, retry_failed=True) == []


def test_describe_names_the_value_and_its_origin():
    assert hpo_compare.describe(_job("x")) == "epsilon=595.2 [ladder L3 c=0.7]"


def test_defaults_are_river_constructor_not_registry_preset():
    default = hpo_compare.default_candidate("DenStream")
    # River's constructor says beta=0.75, mu=2, epsilon=0.02; the registry preset differs.
    assert default["shape"]["epsilon"] == 0.02
    assert default["shape"]["beta"] == 0.75


def test_raw_grid_is_not_rescaled_but_ladder_is():
    probe = {"s": 10.0}
    raw = hpo_compare.raw_grid("DenStream", [0.1, 0.25, 0.5, 1.0])
    ladder = hpo_compare.ladder_grid("DenStream")
    assert [resolve(c, probe)["epsilon"] for c in raw] == [0.1, 0.25, 0.5, 1.0]
    assert resolve(ladder[0], probe)["epsilon"] == ladder[0]["g"] * 10.0
    assert all(GRAN["DenStream"][0] not in c["shape"] for c in ladder)


def test_grids_never_share_a_config_id():
    for algo in hpo_compare.ALGOS:
        cands = (
            hpo_compare.raw_grid(algo, hpo_compare.M1_GRID[algo])
            + hpo_compare.ladder_grid(algo)
            + [hpo_compare.default_candidate(algo)]
        )
        assert len({c["config_id"] for c in cands}) == len(cands)


def test_ladder_ids_unchanged_so_earlier_runs_are_reused():
    # The ladder candidates must hash exactly as before the raw grid was added.
    from hpo_pipeline._common import config_id

    c = hpo_compare.ladder_grid("CluStream")[0]
    assert c["config_id"] == config_id("CluStream", 1, c["g"], c["shape"])
