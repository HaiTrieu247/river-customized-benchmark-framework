"""Tests for the HPO pilot's selection logic and grids. No datasets, no runs."""

from __future__ import annotations

import pandas as pd

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
