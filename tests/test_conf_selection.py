"""conf_pipeline.selection: what choosing the algorithm under defaults costs under online tuning."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conf_pipeline import _common as C
from conf_pipeline import selection as S

TAIL = "lag_ari_win_tail_mean"


def _s(**kw) -> pd.Series:
    return pd.Series(kw, dtype=float)


# --- the pick set and its readings -----------------------------------------------------------

def test_pick_set_is_the_exact_top_or_everything_within_the_band():
    scores = _s(A=0.50, B=0.50, C=0.47, D=0.30)
    assert S.pick_set(scores, 0.0) == ["A", "B"]
    assert S.pick_set(scores, 0.05) == ["A", "B", "C"]
    assert S.pick_set(_s(A=0.5, B=np.nan), 0.0) == ["A"]
    assert S.pick_set(pd.Series(dtype=float), 0.0) == []


def test_a_tie_at_the_top_reads_expected_best_and_worst_regret():
    target = _s(A=0.9, B=0.6, C=0.8, D=0.1)
    out = S.pick_readings(["A", "B"], target, b=0.0)
    assert out["target_best"] == pytest.approx(0.9)
    assert out["target_best_algos"] == "A"
    assert out["regret_exp"] == pytest.approx((0.0 + 0.3) / 2)
    assert out["regret_best"] == pytest.approx(0.0)
    assert out["regret_worst"] == pytest.approx(0.3)
    # a uniform pick from {A, B} is the target's best half of the time
    assert out["top1_agree"] == pytest.approx(0.5)
    assert out["p_regret_gt_b"] == pytest.approx(0.5)


def test_a_loss_within_b_is_no_practical_loss():
    target = _s(A=0.9, B=0.87, C=0.5)
    out = S.pick_readings(["B"], target, b=0.05)
    assert out["regret_exp"] == pytest.approx(0.03)
    assert out["top1_agree"] == 0.0          # B is not the exact best ...
    assert out["p_regret_gt_b"] == 0.0       # ... but within b of it
    out = S.pick_readings(["C"], target, b=0.05)
    assert out["p_regret_gt_b"] == 1.0


def test_ties_at_the_targets_top_all_count_as_its_best():
    out = S.pick_readings(["B"], _s(A=0.9, B=0.9, C=0.1), b=0.0)
    assert out["regret_exp"] == 0.0
    assert out["top1_agree"] == 1.0
    assert out["target_best_algos"] == "A;B"


def test_an_empty_pick_or_a_pick_without_a_target_reading_is_undefined():
    out = S.pick_readings([], _s(A=0.9), b=0.0)
    assert np.isnan(out["regret_exp"]) and out["note"]
    out = S.pick_readings(["Z"], _s(A=0.9), b=0.0)
    assert np.isnan(out["regret_exp"]) and "Z" in out["note"]


# --- rankings: top-k and Kendall ---------------------------------------------------------------

def test_inclusion_in_the_top_k_shares_the_boundary_seats_among_the_tied():
    p = S.inclusion(_s(A=0.9, B=0.5, C=0.5, D=0.5, E=0.1), k=2)
    assert p.to_dict() == pytest.approx({"A": 1.0, "B": 1 / 3, "C": 1 / 3, "D": 1 / 3, "E": 0.0})
    assert p.sum() == pytest.approx(2.0)


def test_topk_overlap_is_the_expected_share_of_shared_seats():
    sel = _s(A=0.9, B=0.8, C=0.7, D=0.1)
    tgt = _s(A=0.2, B=0.9, C=0.8, D=0.7)
    value, note = S.topk_overlap(sel, tgt, k=2)
    assert value == pytest.approx(0.5) and note is None  # {A, B} against {B, C}
    # a constant score (random choice) overlaps by k/n in expectation
    value, _ = S.topk_overlap(_s(A=0.0, B=0.0, C=0.0, D=0.0), tgt, k=2)
    assert value == pytest.approx(2 / 4)
    value, note = S.topk_overlap(_s(A=1.0, B=0.0), _s(A=1.0, B=0.0), k=3)
    assert np.isnan(value) and "fewer than 3" in note


def test_kendall_tau_b_on_the_shared_algorithms_and_its_undefined_cases():
    tau, note = S.kendall(_s(A=3.0, B=2.0, C=1.0), _s(A=0.3, B=0.2, C=0.1))
    assert tau == pytest.approx(1.0) and note is None
    tau, _ = S.kendall(_s(A=3.0, B=2.0, C=1.0, D=np.nan), _s(A=0.1, B=0.2, C=0.3, D=0.9))
    assert tau == pytest.approx(-1.0)
    tau, note = S.kendall(_s(A=1.0, B=2.0), _s(A=1.0, B=2.0))
    assert np.isnan(tau) and "fewer than 3" in note
    tau, note = S.kendall(_s(A=0.0, B=0.0, C=0.0), _s(A=0.1, B=0.2, C=0.3))
    assert np.isnan(tau) and "selector" in note and "constant" in note
    tau, note = S.kendall(_s(A=0.1, B=0.2, C=0.3), _s(A=0.5, B=0.5, C=0.5))
    assert np.isnan(tau) and "target" in note


# --- global choices made on other blocks ---------------------------------------------------------

def _wide(rows: dict) -> pd.DataFrame:
    """{fold: {algo: value}} -> folds x algos."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "fold"
    return df.astype(float)


def test_lobo_scores_leave_out_every_fold_of_the_held_out_block():
    train = _wide({
        "insects_abrupt_balanced": {"A": 1.0, "B": 0.0},
        "insects_gradual_balanced": {"A": 1.0, "B": 0.0},
        "covtype": {"A": 0.0, "B": 0.4},
        "kdd99": {"A": 0.0, "B": 0.2},
    })
    scores, folds = S.lobo_scores(train, "insects_abrupt_balanced", by="score")
    assert folds == ["covtype", "kdd99"]
    assert scores.to_dict() == pytest.approx({"A": 0.0, "B": 0.3})
    scores, folds = S.lobo_scores(train, "covtype", by="score")
    assert sorted(folds) == ["insects_abrupt_balanced", "insects_gradual_balanced", "kdd99"]
    assert scores.to_dict() == pytest.approx({"A": 2 / 3, "B": 0.2 / 3})


def test_lobo_by_rank_rewards_the_algorithm_that_is_often_ahead_not_far_ahead_once():
    train = _wide({
        "covtype": {"A": 0.9, "B": 0.1},
        "kdd99": {"A": 0.40, "B": 0.41},
        "shuttle": {"A": 0.40, "B": 0.41},
        "electricity": {"A": 0.5, "B": 0.5},
    })
    by_score, _ = S.lobo_scores(train, "evolving", by="score")
    by_rank, _ = S.lobo_scores(train, "evolving", by="rank")
    assert by_score.idxmax() == "A"
    assert by_rank.idxmax() == "B"
    # a tie shares the average rank; the score is minus the mean rank
    assert by_rank.to_dict() == pytest.approx({"A": -(1 + 2 + 2 + 1.5) / 4, "B": -(2 + 1 + 1 + 1.5) / 4})


def test_no_fold_outside_the_held_out_block_gives_no_global_choice():
    train = _wide({"insects_abrupt_balanced": {"A": 1.0}, "insects_gradual_balanced": {"A": 0.5}})
    scores, folds = S.lobo_scores(train, "insects_abrupt_balanced", by="score")
    assert scores.empty and folds == []


# --- one table of per-fold readings ----------------------------------------------------------------

def _toy_tables():
    """Two folds of one block (INSECTS) and two singletons. On covtype the defaults put A
    first, online tuning B; the global choice from the other blocks is A on covtype (kdd99
    favours A, the INSECTS folds too)."""
    sel = _wide({
        "covtype": {"A": 0.30, "B": 0.10, "C": 0.05},
        "kdd99": {"A": 0.20, "B": 0.20, "C": 0.00},
        "insects_abrupt_balanced": {"A": 0.02, "B": 0.01, "C": 0.00},
        "insects_gradual_balanced": {"A": 0.03, "B": 0.01, "C": 0.00},
    })
    tgt = _wide({
        "covtype": {"A": 0.40, "B": 0.90, "C": 0.20},
        "kdd99": {"A": 0.60, "B": 0.50, "C": 0.10},
        "insects_abrupt_balanced": {"A": 0.30, "B": 0.10, "C": 0.20},
        "insects_gradual_balanced": {"A": 0.30, "B": 0.40, "C": 0.20},
    })
    return sel, tgt


def _row(rows, **match):
    sub = rows
    for k, v in match.items():
        sub = sub[sub[k] == v]
    assert len(sub) == 1, (match, len(sub))
    return sub.iloc[0]


def test_fold_selection_reads_regret_of_the_default_pick_and_of_the_baselines():
    sel, tgt = _toy_tables()
    rows = S.fold_selection("same_data", sel, {"confstream": tgt}, {"default": sel, "confstream": tgt},
                            band=0.05, k=2)
    cov = _row(rows, selector="default", target="confstream", band=0.0, fold="covtype")
    assert cov["picked"] == "A" and cov["regret_exp"] == pytest.approx(0.5)
    assert cov["group"] == "real" and cov["block"] == "covtype"
    kdd = _row(rows, selector="default", target="confstream", band=0.0, fold="kdd99")
    assert kdd["picked"] == "A;B" and kdd["n_picked"] == 2
    assert kdd["regret_exp"] == pytest.approx(0.05) and kdd["regret_worst"] == pytest.approx(0.1)
    rnd = _row(rows, selector="random", target="confstream", band=0.0, fold="covtype")
    assert rnd["regret_exp"] == pytest.approx(0.9 - (0.4 + 0.9 + 0.2) / 3)
    assert rnd["regret_worst"] == pytest.approx(0.7) and rnd["n_picked"] == 3
    assert np.isnan(rnd["tau"]) and "constant" in rnd["tau_note"]
    # within 0.05 of the top on the INSECTS default readings, every algorithm is tied
    ins = _row(rows, selector="default", target="confstream", band=0.05, fold="insects_abrupt_balanced")
    assert ins["n_picked"] == 3 and ins["flat_within_b"]
    assert ins["regret_exp"] == pytest.approx(0.3 - 0.2)
    assert not _row(rows, selector="default", target="confstream", band=0.05, fold="covtype")["flat_within_b"]


def test_every_band_row_reads_its_pick_against_the_same_b():
    sel, tgt = _toy_tables()
    rows = S.fold_selection("same_data", sel, {"confstream": tgt}, {"default": sel, "confstream": tgt},
                            band=0.05, k=2)
    assert set(rows["b"]) == {0.05}
    # the exact pick: covtype loses 0.5 > b; kdd99's tie {A, B} loses 0 and 0.1
    assert _row(rows, selector="default", band=0.0, fold="covtype")["p_regret_gt_b"] == 1.0
    assert _row(rows, selector="default", band=0.0, fold="kdd99")["p_regret_gt_b"] == pytest.approx(0.5)
    # the band's pick set on insects_abrupt_balanced: regrets 0, 0.2, 0.1
    ins = _row(rows, selector="default", band=0.05, fold="insects_abrupt_balanced")
    assert ins["p_regret_gt_b"] == pytest.approx(2 / 3)
    # without a band, b is 0: a loss of any size counts, the complement of top-1 agreement
    rows0 = S.fold_selection("same_data", sel, {"confstream": tgt}, {"default": sel, "confstream": tgt},
                             band=0.0, k=2)
    assert set(rows0["band"]) == {0.0} and set(rows0["b"]) == {0.0}
    kdd = _row(rows0, selector="default", fold="kdd99")
    assert kdd["p_regret_gt_b"] == pytest.approx(1 - kdd["top1_agree"])


def test_a_global_choice_never_reads_the_test_fold():
    sel, tgt = _toy_tables()
    rows = S.fold_selection("same_data", sel, {"confstream": tgt}, {"default": sel, "confstream": tgt},
                            band=0.0, k=2)
    # covtype is held out: the online readings of kdd99 and the two INSECTS folds favour A
    # (0.6, 0.3, 0.3 against 0.5, 0.1, 0.4), although B wins covtype itself by 0.5
    cov = _row(rows, selector="lobo_target", target="confstream", band=0.0, fold="covtype")
    assert cov["picked"] == "A" and cov["regret_exp"] == pytest.approx(0.5)
    assert cov["n_train_folds"] == 3
    # an INSECTS fold is held out with its whole block: only covtype and kdd99 choose
    ins = _row(rows, selector="lobo_target", target="confstream", band=0.0, fold="insects_gradual_balanced")
    assert ins["n_train_folds"] == 2
    assert ins["picked"] == "B"  # mean of covtype, kdd99: A 0.5, B 0.7
    assert ins["regret_exp"] == pytest.approx(0.0)
    dflt = _row(rows, selector="lobo_default", target="confstream", band=0.0, fold="insects_gradual_balanced")
    assert dflt["picked"] == "A"  # defaults on covtype, kdd99: A 0.25, B 0.15
    assert dflt["regret_exp"] == pytest.approx(0.1)


def test_a_rank_score_takes_no_band():
    sel, tgt = _toy_tables()
    rows = S.fold_selection("same_data", sel, {"confstream": tgt}, {"default": sel, "confstream": tgt},
                            band=0.5, k=2)
    exact = _row(rows, selector="lobo_target_rank", target="confstream", band=0.0, fold="covtype")
    wide = _row(rows, selector="lobo_target_rank", target="confstream", band=0.5, fold="covtype")
    assert exact["picked"] == wide["picked"]
    assert "rank" in wide["note"]


def test_a_fold_the_target_never_read_has_no_regret_and_says_why():
    sel, tgt = _toy_tables()
    tgt = tgt.drop(index="kdd99")
    rows = S.fold_selection("same_data", sel, {"confstream": tgt}, {"default": sel, "confstream": tgt},
                            band=0.0, k=2)
    kdd = _row(rows, selector="default", target="confstream", band=0.0, fold="kdd99")
    assert np.isnan(kdd["regret_exp"]) and "no target reading" in kdd["note"]


# --- aggregation ---------------------------------------------------------------------------------

def test_groups_are_fold_and_block_weighted_with_their_denominators():
    sel, tgt = _toy_tables()
    rows = S.fold_selection("same_data", sel, {"confstream": tgt}, {"default": sel, "confstream": tgt},
                            band=0.0, k=2)
    groups = S.aggregate(rows)
    g = groups[(groups["selector"] == "default") & (groups["level"] == "group") & (groups["unit"] == "real")]
    assert len(g) == 1
    g = g.iloc[0]
    # regrets: covtype 0.5, kdd99 0.05, insects_abrupt 0.0, insects_gradual 0.1
    assert g["n_folds"] == 4 and g["n_blocks"] == 3
    assert g["regret_exp_fw"] == pytest.approx((0.5 + 0.05 + 0.0 + 0.1) / 4)
    assert g["regret_exp_bw"] == pytest.approx((0.5 + 0.05 + 0.05) / 3)
    assert g["regret_exp_max"] == pytest.approx(0.5)
    assert g["regret_exp_med"] == pytest.approx(0.075)
    assert g["n_flat_folds"] == 0  # band 0: nothing is flat
    b = groups[(groups["selector"] == "default") & (groups["level"] == "block") & (groups["unit"] == "insects")]
    assert b.iloc[0]["n_folds"] == 2 and b.iloc[0]["regret_exp_fw"] == pytest.approx(0.05)
    empty = groups[(groups["selector"] == "default") & (groups["unit"] == "param_drift")].iloc[0]
    assert empty["n_folds"] == 0 and np.isnan(empty["regret_exp_fw"])


# --- the readings each variant selects and evaluates on ----------------------------------------------

def _values(rows):
    return pd.DataFrame(rows, columns=["algo", "fold", "arm", "seed", "value"])


def test_wide_means_over_the_seeds_asked_for():
    vals = _values([("A", "covtype", "default", s, v) for s, v in ((1, 0.1), (2, 0.2), (3, 0.6))]
                   + [("B", "covtype", "default", 1, 0.5)])
    w = S.wide(vals, "default")
    assert w.loc["covtype", "A"] == pytest.approx(0.3)
    w = S.wide(vals, "default", seeds=(2, 3))
    assert w.loc["covtype", "A"] == pytest.approx(0.4) and np.isnan(w.loc["covtype", "B"])


def test_variant_tables_select_on_seed_one_and_evaluate_on_the_held_out_seeds():
    vals = _values(
        [(a, "covtype", "default", s, v) for a, s, v in
         (("A", 1, 0.9), ("A", 2, 0.0), ("A", 3, 0.0), ("B", 1, 0.1), ("B", 2, 0.5), ("B", 3, 0.5))]
        + [(a, "covtype", "confstream", s, v) for a, s, v in
           (("A", 1, 0.9), ("A", 2, 0.1), ("A", 3, 0.1), ("B", 1, 0.0), ("B", 2, 0.6), ("B", 3, 0.6))]
        + [(a, "covtype", "oracle_static", s, 0.5) for a in ("A", "B") for s in (2, 3, 4)])
    tables = S.variant_tables(vals, None, targets=("confstream", "oracle_static"),
                              select_seeds=(1,), eval_seeds=(2, 3))
    sel, tgts, note = tables["heldout_seeds"]
    assert sel.loc["covtype"].to_dict() == pytest.approx({"A": 0.9, "B": 0.1})
    assert tgts["confstream"].loc["covtype"].to_dict() == pytest.approx({"A": 0.1, "B": 0.6})
    assert "oracle_static" in tgts
    assert "prefix_suffix" in tables and tables["prefix_suffix"][0] is None
    assert tables["prefix_suffix"][2]  # the reason it did not run


def _window_rows(algo, fold, arm, seed, values, start=1):
    return [{"algo": algo, "fold": fold, "arm": arm, "seed": seed, "config_id": None,
             "window": start + i, "lag_ari_win": v} for i, v in enumerate(values)]


def test_half_values_split_each_runs_tail_windows_in_time():
    # windows 1..9, tail from 4: tail windows 4..9, prefix 4..6, suffix 7..9; window 8 single-class
    values = [9.0, 9.0, 9.0, 0.1, 0.2, 0.3, 0.7, None, 0.9]
    win = pd.DataFrame(_window_rows("A", "covtype", "default", 1, values))
    halves = S.half_values(win, {("A", "covtype", 1, "default"): 4})
    got = halves.set_index("half")["value"].to_dict()
    assert got == pytest.approx({"prefix": 0.2, "suffix": 0.8})
    # an odd number of tail windows gives the extra one to the suffix
    win = pd.DataFrame(_window_rows("A", "covtype", "default", 1, [0.1, 0.2, 0.3, 0.4, 0.5]))
    halves = S.half_values(win, {("A", "covtype", 1, "default"): 1})
    got = halves.set_index("half")["value"].to_dict()
    assert got == pytest.approx({"prefix": 0.15, "suffix": 0.4})


# --- end to end ----------------------------------------------------------------------------------

def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


ALGOS = ("A", "B", "CluStream", "DenStreamPatchedFast")
TOY_FOLDS = ("covtype", "kdd99", "insects_abrupt_balanced", "scale_drift", "evolving")


def _toy_out_dir(root):
    rng = np.random.default_rng(0)
    runs, windows = [], []
    for fold in TOY_FOLDS:
        for ai, algo in enumerate(ALGOS):
            for arm, seeds in (("default", (1, 2, 3)), ("confstream", (1, 2, 3)),
                               ("frozen_prefix", (1, 2, 3)), ("oracle_static", (2, 3, 4))):
                for seed in seeds:
                    vals = list(np.clip(rng.normal(0.2 + 0.1 * ai + (0.2 if arm != "default" else 0.0), 0.05,
                                                   size=10), -0.1, 1.0))
                    windows += _window_rows(algo, fold, arm, seed, vals)
                    runs.append({"kind": "run", "algo": algo, "fold": fold, "seed": seed, "arm": arm,
                                 "config_id": None, "status": "ok", "tail_from": 4,
                                 TAIL: float(np.mean(vals[3:]))})
    _jsonl(root / "runs.jsonl", runs)
    _jsonl(root / "windows.jsonl", windows)
    return root


def test_write_selection_runs_every_variant_on_a_toy_out_dir(tmp_path):
    out = _toy_out_dir(tmp_path / "out")
    paths = S.write_selection(out, band=0.05, top_k=2)
    rep = out / "report" / "selection"
    assert rep / "selection_regret.csv" in paths and rep / "groups.csv" in paths and rep / "selection.md" in paths
    rows = pd.read_csv(rep / "selection_regret.csv")
    assert set(rows["variant"]) == set(S.VARIANTS)
    assert set(rows["target"]) == {"confstream", "frozen_prefix", "oracle_static"}
    assert set(rows["band"]) == {0.0, 0.05}
    assert set(rows["selector"]) == set(S.SELECTORS)
    assert set(rows["fold"]) == set(TOY_FOLDS)
    # the implementation variant is never a candidate
    assert not rows["picked"].astype(str).str.contains("DenStreamPatchedFast").any()
    assert (rows["n_algos"] == 3).all()
    groups = pd.read_csv(rep / "groups.csv")
    assert {"group", "block"} == set(groups["level"])
    md = (rep / "selection.md").read_text()
    assert "descriptive selection regret" in md
    assert "data-generator seeds" in md
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert (rep / "F_selection_regret.png").exists()


def test_write_selection_says_which_variant_could_not_run(tmp_path):
    out = _toy_out_dir(tmp_path / "out")
    (out / "windows.jsonl").unlink()
    S.write_selection(out, band=0.05, top_k=2)
    rows = pd.read_csv(out / "report" / "selection" / "selection_regret.csv")
    assert "prefix_suffix" not in set(rows["variant"])
    md = (out / "report" / "selection" / "selection.md").read_text()
    assert "prefix_suffix" in md and "did not run" in md


def test_load_windows_keeps_only_the_reruns_of_the_current_oracle_configuration(tmp_path):
    from conf_pipeline import report

    out = tmp_path / "out"
    runs, windows = [], []
    for cid in (3, 7):
        runs.append({"kind": "run", "algo": "A", "fold": "covtype", "seed": 2, "arm": "oracle_static",
                     "config_id": cid, "status": "ok", "tail_from": 1, TAIL: 0.1 * cid})
        windows += [{**w, "config_id": cid} for w in _window_rows("A", "covtype", "oracle_static", 2, [0.1 * cid] * 3)]
    runs.append({"kind": "run", "algo": "DenStreamPatchedFast", "fold": "covtype", "seed": 1, "arm": "default",
                 "config_id": None, "status": "ok", "tail_from": 1, TAIL: 0.5})
    windows += _window_rows("DenStreamPatchedFast", "covtype", "default", 1, [0.5] * 3)
    _jsonl(out / "runs.jsonl", runs)
    _jsonl(out / "windows.jsonl", windows)
    (out / "oracle_static.json").write_text(json.dumps({"A": {"covtype": {"config_id": 7}}}))
    win = S.load_windows(out, report.current_runs(out))
    assert set(win["config_id"]) == {7} and len(win) == 3
    assert S.load_windows(tmp_path / "nowhere", report.current_runs(out)) is None


def test_repeating_defaults_names_the_algorithms_whose_default_ignores_the_seed():
    vals = _values([("A", f, "default", s, 0.3) for f in ("covtype", "kdd99") for s in (1, 2, 3)]
                   + [("B", "covtype", "default", s, 0.1 * s) for s in (1, 2, 3)]
                   + [("B", "kdd99", "default", s, 0.2) for s in (1, 2, 3)]
                   + [("C", "covtype", "default", 1, 0.4)])
    assert S.repeating_defaults(vals) == ["A"]


# --- another selecting arm (E1: scaled_default) ---------------------------------------------------

def test_selector_labels_follow_the_selecting_arm_and_default_keeps_its_names():
    assert S.selectors("default") == S.SELECTORS
    assert S.selectors("scaled_default") == ("scaled_default", "random", "lobo_scaled_default", "lobo_target",
                                             "lobo_scaled_default_rank", "lobo_target_rank")


def test_scaled_default_as_the_selecting_arm_picks_and_trains_on_its_own_readings():
    sel, tgt = _toy_tables()
    # scaled_default reverses the defaults' order on covtype; for the global choice it favours C
    scaled = sel.copy()
    scaled.loc["covtype"] = {"A": 0.05, "B": 0.30, "C": 0.10}
    scaled["C"] = scaled["C"] + 1.0
    scaled.loc["covtype", "C"] = 0.0
    rows = S.fold_selection("same_data", scaled, {"confstream": tgt},
                            {"default": sel, "scaled_default": scaled, "confstream": tgt},
                            band=0.0, k=2, selecting_arm="scaled_default")
    assert set(rows["selector"]) == set(S.selectors("scaled_default"))
    cov = _row(rows, selector="scaled_default", fold="covtype")
    assert cov["picked"] == "B" and cov["regret_exp"] == pytest.approx(0.0)
    glob = _row(rows, selector="lobo_scaled_default", fold="covtype")
    assert glob["picked"] == "C"  # the scaled_default readings of the other blocks, not the defaults'
    assert glob["regret_exp"] == pytest.approx(0.9 - 0.2)


def test_an_algorithm_whose_scaled_default_is_its_default_reads_its_default_runs():
    vals = _values([(a, "covtype", "default", s, v) for a, v in (("HPStream", 0.4), ("DenStream", 0.1))
                    for s in (1, 2)]
                   + [("DenStream", "covtype", "scaled_default", s, 0.6) for s in (1, 2)])
    filled, names = S.fill_identical(vals, "scaled_default")
    assert names == ["HPStream"]  # no absolute-distance parameter: run.py skips its scaled_default
    w = S.wide(filled, "scaled_default")
    assert w.loc["covtype"].to_dict() == pytest.approx({"DenStream": 0.6, "HPStream": 0.4})
    # nothing to fill for the default arm itself, and a scaled run is never overwritten
    assert S.fill_identical(vals, "default")[1] == []
    again, names = S.fill_identical(filled, "scaled_default")
    assert names == [] and len(again) == len(filled)
    # an out dir that never ran the arm gets no arm made of fills alone
    only_default = vals[vals["arm"] == "default"]
    same, names = S.fill_identical(only_default, "scaled_default")
    assert names == [] and set(same["arm"]) == {"default"}


def _with_scaled_default(root):
    out = _toy_out_dir(root)
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    wins = [json.loads(line) for line in (out / "windows.jsonl").read_text().splitlines()]
    extra_r = [{**r, "arm": "scaled_default", TAIL: r[TAIL] + 0.05} for r in runs if r["arm"] == "default"]
    extra_w = [{**w, "arm": "scaled_default", "lag_ari_win": w["lag_ari_win"] + 0.05}
               for w in wins if w["arm"] == "default"]
    _jsonl(out / "runs.jsonl", runs + extra_r)
    _jsonl(out / "windows.jsonl", wins + extra_w)
    return out


def test_write_selection_with_another_selecting_arm_writes_beside_the_published_report(tmp_path):
    out = _with_scaled_default(tmp_path / "out")
    paths = S.write_selection(out, band=0.05, top_k=2, selecting_arm="scaled_default")
    rep = out / "report" / "selection_scaled_default"
    assert rep / "selection_regret.csv" in paths
    assert not (out / "report" / "selection").exists()
    rows = pd.read_csv(rep / "selection_regret.csv")
    assert set(rows["variant"]) == set(S.VARIANTS)
    assert set(rows["selector"]) == set(S.selectors("scaled_default"))
    md = (rep / "selection.md").read_text()
    assert "`scaled_default`" in md


def test_an_out_dir_without_the_selecting_arm_reports_it_undefined_with_the_reason(tmp_path):
    out = _toy_out_dir(tmp_path / "out")
    paths = S.write_selection(out, band=0.05, top_k=2, selecting_arm="scaled_default")
    rep = out / "report" / "selection_scaled_default"
    rows = pd.read_csv(rep / "selection_regret.csv")
    assert rows.empty and list(rows.columns) == S.ROW_COLS
    md = (rep / "selection.md").read_text()
    for v in S.VARIANTS:
        assert f"`{v}` did not run" in md
    assert "no `scaled_default` run to select on" in md
    assert rep / "selection.md" in paths and not (rep / "F_selection_regret.png").exists()
