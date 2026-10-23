"""conf_pipeline.robustness: the analyses of the existing logs (A1-A4) and the group aggregations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conf_pipeline import _common as C
from conf_pipeline import robustness as R

C_TAIL = "lag_ari_win_tail_mean"


# --- stream groups -----------------------------------------------------------------------------

def test_stream_group_maps_every_protocol_fold_to_one_of_three_groups():
    assert C.stream_group("covtype") == "real"
    assert C.stream_group("kdd99") == "real"
    assert C.stream_group("scale_drift") == "param_drift"
    assert C.stream_group("mixed_drift") == "param_drift"
    assert C.stream_group("evolving") == "other_synth"
    assert {C.stream_group(f) for f in C.FOLDS} == set(C.STREAM_GROUPS)


def test_stream_group_refuses_a_fold_nobody_assigned():
    with pytest.raises(ValueError, match="no stream group"):
        C.stream_group("my_new_stream")


def test_source_block_puts_every_insects_variant_in_one_block():
    insects = [f for f in C.FOLDS + C.X7_FOLDS if f.startswith("insects_")]
    assert len(insects) == 9
    assert {C.source_block(f) for f in insects} == {"insects"}


def test_source_block_keeps_variants_of_one_drift_design_together_and_other_folds_apart():
    assert {C.source_block(f) for f in ("scale_drift", "scale_drift_x1_5", "scale_drift_x8",
                                        "scale_drift_gradual")} == {"scale_drift"}
    assert C.source_block("random_drift") == "random_drift"
    for f in ("covtype", "kdd99", "shuttle", "electricity", "rialto", "sensor", "density_drift", "evolving"):
        assert C.source_block(f) == f
    # 18 protocol folds, 7 of them INSECTS: 12 blocks
    assert len({C.source_block(f) for f in C.FOLDS}) == 12


def test_source_block_refuses_a_fold_nobody_assigned():
    with pytest.raises(ValueError, match="no stream group"):
        C.source_block("my_new_stream")


# --- group aggregations ------------------------------------------------------------------------

def _cells(rows):
    """(algo, fold, tunable, default, confstream, frozen_prefix, oracle_static) tuples."""
    cols = ["algo", "fold", "tunable", "default", "confstream", "frozen_prefix", "oracle_static"]
    return pd.DataFrame(rows, columns=cols)


def test_capture_ratio_is_a_ratio_of_sums_over_the_tunable_pairs_of_a_group():
    cells = _cells([
        ("A", "covtype", True, 0.0, 0.5, 0.2, 1.0),
        ("B", "covtype", True, 0.0, 0.1, 0.0, 0.1),   # ratio 1 alone, but a small denominator
        ("A", "kdd99", False, 0.0, 0.9, 0.9, 0.01),   # not tunable: left out
        ("A", "k_drift", True, 0.2, 0.9, 0.5, 0.6),
    ])
    out = R.capture_by_group(cells).set_index(["group", "arm"])
    assert out.loc[("real", "confstream"), "capture"] == pytest.approx(0.6 / 1.1)
    assert out.loc[("real", "confstream"), "n_pairs"] == 2
    assert out.loc[("param_drift", "frozen_prefix"), "capture"] == pytest.approx(0.3 / 0.4)
    assert out.loc[("all", "confstream"), "capture"] == pytest.approx((0.6 + 0.7) / (1.1 + 0.4))
    assert pd.isna(out.loc[("other_synth", "confstream"), "capture"])
    assert out.loc[("other_synth", "confstream"), "n_pairs"] == 0


def test_losses_split_the_gap_to_the_reference_and_average_over_pairs():
    loss = pd.DataFrame({
        "algo": ["A", "B", "A"], "fold": ["covtype", "kdd99", "k_drift"],
        "a_lag": [0.1, 0.2, 0.9], "a_star_lag": [0.15, 0.3, 0.92], "A_static": [0.2, 0.3, 0.8],
    })
    out = R.losses_by_group(loss).set_index("group")
    assert out.loc["real", "selection_loss"] == pytest.approx((0.05 + 0.1) / 2)
    assert out.loc["real", "search_loss"] == pytest.approx((0.05 + 0.0) / 2)
    assert out.loc["param_drift", "search_loss"] == pytest.approx(-0.12)
    row = out.loc["all"]
    assert row["selection_loss"] + row["search_loss"] == pytest.approx(row["A_static"] - row["a_lag"])


FULL = C.PIPELINE_ROOT / "out" / "full"


@pytest.mark.slow
@pytest.mark.skipif(not (FULL / "runs.jsonl").exists(), reason="the full run's logs are not here")
def test_group_aggregations_reproduce_the_published_full_run_numbers():
    inputs = R.load_inputs(FULL)
    cap = R.capture_by_group(R.cells_table(inputs)).set_index(["group", "arm"])["capture"]
    # the values result/otb_full/report/make_dat.py printed for the report, to three decimals
    published = {("real", "confstream"): 0.483, ("param_drift", "confstream"): 1.291,
                 ("other_synth", "confstream"): 0.943, ("all", "confstream"): 0.963,
                 ("real", "frozen_prefix"): 0.335, ("param_drift", "frozen_prefix"): 0.738,
                 ("other_synth", "frozen_prefix"): 0.727, ("all", "frozen_prefix"): 0.626}
    for key, value in published.items():
        assert cap[key] == pytest.approx(value, abs=0.0005), key
    loss = R.losses_by_group(R.loss_cells(inputs)).set_index("group")
    for group, (sel, search) in {"real": (0.043, 0.009), "param_drift": (0.003, -0.161),
                                 "other_synth": (0.049, -0.023)}.items():
        assert loss.loc[group, "selection_loss"] == pytest.approx(sel, abs=0.0005), group
        assert loss.loc[group, "search_loss"] == pytest.approx(search, abs=0.0005), group


# --- A3: bootstrap and corrected tests ---------------------------------------------------------

def _seed_rows(pairs):
    """pairs: {(algo, fold): {arm: [seed values]}} -> the ``Inputs.seeds`` frame."""
    rows = []
    for (algo, fold), arms in pairs.items():
        for arm, values in arms.items():
            first = 2 if arm == "oracle_static" else 1  # the reference is rerun on seeds 2-4
            rows += [{"algo": algo, "fold": fold, "arm": arm, "seed": first + i, "value": v}
                     for i, v in enumerate(values)]
    return pd.DataFrame(rows)


def _paired_pairs():
    # on every seed, confstream sits 0.1 above default and the reference 0.2 above it, while
    # default itself moves a lot between seeds: only a resample that keeps the arms paired
    # by seed reads a capture of exactly 0.5 every time
    pairs = {}
    for i, (algo, fold) in enumerate([("A", "covtype"), ("B", "covtype"), ("A", "kdd99"),
                                      ("A", "k_drift"), ("A", "evolving")]):
        d = np.array([0.0, 0.3, 0.6]) + 0.01 * i
        pairs[(algo, fold)] = {"default": d, "confstream": d + 0.1, "frozen_prefix": d + 0.05,
                               "oracle_static": d + 0.2}
    return pairs


def _loss_rows(pairs):
    rows = []
    for (algo, fold), arms in pairs.items():
        rows += [{"algo": algo, "fold": fold, "seed": s + 1, "a_lag": c, "a_star_lag": c + 0.05}
                 for s, c in enumerate(arms["confstream"])]
    return pd.DataFrame(rows)


def test_bootstrap_keeps_the_arms_paired_by_seed():
    pairs = _paired_pairs()
    tunable = {k: True for k in pairs}
    out = R.bootstrap(_seed_rows(pairs), _loss_rows(pairs), tunable, n_boot=200, seed=0)
    cap = out[(out.statistic == "capture") & (out.arm == "confstream")].set_index("group")
    for group in ("real", "param_drift", "other_synth", "all"):
        assert cap.loc[group, "point"] == pytest.approx(0.5)
        assert cap.loc[group, "low"] == pytest.approx(0.5) and cap.loc[group, "high"] == pytest.approx(0.5)
    sel = out[out.statistic == "selection_loss"].set_index("group")
    assert sel.loc["all", "low"] == pytest.approx(0.05) and sel.loc["all", "high"] == pytest.approx(0.05)


def test_bootstrap_is_reproducible_and_its_interval_brackets_the_point():
    rng = np.random.default_rng(3)
    pairs = {}
    for algo in ("A", "B", "C"):
        for fold in ("covtype", "kdd99", "shuttle", "k_drift", "scale_drift", "evolving"):
            d = rng.uniform(0, 0.3, 3)
            pairs[(algo, fold)] = {"default": d, "confstream": d + rng.uniform(0, 0.4, 3),
                                   "frozen_prefix": d + rng.uniform(0, 0.3, 3),
                                   "oracle_static": d + 0.4 + rng.uniform(0, 0.1, 3)}
    tunable = {k: True for k in pairs}
    a = R.bootstrap(_seed_rows(pairs), _loss_rows(pairs), tunable, n_boot=300, seed=7)
    b = R.bootstrap(_seed_rows(pairs), _loss_rows(pairs), tunable, n_boot=300, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert (a["low"] <= a["point"] + 1e-12).all() and (a["point"] <= a["high"] + 1e-12).all()
    assert (a["high"] - a["low"] > 0).any()
    assert set(a["n_used"]) == {300}


def test_holm_adjusts_step_down_and_leaves_a_missing_p_missing():
    assert R.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert R.holm([0.01, None]) == [pytest.approx(0.01), None]
    assert R.holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_rank_biserial_from_signed_ranks_with_zeros_dropped():
    assert R.rank_biserial([1, 2, -3, 4]) == pytest.approx((7 - 3) / 10)
    assert R.rank_biserial([0, 1]) == pytest.approx(1.0)
    assert np.isnan(R.rank_biserial([0, 0]))


def test_paired_tests_compare_confstream_per_algorithm_with_holm_over_the_family():
    folds = ["covtype", "kdd99", "shuttle", "k_drift", "scale_drift", "evolving"]
    rows = []
    for j, fold in enumerate(folds):
        rows.append(("A", fold, True, 0.1, 0.5 + 0.01 * j, 0.5, 0.6))   # online beats default everywhere
        rows.append(("B", fold, True, 0.1, 0.1, 0.1, 0.3))              # online equals default: all zeros
    out = R.paired_tests(_cells(rows)).set_index(["algo", "comparison"])
    a = out.loc[("A", "confstream vs default")]
    assert a["n"] == 6 and a["n_zero"] == 0
    assert a["p"] == pytest.approx(2 / 64) and a["rank_biserial"] == pytest.approx(1.0)
    assert a["median_diff"] == pytest.approx(0.425)
    b = out.loc[("B", "confstream vs default")]
    assert b["n_zero"] == 6 and pd.isna(b["p"]) and pd.isna(b["p_holm"])
    assert len(out) == 6
    valid = out["p"].dropna()
    assert (out.loc[valid.index, "p_holm"] >= valid - 1e-12).all()


# --- A2: reference selected on even windows ---------------------------------------------------

def _sweep_windows(values, algo="A", fold="covtype"):
    """values: {config_id: {window: lag_ari_win}}."""
    return pd.DataFrame([{"algo": algo, "fold": fold, "seed": 1, "config_id": cid, "window": w,
                          "lag_ari_win": v} for cid, by_w in values.items() for w, v in by_w.items()])


def test_interleaved_reference_selects_on_even_tail_windows_and_reports_on_odd_ones():
    sw = _sweep_windows({
        1: {20: 0.9, 21: 0.1, 22: 0.9, 23: 0.1},               # lucky on even windows
        2: {2: 1.0, 20: 0.6, 21: 0.6, 22: 0.6, 23: 0.6},       # window 2 is before the tail
        3: {20: None, 21: 0.8, 22: None, 23: 0.8},             # no even reading: not a candidate
    })
    ref = R.interleaved_reference(sw, tail_from=20).iloc[0]
    assert ref["config_id"] == 1
    assert ref["select_value"] == pytest.approx(0.9) and ref["report_value"] == pytest.approx(0.1)
    assert ref["n_candidates"] == 2


def test_interleaved_reference_can_leave_out_the_tuners_incumbents():
    inc = C.INCUMBENT_ID_BASE + 5
    sw = _sweep_windows({0: {20: 0.3, 21: 0.3}, 4: {20: 0.5, 21: 0.4}, inc: {20: 0.9, 21: 0.9}})
    assert R.interleaved_reference(sw, tail_from=20).iloc[0]["config_id"] == inc
    ref = R.interleaved_reference(sw, tail_from=20, incumbents=False).iloc[0]
    assert ref["config_id"] == 4 and ref["config_source"] == "sobol"


def test_parity_values_read_each_run_on_the_odd_tail_windows_only():
    win = pd.DataFrame([
        {"algo": "A", "fold": "covtype", "arm": "default", "seed": 1, "config_id": None,
         "window": w, "lag_ari_win": v}
        for w, v in {19: 0.9, 20: 0.2, 21: 0.4, 22: 0.2, 23: None, 25: 0.6}.items()
    ])
    out = R.parity_values(win, tail_from={}, parity=1).iloc[0]
    assert (out["algo"], out["arm"], out["seed"]) == ("A", "default", 1)
    assert out["value"] == pytest.approx((0.4 + 0.6) / 2)


# --- A1: three measures ------------------------------------------------------------------------

def test_measure_values_are_tail_means_per_run_then_seed_means():
    win = pd.DataFrame([
        {"algo": "A", "fold": "covtype", "arm": "confstream", "seed": s, "window": w,
         "lag_ari_win": 0.1 * s, "lag_nmi_win": 0.2 * s, "lag_purity_win": 0.5, "lag_n_pred_labels": 3}
        for s in (1, 2) for w in (19, 20, 21)
    ])
    win.loc[win["window"] == 19, ["lag_ari_win", "lag_nmi_win"]] = 9.0   # before the tail
    out = R.measure_values(win, tail_from={}).iloc[0]
    assert out["ari"] == pytest.approx(0.15) and out["nmi"] == pytest.approx(0.3)
    assert out["purity"] == pytest.approx(0.5) and out["n_pred_labels"] == pytest.approx(3)


def test_measure_agreement_reads_kendall_tau_between_the_rankings_of_two_measures():
    rows = []
    for fold in ("covtype", "k_drift"):
        for algo, (ari, nmi, purity) in {"A": (0.9, 0.8, 0.1), "B": (0.5, 0.5, 0.5), "C": (0.1, 0.2, 0.9)}.items():
            rows.append({"algo": algo, "fold": fold, "arm": "confstream", "ari": ari, "nmi": nmi,
                         "purity": purity})
    per_fold, by_group = R.measure_agreement(pd.DataFrame(rows), arms=("confstream",))
    t = by_group.set_index(["group", "pair"])["tau"]
    assert t[("all", "ari-nmi")] == pytest.approx(1.0)
    assert t[("all", "ari-purity")] == pytest.approx(-1.0)
    assert t[("real", "ari-nmi")] == pytest.approx(1.0)
    assert set(per_fold["fold"]) == {"covtype", "k_drift"}


# --- A4: single-class windows ------------------------------------------------------------------

def test_single_class_table_counts_the_windows_the_rule_removes_per_fold():
    win = pd.DataFrame([
        {"algo": "A", "fold": "kdd99", "arm": "default", "seed": 1, "window": w,
         "external_defined": w % 3 != 0}
        for w in range(1, 31)
    ])
    out = R.single_class_table(win, tail_from={}).set_index("fold").loc["kdd99"]
    assert out["n_windows"] == 30 and out["n_single_class"] == 10
    assert out["n_eval_tail"] == sum(1 for w in range(20, 31) if w % 3 != 0)


def test_rankings_without_kdd99_drop_the_fold_from_friedman_and_kendall():
    rows = []
    for fold in ("covtype", "shuttle", "kdd99"):
        for algo, base in {"A": 0.9, "B": 0.5, "C": 0.1}.items():
            flip = fold == "kdd99"
            rows.append({"algo": algo, "fold": fold, "arm": "default", C_TAIL: 1 - base if flip else base})
            rows.append({"algo": algo, "fold": fold, "arm": "confstream", C_TAIL: base})
    fried, kendall = R.without_folds(pd.DataFrame(rows), arms=("default", "confstream"), drop=("kdd99",))
    f = fried.set_index(["arm", "folds"])
    assert f.loc[("default", "all"), "n_folds"] == 3 and f.loc[("default", "without kdd99"), "n_folds"] == 2
    k = kendall.set_index(["arm_a", "arm_b"]).loc[("default", "confstream")]
    assert k["tau_all"] == pytest.approx((1 + 1 - 1) / 3) and k["tau_without"] == pytest.approx(1.0)


# --- usable streams ----------------------------------------------------------------------------

def test_usable_streams_need_enough_windows_and_reachable_labels():
    single = pd.DataFrame({"fold": ["covtype", "kdd99", "shuttle"], "n_windows": [99, 99, 57],
                           "n_single_class": [0, 71, 1], "n_eval_tail": [79, 23, 37]})
    reach = pd.DataFrame({"fold": ["covtype", "kdd99", "shuttle"], "reach_raw": [0.1, 0.9, 0.6],
                          "reach_std": [0.4, 0.9, 0.7]})
    out = R.usable_streams(single, reach=reach, agreement=None, best=None,
                           min_eval_windows=25, min_reach_ari=0.3).set_index("fold")
    assert not out.loc["covtype", "labels_usable"]       # labels out of reach on raw features
    assert not out.loc["kdd99", "labels_usable"]         # too few windows left
    assert out.loc["shuttle", "labels_usable"]
    none = R.usable_streams(single, reach=None, agreement=None, best=None,
                            min_eval_windows=25, min_reach_ari=0.3)
    assert none["labels_usable"].isna().all()


# --- end to end on a toy out dir ---------------------------------------------------------------

TOY_FOLDS = ("covtype", "kdd99", "k_drift", "evolving")
TOY_QUALITY = {"BIRCH": 0.3, "DenStream": 0.5}
TOY_OFFSET = {"default": -0.2, "confstream": 0.0, "frozen_prefix": -0.1, "oracle_static": 0.1}


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _toy_out_dir(root):
    """Constant readings per job: default 0.2 below confstream, the reference 0.1 above it;
    on kdd99 every fourth window holds one class. The sweep's config 7 (the reference) reads
    0.02 higher on even windows than on odd ones."""
    runs, windows, sweep, sweep_win, oracle = [], [], [], [], {}
    for algo, q in TOY_QUALITY.items():
        for fold in TOY_FOLDS:
            oracle.setdefault(algo, {})[fold] = {"config_id": 7}
            for arm, off in TOY_OFFSET.items():
                seeds = (2, 3, 4) if arm == "oracle_static" else (1, 2, 3)
                for seed in seeds:
                    v = q + off + 0.01 * (seed - seeds[1])
                    cid = 7 if arm == "oracle_static" else None
                    runs.append({"kind": "run", "algo": algo, "fold": fold, "seed": seed, "arm": arm,
                                 "config_id": cid, "status": "ok", "tail_from": 20,
                                 "lag_ari_win_tail_mean": v, "lag_ari_win_mean": v})
                    members = []
                    for w in range(1, 26):
                        single = fold == "kdd99" and w % 4 == 0
                        lag = None if single or w == 1 else v
                        windows.append({
                            "kind": "window", "algo": algo, "fold": fold, "seed": seed, "arm": arm,
                            "config_id": cid, "window": w, "incumbent_id": 0, "external_defined": not single,
                            "ari_win": None if single else v + 0.02, "lag_ari_win": lag,
                            "lag_nmi_win": None if lag is None else 0.9 * v,
                            "lag_purity_win": None if lag is None else 0.5 + v / 2, "lag_n_pred_labels": 3})
                        if arm == "confstream" and not single:
                            members += [{"algo": algo, "fold": fold, "seed": seed, "arm": arm, "config_id": None,
                                         "member_id": m, "window": w, "is_test": False, "feasible": True,
                                         "silhouette_defined": True, "score": s, "ari_win": v + a}
                                        for m, s, a in ((0, 0.5, 0.02), (1, 0.4, 0.05))]
                    if members:
                        _jsonl(root / "members" / f"{algo}__{fold}__seed{seed}__{arm}.jsonl", members)
            for cid, v in ((0, q - 0.2), (7, q + 0.1)):
                sweep.append({"kind": "run", "algo": algo, "fold": fold, "seed": 1, "arm": "sweep",
                              "config_id": cid, "status": "ok", "tail_from": 20,
                              "lag_ari_win_tail_mean": v, "lag_ari_win_mean": v})
                sweep_win += [{"algo": algo, "fold": fold, "seed": 1, "arm": "sweep", "config_id": cid,
                               "window": w, "lag_ari_win": v + (0.02 if cid == 7 and w % 2 == 0 else
                                                                 -0.02 if cid == 7 else 0.0)}
                              for w in range(2, 26)]
    _jsonl(root / "runs.jsonl", runs)
    _jsonl(root / "windows.jsonl", windows)
    _jsonl(root / "sweep.jsonl", sweep)
    _jsonl(root / "sweep_windows.jsonl", sweep_win)
    (root / "oracle_static.json").write_text(json.dumps(oracle))
    return root


def test_write_robustness_runs_every_analysis_on_a_toy_out_dir(tmp_path):
    out = _toy_out_dir(tmp_path / "out")
    rep = tmp_path / "rep"
    paths = R.write_robustness(out, n_boot=50, rep_dir=rep)
    md = (rep / "robustness.md").read_text()
    for heading in ("A1", "A2", "A3", "A4", "A5", "Usable streams"):
        assert heading in md
    assert "label_reach.jsonl" in md                     # A5 not run yet: said so, no numbers
    assert all(p.exists() for p in paths)
    cap = pd.read_csv(rep / "robustness" / "capture_losses.csv")
    real = cap[(cap.group == "real") & (cap.statistic == "capture") & (cap.arm == "confstream")].iloc[0]
    assert real["point"] == pytest.approx(0.2 / 0.3)
    a2 = pd.read_csv(rep / "robustness" / "a2_reference.csv")
    inter = a2[a2.construction == "interleaved"]
    assert set(inter["config_id"]) == {7}
    assert inter["report_value"].to_numpy() == pytest.approx((inter["select_value"] - 0.04).to_numpy())
    single = pd.read_csv(rep / "robustness" / "a4_single_class.csv").set_index("fold")
    assert single.loc["kdd99", "n_single_class"] == 6 and single.loc["covtype", "n_single_class"] == 0
    usable = pd.read_csv(rep / "robustness" / "usable_streams.csv")
    assert len(usable) == 4 and usable["labels_usable"].isna().all()


def _reference_dir(root, value=0.35):
    """Another out dir holding a reference for BIRCH only, picked on seeds 1-3: a rerun on
    seed 2 (a picking seed, 0.9) and two on seeds 4 and 5 (``value``)."""
    runs, oracle = [], {"BIRCH": {}}
    for fold in TOY_FOLDS:
        oracle["BIRCH"][fold] = {"config_id": 9, "select_seeds": [1, 2, 3], "n_ok": 40,
                                 "select_metric": C_TAIL, C_TAIL: 0.36}
        for seed, v in ((2, 0.9), (4, value), (5, value)):
            runs.append({"kind": "run", "algo": "BIRCH", "fold": fold, "seed": seed, "arm": "oracle_static",
                         "config_id": 9, "status": "ok", "tail_from": 20, C_TAIL: v, "lag_ari_win_mean": v})
    _jsonl(root / "runs.jsonl", runs)
    (root / "oracle_static.json").write_text(json.dumps(oracle))
    return root


def test_load_reference_never_reads_a_reference_on_a_seed_that_picked_it(tmp_path):
    ref = R.load_reference(_reference_dir(tmp_path)).set_index(["algo", "fold"])
    row = ref.loc[("BIRCH", "covtype")]
    assert row["report_value"] == pytest.approx(0.35) and row["report_seeds"] == "4,5"
    assert row["select_seeds"] == "1,2,3" and row["select_value"] == pytest.approx(0.36)
    assert row["config_id"] == 9 and row["n_candidates"] == 40


def test_a_reference_dir_adds_an_a2_construction_and_keeps_the_published_one_where_it_has_none(tmp_path):
    out = _toy_out_dir(tmp_path / "out")
    rep = tmp_path / "rep"
    R.write_robustness(out, n_boot=20, rep_dir=rep, reference_dirs={"multi": _reference_dir(tmp_path / "ref")})
    a2 = pd.read_csv(rep / "robustness" / "a2_reference.csv")
    multi = a2[a2.construction == "multi"].set_index(["algo", "fold"])
    assert multi.loc[("BIRCH", "covtype"), "reference_from"] == "multi"
    assert multi.loc[("BIRCH", "covtype"), "report_value"] == pytest.approx(0.35)
    assert multi.loc[("DenStream", "covtype"), "reference_from"] == "current"
    assert multi.loc[("DenStream", "covtype"), "report_value"] == pytest.approx(0.6)
    cur = a2[a2.construction == "current"].set_index(["algo", "fold"])
    assert cur.loc[("BIRCH", "covtype"), "report_seeds"] == "2,3,4"
    cap = pd.read_csv(rep / "robustness" / "a2_capture.csv")
    real = cap[(cap.construction == "multi") & (cap.group == "real") & (cap.arm == "confstream")].iloc[0]
    # confstream sits 0.2 above default everywhere; BIRCH's reference drops from 0.4 to 0.35
    assert real["capture"] == pytest.approx(0.8 / (2 * 0.25 + 2 * 0.3))
    assert "`multi` picks another configuration than `current`" in (rep / "robustness.md").read_text()


# --- P1.2: the statistical unit (folds or source blocks) ------------------------------------

INSECTS3 = ("insects_abrupt_balanced", "insects_gradual_balanced", "insects_incremental_balanced")


def test_unit_index_groups_the_insects_variants_into_one_block_and_keeps_fold_order():
    folds = ["covtype", *INSECTS3, "k_drift", "evolving"]
    by_fold = R.unit_index(folds, "fold")
    assert [(g, u) for g, u, _ in by_fold] == [
        ("real", "covtype"), ("real", INSECTS3[0]), ("real", INSECTS3[1]), ("real", INSECTS3[2]),
        ("param_drift", "k_drift"), ("other_synth", "evolving")]
    by_block = R.unit_index(folds, "block")
    assert [(g, u) for g, u, _ in by_block] == [
        ("real", "covtype"), ("real", "insects"), ("param_drift", "k_drift"), ("other_synth", "evolving")]
    assert list(by_block[1][2]) == [1, 2, 3]
    with pytest.raises(ValueError, match="unit"):
        R.unit_index(folds, "seed")


def test_a_block_draw_brings_every_fold_of_the_block_and_a_fold_draw_matches_the_old_bootstrap():
    folds = ["covtype", *INSECTS3, "k_drift", "density_drift", "evolving"]
    rng = np.random.default_rng(0)
    for _ in range(50):
        pos, ids = R.draw_units(folds, "block", rng)
        drawn = [folds[p] for p in pos]
        n_ins = sum(f.startswith("insects_") for f in drawn)
        assert n_ins % 3 == 0
        for u in set(ids):  # one drawn instance of a block is all its folds, in order
            members = [folds[p] for p, i in zip(pos, ids) if i == u]
            assert members in (["covtype"], list(INSECTS3), ["k_drift"], ["density_drift"], ["evolving"])
    # the fold unit consumes the generator exactly as bootstrap() always has
    groups = np.array([C.stream_group(f) for f in folds])
    a, b = np.random.default_rng(4), np.random.default_rng(4)
    old = np.concatenate([a.choice(idx, size=len(idx), replace=True)
                          for idx in (np.flatnonzero(groups == g) for g in C.STREAM_GROUPS) if len(idx)])
    new, ids = R.draw_units(folds, "fold", b)
    assert list(new) == list(old) and len(set(ids)) == len(folds)


def test_bootstrap_defaults_are_unchanged_by_the_new_options():
    pairs = _paired_pairs()
    tunable = {k: True for k in pairs}
    a = R.bootstrap(_seed_rows(pairs), _loss_rows(pairs), tunable, n_boot=100, seed=2)
    b = R.bootstrap(_seed_rows(pairs), _loss_rows(pairs), tunable, n_boot=100, seed=2, unit="fold",
                    pairing="index")
    pd.testing.assert_frame_equal(a, b)


def _one_block_real_pairs():
    """Every real fold an INSECTS variant (one block), and real capture differing by fold."""
    pairs = {}
    for j, fold in enumerate(INSECTS3):
        d = np.array([0.0, 0.0, 0.0])
        pairs[("A", fold)] = {"default": d, "confstream": d + 0.1 * (j + 1), "frozen_prefix": d + 0.05,
                              "oracle_static": d + 0.5}
    pairs[("A", "k_drift")] = {"default": np.zeros(3), "confstream": np.full(3, 0.3),
                               "frozen_prefix": np.full(3, 0.1), "oracle_static": np.full(3, 0.6)}
    return pairs


def test_a_block_bootstrap_of_one_block_has_no_spread_where_the_fold_bootstrap_has_some():
    pairs = _one_block_real_pairs()
    tunable = {k: True for k in pairs}
    seeds, losses = _seed_rows(pairs), _loss_rows(pairs)
    fold = R.bootstrap(seeds, losses, tunable, n_boot=200, seed=0).set_index(["statistic", "group", "arm"])
    block = R.bootstrap(seeds, losses, tunable, n_boot=200, seed=0, unit="block").set_index(
        ["statistic", "group", "arm"])
    key = ("capture", "real", "confstream")
    assert fold.loc[key, "point"] == pytest.approx(block.loc[key, "point"])  # same point definition
    assert fold.loc[key, "high"] - fold.loc[key, "low"] > 0.05
    assert block.loc[key, "low"] == pytest.approx(block.loc[key, "high"])
    with pytest.raises(ValueError, match="unit"):
        R.bootstrap(seeds, losses, tunable, n_boot=5, unit="seed")
    with pytest.raises(ValueError, match="pairing"):
        R.bootstrap(seeds, losses, tunable, n_boot=5, pairing="any")


def test_seed_pairing_never_pairs_the_reference_reruns_with_the_protocol_seeds_by_index():
    # _paired_pairs: the reference sits 0.2 above default seed by seed *index* (its reruns are
    # seeds 2-4, the protocol's 1-3), so only index pairing reads a capture of exactly 0.5
    pairs = _paired_pairs()
    tunable = {k: True for k in pairs}
    out = R.bootstrap(_seed_rows(pairs), _loss_rows(pairs), tunable, n_boot=300, seed=0,
                      pairing="seed").set_index(["statistic", "group", "arm"])
    cap = out.loc[("capture", "all", "confstream")]
    assert cap["point"] == pytest.approx(0.5)
    assert cap["high"] - cap["low"] > 0.1
    # default and confstream share seeds 1-3 and stay paired: the selection loss is still exact
    sel = out.loc[("selection_loss", "all", "confstream")]
    assert sel["low"] == pytest.approx(0.05) and sel["high"] == pytest.approx(0.05)


# --- P1.2: the checks of result/otb_full/report/make_robustness_extra.py, as API -------------

ALGOS4 = ["A", "B", "C", "D"]


def _wide(rows, folds):
    """rows: one list of four values (algos A-D) per fold."""
    return pd.DataFrame(rows, index=pd.Index(folds, name="fold"), columns=pd.Index(ALGOS4, name="algo"))


def test_tau_by_fold_needs_three_algorithms_and_ranks_ties_by_their_average():
    folds = ["covtype", "kdd99", "k_drift"]
    x = _wide([[0.9, 0.5, 0.1, 0.3], [0.9, np.nan, np.nan, 0.1], [0.4, 0.4, 0.1, 0.2]], folds)
    y = _wide([[0.8, 0.6, 0.2, 0.3], [0.1, 0.5, 0.5, 0.9], [0.4, 0.3, 0.1, 0.2]], folds)
    t = R.tau_by_fold(x, y)
    assert list(t.index) == ["covtype", "k_drift"]            # kdd99: two algorithms only
    assert t["covtype"] == pytest.approx(1.0)
    from scipy import stats
    assert t["k_drift"] == pytest.approx(stats.kendalltau([1.5, 1.5, 4, 3], [1, 2, 4, 3]).statistic)
    keep = _wide([[True, True, True, False], [True] * 4, [True] * 4], folds)
    assert R.tau_by_fold(x, y, keep=keep)["covtype"] == pytest.approx(1.0)
    keep.loc["covtype", "B"] = False                           # two algorithms left on covtype
    assert "covtype" not in R.tau_by_fold(x, y, keep=keep).index


def test_boot_mean_resamples_within_groups_in_alphabetical_group_order_like_the_result_script():
    s = pd.Series([0.1, 0.5, 0.9, -0.2], index=["covtype", "kdd99", "k_drift", "evolving"])
    m, lo, hi = R.boot_mean(s, np.random.default_rng(3), n_boot=50)
    rng = np.random.default_rng(3)
    g = np.array([C.stream_group(f) for f in s.index])
    reps = [s.to_numpy()[np.concatenate([rng.choice(np.where(g == k)[0], size=(g == k).sum())
                                         for k in np.unique(g)])].mean() for _ in range(50)]
    assert m == pytest.approx(s.mean())
    assert (lo, hi) == pytest.approx(tuple(np.percentile(reps, [2.5, 97.5])))
    m2, lo2, hi2 = R.boot_mean(s, np.random.default_rng(3), n_boot=50, by_group=False)
    rng = np.random.default_rng(3)
    reps = [s.to_numpy()[rng.choice(4, size=4)].mean() for _ in range(50)]
    assert (lo2, hi2) == pytest.approx(tuple(np.percentile(reps, [2.5, 97.5])))
    empty = R.boot_mean(pd.Series(dtype=float), np.random.default_rng(0), n_boot=10)
    assert all(np.isnan(v) for v in empty)


def _arm_wides():
    folds = ["covtype", "kdd99", *INSECTS3, "k_drift", "evolving"]
    base = _wide([[0.9, 0.6, 0.3, 0.1]] * len(folds), folds)
    flipped = _wide([[0.1, 0.3, 0.6, 0.9]] * len(folds), folds)
    return folds, {"default": flipped, "confstream": base, "frozen_prefix": base.copy(), "oracle_static": base}


def test_arm_tau_table_reads_every_pair_on_all_folds_and_on_the_real_ones():
    _, wide = _arm_wides()
    out = R.arm_tau_table(wide, np.random.default_rng(0), n_boot=20).set_index(["arm_a", "arm_b", "scope"])
    assert out.loc[("default", "confstream", "all"), "tau"] == pytest.approx(-1.0)
    assert out.loc[("default", "confstream", "real"), "n_folds"] == 5
    assert out.loc[("confstream", "oracle_static", "all"), "low"] == pytest.approx(1.0)
    assert len(out) == 2 * len(R.TAU_PAIRS)


def test_block_rankings_average_the_insects_variants_into_one_block_before_ranking():
    folds, wide = _arm_wides()
    # only the INSECTS variants disagree with the default ranking, and one of them by a lot
    wide = {k: v.copy() for k, v in wide.items()}
    wide["confstream"].loc[INSECTS3[0]] = [0.0, 0.1, 0.2, 3.0]
    tau, ranks = R.block_rankings(wide)
    t = tau.set_index(["arm_a", "arm_b"])
    assert t.loc[("confstream", "frozen_prefix"), "n_blocks"] == 5     # covtype, insects, kdd99, k_drift, evolving
    # insects block mean of confstream: A 0.6, B 0.433, C 0.267, D 1.067 -> D first
    expected = (4 * 1.0 + R.tau_by_fold(
        _wide([[0.6, 1.3 / 3, 0.8 / 3, 3.2 / 3]], ["insects"]),
        _wide([[0.9, 0.6, 0.3, 0.1]], ["insects"]))["insects"]) / 5
    assert t.loc[("confstream", "frozen_prefix"), "tau"] == pytest.approx(expected)
    r = ranks.set_index(["arm", "scope"])
    assert r.loc[("default", "all"), "n_blocks"] == 5 and r.loc[("default", "real"), "n_blocks"] == 3
    assert r.loc[("default", "all"), "rank_D"] == pytest.approx(1.0)


def test_weak_default_filters_keep_only_the_pairs_whose_default_passes():
    folds = ["covtype", "kdd99", "shuttle"]
    default = _wide([[0.5, 0.4, 0.3, 0.0], [0.5, 0.005, 0.3, 0.2], [0.2, 0.1, 0.3, 0.4]], folds)
    online = _wide([[0.1, 0.2, 0.3, 0.9], [0.5, 0.9, 0.3, 0.2], [0.2, 0.1, 0.3, 0.4]], folds)
    undefined = _wide([[0.0, 0.0, 0.0, 0.9], [0.0] * 4, [0.0] * 4], folds)
    wide = {"default": default, "confstream": online, "oracle_static": online}
    out = R.weak_default_tau(wide, undefined, np.random.default_rng(0), n_boot=20)
    row = out.set_index(["filter", "arm"]).loc[("default ARI >= 0.01", "confstream")]
    # covtype keeps A-C (D reads 0.0): 1 tau of -1; kdd99 keeps A, C, D (B reads 0.005): tau 1
    assert row["n_folds"] == 3 and row["tau"] == pytest.approx((-1 + 1 + 1) / 3)
    assert row["algos_per_fold"] == pytest.approx((3 + 3 + 4) / 3)
    broken = out.set_index(["filter", "arm"]).loc[("default not broken (undefined share <= 0.5)", "confstream")]
    assert broken["algos_per_fold"] == pytest.approx((3 + 4 + 4) / 3)
    assert [f for f, *_ in R.WEAK_DEFAULT_FILTERS] == [
        "default not broken (undefined share <= 0.5)", "default ARI >= 0.01", "default ARI >= 0.05"]


def test_sobol_rankings_compare_every_sweep_point_with_the_arms():
    folds = ["covtype", "kdd99", "k_drift"]
    base = _wide([[0.9, 0.6, 0.3, 0.1]] * 3, folds)
    wide = {"default": base, "confstream": base, "oracle_static": base}
    rows = []
    for fold in folds:
        for algo, v in zip(ALGOS4, (0.9, 0.6, 0.3, 0.1)):
            rows.append({"algo": algo, "fold": fold, "config_id": 0, "status": "ok", C_TAIL: v})
            rows.append({"algo": algo, "fold": fold, "config_id": 1, "status": "ok", C_TAIL: v})
            rows.append({"algo": algo, "fold": fold, "config_id": 2, "status": "ok", C_TAIL: -v})
            rows.append({"algo": algo, "fold": fold, "config_id": 3, "status": "ok", C_TAIL: 2 * v})
            rows.append({"algo": algo, "fold": fold, "config_id": R.C.INCUMBENT_ID_BASE + 3, "status": "ok",
                         C_TAIL: 5.0})                              # an incumbent: not a Sobol point
    out = R.sobol_rankings(pd.DataFrame(rows), wide, np.random.default_rng(0), n_boot=10, n_points=3)
    assert out["point_zero_tau"] == pytest.approx(1.0)
    per = out["per_point"].set_index("config_id")
    assert list(per.index) == [1, 2, 3]
    assert per.loc[1, "default"] == pytest.approx(1.0) and per.loc[2, "default"] == pytest.approx(-1.0)
    agg = out["aggregate"].set_index(["aggregate", "arm"])
    # mean over the points is 2v/3, the median v: both rank like the arms
    assert agg.loc[("mean", "default"), "tau"] == pytest.approx(1.0)
    assert agg.loc[("median", "oracle_static"), "tau"] == pytest.approx(1.0)
    assert out["describe"].loc["count", "default"] == 3


def _loo_cells():
    cells = _cells([
        ("A", "covtype", True, 0.0, 0.5, 0.2, 1.0),
        ("A", INSECTS3[0], True, 0.0, 0.1, 0.0, 0.2),
        ("A", INSECTS3[1], True, 0.0, 0.2, 0.1, 0.2),
        ("A", "kdd99", False, 0.0, 0.0, 0.0, 0.01),
        ("A", "k_drift", True, 0.2, 0.9, 0.5, 0.6),
    ])
    loss = pd.DataFrame({"algo": "A", "fold": list(cells["fold"]),
                         "a_lag": [0.5, 0.1, 0.2, 0.0, 0.9], "a_star_lag": [0.6, 0.15, 0.2, 0.0, 0.92],
                         "A_static": [1.0, 0.2, 0.2, 0.0, 0.6]})
    return cells, loss


def test_leave_one_out_drops_a_fold_or_a_whole_source_block_and_recomputes_the_group():
    cells, loss = _loo_cells()
    by_fold = R.leave_one_out(cells, loss, unit="fold", group="real").set_index("dropped")
    assert list(by_fold.index) == ["none", "covtype", INSECTS3[0], INSECTS3[1], "kdd99"]
    none = by_fold.loc["none"]
    assert none["capture_confstream"] == pytest.approx(0.8 / 1.4)
    assert none["capture_den_confstream"] == pytest.approx(1.4) and none["n_tunable_pairs"] == 3
    assert none["selection_loss"] == pytest.approx((0.1 + 0.05 + 0.0 + 0.0) / 4)
    assert none["gap"] == pytest.approx(none["selection_loss"] + none["search_loss"])
    assert none["selection_share"] == pytest.approx(none["selection_loss"] / none["gap"])
    assert by_fold.loc["covtype", "capture_confstream"] == pytest.approx(0.3 / 0.4)
    by_block = R.leave_one_out(cells, loss, unit="block", group="real").set_index("dropped")
    assert list(by_block.index) == ["none", "covtype", "insects", "kdd99"]
    assert by_block.loc["insects", "n_dropped_folds"] == 2
    assert by_block.loc["insects", "capture_confstream"] == pytest.approx(0.5 / 1.0)
    extra = R.leave_one_out(cells, loss, unit="fold", group="real",
                            extra=(("covtype + kdd99", ("covtype", "kdd99")), ("absent", ("rialto",))))
    assert list(extra["dropped"])[-1] == "covtype + kdd99"      # a set none of whose folds ran: no row


def test_per_stream_split_gives_each_streams_part_of_the_gain_and_of_the_selection_loss():
    cells, loss = _loo_cells()
    out = R.per_stream_split(cells, loss, group="real").set_index("fold")
    assert out.loc["covtype", "numerator"] == pytest.approx(0.5)
    assert out.loc["covtype", "denominator_share"] == pytest.approx(1.0 / 1.4)
    assert out.loc["kdd99", "n_tunable"] == 0 and pd.isna(out.loc["kdd99", "numerator"])
    assert out.loc["covtype", "selection_share"] == pytest.approx(0.1 / 0.15)
    assert out["selection_share"].sum() == pytest.approx(1.0)


def test_selection_share_interval_resamples_streams_and_counts_a_non_positive_search_loss():
    cells, loss = _loo_cells()
    out = R.selection_share_interval(loss, np.random.default_rng(1), n_boot=200, group="real")
    sel = np.array([0.1, 0.05, 0.0, 0.0])
    sea = np.array([0.4, 0.05, 0.0, 0.0])
    assert out["n_folds"] == 4
    assert out["share"] == pytest.approx(sel.mean() / (sel.mean() + sea.mean()))
    assert out["low"] <= out["share"] <= out["high"]
    assert 0 < out["p_search_le_0"] < 1          # a draw of the two zero-search streams only
    assert out["n_undefined"] > 0                 # ... where the share has no denominator


# --- P1.2: NMI, with the selections kept (a) or redone (b) -----------------------------------

def test_tail_values_are_one_mean_per_run_over_its_tail_windows_of_any_column():
    win = pd.DataFrame([
        {"algo": "A", "fold": "covtype", "arm": arm, "seed": s, "window": w,
         "lag_nmi_win": (0.5 if arm == "default" else 0.7) + 0.01 * s + (9.0 if w < 20 else 0.0)}
        for arm in ("default", "confstream") for s in (1, 2) for w in (19, 20, 21, 22)
    ])
    win.loc[(win["window"] == 22) & (win["arm"] == "confstream"), "lag_nmi_win"] = None
    out = R.tail_values(win, tail_from={}, column="lag_nmi_win").set_index(["arm", "seed"])["value"]
    assert out[("default", 1)] == pytest.approx(0.51) and out[("confstream", 2)] == pytest.approx(0.72)
    odd = R.tail_values(win, tail_from={("A", "covtype", 1, "default"): 21}, column="lag_nmi_win", parity=1)
    assert len(odd) == 4 and odd.set_index(["arm", "seed"]).loc[("default", 1), "value"] == pytest.approx(0.51)


def _by_window():
    """window -> member_id -> (is_test, usable, score, ari, nmi): on window 1 member 1 is best
    by ARI and member 2 by NMI; member 3 is on test on both windows (a challenger accepted on
    window 1 would be active on 2 and a candidate); member 4 was not usable on window 1."""
    return {
        1: {1: (False, True, 0.2, 0.6, 0.3), 2: (False, True, 0.5, 0.4, 0.8), 3: (True, True, 0.9, 0.9, 0.9),
            4: (False, False, 0.1, 0.95, 0.95)},
        2: {1: (False, True, 0.2, 0.50, 0.20), 2: (False, True, 0.5, 0.30, 0.70), 3: (True, True, 0.9, 0.1, 0.1),
            4: (False, True, 0.1, 0.99, 0.99)},
    }


def test_member_picks_by_ari_reproduce_the_report_and_by_nmi_pick_on_nmi():
    from conf_pipeline import report

    bw = _by_window()
    ari_only = {w: {m: v[:4] for m, v in ms.items()} for w, ms in bw.items()}
    published = report._job_readings(ari_only)
    mine = R.member_picks(bw, pick="ari", read="ari")
    assert mine["star"] == published["star"] and mine["lag_pick"] == published["lag_pick"]
    assert mine["lag_pick"][2] == {1: 0.50}              # labels' pick after window 1 by ARI, read on 2
    a = R.member_picks(bw, pick="ari", read="nmi")
    assert a["lag_pick"][2] == {1: 0.20}                 # the same member, read in NMI
    assert a["star"][2] == pytest.approx(0.99)
    b = R.member_picks(bw, pick="nmi", read="nmi")
    assert b["lag_pick"][2] == {2: 0.70}                 # re-picked by NMI


def _add_nmi(root):
    """Give the toy out dir per-window NMI of the incumbent and of every member: member 0 is
    the better one by NMI (v + 0.06), member 1 by ARI."""
    def rewrite(path, fn):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        path.write_text("".join(json.dumps(fn(r)) + "\n" for r in rows))

    def window(r):
        if r.get("ari_win") is not None:
            r["nmi_win"] = r["ari_win"] + 0.04
        return r

    def member(r):
        r["nmi_win"] = r["ari_win"] + (0.04 if r["member_id"] == 0 else -0.04)
        return r

    rewrite(root / "windows.jsonl", window)
    for path in (root / "members").glob("*.jsonl"):
        rewrite(path, member)
    rewrite(root / "sweep_windows.jsonl", lambda r: {**r, "lag_nmi_win": None if r["lag_ari_win"] is None else
                                                     (0.9 - r["lag_ari_win"] if r["config_id"] == 0 else r["lag_ari_win"])})
    return root


def test_selection_windows_by_ari_reproduce_the_regret_windows_losses(tmp_path):
    from conf_pipeline import report

    out = _add_nmi(_toy_out_dir(tmp_path / "out"))
    inputs = R.load_inputs(out)
    windows = R.current_windows(out, inputs.runs)
    tail_from = R._tail_from(inputs.runs)
    readings = R.load_member_picks(out)
    sel = R.selection_windows(windows, readings, tail_from)
    ari = R.selection_loss_seeds(sel, "ari", "ari")
    published = R.loss_seeds(inputs.regret)
    pd.testing.assert_frame_equal(ari.reset_index(drop=True), published.reset_index(drop=True), check_dtype=False)
    a = R.selection_loss_seeds(sel, "ari", "nmi").set_index(["algo", "fold", "seed"])
    b = R.selection_loss_seeds(sel, "nmi", "nmi").set_index(["algo", "fold", "seed"])
    q = TOY_QUALITY["BIRCH"] + TOY_OFFSET["confstream"]
    # a_lag in NMI is the toy's lag_nmi_win (0.9 v); the ARI pick (member 1) reads v + 0.01 in NMI,
    # the NMI pick (member 0) v + 0.06
    row_a, row_b = a.loc[("BIRCH", "covtype", 2)], b.loc[("BIRCH", "covtype", 2)]
    assert row_a["a_lag"] == pytest.approx(0.9 * q) and row_b["a_lag"] == pytest.approx(0.9 * q)
    assert row_a["a_star_lag"] == pytest.approx(q + 0.01) and row_b["a_star_lag"] == pytest.approx(q + 0.06)


def test_selection_windows_say_when_the_logs_have_no_nmi():
    win = pd.DataFrame([{"algo": "A", "fold": "covtype", "arm": "confstream", "seed": 1, "window": w,
                         "incumbent_id": 0, "lag_ari_win": 0.3, "ari_win": 0.3} for w in (20, 21)])
    sel = R.selection_windows(win, {}, tail_from={})
    assert set(sel["read"]) == {"ari"}
    with pytest.raises(KeyError, match="nmi"):
        R.selection_loss_seeds(sel, "ari", "nmi")


def test_reselected_reference_takes_the_best_candidate_by_the_given_column_ties_to_the_smaller_id():
    sw = pd.DataFrame([{"algo": "A", "fold": "covtype", "seed": 1, "config_id": cid, "window": w,
                        "lag_ari_win": ari, "lag_nmi_win": nmi}
                       for cid, ari, nmi in ((0, 0.1, 0.2), (3, 0.5, 0.4), (5, 0.4, 0.6), (7, 0.2, 0.6), (9, 0.9, 0.9))
                       for w in (19, 20, 21)])
    sw.loc[(sw["window"] == 19), ["lag_ari_win", "lag_nmi_win"]] = 5.0        # before the tail
    cand = {("A", "covtype", c) for c in (0, 3, 5, 7)}                        # 9 is not a candidate
    by_nmi = R.reselect_reference(sw, {("A", "covtype"): 20}, "lag_nmi_win", candidates=cand).iloc[0]
    assert by_nmi["config_id"] == 5 and by_nmi["select_value"] == pytest.approx(0.6)
    assert by_nmi["n_candidates"] == 4
    by_ari = R.reselect_reference(sw, {("A", "covtype"): 20}, "lag_ari_win", candidates=cand).iloc[0]
    assert by_ari["config_id"] == 3
    assert R.reselect_reference(sw, {}, "lag_nmi_win").iloc[0]["config_id"] == 9


def test_tunable_gate_on_another_measure_mirrors_the_headroom_gate():
    seeds = pd.DataFrame([
        {"algo": "A", "fold": "covtype", "arm": "default", "seed": s, "value": v} for s, v in ((1, 0.1), (2, 0.1), (3, 0.1))
    ] + [
        {"algo": "B", "fold": "covtype", "arm": "default", "seed": s, "value": v} for s, v in ((1, 0.0), (2, 0.2), (3, 0.4))
    ])
    ref = pd.DataFrame({"algo": ["A", "B"], "fold": ["covtype", "covtype"], "value": [0.2, 0.35]})
    gate = R.tunable_gate(seeds, ref)
    assert gate[("A", "covtype")]                      # gain 0.1 >= 0.05, no seed noise
    assert not gate[("B", "covtype")]                  # gain 0.15 within 1.645 * 0.2 / sqrt(3) = 0.19
    assert not R.tunable_gate(seeds, ref.assign(value=[0.14, 0.35]))[("A", "covtype")]   # gain 0.04


# --- P1.2: paired effects, ratios and rankings by statistical unit ----------------------------

def _con(pairs, tunable=None, reference="ARI-selected reference", metric="ari", basis="all tail windows"):
    tunable = {k: True for k in pairs} if tunable is None else tunable
    return R.Construction(reference=reference, metric=metric, basis=basis, seeds=_seed_rows(pairs),
                          losses=_loss_rows(pairs), tunable=tunable, gate="ARI gate")


def _block_pairs():
    """Algo A: online - default is 0.0 on covtype and 0.1, 0.1, 0.4 on the three INSECTS
    variants; the reference sits 0.5 above default everywhere; k_drift is the only other group."""
    pairs = {}
    for fold, gain in (("covtype", 0.0), (INSECTS3[0], 0.1), (INSECTS3[1], 0.1), (INSECTS3[2], 0.4),
                       ("k_drift", 0.2)):
        d = np.array([0.1, 0.1, 0.1])
        pairs[("A", fold)] = {"default": d, "confstream": d + gain, "frozen_prefix": d + gain / 2,
                              "oracle_static": d + 0.5}
    return pairs


def test_unit_effects_average_within_a_block_before_averaging_over_blocks():
    eff = R.unit_effects(_con(_block_pairs()), unit="block")
    real = eff[(eff["group"] == "real") & (eff["algo"] == R.POOLED) & (eff["comparison"] == "confstream - default")]
    assert sorted(real["unit"]) == ["covtype", "insects"]
    assert real.set_index("unit").loc["insects", "value"] == pytest.approx(0.2)
    assert real.set_index("unit").loc["insects", "n_pairs"] == 3
    by_fold = R.unit_effects(_con(_block_pairs()), unit="fold")
    real_f = by_fold[(by_fold["group"] == "real") & (by_fold["algo"] == "A")
                     & (by_fold["comparison"] == "confstream - default")]
    assert len(real_f) == 4 and real_f["value"].mean() == pytest.approx(0.15)
    assert set(by_fold["comparison"]) == {"confstream - default", "confstream - frozen_prefix",
                                          "confstream - reference"}


def test_unit_statistics_points_match_the_group_aggregations_and_the_unit_means():
    con = _con(_block_pairs())
    out = {u: R.unit_statistics(con, unit=u, n_boot=50, seed=0).set_index(["statistic", "group", "algo"])
           for u in R.UNITS}
    cells = R.construction_cells(con)
    cap = R.capture_parts(cells[0]).set_index(["group", "arm"])
    for u in R.UNITS:   # ratios of sums and pair means keep their published definition under both units
        row = out[u].loc[("capture_confstream", "real", R.POOLED)]
        assert row["point"] == pytest.approx(cap.loc[("real", "confstream"), "capture"])
        assert row["numerator"] == pytest.approx(0.6) and row["denominator"] == pytest.approx(2.0)
        assert row["n_pairs"] == 4
    assert out["fold"].loc[("confstream - default", "real", R.POOLED), "point"] == pytest.approx(0.15)
    assert out["block"].loc[("confstream - default", "real", R.POOLED), "point"] == pytest.approx(0.1)
    assert out["block"].loc[("confstream - default", "real", R.POOLED), "n_units"] == 2
    assert out["block"].loc[("confstream - default", "real", "A"), "point"] == pytest.approx(0.1)
    sel = out["fold"].loc[("selection_loss", "all", R.POOLED)]
    assert sel["point"] == pytest.approx(0.05) and sel["low"] == pytest.approx(0.05)


def test_ratios_are_undefined_below_the_denominator_threshold_and_never_clipped():
    pairs = {("A", "covtype"): {"default": np.zeros(3), "confstream": np.full(3, 0.3),
                                "frozen_prefix": np.full(3, 0.1), "oracle_static": np.full(3, 0.2)},
             ("A", "k_drift"): {"default": np.zeros(3), "confstream": np.full(3, 0.1),
                                "frozen_prefix": np.zeros(3), "oracle_static": np.full(3, 0.004)}}
    out = R.unit_statistics(_con(pairs), unit="fold", n_boot=20, seed=0,
                            min_denominator=0.01).set_index(["statistic", "group", "algo"])
    real = out.loc[("capture_confstream", "real", R.POOLED)]
    assert real["point"] == pytest.approx(1.5) and real["status"] == "ok"     # above 1: reported as is
    drift = out.loc[("capture_confstream", "param_drift", R.POOLED)]
    assert pd.isna(drift["point"]) and drift["status"] == "undefined"
    assert drift["denominator"] == pytest.approx(0.004) and drift["n_pairs"] == 1
    # _loss_rows: a_lag = online, a_star_lag = online + 0.05: on covtype the gap is 0.2 - 0.3 < 0
    share = out.loc[("selection_share", "real", R.POOLED)]
    assert pd.isna(share["point"]) and share["status"] == "undefined"
    assert share["numerator"] == pytest.approx(0.05) and share["denominator"] == pytest.approx(-0.1)


def test_a_ratio_whose_denominator_vanishes_in_some_replicates_is_flagged_unstable():
    pairs = {}
    for fold, ref in (("covtype", 0.5), ("kdd99", 0.0), ("shuttle", 0.0)):
        pairs[("A", fold)] = {"default": np.zeros(3), "confstream": np.full(3, 0.1),
                              "frozen_prefix": np.zeros(3), "oracle_static": np.full(3, ref)}
    out = R.unit_statistics(_con(pairs), unit="fold", n_boot=200, seed=0).set_index(["statistic", "group", "algo"])
    row = out.loc[("capture_confstream", "real", R.POOLED)]
    assert row["point"] == pytest.approx(0.3 / 0.5)
    assert row["status"] == "unstable" and row["n_used"] < 200


def test_paired_unit_tests_run_wilcoxon_over_the_units_with_holm_per_family():
    pairs = {}
    folds = ["covtype", "kdd99", "shuttle", "electricity", *INSECTS3]
    for j, fold in enumerate(folds):
        for algo in ("A", "B"):
            d = np.full(3, 0.1)
            pairs[(algo, fold)] = {"default": d, "confstream": d + 0.1 + 0.01 * j, "frozen_prefix": d,
                                   "oracle_static": d + 0.3}
    con = _con(pairs)
    tests = R.paired_unit_tests(R.unit_effects(con, unit="block"), unit="block")
    row = tests[(tests["group"] == "real") & (tests["algo"] == "A") & (tests["comparison"] == "confstream - default")].iloc[0]
    assert row["n_units"] == 5 and row["n_zero"] == 0          # 4 streams and the insects block
    assert row["p"] == pytest.approx(2 / 32)
    fam = tests[tests["group"] == "real"]
    assert (fam["p_holm"].dropna() >= fam["p"].dropna() - 1e-12).all()
    fold = R.paired_unit_tests(R.unit_effects(con, unit="fold"), unit="fold")
    assert fold[(fold["group"] == "real") & (fold["algo"] == "A")]["n_units"].iloc[0] == 7


def test_tau_units_read_rankings_on_folds_or_on_block_means():
    folds, wide = _arm_wides()
    wide = {k: v.copy() for k, v in wide.items()}
    wide["confstream"].loc[INSECTS3[0]] = [0.0, 0.1, 0.2, 3.0]
    fold = R.tau_units(wide, "fold", np.random.default_rng(0), n_boot=20).set_index(["arm_a", "arm_b", "group"])
    block = R.tau_units(wide, "block", np.random.default_rng(0), n_boot=20).set_index(["arm_a", "arm_b", "group"])
    assert fold.loc[("confstream", "frozen_prefix", "real"), "n_units"] == 5
    assert block.loc[("confstream", "frozen_prefix", "real"), "n_units"] == 3
    tau_b, _ = R.block_rankings(wide)
    pooled = tau_b.set_index(["arm_a", "arm_b"]).loc[("confstream", "frozen_prefix"), "tau"]
    assert block.loc[("confstream", "frozen_prefix", "all"), "tau"] == pytest.approx(pooled)
    kept = R.tau_units(wide, "fold", np.random.default_rng(0), n_boot=5,
                       keep=wide["default"] >= 0.2).set_index(["arm_a", "arm_b", "group"])
    assert kept.loc[("default", "confstream", "all"), "n_units"] == 7      # three algorithms left per fold


def test_conclusions_cross_conclusion_metric_unit_and_reference_construction():
    pairs = _block_pairs()
    for fold in ("kdd99", "evolving"):
        d = np.array([0.0, 0.1, 0.2])
        pairs[("A", fold)] = {"default": d, "confstream": d + 0.3, "frozen_prefix": d + 0.1, "oracle_static": d + 0.4}
    cons = [_con(pairs), _con(pairs, reference="ARI-selected reference, NMI-evaluated", metric="nmi"),
            _con(pairs, reference="NMI-reselected reference", metric="nmi")]
    undefined = pd.DataFrame(0.0, index=sorted({f for _, f in pairs}), columns=["A"])
    out = R.conclusions_table(cons, n_boot=20, seed=0, undefined=undefined)
    assert {"conclusion", "group", "metric", "unit", "reference", "subset", "value", "ci_low", "ci_high",
            "n_units", "n_pairs", "numerator", "denominator", "n_used", "status", "note"} <= set(out.columns)
    assert set(out["unit"]) == set(R.UNITS) and set(out["metric"]) == {"ari", "nmi"}
    refs = set(out["reference"])
    assert {"ARI-selected reference", "ARI-selected reference, NMI-evaluated", "NMI-reselected reference"} <= refs
    assert not any("optimal" in r.lower() for r in refs)
    od = out[(out["conclusion"] == "online - default") & (out["group"] == "real") & (out["unit"] == "block")]
    assert len(od) == 2 and set(od["reference"]) == {"none (all tail windows)"}       # once per metric
    g = out[(out["conclusion"] == "capture G, online") & (out["group"] == "real") & (out["unit"] == "fold")]
    assert len(g) == 3 and g["denominator"].notna().all()
    assert set(out["subset"]) >= {"all streams", *[f for f, *_ in R.WEAK_DEFAULT_FILTERS]}


# --- P1.2 end to end ---------------------------------------------------------------------------

def test_build_constructions_names_every_reference_and_reselects_by_nmi(tmp_path):
    out = _add_nmi(_toy_out_dir(tmp_path / "out"))
    cons, nmi_ref, notes = R.build_constructions(out)
    names = [c.reference for c in cons]
    assert names == list(R.REFERENCE_CONSTRUCTIONS) and not notes
    by = {c.reference: c for c in cons}
    assert by["ARI-selected reference"].metric == "ari" and by["NMI-reselected reference"].metric == "nmi"
    ref = nmi_ref.set_index(["algo", "fold"]).loc[("BIRCH", "covtype")]
    # the toy's config 0 reads 0.9 - ARI in NMI: the best by NMI is River's default, not config 7
    assert ref["ari_config_id"] == 7 and ref["nmi_config_id"] == 0 and not ref["same_config"]
    assert ref["ari_repick_config_id"] == 7
    assert ref["nmi_ref_nmi_sweep"] == pytest.approx(0.8)
    assert ref["nmi_ref_nmi_sweep"] >= ref["ari_ref_nmi_sweep"] - 1e-12        # a maximum over the same set
    published = R.cells_table(R.load_inputs(out))
    mine, _ = R.construction_cells(by["ARI-selected reference"])
    pd.testing.assert_frame_equal(mine.reset_index(drop=True), published.reset_index(drop=True), check_dtype=False)


def test_build_constructions_says_when_the_logs_carry_no_nmi(tmp_path):
    out = _toy_out_dir(tmp_path / "out")          # no nmi_win in the window or member logs
    cons, nmi_ref, notes = R.build_constructions(out)
    assert [c.metric for c in cons] == ["ari", "ari"]
    assert notes and "nmi" in notes[0].lower()


def test_write_robustness_adds_the_p12_tables_and_section(tmp_path):
    out = _add_nmi(_toy_out_dir(tmp_path / "out"))
    rep = tmp_path / "rep"
    R.write_robustness(out, n_boot=30, rep_dir=rep)
    csv = rep / "robustness"
    for name in ("conclusions.csv", "constructions.csv", "paired_effects.csv", "loo.csv", "nmi_reference.csv",
                 "tau_arms.csv", "block_rankings_tau.csv", "block_rankings_ranks.csv", "tau_weak_default.csv",
                 "real_streams.csv", "selection_share.csv", "unit_statistics.csv"):
        assert (csv / name).exists(), name
    conc = pd.read_csv(csv / "conclusions.csv")
    assert {"ARI-selected reference, NMI-evaluated", "NMI-reselected reference"} <= set(conc["reference"])
    assert set(conc["unit"]) == {"fold", "block"}
    md = (rep / "robustness.md").read_text()
    assert "## P1.2" in md and "NMI-reselected reference" in md and "NMI-optimal" not in md
    # the published sections are still there and unchanged in kind
    for heading in ("A1", "A2", "A3", "A4", "A5", "Usable streams"):
        assert heading in md


def test_a_reference_read_once_is_never_lost_to_an_empty_seed_slot():
    # the reference has one reading (a sweep run, seed label 0) against three protocol seeds;
    # drawing among all three slots would leave it missing in about 30 % of the pairs
    pairs = {}
    for j, fold in enumerate(("covtype", "kdd99", "shuttle")):
        pairs[("A", fold)] = {"default": np.zeros(3), "confstream": np.full(3, 0.1 * (j + 1)),
                              "frozen_prefix": np.zeros(3)}
    seeds = _seed_rows(pairs)
    ref = pd.DataFrame({"algo": "A", "fold": ["covtype", "kdd99", "shuttle"], "arm": "oracle_static", "seed": 0,
                        "value": [0.5, 0.5, 0.5]})
    con = R.Construction(reference="r", metric="ari", basis="all tail windows",
                         seeds=pd.concat([seeds, ref], ignore_index=True), losses=_loss_rows(pairs),
                         tunable={k: True for k in pairs})
    out = R.unit_statistics(con, unit="fold", n_boot=200, seed=0).set_index(["statistic", "group", "algo"])
    row = out.loc[("selection_loss", "real", R.POOLED)]
    assert row["n_used"] == 200
    ref_eff = out.loc[("confstream - reference", "real", R.POOLED)]
    assert ref_eff["point"] == pytest.approx(-0.3)
    # every replicate keeps all three pairs: the online - reference effect varies only by the folds drawn
    assert ref_eff["low"] >= -0.4 - 1e-12 and ref_eff["high"] <= -0.2 + 1e-12
    boot = R.bootstrap(con.seeds, con.losses, con.tunable, n_boot=100, pairing="seed").set_index(
        ["statistic", "group", "arm"])
    assert boot.loc[("search_loss", "real", "confstream"), "n_used"] == 100
    assert boot.loc[("search_loss", "real", "confstream"), "low"] >= 0.5 - 0.3 - 0.05 - 1e-12


# --- P1.2: the compatibility check against robustness_extra.txt -------------------------------

def test_a_printed_number_matches_within_half_a_unit_of_its_last_place():
    from conf_pipeline import robustness_check as K

    assert K.matches("+0.073", 0.07349) and not K.matches("+0.073", 0.0736)
    assert K.matches("2.83", 2.8333) and K.matches("18", 18) and not K.matches("18", 17)
    assert K.matches("3e-06", 3.4e-06) and not K.matches("3e-06", 4e-06)
    assert K.matches("True", True) and not K.matches("True", False)


def test_the_check_reads_the_printed_sections_and_compares_their_numbers():
    from conf_pipeline import robustness_check as K

    text = "\n".join([
        "", "======== tau giữa các chế độ: trung bình qua luồng, khoảng bootstrap 95 % ========",
        "       default vs confstream     18 luồng +0.073 [-0.018; +0.167] | thật -0.114 [-0.247; +0.026]",
        "", "======== thuật toán tất định trong các cặp tunable ========",
        "cặp tunable: 105 | thuộc thuật toán tất định: 64",
        "======== hồi phục: trong 0,05 của ARI trước drift của chính chế độ đó ========", "x",
    ])
    secs = K.sections(text)
    assert list(secs) == ["tau giữa các chế độ: trung bình qua luồng, khoảng bootstrap 95 %",
                          "thuật toán tất định trong các cặp tunable",
                          "hồi phục: trong 0,05 của ARI trước drift của chính chế độ đó"]
    tau = pd.DataFrame([{"arm_a": "default", "arm_b": "confstream", "scope": s, "n_folds": n, "tau": t,
                         "low": lo, "high": hi} for s, n, t, lo, hi in (("all", 18, 0.0731, -0.018, 0.167),
                                                                         ("real", 11, -0.114, -0.247, 0.0275))])
    empty = pd.DataFrame(columns=["arm_a", "arm_b", "scope", "filter", "arm", "aggregate", "dropped", "fold", "n_blocks"])
    got = {"tau": tau, "block_tau": empty, "block_ranks": empty, "weak": empty,
           "sobol": {"describe": pd.DataFrame(), "aggregate": empty, "point_zero_tau": np.nan},
           "share": {}, "loo": empty, "streams": empty,
           "seedless": {"n_tunable": 105, "n_tunable_seedless": 63, "reference_sd_zero_on_seedless": True},
           "streamkmeans": {}}
    out = K.compare(text, got)
    checked = out[out["match"].notna()]
    assert len(checked) == 7 + 2
    bad = checked[~checked["match"].astype(bool)]
    assert list(bad["item"]) == ["default vs confstream: real high", "tunable pairs of seedless algorithms"]
    assert out[out["match"].isna()]["item"].tolist() == ["(not ported)"]


@pytest.mark.slow
@pytest.mark.skipif(not (FULL / "runs.jsonl").exists() or not (C.REPO_ROOT / "result" / "otb_full" / "report" /
                                                                "robustness_extra.txt").exists(),
                    reason="the full run's logs or the published file are not here")
def test_the_ported_checks_reproduce_robustness_extra_txt_on_the_full_run():
    from conf_pipeline import robustness_check as K

    table = K.compare(K.PUBLISHED.read_text(encoding="utf-8"), K.recompute(FULL))
    checked = table[table["match"].notna()]
    assert len(checked) > 200
    bad = checked[~checked["match"].astype(bool)]
    assert bad.empty, bad.to_string()


def test_seedless_pairs_count_protocol_algorithms_only():
    tunable = {("BIRCH", "covtype"): True, ("CluStream", "covtype"): True, ("DenStreamPatchedFast", "covtype"): True,
               ("SOStream", "kdd99"): False}
    summary = pd.DataFrame([{"algo": "BIRCH", "fold": "covtype", "arm": "oracle_static", "lag_ari_win_tail_sd": 0.0}])
    out = R.seedless_pairs(tunable, summary)
    assert out["n_tunable"] == 2 and out["n_tunable_seedless"] == 1     # the implementation variant left out
    assert "DenStreamPatchedFast" not in out["seedless_algos"] and "CluStream" not in out["seedless_algos"]
    assert out["reference_sd_zero_on_seedless"]
