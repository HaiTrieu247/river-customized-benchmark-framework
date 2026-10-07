"""conf_pipeline.analyze: cross-fold rankings, their agreement, and the figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conf_pipeline import analyze, report

TAIL = "lag_ari_win_tail_mean"
TAIL_SD = "lag_ari_win_tail_sd"
SPS = "incumbent_throughput_sps"
SPS_SD = "incumbent_throughput_sps_sd"

#: (arm, fold) -> {algo: tail mean}. C has no row on f4 in any arm. Under
#: confstream (and oracle_static) every fold ranks A > B > C; under default
#: fold f1 ranks B > A > C, so tau(default, confstream) on f1 is 1/3.
TAIL_VALUES = {
    ("confstream", "f1"): {"A": 0.8, "B": 0.6, "C": 0.4},
    ("confstream", "f2"): {"A": 0.7, "B": 0.5, "C": 0.3},
    ("confstream", "f3"): {"A": 0.6, "B": 0.5, "C": 0.2},
    ("confstream", "f4"): {"A": 0.5, "B": 0.4},
    ("default", "f1"): {"A": 0.5, "B": 0.6, "C": 0.3},
    ("default", "f2"): {"A": 0.4, "B": 0.3, "C": 0.2},
    ("default", "f3"): {"A": 0.55, "B": 0.4, "C": 0.1},
    ("default", "f4"): {"A": 0.3, "B": 0.2},
    ("oracle_static", "f1"): {"A": 0.9, "B": 0.7, "C": 0.5},
    ("oracle_static", "f2"): {"A": 0.8, "B": 0.6, "C": 0.4},
    ("oracle_static", "f3"): {"A": 0.7, "B": 0.6, "C": 0.3},
    ("oracle_static", "f4"): {"A": 0.6, "B": 0.5},
}
#: A's gaps to default: 0.3, 0.3, 0.05, 0.2. Every sd is 0.02 except the
#: default's on f3 (0.1), so exactly one fold (f3) falls inside the band.
DEFAULT_F3_SD = 0.1
SPS_BY_ARM = {"default": 1000.0, "confstream": 100.0, "oracle_static": 800.0}
SPS_BY_ALGO = {"A": 1.0, "B": 2.0, "C": 0.5}


def _summary_rows():
    rows = []
    for (arm, fold), values in TAIL_VALUES.items():
        for algo, tail in values.items():
            sd = DEFAULT_F3_SD if (arm, fold) == ("default", "f3") else 0.02
            rows.append({
                "algo": algo, "fold": fold, "arm": arm, "n_seeds": 5, "n_failed": 0,
                "failed_status": None, TAIL: tail, TAIL_SD: sd,
                "lag_ari_win_mean": tail - 0.05, "lag_ari_win_sd": 0.03,
                SPS: SPS_BY_ARM[arm] * SPS_BY_ALGO[algo], SPS_SD: 10.0,
            })
    return rows


@pytest.fixture
def summary():
    return pd.DataFrame(_summary_rows())


# --- rank_matrix ---------------------------------------------------------------------------

def test_rank_matrix_ranks_one_best_and_missing_last(summary):
    r = analyze.rank_matrix(summary, "confstream")
    assert r.loc["f1"].tolist() == [1.0, 2.0, 3.0] and r.loc["f4", "C"] == 3.0   # C has no f4 row


def test_rank_matrix_has_sorted_folds_and_the_union_of_algos(summary):
    s = summary[~((summary["algo"] == "C") & (summary["arm"] == "confstream"))]
    r = analyze.rank_matrix(s.iloc[::-1], "confstream")
    assert r.index.tolist() == ["f1", "f2", "f3", "f4"]
    # C has rows only under the other arms: under confstream it ranks last everywhere
    assert r.columns.tolist() == ["A", "B", "C"] and (r["C"] == 3.0).all()


def test_rank_matrix_averages_ties_and_treats_none_as_missing(summary):
    s = summary.copy()
    s.loc[(s["arm"] == "confstream") & (s["fold"] == "f2") & (s["algo"] == "B"), TAIL] = 0.7
    s.loc[(s["arm"] == "confstream") & (s["fold"] == "f3") & (s["algo"] == "A"), TAIL] = None
    r = analyze.rank_matrix(s, "confstream")
    assert r.loc["f2"].tolist() == [1.5, 1.5, 3.0]
    assert r.loc["f3"].tolist() == [3.0, 1.0, 2.0]


def test_rank_matrix_of_an_empty_summary_is_an_empty_frame():
    r = analyze.rank_matrix(pd.DataFrame(columns=["algo", "fold", "arm"]), "confstream")
    assert r.empty and r.shape == (0, 0)


# --- kendall_matrix ------------------------------------------------------------------------

def test_kendall_matrix_is_symmetric_with_unit_diagonal(summary):
    k = analyze.kendall_matrix(summary, ["default", "confstream", "oracle_static"])
    assert np.allclose(np.diag(k.values), 1.0) and np.allclose(k.values, k.values.T)
    assert k.loc["default", "confstream"] < 1.0     # the fixture flips one ranking


def test_kendall_matrix_is_the_mean_tau_over_folds(summary):
    k = analyze.kendall_matrix(summary, ["default", "confstream", "oracle_static"])
    # f1: (B, A, C) vs (A, B, C) -> tau = 1/3; f2..f4 identical -> 1
    assert k.loc["default", "confstream"] == pytest.approx((1 / 3 + 3) / 4)
    assert k.loc["confstream", "oracle_static"] == 1.0
    assert k.index.tolist() == k.columns.tolist() == ["default", "confstream", "oracle_static"]


def test_kendall_matrix_skips_folds_with_fewer_than_two_ranked_algos(summary):
    s = summary.copy()
    # f1 under default keeps only A: that fold says nothing about the ranking
    s = s[~((s["arm"] == "default") & (s["fold"] == "f1") & (s["algo"] != "A"))]
    k = analyze.kendall_matrix(s, ["default", "confstream"])
    assert k.loc["default", "confstream"] == 1.0


def test_kendall_matrix_without_a_shared_fold_is_nan_off_the_diagonal(summary):
    s = summary[~((summary["arm"] == "default") & (summary["fold"] != "f1"))]
    s = s[~((s["arm"] == "confstream") & (s["fold"] == "f1"))]
    k = analyze.kendall_matrix(s, ["default", "confstream"])
    assert np.isnan(k.loc["default", "confstream"]) and k.loc["default", "default"] == 1.0


def test_kendall_complete_only_ranks_just_the_algos_both_arms_read(summary):
    # A, best on f2, has no confstream reading there. Ranked last it reverses two pairs
    # (tau(f2) = -1/3); compared only on B and C, which agree, f2 reads 1.
    s = summary[~((summary["arm"] == "confstream") & (summary["fold"] == "f2") & (summary["algo"] == "A"))]
    every = analyze.kendall_matrix(s, ["default", "confstream"])
    both = analyze.kendall_matrix(s, ["default", "confstream"], complete_only=True)
    # f1 1/3 (as in the fixture), f3 and f4 1
    assert every.loc["default", "confstream"] == pytest.approx((1 / 3 - 1 / 3 + 2) / 4)
    assert both.loc["default", "confstream"] == pytest.approx((1 / 3 + 3) / 4)


def test_kendall_complete_only_skips_a_fold_with_fewer_than_two_shared_algos(summary):
    # f1: confstream reads A and B, default A and C. Each arm ranks two algorithms, so the
    # fold counts when a missing reading ranks last (tau 1/3), but only A is shared.
    s = summary[~((summary["arm"] == "confstream") & (summary["fold"] == "f1") & (summary["algo"] == "C"))]
    s = s[~((s["arm"] == "default") & (s["fold"] == "f1") & (s["algo"] == "B"))]
    every = analyze.kendall_matrix(s, ["default", "confstream"])
    both = analyze.kendall_matrix(s, ["default", "confstream"], complete_only=True)
    assert every.loc["default", "confstream"] == pytest.approx((1 / 3 + 3) / 4)
    assert both.loc["default", "confstream"] == 1.0     # f1 skipped; f2..f4 agree


# --- degenerate cells ----------------------------------------------------------------------

def test_degenerate_table_counts_cells_whose_lagged_windows_are_mostly_undefined(summary):
    s = summary.assign(lag_undefined_share=0.0)
    s.loc[(s["arm"] == "default") & (s["fold"].isin(["f1", "f2"])), "lag_undefined_share"] = 0.9
    s.loc[(s["arm"] == "default") & (s["fold"] == "f3") & (s["algo"] == "A"), "lag_undefined_share"] = 0.5
    t = analyze.degenerate_table(s).set_index("arm")
    # default: 11 cells, 6 on f1/f2 above 0.5; 0.5 itself is not above the threshold
    assert t.loc["default", "n_cells"] == 11 and t.loc["default", "n_degenerate"] == 6
    assert t.loc["default", "share"] == pytest.approx(6 / 11)
    assert t.loc["confstream", "n_degenerate"] == 0
    assert analyze.degenerate_table(s, threshold=0.4).set_index("arm").loc["default", "n_degenerate"] == 7


# --- friedman_nemenyi ----------------------------------------------------------------------

def test_friedman_nemenyi_returns_cd_for_three_methods(summary):
    out = analyze.friedman_nemenyi(analyze.rank_matrix(summary, "confstream"))
    assert out["cd"] == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 4)))
    assert set(out["avg_rank"].index) == {"A", "B", "C"} and 0 <= out["pvalue"] <= 1


def test_friedman_nemenyi_average_ranks_and_statistic(summary):
    out = analyze.friedman_nemenyi(analyze.rank_matrix(summary, "confstream"))
    assert out["avg_rank"].to_dict() == {"A": 1.0, "B": 2.0, "C": 3.0}
    # every block ranks A, B, C: chi2 = 12 N / (k (k+1)) * sum(R_j^2) - 3 N (k+1) = 8
    assert out["statistic"] == pytest.approx(8.0) and out["pvalue"] == pytest.approx(np.exp(-4.0))
    assert out["n_folds"] == 4 and out["k"] == 3


def test_friedman_is_undefined_for_two_methods_but_cd_is_not(summary):
    ranks = analyze.rank_matrix(summary[summary["algo"] != "C"], "confstream")
    out = analyze.friedman_nemenyi(ranks)
    assert np.isnan(out["statistic"]) and np.isnan(out["pvalue"])
    assert out["cd"] == pytest.approx(1.960 * np.sqrt(2 * 3 / (6 * 4)))


def test_friedman_cd_is_nan_outside_the_table():
    ranks = pd.DataFrame(
        [list(range(1, 12)), list(range(11, 0, -1))], index=["f1", "f2"],
        columns=[f"a{i:02d}" for i in range(11)], dtype=float,
    )
    out = analyze.friedman_nemenyi(ranks)
    assert out["k"] == 11 and np.isnan(out["cd"]) and 0 <= out["pvalue"] <= 1


def test_friedman_of_empty_ranks_is_undefined_not_an_error():
    out = analyze.friedman_nemenyi(pd.DataFrame())
    assert np.isnan(out["statistic"]) and np.isnan(out["cd"]) and out["avg_rank"].empty
    assert out["n_folds"] == 0 and out["k"] == 0


# --- win_tie_loss --------------------------------------------------------------------------

def test_win_tie_loss_uses_the_noise_band(summary):
    w = analyze.win_tie_loss(summary, "confstream", "lag_ari_win_tail_mean").set_index("algo")
    assert w.loc["A"].tolist()[:3] == [3, 1, 0]      # one fold inside the band


def test_win_tie_loss_columns_and_worst_gap(summary):
    w = analyze.win_tie_loss(summary, "confstream")
    assert w.columns.tolist() == ["algo", "wins", "ties", "losses", "n_folds", "worst_gap"]
    w = w.set_index("algo")
    assert w.loc["A", "worst_gap"] == pytest.approx(0.05) and w.loc["A", "n_folds"] == 4
    # C has no f4 row: three comparable folds; its f3 gap (0.1) equals the band -> a tie
    assert w.loc["C", "n_folds"] == 3 and w.loc["C"].tolist()[:3] == [2, 1, 0]


def test_win_tie_loss_counts_a_loss_and_reads_a_missing_sd_as_zero(summary):
    s = summary.drop(columns=[TAIL_SD])
    s.loc[(s["arm"] == "confstream") & (s["fold"] == "f1") & (s["algo"] == "B"), TAIL] = 0.59
    w = analyze.win_tie_loss(s, "confstream").set_index("algo")
    assert w.loc["B"].tolist()[:3] == [3, 0, 1] and w.loc["B", "worst_gap"] == pytest.approx(-0.01)


def test_win_tie_loss_of_an_empty_summary_has_the_declared_columns():
    w = analyze.win_tie_loss(pd.DataFrame(columns=["algo", "fold", "arm"]), "confstream")
    assert w.empty and w.columns.tolist() == ["algo", "wins", "ties", "losses", "n_folds", "worst_gap"]


# --- pareto_table --------------------------------------------------------------------------

def test_pareto_hv_is_within_the_unit_interval(summary):
    p = analyze.pareto_table(summary).set_index(["algo", "arm"])
    assert 0 <= p.loc[("A", "confstream"), "hv"] <= 1


def test_pareto_hv_extremes_on_a_single_fold():
    s = pd.DataFrame([
        {"algo": "X", "fold": "f1", "arm": "confstream", TAIL: 0.9, SPS: 1000.0},
        {"algo": "Y", "fold": "f1", "arm": "confstream", TAIL: 0.1, SPS: 10.0},
        {"algo": "Z", "fold": "f1", "arm": "confstream", TAIL: 0.5, SPS: 100.0},
    ])
    p = analyze.pareto_table(s).set_index(["algo", "arm"])
    assert p.loc[("X", "confstream"), "hv"] == pytest.approx(1.0)
    assert p.loc[("Y", "confstream"), "hv"] == pytest.approx(0.0)
    # Z sits in the middle of both axes (log10 sps: 3, 1, 2) -> 0.5 * 0.5
    assert p.loc[("Z", "confstream"), "hv"] == pytest.approx(0.25)


def test_pareto_hv_is_the_mean_over_folds_not_the_best_fold():
    # P wins both axes on f1 (point (1, 1)) and sits at the origin on f2 (point (0, 0)):
    # the typical-fold reading is 0.5; the union of the two points would read 1.0
    s = pd.DataFrame([
        {"algo": "P", "fold": "f1", "arm": "a", TAIL: 1.0, SPS: 1000.0},
        {"algo": "Q", "fold": "f1", "arm": "a", TAIL: 0.0, SPS: 10.0},
        {"algo": "P", "fold": "f2", "arm": "a", TAIL: 0.0, SPS: 10.0},
        {"algo": "Q", "fold": "f2", "arm": "a", TAIL: 1.0, SPS: 1000.0},
    ])
    p = analyze.pareto_table(s).set_index("algo")
    assert p.loc["P", "hv"] == pytest.approx(0.5) and p.loc["Q", "hv"] == pytest.approx(0.5)


def test_pareto_hv_averages_the_per_fold_areas():
    s = pd.DataFrame([
        {"algo": "P", "fold": "f1", "arm": "a", TAIL: 1.0, SPS: 10.0},
        {"algo": "Q", "fold": "f1", "arm": "a", TAIL: 0.0, SPS: 1000.0},
        {"algo": "P", "fold": "f2", "arm": "a", TAIL: 0.0, SPS: 1000.0},
        {"algo": "Q", "fold": "f2", "arm": "a", TAIL: 1.0, SPS: 10.0},
        {"algo": "P", "fold": "f3", "arm": "a", TAIL: 0.5, SPS: 100.0},
        {"algo": "Q", "fold": "f3", "arm": "a", TAIL: 0.0, SPS: 10.0},
        {"algo": "R", "fold": "f3", "arm": "a", TAIL: 1.0, SPS: 1000.0},
    ])
    p = analyze.pareto_table(s).set_index("algo")
    # P's normalised points: (1, 0), (0, 1), (0.5, 0.5) -> areas 0, 0, 0.25 -> mean 1/12
    assert p.loc["P", "hv"] == pytest.approx(0.25 / 3)
    assert p.loc["P", "n_folds"] == 3 and p.loc["P", "ari"] == pytest.approx(0.5)
    assert p.loc["P", "sps"] == pytest.approx(370.0)
    assert p.loc["R", "hv"] == pytest.approx(1.0) and p.loc["R", "n_folds"] == 1


def test_pareto_table_columns_bounds_and_empty(summary):
    p = analyze.pareto_table(summary)
    assert p.columns.tolist() == ["algo", "arm", "ari", "sps", "hv", "n_folds"]
    assert ((p["hv"] >= 0) & (p["hv"] <= 1)).all() and len(p) == 9
    e = analyze.pareto_table(pd.DataFrame(columns=["algo", "fold", "arm"]))
    assert e.empty and e.columns.tolist() == ["algo", "arm", "ari", "sps", "hv", "n_folds"]


# --- the seed gate (all_seeds_ok) -------------------------------------------------------------

def _gate(summary, arm, fold, algo):
    """``summary`` with ``all_seeds_ok`` False on one cell, and ``summary`` without that cell."""
    cell = (summary["arm"] == arm) & (summary["fold"] == fold) & (summary["algo"] == algo)
    return summary.assign(all_seeds_ok=~cell), summary[~cell]


@pytest.mark.parametrize("arm, fold, algo", [("confstream", "f1", "A"), ("default", "f2", "B")])
def test_a_cell_missing_a_seed_reads_as_no_reading_everywhere(summary, arm, fold, algo):
    gated, dropped = _gate(summary, arm, fold, algo)
    arms = ["default", "confstream", "oracle_static"]
    pd.testing.assert_frame_equal(analyze.rank_matrix(gated, arm), analyze.rank_matrix(dropped, arm))
    pd.testing.assert_frame_equal(analyze.kendall_matrix(gated, arms), analyze.kendall_matrix(dropped, arms))
    pd.testing.assert_frame_equal(analyze.win_tie_loss(gated, "confstream"),
                                  analyze.win_tie_loss(dropped, "confstream"))
    pd.testing.assert_frame_equal(analyze.pareto_table(gated), analyze.pareto_table(dropped))


def test_a_cell_missing_a_seed_is_ranked_last_and_excluded_from_wtl_and_pareto(summary):
    gated, _ = _gate(summary, "confstream", "f1", "A")
    assert analyze.rank_matrix(gated, "confstream").loc["f1", "A"] == 3.0      # was 1.0
    assert analyze.win_tie_loss(gated, "confstream").set_index("algo").loc["A", "n_folds"] == 3
    p = analyze.pareto_table(gated).set_index(["algo", "arm"])
    assert p.loc[("A", "confstream"), "n_folds"] == 3 and p.loc[("A", "default"), "n_folds"] == 4


def test_the_seed_gate_needs_an_explicit_false(summary):
    # None (unknown, e.g. an older summary) keeps the cell; only False drops it
    s = summary.assign(all_seeds_ok=None)
    pd.testing.assert_frame_equal(analyze.rank_matrix(s, "confstream"), analyze.rank_matrix(summary, "confstream"))


# --- regret figure -----------------------------------------------------------------------------

def test_regret_means_read_the_tail_region_of_one_arm():
    rows = [
        ("A", "f1", "confstream", "all", 1.0, 1.0, 1.0),
        ("A", "f1", "confstream", "tail", 0.1, 0.2, None),
        ("A", "f2", "confstream", "tail", 0.3, 0.4, None),
        ("B", "f1", "random_online", "tail", 5.0, 5.0, 5.0),
    ]
    regret = pd.DataFrame(rows, columns=["algo", "fold", "arm", "region", "R_lag", "R_proxy", "R_search"],
                          dtype=object)
    means = analyze.regret_means(regret, "confstream")
    assert list(means.index) == ["A"]
    assert means.loc["A", "R_lag"] == pytest.approx(0.2) and means.loc["A", "R_proxy"] == pytest.approx(0.3)
    assert np.isnan(means.loc["A", "R_search"])
    assert analyze.regret_means(regret, "frozen_prefix").empty


def test_regret_means_read_the_delayed_label_terms_the_figure_plots():
    rows = [("A", "f1", "confstream", "tail", 0.1, 0.5), ("A", "f2", "confstream", "tail", 0.3, -0.1)]
    regret = pd.DataFrame(rows, columns=["algo", "fold", "arm", "region", "R_search_lag", "R_proxy_lag"],
                          dtype=object)
    means = analyze.regret_means(regret, "confstream", terms=analyze.REGRET_TERMS_LAG)
    assert list(means.columns) == ["R_search_lag", "R_proxy_lag"]
    assert means.loc["A", "R_search_lag"] == pytest.approx(0.2) and means.loc["A", "R_proxy_lag"] == pytest.approx(0.2)


# --- fidelity ------------------------------------------------------------------------------

def test_fidelity_is_one_for_the_same_ranking_and_nan_below_three_algos(summary):
    full = summary.copy()
    full[TAIL] = full[TAIL] * 0.5 + 0.1        # monotone: same order of the per-algo means
    out = analyze.fidelity(summary, full, "confstream")
    assert out["n"] == 3 and out["spearman"] == pytest.approx(1.0)
    out = analyze.fidelity(summary[summary["algo"] != "C"], full, "confstream")
    assert out["n"] == 2 and np.isnan(out["spearman"])


def test_fidelity_compares_only_folds_present_in_both(summary):
    full = summary.copy()
    # full has only f1, where under default B > A > C
    full = full[full["fold"] == "f1"]
    out = analyze.fidelity(summary, full, "default")
    assert out["n"] == 3 and out["spearman"] == pytest.approx(1.0)


# --- write_analysis ------------------------------------------------------------------------

def _run(algo, fold, arm, seed, tail, **over):
    row = {
        "kind": "run", "algo": algo, "fold": fold, "seed": seed, "arm": arm, "status": "ok",
        "config_id": None, "n_samples": 4000, "h": 1000, "n_windows": 3, "drift_points": [],
        "incumbent_throughput_sps": 900.0 if arm == "default" else 150.0,
        "lag_ari_win_mean": tail - 0.02, TAIL: tail, "lag_score_mean": 0.3, "tail_from": 2,
    }
    row.update(over)
    return row


def _window(algo, fold, arm, seed, w, lag):
    return {
        "kind": "window", "algo": algo, "fold": fold, "seed": seed, "arm": arm, "window": w,
        "config_id": None, "ari_win": lag + 0.05, "lag_ari_win": lag, "score": 0.4,
        "lag_score": 0.3, "n_active": 1 if arm == "default" else 4, "params": {},
    }


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_out(path: Path, bump: float = 0.0) -> Path:
    """Two algos x three folds (one of them a drift fold) x default/confstream, two seeds."""
    tails = {
        ("Alpha", "covtype"): (0.20, 0.40), ("Beta", "covtype"): (0.30, 0.35),
        ("Alpha", "kdd99"): (0.50, 0.60), ("Beta", "kdd99"): (0.40, 0.45),
        ("Alpha", "scale_drift"): (0.60, 0.70), ("Beta", "scale_drift"): (0.10, 0.50),
    }
    runs, windows = [], []
    for (algo, fold), (d, c) in tails.items():
        extra = {"drift_points": [2000]} if fold == "scale_drift" else {}
        for arm, tail in (("default", d), ("confstream", c)):
            for seed in (1, 2):
                runs.append(_run(algo, fold, arm, seed, tail + bump + 0.01 * seed, **extra))
                windows += [_window(algo, fold, arm, seed, w, tail) for w in (1, 2, 3)]
    _write_jsonl(path / "runs.jsonl", runs)
    _write_jsonl(path / "windows.jsonl", windows)
    return path


@pytest.fixture
def summary_dir(tmp_path):
    return _write_out(tmp_path / "cap")


def test_write_analysis_produces_md_and_figures(summary_dir):
    paths = analyze.write_analysis(summary_dir)
    names = {p.name for p in paths}
    assert "analysis.md" in names and "F2_kendall.png" in names and "F1_heatmap_confstream.png" in names


def test_write_analysis_writes_every_table_and_figure(summary_dir):
    paths = analyze.write_analysis(summary_dir)
    names = {p.name for p in paths}
    assert {
        "ranks_default.csv", "ranks_confstream.csv", "kendall.csv", "friedman.csv",
        "win_tie_loss_confstream.csv", "pareto.csv", "F1_heatmap_default.png",
        "F3_regret.png", "F4_drift_scale_drift.png",
    } <= names
    assert "fidelity.csv" not in names and "win_tie_loss_default.csv" not in names
    assert all(p.exists() and p.parent == summary_dir / "report" for p in paths)
    md = (summary_dir / "report" / "analysis.md").read_text()
    for section in ("Rankings by arm", "Ranking agreement between arms (Kendall τ)",
                    "Win/tie/loss vs default", "Pareto (ARI vs throughput)", "Figures"):
        assert f"## {section}" in md
    assert "Fidelity cap vs full" not in md
    assert "all_seeds_ok" in md.split("## Rankings by arm")[1].split("\n## ")[0]
    kendall = pd.read_csv(summary_dir / "report" / "kendall.csv", index_col=0)
    assert kendall.loc["default", "confstream"] == pytest.approx(1 / 3)


def test_write_analysis_adds_the_complete_case_kendall_and_the_degenerate_count(summary_dir):
    runs = [json.loads(line) for line in (summary_dir / "runs.jsonl").read_text().splitlines()]
    for r in runs:
        r["lag_undefined_share"] = 1.0 if (r["arm"], r["algo"]) == ("default", "Beta") else 0.0
    _write_jsonl(summary_dir / "runs.jsonl", runs)
    names = {p.name for p in analyze.write_analysis(summary_dir)}
    assert {"kendall_complete.csv", "degenerate.csv", "F2_kendall_complete.png"} <= names
    rep = summary_dir / "report"
    both = pd.read_csv(rep / "kendall_complete.csv", index_col=0)
    assert both.loc["default", "confstream"] == pytest.approx(1 / 3)   # every cell has both arms
    deg = pd.read_csv(rep / "degenerate.csv").set_index("arm")
    assert deg.loc["default", "n_degenerate"] == 3 and deg.loc["confstream", "n_degenerate"] == 0
    section = (rep / "analysis.md").read_text().split("## Ranking agreement between arms (Kendall τ)")[1]
    section = section.split("\n## ")[0]
    assert "complete" in section and "degenerate" in section


def test_write_analysis_ranks_a_cell_missing_a_seed_last(summary_dir):
    runs = [json.loads(line) for line in (summary_dir / "runs.jsonl").read_text().splitlines()]
    runs = [r for r in runs if (r["algo"], r["fold"], r["arm"], r["seed"]) != ("Alpha", "covtype", "confstream", 2)]
    _write_jsonl(summary_dir / "runs.jsonl", runs)
    summary = report.summary_table(summary_dir)
    # Alpha (0.40) beats Beta (0.35) on covtype, but has one of confstream's two seeds
    ranks = analyze.rank_matrix(summary, "confstream")
    assert ranks.loc["covtype", "Alpha"] == 2.0 and ranks.loc["covtype", "Beta"] == 1.0
    analyze.write_analysis(summary_dir)
    written = pd.read_csv(summary_dir / "report" / "ranks_confstream.csv", index_col=0)
    assert written.loc["covtype", "Alpha"] == 2.0


def test_write_analysis_with_a_full_run_adds_fidelity(summary_dir, tmp_path):
    full = _write_out(tmp_path / "full", bump=0.1)
    paths = analyze.write_analysis(summary_dir, full_out_dir=full)
    assert "fidelity.csv" in {p.name for p in paths}
    assert "## Fidelity cap vs full" in (summary_dir / "report" / "analysis.md").read_text()
    fid = pd.read_csv(summary_dir / "report" / "fidelity.csv")
    assert set(fid["arm"]) == {"default", "confstream"} and (fid["n"] == 2).all()


def test_write_analysis_without_matplotlib_still_writes_tables(summary_dir, monkeypatch):
    monkeypatch.setattr(analyze, "_pyplot", lambda: None)
    paths = analyze.write_analysis(summary_dir)
    names = {p.name for p in paths}
    assert "analysis.md" in names and not any(n.endswith(".png") for n in names)


def test_write_analysis_on_an_empty_out_dir(tmp_path):
    paths = analyze.write_analysis(tmp_path)
    names = {p.name for p in paths}
    assert "analysis.md" in names and "F3_regret.png" not in names
    assert not any(n.startswith("F4_") for n in names)


# --- implementation variants (spec 4.6, table T7) -----------------------------------------------

@pytest.fixture
def with_variant(summary, monkeypatch):
    """B re-implemented as Bfast: equal to B everywhere except confstream/f1, where it leads."""
    monkeypatch.setattr(analyze.C, "IMPL_VARIANTS", {"Bfast": "B"})
    rows = summary[summary["algo"] == "B"].assign(algo="Bfast")
    rows.loc[(rows["arm"] == "confstream") & (rows["fold"] == "f1"), TAIL] = 0.85
    return pd.concat([summary, rows], ignore_index=True)


def test_an_implementation_variant_is_ranked_with_nobody(summary, with_variant):
    assert analyze.rank_matrix(with_variant, "confstream").columns.tolist() == ["A", "B", "C"]
    pd.testing.assert_frame_equal(analyze.kendall_matrix(with_variant, ["default", "confstream"]),
                                  analyze.kendall_matrix(summary, ["default", "confstream"]))
    assert analyze.win_tie_loss(with_variant, "confstream")["algo"].tolist() == ["A", "B", "C"]
    pd.testing.assert_frame_equal(analyze.pareto_table(with_variant), analyze.pareto_table(summary))
    assert analyze.fidelity(with_variant, with_variant, "confstream")["n"] == 3


def test_impl_table_reads_each_variant_against_its_reference(with_variant):
    t = analyze.impl_table(with_variant).set_index(["arm", "fold"])
    f1 = t.loc[("confstream", "f1")]
    assert (f1["variant"], f1["reference"]) == ("Bfast", "B")
    assert f1["delta"] == pytest.approx(0.25)
    assert (f1["rank_reference"], f1["rank_variant"], f1["rank_change"]) == (2.0, 1.0, 1.0)
    rest = t.drop(index=("confstream", "f1"))
    assert (rest["delta"].abs() < 1e-12).all() and (rest["rank_change"] == 0).all()
    assert len(t) == 12            # 3 arms x 4 folds where both B and Bfast have a reading


def test_impl_table_without_a_variant_is_empty_with_its_columns(summary):
    assert analyze.impl_table(summary).columns.tolist() == analyze.IMPL_COLS
    assert analyze.impl_table(summary).empty


def test_write_analysis_reports_the_variant_in_its_own_section(summary_dir, monkeypatch):
    monkeypatch.setattr(analyze.C, "IMPL_VARIANTS", {"BetaFast": "Beta"})
    runs = [json.loads(line) for line in (summary_dir / "runs.jsonl").read_text().splitlines()]
    _write_jsonl(summary_dir / "runs.jsonl",
                 runs + [{**r, "algo": "BetaFast"} for r in runs if r["algo"] == "Beta"])
    paths = analyze.write_analysis(summary_dir)
    assert "impl_T7.csv" in {p.name for p in paths}
    md = (summary_dir / "report" / "analysis.md").read_text()
    assert "## Implementation variants (T7)" in md
    assert pd.read_csv(summary_dir / "report" / "ranks_confstream.csv", index_col=0).columns.tolist() == ["Alpha", "Beta"]
