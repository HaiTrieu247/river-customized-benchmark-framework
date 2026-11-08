"""conf_pipeline.followup: the log-only follow-up readings (E2, E3, E6-E10) on small synthetic out dirs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conf_pipeline import _common as C
from conf_pipeline import followup as F
from conf_pipeline import robustness as R

TAIL = "lag_ari_win_tail_mean"
FOLDS = ("covtype", "k_drift", "evolving")  # one fold per stream group


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _run(algo, fold, arm, seed, **over):
    tuned = arm not in C.FIXED_ARMS
    row = {"kind": "run", "algo": algo, "fold": fold, "seed": seed, "arm": arm, "config_id": None, "status": "ok",
           "h": 1000, "tail_from": 20, "e_size": 4 if tuned else 1, "e_test": 2 if tuned else 0,
           "freeze_after_window": 20 if arm == "frozen_prefix" else None,
           "frozen_at_window": 20 if arm == "frozen_prefix" else None, "drift_points": [], "runtime_factor": 30.0,
           "max_memory_mb": 2048.0, "max_micro_clusters": 10000}
    row.update(over)
    return row


def _win(algo, fold, arm, seed, w, lag, **over):
    row = {"kind": "window", "algo": algo, "fold": fold, "seed": seed, "arm": arm, "config_id": None, "window": w,
           "incumbent_id": 0, "lag_member_id": 0 if w > 1 else None, "lag_ari_win": lag if w > 1 else None,
           "ari_win": lag}
    row.update(over)
    return row


def _mem(algo, fold, arm, seed, w, mid, ari=0.5, score=0.5, **over):
    row = {"algo": algo, "fold": fold, "seed": seed, "arm": arm, "config_id": None, "member_id": mid, "window": w,
           "is_test": False, "born_window": 0, "params": {"x": 1.0}, "score": score, "silhouette_defined": True,
           "feasible": True, "feasible_reason": None, "error": None, "runtime_s": 0.01, "memory_mb": 0.1,
           "n_micro_clusters": 5, "n_sentinel": 0, "n_pred_labels": 3, "n_true_classes": 3, "ari_win": ari}
    row.update(over)
    return row


def _lag(arm, w):
    """confstream reads 0.5 on every window; frozen_prefix 0.5 up to t0 + 1 = 21 and 0.2 after; default 0.1."""
    if arm == "default":
        return 0.1
    if arm == "frozen_prefix":
        return 0.5 if w <= 21 else 0.2
    return 0.5


def _post_freeze_dir(tmp_path: Path, members: bool = False) -> Path:
    runs, wins, mems = [], [], {}
    for algo in ("DenStream", "BIRCH"):
        for fold in FOLDS:
            for seed in (1, 2):
                for arm in ("default", "confstream", "frozen_prefix"):
                    lags = {w: _lag(arm, w) for w in range(1, 30)}
                    runs.append(_run(algo, fold, arm, seed, **{TAIL: float(np.mean([lags[w] for w in range(20, 30)]))}))
                    wins += [_win(algo, fold, arm, seed, w, lags[w]) for w in range(1, 30)]
                    if members and arm != "default":
                        mems[f"{algo}__{fold}__seed{seed}__{arm}"] = [
                            _mem(algo, fold, arm, seed, w, mid) for w in range(1, 30)
                            for mid in ((0, 1) if arm == "confstream" or w <= 20 else (0,))]
    _write(tmp_path / "runs.jsonl", runs)
    _write(tmp_path / "windows.jsonl", wins)
    for name, rows in mems.items():
        _write(tmp_path / "members" / f"{name}.jsonl", rows)
    return tmp_path


# --- E2: post-freeze ---------------------------------------------------------------------------------

def test_default_starts_are_the_freeze_window_and_the_two_after_it():
    runs = pd.DataFrame([_run("DenStream", "covtype", "confstream", 1)])
    specs = F.start_specs(runs)
    assert [label for label, _ in specs] == ["t>=20", "t>=21", "t>=22"]
    assert [list(spec.values())[0] for _, spec in specs] == [20, 21, 22]
    assert F.start_specs(runs, starts=[25]) == [("t>=25", 25)]


def test_post_freeze_contrasts_read_every_start_and_the_paired_difference(tmp_path):
    out = _post_freeze_dir(tmp_path)
    runs = F._ok_runs(out)
    windows = R.current_windows(out, runs)
    con, per_run = F.post_freeze_contrasts(windows, runs, F.start_specs(runs), n_boot=50)
    pooled = con[(con["algo"] == "pooled") & (con["group"] == "all") & (con["unit"] == "fold")].set_index(
        ["contrast", "start"])["point"]
    # frozen reads 0.5 on windows 20 and 21, then 0.2: its mean is 0.26, 0.2333 and 0.2 from t0, t0+1, t0+2
    assert pooled[("confstream - frozen_prefix", "t>=20")] == pytest.approx(0.5 - 0.26)
    assert pooled[("confstream - frozen_prefix", "t>=21")] == pytest.approx(0.5 - (0.5 + 8 * 0.2) / 9)
    assert pooled[("confstream - frozen_prefix", "t>=22")] == pytest.approx(0.3)
    assert pooled[("confstream - frozen_prefix", "t>=22 minus t>=20")] == pytest.approx(0.3 - 0.24)
    assert pooled[("confstream - default", "t>=22")] == pytest.approx(0.4)
    assert not con["contrast"].str.contains("oracle_static").any()  # no reference run in the dir: no row
    assert set(per_run["start"]) == {"t>=20", "t>=21", "t>=22"}


def test_contrast_reproduces_robustness_unit_statistics_on_the_published_cube():
    rng = np.random.default_rng(3)
    rows = []
    for algo in ("A", "B"):
        for fold in ("covtype", "kdd99", "k_drift", "evolving"):
            base = rng.random()
            for arm in R.ARMS:
                seeds = (2, 3, 4) if arm == "oracle_static" else (1, 2, 3)
                rows += [{"algo": algo, "fold": fold, "arm": arm, "seed": s, "value": base + rng.random() * 0.1}
                         for s in seeds]
    seeds = pd.DataFrame(rows)
    st = R.unit_statistics(R.Construction("x", "ari", "x", seeds, F._NO_LOSS, {}), "block", 100, 0)
    mine = F.contrast(seeds, "frozen_prefix", "block", 100, 0)
    ref = st[st["statistic"] == "confstream - frozen_prefix"].reset_index(drop=True)
    assert np.allclose(mine[["point", "low", "high"]].to_numpy(float), ref[["point", "low", "high"]].to_numpy(float))
    # an arm outside the published comparisons is read in default's place
    other = F.contrast(seeds.assign(arm=seeds["arm"].replace({"default": "frozen_reinit"})), "frozen_reinit", "fold", 50, 0)
    base = F.contrast(seeds, "default", "fold", 50, 0)
    assert np.allclose(other["point"].to_numpy(float), base["point"].to_numpy(float))


def test_post_freeze_writes_only_its_directory_and_says_the_lifecycle_is_missing(tmp_path):
    out = _post_freeze_dir(tmp_path, members=True)
    before = {p for p in out.rglob("*") if p.is_file()}
    F.main(["post-freeze", "--out-dir", str(out), "--n-boot", "20", "--workers", "1"])
    new = {p for p in out.rglob("*") if p.is_file()} - before
    assert new and all(p.parent == out / "report" / "followup" / "post-freeze" for p in new)
    md = (out / "report" / "followup" / "post-freeze" / "post-freeze.md").read_text()
    assert "`lifecycle/` not found" in md
    assert "t >= t0 + 2 is the strictly post-freeze region" in md
    assert "`lag_pred_sha`" in md and "not checked on predictions here" in md
    audit = pd.read_csv(out / "report" / "followup" / "post-freeze" / "timing_audit.csv")
    assert (audit["n_violations"] == 0).all()


def test_member_timing_checks_catch_early_scoring_and_a_broken_lag_link(tmp_path):
    rows = [_mem("DenStream", "covtype", "confstream", 1, w, 0) for w in (1, 2, 3)]
    rows += [_mem("DenStream", "covtype", "confstream", 1, 2, 5, is_test=True, born_window=2),  # scored on birth
             _mem("DenStream", "covtype", "confstream", 1, 3, 6, is_test=True, born_window=1)]  # not birth + 1
    _write(tmp_path / "members" / "DenStream__covtype__seed1__confstream.jsonl", rows)
    m = F.load_members(tmp_path, workers=1)
    wrows = pd.DataFrame({"job": m["job"].iat[0], "window": [1, 2, 3], "incumbent_id": [0, 0, 0],
                          "lag_member_id": [None, 0, 9]})
    got = {r["check"]: r["n_violations"] for r in F.member_timing_checks(m, wrows)}
    assert got["no member scored on or before its birth window"] == 1
    assert got["a member on test is scored on its birth window + 1"] == 2
    assert got["the lagged member read on t + 1 is the incumbent named on t"] == 1
    assert got["no member has two rows on one window"] == 0


def test_lifecycle_checks_flag_an_early_challenge_an_early_incumbent_and_life_after_a_freeze():
    job = "DenStream|covtype|1|frozen_prefix|None"
    ev = pd.DataFrame([
        {"job": job, "event": "birth", "window": 0, "member_id": 0, "origin": "start"},
        {"job": job, "event": "birth", "window": 1, "member_id": 3, "origin": "evolutionary"},
        {"job": job, "event": "challenge", "window": 1, "challenger_id": 3, "outcome": "accepted"},  # at its birth
        {"job": job, "event": "decision", "window": 0, "incumbent_id": 3},  # before its admission on window 1
        {"job": job, "event": "decision", "window": 1, "incumbent_id": 0},
        {"job": job, "event": "freeze", "window": 2, "member_id": 0, "reinit_after_freeze": False},
        {"job": job, "event": "birth", "window": 3, "member_id": 7, "origin": "evolutionary"},
        {"job": job, "event": "decision", "window": 3, "incumbent_id": 7},
    ])
    got = {r["check"]: r["n_violations"] for r in F.lifecycle_checks(ev, pd.DataFrame(), pd.DataFrame())}
    assert got["a challenger is challenged on its birth window + 1, never earlier"] == 1
    assert got["no incumbent before the window of its admission (read on the window after it)"] == 2
    assert got["a one-member freeze has no birth after it"] == 1
    assert got["a one-member freeze names its member at every later decision"] == 1
    assert got["every member is born once"] == 0


# --- E3: prefix identity -------------------------------------------------------------------------------

def _branch_dir(tmp_path: Path, sha: bool = True) -> Path:
    runs, wins = [], []
    for seed in (1, 2):
        for arm in ("confstream", "frozen_prefix"):
            runs.append(_run("DenStream", "covtype", arm, seed, freeze_after_window=3 if arm != "confstream" else None,
                             frozen_at_window=3 if arm != "confstream" else None, tail_from=3))
            rows = []
            for w in range(1, 6):
                for mid in (0, 1):
                    diverge = (seed == 2 and arm == "frozen_prefix" and w == 2 and mid == 1)
                    after = arm == "frozen_prefix" and w > 3
                    rows.append(_mem("DenStream", "covtype", arm, seed, w, mid,
                                     pred_sha=(f"s{w}{mid}{'x' if diverge or after else ''}" if sha else None),
                                     feasible=not diverge, feasible_reason="runtime" if diverge else None,
                                     score=-1.0 if diverge else 0.5))
                wins.append(_win("DenStream", "covtype", arm, seed, w, 0.5,
                                 lag_pred_sha=(None if w == 1 else f"L{w}{'x' if arm == 'frozen_prefix' and w >= 5 else ''}")))
            _write(tmp_path / "members" / f"DenStream__covtype__seed{seed}__{arm}.jsonl", rows)
    _write(tmp_path / "runs.jsonl", runs)
    _write(tmp_path / "windows.jsonl", wins)
    return tmp_path


def test_prefix_identity_certifies_a_shared_prefix_and_finds_the_first_divergent_window(tmp_path):
    out = _branch_dir(tmp_path)
    runs = F._ok_runs(out, protocol=False)
    t = F.prefix_table(F.load_members(out, workers=1), F._window_rows(out), runs).set_index("seed_key")
    assert t.loc[1, "sha_status"] == "logged" and bool(t.loc[1, "identical_through_freeze"])
    assert pd.isna(t.loc[1, "sha_first_divergent"]) and t.loc[1, "n_windows"] == 3  # windows after F not compared
    assert t.loc[2, "sha_first_divergent"] == 2 and not bool(t.loc[2, "identical_through_freeze"])
    assert t.loc[2, "readings_first_divergent"] == 2 and t.loc[2, "readings_cause"] == "runtime gate"
    # lagged predictions: identical through t0 + 1 = 4, first differ on t0 + 2 = 5
    sha = F.lag_sha_divergence(F._window_rows(out), runs)
    assert set(sha["first_divergent_window"]) == {5} and sha["identical_through_freeze_plus_1"].all()
    assert set(F.identical_prefix_pairs(F.prefix_table(F.load_members(out, workers=1), F._window_rows(out),
                                                       runs))["frozen_prefix"]) == {("DenStream", "covtype", 1)}


def test_a_tie_broken_differently_parts_branches_whose_readings_agree(tmp_path):
    out = _branch_dir(tmp_path)
    runs = F._ok_runs(out, protocol=False)
    wrows = F._window_rows(out)
    wrows.loc[(wrows["arm"] == "frozen_prefix") & (wrows["seed"] == 1) & (wrows["window"] == 3), "incumbent_id"] = 1
    t = F.prefix_table(F.load_members(out, workers=1), wrows, runs).set_index("seed_key")
    assert pd.isna(t.loc[1, "readings_first_divergent"]) and t.loc[1, "first_divergent"] == 3
    assert t.loc[1, "cause"] == F.TIE_CAUSE and not bool(t.loc[1, "identical_through_freeze"])
    assert t.loc[2, "first_divergent"] == 2 and t.loc[2, "cause"] == "runtime gate"
    assert "frozen_prefix" not in F.identical_prefix_pairs(t.reset_index())


def test_prefix_identity_without_pred_sha_reports_not_logged(tmp_path):
    out = _branch_dir(tmp_path, sha=False)
    F.cmd_prefix_identity(out, workers=1)
    md = (out / "report" / "followup" / "prefix-identity" / "prefix-identity.md").read_text()
    assert "**not logged**" in md
    t = pd.read_csv(out / "report" / "followup" / "prefix-identity" / "prefix.csv")
    assert set(t["sha_status"]) == {"not logged"}
    assert F.lag_sha_divergence(pd.DataFrame({"job": [], "window": []}), F._ok_runs(out)).empty


# --- E7: survivors ---------------------------------------------------------------------------------------

def test_survivors_sort_every_eligible_member_and_find_a_lost_best(tmp_path):
    a = ("DenStream", "covtype", "confstream", 1)
    rows = [
        # window 1: active 0, 1, 2 and challenger 3; 2 has the best ARI and is replaced by 3 after window 1
        _mem(*a, 1, 0, ari=0.5), _mem(*a, 1, 1, ari=0.8), _mem(*a, 1, 2, ari=0.9),
        _mem(*a, 1, 3, ari=0.4, is_test=True, born_window=0),
        # window 2: 1 raised (no ARI), challenger 4 is rejected after it
        _mem(*a, 2, 0, ari=0.6), _mem(*a, 2, 1, ari=None, error="boom", feasible=False, feasible_reason="error"),
        _mem(*a, 2, 3, ari=0.7), _mem(*a, 2, 4, ari=0.3, is_test=True, born_window=1),
        # window 3: one true class, no ARI for anyone
        *[_mem(*a, 3, mid, ari=None, n_true_classes=1) for mid in (0, 1, 3)],
    ]
    _write(tmp_path / "members" / "DenStream__covtype__seed1__confstream.jsonl", rows)
    runs = pd.DataFrame([_run(*a, tail_from=2)])
    wrows = pd.DataFrame({"job": "DenStream|covtype|1|confstream|None", "window": [1, 2, 3], "incumbent_id": [0, 0, 0]})
    e, wins = F.survivor_rows(F.load_members(tmp_path, workers=1), runs, wrows)
    cat = e.set_index(["window", "member_id"])["category"].to_dict()
    assert cat == {(1, 0): "reading", (1, 1): "error_next", (1, 2): "replaced", (1, 3): "reading",
                   (2, 0): "single_class_next", (2, 3): "single_class_next", (2, 4): "challenger_not_admitted"}
    assert F.CATEGORIES["replaced"] == "decision after t" and F.CATEGORIES["error_next"] == "on t+1"
    w = wins.set_index("window")
    assert bool(w.loc[1, "best_lost"]) and w.loc[1, "best_gap"] == pytest.approx(0.1)
    assert bool(w.loc[1, "dropped"])  # the pool's best (member 1) has no ARI on window 2
    assert bool(w.loc[2, "dropped"])  # single-class window 3
    s = F.survivor_summary(e, wins)
    row = s[(s["region"] == "all") & (s["level"] == "group") & (s["unit"] == "real")].iloc[0]
    assert row["share_replaced"] == pytest.approx(1 / 7) and row["share_decided_after_t"] == pytest.approx(2 / 7)


def test_survivors_without_member_logs_say_so(tmp_path):
    _write(tmp_path / "runs.jsonl", [_run("DenStream", "covtype", "confstream", 1)])
    F.cmd_survivors(tmp_path, workers=1)
    assert "No member log" in (tmp_path / "report" / "followup" / "survivors" / "survivors.md").read_text()


# --- E7: lagged validity -----------------------------------------------------------------------------------

def test_group_spearman_matches_scipy_and_applies_its_gates():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.repeat([1, 2, 3, 4], 6), "x": rng.integers(0, 4, 24).astype(float),
                       "y": rng.random(24)})
    df.loc[df["g"] == 3, "y"] = 0.5  # constant: undefined
    df.loc[df["g"] == 4, "y"] = np.linspace(0.5, 0.52, 6)  # spread 0.02 < 0.05: undefined
    out = F.group_spearman(df, ["g"], "x", "y", min_spread=0.05).set_index("g")
    for g in (1, 2):
        sub = df[df["g"] == g]
        assert out.loc[g, "rho"] == pytest.approx(spearmanr(sub["x"], sub["y"]).statistic)
    assert pd.isna(out.loc[3, "rho"]) and pd.isna(out.loc[4, "rho"])
    two = F.group_spearman(df[df["g"] == 1].head(2), ["g"], "x", "y")
    assert not two["defined"].any()


def test_window_rhos_read_lagged_and_same_window_correlations_on_the_same_members():
    rows = []
    for mid, (score, ari, nxt) in enumerate([(0.1, 0.9, 0.1), (0.2, 0.5, 0.4), (0.3, 0.2, 0.8)]):
        rows.append({"job": "j", "algo": "DenStream", "fold": "covtype", "seed": 1, "arm": "confstream",
                     "window": 21, "member_id": mid, "score": score, "ari_win": ari, "next_ari": nxt, "tail": True})
    r = F.window_rhos(pd.DataFrame(rows)).iloc[0]
    assert r["rho_lag"] == pytest.approx(1.0) and r["rho_same"] == pytest.approx(-1.0)
    assert r["rho_diff_paired"] == pytest.approx(2.0)


def test_period_labels_mark_the_pairs_across_a_boundary():
    bounds = [(33333, 33, True), (66666, 66, True)]
    assert F.period_label(10, bounds) == "segment 0"
    assert F.period_label(32, bounds) == "across drift 1"
    assert F.period_label(33, bounds) == "across drift 1"  # its window holds the change inside it
    assert F.period_label(34, bounds) == "segment 1"
    assert F.period_label(70, bounds) == "segment 2"
    exact = [(50000, 50, False)]
    assert F.period_label(49, exact) == "across drift 1" and F.period_label(50, exact) == "segment 1"


def test_unit_bootstrap_averages_cells_within_units_then_units():
    cells = pd.DataFrame({"algo": "A", "fold": ["covtype", "insects_abrupt_balanced", "insects_gradual_balanced"],
                          "value": [0.2, 0.4, 0.6]})
    fold = F.unit_bootstrap(cells, unit="fold", n_boot=50).set_index(["group", "algo"])
    block = F.unit_bootstrap(cells, unit="block", n_boot=50).set_index(["group", "algo"])
    assert fold.loc[("real", "pooled"), "point"] == pytest.approx(0.4)
    assert block.loc[("real", "pooled"), "point"] == pytest.approx((0.2 + 0.5) / 2)
    assert block.loc[("real", "pooled"), "n_units"] == 2
    assert pd.isna(block.loc[("param_drift", "pooled"), "point"])


# --- E8: recovery ------------------------------------------------------------------------------------------

def test_recovery_delay_and_censoring():
    lag = {10: 0.2, 11: 0.5, 12: 0.8, 13: 0.8, 14: 0.8, 15: 0.3}
    assert F._recovery(lag, 10, 20, 0.8) == 2
    assert F._recovery(lag, 10, 14, 0.8) is None  # needs windows 12-14 < next boundary 14: censored
    assert F._recovery(lag, 10, 20, None) is None
    ev = F.recovery_events(pd.DataFrame([
        {"algo": "A", "fold": "k_drift", "arm": "confstream", "boundary": 1, "tw": 1, "straddles": True, "level": 0.5,
         "pre": 0.4, "shortfall": 0.0, "rec_static": d, "rec_pre": 0} for d in (1, None, None)]))
    assert bool(ev["censored_static"].iat[0]) and ev["n_recovered_static"].iat[0] == 1
    assert ev["delays_static"].iat[0] == "1,censored,censored" and not bool(ev["censored_pre"].iat[0])


def test_recovery_rows_drop_the_straddling_window_from_every_reading():
    win = pd.DataFrame([{"algo": "DenStream", "fold": "k_drift", "arm": "confstream", "seed": 1, "window": w,
                         "lag_ari_win": v} for w, v in zip(range(1, 9), [0.9, 0.9, 0.95, 0.9, 0.9, 0.9, 0.9, 0.9])])
    sweep = pd.DataFrame([{"algo": "DenStream", "fold": "k_drift", "seed": 1, "config_id": c, "window": w,
                           "lag_ari_win": v} for c, vals in ((0, [0.5] * 8), (1, [0.9, 0.9, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9]))
                          for w, v in zip(range(1, 9), vals)])
    bounds = {"k_drift": [(3500, 3, True)]}
    kept = F.recovery_rows(win, sweep, bounds).iloc[0]
    dropped = F.recovery_rows(win, sweep, bounds, drop_straddling=True).iloc[0]
    assert kept["level"] == pytest.approx(np.mean([0.0, 0.9, 0.9, 0.9, 0.9, 0.9]))
    assert dropped["level"] == pytest.approx(0.9)
    assert kept["rec_static"] == 0 and dropped["rec_static"] == 1
    assert dropped["n_segment_windows"] == kept["n_segment_windows"] - 1


def test_recovery_check_parses_the_printed_sections():
    text = "\n".join([
        "======== hồi phục: trong 0,05 của mức best static từng đoạn, 3 cửa sổ liền (seed trung vị) ========",
        "          arm  sự_kiện  không_hồi_phục  trung_vị  lớn_nhất",
        "   confstream       56               0       1.0       9.0",
        "======== hồi phục: trong 0,05 của ARI trước drift của chính chế độ đó ========",
        "          arm  sự_kiện  không_hồi_phục  trung_vị  lớn_nhất",
        "   confstream       56               2       1.0      12.0",
        "======== ARI trung bình của đoạn trừ mức best static của đoạn ========",
        "fold           density_drift  k_drift",
        "arm",
        "confstream             0.035    0.125",
    ])
    summ = {"static": pd.DataFrame([{"arm": "confstream", "events": 56, "censored": 0, "median_delay": 1.0,
                                     "max_delay": 9.0}]),
            "pre": pd.DataFrame([{"arm": "confstream", "events": 56, "censored": 3, "median_delay": 1.0,
                                  "max_delay": 12.0}])}
    short = pd.DataFrame({"density_drift": [0.0351], "k_drift": [0.2]}, index=["confstream"])
    chk = F.recovery_check(text, summ, short)
    assert len(chk) == 10
    bad = set(chk.loc[~chk["match"], "item"])
    assert bad == {"confstream censored", "confstream k_drift"}


# --- E6: equivalence ------------------------------------------------------------------------------------------

def test_tost_verdicts():
    assert F.tost_verdict(-0.01, 0.03, 0.05) == "equivalent"
    assert F.tost_verdict(0.06, 0.2, 0.05) == "different beyond the margin"
    assert F.tost_verdict(0.01, 0.07, 0.05) == "not shown equivalent"
    assert F.tost_verdict(np.nan, 0.1, 0.05) == "undefined"


def test_equivalence_uses_a_90_interval_inside_the_95_one_and_restores_robustness():
    rng = np.random.default_rng(1)
    rows = []
    for algo in ("A", "B"):
        for fold in ("covtype", "kdd99", "shuttle", "k_drift"):
            for s in (1, 2, 3):
                base = rng.random()
                rows += [{"algo": algo, "fold": fold, "arm": "frozen_prefix", "seed": s, "value": base},
                         {"algo": algo, "fold": fold, "arm": "confstream", "seed": s, "value": base + 0.01 * rng.random()},
                         {"algo": algo, "fold": fold, "arm": "default", "seed": s, "value": base - 0.5}]
    eq = F.equivalence_table(pd.DataFrame(rows), margin=0.05, n_boot=200)
    assert R.CI == (2.5, 97.5)
    p = eq[(eq["algo"] == "pooled") & (eq["group"] == "real")].set_index(["contrast", "unit"])
    assert p.loc[("confstream - frozen_prefix", "fold"), "verdict"] == "equivalent"
    assert p.loc[("confstream - default", "fold"), "verdict"] == "different beyond the margin"
    assert (p["low90"] >= p["low95"] - 1e-12).all() and (p["high90"] <= p["high95"] + 1e-12).all()


def test_the_noise_band_and_equivalence_are_different_readings():
    seeds = pd.DataFrame([
        *[{"algo": "A", "fold": "covtype", "arm": "confstream", "seed": s, "value": v} for s, v in zip((1, 2, 3), (0.3, 0.5, 0.7))],
        *[{"algo": "A", "fold": "covtype", "arm": "frozen_prefix", "seed": s, "value": 0.4} for s in (1, 2, 3)],
        *[{"algo": "A", "fold": "kdd99", "arm": "confstream", "seed": s, "value": v} for s, v in zip((1, 2, 3), (0.51, 0.52, 0.53))],
        *[{"algo": "A", "fold": "kdd99", "arm": "frozen_prefix", "seed": s, "value": 0.5} for s in (1, 2, 3)],
    ])
    summary = seeds.groupby(["algo", "fold", "arm"])["value"].agg(["mean", "std"]).reset_index().rename(
        columns={"mean": TAIL, "std": "lag_ari_win_tail_sd"})
    summary.loc[summary["arm"] == "frozen_prefix", "lag_ari_win_tail_sd"] = 0.0
    p = F.pair_equivalence(summary, seeds, margin=0.05).set_index("fold")
    assert bool(p.loc["covtype", "within_band"]) and p.loc["covtype", "equivalent"] == False  # noqa: E712
    assert not bool(p.loc["kdd99", "within_band"]) and p.loc["kdd99", "equivalent"] == True  # noqa: E712
    assert p.loc["kdd99", "low90"] > 0  # the gap is real and small


# --- E10: figure ----------------------------------------------------------------------------------------------

def test_figure_regenerates_the_dat_files_and_compares_them(tmp_path):
    wins = []
    for fold in F.FIGURE_FOLDS:
        for algo, bias in (("DenStream", 0.0), ("BIRCH", 0.2), ("DenStreamPatchedFast", 0.9)):
            for arm, a in zip(F.FIGURE_ARMS, (0.1, 0.5, 0.4, 0.6)):
                for seed in (1, 2):
                    wins += [_win(algo, fold, arm, seed, w, a + bias + 0.01 * w + 0.001 * seed) for w in range(1, 6)]
    _write(tmp_path / "windows.jsonl", wins)
    pub = tmp_path / "paper"
    pub.mkdir()
    from conf_pipeline import report

    curves = report.curves(tmp_path)
    for fold in F.FIGURE_FOLDS:
        F.write_dat(F.figure_data(curves, fold), pub / f"f3_{fold}.dat")
    expected = 0.5 + 0.1 + 0.01 * 2 + 0.0015  # confstream on window 2: mean of the two protocol algos and seeds
    assert F.figure_data(curves, "k_drift").loc[2, "confstream"] == pytest.approx(expected)
    text = (pub / "f3_k_drift.dat").read_text().replace("0.6215", "0.7215")
    (pub / "f3_k_drift.dat").write_text(text)
    F.cmd_figure(tmp_path, pub)
    cmp = pd.read_csv(tmp_path / "report" / "followup" / "figure" / "comparison.csv")
    k = cmp[(cmp["file"] == "f3_k_drift.dat") & (cmp["column"] == "confstream")].iloc[0]
    s = cmp[(cmp["file"] == "f3_scale_drift.dat") & cmp["column"].isin(F.FIGURE_ARMS)]
    assert k["max_abs_diff"] == pytest.approx(0.1, abs=1e-4)
    assert (s["max_abs_diff"] <= 5e-5).all()
    assert (tmp_path / "report" / "followup" / "figure" / "f3_k_drift.dat").exists()
    assert (pub / "f3_k_drift.dat").read_text() == text  # the published file is never overwritten


# --- E10: feasibility -------------------------------------------------------------------------------------------

def test_feasibility_replays_the_runtime_gate_and_its_counterfactual(tmp_path):
    a = ("DenStream", "covtype", "confstream", 1)
    rows = [
        _mem(*a, 1, 0, runtime_s=0.010, score=0.1, ari=0.1),  # fastest, and the poorest silhouette
        _mem(*a, 1, 1, runtime_s=0.020, score=0.6, ari=0.6),
        _mem(*a, 1, 2, runtime_s=0.500, score=-1.0, feasible=False, feasible_reason="runtime"),
        _mem(*a, 1, 3, runtime_s=0.400, score=-1.0, feasible=False, feasible_reason="runtime"),
        _mem(*a, 1, 4, runtime_s=0.001, score=-1.0, silhouette_defined=False),  # no partition: not the reference
        _mem(*a, 1, 5, runtime_s=0.0, score=-1.0, feasible=False, feasible_reason="error", error="boom"),
        _mem(*a, 1, 6, runtime_s=0.030, score=0.5, ari=0.5),
    ]
    _write(tmp_path / "members" / "DenStream__covtype__seed1__confstream.jsonl", rows)
    runs = pd.DataFrame([_run(*a)])
    m = F.replay_feasibility(F.load_members(tmp_path, workers=1), runs)
    assert (m["replayed"] == m["logged"]).all()
    assert set(m.loc[m["flips"], "member_id"]) == {2, 3}  # without member 0 the cap is 30 x 0.02 = 0.6
    w = F.window_feasibility(m).iloc[0]
    assert w["n_flips"] == 2 and bool(w["fastest_below_median_sil"])
    s = F.feasibility_summary(m, F.window_feasibility(m)).iloc[0]
    assert s["replay_agreement"] == 1.0 and s["flipped_over_shed"] == 1.0


def test_feasibility_without_member_logs_says_so(tmp_path):
    _write(tmp_path / "runs.jsonl", [_run("DenStream", "covtype", "confstream", 1)])
    F.cmd_feasibility(tmp_path, workers=1)
    assert "No member log" in (tmp_path / "report" / "followup" / "feasibility" / "feasibility.md").read_text()


# --- E9: substitute -----------------------------------------------------------------------------------------------

def test_substitute_reads_the_patched_variant_in_its_reference_place(tmp_path):
    table = {  # fold -> arm -> algo -> tail ARI
        "covtype": {"default": {"DenStream": 0.5, "BIRCH": 0.4, "SOStream": 0.1, "DenStreamPatchedFast": 0.3},
                    "confstream": {"DenStream": 0.6, "BIRCH": 0.8, "SOStream": 0.2, "DenStreamPatchedFast": 0.6}},
        "k_drift": {"default": {"DenStream": 0.9, "BIRCH": 0.4, "SOStream": 0.1, "DenStreamPatchedFast": 0.9},
                    "confstream": {"DenStream": 0.9, "BIRCH": 0.5, "SOStream": 0.2, "DenStreamPatchedFast": 0.9}},
    }
    runs = []
    for fold, arms in table.items():
        for arm, algos in arms.items():
            for algo, v in algos.items():
                runs += [_run(algo, fold, arm, s, **{TAIL: v}) for s in (1, 2)]
                runs += [_run(algo, fold, "frozen_prefix", s, **{TAIL: v}) for s in (1, 2)] if arm == "confstream" else []
    _write(tmp_path / "runs.jsonl", runs)
    F.cmd_substitute(tmp_path)
    r = pd.read_csv(tmp_path / "report" / "followup" / "substitute" / "readings.csv").set_index("reading")
    pub, sub = r.index
    assert r.loc[pub, "regret_real_fw"] == pytest.approx(0.2)  # defaults pick DenStream, online's best is BIRCH
    assert r.loc[sub, "regret_real_fw"] == pytest.approx(0.0)  # the patched DenStream's default reads 0.3: BIRCH
    assert r.loc[sub, "regret_param_drift_fw"] == pytest.approx(0.0)
    md = (tmp_path / "report" / "followup" / "substitute" / "substitute.md").read_text()
    assert "1 of 2" in md


def test_substitute_without_the_variant_writes_the_reason(tmp_path):
    _write(tmp_path / "runs.jsonl", [_run("DenStream", "covtype", "default", 1, **{TAIL: 0.1})])
    F.cmd_substitute(tmp_path)
    assert "nothing to substitute" in (tmp_path / "report" / "followup" / "substitute" / "substitute.md").read_text()


# --- loading ------------------------------------------------------------------------------------------------------

def test_member_logs_read_the_same_in_parallel_and_in_one_process(tmp_path):
    for k in range(6):
        _write(tmp_path / "members" / f"DenStream__covtype__seed{k}__confstream.jsonl",
               [_mem("DenStream", "covtype", "confstream", k, w, 0) for w in (1, 2)])
    one = F.load_members(tmp_path, workers=1)
    two = F.load_members(tmp_path, workers=2)
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == 12 and one["usable"].all() and one["has_partition"].all()
    assert F.load_members(tmp_path / "nowhere", workers=1).empty


def test_recovery_without_drift_points_says_so(tmp_path):
    _write(tmp_path / "runs.jsonl", [_run("DenStream", "covtype", "confstream", 1)])
    F.cmd_recovery(tmp_path, published=None)
    assert "no designed drift" in (tmp_path / "report" / "followup" / "recovery" / "recovery.md").read_text()
