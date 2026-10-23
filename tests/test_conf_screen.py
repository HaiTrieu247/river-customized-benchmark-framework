"""conf_pipeline.screen: what each logged selection signal would have picked, one window ahead."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conf_pipeline import screen

WINDOWS = range(18, 25)


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _member(fold, seed, w, mid, ari, score, **sig):
    return {"algo": "DenStream", "fold": fold, "seed": seed, "arm": "confstream", "config_id": None,
            "member_id": mid, "window": w, "is_test": False, "feasible": True,
            "silhouette_defined": True, "score": score, "ari_win": ari,
            **{f"sig_{k}": v for k, v in sig.items()}}


def _out_dir(root, fold="covtype"):
    """Member 0 is good (ARI 0.8) and member 1 poor (0.2) on every window. The silhouette
    prefers member 1; signal ``good`` prefers member 0, ``bad`` member 1; ``none`` has no
    reading before window 22, so it picks nothing on windows 20 to 22."""
    runs, windows = [], []
    for seed in (1, 2):
        runs.append({"kind": "run", "algo": "DenStream", "fold": fold, "seed": seed, "arm": "confstream",
                     "config_id": None, "status": "ok", "tail_from": 20})
        windows += [{"kind": "window", "algo": "DenStream", "fold": fold, "seed": seed, "arm": "confstream",
                     "config_id": None, "window": w, "incumbent_id": 1} for w in WINDOWS]
        rows = []
        for w in WINDOWS:
            late = w >= 22
            rows.append(_member(fold, seed, w, 0, 0.8, 0.1, good=0.9, bad=0.1, none=0.9 if late else None))
            rows.append(_member(fold, seed, w, 1, 0.2, 0.9, good=0.1, bad=0.9, none=0.1 if late else None))
        _jsonl(root / "members" / f"DenStream__{fold}__seed{seed}__confstream.jsonl", rows)
    _jsonl(root / "runs.jsonl", runs)
    _jsonl(root / "windows.jsonl", windows)
    return root


def test_signal_picks_read_the_member_each_signal_chose_one_window_earlier(tmp_path):
    picks = screen.signal_picks(_out_dir(tmp_path))
    assert set(picks["signal"]) == {"labels", "silhouette", "good", "bad", "none"}
    p = picks.set_index(["seed", "window", "signal"])["ari"]
    assert p[(1, 21, "labels")] == pytest.approx(0.8)
    assert p[(1, 21, "silhouette")] == pytest.approx(0.2)
    assert p[(1, 21, "good")] == pytest.approx(0.8) and p[(1, 21, "bad")] == pytest.approx(0.2)
    assert (1, 21, "none") not in p.index and p[(1, 23, "none")] == pytest.approx(0.8)
    assert picks["window"].min() == 19  # window 18 has no previous window to pick on


def test_screen_table_reports_the_share_of_the_selection_loss_each_signal_recovers(tmp_path):
    out = _out_dir(tmp_path)
    pairs, groups = screen.screen_tables(screen.signal_picks(out), screen.tail_from(out))
    g = groups.set_index(["group", "signal"])
    assert g.loc[("real", "good"), "recovered"] == pytest.approx(1.0)
    assert g.loc[("real", "bad"), "recovered"] == pytest.approx(0.0)
    assert g.loc[("real", "good"), "selection_loss"] == pytest.approx(0.0)
    assert g.loc[("real", "silhouette"), "selection_loss"] == pytest.approx(0.6)
    # "none" is compared only on the tail windows where it picked: 23 and 24
    row = pairs.set_index("signal").loc["none"]
    assert row["n_windows"] == 2 * 2 and row["a_signal"] == pytest.approx(0.8)


def test_write_screen_writes_the_tables_and_says_what_they_can_not_show(tmp_path):
    out = _out_dir(tmp_path / "out")
    paths = screen.write_screen(out, rep_dir=tmp_path / "rep")
    md = (tmp_path / "rep" / "screen.md").read_text()
    assert "good" in md and "signal arm" in md
    assert all(p.exists() for p in paths)
    groups = pd.read_csv(tmp_path / "rep" / "screen" / "groups.csv")
    assert set(groups["signal"]) >= {"good", "bad"}


def test_a_run_without_logged_signals_still_reads_labels_and_silhouette(tmp_path):
    out = _out_dir(tmp_path)
    for path in (out / "members").glob("*.jsonl"):
        rows = [{k: v for k, v in json.loads(line).items() if not k.startswith("sig_")}
                for line in path.read_text().splitlines()]
        _jsonl(path, rows)
    picks = screen.signal_picks(out)
    assert set(picks["signal"]) == {"labels", "silhouette"}


def test_a_lower_is_better_reading_picks_its_lowest_member(tmp_path):
    out = _out_dir(tmp_path)
    for path in (out / "members").glob("*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:  # Davies-Bouldin: the good member 0 has the lower value
            row["sig_davies_bouldin_std"] = 0.3 if row["member_id"] == 0 else 2.0
        _jsonl(path, rows)
    pairs, groups = screen.screen_tables(screen.signal_picks(out), screen.tail_from(out))
    g = groups.set_index(["group", "signal"])
    assert g.loc[("real", "davies_bouldin_std"), "recovered"] == pytest.approx(1.0)


# --- paired supports, coverage, missing readings, cost ---------------------------------------

def _support(out):
    log = screen.pick_log(out)
    return log, screen.support_tables(log, screen.tail_from(out), screen.pairs_run(out))


def test_the_pick_log_says_why_a_signal_picked_nobody(tmp_path):
    log = screen.pick_log(_out_dir(tmp_path)).set_index(["seed", "window", "signal"])
    assert log.loc[(1, 21, "none"), "status"] == "undefined"      # no reading on window 20
    assert log.loc[(1, 23, "none"), "status"] == "picked"
    assert log.loc[(1, 21, "good"), "status"] == "picked"
    assert log.loc[(1, 21, "none"), "n_candidates"] == 2 and log.loc[(1, 21, "none"), "n_defined"] == 0
    assert log.loc[(1, 23, "none"), "n_defined"] == 2
    # the picks the screen has always read are the log's "picked" rows
    picks = screen.signal_picks(_out_dir(tmp_path / "again"))
    assert len(picks) == int((log["status"] == "picked").sum())


def test_only_the_common_support_reads_every_signal_on_the_same_windows(tmp_path):
    log, (pairs, groups, missing) = _support(_out_dir(tmp_path))
    g = groups[groups["group"] == "real"].set_index(["support", "signal"])
    # own support: "good" on tail windows 20..24 of both seeds, "none" only on 23 and 24
    assert g.loc[("own", "good"), "n_windows"] == 10 and g.loc[("own", "none"), "n_windows"] == 4
    # common support: the windows where every logged signal picked, the same for all of them
    common = g.loc["common"]
    assert set(common["n_windows"]) == {4}
    assert common["recovered_den"].to_numpy() == pytest.approx([0.6] * len(common))
    assert common.loc["good", "recovered"] == pytest.approx(1.0)
    assert common.loc["bad", "recovered"] == pytest.approx(0.0)
    assert common.loc["silhouette", "recovered"] == pytest.approx(0.0)
    # the own-support reading is the published one
    _, old = screen.screen_tables(screen.signal_picks(tmp_path), screen.tail_from(tmp_path))
    old = old.set_index(["group", "signal"])
    for signal in ("good", "bad", "none", "silhouette"):
        assert g.loc[("own", signal), "recovered"] == pytest.approx(old.loc[("real", signal), "recovered"])


def test_a_signal_without_a_reading_falls_back_to_the_silhouette_on_the_fallback_support(tmp_path):
    _, (pairs, groups, missing) = _support(_out_dir(tmp_path))
    g = groups[groups["group"] == "real"].set_index(["support", "signal"])
    row = g.loc[("fallback", "none")]
    # windows 20..22 fall back to the silhouette's 0.2, 23 and 24 read 0.8: (0.6 + 1.6) / 5
    assert row["n_windows"] == 10 and row["n_fallback"] == 6
    assert row["a_signal"] == pytest.approx(0.44)
    assert row["recovered"] == pytest.approx(0.24 / 0.6)
    assert g.loc[("fallback", "good"), "n_fallback"] == 0


def test_the_pilot_table_reports_gain_coverage_and_the_defined_rate(tmp_path):
    _, (pairs, groups, missing) = _support(_out_dir(tmp_path))
    g = groups[groups["group"] == "real"].set_index(["support", "signal"])
    row = g.loc[("own", "good")]
    assert row["n_pairs_run"] == 1 and row["n_pairs_covered"] == 1 and row["coverage"] == pytest.approx(1.0)
    assert row["gain"] == pytest.approx(0.6) and row["window_gain_mean"] == pytest.approx(0.6)
    assert (row["n_better"], row["n_worse"], row["n_tied"]) == (10, 0, 0)
    assert row["recovered_num"] == pytest.approx(0.6) and row["recovered_den"] == pytest.approx(0.6)
    # "none" has a reading on 4 of the 10 candidate member-windows of each seed's tail
    assert g.loc[("own", "none"), "defined_rate"] == pytest.approx(0.4)
    assert g.loc[("own", "none"), "n_candidates"] == 20
    m = missing[(missing["level"] == "pair") & (missing["signal"] == "none")].iloc[0]
    assert (m["n_tail"], m["picked"], m["undefined"], m["n_fallback"]) == (10, 4, 6, 6)
    assert m["n_lost_to_common"] == 0
    assert missing.loc[(missing["level"] == "pair") & (missing["signal"] == "good"), "n_lost_to_common"].iloc[0] == 6


def test_a_recovery_without_a_selection_loss_is_undefined_and_says_why(tmp_path):
    out = _out_dir(tmp_path)
    for path in (out / "members").glob("*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:  # the silhouette now prefers the good member: nothing to recover
            row["score"] = 0.9 if row["member_id"] == 0 else 0.1
        _jsonl(path, rows)
    _, (pairs, groups, missing) = _support(out)
    row = groups[(groups["group"] == "real")].set_index(["support", "signal"]).loc[("common", "good")]
    assert np.isnan(row["recovered"]) and row["recovered_den"] == pytest.approx(0.0)
    assert "no selection loss" in row["recovered_note"]


def test_a_job_that_did_not_log_a_signal_has_no_common_window(tmp_path):
    out = _out_dir(tmp_path)
    path = out / "members" / "DenStream__covtype__seed2__confstream.jsonl"
    rows = [{k: v for k, v in json.loads(line).items() if k != "sig_none"} for line in path.read_text().splitlines()]
    _jsonl(path, rows)
    log, (pairs, groups, missing) = _support(out)
    assert set(log.loc[(log["seed"] == 2) & (log["signal"] == "none"), "status"]) == {"not_logged"}
    g = groups[groups["group"] == "real"].set_index(["support", "signal"])
    assert g.loc[("common", "good"), "n_windows"] == 2                      # seed 1, windows 23 and 24
    # the fallback support also leaves seed 2 out: not computing a signal is not failing to read it
    assert g.loc[("fallback", "good"), "n_windows"] == 5 and g.loc[("fallback", "none"), "n_fallback"] == 3
    m = missing[(missing["level"] == "pair") & (missing["signal"] == "none")].iloc[0]
    assert m["not_logged"] == 5


def test_cost_is_the_signals_share_of_wall_time_and_is_not_split_by_signal(tmp_path):
    out = _out_dir(tmp_path)
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    for r in runs:
        r.update(wall_s=100.0, signals_runtime_s=25.0, ensemble_runtime_s=50.0, n_windows=7,
                 log_signals=["good", "bad", "none"])
    _jsonl(out / "runs.jsonl", runs)
    cost = screen.cost_table(out)
    run = cost[cost["level"] == "run"]
    assert len(run) == 2 and run["signals_share_of_wall"].to_numpy() == pytest.approx([0.25, 0.25])
    assert set(run["n_member_windows"]) == {14}
    total = cost[cost["level"] == "all"].iloc[0]
    assert total["signals_runtime_s"] == pytest.approx(50.0) and total["signals_share_of_wall"] == pytest.approx(0.25)


def test_sentinel_exposure_is_counted_on_the_windows_compared(tmp_path):
    out = _out_dir(tmp_path)
    path = out / "members" / "DenStream__covtype__seed1__confstream.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["n_sentinel"] = 3 if (row["window"] == 22 and row["member_id"] == 1) else 0
    _jsonl(path, rows)
    log = screen.pick_log(out)
    one = log[(log["seed"] == 1) & (log["signal"] == "good")].set_index("window")["n_sentinel_members"]
    assert one.loc[23] == 1 and one.loc[22] == 0
    assert log.loc[log["seed"] == 2, "n_sentinel_members"].isna().all()   # not in that log


def test_write_screen_adds_an_exploratory_pilot_section_and_keeps_the_published_tables(tmp_path):
    out = _out_dir(tmp_path / "out")
    runs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    for r in runs:
        r.update(wall_s=100.0, signals_runtime_s=25.0, ensemble_runtime_s=50.0, log_signals=["good"])
    _jsonl(out / "runs.jsonl", runs)
    screen.write_screen(out, rep_dir=tmp_path / "rep")
    md = (tmp_path / "rep" / "screen.md").read_text()
    head = md.split("## ")[0]
    assert "**Share of the silhouette's selection loss recovered**" in head
    assert "Exploratory" in md and "held-out" in md and "common support" in md
    assert "cannot be separated" in md and "predictions are not logged" in md
    for name in ("support_pairs.csv", "support_groups.csv", "missing.csv", "cost.csv", "pick_log.csv"):
        assert (tmp_path / "rep" / "screen" / name).exists()
    pairs, groups = screen.screen_tables(screen.signal_picks(out), screen.tail_from(out))
    assert pd.read_csv(tmp_path / "rep" / "screen" / "groups.csv").shape == groups.shape


def test_runs_without_any_readable_window_are_reported_as_such(tmp_path):
    out = _out_dir(tmp_path / "out")
    for path in (out / "members").glob("*.jsonl"):  # single-class windows: no ARI anywhere
        _jsonl(path, [dict(json.loads(line), ari_win=None) for line in path.read_text().splitlines()])
    log = screen.pick_log(out)
    assert set(log.loc[log["window"] > 18, "status"]) == {"no_ari", "undefined"}
    _, groups, missing = screen.support_tables(log, screen.tail_from(out), screen.pairs_run(out))
    row = groups.set_index(["group", "support", "signal"]).loc[("real", "common", "good")]
    assert row["n_pairs_covered"] == 0 and row["coverage"] == 0.0 and row["recovered_note"] == "no window compared"
    screen.write_screen(out, rep_dir=tmp_path / "rep")
    assert "nothing to compare" not in (tmp_path / "rep" / "screen.md").read_text()
