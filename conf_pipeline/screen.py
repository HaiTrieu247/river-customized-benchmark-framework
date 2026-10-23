"""Screening label-free selection signals on the member logs of existing runs.

On real streams most of the gap between online tuning and the static reference is
selection loss: the ensemble holds a good configuration and the silhouette does not pick
it. A run started with ``run.py --log-members --signals ...`` writes, for every member on
every window, the candidate signals of :mod:`benchmark.tuning.signals` as ``sig_<name>``.
This module asks, for each logged signal S, what the tuner would have handed a user had it
picked by S instead: after window t-1, among the members active on t that were usable on
t-1 (feasible, with a partition), the one with the highest S on t-1, read by its ARI on t.
Labels and the silhouette pick the same way and under the same delay, so the three are
comparable window by window:

- ``selection_loss`` = labels' pick minus S's pick (the silhouette's is the published one)
- ``recovered`` = share of the silhouette's selection loss that S wins back, as a ratio of
  sums over the pairs of a stream group: 1 is as good as labels, 0 no better than the
  silhouette, negative worse

A reading where lower is better (``signals.LOWER_IS_BETTER``, e.g. Davies-Bouldin) picks
its lowest member. A tie goes to the incumbent named after t-1, then to the smallest member
id, as in ``report.member_readings``. The screen reads the ensemble the silhouette evolved, so it
says which signal would pick better among those configurations, not what an ensemble
driven by S would have grown: that is what the signal arms (``confstream_<signal>``) run.

Supports. ``screen_tables`` (``pairs.csv``, ``groups.csv``, the first tables of ``screen.md``)
reads every signal on its OWN support: the tail windows where labels, the silhouette and that
signal picked. Two signals undefined on different windows are then scored on different
windows, against different selection losses, and their recovery ratios do not compare.
``support_tables`` adds two paired supports, the same windows for every signal:

- ``common``   the tail windows where labels, the silhouette and EVERY logged signal picked
               (a job that did not log some signal has no common window): the only support on
               which recovery ratios compare signals directly without imputing anything
- ``fallback`` the tail windows where labels and the silhouette picked, in the jobs that logged
               every signal; a signal that picked nobody takes the silhouette's pick there, so
               it is charged for the windows it cannot read (a tuner that reads S where S is
               defined and the silhouette elsewhere; a tuner selecting by S alone would keep
               its incumbent instead)

``pick_log`` says, for every window and signal, whether the signal picked and why not
(``STATUSES``), how many candidates it had a reading on, and how many feasible members of
the reading window held a sentinel label (DenStream's ``-1``), the points a noise policy
(``signals.NOISE_POLICIES``) would read differently. ``cost_table`` reads the signals' share
of the wall time from ``runs.jsonl``; the runs time all logged signals together, so a cost per
signal cannot be read from them. ``write_screen`` writes all of it, labelled exploratory: the
streams screened are development data for the choice of a signal.

Run from the repo root::

    python -m conf_pipeline.screen --out-dir conf_pipeline/out/<dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from . import report

from benchmark.tuning.signals import LOWER_IS_BETTER  # noqa: E402  (after _common: river on sys.path)

LABELS, SILHOUETTE = "labels", "silhouette"
PICK_COLS = ["algo", "fold", "seed", "arm", "window", "signal", "member_id", "ari"]
#: Why a signal has a pick on a window, or has none (``pick_log``):
#: ``picked`` a pick with an ARI on the window; ``no_ari`` a pick whose ARI is undefined (a
#: single-class window); ``undefined`` candidates, but no reading on any of them;
#: ``not_logged`` the job's member log carries no reading of this signal at all;
#: ``no_candidate`` no member usable on t-1 and active on t; ``no_previous`` t-1 not logged.
STATUSES: tuple[str, ...] = ("picked", "no_ari", "undefined", "not_logged", "no_candidate", "no_previous")
SUPPORTS: tuple[str, ...] = ("own", "common", "fallback")
LOG_COLS = ["algo", "fold", "seed", "arm", "window", "signal", "status", "member_id", "ari",
            "n_candidates", "n_defined", "n_sentinel_members"]
JOB_WINDOW = ["algo", "fold", "seed", "arm", "window"]
PAIR = ["algo", "fold", "arm"]


def tail_from(out_dir: Path) -> dict:
    """(algo, fold, seed, arm) -> the run's ``tail_from``."""
    return report._tail_from_by_job(report.current_runs(out_dir, diagnostic=True))


def _incumbents(out_dir: Path, arms) -> dict:
    win = report.load_windows(out_dir, diagnostic=True)
    if win.empty or "incumbent_id" not in win:
        return {}
    win = win[win["arm"].isin(list(arms))]
    return {(a, f, s, arm, int(w)): i for a, f, s, arm, w, i in
            zip(win["algo"], win["fold"], win["seed"], win["arm"], win["window"], win["incumbent_id"])}


def _jobs(out_dir: Path, arms):
    """Per job of the member logs, in file order: ``(key, logged signal names, {window: {member:
    row}})``; the names leave out the silhouette, which every log carries as ``score``."""
    for path in report._member_files(out_dir, arms=set(arms)):
        jobs: dict[tuple, dict[int, dict]] = {}
        for r in C.read_jsonl(path):
            key = (r["algo"], r["fold"], r["seed"], r["arm"])
            jobs.setdefault(key, {}).setdefault(int(r["window"]), {})[r["member_id"]] = r
        for key, by_window in jobs.items():
            names = sorted({k[4:] for ms in by_window.values() for m in ms.values() for k in m
                            if k.startswith("sig_")} - {SILHOUETTE})
            yield key, names, by_window


def _readers(names) -> list[tuple[str, str]]:
    return [(LABELS, "ari_win"), (SILHOUETTE, "score")] + [(n, f"sig_{n}") for n in names]


def _candidates(prev: dict, members: dict) -> list:
    """The members active on t (not under test) that were usable on t-1: feasible, with a
    partition."""
    return [mid for mid, m in members.items() if not m.get("is_test") and mid in prev
            and prev[mid].get("feasible") is not False
            and prev[mid].get("silhouette_defined") is not False]


def _pick(signal: str, field: str, prev: dict, cands: list, inc) -> tuple[int, object]:
    """``(n_defined, pick)``: how many candidates have a reading of ``field`` on t-1, and the one
    with the best reading (None without any); a tie goes to ``inc``, then to the smallest id."""
    sign = -1.0 if signal in LOWER_IS_BETTER else 1.0
    vals = {mid: report._num(prev[mid].get(field)) for mid in cands}
    vals = {mid: sign * v for mid, v in vals.items() if v is not None}
    if not vals:
        return 0, None
    top = max(vals.values())
    tied = [mid for mid, v in vals.items() if v == top]
    return len(vals), (inc if inc in tied else min(tied))


def signal_picks(out_dir: Path, arms=("confstream",)) -> pd.DataFrame:
    """One row per (job, window, signal): the member that signal picked after the previous
    window, and its ARI on this one. Signals: ``labels``, ``silhouette`` (the logged score)
    and every ``sig_<name>`` in the member logs. A window with no pick, or with no ARI (a
    single-class window), has no row."""
    out_dir = Path(out_dir)
    incumbents = _incumbents(out_dir, arms)
    rows = []
    for key, names, by_window in _jobs(out_dir, arms):
        readers = _readers(names)
        for w in sorted(by_window):
            prev = by_window.get(w - 1)
            if not prev:
                continue
            members = by_window[w]
            cands = _candidates(prev, members)
            inc = incumbents.get((*key, w - 1))
            for signal, field in readers:
                _, pick = _pick(signal, field, prev, cands, inc)
                if pick is None:
                    continue
                ari = report._num(members[pick].get("ari_win"))
                if ari is not None:
                    rows.append((*key, w, signal, pick, ari))
    return pd.DataFrame(rows, columns=PICK_COLS)


def _sentinel_members(prev: dict, logged: bool) -> int | None:
    """Feasible members of t-1 whose predictions held a sentinel label: the pool every reading
    of t-1 draws on (``rank_agg`` ranks all feasible members, under test or not). None when the
    log has no ``n_sentinel`` (older logs)."""
    if not logged:
        return None
    return sum(1 for m in prev.values() if m.get("feasible") is not False and not m.get("error")
               and (report._num(m.get("n_sentinel")) or 0) > 0)


def pick_log(out_dir: Path, arms=("confstream",)) -> pd.DataFrame:
    """Every (job, window, signal) of the member logs, picked or not (``LOG_COLS``): its
    ``status`` (``STATUSES``), the pick and its ARI when there is one, ``n_candidates`` (members
    usable on t-1 and active on t), ``n_defined`` (those with a reading of the signal on t-1)
    and ``n_sentinel_members`` (feasible members of t-1 whose predictions held a sentinel).

    A signal that some job logged and another did not is ``not_logged`` on every window of the
    other. The rows with ``status == "picked"`` are exactly ``signal_picks``."""
    out_dir = Path(out_dir)
    incumbents = _incumbents(out_dir, arms)
    rows, windows, logged = [], [], {}
    for key, names, by_window in _jobs(out_dir, arms):
        logged[key] = set(names)
        readers = _readers(names)
        has_sentinel = any("n_sentinel" in m for ms in by_window.values() for m in ms.values())
        for w in sorted(by_window):
            prev = by_window.get(w - 1)
            if not prev:
                rows += [(*key, w, s, "no_previous", None, None, 0, 0, None) for s, _ in readers]
                continue
            members = by_window[w]
            cands = _candidates(prev, members)
            inc = incumbents.get((*key, w - 1))
            n_sent = _sentinel_members(prev, has_sentinel)
            windows.append((key, w, len(cands), n_sent))
            for signal, field in readers:
                n_def, pick = _pick(signal, field, prev, cands, inc)
                ari = report._num(members[pick].get("ari_win")) if pick is not None else None
                status = ("no_candidate" if not cands else "undefined" if pick is None
                          else "no_ari" if ari is None else "picked")
                rows.append((*key, w, signal, status, pick, ari, len(cands), n_def, n_sent))
    union = set().union(*logged.values()) if logged else set()
    for key, w, n_cands, n_sent in windows:
        rows += [(*key, w, s, "not_logged", None, None, n_cands, 0, n_sent) for s in sorted(union - logged[key])]
    log = pd.DataFrame(rows, columns=LOG_COLS)
    for col in ("member_id", "n_sentinel_members"):
        log[col] = log[col].astype("Int64")
    return log


def pairs_run(out_dir: Path, arms=("confstream",)) -> set[tuple]:
    """(algo, fold, arm) of every ``ok`` run of ``arms`` in ``runs.jsonl``: the denominator of
    pair coverage, so that a pair run without a member log counts as not covered."""
    runs = report.current_runs(out_dir, diagnostic=True)
    if runs.empty or "status" not in runs:
        return set()
    runs = runs[(runs["status"] == "ok") & runs["arm"].isin(list(arms))]
    return set(zip(runs["algo"], runs["fold"], runs["arm"]))


def _in_tail(log: pd.DataFrame, tail: dict) -> pd.DataFrame:
    starts = [tail.get(k, C.FREEZE_AFTER_WINDOW) for k in zip(log["algo"], log["fold"], log["seed"], log["arm"])]
    return log[log["window"].to_numpy() >= np.array(starts, dtype=float)]


def _recovered(num: float, den: float, n_pairs: int) -> tuple[float, str]:
    """``num / den`` with a reason in place of a ratio that has no meaning (never clipped)."""
    if n_pairs == 0:
        return np.nan, "no window compared"
    if den > 0:
        return num / den, ""
    if den == 0:
        return np.nan, "no selection loss to recover: the silhouette's pick read as well as the labels' (den = 0)"
    return np.nan, "the silhouette's pick read better than the labels' (den < 0)"


def _rows(df: pd.DataFrame, mask) -> pd.DataFrame:
    """The rows of ``df`` where ``mask`` holds; an empty mask selects no row (``df[[]]`` would
    select no column instead)."""
    return df.loc[np.asarray(mask, dtype=bool)] if len(df) else df


def _window_stats(diff: pd.Series) -> dict:
    """The paired per-window difference (signal's pick minus the silhouette's), pooled."""
    n = len(diff)
    return {"window_gain_mean": float(diff.mean()) if n else np.nan,
            "window_gain_sd": float(diff.std(ddof=1)) if n > 1 else np.nan,
            "n_better": int((diff > 0).sum()), "n_worse": int((diff < 0).sum()), "n_tied": int((diff == 0).sum())}


SUPPORT_PAIR_COLS = [*PAIR, "group", "signal", "support", "n_seeds", "n_windows", "n_fallback", "a_labels",
                     "a_silhouette", "a_signal", "gain", "selection_loss", "silhouette_loss", "recovered",
                     "recovered_note", "window_gain_mean", "window_gain_sd", "n_better", "n_worse", "n_tied",
                     "n_candidates", "n_defined", "defined_rate"]
SUPPORT_GROUP_COLS = ["group", "signal", "support", "n_pairs_run", "n_pairs_covered", "coverage", "n_windows",
                      "n_fallback", "n_candidates", "n_defined", "defined_rate", "a_labels", "a_silhouette",
                      "a_signal", "gain", "recovered_num", "recovered_den", "recovered", "recovered_note",
                      "window_gain_mean", "window_gain_sd", "n_better", "n_worse", "n_tied"]
MISSING_COLS = ["level", "group", *PAIR, "signal", "n_tail", *STATUSES, "n_partial", "n_base", "n_own", "n_common",
                "n_lost_to_common", "n_fallback", "n_sentinel_windows", "n_sentinel_common"]


def support_tables(log: pd.DataFrame, tail: dict, run_pairs: set | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The screen on the three ``SUPPORTS``, from ``pick_log`` restricted to each job's tail.
    Returns ``(pairs, groups, missing)``:

    - ``pairs``: per (algo, fold, arm, signal, support), seed means of the per-job means as in
      ``screen_tables``; ``gain`` = a_signal - a_silhouette, ``silhouette_loss`` = a_labels -
      a_silhouette (the recovery's denominator), ``recovered`` their ratio, undefined with a
      ``recovered_note`` when the denominator is not positive; the paired per-window difference
      pooled over the pair's windows; the defined rate over the pair's tail candidates.
    - ``groups``: per stream group (and ``all``), signal and support: pair coverage (pairs with
      at least one window on the support / ``run_pairs`` of the group, the pairs seen in the log
      when ``run_pairs`` is None), ``recovered_num`` / ``recovered_den`` (sums over the covered
      pairs) and their ratio, the mean gain over pairs and the pooled per-window difference.
    - ``missing``: per pair and per group, the tail windows by ``STATUSES``; ``n_partial`` picks
      made among only some candidates; ``n_base`` windows where labels and the silhouette
      picked; ``n_own`` / ``n_common`` the signal's windows on each support and
      ``n_lost_to_common`` the difference; ``n_fallback`` fallback-support windows where the
      signal picked nobody; ``n_sentinel_windows`` / ``n_sentinel_common`` tail / common windows with a
      sentinel among the feasible members read (NaN when the logs do not count sentinels).

    The defined rate leaves ``not_logged`` rows out: a job that never computed a signal says
    nothing about where it is defined.
    """
    empty = (pd.DataFrame(columns=SUPPORT_PAIR_COLS), pd.DataFrame(columns=SUPPORT_GROUP_COLS),
             pd.DataFrame(columns=MISSING_COLS))
    if log.empty:
        return empty
    log = _in_tail(log, tail)
    log = log[log["status"] != "no_previous"]
    if log.empty:
        return empty
    signals = sorted(set(log["signal"]) - {LABELS})
    picked = log[log["status"] == "picked"]
    wide = picked.pivot_table(index=JOB_WINDOW, columns="signal", values="ari", aggfunc="first")
    wide = wide.reindex(pd.MultiIndex.from_frame(log[JOB_WINDOW].drop_duplicates()))
    for s in (LABELS, SILHOUETTE, *signals):
        if s not in wide:
            wide[s] = np.nan
    base = wide[LABELS].notna() & wide[SILHOUETTE].notna()
    every = base & wide[signals].notna().all(axis=1)
    # the fallback support: base windows of jobs that logged every signal, so that it stays the
    # same windows for all signals and a signal is charged for failing to read, not for not
    # having been computed
    unlogged = pd.MultiIndex.from_frame(log.loc[log["status"] == "not_logged", JOB_WINDOW].drop_duplicates())
    fallback = base & ~wide.index.isin(unlogged)
    sentinel = log.drop_duplicates(JOB_WINDOW).set_index(JOB_WINDOW)["n_sentinel_members"].reindex(wide.index)
    pair_of = {p: C.stream_group(p[1]) for p in set(zip(log["algo"], log["fold"], log["arm"]))}
    run_pairs = set(run_pairs) if run_pairs is not None else set(pair_of)
    groups_of = {**{p: C.stream_group(p[1]) for p in run_pairs}, **pair_of}

    defined = (log[log["status"] != "not_logged"].groupby([*PAIR, "signal"])[["n_candidates", "n_defined"]].sum())

    pair_rows, windows_of = [], {}
    for signal in signals:
        sig = wide[signal]
        masks = {"own": base & sig.notna(), "common": every, "fallback": fallback}
        for support, mask in masks.items():
            value = sig.where(sig.notna(), wide[SILHOUETTE]) if support == "fallback" else sig
            frame = pd.DataFrame({"lab": wide.loc[mask, LABELS], "sil": wide.loc[mask, SILHOUETTE],
                                  "sig": value[mask], "fb": (mask & sig.isna())[mask]})
            windows_of[(signal, support)] = frame
            for (algo, fold, arm), g in frame.groupby(level=PAIR, sort=True):
                per_job = g.groupby(level="seed")[["lab", "sil", "sig"]].mean()
                a_lab, a_sil, a_sig = (float(per_job[c].mean()) for c in ("lab", "sil", "sig"))
                rec, note = _recovered(a_sig - a_sil, a_lab - a_sil, 1)
                n_c, n_d = (defined.loc[(algo, fold, arm, signal)] if (algo, fold, arm, signal) in defined.index
                            else (0, 0))
                pair_rows.append({
                    "algo": algo, "fold": fold, "arm": arm, "group": groups_of[(algo, fold, arm)],
                    "signal": signal, "support": support, "n_seeds": int(len(per_job)), "n_windows": int(len(g)),
                    "n_fallback": int(g["fb"].sum()), "a_labels": a_lab, "a_silhouette": a_sil, "a_signal": a_sig,
                    "gain": a_sig - a_sil, "selection_loss": a_lab - a_sig, "silhouette_loss": a_lab - a_sil,
                    "recovered": rec, "recovered_note": note, **_window_stats(g["sig"] - g["sil"]),
                    "n_candidates": int(n_c), "n_defined": int(n_d),
                    "defined_rate": n_d / n_c if n_c else np.nan,
                })
    pairs = pd.DataFrame(pair_rows, columns=SUPPORT_PAIR_COLS)

    group_names = [g for g in C.STREAM_GROUPS if g in set(groups_of.values())] + ["all"]
    group_rows = []
    for signal in signals:
        d_sig = defined.xs(signal, level="signal") if signal in defined.index.get_level_values("signal") else None
        for support in SUPPORTS:
            sub_all = pairs[(pairs["signal"] == signal) & (pairs["support"] == support)]
            frame = windows_of[(signal, support)]
            for group in group_names:
                in_group = (lambda p: True) if group == "all" else (lambda p, g=group: groups_of[p] == g)
                sub = _rows(sub_all, [in_group(p) for p in zip(sub_all["algo"], sub_all["fold"], sub_all["arm"])])
                win = _rows(frame, [in_group(p) for p in zip(frame.index.get_level_values("algo"),
                                                             frame.index.get_level_values("fold"),
                                                             frame.index.get_level_values("arm"))])
                n_run = sum(1 for p in run_pairs if in_group(p))
                if d_sig is not None:
                    d = _rows(d_sig, [in_group(p) for p in d_sig.index])
                    n_c, n_d = int(d["n_candidates"].sum()), int(d["n_defined"].sum())
                else:
                    n_c = n_d = 0
                num = float((sub["a_signal"] - sub["a_silhouette"]).sum())
                den = float((sub["a_labels"] - sub["a_silhouette"]).sum())
                rec, note = _recovered(num, den, len(sub))
                group_rows.append({
                    "group": group, "signal": signal, "support": support, "n_pairs_run": n_run,
                    "n_pairs_covered": int(len(sub)), "coverage": len(sub) / n_run if n_run else np.nan,
                    "n_windows": int(len(win)), "n_fallback": int(win["fb"].sum()),
                    "n_candidates": n_c, "n_defined": n_d, "defined_rate": n_d / n_c if n_c else np.nan,
                    "a_labels": float(sub["a_labels"].mean()) if len(sub) else np.nan,
                    "a_silhouette": float(sub["a_silhouette"].mean()) if len(sub) else np.nan,
                    "a_signal": float(sub["a_signal"].mean()) if len(sub) else np.nan,
                    "gain": float(sub["gain"].mean()) if len(sub) else np.nan,
                    "recovered_num": num if len(sub) else np.nan, "recovered_den": den if len(sub) else np.nan,
                    "recovered": rec, "recovered_note": note, **_window_stats(win["sig"] - win["sil"]),
                })
    groups = pd.DataFrame(group_rows, columns=SUPPORT_GROUP_COLS)
    return pairs, groups, _missing(log, wide, base, every, fallback, sentinel, signals, groups_of, group_names)


def _missing(log, wide, base, every, fallback, sentinel, signals, groups_of, group_names) -> pd.DataFrame:
    """``support_tables``' missing/fallback table (see there)."""
    sent_known = sentinel.notna()
    rows = []
    for signal in signals:
        lg = log[log["signal"] == signal]
        own = base & wide[signal].notna()
        per_window = pd.DataFrame({"own": own, "common": every, "fb": fallback & wide[signal].isna(), "base": base,
                                   "sent": sentinel.fillna(0) > 0, "sent_known": sent_known})
        units = [("pair", p) for p in sorted(set(zip(lg["algo"], lg["fold"], lg["arm"])))]
        units += [("group", g) for g in group_names]
        for level, unit in units:
            if level == "pair":
                keep_l = (lg["algo"] == unit[0]) & (lg["fold"] == unit[1]) & (lg["arm"] == unit[2])
                idx = per_window.index
                keep_w = ((idx.get_level_values("algo") == unit[0]) & (idx.get_level_values("fold") == unit[1])
                          & (idx.get_level_values("arm") == unit[2]))
                head = {"level": "pair", "group": groups_of[unit], "algo": unit[0], "fold": unit[1], "arm": unit[2]}
            else:
                keep_l = np.asarray([unit == "all" or groups_of[p] == unit
                                     for p in zip(lg["algo"], lg["fold"], lg["arm"])], dtype=bool)
                idx = per_window.index
                keep_w = np.asarray([unit == "all" or groups_of[p] == unit for p in zip(
                    idx.get_level_values("algo"), idx.get_level_values("fold"), idx.get_level_values("arm"))],
                    dtype=bool)
                head = {"level": "group", "group": unit, "algo": None, "fold": None, "arm": None}
            sl, sw = lg[keep_l], per_window[keep_w]
            counts = sl["status"].value_counts()
            known = bool(sw["sent_known"].any())
            rows.append({
                **head, "signal": signal, "n_tail": int(len(sl)), **{s: int(counts.get(s, 0)) for s in STATUSES},
                "n_partial": int(((sl["status"] == "picked") & (sl["n_defined"] < sl["n_candidates"])).sum()),
                "n_base": int(sw["base"].sum()), "n_own": int(sw["own"].sum()), "n_common": int(sw["common"].sum()),
                "n_lost_to_common": int(sw["own"].sum() - sw["common"].sum()), "n_fallback": int(sw["fb"].sum()),
                "n_sentinel_windows": int(sw["sent"].sum()) if known else np.nan,
                "n_sentinel_common": int((sw["sent"] & sw["common"]).sum()) if known else np.nan,
            })
    return pd.DataFrame(rows, columns=MISSING_COLS)


COST_COLS = ["level", "group", "algo", "fold", "seed", "arm", "n_runs", "n_windows", "n_member_windows", "wall_s",
             "ensemble_runtime_s", "signals_runtime_s", "signals_share_of_wall", "signals_over_ensemble",
             "signals_ms_per_member_window", "log_signals"]


def cost_table(out_dir: Path, arms=("confstream",)) -> pd.DataFrame:
    """The logged signals' cost per ``ok`` run of ``arms`` and summed per stream group and over
    all runs: ``signals_runtime_s`` (seconds the runs spent on every logged reading of every
    member, together) against ``wall_s`` and against ``ensemble_runtime_s`` (the members' own
    learning and prediction), and per member-window (member rows of the run's log).

    What the runs time: ``signals_runtime_s`` wraps ``member_signals`` (and ``stability`` when
    logged) per member, including the window's distance matrices, built once and charged to the
    first member that needs them; ``rank_agg``'s ranking across members, the window's feature
    matrix and the tuner's own silhouette run outside it. A cost per signal cannot be separated
    from these logs."""
    out_dir = Path(out_dir)
    runs = report.current_runs(out_dir, diagnostic=True)
    if runs.empty or "status" not in runs:
        return pd.DataFrame(columns=COST_COLS)
    runs = runs[(runs["status"] == "ok") & runs["arm"].isin(list(arms))]
    n_rows: dict[tuple, int] = {}
    for path in report._member_files(out_dir, arms=set(arms)):
        for r in C.read_jsonl(path):
            key = (r["algo"], r["fold"], r["seed"], r["arm"])
            n_rows[key] = n_rows.get(key, 0) + 1

    def num(row, col):
        return report._num(row.get(col)) if col in row else None

    rows = []
    for row in runs.to_dict("records"):
        key = (row["algo"], row["fold"], row["seed"], row["arm"])
        wall, sig, ens = num(row, "wall_s"), num(row, "signals_runtime_s"), num(row, "ensemble_runtime_s")
        n_mw = n_rows.get(key, 0)
        logged = row.get("log_signals")
        rows.append({
            "level": "run", "group": C.stream_group(row["fold"]), "algo": row["algo"], "fold": row["fold"],
            "seed": row["seed"], "arm": row["arm"], "n_runs": 1, "n_windows": num(row, "n_windows"),
            "n_member_windows": n_mw, "wall_s": wall, "ensemble_runtime_s": ens, "signals_runtime_s": sig,
            "log_signals": ", ".join(logged) if isinstance(logged, (list, tuple)) else None,
        })
    out = pd.DataFrame(rows, columns=COST_COLS)
    if out.empty:
        return out
    for col in ("wall_s", "ensemble_runtime_s", "signals_runtime_s", "n_windows"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    summary = []  # a total with a run missing its reading is unknown (NaN), not a partial sum
    for group in [*(g for g in C.STREAM_GROUPS if g in set(out["group"])), "all"]:
        sub = out if group == "all" else out[out["group"] == group]
        summary.append({"level": "group" if group != "all" else "all", "group": group, "n_runs": int(len(sub)),
                        **{c: float(sub[c].sum(min_count=len(sub))) for c in
                           ("n_windows", "wall_s", "ensemble_runtime_s", "signals_runtime_s")},
                        "n_member_windows": int(sub["n_member_windows"].sum())})
    out = pd.concat([out, pd.DataFrame(summary, columns=COST_COLS)], ignore_index=True)
    for col in ("wall_s", "ensemble_runtime_s", "signals_runtime_s", "n_windows"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["signals_share_of_wall"] = out["signals_runtime_s"] / out["wall_s"]
    out["signals_over_ensemble"] = out["signals_runtime_s"] / out["ensemble_runtime_s"]
    out["signals_ms_per_member_window"] = np.where(out["n_member_windows"] > 0, 1e3 * out["signals_runtime_s"]
                                                   / out["n_member_windows"].where(out["n_member_windows"] > 0), np.nan)
    return out


def screen_tables(picks: pd.DataFrame, tail: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (algo, fold, arm, signal) and per stream group: the picks compared on the tail
    windows (from each job's ``tail_from``) where labels, the silhouette and the signal all
    picked -- each signal's OWN support (``support_tables`` has the paired ones). Returns
    ``(pairs, groups)``."""
    pair_cols = ["algo", "fold", "arm", "signal", "n_seeds", "n_windows", "a_labels", "a_silhouette",
                 "a_signal", "selection_loss"]
    if picks.empty:
        return pd.DataFrame(columns=pair_cols), pd.DataFrame(
            columns=["group", "signal", "n_pairs", "a_signal", "selection_loss", "recovered"])
    starts = [tail.get(k, C.FREEZE_AFTER_WINDOW)
              for k in zip(picks["algo"], picks["fold"], picks["seed"], picks["arm"])]
    picks = picks[picks["window"] >= np.array(starts)]
    wide = picks.pivot_table(index=["algo", "fold", "seed", "arm", "window"], columns="signal",
                             values="ari", aggfunc="first")
    signals = [s for s in wide.columns if s != LABELS]
    rows = []
    for signal in signals:
        cols = list(dict.fromkeys([LABELS, SILHOUETTE, signal]))
        if not set(cols) <= set(wide.columns):
            continue
        sub = wide[cols].dropna()
        if sub.empty:
            continue
        per_job = sub.groupby(level=["algo", "fold", "seed", "arm"]).agg(["mean", "size"])
        for (algo, fold, arm), g in per_job.groupby(level=["algo", "fold", "arm"]):
            a_lab = float(g[(LABELS, "mean")].mean())
            a_sil = float(g[(SILHOUETTE, "mean")].mean())
            a_sig = float(g[(signal, "mean")].mean())
            rows.append({"algo": algo, "fold": fold, "arm": arm, "signal": signal, "n_seeds": int(len(g)),
                         "n_windows": int(g[(LABELS, "size")].sum()), "a_labels": a_lab,
                         "a_silhouette": a_sil, "a_signal": a_sig, "selection_loss": a_lab - a_sig})
    pairs = pd.DataFrame(rows, columns=pair_cols)
    out = []
    pairs_g = pairs.assign(group=[C.stream_group(f) for f in pairs["fold"]])
    for signal, g in pairs_g.groupby("signal", sort=False):
        for group in (*C.STREAM_GROUPS, "all"):
            sub = g if group == "all" else g[g["group"] == group]
            den = float((sub["a_labels"] - sub["a_silhouette"]).sum())
            out.append({"group": group, "signal": signal, "n_pairs": int(len(sub)),
                        "a_signal": float(sub["a_signal"].mean()) if len(sub) else np.nan,
                        "selection_loss": float(sub["selection_loss"].mean()) if len(sub) else np.nan,
                        "recovered": float((sub["a_signal"] - sub["a_silhouette"]).sum()) / den
                        if len(sub) and den > 0 else np.nan})
    return pairs, pd.DataFrame(out)


# --- the exploratory pilot section of screen.md -----------------------------------------------

def _f(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{digits}f}"


def _pilot_markdown(log, tail, pairs, groups, missing, cost, n_signals: int) -> list[str]:
    from . import analyze

    jobs = log[["algo", "fold", "seed", "arm"]].drop_duplicates() if len(log) else log
    runs_desc = (f"{len(jobs)} runs: {', '.join(sorted(set(jobs['algo'])))} × "
                 f"{', '.join(sorted(set(jobs['fold'])))}, seeds {', '.join(str(s) for s in sorted(set(jobs['seed'])))}"
                 if len(jobs) else "no runs")
    lines = [
        "## Paired supports, coverage, missing readings and cost",
        "",
        f"**Exploratory.** Read from {runs_desc}. The streams screened here are development data for the "
        "choice of a signal: whatever this section suggests has to be confirmed on streams the screen never "
        "read, and these runs are never to be renamed held-out."
        + (" With one seed, no spread across seeds is in these numbers, and windows of one run are not "
           "independent draws." if len(jobs) and jobs["seed"].nunique() == 1 else ""),
        "",
        "The tables above read each signal on its own support (the tail windows where labels, the silhouette and "
        "that signal picked), so their shares are not directly comparable across signals. Here every signal is "
        "also read on two supports that are the same windows for all of them: `common`, the tail windows where "
        f"labels, the silhouette and every logged signal ({n_signals} of them) picked, the only support on which "
        "recovery ratios compare signals directly without imputing anything; and `fallback`, the tail windows "
        "where labels and the silhouette picked in the runs that logged every signal, a signal that picked nobody "
        "taking the silhouette's pick (it is charged for the windows it cannot read). `recovered` = num / den, num = Σ over the covered pairs of (a_signal − "
        "a_silhouette), den = Σ of (a_labels − a_silhouette), each a seed mean of per-run means over the "
        "support's windows; undefined (—) with its reason when den ≤ 0. `gain` = a_signal − a_silhouette, the "
        "mean over pairs; `per window` is the paired difference of the two picks' ARI pooled over the windows, "
        "with the windows the signal's pick read better / worse / the same. `defined` is the share of the tail's "
        "candidate member-windows (members usable on t-1 and active on t) on which the signal has a reading. "
        "Pair coverage is the pairs with at least one window on the support over the pairs run.",
        "",
    ]
    if groups.empty:
        return lines + ["No tail window has a pick by labels and the silhouette: nothing to compare.", ""]
    shown = [g for g in groups["group"].unique() if g != "all"]
    for group in (shown if len(shown) == 1 else [*shown, "all"]):
        gsub = groups[groups["group"] == group]
        table = pd.DataFrame([{
            "signal": r["signal"], "support": r["support"], "pairs": f"{r['n_pairs_covered']}/{r['n_pairs_run']}",
            "windows": r["n_windows"], "defined (of n)": f"{_f(r['defined_rate'])} ({r['n_candidates']})",
            "recovered": _f(r["recovered"]) if not r["recovered_note"] else f"— ({r['recovered_note'].split(':')[0]})",
            "num / den": f"{_f(r['recovered_num'])} / {_f(r['recovered_den'])}",
            "gain": _f(r["gain"]), "per window": _f(r["window_gain_mean"]),
            "better / worse / same": f"{r['n_better']} / {r['n_worse']} / {r['n_tied']}",
            "fell back": r["n_fallback"] if r["support"] == "fallback" else "",
        } for r in gsub.to_dict("records")])
        label = "all pairs" if group == "all" else f"group `{group}`" + (" (every pair screened)" if len(shown) == 1 else "")
        lines += [f"**Recovery by support, {label}**", "", analyze._md_table(table), ""]
    m_all = missing[(missing["level"] == "group") & (missing["group"] == "all")]
    if len(m_all) and (m_all["n_lost_to_common"] == 0).all() and (m_all["n_fallback"] == 0).all():
        lines += [f"On these logs every signal picked on every tail window where labels and the silhouette did "
                  f"({int(m_all['n_base'].iloc[0])} windows): the three supports are the same windows, so the "
                  "own-support shares above are paired after all and compare directly.", ""]
    elif len(m_all):
        lost = m_all["n_lost_to_common"]
        lines += [f"The common support drops {int(lost.min())} to {int(lost.max())} windows of a signal's own "
                  "support; only its ratios, and the fallback ones, compare signals on the same windows.", ""]

    common = pairs[pairs["support"] == "common"]
    if len(common):
        piv = common.pivot_table(index=["algo", "fold"], columns="signal", values="recovered", aggfunc="first")
        den = common.groupby(["algo", "fold"])[["silhouette_loss", "n_windows"]].first()
        piv = den.rename(columns={"silhouette_loss": "den", "n_windows": "windows"}).join(piv).reset_index()
        lines += ["**Per pair, common support: recovered** (den = a_labels − a_silhouette of the pair; — where "
                  "den ≤ 0; one pair can carry a group's ratio when its den dominates the sum)", "",
                  analyze._md_table(piv), ""]
        by = common.assign(num=common["a_signal"] - common["a_silhouette"]).groupby(["signal", "fold"])[
            ["num", "silhouette_loss"]].sum()
        rec = by["num"].where(by["silhouette_loss"] > 0) / by["silhouette_loss"].where(by["silhouette_loss"] > 0)
        per_fold = rec.unstack("fold")
        den_row = by["silhouette_loss"].xs(SILHOUETTE, level="signal") if SILHOUETTE in per_fold.index else None
        table = per_fold.reset_index()
        if den_row is not None:
            table = pd.concat([pd.DataFrame([{"signal": "(den)", **den_row.to_dict()}]), table], ignore_index=True)
        lines += ["**Per stream, common support: recovered** (sums over the stream's pairs; the first row is the "
                  "stream's den, its share of the group's)", "", analyze._md_table(table), ""]

    m = missing[missing["level"] == "group"]
    m = m[m["group"] == ("all" if len(shown) != 1 else shown[0])]
    if len(m):
        mt = m[["signal", "n_tail", "picked", "undefined", "no_ari", "not_logged", "no_candidate", "n_partial",
                "n_base", "n_own", "n_common", "n_lost_to_common", "n_fallback"]].rename(columns={
                    "n_tail": "tail windows", "no_ari": "no ARI", "not_logged": "not logged",
                    "no_candidate": "no candidate", "n_partial": "picked among some", "n_base": "labels+sil.",
                    "n_own": "own", "n_common": "common", "n_lost_to_common": "lost to common",
                    "n_fallback": "fell back"})
        lines += ["**Missing readings and fallbacks** (tail windows; `undefined`: candidates but no reading on any; "
                  "`picked among some`: a pick made while some candidates had no reading; `labels+sil.`: windows "
                  "where labels and the silhouette picked; `fell back`: windows of the fallback support where the signal "
                  "picked nobody; `lost to common`: windows of the "
                  "signal's own support that the common support drops because another signal did not pick)", "",
                  analyze._md_table(mt), ""]

    lines += _cost_markdown(cost)
    lines += _noise_markdown(log, tail, missing)
    lines += [
        "DBCV readings (`dbcv`, `dbcv_std`) are the paper's value for the minimum spanning tree scipy returns; "
        "mutual reachability distances tie by construction, so another tree of the same weight can give another "
        "value (`benchmark.tuning.signals.dbcv`). How far the screen moves under another tie rule is not "
        "measured: it needs new runs.",
        "",
    ]
    return lines


def _cost_markdown(cost: pd.DataFrame) -> list[str]:
    from . import analyze

    if cost.empty or cost["signals_runtime_s"].isna().all():
        return ["**Cost.** The runs record no `signals_runtime_s`: the cost of the signals is not known.", ""]
    run = cost[cost["level"] == "run"]
    table = run[["algo", "fold", "seed", "n_member_windows", "wall_s", "ensemble_runtime_s", "signals_runtime_s",
                 "signals_share_of_wall", "signals_over_ensemble", "signals_ms_per_member_window"]].rename(columns={
                     "n_member_windows": "member-windows", "wall_s": "wall s", "ensemble_runtime_s": "members s",
                     "signals_runtime_s": "signals s", "signals_share_of_wall": "signals / wall",
                     "signals_over_ensemble": "signals / members", "signals_ms_per_member_window": "ms per member-window"})
    total = cost[cost["level"] == "all"].iloc[0]
    share = run["signals_share_of_wall"]
    logged = sorted(set(run["log_signals"].dropna()))
    what = "; ".join(logged) or "none recorded"
    if any("rank_agg" in s for s in logged):
        what += ", with rank_agg's components calinski_harabasz_std and davies_bouldin_std"
    return [
        "**Cost of the logged signals**", "",
        analyze._md_table(table, digits=2), "",
        f"Over the {int(total['n_runs'])} runs the logged signals ({what}) took "
        f"{total['signals_runtime_s']:.0f} s of {total['wall_s']:.0f} s wall ({total['signals_share_of_wall']:.2f}; "
        f"per run {share.min():.2f} to {share.max():.2f}), {total['signals_over_ensemble']:.2f} times what the "
        "members spent learning and predicting. The runs time every logged reading of a member together, the "
        "window's distance matrices included (built once, charged to the first member that needs them); "
        "`rank_agg`'s ranking across members and the tuner's own silhouette run outside that timer. The cost of "
        "one signal alone cannot be separated from these logs.", "",
    ]


def _noise_markdown(log: pd.DataFrame, tail: dict, missing: pd.DataFrame) -> list[str]:
    head = ("**Noise policy.** The readings logged here keep a sentinel or noise label as an ordinary cluster "
            "label (`signals.member_signals`'s default `noise=\"label\"`). A screen under another policy "
            "(`signals.NOISE_POLICIES`) needs new runs: member predictions are not logged, so no reading can be "
            "recomputed from these logs.")
    known = log["n_sentinel_members"].notna() if len(log) else pd.Series(dtype=bool)
    if not len(log) or not known.any():
        return [head + " These logs do not count sentinels (`n_sentinel`), so how many readings a policy "
                "would move is not known either.", ""]
    windows = log[known].drop_duplicates(JOB_WINDOW)
    exposed = windows[windows["n_sentinel_members"] > 0]
    tail_w = _in_tail(windows, tail)
    tail_exposed = tail_w[tail_w["n_sentinel_members"] > 0]
    common = missing.loc[(missing["level"] == "group") & (missing["group"] == "all"), "n_sentinel_common"]
    n_common = int(common.max()) if len(common) and common.notna().any() else 0
    where = (f"on {len(exposed)} of {len(windows)} reading windows, the latest window read being "
             f"{int(exposed['window'].max()) - 1}" if len(exposed) else f"on none of the {len(windows)} reading windows")
    body = (f" The logs do bound the exposure: a feasible member's predictions held a sentinel {where}; "
            f"{len(tail_exposed)} of the {len(tail_w)} tail windows and {n_common} common-support windows had one.")
    if len(tail_exposed) == 0:
        body += (" No reading the screen compares involved a sentinel, so on these logs the policy could not have "
                 "moved the screen.")
    body += (" `n_sentinel` counts the labels `adapter.SENTINEL_LABELS` names (DenStream's -1); a model emitting "
             "another noise label would not be counted.")
    return [head + body, ""]


def write_screen(out_dir: Path, arms=("confstream",), rep_dir: Path | None = None) -> list[Path]:
    """``screen_tables`` as ``<rep>/screen/{pairs,groups}.csv`` and ``<rep>/screen.md``,
    ``<rep>`` being ``<out_dir>/report`` unless ``rep_dir``; then the paired supports, the
    missing readings and the cost (``support_tables``, ``cost_table``) as
    ``<rep>/screen/{support_pairs,support_groups,missing,cost,pick_log}.csv`` and an exploratory
    section appended to ``screen.md``. The first two tables and their files are unchanged."""
    from . import analyze

    out_dir = Path(out_dir)
    rep = Path(rep_dir) if rep_dir is not None else out_dir / C.REPORT_DIR.name
    (rep / "screen").mkdir(parents=True, exist_ok=True)
    pairs, groups = screen_tables(signal_picks(out_dir, arms), tail_from(out_dir))
    paths = [rep / "screen" / "pairs.csv", rep / "screen" / "groups.csv", rep / "screen.md"]
    pairs.to_csv(paths[0], index=False)
    groups.to_csv(paths[1], index=False)
    logged = sorted(set(groups["signal"]) - {SILHOUETTE}) if len(groups) else []
    lines = [
        "# Screening of selection signals",
        "",
        f"Read from the member logs of `{out_dir}` (arms {', '.join(arms)}). After window t-1, each signal picks "
        "the member with its highest reading among those active on t and usable on t-1; the pick is read by "
        "its ARI on t, the delay the silhouette has. `selection_loss` is the labels' pick minus the signal's; "
        "`recovered` is the share of the silhouette's selection loss the signal wins back (ratio of sums over "
        "the pairs of a group: 1 as good as labels, 0 no better than the silhouette). Only tail windows on "
        "which labels, the silhouette and the signal all picked are compared.",
        "",
    ]
    if not logged:
        lines += ["No `sig_<name>` reading is in these member logs: run `run.py --log-members --signals ...` "
                  "to screen candidate signals. Only the silhouette is read.", ""]
    piv = groups.pivot_table(index="signal", columns="group", values="recovered").reset_index() if len(groups) else groups
    if len(groups):
        lines += ["**Share of the silhouette's selection loss recovered**", "",
                  analyze._md_table(piv.reindex(columns=["signal", *C.STREAM_GROUPS, "all"])), "",
                  "**Selection loss**", "",
                  analyze._md_table(groups.pivot_table(index="signal", columns="group", values="selection_loss")
                                    .reset_index().reindex(columns=["signal", *C.STREAM_GROUPS, "all"])), ""]
    lines += [
        "The screen reads the ensemble that the silhouette evolved, so it says which signal picks better among "
        "those configurations, not what an ensemble driven by the signal would have grown. A signal that "
        "screens well is then run as a signal arm (`--arms confstream_<signal>`), which selects by it "
        "throughout.",
        "",
    ]
    if len(groups):
        lines += ["Each signal is read above on its own support, so the shares are not directly comparable across "
                  "signals; the common-support table below is.", ""]

    log = pick_log(out_dir, arms)
    tail, run_pairs = tail_from(out_dir), pairs_run(out_dir, arms)
    s_pairs, s_groups, missing = support_tables(log, tail, run_pairs)
    cost = cost_table(out_dir, arms)
    extra = {"support_pairs.csv": s_pairs, "support_groups.csv": s_groups, "missing.csv": missing,
             "cost.csv": cost, "pick_log.csv": log}
    for name, frame in extra.items():
        frame.to_csv(rep / "screen" / name, index=False)
        paths.insert(-1, rep / "screen" / name)
    n_signals = len(set(log["signal"]) - {LABELS, SILHOUETTE}) if len(log) else 0
    lines += _pilot_markdown(log, tail, s_pairs, s_groups, missing, cost, n_signals)
    paths[-1].write_text("\n".join(lines), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(C.OUT))
    parser.add_argument("--arms", default="confstream", help="comma-separated online arms whose member logs to read")
    args = parser.parse_args()
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for p in write_screen(Path(args.out_dir), arms=arms):
        print(p)


if __name__ == "__main__":
    main()
