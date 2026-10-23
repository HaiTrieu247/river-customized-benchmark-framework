"""E4: does the silhouette limit the tuner only by what it picks, or also by what it grows?

Three readings of the same (algo, fold, seed), on the same tail windows:

- ``a_silhouette``: the online tuner as run (``confstream``), its incumbent read one window later;
- ``a_label_replay``: within that same silhouette-grown ensemble, the member labels would have
  picked after window t-1, read on t (``report.regret_windows``' ``a_star_lag``; needs the
  confstream runs' member logs);
- ``a_oracle``: the diagnostic tuner whose every selection -- incumbent, victims, parents -- is
  made by the ARI of the window just processed (``--arms confstream_ari_oracle[_constrained]``,
  see ``benchmark.tuning.confstream.ORACLE_SIGNALS``), read the same way; every oracle arm the
  dir holds is read, one row each (``oracle_arm``).

The labels' pick within the silhouette's ensemble is made among the members that had a
partition on t-1 (the silhouette's gate), ties going to the incumbent, then the smallest id.
``confstream_ari_oracle_constrained`` keeps that gate, so its ``d_search`` compares like with
like (``like_for_like``); the unconstrained oracle also admits one-label members, and its
``d_search`` mixes that wider eligibility with the search. The oracle tuner breaks ties as the
tuner does (fewer micro-clusters, the incumbent, the oldest), not as the replay does.

``d_replay = a_label_replay - a_silhouette`` is the conditional selection gap under the
silhouette-driven search; ``d_search = a_oracle - a_label_replay`` what letting labels drive the
search as well changes; ``d_trajectory = a_oracle - a_silhouette`` the two together. Both arms
start from the same seed, so their first windows match, and they part at their first different
selection. The differences are operational comparisons of two runs and are reported signed: none
of them is a non-negative loss by construction, nor a causal split of the gap into "signal" and
"search" -- the oracle reads labels a deployed tuner never has, and a label-driven ensemble can
grow members whose silhouette would have shed them. Only windows where all three read (from each
run's ``tail_from``, external metrics defined) are compared.

Run from the repo root::

    python -m conf_pipeline.oracle_compare --out-dir conf_pipeline/out/<dir> [--oracle-arms ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from . import report

READINGS = ("a_silhouette", "a_label_replay", "a_oracle")
DIFFS = ("d_replay", "d_search", "d_trajectory")
JOB = ["algo", "fold", "seed", "oracle_arm"]
WINDOW_COLS = [*JOB, "window", *READINGS]
#: The oracle arm whose eligibility matches the replay's (the silhouette's gate kept).
LIKE_FOR_LIKE = "confstream_ari_oracle_constrained"


def _group(fold: str) -> str:
    try:
        return C.stream_group(fold)
    except ValueError:
        return "unassigned"


def window_readings(out_dir: Path, arm: str = "confstream", oracle_arms=C.DIAGNOSTIC_ARMS) -> pd.DataFrame:
    """One row per (oracle arm, tail window) where the three readings exist."""
    out_dir = Path(out_dir)
    runs = report.current_runs(out_dir, diagnostic=True)
    present = [a for a in oracle_arms if not runs.empty and a in set(runs["arm"])]
    if runs.empty or arm not in set(runs["arm"]) or not present:
        return pd.DataFrame(columns=WINDOW_COLS)
    reg = report.regret_windows(out_dir, arms=(arm,), region="tail")
    reg = reg[reg["used"].fillna(False).astype(bool)]
    base = reg.rename(columns={"lag_ari_win": "a_silhouette", "a_star_lag": "a_label_replay"})
    base = base[["algo", "fold", "seed", "window", "a_silhouette", "a_label_replay"]]
    win = report.load_windows(out_dir, diagnostic=True)
    tail = report._tail_from_by_job(runs)
    frames = []
    for oracle_arm in present:
        w = win[win["arm"] == oracle_arm]
        keep = [i >= tail.get((a, f, s, oracle_arm), C.FREEZE_AFTER_WINDOW)
                for a, f, s, i in zip(w["algo"], w["fold"], w["seed"], w["window"])]
        ora = w[np.asarray(keep, dtype=bool)][["algo", "fold", "seed", "window", "lag_ari_win"]]
        merged = base.merge(ora.rename(columns={"lag_ari_win": "a_oracle"}), on=["algo", "fold", "seed", "window"])
        frames.append(merged.assign(oracle_arm=oracle_arm))
    merged = pd.concat(frames, ignore_index=True)
    for col in READINGS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged.dropna(subset=list(READINGS)).sort_values([*JOB, "window"]).reset_index(drop=True)[WINDOW_COLS]


def job_means(windows: pd.DataFrame) -> pd.DataFrame:
    cols = [*JOB, "like_for_like", "n_windows", *READINGS, *DIFFS]
    if windows.empty:
        return pd.DataFrame(columns=cols)
    g = windows.groupby(JOB, sort=True)
    out = g[list(READINGS)].mean().reset_index()
    out["n_windows"] = g.size().to_numpy()
    out["like_for_like"] = out["oracle_arm"] == LIKE_FOR_LIKE
    out["d_replay"] = out["a_label_replay"] - out["a_silhouette"]
    out["d_search"] = out["a_oracle"] - out["a_label_replay"]
    out["d_trajectory"] = out["a_oracle"] - out["a_silhouette"]
    return out[cols]


def pair_means(jobs: pd.DataFrame) -> pd.DataFrame:
    """Per (algo, fold): the mean over its seeds, each seed one paired job."""
    cols = ["oracle_arm", "algo", "fold", "group", "n_seeds", "n_windows", *READINGS, *DIFFS]
    if jobs.empty:
        return pd.DataFrame(columns=cols)
    g = jobs.groupby(["oracle_arm", "algo", "fold"], sort=True)
    out = g[list(READINGS) + list(DIFFS)].mean().reset_index()
    out["n_seeds"] = g.size().to_numpy()
    out["n_windows"] = g["n_windows"].sum().to_numpy()
    out["group"] = [_group(f) for f in out["fold"]]
    return out[cols]


def group_means(pairs: pd.DataFrame) -> pd.DataFrame:
    cols = ["oracle_arm", "group", "n_pairs", *READINGS, *DIFFS, "share_d_search_negative"]
    rows = []
    for oracle_arm, by_arm in pairs.groupby("oracle_arm", sort=True):
        for group in sorted(set(by_arm["group"])) + ["all"]:
            sub = by_arm if group == "all" else by_arm[by_arm["group"] == group]
            if sub.empty:
                continue
            rows.append({"oracle_arm": oracle_arm, "group": group, "n_pairs": len(sub),
                         **{c: float(sub[c].mean()) for c in (*READINGS, *DIFFS)},
                         "share_d_search_negative": float((sub["d_search"] < 0).mean())})
    return pd.DataFrame(rows, columns=cols)


def compare(out_dir: Path, arm: str = "confstream", oracle_arms=C.DIAGNOSTIC_ARMS):
    """``(jobs, windows)``: the paired readings per (job, oracle arm) and per window."""
    windows = window_readings(out_dir, arm, oracle_arms)
    return job_means(windows), windows


def write_compare(out_dir: Path, arm: str = "confstream", oracle_arms=C.DIAGNOSTIC_ARMS) -> list[Path]:
    from .analyze import _md_table

    jobs, windows = compare(out_dir, arm, oracle_arms)
    pairs = pair_means(jobs)
    groups = group_means(pairs) if len(pairs) else pd.DataFrame()
    rep = Path(out_dir) / C.REPORT_DIR.name / "oracle_compare"
    rep.mkdir(parents=True, exist_ok=True)
    paths = [rep / "jobs.csv", rep / "pairs.csv", rep / "groups.csv", rep / "windows.csv", rep / "oracle_compare.md"]
    for frame, path in zip((jobs, pairs, groups, windows), paths):
        frame.to_csv(path, index=False)
    lines = [
        "# The silhouette's tuner, the labels' pick within it, and the labels' tuner (E4)", "",
        f"Read from `{out_dir}`: `{arm}` (online tuning by the silhouette, its member logs giving the labels' "
        f"pick one window ahead) and the oracle arms {', '.join(f'`{a}`' for a in oracle_arms)} (the diagnostic "
        "tuner selecting by the ARI of the window just processed), paired by (algo, fold, seed) on the tail "
        "windows where all three read. The labels' pick within the silhouette's ensemble keeps the silhouette's "
        "gate (members with a partition on t-1) and breaks ties to the incumbent, then the smallest id; "
        f"`{LIKE_FOR_LIKE}` keeps the same gate, so its `d_search` compares like with like (`like_for_like`), while "
        "the unconstrained oracle also admits one-label members, and both oracles break ties as the tuner does. "
        "`d_replay` = labels' pick within the silhouette's ensemble minus the silhouette's pick (the conditional "
        "selection gap under silhouette-driven search); `d_search` = the labels' tuner minus that pick; "
        "`d_trajectory` = the labels' tuner minus the silhouette's. These are signed operational comparisons of "
        "two runs, not losses that cannot be negative and not a causal split into signal and search.", "",
    ]
    if jobs.empty:
        lines.append(f"No (algo, fold, seed) has both `{arm}` with member logs and an oracle arm: nothing compared.")
    else:
        lines += ["## By stream group (mean over pairs)", "", _md_table(groups, 3), "",
                  "## Per (algo, fold)", "", _md_table(pairs, 3), ""]
    paths[4].write_text("\n".join(lines), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--arm", default="confstream", help="the silhouette-driven arm, with member logs")
    parser.add_argument("--oracle-arms", default=",".join(C.DIAGNOSTIC_ARMS),
                        help=f"comma-separated, from {C.DIAGNOSTIC_ARMS} (default: every one the dir holds)")
    args = parser.parse_args()
    arms = tuple(a.strip() for a in args.oracle_arms.split(",") if a.strip())
    bad = [a for a in arms if a not in C.DIAGNOSTIC_ARMS]
    if bad:
        raise SystemExit(f"unknown oracle arms {bad}; known: {C.DIAGNOSTIC_ARMS}")
    for p in write_compare(Path(args.out_dir), args.arm, arms):
        print(p)


if __name__ == "__main__":
    main()
