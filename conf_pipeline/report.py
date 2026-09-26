"""Tables and curves from ``out/runs.jsonl`` and ``out/windows.jsonl``.

Everything reported about quality is the *lagged* reading: the configuration
that was incumbent after window ``t-1``, scored on window ``t``. The same-window
score of the incumbent is the maximum of ``e_size`` noisy values and is kept in
the rows only so the gap between the two can be seen.

Run from the repo root::

    python -m conf_pipeline.report [--out-dir conf_pipeline/out]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C

#: Per-run fields averaged over seeds, and shown in the summary table.
SUMMARY_FIELDS = (
    "lag_ari_win_mean", "lag_score_mean", "lag_ari_cumulative",
    "incumbent_throughput_sps", "ensemble_runtime_s", "wall_s",
    "lag_memory_mb_peak", "lag_n_pred_labels_mean", "lag_n_macro_clusters_mean",
    "n_switches", "n_accepted_total", "n_reset_total",
    "lag_undefined_share", "lag_infeasible_share", "single_class_window_share",
    "lag_n_sentinel_total", "s_online", "lam",
)
GROUP = ["algo", "fold", "arm"]


def load_runs(out_dir: Path) -> pd.DataFrame:
    rows = list(C.read_jsonl(Path(out_dir) / "runs.jsonl"))
    if not rows:
        return pd.DataFrame(columns=["algo", "fold", "seed", "arm", "status"])
    return pd.DataFrame(rows)


def load_windows(out_dir: Path) -> pd.DataFrame:
    rows = list(C.read_jsonl(Path(out_dir) / "windows.jsonl"))
    if not rows:
        return pd.DataFrame(columns=["algo", "fold", "seed", "arm", "window"])
    return pd.DataFrame(rows)


def summary_table(out_dir: Path) -> pd.DataFrame:
    """One row per (algo, fold, arm): mean and sd over the seeds that finished."""
    runs = load_runs(out_dir)
    if runs.empty:
        return pd.DataFrame(columns=GROUP)
    ok = runs[runs["status"] == "ok"]
    failed = runs[runs["status"] != "ok"]
    out = []
    for key, grp in runs.groupby(GROUP, sort=True):
        good = ok[(ok[GROUP] == key).all(axis=1)] if not ok.empty else ok
        bad = failed[(failed[GROUP] == key).all(axis=1)] if not failed.empty else failed
        row = dict(zip(GROUP, key))
        row["n_seeds"] = int(len(good))
        row["n_failed"] = int(len(bad))
        row["failed_status"] = ",".join(sorted(set(bad["status"]))) if len(bad) else None
        for f in SUMMARY_FIELDS:
            vals = pd.to_numeric(good[f], errors="coerce").dropna() if f in good else pd.Series(dtype=float)
            row[f] = float(vals.mean()) if len(vals) else None
            row[f"{f.removesuffix('_mean')}_sd" if f.endswith("_mean") else f"{f}_sd"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else None
            )
        out.append(row)
    table = pd.DataFrame(out)
    # ``lag_ari_win_mean`` -> sd column named ``lag_ari_win_sd`` for readability
    return table


def gain_table(out_dir: Path) -> pd.DataFrame:
    """The tuner against the default arm, per (algo, fold)."""
    table = summary_table(out_dir)
    if table.empty:
        return pd.DataFrame(columns=["algo", "fold"])
    metrics = ("lag_ari_win_mean", "lag_score_mean", "lag_ari_cumulative", "incumbent_throughput_sps")
    conf = table[table["arm"] == "confstream"].set_index(["algo", "fold"])
    base = table[table["arm"] == "default"].set_index(["algo", "fold"])
    keys = sorted(set(conf.index) | set(base.index))
    out = []
    for algo, fold in keys:
        row = {"algo": algo, "fold": fold}
        for m in metrics:
            c = conf[m].get((algo, fold)) if (algo, fold) in conf.index else None
            b = base[m].get((algo, fold)) if (algo, fold) in base.index else None
            row[f"{m}_confstream"] = c
            row[f"{m}_default"] = b
        c, b = row["lag_ari_win_mean_confstream"], row["lag_ari_win_mean_default"]
        row["gain_lag_ari_win"] = (c - b) if c is not None and b is not None else np.nan
        c, b = row["incumbent_throughput_sps_confstream"], row["incumbent_throughput_sps_default"]
        row["throughput_ratio"] = (c / b) if c and b else np.nan
        out.append(row)
    return pd.DataFrame(out)


def curves(out_dir: Path) -> dict[str, pd.DataFrame]:
    """Per fold: lagged ARI and silhouette vs window, averaged over seeds."""
    win = load_windows(out_dir)
    if win.empty:
        return {}
    cols = [c for c in ("lag_ari_win", "lag_score", "score", "lag_ari_cum", "lag_n_pred_labels",
                        "ensemble_runtime_s", "wall_s") if c in win]
    out = {}
    for fold, grp in win.groupby("fold"):
        agg = (
            grp.groupby(["algo", "arm", "window"], sort=True)[cols]
            .agg(["mean", "count"])
        )
        agg.columns = [f"{a}_{b}" if b != "mean" else a for a, b in agg.columns]
        out[fold] = agg.reset_index()
    return out


def _md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n" for r in df.to_dict("records"))
    return head + body


def write_report(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    rep = out_dir / "report"
    rep.mkdir(parents=True, exist_ok=True)
    written = []

    table = summary_table(out_dir)
    table.to_csv(rep / "summary.csv", index=False)
    written.append(rep / "summary.csv")
    gain = gain_table(out_dir)
    gain.to_csv(rep / "gain.csv", index=False)
    written.append(rep / "gain.csv")
    for fold, df in curves(out_dir).items():
        p = rep / f"curves_{fold}.csv"
        df.to_csv(p, index=False)
        written.append(p)

    runs = load_runs(out_dir)
    lines = ["# conf_pipeline report", ""]
    lines.append(
        "Every quality number is the *lagged* reading: the incumbent named after "
        "window t-1, scored on window t. `score` is the batch silhouette in [-1, 1] "
        "(-1 when undefined); `ari_win` is the ARI inside the window; "
        "`lag_ari_cumulative` is the ARI over the whole stream of lagged incumbents."
    )
    lines += ["", "## Summary (mean over seeds)", ""]
    if not table.empty:
        cols = ["algo", "fold", "arm", "n_seeds", "lag_ari_win_mean", "lag_ari_win_sd",
                "lag_score_mean", "lag_ari_cumulative", "incumbent_throughput_sps",
                "ensemble_runtime_s", "lag_memory_mb_peak", "lag_n_pred_labels_mean",
                "n_switches", "lag_undefined_share", "single_class_window_share"]
        lines.append(_md_table(table, [c for c in cols if c in table]))
    lines += ["", "## Tuner vs default", ""]
    if not gain.empty:
        cols = ["algo", "fold", "lag_ari_win_mean_confstream", "lag_ari_win_mean_default",
                "gain_lag_ari_win", "lag_score_mean_confstream", "lag_score_mean_default",
                "throughput_ratio"]
        lines.append(_md_table(gain, [c for c in cols if c in gain]))
    failed = runs[runs["status"] != "ok"] if not runs.empty else runs
    lines += ["", "## Runs that did not finish", ""]
    if len(failed):
        lines.append(_md_table(failed, ["algo", "fold", "seed", "arm", "status", "error"]))
    else:
        lines.append("none")
    lines += ["", "## Files", ""]
    lines += [f"- `{p.name}`" for p in written]
    md = rep / "report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(md)

    written += _plots(out_dir, rep)
    return written


def _plots(out_dir: Path, rep: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib is optional (requirements.txt)
        return []
    written = []
    for fold, df in curves(out_dir).items():
        if "lag_ari_win" not in df:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for (algo, arm), grp in df.groupby(["algo", "arm"]):
            style = "-" if arm == "confstream" else "--"
            axes[0].plot(grp["window"], grp["lag_ari_win"], style, label=f"{algo} ({arm})")
            if "lag_score" in grp:
                axes[1].plot(grp["window"], grp["lag_score"], style, label=f"{algo} ({arm})")
        axes[0].set_title(f"{fold}: lagged ARI per window")
        axes[1].set_title(f"{fold}: lagged silhouette per window")
        for ax in axes:
            ax.set_xlabel("window")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        p = rep / f"curves_{fold}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(C.OUT))
    args = parser.parse_args()
    for p in write_report(Path(args.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
