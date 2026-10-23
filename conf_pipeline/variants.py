"""Protocol constants varied one at a time (X6): do the conclusions hang on h, E, C or t0?

Every number of the study is read under four constants the protocol fixes: the window
``h``, the ensemble size ``E`` (``e_size``), the challengers per window ``C``
(``e_test``) and the window ``t0`` where ``frozen_prefix`` freezes and every arm's tail
starts (``freeze_after_window``). ``VARIANTS`` changes one of them per run, each to half
and to twice its value; the number of challengers stays at most half the ensemble, as in
the protocol (10 of 20), so ``E = 10`` comes with ``C = 5`` and ``C = 20`` with
``E = 40``. ``t0`` stays counted in windows when ``h`` changes. These are runnable
placeholders, not a recommended design.

``run`` drives the existing commands, one out dir per variant under ``--out-dir``, with
the arms and seeds asked for. ``compare`` reads them beside the base run (``--base-dir``,
the run made with the protocol's constants) and writes, per stream group, the mean tail
ARI of every arm, the gain of online tuning over the defaults, the capture ratio against
the static reference and Kendall's tau between the rankings of two arms.

The reference needs a sweep, which no variant pays for by default. A variant that keeps
``h`` and ``t0`` is read against the base run's reference: the reference is one fixed
configuration that never meets ``E`` or ``C``, and its reruns read the same windows. A
variant that changes ``h`` or ``t0`` changes the windows the reference is read on, so its
capture ratio is left empty unless ``--own-reference`` sweeps one in its out dir.

The full run broke ties between members by runtime; a run made since breaks them by the
number of micro-clusters, and changes about 15% of the online windows (README,
"Tie-break"). Variants are run with the rule the base run recorded, so the comparison
moves one constant at a time.

Run from the repo root::

    python -m conf_pipeline.variants run --out-dir conf_pipeline/out/x6 --base-dir conf_pipeline/out/full --workers 8
    python -m conf_pipeline.variants compare --out-dir conf_pipeline/out/x6 --base-dir conf_pipeline/out/full

**Can the ensemble be smaller?** ``ec-run`` runs the online tuner (``confstream``) under every
setting of ``EC_GRID`` -- the protocol's 20 members and 10 challengers first, then smaller ones --
each into ``<out-dir>/<setting>/``, gate-free, so its selections do not depend on measured time and
the settings are compared on the same footing. The base setting is run again there, rather than
read from a run made with the runtime gate. ``ec-compare`` pairs every setting with the base on the
same (algo, fold, seed) and writes, per setting and stream group, the paired change in tail ARI, the
share of pairs that lose more than a band, the speed-up in wall time and in model time, the samples
per second of the whole tuner, and a verdict under thresholds that are flags (placeholders)::

    python -m conf_pipeline.variants ec-run --out-dir conf_pipeline/out/ec_variants --workers 8
    python -m conf_pipeline.variants ec-compare --out-dir conf_pipeline/out/ec_variants
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from . import report

#: name -> the constants it changes (run.py's job keys). Placeholders: halved and doubled.
VARIANTS: dict[str, dict] = {
    "h500": {"h": 500},
    "h2000": {"h": 2000},
    "e10": {"e_size": 10, "e_test": 5},
    "e40": {"e_size": 40},
    "c5": {"e_test": 5},
    "c20": {"e_size": 40, "e_test": 20},
    "t10": {"freeze_after_window": 10},
    "t40": {"freeze_after_window": 40},
}
#: The protocol's value of every constant a variant may change.
BASE = {"h": C.H, "e_size": C.E_SIZE, "e_test": C.E_TEST, "freeze_after_window": C.FREEZE_AFTER_WINDOW}
#: The six cheaper algorithms (CluStream and RepStream cost 85% of the online budget).
VARIANT_ALGOS: tuple[str, ...] = tuple(a for a in C.ALGOS if a not in ("CluStream", "RepStream"))
VARIANT_ARMS: tuple[str, ...] = C.ARMS
#: The flag of each constant on run.py's command line.
FLAGS = {"h": "--h", "e_size": "--e-size", "e_test": "--e-test", "freeze_after_window": "--freeze-after-window"}
#: The rankings compared: does the ranking under the defaults differ from online tuning's, and
#: does online tuning's differ from tuning once and freezing.
TAU_PAIRS: tuple[tuple[str, str], ...] = (("default", "confstream"), ("confstream", "frozen_prefix"))
COMPARE_COLS = [
    "setting", "changed", "group", "n_folds", "n_pairs", "default", "confstream", "frozen_prefix",
    "online_minus_default", "capture_confstream", "capture_frozen_prefix",
    "tau_default_confstream", "tau_confstream_frozen_prefix", "reference",
]


def constants(name: str) -> dict:
    """The four constants of variant ``name`` (``"base"``: the protocol's)."""
    if name == "base":
        return dict(BASE)
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; variants: {sorted(VARIANTS)}")
    unknown = set(VARIANTS[name]) - set(BASE)
    if unknown:
        raise KeyError(f"variant {name!r} changes {sorted(unknown)}, not one of {sorted(BASE)}")
    return {**BASE, **VARIANTS[name]}


def changed(name: str) -> str:
    """``"h=500"``, ``"e_size=10, e_test=5"``...: what the variant changes."""
    if name == "base":
        return ""
    return ", ".join(f"{k}={v}" for k, v in VARIANTS[name].items())


def keeps_reference(name: str) -> bool:
    """Whether the base run's reference reads on the variant's windows: same ``h`` and ``t0``."""
    c = constants(name)
    return c["h"] == BASE["h"] and c["freeze_after_window"] == BASE["freeze_after_window"]


def base_tie_break(base_dir: Path) -> str:
    """The tie-break rule the base run's online rows recorded; a row without the field
    predates it and broke ties by runtime."""
    runs = report.current_runs(Path(base_dir))
    if runs.empty:
        return "micro_clusters"
    online = runs[runs["arm"].isin(["confstream", "frozen_prefix"])]
    rules = online["tie_break"].fillna("runtime") if "tie_break" in online else pd.Series(["runtime"] * len(online))
    return str(rules.mode().iloc[0]) if len(rules) else "micro_clusters"


# --- running ----------------------------------------------------------------------------------

def commands(name: str, out_dir: Path, algos, folds, seeds, arms, workers: int, tie_break: str,
             n_samples: int | None = C.N_SAMPLES_CAP, log_members: bool = False,
             own_reference: bool = False, oracle_seeds=(2, 3, 4), sweep_n_points: int = C.SWEEP_N_POINTS,
             ) -> list[list[str]]:
    """The command lines that run variant ``name`` into ``out_dir / name``: the arms, then,
    with ``own_reference``, the sweep (with the arms' incumbents) and the reference's reruns."""
    c = constants(name)
    vdir = Path(out_dir) / name
    shape = ["--algos", ",".join(algos), "--folds", ",".join(folds), "--n-samples", str(n_samples or 0),
             "--h", str(c["h"]), "--freeze-after-window", str(c["freeze_after_window"]),
             "--out-dir", str(vdir), "--workers", str(workers)]
    tuner = ["--e-size", str(c["e_size"]), "--e-test", str(c["e_test"]), "--tie-break", tie_break]
    py = [sys.executable, "-m"]
    # The base run (out/full) seeded tuner and models with one job seed: the variants do the same.
    shared = ["--seed-policy", "shared"]
    cmds = [[*py, "conf_pipeline.run", *shape, *tuner, *shared, "--seeds", ",".join(map(str, seeds)),
             "--arms", ",".join(arms), *(["--log-members"] if log_members else [])]]
    if own_reference:
        cmds.append([*py, "conf_pipeline.sweep", *shape, *shared, "--n-points", str(sweep_n_points), "--incumbents"])
        cmds.append([*py, "conf_pipeline.run", *shape, *tuner, *shared, "--seeds", ",".join(map(str, oracle_seeds)),
                     "--arms", "oracle_static"])
    return cmds


def run_variants(names, out_dir: Path, dry_run: bool = False, **kw) -> None:
    for name in names:
        for cmd in commands(name, out_dir, **kw):
            print("$", " ".join(cmd[1:]), flush=True)
            if dry_run:
                continue
            proc = subprocess.run(cmd, cwd=str(C.REPO_ROOT), env=C.child_env())
            if proc.returncode != 0:
                raise SystemExit(f"variant {name}: {cmd[2]} exited with {proc.returncode}")


# --- comparing ----------------------------------------------------------------------------------

def arm_means(out_dir: Path, seeds=None, algos=None, folds=None) -> pd.DataFrame:
    """Per (algo, fold, arm) of the protocol arms: the mean tail ARI over the ok runs on
    ``seeds`` (all when None), and whether every seed asked for is there (``all_seeds_ok``)."""
    cols = ["algo", "fold", "arm", report.TAIL, "n_seeds", "all_seeds_ok", "n_runs"]
    runs = report.current_runs(Path(out_dir))
    if runs.empty:
        return pd.DataFrame(columns=cols)
    runs = runs[runs["arm"].isin(list(VARIANT_ARMS)) & ~runs["algo"].isin(list(C.IMPL_VARIANTS))]
    if report.is_split(runs):
        # A fixed arm's runs are not tuner seeds: its mean stands beside every tuner seed (report.replicate_view).
        ok = runs[runs["status"] == "ok"]
        values = ok[["algo", "fold", "arm", "seed"]].assign(value=pd.to_numeric(ok[report.TAIL], errors="coerce"))
        view = report.replicate_view(values.dropna(subset=["value"]), split=True)
        if seeds is not None:
            view = view[view["seed"].isin(list(seeds))]
        if algos is not None:
            view = view[view["algo"].isin(list(algos))]
        if folds is not None:
            view = view[view["fold"].isin(list(folds))]
        g = view.groupby(["algo", "fold", "arm"], sort=True)
        out = g.agg(**{report.TAIL: ("value", "mean"), "n_seeds": ("seed", "nunique"), "n_runs": ("n_runs", "max")})
        out = out.reset_index()
        want = len(set(seeds)) if seeds is not None else (int(view.groupby("arm")["seed"].nunique().max())
                                                         if len(view) else 0)
        out["all_seeds_ok"] = out["n_seeds"] >= want
        return out.reindex(columns=cols)
    if seeds is not None:
        runs = runs[runs["seed"].isin(list(seeds))]
    if algos is not None:
        runs = runs[runs["algo"].isin(list(algos))]
    if folds is not None:
        runs = runs[runs["fold"].isin(list(folds))]
    ok = runs[runs["status"] == "ok"].copy()
    ok[report.TAIL] = pd.to_numeric(ok[report.TAIL], errors="coerce") if report.TAIL in ok else np.nan
    g = ok.dropna(subset=[report.TAIL]).groupby(["algo", "fold", "arm"], sort=True)[report.TAIL]
    out = g.agg(["mean", "count"]).reset_index().rename(columns={"mean": report.TAIL, "count": "n_seeds"})
    want = len(set(seeds)) if seeds is not None else (runs.groupby("arm")["seed"].nunique().max() if len(runs) else 0)
    out["all_seeds_ok"] = out["n_seeds"] >= want
    out["n_runs"] = out["n_seeds"]
    return out.reindex(columns=cols)


def reference(ref_dir: Path) -> pd.DataFrame:
    """Per (algo, fold) of ``ref_dir``: the static reference's tail reading and whether the
    pair is tunable (``report.headroom_table``'s gate)."""
    head = report.headroom_table(Path(ref_dir))
    if head.empty:
        return pd.DataFrame(columns=["algo", "fold", "oracle_static", "tunable"])
    ceil = report.oracle_table(Path(ref_dir))[["algo", "fold", "ceiling"]]
    out = head[["algo", "fold", "tunable"]].merge(ceil, on=["algo", "fold"], how="left")
    return out.rename(columns={"ceiling": "oracle_static"})


def _has_reference(d: Path) -> bool:
    if not (Path(d) / C.ORACLE_JSON.name).exists():
        return False
    runs = report.current_runs(Path(d))
    return not runs.empty and bool(((runs["arm"] == "oracle_static") & (runs["status"] == "ok")).any())


def setting_rows(setting: str, means: pd.DataFrame, ref: pd.DataFrame | None, ref_name: str | None) -> list[dict]:
    """The comparison rows of one setting, one per stream group and pooled."""
    from . import analyze
    from .robustness import ALL, capture_by_group

    wide = means.pivot_table(index=["algo", "fold"], columns="arm", values=report.TAIL, aggfunc="mean")
    wide = wide.reindex(columns=list(VARIANT_ARMS)).reset_index()
    wide.columns.name = None
    if ref is not None and not ref.empty:
        wide = wide.merge(ref, on=["algo", "fold"], how="left")
        wide["tunable"] = wide["tunable"].fillna(False).astype(bool)
    else:
        wide["oracle_static"], wide["tunable"] = np.nan, False
    capture = capture_by_group(wide, arms=("confstream", "frozen_prefix")).set_index(["group", "arm"])
    wide["group"] = [C.stream_group(f) for f in wide["fold"]]
    rows = []
    for group in (*C.STREAM_GROUPS, ALL):
        sub = wide if group == ALL else wide[wide["group"] == group]
        both = sub[["default", "confstream"]].dropna()
        folds = set(sub["fold"])
        summ = means[means["fold"].isin(folds)]
        taus = analyze.kendall_matrix(summ, list(VARIANT_ARMS)) if len(summ) else None
        row = {"setting": setting, "changed": changed(setting), "group": group,
               "n_folds": len(folds), "n_pairs": int(len(both)),
               **{a: float(sub[a].mean()) if sub[a].notna().any() else np.nan for a in VARIANT_ARMS},
               "online_minus_default": float((both["confstream"] - both["default"]).mean()) if len(both) else np.nan,
               "reference": ref_name}
        for arm in ("confstream", "frozen_prefix"):
            row[f"capture_{arm}"] = (float(capture.loc[(group, arm), "capture"]) if ref_name is not None
                                     else np.nan)
        for a, b in TAU_PAIRS:
            row[f"tau_{a}_{b}"] = float(taus.loc[a, b]) if taus is not None else np.nan
        rows.append(row)
    return rows


def compare(out_dir: Path, base_dir: Path, names=None, seeds=None, algos=VARIANT_ALGOS, folds=None) -> pd.DataFrame:
    """The comparison table: the base run, then every variant with an out dir, read on the
    same ``seeds``, ``algos`` and ``folds`` (all when None)."""
    names = list(VARIANTS) if names is None else list(names)
    base_ref = reference(base_dir) if _has_reference(base_dir) else None
    rows = setting_rows("base", arm_means(base_dir, seeds, algos, folds), base_ref,
                        "base" if base_ref is not None else None)
    for name in names:
        vdir = Path(out_dir) / name
        if not (vdir / "runs.jsonl").exists():
            continue
        if _has_reference(vdir):
            ref, ref_name = reference(vdir), "own"
        elif keeps_reference(name) and base_ref is not None:
            ref, ref_name = base_ref, "base"
        else:
            ref, ref_name = None, None
        rows += setting_rows(name, arm_means(vdir, seeds, algos, folds), ref, ref_name)
    return pd.DataFrame(rows, columns=COMPARE_COLS)


def write_compare(out_dir: Path, base_dir: Path, **kw) -> list[Path]:
    from . import analyze

    table = compare(out_dir, base_dir, **kw)
    rep = Path(out_dir) / C.REPORT_DIR.name
    rep.mkdir(parents=True, exist_ok=True)
    csv = rep / "variants.csv"
    table.to_csv(csv, index=False)
    lines = [
        "# Protocol constants varied one at a time", "",
        f"Base run: `{base_dir}`. Every setting is read on the same seeds, algorithms and folds; "
        "`default`, `confstream` and `frozen_prefix` are mean tail ARIs over the (algo, fold) pairs of the "
        "group, `online_minus_default` the mean paired gain, `capture_*` the share of the reference's gain "
        "over the defaults (tunable pairs, sums before dividing), `tau_*` Kendall's tau between two arms' "
        "rankings, mean over the folds. `reference` says whose reference the capture is read against: "
        "`base` (the variant keeps h and t0), `own` (swept in the variant's out dir) or none.", "",
        analyze._md_table(table, 3) if len(table) else "_(no rows)_", "",
    ]
    md = rep / "variants.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return [csv, md]


# --- E/C grid -----------------------------------------------------------------------------------

#: name -> (ensemble size, challengers per window); the protocol's first. Placeholders: the protocol
#: shrunk at its ratio, and its challengers cut at a full ensemble.
EC_GRID: dict[str, dict] = {
    "ec20_10": {"e_size": 20, "e_test": 10},
    "ec15_7": {"e_size": 15, "e_test": 7},
    "ec10_5": {"e_size": 10, "e_test": 5},
    "ec5_2": {"e_size": 5, "e_test": 2},
    "ec20_5": {"e_size": 20, "e_test": 5},
    "ec20_2": {"e_size": 20, "e_test": 2},
}
EC_BASE = "ec20_10"
EC_ARMS: tuple[str, ...] = ("confstream",)
#: ``--oracles``: the diagnostic tuners selecting by labels (E4) run beside the online arm in every
#: setting, to see what a smaller ensemble costs a tuner with a perfect signal.
EC_ORACLES: dict[str, tuple[str, ...]] = {
    "none": (),
    "unconstrained": ("confstream_ari_oracle",),
    "constrained": ("confstream_ari_oracle_constrained",),
    "both": C.DIAGNOSTIC_ARMS,
}


def ec_arms(oracles: str = "none") -> tuple[str, ...]:
    """The arms every E/C setting runs: the online tuner, and the oracles ``oracles`` names."""
    return EC_ARMS + EC_ORACLES[oracles]
#: A gate-free base job of the slow algorithms can outlast the protocol's 6 h.
EC_TIMEOUT_S = 24 * 3600
#: Verdict thresholds (placeholders, all flags): the study's smallest effect of interest per pair, a
#: mean loss of about the online tuner's seed noise, the share of pairs allowed beyond the band, and
#: the slowest whole-tuner rate the confStream paper reports (CluStream on powersupply, 15 samples/s).
EC_BAND = C.MIN_CEILING_GAIN
EC_MAX_MEAN_LOSS = 0.01
EC_MAX_SHARE_OVER_BAND = 0.10
EC_MIN_SPS = 15.0
EC_RUN_COLS = ["setting", "e_size", "e_test", "arm", "algo", "fold", "seed", "status", "tail", "wall_s", "model_s",
               "n_samples", "sps", "incumbent_sps"]
EC_PAIR_COLS = ["setting", "arm", "algo", "fold", "seed", "group", "status", "tail", "base_tail", "d_tail",
                "speedup_wall", "speedup_model", "sps", "base_sps"]
EC_SUMMARY_COLS = ["setting", "e_size", "e_test", "arm", "group", "n_pairs", "n_unfinished", "mean_tail", "mean_base_tail",
                   "mean_d_tail", "min_d_tail", "share_loss_over_band", "tau_vs_base", "median_speedup_wall",
                   "median_speedup_model", "median_sps", "share_sps_at_least_min", "keeps_quality", "faster",
                   "verdict"]


def ec_command(name: str, out_dir: Path, algos, folds, tuner_seeds, workers: int, timeout: int = EC_TIMEOUT_S,
               n_samples: int | None = C.N_SAMPLES_CAP, arms=EC_ARMS, log_members: bool = False,
               h: int = C.H, freeze_after_window: int = C.FREEZE_AFTER_WINDOW,
               model_seed: int = C.MODEL_SEED) -> list[str]:
    """The ``run.py`` command of E/C setting ``name``, into ``out_dir / name``, under the split seed policy:
    one run per tuner seed, the models on ``model_seed`` in every setting, so the settings differ only in
    the ensemble's budget. ``h`` and ``freeze_after_window`` (where the tail starts) are the protocol's
    unless given."""
    c = EC_GRID[name]
    return [sys.executable, "-m", "conf_pipeline.run", "--algos", ",".join(algos), "--folds", ",".join(folds),
            "--seed-policy", "split", "--tuner-seeds", ",".join(map(str, tuner_seeds)),
            "--model-seed", str(model_seed),
            "--arms", ",".join(arms), "--n-samples", str(n_samples or 0),
            "--h", str(h), "--freeze-after-window", str(freeze_after_window),
            "--e-size", str(c["e_size"]), "--e-test", str(c["e_test"]), "--no-runtime-gate",
            "--timeout", str(timeout), "--workers", str(workers), "--out-dir", str(Path(out_dir) / name),
            *(["--log-members"] if log_members else [])]


def run_ec(names, out_dir: Path, dry_run: bool = False, **kw) -> None:
    """Run the settings ``names`` one after the other (each its own out dir, resumable)."""
    for name in names:
        cmd = ec_command(name, out_dir, **kw)
        print("$", " ".join(cmd[1:]), flush=True)
        if dry_run:
            continue
        proc = subprocess.run(cmd, cwd=str(C.REPO_ROOT), env=C.child_env())
        if proc.returncode != 0:
            raise SystemExit(f"setting {name}: conf_pipeline.run exited with {proc.returncode}")


def ec_runs(out_dir: Path, names=None) -> pd.DataFrame:
    """One row per (setting, arm, algo, fold, seed) of the online arm and of the oracle arms run beside it
    (read on purpose: the default readers leave the diagnostic arms out): status, tail ARI and costs."""
    rows = []
    for name in (list(EC_GRID) if names is None else list(names)):
        d = Path(out_dir) / name
        if not (d / "runs.jsonl").exists():
            continue
        runs = report.current_runs(d, diagnostic=True)
        if runs.empty:
            continue
        runs = runs[runs["arm"].isin(list(EC_ARMS) + list(C.DIAGNOSTIC_ARMS))]
        for r in runs.to_dict("records"):
            ok = r.get("status") == "ok"
            wall = r.get("wall_s") if ok else None
            n = r.get("n_samples") if ok else None
            rows.append({"setting": name, **EC_GRID[name], "arm": r["arm"], "algo": r["algo"], "fold": r["fold"],
                         "seed": r["seed"],
                         "status": r.get("status"), "tail": report._num(r.get(report.TAIL)) if ok else None,
                         "wall_s": wall, "model_s": r.get("ensemble_runtime_s") if ok else None, "n_samples": n,
                         "sps": (n / wall) if ok and wall else None,
                         "incumbent_sps": r.get("incumbent_throughput_sps") if ok else None})
    return pd.DataFrame(rows, columns=EC_RUN_COLS)


def _ratio(a, b):
    a, b = report._num(a), report._num(b)
    return a / b if a is not None and b else None


def ec_pairs(runs: pd.DataFrame, base: str = EC_BASE) -> pd.DataFrame:
    """Every setting but ``base`` paired with ``base`` on the same (arm, algo, fold, seed): the change in tail ARI
    and the speed-up (base time over the setting's). A pair whose setting did not finish keeps its status and
    reads no change; one whose base did not finish reads ``base: <status>``."""
    if runs.empty:
        return pd.DataFrame(columns=EC_PAIR_COLS)
    key = ["arm", "algo", "fold", "seed"]
    b = runs[runs["setting"] == base].set_index(key)
    rows = []
    for r in runs[runs["setting"] != base].to_dict("records"):
        k = (r["arm"], r["algo"], r["fold"], r["seed"])
        br = b.loc[k].to_dict() if k in b.index else None
        if br is None or br["status"] != "ok":
            status = f"base: {br['status'] if br else 'missing'}"
        else:
            status = r["status"]
        done = status == "ok" and br is not None
        try:
            group = C.stream_group(r["fold"])
        except ValueError:
            group = "unassigned"
        rows.append({"setting": r["setting"], "arm": r["arm"], "algo": r["algo"], "fold": r["fold"], "seed": r["seed"],
                     "group": group,
                     "status": status, "tail": r["tail"], "base_tail": br["tail"] if br else None,
                     "d_tail": (r["tail"] - br["tail"]) if done and r["tail"] is not None and br["tail"] is not None
                     else None,
                     "speedup_wall": _ratio(br["wall_s"], r["wall_s"]) if done else None,
                     "speedup_model": _ratio(br["model_s"], r["model_s"]) if done else None,
                     "sps": r["sps"], "base_sps": br["sps"] if br else None})
    return pd.DataFrame(rows, columns=EC_PAIR_COLS)


def _tau_vs_base(sub: pd.DataFrame) -> float:
    """Kendall's tau between the setting's and the base's ranking of the algorithms, mean over the folds
    with at least three algorithms read on both."""
    from .robustness import _tau

    taus = []
    for _, g in sub.dropna(subset=["tail", "base_tail"]).groupby("fold"):
        if len(g) >= 3:
            t = _tau(g["tail"].to_numpy(float), g["base_tail"].to_numpy(float))
            if t == t:
                taus.append(t)
    return float(np.mean(taus)) if taus else np.nan


def ec_summary(pairs: pd.DataFrame, band: float = EC_BAND, max_mean_loss: float = EC_MAX_MEAN_LOSS,
               max_share_over_band: float = EC_MAX_SHARE_OVER_BAND, min_sps: float = EC_MIN_SPS) -> pd.DataFrame:
    """Per setting and stream group (and ``all``): what the setting keeps and what it saves.

    ``keeps_quality``: the mean paired change is no worse than ``-max_mean_loss`` and at most
    ``max_share_over_band`` of the pairs lose more than ``band``; ``faster``: the median wall-time
    speed-up is above 1. ``verdict``: ``reducible`` (both), ``quality drops``, ``no speed-up``, or
    ``undecided`` when no pair finished on both sides. Unfinished pairs are counted, never read as a
    change. The thresholds are placeholders, every one a flag.
    """
    rows = []
    if pairs.empty:
        return pd.DataFrame(columns=EC_SUMMARY_COLS)
    for (setting, arm), by_setting in pairs.groupby(["setting", "arm"], sort=False):
        groups = [g for g in (*C.STREAM_GROUPS, "unassigned") if g in set(by_setting["group"])]
        for group in (*groups, "all"):
            sub = by_setting if group == "all" else by_setting[by_setting["group"] == group]
            done = sub[sub["d_tail"].notna()]
            d = done["d_tail"].astype(float)
            sps = pd.to_numeric(sub["sps"], errors="coerce").dropna()
            keeps = bool(len(done)) and d.mean() >= -max_mean_loss and (d < -band).mean() <= max_share_over_band
            faster = bool(len(done)) and float(done["speedup_wall"].median()) > 1.0
            verdict = ("undecided" if not len(done) else "reducible" if keeps and faster
                       else "quality drops" if not keeps else "no speed-up")
            c = EC_GRID.get(setting, {})
            rows.append({
                "setting": setting, "e_size": c.get("e_size"), "e_test": c.get("e_test"), "arm": arm, "group": group,
                "n_pairs": int(len(done)), "n_unfinished": int(len(sub) - len(done)),
                "mean_tail": float(done["tail"].mean()) if len(done) else np.nan,
                "mean_base_tail": float(done["base_tail"].mean()) if len(done) else np.nan,
                "mean_d_tail": float(d.mean()) if len(done) else np.nan,
                "min_d_tail": float(d.min()) if len(done) else np.nan,
                "share_loss_over_band": float((d < -band).mean()) if len(done) else np.nan,
                "tau_vs_base": _tau_vs_base(done),
                "median_speedup_wall": float(done["speedup_wall"].median()) if len(done) else np.nan,
                "median_speedup_model": float(done["speedup_model"].median()) if len(done) else np.nan,
                "median_sps": float(sps.median()) if len(sps) else np.nan,
                "share_sps_at_least_min": float((sps >= min_sps).mean()) if len(sps) else np.nan,
                "keeps_quality": keeps, "faster": faster, "verdict": verdict,
            })
    return pd.DataFrame(rows, columns=EC_SUMMARY_COLS)


def write_ec(out_dir: Path, names=None, base: str = EC_BASE, **thresholds) -> list[Path]:
    from . import analyze

    runs = ec_runs(out_dir, names)
    pairs = ec_pairs(runs, base)
    summary = ec_summary(pairs, **thresholds)
    rep = Path(out_dir) / C.REPORT_DIR.name / "ec"
    rep.mkdir(parents=True, exist_ok=True)
    paths = [rep / "runs.csv", rep / "pairs.csv", rep / "summary.csv", rep / "ec.md"]
    runs.to_csv(paths[0], index=False)
    pairs.to_csv(paths[1], index=False)
    summary.to_csv(paths[2], index=False)
    t = {"band": EC_BAND, "max_mean_loss": EC_MAX_MEAN_LOSS, "max_share_over_band": EC_MAX_SHARE_OVER_BAND,
         "min_sps": EC_MIN_SPS, **thresholds}
    status = (runs.groupby(["setting", "arm", "status"]).size().unstack(fill_value=0).reset_index()
              if len(runs) else runs)
    lines = [
        "# Can the online tuner run with fewer members and challengers?", "",
        f"Read from `{out_dir}`: the online arm, and the oracle arms when they ran, under every E/C setting found "
        f"there, each paired with `{base}` of the same arm on the same (algo, fold, seed), all gate-free. `d_tail` is the setting's tail ARI minus the base's; "
        "`speedup_*` the base's time over the setting's (above 1 is faster), in wall time and in the members' "
        "model time; `sps` samples per second of the whole tuner. A pair that did not finish on either side is "
        "counted (`n_unfinished`) and never read as a change. "
        f"Verdict thresholds (placeholders): a setting keeps the quality when its mean change is at least "
        f"-{t['max_mean_loss']:g} and at most {t['max_share_over_band']:.0%} of its pairs lose more than "
        f"{t['band']:g}; the paper's slowest whole-tuner rate was {t['min_sps']:g} samples/s. One seed: changes "
        "smaller than the online tuner's seed noise (about 0.005-0.01 on the full run) are not read.", "",
        "## Jobs by status", "", analyze._md_table(status) if len(status) else "_(no runs yet)_", "",
        "## Verdict per setting and stream group", "",
        analyze._md_table(summary, 3) if len(summary) else "_(nothing to compare yet)_", "",
    ]
    paths[3].write_text("\n".join(lines), encoding="utf-8")
    return paths


# --- CLI ----------------------------------------------------------------------------------------

def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _names(value: str) -> list[str]:
    names = list(VARIANTS) if value == "all" else _csv(value)
    for n in names:
        constants(n)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("run", "compare", "ec-run", "ec-compare"))
    parser.add_argument("--out-dir", required=True, help="one sub-directory per variant goes here")
    parser.add_argument("--base-dir", default=None,
                        help="the run made with the protocol's constants (run, compare; the E/C actions "
                             f"read their base, {EC_BASE}, from --out-dir)")
    parser.add_argument("--grid", default="all", help=f"E/C settings, comma-separated from {list(EC_GRID)}, or all")
    parser.add_argument("--timeout", type=int, default=EC_TIMEOUT_S, help="ec-run: seconds per job")
    parser.add_argument("--oracles", default="none", choices=list(EC_ORACLES),
                        help="ec-run: the oracle tuners (selecting by labels) to run beside the online arm")
    parser.add_argument("--tuner-seeds", default=str(C.TUNER_SEEDS[0]), help="ec-run: the tuner seeds (split policy)")
    parser.add_argument("--model-seed", type=int, default=C.MODEL_SEED, help="ec-run: the models' seed (split policy)")
    parser.add_argument("--h", type=int, default=C.H, help="ec-run: the window")
    parser.add_argument("--freeze-after-window", type=int, default=C.FREEZE_AFTER_WINDOW,
                        help="ec-run: where every run's tail starts")
    parser.add_argument("--band", type=float, default=EC_BAND)
    parser.add_argument("--max-mean-loss", type=float, default=EC_MAX_MEAN_LOSS)
    parser.add_argument("--max-share-over-band", type=float, default=EC_MAX_SHARE_OVER_BAND)
    parser.add_argument("--min-sps", type=float, default=EC_MIN_SPS)
    parser.add_argument("--variants", default="all", help=f"comma-separated, from {list(VARIANTS)}, or all")
    parser.add_argument("--algos", default=",".join(VARIANT_ALGOS))
    parser.add_argument("--folds", default=",".join(C.FOLDS))
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--arms", default=",".join(VARIANT_ARMS))
    parser.add_argument("--n-samples", type=int, default=C.N_SAMPLES_CAP, help="0 = whole stream")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tie-break", default=None,
                        help="default: the rule the base run's online rows recorded")
    parser.add_argument("--log-members", action="store_true")
    parser.add_argument("--own-reference", action="store_true",
                        help="also sweep a reference in every variant's out dir and rerun it")
    parser.add_argument("--oracle-seeds", default="2,3,4")
    parser.add_argument("--sweep-n-points", type=int, default=C.SWEEP_N_POINTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    if args.action in ("ec-run", "ec-compare"):
        grid = list(EC_GRID) if args.grid == "all" else _csv(args.grid)
        unknown = [g for g in grid if g not in EC_GRID]
        if unknown:
            raise SystemExit(f"unknown E/C settings {unknown}; known: {list(EC_GRID)}")
        if args.action == "ec-run":
            run_ec(grid, out, dry_run=args.dry_run, algos=_csv(args.algos), folds=_csv(args.folds),
                   tuner_seeds=[int(s) for s in _csv(args.tuner_seeds)], model_seed=args.model_seed,
                   arms=ec_arms(args.oracles),
                   workers=args.workers, timeout=args.timeout,
                   n_samples=args.n_samples or None, log_members=args.log_members, h=args.h,
                   freeze_after_window=args.freeze_after_window)
            return
        for p in write_ec(out, grid, band=args.band, max_mean_loss=args.max_mean_loss,
                          max_share_over_band=args.max_share_over_band, min_sps=args.min_sps):
            print(p)
        return
    if args.base_dir is None:
        raise SystemExit("--base-dir is needed for run and compare")
    names = _names(args.variants)
    algos, folds, seeds = _csv(args.algos), _csv(args.folds), [int(s) for s in _csv(args.seeds)]
    base = Path(args.base_dir)
    if args.action == "run":
        tie_break = args.tie_break or base_tie_break(base)
        print(f"tie-break: {tie_break}")
        run_variants(names, out, dry_run=args.dry_run, algos=algos, folds=folds, seeds=seeds,
                     arms=_csv(args.arms), workers=args.workers, tie_break=tie_break,
                     n_samples=args.n_samples or None, log_members=args.log_members,
                     own_reference=args.own_reference, oracle_seeds=[int(s) for s in _csv(args.oracle_seeds)],
                     sweep_n_points=args.sweep_n_points)
        return
    for p in write_compare(out, base, names=names, seeds=seeds, algos=algos, folds=folds):
        print(p)


if __name__ == "__main__":
    main()
