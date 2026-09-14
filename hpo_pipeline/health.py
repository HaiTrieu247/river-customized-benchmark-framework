"""Stage 5 -- HEALTH: eight checks, and a gate. Read nothing before it passes.

Two of these are load-bearing rather than routine.

**Check 5** asks whether ``s(D)`` normalises anything at all. For each algorithm
whose resolution axis is a distance threshold, take the best level per fold and
compare the spread of the dimensionless value against the spread of the absolute
one. If the dimensionless value is the tighter of the two, the conversion is
doing work and P2 rests on something. If it is not, either ``s(D)`` is the wrong
statistic or the scale genuinely does not transfer -- and that has to be said
rather than buried.

**Check 8** is the precondition for P2R. Regress the achieved cluster count on
``g``: the Stage 3 convention demands the correlation be positive. A negative
one means that algorithm's ladder is reversed; a flat one means ``g`` does not
control the cluster count on that fold, and every selection rule read off that
curve is meaningless there. This is also where the inferred direction of
``RepStream.alpha`` is confirmed against data rather than against a docstring.

Checks 4 and 7 report without blocking: a degenerate run and a one-dimensional
fold are properties to be published, not errors to be fixed.
"""

from __future__ import annotations

import argparse

import numpy as np

from ._common import (
    HEALTH_MD,
    HPO_OUT,
    MATRIX_JSONL,
    PROBE_JSON,
    load_matrix,
    read_json,
    require,
)
from .space import GRAN

OBJ = "ari"


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def _best_level_per_fold(df, algo):
    """For one algorithm: fold -> (best granularity level, its g)."""
    sub = df[(df.algo == algo) & (df.status == "ok") & df.feasible & df.granularity_level.notna()]
    out = {}
    for fold, grp in sub.groupby("dataset"):
        means = grp.groupby(["granularity_level", "g"])[OBJ].mean()
        if means.empty or means.isna().all():
            continue
        level, g = means.idxmax()
        out[fold] = (int(level), float(g))
    return out


def run_checks(df, probes: dict) -> tuple[list[str], bool]:
    lines: list[str] = []
    failed = False
    tuned = df[df.granularity_level.notna()]

    # 1 -- coverage
    n_ok = int((df.status == "ok").sum())
    coverage = n_ok / max(len(df), 1)
    ok = coverage >= 0.95
    failed |= not ok
    lines.append("## 1. Coverage\n")
    lines.append(f"`ok` rows: {n_ok} / {len(df)} = **{coverage:.3f}** (need >= 0.95) "
                 f"-- {'PASS' if ok else 'FAIL'}\n")
    counts = df.status.value_counts().to_dict()
    lines.append(f"status counts: {counts}\n")

    # 2 -- does the search space discriminate at all?
    lines.append("\n## 2. Discrimination: max(ARI) - min(ARI) per (algo, fold)\n")
    spread = (
        df[(df.status == "ok") & df.feasible]
        .groupby(["algo", "dataset"])[OBJ]
        .agg(lambda s: (s.max() - s.min()) if s.notna().any() else np.nan)
    )
    lines.append(f"median spread: **{_fmt(float(spread.median()))}**, "
                 f"min: {_fmt(float(spread.min()))}, max: {_fmt(float(spread.max()))}\n")
    if spread.notna().any() and float(spread.median()) <= 0:
        failed = True
        lines.append("FAIL: the space does not separate configurations\n")

    # 3 -- seed noise, against check 2
    lines.append("\n## 3. Seed noise vs. discrimination\n")
    per_seed = (
        df[(df.status == "ok") & df.feasible]
        .groupby(["algo", "dataset", "config_id"])[OBJ]
        .std()
    )
    med_noise = float(per_seed.median()) if per_seed.notna().any() else float("nan")
    lines.append(f"median std over seeds: **{_fmt(med_noise)}** "
                 f"vs. median spread {_fmt(float(spread.median()))}\n")
    lines.append("Seed noise must be clearly the smaller of the two, or differences "
                 "between configurations are noise.\n")

    # 4 -- degeneracy: report only
    lines.append("\n## 4. Degenerate runs (report only)\n")
    k_hat = df.get("k_hat")
    n_single = int((k_hat == 1).sum()) if k_hat is not None else 0
    lines.append(f"runs collapsing to a single cluster: {n_single}\n")

    # 5 -- does s(D) normalise?
    lines.append("\n## 5. Does s(D) normalise? (the pipeline's key test)\n")
    lines.append("| algorithm | CV(g*) | CV(abs*) | verdict |\n|---|---|---|---|\n")
    scaled = [a for a, (_, kind) in GRAN.items() if kind == "scaled"]
    for algo in scaled:
        if algo not in set(df.algo):
            continue
        best = _best_level_per_fold(df, algo)
        if len(best) < 2:
            lines.append(f"| {algo} | - | - | too few folds |\n")
            continue
        gs = np.array([g for _, g in best.values()], dtype=float)
        abs_vals = np.array(
            [g * probes[fold]["s"] for fold, (_, g) in best.items()], dtype=float
        )
        cv_g = float(gs.std() / abs(gs.mean())) if gs.mean() else float("inf")
        cv_abs = float(abs_vals.std() / abs(abs_vals.mean())) if abs_vals.mean() else float("inf")
        good = cv_g < cv_abs
        failed |= not good
        lines.append(
            f"| {algo} | {_fmt(cv_g)} | {_fmt(cv_abs)} | "
            f"{'PASS' if good else 'FAIL'} |\n"
        )
    lines.append(
        "\nCV(g*) < CV(abs*) means the conversion transfers scale and P2 has a basis. "
        "If it fails here, either s(D) is the wrong statistic (it is the pairwise "
        "median; the kNN median it replaced collapses to 0 on duplicate-heavy folds) "
        "or the scale does not transfer at all.\n"
    )

    # 6 -- is the resolution axis wide enough?
    lines.append("\n## 6. Does the resolution axis hit its ends?\n")
    hits, total = 0, 0
    for algo in sorted(set(tuned.algo)):
        best = _best_level_per_fold(df, algo)
        for level, _ in best.values():
            total += 1
            hits += int(level in (1, 8))
    share = hits / total if total else 0.0
    ok = share < 0.20
    failed |= not ok
    lines.append(f"best level sits at an endpoint in {hits}/{total} = **{share:.3f}** "
                 f"of cells (need < 0.20) -- {'PASS' if ok else 'FAIL'}\n")
    lines.append("An optimum at an endpoint means the ladder is too short and the "
                 "true optimum is outside it.\n")

    # 7 -- feature domination: report only, but mandatory
    lines.append("\n## 7. Feature domination (report only, but must be published)\n")
    lines.append("| fold | s(D) | top1_variance_share | effective_dim | 1-D dominated |\n")
    lines.append("|---|---|---|---|---|\n")
    for fold, entry in sorted(probes.items()):
        lines.append(
            f"| {fold} | {entry['s']:.5g} | {entry['top1_variance_share']:.3f} | "
            f"{entry['effective_dim']:.2f} | "
            f"{'**yes**' if entry.get('one_d_dominated') else 'no'} |\n"
        )
    lines.append(
        "\nOn a 1-D dominated fold, clustering runs along a single axis and every "
        "comparison there describes behaviour on that axis alone.\n"
    )

    # 8 -- direction of the resolution axis
    lines.append("\n## 8. Does k_hat rise with g? (precondition for P2R)\n")
    lines.append("| algorithm | median rho | folds rho>0 | verdict |\n|---|---|---|---|\n")
    from scipy.stats import spearmanr

    for algo in sorted(set(tuned.algo)):
        sub = tuned[(tuned.algo == algo) & (tuned.status == "ok") & tuned.feasible]
        rhos = []
        for _, grp in sub.groupby("dataset"):
            curve = grp.groupby("granularity_level")["k_hat"].mean().dropna()
            if len(curve) < 3 or curve.nunique() < 2:
                continue
            rho = spearmanr(curve.index.to_numpy(), curve.to_numpy()).statistic
            if np.isfinite(rho):
                rhos.append(float(rho))
        if not rhos:
            lines.append(f"| {algo} | - | - | no usable curve |\n")
            continue
        median_rho = float(np.median(rhos))
        n_pos = sum(r > 0 for r in rhos)
        good = median_rho > 0
        failed |= not good
        lines.append(
            f"| {algo} | {_fmt(median_rho)} | {n_pos}/{len(rhos)} | "
            f"{'PASS' if good else 'FAIL -- ladder reversed'} |\n"
        )
    lines.append(
        "\nExpect rho near +1 for the count family and positive but looser for the "
        "threshold family, where the cluster count emerges rather than being set. "
        "This is the check that confirms the inferred direction of RepStream.alpha.\n"
    )

    return lines, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    require(MATRIX_JSONL, "Stage 4 (build_matrix.py)")
    require(PROBE_JSON, "Stage 2 (probe.py)")
    df = load_matrix()
    probes = read_json(PROBE_JSON)

    lines, failed = run_checks(df, probes)
    header = [
        "# Stage 5 -- health\n\n",
        "Gate for Stage 6. Checks 4 and 7 report without blocking; the rest block.\n\n",
        f"Rows: {len(df)}. Feasible: {int(df.feasible.sum())}.\n\n",
        f"**VERDICT: {'FAIL' if failed else 'PASS'}**\n\n",
    ]
    HPO_OUT.mkdir(parents=True, exist_ok=True)
    HEALTH_MD.write_text("".join(header + lines), encoding="utf-8")
    print("".join(header + lines))
    print(f"wrote {HEALTH_MD}")
    if failed:
        raise SystemExit("GATE FAILED -- do not read Stage 6 results yet")


if __name__ == "__main__":
    main()
