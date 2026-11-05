"""Two runs of one protocol and seed, compared window by window: is a run repeatable, and if
not, where does it first part from the other and why?

``run.py`` pins the hash seed, the threads and the tie-break, yet two processes can still part
ways: the runtime cap sheds a member against the fastest one of the window, and measured time
varies with the machine's load. This module reads two out dirs (``--a``, ``--b``), pairs their
jobs by (algo, fold, seed, arm, config_id) and compares, per window,

- the trajectory (``windows.jsonl``): ``WINDOW_FIELDS`` in that order, the incumbent's and the
  lagged member's predictions included as a hash (``pred_sha``; runs made before 2026-10-23
  have none and compare on the rest);
- the ensemble (member logs, when both runs wrote them): which members were alive, and each
  one's ``MEMBER_FIELDS``;
- the lifecycle (``lifecycle/``, when both runs wrote it): every birth, challenge, decision and
  freeze event, field by field (``LIFECYCLE_IGNORED`` aside).

A job is keyed by ``KEY``, its model seed included, so a fixed configuration run on two model
seeds is two jobs; a deterministic fixed configuration (no seed) reads its member and lifecycle
logs under ``seedx``, as ``run.job_name`` writes them. ``--strict`` exits non-zero unless every job
is on both sides, ok, identical, with evaluation windows and (``--require-members``) member logs:
the check a reproducibility smoke runs.

A job's ``cause`` is read at the first window where the members part: ``runtime gate`` when a
member was shed for its runtime on one side only, ``model or scoring`` when the same member with
the same parameters predicted or scored differently, ``sampling`` when the members themselves
differ (another child drawn), ``tie-break`` when every member read the same and the selection
still differs, ``not logged`` when the trajectories differ and no member log says why. Runtimes
are expected to differ and are only reported (``wall_ratio``). ``*_share_over_factor`` replays
the runtime cap on a side's member logs: the share of members with a runtime ratio above
``factor``, i.e. what a cap would have shed from a run made without it.

Run from the repo root::

    python -m conf_pipeline.repeat --a conf_pipeline/out/rep_a --b conf_pipeline/out/rep_b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from .manifest import canonical

KEY = ("algo", "fold", "seed", "arm", "config_id", "model_seed")
#: Lifecycle fields that are not part of the trajectory (none today: events carry no timing).
LIFECYCLE_IGNORED: tuple[str, ...] = ()
WINDOW_FIELDS = ("incumbent_id", "params", "pred_sha", "score", "feasible_reason",
                 "lag_member_id", "lag_pred_sha", "lag_ari_win")
MEMBER_FIELDS = ("params", "pred_sha", "score", "feasible_reason")
JOB_COLS = [*KEY, "status", "identical", "n_windows", "first_divergence_window", "first_divergence_field",
            "n_windows_differing", "max_abs_lag_ari_diff", "a_tail", "b_tail", "tail_diff", "wall_ratio",
            "members_logged", "member_first_divergence_window", "member_first_divergence_field", "cause",
            "lifecycle_logged", "n_lifecycle_events", "lifecycle_first_divergence",
            "a_share_over_factor", "b_share_over_factor"]
WINDOW_COLS = [*KEY, "window", "fields", "a_incumbent_id", "b_incumbent_id", "a_lag_ari_win", "b_lag_ari_win"]


def _key(row: dict) -> tuple:
    from .report import _cid, _seed_key
    from .run import _model_seed_of

    return (row.get("algo"), row.get("fold"), _seed_key(row.get("seed")), row.get("arm"),
            _cid(row.get("config_id")), _seed_key(_model_seed_of(row)))


def _summaries(out_dir: Path) -> dict:
    """The last summary row of every job (a resumed job's latest attempt)."""
    out = {}
    for r in C.read_jsonl(Path(out_dir) / "runs.jsonl"):
        out[_key(r)] = r
    return out


def _windows(out_dir: Path) -> dict:
    out: dict = {}
    for r in C.read_jsonl(Path(out_dir) / "windows.jsonl"):
        if r.get("kind", "window") == "window":
            out.setdefault(_key(r), {})[int(r["window"])] = r
    return out


def _log_name(key: tuple) -> str:
    """The file stem ``run.job_name`` gives the job ``key`` (``seedx`` for a run without a seed)."""
    from .run import job_name

    algo, fold, seed, arm, cid, _ = key
    return job_name({"algo": algo, "fold": fold, "seed": seed, "arm": arm, "config_id": cid})


def _member_path(out_dir: Path, key: tuple) -> Path:
    return Path(out_dir) / C.MEMBERS_DIR.name / f"{_log_name(key)}.jsonl"


def _lifecycle(out_dir: Path, key: tuple) -> list[dict] | None:
    from .run import LIFECYCLE_DIR

    path = Path(out_dir) / LIFECYCLE_DIR / f"{_log_name(key)}.jsonl"
    if not path.exists():
        return None
    return [{k: v for k, v in r.items() if k not in LIFECYCLE_IGNORED} for r in C.read_jsonl(path)]


def _lifecycle_divergence(la: list[dict], lb: list[dict]) -> str | None:
    """The first lifecycle event the two runs disagree on, as ``"<index> <event> <field>"``."""
    for i in range(max(len(la), len(lb))):
        if i >= len(la) or i >= len(lb):
            return f"{i} missing on {'a' if i >= len(la) else 'b'}"
        a, b = la[i], lb[i]
        field = next((k for k in sorted(set(a) | set(b), key=str) if not _same(a.get(k), b.get(k))), None)
        if field is not None:
            return f"{i} {a.get('event')} {field}"
    return None


def _members(out_dir: Path, key: tuple) -> dict | None:
    path = _member_path(out_dir, key)
    if not path.exists():
        return None
    out: dict = {}
    for r in C.read_jsonl(path):
        out.setdefault(int(r["window"]), {})[r["member_id"]] = r
    return out


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return canonical(a) == canonical(b)


def _first_field(a: dict | None, b: dict | None, fields) -> str | None:
    if a is None or b is None:
        return "window missing"
    return next((f for f in fields if not _same(a.get(f), b.get(f))), None)


def _member_divergence(ma: dict, mb: dict) -> tuple[int | None, str | None, str | None]:
    """The first window where the ensembles part, the field, and the cause read there."""
    for w in sorted(set(ma) | set(mb)):
        a, b = ma.get(w, {}), mb.get(w, {})
        if set(a) != set(b):
            return w, "member_ids", _cause(a, b)
        for mid in sorted(a, key=str):
            field = _first_field(a[mid], b[mid], MEMBER_FIELDS)
            if field is not None:
                return w, field, _cause(a, b)
    return None, None, None


def _cause(a: dict, b: dict) -> str:
    """Why two ensembles part on one window (see the module docstring)."""
    for mid in set(a) & set(b):
        ra, rb = a[mid].get("feasible_reason"), b[mid].get("feasible_reason")
        if ra != rb and "runtime" in (ra, rb):
            return "runtime gate"
    if set(a) != set(b):
        return "sampling"
    for mid in a:
        if not _same(a[mid].get("params"), b[mid].get("params")):
            return "sampling"
    return "model or scoring"


def _share_over(members: dict | None, factor: float) -> float | None:
    if members is None:
        return None
    ratios = [m.get("runtime_ratio") for ms in members.values() for m in ms.values()]
    ratios = [r for r in ratios if r is not None]
    return float(np.mean([r > factor for r in ratios])) if ratios else None


def compare(dir_a: Path, dir_b: Path, factor: float = C.RUNTIME_FACTOR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(jobs, windows)``: one row per job of either dir, one per window where they differ."""
    sa, sb = _summaries(dir_a), _summaries(dir_b)
    wa, wb = _windows(dir_a), _windows(dir_b)
    jobs, diffs = [], []
    for key in sorted(set(sa) | set(sb), key=lambda k: tuple(str(v) for v in k)):
        row = dict(zip(KEY, key))
        if key not in sb or key not in sa:
            jobs.append({**row, "status": "only in a" if key in sa else "only in b", "identical": False})
            continue
        a, b = sa[key], sb[key]
        if a.get("status") == "skipped" and b.get("status") == "skipped":
            # a job both runs wrote down as not run (an identical configuration): nothing to compare
            jobs.append({**row, "status": "skipped on both", "identical": a.get("error") == b.get("error")})
            continue
        if a.get("status") != "ok" or b.get("status") != "ok":
            jobs.append({**row, "status": f"a: {a.get('status')}, b: {b.get('status')}", "identical": False})
            continue
        ta, tb = wa.get(key, {}), wb.get(key, {})
        first_w = first_f = None
        n_diff, max_ari = 0, 0.0
        for w in sorted(set(ta) | set(tb)):
            field = _first_field(ta.get(w), tb.get(w), WINDOW_FIELDS)
            if field is None:
                continue
            n_diff += 1
            ra, rb = ta.get(w) or {}, tb.get(w) or {}
            fields = [f for f in WINDOW_FIELDS if not _same(ra.get(f), rb.get(f))] if ra and rb else [field]
            xa, xb = ra.get("lag_ari_win"), rb.get("lag_ari_win")
            if xa is not None and xb is not None:
                max_ari = max(max_ari, abs(float(xa) - float(xb)))
            diffs.append({**row, "window": w, "fields": ",".join(fields),
                          "a_incumbent_id": ra.get("incumbent_id"), "b_incumbent_id": rb.get("incumbent_id"),
                          "a_lag_ari_win": xa, "b_lag_ari_win": xb})
            if first_w is None:
                first_w, first_f = w, field
        ma, mb = _members(dir_a, key), _members(dir_b, key)
        logged = ma is not None and mb is not None
        mw, mf, cause = _member_divergence(ma, mb) if logged else (None, None, None)
        if cause is None:
            cause = ("none" if first_w is None else "tie-break") if logged else (
                "none" if first_w is None else "not logged")
        la, lb = _lifecycle(dir_a, key), _lifecycle(dir_b, key)
        life_logged = la is not None and lb is not None
        life_div = _lifecycle_divergence(la, lb) if life_logged else None
        tail_a, tail_b = a.get("lag_ari_win_tail_mean"), b.get("lag_ari_win_tail_mean")
        jobs.append({
            **row, "status": "both", "identical": first_w is None and mw is None and life_div is None,
            "n_windows": len(ta), "lifecycle_logged": life_logged,
            "n_lifecycle_events": len(la) if life_logged else None, "lifecycle_first_divergence": life_div,
            "first_divergence_window": first_w, "first_divergence_field": first_f, "n_windows_differing": n_diff,
            "max_abs_lag_ari_diff": max_ari, "a_tail": tail_a, "b_tail": tail_b,
            "tail_diff": (tail_b - tail_a) if tail_a is not None and tail_b is not None else None,
            "wall_ratio": (b["wall_s"] / a["wall_s"]) if a.get("wall_s") and b.get("wall_s") else None,
            "members_logged": logged, "member_first_divergence_window": mw, "member_first_divergence_field": mf,
            "cause": cause, "a_share_over_factor": _share_over(ma, factor), "b_share_over_factor": _share_over(mb, factor),
        })
    return pd.DataFrame(jobs).reindex(columns=JOB_COLS), pd.DataFrame(diffs).reindex(columns=WINDOW_COLS)


def strict_failures(jobs: pd.DataFrame, require_members: bool = False) -> list[str]:
    """Why a reproducibility check fails, one line per job: on one side only or not ok, not identical,
    without evaluation windows, or (``require_members``) without member logs on both sides. A job both
    sides skipped for the same reason is no failure; no job at all is one."""
    if jobs.empty:
        return ["no job on either side"]
    out = []
    for r in jobs.to_dict("records"):
        name = " ".join(str(r.get(k)) for k in KEY)
        if r.get("status") == "skipped on both" and r.get("identical"):
            continue
        if r.get("status") != "both":
            out.append(f"{name}: {r.get('status')}")
            continue
        if not r.get("n_windows"):
            out.append(f"{name}: no evaluation window")
        if require_members and not r.get("members_logged"):
            out.append(f"{name}: member logs missing on a side")
        if not r.get("identical"):
            out.append(f"{name}: differs (window {r.get('first_divergence_window')}, member window "
                       f"{r.get('member_first_divergence_window')}, lifecycle {r.get('lifecycle_first_divergence')})")
    return out


def write_repeat(dir_a: Path, dir_b: Path, out: Path, factor: float = C.RUNTIME_FACTOR) -> list[Path]:
    from .analyze import _md_table

    jobs, windows = compare(dir_a, dir_b, factor)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "jobs.csv", out / "windows.csv", out / "repeat.md"]
    jobs.to_csv(paths[0], index=False)
    windows.to_csv(paths[1], index=False)
    both = jobs[jobs["status"] == "both"]
    lines = [
        "# Repeatability: two runs of one protocol", "",
        f"`a` = `{dir_a}`, `b` = `{dir_b}`. Jobs are paired by (algo, fold, seed, arm, config_id, model_seed); per window "
        f"the trajectory fields {', '.join(f'`{f}`' for f in WINDOW_FIELDS)} are compared exactly, and with "
        f"member logs on both sides every member's {', '.join(f'`{f}`' for f in MEMBER_FIELDS)}. Runtimes "
        "are expected to differ and are only reported.", "",
        f"- jobs on both sides: {len(both)}; identical: {int(both['identical'].sum()) if len(both) else 0}; "
        f"on one side only or not ok on both: {len(jobs) - len(both)}",
    ]
    if len(both):
        causes = both.loc[~both["identical"], "cause"].value_counts()
        lines.append("- causes of the divergences: " + (", ".join(f"{c} {n}" for c, n in causes.items()) or "none"))
        lines.append(f"- windows differing: {int(both['n_windows_differing'].sum())} of {int(both['n_windows'].sum())}")
        lines.append(f"- largest |tail ARI difference|: {float(both['tail_diff'].abs().max()):.4f}; "
                     f"wall-time ratio b/a: median {float(both['wall_ratio'].median()):.2f}, "
                     f"range {float(both['wall_ratio'].min()):.2f}-{float(both['wall_ratio'].max()):.2f}")
        for side in ("a", "b"):
            col = f"{side}_share_over_factor"
            if both[col].notna().any():
                lines.append(f"- {side}: share of member-windows with a runtime ratio above {factor:g} "
                             f"(what a runtime cap at {factor:g} sheds): mean {float(both[col].mean()):.4f}, "
                             f"max {float(both[col].max()):.4f}")
        lines += ["", "## Jobs", "", _md_table(jobs, 4), ""]
    paths[2].write_text("\n".join(lines), encoding="utf-8")
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, help="one out dir")
    parser.add_argument("--b", required=True, help="the other, made under the same protocol")
    parser.add_argument("--out", default=None, help="where the tables go (default: <b>/report/repeat)")
    parser.add_argument("--factor", type=float, default=C.RUNTIME_FACTOR,
                        help="the runtime cap replayed on the member logs")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 unless every job is on both sides, ok, identical and has evaluation windows")
    parser.add_argument("--require-members", action="store_true",
                        help="with --strict: also fail a job without member logs on both sides")
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.b) / C.REPORT_DIR.name / "repeat"
    for p in write_repeat(Path(args.a), Path(args.b), out, args.factor):
        print(p)
    if args.strict:
        jobs, _ = compare(Path(args.a), Path(args.b), args.factor)
        failures = strict_failures(jobs, args.require_members)
        if failures:
            print("NOT REPRODUCED:\n  " + "\n  ".join(failures))
            raise SystemExit(1)
        print(f"reproduced: {len(jobs)} jobs identical")


if __name__ == "__main__":
    main()
