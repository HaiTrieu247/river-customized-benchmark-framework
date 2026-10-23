"""A5 driver: batch clustering with the true number of classes on every window of every fold.

Cuts each fold into windows of ``h`` samples, as ``run.py`` does (window ``w`` holds
samples ``w*h`` to ``(w+1)*h - 1``), and writes what
:func:`benchmark.evaluation.label_reach.window_readings` reads on each into
``<out-dir>/label_reach.jsonl``: one ``kind = "reading"`` row per (window, method,
representation), then one ``kind = "fold"`` row once the fold is finished. A window
whose labels hold one class has no reading; the fold row counts such windows. A fold
with its ``fold`` row on file is skipped, so an interrupted command resumes by running
it again. No tuning run is involved; ``robustness`` reads the file.

Run from the repo root::

    .venv/bin/python -m conf_pipeline.label_reach --out-dir conf_pipeline/out/full --workers 8
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import _common as C  # first: it puts the river tree on sys.path

from benchmark.evaluation import label_reach as LR  # noqa: E402

OUT_NAME = "label_reach.jsonl"


def fold_rows(fold: str, h: int = C.H, n_samples: int | None = C.N_SAMPLES_CAP,
              eps_grid=LR.EPS_GRID, min_samples: int = LR.MIN_SAMPLES) -> list[dict]:
    """Every reading of one fold, then its ``fold`` row."""
    t0 = time.perf_counter()
    base = {"fold": fold, "h": h, "n_samples_cap": n_samples, "min_samples": min_samples}
    rows, xs, ys = [], [], []
    n_seen = window = n_single = 0
    for x, y in C.load_fold(fold, n_samples):
        if n_samples is not None and n_seen >= n_samples:
            break
        xs.append(dict(x))
        ys.append(y)
        n_seen += 1
        if len(xs) < h:
            continue
        readings = LR.window_readings(xs, ys, eps_grid=eps_grid, min_samples=min_samples)
        n_single += not readings
        rows += [{"kind": "reading", **base, "window": window, "start": n_seen - h, "end": n_seen - 1, **r}
                 for r in readings]
        xs, ys = [], []
        window += 1
    rows.append({"kind": "fold", **base, "n_windows": window, "n_single_class": n_single,
                 "n_samples": n_seen, "eps_grid": list(eps_grid), "wall_s": time.perf_counter() - t0})
    return rows


def done_folds(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["fold"] for r in C.read_jsonl(path) if r.get("kind") == "fold"}


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", default=",".join(C.FOLDS))
    parser.add_argument("--h", type=int, default=C.H)
    parser.add_argument("--n-samples", type=int, default=C.N_SAMPLES_CAP, help="0 = whole stream")
    parser.add_argument("--min-samples", type=int, default=LR.MIN_SAMPLES,
                        help="DBSCAN's min_samples (a placeholder default)")
    parser.add_argument("--eps-grid", default=",".join(f"{v:.6g}" for v in LR.EPS_GRID),
                        help="DBSCAN radii as multiples of the window's median pairwise distance")
    parser.add_argument("--out-dir", default=str(C.OUT))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="override a lock left by a dead process")
    args = parser.parse_args()

    path = Path(args.out_dir) / OUT_NAME
    n_samples = args.n_samples or None
    eps_grid = tuple(float(v) for v in _csv(args.eps_grid))
    with C.JsonlLock(path, force=args.force):
        todo = [f for f in _csv(args.folds) if f not in done_folds(path)]
        print(f"{len(todo)} folds to read ({path})")
        kwargs = dict(h=args.h, n_samples=n_samples, eps_grid=eps_grid, min_samples=args.min_samples)
        if args.workers <= 1:
            results = ((f, fold_rows(f, **kwargs)) for f in todo)
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers)
            futures = {pool.submit(fold_rows, f, **kwargs): f for f in todo}
            results = ((futures[fut], fut.result()) for fut in as_completed(futures))
        for fold, rows in results:
            for row in rows:  # the fold row comes last: a fold is done only once all of it is on file
                C.append_jsonl(path, row)
            print(f"  {fold}: {rows[-1]['n_windows']} windows, {rows[-1]['n_single_class']} single-class, "
                  f"{rows[-1]['wall_s']:.0f}s")
        if args.workers > 1:
            pool.shutdown()


if __name__ == "__main__":
    main()
