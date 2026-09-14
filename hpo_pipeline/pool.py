"""Stage 0 -- PREFLIGHT: freeze the algorithm pool and the fold pool.

The point is not to discover anything; it is to fail in the first minute rather
than in the third hour, on an algorithm that cannot be constructed or a fold
that cannot be iterated.

Gate: every name in the pool constructs with ``scaler="none"``, and every fold
iterates and yields at least one sample.
"""

from __future__ import annotations

import argparse
import itertools

from ._common import FOLD_GROUPS, POOL_JSON, SCALER, load_fold, write_json


def numeric_pool() -> tuple[list[str], list[dict]]:
    """Algorithms that partition samples, take coordinates, and construct."""
    from benchmark.algorithms.registry import ALGORITHMS, get_algorithm

    pool: list[str] = []
    dropped: list[dict] = []
    for name, spec in ALGORITHMS.items():
        if not spec.partitions_samples:
            # ODAC clusters variables, not samples: no per-sample label exists
            # to compare against ground truth.
            dropped.append({"algorithm": name, "reason": "does not partition samples"})
            continue
        if spec.input_kind != "numeric":
            # TextClust consumes token counts, not coordinates.
            dropped.append({"algorithm": name, "reason": f"input_kind={spec.input_kind}"})
            continue
        try:
            get_algorithm(name, scaler=SCALER)
        except Exception as exc:
            dropped.append({"algorithm": name, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        pool.append(name)
    return pool, dropped


def check_folds() -> list[dict]:
    """Iterate one sample out of every fold."""
    report = []
    for name, group in FOLD_GROUPS.items():
        entry = {"name": name, "group": group}
        try:
            dataset = load_fold(name)
            head = list(itertools.islice(iter(dataset), 1))
            if not head:
                entry["status"] = "empty"
            else:
                x, _ = head[0]
                entry["status"] = "ok"
                entry["n_features"] = dataset.n_features
                entry["n_keys_first_record"] = len(x)
                length = len(dataset)
                entry["length"] = length
        except Exception as exc:
            entry["status"] = f"{type(exc).__name__}: {exc}"
        report.append(entry)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-folds",
        action="store_true",
        help="only check the algorithm pool (folds may need downloading)",
    )
    args = parser.parse_args()

    pool, dropped = numeric_pool()
    print(f"pool ({len(pool)}): {pool}")
    for entry in dropped:
        print(f"  dropped {entry['algorithm']}: {entry['reason']}")

    folds = [] if args.skip_folds else check_folds()
    for entry in folds:
        print(f"  fold {entry['name']:16s} {entry['status']}")

    failed = [f for f in folds if f["status"] != "ok"]
    payload = {
        "scaler": SCALER,
        "shuffle": True,
        "algorithms": pool,
        "dropped": dropped,
        "folds": [{"name": n, "group": g} for n, g in FOLD_GROUPS.items()],
        "fold_checks": folds,
    }
    write_json(POOL_JSON, payload)
    print(f"wrote {POOL_JSON}")

    if not pool:
        raise SystemExit("GATE FAILED: no algorithm survived the pool filter")
    if failed:
        raise SystemExit(f"GATE FAILED: folds not iterable: {[f['name'] for f in failed]}")
    print("GATE PASSED")


if __name__ == "__main__":
    main()
