"""Stage 1 -- STREAM: raw, then shuffle, then stream.

Shuffling forces the dataset to be materialised once. A ``Dataset`` in this
framework is a lazy factory that builds a fresh iterator per ``iter()``; if the
shuffle lived inside the factory, each of the thousands of runs would re-read
and re-shuffle the source, which is both slow and a violation of the rule the
framework is built on -- every setting must consume the *identical* stream.

So: materialise once per (fold, shuffle_seed), cache as ``.npy``, memory-map on
use. RSS then does not grow with the number of workers.

Shuffling buys a lot and costs a lot, and both belong in the write-up. It
removes the block-ordered arrival that makes per-window metrics degenerate, and
it makes a prefix a representative sample so Stage 2 can probe one. It also
erases every temporal property the stream had, which is why the drift scenarios
are not folds here and why the decay parameters must not be read as statements
about adaptation.

Gate: the shuffled array has the same length as the source and the same label
distribution -- a permutation must not lose a sample.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from ._common import (
    CACHE,
    FOLDS,
    SEEDS,
    load_fold,
    shuffle_seed_for,
)


def _paths(fold: str, shuffle_seed: int):
    return (
        CACHE / f"{fold}_{shuffle_seed}_X.npy",
        CACHE / f"{fold}_{shuffle_seed}_y.npy",
        CACHE / f"{fold}_features.json",
        CACHE / f"{fold}_stats.json",
    )


def materialise(fold: str, shuffle_seed: int, force: bool = False) -> dict:
    """Raw -> shuffle -> cached arrays. Runs once per (fold, shuffle_seed)."""
    xp, yp, fp, sp = _paths(fold, shuffle_seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    if xp.exists() and yp.exists() and fp.exists() and sp.exists() and not force:
        return json.loads(sp.read_text(encoding="utf-8"))

    dataset = load_fold(fold)
    rows: list[list[float]] = []
    labels: list = []
    feats: list[str] | None = None
    for x, y in dataset:
        if feats is None:
            feats = sorted(x)
        elif len(x) != len(feats):
            raise SystemExit(
                f"fold {fold!r} has a ragged schema: first record had {len(feats)} "
                f"features, a later one has {len(x)}"
            )
        rows.append([float(x[f]) for f in feats])
        labels.append(y)

    if feats is None:
        raise SystemExit(f"fold {fold!r} yielded no records")

    X = np.asarray(rows, dtype=np.float64)
    del rows
    # Object dtype keeps `None` as itself. A numeric sentinel would collide with
    # a real label the moment a dataset uses that value.
    y = np.empty(len(labels), dtype=object)
    y[:] = labels

    before = collections.Counter(labels)
    rng = np.random.default_rng(shuffle_seed)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]
    after = collections.Counter(y.tolist())

    if before != after:
        raise SystemExit(f"GATE FAILED: {fold} label distribution changed under shuffle")

    np.save(xp, X)
    np.save(yp, y, allow_pickle=True)
    fp.write_text(json.dumps(feats), encoding="utf-8")

    stats = {
        "fold": fold,
        "shuffle_seed": shuffle_seed,
        "n_samples": int(len(X)),
        "n_features": len(feats),
        "n_classes": len([k for k in before if k is not None]),
        "n_unlabeled": int(before.get(None, 0)),
        "label_counts": {str(k): int(v) for k, v in before.items()},
    }
    sp.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats


def features_of(fold: str) -> list[str]:
    _, _, fp, _ = _paths(fold, 0)
    return json.loads(fp.read_text(encoding="utf-8"))


def stats_of(fold: str) -> dict:
    _, _, _, sp = _paths(fold, 0)
    return json.loads(sp.read_text(encoding="utf-8"))


def load_arrays(fold: str, shuffle_seed: int):
    """Memory-mapped X and the label array."""
    xp, yp, _, _ = _paths(fold, shuffle_seed)
    X = np.load(xp, mmap_mode="r")
    y = np.load(yp, allow_pickle=True)
    return X, y


def shuffled_stream(fold: str, shuffle_seed: int, feats: list[str] | None = None):
    """A re-iterable Dataset reading the cached, shuffled arrays."""
    from benchmark.datasets.base import IterableDataset

    xp, yp, fp, _ = _paths(fold, shuffle_seed)
    if feats is None:
        feats = json.loads(fp.read_text(encoding="utf-8"))
    length = int(np.load(xp, mmap_mode="r").shape[0])

    def factory():
        X = np.load(xp, mmap_mode="r")
        y = np.load(yp, allow_pickle=True)
        for i in range(len(X)):
            yield {f: float(v) for f, v in zip(feats, X[i])}, y[i]

    return IterableDataset(
        factory,
        name=f"{fold}@s{shuffle_seed}",
        n_features=len(feats),
        has_labels=True,
        length=length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for fold in folds:
        for seed in seeds:
            shuffle_seed = shuffle_seed_for(seed)
            stats = materialise(fold, shuffle_seed, force=args.force)
            print(
                f"{fold:16s} s{shuffle_seed} n={stats['n_samples']:>7} "
                f"d={stats['n_features']:>3} classes={stats['n_classes']:>3} "
                f"unlabeled={stats['n_unlabeled']}"
            )
    print("GATE PASSED: every permutation preserved its label distribution")


if __name__ == "__main__":
    main()
