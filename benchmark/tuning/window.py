"""One evaluation window: its points, a dataset view of them, and a batch score.

The ensemble processes the stream in windows of ``h`` samples. Every member is
run over the *same* window through :func:`benchmark.evaluation.runner.run_once`,
so the window has to be re-iterable and has to hand each member its own copy of
each point (River clusterers keep a reference to the dict they are given,
RIVER_BUGS.md BUG-3). The buffer is also what a freshly created member is
warm-started on, which is the one piece of history a new configuration gets.

The score is the classic silhouette in [-1, 1] over the window's points and the
member's own predictions, computed in batch with scikit-learn. River's streaming
``metrics.Silhouette`` is a cohesion/separation ratio (lower is better, infinite
at one cluster) and cannot feed a roulette wheel that needs "higher is better".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import silhouette_score

from ..algorithms.adapter import ClustererAdapter
from ..datasets.base import IterableDataset

#: The score a window gets when the silhouette is undefined on it: a single
#: predicted cluster, or every point in a cluster of its own. It is the floor of
#: the silhouette's range, so such a configuration ranks below any configuration
#: that produced a partition at all -- and it is still a number, so the roulette
#: wheel stays well-defined when every member is in that state.
UNDEFINED_SCORE = -1.0


def batch_silhouette(xs: Sequence[dict], labels: Sequence) -> tuple[float, bool]:
    """Silhouette of ``labels`` over ``xs``; ``(score, defined)``.

    A ``None`` label (a prediction the adapter could not make) drops that point
    from the score rather than becoming a cluster of its own. With fewer than two
    distinct labels, or with as many labels as points, the silhouette has no
    value and :data:`UNDEFINED_SCORE` is returned with ``defined=False``.
    """
    keep = [i for i, lab in enumerate(labels) if lab is not None]
    if len(keep) < 2:
        return UNDEFINED_SCORE, False
    kept_labels = [labels[i] for i in keep]
    n_labels = len(set(kept_labels))
    if n_labels < 2 or n_labels >= len(keep):
        return UNDEFINED_SCORE, False
    keys = sorted(xs[keep[0]].keys())
    X = np.asarray([[float(xs[i].get(k, 0.0)) for k in keys] for i in keep], dtype=float)
    # Labels can be any hashable (DenStream hands out ints, TextClust strings);
    # scikit-learn wants an array it can compare, so map them to dense ids.
    ids = {lab: j for j, lab in enumerate(dict.fromkeys(kept_labels))}
    y = np.asarray([ids[lab] for lab in kept_labels])
    return float(silhouette_score(X, y)), True


class WindowBuffer:
    """The ``h`` most recent ``(x, y)`` pairs, filled once per window."""

    def __init__(self, h: int):
        if h < 1:
            raise ValueError(f"window size h must be >= 1, got {h}")
        self.h = h
        self.xs: list[dict] = []
        self.ys: list = []

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def is_full(self) -> bool:
        return len(self.xs) >= self.h

    def push(self, x: dict, y) -> bool:
        """Append one sample; returns ``True`` once the window is full."""
        if self.is_full:
            raise ValueError(f"window already holds {self.h} samples; clear() it first")
        self.xs.append(dict(x))
        self.ys.append(y)
        return self.is_full

    def clear(self) -> None:
        self.xs = []
        self.ys = []

    def as_dataset(self, name: str = "window") -> IterableDataset:
        """A re-iterable view; every iteration yields fresh copies of the points."""
        xs, ys = self.xs, self.ys
        has_labels = any(y is not None for y in ys)
        n_features = len(xs[0]) if xs else None
        return IterableDataset(
            lambda: ((dict(x), y) for x, y in zip(xs, ys)),
            name=name, n_features=n_features, has_labels=has_labels, length=len(xs),
        )


def warm_start(adapter: ClustererAdapter, buffer: WindowBuffer) -> int:
    """Replay the buffer into ``adapter`` (learn only); returns the sample count."""
    for x in buffer.xs:
        adapter.learn_only(dict(x))
    return len(buffer)
