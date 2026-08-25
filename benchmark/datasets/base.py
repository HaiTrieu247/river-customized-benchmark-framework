"""Common dataset interface for the benchmark.

A :class:`Dataset` is a *re-iterable* source of ``(x, y)`` pairs, where ``x`` is
a feature dict (what the model sees) and ``y`` is an optional ground-truth label
used only to score clustering quality (``None`` when the dataset is unlabeled).

Re-iterability matters: every algorithm/setting must consume the *same* stream
from the start, so datasets yield a fresh iterator each time (Principle: fixed
data stream, no shuffling — record order is the streaming order).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Tuple

Sample = Tuple[dict, Optional[object]]


class Dataset:
    """Base class: subclasses implement :meth:`__iter__` and ``n_features``."""

    name: str = "unnamed"
    n_features: int | None = None
    has_labels: bool = False

    def __iter__(self) -> Iterator[Sample]:  # pragma: no cover - interface
        raise NotImplementedError

    def __len__(self) -> int | None:  # optional
        return None

    def take(self, n: int) -> Iterator[Sample]:
        for i, item in enumerate(self):
            if i >= n:
                break
            yield item


class IterableDataset(Dataset):
    """Wrap any zero-arg factory returning an iterable of ``(x, y)``.

    The factory is called on every ``iter()`` so the stream restarts cleanly.
    """

    def __init__(self, factory, name: str, n_features: int | None = None,
                 has_labels: bool = False, length: int | None = None):
        self._factory = factory
        self.name = name
        self.n_features = n_features
        self.has_labels = has_labels
        self._length = length

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._factory())

    def __len__(self) -> int | None:
        return self._length
