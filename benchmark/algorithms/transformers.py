"""Small transformers that fill gaps River leaves open for clustering.

Everything here follows River's ``learn_one`` / ``transform_one`` protocol, so
it drops into the same preprocessing slot as any River transformer and composes
with ``|``.
"""

from __future__ import annotations

import math
import numbers

from .adapter import build_river_path

build_river_path()

from river import base, preprocessing  # noqa: E402


class OneHotCategorical(base.Transformer):
    """One-hot encode only the non-numeric features; pass numbers through.

    River's :class:`~river.preprocessing.OneHotEncoder` encodes *every* key it
    is given, which is correct for a record that is entirely categorical but
    destructive on a mixed one: a numeric column turns into one indicator per
    distinct value it has ever taken. On KDDCup99's ``duration`` that is tens of
    thousands of columns, each carrying a single observation, and every distance
    between two points collapses towards the same value.

    The River idiom for avoiding it is to route with ``compose.Select`` and
    rejoin the branches, which needs the categorical column names known up front.
    A stream does not necessarily offer them, so this transformer decides per
    value instead: anything that is a number stays as it is, anything else is
    one-hot encoded.

    Parameters
    ----------
    drop_first
        Passed to the underlying encoder; drops one indicator per feature to
        avoid the collinear dummy.

    Examples
    --------
    >>> from benchmark.algorithms.transformers import OneHotCategorical
    >>> enc = OneHotCategorical()
    >>> for proto in ("tcp", "udp", "tcp"):
    ...     _ = enc.learn_one({"duration": 1.5, "proto": proto})
    >>> sorted(enc.transform_one({"duration": 2.5, "proto": "udp"}).items())
    [('duration', 2.5), ('proto_tcp', 0), ('proto_udp', 1)]
    """

    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first
        self._encoder = preprocessing.OneHotEncoder(drop_first=drop_first)

    @staticmethod
    def _split(x: dict) -> tuple[dict, dict]:
        numeric, categorical = {}, {}
        for key, value in x.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Number):
                categorical[key] = value
            else:
                numeric[key] = value
        return numeric, categorical

    def learn_one(self, x: dict) -> None:
        _, categorical = self._split(x)
        if categorical:
            self._encoder.learn_one(categorical)

    def transform_one(self, x: dict) -> dict:
        numeric, categorical = self._split(x)
        if not categorical:
            return dict(numeric)
        return {**numeric, **self._encoder.transform_one(categorical)}


class RealValued(base.Transformer):
    """Wrap a transformer and coerce any non-real or non-finite output to ``0.0``.

    A workaround for RIVER_BUGS.md BUG-1, kept on this side of the boundary
    because the river tree is never modified.

    ``StandardScaler(window_size=...)`` divides by ``variance ** 0.5``, guarded
    only against an exactly-zero variance. On a constant feature the rolling
    variance drifts slightly *negative* through floating-point cancellation, and
    a negative float raised to ``0.5`` is a **complex number** in Python, not an
    error. The complex value travels into the model and finally fails inside the
    distance function, a long way from where it was produced.

    This wrapper stops it at the source: anything that is not a finite real
    number becomes ``0.0``, which is what the original guard intended for a
    feature with no spread.

    Parameters
    ----------
    transformer
        The wrapped river transformer.

    Examples
    --------
    >>> from river import preprocessing
    >>> from benchmark.algorithms.transformers import RealValued
    >>> safe = RealValued(preprocessing.StandardScaler())
    >>> safe.learn_one({"a": 1.0})
    >>> safe.transform_one({"a": 1.0})
    {'a': 0.0}
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def learn_one(self, x: dict) -> None:
        self.transformer.learn_one(x)

    def transform_one(self, x: dict) -> dict:
        out = self.transformer.transform_one(x)
        return {k: v if _is_finite_real(v) else 0.0 for k, v in out.items()}


def _is_finite_real(value) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, numbers.Real):
        return math.isfinite(value)
    return False


def safe_standard_scaler(**params):
    """``StandardScaler`` wrapped so it cannot emit complex values.

    See RIVER_BUGS.md BUG-1. Registered as the ``standard_safe`` preprocessor;
    plain ``standard`` stays untouched river behaviour so the two are comparable.
    """
    return RealValued(preprocessing.StandardScaler(**params))
