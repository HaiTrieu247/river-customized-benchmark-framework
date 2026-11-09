"""Stream-level scaling: one transformation of the stream, shared by every model that reads it.

A scaler inside each model's adapter would be learned anew by every model, from the points that model
saw: a challenger built after window ``t - 1`` would scale with the statistics of one window, the
members it competes with with those of the whole stream so far, and the members would not share their
coordinates. Scaling the stream itself, once, before any model, any distance unit or any score reads
it, gives every reader the same coordinates.

``RollingStandardised`` applies River's ``StandardScaler(window_size=w)`` to every numeric feature,
each point first learned and then transformed (River's own idiom, as in the scaler's docstring and the
benchmark adapter's ``_scale``): a point is scaled with the mean and the population variance of itself
and the ``w - 1`` points before it, never with a later one. The scaler is the benchmark's
``standard_safe`` (``transformers.safe_standard_scaler``), the same River scaler with a non-finite or
complex output set to 0.0 (RIVER_BUGS.md BUG-1: on a constant feature the rolling variance can drift
below zero). A non-numeric value passes through unchanged; the labels are untouched.
"""

from __future__ import annotations

import numbers

#: The stream scalings a driver knows: ``none`` keeps the supplied coordinates.
STREAM_SCALINGS: tuple[str, ...] = ("none", "rolling_standard")


def _numeric(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


class RollingStandardised:
    """``dataset`` (an iterable of ``(x, y)``) with every numeric feature standardised by River's
    ``StandardScaler(window_size=window_size)`` (see the module docstring)."""

    def __init__(self, dataset, window_size: int):
        if window_size is None or int(window_size) < 2:
            raise ValueError(f"window_size must be an integer >= 2, got {window_size!r}")
        self.dataset, self.window_size = dataset, int(window_size)
        self.n_seen = 0
        self.n_passthrough = 0
        #: The live scaler while the stream is read: a caller may transform side points (a probe set) with
        #: the statistics of the points read so far, without teaching it anything.
        self.scaler = None

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __iter__(self):
        from ..algorithms.transformers import safe_standard_scaler

        scaler = self.scaler = safe_standard_scaler(window_size=self.window_size)
        for x, y in self.dataset:
            num = {k: float(v) for k, v in x.items() if _numeric(v)}
            rest = {k: v for k, v in x.items() if not _numeric(v)}
            scaler.learn_one(num)
            out = scaler.transform_one(num)
            self.n_seen += 1
            self.n_passthrough += len(rest)
            yield ({**out, **rest} if rest else out), y

    def summary(self) -> dict:
        return {"scaler": "river.preprocessing.StandardScaler (standard_safe)", "window_size": self.window_size,
                "order": "learn then transform", "n_seen": self.n_seen, "n_passthrough": self.n_passthrough}


def transform_side(stream, x: dict) -> dict:
    """``x`` under the statistics a ``RollingStandardised`` stream has read so far (``x`` unchanged for any
    other stream); the scaler learns nothing from it."""
    scaler = getattr(stream, "scaler", None) if isinstance(stream, RollingStandardised) else None
    if scaler is None:
        return dict(x)
    num = {k: float(v) for k, v in x.items() if _numeric(v)}
    return {**scaler.transform_one(num), **{k: v for k, v in x.items() if not _numeric(v)}}


def scale_stream(dataset, scaling: str | None, window_size: int | None):
    """``dataset`` under ``scaling`` (``STREAM_SCALINGS``; None reads as ``none``)."""
    if scaling in (None, "none"):
        return dataset
    if scaling == "rolling_standard":
        return RollingStandardised(dataset, window_size)
    raise ValueError(f"stream scaling must be one of {STREAM_SCALINGS}, got {scaling!r}")
