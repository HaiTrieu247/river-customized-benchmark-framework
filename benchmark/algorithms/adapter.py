"""Uniform adapter over any ``river.base.Clusterer``.

The adapter gives every clusterer the same interface for the benchmark runner
and, optionally, chains an **online scaler** in front of the model (the README
stresses that the scaler strongly affects the resulting number of clusters and
quality). It also reports, best-effort, the micro-cluster count, macro-cluster
count and model memory so efficiency can be compared across algorithms.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any


def build_river_path() -> str:
    """Return (and register on ``sys.path``) the sibling river source tree.

    Layout expected (see README): the benchmark repo sits next to ``river/``::

        OnlineML/
        ├── river/                                   # river source tree
        └── river-customized-benchmark-framework/
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    river_root = os.path.abspath(os.path.join(repo_root, "..", "river"))
    if river_root not in sys.path:
        sys.path.insert(0, river_root)
    return river_root


@dataclass
class ClustererAdapter:
    """Wrap a river clusterer, exposing benchmark-friendly hooks.

    Attributes
    ----------
    name:
        Display name (may include a ``#2`` suffix when comparing configs).
    model:
        The underlying river clusterer instance.
    params:
        The constructor params used (recorded for reproducibility).
    scaler:
        Optional online scaler applied to each ``x`` before predict/learn. Must
        implement river's ``learn_one`` + ``transform_one``. ``None`` => raw.
    scaler_name:
        Human-readable scaler name for reporting (e.g. ``"StandardScaler"``).
    """

    name: str
    model: Any
    params: dict = field(default_factory=dict)
    scaler: Any = None
    scaler_name: str = "none"

    # -- scaling -------------------------------------------------------------
    def _scale(self, x: dict) -> dict:
        """Learn-then-transform the point with the online scaler, if present."""
        if self.scaler is None:
            return x
        self.scaler.learn_one(x)
        return self.scaler.transform_one(x)

    # -- online protocol -----------------------------------------------------
    def predict_one(self, x: dict) -> int:
        """Assign ``x`` to a cluster.

        Some clusterers (e.g. STREAMKMeans) raise before they have initialised
        any center. During that cold-start phase we return label ``0`` so the
        stream keeps flowing; those early samples are scored like any other.
        """
        try:
            return self.model.predict_one(x)
        except (ValueError, ZeroDivisionError, KeyError):
            return 0

    def learn_one(self, x: dict) -> None:
        self.model.learn_one(x)

    def process_one(self, x: dict) -> int:
        """Full per-sample step used by the runner: scale -> predict -> learn.

        Returns the predicted cluster label. Prediction happens before learning
        (test-then-train), matching a live streaming deployment.
        """
        xs = self._scale(x)
        y_pred = self.predict_one(xs)
        self.learn_one(xs)
        return y_pred

    # -- introspection (best effort) ----------------------------------------
    def n_micro_clusters(self) -> int | None:
        """Number of micro-clusters, if the model exposes them."""
        m = self.model
        for attr in ("micro_clusters", "_micro_clusters"):
            mc = getattr(m, attr, None)
            if mc is not None:
                try:
                    return len(mc)
                except TypeError:
                    pass
        # DenStream keeps potential + outlier micro-clusters separately
        p = getattr(m, "p_micro_clusters", None)
        o = getattr(m, "o_micro_clusters", None)
        if p is not None or o is not None:
            total = 0
            for group in (p, o):
                try:
                    total += len(group) if group is not None else 0
                except TypeError:
                    pass
            return total
        return None

    def n_macro_clusters(self) -> int | None:
        """Number of macro clusters, if exposed."""
        for attr in ("n_clusters", "_n_clusters", "n_macro_clusters"):
            v = getattr(self.model, attr, None)
            if isinstance(v, int):
                return v
        centers = self.centers()
        return len(centers) if centers is not None else None

    def centers(self) -> dict | None:
        c = getattr(self.model, "centers", None)
        if callable(c):
            try:
                c = c()
            except Exception:
                return None
        return c if isinstance(c, dict) else None

    def memory_bytes(self) -> int | None:
        """Model object size via river's ``_raw_memory_usage`` (README).

        Includes the scaler's footprint when one is attached.
        """
        total = 0
        found = False
        for obj in (self.model, self.scaler):
            v = getattr(obj, "_raw_memory_usage", None) if obj is not None else None
            if isinstance(v, (int, float)):
                total += int(v)
                found = True
        return total if found else None


def make_adapter(
    name: str,
    model: Any,
    params: dict | None = None,
    scaler: Any = None,
    scaler_name: str = "none",
) -> ClustererAdapter:
    return ClustererAdapter(
        name=name, model=model, params=params or {},
        scaler=scaler, scaler_name=scaler_name,
    )
