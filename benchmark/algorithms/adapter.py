"""Uniform adapter over any ``river.base.Clusterer``.

The adapter gives every clusterer in ``river.cluster`` the same interface for
the benchmark runner, and optionally chains a **preprocessing step** in front of
the model — usually an online scaler, which strongly affects the resulting
number of clusters and the quality of coordinate-based methods, but equally a
bag-of-words step for the text-based ones. It also reports, best-effort, the
micro-cluster count, macro-cluster count, structure summary and model memory so
efficiency can be compared across algorithms.

"Any clusterer" means the protocol cannot be assumed. River's seven span three
shapes, and the adapter handles all of them rather than assuming the first:

* **partition a sample into a cluster** — CluStream, DenStream, DBSTREAM,
  STREAMKMeans, KMeans, TextClust;
* **cluster something other than samples** — ODAC builds a hierarchy over the
  *variables* and raises ``NotImplementedError`` from ``predict_one`` on
  purpose; there is no per-sample label to score, so it is measured on
  structure and resources;
* **consume something other than coordinates** — TextClust takes term-frequency
  dictionaries and compares them by tf-idf cosine, and needs the stream position
  passed as ``t`` when its real-time fading is enabled.
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import Any


#: Values a clusterer's ``predict_one`` returns that are NOT cluster ids.
#:
#: DenStream's ``_get_closest_cluster_key`` returns ``-1`` when
#: ``p_micro_clusters`` is empty (``denstream.py:193``): there is no cluster to
#: assign to, so it signals that rather than raising. The value then travels out
#: of ``predict_one`` unchanged and, unless something intervenes, straight into
#: ``AdjustedRand`` as though it were a cluster label. A whole stream of them
#: looks like "one big cluster that scores 0" when what happened is "no
#: clustering at all" -- and it is nearly free to compute, so it also tops the
#: throughput table. See RIVER_BUGS.md BUG-6.
#:
#: Keyed by River class name so a workaround is scoped to the model that needs
#: it, never applied globally.
SENTINEL_LABELS: dict[str, tuple] = {
    "DenStream": (-1,),
}


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
        Optional preprocessing step applied to each ``x`` before predict/learn.
        Must implement river's ``learn_one`` + ``transform_one``. Usually a
        scaler, but the same slot holds a bag-of-words step for the text-based
        clusterers. ``None`` => raw.
    scaler_name:
        Human-readable name for reporting (e.g. ``"standard"``).
    scaler_params:
        The constructor params the preprocessor was built with (recorded so a
        run is reproducible: ``standard`` with a rolling window is a different
        preprocessing step from plain ``standard``).
    partitions_samples:
        Whether ``predict_one`` assigns *this sample* to a cluster. ``False``
        for algorithms that cluster something else — ODAC builds a hierarchy
        over the variables — in which case there is no per-sample label to
        score and the run is measured on structure and resources instead.
    """

    name: str
    model: Any
    params: dict = field(default_factory=dict)
    scaler: Any = None
    scaler_name: str = "none"
    scaler_params: dict = field(default_factory=dict)
    partitions_samples: bool = True
    #: How many times ``predict_one`` had to fall back to label 0 because the
    #: model was not initialised yet. Reported so a cold start cannot silently
    #: inflate or deflate the external metrics without anyone noticing.
    n_fallback_predictions: int = 0
    #: How many times the model returned a *sentinel* instead of a cluster id
    #: (RIVER_BUGS.md BUG-6). Always counted; only suppressed when
    #: ``drop_sentinel_predictions`` is on. A run with a high count here did not
    #: cluster badly -- it did not cluster.
    n_sentinel_predictions: int = 0
    #: Opt-in workaround for BUG-6. Off by default, so the measured behaviour is
    #: River's own: the sentinel goes to the metrics as if it were a cluster id,
    #: which is what a River user gets. Turning it on returns ``None`` instead,
    #: and the runner then skips that sample for the external metrics. The two
    #: are different measurements, so the choice has to be explicit and on the
    #: record rather than a silent default.
    drop_sentinel_predictions: bool = False
    #: The most recent point *after* preprocessing — the space the model works in.
    last_x_scaled: dict | None = None

    def __post_init__(self):
        # Some clusterers take the stream position as `t` (TextClust uses it to
        # fade term weights) and fail with a TypeError if it is missing. Decide
        # once, by signature, rather than guessing per sample.
        self._learn_takes_t = self._accepts(self.model.learn_one, "t")
        self._n_seen = 0

    @staticmethod
    def _accepts(method, kwarg: str) -> bool:
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):  # pragma: no cover - C-implemented
            return False
        if kwarg in params:
            return True
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    # -- preprocessing -------------------------------------------------------
    def _scale(self, x: dict) -> dict:
        """Learn-then-transform the point with the preprocessor, if present."""
        if self.scaler is None:
            return x
        self.scaler.learn_one(x)
        return self.scaler.transform_one(x)

    # -- online protocol -----------------------------------------------------
    def predict_one(self, x: dict) -> int | None:
        """Assign ``x`` to a cluster, or return ``None`` if that is not defined.

        Two distinct situations produce no usable label:

        * **cold start** — clusterers such as STREAMKMeans raise before any
          centre exists. Label ``0`` keeps the stream flowing, and the count is
          reported so it cannot distort the metrics unnoticed.
        * **no per-sample partition** — ODAC clusters variables and its
          ``predict_one`` raises ``NotImplementedError`` by design. There is
          nothing to score, so ``None`` is returned and the runner skips
          external quality for the whole run.
        * **sentinel** — DenStream returns ``-1`` when it holds no
          p-micro-clusters, i.e. when it has produced no clustering at all
          (RIVER_BUGS.md BUG-6). It is always counted in
          ``n_sentinel_predictions``; whether it is also suppressed depends on
          ``drop_sentinel_predictions``, which defaults to off so that what is
          measured stays River's own behaviour.
        """
        if not self.partitions_samples:
            return None
        try:
            label = self.model.predict_one(x)
        except NotImplementedError:
            # The registry said this model partitions samples but River says
            # otherwise; trust River and stop scoring it.
            self.partitions_samples = False
            return None
        except (ValueError, ZeroDivisionError, KeyError):
            self.n_fallback_predictions += 1
            return 0
        if label in SENTINEL_LABELS.get(type(self.model).__name__, ()):
            self.n_sentinel_predictions += 1
            if self.drop_sentinel_predictions:
                return None
        return label

    def learn_one(self, x: dict) -> None:
        if self._learn_takes_t:
            self.model.learn_one(x, t=self._n_seen)
        else:
            self.model.learn_one(x)

    def process_one(self, x: dict) -> int | None:
        """Full per-sample step used by the runner: preprocess -> predict -> learn.

        Returns the predicted cluster label, or ``None`` when the algorithm does
        not assign samples to clusters. Prediction happens before learning
        (test-then-train), matching a live streaming deployment.

        The preprocessed point is kept on :attr:`last_x_scaled` because the
        model — and therefore its centers — lives in that space. Internal
        metrics must measure distances in the same space; feeding them the raw
        ``x`` would compare points against centers on a different scale.
        """
        xs = self._scale(x)
        self.last_x_scaled = xs
        y_pred = self.predict_one(xs)
        self.learn_one(xs)
        self._n_seen += 1
        return y_pred

    # -- structure (for algorithms that cluster something other than samples) -
    def structure(self) -> dict | None:
        """Model structure summary, when the clusterer exposes one.

        ODAC reports ``n_clusters`` (variable groups), ``n_active_clusters`` and
        tree ``height``. This is what stands in for a quality score when there
        is no per-sample partition to compare against ground truth.
        """
        summary = getattr(self.model, "summary", None)
        if isinstance(summary, dict):
            return dict(summary)
        return None

    # -- introspection (best effort) ----------------------------------------
    def n_micro_clusters(self) -> int | None:
        """Number of micro-clusters (or active density grids in DStream), if exposed."""
        m = self.model
        for attr in ("micro_clusters", "_micro_clusters", "_grids", "grids"):
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
        """Number of macro clusters over *samples*, if exposed.

        Deliberately ``None`` for a model that clusters something else. ODAC has
        an ``n_clusters`` attribute too, but it counts groups of *variables*;
        printing it in the same column as everyone else's sample-cluster count
        would put two different units side by side and invite the reader to
        compare them. Its real numbers live in :meth:`structure` and are
        reported under their own heading.
        """
        if not self.partitions_samples:
            return None
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

        **Cost warning:** ``_raw_memory_usage`` is a property that walks the
        whole object graph with ``gc.get_referents``, so it is O(model size) —
        for a clusterer holding a 2000-point chunk buffer it costs ~14 ms, i.e.
        two orders of magnitude more than processing a sample. Call it at
        checkpoints only, never once per sample.
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
    scaler_params: dict | None = None,
    partitions_samples: bool = True,
) -> ClustererAdapter:
    return ClustererAdapter(
        name=name, model=model, params=params or {},
        scaler=scaler, scaler_name=scaler_name,
        scaler_params=scaler_params or {},
        partitions_samples=partitions_samples,
    )
