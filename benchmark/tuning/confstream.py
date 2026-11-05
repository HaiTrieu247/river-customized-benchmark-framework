"""The confStream ensemble for one algorithm (Carnein et al., 2020, Sec. 3).

An ensemble of ``e_size`` configurations of the same clusterer runs over the
stream in lock-step, one window at a time. After every window:

1. every member -- active and on test -- is scored on the window it has just
   processed, with the batch silhouette over its own predictions;
2. the members on test (created after the previous window and trained on this
   one alongside the active set) are resolved: each is matched against a
   victim drawn from the active set by inverse fitness and replaces it only if
   it scored higher on data neither had seen; the current best is never the
   victim;
3. the active member with the highest score is the *incumbent*: the
   configuration the ensemble would hand a user right now;
4. ``e_test`` new configurations are drawn: a parent by roulette over the
   active scores (shifted so the worst still has a chance), a child by the
   sampler in :mod:`benchmark.tuning.space`, a fresh model instance for it,
   warm-started on the window just processed.

Two departures from the paper, both deliberate. A child is always a fresh
instance replayed on the last window, whatever its parameters: River models
bake most parameters into their state at construction, and one warm-start rule
for every algorithm is the only one that treats them alike. And a member that
trips a feasibility cap (runtime relative to the fastest member that produced
a partition, memory, micro-cluster count) is scored as if it had produced no
partition, so the ensemble sheds unusable configurations instead of paying for
them. The runtime cap sheds *slow* members; a member that does nothing is the
fastest of all, so it is never the reference, and when members tie on the
undefined score the runtime does not break the tie either: the tie goes to the
previous incumbent, then to the oldest surviving member, so a window on which
nothing in the ensemble works leaves the incumbent where it was.

The ensemble scores; it does not write. The caller decides what a window
report becomes.

``"cmm"`` among ``external_metrics`` adds MOA's CMM (Kremer et al., 2011;
:mod:`benchmark.evaluation.cmm`) to every member's window readings, and
``"cmm_strict"`` the same measure with MOA's merging of connected classes and its
excuse for points inside another class's sphere both turned off (not MOA's
number: on real windows MOA's merges the classes, all six on an INSECTS window,
and then scores every partition 1). The ground-truth side depends on the window
alone and is built once per window; each member is then scored on its own
predictions, a River sentinel (DenStream's ``-1`` before it has clusters)
counting as a point no cluster took. The window's class analysis (how many
classes MOA's merged, how many points its spheres excuse) goes to the report as
``cmm_window``. Like the other external readings CMM never takes part in
selection, and a failure reads None with its message in ``cmm_error`` rather
than stopping the run.
"""

from __future__ import annotations

import copy
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..algorithms.adapter import ClustererAdapter, sentinel_labels
from ..evaluation.cmm import CMMWindow

#: The CMM readings ``external_metrics`` may name, and the options each builds its window with.
CMM_VARIANTS: dict[str, dict] = {
    "cmm": {},
    "cmm_strict": {"enable_class_merge": False, "enable_model_error": False},
}
from ..evaluation.runner import run_once
from . import signals as SIG
from .space import ParamSpace, SamplerState, SpaceContext
from .window import (UNDEFINED_SCORE, WindowBuffer, external_support, silhouette_status, silhouette_support,
                     support_fingerprint, warm_start)


#: The two ways a child configuration can be drawn: ``"evolutionary"`` (the
#: paper's rule -- a parent by roulette, a child by Gaussian perturbation with
#: shrinking sigma) or ``"uniform"`` (a fresh draw over the whole space, no
#: parent, no sigma shrink).
SAMPLERS = ("evolutionary", "uniform")
#: How a tie between members with a partition is broken, after the score: ``"micro_clusters"``
#: (fewer micro-clusters, a deterministic stand-in for cost, then the incumbent, then the
#: oldest) or ``"runtime"`` (the faster, then the oldest: the rule of the runs made before
#: 2026-10-22, kept to compare with them). Measured runtime varies between two runs of one
#: seed, and on the full run 15% of the online regime's windows had a tie whose members read
#: different ARIs on the next window, so ``"runtime"`` does not reproduce a run exactly.
TIE_BREAKS = ("micro_clusters", "runtime")
#: Diagnostic selection by labels (E4), never a deployable tuner: every member is ranked by its
#: ARI on the window it has just processed, for the incumbent, the victims and the roulette
#: alike, so the ensemble is grown by what labels would pick and a pick after window t is read
#: on t+1, as the silhouette's is. A window with fewer than ``min_true_classes`` true classes
#: gives no reading (the primary metric drops it too): every member is undefined there and the
#: incumbent stays. ``ari_oracle`` reads every member that ran, a one-label output included
#: (ARI 0 on a window of several classes); ``ari_oracle_constrained`` also keeps the
#: silhouette's gate, a member without a partition staying undefined, which isolates the signal
#: from that gate. The silhouette is still logged (``sig_silhouette``).
ORACLE_SIGNALS: dict[str, bool] = {"ari_oracle": False, "ari_oracle_constrained": True}
#: What a member's selection score rests on, after the caps (``WindowOutcome.score_status``):
#: ``valid`` (a partition with a silhouette, within every cap, and a reading of the selection
#: signal), ``member_error`` (the model raised), ``infeasible_<cap>`` (memory, micro_clusters,
#: runtime), ``undefined_<reason>`` (the sequential partition has no silhouette,
#: ``window.SILHOUETTE_REASONS``) and ``signal_undefined`` (a partition, but the selection signal
#: read nothing). Every status but ``valid`` scores ``UNDEFINED_SCORE``; the status says why.
SCORE_STATUSES: tuple[str, ...] = (
    "valid", "member_error", "infeasible_memory", "infeasible_micro_clusters", "infeasible_runtime",
    "undefined_insufficient_points", "undefined_single_label", "undefined_one_label_per_point",
    "signal_undefined",
)
#: The order a window's incumbent is named by (``ConfStreamEnsemble.incumbent_key``), level by
#: level: the first level at which the winner and the runner-up differ is ``decided_by``.
TIE_LEVELS: dict[str, tuple[str, ...]] = {
    "micro_clusters": ("score", "usable", "micro_clusters", "incumbent", "member_id"),
    "runtime": ("score", "usable", "runtime", "member_id"),
    "unusable": ("score", "usable", "incumbent", "member_id"),
}
#: Schema of the lifecycle events (``ConfStreamEnsemble.events``): births, challenges, decisions
#: and the freeze, one dict each.
LIFECYCLE_SCHEMA = 1
#: Where a member came from (``birth`` events): one of the explicit starts, an offspring of a start
#: drawn when the ensemble is filled, a child of the evolutionary or the uniform sampler, or a fresh
#: copy of the frozen configuration (``reinit_after_freeze``).
BIRTH_ORIGINS: tuple[str, ...] = ("start", "initial_offspring", "evolutionary", "uniform", "reinit_copy")


@dataclass
class EnsembleConfig:
    e_size: int = 20
    e_test: int = 10
    #: Fading factor: sigma shrinks by ``2 ** -lam`` per generation (Eq. 2).
    lam: float = 0.05
    #: Probability that a child starts again at the initial exploration width.
    p_reset: float = 0.01
    #: Feasibility caps, per window. A member slower than ``runtime_factor``
    #: times the fastest member of the same window that produced a partition
    #: (a defined silhouette), heavier than ``max_memory_mb`` or holding more
    #: than ``max_micro_clusters`` gets the undefined score for that window.
    #: ``runtime_factor=None`` turns the runtime cap off: measured time varies between
    #: two runs of one seed, so a selection that hangs on it is not repeatable. Every
    #: member's ``runtime_ratio`` is recorded either way, so the cap can be replayed.
    runtime_factor: float | None = 30.0
    max_memory_mb: float = 2048.0
    max_micro_clusters: int = 10_000
    external_metrics: tuple[str, ...] = ("ari", "nmi", "purity")
    #: How children are drawn; see ``SAMPLERS``.
    sampler: str = "evolutionary"
    #: Window index at which the ensemble collapses onto its incumbent and
    #: stops spawning children. ``None`` disables freezing.
    freeze_after_window: int | None = None
    #: What members are ranked by (``signals.SELECTABLE``): the silhouette, as in the
    #: paper, or a candidate replacement. It drives the victims, the roulette and the
    #: incumbent alike.
    selection_signal: str = "silhouette"
    #: Readings (``signals.SIGNALS``) computed for every member on every window and
    #: written beside the score, without changing what the ensemble selects.
    log_signals: tuple[str, ...] = ()
    #: After ``freeze_after_window``, keep spawning ``e_test`` children per window, each a
    #: fresh model of the frozen configuration warm-started on the last window, which
    #: compete exactly as the tuner's children do: the parameters stop changing, the
    #: models keep being renewed (the re-initialisation control).
    reinit_after_freeze: bool = False
    #: See ``TIE_BREAKS``.
    tie_break: str = "micro_clusters"
    #: True classes a window needs for the oracle signals to read it (``ORACLE_SIGNALS``).
    min_true_classes: int = 2
    #: How the label-free signals read a model's sentinel labels (DenStream's -1):
    #: ``signals.NOISE_POLICIES``. ``"label"``, the default, reads them as one more cluster, as
    #: the tuner's own silhouette always does; the score the tuner selects by is not changed.
    signal_noise: str = "label"

    def __post_init__(self):
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {self.sampler!r}")
        allowed = SIG.SELECTABLE + tuple(ORACLE_SIGNALS)
        if self.selection_signal not in allowed:
            raise ValueError(f"selection_signal must be one of {allowed}, got {self.selection_signal!r}")
        if self.selection_signal in ORACLE_SIGNALS and "ari" not in self.external_metrics:
            raise ValueError(f"{self.selection_signal} ranks members by their ari: put 'ari' in external_metrics")
        self.log_signals = SIG.check_signals(self.log_signals)
        SIG.check_noise(self.signal_noise)
        if self.tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}, got {self.tie_break!r}")
        if self.reinit_after_freeze and self.freeze_after_window is None:
            raise ValueError("reinit_after_freeze needs freeze_after_window")

    @property
    def signals_needed(self) -> tuple[str, ...]:
        """Every reading a window needs: the logged ones, and the selection signal unless
        it is the silhouette, which the score already is."""
        if self.selection_signal in ORACLE_SIGNALS:
            return tuple(dict.fromkeys(self.log_signals))
        extra = () if self.selection_signal == "silhouette" else (self.selection_signal,)
        return tuple(dict.fromkeys(self.log_signals + extra))


@dataclass
class WindowOutcome:
    """What one member did on one window."""

    member_id: int
    score: float
    defined: bool
    runtime_s: float
    memory_bytes: int | None = None
    n_micro_clusters: int | None = None
    n_macro_clusters: int | None = None
    n_pred_labels: int | None = None
    n_sentinel: int = 0
    n_fallback: int = 0
    window_metrics: dict = field(default_factory=dict)
    n_true_classes: int | None = None
    majority_class_share: float | None = None
    error: str | None = None
    feasible: bool = True
    feasible_reason: str | None = None
    window_index: int | None = None
    params: dict | None = None
    parent_id: int | None = None
    born_window: int | None = None
    is_test: bool = False
    #: The member's per-sample predictions on the window. Kept only for the
    #: member that was incumbent on the previous window, so a caller can score
    #: the configuration a user would actually have been running.
    predictions: list | None = None
    #: ``EnsembleConfig.signals_needed`` read on this window; empty unless asked for.
    signals: dict = field(default_factory=dict)
    #: Seconds spent computing ``signals`` (the window's distance matrices included,
    #: charged to the first member that needed them).
    signals_runtime_s: float = 0.0
    #: Seconds spent on CMM (the window's ground-truth analysis charged to the first member),
    #: and why it read None when it failed.
    cmm_runtime_s: float = 0.0
    cmm_error: str | None = None
    #: The member's runtime over the fastest member of the window with a partition (no error,
    #: a defined silhouette); None for a member without one, or with no such member.
    runtime_ratio: float | None = None
    #: Seconds of the batch silhouette on the member's predictions (not in ``runtime_s``).
    scoring_runtime_s: float = 0.0
    #: The first 16 hex digits of a SHA-256 of the member's predictions on the window, in
    #: order: two runs that agree on it handed out the same labels to the same points.
    pred_sha: str | None = None
    #: Whether the member produced a partition (no error, a defined silhouette), whatever its
    #: selection score says; None means ``defined`` tells. The runtime reference reads it.
    partitioned: bool | None = None
    #: The silhouette of the sequential (predict-before-learn) partition, before any cap or signal
    #: replaces the score: None when undefined; ``silhouette_reason`` says why
    #: (``window.SILHOUETTE_REASONS``), None for a member that raised.
    silhouette_raw: float | None = None
    silhouette_reason: str | None = None
    #: Points the silhouette kept, ``None`` predictions it left out, and the share of the kept
    #: points under the largest label.
    n_points_scored: int | None = None
    n_none: int | None = None
    max_label_share: float | None = None
    #: The points of the window handed to the member, and which of them each reading used, by
    #: position (``window.support_fingerprint``): ``support_sha`` the silhouette's
    #: (``window.silhouette_support``), ``n_points_external`` and ``external_support_sha`` the windowed
    #: external metrics' (``window.external_support``). Records only: nothing reads them back.
    n_points_input: int | None = None
    support_sha: str | None = None
    n_points_external: int | None = None
    external_support_sha: str | None = None
    #: ``SCORE_STATUSES``: what the selection score rests on, set once the caps are applied.
    score_status: str | None = None
    #: Readings of the end-of-window snapshot (the window predicted again by a copy of the model
    #: once it has learned it), computed only when ``stability`` is asked for; diagnostics, never
    #: selected by. Keys ``snap_*``.
    snapshot: dict = field(default_factory=dict)
    #: A diagnostic reading that failed (its message); the member and the selection go on.
    diag_error: str | None = None

    @property
    def memory_mb(self) -> float | None:
        return self.memory_bytes / (1024 * 1024) if self.memory_bytes else None

    def to_dict(self) -> dict:
        return {
            "member_id": self.member_id,
            "window": self.window_index,
            "is_test": self.is_test,
            "parent_id": self.parent_id,
            "born_window": self.born_window,
            "params": self.params,
            "score": round(self.score, 6),
            "silhouette_defined": self.defined,
            #: ``silhouette_defined`` is the selection score's: under an oracle signal it says
            #: whether the ARI read; whether the member produced a partition is this.
            "partitioned": self.partitioned,
            "feasible": self.feasible,
            "feasible_reason": self.feasible_reason,
            "error": self.error,
            "runtime_s": round(self.runtime_s, 6),
            "runtime_ratio": round(self.runtime_ratio, 6) if self.runtime_ratio is not None else None,
            "scoring_runtime_s": round(self.scoring_runtime_s, 6),
            "pred_sha": self.pred_sha,
            "memory_mb": round(self.memory_mb, 4) if self.memory_mb is not None else None,
            "n_micro_clusters": self.n_micro_clusters,
            "n_macro_clusters": self.n_macro_clusters,
            "n_pred_labels": self.n_pred_labels,
            "n_sentinel": self.n_sentinel,
            "n_fallback": self.n_fallback,
            "n_true_classes": self.n_true_classes,
            "majority_class_share": self.majority_class_share,
            "silhouette_raw": round(self.silhouette_raw, 6) if self.silhouette_raw is not None else None,
            "silhouette_reason": self.silhouette_reason,
            "n_points_scored": self.n_points_scored,
            "n_none": self.n_none,
            "n_points_input": self.n_points_input,
            "support_sha": self.support_sha,
            "n_points_external": self.n_points_external,
            "external_support_sha": self.external_support_sha,
            "max_label_share": round(self.max_label_share, 6) if self.max_label_share is not None else None,
            "score_status": self.score_status,
            **{f"{k}_win": v for k, v in self.window_metrics.items()},
            **self.snapshot,
            **({"diag_error": self.diag_error} if self.diag_error is not None else {}),
            **({"cmm_error": self.cmm_error} if self.cmm_error is not None else {}),
            **{f"sig_{k}": (round(v, 6) if v is not None else None) for k, v in self.signals.items()},
        }


@dataclass
class Member:
    id: int
    u: dict
    params: dict
    adapter: ClustererAdapter
    state: SamplerState
    parent_id: int | None
    born_window: int
    score: float = UNDEFINED_SCORE
    last: WindowOutcome | None = None
    error: str | None = None
    #: Test hook: a score to report instead of the measured one, used once.
    outcome_override: float | None = None


@dataclass
class WindowReport:
    window_index: int
    incumbent: WindowOutcome
    lag: WindowOutcome | None
    outcomes: list[WindowOutcome]
    n_active: int
    n_tested: int
    n_accepted: int
    n_discarded: int
    n_spawned: int
    n_reset: int
    frozen: bool = False
    #: MOA's class analysis of the window (``ca_separability``, ``ca_noise``, ``ca_model``),
    #: empty unless CMM was read.
    cmm_window: dict = field(default_factory=dict)
    #: Seconds spent building and warm-starting the next test set (no member's runtime).
    spawn_runtime_s: float = 0.0
    #: How the incumbent was named (``ConfStreamEnsemble.decision_info``).
    decision: dict = field(default_factory=dict)
    #: Children built for the next window whose construction or warm start raised.
    n_birth_errors: int = 0

    @property
    def scoring_runtime_s(self) -> float:
        return float(sum(o.scoring_runtime_s for o in self.outcomes))

    @property
    def ensemble_runtime_s(self) -> float:
        return float(sum(o.runtime_s for o in self.outcomes))

    @property
    def n_feasible(self) -> int:
        return sum(1 for o in self.outcomes if o.feasible)

    @property
    def signals_runtime_s(self) -> float:
        return float(sum(o.signals_runtime_s for o in self.outcomes))

    @property
    def cmm_runtime_s(self) -> float:
        return float(sum(o.cmm_runtime_s for o in self.outcomes))

    @property
    def status_counts(self) -> dict[str, int]:
        """Members of the window per ``SCORE_STATUSES`` entry (only the statuses seen)."""
        out: dict[str, int] = {}
        for o in self.outcomes:
            key = o.score_status or "unknown"
            out[key] = out.get(key, 0) + 1
        return out


class _CMMState:
    """CMM for one window: the ground-truth analysis of every variant built on the first
    call, its cost and any failure charged to the first member scored."""

    def __init__(self, buffer: WindowBuffer, variants):
        self.buffer = buffer
        self.variants = list(variants)
        self.windows: dict[str, CMMWindow | None] = {}
        self.errors: dict[str, str] = {}
        self._built = False

    def _build(self) -> float:
        t0 = time.perf_counter()
        self._built = True
        try:
            keys = sorted(self.buffer.xs[0])
            X = np.asarray([[float(x.get(k, 0.0)) for k in keys] for x in self.buffer.xs], dtype=float)
        except Exception as exc:
            self.errors = {v: f"{type(exc).__name__}: {exc}" for v in self.variants}
            return time.perf_counter() - t0
        for v in self.variants:
            try:
                self.windows[v] = CMMWindow.build(X, list(self.buffer.ys), **CMM_VARIANTS[v])
            except Exception as exc:
                self.errors[v] = f"{type(exc).__name__}: {exc}"
        return time.perf_counter() - t0

    def score_into(self, outcome: WindowOutcome, preds: list, noise_labels: frozenset) -> None:
        cost = 0.0 if self._built else self._build()
        t0 = time.perf_counter()
        errors = dict(self.errors)
        for v in self.variants:
            value, win = None, self.windows.get(v)
            if win is not None:
                try:
                    value = win.score(preds, noise_labels=noise_labels)
                except Exception as exc:
                    errors[v] = f"{type(exc).__name__}: {exc}"
            outcome.window_metrics[v] = value
        outcome.cmm_error = next((f"{v}: {errors[v]}" for v in self.variants if v in errors), None)
        outcome.cmm_runtime_s = cost + time.perf_counter() - t0

    def window_readings(self) -> dict:
        """MOA's class analysis of the window, from its default build (none without it)."""
        win = self.windows.get("cmm")
        if win is None:
            return {}
        out = {}
        for k in ("ca_separability", "ca_noise", "ca_model"):
            v = getattr(win, k, None)
            out[k] = None if v is None or v != v else float(v)
        return out


# --- pure helpers ---------------------------------------------------------------


def prediction_hash(preds) -> str:
    """The identity of a sequence of predictions (``WindowOutcome.pred_sha``)."""
    return hashlib.sha256("\0".join(map(str, preds)).encode()).hexdigest()[:16]


def roulette_weights(scores) -> np.ndarray:
    """Selection probabilities, higher score -> more likely.

    Silhouettes can be negative, so the scores are shifted by ``|min| - min``
    before being used as weights (the paper's rule: ``{-0.5, 0.8, 1}`` becomes
    ``{0.5, 1.8, 2}``). Equal scores, all-undefined included, give a uniform
    draw rather than a division by zero.
    """
    x = np.asarray(scores, dtype=float)
    lo = x.min()
    w = x + (abs(lo) - lo if lo < 0 else 0.0)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(x), 1.0 / len(x))
    return w / total


def victim_weights(scores) -> np.ndarray:
    """Replacement probabilities, lower score -> more likely; the best gets 0."""
    x = np.asarray(scores, dtype=float)
    w = x.max() - x
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(x), 1.0 / len(x))
    return w / total


def apply_feasibility(outcomes: list[WindowOutcome], cfg: EnsembleConfig) -> None:
    """Mark the outcomes that tripped a cap; an infeasible one scores undefined.

    The runtime reference is the fastest member that produced a partition
    (no error, a defined silhouette). A member that buffered the whole window,
    or otherwise predicted one label, did no clustering work: measuring the
    others against it would shed every real configuration. With no defined
    member on the window there is no reference, and the runtime cap is not
    applied; the memory and micro-cluster caps are. ``runtime_factor=None`` never applies it.
    Every member with a partition gets its ``runtime_ratio`` against that reference.
    """
    def has_partition(o: WindowOutcome) -> bool:
        return o.error is None and (o.partitioned if o.partitioned is not None else o.defined)

    fastest = min((o.runtime_s for o in outcomes if has_partition(o)), default=None)
    for o in outcomes:
        if fastest is not None and has_partition(o):
            o.runtime_ratio = o.runtime_s / fastest if fastest > 0 else None
        reason = None
        if o.error is not None:
            reason = "error"
        elif o.memory_mb is not None and o.memory_mb > cfg.max_memory_mb:
            reason = "memory"
        elif o.n_micro_clusters is not None and o.n_micro_clusters > cfg.max_micro_clusters:
            reason = "micro_clusters"
        elif (cfg.runtime_factor is not None and fastest is not None
              and o.runtime_s > cfg.runtime_factor * fastest):
            reason = "runtime"
        if reason is not None:
            o.feasible = False
            o.feasible_reason = reason
            o.score = UNDEFINED_SCORE


def score_status(o: WindowOutcome) -> str:
    """``SCORE_STATUSES``: the first of error, cap, undefined partition, unread signal that holds."""
    if o.error is not None:
        return "member_error"
    if not o.feasible:
        return f"infeasible_{o.feasible_reason}"
    if o.silhouette_reason is not None and o.silhouette_reason != "valid":
        return f"undefined_{o.silhouette_reason}"
    if not o.defined:
        return "signal_undefined"
    return "valid"


class _GlobalRandomState:
    """Python's and NumPy's global generators as they were on entry, restored on exit: a
    diagnostic reading can never move a draw anything else makes from them."""

    def __enter__(self):
        self._py, self._np = random.getstate(), np.random.get_state()
        return self

    def __exit__(self, *exc):
        random.setstate(self._py)
        np.random.set_state(self._np)
        return False


# --- the ensemble ---------------------------------------------------------------


class ConfStreamEnsemble:
    def __init__(
        self,
        space: ParamSpace,
        ctx: SpaceContext,
        make_adapter: Callable[[dict, int], ClustererAdapter],
        start_params: list[dict],
        config: EnsembleConfig,
        rng: np.random.Generator,
        seed: int,
        post_resolve: Callable[[dict], dict] | None = None,
    ):
        """``post_resolve`` rewrites every child's resolved parameters before
        the member records them and its model is built (e.g. the driver caps
        initialisation buffers at the window size); the identity by default.
        The explicit starts are taken as given."""
        if config.e_size < 1:
            raise ValueError(f"e_size must be >= 1, got {config.e_size}")
        if not start_params:
            raise ValueError("at least one starting configuration is needed")
        self.space = space
        self.ctx = ctx
        self.make_adapter = make_adapter
        self.start_params = [dict(p) for p in start_params]
        self.cfg = config
        self.rng = rng
        self.seed = seed
        self.post_resolve = post_resolve
        self.active: list[Member] = []
        self.test: list[Member] = []
        self._next_id = 0
        self._last_incumbent_id: int | None = None
        self.frozen = False
        #: Lifecycle events since the last ``drain_events`` (``LIFECYCLE_SCHEMA``): what a log of
        #: births, challenges, decisions and the freeze is written from. Records only; nothing
        #: reads them back.
        self.events: list[dict] = []

    def drain_events(self) -> list[dict]:
        """The lifecycle events recorded since the last call, which are then forgotten."""
        out, self.events = self.events, []
        return out

    # -- members -----------------------------------------------------------------
    def _resolve(self, u: dict) -> dict:
        """A child's constructor kwargs: the space's reading of ``u``, then ``post_resolve``."""
        params = self.space.resolve(u, self.ctx)
        return self.post_resolve(params) if self.post_resolve is not None else params

    def _new_member(self, u, params, state, parent_id, born_window, buffer, origin: str = "start",
                    reset: bool | None = None) -> Member:
        member = Member(
            id=self._next_id, u=dict(u), params=dict(params), adapter=None,  # type: ignore[arg-type]
            state=state, parent_id=parent_id, born_window=born_window,
        )
        self._next_id += 1
        n_warm = 0
        try:
            member.adapter = self.make_adapter(dict(params), self.seed)
            n_warm = warm_start(member.adapter, buffer)
        except Exception as exc:  # a configuration River cannot run is a fact to record
            member.error = f"{type(exc).__name__}: {exc}"
        self.events.append({
            "event": "birth", "window": born_window, "member_id": member.id, "parent_id": parent_id,
            "origin": origin, "reset": reset, "params": dict(params), "warm_start_samples": n_warm,
            "error": member.error,
        })
        return member

    def find(self, member_id: int) -> Member | None:
        """The live member with this id, active or on test; ``None`` if gone."""
        return next((m for m in self.active + self.test if m.id == member_id), None)

    def initialise(self, buffer: WindowBuffer, window_index: int) -> None:
        """Fill the ensemble: the explicit starts first, then their offspring."""
        starts = self.start_params[: self.cfg.e_size]
        for sp in starts:
            params = dict(sp)
            for k, v in self.space.fixed.items():
                params.setdefault(k, v)
            u = self.space.unit_point(params, self.ctx)
            self.active.append(self._new_member(
                u, params, SamplerState.initial(self.space), None, window_index, buffer, origin="start",
            ))
        while len(self.active) < self.cfg.e_size:
            parent = self.active[int(self.rng.integers(len(starts)))]
            child_u, child_state = parent.state.offspring(parent.u, self.rng, self.cfg.lam)
            params = self._resolve(child_u)
            self.active.append(self._new_member(
                child_u, params, child_state, parent.id, window_index, buffer, origin="initial_offspring",
            ))

    # -- one window ----------------------------------------------------------------
    def _run_member(self, m: Member, buffer: WindowBuffer, window_index: int,
                    window: SIG.WindowData | None = None, cmm: _CMMState | None = None) -> WindowOutcome:
        base = dict(
            member_id=m.id, window_index=window_index, params=dict(m.params),
            parent_id=m.parent_id, born_window=m.born_window, is_test=m in self.test,
        )
        if m.error is not None:
            return WindowOutcome(score=UNDEFINED_SCORE, defined=False, runtime_s=0.0,
                                 error=m.error, feasible=False, feasible_reason="error", **base)
        adapter = m.adapter
        sent0, fb0 = adapter.n_sentinel_predictions, adapter.n_fallback_predictions
        preds: list = []
        try:
            res = run_once(
                adapter, buffer.as_dataset(f"window_{window_index}"),
                n_checkpoints=1, window_size=len(buffer),
                external_metrics=self.cfg.external_metrics, internal_metrics=(),
                predictions=preds,
            )
        except Exception as exc:
            m.error = f"{type(exc).__name__}: {exc}"
            return WindowOutcome(score=UNDEFINED_SCORE, defined=False, runtime_s=0.0,
                                 error=m.error, feasible=False, feasible_reason="error", **base)
        t_score = time.perf_counter()
        sil = silhouette_status(buffer.xs, preds)
        score, defined = sil.score, sil.defined
        scoring_s = time.perf_counter() - t_score
        win = res.windows[0] if res.windows else None
        outcome = WindowOutcome(
            score=score, defined=defined,
            runtime_s=res.efficiency.total_runtime_s,
            memory_bytes=res.efficiency.peak_memory_bytes,
            n_micro_clusters=res.efficiency.max_micro_clusters,
            n_macro_clusters=res.efficiency.final_macro_clusters,
            n_pred_labels=len({p for p in preds if p is not None}),
            n_sentinel=adapter.n_sentinel_predictions - sent0,
            n_fallback=adapter.n_fallback_predictions - fb0,
            window_metrics=dict(win.window_metrics) if win else {},
            n_true_classes=win.n_true_classes if win else None,
            majority_class_share=win.majority_class_share if win else None,
            scoring_runtime_s=scoring_s, pred_sha=prediction_hash(preds), partitioned=defined,
            silhouette_raw=score if defined else None, silhouette_reason=sil.reason,
            n_points_scored=sil.n_points, n_none=sil.n_none, max_label_share=sil.max_label_share,
            **base,
        )
        ext_mask = external_support(buffer.ys, preds)
        outcome.n_points_input = len(preds)
        outcome.support_sha = support_fingerprint(silhouette_support(preds))
        outcome.n_points_external = sum(ext_mask)
        outcome.external_support_sha = support_fingerprint(ext_mask)
        if m.id == self._last_incumbent_id:
            outcome.predictions = preds
        if cmm is not None:
            cmm.score_into(outcome, preds, frozenset(sentinel_labels(adapter.model)))
        if self.cfg.selection_signal in ORACLE_SIGNALS:
            ari = outcome.window_metrics.get("ari")
            readable = ari is not None and (outcome.n_true_classes is None
                                            or outcome.n_true_classes >= self.cfg.min_true_classes)
            if ORACLE_SIGNALS[self.cfg.selection_signal]:
                readable = readable and defined
            outcome.score, outcome.defined = (float(ari), True) if readable else (UNDEFINED_SCORE, False)
        if window is not None:
            t0 = time.perf_counter()
            needed = SIG.per_member_needed(self.cfg.signals_needed)
            noise = {"noise": self.cfg.signal_noise, "noise_labels": tuple(sentinel_labels(adapter.model))}
            selects = self.cfg.selection_signal != "silhouette" and self.cfg.selection_signal in SIG.SIGNALS
            with _GlobalRandomState():
                try:
                    outcome.signals = SIG.member_signals(window, preds, needed, **noise)
                except Exception as exc:
                    if selects:
                        raise
                    outcome.signals = {}
                    outcome.diag_error = f"signals: {type(exc).__name__}: {exc}"
                if "silhouette" in self.cfg.log_signals:
                    outcome.signals["silhouette"] = score if defined else None
                if "stability" in self.cfg.signals_needed:
                    self._snapshot_into(outcome, adapter, buffer, preds, noise, selects)
            outcome.signals_runtime_s = time.perf_counter() - t0
            sel = self.cfg.selection_signal
            if sel != "silhouette" and sel in SIG.PER_MEMBER + SIG.MODEL_SIGNALS:
                value = outcome.signals.get(sel)
                outcome.score = value if value is not None else UNDEFINED_SCORE
                outcome.defined = defined and value is not None
        if self.cfg.selection_signal in ORACLE_SIGNALS:
            # the score is the ARI now; the silhouette it replaced is logged beside the other readings
            outcome.signals["silhouette"] = score if defined else None
        if m.outcome_override is not None:
            outcome.score, outcome.defined = float(m.outcome_override), True
            m.outcome_override = None
        return outcome

    def _snapshot_into(self, outcome: WindowOutcome, adapter: ClustererAdapter, buffer: WindowBuffer,
                       preds: list, noise: dict, selects: bool) -> None:
        """``stability`` and the snapshot's readings, from ONE prediction pass of a copy of the
        model at the end of the window (``_repredict``). The snapshot's silhouette is the same
        classic silhouette as the score, on the snapshot's labels (``snap_silhouette``,
        ``snap_silhouette_defined``, ``snap_silhouette_reason``); ``snap_label_agreement`` is the
        share of points both predictions labelled with the same raw label (a renaming lowers it
        and leaves ``stability``, an ARI, at 1). Never selected by. A failure is recorded in
        ``diag_error``; it is raised only when the tuner selects by ``stability`` itself."""
        try:
            again = self._repredict(adapter, buffer)
            outcome.signals["stability"] = SIG.stability(preds, again, **noise)
        except Exception as exc:
            if selects and self.cfg.selection_signal == "stability":
                raise
            outcome.signals["stability"] = None
            outcome.diag_error = f"stability: {type(exc).__name__}: {exc}"
            outcome.snapshot = {"snap_silhouette": None, "snap_silhouette_defined": None,
                                "snap_silhouette_reason": None, "snap_n_pred_labels": None,
                                "snap_pred_sha": None, "snap_label_agreement": None}
            return
        both = [(a, b) for a, b in zip(preds, again) if a is not None and b is not None]
        snap = {"snap_n_pred_labels": len({b for b in again if b is not None}),
                "snap_pred_sha": prediction_hash(again),
                "snap_label_agreement": (round(sum(a == b for a, b in both) / len(both), 6) if both else None)}
        try:
            st = silhouette_status(buffer.xs, again)
            snap.update(snap_silhouette=round(st.score, 6) if st.defined else None,
                        snap_silhouette_defined=st.defined, snap_silhouette_reason=st.reason)
        except Exception as exc:
            snap.update(snap_silhouette=None, snap_silhouette_defined=None, snap_silhouette_reason="scoring_error")
            outcome.diag_error = f"snapshot silhouette: {type(exc).__name__}: {exc}"
        outcome.snapshot = snap

    @staticmethod
    def _repredict(adapter: ClustererAdapter, buffer: WindowBuffer) -> list:
        """The window predicted again by a copy of a model that has just learned it. A copy,
        because a prediction may change a model (RIVER_BUGS.md BUG-7), and every point is
        handed over as a copy of its own (BUG-3)."""
        probe = copy.deepcopy(adapter)
        out = []
        for x in buffer.xs:
            x = dict(x)
            out.append(probe.predict_one(probe.scaler.transform_one(x) if probe.scaler is not None else x))
        return out

    def replay(self, buffer: WindowBuffer) -> int:
        """Warm-start every active member on ``buffer`` (no scoring)."""
        n = 0
        for m in self.active:
            if m.error is None:
                n = warm_start(m.adapter, buffer)
        return n

    def _rank_aggregate(self, outcomes: list[WindowOutcome], overridden: set[int]) -> None:
        """``rank_agg`` of every member, ranked among the feasible members of the window
        (a member shed by a cap takes no part); the score too when it selects, except for a
        member whose score a test pinned."""
        readings = [o.signals if o.feasible and o.error is None else {} for o in outcomes]
        for o, value in zip(outcomes, SIG.rank_aggregate(readings)):
            o.signals["rank_agg"] = value
            if self.cfg.selection_signal == "rank_agg" and o.member_id not in overridden:
                o.score = value if value is not None else UNDEFINED_SCORE
                o.defined = o.defined and value is not None

    @staticmethod
    def pick_incumbent(
        outcomes: list[WindowOutcome], last_incumbent_id: int | None = None,
        tie_break: str = "micro_clusters",
    ) -> WindowOutcome:
        """Highest score first; a member with a usable partition (a defined
        silhouette, within every cap) beats one without at an equal score.

        Ties between usable members follow ``tie_break`` (``TIE_BREAKS``): by
        default the member with fewer micro-clusters, then ``last_incumbent_id``,
        then the older member; with ``"runtime"``, the faster one, then the older.
        Ties between unusable ones -- all on the undefined score -- go to
        ``last_incumbent_id``, then the older member, never the faster one:
        doing nothing is fastest, and the runtime of a member with no
        partition says nothing about its configuration.
        """
        return min(outcomes, key=lambda o: ConfStreamEnsemble.incumbent_key(o, last_incumbent_id, tie_break))

    @staticmethod
    def incumbent_key(o: WindowOutcome, last_incumbent_id: int | None, tie_break: str = "micro_clusters") -> tuple:
        """The sort key ``pick_incumbent`` minimises (``TIE_LEVELS`` names its levels). A member
        whose micro-cluster count is unknown (None) counts as 0 micro-clusters; ``member_id``
        (the older member) is the last level, so the key is total and the pick deterministic."""
        if o.defined and o.feasible:
            if tie_break == "runtime":
                return (-o.score, 0, o.runtime_s, o.member_id)
            n_micro = o.n_micro_clusters if o.n_micro_clusters is not None else 0
            return (-o.score, 0, n_micro, 0 if o.member_id == last_incumbent_id else 1, o.member_id)
        return (-o.score, 1, 0 if o.member_id == last_incumbent_id else 1, o.member_id, 0)

    @staticmethod
    def decision_info(outcomes: list[WindowOutcome], last_incumbent_id: int | None,
                      tie_break: str = "micro_clusters") -> dict:
        """How ``pick_incumbent`` named its winner: ``decided_by``, the first ``TIE_LEVELS`` level at
        which the winner's key differs from the runner-up's (``only_member`` with one member),
        ``n_top_ties``, the members on the winner's score and usability, and ``tie_break``."""
        keys = sorted((ConfStreamEnsemble.incumbent_key(o, last_incumbent_id, tie_break), o.member_id)
                      for o in outcomes)
        if not keys:
            return {}
        win = keys[0][0]
        usable = win[1] == 0
        levels = TIE_LEVELS[tie_break if usable else "unusable"]
        decided_by = "only_member"
        if len(keys) > 1:
            other = keys[1][0]
            decided_by = next((levels[i] for i in range(min(len(win), len(levels))) if win[i] != other[i]),
                              levels[-1])
        n_ties = sum(1 for k, _ in keys if k[0] == win[0] and k[1] == win[1])
        return {"decided_by": decided_by, "n_top_ties": n_ties, "tie_break": tie_break, "usable": usable}

    def process_window(self, buffer: WindowBuffer, window_index: int) -> WindowReport:
        members = self.active + self.test
        window = SIG.WindowData(buffer.xs) if self.cfg.signals_needed else None
        overridden = {m.id for m in members if m.outcome_override is not None}
        variants = [v for v in CMM_VARIANTS if v in self.cfg.external_metrics]
        cmm = _CMMState(buffer, variants) if variants else None
        outcomes = [self._run_member(m, buffer, window_index, window, cmm) for m in members]
        apply_feasibility(outcomes, self.cfg)
        if "rank_agg" in self.cfg.signals_needed:
            self._rank_aggregate(outcomes, overridden)
        for o in outcomes:
            o.score_status = score_status(o)
        for m, o in zip(members, outcomes):
            m.score, m.last = o.score, o

        lag = next((o for o in outcomes if o.member_id == self._last_incumbent_id), None)

        # Resolve the test set against the active set, one challenger at a time.
        n_tested, n_accepted, n_discarded = len(self.test), 0, 0
        for challenger in self.test:
            event = {"event": "challenge", "window": window_index, "challenger_id": challenger.id,
                     "challenger_score": challenger.score, "victim_id": None, "victim_score": None}
            if len(self.active) < self.cfg.e_size:
                # only after a freeze with renewal: the copies fill the ensemble back up
                self.active.append(challenger)
                n_accepted += 1
                self.events.append({**event, "outcome": "filled"})
                continue
            best = self.pick_incumbent([m.last for m in self.active], self._last_incumbent_id, self.cfg.tie_break)
            candidates = [m for m in self.active if m.id != best.member_id]
            if not candidates:
                n_discarded += 1
                self.events.append({**event, "outcome": "discarded_no_victim"})
                continue
            w = victim_weights([m.score for m in candidates])
            victim = candidates[int(self.rng.choice(len(candidates), p=w))]
            event.update(victim_id=victim.id, victim_score=victim.score)
            if challenger.score > victim.score:
                self.active[self.active.index(victim)] = challenger
                n_accepted += 1
                self.events.append({**event, "outcome": "accepted"})
            else:
                n_discarded += 1
                self.events.append({**event, "outcome": "rejected"})
        self.test = []

        active_outcomes = [m.last for m in self.active]
        previous = self._last_incumbent_id
        incumbent = self.pick_incumbent(active_outcomes, previous, self.cfg.tie_break)
        decision = {**self.decision_info(active_outcomes, previous, self.cfg.tie_break),
                    "incumbent_id": incumbent.member_id, "previous_incumbent_id": previous,
                    "switched": previous is not None and incumbent.member_id != previous,
                    "score": incumbent.score, "score_status": incumbent.score_status,
                    "n_active": len(self.active), "n_evaluated": len(outcomes),
                    "n_evaluated_ok": sum(1 for o in outcomes if o.error is None)}
        self.events.append({"event": "decision", "window": window_index, **decision})
        self._last_incumbent_id = incumbent.member_id

        if (self.cfg.freeze_after_window is not None and not self.frozen
                and window_index >= self.cfg.freeze_after_window):
            self.active = [m for m in self.active if m.id == incumbent.member_id]
            self.frozen = True
            keep = self.active[0]
            self._frozen_member = (dict(keep.u), dict(keep.params), keep.state, keep.id)
            self.events.append({"event": "freeze", "window": window_index, "member_id": keep.id,
                                "params": dict(keep.params), "reinit_after_freeze": self.cfg.reinit_after_freeze})

        # Spawn the next test set from the (possibly refreshed, possibly
        # frozen) active set. A frozen ensemble spawns nothing.
        t_spawn = time.perf_counter()
        n_reset = 0
        n_spawn = 0 if self.frozen and not self.cfg.reinit_after_freeze else self.cfg.e_test
        parent_w = roulette_weights([m.score for m in self.active])
        for _ in range(n_spawn):
            if self.frozen:
                # re-initialisation control: a fresh model of the frozen configuration
                u, params, state, parent_id = self._frozen_member
                self.test.append(self._new_member(u, params, state, parent_id, window_index, buffer,
                                                  origin="reinit_copy"))
                continue
            reset = None
            if self.cfg.sampler == "uniform":
                parent_id = None
                child_u = {p.name: float(self.rng.random()) for p in self.space.params}
                child_state = SamplerState.initial(self.space)
            else:
                parent = self.active[int(self.rng.choice(len(self.active), p=parent_w))]
                parent_id = parent.id
                # A reset child is drawn at the initial width, not just given
                # it for its own offspring (paper, Sec. 3; MOA resets before
                # it samples).
                state = parent.state
                reset = bool(self.rng.random() < self.cfg.p_reset)
                if reset:
                    state = state.reset(self.space)
                    n_reset += 1
                child_u, child_state = state.offspring(parent.u, self.rng, self.cfg.lam)
            params = self._resolve(child_u)
            self.test.append(self._new_member(
                child_u, params, child_state, parent_id, window_index, buffer,
                origin=self.cfg.sampler, reset=reset,
            ))

        return WindowReport(
            window_index=window_index, incumbent=incumbent, lag=lag, outcomes=outcomes,
            n_active=len(self.active), n_tested=n_tested, n_accepted=n_accepted,
            n_discarded=n_discarded, n_spawned=len(self.test), n_reset=n_reset,
            frozen=self.frozen, cmm_window=cmm.window_readings() if cmm is not None else {},
            spawn_runtime_s=(time.perf_counter() - t_spawn) if n_spawn else 0.0,
            decision=decision, n_birth_errors=sum(1 for m in self.test if m.error is not None),
        )
