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
trips a feasibility cap (runtime relative to the fastest member, memory,
micro-cluster count) is scored as if it had produced no partition, so the
ensemble sheds unusable configurations instead of paying for them.

The ensemble scores; it does not write. The caller decides what a window
report becomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..algorithms.adapter import ClustererAdapter
from ..evaluation.runner import run_once
from .space import ParamSpace, SamplerState, SpaceContext
from .window import UNDEFINED_SCORE, WindowBuffer, batch_silhouette, warm_start


@dataclass
class EnsembleConfig:
    e_size: int = 20
    e_test: int = 10
    #: Fading factor: sigma shrinks by ``2 ** -lam`` per generation (Eq. 2).
    lam: float = 0.05
    #: Probability that a child starts again at the initial exploration width.
    p_reset: float = 0.01
    #: Feasibility caps, per window. A member slower than ``runtime_factor``
    #: times the fastest member of the same window, heavier than
    #: ``max_memory_mb`` or holding more than ``max_micro_clusters`` gets the
    #: undefined score for that window.
    runtime_factor: float = 30.0
    max_memory_mb: float = 2048.0
    max_micro_clusters: int = 10_000
    external_metrics: tuple[str, ...] = ("ari", "nmi", "purity")


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
            "feasible": self.feasible,
            "feasible_reason": self.feasible_reason,
            "error": self.error,
            "runtime_s": round(self.runtime_s, 6),
            "memory_mb": round(self.memory_mb, 4) if self.memory_mb is not None else None,
            "n_micro_clusters": self.n_micro_clusters,
            "n_macro_clusters": self.n_macro_clusters,
            "n_pred_labels": self.n_pred_labels,
            "n_sentinel": self.n_sentinel,
            "n_fallback": self.n_fallback,
            "n_true_classes": self.n_true_classes,
            "majority_class_share": self.majority_class_share,
            **{f"{k}_win": v for k, v in self.window_metrics.items()},
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

    @property
    def ensemble_runtime_s(self) -> float:
        return float(sum(o.runtime_s for o in self.outcomes))

    @property
    def n_feasible(self) -> int:
        return sum(1 for o in self.outcomes if o.feasible)


# --- pure helpers ---------------------------------------------------------------


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
    """Mark the outcomes that tripped a cap; an infeasible one scores undefined."""
    clean = [o for o in outcomes if o.error is None]
    fastest = min((o.runtime_s for o in clean), default=None)
    for o in outcomes:
        reason = None
        if o.error is not None:
            reason = "error"
        elif o.memory_mb is not None and o.memory_mb > cfg.max_memory_mb:
            reason = "memory"
        elif o.n_micro_clusters is not None and o.n_micro_clusters > cfg.max_micro_clusters:
            reason = "micro_clusters"
        elif fastest is not None and o.runtime_s > cfg.runtime_factor * fastest:
            reason = "runtime"
        if reason is not None:
            o.feasible = False
            o.feasible_reason = reason
            o.score = UNDEFINED_SCORE


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
    ):
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
        self.active: list[Member] = []
        self.test: list[Member] = []
        self._next_id = 0
        self._last_incumbent_id: int | None = None

    # -- members -----------------------------------------------------------------
    def _new_member(self, u, params, state, parent_id, born_window, buffer) -> Member:
        member = Member(
            id=self._next_id, u=dict(u), params=dict(params), adapter=None,  # type: ignore[arg-type]
            state=state, parent_id=parent_id, born_window=born_window,
        )
        self._next_id += 1
        try:
            member.adapter = self.make_adapter(dict(params), self.seed)
            warm_start(member.adapter, buffer)
        except Exception as exc:  # a configuration River cannot run is a fact to record
            member.error = f"{type(exc).__name__}: {exc}"
        return member

    def initialise(self, buffer: WindowBuffer, window_index: int) -> None:
        """Fill the ensemble: the explicit starts first, then their offspring."""
        starts = self.start_params[: self.cfg.e_size]
        for sp in starts:
            params = dict(sp)
            for k, v in self.space.fixed.items():
                params.setdefault(k, v)
            u = self.space.unit_point(params, self.ctx)
            self.active.append(self._new_member(
                u, params, SamplerState.initial(self.space), None, window_index, buffer,
            ))
        while len(self.active) < self.cfg.e_size:
            parent = self.active[int(self.rng.integers(len(starts)))]
            child_u, child_state = parent.state.offspring(parent.u, self.rng, self.cfg.lam)
            params = self.space.resolve(child_u, self.ctx)
            self.active.append(self._new_member(
                child_u, params, child_state, parent.id, window_index, buffer,
            ))

    # -- one window ----------------------------------------------------------------
    def _run_member(self, m: Member, buffer: WindowBuffer, window_index: int) -> WindowOutcome:
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
        score, defined = batch_silhouette(buffer.xs, preds)
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
            **base,
        )
        if m.id == self._last_incumbent_id:
            outcome.predictions = preds
        if m.outcome_override is not None:
            outcome.score, outcome.defined = float(m.outcome_override), True
            m.outcome_override = None
        return outcome

    def replay(self, buffer: WindowBuffer) -> int:
        """Warm-start every active member on ``buffer`` (no scoring)."""
        n = 0
        for m in self.active:
            if m.error is None:
                n = warm_start(m.adapter, buffer)
        return n

    @staticmethod
    def pick_incumbent(outcomes: list[WindowOutcome]) -> WindowOutcome:
        """Highest score; ties go to the faster member, then the older one."""
        return min(outcomes, key=lambda o: (-o.score, o.runtime_s, o.member_id))

    def process_window(self, buffer: WindowBuffer, window_index: int) -> WindowReport:
        members = self.active + self.test
        outcomes = [self._run_member(m, buffer, window_index) for m in members]
        apply_feasibility(outcomes, self.cfg)
        for m, o in zip(members, outcomes):
            m.score, m.last = o.score, o

        lag = next((o for o in outcomes if o.member_id == self._last_incumbent_id), None)

        # Resolve the test set against the active set, one challenger at a time.
        n_tested, n_accepted, n_discarded = len(self.test), 0, 0
        for challenger in self.test:
            best = self.pick_incumbent([m.last for m in self.active])
            candidates = [m for m in self.active if m.id != best.member_id]
            if not candidates:
                n_discarded += 1
                continue
            w = victim_weights([m.score for m in candidates])
            victim = candidates[int(self.rng.choice(len(candidates), p=w))]
            if challenger.score > victim.score:
                self.active[self.active.index(victim)] = challenger
                n_accepted += 1
            else:
                n_discarded += 1
        self.test = []

        incumbent = self.pick_incumbent([m.last for m in self.active])
        self._last_incumbent_id = incumbent.member_id

        # Spawn the next test set from the (possibly refreshed) active set.
        n_reset = 0
        parent_w = roulette_weights([m.score for m in self.active])
        for _ in range(self.cfg.e_test):
            parent = self.active[int(self.rng.choice(len(self.active), p=parent_w))]
            child_u, child_state = parent.state.offspring(parent.u, self.rng, self.cfg.lam)
            if self.rng.random() < self.cfg.p_reset:
                child_state = child_state.reset(self.space)
                n_reset += 1
            params = self.space.resolve(child_u, self.ctx)
            self.test.append(self._new_member(
                child_u, params, child_state, parent.id, window_index, buffer,
            ))

        return WindowReport(
            window_index=window_index, incumbent=incumbent, lag=lag, outcomes=outcomes,
            n_active=len(self.active), n_tested=n_tested, n_accepted=n_accepted,
            n_discarded=n_discarded, n_spawned=len(self.test), n_reset=n_reset,
        )
