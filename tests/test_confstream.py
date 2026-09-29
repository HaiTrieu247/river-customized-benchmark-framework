"""The confStream ensemble: selection, replacement, spawning, reporting."""

from __future__ import annotations

import random
import time

import numpy as np
import pytest

from benchmark.algorithms.registry import get_algorithm
from benchmark.tuning.confstream import (
    ConfStreamEnsemble,
    EnsembleConfig,
    WindowOutcome,
    apply_feasibility,
    roulette_weights,
    victim_weights,
)
from benchmark.tuning.space import Param, ParamSpace, SpaceContext
from benchmark.tuning.window import UNDEFINED_SCORE, WindowBuffer

SPACE = ParamSpace(
    "KMeans",
    (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)),
)
CTX = SpaceContext(s=1.0, d=2)
START = {"n_clusters": 2, "halflife": 0.5}


def _make_adapter(params: dict, seed: int):
    return get_algorithm("KMeans", seed=seed, **params)


def _blob_window(h: int, seed: int) -> WindowBuffer:
    rng = random.Random(seed)
    buf = WindowBuffer(h)
    for i in range(h):
        c = i % 2
        buf.push({"a": 5.0 * c + rng.gauss(0, 0.1), "b": rng.gauss(0, 0.1)}, c)
    return buf


def _ensemble(e_size=4, e_test=2, p_reset=0.0, start=None, seed=0, **kw):
    cfg = EnsembleConfig(e_size=e_size, e_test=e_test, lam=0.05, p_reset=p_reset, **kw)
    return ConfStreamEnsemble(
        space=SPACE, ctx=CTX, make_adapter=_make_adapter,
        start_params=start or [START], config=cfg, rng=np.random.default_rng(seed), seed=seed,
    )


# --- pure helpers ---------------------------------------------------------------


def test_roulette_weights_shift_by_abs_min_minus_min_as_in_the_paper():
    w = roulette_weights([-0.5, 0.8, 1.0])
    assert np.allclose(w, np.array([0.5, 1.8, 2.0]) / 4.3)


def test_roulette_weights_are_uniform_when_every_score_is_equal():
    assert np.allclose(roulette_weights([-1.0, -1.0, -1.0]), [1 / 3] * 3)
    assert np.allclose(roulette_weights([0.0, 0.0]), [0.5, 0.5])


def test_roulette_weights_do_not_shift_when_all_scores_are_positive():
    assert np.allclose(roulette_weights([1.0, 3.0]), [0.25, 0.75])


def test_victim_weights_favour_the_worst_and_give_the_best_nothing():
    w = victim_weights([0.2, 0.9, 0.5])
    assert w[1] == 0.0
    assert w[0] > w[2] > 0.0
    assert np.allclose(w.sum(), 1.0)


def test_victim_weights_are_uniform_when_every_score_is_equal():
    assert np.allclose(victim_weights([0.3, 0.3]), [0.5, 0.5])


def _outcome(member_id, runtime, score=0.5, memory_mb=1.0, n_micro=10, error=None, defined=True):
    return WindowOutcome(
        member_id=member_id, score=score if defined else UNDEFINED_SCORE, defined=defined,
        runtime_s=runtime, memory_bytes=int(memory_mb * 1024 * 1024), n_micro_clusters=n_micro,
        error=error,
    )


def test_apply_feasibility_caps_runtime_relative_to_the_fastest_member():
    cfg = EnsembleConfig(runtime_factor=3.0)
    outs = [_outcome(0, 1.0), _outcome(1, 2.9), _outcome(2, 3.1)]
    apply_feasibility(outs, cfg)
    assert [o.feasible for o in outs] == [True, True, False]
    assert outs[2].score == UNDEFINED_SCORE and outs[2].feasible_reason == "runtime"
    assert outs[1].score == 0.5


def test_apply_feasibility_caps_memory_and_micro_clusters():
    cfg = EnsembleConfig(max_memory_mb=10.0, max_micro_clusters=100)
    outs = [_outcome(0, 1.0), _outcome(1, 1.0, memory_mb=11.0), _outcome(2, 1.0, n_micro=101)]
    apply_feasibility(outs, cfg)
    assert [o.feasible for o in outs] == [True, False, False]
    assert outs[1].feasible_reason == "memory" and outs[2].feasible_reason == "micro_clusters"


def test_apply_feasibility_ignores_errored_members_when_finding_the_fastest():
    cfg = EnsembleConfig(runtime_factor=3.0)
    outs = [_outcome(0, 0.01, error="boom"), _outcome(1, 1.0), _outcome(2, 2.0)]
    apply_feasibility(outs, cfg)
    assert outs[0].feasible is False and outs[0].feasible_reason == "error"
    assert outs[1].feasible and outs[2].feasible


def test_apply_feasibility_measures_runtime_against_the_fastest_defined_member():
    # A member that produced no partition (it buffered the whole window, say)
    # is the fastest by doing nothing; it must not be the runtime reference.
    outs = [_outcome(0, 1e-4, defined=False), _outcome(1, 5e-3, score=0.7)]
    apply_feasibility(outs, EnsembleConfig())
    assert outs[1].feasible and outs[1].feasible_reason is None and outs[1].score == 0.7


def test_apply_feasibility_still_sheds_a_slow_defined_member():
    outs = [_outcome(0, 1e-3), _outcome(1, 2e-3), _outcome(2, 1.0)]
    apply_feasibility(outs, EnsembleConfig())
    assert [o.feasible for o in outs] == [True, True, False]
    assert outs[2].feasible_reason == "runtime" and outs[2].score == UNDEFINED_SCORE


def test_apply_feasibility_skips_the_runtime_cap_when_no_member_is_defined():
    cfg = EnsembleConfig(max_memory_mb=10.0)
    outs = [_outcome(0, 1e-4, defined=False), _outcome(1, 1.0, defined=False),
            _outcome(2, 1.0, defined=False, memory_mb=11.0)]
    apply_feasibility(outs, cfg)
    assert outs[0].feasible and outs[1].feasible
    assert outs[2].feasible is False and outs[2].feasible_reason == "memory"


# --- ensemble life cycle ----------------------------------------------------------


def test_initialise_fills_the_ensemble_and_warm_starts_every_member():
    ens = _ensemble(e_size=4, e_test=2)
    buf = _blob_window(40, seed=0)
    ens.initialise(buf, window_index=0)
    assert len(ens.active) == 4 and ens.test == []
    first = ens.active[0]
    assert first.params == {**START} and first.parent_id is None and first.id == 0
    assert all(m.adapter._n_seen == 40 for m in ens.active)
    assert {m.id for m in ens.active} == {0, 1, 2, 3}
    assert all(m.parent_id == 0 for m in ens.active[1:])


def test_initialise_accepts_several_explicit_starting_configurations():
    ens = _ensemble(e_size=3, e_test=1, start=[START, {"n_clusters": 7, "halflife": 0.1}])
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    assert ens.active[1].params == {"n_clusters": 7, "halflife": 0.1}
    assert ens.active[2].parent_id in (0, 1)


def test_process_window_scores_every_member_and_names_an_incumbent():
    ens = _ensemble(e_size=3, e_test=2, start=[START, {"n_clusters": 8, "halflife": 0.5}])
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    # River's KMeans leaves unused centroids idle, so 8 clusters can collapse to
    # the same two labels as 2 clusters; pin member 1 below to make the
    # ordering deterministic. Member 0's score is measured for real.
    ens.active[1].outcome_override = 0.2
    report = ens.process_window(_blob_window(40, seed=1), window_index=1)
    assert report.window_index == 1
    assert len(report.outcomes) == 3  # only the active members: no test set yet
    by_id = {o.member_id: o for o in report.outcomes}
    assert by_id[0].defined and by_id[0].score > 0.9
    assert by_id[0].n_pred_labels == 2
    assert by_id[1].score == 0.2
    assert report.incumbent.member_id == 0
    assert report.incumbent.window_metrics["ari"] == pytest.approx(1.0)
    assert report.lag is None  # no incumbent before the first scored window
    assert len(ens.test) == 2 and all(m.adapter._n_seen == 40 for m in ens.test)


def test_lag_outcome_is_the_previous_incumbent_scored_on_the_current_window():
    ens = _ensemble(e_size=3, e_test=1)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    r1 = ens.process_window(_blob_window(40, seed=1), window_index=1)
    r2 = ens.process_window(_blob_window(40, seed=2), window_index=2)
    assert r2.lag is not None
    assert r2.lag.member_id == r1.incumbent.member_id
    assert r2.lag.window_index == 2


def test_test_members_are_resolved_on_the_next_window_and_the_sizes_hold():
    ens = _ensemble(e_size=4, e_test=2)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    ens.process_window(_blob_window(40, seed=1), window_index=1)
    r2 = ens.process_window(_blob_window(40, seed=2), window_index=2)
    assert len(r2.outcomes) == 6  # 4 active + 2 test were all scored
    assert r2.n_tested == 2 and r2.n_accepted + r2.n_discarded == 2
    assert len(ens.active) == 4 and len(ens.test) == 2


def test_a_test_member_replaces_a_victim_only_when_it_scores_higher():
    ens = _ensemble(e_size=2, e_test=1, start=[START, {"n_clusters": 8, "halflife": 0.5}])
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    ens.process_window(_blob_window(40, seed=1), window_index=1)
    # force the single test member to a hopeless configuration
    ens.test[0].outcome_override = UNDEFINED_SCORE
    r2 = ens.process_window(_blob_window(40, seed=2), window_index=2)
    assert r2.n_discarded == 1 and r2.n_accepted == 0
    assert {m.id for m in ens.active} == {0, 1}


def test_the_incumbent_is_never_the_victim():
    # Two active members, one clearly better. Every replacement round must pick
    # the other one as the victim, whatever the roulette draws.
    for seed in range(5):
        ens = _ensemble(e_size=2, e_test=1, seed=seed,
                        start=[START, {"n_clusters": 8, "halflife": 0.5}])
        ens.initialise(_blob_window(40, seed=0), window_index=0)
        ens.active[1].outcome_override = 0.2
        ens.process_window(_blob_window(40, seed=1), window_index=1)
        ens.active[1].outcome_override = 0.2
        ens.test[0].outcome_override = 0.5  # beats member 1, not member 0
        ens.process_window(_blob_window(40, seed=2), window_index=2)
        assert 0 in {m.id for m in ens.active}
        assert 1 not in {m.id for m in ens.active}


def test_reset_probability_one_gives_every_child_the_initial_width():
    ens = _ensemble(e_size=2, e_test=3, p_reset=1.0)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    r1 = ens.process_window(_blob_window(40, seed=1), window_index=1)
    assert r1.n_reset == 3
    assert all(m.state.sigma == {"n_clusters": 0.5, "halflife": 0.5} for m in ens.test)


def test_children_shrink_sigma_when_not_reset():
    ens = _ensemble(e_size=2, e_test=2, p_reset=0.0)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    ens.process_window(_blob_window(40, seed=1), window_index=1)
    parent = {m.id: m for m in ens.active}
    for child in ens.test:
        assert child.state.sigma["halflife"] == pytest.approx(
            parent[child.parent_id].state.sigma["halflife"] * 2 ** -0.05
        )
        assert child.born_window == 1 and child.params["n_clusters"] in range(2, 9)


def test_baseline_arm_with_a_single_member_and_no_offspring():
    ens = _ensemble(e_size=1, e_test=0)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    r = ens.process_window(_blob_window(40, seed=1), window_index=1)
    assert r.incumbent.member_id == 0 and len(ens.test) == 0 and len(r.outcomes) == 1


def test_a_member_that_raises_is_scored_undefined_and_the_ensemble_goes_on():
    class Exploding:
        def __init__(self):
            self.calls = 0

        def __call__(self, params, seed):
            self.calls += 1
            adapter = get_algorithm("KMeans", seed=seed, **params)
            if self.calls == 2:  # the second member built will blow up on learn
                def boom(x, **_):
                    raise RuntimeError("boom")
                adapter.model.learn_one = boom
            return adapter

    cfg = EnsembleConfig(e_size=2, e_test=0, lam=0.05, p_reset=0.0)
    ens = ConfStreamEnsemble(
        space=SPACE, ctx=CTX, make_adapter=Exploding(), start_params=[START, START],
        config=cfg, rng=np.random.default_rng(0), seed=0,
    )
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    r = ens.process_window(_blob_window(40, seed=1), window_index=1)
    by_id = {o.member_id: o for o in r.outcomes}
    assert by_id[1].error is not None and by_id[1].score == UNDEFINED_SCORE
    assert by_id[1].feasible is False
    assert r.incumbent.member_id == 0


def test_replay_trains_every_active_member_on_another_buffer():
    ens = _ensemble(e_size=3, e_test=1)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    n = ens.replay(_blob_window(25, seed=5))
    assert n == 25
    assert all(m.adapter._n_seen == 65 for m in ens.active)


def test_process_window_keeps_the_previous_incumbents_predictions():
    ens = _ensemble(e_size=2, e_test=1)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    r1 = ens.process_window(_blob_window(40, seed=1), window_index=1)
    assert all(o.predictions is None for o in r1.outcomes)
    r2 = ens.process_window(_blob_window(40, seed=2), window_index=2)
    assert r2.lag is not None and len(r2.lag.predictions) == 40
    others = [o for o in r2.outcomes if o.member_id != r2.lag.member_id]
    assert all(o.predictions is None for o in others)


def test_incumbent_tie_breaks_on_runtime_then_id():
    outs = [_outcome(3, 2.0, score=0.7), _outcome(1, 1.0, score=0.7), _outcome(2, 1.0, score=0.7)]
    ens = _ensemble(e_size=1, e_test=0)
    assert ens.pick_incumbent(outs).member_id == 1


def test_undefined_ties_keep_the_previous_incumbent_not_the_fastest():
    # Doing nothing is fastest: among members with no usable partition the
    # runtime says nothing about quality, so the incumbent stays put.
    outs = [_outcome(0, 3.0, defined=False), _outcome(1, 2.0, defined=False),
            _outcome(2, 1e-4, defined=False)]
    ens = _ensemble(e_size=1, e_test=0)
    assert ens.pick_incumbent(outs, last_incumbent_id=1).member_id == 1


def test_undefined_ties_without_a_previous_incumbent_go_to_the_oldest_member():
    outs = [_outcome(2, 1e-4, defined=False), _outcome(0, 3.0, defined=False),
            _outcome(1, 2.0, defined=False)]
    ens = _ensemble(e_size=1, e_test=0)
    assert ens.pick_incumbent(outs).member_id == 0
    assert ens.pick_incumbent(outs, last_incumbent_id=None).member_id == 0
    # a previous incumbent that is no longer among the outcomes changes nothing
    assert ens.pick_incumbent(outs, last_incumbent_id=7).member_id == 0


def test_a_defined_member_beats_an_undefined_one_whatever_the_runtime():
    outs = [_outcome(0, 1e-4, defined=False), _outcome(1, 5.0, score=-0.2)]
    ens = _ensemble(e_size=1, e_test=0)
    assert ens.pick_incumbent(outs).member_id == 1
    assert ens.pick_incumbent(outs, last_incumbent_id=0).member_id == 1
    # even at the floor: a real silhouette of -1 still outranks no partition
    outs = [_outcome(0, 1e-4, defined=False), _outcome(1, 5.0, score=UNDEFINED_SCORE)]
    assert ens.pick_incumbent(outs, last_incumbent_id=0).member_id == 1


def test_a_capped_member_ties_with_the_undefined_ones_not_with_the_defined_ones():
    # An infeasible member carries the undefined score; the runtime that got it
    # capped must not decide the tie either.
    capped = _outcome(1, 9.0, score=0.8)
    capped.feasible, capped.feasible_reason, capped.score = False, "runtime", UNDEFINED_SCORE
    outs = [_outcome(0, 1e-4, defined=False), capped]
    ens = _ensemble(e_size=1, e_test=0)
    assert ens.pick_incumbent(outs, last_incumbent_id=1).member_id == 1
    assert ens.pick_incumbent(outs).member_id == 0


def test_process_window_keeps_the_incumbent_when_every_member_is_undefined():
    class SlowThird:
        """Member 2 is built to be the slowest, so a runtime tie-break never picks it."""

        def __init__(self):
            self.calls = 0

        def __call__(self, params, seed):
            self.calls += 1
            adapter = get_algorithm("KMeans", seed=seed, **params)
            if self.calls == 3:
                inner = adapter.model.learn_one

                def slow(x, **kw):
                    time.sleep(0.002)
                    return inner(x, **kw)
                adapter.model.learn_one = slow
            return adapter

    cfg = EnsembleConfig(e_size=3, e_test=0, lam=0.05, p_reset=0.0)
    ens = ConfStreamEnsemble(
        space=SPACE, ctx=CTX, make_adapter=SlowThird(), start_params=[START, START, START],
        config=cfg, rng=np.random.default_rng(0), seed=0,
    )
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    ens._last_incumbent_id = 2
    # a window of identical points: no silhouette is defined for anyone
    flat = WindowBuffer(40)
    for i in range(40):
        flat.push({"a": 1.0, "b": 1.0}, i % 2)
    r = ens.process_window(flat, window_index=1)
    assert all(not o.defined for o in r.outcomes)
    assert all(o.feasible for o in r.outcomes)   # no defined reference, no runtime cap
    assert r.incumbent.member_id == 2


def test_post_resolve_is_applied_to_every_child_but_not_to_the_starts():
    seen = []

    def cap(params):
        seen.append(dict(params))
        return {**params, "n_clusters": 2}

    cfg = EnsembleConfig(e_size=3, e_test=2, lam=0.05, p_reset=0.0)
    ens = ConfStreamEnsemble(
        space=SPACE, ctx=CTX, make_adapter=_make_adapter,
        start_params=[{"n_clusters": 8, "halflife": 0.5}], config=cfg,
        rng=np.random.default_rng(0), seed=0, post_resolve=cap,
    )
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    assert ens.active[0].params["n_clusters"] == 8           # an explicit start is kept
    assert ens.active[0].adapter.model.n_clusters == 8
    ens.process_window(_blob_window(40, seed=1), window_index=1)
    children = ens.active[1:] + ens.test
    assert len(children) == 4 and len(seen) == 4
    for m in children:
        assert m.params["n_clusters"] == 2                   # recorded as clamped
        assert m.adapter.model.n_clusters == 2               # and built as clamped


def test_post_resolve_defaults_to_the_identity():
    ens = _ensemble(e_size=2, e_test=0)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    child = ens.active[1]
    assert child.params == SPACE.resolve(child.u, CTX)


# --- ensemble regimes: uniform sampler and freeze --------------------------------


def test_uniform_sampler_spawns_parentless_children_at_initial_width():
    ens = _ensemble(e_size=3, e_test=2, sampler="uniform")
    ens.initialise(_blob_window(60, 0), 0)
    rep = ens.process_window(_blob_window(60, 1), 1)
    assert rep.n_spawned == 2
    assert all(m.parent_id is None for m in ens.test)
    assert all(m.state.sigma["n_clusters"] == 0.5 for m in ens.test)


def test_unknown_sampler_is_rejected():
    with pytest.raises(ValueError):
        _ensemble(sampler="genetic")


def test_freeze_collapses_onto_the_incumbent_and_stops_spawning():
    ens = _ensemble(e_size=3, e_test=2, freeze_after_window=2)
    ens.initialise(_blob_window(60, 0), 0)
    r1 = ens.process_window(_blob_window(60, 1), 1)
    assert not r1.frozen and r1.n_spawned == 2
    r2 = ens.process_window(_blob_window(60, 2), 2)
    assert r2.frozen and r2.n_spawned == 0 and len(ens.active) == 1
    assert ens.active[0].id == r2.incumbent.member_id
    r3 = ens.process_window(_blob_window(60, 3), 3)
    assert r3.lag is not None and r3.lag.member_id == r3.incumbent.member_id


def test_find_returns_the_live_member():
    ens = _ensemble(); ens.initialise(_blob_window(60, 0), 0)
    assert ens.find(ens.active[0].id) is ens.active[0] and ens.find(999) is None
