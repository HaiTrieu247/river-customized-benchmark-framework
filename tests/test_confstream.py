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


def _narrow_every_parent(ens, sigma=1e-12):
    for m in ens.active:
        m.state.sigma = {name: sigma for name in m.state.sigma}


def test_reset_probability_one_draws_every_child_at_the_initial_width():
    # Parents narrowed to nothing: a child drawn at its parent's width lands on
    # the parent. A reset child is drawn at the initial width (paper, Sec. 3;
    # MOA's NumericalParameter resets before it samples), so it moves away, and
    # its own width then shrinks like any child's.
    ens = _ensemble(e_size=2, e_test=3, p_reset=1.0)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    _narrow_every_parent(ens)
    r1 = ens.process_window(_blob_window(40, seed=1), window_index=1)
    parent = {m.id: m for m in ens.active}
    assert r1.n_reset == 3
    for child in ens.test:
        assert abs(child.u["halflife"] - parent[child.parent_id].u["halflife"]) > 1e-6
        assert child.state.sigma == pytest.approx({"n_clusters": 0.5 * 2 ** -0.05,
                                                   "halflife": 0.5 * 2 ** -0.05})


def test_a_child_that_is_not_reset_is_drawn_at_its_parents_width():
    ens = _ensemble(e_size=2, e_test=3, p_reset=0.0)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    _narrow_every_parent(ens)
    r1 = ens.process_window(_blob_window(40, seed=1), window_index=1)
    parent = {m.id: m for m in ens.active}
    assert r1.n_reset == 0
    for child in ens.test:
        assert child.u["halflife"] == pytest.approx(parent[child.parent_id].u["halflife"], abs=1e-9)


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


def test_runtime_tie_break_is_kept_for_comparisons_with_earlier_runs():
    outs = [_outcome(3, 2.0, score=0.7), _outcome(1, 1.0, score=0.7), _outcome(2, 1.0, score=0.7)]
    ens = _ensemble(e_size=1, e_test=0)
    assert ens.pick_incumbent(outs, tie_break="runtime").member_id == 1


def test_incumbent_ties_go_to_fewer_micro_clusters_then_the_incumbent_then_the_oldest():
    # the measured runtime varies between two runs of the same seed; the tie-break must not
    outs = [_outcome(3, 0.1, score=0.7, n_micro=5), _outcome(1, 9.0, score=0.7, n_micro=8),
            _outcome(2, 5.0, score=0.7, n_micro=5)]
    assert ConfStreamEnsemble.pick_incumbent(outs).member_id == 2
    assert ConfStreamEnsemble.pick_incumbent(outs, last_incumbent_id=3).member_id == 3
    assert ConfStreamEnsemble.pick_incumbent(outs, last_incumbent_id=1).member_id == 2


def test_the_tie_break_is_a_setting_and_defaults_to_the_deterministic_one():
    assert EnsembleConfig().tie_break == "micro_clusters"
    with pytest.raises(ValueError, match="tie_break"):
        EnsembleConfig(tie_break="coin")


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


# --- other selection signals ------------------------------------------------------------


def test_logged_signals_are_read_for_every_member_and_written_in_the_member_row():
    ens = _ensemble(e_size=3, e_test=1, log_signals=("silhouette_std", "dbcv", "rank_agg"))
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    ens.process_window(_blob_window(40, seed=1), window_index=1)
    report = ens.process_window(_blob_window(40, seed=2), window_index=2)
    for o in report.outcomes:
        assert set(o.signals) >= {"silhouette_std", "dbcv", "rank_agg"}
        row = o.to_dict()
        assert "sig_silhouette_std" in row and "sig_rank_agg" in row
    # logging a signal never changes what the ensemble selects: a twin without logging,
    # on the same seed, names the same incumbents with the same scores
    twin = _ensemble(e_size=3, e_test=1)
    twin.initialise(_blob_window(40, seed=0), window_index=0)
    twin.process_window(_blob_window(40, seed=1), window_index=1)
    twin_report = twin.process_window(_blob_window(40, seed=2), window_index=2)
    assert twin_report.incumbent.member_id == report.incumbent.member_id
    assert [o.score for o in twin_report.outcomes] == pytest.approx([o.score for o in report.outcomes])


def test_without_logged_signals_no_reading_is_computed():
    ens = _ensemble(e_size=2, e_test=1)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    report = ens.process_window(_blob_window(40, seed=1), window_index=1)
    assert all(o.signals == {} for o in report.outcomes)
    assert not any(k.startswith("sig_") for k in report.outcomes[0].to_dict())


@pytest.mark.parametrize("signal", ["silhouette_std", "dbcv", "rank_agg"])
def test_an_ensemble_selects_by_the_signal_it_is_given(signal):
    ens = _ensemble(e_size=4, e_test=2, selection_signal=signal,
                    start=[START, {"n_clusters": 5, "halflife": 0.3}])
    ens.initialise(_blob_window(60, seed=0), window_index=0)
    ens.process_window(_blob_window(60, seed=1), window_index=1)
    report = ens.process_window(_blob_window(60, seed=2), window_index=2)
    for o in report.outcomes:
        if o.feasible and o.defined:
            assert o.score == pytest.approx(o.signals[signal])
        else:
            assert o.score == UNDEFINED_SCORE
    active = {m.id for m in ens.active}
    best = max((o for o in report.outcomes if o.member_id in active), key=lambda o: o.score)
    assert report.incumbent.score == pytest.approx(best.score)


def test_a_signal_that_cannot_select_is_refused():
    with pytest.raises(ValueError, match="selection_signal"):
        EnsembleConfig(selection_signal="davies_bouldin_std")
    with pytest.raises(ValueError, match="unknown signal"):
        EnsembleConfig(log_signals=("entropy",))


def test_stability_is_read_on_a_copy_after_the_member_learned_the_window():
    ens = _ensemble(e_size=3, e_test=1, log_signals=("stability",))
    ens.initialise(_blob_window(60, seed=0), window_index=0)
    ens.process_window(_blob_window(60, seed=1), window_index=1)
    report = ens.process_window(_blob_window(60, seed=2), window_index=2)
    defined = [o for o in report.outcomes if o.defined]
    assert defined and all(o.signals["stability"] > 0.9 for o in defined)
    assert all(o.signals_runtime_s > 0 for o in defined)


def test_reinit_after_freeze_keeps_spawning_fresh_copies_of_the_frozen_configuration():
    ens = _ensemble(e_size=3, e_test=2, freeze_after_window=1, reinit_after_freeze=True)
    ens.initialise(_blob_window(40, seed=0), window_index=0)
    r1 = ens.process_window(_blob_window(40, seed=1), window_index=1)
    frozen = r1.incumbent.params
    assert r1.frozen and len(ens.active) == 1 and r1.n_spawned == 2
    assert all(m.params == frozen and m.adapter._n_seen == 40 for m in ens.test)
    for w in range(2, 5):
        r = ens.process_window(_blob_window(40, seed=w), window_index=w)
    assert len(ens.active) == 3                       # the copies fill the ensemble back up
    assert all(m.params == frozen for m in ens.active + ens.test)
    assert r.n_spawned == 2 and r.n_tested == 2


def test_reinit_needs_a_freezing_point():
    with pytest.raises(ValueError, match="freeze_after_window"):
        EnsembleConfig(reinit_after_freeze=True)


# --- CMM among the external readings -----------------------------------------------------

def test_cmm_is_read_for_every_member_against_one_ground_truth_analysis_per_window(monkeypatch):
    from benchmark.evaluation.cmm import CMMWindow
    from benchmark.tuning import confstream as CS

    scored = []
    real = CS._CMMState.score_into

    def spy(self, outcome, preds, noise_labels):
        scored.append(list(preds))
        return real(self, outcome, preds, noise_labels)

    monkeypatch.setattr(CS._CMMState, "score_into", spy)
    ens = _ensemble(external_metrics=("ari", "cmm", "cmm_strict"))
    ens.initialise(_blob_window(60, 0), window_index=0)
    buf = _blob_window(60, 1)
    rep = ens.process_window(buf, 1)
    keys = sorted(buf.xs[0])
    X = np.array([[x[k] for k in keys] for x in buf.xs])
    for o in rep.outcomes:
        assert set(o.window_metrics) == {"ari", "cmm", "cmm_strict"} and o.cmm_error is None
    assert len(scored) == len(rep.outcomes)
    moa = CMMWindow.build(X, buf.ys)
    strict = CMMWindow.build(X, buf.ys, enable_class_merge=False, enable_model_error=False)
    for o, preds in zip(rep.outcomes, scored):                  # the window's test-then-train predictions
        assert o.window_metrics["cmm"] == pytest.approx(moa.score(preds))
        assert o.window_metrics["cmm_strict"] == pytest.approx(strict.score(preds))
    built = sum(1 for o in rep.outcomes if o.cmm_runtime_s > 0)
    assert built == len(rep.outcomes)                                   # every score timed...
    assert max(o.cmm_runtime_s for o in rep.outcomes) == rep.outcomes[0].cmm_runtime_s  # ...the build once
    assert set(rep.cmm_window) == {"ca_separability", "ca_noise", "ca_model"}
    assert rep.cmm_runtime_s == pytest.approx(sum(o.cmm_runtime_s for o in rep.outcomes))


def test_without_cmm_in_the_external_metrics_nothing_is_read_or_timed():
    ens = _ensemble(external_metrics=("ari",))
    ens.initialise(_blob_window(60, 0), window_index=0)
    rep = ens.process_window(_blob_window(60, 1), 1)
    assert all("cmm" not in o.window_metrics and o.cmm_runtime_s == 0.0 for o in rep.outcomes)
    assert rep.cmm_window == {}


def test_a_cmm_failure_reads_none_and_says_why_without_stopping_the_window(monkeypatch):
    from benchmark.evaluation import cmm as CMM

    def boom(*a, **k):
        raise ValueError("no sphere")

    monkeypatch.setattr(CMM.CMMWindow, "build", classmethod(lambda cls, *a, **k: boom()))
    ens = _ensemble(external_metrics=("ari", "cmm"))
    ens.initialise(_blob_window(60, 0), window_index=0)
    rep = ens.process_window(_blob_window(60, 1), 1)
    assert all(o.window_metrics["cmm"] is None and "no sphere" in o.cmm_error for o in rep.outcomes)
    assert all(o.window_metrics["ari"] is not None for o in rep.outcomes)


def test_a_river_sentinel_counts_as_a_point_no_cluster_took(monkeypatch):
    from benchmark.tuning import confstream as CS

    seen = []
    real = CS._CMMState.score_into

    def spy(self, outcome, preds, noise_labels):
        seen.append(noise_labels)
        return real(self, outcome, preds, noise_labels)

    monkeypatch.setattr(CS._CMMState, "score_into", spy)
    monkeypatch.setattr(CS, "sentinel_labels", lambda model: (-1,))
    ens = _ensemble(external_metrics=("cmm",))
    ens.initialise(_blob_window(60, 0), window_index=0)
    ens.process_window(_blob_window(60, 1), 1)
    assert seen and all(n == frozenset({-1}) for n in seen)


# --- repeatability: the runtime gate, the partition's identity, the time split ------------------

def test_without_a_runtime_factor_no_member_is_shed_for_its_runtime():
    cfg = EnsembleConfig(runtime_factor=None, max_memory_mb=10.0)
    outs = [_outcome(0, 1.0), _outcome(1, 50.0), _outcome(2, 1.0, memory_mb=11.0)]
    apply_feasibility(outs, cfg)
    assert [o.feasible for o in outs] == [True, True, False]
    assert outs[1].score == 0.5 and outs[2].feasible_reason == "memory"


def test_every_outcome_records_its_runtime_against_the_fastest_defined_member_gate_or_not():
    for factor in (3.0, None):
        outs = [_outcome(0, 2.0), _outcome(1, 5.0), _outcome(2, 1e-4, defined=False), _outcome(3, 1.0, error="x")]
        apply_feasibility(outs, EnsembleConfig(runtime_factor=factor))
        assert [o.runtime_ratio for o in outs] == [1.0, 2.5, None, None]
    outs = [_outcome(0, 1.0, defined=False)]
    apply_feasibility(outs, EnsembleConfig())
    assert outs[0].runtime_ratio is None                    # no defined member, no reference


def test_a_member_row_carries_a_hash_of_its_predictions_and_its_runtime_ratio():
    ens = _ensemble(e_size=2, e_test=1)
    ens.initialise(_blob_window(200, 0), window_index=0)
    rep = ens.process_window(_blob_window(200, 1), window_index=1)
    rows = [o.to_dict() for o in rep.outcomes]
    assert all(len(r["pred_sha"]) == 16 for r in rows)
    assert all("runtime_ratio" in r and r["scoring_runtime_s"] >= 0 for r in rows)
    again = _ensemble(e_size=2, e_test=1)
    again.initialise(_blob_window(200, 0), window_index=0)
    rep2 = again.process_window(_blob_window(200, 1), window_index=1)
    assert [o.pred_sha for o in rep.outcomes] == [o.pred_sha for o in rep2.outcomes]


def test_the_window_report_times_scoring_and_spawning_apart_from_the_models():
    ens = _ensemble(e_size=3, e_test=2)
    ens.initialise(_blob_window(200, 0), window_index=0)
    rep = ens.process_window(_blob_window(200, 1), window_index=1)
    assert rep.scoring_runtime_s > 0 and rep.spawn_runtime_s > 0
    assert rep.scoring_runtime_s == pytest.approx(sum(o.scoring_runtime_s for o in rep.outcomes))
    frozen = _ensemble(e_size=1, e_test=0)
    frozen.initialise(_blob_window(200, 0), window_index=0)
    assert frozen.process_window(_blob_window(200, 1), window_index=1).spawn_runtime_s == 0.0


# --- E4: the label-driven diagnostic tuner ------------------------------------------------------

def _one_class_window(h: int, seed: int) -> WindowBuffer:
    rng = random.Random(seed)
    buf = WindowBuffer(h)
    for i in range(h):
        c = i % 2
        buf.push({"a": 5.0 * c + rng.gauss(0, 0.1), "b": rng.gauss(0, 0.1)}, 0)
    return buf


def test_the_oracle_ranks_every_member_by_its_ari_on_the_window_just_processed():
    ens = _ensemble(e_size=4, e_test=2, selection_signal="ari_oracle",
                    start=[{"n_clusters": k, "halflife": 0.5} for k in (2, 3, 5, 8)])
    ens.initialise(_blob_window(300, 0), window_index=0)
    rep = ens.process_window(_blob_window(300, 1), window_index=1)
    for o in rep.outcomes:
        assert o.score == pytest.approx(o.window_metrics["ari"]) and o.defined
        assert o.signals["silhouette"] is not None          # the silhouette is still logged beside it
    active = {m.id for m in ens.active}
    assert rep.incumbent.score == max(o.score for o in rep.outcomes if o.member_id in active)


def test_on_a_one_class_window_the_oracle_has_no_reading_and_the_incumbent_stays():
    ens = _ensemble(e_size=3, e_test=1, selection_signal="ari_oracle")
    ens.initialise(_blob_window(200, 0), window_index=0)
    first = ens.process_window(_blob_window(200, 1), window_index=1).incumbent.member_id
    rep = ens.process_window(_one_class_window(200, 2), window_index=2)
    assert all(not o.defined and o.score == UNDEFINED_SCORE for o in rep.outcomes)
    assert rep.incumbent.member_id == first


def test_a_one_label_member_is_never_the_oracles_runtime_reference():
    fast_nothing = _outcome(0, 1e-5, score=0.0)
    fast_nothing.partitioned = False                         # one label: ARI 0, defined, no partition
    slow_real = _outcome(1, 1e-2, score=0.6)
    apply_feasibility([fast_nothing, slow_real], EnsembleConfig(runtime_factor=30.0))
    assert slow_real.feasible and slow_real.runtime_ratio == 1.0


def test_the_constrained_oracle_keeps_the_silhouettes_gate():
    ens = _ensemble(e_size=2, e_test=0, selection_signal="ari_oracle_constrained",
                    start=[{"n_clusters": 2, "halflife": 0.5}, {"n_clusters": 3, "halflife": 0.5}])
    ens.initialise(_blob_window(200, 0), window_index=0)
    rep = ens.process_window(_blob_window(200, 1), window_index=1)
    assert all(o.defined == (o.partitioned and o.window_metrics["ari"] is not None) for o in rep.outcomes)


def test_the_oracle_needs_ari_among_the_external_metrics():
    with pytest.raises(ValueError, match="ari"):
        EnsembleConfig(selection_signal="ari_oracle", external_metrics=("nmi",))


def test_the_signal_noise_policy_reaches_every_reading_with_the_models_own_sentinels(monkeypatch):
    from benchmark.tuning import confstream as CS

    seen = []
    real = CS.SIG.member_signals

    def spy(window, labels, names, **kw):
        seen.append(kw)
        return real(window, labels, names, **kw)

    monkeypatch.setattr(CS.SIG, "member_signals", spy)
    monkeypatch.setattr(CS, "sentinel_labels", lambda model: (-7,))
    ens = _ensemble(e_size=2, e_test=0, log_signals=("silhouette_std",), signal_noise="drop")
    ens.initialise(_blob_window(200, 0), window_index=0)
    ens.process_window(_blob_window(200, 1), window_index=1)
    assert seen and all(kw == {"noise": "drop", "noise_labels": (-7,)} for kw in seen)


def test_an_unknown_signal_noise_policy_is_refused_and_the_default_is_the_old_reading():
    assert EnsembleConfig().signal_noise == "label"
    with pytest.raises(ValueError):
        EnsembleConfig(signal_noise="ignore")


def test_the_oracle_keeps_the_silhouette_beside_other_logged_signals():
    ens = _ensemble(e_size=2, e_test=0, selection_signal="ari_oracle", log_signals=("silhouette_std",),
                    start=[{"n_clusters": 2, "halflife": 0.5}, {"n_clusters": 3, "halflife": 0.5}])
    ens.initialise(_blob_window(200, 0), window_index=0)
    rep = ens.process_window(_blob_window(200, 1), window_index=1)
    assert all(o.signals.get("silhouette") is not None and "silhouette_std" in o.signals for o in rep.outcomes)
