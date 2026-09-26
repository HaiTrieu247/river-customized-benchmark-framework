"""Unit-space parameter declarations and the confStream offspring sampler."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmark.tuning.space import Param, ParamSpace, SamplerState, SpaceContext

CTX = SpaceContext(s=100.0, d=10)


# --- one parameter: unit interval -> value -----------------------------------


@pytest.mark.parametrize(
    "param, u, expected",
    [
        (Param("x", "lin", 2.0, 6.0), 0.0, 2.0),
        (Param("x", "lin", 2.0, 6.0), 1.0, 6.0),
        (Param("x", "lin", 2.0, 6.0), 0.25, 3.0),
        (Param("x", "log", 0.01, 100.0), 0.0, 0.01),
        (Param("x", "log", 0.01, 100.0), 0.5, 1.0),
        (Param("x", "log", 0.01, 100.0), 1.0, 100.0),
        (Param("k", "log_int", 2, 60), 0.0, 2),
        (Param("k", "log_int", 2, 60), 1.0, 60),
        (Param("c", "choice", values=(200, 500, 1000)), 0.0, 200),
        (Param("c", "choice", values=(200, 500, 1000)), 0.5, 500),
        (Param("c", "choice", values=(200, 500, 1000)), 0.999, 1000),
        # scaled: c * s with c log-uniform; the unit s comes from the context
        (Param("eps", "scaled", 0.05, 2.0), 0.0, 5.0),
        (Param("eps", "scaled", 0.05, 2.0), 1.0, 200.0),
        # a fraction of the data dimension, at least 1
        (Param("l", "frac_of_d", 0.2, 1.0), 0.0, 2),
        (Param("l", "frac_of_d", 0.2, 1.0), 1.0, 10),
        (Param("l", "frac_of_d", 0.01, 1.0), 0.0, 1),
    ],
)
def test_param_value_maps_the_unit_interval_onto_the_declared_range(param, u, expected):
    value = param.value(u, CTX)
    assert value == pytest.approx(expected)
    assert isinstance(value, int) == isinstance(expected, int)


def test_param_rejects_a_log_range_that_touches_zero():
    with pytest.raises(ValueError):
        Param("x", "log", 0.0, 1.0)


@pytest.mark.parametrize(
    "param, value",
    [
        (Param("x", "lin", 2.0, 6.0), 4.5),
        (Param("x", "log", 0.01, 100.0), 3.0),
        (Param("k", "log_int", 2, 60), 17),
        (Param("c", "choice", values=(200, 500, 1000)), 500),
        (Param("eps", "scaled", 0.05, 2.0), 40.0),
        (Param("l", "frac_of_d", 0.2, 1.0), 6),
    ],
)
def test_param_unit_of_inverts_value(param, value):
    u = param.unit_of(value, CTX)
    assert 0.0 <= u <= 1.0
    assert param.value(u, CTX) == pytest.approx(value, rel=1e-6)


def test_param_unit_of_clips_a_value_outside_the_range():
    p = Param("x", "lin", 2.0, 6.0)
    assert p.unit_of(100.0, CTX) == 1.0
    assert p.unit_of(-100.0, CTX) == 0.0


# --- a space: resolve a unit point into constructor kwargs --------------------


SPACE = ParamSpace(
    "Toy",
    (
        Param("eps", "scaled", 0.05, 2.0),
        Param("k", "log_int", 2, 60),
        Param("init", "choice", values=(200, 500, 1000)),
    ),
    fixed={"beta": 0.75},
)


def test_space_resolve_returns_every_parameter_plus_the_fixed_ones():
    params = SPACE.resolve({"eps": 0.0, "k": 0.0, "init": 0.5}, CTX)
    assert params == {"eps": 5.0, "k": 2, "init": 500, "beta": 0.75}


def test_space_resolve_refuses_a_point_missing_a_dimension():
    with pytest.raises(KeyError):
        SPACE.resolve({"eps": 0.0}, CTX)


def test_space_unit_point_of_a_configuration_round_trips():
    u = SPACE.unit_point({"eps": 40.0, "k": 17, "init": 1000, "beta": 0.75}, CTX)
    assert set(u) == {"eps", "k", "init"}
    assert SPACE.resolve(u, CTX) == {"eps": pytest.approx(40.0), "k": 17, "init": 1000, "beta": 0.75}


def test_space_names_and_dimension():
    assert SPACE.names == ("eps", "k", "init")
    assert SPACE.dim == 3


# --- the sampler: truncated normal around the parent, sigma decaying ----------


def test_initial_sampler_state_has_half_range_sigma_and_uniform_choices():
    st = SamplerState.initial(SPACE)
    assert st.sigma == {"eps": 0.5, "k": 0.5}
    assert np.allclose(st.probs["init"], [1 / 3, 1 / 3, 1 / 3])


def test_offspring_stays_inside_the_unit_cube_and_is_centred_on_the_parent():
    rng = np.random.default_rng(0)
    st = SamplerState.initial(SPACE)
    st.sigma = {"eps": 0.1, "k": 0.1}
    parent = {"eps": 0.5, "k": 0.9, "init": 0.5}
    eps, ks = [], []
    for _ in range(2000):
        child, _ = st.offspring(parent, rng, lam=0.05)
        assert 0.0 <= child["eps"] <= 1.0 and 0.0 <= child["k"] <= 1.0
        eps.append(child["eps"])
        ks.append(child["k"])
    assert np.mean(eps) == pytest.approx(0.5, abs=0.01)
    # truncated at 1.0, so the mass above the parent is cut and the mean drops
    assert 0.8 < np.mean(ks) < 0.9


def test_offspring_shrinks_sigma_by_two_to_the_minus_lambda():
    rng = np.random.default_rng(0)
    st = SamplerState.initial(SPACE)
    _, child_state = st.offspring({"eps": 0.5, "k": 0.5, "init": 0.5}, rng, lam=0.05)
    assert child_state.sigma["eps"] == pytest.approx(0.5 * 2 ** -0.05)
    assert child_state.sigma["k"] == pytest.approx(0.5 * 2 ** -0.05)
    # the parent's own state is untouched
    assert st.sigma["eps"] == 0.5


def test_offspring_raises_the_probability_of_the_drawn_category():
    rng = np.random.default_rng(1)
    st = SamplerState.initial(SPACE)
    child, child_state = st.offspring({"eps": 0.5, "k": 0.5, "init": 0.5}, rng, lam=0.05)
    drawn = int(child["init"] * 3)
    before = np.array([1 / 3, 1 / 3, 1 / 3])
    expected = before.copy()
    expected[drawn] *= 2 - 2 ** -0.05
    expected /= expected.sum()
    assert np.allclose(child_state.probs["init"], expected)
    assert np.allclose(st.probs["init"], before)


def test_offspring_draws_the_category_according_to_the_probabilities():
    rng = np.random.default_rng(2)
    st = SamplerState.initial(SPACE)
    st.probs["init"] = np.array([0.0, 0.0, 1.0])
    for _ in range(20):
        child, _ = st.offspring({"eps": 0.5, "k": 0.5, "init": 0.0}, rng, lam=0.05)
        assert SPACE.resolve(child, CTX)["init"] == 1000


def test_reset_restores_the_initial_state():
    st = SamplerState.initial(SPACE)
    st.sigma = {"eps": 0.01, "k": 0.02}
    st.probs["init"] = np.array([0.9, 0.05, 0.05])
    fresh = st.reset(SPACE)
    assert fresh.sigma == {"eps": 0.5, "k": 0.5}
    assert np.allclose(fresh.probs["init"], [1 / 3, 1 / 3, 1 / 3])


def test_offspring_is_deterministic_under_a_seeded_generator():
    st = SamplerState.initial(SPACE)
    parent = {"eps": 0.3, "k": 0.6, "init": 0.5}
    a, _ = st.offspring(parent, np.random.default_rng(7), lam=0.05)
    b, _ = st.offspring(parent, np.random.default_rng(7), lam=0.05)
    assert a == b
