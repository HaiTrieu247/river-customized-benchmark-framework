"""The preprocessing slot takes any unsupervised river transformer, not just scalers."""

from __future__ import annotations

import random

import pytest

from benchmark.algorithms.registry import PREPROCESSORS, make_preprocessor
from benchmark.algorithms.transformers import OneHotCategorical

_NUMERIC = {f"f{i}": float(i) for i in range(6)}
_MIXED = {"duration": 1.5, "proto": "tcp"}
_TEXT = {"body": "machine learning on streaming data"}

#: Input and constructor params each step needs to be exercised meaningfully.
_FIXTURES: dict[str, tuple[dict, dict]] = {
    "one_hot": (_MIXED, {}),
    "one_hot_categorical": (_MIXED, {}),
    "ordinal": (_MIXED, {}),
    "feature_hasher": ({"a": "x", "b": "y"}, {}),
    "gap_encoder": (_TEXT, {"on": "body"}),
    "bag_of_words": (_TEXT, {"on": "body"}),
    "tfidf": (_TEXT, {"on": "body"}),
    "lda": ({"machine": 1, "learning": 1, "data": 1}, {"n_components": 3}),
    "sparse_projection": (_NUMERIC, {"n_components": 3}),
    "gaussian_projection": (_NUMERIC, {"n_components": 3}),
    "select": (_MIXED, {}),
    "discard": (_MIXED, {}),
    "select_type": (_MIXED, {}),
}


@pytest.mark.parametrize("name", [n for n in PREPROCESSORS if n != "none"])
def test_every_registered_preprocessor_runs_in_the_slot(name):
    """A registered step that cannot learn_one/transform_one is not usable."""
    x, params = _FIXTURES.get(name, (_NUMERIC, {}))
    step = make_preprocessor(name, **params)
    for _ in range(5):
        step.learn_one(x)
        out = step.transform_one(x)
    assert isinstance(out, dict)


def test_none_is_a_passthrough_and_rejects_params():
    assert make_preprocessor("none") is None
    with pytest.raises(TypeError, match="takes no params"):
        make_preprocessor("none", window_size=10)


# --- chaining ---------------------------------------------------------------
def test_steps_chain_left_to_right():
    """The useful pipelines are rarely a single step."""
    chain = make_preprocessor("impute|one_hot_categorical|standard")
    rows = [
        {"dur": 1.0, "proto": "tcp"},
        {"proto": "udp"},                 # `dur` missing -> imputed
        {"dur": 5.0, "proto": "tcp"},
    ]
    for r in rows:
        chain.learn_one(r)
    out = chain.transform_one({"dur": 7.0, "proto": "udp"})
    assert "dur" in out                    # survived the encoder as a number
    assert {"proto_tcp", "proto_udp"} <= set(out)
    assert all(isinstance(v, float) for v in out.values())


def test_chain_params_are_nested_per_step():
    chain = make_preprocessor(
        "variance_threshold|sparse_projection|standard",
        sparse_projection={"n_components": 4, "seed": 1},
    )
    rng = random.Random(0)
    for _ in range(50):
        x = {f"f{j}": rng.gauss(0, 1) for j in range(20)}
        x["constant"] = 0.0
        chain.learn_one(x)
        out = chain.transform_one(x)
    assert len(out) == 4                   # projected down
    assert "constant" not in out           # dropped before projection


def test_flat_params_on_a_chain_are_rejected_rather_than_guessed():
    """Two steps can share a parameter name; guessing an owner would be wrong."""
    with pytest.raises(KeyError, match="nested under a step name"):
        make_preprocessor("impute|standard", window_size=10)
    with pytest.raises(TypeError, match="must be a dict"):
        make_preprocessor("impute|standard", standard=5)


def test_unknown_step_in_a_chain_fails_loudly():
    with pytest.raises(KeyError, match="unknown preprocessor"):
        make_preprocessor("impute|standrad")


def test_none_inside_a_chain_is_a_no_op():
    chain = make_preprocessor("none|standard")
    chain.learn_one(_NUMERIC)
    assert set(chain.transform_one(_NUMERIC)) == set(_NUMERIC)


# --- the one-hot trap -------------------------------------------------------
def test_plain_one_hot_shreds_numeric_columns_and_the_guarded_one_does_not():
    """River's OneHotEncoder encodes *every* key it is given.

    On a mixed record that turns a numeric column into one indicator per
    distinct value it has ever taken, which destroys every distance a clusterer
    would compute from it.
    """
    rows = [{"dur": 1.0, "proto": "tcp"}, {"dur": 5.0, "proto": "udp"},
            {"dur": 3.0, "proto": "tcp"}]

    plain = make_preprocessor("one_hot")
    guarded = make_preprocessor("one_hot_categorical")
    for r in rows:
        plain.learn_one(r)
        guarded.learn_one(r)

    probe = {"dur": 7.0, "proto": "udp"}
    plain_out = plain.transform_one(probe)
    guarded_out = guarded.transform_one(probe)

    # plain: `dur` became indicators, one per value seen
    assert "dur" not in plain_out
    assert {"dur_1.0", "dur_3.0", "dur_5.0"} <= set(plain_out)
    # guarded: the number is passed straight through
    assert guarded_out["dur"] == 7.0
    assert guarded_out["proto_udp"] == 1
    assert guarded_out["proto_tcp"] == 0


def test_one_hot_categorical_passes_through_a_fully_numeric_record():
    enc = OneHotCategorical()
    enc.learn_one(_NUMERIC)
    assert enc.transform_one(_NUMERIC) == _NUMERIC


def test_one_hot_categorical_treats_booleans_as_categorical():
    """A bool is a Number in Python; encoding it as one is rarely what is meant."""
    enc = OneHotCategorical()
    for flag in (True, False, True):
        enc.learn_one({"x": 1.0, "flag": flag})
    out = enc.transform_one({"x": 1.0, "flag": False})
    assert out["x"] == 1.0
    assert "flag" not in out
    assert set(out) == {"x", "flag_True", "flag_False"}


# --- end to end -------------------------------------------------------------
def test_a_chain_reaches_the_result_and_the_label():
    from benchmark.experiments import BenchmarkConfig, Setting, run_setting
    from benchmark.datasets.synthetic import StreamConfig, SyntheticStream

    setting = Setting(
        "DBSTREAM",
        scaler="variance_threshold|standard",
        scaler_params={"standard": {"window_size": 500}},
    )
    assert "variance_threshold|standard" in setting.display()

    stream = SyntheticStream(StreamConfig(n_samples=3000, n_clusters=4, n_features=8, seed=1))
    res = run_setting(setting, stream, BenchmarkConfig(n_checkpoints=3, window_size=500), seed=1)
    assert res.scaler == "variance_threshold|standard"
    assert res.scaler_params == {"standard": {"window_size": 500}}
    assert res.quality["ari"] is not None
