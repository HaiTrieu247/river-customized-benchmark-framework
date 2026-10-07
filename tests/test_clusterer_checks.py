"""benchmark.algorithms.checks closes RIVER_BUGS.md GAP-1 and GAP-2 for clusterers.

GAP-1: ``river.checks._yield_datasets`` has no ``base.Clusterer`` branch, so
``yield_checks`` never pairs a clusterer with a dataset and ``check_estimator``
passes one without feeding it a single data point. GAP-2:
``river.checks.common._inference_methods`` has no Clusterer branch either, so
``check_predict_one_pure`` and ``check_predict_one_before_any_learn`` return
before asserting anything even when they are given data.

The upstream tests assert the gap is still there; the day river closes it they
fail, and the harness should be revisited. The composition, drift and body
tests pin the harness to river's own lists and check bodies. The planted-defect
tests are the decisive proof: each test clusterer carries exactly one defect
that river's checks let through and the framework's catch. The matrix records
what the harness measures on unmodified river for every registered clusterer;
each expected failure is a strict xfail whose reason cites the RIVER_BUGS.md
entry behind it, or says "unfiled" when there is none. A framework subclass
(DenStreamPatched, DenStreamPatchedFast, CluStreamPyramidal) may only inherit
the failures of the entry it extends.
"""

from __future__ import annotations

import ast
import functools
import inspect
import math
import os
import pathlib
import random
import subprocess
import sys
import textwrap
import types

import pytest

# Import order matters: benchmark.algorithms puts the sibling river tree on
# sys.path as a side effect, so it has to come before `river`.
from benchmark.algorithms import checks as fw_checks
from benchmark.algorithms.registry import ALGORITHMS, NUMERIC, TOKENS, discover_clusterers
from river import (
    anomaly,
    base,
    checks,
    cluster,
    feature_extraction,
    linear_model,
    model_selection,
    multioutput,
    preprocessing,
    time_series,
    tree,
)
from river.checks import common

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
#: Every river clusterer plus every registry entry, so a framework subclass
#: (DenStreamPatched, DenStreamPatchedFast, CluStreamPyramidal) joins the matrix
#: the moment it is registered and has to bring its expectations with it.
_ALL = sorted(set(discover_clusterers()) | set(ALGORITHMS))


@pytest.fixture(autouse=True)
def _seeded_random():
    # River's shuffle and disappearing-feature checks draw from the global
    # `random`; pin it so a strict xfail cannot flip between runs.
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)


def _dataset_of(check):
    """The dataset a yielded check was bound to, or None for a static check."""
    if isinstance(check, functools.partial):
        return check.keywords.get("dataset")
    return None


def _build(name):
    return ALGORITHMS[name].factory()[0]


def _phishing():
    return next(iter(fw_checks.clusterer_datasets()))


_VOCAB = "apple banana cherry date elder fig grape honey ice jam kiwi lemon mango nut olive pear"


def _token_datasets(scale=1):
    """Token counts for TextClust, made by BagOfWords as in its docstring: two topics of eight words."""

    def docs():
        rng = random.Random(1)
        bow = feature_extraction.BagOfWords()
        words = _VOCAB.split()
        for i in range(200 * scale):
            topic = words[8 * (i % 2) : 8 * (i % 2) + 8]
            yield bow.transform_one(" ".join(rng.choice(topic) for _ in range(6))), i % 2

    yield docs()


#: input_kind -> (datasets, points per short stream, points per scale=10 stream).
#: Phishing has 1250 rows, so take(2000) stops there.
_STREAMS = {
    NUMERIC: (fw_checks.clusterer_datasets, 200, 1250),
    TOKENS: (_token_datasets, 200, 2000),
}


def _make(name):
    """The datasets for this algorithm's input kind; Phishing for a kind with no test stream."""
    return _STREAMS.get(ALGORITHMS[name].input_kind, (fw_checks.clusterer_datasets,))[0]


# --- upstream: the gap is still there -----------------------------------------


def test_upstream_gap1_yield_datasets_has_no_clusterer_branch():
    assert list(checks._yield_datasets(cluster.DBSTREAM())) == []
    # Same function, same tree: a classifier does get data, so what is missing
    # is the Clusterer branch, not the datasets.
    assert list(checks._yield_datasets(linear_model.LogisticRegression()))


def test_upstream_gap2_inference_methods_has_no_clusterer_branch():
    assert list(common._inference_methods(cluster.DBSTREAM())) == []
    assert list(common._inference_methods(anomaly.HalfSpaceTrees())) == ["score_one"]


@pytest.mark.parametrize("name", sorted(discover_clusterers()))
def test_upstream_yield_checks_gives_no_clusterer_a_dataset_check(name):
    assert [c for c in checks.yield_checks(_build(name)) if _dataset_of(c) is not None] == []


# --- gap closed: what the harness yields ---------------------------------------

_GAP2_SWAP = {
    common.check_predict_one_pure: fw_checks.check_predict_one_pure,
    common.check_predict_one_before_any_learn: fw_checks.check_predict_one_before_any_learn,
}


def _length(check):
    dataset = _dataset_of(check)
    if dataset is None:
        return None
    assert iter(dataset) is dataset, f"{check.__name__} got a restartable dataset"
    return sum(1 for _ in dataset)


@pytest.mark.parametrize("name", _ALL)
def test_harness_is_river_checks_then_every_dataset_check_then_memory(name):
    model = _build(name)
    make, short, long = _STREAMS[ALGORITHMS[name].input_kind]
    yielded = list(fw_checks.yield_clusterer_checks(model, make))

    # River's own checks first and unchanged, then the replica with the two
    # GAP-2 checks swapped, one short stream each, then memory on the long one.
    want = [(c.__name__, inspect.unwrap(c), None) for c in checks.yield_checks(model)]
    want += [(c.__name__, _GAP2_SWAP.get(c, c), short) for c in fw_checks.river_dataset_checks(model)]
    want += [("check_bounded_memory_growth", common.check_bounded_memory_growth, long)]
    assert [(c.__name__, inspect.unwrap(c), _length(c)) for c in yielded] == want

    by_name = {c.__name__: inspect.unwrap(c) for c in yielded}
    assert by_name["check_predict_one_pure"] is fw_checks.check_predict_one_pure
    assert by_name["check_predict_one_before_any_learn"] is fw_checks.check_predict_one_before_any_learn
    # Stable names: what `_unit_test_skips` and the report key on.
    again = [c.__name__ for c in fw_checks.yield_clusterer_checks(model, make)]
    assert [c.__name__ for c in yielded] == again
    assert all(n.startswith("check_") for n in again)


# The dataset-check list is a literal inside river's `yield_checks`, so the
# framework replicates it. These estimators all get datasets from river, which
# makes river's own output the reference: if river adds, drops, reorders or
# rewraps a check, the replica no longer matches and this fails.
_REFERENCE_ESTIMATORS = {
    "binary_classifier": lambda: linear_model.LogisticRegression(),
    "multiclass_tree_with_debug_one": lambda: tree.HoeffdingTreeClassifier(),
    "regressor": lambda: linear_model.LinearRegression(),
    "stochastic_anomaly_detector": lambda: anomaly.HalfSpaceTrees(),
    "minibatch_transformer": lambda: preprocessing.StandardScaler(),
    "multilabel": lambda: multioutput.ClassifierChain(linear_model.LogisticRegression()),
    "model_selector": lambda: model_selection.GreedyRegressor(
        [linear_model.LinearRegression(), linear_model.LinearRegression(l2=0.1)]
    ),
    "forecaster": lambda: time_series.SNARIMAX(p=1, d=0, q=0),
}


def _identity(check):
    # An `_allow_exception` wrapper keeps the wrapped __name__ but not its
    # __qualname__, so this also sees a check being wrapped or unwrapped.
    return check.__module__, check.__qualname__, check.__name__


@pytest.mark.parametrize("kind", sorted(_REFERENCE_ESTIMATORS))
def test_replicated_dataset_check_list_matches_river(kind):
    model = _REFERENCE_ESTIMATORS[kind]()
    n = len(list(checks._yield_datasets(model)))
    assert n, "reference estimator must be one river feeds datasets to"
    river = [_identity(c) for c in checks.yield_checks(model) if _dataset_of(c) is not None]
    mine = [_identity(c) for c in fw_checks.river_dataset_checks(model) for _ in range(n)]
    if not isinstance(model, time_series.base.Forecaster):
        memory = _identity(common.check_bounded_memory_growth)
        mine += [memory] * len(list(checks._yield_datasets(model, scale=10)))
    assert river == mine


def test_dataset_checks_get_one_shot_streams():
    """check_radically_disappearing_features learns 20 points, then resumes 10 further on.

    On a one-shot stream that is 20 + 170 learns; a restartable one gives 20 + 190.
    """
    model = _CountsLearns()
    (check,) = [
        c for c in fw_checks.yield_clusterer_checks(model)
        if c.__name__ == "check_radically_disappearing_features"
    ]
    check(model)
    assert model.n_learned == 190


# --- the two GAP-2 checks assert what river's assert ---------------------------


class _QualifiedLearn(ast.NodeTransformer):
    # The framework calls `common._learn`; river's module calls it as `_learn`.
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "common" and node.attr == "_learn":
            return ast.copy_location(ast.Name(id="_learn", ctx=node.ctx), node)
        return self.generic_visit(node)


def _code(func):
    (fn,) = ast.parse(textwrap.dedent(inspect.getsource(func))).body
    body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
    return ast.dump(fn.args), ast.dump(_QualifiedLearn().visit(ast.Module(body=body, type_ignores=[])))


@pytest.mark.parametrize("check_name", ["check_predict_one_pure", "check_predict_one_before_any_learn"])
def test_framework_predict_one_checks_have_river_code(check_name):
    """Same signature and statements as river's, docstring and comments aside.

    `_inference_methods` resolves to the framework's own, which is the whole
    difference; if river changes either body, this fails until the copy follows.
    """
    assert _code(getattr(fw_checks, check_name)) == _code(getattr(common, check_name))


class _Regressor(base.Regressor):
    """Predicts 0; river's own GAP-2 checks are active for a regressor."""

    def learn_one(self, x, y):
        pass

    def predict_one(self, x):
        return 0.0


class _RegressorMutatesInput(_Regressor):
    def predict_one(self, x):
        x["_predicted"] = 0.0
        return 0.0


class _RegressorRaisesBeforeLearn(_Regressor):
    def predict_one(self, x):
        raise RuntimeError("no state yet")


@pytest.mark.parametrize(
    ("check_name", "model_cls", "raises"),
    [
        ("check_predict_one_pure", _Regressor, None),
        ("check_predict_one_pure", _RegressorMutatesInput, AssertionError),
        ("check_predict_one_before_any_learn", _Regressor, None),
        ("check_predict_one_before_any_learn", _RegressorRaisesBeforeLearn, RuntimeError),
    ],
)
def test_framework_predict_one_checks_agree_with_river_where_river_is_active(
    check_name, model_cls, raises
):
    for check in (getattr(common, check_name), getattr(fw_checks, check_name)):
        if raises is None:
            check(model_cls(), dataset=_phishing())
        else:
            with pytest.raises(raises):
                check(model_cls(), dataset=_phishing())


# --- planted defects: the decisive proof ---------------------------------------


def _sq_dist(a, b):
    # fsum is exactly rounded, hence independent of key order: a correct
    # clusterer must not fail the shuffle check through float noise.
    return math.fsum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in {*a, *b})


class _Leader(base.Clusterer):
    """The first ``n_clusters`` points seen become fixed centres; predict the nearest."""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.centers: list[dict] = []

    def _keep(self, x):
        return dict(x)

    def learn_one(self, x):
        if len(self.centers) < self.n_clusters:
            self.centers.append(self._keep(x))

    def predict_one(self, x):
        if not self.centers:
            return 0
        return min(range(len(self.centers)), key=lambda i: _sq_dist(self.centers[i], x))


class _CountsLearns(_Leader):
    """A correct clusterer that counts its learn_one calls."""

    def __init__(self, n_clusters: int = 3):
        super().__init__(n_clusters)
        self.n_learned = 0

    def learn_one(self, x):
        self.n_learned += 1
        super().learn_one(x)


class _MutatesInput(_Leader):
    """Defect: predict_one writes into the caller's dict."""

    def predict_one(self, x):
        y = super().predict_one(x)
        x["_predicted"] = y
        return y


class _RaisesBeforeLearn(_Leader):
    """Defect: predict_one has no answer before the first learn_one (BUG-5's shape)."""

    def predict_one(self, x):
        return min(range(len(self.centers)), key=lambda i: _sq_dist(self.centers[i], x))


class _AliasesInput(_Leader):
    """Defect: learn_one keeps the caller's dict as a centre (BUG-3's shape)."""

    def _keep(self, x):
        return x


class _PredictsNothing(_Leader):
    """Clusters something other than samples, as ODAC does: predict_one is not implemented."""

    def predict_one(self, x):
        raise NotImplementedError


class _PredictsNothingNeedsFirstFeatures(_PredictsNothing):
    """Defect: learn_one sums the first point's features and indexes them in every later point.

    ODAC's shape: a missing feature crashes learn_one, behind a predict_one that
    raises NotImplementedError before any check gets that far.
    """

    def learn_one(self, x):
        if not self.centers:
            self.centers.append(dict.fromkeys(x, 0.0))
        sums = self.centers[0]
        for k in sums:
            sums[k] += x[k]


class _PredictMovesCentre(_Leader):
    """predict_one drags the nearest centre halfway to x: it learns while it infers."""

    def predict_one(self, x):
        i = super().predict_one(x)
        if self.centers:
            c = self.centers[i]
            for k in {*c, *x}:
                c[k] = 0.5 * (c.get(k, 0.0) + x.get(k, 0.0))
        return i


def _failing(model):
    return {k for k, v in fw_checks.run_clusterer_checks(model).items() if v != "pass"}


def test_planted_predict_one_mutates_its_input_river_silent_framework_raises():
    # River's check_predict_one_pure asserts that the caller's x is unchanged;
    # that is the assertion the framework's makes reachable for a clusterer.
    assert common.check_predict_one_pure(_MutatesInput(), dataset=_phishing()) is None
    with pytest.raises(AssertionError, match="mutated its input"):
        fw_checks.check_predict_one_pure(_MutatesInput(), dataset=_phishing())
    checks.check_estimator(_MutatesInput())
    assert _failing(_MutatesInput()) == {"check_predict_one_pure"}


def test_planted_predict_one_raises_before_learn_river_silent_framework_fails():
    assert common.check_predict_one_before_any_learn(_RaisesBeforeLearn(), dataset=_phishing()) is None
    with pytest.raises(ValueError):
        fw_checks.check_predict_one_before_any_learn(_RaisesBeforeLearn(), dataset=_phishing())
    checks.check_estimator(_RaisesBeforeLearn())
    # The same defect trips every check that infers before it first learns --
    # exactly the set STREAMKMeans fails in the matrix below through BUG-5.
    assert _failing(_RaisesBeforeLearn()) == {
        "check_pickling",
        "check_shuffle_features_no_impact",
        "check_emerging_features",
        "check_disappearing_features",
        "check_radically_disappearing_features",
        "check_predict_one_before_any_learn",
    }


def test_planted_aliasing_passes_check_estimator_fails_check_clusterer():
    checks.check_estimator(_AliasesInput())
    with pytest.raises(AssertionError) as info:
        fw_checks.check_clusterer(_AliasesInput())
    assert "check_no_state_aliasing_with_input" in "".join(getattr(info.value, "__notes__", []))
    assert _failing(_AliasesInput()) == {"check_no_state_aliasing_with_input"}


def test_planted_learn_defect_behind_unimplemented_predict_one_is_reached():
    # River's checks stop at the first predict_one, so the defect never runs.
    for check in (common.check_emerging_features, common.check_radically_disappearing_features):
        with pytest.raises(NotImplementedError):
            check(_PredictsNothingNeedsFirstFeatures(), dataset=_phishing())
    report = fw_checks.run_clusterer_checks(_PredictsNothingNeedsFirstFeatures())
    failing = {k: v for k, v in report.items() if v != "pass"}
    assert set(failing) == {"check_emerging_features", "check_radically_disappearing_features"}
    assert all(v.startswith("fail: KeyError") for v in failing.values())


class _PredictsNothingWrapsScaler(_PredictsNothing):
    """Predicts nothing and holds a nested estimator, which the rerun has to keep intact."""

    def __init__(self, n_clusters: int = 3, scaler: base.Transformer | None = None):
        super().__init__(n_clusters)
        self.scaler = scaler if scaler is not None else preprocessing.StandardScaler()

    def learn_one(self, x):
        self.scaler.learn_one(x)
        super().learn_one(self.scaler.transform_one(x))


@pytest.mark.parametrize("model_cls", [_PredictsNothing, _PredictsNothingWrapsScaler])
def test_unimplemented_predict_one_alone_passes_every_check(model_cls):
    fw_checks.check_clusterer(model_cls())
    assert set(fw_checks.run_clusterer_checks(model_cls()).values()) == {"pass"}


def test_correct_clusterer_passes_everything():
    checks.check_estimator(_Leader())
    fw_checks.check_clusterer(_Leader())
    report = fw_checks.run_clusterer_checks(_Leader())
    assert set(report.values()) == {"pass"}
    assert len(report) == len({c.__name__ for c in fw_checks.yield_clusterer_checks(_Leader())})


def test_known_limit_predict_one_that_changes_model_state_is_not_caught():
    """No river check compares the model's state across predict_one, and the harness adds none.

    BUG-7 has this shape and is caught by its own tests, not here. This test
    fails if a check of that kind is ever added, so the limit stays documented.
    """
    model = _PredictMovesCentre()
    for x, _ in _phishing():
        model.learn_one(x)
    before = [dict(c) for c in model.centers]
    model.predict_one({"url_length": 9.0})
    assert model.centers != before
    fw_checks.check_clusterer(_PredictMovesCentre())


class _SkipsAliasing(_AliasesInput):
    """Declares its one defect a known skip, as river estimators do."""

    @classmethod
    def _unit_test_skips(cls):
        return {"check_no_state_aliasing_with_input"}


def test_check_clusterer_honours_unit_test_skips():
    fw_checks.check_clusterer(_SkipsAliasing())
    report = fw_checks.run_clusterer_checks(_SkipsAliasing())
    assert report["check_no_state_aliasing_with_input"] == "skip: listed in _unit_test_skips"


def test_report_message_names_the_exception_and_where_it_was_raised():
    msg = fw_checks.run_clusterer_checks(_RaisesBeforeLearn())["check_predict_one_before_any_learn"]
    assert msg.startswith("fail: ValueError")
    assert "test_clusterer_checks.py:" in msg


# --- measured matrix on unmodified river ---------------------------------------

_BUG3 = "BUG-3: learn_one stores the caller's x by reference"
_BUG3_INHERITED = (
    "BUG-3: learn_one is inherited unchanged from DenStream, which stores the caller's x "
    "by reference; DenStreamPatched fixes only BUG-7/BUG-8"
)
_BUG3_INHERITED_FAST = (
    "BUG-3: learn_one is inherited unchanged from DenStream through DenStreamPatched, which "
    "stores the caller's x by reference; DenStreamPatchedFast changes only predict_one (GAP-3)"
)
_BUG3_INHERITED_PYRAMIDAL = (
    "BUG-3: learn_one runs CluStream.learn_one, which keeps the caller's x as "
    "CluStreamMicroCluster.x; the snapshots CluStreamPyramidal adds (GAP-4) copy their statistics"
)
_BUG4 = "BUG-4: CluStream.learn_one raises KeyError when a new feature appears mid-stream"
_BUG4_INHERITED = (
    "BUG-4: learn_one runs CluStream.learn_one, which raises KeyError when a new feature "
    "appears mid-stream; CluStreamPyramidal adds the pyramidal time frame only (GAP-4)"
)
_BUG5 = (
    "BUG-5: STREAMKMeans.predict_one raises ValueError until the first learn_one, and this "
    "check infers before it learns"
)
_BUG5_MASKS_BUG2 = _BUG5 + (
    ". It masks a second failure: with BUG-5 out of the way this check still fails for some "
    "shuffle seeds (seed 0 among them), through the KMeans inside STREAMKMeans (BUG-2), so "
    "expect an AssertionError here once BUG-5 is fixed"
)
_NO_DOC = (
    "unfiled: {} has no class docstring, so river's own check_doc fails and "
    "check_estimator raises before reaching any other check"
)
_ODAC_SINGLETON = (
    "unfiled: ODAC.learn_one raises TypeError when x lacks the one series of a singleton "
    "cluster: update_statistics calls float(x.get(name)) there without the None guard its "
    "multi-series branch has (odac.py:480)"
)

#: (algorithm, check) -> (exception, reason), measured under PYTHONHASHSEED=0
#: with `random` seeded, 2026-10-06, river 873fa6e0.
EXPECTED_FAILURES = {
    ("BIRCH", "check_doc"): (AssertionError, _NO_DOC.format("BIRCH")),
    ("ImprovedBIRCH", "check_doc"): (AssertionError, _NO_DOC.format("ImprovedBIRCH")),
    ("CluStream", "check_emerging_features"): (KeyError, _BUG4),
    ("CluStream", "check_no_state_aliasing_with_input"): (AssertionError, _BUG3),
    ("CluStreamPyramidal", "check_emerging_features"): (KeyError, _BUG4_INHERITED),
    ("CluStreamPyramidal", "check_no_state_aliasing_with_input"): (
        AssertionError, _BUG3_INHERITED_PYRAMIDAL,
    ),
    ("DBSTREAM", "check_no_state_aliasing_with_input"): (AssertionError, _BUG3),
    ("DenStream", "check_no_state_aliasing_with_input"): (AssertionError, _BUG3),
    ("DenStreamPatched", "check_no_state_aliasing_with_input"): (AssertionError, _BUG3_INHERITED),
    ("DenStreamPatchedFast", "check_no_state_aliasing_with_input"): (
        AssertionError, _BUG3_INHERITED_FAST,
    ),
    ("ODAC", "check_emerging_features"): (TypeError, _ODAC_SINGLETON),
    ("STREAMKMeans", "check_no_state_aliasing_with_input"): (AssertionError, _BUG3),
    ("STREAMKMeans", "check_shuffle_features_no_impact"): (ValueError, _BUG5_MASKS_BUG2),
    **{
        ("STREAMKMeans", c): (ValueError, _BUG5)
        for c in (
            "check_pickling",
            "check_emerging_features",
            "check_disappearing_features",
            "check_radically_disappearing_features",
            "check_predict_one_before_any_learn",
        )
    },
}

#: Cells whose outcome depends on the interpreter's hash seed. They pass under
#: PYTHONHASHSEED=0, which is how this suite is run. Under another hash seed
#: they fail for some shuffle seeds and not others, so there they are a
#: non-strict xfail.
HASH_DEPENDENT = {
    ("KMeans", "check_shuffle_features_no_impact"): (
        "BUG-2: KMeans iterates a set of feature names while drawing lazy centre "
        "coordinates from its RNG, so feature order changes the centres"
    ),
}


def _matrix():
    params = []
    for name in _ALL:
        if name not in ALGORITHMS:
            params.append(pytest.param(name, None, marks=pytest.mark.skip(reason="not in registry")))
            continue
        kind = ALGORITHMS[name].input_kind
        seen = {}
        for check in fw_checks.yield_clusterer_checks(_build(name), _make(name)):
            seen.setdefault(check.__name__, _dataset_of(check) is not None)
        for check_name, dataset_driven in seen.items():
            marks = []
            if dataset_driven and kind not in _STREAMS:
                marks.append(pytest.mark.skip(reason=f"no test stream for input_kind={kind}"))
            elif (name, check_name) in EXPECTED_FAILURES:
                exc, reason = EXPECTED_FAILURES[(name, check_name)]
                marks.append(pytest.mark.xfail(strict=True, raises=exc, reason=reason))
            elif (name, check_name) in HASH_DEPENDENT and os.environ.get("PYTHONHASHSEED") != "0":
                marks.append(pytest.mark.xfail(strict=False, reason=HASH_DEPENDENT[(name, check_name)]))
            params.append(pytest.param(name, check_name, marks=marks, id=f"{name}-{check_name}"))
    return params


@pytest.mark.parametrize(("name", "check_name"), _matrix())
def test_measured_matrix(name, check_name):
    model = _build(name)
    if check_name in model._unit_test_skips():
        pytest.skip(f"{name} lists {check_name} in _unit_test_skips")
    selected = [
        c for c in fw_checks.yield_clusterer_checks(model, _make(name))
        if c.__name__ == check_name
    ]
    assert selected
    for check in selected:
        check(model.clone())


def test_matrix_expectations_name_real_cells():
    cells = {tuple(p.values) for p in _matrix()}
    assert set(EXPECTED_FAILURES) <= cells
    assert set(HASH_DEPENDENT) <= cells


#: Framework subclass -> the registered entry it extends.
_EXTENDS = {
    "DenStreamPatched": "DenStream",
    "DenStreamPatchedFast": "DenStreamPatched",
    "CluStreamPyramidal": "CluStream",
}


def test_framework_subclasses_fail_only_what_their_reference_fails():
    """A workaround may inherit its reference's failures, never add one.

    Each inherited failure must also raise the same exception. A new failing cell
    for a subclass is a defect in the subclass, not a cell to xfail.
    """
    assert set(_EXTENDS) == set(ALGORITHMS) - set(discover_clusterers())
    for entry, ref in _EXTENDS.items():
        own = {c: exc for (n, c), (exc, _) in EXPECTED_FAILURES.items() if n == entry}
        inherited = {c: exc for (n, c), (exc, _) in EXPECTED_FAILURES.items() if n == ref}
        assert own.items() <= inherited.items(), entry
        assert {c for n, c in HASH_DEPENDENT if n == entry} <= {c for n, c in HASH_DEPENDENT if n == ref}


# River yields check_seeding_is_idempotent only for a model whose seed is
# None. The registry pins every seed, so the matrix never gets it.
_SEEDED = [n for n in _ALL if "seed" in _build(n)._get_params()]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n, marks=[pytest.mark.xfail(strict=True, raises=ValueError, reason=_BUG5)]
            if n == "STREAMKMeans" else []
        )
        for n in _SEEDED
    ],
)
def test_seeding_is_idempotent_with_seed_unset(name):
    model = _build(name).clone({"seed": None})
    selected = [
        c for c in fw_checks.yield_clusterer_checks(model, _make(name))
        if c.__name__ == "check_seeding_is_idempotent"
    ]
    assert selected
    for check in selected:
        check(model.clone())


_KMEANS_SHUFFLE = """
import random
from benchmark.algorithms import checks
from benchmark.algorithms.registry import ALGORITHMS
model = ALGORITHMS["KMeans"].factory()[0]
random.seed(5)
print(checks.run_clusterer_checks(model)["check_shuffle_features_no_impact"])
"""


def test_bug2_kmeans_shuffle_check_depends_on_the_hash_seed():
    """BUG-2 through the harness: same code, same seeds, only PYTHONHASHSEED differs."""
    out = {}
    for hash_seed in ("0", "3"):
        proc = subprocess.run(
            [sys.executable, "-c", _KMEANS_SHUFFLE],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
        )
        out[hash_seed] = proc.stdout.strip()
    assert out["0"] == "pass"
    assert out["3"].startswith("fail: AssertionError")


def _twin(learn_defect):
    """A class sharing its module and qualname with every other twin, as a class factory or a re-run cell makes."""
    parent = _PredictsNothingNeedsFirstFeatures if learn_defect else _PredictsNothing
    namespace = {"__module__": __name__, "__qualname__": "_Twin", "__doc__": parent.__doc__}
    return types.new_class("_Twin", (parent,), exec_body=lambda ns: ns.update(namespace))


@pytest.mark.parametrize("defective_first", [True, False])
def test_rerun_keeps_apart_classes_that_share_a_qualified_name(defective_first):
    good, bad = _twin(learn_defect=False), _twin(learn_defect=True)
    order = (bad, good) if defective_first else (good, bad)
    got = {cls: fw_checks.run_clusterer_checks(cls())["check_radically_disappearing_features"] for cls in order}
    assert got[good] == "pass"
    assert got[bad].startswith("fail: KeyError")
