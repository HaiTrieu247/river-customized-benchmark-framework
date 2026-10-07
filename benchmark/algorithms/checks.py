"""The behavioural checks ``river.checks`` silently skips for a clusterer.

Per AGENTS.md rule 1 the vendored ``river`` tree is never modified, and
nothing here monkeypatches it: this module composes river's own checks. It
works around two gaps in ``river.checks`` (see ``RIVER_BUGS.md``):

* **GAP-1** -- ``_yield_datasets`` has no ``base.Clusterer`` branch, so
  ``yield_checks`` pairs no dataset-driven check with a clusterer and
  ``check_estimator`` passes one after static checks alone, none of which
  feeds it a data point. :func:`yield_clusterer_checks` yields everything river
  yields, then river's dataset-driven checks bound to :func:`clusterer_datasets`.
* **GAP-2** -- ``common._inference_methods`` has no Clusterer branch, so
  ``check_predict_one_pure`` and ``check_predict_one_before_any_learn`` return
  before asserting anything even when given data. The two functions of the
  same name here are river's, with ``predict_one`` as a clusterer's inference
  method; :func:`yield_clusterer_checks` uses them in place of river's.

River lets an inference method answer ``NotImplementedError``: its two
predict_one checks skip it, and a classifier's ``_infer`` falls back from
``predict_proba_one``. The other dataset checks call ``_infer`` unguarded, so
on a clusterer that predicts nothing (ODAC clusters variables, not samples)
they would stop at the first point and never reach ``learn_one``. When
``NotImplementedError`` escapes a dataset check, the harness reruns it on a
fresh clone and the same data with ``predict_one`` answering ``None`` instead.

The dataset-check list is a literal inside ``river.checks.yield_checks``, so
:func:`river_dataset_checks` replicates it; ``tests/test_clusterer_checks.py``
compares the replica with river's own output for estimator types river does
feed datasets to, and fails when the two drift apart.
"""

from __future__ import annotations

import copy
import functools
import os
import traceback
import types
import typing

from .adapter import build_river_path

# Ensure the sibling river tree is importable before we touch river.*, exactly
# as benchmark/algorithms/registry.py does.
build_river_path()

from river import base, datasets  # noqa: E402
from river import checks as river_checks  # noqa: E402
from river.checks import common  # noqa: E402

__all__ = [
    "check_clusterer",
    "check_predict_one_before_any_learn",
    "check_predict_one_pure",
    "clusterer_datasets",
    "river_dataset_checks",
    "run_clusterer_checks",
    "yield_clusterer_checks",
]


def clusterer_datasets(scale: int = 1):
    """Datasets for clusterers, in the shape of river's ``_yield_datasets``.

    Each call yields fresh one-shot streams, so every check gets its own, as in
    river. Phishing is the labelled numeric stream RIVER_BUGS.md GAP-1 measured
    with and already serves river's classifier branch; clusterers ignore ``y``.
    It has 1250 rows, so ``scale=10`` gives 1250 points, not 2000, exactly as
    in river's classifier branch.

    Pass another function of the same shape as ``make_datasets`` to the
    functions below to check a clusterer that does not take numeric vectors.
    """
    yield datasets.Phishing().take(200 * scale)


def _inference_methods(model):
    if isinstance(model, base.Clusterer):
        yield "predict_one"
    else:
        yield from common._inference_methods(model)


def check_predict_one_pure(model, dataset):
    """river's check of the same name, with a clusterer's ``predict_one`` covered (GAP-2)."""

    methods = list(_inference_methods(model))
    if not methods:
        return

    for x, y in dataset:
        common._learn(model, x, y)
        for name in methods:
            method = getattr(model, name, None)
            if method is None:
                continue
            xx = copy.deepcopy(x)
            try:
                method(x)
            except NotImplementedError:
                continue
            assert x == xx, f"{name} mutated its input"


def check_predict_one_before_any_learn(model, dataset):
    """river's check of the same name, with a clusterer's ``predict_one`` covered (GAP-2)."""

    methods = list(_inference_methods(model))
    if not methods:
        return

    for x, _ in dataset:
        for name in methods:
            method = getattr(model, name, None)
            if method is None:
                continue
            try:
                method(x)
            except NotImplementedError:
                continue
        break


def river_dataset_checks(model) -> list[typing.Callable]:
    """The checks ``river.checks.yield_checks`` pairs with each dataset for ``model``.

    Replicated from the literal in ``yield_checks``, branch for branch, because
    river does not expose it. ``check_bounded_memory_growth`` is not in the list:
    river runs it separately, on the datasets for ``scale=10``.
    """
    from river import utils
    from river.checks import anomaly, clf, model_selection, time_series
    from river.model_selection.base import ModelSelector
    from river.time_series.base import Forecaster

    dataset_checks = [
        common.check_learn_one,
        common.check_pickling,
        common.check_shuffle_features_no_impact,
        common.check_emerging_features,
        common.check_disappearing_features,
        common.check_radically_disappearing_features,
        common.check_predict_one_pure,
        common.check_predict_one_before_any_learn,
        common.check_no_state_aliasing_with_input,
        common.check_clone_is_independent,
    ]

    if isinstance(model, (base.Transformer, base.SupervisedTransformer)):
        dataset_checks.append(common.check_transform_one)

    if utils.pandas.PANDAS_INSTALLED:
        if isinstance(model, (base.MiniBatchClassifier, base.MiniBatchRegressor)):
            dataset_checks.append(common.check_predict_many_matches_predict_one)
            dataset_checks.append(common.check_learn_many_matches_learn_one)
        if isinstance(model, base.MiniBatchClassifier):
            dataset_checks.append(common.check_predict_proba_many_matches_predict_proba_one)
        if isinstance(model, (base.MiniBatchTransformer, base.MiniBatchSupervisedTransformer)):
            dataset_checks.append(common.check_transform_many_matches_transform_one)
            dataset_checks.append(common.check_learn_many_matches_learn_one)

    if hasattr(model, "debug_one"):
        dataset_checks.append(common.check_debug_one)

    if model._is_stochastic:
        dataset_checks.append(common.check_seeding_is_idempotent)

    if isinstance(model, base.Classifier) and not isinstance(model, base.MultiLabelClassifier):
        allow = river_checks._allow_exception
        dataset_checks.append(allow(clf.check_predict_proba_one, NotImplementedError))
        dataset_checks.append(clf.check_classifier_tracks_seen_labels)
        if not model._multiclass:
            dataset_checks.append(allow(clf.check_predict_proba_one_binary, NotImplementedError))

    if isinstance(model, ModelSelector):
        dataset_checks.append(model_selection.check_model_selection_order_does_not_matter)

    if isinstance(model, base.AnomalyDetector):
        dataset_checks.append(anomaly.check_roc_auc)

    if isinstance(model, Forecaster):
        dataset_checks = [time_series.check_learn_one, time_series.check_forecast]

    return dataset_checks


#: river's GAP-2-gated checks -> the clusterer-aware ones that replace them.
_REPLACEMENTS = {
    common.check_predict_one_pure: check_predict_one_pure,
    common.check_predict_one_before_any_learn: check_predict_one_before_any_learn,
}


class _NoPredictOne:
    """Holds the classes :func:`_without_predict_one` builds, so pickle can find them."""


#: model class -> its rerun subclass. Keyed by the class object, not its name:
#: two classes can share a module and qualname (a class factory, a re-run
#: notebook cell), and each must rerun its own code.
_RERUN_CLASSES: dict[type, type] = {}


def _without_predict_one(model):
    """A fresh clone of ``model`` whose ``predict_one`` returns None where it raised NotImplementedError."""
    cls = type(model)
    sub = _RERUN_CLASSES.get(cls)
    if sub is None:
        stem = "".join(c if c.isalnum() else "_" for c in f"{cls.__module__}.{cls.__qualname__}")
        key = f"{stem}_{len(_RERUN_CLASSES)}"

        def predict_one(self, x, *args, **kwargs):
            try:
                return cls.predict_one(self, x, *args, **kwargs)
            except NotImplementedError:
                return None

        namespace = {
            "predict_one": predict_one,
            "__doc__": cls.__doc__,
            "__module__": __name__,
            "__qualname__": f"{_NoPredictOne.__qualname__}.{key}",
            "__slots__": (),
        }
        sub = types.new_class(cls.__name__, (cls,), exec_body=lambda ns: ns.update(namespace))
        setattr(_NoPredictOne, key, sub)
        _RERUN_CLASSES[cls] = sub
    # Swap the class of a real clone rather than calling sub(**_get_params()):
    # _get_params encodes a nested estimator as a (class, params) tuple.
    fresh = model.clone()
    fresh.__class__ = sub
    return fresh


@functools.cache
def _rerun_without_predict_one(check):
    """``check``, rerun as described in the module docstring when NotImplementedError escapes it."""

    @functools.wraps(check)
    def run(model, dataset):
        # Materialised so the rerun gets the same points untouched; iter()
        # keeps them one-shot, which check_radically_disappearing_features
        # relies on (its second islice continues where the first stopped).
        data = list(dataset)
        pristine = copy.deepcopy(data)
        try:
            check(model, iter(data))
        except NotImplementedError:
            check(_without_predict_one(model), iter(pristine))

    return run


def yield_clusterer_checks(
    model: base.Clusterer, make_datasets=clusterer_datasets
) -> typing.Iterator[typing.Callable]:
    """Every check river yields for ``model``, then the dataset-driven ones it skips.

    The second part is what river would yield if ``_yield_datasets`` had a
    Clusterer branch (GAP-1), with the two GAP-2 checks swapped for the
    clusterer-aware ones, each check bound to every dataset of
    ``make_datasets()`` and then ``check_bounded_memory_growth`` to every
    dataset of ``make_datasets(scale=10)``. Every callable keeps the
    ``__name__`` of the check it wraps, so ``_unit_test_skips`` and reports key
    on it as in river.
    """
    yield from river_checks.yield_checks(model)

    for check in river_dataset_checks(model):
        check = _rerun_without_predict_one(_REPLACEMENTS.get(check, check))
        for dataset in make_datasets():
            yield river_checks._wrapped_partial(check, dataset=dataset)

    memory = _rerun_without_predict_one(common.check_bounded_memory_growth)
    for dataset in make_datasets(scale=10):
        yield river_checks._wrapped_partial(memory, dataset=dataset)


def check_clusterer(model: base.Clusterer, make_datasets=clusterer_datasets):
    """``check_estimator`` for a clusterer: raises on the first failing check.

    The exception carries a note naming the check that raised it.
    """
    for check in yield_clusterer_checks(model, make_datasets):
        if check.__name__ in model._unit_test_skips():
            continue
        try:
            check(model.clone())
        except Exception as exc:
            exc.add_note(f"raised by {check.__name__} on {model.__class__.__name__}")
            raise


def _short(exc: Exception) -> str:
    frame = traceback.extract_tb(exc.__traceback__)[-1]
    lines = str(exc).strip().splitlines()
    head = f"{type(exc).__name__}: {lines[0][:100]}" if lines else type(exc).__name__
    return f"{head} @ {os.path.basename(frame.filename)}:{frame.lineno}"


def run_clusterer_checks(model: base.Clusterer, make_datasets=clusterer_datasets) -> dict[str, str]:
    """Run every check and report each check name's outcome.

    The value is ``"pass"``, ``"fail: <exception>: <message> @ <file>:<line>"``,
    or ``"skip: listed in _unit_test_skips"``. A name run on several datasets
    fails if any of them does; the first failure is kept.
    """
    report: dict[str, str] = {}
    for check in yield_clusterer_checks(model, make_datasets):
        name = check.__name__
        if name in model._unit_test_skips():
            report[name] = "skip: listed in _unit_test_skips"
            continue
        try:
            check(model.clone())
        except Exception as exc:
            if not report.get(name, "").startswith("fail"):
                report[name] = f"fail: {_short(exc)}"
        else:
            report.setdefault(name, "pass")
    return report
