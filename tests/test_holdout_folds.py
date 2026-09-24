"""The out-of-sample fold pool: loadable, never selectable."""
from __future__ import annotations

import pytest

from hpo_pipeline._common import FOLD_GROUPS, FOLDS, HOLDOUT_FOLDS, load_fold


def test_holdout_folds_are_disjoint_from_the_tuning_pool():
    """This is the only property that makes the final comparison a test.

    A fold that tuning can see is a fold the shipped set was chosen on, and
    scoring on it measures the reading twice rather than measuring transfer.
    """
    assert not set(HOLDOUT_FOLDS) & set(FOLD_GROUPS)
    assert not set(HOLDOUT_FOLDS) & set(FOLDS)


def test_the_pool_covers_the_evidence_the_tuning_folds_lack():
    """Multi-dimensional, real, and class-imbalanced -- nothing in FOLD_GROUPS is
    all three at once."""
    assert "insects_abrupt_imbalanced" in HOLDOUT_FOLDS
    assert "insects_gradual_imbalanced" in HOLDOUT_FOLDS


@pytest.mark.parametrize("name", ["dstream_30k_nonconvex"])
def test_synthetic_holdout_folds_load(name):
    dataset = load_fold(name)
    assert dataset.has_labels
    assert dataset.n_features >= 2


def test_every_holdout_name_is_routed_by_load_fold():
    """A name in the tuple that load_fold cannot build is a trap: it only fails
    once someone runs the out-of-sample phase."""
    import inspect

    from hpo_pipeline import _common

    source = inspect.getsource(_common.load_fold)
    for name in HOLDOUT_FOLDS:
        assert f'"{name}"' in source, name


def test_an_unknown_fold_names_both_pools():
    with pytest.raises(KeyError, match="out-of-sample folds"):
        load_fold("does_not_exist")
