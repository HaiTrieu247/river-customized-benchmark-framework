"""Tests for the four carried-over pipeline defects.

Each one is pinned by a case that fails against the previous behaviour, not
just by "the new code returns something".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from hpo_pipeline import analyze
from hpo_pipeline._common import JsonlLock, river_defaults


# --- P0 reads River's constructor, not this framework's preset ---------------
def test_p0_default_is_rivers_own_constructor():
    """The registry preset and River's constructor disagree for DenStream.

    P0 answers "what does a River user get today", so a preset there measures
    something else and shifts every P1 - P0 gap with it.
    """
    from benchmark.algorithms.registry import ALGORITHMS
    from hpo_pipeline.space import build_space

    ctor = river_defaults("DenStream")
    _, preset = ALGORITHMS["DenStream"].factory()
    assert preset != ctor, "if these ever agree, this test has stopped testing anything"

    default = [c for c in build_space(["DenStream"])["DenStream"] if c["is_default"]]
    assert len(default) == 1
    assert default[0]["shape"] == ctor
    assert default[0]["shape"]["epsilon"] == 0.02   # River's, not the preset's 0.5
    assert default[0]["shape"]["beta"] == 0.75      # River's, not the preset's 0.5
    assert default[0]["shape"]["mu"] == 2           # River's, not the preset's 2.5


def test_river_defaults_leaves_out_the_injected_seed():
    assert "seed" not in river_defaults("CluStream")
    assert river_defaults("CluStream")["micro_cluster_r_factor"] == 2


# --- the matrix lock ---------------------------------------------------------
def test_lock_is_exclusive_and_released(tmp_path):
    target = tmp_path / "matrix.jsonl"
    with JsonlLock(target):
        assert (tmp_path / "matrix.jsonl.lock").exists()
        with pytest.raises(SystemExit, match="another run holds this matrix"):
            with JsonlLock(target):
                pass
    assert not (tmp_path / "matrix.jsonl.lock").exists()


def test_lock_records_the_owning_pid(tmp_path):
    target = tmp_path / "matrix.jsonl"
    with JsonlLock(target):
        assert f"pid={os.getpid()}" in (tmp_path / "matrix.jsonl.lock").read_text()


def test_force_overrides_a_stale_lock(tmp_path):
    target = tmp_path / "matrix.jsonl"
    (tmp_path / "matrix.jsonl.lock").write_text("pid=999999 (dead)\n")
    with JsonlLock(target, force=True):
        assert (tmp_path / "matrix.jsonl.lock").exists()
    assert not (tmp_path / "matrix.jsonl.lock").exists()


def test_lock_survives_an_exception_in_the_body(tmp_path):
    target = tmp_path / "matrix.jsonl"
    with pytest.raises(RuntimeError):
        with JsonlLock(target):
            raise RuntimeError("boom")
    assert not (tmp_path / "matrix.jsonl.lock").exists()


# --- hypervolume over the front, not over one point --------------------------
def test_pareto_front_keeps_only_non_dominated_rows():
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.4, 0.4]])
    front = analyze.pareto_front(pts)
    assert len(front) == 3
    assert not any(np.allclose(p, [0.4, 0.4]) for p in front)


def test_pareto_front_keeps_duplicates_of_a_non_dominated_point():
    pts = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert len(analyze.pareto_front(pts)) == 2


def test_hv3_of_one_box_is_its_volume():
    assert analyze._hv3(np.array([[0.5, 0.4, 0.2]])) == pytest.approx(0.04)


def test_hv3_of_two_disjoint_corners_is_their_union_not_the_larger():
    """(1, 0.1, 1) and (0.1, 1, 1): each box has volume 0.1, the union 0.19.

    A max-over-points implementation returns 0.1 here. That is the defect: a
    second, differently-shaped non-dominated configuration has to make the
    number go up, or the hypervolume is not measuring the front.
    """
    pts = np.array([[1.0, 0.1, 1.0], [0.1, 1.0, 1.0]])
    assert analyze._hv3(pts) == pytest.approx(0.19)
    assert analyze._hv3(pts) > max(float(np.prod(p)) for p in pts)


def test_hv3_ignores_a_dominated_point():
    front = np.array([[1.0, 1.0, 1.0]])
    withdom = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    assert analyze._hv3(front) == pytest.approx(analyze._hv3(withdom)) == pytest.approx(1.0)


def test_hv3_of_nothing_is_zero():
    assert analyze._hv3(np.empty((0, 3))) == 0.0


def test_hypervolume_grows_when_a_cheaper_configuration_is_added():
    """The end-to-end version of the same property, in the real columns."""
    base = [{"algo": "A", "dataset": "d1", "status": "ok", "feasible": True,
             "ari": 0.5, "runtime_s": 450.0, "peak_memory_mb": 1024.0}]
    cheaper = base + [{"algo": "A", "dataset": "d1", "status": "ok", "feasible": True,
                       "ari": 0.45, "runtime_s": 10.0, "peak_memory_mb": 10.0}]
    one = analyze.hypervolume(pd.DataFrame(base))
    two = analyze.hypervolume(pd.DataFrame(cheaper))
    assert two.hypervolume.iloc[0] > one.hypervolume.iloc[0]
    assert int(one.front_size.iloc[0]) == 1
    assert int(two.front_size.iloc[0]) == 2


def test_hypervolume_reports_the_front_size_beside_the_volume():
    df = pd.DataFrame([{"algo": "A", "dataset": "d1", "status": "ok", "feasible": True,
                        "ari": 0.5, "runtime_s": 1.0, "peak_memory_mb": 1.0}])
    out = analyze.hypervolume(df)
    assert set(["hypervolume", "front_size", "n_feasible"]) <= set(out.columns)


# --- progress is not held back by one slow run -------------------------------
def test_build_matrix_consumes_results_as_they_complete():
    """`pool.map` yields in submission order; `as_completed` does not.

    Checked on the module source rather than by timing a real grid: the failure
    it prevents (a slow run freezing both the log and the file writes behind it)
    only shows up at a scale a unit test should not run.
    """
    from hpo_pipeline import build_matrix

    text = open(build_matrix.__file__, encoding="utf-8").read()
    assert "as_completed(futures)" in text
    assert "pool.map(" not in text
