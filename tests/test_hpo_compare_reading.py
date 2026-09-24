"""Tests for the reading changes: the tunability gate, headroom pooling, the
label-free-criterion gate, the degeneracy flag, and the pinned-parameter
mechanism. No datasets, no runs -- every frame here is hand-built so the
expected number can be computed by hand in the docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hpo_pipeline import hpo_compare as H


def frame(rows, grid="ladder"):
    """rows: (algo, dataset, config_id, level, theta, ari, [is_default, criterion])."""
    out = []
    for r in rows:
        algo, ds, cid, lv, th, ari = r[:6]
        is_default = r[6] if len(r) > 6 else False
        row = {
            "algo": algo, "dataset": ds, "config_id": cid, "granularity_level": lv,
            "theta": th, "ari": ari, "runtime_s": 1.0, "status": "ok",
            "grid": "default" if is_default else grid, "is_default": is_default,
            "throughput_sps": 5000.0,
        }
        if len(r) > 7:
            row["matr_trace_win"] = r[7]
        out.append(row)
    return pd.DataFrame(out)


# --- B2: the tunability gate -------------------------------------------------
def test_gate_needs_both_a_ceiling_and_a_gain_over_the_default():
    """easy: ceiling 0.60 over a 0.10 default -> tunable.
    flat:  ceiling 0.60 over a 0.59 default -> gain 0.01 < 0.02, not tunable.
    tiny:  ceiling 0.04 over a 0.00 default -> ceiling < 0.05, not tunable.
    """
    df = frame([
        ("A", "easy", "d", 1, 1, 0.10, True), ("A", "easy", "c1", 1, 1, 0.60),
        ("A", "flat", "d", 1, 1, 0.59, True), ("A", "flat", "c1", 1, 1, 0.60),
        ("A", "tiny", "d", 1, 1, 0.00, True), ("A", "tiny", "c1", 1, 1, 0.04),
    ])
    ref = H.fold_reference(df)
    assert bool(ref.loc["easy", "tunable"]) is True
    assert bool(ref.loc["flat", "tunable"]) is False
    assert bool(ref.loc["tiny", "tunable"]) is False
    assert H.tunable_folds(df, ["easy", "flat", "tiny"]) == ["easy"]


def test_gate_is_false_not_a_crash_when_the_default_never_ran():
    df = frame([("A", "d1", "c1", 1, 1, 0.9)])
    ref = H.fold_reference(df)
    assert np.isnan(ref.loc["d1", "default"])
    assert bool(ref.loc["d1", "tunable"]) is False


def test_gate_thresholds_are_the_documented_constants():
    # They are this project's choice, so a change must be deliberate, not a drift.
    assert H.MIN_CEILING == 0.05
    assert H.MIN_CEILING_GAIN == 0.02


# --- B1: headroom pooling ----------------------------------------------------
def _pool(df, folds, how):
    return H._aggregate(df[df.algo == "A"], folds, how, H.fold_reference(df)).set_index(
        "config_id")["score"]


def test_headroom_normalises_each_dataset_by_its_own_range():
    """big: default 0.0, ceiling 1.0. small: default 0.0, ceiling 0.1.

    ``lo`` scores 0.50 on big and 0.01 on small -> headroom 0.50 and 0.10.
    ``hi`` scores 0.20 on big and 0.09 on small -> headroom 0.20 and 0.90.
    Mean ARI crowns ``lo`` (0.255 vs 0.145); headroom crowns ``hi``
    (0.55 vs 0.30), which is the whole point of the option.
    """
    df = frame([
        ("A", "big", "d", 1, 1, 0.00, True), ("A", "small", "d", 1, 1, 0.00, True),
        ("A", "big", "lo", 1, 1, 0.50), ("A", "small", "lo", 1, 2, 0.01),
        ("A", "big", "hi", 1, 1, 0.20), ("A", "small", "hi", 1, 2, 0.09),
        ("A", "big", "top", 1, 1, 1.00), ("A", "small", "top", 1, 2, 0.10),
    ])
    folds = ["big", "small"]
    mean = _pool(df, folds, "mean")
    assert mean["lo"] > mean["hi"]
    head = _pool(df, folds, "headroom")
    assert head["hi"] == pytest.approx(0.55)
    assert head["lo"] == pytest.approx(0.30)
    assert head["hi"] > head["lo"]
    # the ceiling config is headroom 1.0 on both by construction
    assert head["top"] == pytest.approx(1.0)


def test_headroom_drops_untunable_datasets_from_the_pool():
    """``noise`` fails the gate, so it must not move the pooled score at all."""
    base = [
        ("A", "good", "d", 1, 1, 0.00, True), ("A", "good", "c1", 1, 1, 0.50),
        ("A", "good", "top", 1, 1, 1.00),
        ("A", "noise", "d", 1, 1, 0.00, True), ("A", "noise", "top", 1, 1, 0.01),
    ]
    with_noise = frame(base + [("A", "noise", "c1", 1, 1, 0.009)])
    against = frame(base + [("A", "noise", "c1", 1, 1, 0.000)])
    a = _pool(with_noise, ["good", "noise"], "headroom")["c1"]
    b = _pool(against, ["good", "noise"], "headroom")["c1"]
    assert a == pytest.approx(b) == pytest.approx(0.50)


def test_headroom_returns_nothing_when_no_dataset_passes_the_gate():
    """Refusing to pick is the honest answer, not falling back to raw ARI."""
    df = frame([
        ("A", "d1", "d", 1, 1, 0.00, True), ("A", "d1", "c1", 1, 1, 0.01),
        ("A", "d2", "d", 1, 1, 0.00, True), ("A", "d2", "c1", 1, 1, 0.02),
    ])
    assert H._aggregate(df, ["d1", "d2"], "headroom", H.fold_reference(df)).empty
    assert H.pick_all(df, ["d1", "d2"], "headroom",
                      {"A": H.fold_reference(df)}) == {}


def test_headroom_is_negative_below_the_default():
    df = frame([
        ("A", "d1", "d", 1, 1, 0.20, True), ("A", "d1", "top", 1, 1, 0.40),
        ("A", "d1", "bad", 1, 1, 0.10),
    ])
    assert _pool(df, ["d1"], "headroom")["bad"] == pytest.approx(-0.5)


def test_headroom_is_a_registered_aggregate_and_bad_names_still_raise():
    assert "headroom" in H.AGGREGATES
    with pytest.raises(ValueError):
        H._aggregate(frame([("A", "d1", "c", 1, 1, 0.1)]), ["d1"], "nonsense")


def test_the_reference_is_built_before_the_fold_filter():
    """A LODO round must use the same yardstick as every other round.

    Scoring ``c1`` on ``d1`` alone has to give the same headroom whether the
    round's optimisation set was {d1} or {d1, d2}, because the reference comes
    from the whole frame either way.
    """
    df = frame([
        ("A", "d1", "d", 1, 1, 0.00, True), ("A", "d1", "top", 1, 1, 1.00),
        ("A", "d1", "c1", 1, 1, 0.40),
        ("A", "d2", "d", 1, 1, 0.00, True), ("A", "d2", "top", 1, 1, 1.00),
        ("A", "d2", "c1", 1, 1, 0.80),
    ])
    ref = H.fold_reference(df)
    one = H._aggregate(df, ["d1"], "headroom", ref).set_index("config_id")["score"]
    both = H._aggregate(df, ["d1", "d2"], "headroom", ref).set_index("config_id")["score"]
    assert one["c1"] == pytest.approx(0.40)
    assert both["c1"] == pytest.approx(0.60)


# --- B4: is the label-free criterion usable here? ----------------------------
def _curve_frame(pairs, dataset="d1"):
    """pairs: (level, ari, criterion) at a single shape point."""
    return frame([("A", dataset, f"c{lv}", lv, 1, ari, False, crit)
                  for lv, ari, crit in pairs])


def test_criterion_with_its_peak_on_the_best_rung_has_zero_regret():
    df = _curve_frame([(1, 0.1, -9.0), (2, 0.5, -1.0), (3, 0.2, -5.0)])
    r = H.criterion_regret(df, "A", "d1", "matr_trace_win", 3)
    assert r["n_curves"] == 1
    assert r["median"] == pytest.approx(0.0)
    assert r["n_negative"] == 0


def test_criterion_pointing_at_the_worst_rung_is_caught():
    """ARI peaks at level 2 (0.50); the criterion peaks at level 3 (ARI 0.02).

    regret = 0.50 - 0.02 = 0.48, well past the 0.05 bar -- this is the shuttle
    failure in miniature, and it is invisible without labels.
    """
    df = _curve_frame([(1, 0.10, -9.0), (2, 0.50, -8.0), (3, 0.02, -1.0)])
    r = H.criterion_regret(df, "A", "d1", "matr_trace_win", 3)
    assert r["median"] == pytest.approx(0.48)
    assert r["median"] > H.MAX_CRITERION_REGRET
    assert r["n_negative"] == 1


def test_regret_is_measured_per_curve_not_over_the_whole_grid():
    """Two shape points: one clean, one inverted. The median must see both."""
    clean = [("A", "d1", "a1", lv, 1, ari, False, crit)
             for lv, ari, crit in [(1, 0.1, -9.0), (2, 0.5, -1.0)]]
    broken = [("A", "d1", "b1", lv, 2, ari, False, crit)
              for lv, ari, crit in [(1, 0.5, -9.0), (2, 0.1, -1.0)]]
    r = H.criterion_regret(frame(clean + broken), "A", "d1", "matr_trace_win", 2)
    assert r["n_curves"] == 2
    assert r["max"] == pytest.approx(0.4)
    assert r["median"] == pytest.approx(0.2)


def test_regret_is_empty_rather_than_zero_when_a_curve_is_broken():
    df = _curve_frame([(1, 0.1, -9.0), (2, 0.5, float("nan"))])
    r = H.criterion_regret(df, "A", "d1", "matr_trace_win", 2)
    assert r["n_curves"] == 0
    assert np.isnan(r["median"])


# --- A1: the degeneracy flag -------------------------------------------------
def test_one_label_for_the_whole_stream_is_degenerate():
    assert H._degenerate({"largest_cluster_share": 1.0}) is True
    assert H._degenerate({"n_predicted_clusters": 2.0, "homogeneity": 0.0}) is True
    assert H._degenerate({"n_predicted_clusters": 1.0, "homogeneity": 0.0}) is True


def test_a_real_partition_is_not_degenerate():
    assert H._degenerate({"largest_cluster_share": 0.65,
                          "n_predicted_clusters": 3.0, "homogeneity": 0.02}) is False
    # two clusters that actually carry class information are fine
    assert H._degenerate({"n_predicted_clusters": 2.0, "homogeneity": 0.3}) is False


def test_degeneracy_is_false_on_missing_columns_rather_than_raising():
    assert H._degenerate({}) is False
    assert H._degenerate({"largest_cluster_share": float("nan")}) is False


# --- C1: pinned parameters ---------------------------------------------------
def test_denstream_pins_beta_and_sweeps_lambda():
    spec = H.GRIDS["DenStream"]
    assert spec["fixed"] == {"beta": 0.75}
    for grid in ("raw", "ladder"):
        assert "beta" not in spec["shape"][grid]
        # The lambda ladder starts at River's default and goes UP: the pilot
        # measured no movement below 0.25 and the Pareto gains above 1.
        assert spec["shape"][grid]["decaying_factor"] == [0.25, 1.0, 2.0]


def test_a_pinned_value_reaches_the_shape_and_so_the_config_id():
    """Pinned is not the same as inherited: it has to be visible in the row."""
    accepted, _ = H.build_grid("DenStream", "ladder", H.GRIDS["DenStream"])
    assert accepted, "grid is empty"
    assert all(c["shape"]["beta"] == 0.75 for c in accepted)
    # and changing the pin changes every config_id
    other = dict(H.GRIDS["DenStream"], fixed={"beta": 0.5})
    moved, _ = H.build_grid("DenStream", "ladder", other)
    assert {c["config_id"] for c in accepted}.isdisjoint({c["config_id"] for c in moved})


def test_pinning_beta_leaves_no_grid_point_rejected():
    """beta*mu > 1 held at beta=0.75 for every mu on the ladder, so the
    constraint can no longer throw away a third of the DenStream budget."""
    for grid in ("raw", "ladder"):
        accepted, rejected = H.build_grid("DenStream", grid, H.GRIDS["DenStream"])
        assert rejected == []
        assert len(accepted) == (len(H.GRIDS["DenStream"]["levels"][grid])
                                 * 4 * 3)  # 4 mu x 3 lambda


def test_a_parameter_cannot_be_pinned_and_swept_at_once():
    bad = dict(H.GRIDS["DenStream"],
               shape={g: {**v, "beta": [0.2, 0.5]}
                      for g, v in H.GRIDS["DenStream"]["shape"].items()})
    with pytest.raises(ValueError, match="pinned and swept"):
        H.build_grid("DenStream", "ladder", bad)


def test_the_constraint_still_guards_a_grid_override():
    """It cannot fire on the shipped grid, but --grids must not get past it."""
    bad = dict(H.GRIDS["DenStream"], fixed={"beta": 0.2},
               shape={g: {"mu": [2], "decaying_factor": [0.25]}
                      for g in ("raw", "ladder")})
    accepted, rejected = H.build_grid("DenStream", "ladder", bad)
    assert accepted == []
    # one rejection per resolution level, since the shape point is the same
    assert len(rejected) == len(H.GRIDS["DenStream"]["levels"]["ladder"])
    assert {tuple(r["violates"]) for r in rejected} == {("beta * mu > 1",)}


def test_merge_grids_handles_the_flat_fixed_key():
    merged = H.merge_grids(H.GRIDS, {"DenStream": {"fixed": {"beta": 0.5}}})
    assert merged["DenStream"]["fixed"] == {"beta": 0.5}
    assert H.GRIDS["DenStream"]["fixed"] == {"beta": 0.75}, "base was mutated"


def test_clustream_grid_is_unchanged_by_the_denstream_work():
    spec = H.GRIDS["CluStream"]
    assert sorted(spec["shape"]["ladder"]) == ["micro_cluster_r_factor", "micro_ratio"]
    assert spec["levels"]["ladder"] == [2, 4, 8, 16, 32, 64]
