"""conf_pipeline.fidelity: controlled scenarios, the label-identity proxy and its validation on HPStream,
and the command's outputs. Tiny sizes throughout; run with PYTHONHASHSEED=0."""

from __future__ import annotations

import csv
import dataclasses
import json

import numpy as np
import pytest

from conf_pipeline import fidelity as F
from conf_pipeline.spaces import start_params

TINY = F.StreamShape(h=60, n_windows=5, d=2)


def _first(y, label):
    idx = np.flatnonzero(y == label)
    return int(idx[0]) if len(idx) else None


def _last(y, label):
    idx = np.flatnonzero(y == label)
    return int(idx[-1]) if len(idx) else None


# --- scenarios ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", F.SCENARIOS)
def test_a_scenario_is_deterministic_per_seed_and_differs_across_seeds(name):
    a, b, c = F.make_scenario(name, 1, TINY), F.make_scenario(name, 1, TINY), F.make_scenario(name, 2, TINY)
    assert a.X.shape == (TINY.n, TINY.d) and len(a.y) == TINY.n
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and a.events == b.events
    assert not np.array_equal(a.X, c.X)
    assert a.phases[0] == "warmup" and len(a.phases) == TINY.n_windows


def test_birth_and_deletion_happen_at_the_first_point_of_the_event_window():
    w = TINY.event_window()
    t = w * TINY.h
    birth = F.make_scenario("birth", 3, TINY)
    assert birth.events == [{"event": "birth", "t": t, "window": w, "labels": [2]}]
    assert _first(birth.y, 2) >= t and (birth.y[t:t + TINY.h] == 2).any()
    assert set(birth.y[:t].tolist()) == {0, 1}
    deletion = F.make_scenario("deletion", 3, TINY)
    assert deletion.events[0]["event"] == "deletion" and deletion.events[0]["t"] == t
    assert _last(deletion.y, 2) < t and set(deletion.y[t:].tolist()) == {0, 1}
    assert deletion.probe.shape == (TINY.probe_points, TINY.d)
    assert birth.phases == ["warmup"] + ["pre"] * (w - 1) + ["post"] * (TINY.n_windows - w)


def test_merge_ends_in_one_class_at_the_midpoint_and_split_in_two_children():
    first, end = TINY.transition()
    m = F.make_scenario("merge", 4, TINY)
    tm = end * TINY.h
    assert _last(m.y, 0) < tm and _last(m.y, 1) < tm and _first(m.y, 3) >= tm
    assert set(m.y[tm:].tolist()) == {2, 3}
    # the merged class sits at the midpoint of where the pair started, far from the static cluster in every
    # coordinate
    start_a = m.X[: first * TINY.h][m.y[: first * TINY.h] == 0].mean(0)
    start_b = m.X[: first * TINY.h][m.y[: first * TINY.h] == 1].mean(0)
    merged, other = m.X[m.y == 3].mean(0), m.X[m.y == 2].mean(0)
    assert np.abs(merged - (start_a + start_b) / 2).max() < 1.5 * TINY.sigma
    assert np.abs(merged - other).min() > 1.0 * TINY.separation * TINY.sigma
    assert m.phases[first] == "transition" and m.phases[end] == "post"
    s = F.make_scenario("split", 4, TINY)
    ts = first * TINY.h
    assert _last(s.y, 0) < ts and _first(s.y, 3) >= ts and _first(s.y, 4) >= ts
    late = s.y[end * TINY.h:]
    q1, q2 = s.X[end * TINY.h:][late == 3].mean(0), s.X[end * TINY.h:][late == 4].mean(0)
    assert np.abs(q1 - q2).min() > 0.5 * TINY.separation * TINY.sigma


def test_identity_stress_retires_the_oldest_and_never_reuses_a_true_label():
    sc = F.make_scenario("identity_stress", 5, TINY)
    births = sc.meta["births"]
    assert [b["t"] for b in births] == list(range(TINY.h, TINY.n, TINY.period))
    assert len({b["label"] for b in births}) == len(births)
    for b in births:
        assert _first(sc.y, b["label"]) >= b["t"]
    for e in (e for e in sc.events if e["event"] == "death"):
        assert _last(sc.y, e["labels"][0]) < e["t"]
    # three clusters at a time
    for t in range(0, TINY.n, TINY.period):
        assert len(set(sc.y[t:t + TINY.period].tolist())) <= TINY.stress_active


def test_the_latin_layout_separates_every_pair_in_every_coordinate():
    L = F._latin(6, 3, 8.0, np.random.default_rng(0))
    for i in range(6):
        for j in range(i + 1, 6):
            assert np.abs(L[i] - L[j]).min() >= 8.0


# --- the identity proxy ----------------------------------------------------------------------------
def test_the_proxy_flags_a_label_whose_majority_class_changes_and_nothing_else():
    truth = ["a"] * 20 + ["b"] * 20
    out = F.label_class_changes([0] * 40, truth, segments=4, min_support=5, min_share=0.6)
    assert out["flagged"] == {0} and out["points"] == 40
    mixed = F.label_class_changes([0] * 40, ["a", "b"] * 20, segments=4, min_support=5, min_share=0.6)
    assert mixed["flagged"] == set()  # one label over two classes all along is not a change
    clean = F.label_class_changes([0] * 20 + [1] * 20, truth, segments=4, min_support=5, min_share=0.6)
    assert clean["flagged"] == set()


def test_exact_reuse_reads_an_hpstream_label_that_covers_two_variant_labels():
    river = [0, 0, 0, 0, 1, 1]
    stable = [0, 0, 10, 10, 1, 1]
    truth = ["a", "a", "b", "b", "c", "c"]
    ex = F.exact_reuse(river, stable, truth, min_support=2, min_share=0.6)
    assert ex["reused"] == {0} and ex["cross_any"] == {0} and ex["cross_supported"] == {0}
    assert ex["function"]
    same_class = F.exact_reuse(river, stable, ["a"] * 4 + ["c", "c"], min_support=2, min_share=0.6)
    assert same_class["reused"] == {0} and same_class["cross_any"] == set()
    assert not F.exact_reuse([0, 1], [5, 5], ["a", "a"])["function"]


def _cells(algos, scenario, seeds, shape, config="default"):
    shd = dataclasses.asdict(shape)
    out = {}
    for seed in seeds:
        for algo in algos:
            r = F.run_cell({"scenario": scenario, "seed": seed, "config": config, "algo": algo, "shape": shd,
                            "model_seed": 201})
            out[r["key"]] = r
    return out


def test_the_proxy_is_validated_on_hpstream_against_its_stable_variant():
    shape = F.StreamShape(h=100, n_windows=5, d=5)
    cells = _cells(("HPStream", "HPStreamStableLabels"), "identity_stress", (1,), shape)
    summary, detail = F.identity_validation(cells, shape)
    assert len(summary) == 1 and detail
    s = summary[0]
    # the premise: both models hold the same clusters, so every variant label sits under one HPStream label
    assert s["stable_to_river_is_function"]
    assert s["n_reused"] > 0, "HPStream reuses no label on this case: the validation has nothing to read"
    assert s["n_cross_any"] <= s["n_reused"] and s["n_cross_supported"] <= s["n_cross_any"]
    assert s["n_cross_readable"] <= s["n_cross_any"] and s["n_proxy"] <= s["n_proxy_readable"]
    for ref in ("reused", "cross_any", "cross_supported", "cross_readable"):
        assert s[f"tp_{ref}"] + s[f"fn_{ref}"] == s[f"n_{ref}"]
        assert s[f"tp_{ref}"] + s[f"fp_{ref}"] == s["n_proxy"]
    # a flag is a readable label, so every cross-class flag is a readable cross-class reuse
    assert s["tp_cross_readable"] == s["tp_cross_any"]
    # the variant never reuses a label: on its own sequential labels the proxy is read against nothing
    assert all(r["function"] for r in detail)
    stable_windows = cells[("identity_stress", 1, "default", "HPStreamStableLabels")]["windows"]
    assert sum(r["proxy_flags_seq"] for r in stable_windows) == s["n_proxy_stable"]


# --- cells and checks ------------------------------------------------------------------------------
def test_a_cell_is_built_as_run_py_builds_a_member():
    sc = F.make_scenario("stationary", 1, TINY)
    w0 = sc.points()[:TINY.h]
    params, s, rec = F.cell_params("DenStream", "default", 300, 2, w0)
    assert params == start_params("DenStream", 300, d=2) and params["n_samples_init"] == 200
    scaled, s2, rec2 = F.cell_params("DenStream", "scaled_default", 300, 2, w0)
    assert s2 == s > 0 and scaled["epsilon"] == pytest.approx(params["epsilon"] * s)
    assert rec2 == {"epsilon": {"default": params["epsilon"], "s": s, "value": scaled["epsilon"]}}


def test_checks_say_not_applicable_by_design_and_by_capacity():
    shd = dataclasses.asdict(TINY)
    stk = F.run_cell({"scenario": "stationary", "seed": 1, "config": "default", "algo": "STREAMKMeans",
                      "shape": shd, "model_seed": 201})
    assert stk["cell"]["capacity"] == 2  # River's default n_clusters, kept by the protocol's start
    assert {c["status"] for c in stk["checks"]} == {"not-applicable"}
    assert all("capacity" in c["reason"] for c in stk["checks"])
    birch = F.run_cell({"scenario": "deletion", "seed": 1, "config": "default", "algo": "BIRCH",
                        "shape": shd, "model_seed": 201})
    (ck,) = birch["checks"]
    assert ck["status"] == "not-applicable" and "forgetting" in ck["reason"]
    clu = F.run_cell({"scenario": "stationary", "seed": 1, "config": "default", "algo": "CluStream",
                      "shape": shd, "model_seed": 201})
    assert {c["status"] for c in clu["checks"]} <= set(F.STATUSES) - {"not-applicable"}
    w = clu["windows"]
    assert [r["window"] for r in w] == list(range(1, TINY.n_windows))
    assert all(r["n_points"] == TINY.h for r in w)


def _fake_snaps(sc, label_of, probe_label=None):
    """Snapshots that label every point by ``label_of(window, true class)`` and every probe ``probe_label(w)``."""
    out = {}
    for w in range(1, sc.shape.n_windows):
        truth = sc.y[w * sc.shape.h:(w + 1) * sc.shape.h].tolist()
        out[w] = {"snap": [label_of(w, t) for t in truth], "truth": truth,
                  "probe": [probe_label(w)] * sc.shape.probe_points if probe_label else None}
    return out


def test_the_checks_read_births_and_deletions_off_the_snapshots():
    th = dict(F.THRESHOLDS)
    birth = F.make_scenario("birth", 1, TINY)
    w0 = birth.meta["window"]
    told = F.evaluate_checks(birth, _fake_snaps(birth, lambda w, t: t), [], "CluStream", "default",
                             {"n_macro_clusters": 5}, th)
    assert told[0]["status"] == "pass" and told[0]["value"] == 0
    late = F.evaluate_checks(birth, _fake_snaps(birth, lambda w, t: t if w >= w0 + 2 or t != 2 else 0), [],
                             "CluStream", "default", {"n_macro_clusters": 5}, th)
    assert late[0]["status"] == "fail" and late[0]["value"] == 2   # told apart, but after the allowed delay
    dele = F.make_scenario("deletion", 1, TINY)
    d0 = dele.meta["window"]
    gone = F.evaluate_checks(dele, _fake_snaps(dele, lambda w, t: t, lambda w: 0 if w >= d0 else 2), [],
                             "DBSTREAM", "default", {}, th)
    assert gone[0]["status"] == "pass" and gone[0]["value"] == 0
    kept = F.evaluate_checks(dele, _fake_snaps(dele, lambda w, t: t, lambda w: 2), [], "DBSTREAM", "default", {}, th)
    assert kept[0]["status"] == "fail" and kept[0]["value"] is None
    never_apart = F.evaluate_checks(dele, _fake_snaps(dele, lambda w, t: 0, lambda w: 0), [], "DBSTREAM",
                                    "default", {}, th)
    assert never_apart[0]["status"] == "undetermined"


def test_the_audit_references_point_at_the_code_they_quote():
    bad = [r for r in F.verify_refs() if r["status"] != "ok"]
    assert not bad, bad
    assert {e["algo"] for e in F.AUDIT} == set(F.ALGOS)


# --- the command ------------------------------------------------------------------------------------
def test_the_command_writes_its_outputs_and_never_overwrites(tmp_path):
    argv = ["--out-dir", str(tmp_path), "--algos", "HPStream,HPStreamStableLabels,STREAMKMeans,BIRCH",
            "--seeds", "1", "--config", "default,scaled_default", "--h", "60", "--windows", "5",
            "--birch-factors", "1", "--workers", "1", "--scenarios", "stationary,identity_stress,deletion"]
    report = F.main(argv)
    fid = tmp_path / "fidelity"
    for name in ("windows.csv", "checks.csv", "cells.csv", "scenarios.csv", "identity_proxy.csv",
                 "identity_windows.csv", "birch_crosslib.csv", "independent_implementations.csv", "audit.csv",
                 "audit_refs.csv", "river_checks.csv", "run.json", "fidelity.md"):
        assert (fid / name).exists(), name
    with open(fid / "checks.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and {r["status"] for r in rows} <= set(F.STATUSES)
    # HPStream has no distance parameter: its scaled_default cells read as its default ones, marked so
    hp = [r for r in rows if r["algo"] == "HPStream" and r["config"] == "scaled_default"]
    assert hp and all(r["identical_to"] == "default" for r in hp)
    with open(fid / "identity_proxy.csv", newline="") as fh:
        ident = list(csv.DictReader(fh))
    assert {r["config"] for r in ident} == {"default"} and len(ident) == 3
    with open(fid / "birch_crosslib.csv", newline="") as fh:
        birch = list(csv.DictReader(fh))
    assert {r["variant"] for r in birch} == set(F.BIRCH_VARIANTS) and not any(r["error"] for r in birch)
    meta = json.loads((fid / "run.json").read_text())
    assert meta["shape"]["h"] == 60 and meta["thresholds"] == F.THRESHOLDS
    md = report.read_text()
    for head in ("## 1. What was run", "## 2. Expected-behaviour checks", "## 3. Label identity",
                 "## 4. Cross-library agreement", "## 5. Audit per implementation", "## 6. What this does not show"):
        assert head in md
    with pytest.raises(SystemExit):
        F.main(argv)


def test_the_command_rejects_unknown_names(tmp_path):
    with pytest.raises(SystemExit):
        F.main(["--out-dir", str(tmp_path), "--algos", "NotAnAlgorithm"])
    with pytest.raises(SystemExit):
        F.main(["--out-dir", str(tmp_path / "b"), "--threshold", "nonsense=1"])
