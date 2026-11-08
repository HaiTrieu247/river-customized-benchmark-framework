"""Point support: which points a reading rests on, recorded per window, and how two arms' common
windows compare on it (``window.support_fingerprint``, ``analysis_support.point_support_status``)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from benchmark.algorithms.adapter import make_adapter
from benchmark.tuning.confstream import ConfStreamEnsemble, EnsembleConfig
from benchmark.tuning.space import Param, ParamSpace, SpaceContext
from benchmark.tuning.window import (WindowBuffer, external_support, silhouette_status, silhouette_support,
                                     support_fingerprint)
from conf_pipeline import analysis_support as AS
from conf_pipeline import report


# --- the masks follow the scoring policy as it is -------------------------------------------------

def test_the_masks_leave_out_none_and_keep_a_sentinel_and_a_fallback_label():
    labels = [0, None, -1, 0, 1, None, 1]          # -1: a sentinel kept as a label (the default policy)
    ys = ["a", "a", None, "b", "b", "b", "a"]      # a point without a true label is not read externally
    assert silhouette_support(labels) == [True, False, True, True, True, False, True]
    assert external_support(ys, labels) == [True, False, False, True, True, False, True]
    xs = [{"u": float(i % 3), "v": float(i)} for i in range(len(labels))]
    assert silhouette_status(xs, labels).n_points == sum(silhouette_support(labels))


def test_a_fingerprint_names_the_positions_not_the_count_and_is_the_same_in_another_process():
    a = [True, True, False, True]
    b = [True, False, True, True]                  # three points as well, other positions
    assert sum(a) == sum(b) and support_fingerprint(a) != support_fingerprint(b)
    assert support_fingerprint(a) == support_fingerprint(list(a))
    assert support_fingerprint([True] * 3) != support_fingerprint([True] * 4)   # the length is in it
    code = ("from benchmark.tuning.window import support_fingerprint as f;"
            "print(f([True, True, False, True]))")
    for seed in ("0", "1", "random"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             env={"PYTHONHASHSEED": seed, "PATH": ""}, cwd=str(AS.C.PIPELINE_ROOT.parent))
        assert out.stdout.strip() == support_fingerprint(a)
    assert support_fingerprint(a).startswith("p1-") and len(support_fingerprint(a)) == 3 + 16


class DenStream:
    """A scripted model named after DenStream, so the adapter reads ``-1`` as its sentinel."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def predict_one(self, x):
        v = self.script[self.i % len(self.script)]
        self.i += 1
        if v == "raise":
            raise ValueError("cold")
        return v

    def learn_one(self, x):
        pass


SPACE = ParamSpace("KMeans", (Param("n_clusters", "log_int", 2, 8),))


def _member_outcome(script, ys, drop_sentinel=False):
    def make(params, seed):
        ad = make_adapter("DenStream", DenStream(script))
        ad.drop_sentinel_predictions = drop_sentinel
        return ad

    ens = ConfStreamEnsemble(space=SPACE, ctx=SpaceContext(s=1.0, d=2), make_adapter=make,
                             start_params=[{"n_clusters": 2}],
                             # the support of the sequential labels is what this reads
                             config=EnsembleConfig(e_size=1, e_test=0, score_partition="sequential"),
                             rng=np.random.default_rng(0), seed=0)
    buf = WindowBuffer(len(ys))
    for i, y in enumerate(ys):
        buf.push({"u": float(i % 3), "v": float(i // 3)}, y)
    ens.initialise(buf, 0)
    m = ens.active[0]
    m.adapter.model.i = 0                          # the warm start drew nothing from predict_one; start the script
    return ens._run_member(m, buf, 1), buf


@pytest.mark.parametrize("drop_sentinel", [False, True])
def test_a_member_window_records_the_points_its_readings_used(drop_sentinel):
    script = [0, 1, -1, "raise", 1, 0, -1, 1]
    ys = [0, 1, 0, None, 1, 0, 1, 1]
    o, buf = _member_outcome(script, ys, drop_sentinel)
    preds = [0, 1, None if drop_sentinel else -1, 0, 1, 0, None if drop_sentinel else -1, 1]   # fallback 0
    assert o.n_points_input == len(ys) and o.n_fallback == 1 and o.n_sentinel == 2
    assert o.n_points_scored == sum(silhouette_support(preds)) == (6 if drop_sentinel else 8)
    assert o.support_sha == support_fingerprint(silhouette_support(preds))
    mask = external_support(ys, preds)
    assert o.n_points_external == sum(mask) and o.external_support_sha == support_fingerprint(mask)
    # the ARI of the window is the ARI of exactly those points
    from river import metrics

    ari = metrics.AdjustedRand()
    for y, p, keep in zip(ys, preds, mask):
        if keep:
            ari.update(y, p)
    assert o.window_metrics["ari"] == pytest.approx(ari.get())
    assert o.to_dict()["external_support_sha"] == o.external_support_sha


# --- two arms' common windows by point support ----------------------------------------------------

MASK_A = support_fingerprint([True, True, False, True])
MASK_B = support_fingerprint([True, False, True, True])   # the same count of points as MASK_A


def _tw(rows):
    base = {"algo": "DenStream", "fold": "covtype", "config_id": None, "model_seed": None}
    out = [{**base, **r} for r in rows]
    df = pd.DataFrame(out)
    for k in ("seed", "model_seed", "config_id"):
        df[k] = pd.Series([AS.seed_key(r.get(k)) for r in out], dtype=object)
    return df


def _row(arm, seed, window, value, sha, start=None):
    start = window * 10 if start is None else start
    return {"arm": arm, "seed": seed, "window": window, "lag_ari_win": value, "lag_external_support_sha": sha,
            "start": start, "end": start + 9}


def test_the_same_count_of_points_on_other_positions_is_a_differing_support():
    tw = _tw([_row("confstream", 101, 20, 0.5, MASK_A), _row("random_online", 101, 20, 0.3, MASK_B),
              _row("confstream", 101, 21, 0.6, MASK_A), _row("random_online", 101, 21, 0.2, MASK_A)])
    p = AS.paired_contrast(tw, "confstream", "random_online", "lag_ari_win").iloc[0]
    assert p["n_common"] == 2 and p["n_point_differs"] == 1 and p["n_point_identical"] == 1
    assert p["delta"] == pytest.approx(0.3)                      # the primary reading: every common window
    assert p["delta_point_identical"] == pytest.approx(0.4)      # the supplementary one: window 21 only


def test_the_same_mask_on_both_sides_is_verified_identical():
    tw = _tw([_row("confstream", 101, w, 0.5, MASK_A) for w in (20, 21)]
             + [_row("random_online", 101, w, 0.4, MASK_A) for w in (20, 21)])
    p = AS.paired_contrast(tw, "confstream", "random_online", "lag_ari_win").iloc[0]
    assert p["n_point_identical"] == 2 and p["n_point_differs"] == p["n_point_undetermined"] == 0
    assert p["delta_point_identical"] == pytest.approx(p["delta"])
    rows = report.point_support_rows(AS.paired_contrast(tw, "confstream", "random_online", "lag_ari_win"),
                                     "confstream-random_online", "ari")
    assert {r["status"] for r in rows} == {"verified identical"}


def test_logs_without_the_record_leave_the_support_undetermined_and_never_identical():
    tw = _tw([_row("confstream", 101, 20, 0.5, None), _row("random_online", 101, 20, 0.4, MASK_A),
              _row("confstream", 101, 21, 0.5, None), _row("random_online", 101, 21, 0.4, None)])
    p = AS.paired_contrast(tw, "confstream", "random_online", "lag_ari_win")
    assert p.iloc[0]["n_point_undetermined"] == 2 and p.iloc[0]["n_point_identical"] == 0
    assert np.isnan(p.iloc[0]["delta_point_identical"])
    rows = report.point_support_rows(p, "confstream-random_online", "ari")
    assert {r["status"] for r in rows} == {"undetermined"}
    # a fixture without the field at all reads the same
    p2 = AS.paired_contrast(tw.drop(columns=["lag_external_support_sha"]), "confstream", "random_online", "lag_ari_win")
    assert p2.iloc[0]["n_point_undetermined"] == 2


def test_another_window_identity_or_a_fixed_arm_whose_runs_disagree_is_a_differing_support():
    tw = _tw([_row("confstream", 101, 20, 0.5, MASK_A), _row("frozen_reinit", 101, 20, 0.4, MASK_A, start=999)])
    p = AS.paired_contrast(tw, "confstream", "frozen_reinit", "lag_ari_win").iloc[0]
    assert p["n_point_differs"] == 1 and p["n_window_id_differs"] == 1
    # a fixed arm on two model seeds that read the window on other points has no one support
    fixed = _tw([{**_row("default", 201, 20, 0.2, MASK_A), "algo": "HPStream", "model_seed": 201},
                 {**_row("default", 202, 20, 0.3, MASK_B), "algo": "HPStream", "model_seed": 202},
                 {**_row("confstream", 101, 20, 0.5, MASK_A), "algo": "HPStream", "model_seed": 201}])
    aw = AS.arm_windows(fixed, "default", "lag_ari_win")
    assert aw.iloc[0]["support"] == AS.MIXED and aw.iloc[0]["n_runs"] == 2
    q = AS.paired_contrast(fixed, "confstream", "default", "lag_ari_win").iloc[0]
    assert q["n_point_differs"] == 1 and q["n_point_identical"] == 0


def test_the_contrast_report_counts_each_kind_of_window_and_keeps_the_supplementary_reading_apart(tmp_path):
    from tests._split_fixture import make_out_dir

    out = make_out_dir(tmp_path / "o", arms=("default", "scaled_default", "confstream", "frozen_prefix",
                                             "frozen_reinit", "random_online"))
    rows = [json.loads(x) for x in (out / "windows.jsonl").read_text().splitlines() if x]
    for r in rows:                                    # new logs on every arm but frozen_prefix (old logs)
        r["start"], r["end"] = r["window"] * 1000, r["window"] * 1000 + 999
        if r["arm"] != "frozen_prefix":
            other = r["arm"] == "random_online" and r["window"] == 23 and r["algo"] == "DenStream"
            r["lag_external_support_sha"] = MASK_B if other else MASK_A
    (out / "windows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    t = report.contrast_tables(out)
    ps = t["point_support"].query("metric == 'ari' and group == 'all'").set_index("contrast")
    assert ps.loc["frozen_reinit-frozen_prefix", "status"] == "undetermined"
    assert ps.loc["frozen_reinit-frozen_prefix", "n_point_undetermined"] == ps.loc["frozen_reinit-frozen_prefix", "n_pair_windows"]
    assert ps.loc["confstream-scaled_default", "status"] == "verified identical"
    e5 = ps.loc["confstream-random_online"]
    assert e5["status"] == "differs on some windows"
    # DenStream on the 6 real and the synthetic folds, tuner seeds 101 and 102, window 23: 7 x 2 windows differ
    assert e5["n_point_differs"] == 14 and e5["n_point_identical"] == e5["n_pair_windows"] - 14
    g = t["groups"].query("metric == 'ari' and level == 'group' and unit == 'all'")
    assert set(g["subset"]) >= {"all", report.POINT_IDENTICAL_SUBSET}
    supp = g[(g["subset"] == report.POINT_IDENTICAL_SUBSET) & (g["contrast"] == "frozen_reinit-frozen_prefix")]
    assert supp.empty                                 # nothing verified: no supplementary number
    paths = report.write_contrasts(out)
    md = (out / "report" / "contrasts" / "contrasts.md").read_text()
    assert "Point support of the common windows" in md and "Supplementary, windows with verified identical" in md
    assert "point_support.csv" in {p.name for p in paths}


def test_a_run_writes_the_lagged_incumbents_points_into_every_window_row():
    from benchmark.datasets import make_scenario
    from conf_pipeline.run import run_ensemble

    toy = ParamSpace("KMeans", (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)))
    ds = make_scenario("static_easy", n_samples=1500, n_features=2, n_clusters=3, seed=3)
    cfg = EnsembleConfig(e_size=3, e_test=1, lam=0.1, p_reset=0.0, runtime_factor=None)
    rows = run_ensemble(algo="KMeans", space=toy, start=[{"n_clusters": 2, "halflife": 0.5}], dataset=ds,
                        fold="toy", seed=1, arm="confstream", cfg=cfg, h=300, n_samples=1500)
    wins = [r for r in rows if r["kind"] == "window" and r.get("lag_member_id") is not None]
    assert wins
    full = support_fingerprint([True] * 300)          # every point has a label and a prediction here
    for r in wins:
        assert r["lag_n_points_input"] == 300 == r["lag_n_points_external"] == r["lag_n_points_scored"]
        assert r["lag_external_support_sha"] == r["lag_support_sha"] == full
        assert r["end"] - r["start"] == 299


def test_a_mixed_window_identity_is_never_called_the_same_points():
    assert AS.point_support_status(MASK_A, MASK_A, AS.MIXED, AS.MIXED) == "differs"
    assert AS.point_support_status(MASK_A, MASK_A, "0-9", AS.MIXED) == "differs"
    assert AS.point_support_status(MASK_A, MASK_A, "0-9", "0-9") == "identical"


def test_recording_the_points_moves_no_prediction_score_decision_or_draw(monkeypatch):
    """The same run with the support record replaced by nothing: every trajectory field is the same."""
    from benchmark.datasets import make_scenario
    from benchmark.tuning import confstream as CS
    from conf_pipeline.run import run_ensemble

    toy = ParamSpace("KMeans", (Param("n_clusters", "log_int", 2, 8), Param("halflife", "log", 0.01, 0.9)))

    def go():
        ds = make_scenario("static_easy", n_samples=1500, n_features=2, n_clusters=3, seed=3)
        cfg = EnsembleConfig(e_size=3, e_test=1, lam=0.1, p_reset=0.1, runtime_factor=None)
        rows = run_ensemble(algo="KMeans", space=toy, start=[{"n_clusters": 2, "halflife": 0.5}], dataset=ds,
                            fold="toy", seed=1, arm="confstream", cfg=cfg, h=300, n_samples=1500)
        keep = ("incumbent_id", "params", "pred_sha", "score", "lag_member_id", "lag_pred_sha", "lag_ari_win",
                "n_spawned", "n_accepted", "n_reset")
        return [tuple(str(r.get(k)) for k in keep) for r in rows if r["kind"] == "window"]

    with_record = go()
    monkeypatch.setattr(CS, "support_fingerprint", lambda mask: None)
    monkeypatch.setattr(CS, "external_support", lambda ys, labels: [])
    monkeypatch.setattr(CS, "silhouette_support", lambda labels: [])
    without = go()
    assert with_record == without and len(with_record) == 4
