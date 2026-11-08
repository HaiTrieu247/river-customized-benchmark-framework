"""E1: label-free detection of label reuse inside a window, and its validation on HPStream."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conf_pipeline import identity as I
from conf_pipeline import run as R


def _two_blobs(n=40, gap=10.0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.1, size=(n // 2, 2))
    b = rng.normal(gap, 0.1, size=(n // 2, 2))
    return np.vstack([a, b])


def test_a_label_handed_to_a_far_cluster_is_detected_and_split():
    X = _two_blobs()
    labels = [0] * 40  # label 0 names blob A for 20 points, then blob B
    fixed, events = I.detect_identity_changes(X, labels, s=1.0, kappa=1.0, min_points=5)
    assert events == [20]
    assert len(set(fixed[:20])) == 1 and len(set(fixed[20:])) == 1 and fixed[0] != fixed[20]


def test_no_change_is_detected_inside_one_compact_cluster_or_without_a_unit():
    X = _two_blobs()
    labels = [0] * 20 + [1] * 20
    assert I.detect_identity_changes(X, labels, s=1.0, kappa=1.0)[1] == []
    assert I.detect_identity_changes(X, [0] * 40, s=0.0, kappa=1.0)[1] == []
    fixed, _ = I.detect_identity_changes(X, [None, *labels[1:]], s=1.0)
    assert fixed[0] is None


def test_exact_reuse_events_follow_the_stable_labels_and_matching_counts_hits():
    raw = [0, 0, 1, 0, 0, 1]
    stable = [0, 0, 1, 5, 5, 1]  # label 0 is reused for another cluster from position 3
    assert I.exact_reuse_events(raw, stable) == [3]
    m = I.match_events([3, 4], [3], tol=0)
    assert m["n_hits"] == 1 and m["precision"] == 0.5 and m["recall"] == 1.0
    assert np.isnan(I.match_events([], [])["precision"])


def test_end_to_end_on_a_small_run_with_the_hpstream_validation(tmp_path):
    out = tmp_path / "e1"
    R.main(["--algos", "HPStream,HPStreamStableLabels", "--folds", "static_easy", "--arms", "default",
            "--n-samples", "2000", "--h", "500", "--freeze-after-window", "2", "--no-runtime-gate",
            "--log-members", "--log-predictions", "--out-dir", str(out)])
    paths = I.write(out, kappas=(0.5, 1.0))
    names = {p.name for p in paths}
    assert {"member_windows.csv", "replay.csv", "summary.csv", "validation.csv", "identity.md"} <= names
    mw = pd.read_csv(out / "report" / "identity" / "member_windows.csv")
    assert set(mw["algo"]) == {"HPStream", "HPStreamStableLabels"}
    assert {"raw_ari", "ari_k0.5", "n_changes_k1"} <= set(mw.columns)
    val = pd.read_csv(out / "report" / "identity" / "validation.csv")
    assert len(val) and (val["n_true_reuses"] >= 0).all()
    replay = pd.read_csv(out / "report" / "identity" / "replay.csv")
    # one member, no selection: the raw replay is the logged lagged ARI
    both = replay.dropna(subset=["lag_ari_raw", "logged_lag_ari"])
    assert np.allclose(both["lag_ari_raw"], both["logged_lag_ari"], atol=1e-9)
