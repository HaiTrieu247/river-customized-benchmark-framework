"""The cluster identity audit driver (``conf_pipeline.identity_audit``) end to end on small streams."""

from __future__ import annotations

import pandas as pd

from conf_pipeline.identity_audit import CAUSES, cause_of, main


def test_cause_tells_renames_from_micro_and_macro_changes():
    assert cause_of({"kind": "none"}) == "none"
    assert cause_of({"kind": "rename_only"}) == "rename_only"
    assert cause_of({"kind": "structural", "micro_born": 1, "micro_died": 0}) == "structural_micro"
    assert cause_of({"kind": "structural", "micro_born": 0, "micro_died": 0}) == "structural_macro_only"
    assert cause_of({"kind": "structural", "micro_born": None, "micro_died": None,
                     "n_micro_before": 3, "n_micro_after": 4}) == "structural_micro"
    assert cause_of({"kind": "structural", "micro_born": None, "micro_died": None,
                     "n_micro_before": None, "n_micro_after": None}) == "structural_unattributed"


def test_the_audit_writes_every_reading_on_a_scenario_and_a_fold(tmp_path):
    out = tmp_path / "audit"
    report = main(["--out", str(out), "--algos", "SOStream,HPStream,HPStreamStableLabels", "--configs", "default",
                   "--scenarios", "birth", "--seeds", "1", "--h", "100", "--windows", "5", "--probe-every", "20",
                   "--folds", "static_overlap", "--fold-h", "100", "--fold-windows", "2"])
    assert report.exists() and "## 4. The tracker where the exact answer is known" in report.read_text()
    cells = pd.read_csv(out / "cells.csv")
    assert len(cells) == 6 and cells["error"].isna().all()
    windows = pd.read_csv(out / "windows.csv")
    assert len(windows) == 3 * 4 + 3 * 2
    assert (windows[[f"iv_cause_{c}" for c in CAUSES]].sum(axis=1) == windows["n_intervals"]).all()
    intervals = pd.read_csv(out / "intervals.csv")
    assert set(intervals["cause"]) <= set(CAUSES) and len(intervals) == windows["n_intervals"].sum()
    birth = intervals[(intervals["source"] == "scenario") & intervals["events"].fillna("").str.contains("birth")]
    assert len(birth) == 3                                   # one interval per algorithm holds the birth
    pairs = pd.read_csv(out / "pairs.csv")
    assert set(pairs["labels"] + ">" + pairs["reference"]) == {
        "seq>post", "tracked>post", "seq>truth", "tracked>truth", "pre>truth", "post>truth"}
    val = pd.read_csv(out / "validation.csv")
    assert len(val) == 4 + 2 and val["ari_stable_tracked_vs_stable"].notna().all()
    for name in ("by_cause", "pairs", "windows", "full_run", "validation"):
        assert (out / "summary" / f"{name}.csv").exists()


def test_a_dry_run_lists_cells_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "dry"
    assert main(["--out", str(out), "--algos", "DBSTREAM", "--scenarios", "merge,split", "--seeds", "1,2",
                 "--dry-run"]) is None
    assert not out.exists()
    assert "identity_audit: 8 cells" in capsys.readouterr().out
