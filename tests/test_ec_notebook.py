"""otb_ec_variants.ipynb: the notebook comparing ensemble sizes and challenger counts of the online tuner."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def test_the_notebook_on_disk_is_the_generated_one():
    from conf_pipeline import make_ec_notebook

    on_disk = nbformat.read(REPO / "otb_ec_variants.ipynb", as_version=4)
    assert [(c.cell_type, c.source) for c in on_disk.cells] == [
        (c.cell_type, c.source) for c in make_ec_notebook.build().cells
    ]


def _panel(preset, out_dir, monkeypatch):
    from conf_pipeline import _common as C
    from conf_pipeline import make_ec_notebook

    monkeypatch.setenv("OTB_EC_PRESET", preset)
    monkeypatch.setenv("OTB_EC_OUT_DIR", str(out_dir))
    ns = {"os": os, "Path": Path, "REPO": REPO, "C": C}
    exec(make_ec_notebook.PANEL, ns)
    return ns


def test_the_full_preset_is_every_algorithm_on_the_locked_streams_with_one_seed_and_eight_workers(tmp_path, monkeypatch):
    from conf_pipeline import _common as C
    from conf_pipeline import variants as V

    ns = _panel("full", tmp_path, monkeypatch)
    assert set(ns["ALGOS_FIRST"]) | set(ns["ALGOS_SECOND"]) == set(C.ALGOS)
    assert set(ns["ALGOS_SECOND"]) == {"CluStream", "RepStream"}
    assert ns["FOLDS"] == ["covtype", "kdd99", "electricity", "shuttle", "insects_abrupt_imbalanced",
                           "scale_drift", "density_drift", "k_drift", "mixed_drift", "static_overlap",
                           "incremental_drift", "evolving"]
    assert ns["TUNER_SEEDS"] == [101] and ns["MODEL_SEED"] == 201 and ns["WORKERS"] == 8
    assert ns["GRID"] == list(V.EC_GRID)
    assert ns["N_SAMPLES"] == C.N_SAMPLES_CAP and ns["H"] == C.H and ns["TIMEOUT_S"] == V.EC_TIMEOUT_S
    args = [str(a) for a in ns["RUN_ARGS_FIRST"]]
    assert args[0] == "ec-run" and "--grid" in args and args[args.index("--workers") + 1] == "8"


def test_every_preset_leaves_a_tail(tmp_path, monkeypatch):
    for preset in ("full", "tiny"):
        ns = _panel(preset, tmp_path, monkeypatch)
        assert ns["N_SAMPLES"] // ns["H"] > ns["FREEZE_AFTER_WINDOW"]


@pytest.mark.slow
def test_the_notebook_executes_in_tiny_mode(tmp_path, monkeypatch):
    import subprocess
    import sys

    from nbclient import NotebookClient

    subprocess.run([sys.executable, "-m", "conf_pipeline.make_ec_notebook"], cwd=REPO, check=True)
    nb = nbformat.read(REPO / "otb_ec_variants.ipynb", as_version=4)
    monkeypatch.setenv("OTB_EC_PRESET", "tiny")
    monkeypatch.setenv("OTB_EC_OUT_DIR", str(tmp_path))
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(REPO)}}).execute()
    summary = pd.read_csv(tmp_path / "report" / "ec" / "summary.csv")
    assert set(summary["setting"]) == {"ec15_7", "ec10_5", "ec5_2", "ec20_5", "ec20_2"}
    assert (tmp_path / "ec20_10" / "manifest.json").exists()
