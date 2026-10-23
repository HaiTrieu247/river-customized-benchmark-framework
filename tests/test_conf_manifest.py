"""conf_pipeline.manifest: the protocol an out dir's results were made under, and the resume guard."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.reports.environment import code_hash, git_provenance, package_lock
from conf_pipeline import _common as C
from conf_pipeline import manifest as M
from conf_pipeline import run, sweep

REPO = Path(__file__).resolve().parent.parent


def _job(**over):
    job = {
        "algo": "DenStream", "fold": "static_easy", "seed": 1, "arm": "confstream", "n_samples": 900,
        "h": 300, "e_size": 2, "e_test": 1, "p_reset": 0.01, "half_life_frac": 0.1, "lam": None,
        "runtime_factor": 30.0, "max_memory_mb": 2048.0, "max_micro_clusters": 10_000,
        "freeze_after_window": 20, "member_log": None, "config_id": None, "unit_override": None,
        "params_override": None, "log_signals": [], "tie_break": "micro_clusters", "cmm": True,
        "data_seed": C.DATA_SEED,
    }
    job.update(over)
    return job


def _built(jobs, **kw):
    return M.build(jobs, provenance={"framework": {"commit": "abc"}}, **kw)


# --- provenance ----------------------------------------------------------------------------

def test_git_provenance_names_the_commit_and_hashes_the_uncommitted_changes(tmp_path):
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)

    clean = git_provenance(repo, paths=("src",))
    assert len(clean["commit"]) == 40 and clean["dirty"] is False and clean["diff_sha256"] is None

    (repo / "src" / "a.py").write_text("x = 2\n")
    edited = git_provenance(repo, paths=("src",))
    assert edited["dirty"] is True and edited["changed_files"] == ["src/a.py"]
    (repo / "src" / "new.py").write_text("y = 1\n")
    more = git_provenance(repo, paths=("src",))
    assert more["changed_files"] == ["src/a.py", "src/new.py"]
    assert more["diff_sha256"] != edited["diff_sha256"]
    # a change outside the watched paths is not code the numbers came from
    (repo / "notes.md").write_text("hello\n")
    assert git_provenance(repo, paths=("src",))["diff_sha256"] == more["diff_sha256"]


def test_the_code_hash_reads_the_python_files_not_the_commit_or_the_docs(tmp_path):
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    before = code_hash(repo, paths=("src",))
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    assert code_hash(repo, paths=("src",)) == before                    # committing changes no code
    (repo / "src" / "README.md").write_text("notes\n")
    assert code_hash(repo, paths=("src",)) == before                    # nor does a doc
    (repo / "src" / "b.py").write_text("y = 1\n")
    after = code_hash(repo, paths=("src",))
    assert after != before
    (repo / "src" / "a.py").write_text("x = 2\n")
    assert code_hash(repo, paths=("src",)) not in (before, after)


def test_git_provenance_of_a_directory_that_is_no_repository_is_unknown(tmp_path):
    prov = git_provenance(tmp_path)
    assert prov["commit"] == "unknown" and prov["diff_sha256"] is None


def test_package_lock_lists_installed_versions_and_hashes_them():
    lock = package_lock()
    assert lock["n"] == len(lock["packages"]) > 0
    assert any(p.lower().startswith("numpy==") for p in lock["packages"])
    assert len(lock["sha256"]) == 64 and package_lock()["sha256"] == lock["sha256"]


# --- what a manifest holds -------------------------------------------------------------------

def test_the_manifest_records_protocol_arms_algorithms_and_streams():
    m = _built([_job(), _job(arm="default"), _job(seed=2)])
    assert m["protocol"]["h"] == 300 and m["protocol"]["freeze_after_window"] == 20
    assert m["protocol"]["data_seed"] == C.DATA_SEED and m["protocol"]["cmm"] is True
    assert m["constants"]["SCALER"] == C.SCALER and m["constants"]["PYTHONHASHSEED"] == C.PYTHONHASHSEED
    assert set(m["arms"]) == {"confstream", "default"}
    assert m["arms"]["default"]["e_size"] == 1 and m["arms"]["confstream"]["e_size"] == 2
    assert m["arms"]["confstream"]["tie_break"] == "micro_clusters"
    algo = m["algorithms"]["DenStream"]
    assert algo["river_defaults"]["epsilon"] == C.river_defaults("DenStream")["epsilon"]
    assert {p["name"] for p in algo["space"]["params"]} >= {"epsilon", "mu"}
    assert algo["space"]["fixed"] == {"beta": 0.75, "stream_speed": 100}
    assert algo["model_seed"] == "none (deterministic)"
    (stream,) = m["streams"].values()
    assert stream["fold"] == "static_easy" and stream["data_seed"] == C.DATA_SEED
    assert stream["n_samples_cap"] == 900 and stream["content_sha256"] is None
    assert m["seeds"]["tuner"] == [1, 2]


def test_a_seeded_algorithm_records_that_the_job_seed_reaches_the_model():
    m = _built([_job(algo="CluStream")])
    assert m["algorithms"]["CluStream"]["model_seed"] == "job seed"
    assert m["algorithms"]["CluStream"]["start_overrides"] == {"sigma": 1.0}


def test_a_real_stream_has_no_data_seed():
    spec = M.stream_spec("covtype", 100_000, C.DATA_SEED)
    assert spec["data_seed"] is None and spec["group"] == "real"
    assert spec["source"]["loader"] == "load_covtype"


def test_every_protocol_and_x7_fold_has_a_source():
    for fold in C.FOLDS + C.X7_FOLDS:
        assert M.stream_spec(fold, C.N_SAMPLES_CAP, C.DATA_SEED)["source"]["loader"]


def test_the_fingerprint_moves_with_every_protocol_constant_and_only_with_them():
    base = _built([_job()])
    assert _built([_job()])["protocol_fingerprint"] == base["protocol_fingerprint"]
    for over in ({"h": 600}, {"freeze_after_window": 10}, {"n_samples": 1200}, {"cmm": False},
                 {"data_seed": 7}, {"runtime_factor": 10.0}, {"half_life_frac": 0.2}):
        assert _built([_job(**over)])["protocol_fingerprint"] != base["protocol_fingerprint"], over
    # seeds and arms are the design, not the protocol
    assert _built([_job(seed=5), _job(arm="default")])["protocol_fingerprint"] == base["protocol_fingerprint"]


def test_one_protocol_constant_with_two_values_among_the_jobs_is_refused():
    with pytest.raises(ValueError, match="h"):
        _built([_job(), _job(h=600, seed=2)])


def test_job_fingerprints_tell_arms_algorithms_and_streams_apart():
    m = _built([_job(), _job(arm="default"), _job(algo="BIRCH"), _job(fold="static_overlap")])
    fps = {M.job_fingerprint(m, _job(**o)) for o in ({}, {"arm": "default"}, {"algo": "BIRCH"},
                                                       {"fold": "static_overlap"})}
    assert len(fps) == 4
    assert M.job_fingerprint(m, _job(seed=3)) == M.job_fingerprint(m, _job())


# --- comparing two manifests -----------------------------------------------------------------

def test_comparing_a_manifest_with_itself_finds_nothing():
    m = _built([_job()])
    assert M.compare(m, m) == ([], [], [])


def test_a_changed_protocol_constant_is_a_conflict_naming_the_field():
    conflicts, _, _ = M.compare(_built([_job()]), _built([_job(h=600)]))
    assert any(c.startswith("protocol.h:") and "300" in c and "600" in c for c in conflicts)


def test_a_new_arm_algorithm_or_stream_extends_the_manifest():
    old = _built([_job()])
    conflicts, added, _ = M.compare(old, _built([_job(arm="default"), _job(algo="BIRCH"),
                                                 _job(fold="static_overlap")]))
    assert conflicts == []
    assert set(added) >= {"arms.default", "algorithms.BIRCH",
                          f"streams.{M.stream_key('static_overlap', C.DATA_SEED)}"}


def test_an_arm_run_again_under_another_rule_is_a_conflict():
    conflicts, _, _ = M.compare(_built([_job()]), _built([_job(tie_break="runtime")]))
    assert any(c.startswith("arms.confstream.tie_break") for c in conflicts)


def test_a_changed_search_space_is_a_conflict(monkeypatch):
    import dataclasses

    old = _built([_job()])
    monkeypatch.setitem(M.SPACES, "DenStream",
                        dataclasses.replace(M.SPACES["DenStream"], params=M.SPACES["DenStream"].params[:2]))
    conflicts, _, _ = M.compare(old, _built([_job()]))
    assert any(c.startswith("algorithms.DenStream.space") for c in conflicts)


def test_code_changes_are_reported_apart_from_protocol_conflicts():
    old = M.build([_job()], provenance={"framework": {"commit": "abc"}, "code_sha256": "1"})
    new = M.build([_job()], provenance={"framework": {"commit": "def"}, "code_sha256": "2"})
    conflicts, _, code = M.compare(old, new)
    assert conflicts == [] and any("code_sha256" in c for c in code)


def test_a_new_commit_of_the_same_code_is_no_code_change():
    old = M.build([_job()], provenance={"framework": {"commit": "abc"}, "code_sha256": "1"})
    new = M.build([_job()], provenance={"framework": {"commit": "def"}, "code_sha256": "1"})
    assert M.compare(old, new)[2] == []


def test_a_run_made_by_other_code_than_its_command_started_with_is_not_appended(tmp_path):
    m = M.build([_job()], provenance={"code_sha256": "started"})
    session = M.Session(tmp_path, {**m, "history": []})
    ok = {"kind": "run", "status": "ok", "algo": "DenStream", "fold": "static_easy", "code_sha256": "started"}
    session.accept(_job(), [dict(ok)])
    with pytest.raises(SystemExit, match="code changed"):
        session.accept(_job(), [{**ok, "code_sha256": "edited"}])
    session.accept(_job(), [{"kind": "run", "status": "error"}])     # a failed child says nothing


def test_a_recorded_stream_hash_must_match_but_an_unrecorded_one_is_no_conflict():
    old = _built([_job()])
    key = M.stream_key("static_easy", C.DATA_SEED)
    old["streams"][key]["content_sha256"] = "aaa"
    assert M.compare(old, _built([_job()]))[0] == []
    new = _built([_job()])
    new["streams"][key]["content_sha256"] = "bbb"
    assert any("content_sha256" in c for c in M.compare(old, new)[0])


# --- the stream digest ---------------------------------------------------------------------------

def test_the_stream_digest_reads_content_and_order():
    a, b, c = M.StreamDigest(), M.StreamDigest(), M.StreamDigest()
    rows = [({"f0": 1.0, "f1": 2.0}, 0), ({"f0": 3.0, "f1": 4.0}, 1)]
    for x, y in rows:
        a.update(x, y)
        b.update(x, y)
    for x, y in reversed(rows):
        c.update(x, y)
    assert a.hexdigest() == b.hexdigest() != c.hexdigest()
    d = M.StreamDigest()
    d.update({"f0": 1.0, "f1": 2.0}, 1)
    d.update({"f0": 3.0, "f1": 4.0}, 1)
    assert d.hexdigest() != a.hexdigest()                     # labels are content too


def test_a_capped_stream_is_digested_up_to_the_cap_only():
    full = M.CappedStream(C.load_fold("static_easy", n_samples=600), 600)
    head = M.CappedStream(C.load_fold("static_easy", n_samples=600), 300)
    assert len(list(full)) == 600 and len(list(head)) == 300
    again = M.CappedStream(C.load_fold("static_easy", n_samples=600), 300)
    list(again)
    assert again.digest.hexdigest() == head.digest.hexdigest() != full.digest.hexdigest()


# --- the resume guard, end to end -----------------------------------------------------------------

def _cli(*args, module="conf_pipeline.run"):
    # these tests exercise the job seeds of the shared policy, the runs made before the seed split
    if module != "conf_pipeline.manifest" and "--seed-policy" not in args:
        args = (*args, "--seed-policy", "shared")
    return subprocess.run(
        [sys.executable, "-m", module, *args], capture_output=True, text=True, cwd=str(REPO),
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )


def _args(seeds="1", arms="default", h="300"):
    return ["--algos", "DenStream", "--folds", "static_easy", "--seeds", seeds, "--arms", arms,
            "--n-samples", "900", "--h", h, "--workers", "1"]


COMMON = _args()
#: For the tests of something else than code provenance: the tree may change between two commands
#: of one test (another edit in progress), which the guard rightly refuses.
ANY_CODE = "--allow-code-change"


def test_a_run_writes_its_manifest_and_stamps_its_rows(tmp_path):
    proc = _cli(*COMMON, "--out-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["schema"] == M.SCHEMA and m["history"][0]["event"] == "created"
    (row,) = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert row["protocol_fingerprint"] == m["protocol_fingerprint"]
    assert row["job_fingerprint"] == M.job_fingerprint(m, {"algo": "DenStream", "fold": "static_easy",
                                                            "arm": "default", "data_seed": C.DATA_SEED})
    key = M.stream_key("static_easy", C.DATA_SEED)
    assert row["stream_sha256"] == m["streams"][key]["content_sha256"] and len(row["stream_sha256"]) == 64
    assert row["code_sha256"] == m["code"]["code_sha256"]                # the child ran the code on file


def test_a_resume_under_another_protocol_stops_before_reading_what_is_done(tmp_path):
    assert _cli(*COMMON, "--out-dir", str(tmp_path)).returncode == 0
    other = _args(h="450")
    proc = _cli(*other, "--out-dir", str(tmp_path))
    assert proc.returncode != 0
    assert "protocol.h" in proc.stderr and "new --out-dir" in proc.stderr
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1
    # a dry run says the same, and nothing is written
    dry = _cli(*other, "--out-dir", str(tmp_path), "--dry-run")
    assert dry.returncode != 0 and "protocol.h" in dry.stderr


def test_more_seeds_and_another_arm_extend_the_same_out_dir(tmp_path):
    assert _cli(*COMMON, "--out-dir", str(tmp_path)).returncode == 0
    more = _args(seeds="1,2", arms="default,confstream")
    proc = _cli(*more, "--out-dir", str(tmp_path), "--e-size", "2", "--e-test", "1", "--dry-run", ANY_CODE)
    assert proc.returncode == 0, proc.stderr
    assert "3 to run" in proc.stdout and "extends" in proc.stdout


def test_a_directory_of_results_without_a_manifest_is_not_resumed_unless_adopted(tmp_path):
    # a row of another seed, so the adopting command has a job of its own to write
    row = {"kind": "run", "algo": "DenStream", "fold": "static_easy", "seed": 2, "arm": "default",
           "config_id": None, "n_samples_cap": 900, "status": "ok"}
    (tmp_path / "runs.jsonl").write_text(json.dumps(row) + "\n")
    proc = _cli(*COMMON, "--out-dir", str(tmp_path), "--dry-run")
    assert proc.returncode != 0 and "no manifest" in proc.stderr
    adopted = _cli(*COMMON, "--out-dir", str(tmp_path), "--adopt-legacy")
    assert adopted.returncode == 0, adopted.stderr
    m = json.loads((tmp_path / "manifest.json").read_text())
    first = m["history"][0]
    assert first["event"] == "adopted_legacy" and first["prior_protocol"] == "unknown"
    assert first["prior_rows"] == {"runs.jsonl": 1}


def test_a_code_change_needs_to_be_allowed_and_is_recorded(tmp_path):
    assert _cli(*COMMON, "--out-dir", str(tmp_path)).returncode == 0
    path = tmp_path / "manifest.json"
    m = json.loads(path.read_text())
    m["code"]["code_sha256"] = "0" * 64
    path.write_text(json.dumps(m))
    more = _args(seeds="1,2")
    proc = _cli(*more, "--out-dir", str(tmp_path))
    assert proc.returncode != 0 and "code.code_sha256" in proc.stderr and "--allow-code-change" in proc.stderr
    ok = _cli(*more, "--out-dir", str(tmp_path), "--allow-code-change")
    assert ok.returncode == 0, ok.stderr
    events = [h["event"] for h in json.loads(path.read_text())["history"]]
    assert events[-1] == "code_changed"


def test_a_stream_whose_content_changed_stops_the_command_and_is_not_appended(tmp_path):
    assert _cli(*COMMON, "--out-dir", str(tmp_path)).returncode == 0
    path = tmp_path / "manifest.json"
    m = json.loads(path.read_text())
    m["streams"][M.stream_key("static_easy", C.DATA_SEED)]["content_sha256"] = "f" * 64
    path.write_text(json.dumps(m))
    more = _args(seeds="1,2")
    proc = _cli(*more, "--out-dir", str(tmp_path), ANY_CODE)
    assert proc.returncode != 0 and "content" in proc.stderr
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 1


def test_the_sweep_shares_the_protocol_and_records_its_design_seed(tmp_path):
    assert _cli(*COMMON, "--out-dir", str(tmp_path), "--freeze-after-window", "1").returncode == 0
    sweep_common = ["--algos", "DenStream", "--folds", "static_easy", "--n-samples", "900", "--h", "300",
                    "--out-dir", str(tmp_path), "--workers", "1", "--n-points", "1", ANY_CODE]
    wrong_t0 = _cli(*sweep_common, module="conf_pipeline.sweep")
    assert wrong_t0.returncode != 0 and "protocol.freeze_after_window" in wrong_t0.stderr
    ok = _cli(*sweep_common, "--freeze-after-window", "1", module="conf_pipeline.sweep")
    assert ok.returncode == 0, ok.stderr
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["sweep"] == {"sobol_seed": 1} and "sweep" in m["arms"]
    other_design = _cli(*sweep_common, "--freeze-after-window", "1", "--seed", "2", module="conf_pipeline.sweep")
    assert other_design.returncode != 0 and "sweep.sobol_seed" in other_design.stderr


def test_select_only_needs_no_manifest(tmp_path):
    (tmp_path / "sweep.jsonl").write_text(json.dumps(
        {"algo": "A", "fold": "f", "seed": 1, "config_id": 0, "status": "ok", "lag_ari_win_mean": 0.1,
         "unit_override": None, "final_params": {}}) + "\n")
    proc = _cli("--out-dir", str(tmp_path), "--select-only", module="conf_pipeline.sweep")
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "manifest.json").exists()


# --- describing a dir made before manifests ------------------------------------------------------

def test_describe_reports_what_rows_say_and_unknown_for_the_rest(tmp_path):
    rows = [
        {"kind": "run", "algo": "A", "fold": "f", "seed": 1, "arm": "confstream", "status": "ok", "h": 1000,
         "e_size": 20, "e_test": 10, "freeze_after_window": 20, "n_samples_cap": 100_000},
        {"kind": "run", "algo": "A", "fold": "f", "seed": 2, "arm": "confstream", "status": "ok", "h": 1000,
         "e_size": 20, "e_test": 10, "freeze_after_window": 20, "n_samples_cap": 100_000, "tie_break": "runtime"},
        {"kind": "run", "algo": "A", "fold": "f", "seed": 1, "arm": "default", "status": "error"},
    ]
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    d = M.describe_dir(tmp_path)
    assert d["manifest"] == "absent" and d["code"] == "unknown" and d["environment"] == "unknown"
    files = {f["name"]: f for f in d["files"]}
    assert files["runs.jsonl"]["rows"] == 3 and len(files["runs.jsonl"]["sha256"]) == 64
    conf = d["recorded"]["confstream"]
    assert conf["h"] == {"1000": 2} and conf["e_size"] == {"20": 2}
    assert conf["tie_break"] == {"<absent>": 1, '"runtime"': 1}     # values as JSON
    assert "default" not in d["recorded"]                    # only ok rows say how a run was made
    assert not (tmp_path / "manifest.json").exists()


def test_describe_cli_writes_into_the_report_dir_only(tmp_path):
    (tmp_path / "runs.jsonl").write_text(json.dumps(
        {"kind": "run", "algo": "A", "fold": "f", "seed": 1, "arm": "default", "status": "ok", "h": 1000}) + "\n")
    proc = _cli("describe", "--out-dir", str(tmp_path), module="conf_pipeline.manifest")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "report" / "provenance.json").exists() and (tmp_path / "report" / "provenance.md").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report", "runs.jsonl"]


def test_a_sweep_into_a_dir_run_without_the_runtime_gate_drops_it_too(tmp_path):
    assert _cli(*COMMON, "--out-dir", str(tmp_path), "--no-runtime-gate").returncode == 0
    assert json.loads((tmp_path / "manifest.json").read_text())["protocol"]["runtime_factor"] is None
    sweep_common = ["--algos", "DenStream", "--folds", "static_easy", "--n-samples", "900", "--h", "300",
                    "--out-dir", str(tmp_path), "--n-points", "1", "--dry-run", ANY_CODE]
    refused = _cli(*sweep_common, module="conf_pipeline.sweep")
    assert refused.returncode != 0 and "protocol.runtime_factor" in refused.stderr
    ok = _cli(*sweep_common, "--no-runtime-gate", module="conf_pipeline.sweep")
    assert ok.returncode == 0, ok.stderr


def test_an_oracle_rerun_whose_file_names_another_point_under_the_same_id_is_refused(tmp_path):
    oracle = tmp_path / "oracle_static.json"
    oracle.write_text(json.dumps({"DenStream": {"static_easy": {"config_id": 3, "unit": {"epsilon": 0.9}}}}))
    args = ["--algos", "DenStream", "--folds", "static_easy", "--seeds", "1", "--arms", "oracle_static",
            "--n-samples", "900", "--h", "300", "--workers", "1", "--out-dir", str(tmp_path), ANY_CODE]
    assert _cli(*args).returncode == 0
    assert "0 to run" in _cli(*args).stdout                              # the same point: resumed
    oracle.write_text(json.dumps({"DenStream": {"static_easy": {"config_id": 3, "unit": {"epsilon": 0.1}}}}))
    proc = _cli(*args)
    assert proc.returncode != 0 and "another configuration" in proc.stderr and "config_id 3" in proc.stderr


def test_a_second_command_into_a_dir_whose_manifest_is_held_is_refused(tmp_path):
    (tmp_path / "manifest.json.lock").write_text("pid=1 started=now\n")
    proc = _cli(*COMMON, "--out-dir", str(tmp_path))
    assert proc.returncode != 0 and "manifest.json.lock" in proc.stderr
    assert not (tmp_path / "runs.jsonl").exists()
    sweep = _cli("--algos", "DenStream", "--folds", "static_easy", "--n-samples", "900", "--h", "300",
                 "--out-dir", str(tmp_path), "--n-points", "1", module="conf_pipeline.sweep")
    assert sweep.returncode != 0 and "manifest.json.lock" in sweep.stderr


def test_the_manifest_is_checked_again_once_its_lock_is_held(tmp_path):
    first = M.build([_job()], provenance={"code_sha256": "1"})
    M.write(tmp_path, {**first, "history": []})
    # another command changed the dir's protocol after this one's first check
    M.write(tmp_path, {**M.build([_job(h=600)], provenance={"code_sha256": "1"}), "history": []})
    with pytest.raises(SystemExit, match="protocol.h"):
        with M.locked_session(tmp_path, first, "conf_pipeline.run"):
            pass
    assert not (tmp_path / "manifest.json.lock").exists()
