"""A campaign: one JSON config (``conf_pipeline/campaigns/*.json``) turned into the exact ``run``
commands, the inventory of jobs they make, a campaign manifest, and a strict resume.

The config names the streams, the arms, groups of algorithms each with its tuner seeds, the protocol
(seed policy, model and data seeds, h, E, C, t0, cap, gate, tie-break, logging) and the operational
limits (the timeout). The inventory is built by ``run.plan_jobs``, the code ``run`` itself executes,
so the jobs listed are the jobs run: one per online arm and tuner seed, a fixed configuration once
per fixed model seed (once at all for a deterministic algorithm), a ``scaled_default`` identical to
``default`` written down as skipped. Nothing is loaded and nothing is downloaded to build it.

Without ``--execute`` the command is a dry run: it prints the inventory, the commands and whether an
existing campaign dir would accept them, and writes nothing (``--write-inventory`` writes the
inventory to a file outside the out dir). With ``--execute`` it writes ``<out>/campaign/manifest.json``
(config and its hash, code commit, diff hash and code hash, the full package lock, the machine,
protocol, seeds, flags, operational limits and whether the address-space limit is enforced here, the
inventory and its hash, the output paths) and runs every group through ``run.main`` in order.

Resume is strict. A campaign dir accepts a command only when the protocol, the operational limits,
the code hash and the package lock are those on file, and the new inventory contains every job of
the old one: more tuner seeds, arms or streams extend it (recorded in the manifest's history); a
change of anything else, or a dir not made by a campaign, is refused. There is no flag that bypasses
it: ``--force``, ``--allow-code-change`` and ``--adopt-legacy`` are never passed to ``run``, and
``conf_pipeline/out/full`` is never written. A new data seed or preprocessing condition is another
protocol and goes to another out dir. Data fingerprints are recorded per stream by the run manifest
as each job consumes its stream (``manifest.json``, ``streams.<fold>.content_sha256``); a stream whose
content differs stops ``run`` before its row is written.

    PYTHONHASHSEED=0 python -m conf_pipeline.campaign --config conf_pipeline/campaigns/mechanism_pilot.json
    PYTHONHASHSEED=0 python -m conf_pipeline.campaign --config ... --execute --workers 8
    python -m conf_pipeline.campaign --config ... --status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import _common as C
from . import manifest as M
from . import run as R

SCHEMA = 1
CAMPAIGN_DIR = "campaign"
CAMPAIGN_MANIFEST = R.CAMPAIGN_MANIFEST
#: Out dirs a campaign never writes into: the published full run.
FORBIDDEN_OUT_DIRS: tuple[Path, ...] = (C.OUT / "full",)
#: ``run`` flags a campaign never passes: each one bypasses a guard.
BYPASS_FLAGS: tuple[str, ...] = ("--force", "--allow-code-change", "--adopt-legacy")
#: Config keys of ``protocol`` and the repo constant each one is locked to (``locked_to_repo_constants``).
REPO_CONSTANTS: dict[str, str] = {
    "h": "H", "e_size": "E_SIZE", "e_test": "E_TEST", "freeze_after_window": "FREEZE_AFTER_WINDOW",
    "n_samples": "N_SAMPLES_CAP", "p_reset": "P_RESET", "half_life_frac": "HALF_LIFE_FRAC",
    "max_memory_mb": "MAX_MEMORY_MB", "max_micro_clusters": "MAX_MICRO_CLUSTERS", "data_seed": "DATA_SEED",
    "model_seed": "MODEL_SEED",
}
PROTOCOL_KEYS = ("seed_policy", "model_seed", "fixed_model_seeds", "data_seed", "h", "e_size", "e_test",
                 "freeze_after_window", "n_samples", "p_reset", "half_life_frac", "lam", "runtime_gate",
                 "max_memory_mb", "max_micro_clusters", "tie_break", "log_members", "signals", "signal_noise", "cmm")


# --- the config -----------------------------------------------------------------------------------

#: What is known to limit what a campaign's readings of an algorithm can stand for (docs/SEQUENTIAL_PARTITION_AUDIT.md,
#: RIVER_BUGS.md GAP-6). No algorithm leaves the inventory for it and every job still runs: dry runs, runs and
#: diagnostics are unaffected. An entry leaves this table only with the code change that resolves it, and its test.
MEASUREMENT_ISSUES: dict[str, dict] = {
    "HPStream": {
        "issue": "GAP-6: a label reused within one window",
        "mechanism": "_free_label hands the evicted cluster's label to the cluster opened in its place (hpstream.py:620-647)",
        "evidence": "tests/test_hpstream_stable_labels.py (a window where the blob at 0 and the newcomer share label 0); "
                    "audit: renumbering in 684 of 800 probe intervals at River's defaults",
        "scope": "within a window: the sequential silhouette (selection) and the lagged per-window ARI/NMI",
        "fix": "HPStreamStableLabels (registered beside HPStream): the same clusters and predictions at every point, "
               "labels never reused; exact by test",
        "unblock": "run HPStreamStableLabels in HPStream's place (mechanism_pilot.json does, by the user's decision of "
                   "2026-11-04), or report HPStream's readings with this limitation",
    },
    "DenStream": {
        "issue": "GAP-6: clusters renumbered within one window",
        "mechanism": "predict_one numbers the DBSCAN macro clusters from 0 in p_micro_clusters order on every call; the "
                     "periodic pruning of learn_one shifts every later number (denstream.py:351)",
        "evidence": "tests/test_partition_audit.py: pruning the p-micro-cluster at 0 moves label 1 from the cluster at 10, "
                    "whose p-micro-cluster is the same object, to the cluster at 20",
        "scope": "within a window: the sequential silhouette (selection) and the lagged per-window ARI/NMI",
        "fix": "none exact: macro clusters are rebuilt from scratch and carry no identity; naming them needs a rule for "
               "merges and splits, a design decision",
        "unblock": "a naming rule chosen by the user, registered as a variant and tested; or the readings reported with "
                   "this limitation",
    },
    "BIRCH": {
        "issue": "GAP-6: clusters renamed within one window",
        "mechanism": "labels are positions in the leaves (dict(enumerate(features)) at each reclustering, birch.py:392); "
                     "an entry filed earlier shifts every later label",
        "evidence": "tests/test_partition_audit.py: under River's defaults (threshold 0, leaf_capacity 50) the probes keep "
                    "their partition and a label passes to other points; audit: 3 renames with a reused label in 800",
        "scope": "within a window: the sequential silhouette (selection) and the lagged per-window ARI/NMI",
        "fix": "none exact in the framework yet: leaf entries merge and the tree is rebuilt, so their identity needs a "
               "rule of its own",
        "unblock": "an identity rule for leaf entries chosen by the user, registered as a variant and tested; or the "
                   "readings reported with this limitation",
    },
    "DBSTREAM": {
        "issue": "GAP-6: clusters renumbered within one window",
        "mechanism": "macro labels from _generate_clusters_from_labels at each reclustering (dbstream.py:384-389); a "
                     "micro-cluster cleaned up shifts the others",
        "evidence": "tests/test_partition_audit.py: a cleanup renumbers the macro clusters whose micro-clusters stay; "
                    "audit: 4 renames with a reused label in 800",
        "scope": "within a window: the sequential silhouette (selection) and the lagged per-window ARI/NMI",
        "fix": "none exact: macro clusters carry no identity across reclusterings",
        "unblock": "a naming rule chosen by the user, registered as a variant and tested; or the readings reported with "
                   "this limitation",
    },
    "CluStream": {
        "issue": "GAP-6: clusters renamed within one window",
        "mechanism": "a new KMeans over the micro-cluster centres every time_gap points (clustream.py:251-259), its "
                     "labels unrelated to the previous one's",
        "evidence": "tests/test_partition_audit.py: at a time_gap boundary the cluster at 0 stays and its label passes to "
                    "the cluster at 10; audit: 7 renames with a reused label in 800",
        "scope": "within a window: the sequential silhouette (selection) and the lagged per-window ARI/NMI",
        "fix": "none exact: each offline k-means is new, with no identity to carry",
        "unblock": "a naming rule chosen by the user, registered as a variant and tested; or the readings reported with "
                   "this limitation",
    },
}
#: Algorithms audited without a known measurement issue (labels from a monotone counter or a persistent key).
AUDITED_NO_ISSUE: tuple[str, ...] = ("RepStream", "SOStream", "STREAMKMeans")
#: Algorithms whose GAP-6 issue a variant resolves exactly, and the evidence.
RESOLVED: dict[str, str] = {
    "HPStreamStableLabels": "HPStream's GAP-6 resolved: the same model as HPStream with labels never reused "
                            "(tests/test_hpstream_stable_labels.py)",
}
#: Framework subclasses that inherit the labelling of the River class they extend (RIVER_BUGS.md GAP-6 applies
#: to them as to it): DenStreamPatched and DenStreamPatchedFast number DBSCAN's macro clusters from 0 on every
#: call, CluStreamPyramidal keeps CluStream's offline k-means.
INHERITS_LABELLING: dict[str, str] = {**C.RIVER_CLASS, "CluStreamPyramidal": "CluStream"}
#: Conditions on the campaign as a whole, open as of the code that carries this list (docs/MECHANISM_CAMPAIGN.md).
OPEN_CONDITIONS: tuple[str, ...] = (
    "repeatability across two machines not checked (one-machine smoke only)",
    "no smoke on a real stream at the campaign's h, E, C",
    "the 12 h timeout is a projection for CluStream on kdd99 without the gate and with stability (margin x1.25)",
)


def _algo_status(algo: str) -> dict:
    """``open`` (an issue of its own or of the River class it inherits its labels from), ``resolved``,
    ``no known issue`` (audited), or ``not audited`` (never read: it blocks like an open issue)."""
    if algo in RESOLVED:
        return {"status": "resolved", "evidence": RESOLVED[algo]}
    base = algo if algo in MEASUREMENT_ISSUES else INHERITS_LABELLING.get(algo)
    if base in MEASUREMENT_ISSUES:
        out = {"status": "open", **MEASUREMENT_ISSUES[base]}
        if base != algo:
            out["inherited_from"] = base
        return out
    if algo in AUDITED_NO_ISSUE:
        return {"status": "no known issue"}
    return {"status": "not audited", "issue": "not audited for GAP-6: whether its labels keep their meaning within "
                                               "a window is unknown"}


def measurement_status(cfg: dict) -> dict:
    """Per algorithm of the config, its known measurement issue or none; and whether the campaign's
    results may stand as confirmation for the paper: never while an algorithm in the inventory has an
    open issue or a condition of ``OPEN_CONDITIONS`` is open. Runs, dry runs and diagnostics go ahead."""
    algos = [a for g in cfg["groups"] for a in g["algos"]]
    per = {a: _algo_status(a) for a in algos}
    blocked = sorted(a for a, v in per.items() if v["status"] not in ("no known issue", "resolved"))
    reasons = [f"{a}: {per[a].get('issue', per[a]['status'])}" for a in blocked] + list(OPEN_CONDITIONS)
    return {"paper_confirmation": "not eligible" if reasons else "eligible on the known measurement issues",
            "reasons": reasons, "algorithms": per,
            "note": "the status says what results can stand for; it removes no algorithm and stops no job"}


def load_config(path: Path) -> dict:
    """The config at ``path``, checked (``check_config``)."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    check_config(cfg)
    return cfg


def config_sha256(cfg: dict) -> str:
    return M.fingerprint(cfg)


def check_config(cfg: dict) -> None:
    """Refuse a config that ``run`` would not run as written: an unknown arm, algorithm or stream, an
    algorithm in two groups, a protocol key missing, a value locked to the repo's constants that
    differs from them, a runtime gate (the campaign protocol runs gate-free only when it says so)."""
    from .spaces import SPACES

    missing = [k for k in ("name", "folds", "arms", "groups", "protocol", "operational") if k not in cfg]
    if missing:
        raise SystemExit(f"campaign config lacks {missing}")
    proto = cfg["protocol"]
    gap = [k for k in PROTOCOL_KEYS if k not in proto]
    if gap:
        raise SystemExit(f"campaign protocol lacks {gap}")
    runnable = C.ARMS + C.OPTIONAL_ARMS + C.DIAGNOSTIC_ARMS
    bad = [a for a in cfg["arms"] if a not in runnable]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    for f in cfg["folds"]:
        C.stream_group(f)  # raises on a stream without a group
    seen: dict[str, str] = {}
    for g in cfg["groups"]:
        for a in g["algos"]:
            if a not in SPACES:
                raise SystemExit(f"group {g['name']}: no search space for {a}")
            if a in seen:
                raise SystemExit(f"{a} is in groups {seen[a]} and {g['name']}: one group per algorithm")
            seen[a] = g["name"]
        if not g.get("tuner_seeds"):
            raise SystemExit(f"group {g['name']} names no tuner seed")
    for key in cfg.get("locked_to_repo_constants", []):
        want = getattr(C, REPO_CONSTANTS[key])
        if proto[key] != want:
            raise SystemExit(f"protocol.{key} = {proto[key]!r} but the repo's locked value is {want!r} "
                             f"(_common.{REPO_CONSTANTS[key]}); a campaign does not retune the protocol")
    if not isinstance(cfg["operational"].get("timeout_s"), int) or cfg["operational"]["timeout_s"] <= 0:
        raise SystemExit("operational.timeout_s must be a positive integer of seconds")


# --- commands and inventory --------------------------------------------------------------------------

def run_argv(cfg: dict, group: dict, out_dir: Path, workers: int, dry_run: bool = False) -> list[str]:
    """The ``conf_pipeline.run`` arguments of one group, exactly as the campaign passes them."""
    p = cfg["protocol"]
    argv = ["--algos", ",".join(group["algos"]), "--folds", ",".join(cfg["folds"]), "--arms", ",".join(cfg["arms"]),
            "--seed-policy", p["seed_policy"], "--tuner-seeds", ",".join(map(str, group["tuner_seeds"])),
            "--model-seed", str(p["model_seed"]), "--fixed-model-seeds", ",".join(map(str, p["fixed_model_seeds"])),
            "--data-seed", str(p["data_seed"]), "--n-samples", str(p["n_samples"]),
            "--freeze-after-window", str(p["freeze_after_window"]), "--h", str(p["h"]),
            "--e-size", str(p["e_size"]), "--e-test", str(p["e_test"]), "--p-reset", repr(float(p["p_reset"])),
            "--half-life-frac", repr(float(p["half_life_frac"])),
            "--max-memory-mb", repr(float(p["max_memory_mb"])), "--max-micro-clusters", str(p["max_micro_clusters"]),
            "--tie-break", p["tie_break"], "--signal-noise", p["signal_noise"],
            "--out-dir", str(out_dir), "--workers", str(int(workers)), "--timeout", str(cfg["operational"]["timeout_s"])]
    if p.get("lam") is not None:
        argv += ["--lam", repr(float(p["lam"]))]
    if not p["runtime_gate"]:
        argv.append("--no-runtime-gate")
    if p["log_members"]:
        argv.append("--log-members")
    if p["signals"]:
        argv += ["--signals", ",".join(p["signals"])]
    if not p["cmm"]:
        argv.append("--no-cmm")
    if dry_run:
        argv.append("--dry-run")
    assert not set(argv) & set(BYPASS_FLAGS)
    return argv


def job_key(job: dict) -> tuple:
    return (job["algo"], job["fold"], job["arm"], job["seed"], job.get("model_seed"), job.get("config_id"))


def inventory(cfg: dict, out_dir: Path) -> dict:
    """Every job the campaign's commands run and every one they write down as skipped, per group, built by
    ``run.plan_jobs`` (no data read). ``sha256`` hashes the job keys, the skips and the paths."""
    jobs, skipped = [], []
    for g in cfg["groups"]:
        args = R.build_parser().parse_args(run_argv(cfg, g, out_dir, 1, dry_run=True))
        planned, skips, _, _ = R.plan_jobs(args)
        for j in planned:
            jobs.append({"group": g["name"], "algo": j["algo"], "fold": j["fold"], "arm": j["arm"], "seed": j["seed"],
                         "tuner_seed": j["tuner_seed"], "model_seed": j["model_seed"], "config_id": j["config_id"],
                         "n_samples": j["n_samples"], "member_log": j["member_log"], "lifecycle_log": j["lifecycle_log"]})
        for j, row in skips:
            skipped.append({"group": g["name"], "algo": j["algo"], "fold": j["fold"], "arm": j["arm"],
                            "seed": j["seed"], "model_seed": j["model_seed"], "reason": row["error"]})
    keys = [job_key(j) for j in jobs]
    if len(set(keys)) != len(keys):
        raise SystemExit("two groups plan the same job; an algorithm belongs to one group")
    count = lambda xs, k: dict(sorted(((v, sum(1 for x in xs if x[k] == v)) for v in {x[k] for x in xs})))  # noqa: E731
    return {"n_runnable": len(jobs), "n_skipped": len(skipped), "by_arm": count(jobs, "arm"),
            "by_algo": count(jobs, "algo"), "by_group": count(jobs, "group"),
            "skipped_by_arm": count(skipped, "arm"), "jobs": jobs, "skipped": skipped,
            "sha256": M.fingerprint({"jobs": jobs, "skipped": skipped})}


def e1_configs(cfg: dict) -> list[dict]:
    """What E1 changes, read off the search spaces before any run: per algorithm, its absolute-distance
    parameters and River's start values of them (``spaces.start_params``). A parameter whose start is 0 stays
    0 in units of s (BIRCH's ``threshold``); an algorithm without one has no ``scaled_default`` run (its
    ``default`` stands in). The value of s and the scaled values are per stream, on each run's summary
    (``scaled_params``, ``scaled_changed``)."""
    from .spaces import distance_params, start_params

    out = []
    for g in cfg["groups"]:
        for algo in g["algos"]:
            names = distance_params(algo)
            start = start_params(algo, cfg["protocol"]["h"], d=10)
            values = {n: start.get(n) for n in names}
            changed = [n for n, v in values.items() if v not in (None, 0, 0.0)]
            out.append({"algo": algo, "distance_params": list(names), "start_values": values,
                        "scaled_default": ("not run: identical to default" if not names else
                                           "runs; every distance parameter starts at 0, so it equals default" if not changed
                                           else "runs; scaled: " + ", ".join(changed))})
    return out


# --- the manifest and the guard ------------------------------------------------------------------------

def operational(cfg: dict) -> dict:
    """The limits a job runs under (``run.operational_limits``'s keys, which ``run`` checks against this dir's
    campaign manifest), and how they are enforced here."""
    p = cfg["protocol"]
    return {"timeout_s": int(cfg["operational"]["timeout_s"]), "memory_limit_bytes": int(C.MEMORY_LIMIT_BYTES),
            "max_memory_mb": float(p["max_memory_mb"]), "max_micro_clusters": int(p["max_micro_clusters"]),
            "runtime_factor": None if not p["runtime_gate"] else float(C.RUNTIME_FACTOR)}


def rlimit_as_enforced() -> bool | None:
    """Whether a child can set the address-space limit ``run`` asks for on this machine (macOS usually
    refuses it, and ``run`` then goes on without one); None when the probe itself fails."""
    code = ("import resource,sys\n"
            f"L={int(C.MEMORY_LIMIT_BYTES)}\n"
            "try:\n resource.setrlimit(resource.RLIMIT_AS,(L,L)); print('yes')\n"
            "except (ValueError, OSError): print('no')\n")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    return {"yes": True, "no": False}.get(out.stdout.strip())


def build_manifest(cfg: dict, config_path: Path, out_dir: Path, inv: dict) -> dict:
    """The campaign manifest (see the module docstring); ``history`` is added by ``check_resume``."""
    from benchmark.reports.environment import capture_environment, package_lock

    lock = package_lock()
    return M._plain({
        "schema": SCHEMA, "name": cfg["name"], "config_path": str(config_path), "config": cfg,
        "config_sha256": config_sha256(cfg),
        "protocol": cfg["protocol"], "folds": cfg["folds"], "arms": cfg["arms"], "groups": cfg["groups"],
        "excluded_arms": cfg.get("excluded_arms", {}),
        "operational": operational(cfg),
        "enforcement": {"rlimit_as": rlimit_as_enforced(), "pythonhashseed_children": C.PYTHONHASHSEED,
                        "thread_pins_children": C.THREAD_PINS, "timeout_source": cfg["operational"].get("timeout_source"),
                        "timeout_status": "a job past the timeout is written with status timeout, never scored"},
        "commands": {g["name"]: run_argv(cfg, g, out_dir, cfg["operational"].get("workers", 1)) for g in cfg["groups"]},
        "code": M.code_provenance(), "packages": lock, "hardware": capture_environment(),
        "e1_configs": e1_configs(cfg),
        "measurement_status": measurement_status(cfg),
        "inventory": {k: inv[k] for k in ("n_runnable", "n_skipped", "by_arm", "by_algo", "by_group",
                                           "skipped_by_arm", "jobs", "skipped", "sha256")},
        "data": {"fingerprints": "per stream in manifest.json (streams.<fold>.content_sha256), recorded as each job "
                                 "consumes its stream; pending until then", "data_seed": cfg["protocol"]["data_seed"]},
        "outputs": {"runs": "runs.jsonl", "windows": "windows.jsonl", "members": f"{C.MEMBERS_DIR.name}/",
                    "lifecycle": f"{R.LIFECYCLE_DIR}/", "run_manifest": M.MANIFEST, "environment": "environment.jsonl",
                    "campaign_manifest": str(CAMPAIGN_MANIFEST)},
    })


def load_manifest(out_dir: Path) -> dict | None:
    path = Path(out_dir) / CAMPAIGN_MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def guard_out_dir(out_dir: Path) -> None:
    """Refuse the published full run's dir (or one inside it), and a dir holding results that no campaign made."""
    out = Path(out_dir).resolve()
    for bad in FORBIDDEN_OUT_DIRS:
        bad = bad.resolve()
        if out == bad or bad in out.parents:
            raise SystemExit(f"{out_dir} is (inside) {bad}: a campaign never writes into the published full run")
    if load_manifest(out_dir) is None and any((Path(out_dir) / f).exists() for f in (*M.RESULT_FILES, M.MANIFEST)):
        raise SystemExit(f"{out_dir} holds results that no campaign made: write the campaign into a new --out-dir")


def check_resume(old: dict | None, new: dict, workers: int, argv=None) -> tuple[dict, str, list[str]]:
    """The manifest to write, the event (``created``, ``resumed``, ``extended``) and the jobs it adds.
    Raises ``SystemExit`` with every difference when the campaign on file may not be resumed by ``new``."""
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": list(sys.argv[1:] if argv is None else argv),
             "workers": int(workers), "config_sha256": new["config_sha256"], "inventory_sha256": new["inventory"]["sha256"]}
    if old is None:
        return {**new, "history": [{**event, "event": "created"}]}, "created", []
    diffs = []
    for key in ("name", "protocol", "operational"):
        M._diff(key, old.get(key), new.get(key), diffs)
    for key in ("code_sha256", "packages_sha256"):
        if old.get("code", {}).get(key) != new["code"].get(key):
            diffs.append(f"code.{key}: {old.get('code', {}).get(key)} -> {new['code'].get(key)}")
    old_jobs = {M.canonical(job_key(j)) for j in old["inventory"]["jobs"]}
    new_jobs = {M.canonical(job_key(j)) for j in new["inventory"]["jobs"]}
    dropped = sorted(old_jobs - new_jobs)
    if dropped:
        diffs.append(f"inventory: {len(dropped)} jobs on file are not in the new inventory (first: {dropped[0]})")
    if diffs:
        raise SystemExit("this campaign dir was made under another campaign protocol, code or inventory:\n  "
                         + "\n  ".join(diffs) + "\nWrite into a new --out-dir; no flag bypasses this check.")
    added = sorted(new_jobs - old_jobs)
    kind = "extended" if added or old.get("config_sha256") != new["config_sha256"] else "resumed"
    merged = {**new, "history": [*old.get("history", []), {**event, "event": kind, "n_jobs_added": len(added)}]}
    return merged, kind, added


def write_manifest(out_dir: Path, m: dict) -> Path:
    path = Path(out_dir) / CAMPAIGN_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True, default=C._json_default), encoding="utf-8")
    os.replace(tmp, path)
    return path


def refresh_fingerprints(out_dir: Path) -> dict:
    """The stream content hashes the run manifest has recorded so far (``None`` where still pending)."""
    rm = M.load(out_dir) or {}
    return {k: v.get("content_sha256") for k, v in sorted(rm.get("streams", {}).items())}


# --- status ------------------------------------------------------------------------------------------------

def status(out_dir: Path, inv: dict) -> dict:
    """Per arm: inventory jobs and the latest status of each (ok, error, timeout, oom, not run yet)."""
    from .analysis_support import latest_attempts, seed_key

    latest = latest_attempts(C.read_jsonl(Path(out_dir) / "runs.jsonl"))
    out: dict = {}
    for j in inv["jobs"]:
        key = (j["algo"], j["fold"], j["arm"], seed_key(j["seed"]), seed_key(j["model_seed"]), j["config_id"])
        st = latest.get(key, {}).get("status", "not run")
        arm = out.setdefault(j["arm"], {})
        arm[st] = arm.get(st, 0) + 1
    return out


# --- CLI -----------------------------------------------------------------------------------------------------

def _print_inventory(inv: dict, cfg: dict, out_dir: Path, workers: int) -> None:
    print(f"campaign {cfg['name']}: {inv['n_runnable']} runnable jobs, {inv['n_skipped']} written down as skipped "
          f"(identical configuration); inventory {inv['sha256'][:16]}")
    print("  per arm: " + ", ".join(f"{k} {v}" for k, v in inv["by_arm"].items()))
    print("  per group: " + ", ".join(f"{k} {v}" for k, v in inv["by_group"].items()))
    print("  skipped per arm: " + (", ".join(f"{k} {v}" for k, v in inv["skipped_by_arm"].items()) or "none"))
    p = cfg["protocol"]
    print(f"  protocol: h={p['h']} E={p['e_size']} C={p['e_test']} t0={p['freeze_after_window']} cap={p['n_samples']} "
          f"seed policy {p['seed_policy']}, model seed {p['model_seed']}, fixed model seeds {p['fixed_model_seeds']}, "
          f"data seed {p['data_seed']}, runtime gate {'on' if p['runtime_gate'] else 'off'}, tie-break {p['tie_break']}, "
          f"lambda {'per fold from half_life_frac=' + str(p['half_life_frac']) if p['lam'] is None else p['lam']}, "
          f"signals {p['signals']}")
    print(f"  operational: timeout {cfg['operational']['timeout_s']} s ({cfg['operational'].get('timeout_source')}), "
          f"member caps {p['max_memory_mb']} MiB / {p['max_micro_clusters']} micro-clusters, child address space "
          f"{C.MEMORY_LIMIT_BYTES // 1024**3} GiB (RLIMIT_AS, where the platform enforces it)")
    for g in cfg["groups"]:
        print(f"  command [{g['name']}]: PYTHONHASHSEED=0 {sys.executable} -m conf_pipeline.run "
              + " ".join(run_argv(cfg, g, out_dir, workers)))
    _print_measurement(measurement_status(cfg))


def _print_measurement(ms: dict) -> None:
    print(f"  measurement status: paper confirmation {ms['paper_confirmation'].upper()} ({ms['note']})")
    for a, v in ms["algorithms"].items():
        if v["status"] == "open":
            via = f" (inherited from {v['inherited_from']})" if "inherited_from" in v else ""
            print(f"    {a}: open{via}, {v['issue']}; fix: {v['fix']}; unblock: {v['unblock']}")
        elif v["status"] == "not audited":
            print(f"    {a}: not audited, {v['issue']}")
        elif v["status"] == "resolved":
            print(f"    {a}: resolved, {v['evidence']}")
        else:
            print(f"    {a}: {v['status']}")
    for c in OPEN_CONDITIONS:
        print(f"    campaign: open, {c}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="a campaign config (conf_pipeline/campaigns/*.json)")
    parser.add_argument("--out-dir", default=None, help="default: the config's out_dir")
    parser.add_argument("--workers", type=int, default=None, help="default: the config's operational.workers")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="print the inventory and the commands (the default)")
    group.add_argument("--execute", action="store_true", help="write the campaign manifest and run every group")
    group.add_argument("--status", action="store_true", help="per arm, the latest status of every inventory job")
    parser.add_argument("--write-inventory", default=None,
                        help="dry run: write the inventory as JSON to this path (outside the out dir)")
    args = parser.parse_args(argv)
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    out_dir = Path(args.out_dir or cfg["out_dir"])
    workers = int(args.workers or cfg["operational"].get("workers", 1))
    inv = inventory(cfg, out_dir)
    if args.status:
        print(json.dumps({"status": status(out_dir, inv), "fingerprints": refresh_fingerprints(out_dir),
                          "measurement_status": measurement_status(cfg)}, indent=2))
        return
    _print_inventory(inv, cfg, out_dir, workers)
    if not args.execute:
        guard_out_dir(out_dir)
        old = load_manifest(out_dir)
        if old is not None:
            new = build_manifest(cfg, cfg_path, out_dir, inv)
            _, kind, added = check_resume(old, new, workers, argv)
            print(f"  out dir {out_dir}: an existing campaign would be {kind} ({len(added)} jobs added)")
        else:
            print(f"  out dir {out_dir}: new campaign")
        if args.write_inventory:
            target = Path(args.write_inventory).resolve()
            if out_dir.resolve() in target.parents:
                raise SystemExit("--write-inventory writes outside the out dir (a dry run writes nothing there)")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"config_sha256": config_sha256(cfg), "inventory": inv,
                                          "e1_configs": e1_configs(cfg)}, indent=2, default=C._json_default),
                              encoding="utf-8")
            print(f"  inventory written to {target}")
        print("dry run: nothing executed, nothing written to the out dir")
        return
    if os.environ.get("PYTHONHASHSEED") != C.PYTHONHASHSEED:
        raise SystemExit(f"run the campaign with PYTHONHASHSEED={C.PYTHONHASHSEED} (every child is pinned anyway; the "
                         "documented command pins the parent too)")
    guard_out_dir(out_dir)
    new = build_manifest(cfg, cfg_path, out_dir, inv)
    merged, kind, added = check_resume(load_manifest(out_dir), new, workers, argv)
    write_manifest(out_dir, merged)
    print(f"campaign manifest: {kind} ({len(added)} jobs added) -> {out_dir / CAMPAIGN_MANIFEST}")
    for g in cfg["groups"]:
        R.main(run_argv(cfg, g, out_dir, workers))
    merged = load_manifest(out_dir)
    merged["data"]["recorded"] = refresh_fingerprints(out_dir)
    merged["history"][-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    merged["history"][-1]["status"] = status(out_dir, inv)
    write_manifest(out_dir, merged)
    print(json.dumps(merged["history"][-1]["status"], indent=2))


if __name__ == "__main__":
    main()
