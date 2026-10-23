"""The protocol an out dir's results were made under, and the guard that keeps it one protocol.

``run.py`` and ``sweep.py`` resume by job key: (algo, fold, seed, arm, cap, config). The key
names none of the constants a job ran under (h, E, C, t0, the tie-break, the signals, CMM,
the search spaces), so a command with other flags pointed at an existing out dir used to
skip every job it found and report the old numbers as if they answered the new question.
Every command that runs jobs now writes ``<out-dir>/manifest.json`` and checks it first:

- ``protocol`` and ``constants``: what every job of the dir shares (window, cap, fading
  half-life, feasibility caps, t0 and the tail it starts, CMM, the data-generator seed, and
  the module constants behind them). They must match exactly.
- ``arms``, ``algorithms``, ``streams``: the effective ensemble configuration of every arm,
  every algorithm's implementation, constructor defaults, pinned values and search space,
  every stream's source, order, cap and data seed. An entry already on file must match;
  a new one extends the dir (more seeds, another arm, another fold stay legal).
- ``sweep``: the Sobol design's seed. Its point count is design, not protocol: a Sobol
  sequence of one seed is prefix-stable, so more points extend the same design.
- ``code``: the framework's and River's commits with a hash of their uncommitted changes (for
  the record), a hash of the content of their Python files (``code_sha256``) and of the
  installed packages. A change of either hash blocks a resume unless ``--allow-code-change``,
  and is then recorded in ``history``; a new commit of the same code, or an edited document,
  is not a change. Every child hashes the code it imported and reports it on its run row: a
  command whose code is edited while it runs stops before the first row made by other code.

A stream's content hash is read by the child over the samples it consumed
(``stream_sha256`` on the run's summary); the first one is recorded, and a later run that
saw other content stops the command before its row is written. A dir holding results but
no manifest was made before manifests existed: its protocol is unknown and is never
reconstructed; it is resumed only with ``--adopt-legacy``, which says so in ``history``.

Every summary row carries ``protocol_fingerprint`` (``protocol`` + ``constants``) and
``job_fingerprint`` (that, plus its arm, algorithm and stream), so a number traces to the
manifest it was made under.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import struct
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import _common as C
from .spaces import HPSTREAM_DEFAULT_PROJECTION, INIT_BUFFER_PARAMS, SPACES, START_OVERRIDES

#: 2 since the seed split (2026-10-24): the protocol names its seed policy, the online arms' model seed
#: and the calibration seed; a manifest of schema 1 is another protocol version and never resumed.
SCHEMA = 2
MANIFEST = "manifest.json"
#: The files whose rows mean "results are here": a dir with any of them and no manifest is legacy.
RESULT_FILES = ("runs.jsonl", "sweep.jsonl")
#: Job fields every job of a dir shares; ``lam`` None means "from the half-life, per fold".
PROTOCOL_KEYS = ("h", "n_samples", "half_life_frac", "lam", "runtime_factor", "max_memory_mb",
                 "max_micro_clusters", "freeze_after_window", "cmm", "data_seed",
                 "seed_policy", "online_model_seed", "calibration_seed")
#: Paths of the framework whose changes can change a number (tests, docs, notebooks cannot).
FRAMEWORK_CODE = ("benchmark", "conf_pipeline")
#: Sections compared entry by entry: an entry on file must match, a new one extends.
EXTENSIBLE = ("arms", "algorithms", "streams")


# --- canonical form ---------------------------------------------------------------------------

def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=C._json_default)


def fingerprint(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def _plain(obj):
    """``obj`` as the JSON a manifest file reads back, so a built manifest compares equal to a
    loaded one (tuples become lists, numpy scalars numbers)."""
    return json.loads(canonical(obj))


# --- what is recorded -------------------------------------------------------------------------

def constants() -> dict:
    from benchmark.tuning.space import INITIAL_SIGMA
    from benchmark.tuning.window import UNDEFINED_SCORE

    return _plain({
        "SCALER": C.SCALER, "EXTERNAL_METRICS": C.EXTERNAL_METRICS, "MIN_TRUE_CLASSES": C.MIN_TRUE_CLASSES,
        "SCALE_SUBSAMPLE": C.SCALE_SUBSAMPLE, "SCALE_MAX_WINDOWS": C.SCALE_MAX_WINDOWS,
        "TABLE_A_EPS_FACTOR": C.TABLE_A_EPS_FACTOR, "ORACLE_METRICS": C.ORACLE_METRICS,
        "SYNTH_N_SAMPLES": C.SYNTH_N_SAMPLES, "PYTHONHASHSEED": C.PYTHONHASHSEED,
        "THREAD_PINS": C.THREAD_PINS, "UNDEFINED_SCORE": UNDEFINED_SCORE, "INITIAL_SIGMA": INITIAL_SIGMA,
        "INIT_BUFFER_PARAMS": INIT_BUFFER_PARAMS, "lag_windows": 1, "tail_from": "freeze_after_window",
    })


def algorithm_spec(algo: str) -> dict:
    """How ``algo`` is built: the registered factory and model class, River's constructor
    defaults, the values the protocol pins over them, its search space and its seed policy."""
    from benchmark.algorithms.registry import accepts_seed, get_spec

    spec = get_spec(algo)
    model = type(spec.factory()[0])
    space = SPACES[algo]
    seeded = accepts_seed(algo)
    out = {
        "factory": f"{spec.factory.__module__}.{spec.factory.__qualname__}",
        "model_class": f"{model.__module__}.{model.__qualname__}",
        "provenance": spec.provenance,
        "river_defaults": C.river_defaults(algo),
        "start_overrides": START_OVERRIDES.get(algo, {}),
        "space": {"params": [dataclasses.asdict(p) for p in space.params], "fixed": space.fixed},
        "accepts_seed": seeded,
        "model_seed": "job seed" if seeded else "none (deterministic)",
    }
    if algo == "HPStream":
        out["default_projection"] = HPSTREAM_DEFAULT_PROJECTION
    return _plain(out)


def stream_key(fold: str, data_seed: int | None) -> str:
    """A stream's entry in ``streams``: the fold, plus the data seed of a synthetic one."""
    return f"{fold}/data_seed={data_seed}" if C.is_synthetic(fold) else fold


def stream_spec(fold: str, n_samples: int | None, data_seed: int | None) -> dict:
    synthetic = C.is_synthetic(fold)
    try:
        group = C.stream_group(fold)
    except ValueError:
        group = None
    return _plain({
        "fold": fold, "group": group, "source": C.fold_source(fold), "order": "natural",
        "n_samples_cap": n_samples,
        "generated_n_samples": (n_samples or C.SYNTH_N_SAMPLES) if synthetic else None,
        "data_seed": data_seed if synthetic else None,
        "content_sha256": None,
    })


def _river_root() -> Path | None:
    try:
        import river

        return Path(river.__file__).resolve().parent.parent
    except Exception:  # pragma: no cover
        return None


def code_fingerprint() -> str:
    """One hash of the Python code of the framework (``FRAMEWORK_CODE``) and of River: what a
    child reports on its run row, and ``code.code_sha256`` of the manifest."""
    from benchmark.reports.environment import code_hash

    root = _river_root()
    return fingerprint([code_hash(C.REPO_ROOT, FRAMEWORK_CODE), code_hash(root) if root else None])


def code_provenance() -> dict:
    """The code the numbers came from (see the module docstring)."""
    from benchmark.reports.environment import git_provenance, package_lock

    root = _river_root()
    lock = package_lock()
    return _plain({
        "framework": git_provenance(C.REPO_ROOT, paths=FRAMEWORK_CODE),
        "river": git_provenance(root) if root else {"commit": "unknown"},
        "code_sha256": code_fingerprint(),
        "python": sys.version.split()[0],
        "packages_sha256": lock["sha256"],
    })


def _shared(jobs: list[dict], key: str):
    values = {canonical(j.get(key, _JOB_DEFAULTS.get(key))) for j in jobs}
    if len(values) > 1:
        raise ValueError(f"the jobs disagree on the protocol constant {key!r}: {sorted(values)}")
    return json.loads(values.pop())


#: What the child assumes when a job dict leaves a field out (the sweep's jobs do).
_JOB_DEFAULTS = {"cmm": True, "data_seed": C.DATA_SEED, "lam": None, "seed_policy": "shared",
                 "online_model_seed": None, "calibration_seed": None}


def _seeds(jobs: list[dict]) -> dict:
    """The tuner and model seeds the jobs run on (a job dict from before the split: its job seed is both)."""
    from . import run

    fields = [run.seed_fields(j) for j in jobs]
    return {"tuner": sorted({f["tuner_seed"] for f in fields if f["tuner_seed"] is not None}),
            "model": sorted({f["model_seed"] for f in fields if f["model_seed"] is not None})}


def arm_spec(job: dict) -> dict:
    """The ensemble configuration ``job``'s arm runs under, as the child builds it."""
    from . import run

    t0 = job.get("freeze_after_window")
    cfg = run.arm_config(job["arm"], run.job_config(job), C.FREEZE_AFTER_WINDOW if t0 is None else t0)
    return _plain({k: v for k, v in dataclasses.asdict(cfg).items() if k != "lam"})


def build(jobs: list[dict], provenance: dict | None = None, sweep: dict | None = None) -> dict:
    """The manifest of ``jobs`` (no history: ``check`` adds it). ``provenance`` defaults to
    this process's code; ``sweep`` is the Sobol design (``{"sobol_seed": ...}``)."""
    if not jobs:
        raise ValueError("no jobs to describe")
    protocol = {k: _shared(jobs, k) for k in PROTOCOL_KEYS}
    m = {
        "schema": SCHEMA,
        "protocol": protocol,
        "constants": constants(),
        "arms": {}, "algorithms": {}, "streams": {},
        "seeds": _seeds(jobs),
    }
    for j in jobs:
        if j["arm"] not in m["arms"]:
            m["arms"][j["arm"]] = arm_spec(j)
        if j["algo"] not in m["algorithms"]:
            m["algorithms"][j["algo"]] = algorithm_spec(j["algo"])
        key = stream_key(j["fold"], protocol["data_seed"])
        if key not in m["streams"]:
            m["streams"][key] = stream_spec(j["fold"], protocol["n_samples"], protocol["data_seed"])
    if sweep is not None:
        m["sweep"] = _plain(sweep)
    m["code"] = _plain(code_provenance() if provenance is None else provenance)
    m["protocol_fingerprint"] = fingerprint({"protocol": m["protocol"], "constants": m["constants"]})
    return m


def job_fingerprint(m: dict, job: dict) -> str:
    """The fingerprint of the protocol ``job`` ran under: the dir's, plus its arm, algorithm and stream."""
    key = stream_key(job["fold"], m["protocol"]["data_seed"])
    stream = {k: v for k, v in m["streams"][key].items() if k != "content_sha256"}
    return fingerprint({
        "protocol": m["protocol"], "constants": m["constants"], "arm": m["arms"][job["arm"]],
        "algorithm": m["algorithms"][job["algo"]], "stream": stream,
        "sweep": m.get("sweep") if job["arm"] == "sweep" else None,
    })


# --- comparing ----------------------------------------------------------------------------------

def _diff(prefix: str, old, new, out: list[str]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            _diff(f"{prefix}.{k}", old.get(k, "<absent>"), new.get(k, "<absent>"), out)
    elif canonical(old) != canonical(new):
        out.append(f"{prefix}: {canonical(old)} -> {canonical(new)}")


def compare(old: dict, new: dict) -> tuple[list[str], list[str], list[str]]:
    """``(conflicts, added, code_changes)`` between the manifest on file and a command's.

    A conflict is a field that differs where both say something: the shared protocol, an
    arm, algorithm or stream on file, the sweep design, a stream content hash recorded on
    both sides. ``added`` names the entries only ``new`` has. Code changes are listed apart
    (the commits, the diff hashes, the package lock), as ``--allow-code-change`` can accept them.
    """
    conflicts: list[str] = []
    _diff("schema", old.get("schema"), new.get("schema"), conflicts)
    _diff("protocol", old.get("protocol", {}), new.get("protocol", {}), conflicts)
    _diff("constants", old.get("constants", {}), new.get("constants", {}), conflicts)
    added: list[str] = []
    for section in EXTENSIBLE:
        o, n = old.get(section, {}), new.get(section, {})
        for key in sorted(n):
            if key not in o:
                added.append(f"{section}.{key}")
                continue
            if section == "streams":
                a, b = dict(o[key]), dict(n[key])
                ha, hb = a.pop("content_sha256", None), b.pop("content_sha256", None)
                _diff(f"streams.{key}", a, b, conflicts)
                if ha is not None and hb is not None and ha != hb:
                    conflicts.append(f"streams.{key}.content_sha256: {ha} -> {hb}")
            else:
                _diff(f"{section}.{key}", o[key], n[key], conflicts)
    if old.get("sweep") is not None and new.get("sweep") is not None:
        _diff("sweep", old["sweep"], new["sweep"], conflicts)
    elif new.get("sweep") is not None:
        added.append("sweep")
    code: list[str] = []
    oc, nc = old.get("code", {}), new.get("code", {})
    for k in ("code_sha256", "packages_sha256"):
        if oc.get(k) != nc.get(k):
            code.append(f"code.{k}: {oc.get(k)} -> {nc.get(k)}")
    return conflicts, added, code


# --- the guard --------------------------------------------------------------------------------

def load(out_dir: Path) -> dict | None:
    path = Path(out_dir) / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write(out_dir: Path, m: dict) -> None:
    path = Path(out_dir) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True, default=C._json_default), encoding="utf-8")
    os.replace(tmp, path)


def _prior_rows(out_dir: Path) -> dict:
    """Rows already in the dir's result files; a ``skipped`` row (an oracle not yet swept) is no result."""
    out = {}
    for name in RESULT_FILES:
        n = sum(1 for r in C.read_jsonl(Path(out_dir) / name) if r.get("status") != "skipped")
        if n:
            out[name] = n
    return out


def _event(event: str, command: str, argv, **extra) -> dict:
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, "command": command,
            "argv": list(sys.argv[1:] if argv is None else argv), **extra}


def check(out_dir: Path, new: dict, command: str, argv=None, adopt_legacy: bool = False,
          allow_code_change: bool = False, **history) -> tuple[dict, str]:
    """The manifest the command may run under, merged with the one on file, and what happened
    (``created``, ``adopted_legacy``, ``resumed``, ``extended``, ``code_changed``); nothing is
    written. Raises ``SystemExit`` with every differing field when the command may not resume
    into ``out_dir``. ``history`` adds fields to the event (workers, timeout...)."""
    out_dir = Path(out_dir)
    old = load(out_dir)
    if old is None:
        prior = _prior_rows(out_dir)
        if prior and not adopt_legacy:
            raise SystemExit(
                f"{out_dir} holds results ({', '.join(f'{k}: {v} rows' for k, v in prior.items())}) but no "
                f"manifest: they were made before manifests existed and their protocol is unknown.\n"
                "Write into a new --out-dir, or pass --adopt-legacy to resume here and record that the "
                "earlier rows' protocol is unknown."
            )
        event = "adopted_legacy" if prior else "created"
        extra = {"prior_rows": prior, "prior_protocol": "unknown"} if prior else {}
        return {**new, "history": [_event(event, command, argv, **extra, **history)]}, event

    conflicts, added, code = compare(old, new)
    if conflicts:
        raise SystemExit(
            f"{out_dir} was made under another protocol ({MANIFEST}); this command differs in:\n  "
            + "\n  ".join(conflicts)
            + "\nWrite into a new --out-dir: resuming would report the old results as answers to this command."
        )
    if code and not allow_code_change:
        raise SystemExit(
            f"{out_dir} was made by other code ({MANIFEST}):\n  " + "\n  ".join(code)
            + "\nWrite into a new --out-dir, or pass --allow-code-change to resume here and record the change."
        )
    merged = json.loads(json.dumps(old))
    for item in added:
        section, _, key = item.partition(".")
        if key:
            merged.setdefault(section, {})[key] = new[section][key]
        else:
            merged[section] = new[section]
    merged["seeds"] = {k: sorted(set(old.get("seeds", {}).get(k, [])) | set(new["seeds"].get(k, [])))
                       for k in ("tuner", "model")}
    event = "code_changed" if code else ("extended" if added else "resumed")
    extra = {"added": added} if added else {}
    if code:
        extra["code_changes"] = code
    commits = [f"{part}: {(old.get('code', {}).get(part) or {}).get('commit')} -> {(new['code'].get(part) or {}).get('commit')}"
               for part in ("framework", "river")
               if (old.get("code", {}).get(part) or {}).get("commit") != (new["code"].get(part) or {}).get("commit")]
    if commits:
        extra["commits"] = commits
    merged["code"] = new["code"]
    merged.setdefault("history", []).append(_event(event, command, argv, **extra, **history))
    return merged, event


def describe(event: str, m: dict) -> str:
    added = m["history"][-1].get("added") if m.get("history") else None
    return f"manifest: {event}" + (f" (extends {', '.join(added)})" if added else "")


def record_stream(m: dict, fold: str, sha: str | None) -> bool:
    """Record a run's stream content hash; True when the manifest changed. Raises
    ``SystemExit`` when the stream already has another hash on file."""
    if sha is None:
        return False
    entry = m["streams"][stream_key(fold, m["protocol"]["data_seed"])]
    if entry.get("content_sha256") is None:
        entry["content_sha256"] = sha
        return True
    if entry["content_sha256"] != sha:
        raise SystemExit(
            f"the content of stream {fold!r} differs from what this out dir's runs consumed "
            f"({entry['content_sha256']} on file, {sha} now): the data or its loader changed. "
            "Write into a new --out-dir."
        )
    return False


class Session:
    """The manifest a command runs under, once it holds its out dir's lock: written on
    entry, then updated as runs report the content they consumed."""

    def __init__(self, out_dir: Path, m: dict):
        self.out_dir = Path(out_dir)
        self.m = m
        write(self.out_dir, self.m)

    def accept(self, job: dict, rows: list[dict]) -> None:
        """Check ``job``'s summary against the manifest and stamp it with its fingerprints.
        Raises ``SystemExit`` (nothing appended yet by the caller) on a stream whose content
        differs from what the dir's runs consumed."""
        summary = rows[-1]
        ran, started = summary.get("code_sha256"), self.m.get("code", {}).get("code_sha256")
        if ran is not None and ran != started:
            raise SystemExit(
                f"the code changed while this command ran: {job['algo']} {job['fold']} seed {job['seed']} "
                f"arm {job['arm']} ran code {ran}, the command started with {started}. Its row is not written; "
                "finish the edit, then resume (with --allow-code-change) or write into a new --out-dir."
            )
        if summary.get("status") == "ok" and record_stream(self.m, job["fold"], summary.get("stream_sha256")):
            write(self.out_dir, self.m)
        self.stamp(job, summary)

    def stamp(self, job: dict, row: dict) -> dict:
        row["protocol_fingerprint"] = self.m["protocol_fingerprint"]
        row["job_fingerprint"] = job_fingerprint(self.m, job)
        return row


@contextmanager
def locked_session(out_dir: Path, new: dict, command: str, force: bool = False, argv=None, **check_kw):
    """A ``Session`` held under ``manifest.json.lock`` for the whole command: the manifest is
    checked again once the lock is held (another command may have changed it since the first
    check), written, and only then are jobs run. A second command into the same dir -- ``run``
    beside ``sweep`` included -- is refused while the lock is held, as the result files' own
    locks refuse a second writer; ``force`` overrides it as theirs do."""
    with C.JsonlLock(Path(out_dir) / MANIFEST, force=force):
        m, _ = check(out_dir, new, command, argv=argv, **check_kw)
        yield Session(out_dir, m)


# --- the content a run consumed -----------------------------------------------------------------

class StreamDigest:
    """SHA-256 over a stream's samples in order: the feature names whenever they change, each
    sample's values (packed as doubles, their text when one is not a number) and its label."""

    def __init__(self):
        self._h = hashlib.sha256()
        self._keys = None
        self.n = 0

    def update(self, x: dict, y) -> None:
        keys = tuple(x)
        if keys != self._keys:
            self._h.update(b"\1" + "\0".join(map(str, keys)).encode())
            self._keys = keys
        values = tuple(x.values())
        try:
            self._h.update(struct.pack(f"<{len(values)}d", *values))
        except (struct.error, TypeError):
            self._h.update(b"\2" + "\0".join(map(str, values)).encode())
        self._h.update(b"\3" + str(y).encode())
        self.n += 1

    def hexdigest(self) -> str:
        return self._h.hexdigest()


class CappedStream:
    """``dataset`` cut at ``cap`` samples (None: all), digesting what it hands out."""

    def __init__(self, dataset, cap: int | None):
        self.dataset = dataset
        self.cap = cap
        self.digest = StreamDigest()

    def __len__(self):
        n = len(self.dataset)
        return n if self.cap is None else min(n, self.cap)

    def __iter__(self):
        for i, (x, y) in enumerate(self.dataset):
            if self.cap is not None and i >= self.cap:
                return
            self.digest.update(x, y)
            yield x, y

    def __getattr__(self, name):
        return getattr(self.dataset, name)


# --- what a dir made before manifests can still say ----------------------------------------------

#: Summary-row fields that say how a run was made, read per arm by ``describe_dir``.
RECORDED_FIELDS = ("h", "n_samples_cap", "half_life_frac", "e_size", "e_test", "p_reset", "runtime_factor",
                   "max_memory_mb", "max_micro_clusters", "freeze_after_window", "tail_from", "sampler",
                   "selection_signal", "log_signals", "tie_break", "reinit_after_freeze", "data_seed",
                   "seed_policy", "tuner_seed", "model_seed", "calibration_seed")
ALL_FILES = RESULT_FILES + ("windows.jsonl", "sweep_windows.jsonl", "oracle_static.json", "environment.jsonl")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe_dir(out_dir: Path) -> dict:
    """What an out dir's own files say about how its results were made, and nothing more.

    For a dir made before manifests this is the most that can be recovered: the hash and row
    count of every file, and per arm the values its ok summary rows recorded for each field of
    ``RECORDED_FIELDS`` with their counts (``<absent>`` where a row predates the field).
    Whatever no row records -- the code, River's version, the machine when there is no
    ``environment.jsonl`` -- is ``"unknown"``. Nothing is written into the dir itself.
    """
    out_dir = Path(out_dir)
    files = []
    for name in ALL_FILES:
        path = out_dir / name
        if path.exists():
            rows = sum(1 for _ in C.read_jsonl(path)) if name.endswith(".jsonl") else None
            files.append({"name": name, "bytes": path.stat().st_size, "rows": rows, "sha256": _sha256_file(path)})
    members = out_dir / C.MEMBERS_DIR.name
    if members.is_dir():
        logs = sorted(members.glob("*.jsonl"))
        digest = hashlib.sha256()
        for p in logs:
            digest.update(p.name.encode() + b"\0" + _sha256_file(p).encode())
        files.append({"name": f"{C.MEMBERS_DIR.name}/", "files": len(logs), "sha256": digest.hexdigest()})
    recorded: dict = {}
    for name in RESULT_FILES:
        for r in C.read_jsonl(out_dir / name):
            if r.get("status") != "ok":
                continue
            per_arm = recorded.setdefault(r.get("arm"), {})
            for f in RECORDED_FIELDS:
                value = canonical(r[f]) if f in r else "<absent>"
                counts = per_arm.setdefault(f, {})
                counts[value] = counts.get(value, 0) + 1
    env = [r for r in C.read_jsonl(out_dir / "environment.jsonl")]
    m = load(out_dir)
    return {
        "out_dir": str(out_dir),
        "manifest": "present" if m is not None else "absent",
        "protocol_fingerprint": m.get("protocol_fingerprint") if m else None,
        "code": m.get("code") if m else "unknown",
        "environment": env if env else "unknown",
        "files": files,
        "recorded": recorded,
    }


def write_description(out_dir: Path) -> list[Path]:
    d = describe_dir(out_dir)
    rep = Path(out_dir) / C.REPORT_DIR.name
    rep.mkdir(parents=True, exist_ok=True)
    js, md = rep / "provenance.json", rep / "provenance.md"
    js.write_text(json.dumps(d, indent=2, sort_keys=True, default=C._json_default), encoding="utf-8")
    lines = [
        "# Provenance of this out dir", "",
        f"Manifest: **{d['manifest']}**." + (
            f" Protocol fingerprint `{d['protocol_fingerprint']}`." if d["protocol_fingerprint"] else
            " Without one the protocol is what the rows below record, and the code that produced "
            "them is unknown: nothing here is reconstructed."), "",
        f"Code: {'unknown' if d['code'] == 'unknown' else 'see provenance.json'}. "
        f"Environment records: {len(d['environment']) if d['environment'] != 'unknown' else 'unknown'}.", "",
        "## Files", "", "| file | rows/files | bytes | sha256 |", "|---|---:|---:|---|",
    ]
    for f in d["files"]:
        lines.append(f"| {f['name']} | {f.get('rows', f.get('files', ''))} | {f.get('bytes', '')} | `{f['sha256']}` |")
    lines += ["", "## What the ok summary rows recorded, per arm", "",
              "Values with the number of rows recording them; `<absent>` counts rows made before the field existed.", ""]
    for arm, fields in sorted(d["recorded"].items(), key=lambda kv: str(kv[0])):
        lines += [f"**{arm}**", "", "| field | values |", "|---|---|"]
        for f, counts in fields.items():
            lines.append(f"| {f} | " + ", ".join(f"`{v}` × {n}" for v, n in sorted(counts.items())) + " |")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return [js, md]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("describe",))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    for p in write_description(Path(args.out_dir)):
        print(p)


if __name__ == "__main__":
    main()
