"""Choosing each arm's one set with irace instead of an argmax, scored in headroom.

``hpo_compare`` picks an arm's set by an argmax over ONE tuning draw per grid
point. On a fold whose seed noise is larger than the gaps being measured
(kdd99, shuttle) that argmax is partly a draw of the noise. This module hands
the same choice to irace (Lopez-Ibanez et al., "The irace package: Iterated
racing for automatic algorithm configuration", Operations Research
Perspectives 3, 2016): candidates race over (dataset, seed) instances,
statistically worse ones are dropped early, and only the survivors pay for
more seeds.

Every setting below follows the irace user guide (CRAN vignette of irace 4.5),
cited by section:

* **cost = -headroom.** irace minimises. Headroom is
  ``(ari - default) / (ceiling - default)`` against the per-dataset reference
  ``hpo_compare.fold_reference`` builds from the grid at the tuning seeds, so
  the grid arms and the irace arms are measured with one yardstick. Only
  datasets that pass the tunability gate become instances.
* **t-test, not irace's default F-test.** Friedman ranks configurations within
  each instance, and a rank is unchanged by any per-dataset affine map -- under
  the F-test, headroom would change nothing at all. Sec. 10.7 recommends the
  t-test "whenever the best configurations should obtain the best average
  solution cost", and a mean over instances is only meaningful when the costs
  are commensurable (Demsar, JMLR 2006, Sec. 3.1.1), which is what headroom is
  for.
* **One block = one seed across every optimisation dataset** (``blockSize``,
  Sec. 10.6): a configuration is never eliminated before it has been seen on
  every dataset the same number of times. ``firstTest``/``eachTest`` are
  counted in blocks, and irace 4.5 requires ``mu`` to be a multiple of
  ``blockSize`` (not in the guide; it stops with an error otherwise).
* **Deterministic, instances in file order.** An instance is ``fold@seed``; the
  seed fixes both the permutation and the model seed, so a (configuration,
  instance) pair re-run would repeat the same number (Sec. 5.4). Seed blocks
  are exchangeable, so the file order costs nothing statistically, and it lets
  seed 1 -- where the grid has already run every point -- come first.
* **``grid`` space = one race over every grid point.** FAQ 12.11: when the space
  is small enough to enumerate, give irace all configurations and run a single
  race (``nbIterations=1``). The picks are then grid points, the grid's runs
  are reused through the shared ``runs.jsonl``, and the comparison with the
  grid arms isolates the selector. ``continuous`` races over the grid's
  bounding box with iterated racing instead.
* **A failed run costs Inf**: immediate rejection (Sec. 10.9), the same verdict
  ``seed_mean`` gives a grid cell that failed on any seed.

The target runner is this module (``python -m hpo_pipeline.irace_tune target``),
launched through a generated shell wrapper. It looks every run up in the shared
``runs.jsonl`` first and only executes what is missing, through
``hpo_compare.run_one`` -- the same subprocess, hash seed and row format as
every other run of the pilot.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: Instance seeds, one block each. Seed 1 is `hpo_compare.TUNE_SEEDS` -- the
#: grid already holds every point there -- and the rest must stay disjoint from
#: `hpo_compare.FINAL_SEEDS`, or a set would be scored on a draw that chose it.
#: A runnable placeholder: the number of seeds caps how many instances a
#: configuration can be evaluated on (seeds x datasets).
IRACE_SEEDS = (1, 7, 8, 9, 10)
#: maxExperiments per irace run, i.e. target-runner calls (a P2R call runs a
#: whole ladder). Sec. 10.1 calls 1 000 - 100 000 typical; this is the low end,
#: and a placeholder. A grid race cannot spend more than configurations x
#: instances whatever the budget.
IRACE_BUDGET = 1000
#: First elimination test after this many blocks, then one every IRACE_EACH_TEST
#: blocks. Sec. 10.5 uses firstTest=2 for ML folds; placeholders.
IRACE_FIRST_TEST = 2
IRACE_EACH_TEST = 1
IRACE_TESTS = ("t-test", "t-test-bonferroni", "t-test-holm", "F-test")
IRACE_TEST = "t-test"
#: irace's own default.
IRACE_CONFIDENCE = 0.95
#: irace's RNG seed, fixed so that a race can be reproduced.
IRACE_RNG_SEED = 20260924
IRACE_SPACES = ("grid", "continuous")
IRACE_ARMS = ("raw", "ladder", "p2r")
#: The P2R reading irace races by default: the one `final_eval.ARMS` evaluates.
#: A placeholder like the rest; `--irace-p2r` names another.
IRACE_P2R_DEFAULT = "R_knee:matr_trace_win"
#: In `continuous` space a numeric axis is sampled on a log scale when its grid
#: spans at least this ratio -- the grids here are geometric ladders.
LOG_RATIO = 4.0
#: What the target runner prints for a run that failed or a curve that broke.
REJECT = math.inf


def _hpo():
    # Imported lazily: `hpo_compare` imports this module for its irace phase.
    from . import hpo_compare

    return hpo_compare


# --- parameters ----------------------------------------------------------------
def axes_of(algo: str, arm: str, grids_spec: dict, rule: str | None = None) -> list[dict]:
    """The irace parameters of one arm, in a fixed order.

    ``raw`` / ``ladder``: the resolution axis plus the shape. On a scaled ladder
    the axis is the dimensionless ``c`` of ``epsilon = c * s(D)``, not the
    threshold. ``p2r``: the shape only -- the level is read off each dataset's
    own label-free curve, which is the point of P2R -- plus ``phi`` when the
    rule has one to fit.
    """
    from .rules import RULES
    from .space import GRAN

    if arm not in IRACE_ARMS:
        raise ValueError(f"irace arm must be one of {IRACE_ARMS}, got {arm!r}")
    axis, kind = GRAN[algo]
    spec = grids_spec[algo]
    grid = "raw" if arm == "raw" else "ladder"
    out = []
    if arm != "p2r":
        name = "c" if (grid == "ladder" and kind == "scaled") else axis
        out.append({"name": name, "role": "axis", "values": list(spec["levels"][grid])})
    for key in sorted(spec["shape"][grid]):
        out.append({"name": key, "role": "shape", "values": list(spec["shape"][grid][key])})
    if arm == "p2r":
        _, phis = RULES[rule]
        if any(p is not None for p in phis):
            out.append({"name": "phi", "role": "phi", "values": list(phis)})
    return out


def _token(value) -> str:
    """Exact text of a grid value; `values_of` maps it back by lookup, never by parsing."""
    return str(value) if isinstance(value, (int, str)) else repr(value)


def _is_int(values) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def parameters_text(axes: list[dict], space: str) -> str:
    """irace parameter file (guide Sec. 5.1.7). Each value reaches the target as ``--name=value``."""
    if space not in IRACE_SPACES:
        raise ValueError(f"irace space must be one of {IRACE_SPACES}, got {space!r}")
    lines = ["# name label type domain -- written by hpo_pipeline.irace_tune"]
    for ax in axes:
        label = f'"--{ax["name"]}="'
        values = ax["values"]
        if space == "grid" or ax["role"] == "phi":
            # irace 4.5 rejects an ordinal domain that reads as numbers unless it
            # is increasing ("appears to be a discretization of a numerical
            # range"); the grid lists thresholds coarse -> fine, i.e. decreasing.
            # A single race never samples, so the order carries no meaning here.
            ordered = sorted(values) if all(isinstance(v, (int, float)) for v in values) \
                else list(values)
            domain = ", ".join(f'"{_token(v)}"' for v in ordered)
            lines.append(f'{ax["name"]} {label} o ({domain})')
            continue
        lo, hi = min(values), max(values)
        kind = "i" if _is_int(values) else "r"
        if lo > 0 and hi / lo >= LOG_RATIO:
            kind += ",log"
        if kind.startswith("i"):
            lines.append(f'{ax["name"]} {label} {kind} ({int(lo)}, {int(hi)})')
        else:
            lines.append(f'{ax["name"]} {label} {kind} ({float(lo)!r}, {float(hi)!r})')
    return "\n".join(lines) + "\n"


def grid_points(algo: str, arm: str, grids_spec: dict, rule: str | None = None) -> list[dict]:
    """Every grid configuration of an arm, as {parameter: value}.

    Built from `build_grid`'s accepted candidates, so a point the constraints
    reject is never raced. A P2R shape is kept only if its whole ladder was
    accepted: a rule cannot be applied to a curve with a hole in it.
    """
    from .rules import RULES
    from .space import GRAN

    H = _hpo()
    axis, kind = GRAN[algo]
    grid = "raw" if arm == "raw" else "ladder"
    accepted, _ = H.build_grid(algo, grid, grids_spec[algo])
    shape_keys = sorted(grids_spec[algo]["shape"][grid])
    if arm != "p2r":
        name = "c" if (grid == "ladder" and kind == "scaled") else axis
        return [{name: c["g"], **{k: c["shape"][k] for k in shape_keys}} for c in accepted]
    n_levels = len(grids_spec[algo]["levels"][grid])
    by_theta: dict[int, list[dict]] = {}
    for c in accepted:
        by_theta.setdefault(c["theta"], []).append(c)
    points = []
    for theta in sorted(by_theta):
        cells = by_theta[theta]
        if len(cells) != n_levels:
            continue
        points.append({k: cells[0]["shape"][k] for k in shape_keys})
    _, phis = RULES[rule]
    if any(p is not None for p in phis):
        return [{**p, "phi": phi} for p in points for phi in phis]
    return points


def configurations_text(axes: list[dict], points: list[dict]) -> str:
    """irace configurations file (Sec. 5.5): the header, then one row per point."""
    names = [ax["name"] for ax in axes]
    rows = [" ".join(names)]
    for point in points:
        rows.append(" ".join(f'"{_token(point[n])}"' for n in names))
    return "\n".join(rows) + "\n"


def values_of(tokens: dict[str, str], axes: list[dict], space: str) -> dict:
    """``--name=value`` strings (or a results table row) back to Python values."""
    out = {}
    for ax in axes:
        raw = tokens.get(ax["name"])
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            raise ValueError(f"irace passed no value for {ax['name']!r}")
        text = str(raw).strip().strip('"')
        if space == "grid" or ax["role"] == "phi":
            lookup = {_token(v): v for v in ax["values"]}
            if text not in lookup:
                # `read.csv` in the result dump turns an ordinal "5" into 5.0
                # and "0.25" into 0.25; match those by value, never by position.
                hits = [v for v in ax["values"]
                        if not isinstance(v, str) and _close(float(v), float(text))]
                if len(hits) != 1:
                    raise ValueError(f"{ax['name']}={text!r} is not one of {list(lookup)}")
                out[ax["name"]] = hits[0]
            else:
                out[ax["name"]] = lookup[text]
        elif _is_int(ax["values"]):
            out[ax["name"]] = int(round(float(text)))
        else:
            out[ax["name"]] = float(text)
    return out


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def parse_switches(switches: list[str]) -> dict[str, str]:
    out = {}
    for item in switches:
        if not item.startswith("--") or "=" not in item:
            raise ValueError(f"unexpected irace argument {item!r}")
        name, _, value = item[2:].partition("=")
        out[name] = value
    return out


def parse_instance(instance: str) -> tuple[str, int]:
    fold, sep, seed = instance.strip().strip('"').rpartition("@")
    if not sep:
        raise ValueError(f"instance {instance!r} is not fold@seed")
    return fold, int(seed)


def instances_text(folds: list[str], seeds) -> str:
    """One block per seed, every optimisation dataset once inside each block."""
    return "".join(f"{fold}@{seed}\n" for seed in seeds for fold in folds)


# --- candidates ----------------------------------------------------------------
def candidates_for(algo: str, arm: str, space: str, grids_spec: dict, values: dict) -> list[dict]:
    """The runs one irace configuration stands for: one candidate, or a ladder (P2R).

    In ``grid`` space the candidate is rebuilt with exactly the arguments
    `build_grid` used, so it hashes to the grid's `config_id` and every run the
    grid already made is reused. In ``continuous`` space the id carries an
    ``irace-`` level tag, so a point that merely rounds to a grid value is not
    mistaken for it; ``hpo_compare`` keeps those ids out of every grid table.
    A P2R ladder keeps the grid's level index in its id -- the ladder is the
    grid's -- so a continuous shape that equals a grid shape reuses its runs.
    """
    from .space import GRAN

    H = _hpo()
    axis, kind = GRAN[algo]
    spec = grids_spec[algo]
    grid = "raw" if arm == "raw" else "ladder"
    shape_spec = spec["shape"][grid]
    point = {k: values[k] for k in sorted(shape_spec)}
    fixed = H.fixed_params(algo, grid, spec)
    points = H.shape_points(shape_spec)
    theta = points.index(point) + 1 if point in points else None
    levels = list(spec["levels"][grid])
    continuous = space == "continuous"
    if arm == "p2r":
        cells = [H._candidate(algo, level, g, {**fixed, **point}, grid="ladder", theta=theta)
                 for level, g in enumerate(levels, start=1)]
        return [_mark(c, continuous and theta is None) for c in cells]
    g = values["c" if (grid == "ladder" and kind == "scaled") else axis]
    level = levels.index(g) + 1 if g in levels else None
    if not continuous and (level is None or theta is None):
        raise ValueError(f"{algo} {arm}: {values} is not a grid point")
    if grid == "raw":
        shape = {**fixed, **point, axis: g}
        id_level = "irace-raw" if continuous else f"raw{level}"
    else:
        shape = {**fixed, **point}
        id_level = "irace-ladder" if continuous else None
    cand = H._candidate(algo, None if continuous else level, g, shape, grid=grid,
                        theta=None if continuous else theta, id_level=id_level)
    return [_mark(cand, continuous)]


def _mark(cand: dict, off_grid: bool) -> dict:
    if off_grid:
        cand["irace"] = True
    return cand


# --- the target runner ---------------------------------------------------------
def locked_append(path: Path, row: dict) -> None:
    """Append one JSON line under an exclusive lock: irace runs targets in parallel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _latest_rows(runs_path: Path) -> dict:
    """Newest row per run key, read under a shared lock so a line another
    target is still writing is never half-read."""
    H = _hpo()
    latest = {}
    runs_path = Path(runs_path)
    if not runs_path.exists():
        return latest
    with open(runs_path, encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            lines = handle.readlines()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    for line in lines:
        line = line.strip()
        if line:
            row = json.loads(line)
            latest[H._row_key(row)] = row
    return latest


def fetch_or_run(ctx: dict, cands: list[dict], fold: str, seed: int,
                 runner=None) -> tuple[list[dict], int]:
    """The rows for ``cands`` on (fold, seed): from ``runs.jsonl`` when there,
    executed and appended when not. Returns (rows, number of runs executed).

    A failed row on disk counts as done, exactly as in `hpo_compare.pending`:
    a timeout is a fact about the configuration.
    """
    H = _hpo()
    runner = runner or (lambda job: H.run_one(job, ctx["timeout"]))
    runs_path = Path(ctx["out_dir"]) / "runs.jsonl"
    known = _latest_rows(runs_path)
    probes = {fold: ctx["probes"][fold]}
    rows, executed = [], 0
    for cand in cands:
        job = H._job(ctx["algo"], cand, fold, seed, ctx["n_samples"], probes, ctx["tuned"])
        row = known.get(H._job_key(job))
        if row is None:
            row = runner(job)
            locked_append(runs_path, row)
            executed += 1
        rows.append(row)
    return rows, executed


def cost_of(ctx: dict, values: dict, fold: str, rows: list[dict]) -> tuple[float, dict]:
    """-headroom of the configuration on this instance, or REJECT.

    For P2R the rows are the ladder at the configuration's shape; the rule reads
    the label-free curve and the cost is the headroom at the level it picks.
    """
    import numpy as np

    from .rules import RULES, orient

    ref = ctx["reference"][fold]
    if any(r.get("status") != "ok" for r in rows):
        return REJECT, {"why": "run failed"}
    if ctx["arm"] == "p2r":
        crit = ctx["criterion"]
        curve = np.array([_float(r.get(crit)) for r in rows], dtype=float)
        if not np.isfinite(curve).all():
            return REJECT, {"why": f"{crit} curve broken"}
        rule, _ = RULES[ctx["rule"]]
        index = int(rule(orient(curve, crit), values.get("phi")))
        ari, detail = _float(rows[index].get("ari")), {"level": index + 1}
    else:
        ari, detail = _float(rows[0].get("ari")), {}
    if not math.isfinite(ari):
        return REJECT, {"why": "ari missing"}
    headroom = (ari - ref["default"]) / ref["gain"]
    return -headroom, {**detail, "ari": ari, "headroom": headroom}


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def evaluate(ctx: dict, values: dict, fold: str, seed: int, runner=None) -> tuple[float, dict]:
    """One target-runner call: resolve, check constraints, fetch or run, score."""
    H = _hpo()
    cands = candidates_for(ctx["algo"], ctx["arm"], ctx["space"], ctx["grids_spec"], values)
    probe = ctx["probes"][fold]
    for cand in cands:
        broken = H._violations(ctx["algo"], H.resolve_candidate(cand, probe))
        if broken:
            return REJECT, {"why": f"violates {broken}", "executed": 0}
    rows, executed = fetch_or_run(ctx, cands, fold, seed, runner)
    cost, detail = cost_of(ctx, values, fold, rows)
    return cost, {**detail, "executed": executed,
                  "config_ids": [c["config_id"] for c in cands]}


def _format_cost(cost: float) -> str:
    return "Inf" if math.isinf(cost) and cost > 0 else f"{cost:.12g}"


def target_main(argv: list[str]) -> int:
    """``target <context.json> <id_conf> <id_inst> <seed> <instance> --name=value ...``.

    irace's own seed is ignored: the instance names the seed (deterministic
    mode). A Python error is not data -- it exits non-zero so irace stops, as
    it does by default on a failing target runner (Sec. 10.9).
    """
    ctx_path, conf_id, inst_id, _irace_seed, instance, *switches = argv
    ctx = json.loads(Path(ctx_path).read_text(encoding="utf-8"))
    fold, seed = parse_instance(instance)
    values = values_of(parse_switches(switches), ctx["axes"], ctx["space"])
    started = time.time()
    cost, detail = evaluate(ctx, values, fold, seed)
    locked_append(Path(ctx["run_dir"]) / "target.jsonl", {
        "configuration": conf_id, "instance_id": inst_id, "instance": instance,
        "values": values, "cost": _format_cost(cost), "seconds": round(time.time() - started, 3),
        **{k: v for k, v in detail.items()},
    })
    print(_format_cost(cost))
    return 0


# --- running irace ---------------------------------------------------------------
def find_rscript(explicit: str | None = None) -> str:
    """``--rscript``, then ``$RSCRIPT``, then ``Rscript`` on PATH."""
    for candidate in (explicit, os.environ.get("RSCRIPT"), shutil.which("Rscript")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise SystemExit(
        "irace needs R: no Rscript found (--rscript, $RSCRIPT or PATH). Install R and "
        "then, in R, install.packages(\"irace\") -- e.g. on Debian/Ubuntu "
        "`sudo apt install r-base` followed by "
        "`Rscript -e 'install.packages(\"irace\", repos=\"https://cloud.r-project.org\")'`."
    )


def irace_version(rscript: str) -> str:
    proc = subprocess.run(
        [rscript, "-e", 'cat(as.character(utils::packageVersion("irace")))'],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"R found at {rscript} but the irace package is not installed:\n"
            f"{proc.stderr.strip()[-400:]}\n"
            "Install it with: Rscript -e 'install.packages(\"irace\", "
            "repos=\"https://cloud.r-project.org\")'"
        )
    return proc.stdout.strip()


def _wrapper_text(ctx_path: Path) -> str:
    """The executable irace calls. Absolute paths only: irace runs it from execDir."""
    from ._common import REPO_ROOT

    python = sys.executable
    return (
        "#!/bin/sh\n"
        f'PYTHONPATH="{REPO_ROOT}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f'exec "{python}" -m hpo_pipeline.irace_tune target "{ctx_path}" "$@"\n'
    )


def scenario_text(run_dir: Path, *, n_blocks_folds: int, budget: int, workers: int,
                  test: str, confidence: float, first_test: int, each_test: int,
                  rng_seed: int, n_grid_points: int | None) -> str:
    """irace scenario file (Sec. 11). ``n_grid_points`` set means one race over all of them."""
    lines = [
        f'parameterFile = "{run_dir / "parameters.txt"}"',
        f'trainInstancesFile = "{run_dir / "instances.txt"}"',
        'trainInstancesDir = ""',
        f'targetRunner = "{run_dir / "target-runner"}"',
        f'execDir = "{run_dir / "exec"}"',
        f'logFile = "{run_dir / "exec" / "irace.Rdata"}"',
        "deterministic = 1",
        "sampleInstances = 0",
        f"blockSize = {n_blocks_folds}",
        f"firstTest = {first_test}",
        f"eachTest = {each_test}",
        # irace 4.5 stops unless mu is a multiple of blockSize; mu is also raised
        # to firstTest by irace itself, so this is the smallest legal value.
        f"mu = {first_test * n_blocks_folds}",
        f'testType = "{test}"',
        f"confidence = {confidence}",
        f"maxExperiments = {budget}",
        f"parallel = {workers if workers > 1 else 0}",
        f"seed = {rng_seed}",
    ]
    if n_grid_points is not None:
        lines += [
            f'configurationsFile = "{run_dir / "configurations.txt"}"',
            "nbIterations = 1",
            f"nbConfigurations = {n_grid_points}",
        ]
    return "\n".join(lines) + "\n"


#: R code that turns the irace log into plain CSVs; base R only.
_DUMP_R = r"""
args <- commandArgs(trailingOnly = TRUE)
suppressMessages(library(irace))
r <- read_logfile(args[1])
out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
conf <- r$allConfigurations
write.csv(conf, file.path(out, "configurations.csv"), row.names = FALSE)
elites <- if (!is.null(r$psrace_log) && length(r$psrace_log$elites) > 0) {
  r$psrace_log$elites } else { r$allElites[[length(r$allElites)]] }
write.csv(data.frame(rank = seq_along(elites), id = elites),
          file.path(out, "elites.csv"), row.names = FALSE)
exps <- r$experiments
rows <- as.integer(rownames(exps))
long <- data.frame(row = rep(rows, times = ncol(exps)),
                   id = rep(colnames(exps), each = nrow(exps)),
                   cost = as.vector(exps))
long <- long[!is.na(long$cost), ]
inst <- r$state$instances_log
long$instance <- r$scenario$instances[inst$instanceID[long$row]]
write.csv(long, file.path(out, "experiments.csv"), row.names = FALSE)
writeLines(as.character(r$state$rejected_ids), file.path(out, "rejected.txt"))
writeLines(as.character(r$irace_version), file.path(out, "version.txt"))
"""


def _hash(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]


def race(*, algo: str, arm: str, space: str, opt_folds: list[str], grids_spec: dict,
         reference: dict, probes: dict, tuned: list[str], n_samples, timeout: int,
         workers: int, run_dir: Path, out_dir: Path, rscript: str,
         rule: str | None = None, criterion: str | None = None,
         seeds=IRACE_SEEDS, budget: int = IRACE_BUDGET, test: str = IRACE_TEST,
         confidence: float = IRACE_CONFIDENCE, first_test: int = IRACE_FIRST_TEST,
         each_test: int = IRACE_EACH_TEST, rng_seed: int = IRACE_RNG_SEED,
         force: bool = False, run: bool = True) -> dict | None:
    """One irace run: write the scenario, race, and return the parsed result.

    A finished run whose inputs are unchanged is reused (``result.json`` carries
    their hash); every target-runner call is itself cached in ``runs.jsonl``,
    so a re-race after a change of settings only pays for runs it has not
    made before. ``run=False`` only reads: the finished result when its inputs
    still match, else None -- a stale race is never silently reported.
    """
    if test not in IRACE_TESTS:
        raise ValueError(f"irace test must be one of {IRACE_TESTS}, got {test!r}")
    if first_test > len(seeds):
        raise SystemExit(f"--irace-first-test {first_test} needs at least that many seeds; "
                         f"got {list(seeds)}")
    # irace is started with cwd=run_dir and starts the target runner from its
    # execDir, so every path handed to either must be absolute; a relative
    # run_dir or out_dir would otherwise be resolved twice.
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    axes = axes_of(algo, arm, grids_spec, rule)
    points = grid_points(algo, arm, grids_spec, rule) if space == "grid" else None
    block = len(opt_folds)
    if points is not None:
        floor = len(points) * first_test * block
        if budget < floor:
            raise SystemExit(
                f"irace budget {budget} is below what one race needs to reach its first test "
                f"({len(points)} configurations x {first_test} blocks x {block} datasets = "
                f"{floor}); raise --irace-budget or --irace-first-test down")
    ctx = {
        "algo": algo, "arm": arm, "space": space, "rule": rule, "criterion": criterion,
        "axes": axes, "grids_spec": grids_spec,
        "reference": {f: reference[f] for f in opt_folds},
        "probes": {f: probes[f] for f in opt_folds},
        "tuned": list(tuned), "n_samples": n_samples, "timeout": timeout,
        "out_dir": str(out_dir), "run_dir": str(run_dir),
    }
    files = {
        "parameters.txt": parameters_text(axes, space),
        "instances.txt": instances_text(opt_folds, seeds),
        "scenario.txt": scenario_text(
            run_dir, n_blocks_folds=block, budget=budget, workers=workers, test=test,
            confidence=confidence, first_test=first_test, each_test=each_test,
            rng_seed=rng_seed, n_grid_points=None if points is None else len(points)),
    }
    if points is not None:
        files["configurations.txt"] = configurations_text(axes, points)
    # `parallel` and `timeout` change how the race runs, not what it computes.
    inputs = _hash({k: v for k, v in files.items() if k != "scenario.txt"},
                   [line for line in files["scenario.txt"].splitlines()
                    if not line.startswith("parallel")],
                   {k: v for k, v in ctx.items() if k not in ("timeout",)})
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("inputs") == inputs:
            if run:
                print(f"  irace {algo} {arm} [{', '.join(opt_folds)}]: reusing {result_path}")
            return previous
    if not run:
        return None
    if (run_dir / "exec").exists():
        shutil.rmtree(run_dir / "exec")
    (run_dir / "exec").mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (run_dir / name).write_text(text, encoding="utf-8")
    ctx_path = run_dir / "context.json"
    ctx_path.write_text(json.dumps(ctx, indent=1, default=str), encoding="utf-8")
    wrapper = run_dir / "target-runner"
    wrapper.write_text(_wrapper_text(ctx_path), encoding="utf-8")
    wrapper.chmod(0o755)
    (run_dir / "target.jsonl").unlink(missing_ok=True)

    started = time.time()
    print(f"  irace {algo} {arm} [{', '.join(opt_folds)}]: "
          f"{'1 race over ' + str(len(points)) + ' grid points' if points else 'iterated racing'}, "
          f"{len(seeds)} seed blocks x {block} datasets, budget {budget}, {test}")
    with open(run_dir / "irace.out", "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [rscript, "-e",
             "suppressMessages(library(irace)); irace_cmdline(commandArgs(trailingOnly = TRUE))",
             "--scenario", str(run_dir / "scenario.txt")],
            cwd=str(run_dir), stdout=log, stderr=subprocess.STDOUT, text=True,
        )
    tail = (run_dir / "irace.out").read_text(encoding="utf-8")[-1500:]
    if proc.returncode != 0:
        raise SystemExit(f"irace failed for {algo} {arm} (exit {proc.returncode}); "
                         f"see {run_dir / 'irace.out'}:\n{tail}")
    dump = subprocess.run(
        [rscript, "-e", _DUMP_R, str(run_dir / "exec" / "irace.Rdata"), str(run_dir / "results")],
        capture_output=True, text=True,
    )
    if dump.returncode != 0:
        raise SystemExit(f"could not read the irace log for {algo} {arm}:\n{dump.stderr[-800:]}")
    result = read_result(run_dir / "results", axes, space, run_dir / "target.jsonl")
    result.update({"inputs": inputs, "algo": algo, "arm": arm, "space": space,
                   "rule": rule, "criterion": criterion, "opt_folds": list(opt_folds),
                   "seeds": list(seeds), "budget": budget, "test": test,
                   "confidence": confidence, "first_test": first_test,
                   "n_grid_points": None if points is None else len(points),
                   "wall_s": round(time.time() - started, 1)})
    result_path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def read_result(results_dir: Path, axes: list[dict], space: str,
                target_log: Path | None = None) -> dict:
    """Parse the CSVs `_DUMP_R` wrote into the pieces `hpo_compare` needs.

    ``elites`` are best first, as irace ranks them (post-selection race when
    it ran, else the last iteration). Every elite carries how many instances it
    was evaluated on and its mean cost there, because an elite seen on one
    block is a much weaker pick than one seen on five.
    """
    import pandas as pd

    results_dir = Path(results_dir)
    conf = pd.read_csv(results_dir / "configurations.csv")
    elites = pd.read_csv(results_dir / "elites.csv").sort_values("rank")
    exps = pd.read_csv(results_dir / "experiments.csv")
    exps["cost"] = pd.to_numeric(exps["cost"], errors="coerce")
    names = [ax["name"] for ax in axes]
    by_id = {}
    for rec in conf.to_dict("records"):
        tokens = {n: rec.get(n) for n in names}
        by_id[str(int(rec[".ID."]))] = values_of(tokens, axes, space)
    stats = exps.groupby(exps["id"].astype(str))["cost"].agg(["count", "mean"])
    rejected_file = results_dir / "rejected.txt"
    rejected = [line.strip() for line in rejected_file.read_text().splitlines()
                if line.strip()] if rejected_file.exists() else []
    elite_rows = []
    for eid in elites["id"].astype(str):
        n_inst = int(stats.loc[eid, "count"]) if eid in stats.index else 0
        mean_cost = float(stats.loc[eid, "mean"]) if eid in stats.index else math.nan
        elite_rows.append({"id": eid, "values": by_id[eid], "n_instances": n_inst,
                           "mean_headroom": -mean_cost if math.isfinite(mean_cost) else None})
    executed = calls = 0
    if target_log is not None and Path(target_log).exists():
        from ._common import read_jsonl

        for rec in read_jsonl(Path(target_log)):
            calls += 1
            executed += int(rec.get("executed") or 0)
    version = (results_dir / "version.txt").read_text().strip() \
        if (results_dir / "version.txt").exists() else None
    return {
        "elites": elite_rows,
        "best": elite_rows[0]["values"] if elite_rows else None,
        "n_configurations": int(len(conf)),
        "n_experiments": int(len(exps)),
        "n_rejected": len(rejected),
        "runs_executed": executed,
        "target_calls": calls,
        "irace_version": version,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "target":
        print("usage: python -m hpo_pipeline.irace_tune target <context.json> <irace args...>\n"
              "(called by irace; the irace phase is run through hpo_compare --irace ...)",
              file=sys.stderr)
        return 2
    return target_main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
