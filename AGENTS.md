# Agent rules

Ground rules for anyone (human or agent) working in this repo.

## 1. Never modify the river repo

The `river/` tree next to this one is a **vendored upstream dependency**. Do not
edit it — not to fix a typo, not to fix a real bug, not "temporarily".

When you find a river bug:

1. Write it up in [`RIVER_BUGS.md`](RIVER_BUGS.md): symptom, minimal repro, root
   cause with `file:line`, impact on benchmarking, suggested upstream fix.
2. Work around it **inside this framework**.
3. Register any workaround under its own name (e.g. `standard_safe` alongside the
   untouched `standard`) so the pure-river behaviour stays available and choosing
   the workaround is explicit.

Why this rule exists: patching river locally means the benchmark measures a
modified library instead of the real one, and the patches vanish on the next
update or revert. That has already happened once — two fixes were reverted and
both bugs came straight back.

## 2. Scope: build the framework, not the results

The deliverable is a **framework for benchmarking and evaluating `river.cluster`**.
Hyperparameters and dataset configuration are the user's research decisions.

- Do: build measurement machinery, expose knobs, add diagnostics, keep defaults
  documented and easy to override.
- Do not: tune hyperparameters, curate "recommended" dataset lists, or bake in
  opinionated presets and present them as answers.
- When tempted to tune something, expose the knob and the diagnostic instead.

Defaults that ship with the framework are **runnable placeholders**, and should
say so.

## 3. Keep it small

One basic benchmark script (`run_benchmark.py`), not a family of scenario-specific
ones. New capability belongs in `benchmark/`, not in another top-level runner.

## 4. Report honestly

- If a measurement is degenerate or a metric undefined, say so in the output
  rather than printing a number that reads as a result.
- State what was measured and under which conditions; the environment record
  exists for this.
- Do not claim something is verified unless it was run.
