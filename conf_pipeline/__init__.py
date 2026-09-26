"""Online tuning + benchmark in one pass: confStream per algorithm.

The HPO pipeline tunes offline on shuffled prefixes and then scores a fixed
configuration. This pipeline does what a deployment would have to do: every
algorithm runs as a confStream ensemble (Carnein et al., 2020) over the stream
in its natural order, re-evaluating and re-sampling its own configuration after
every window, and what gets reported is the configuration the ensemble would
have handed a user at that point -- scored on the *next* window, so the report
is never the winner's own training window.

Protocol, per (algorithm, fold, seed, arm)::

    window 0        calibration: estimate the distance unit s (median pairwise
                    distance), build the ensemble at River's defaults, replay
    window t >= 1   every member processes the window; batch silhouette per
                    member; test members resolved; incumbent named; e_test new
                    members spawned and replayed on this window

Arms::

    confstream      e_size members, e_test challengers per window (the tuner)
    default         one member at River's defaults, never changed (the baseline
                    at 1/e_size of the cost)

Run from the repo root::

    python -m conf_pipeline.run --dry-run
    python -m conf_pipeline.run --workers 8
    python -m conf_pipeline.report

Layout::

    conf_pipeline/
    |-- _common.py   paths, protocol constants, helpers
    |-- spaces.py    the search space of every algorithm in the protocol
    |-- run.py       driver (one ensemble over one fold) + CLI (parent/child)
    |-- report.py    tables and figures from out/
    `-- out/         windows.jsonl, runs.jsonl, members/, report/

The protocol constants in ``_common.py`` are runnable placeholders (see
AGENTS.md rule 2): every one of them is a CLI flag.
"""
