# River clustering benchmark framework

A measurement framework for [River](https://riverml.xyz)'s `river.cluster` module.

It runs whatever configuration you ask for under one consistent streaming
protocol and reports the result honestly — including when a metric is
meaningless for the configuration you chose.

**What this is not:** a set of tuned results or recommended hyperparameters.
Choosing and optimising parameters and dataset settings is your call; the
framework's job is to make those choices measurable and comparable. Defaults that
ship here are runnable placeholders, and say so.

---

## Two rules

1. **The `river/` tree is never modified.** It is a vendored upstream dependency.
   Suspected bugs go in [RIVER_BUGS.md](RIVER_BUGS.md) and are worked around on
   this side of the boundary. Patching river locally would mean benchmarking a
   modified library instead of the real one — and the patch disappears on the
   next revert, which has already happened once.
2. **Build the framework, not the results.** Expose knobs and diagnostics rather
   than baking in opinions.

Full working rules: [AGENTS.md](AGENTS.md).

---

## Quick start

```bash
pip install -r requirements.txt      # river is imported from ../river, not pip

python run_benchmark.py              # synthetic stream, sensible defaults
python -m pytest tests -q            # 92 tests
```

`run_benchmark.py` is the only entry point; everything is driven from the command
line.

```bash
# your parameters, as JSON — anything unnamed falls back to river's own defaults
python run_benchmark.py --dataset kdd99 --n-samples 50000 \
    --algorithms DenStream,DBSTREAM \
    --params '{"DenStream": {"epsilon": 0.5, "mu": 2.5}}' \
    --scalers standard

# compare preprocessing chains for one algorithm
python run_benchmark.py --algorithms DBSTREAM \
    --scalers 'standard,standard_safe,variance_threshold|standard'

# a text stream and the clusterer that consumes token counts
python run_benchmark.py --dataset sms_spam --algorithms TextClust \
    --scalers bag_of_words --scaler-params '{"bag_of_words": {"on": "body"}}'

# reproducible figures — see RIVER_BUGS.md BUG-2 for why this matters
PYTHONHASHSEED=0 python run_benchmark.py --dataset insects --seeds 1,2,3
```

Results land in `benchmark_out/<dataset>/`: `report.md`, `results.json` (full
per-window and per-checkpoint trajectories), and PNG plots. `--help` lists every
flag.

---

## What it measures

| Group | Metrics |
|---|---|
| Quality (needs labels) | ARI, NMI, AMI, purity — tracked **cumulatively and per evaluation window** |
| Quality (no labels) | river's streaming silhouette (`--internal`) |
| Efficiency | runtime, throughput, ms/sample, model memory, process RSS |
| Model complexity | micro-cluster and macro-cluster counts, sampled along the stream |
| Structure | for algorithms that cluster something other than samples |
| Stability | mean ± std over seeds, with the real run count stated |

Each sample is predicted before it is learned from (test-then-train), every
setting consumes the identical stream in record order, and the environment is
recorded with every run.

Two measurement traps the framework handles explicitly:

- **Averaging the cumulative curve is not a time average.** It weights the first
  window most — exactly the cold start. The time-aware figure uses per-window
  values; the cumulative curve is reported separately.
- **A window holding a single class makes external metrics degenerate.** ARI
  collapses to a "did you lump the whole window together?" test — 1.0 for
  lumping, 0.0 for any split, regardless of clustering quality, so an
  under-segmenting model scores well for the wrong reason. The framework measures
  what share of windows are like this and warns above 30%. On KDDCup99 it is 72%.

Details and formulas: [BENCHMARK.md](BENCHMARK.md) and
[docs/benchmark_framework_report.tex](docs/benchmark_framework_report.tex).

---

## What it covers

**All 7 clusterers in `river.cluster`**, across the three protocols they actually
have:

| Protocol | Algorithms |
|---|---|
| assigns each sample to a cluster | CluStream, DenStream, DBSTREAM, STREAMKMeans, KMeans |
| clusters *variables*, not samples | ODAC — scored on structure; external metrics are undefined and omitted |
| consumes token counts, not coordinates | TextClust — needs a bag-of-words step and a text stream |

**26 unsupervised preprocessing steps**, chainable with `|`: scalers, imputers,
categorical encoders, random projections, a variance filter, text extractors,
kernel approximations, and column routing. Anything needing a target is excluded
by construction — a clustering benchmark has no `y` at learn time.

**Datasets:** a synthetic generator following MOA's `RandomRBFGeneratorEvents`
(drift, cluster merge/split/appear/disappear, controllable overlap, 7 named
scenarios), plus KDDCup99, CoverType, INSECTS, Keystroke, Shuttle, Electricity,
SMS Spam and KDDCup98. All are downloaded and cached on first use except
KDDCup98, which needs a local `cup98LRN.txt` via `--kdd98-path`. Adding your own
is a small function — see [STRUCTURE.md](STRUCTURE.md).

---

## Known river bugs

Found while building this, **recorded but not patched**:

| | Effect | Workaround here |
|---|---|---|
| BUG-1 | `StandardScaler(window_size=...)` emits **complex numbers** on a constant feature, crashing far from the cause | use the `standard_safe` step |
| BUG-2 | `KMeans` results depend on `PYTHONHASHSEED`, so `seed` does **not** make a run reproducible across processes | pin `PYTHONHASHSEED`; the environment record carries it |

Reproductions, root causes and suggested upstream fixes: [RIVER_BUGS.md](RIVER_BUGS.md).

---

## Layout

River is imported from the sibling source tree, not from pip:

```
OnlineML/
├── river/                                  # river source tree — never modified
└── river-customized-benchmark-framework/
    ├── run_benchmark.py                    # the only runner
    ├── benchmark/                          # the framework package
    │   ├── algorithms/   datasets/   evaluation/   experiments/   reports/
    ├── tests/
    └── docs/
```

`benchmark/algorithms/adapter.py` registers `../river` on `sys.path` at import
time, so no installation step is needed. If your layout differs, adjust
`build_river_path()` there.

Requirements: Python 3.10+, plus `requirements.txt`. `matplotlib` and `psutil`
are listed there but genuinely optional — without them you lose the PNG plots
and the process-RSS figures respectively, and everything else still runs.

---

## Documentation

| File | Contents |
|---|---|
| [STRUCTURE.md](STRUCTURE.md) | what each module does, how to use it, how to add a dataset |
| [BENCHMARK.md](BENCHMARK.md) | what is measured and why, in detail |
| [RIVER_BUGS.md](RIVER_BUGS.md) | suspected river bugs, reproductions, workarounds |
| [AGENTS.md](AGENTS.md) | working rules for this repo |
| [streaming_clustering_benchmark_evaluation_criteria.md](streaming_clustering_benchmark_evaluation_criteria.md) | the evaluation protocol this implements |
| [docs/benchmark_framework_report.tex](docs/benchmark_framework_report.tex) | technical report: protocol, formulas, pseudocode, validation |

---

## Legacy notebooks

`benchmark_cluster_initial.ipynb` and `evaluate_river_resource_benchmark.ipynb`
predate the `benchmark/` package and are **superseded by it**. They are kept for
reference only; they have not been updated for the current framework and are not
covered by the test suite.

## Online-tuning benchmark (`conf_pipeline/`)

Four tuning regimes on one driver (`default`, `confstream`, `random_online`, `frozen_prefix`) plus an a-posteriori oracle from a Sobol sweep, over 8 algorithms × 18 streams, with headroom, regret decomposition, drift adaptation, Table-A common reclustering and cross-fold statistics. Open `otb_benchmark.ipynb` at this directory's root with `OTB_PRESET=tiny|pilot|full`; see [conf_pipeline/README.md](conf_pipeline/README.md).
