# Framework structure

What each module is for, and how to use it. The framework measures
`river.cluster`; it does not choose your hyperparameters or datasets.

```
river-customized-benchmark-framework/
├── run_benchmark.py            ← main streaming cluster benchmark runner
├── run_paper_benchmark.py      ← unified runner for scientific paper reproductions
├── AGENTS.md                   ← working rules (never modify river/, scope)
├── RIVER_BUGS.md               ← suspected river bugs + framework workarounds
├── STRUCTURE.md                ← this file
├── BENCHMARK.md                ← what is measured and why
├── streaming_clustering_benchmark_evaluation_criteria.md   ← the protocol spec
├── docs/benchmark_framework_report.tex   ← technical report (LaTeX)
├── benchmark/
│   ├── algorithms/   → wrap a clusterer, decide its protocol
│   ├── datasets/     → produce a re-iterable stream of (x, y)
│   ├── evaluation/   → run the stream, measure quality and resources
│   ├── experiments/  → run many settings / seeds on one stream
│   │   └── papers/   → paper reproductions framework (BasePaperBenchmark)
│   └── reports/      → tables, plots, environment record
└── tests/            → 90 tests
```

---

## `benchmark/algorithms/` — what is being measured

| File | Responsibility |
|---|---|
| `registry.py` | Names → clusterers and preprocessing steps, with capability metadata |
| `adapter.py` | One uniform interface over any `river.base.Clusterer` |
| `transformers.py` | Framework-side transformers river does not provide |

### `registry.py`

Holds `ALGORITHMS` (all 7 river clusterers) and `PREPROCESSORS` (25 unsupervised
transformers). `AlgorithmSpec` records what each algorithm actually supports, so
the runner adapts instead of crashing:

```python
from benchmark.algorithms.registry import get_spec, algorithms_for

get_spec("ODAC").partitions_samples      # False — clusters variables, not samples
get_spec("TextClust").input_kind         # "tokens" — needs token counts
algorithms_for()                         # numeric + sample-partitioning
algorithms_for(partitions_samples=False) # ["ODAC"]
```

`make_preprocessor(spec, **params)` builds a step or a **chain**:

```python
make_preprocessor("standard", window_size=10_000)
make_preprocessor("impute|one_hot_categorical|standard")
make_preprocessor("variance_threshold|sparse_projection|standard",
                  sparse_projection={"n_components": 16})
```

A single step takes flat params; a chain requires them nested by step name,
because two steps easily share a parameter name.

### `adapter.py`

`ClustererAdapter` gives every clusterer the same shape and handles the three
protocols river's clusterers actually have:

- assigns each sample to a cluster (5 of them, plus TextClust);
- clusters something else — ODAC builds a hierarchy over the *variables* and
  raises `NotImplementedError` from `predict_one`, so quality metrics are skipped
  and it is scored on structure;
- consumes something other than coordinates — TextClust takes token counts, and
  needs the stream position passed as `t`.

Per sample: `preprocess → predict → learn` (test-then-train). Also exposes
`n_micro_clusters()`, `n_macro_clusters()`, `structure()`, `memory_bytes()`.

### `transformers.py`

| Class | Why it exists |
|---|---|
| `OneHotCategorical` | River's `OneHotEncoder` encodes **every** key, turning a numeric column into one indicator per distinct value. This one leaves numbers alone. |
| `RealValued` | Coerces non-real/non-finite output to `0.0` — the workaround for RIVER_BUGS.md BUG-1. Registered as `standard_safe`. |

---

## `benchmark/datasets/` — what it runs on

A dataset is a **re-iterable** source of `(x, y)`. `x` is a feature dict; `y` is
ground truth used only for scoring, and `None` means "do not score this sample".
Re-iterability is what lets every setting consume the identical stream.

| File | Provides |
|---|---|
| `base.py` | `Dataset`, `IterableDataset` |
| `synthetic/generator.py` | `SyntheticStream`, `StreamConfig`, `SCENARIOS`, `make_scenario` |
| `real/kdd.py` | `load_kddcup99` (auto-download), `load_kddcup98` (needs a local `cup98LRN.txt`, CLI flag `--kdd98-path`) |
| `real/covtype.py` | `load_covtype` |
| `real/river_streams.py` | `from_river_dataset` + loaders for INSECTS, shuttle, keystroke, electricity, sms_spam |

### Synthetic generator

Follows MOA's `RandomRBFGeneratorEvents`. Ground truth is the **generating
cluster id**, which is the right target for clustering.

| Knob | Effect |
|---|---|
| `separation` | distance between the closest pair of centres, in cluster radii. The main difficulty knob: `1` overlaps heavily, `6` is trivial |
| `radius_range`, `weight_range` | unequal cluster sizes and densities |
| `noise`, `noise_in_cluster` | uniform background, labelled `None` |
| `drift_speed`, `drift_every` | centres move and bounce off the unit cube |
| `event_frequency`, `events` | clusters merge, split, appear, disappear |
| `block_ordered` | **control**: emit contiguous same-class blocks |

`separation` *derives* the radius from the layout rather than bounding it,
because the unit cube gets roomier as `d` grows and a minimum-distance
constraint stops binding — before this, `separation=0.8` and `separation=3.0`
produced identical results at `d=10`.

```python
from benchmark.datasets import make_scenario
ds = make_scenario("evolving", n_samples=50_000, n_features=20, n_clusters=8)
```

### Adding your own dataset

```python
from benchmark.datasets import IterableDataset

def load_mine(path, n_samples=None):
    def factory():                    # called fresh on every iteration
        for row in read(path):
            yield {"a": row.a, "b": row.b}, row.label
    return IterableDataset(factory, name="mine", n_features=2,
                           has_labels=True, length=n_samples)
```

Then add it to `DATASETS` in `run_benchmark.py`. For a river dataset,
`from_river_dataset(dataset, name=...)` handles it — including datasets whose
numeric columns arrive as strings.

---

## `benchmark/evaluation/` — the measurement

| File | Responsibility |
|---|---|
| `runner.py` | `run_once` — the streaming loop; `RunResult`, `Checkpoint`, `WindowQuality` |
| `clustering_quality.py` | Streaming metrics + batch helpers for cross-checking |
| `efficiency.py` | `EfficiencyMetrics` — runtime, throughput, memory, cluster counts |
| `aggregation.py` | mean ± std across seeds |

### Two quality views, tracked at once

| Field | How | Use |
|---|---|---|
| `quality` | cumulative over all scored samples | the single summary number |
| `quality_stream` | **per window**, metric reset at each boundary | the time-aware protocol figure |
| `quality_cumulative_stream` | the converging from-start curve | reference; its `final` equals `quality` |

Averaging the cumulative curve is not a time average — it weights the first
window most, which is exactly the cold start.

### Window diagnostics

`RunResult.window_label_diagnostics` reports what share of windows hold a single
class. This matters because on such a window ARI collapses to a *"did you lump
the window together?"* test — 1.0 for lumping, 0.0 for any split, regardless of
clustering quality. A model that under-segments then scores well for the wrong
reason. On KDDCup99 72% of 1000-sample windows are like this. The runner reports
it and `run_benchmark.py` warns above 30%.

### Cost note

Memory is probed **at checkpoints only**. `_raw_memory_usage` walks the whole
object graph — roughly 14 ms for a model holding a 2000-point buffer, two orders
of magnitude more than processing a sample. Peak memory is therefore a max over
checkpoints, a lower bound for models that prune.

---

## `benchmark/experiments/` — orchestration

| File | Responsibility |
|---|---|
| `harness.py` | `Setting`, `BenchmarkConfig`, `run_setting`, `run_benchmark_repeated` |
| `scalability.py` | `sweep_samples`, `sweep_features` |

```python
from benchmark.experiments import BenchmarkConfig, Setting, run_setting

setting = Setting("DBSTREAM",
                  params={"clustering_threshold": 1.0},
                  scaler="standard",
                  scaler_params={"window_size": 10_000})
result = run_setting(setting, dataset,
                     BenchmarkConfig(window_size=1000, n_checkpoints=25), seed=1)
```

Seeds are injected only into algorithms whose constructor accepts one (decided by
signature, including `**kwargs` forwarding — STREAMKMeans passes `seed` to the
KMeans it wraps).

---

## `benchmark/reports/` — output

| File | Provides |
|---|---|
| `tables.py` | efficiency, quality, per-window, structure, window-diagnostics tables |
| `plots.py` | runtime / memory / cluster-count / throughput / quality curves |
| `environment.py` | CPU, RAM, OS, Python, river version, **`PYTHONHASHSEED`** |

The hash seed is recorded because river's KMeans is sensitive to it
(RIVER_BUGS.md BUG-2): without pinning it, the same `seed` gives different
results per process.

---

## Running

```bash
# defaults — synthetic stream, sample-partitioning algorithms, no preprocessing
python run_benchmark.py

# your parameters
python run_benchmark.py --dataset kdd99 --n-samples 50000 \
    --algorithms DenStream,DBSTREAM \
    --params '{"DenStream": {"epsilon": 0.5, "mu": 2.5}}' \
    --scalers standard

# compare preprocessing chains
python run_benchmark.py --algorithms DBSTREAM \
    --scalers 'standard,standard_safe,variance_threshold|standard'

# text stream + the token-based clusterer
python run_benchmark.py --dataset sms_spam --algorithms TextClust \
    --scalers bag_of_words --scaler-params '{"bag_of_words": {"on": "body"}}'

# reproducible figures
PYTHONHASHSEED=0 python run_benchmark.py --dataset insects --seeds 1,2,3
```

Outputs land in `benchmark_out/<dataset>/`: `report.md`, `results.json` (full
window and checkpoint trajectories), and PNG plots.

## Tests

```bash
python -m pytest tests -q          # 90 tests
```

| File | Covers |
|---|---|
| `test_evaluation.py` | protocol, metrics, window semantics, seeding, diagnostics |
| `test_algorithms.py` | all 7 clusterers, the three protocols, hash-seed sensitivity |
| `test_datasets.py` | generator geometry, drift, events, ordering, river adapter |
| `test_preprocessing.py` | every registered step, chaining, the one-hot trap |
