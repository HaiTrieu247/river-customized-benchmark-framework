# `benchmark/` — Streaming Clustering Resource Benchmark

Run many River clustering algorithms — in several **settings** (params +
scaler) — on the **same dataset**, and compare them on:

- **Resource cost:** execution time, throughput (samples/s), model RAM,
  micro-cluster count, macro-cluster count, scalability as the stream grows.
- **Clustering quality** (labeled datasets only): ARI, NMI, purity.

```
Dataset ──▶ [ algorithm + params + scaler ] × N ──▶ Runner ──▶ Report
                                                       │
                     (runtime, RAM, cluster counts, quality)
```

## Layout

| Package | Role |
|---|---|
| `datasets/` | `Dataset` interface, synthetic static-blob generator, KDD99/KDD98 loaders |
| `algorithms/` | `ClustererAdapter` over River clusterers (+ optional online scaler), preset registry |
| `evaluation/` | quality metrics, efficiency metrics, the online runner, seed aggregation |
| `reports/` | comparison tables (efficiency / quality) and plots |
| `experiments/` | `Setting` + `BenchmarkConfig` harness that runs many settings on one dataset |

## Requirements

River is imported from the **sibling source tree** `../river` (auto-added to
`sys.path` by `algorithms/adapter.py:build_river_path`). Install the rest:

```bash
pip install -r requirements.txt
```

## Running (CLI)

```bash
# Synthetic labeled Gaussian blobs — compare all default algorithms
python -m benchmark synthetic --n-samples 20000 --n-clusters 5

# Compare the effect of scalers on one algorithm (README: scaler matters a lot)
python -m benchmark synthetic --algorithms DBSTREAM --scalers none,standard,minmax

# KDDCup99 (34 continuous features, auto-fetched via sklearn)
python -m benchmark kdd99 --n-samples 50000 --scalers standard

# KDDCup98 (local file, 56 RFM fields)
python -m benchmark kdd98 --path /path/to/cup98LRN.txt --n-samples 20000

# Repeat over seeds → mean ± std, and write JSON + plots
python -m benchmark synthetic --seeds 1,2,3 --out results/
```

### CLI options (common to all datasets)

| Flag | Meaning |
|---|---|
| `--algorithms` | comma-separated; default: CluStream, DenStream, DBSTREAM, STREAMKMeans, KMeans |
| `--scalers` | comma-separated online scalers: `none, standard, minmax, maxabs, robust, adaptive_standard` |
| `--seeds` | comma-separated; more than one → aggregate to mean ± std |
| `--n-samples` | cap number of streamed samples |
| `--checkpoints` | number of time/RAM/cluster measurements across the stream |
| `--out` | directory to write `results.json` + plots |

## Metrics

- **Efficiency:** `total_runtime_s`, `throughput_sps`, `ms_per_sample`,
  `peak_memory_mb` (via river's `_raw_memory_usage`, includes the scaler),
  `mean/max micro-clusters`, `final macro-clusters`. Sampled at checkpoints so
  scalability over the stream is visible, not just a final number.
- **Quality:** ARI, NMI, macro-average purity (README formula), computed over
  all labeled samples. Skipped automatically for unlabeled datasets.

## Programmatic use

```python
from benchmark.datasets import SyntheticStream, StreamConfig
from benchmark.experiments import BenchmarkConfig, Setting, run_benchmark
from benchmark.reports import efficiency_table, quality_table

dataset = SyntheticStream(StreamConfig(n_samples=20_000, n_clusters=5, seed=42))
config = BenchmarkConfig(
    settings=[
        Setting("DBSTREAM", scaler="standard"),
        Setting("DBSTREAM", params={"clustering_threshold": 2.0}, scaler="standard",
                label="DBSTREAM τ=2"),
        Setting("STREAMKMeans", params={"n_clusters": 5}),
        Setting("CluStream", params={"n_macro_clusters": 5}),
    ],
)
results = run_benchmark(dataset, config)
print(efficiency_table(results))
print(quality_table(results))
```

## Reproducibility rules enforced

- Every setting consumes the **same** dataset stream from the start (record
  order is the streaming order — no shuffling).
- Datasets are re-iterable so each algorithm gets an identical sequence.
- Runs are repeated over seeds and aggregated (`--seeds 1,2,3`).
- Each result records its full params + scaler.

## Notes on results (from the existing repo README)

- **The scaler matters a lot.** KDD99 has features with an extreme range; a
  `minmax` scaler can squash everything into one micro-cluster. Prefer
  `standard`.
- **`epsilon`/`clustering_threshold` must match the scale.** DenStream/DBSTREAM
  presets are starting points; tune the radius param to the (scaled) data.
- **River's DenStream is an online-adapted implementation**, not classic DBSCAN.
