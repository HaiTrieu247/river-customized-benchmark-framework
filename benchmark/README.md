# `benchmark/` — Streaming Clustering Concept-Drift Benchmark

Implementation of the design in
[`../streaming_clustering_concept_drift_benchmark_direction.md`](../streaming_clustering_concept_drift_benchmark_direction.md).

The framework answers not just *"which algorithm has the highest final clustering
quality?"* but *"which adapts best when the cluster structure changes over time,
at a reasonable compute/memory cost?"*

```
Synthetic Stream → Ground-Truth Cluster Evolution → Drift Events / States
    → Streaming Model → Observed Clusters → Evaluator
    → (Cluster Quality, Drift Adaptation, Efficiency)
```

## Layout

| Package | Role | Doc section |
|---|---|---|
| `ground_truth/` | `ClusterState`, `DriftEvent`, `GroundTruthTracker` — describe *how* structure evolves | §3–§7 |
| `datasets/synthetic/` | `SyntheticStream` generator + the `S1..S10` scenario matrix | §6, §16 |
| `algorithms/` | Uniform `ClustererAdapter` over River clusterers + preset registry | §9, §22 |
| `evaluation/` | Quality metrics, adaptation metrics, efficiency, the online runner, aggregation | §10–§15, §20 |
| `reports/` | Comparison tables (§18) and plots (§23) | §18, §23 |
| `experiments/` | Harness + per-phase entry points (stationary / drift / efficiency / hyperparameter) | §14–§20 |

## Requirements

River is imported from the **sibling source tree** `../river` (auto-added to
`sys.path` by `algorithms/adapter.py:build_river_path`). Install the rest with:

```bash
pip install -r requirements.txt
```

## Running

```bash
# Phase 2 — stationary baseline (§17)
python -m benchmark stationary

# Phase 3-4 — drift adaptation on any scenario (§12, §18)
python -m benchmark drift S2_abrupt_movement
python -m benchmark drift S6_split --n-samples 20000

# Phase 5 — hyperparameter calibration on disjoint streams (§19)
python -m benchmark calibrate

# Phase 6 — efficiency & scalability (§13)
python -m benchmark efficiency

# Full S1..S10 matrix, repeated over seeds → JSON digest (§16, §20)
python -m benchmark matrix --seeds 1,2,3 --out results.json
```

## Scenario matrix (§16)

| Id | Scenario | Purpose |
|---|---|---|
| `S1_stationary` | no drift | baseline |
| `S2_abrupt_movement` | abrupt move | tracking |
| `S3_gradual_movement` | gradual move (interval) | slow adaptation |
| `S4_birth` | new cluster | detect new structure |
| `S5_death` | cluster disappears | drop obsolete structure |
| `S6_split` | one → many | structural adaptation |
| `S7_merge` | many → one | structural adaptation |
| `S8_recurring` | concept returns | recurrence |
| `S9_multiple_drift` | several events | robustness over time |
| `S10_noise_drift` | drift + background noise | robustness |

## Metrics

- **Quality (§11):** ARI, NMI, macro-average purity — computed *per time window*,
  not just at the end.
- **Adaptation (§12):** performance drop, recovery time, adaptation delay,
  post-drift quality, area under recovery curve — all derived from the
  quality-over-time curve, so algorithms **without** a drift detector are still
  evaluated on how well they adapt (§9).
- **Efficiency (§13):** runtime, throughput, ms/sample, model memory
  (`_raw_memory_usage`), micro-cluster count.

## Key principles enforced

1. Ground truth never leaks into the model — the runner passes only the feature
   dict; ground truth is used solely by the evaluator (§25 Principle 1).
2. Evaluation is over time, not a single final score (Principle 2).
3. Drift *adaptation* is central; a drift *detector* is not required (Principle 3).
4. Hyperparameters are calibrated on separate streams then frozen (Principle 4).
5. Fixed seed / stream / windows / params make runs reproducible (Principle 5).

## Programmatic use

```python
from benchmark.experiments import BenchmarkConfig, run_scenario
from benchmark.reports import before_during_after_table, plot_quality_over_time

config = BenchmarkConfig(seeds=(1,), n_samples=20_000, window_size=1000)
results = run_scenario("S2_abrupt_movement", config)
print(before_during_after_table(results))
fig = plot_quality_over_time(results, "ari")
fig.savefig("s2_ari.png")
```
