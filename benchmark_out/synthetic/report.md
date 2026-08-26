# river.cluster benchmark — static_easy_d10_n6000

- n_samples: 6,000
- n_features: 10
- window_size: 1000
- checkpoints: 20
- seeds: [1, 2]
- labelled: True

## Environment

| Key                   | Value                                                 |
|-----------------------|-------------------------------------------------------|
| os                    | Windows 11                                            |
| platform              | Windows-11-10.0.26200-SP0                             |
| machine               | AMD64                                                 |
| processor             | AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD     |
| cpu_count_physical    | 8                                                     |
| cpu_count_logical     | 16                                                    |
| ram_total_gb          | 27.8600                                               |
| python                | 3.13.3                                                |
| python_implementation | CPython                                               |
| python_hash_seed      | 0                                                     |
| river_version         | 0.26.1                                                |
| river_path            | D:\WORK\CTY-ANH KHÔI\OnlineML\river\river\__init__.py |
| lib:river             | 0.26.0                                                |
| lib:numpy             | 2.4.3                                                 |
| lib:scipy             | 1.17.1                                                |
| lib:pandas            | 2.3.3                                                 |
| lib:scikit-learn      | 1.8.0                                                 |
| lib:matplotlib        | 3.10.9                                                |

## Configuration

- **DBSTREAM [standard]** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.01, 'cleanup_interval': 2, 'intersection_factor': 0.3, 'minimum_weight': 1.0}` | preprocessing: `standard` params: `{}`
- **STREAMKMeans [standard]** — params: `{'chunk_size': 50, 'n_clusters': 5, 'seed': 1}` | preprocessing: `standard` params: `{}`
- **ODAC [standard]** — params: `{'confidence_level': 0.9, 'n_min': 100, 'tau': 0.1}` | preprocessing: `standard` params: `{}`

> Parameters not named on the command line are River's own defaults. The framework does not tune them.

## Efficiency

| Setting                 | Runs | Runtime (s)     | Throughput (sps)      | Model MB        | Mean MC          | Max MC           | Macro K         |
|-------------------------|------|-----------------|-----------------------|-----------------|------------------|------------------|-----------------|
| DBSTREAM [standard]     | 1    | 0.4339 ± 0.0000 | 13828.0500 ± 0.0000   | 0.0240 ± 0.0000 | 12.6500 ± 0.0000 | 18.0000 ± 0.0000 | 5.0000 ± 0.0000 |
| STREAMKMeans [standard] | 2    | 0.1799 ± 0.0008 | 33347.3150 ± 137.7650 | 0.0360 ± 0.0000 | -                | -                | 5.0000 ± 0.0000 |
| ODAC [standard]         | 1    | 0.1614 ± 0.0000 | 37172.0000 ± 0.0000   | 0.0130 ± 0.0000 | -                | -                | -               |

> `Model MB` is the model (+preprocessing) footprint via River's `_raw_memory_usage`, the comparable figure across algorithms. `RSS MB` is whole-process memory, for context. Peak memory is a max over checkpoints, so it is a lower bound for models that prune.

## Structure — algorithms that do not partition samples

| Setting         | Variable clusters | Active | Tree height | Throughput (sps) |
|-----------------|-------------------|--------|-------------|------------------|
| ODAC [standard] | 7                 | 4      | 2           | 37172.0000       |

> ODAC clusters **variables**, not samples; its `predict_one` raises `NotImplementedError` by design. External metrics are undefined for it, so it is absent from the quality tables below — that is correct, not a failure.

## Quality — whole stream

| Setting                 | Runs | ARI             | NMI             | AMI             | PURITY          |
|-------------------------|------|-----------------|-----------------|-----------------|-----------------|
| DBSTREAM [standard]     | 1    | 0.9587 ± 0.0000 | 0.9429 ± 0.0000 | 0.9429 ± 0.0000 | 0.9843 ± 0.0000 |
| STREAMKMeans [standard] | 2    | 0.7068 ± 0.1077 | 0.7696 ± 0.0675 | 0.7694 ± 0.0676 | 0.8253 ± 0.0874 |

## Window diagnostics — read BEFORE the time-aware tables

| Windows | Single-class windows | Share | Median majority-class share |
|---------|----------------------|-------|-----------------------------|
| 6       | 0                    | 0%    | 0.2160                      |

## Quality over time — ARI per window

| Setting                 | Runs | Mean ARI        | Median ARI      | Final ARI       |
|-------------------------|------|-----------------|-----------------|-----------------|
| DBSTREAM [standard]     | 1    | 0.9613 ± 0.0000 | 0.9987 ± 0.0000 | 0.9987 ± 0.0000 |
| STREAMKMeans [standard] | 2    | 0.8149 ± 0.1076 | 0.8756 ± 0.1244 | 0.8772 ± 0.1228 |

## Quality over time — NMI per window

| Setting                 | Runs | Mean NMI        | Median NMI      | Final NMI       |
|-------------------------|------|-----------------|-----------------|-----------------|
| DBSTREAM [standard]     | 1    | 0.9623 ± 0.0000 | 0.9980 ± 0.0000 | 0.9980 ± 0.0000 |
| STREAMKMeans [standard] | 2    | 0.8814 ± 0.0596 | 0.9253 ± 0.0747 | 0.9237 ± 0.0763 |

## Quality over time — AMI per window

| Setting                 | Runs | Mean AMI        | Median AMI      | Final AMI       |
|-------------------------|------|-----------------|-----------------|-----------------|
| DBSTREAM [standard]     | 1    | 0.9620 ± 0.0000 | 0.9980 ± 0.0000 | 0.9980 ± 0.0000 |
| STREAMKMeans [standard] | 2    | 0.8809 ± 0.0598 | 0.9250 ± 0.0750 | 0.9234 ± 0.0766 |

## Quality over time — PURITY per window

| Setting                 | Runs | Mean PURITY     | Median PURITY   | Final PURITY    |
|-------------------------|------|-----------------|-----------------|-----------------|
| DBSTREAM [standard]     | 1    | 0.9847 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| STREAMKMeans [standard] | 2    | 0.8603 ± 0.0907 | 0.8975 ± 0.1025 | 0.9075 ± 0.0925 |

> Real run counts — DBSTREAM [standard]: 1, STREAMKMeans [standard]: 2, ODAC [standard]: 1. Deterministic algorithms run once, so their standard deviation is zero by definition, not by stability.
