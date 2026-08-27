# river.cluster benchmark — insects_abrupt_balanced

- n_samples: 52,848
- n_features: 33
- window_size: 1000
- checkpoints: 20
- seeds: [1, 2, 3]
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

- **CluStream** — params: `{'n_macro_clusters': 6, 'max_micro_clusters': 100, 'time_gap': 100, 'seed': 1}` | preprocessing: `none` params: `{}`
- **DBSTREAM** — params: `{'clustering_threshold': 1200, 'fading_factor': 0.05, 'cleanup_interval': 4, 'intersection_factor': 0.3, 'minimum_weight': 1.0}` | preprocessing: `none` params: `{}`
- **STREAMKMeans** — params: `{'chunk_size': 100, 'n_clusters': 6, 'seed': 1}` | preprocessing: `none` params: `{}`

> Parameters not named on the command line are River's own defaults. The framework does not tune them.

## Efficiency

| Setting      | Runs | Runtime (s)       | Throughput (sps)      | Model MB        | Mean MC           | Max MC            | Macro K         |
|--------------|------|-------------------|-----------------------|-----------------|-------------------|-------------------|-----------------|
| CluStream    | 3    | 231.0713 ± 4.7247 | 228.8033 ± 4.6718     | 0.9690 ± 0.0000 | 100.0000 ± 0.0000 | 100.0000 ± 0.0000 | 6.0000 ± 0.0000 |
| DBSTREAM     | 1    | 3.2908 ± 0.0000   | 16059.5400 ± 0.0000   | 0.0400 ± 0.0000 | 9.2400 ± 0.0000   | 14.0000 ± 0.0000  | 8.0000 ± 0.0000 |
| STREAMKMeans | 3    | 3.8896 ± 0.0860   | 13593.6800 ± 296.4213 | 0.1720 ± 0.0000 | -                 | -                 | 6.0000 ± 0.0000 |

> `Model MB` is the model (+preprocessing) footprint via River's `_raw_memory_usage`, the comparable figure across algorithms. `RSS MB` is whole-process memory, for context. Peak memory is a max over checkpoints, so it is a lower bound for models that prune.

## Quality — whole stream

| Setting      | Runs | ARI             | NMI             | AMI             | PURITY          |
|--------------|------|-----------------|-----------------|-----------------|-----------------|
| CluStream    | 3    | 0.0028 ± 0.0010 | 0.0058 ± 0.0013 | 0.0056 ± 0.0013 | 0.1900 ± 0.0046 |
| DBSTREAM     | 1    | 0.0125 ± 0.0000 | 0.0159 ± 0.0000 | 0.0156 ± 0.0000 | 0.2128 ± 0.0000 |
| STREAMKMeans | 3    | 0.0070 ± 0.0005 | 0.0102 ± 0.0010 | 0.0100 ± 0.0010 | 0.2082 ± 0.0039 |

## Window diagnostics — read BEFORE the time-aware tables

| Windows | Single-class windows | Share | Median majority-class share |
|---------|----------------------|-------|-----------------------------|
| 53      | 1                    | 2%    | 0.2800                      |

## Quality over time — ARI per window

| Setting      | Runs | Mean ARI        | Median ARI      | Final ARI        |
|--------------|------|-----------------|-----------------|------------------|
| CluStream    | 3    | 0.0089 ± 0.0011 | 0.0054 ± 0.0006 | -0.0012 ± 0.0013 |
| DBSTREAM     | 1    | 0.0271 ± 0.0000 | 0.0234 ± 0.0000 | -0.0025 ± 0.0000 |
| STREAMKMeans | 3    | 0.0352 ± 0.0016 | 0.0288 ± 0.0035 | -0.0035 ± 0.0019 |

## Quality over time — NMI per window

| Setting      | Runs | Mean NMI        | Median NMI      | Final NMI       |
|--------------|------|-----------------|-----------------|-----------------|
| CluStream    | 3    | 0.0211 ± 0.0015 | 0.0169 ± 0.0009 | 0.0131 ± 0.0036 |
| DBSTREAM     | 1    | 0.0360 ± 0.0000 | 0.0318 ± 0.0000 | 0.0130 ± 0.0000 |
| STREAMKMeans | 3    | 0.0509 ± 0.0017 | 0.0471 ± 0.0021 | 0.0088 ± 0.0004 |

## Quality over time — AMI per window

| Setting      | Runs | Mean AMI        | Median AMI      | Final AMI        |
|--------------|------|-----------------|-----------------|------------------|
| CluStream    | 3    | 0.0140 ± 0.0016 | 0.0094 ± 0.0010 | 0.0024 ± 0.0036  |
| DBSTREAM     | 1    | 0.0234 ± 0.0000 | 0.0196 ± 0.0000 | -0.0015 ± 0.0000 |
| STREAMKMeans | 3    | 0.0442 ± 0.0017 | 0.0397 ± 0.0020 | -0.0017 ± 0.0003 |

## Quality over time — PURITY per window

| Setting      | Runs | Mean PURITY     | Median PURITY   | Final PURITY    |
|--------------|------|-----------------|-----------------|-----------------|
| CluStream    | 3    | 0.3566 ± 0.0008 | 0.3107 ± 0.0033 | 0.9693 ± 0.0000 |
| DBSTREAM     | 1    | 0.3870 ± 0.0000 | 0.3480 ± 0.0000 | 0.9693 ± 0.0000 |
| STREAMKMeans | 3    | 0.3912 ± 0.0008 | 0.3613 ± 0.0026 | 0.9693 ± 0.0000 |

> Real run counts — CluStream: 3, DBSTREAM: 1, STREAMKMeans: 3. Deterministic algorithms run once, so their standard deviation is zero by definition, not by stability.
