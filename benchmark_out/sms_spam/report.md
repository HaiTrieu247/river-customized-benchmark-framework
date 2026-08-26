# river.cluster benchmark — sms_spam

- n_samples: 400
- n_features: 1
- window_size: 200
- checkpoints: 2
- seeds: [1]
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
| python_hash_seed      | random (unpinned)                                     |
| river_version         | 0.26.1                                                |
| river_path            | D:\WORK\CTY-ANH KHÔI\OnlineML\river\river\__init__.py |
| lib:river             | 0.26.0                                                |
| lib:numpy             | 2.4.3                                                 |
| lib:scipy             | 1.17.1                                                |
| lib:pandas            | 2.3.3                                                 |
| lib:scikit-learn      | 1.8.0                                                 |
| lib:matplotlib        | 3.10.9                                                |

> ⚠ `PYTHONHASHSEED` is unpinned. River's KMeans gives different results per process even with a fixed `seed` (RIVER_BUGS.md BUG-2), so these numbers are not reproducible as they stand. Re-run with `PYTHONHASHSEED=0` for figures you intend to publish.

## Configuration

- **TextClust [bag_of_words/on=body]** — params: `{'radius': 0.3, 'fading_factor': 0.001, 'tgap': 100, 'real_time_fading': False}` | preprocessing: `bag_of_words` params: `{'on': 'body'}`

> Parameters not named on the command line are River's own defaults. The framework does not tune them.

## Efficiency

| Setting                          | Samples | Runtime (s) | Throughput (sps) | ms/sample | Model MB | RSS MB   | Mean MC  | Max MC | Macro K | Cold-start |
|----------------------------------|---------|-------------|------------------|-----------|----------|----------|----------|--------|---------|------------|
| TextClust [bag_of_words/on=body] | 400     | 0.8182      | 488.8700         | 2.0455    | 0.3730   | 198.8000 | 100.5000 | 103    | -       | 0          |

> `Model MB` is the model (+preprocessing) footprint via River's `_raw_memory_usage`, the comparable figure across algorithms. `RSS MB` is whole-process memory, for context. Peak memory is a max over checkpoints, so it is a lower bound for models that prune.

## Quality — whole stream

| Setting                          | ARI    | NMI    | AMI    | PURITY |
|----------------------------------|--------|--------|--------|--------|
| TextClust [bag_of_words/on=body] | 0.0011 | 0.1190 | 0.0334 | 0.9624 |

## Window diagnostics — read BEFORE the time-aware tables

| Windows | Single-class windows | Share | Median majority-class share |
|---------|----------------------|-------|-----------------------------|
| 2       | 0                    | 0%    | 0.8750                      |

## Quality over time — ARI per window

| Setting                          | Mean ARI (win.) | Median ARI (win.) | Final ARI (win.) |
|----------------------------------|-----------------|-------------------|------------------|
| TextClust [bag_of_words/on=body] | 0.0025          | 0.0025            | 0.0018           |

## Quality over time — NMI per window

| Setting                          | Mean NMI (win.) | Median NMI (win.) | Final NMI (win.) |
|----------------------------------|-----------------|-------------------|------------------|
| TextClust [bag_of_words/on=body] | 0.1397          | 0.1397            | 0.1324           |

## Quality over time — AMI per window

| Setting                          | Mean AMI (win.) | Median AMI (win.) | Final AMI (win.) |
|----------------------------------|-----------------|-------------------|------------------|
| TextClust [bag_of_words/on=body] | 0.0410          | 0.0410            | 0.0317           |

## Quality over time — PURITY per window

| Setting                          | Mean PURITY (win.) | Median PURITY (win.) | Final PURITY (win.) |
|----------------------------------|--------------------|----------------------|---------------------|
| TextClust [bag_of_words/on=body] | 0.9674             | 0.9674               | 0.9800              |
