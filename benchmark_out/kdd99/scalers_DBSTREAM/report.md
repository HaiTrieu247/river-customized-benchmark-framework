# Streaming clustering benchmark — kddcup99

- n_samples: 20,000
- n_features: 34
- window_size: 1000
- checkpoints: 20
- seeds: [1]
- có ground truth: True

## Môi trường chạy

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
| river_version         | 0.26.1                                                |
| river_path            | D:\WORK\CTY-ANH KHÔI\OnlineML\river\river\__init__.py |
| lib:river             | 0.26.0                                                |
| lib:numpy             | 2.4.3                                                 |
| lib:scipy             | 1.17.1                                                |
| lib:pandas            | 2.3.3                                                 |
| lib:scikit-learn      | 1.8.0                                                 |
| lib:matplotlib        | 3.10.9                                                |

## Cấu hình (đóng băng trước khi chạy)

- **DBSTREAM** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.001, 'cleanup_interval': 2, 'intersection_factor': 0.0, 'minimum_weight': 0.0}` | scaler: `none` params: `{}`
- **DBSTREAM [standard]** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.001, 'cleanup_interval': 2, 'intersection_factor': 0.0, 'minimum_weight': 0.0}` | scaler: `standard` params: `{}`
- **DBSTREAM [standard/window_size=10000]** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.001, 'cleanup_interval': 2, 'intersection_factor': 0.0, 'minimum_weight': 0.0}` | scaler: `standard` params: `{'window_size': 10000}`
- **DBSTREAM [adaptive_standard/fading_factor=0.01]** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.001, 'cleanup_interval': 2, 'intersection_factor': 0.0, 'minimum_weight': 0.0}` | scaler: `adaptive_standard` params: `{'fading_factor': 0.01}`
- **DBSTREAM [minmax]** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.001, 'cleanup_interval': 2, 'intersection_factor': 0.0, 'minimum_weight': 0.0}` | scaler: `minmax` params: `{}`
- **DBSTREAM [robust]** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.001, 'cleanup_interval': 2, 'intersection_factor': 0.0, 'minimum_weight': 0.0}` | scaler: `robust` params: `{}`

## Hiệu năng

| Setting                                         | Samples | Runtime (s) | Throughput (sps) | ms/sample | Model MB | RSS MB   | Mean MC  | Max MC | Macro K | Cold-start |
|-------------------------------------------------|---------|-------------|------------------|-----------|----------|----------|----------|--------|---------|------------|
| DBSTREAM                                        | 20000   | 0.3330      | 60067.2400       | 0.0167    | 0.0170   | 332.3000 | 3.1000   | 6      | 3       | 0          |
| DBSTREAM [standard]                             | 20000   | 16.3885     | 1220.3700        | 0.8194    | 0.3800   | 332.3000 | 99.8500  | 132    | 40      | 0          |
| DBSTREAM [standard/window_size=10000]           | 20000   | 18.8378     | 1061.7000        | 0.9419    | 122.5070 | 464.5000 | 100.4000 | 132    | 44      | 0          |
| DBSTREAM [adaptive_standard/fading_factor=0.01] | 20000   | 26.7605     | 747.3700         | 1.3380    | 0.6600   | 440.6000 | 161.0000 | 230    | 42      | 0          |
| DBSTREAM [minmax]                               | 20000   | 4.1716      | 4794.3800        | 0.2086    | 0.0850   | 439.6000 | 18.9500  | 31     | 3       | 0          |
| DBSTREAM [robust]                               | 20000   | 8.6803      | 2304.0600        | 0.4340    | 0.8130   | 447.4000 | 105.1500 | 287    | 94      | 0          |

> `Model MB` là footprint của model (+scaler) theo `_raw_memory_usage` của river — đây là con số so sánh được giữa các thuật toán. `RSS MB` là RAM của cả tiến trình, chỉ để tham chiếu. `Cold-start` là số lần `predict_one` phải trả nhãn 0 vì model chưa khởi tạo xong.

## Chất lượng — toàn stream (luỹ kế đến mẫu cuối)

| Setting                                         | ARI     | NMI    | AMI    | PURITY |
|-------------------------------------------------|---------|--------|--------|--------|
| DBSTREAM                                        | 0.0253  | 0.1840 | 0.1833 | 0.8194 |
| DBSTREAM [standard]                             | 0.0928  | 0.1305 | 0.1282 | 0.8508 |
| DBSTREAM [standard/window_size=10000]           | 0.0856  | 0.1229 | 0.1205 | 0.8478 |
| DBSTREAM [adaptive_standard/fading_factor=0.01] | -0.0376 | 0.1338 | 0.1312 | 0.8582 |
| DBSTREAM [minmax]                               | -0.0702 | 0.1574 | 0.1564 | 0.8090 |
| DBSTREAM [robust]                               | 0.0011  | 0.0731 | 0.0676 | 0.8580 |

## Chẩn đoán window — đọc TRƯỚC bảng time-aware

| Windows | Single-class windows | Share | Median majority-class share |
|---------|----------------------|-------|-----------------------------|
| 20      | 14                   | 70%   | 1.0000                      |

> ⚠ **70% số window chỉ chứa đúng một lớp.** Với những window đó, bất kỳ model nào không xé nhỏ window đều đạt ARI = 1.0, nên điểm per-window đo *thứ tự đến của dữ liệu* chứ không đo chất lượng cluster. Trên stream kiểu này (KDDCup99 đến theo từng khối cùng lớp), hãy đọc bảng **toàn stream** làm chỉ số chính, và chỉ dùng bảng time-aware để xem xu hướng. Muốn chỉ số time-aware có nghĩa thì cần window lớn hơn hoặc stream trộn lớp.

## Chất lượng theo thời gian — ARI trên từng window

| Setting                                         | Mean ARI (win.) | Median ARI (win.) | Final ARI (win.) |
|-------------------------------------------------|-----------------|-------------------|------------------|
| DBSTREAM                                        | 0.0326          | 0.0000            | 0.0200           |
| DBSTREAM [standard]                             | 0.0293          | 0.0000            | 0.0537           |
| DBSTREAM [standard/window_size=10000]           | 0.0287          | 0.0000            | 0.0463           |
| DBSTREAM [adaptive_standard/fading_factor=0.01] | 0.0242          | 0.0000            | 0.0257           |
| DBSTREAM [minmax]                               | 0.4271          | 0.0883            | 0.8462           |
| DBSTREAM [robust]                               | 0.0116          | 0.0000            | 0.0073           |

## Chất lượng theo thời gian — NMI trên từng window

| Setting                                         | Mean NMI (win.) | Median NMI (win.) | Final NMI (win.) |
|-------------------------------------------------|-----------------|-------------------|------------------|
| DBSTREAM                                        | 0.0428          | 0.0000            | 0.0710           |
| DBSTREAM [standard]                             | 0.0451          | 0.0000            | 0.1615           |
| DBSTREAM [standard/window_size=10000]           | 0.0444          | 0.0000            | 0.1636           |
| DBSTREAM [adaptive_standard/fading_factor=0.01] | 0.0380          | 0.0000            | 0.1421           |
| DBSTREAM [minmax]                               | 0.4223          | 0.0884            | 0.7642           |
| DBSTREAM [robust]                               | 0.0316          | 0.0000            | 0.1249           |

## Chất lượng theo thời gian — AMI trên từng window

| Setting                                         | Mean AMI (win.) | Median AMI (win.) | Final AMI (win.) |
|-------------------------------------------------|-----------------|-------------------|------------------|
| DBSTREAM                                        | 0.0420          | 0.0000            | 0.0692           |
| DBSTREAM [standard]                             | 0.0424          | 0.0000            | 0.1485           |
| DBSTREAM [standard/window_size=10000]           | 0.0416          | 0.0000            | 0.1497           |
| DBSTREAM [adaptive_standard/fading_factor=0.01] | 0.0353          | 0.0000            | 0.1292           |
| DBSTREAM [minmax]                               | 0.4213          | 0.0839            | 0.7632           |
| DBSTREAM [robust]                               | 0.0280          | 0.0000            | 0.0982           |

> Đây là chỉ số time-aware: mỗi window được chấm **chỉ trên các mẫu trong window đó**, metric reset ở mỗi biên. Bảng dưới đây là đường cong luỹ kế từ đầu stream, hội tụ dần về giá trị toàn stream.

## Chất lượng theo thời gian — ARI luỹ kế (đối chiếu)

| Setting                                         | Mean ARI (cum.) | Median ARI (cum.) | Final ARI (cum.) |
|-------------------------------------------------|-----------------|-------------------|------------------|
| DBSTREAM                                        | 0.0128          | 0.0098            | 0.0253           |
| DBSTREAM [standard]                             | 0.0468          | 0.0667            | 0.0928           |
| DBSTREAM [standard/window_size=10000]           | 0.0437          | 0.0649            | 0.0856           |
| DBSTREAM [adaptive_standard/fading_factor=0.01] | -0.0024         | 0.0000            | -0.0376          |
| DBSTREAM [minmax]                               | 0.0706          | 0.0019            | -0.0702          |
| DBSTREAM [robust]                               | 0.0026          | 0.0013            | 0.0011           |
