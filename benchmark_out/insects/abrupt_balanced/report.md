# Streaming clustering benchmark — insects_abrupt_balanced

- n_samples: 20,000
- n_features: 33
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

- **CluStream [standard]** — params: `{'n_macro_clusters': 5, 'max_micro_clusters': 100, 'time_gap': 100, 'seed': 1}` | scaler: `standard` params: `{}`
- **DenStream [standard]** — params: `{'decaying_factor': 0.25, 'beta': 0.5, 'mu': 2.5, 'epsilon': 0.5, 'n_samples_init': 1000, 'stream_speed': 100}` | scaler: `standard` params: `{}`
- **DBSTREAM [standard]** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.01, 'cleanup_interval': 2, 'intersection_factor': 0.3, 'minimum_weight': 1.0}` | scaler: `standard` params: `{}`
- **STREAMKMeans [standard]** — params: `{'chunk_size': 500, 'n_clusters': 5, 'seed': 1}` | scaler: `standard` params: `{}`

## Hiệu năng

| Setting                 | Samples | Runtime (s) | Throughput (sps) | ms/sample | Model MB | RSS MB   | Mean MC  | Max MC | Macro K | Cold-start |
|-------------------------|---------|-------------|------------------|-----------|----------|----------|----------|--------|---------|------------|
| CluStream [standard]    | 20000   | 78.3106     | 255.3900         | 3.9155    | 0.8960   | 196.8000 | 100.0000 | 100    | 5       | 0          |
| DenStream [standard]    | 20000   | 3.6787      | 5436.7300        | 0.1839    | 0.0580   | 196.9000 | 7.5500   | 8      | 8       | 0          |
| DBSTREAM [standard]     | 20000   | 0.7380      | 27098.9000       | 0.0369    | 0.0140   | 196.9000 | 2.0500   | 3      | 2       | 0          |
| STREAMKMeans [standard] | 20000   | 1.3460      | 14858.5300       | 0.0673    | 0.8220   | 197.5000 | -        | -      | 5       | 1          |

> `Model MB` là footprint của model (+scaler) theo `_raw_memory_usage` của river — đây là con số so sánh được giữa các thuật toán. `RSS MB` là RAM của cả tiến trình, chỉ để tham chiếu. `Cold-start` là số lần `predict_one` phải trả nhãn 0 vì model chưa khởi tạo xong.

## Chất lượng — toàn stream (luỹ kế đến mẫu cuối)

| Setting                 | ARI    | NMI    | AMI    | PURITY |
|-------------------------|--------|--------|--------|--------|
| CluStream [standard]    | 0.0051 | 0.0068 | 0.0065 | 0.2021 |
| DenStream [standard]    | 0.1192 | 0.1865 | 0.1860 | 0.3832 |
| DBSTREAM [standard]     | 0.0002 | 0.0025 | 0.0021 | 0.1782 |
| STREAMKMeans [standard] | 0.0359 | 0.0658 | 0.0655 | 0.2866 |

## Chẩn đoán window — đọc TRƯỚC bảng time-aware

| Windows | Single-class windows | Share | Median majority-class share |
|---------|----------------------|-------|-----------------------------|
| 20      | 1                    | 5%    | 0.2160                      |

## Chất lượng theo thời gian — ARI trên từng window

| Setting                 | Mean ARI (win.) | Median ARI (win.) | Final ARI (win.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.0134          | 0.0104            | 0.0780           |
| DenStream [standard]    | 0.1808          | 0.1935            | 0.1509           |
| DBSTREAM [standard]     | -0.0008         | -0.0004           | -0.0046          |
| STREAMKMeans [standard] | 0.0364          | 0.0251            | 0.0447           |

## Chất lượng theo thời gian — NMI trên từng window

| Setting                 | Mean NMI (win.) | Median NMI (win.) | Final NMI (win.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.0296          | 0.0254            | 0.1015           |
| DenStream [standard]    | 0.2792          | 0.3258            | 0.2211           |
| DBSTREAM [standard]     | 0.0053          | 0.0051            | 0.0108           |
| STREAMKMeans [standard] | 0.1280          | 0.1182            | 0.1341           |

## Chất lượng theo thời gian — AMI trên từng window

| Setting                 | Mean AMI (win.) | Median AMI (win.) | Final AMI (win.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.0237          | 0.0191            | 0.0949           |
| DenStream [standard]    | 0.2730          | 0.3188            | 0.2121           |
| DBSTREAM [standard]     | 0.0009          | -0.0001           | 0.0046           |
| STREAMKMeans [standard] | 0.1215          | 0.1102            | 0.1259           |

> Đây là chỉ số time-aware: mỗi window được chấm **chỉ trên các mẫu trong window đó**, metric reset ở mỗi biên. Bảng dưới đây là đường cong luỹ kế từ đầu stream, hội tụ dần về giá trị toàn stream.

## Chất lượng theo thời gian — ARI luỹ kế (đối chiếu)

| Setting                 | Mean ARI (cum.) | Median ARI (cum.) | Final ARI (cum.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.0035          | 0.0031            | 0.0051           |
| DenStream [standard]    | 0.1342          | 0.1448            | 0.1192           |
| DBSTREAM [standard]     | 0.0002          | 0.0004            | 0.0002           |
| STREAMKMeans [standard] | 0.0273          | 0.0249            | 0.0359           |
