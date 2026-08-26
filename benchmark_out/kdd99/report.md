# Streaming clustering benchmark — kddcup99

- n_samples: 50,000
- n_features: 34
- window_size: 1000
- checkpoints: 25
- seeds: [1, 2, 3]
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
- **DenStream [standard]** — params: `{'decaying_factor': 0.25, 'beta': 0.2, 'mu': 10.0, 'epsilon': 16.0, 'n_samples_init': 1000, 'stream_speed': 1000}` | scaler: `standard` params: `{}`
- **DBSTREAM [standard]** — params: `{'clustering_threshold': 1.0, 'fading_factor': 0.001, 'cleanup_interval': 2, 'intersection_factor': 0.0, 'minimum_weight': 0.0}` | scaler: `standard` params: `{}`
- **STREAMKMeans [standard]** — params: `{'chunk_size': 2000, 'n_clusters': 10, 'seed': 1}` | scaler: `standard` params: `{}`

## Hiệu năng

| Setting                 | Runs | Runtime (s)       | Throughput (sps)    | Model MB        | Mean MC           | Max MC            | Macro K          |
|-------------------------|------|-------------------|---------------------|-----------------|-------------------|-------------------|------------------|
| CluStream [standard]    | 3    | 162.7102 ± 5.0622 | 307.5933 ± 9.6604   | 0.9130 ± 0.0000 | 100.0000 ± 0.0000 | 100.0000 ± 0.0000 | 5.0000 ± 0.0000  |
| DenStream [standard]    | 1    | 2.5099 ± 0.0000   | 19921.0700 ± 0.0000 | 0.0120 ± 0.0000 | 1.0000 ± 0.0000   | 1.0000 ± 0.0000   | 1.0000 ± 0.0000  |
| DBSTREAM [standard]     | 1    | 53.1502 ± 0.0000  | 940.7300 ± 0.0000   | 0.6570 ± 0.0000 | 109.5200 ± 0.0000 | 222.0000 ± 0.0000 | 1.0000 ± 0.0000  |
| STREAMKMeans [standard] | 3    | 5.6012 ± 0.0550   | 8927.5767 ± 88.1908 | 3.2460 ± 0.0000 | -                 | -                 | 10.0000 ± 0.0000 |

> `Model MB` là footprint của model (+scaler) theo `_raw_memory_usage` của river — đây là con số so sánh được giữa các thuật toán. `RSS MB` là RAM của cả tiến trình, chỉ để tham chiếu. `Cold-start` là số lần `predict_one` phải trả nhãn 0 vì model chưa khởi tạo xong.

## Chất lượng — toàn stream (luỹ kế đến mẫu cuối)

| Setting                 | Runs | ARI             | NMI             | AMI             | PURITY          |
|-------------------------|------|-----------------|-----------------|-----------------|-----------------|
| CluStream [standard]    | 3    | 0.0282 ± 0.0092 | 0.1303 ± 0.0252 | 0.1298 ± 0.0252 | 0.7703 ± 0.0152 |
| DenStream [standard]    | 1    | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7593 ± 0.0000 |
| DBSTREAM [standard]     | 1    | 0.1458 ± 0.0000 | 0.1564 ± 0.0000 | 0.1530 ± 0.0000 | 0.8201 ± 0.0000 |
| STREAMKMeans [standard] | 3    | 0.0734 ± 0.0719 | 0.2252 ± 0.0543 | 0.2245 ± 0.0543 | 0.8114 ± 0.0520 |

## Chẩn đoán window — đọc TRƯỚC bảng time-aware

| Windows | Single-class windows | Share | Median majority-class share |
|---------|----------------------|-------|-----------------------------|
| 50      | 36                   | 72%   | 1.0000                      |

> ⚠ **72% số window chỉ chứa đúng một lớp.** Khi window chỉ có 1 lớp, ARI thoái hoá thành phép thử 'model có gộp nguyên window thành 1 cluster không': gộp thì được 1.0, xé ra dù chỉ thành 2 cluster thì được 0.0 — bất kể clustering thực sự tốt hay xấu. Nghĩa là model **under-segment sẽ ăn điểm cao vì lý do sai**. Trên stream kiểu này (KDDCup99 đến theo từng khối cùng lớp), hãy đọc bảng **toàn stream** làm chỉ số chính, và chỉ dùng bảng time-aware để xem xu hướng. Muốn chỉ số time-aware có nghĩa thì cần window lớn hơn hoặc stream trộn lớp.

## Chất lượng theo thời gian — ARI trên từng window

| Setting                 | Runs | Mean ARI        | Median ARI      | Final ARI       |
|-------------------------|------|-----------------|-----------------|-----------------|
| CluStream [standard]    | 3    | 0.1533 ± 0.0466 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| DenStream [standard]    | 1    | 0.7200 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| DBSTREAM [standard]     | 1    | 0.0228 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| STREAMKMeans [standard] | 3    | 0.3488 ± 0.0076 | 0.0001 ± 0.0001 | 1.0000 ± 0.0000 |

## Chất lượng theo thời gian — NMI trên từng window

| Setting                 | Runs | Mean NMI        | Median NMI      | Final NMI       |
|-------------------------|------|-----------------|-----------------|-----------------|
| CluStream [standard]    | 3    | 0.1699 ± 0.0428 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| DenStream [standard]    | 1    | 0.7200 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| DBSTREAM [standard]     | 1    | 0.0433 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| STREAMKMeans [standard] | 3    | 0.3503 ± 0.0057 | 0.0177 ± 0.0007 | 1.0000 ± 0.0000 |

## Chất lượng theo thời gian — AMI trên từng window

| Setting                 | Runs | Mean AMI        | Median AMI      | Final AMI       |
|-------------------------|------|-----------------|-----------------|-----------------|
| CluStream [standard]    | 3    | 0.1690 ± 0.0428 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| DenStream [standard]    | 1    | 0.7200 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| DBSTREAM [standard]     | 1    | 0.0397 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| STREAMKMeans [standard] | 3    | 0.3491 ± 0.0058 | 0.0136 ± 0.0009 | 1.0000 ± 0.0000 |

> Đây là chỉ số time-aware: mỗi window được chấm **chỉ trên các mẫu trong window đó**, metric reset ở mỗi biên. Bảng dưới đây là đường cong luỹ kế từ đầu stream, hội tụ dần về giá trị toàn stream.

## Chất lượng theo thời gian — ARI luỹ kế (đối chiếu)

| Setting                 | Mean ARI (cum.) | Median ARI (cum.) | Final ARI (cum.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.0172          | 0.0123            | 0.0283           |
| DenStream [standard]    | 0.0000          | 0.0000            | 0.0000           |
| DBSTREAM [standard]     | 0.0848          | 0.0998            | 0.1458           |
| STREAMKMeans [standard] | 0.0086          | 0.0053            | 0.0078           |

> Số lần chạy thật của từng setting — CluStream [standard]: 3, DenStream [standard]: 1, DBSTREAM [standard]: 1, STREAMKMeans [standard]: 3. Thuật toán tất định (DenStream, DBSTREAM) chỉ chạy 1 seed vì lặp lại trên cùng stream sẽ ra đúng cùng kết quả, nên std của chúng bằng 0 theo định nghĩa.
