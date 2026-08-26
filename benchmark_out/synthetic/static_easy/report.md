# Streaming clustering benchmark — static_easy_d10_n8000

- n_samples: 8,000
- n_features: 10
- window_size: 1000
- checkpoints: 10
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
- **KMeans [standard]** — params: `{'n_clusters': 5, 'halflife': 0.5, 'seed': 1}` | scaler: `standard` params: `{}`
- **ODAC** — params: `{'confidence_level': 0.9, 'n_min': 100, 'tau': 0.1}` | scaler: `none` params: `{}`

## Hiệu năng

| Setting                 | Samples | Runtime (s) | Throughput (sps) | ms/sample | Model MB | RSS MB   | Mean MC  | Max MC | Macro K | Cold-start |
|-------------------------|---------|-------------|------------------|-----------|----------|----------|----------|--------|---------|------------|
| CluStream [standard]    | 8000    | 7.7584      | 1031.1400        | 0.9698    | 0.3060   | 196.0000 | 100.0000 | 100    | 5       | 0          |
| DenStream [standard]    | 8000    | 17.4262     | 459.0800         | 2.1783    | 0.4790   | 196.3000 | 37.8000  | 42     | 42      | 0          |
| DBSTREAM [standard]     | 8000    | 0.7589      | 10541.2600       | 0.0949    | 0.0330   | 196.3000 | 15.3000  | 24     | 6       | 0          |
| STREAMKMeans [standard] | 8000    | 0.2560      | 31253.3000       | 0.0320    | 0.2830   | 196.5000 | -        | -      | 5       | 1          |
| KMeans [standard]       | 8000    | 0.3626      | 22063.5100       | 0.0453    | 0.0080   | 196.5000 | -        | -      | 5       | 0          |
| ODAC                    | 8000    | 0.1050      | 76164.7300       | 0.0131    | 0.0080   | 196.5000 | -        | -      | 13      | 0          |

> `Model MB` là footprint của model (+scaler) theo `_raw_memory_usage` của river — đây là con số so sánh được giữa các thuật toán. `RSS MB` là RAM của cả tiến trình, chỉ để tham chiếu. `Cold-start` là số lần `predict_one` phải trả nhãn 0 vì model chưa khởi tạo xong.

## Cấu trúc — thuật toán không phân hoạch mẫu

| Setting | Variable clusters | Active | Tree height | Throughput (sps) |
|---------|-------------------|--------|-------------|------------------|
| ODAC    | 13                | 7      | 4           | 76164.7000       |

> ODAC gom cụm **các biến**, không phải các mẫu: `predict_one` của nó raise `NotImplementedError` theo thiết kế. Vì vậy ARI/NMI/AMI/purity không định nghĩa được cho nó và nó vắng mặt ở các bảng chất lượng bên dưới — đó là *đúng*, không phải lỗi. Nó vẫn được đo về tài nguyên và cấu trúc.

## Chất lượng — toàn stream (luỹ kế đến mẫu cuối)

| Setting                 | ARI    | NMI    | AMI    | PURITY |
|-------------------------|--------|--------|--------|--------|
| CluStream [standard]    | 0.4432 | 0.5307 | 0.5304 | 0.6531 |
| DenStream [standard]    | 0.3022 | 0.5807 | 0.5788 | 0.8979 |
| DBSTREAM [standard]     | 0.9686 | 0.9540 | 0.9539 | 0.9880 |
| STREAMKMeans [standard] | 0.8505 | 0.8550 | 0.8549 | 0.9384 |
| KMeans [standard]       | 0.9981 | 0.9963 | 0.9963 | 0.9992 |

## Chẩn đoán window — đọc TRƯỚC bảng time-aware

| Windows | Single-class windows | Share | Median majority-class share |
|---------|----------------------|-------|-----------------------------|
| 8       | 0                    | 0%    | 0.2150                      |

## Chất lượng theo thời gian — ARI trên từng window

| Setting                 | Mean ARI (win.) | Median ARI (win.) | Final ARI (win.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.6159          | 0.6452            | 0.8636           |
| DenStream [standard]    | 0.3542          | 0.4028            | 0.3959           |
| DBSTREAM [standard]     | 0.9707          | 0.9987            | 0.9974           |
| STREAMKMeans [standard] | 0.8979          | 1.0000            | 1.0000           |
| KMeans [standard]       | 0.9982          | 1.0000            | 1.0000           |

## Chất lượng theo thời gian — NMI trên từng window

| Setting                 | Mean NMI (win.) | Median NMI (win.) | Final NMI (win.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.7025          | 0.7243            | 0.8804           |
| DenStream [standard]    | 0.5854          | 0.6694            | 0.6653           |
| DBSTREAM [standard]     | 0.9712          | 0.9980            | 0.9957           |
| STREAMKMeans [standard] | 0.9311          | 1.0000            | 1.0000           |
| KMeans [standard]       | 0.9973          | 1.0000            | 1.0000           |

## Chất lượng theo thời gian — AMI trên từng window

| Setting                 | Mean AMI (win.) | Median AMI (win.) | Final AMI (win.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.7010          | 0.7229            | 0.8798           |
| DenStream [standard]    | 0.5738          | 0.6561            | 0.6520           |
| DBSTREAM [standard]     | 0.9710          | 0.9980            | 0.9956           |
| STREAMKMeans [standard] | 0.9308          | 1.0000            | 1.0000           |
| KMeans [standard]       | 0.9973          | 1.0000            | 1.0000           |

> Đây là chỉ số time-aware: mỗi window được chấm **chỉ trên các mẫu trong window đó**, metric reset ở mỗi biên. Bảng dưới đây là đường cong luỹ kế từ đầu stream, hội tụ dần về giá trị toàn stream.

## Chất lượng theo thời gian — ARI luỹ kế (đối chiếu)

| Setting                 | Mean ARI (cum.) | Median ARI (cum.) | Final ARI (cum.) |
|-------------------------|-----------------|-------------------|------------------|
| CluStream [standard]    | 0.4604          | 0.4251            | 0.4432           |
| DenStream [standard]    | 0.1884          | 0.2211            | 0.3022           |
| DBSTREAM [standard]     | 0.9190          | 0.9448            | 0.9686           |
| STREAMKMeans [standard] | 0.6533          | 0.7363            | 0.8505           |
| KMeans [standard]       | 0.9950          | 0.9967            | 0.9981           |
