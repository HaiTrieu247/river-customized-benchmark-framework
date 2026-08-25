# Streaming Clustering Benchmark — Evaluation Criteria

## Mục tiêu

Benchmark các **streaming clustering algorithms** trên data stream, không tập trung vào concept drift detection.

Câu hỏi chính:

> Thuật toán nào tạo clustering tốt, xử lý stream hiệu quả, sử dụng tài nguyên hợp lý và scale tốt?

## Bộ tiêu chí cốt lõi

### 1. Clustering Quality

Nếu có ground truth:

- **ARI (Adjusted Rand Index)** — metric chính.
- **NMI (Normalized Mutual Information)** — metric bổ sung.
- **AMI (Adjusted Mutual Information)** — metric bổ sung.

Nên đánh giá theo thời gian thay vì chỉ lấy kết quả cuối:

- Mean ARI/NMI/AMI over stream
- Final ARI/NMI/AMI
- Có thể thêm median.

> Mean quality over stream quan trọng hơn chỉ Final quality vì nó phản ánh toàn bộ quá trình streaming.

### 2. Computational Efficiency

- **Throughput:** samples/second.
- **Runtime:** tổng thời gian hoặc ms/sample.
- **Peak Memory:** RAM tối đa sử dụng.

Đây là nhóm quan trọng vì streaming clustering phải xử lý dữ liệu liên tục.

### 3. Model Complexity

Đặc biệt với CluStream, DenStream, DBSTREAM:

- **Number of micro-clusters**
- Có thể theo dõi số micro-cluster theo thời gian.

Metric này giúp giải thích memory, runtime và scalability.

### 4. Scalability

Kiểm tra khi stream tăng:

```text
10K → 100K → 1M → 10M samples
```

và khi số chiều tăng:

```text
2D → 10D → 50D → 100D
```

Theo dõi:

- Quality
- Runtime
- Throughput
- Memory

### 5. Stability / Reproducibility

Nếu algorithm có randomness, chạy nhiều seeds:

```text
seed = 1
seed = 2
seed = 3
seed = 4
seed = 5
```

Báo:

```text
Mean ± Standard Deviation
```

Ví dụ:

```text
DenStream: ARI = 0.89 ± 0.02
```

## Bộ metric đề xuất

| Category | Metric | Priority |
|---|---|---|
| Quality | ARI | ⭐⭐⭐⭐⭐ |
| Quality | NMI | ⭐⭐⭐⭐ |
| Quality | AMI | ⭐⭐⭐⭐ |
| Streaming Quality | Mean score over stream | ⭐⭐⭐⭐⭐ |
| Streaming Quality | Final score | ⭐⭐⭐ |
| Efficiency | Throughput | ⭐⭐⭐⭐⭐ |
| Efficiency | Runtime | ⭐⭐⭐⭐ |
| Efficiency | Peak Memory | ⭐⭐⭐⭐⭐ |
| Complexity | Number of micro-clusters | ⭐⭐⭐⭐ |
| Scalability | Runtime vs #samples | ⭐⭐⭐⭐⭐ |
| Scalability | Memory vs #samples | ⭐⭐⭐⭐⭐ |
| Scalability | Performance vs dimensionality | ⭐⭐⭐⭐ |
| Stability | Mean ± Std over seeds | ⭐⭐⭐⭐ |

## Core metrics cho phiên bản đầu tiên

Nên bắt đầu với:

1. ARI
2. NMI
3. AMI
4. Mean quality over stream
5. Final quality
6. Throughput
7. Runtime
8. Peak Memory
9. Number of micro-clusters
10. Mean ± Std over multiple seeds

Sau khi framework ổn định mới mở rộng các scalability experiments.

## Ground Truth

Ground truth vẫn cần, nhưng không phải ground-truth drift points.

Đối với clustering benchmark, ground truth chủ yếu là:

```text
Ground-truth cluster assignment
```

Ví dụ:

```text
Samples:       A B C D E F G H I
Ground truth:  1 1 1 2 2 2 3 3 3
```

Model có thể trả về:

```text
3 3 3 1 1 1 2 2 2
```

và vẫn được xem là clustering giống nhau vì cluster ID không có ý nghĩa.

Ground truth được dùng để tính ARI/NMI/AMI.

## Không có Ground Truth

Với real-world stream không có labels, có thể dùng internal metrics:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

Tuy nhiên cần cẩn thận vì internal metrics không phải lúc nào cũng phản ánh đúng chất lượng clustering thực tế.

## Experimental Protocol

Mọi algorithm phải chạy trong điều kiện comparable:

- Same dataset
- Same stream order
- Same number of samples
- Same evaluation windows
- Same hardware/environment
- Same random seeds

Ví dụ nếu `window_size = 1000`, mọi algorithm đều được đánh giá tại:

```text
0–999
1000–1999
2000–2999
...
```

Runtime và memory phải được đo trong cùng môi trường.

Nên ghi lại:

- CPU
- RAM
- OS
- Python version
- River version
- Dependency versions

## Hyperparameter Protocol

Không tune trực tiếp trên test dataset.

Không nên:

```text
Test dataset
    ↓
thử nhiều parameters
    ↓
chọn parameter có ARI cao nhất
    ↓
báo kết quả
```

Nên:

```text
Calibration streams
        ↓
Hyperparameter tuning
        ↓
Freeze parameters
        ↓
Test streams
        ↓
Final benchmark
```

## Repeated Runs

Mỗi configuration nên chạy nhiều seed nếu algorithm có randomness:

```text
Algorithm + Dataset + Parameters
        ↓
seed 1
seed 2
seed 3
seed 4
seed 5
        ↓
Mean ± Std
```

## Benchmark Scenarios

Có thể chia:

```text
Stationary Streams
        ↓
Baseline performance

Large Streams
        ↓
Scalability

High-dimensional Streams
        ↓
Dimensional scalability

Noisy Streams
        ↓
Robustness
```

Nếu sau này nghiên cứu khả năng adaptation trước changing streams, có thể thêm scenario tương ứng, nhưng **concept drift detection không phải core của benchmark này**.

## Những metric không cần đưa vào Core Benchmark

Nếu không nghiên cứu concept drift detection, không cần đưa vào core:

- Detection Delay
- False Positive Rate của drift detector
- False Negative Rate của drift detector
- Ground-truth drift points
- Drift detection Precision/Recall
- Drift detection F1

Đây là metric của bài toán **Concept Drift Detection**, không phải core của **Streaming Clustering Benchmark**.

## Cấu trúc tổng thể

```text
                    STREAMING CLUSTERING
                         BENCHMARK
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
   QUALITY                EFFICIENCY            SCALABILITY
       │                      │                      │
 ARI / NMI / AMI        Throughput             # Samples
 Mean over stream       Runtime                # Features
 Final score            Memory
                        Micro-clusters
       │
       └──────────────────────┬──────────────────────┐
                              ▼                      │
                         STABILITY                  │
                              │                      │
                         Mean ± Std                 │
                              │                      │
                              └──────────────────────┘
```

## Kết luận

Bộ tiêu chí cốt lõi:

### Quality

```text
ARI
NMI
AMI
Mean quality over stream
Final quality
```

### Efficiency

```text
Throughput
Runtime
Peak Memory
```

### Model Complexity

```text
Number of micro-clusters
```

### Scalability

```text
Runtime vs number of samples
Memory vs number of samples
Quality vs dimensionality
```

### Stability

```text
Mean ± Std across multiple seeds
```

Không cần đưa concept drift detection vào core.

Một benchmark tốt không phải là benchmark có thật nhiều metric. Quan trọng hơn là:

```text
Same data
    +
Same stream order
    +
Same evaluation protocol
    +
Fair hyperparameter selection
    +
Repeated runs
    +
Time-aware evaluation
    +
Quality + Efficiency
```

Từ đó có thể kết luận:

> **Algorithm nào clustering tốt hơn, nhanh hơn, ít tốn memory hơn và scale tốt hơn trên streaming data.**
