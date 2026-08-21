# Hướng đi Benchmark Streaming Clustering dưới Concept Drift

## 1. Mục tiêu

Xây dựng một framework benchmark dành cho **streaming clustering** trên dữ liệu dạng data stream, tập trung vào khả năng của thuật toán khi **cluster structure thay đổi theo thời gian**.

Các thuật toán mục tiêu có thể bao gồm:

- CluStream
- DenStream
- DBSTREAM
- Các streaming clustering algorithm khác trong River/custom implementation

Mục tiêu không chỉ là trả lời:

> "Thuật toán nào có clustering quality cao nhất?"

mà phải trả lời được:

> "Thuật toán nào thích nghi tốt nhất khi cấu trúc cluster thay đổi theo thời gian, với chi phí tính toán và bộ nhớ hợp lý?"

---

# 2. Ý tưởng cốt lõi

Trong streaming clustering, không nên định nghĩa concept drift chỉ đơn giản là:

```text
drift = một timestamp
```

Thay vào đó, nên mô hình hóa:

```text
Ground Truth
    ↓
Evolution của cluster structure theo thời gian
    ↓
Drift events / drift intervals
    ↓
Streaming clustering algorithm
    ↓
Cluster evolution thực tế của model
    ↓
Evaluation
```

Ground truth cần mô tả **cluster structure thay đổi như thế nào**, chứ không chỉ nói "drift xảy ra ở sample thứ N".

---

# 3. Ground Truth cho Streaming Clustering

## 3.1. Ground truth là gì?

Ground truth là trạng thái cluster được biết trước khi benchmark.

Ví dụ:

```text
t = 0 → 5000

C1 = center (0, 0)
C2 = center (10, 10)

t = 5001 → 10000

C1 = center (5, 5)
C2 = center (10, 10)
```

Ground truth cho biết:

```text
t = 5001
C1 moved:
(0, 0) → (5, 5)
```

Model không được nhìn thấy thông tin ground truth này trong quá trình online learning.

Ground truth chỉ được dùng để **đánh giá sau đó**.

---

# 4. Ground Truth nên mô tả Cluster Evolution

Nên xây dựng một representation cho cluster state.

Ví dụ:

```python
ClusterState(
    cluster_id="C1",
    center=(0.0, 0.0),
    spread=1.0,
    weight=1000,
    active=True
)
```

Một stream có thể có:

```text
State S1
    C1
    C2

State S2
    C1 moved
    C2 unchanged

State S3
    C1 disappeared
    C2 remains
    C3 appeared
```

Từ đó tạo ra các **drift events**.

---

# 5. Taxonomy của Drift cho Clustering

Framework nên hỗ trợ nhiều loại thay đổi khác nhau.

## 5.1. Cluster Movement

Cluster thay đổi vị trí.

```text
Before:

   C1

After:

              C1
```

Ví dụ:

```text
μ1: (0, 0) → (5, 5)
```

Dùng để đánh giá khả năng tracking cluster.

---

## 5.2. Cluster Birth

Cluster mới xuất hiện.

```text
Before:

C1    C2

After:

C1    C2    C3
```

Đánh giá khả năng phát hiện và hình thành cluster mới.

---

## 5.3. Cluster Death

Cluster cũ biến mất.

```text
Before:

C1    C2    C3

After:

C1    C2
```

Đánh giá khả năng loại bỏ obsolete/outdated structure.

---

## 5.4. Cluster Split

Một cluster tách thành nhiều cluster.

```text
Before:

       C1

After:

   C1a       C1b
```

Đây là một dạng structural drift quan trọng.

---

## 5.5. Cluster Merge

Nhiều cluster hợp thành một cluster.

```text
Before:

C1       C2

After:

      C1
```

Đánh giá khả năng cập nhật cấu trúc cluster.

---

## 5.6. Recurring Drift

Một concept cũ quay trở lại.

```text
A → B → C → A
```

Ví dụ:

```text
C1 position:

(0,0)
  ↓
(10,10)
  ↓
(20,0)
  ↓
(0,0)
```

Dạng này rất hữu ích để đánh giá khả năng tận dụng hoặc thích nghi với recurring patterns.

---

## 5.7. Gradual Drift

Thay đổi xảy ra trong một khoảng thời gian thay vì một timestamp duy nhất.

Ví dụ:

```text
Concept A
██████████
          \
           \
            \
             ██████████
                 Concept B
```

Ground truth nên lưu:

```text
drift_start = 10000
drift_end   = 12000
```

Không nên ép mọi drift thành một timestamp duy nhất.

---

# 6. Synthetic Stream là thành phần quan trọng

Synthetic data nên là nền tảng đầu tiên của benchmark vì có thể kiểm soát chính xác ground truth.

Framework nên cho phép cấu hình:

```python
StreamConfig(
    n_samples=100_000,
    n_features=2,
    n_clusters=5,
    noise=0.05,
    seed=42
)
```

Và drift:

```python
DriftConfig(
    type="movement",
    start=20_000,
    duration=0,
    magnitude=5.0
)
```

Có thể tạo nhiều drift:

```text
0 ─────── 20k ─────── 40k ─────── 60k ─────── 80k

       movement        birth       death       merge
```

Mỗi event phải được lưu trong ground-truth metadata.

---

# 7. Ground Truth Metadata

Một drift event có thể có schema:

```python
DriftEvent(
    drift_id=1,
    type="movement",
    start=20_000,
    end=20_000,
    affected_clusters=["C1"],
    magnitude=5.0
)
```

Ví dụ cluster birth:

```python
DriftEvent(
    drift_id=2,
    type="birth",
    start=40_000,
    end=40_000,
    affected_clusters=["C3"]
)
```

Ví dụ gradual drift:

```python
DriftEvent(
    drift_id=3,
    type="movement",
    start=60_000,
    end=65_000,
    affected_clusters=["C1"],
    magnitude=10.0
)
```

---

# 8. Ground Truth có ý nghĩa gì?

Ground truth **không phải metric**.

Nó là reference để đánh giá model.

Ví dụ:

```text
Ground truth:

Drift 1 = 20,000
Drift 2 = 40,000
```

Model thực tế:

```text
Drift 1 detected/adapted at 20,300
Drift 2 detected/adapted at 41,000
```

Từ đó có thể tính:

```text
Detection/Adaptation delay

Drift 1:
20,300 - 20,000 = 300 samples

Drift 2:
41,000 - 40,000 = 1,000 samples
```

Nếu model không có cơ chế phát ra drift event, vẫn có thể đánh giá **adaptation** thông qua cluster quality theo thời gian.

---

# 9. Không phải Streaming Clustering Algorithm nào cũng có Drift Detector

Đây là điểm quan trọng.

Một thuật toán như DenStream hoặc CluStream có thể không trả về:

```text
DRIFT DETECTED
```

Thay vào đó model thích nghi bằng cách:

```text
micro-cluster cũ
      ↓
decay / fading
      ↓
disappear

micro-cluster mới
      ↓
appear
```

Do đó benchmark không nên yêu cầu mọi thuật toán phải có `detect_drift()`.

Nên phân biệt hai khái niệm:

## Drift Detection

Model có phát hiện drift event hay không?

## Drift Adaptation

Model có thích nghi được với cluster structure mới hay không?

Đối với streaming clustering, **drift adaptation nên là tiêu chí trung tâm**.

---

# 10. Evaluation Framework

Nên chia evaluation thành 3 nhóm lớn.

```text
Benchmark
│
├── 1. Cluster Quality
│
├── 2. Drift Adaptation
│
└── 3. Efficiency
```

---

# 11. Nhóm 1 — Cluster Quality

Khi synthetic dataset có ground-truth cluster labels, có thể dùng:

- ARI
- NMI
- Purity
- Silhouette Score
- Adjusted Mutual Information nếu phù hợp

Không nên chỉ lấy metric tại sample cuối cùng.

Thay vào đó tính theo thời gian.

Ví dụ:

```text
time       ARI
----------------
0          0.91
5000       0.92
10000      0.40   ← drift
10500      0.55
11000      0.72
12000      0.86
```

Điều này cho thấy cả:

- chất lượng trước drift
- mức độ suy giảm
- tốc độ recovery
- chất lượng sau adaptation

---

# 12. Nhóm 2 — Drift Adaptation

Đây là phần quan trọng nhất.

## 12.1. Performance Drop

Đo mức suy giảm quality khi drift xảy ra.

Ví dụ:

```text
ARI_before = 0.92
ARI_min    = 0.40

Performance Drop = 0.92 - 0.40
                 = 0.52
```

Performance drop càng thấp càng tốt.

---

## 12.2. Recovery Time

Thời gian model cần để quay lại một mức quality nhất định.

Ví dụ:

```text
ARI trước drift = 0.92

sau drift:
0.40
0.55
0.72
0.86
0.90
```

Nếu recovery threshold = 0.90:

```text
Recovery Time = 2,000 samples
```

---

## 12.3. Adaptation Delay

Nếu ground truth có drift start:

```text
drift_start = 20,000
```

và model bắt đầu phản ứng/đạt adaptation criterion tại:

```text
20,500
```

thì:

```text
Adaptation Delay = 500 samples
```

---

## 12.4. Post-drift Quality

Đo chất lượng sau khi model đã ổn định.

Ví dụ:

```text
Pre-drift ARI  = 0.92
Post-drift ARI = 0.88
```

Thuật toán có thể chậm nhưng cuối cùng vẫn đạt chất lượng tốt.

---

## 12.5. Area Under Recovery Curve

Thay vì chỉ lấy một điểm recovery, có thể đo toàn bộ quá trình:

```text
quality
1.0 |████████
    |       \
    |        \
0.5 |         \____
    |
    +-----------------> time
          drift
```

Metric dạng area phản ánh tổng chất lượng bị mất trong quá trình adaptation.

Đây có thể là một metric rất tốt cho benchmark.

---

# 13. Nhóm 3 — Efficiency

Streaming clustering không chỉ cần chính xác.

Cần đo:

## Runtime

```text
samples / second
```

hoặc:

```text
ms / sample
```

## Memory

```text
MB
```

## Number of Micro-clusters

Đặc biệt quan trọng với:

- DenStream
- CluStream
- DBSTREAM

## Scalability

Đánh giá khi:

```text
n_samples tăng
n_features tăng
n_clusters tăng
```

---

# 14. Benchmark Protocol

Mỗi algorithm nên chạy cùng một stream.

Ví dụ:

```text
Dataset D1
    ↓
CluStream
DenStream
DBSTREAM
...
```

Không được để từng algorithm dùng stream khác nhau.

Các điều kiện nên cố định:

```text
Random seed
Data stream
Number of samples
Feature dimensions
Drift events
Evaluation intervals
```

---

# 15. Evaluation theo Time Window

Không nên chỉ đánh giá cuối stream.

Có thể dùng window:

```text
window_size = 1000
```

Ví dụ:

```text
samples 0–999
samples 1000–1999
samples 2000–2999
...
```

Mỗi window lưu:

```python
{
    "start": 1000,
    "end": 1999,
    "ari": 0.91,
    "n_micro_clusters": 8,
    "runtime": 0.12,
    "memory": 42.5
}
```

Từ đó có thể vẽ performance curve.

---

# 16. Benchmark Scenario Matrix

Nên xây dựng nhiều scenario thay vì chỉ một dataset.

| Scenario | Drift | Mục đích |
|---|---|---|
| S1 | Stationary | Baseline |
| S2 | Abrupt movement | Khả năng tracking |
| S3 | Gradual movement | Adaptation chậm |
| S4 | Birth | Phát hiện cluster mới |
| S5 | Death | Loại bỏ cluster cũ |
| S6 | Split | Structural adaptation |
| S7 | Merge | Structural adaptation |
| S8 | Recurring | Concept recurrence |
| S9 | Multiple drift | Nhiều event |
| S10 | Noise + drift | Robustness |

---

# 17. Baseline Stationary

Trước khi benchmark drift, phải có stationary baseline.

```text
No drift
    ↓
CluStream
DenStream
DBSTREAM
    ↓
Quality + Runtime + Memory
```

Mục đích:

- kiểm tra thuật toán có hoạt động đúng hay không
- biết performance cơ bản
- tránh nhầm adaptation problem với clustering problem

---

# 18. So sánh Before / During / After Drift

Mỗi drift event nên được đánh giá theo 3 giai đoạn:

```text
                Drift
                  │
                  ▼
───────────────│───────────────
     BEFORE    │   AFTER
               │
```

Có thể định nghĩa:

### Before

Quality trung bình trước drift.

### During

Quality trong quá trình drift/adaptation.

### After

Quality sau khi model ổn định.

Ví dụ:

| Algorithm | Before ARI | Min ARI | Recovery | After ARI |
|---|---:|---:|---:|---:|
| CluStream | 0.92 | 0.41 | 1800 | 0.86 |
| DenStream | 0.90 | 0.55 | 900 | 0.89 |
| DBSTREAM | 0.91 | 0.48 | 1200 | 0.88 |

Bảng này có ý nghĩa hơn nhiều so với:

| Algorithm | Final ARI |
|---|---:|
| CluStream | 0.86 |
| DenStream | 0.89 |
| DBSTREAM | 0.88 |

---

# 19. Hyperparameter Benchmark

Không nên benchmark bằng cách chọn parameter tốt nhất riêng cho từng stream rồi báo kết quả.

Điều đó dễ dẫn tới overfitting.

Nên có:

```text
Training / calibration streams
            ↓
    hyperparameter selection
            ↓
      fixed parameters
            ↓
       test streams
```

Ví dụ:

```text
Calibration:
S1, S2, S3

Test:
S4, S5, S6, S7
```

Hyperparameters được chọn trước và giữ cố định trên test set.

---

# 20. Repeated Runs

Streaming algorithms có thể phụ thuộc vào:

- random initialization
- seed
- stream ordering
- sampling

Do đó nên chạy nhiều seed:

```text
seed = 1
seed = 2
seed = 3
seed = 4
seed = 5
```

Sau đó báo:

```text
mean ± std
```

hoặc confidence interval nếu framework cần mức thống kê cao hơn.

---

# 21. Architecture đề xuất

```text
benchmark/
│
├── datasets/
│   ├── synthetic/
│   │   ├── stationary.py
│   │   ├── movement.py
│   │   ├── birth.py
│   │   ├── death.py
│   │   ├── split.py
│   │   ├── merge.py
│   │   └── recurring.py
│   │
│   └── real/
│
├── ground_truth/
│   ├── schema.py
│   ├── events.py
│   └── tracker.py
│
├── algorithms/
│   ├── clustream.py
│   ├── denstream.py
│   └── dbstream.py
│
├── evaluation/
│   ├── clustering_quality.py
│   ├── adaptation.py
│   ├── efficiency.py
│   └── aggregation.py
│
├── experiments/
│   ├── stationary.py
│   ├── abrupt_drift.py
│   ├── gradual_drift.py
│   └── recurring_drift.py
│
└── reports/
    ├── tables.py
    └── plots.py
```

---

# 22. Một Experiment Record nên lưu gì?

Mỗi experiment nên lưu metadata đầy đủ:

```python
{
    "algorithm": "DenStream",

    "dataset": "synthetic_movement_01",

    "seed": 42,

    "parameters": {
        "epsilon": 0.1,
        "mu": 50,
        "beta": 0.2,
        "lambda": 0.001
    },

    "stream": {
        "n_samples": 100000,
        "n_features": 2
    },

    "drifts": [
        {
            "id": 1,
            "type": "movement",
            "start": 20000,
            "end": 20000,
            "affected_clusters": ["C1"]
        }
    ],

    "metrics": {
        "ari_mean": 0.86,
        "ari_min": 0.41,
        "recovery_time": 1800,
        "runtime": 12.5,
        "memory_mb": 48.2
    }
}
```

Điều này giúp benchmark reproducible.

---

# 23. Visualization

Framework nên sinh ra các biểu đồ chính.

## Cluster Quality over Time

```text
ARI
1.0 |───────────────
    |              \
    |               \
0.5 |                \____
    |
    +--------------------------> time
                    ↑
                  drift
```

## Multiple Algorithms

Vẽ:

```text
ARI vs Time
```

cho:

- CluStream
- DenStream
- DBSTREAM

trên cùng một scenario.

## Drift Timeline

Đánh dấu:

```text
       drift 1          drift 2
          ↓                ↓
──────────│────────────────│──────────
```

## Recovery Curve

Zoom riêng từng drift event.

---

# 24. Real-world Data

Sau khi synthetic benchmark ổn định, mới chuyển sang realistic streams.

Vấn đề:

```text
Real data
    ↓
không có ground truth drift rõ ràng
```

Do đó có thể dùng:

- domain events
- known structural changes
- external events
- manually annotated intervals
- change points từ domain experts

Nhưng cần ghi rõ:

> Đây là observed/annotated drift hoặc proxy ground truth, không nhất thiết là ground truth tuyệt đối.

Không nên coi mọi change point được phát hiện bằng một detector khác là ground truth tuyệt đối.

---

# 25. Nguyên tắc quan trọng

## Principle 1 — Ground truth không được leak vào model

Ground truth chỉ dùng cho evaluation.

```text
Stream ─────→ Model
              ↑
          không thấy GT

Ground Truth ─────→ Evaluator
```

---

## Principle 2 — Không chỉ đánh giá final score

Phải đánh giá performance theo thời gian.

```text
Before → Drift → Degradation → Recovery → Stable
```

---

## Principle 3 — Drift và adaptation là hai khái niệm khác nhau

Một model có thể:

```text
không phát ra drift event
```

nhưng vẫn:

```text
thích nghi cluster structure rất tốt
```

Do đó không được loại model chỉ vì nó không có explicit drift detector.

---

## Principle 4 — Không overfit hyperparameters

Parameter tuning phải tách khỏi test streams.

---

## Principle 5 — Benchmark phải reproducible

Cần cố định:

- seed
- stream generator
- drift configuration
- evaluation windows
- hyperparameters
- software version
- dataset version

---

# 26. Roadmap triển khai

## Phase 1 — Validate Algorithm Correctness

Trước benchmark:

```text
Algorithm
    ↓
Paper / reference implementation
    ↓
Synthetic examples
    ↓
Expected behavior
```

Kiểm tra:

- micro-cluster formation
- fading/decay
- merge/split behavior
- outlier handling
- parameter semantics

---

## Phase 2 — Stationary Benchmark

Chưa có drift.

Đo:

```text
ARI
NMI
Runtime
Memory
Micro-cluster count
```

---

## Phase 3 — Synthetic Drift Benchmark

Implement:

```text
movement
birth
death
split
merge
gradual
recurring
```

Có ground truth đầy đủ.

---

## Phase 4 — Adaptation Metrics

Implement:

```text
performance drop
recovery time
adaptation delay
post-drift quality
area under recovery curve
```

---

## Phase 5 — Hyperparameter Protocol

Tách:

```text
calibration streams
```

và:

```text
test streams
```

Sau đó freeze parameters.

---

## Phase 6 — Efficiency Benchmark

Đo:

```text
runtime
throughput
memory
micro-cluster count
scalability
```

---

## Phase 7 — Real-world Streams

Cuối cùng mới benchmark trên dữ liệu thực tế.

Ground truth có thể là:

```text
annotated / domain-informed / event-based
```

và phải ghi rõ mức độ chắc chắn.

---

# 27. Kết quả cuối cùng nên trả lời được

Sau khi hoàn thành framework, benchmark phải trả lời được các câu hỏi:

### Clustering quality

> Thuật toán nào tạo cluster tốt nhất khi stream stationary?

### Robustness

> Thuật toán nào ít bị suy giảm nhất khi drift xảy ra?

### Adaptation

> Thuật toán nào thích nghi nhanh nhất?

### Recovery

> Thuật toán nào phục hồi nhanh nhất sau drift?

### Structural adaptation

> Thuật toán nào xử lý tốt birth/death/split/merge?

### Recurring behavior

> Thuật toán nào xử lý tốt recurring drift?

### Efficiency

> Thuật toán nào đạt trade-off tốt nhất giữa quality, runtime và memory?

### Scalability

> Khi stream lớn hơn hoặc số feature tăng, thuật toán nào vẫn hoạt động ổn định?

---

# 28. Kết luận

Hướng benchmark phù hợp cho streaming clustering nên được xây dựng quanh ý tưởng:

```text
                Synthetic Stream
                      │
                      ▼
             Ground Truth Cluster
                   Evolution
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
     Drift Events           Cluster States
          │                       │
          └───────────┬───────────┘
                      ▼
              Streaming Model
                      │
                      ▼
              Observed Clusters
                      │
                      ▼
                  Evaluator
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
 Cluster Quality   Adaptation    Efficiency
       │              │              │
      ARI          Drop/Delay      Runtime
      NMI          Recovery        Memory
      ...          Post-quality    Throughput
```

**Trọng tâm của framework không phải chỉ là "drift detection".**

Đối với streaming clustering, trọng tâm nên là:

> **Cluster Quality + Drift Adaptation + Efficiency**

Trong đó **ground truth cluster evolution** là nền tảng để biết chính xác stream đã thay đổi như thế nào và từ đó đánh giá model có thích nghi đúng, nhanh và ổn định hay không.

Đây cũng là cơ sở tốt để xây dựng một benchmark framework có thể mở rộng cho nhiều thuật toán clustering trong River.
