# Streaming Clustering Benchmark Framework

Chạy nhiều thuật toán clustering của River — với nhiều setting (tham số + scaler)
— trên **cùng một dataset**, rồi so sánh hiệu năng và chất lượng.

```
Dataset ──▶ [ thuật toán + params + scaler ] × N ──▶ Runner ──▶ Báo cáo
                                                        │
                      (runtime, RAM, số cluster, chất lượng)
```

Mọi setting tiêu thụ cùng một stream theo đúng thứ tự bản ghi (không xáo trộn),
đúng như hệ thống chạy trên luồng dữ liệu thật: mỗi mẫu được `predict_one` trước
rồi `learn_one` sau (test-then-train). Nhãn (nếu có) chỉ dùng để chấm điểm, model
không bao giờ thấy.

## Đo những gì

**Hiệu năng** (mọi dataset):

| Metric | Ý nghĩa |
|---|---|
| `total_runtime_s` | tổng thời gian xử lý (chỉ tính predict+learn, không tính I/O) |
| `throughput_sps` | số mẫu / giây |
| `ms_per_sample` | mili-giây / mẫu |
| `peak_memory_mb` | RAM của model qua `_raw_memory_usage` của River (gồm cả scaler) |
| `mean_micro_clusters` / `max_micro_clusters` | số micro-cluster trung bình / lớn nhất |
| `final_macro_clusters` | số cluster cuối cùng |

Các chỉ số tài nguyên được lấy mẫu tại nhiều **checkpoint** dọc stream, nên thấy
được xu hướng theo số mẫu (scalability), không chỉ một con số cuối.

**Chất lượng — cần nhãn** (external): `ARI`, `NMI`, `AMI`, `purity` (macro-average).
Được tính theo từng **window** dọc stream (mặc định `window_size=1000`) nên báo cả
`mean / median / final` over stream — quan trọng hơn chỉ điểm cuối vì phản ánh toàn
bộ quá trình streaming — cộng một giá trị whole-stream trên toàn bộ mẫu.

**Chất lượng — không cần nhãn** (internal, bật bằng `--internal`): `silhouette`,
`davies_bouldin`, `calinski_harabasz`. Dùng cho stream thật không có labels; cũng
tính theo window (mean/median/final).

## Dataset

- **Synthetic** — các cụm Gaussian tĩnh, cấu hình được (`n_samples`, `n_features`,
  `n_clusters`, `cluster_std`, `noise`, `seed`), có nhãn nên chấm được chất lượng.
- **KDDCup99** — 34 đặc trưng liên tục, tự tải qua `sklearn.datasets.fetch_kddcup99`.
- **KDDCup98** — 56 trường RFM đọc từ `cup98LRN.txt` (phải chỉ đường dẫn).

## Thuật toán & scaler

Thuật toán: `CluStream`, `DenStream`, `DBSTREAM`, `STREAMKMeans`, `KMeans`.
Scaler online: `none`, `standard`, `minmax`, `maxabs`, `robust`, `adaptive_standard`.

Scaler ảnh hưởng rất lớn: KDD99 có đặc trưng dải giá trị cực rộng, `minmax` dễ ép
mọi điểm về gần nhau khiến model gộp còn 1 cluster — thường nên dùng `standard`.
Bán kính (`epsilon` của DenStream, `clustering_threshold` của DBSTREAM) phải khớp
với thang dữ liệu sau khi scale.

## Chạy

```bash
pip install -r requirements.txt   # river được import từ cây nguồn ../river

# So sánh mọi thuật toán mặc định trên synthetic có nhãn
python -m benchmark synthetic --n-samples 20000 --n-clusters 5

# So sánh ảnh hưởng của scaler trên một thuật toán
python -m benchmark synthetic --algorithms DBSTREAM --scalers none,standard,minmax

# KDDCup99 (tự tải)
python -m benchmark kdd99 --n-samples 50000 --scalers standard

# KDDCup98 (file local)
python -m benchmark kdd98 --path cup98LRN.txt --n-samples 20000

# Bật internal metrics (cho stream không nhãn hoặc để so thêm)
python -m benchmark synthetic --internal silhouette,davies_bouldin

# Lặp nhiều seed -> mean ± std, kèm xuất JSON + biểu đồ
python -m benchmark synthetic --seeds 1,2,3 --out results/

# Scalability: quét runtime/memory/quality theo #samples hoặc #features
python -m benchmark scalability --axis samples --values 10000,100000,1000000
python -m benchmark scalability --axis features --values 2,10,50,100
```

Cờ chung: `--algorithms`, `--scalers`, `--seeds` (nhiều seed thì gộp mean±std),
`--n-samples`, `--window` (kích thước window đánh giá chất lượng), `--checkpoints`
(số lần đo tài nguyên), `--internal`, `--out` (ghi `results.json` + 4 biểu đồ:
runtime/RAM/số cluster theo số mẫu, và ARI cuối theo thuật toán).

## Dùng bằng code

```python
from benchmark.datasets import SyntheticStream, StreamConfig
from benchmark.experiments import BenchmarkConfig, Setting, run_benchmark
from benchmark.reports import efficiency_table, quality_table

dataset = SyntheticStream(StreamConfig(n_samples=20_000, n_clusters=5, seed=42))
config = BenchmarkConfig(settings=[
    Setting("DBSTREAM", scaler="standard"),
    Setting("DBSTREAM", params={"clustering_threshold": 2.0}, scaler="standard",
            label="DBSTREAM τ=2"),
    Setting("STREAMKMeans", params={"n_clusters": 5}),
    Setting("CluStream", params={"n_macro_clusters": 5}),
])
results = run_benchmark(dataset, config)
print(efficiency_table(results))
print(quality_table(results))
```

## Tái lập

Cùng một dataset stream cho mọi setting (không xáo trộn); dataset lặp lại được nên
mỗi thuật toán nhận đúng chuỗi mẫu như nhau; chạy lặp nhiều seed rồi gộp; mỗi kết
quả lưu đầy đủ params + scaler. Kiến trúc: `datasets/` · `algorithms/` ·
`evaluation/` · `reports/` · `experiments/`. Xem `walkthrough.py` để chạy thử có
giải thích từng bước, và `benchmark/README.md` cho chi tiết API.
