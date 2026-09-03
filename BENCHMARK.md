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
| `peak_memory_mb` | RAM của model qua `_raw_memory_usage` của River (gồm cả scaler) — con số so sánh được giữa các thuật toán |
| `peak_process_rss_mb` | RSS của cả tiến trình, chỉ để tham chiếu (gồm cả interpreter + dataset) |
| `mean_micro_clusters` / `max_micro_clusters` | số micro-cluster trung bình / lớn nhất |
| `final_macro_clusters` | số cluster cuối cùng |

Các chỉ số tài nguyên được lấy mẫu tại nhiều **checkpoint** dọc stream, nên thấy
được xu hướng theo số mẫu (scalability), không chỉ một con số cuối. Riêng phép đo
RAM chỉ chạy **tại checkpoint**: `_raw_memory_usage` duyệt toàn bộ object graph của
model nên tốn hơn việc xử lý một mẫu tới hai bậc độ lớn. Hệ quả là `peak_memory_mb`
là max trên các checkpoint — tăng `--checkpoints` nếu cần chặn trên chặt hơn cho
model có prune (DenStream, DBSTREAM).

**Chất lượng — cần nhãn** (external): `ARI`, `NMI`, `AMI`, `purity`. Được theo dõi
đồng thời theo hai cách, vì chúng trả lời hai câu hỏi khác nhau:

| Trường | Cách tính | Dùng khi |
|---|---|---|
| `quality` | luỹ kế trên toàn bộ mẫu đã chấm | điểm tổng kết một con số |
| `quality_stream` | **từng window riêng** (metric reset ở mỗi biên), rồi lấy mean/median/final | chỉ số time-aware của protocol |
| `quality_cumulative_stream` | đường luỹ kế từ đầu stream, lấy mean/median/final | đối chiếu; `final` của nó chính là `quality` |

Lấy trung bình đường **luỹ kế** không phải là "mean over stream": nó chỉ đánh trọng
số lại phần đầu stream chứ không đo hành vi theo thời gian. Vì vậy `quality_stream`
— thứ mà `stream_quality_table()` in ra mặc định — dùng giá trị per-window.

Mẫu không có ground truth (ví dụ nhiễu nền do generator chèn vào) bị **bỏ qua** khi
tính external metric, chứ không bị coi là một lớp riêng.

**Chất lượng — không cần nhãn** (internal, bật bằng `--internal`): `silhouette`
streaming của River. Nó cần `centers` của model nên chỉ chạy với thuật toán có expose
centers, và được chấm trong **không gian đã scale** — cùng không gian model làm việc.
Lưu ý định nghĩa của River là tỉ số cohesion/separation nên **nhỏ hơn là tốt hơn**
(`bigger_is_better = False`), ngược với silhouette batch của scikit-learn.

## Dataset

### Synthetic

Bộ sinh theo cấu trúc của `RandomRBFGeneratorEvents` trong MOA (Kranen et al.
2010) — chuẩn de-facto mà DenStream, DBSTREAM, evoStream đều đánh giá trên đó.
Cụm nằm trong hypercube đơn vị; điểm sinh theo dạng *hướng ngẫu nhiên × biên độ
Gaussian*, nên **kích thước cụm không phụ thuộc `d`** — điều kiện để phép quét
theo số chiều chỉ thay đổi đúng số chiều.

Nhãn `y` là **id của cụm sinh ra điểm**, không phải class label gộp nhiều cụm:
với bài toán clustering thì khôi phục đúng thành phần hỗn hợp mới là mục tiêu.

| Tham số | Ý nghĩa |
|---|---|
| `separation` | khoảng cách giữa cặp tâm gần nhất, tính theo **bán kính cụm**. Núm điều khiển độ khó chính. `1` = chồng lấn nặng, `3` = tách rõ, `6` = tầm thường |
| `cluster_radius` | chỉ dùng khi `separation=None` (chế độ tương thích MOA) |
| `radius_range`, `weight_range` | cụm không đều về mật độ và kích thước |
| `noise`, `noise_in_cluster` | nhiễu nền, nhãn `None` nên metric bỏ qua |
| `drift_speed`, `drift_every` | tâm cụm trôi, dội lại ở biên hypercube |
| `event_frequency`, `events` | cụm gộp / tách / sinh ra / biến mất |
| `block_ordered` | **control**: phát theo khối cùng lớp, để tái hiện hiện tượng thoái hoá cửa sổ |

`separation` **suy ra** bán kính từ layout (`radius = d_min / (2·separation)`)
chứ không dùng làm chặn dưới. Lý do: hypercube càng rộng khi `d` tăng, nên ràng
buộc khoảng cách tối thiểu không bao giờ bị chạm và mọi layout đều dễ như nhau —
đo được: ở `d=10` hai cấu hình `separation=0.8` và `3.0` cho kết quả **giống hệt**
trước khi sửa.

Bảy kịch bản đặt tên sẵn trong `SCENARIOS`: `static_easy`, `static_overlap`,
`unbalanced`, `noisy`, `incremental_drift`, `evolving`, `block_ordered`.

```python
from benchmark.datasets import make_scenario
ds = make_scenario("evolving", n_samples=50_000, n_features=20, n_clusters=8)
```

### Real

Cột `ρ_deg` là tỉ lệ cửa sổ 1000 mẫu chỉ chứa **một lớp**, đo trên 50k bản ghi
đầu. `ρ_deg` cao nghĩa là chỉ số per-window thoái hoá trên bộ đó và **chỉ được
đọc con số toàn stream**.

| Dataset | n | d | classes | ρ_deg | med. lớp áp đảo | Vai trò |
|---|---|---|---|---|---|---|
| `kdd99` | 494k | 34 | 23 | **72%** | 1.000 | benchmark mặc định của ngành; chỉ đọc toàn stream |
| `covtype` | 581k | 10 | 7 | 0% | 0.735 | bộ dùng nhiều thứ 2; quy mô lớn, cửa sổ dùng được |
| `insects` | 24k–355k | 33 | 6 | **2%** | 0.276 | **drift có tài liệu**; cửa sổ cân bằng nhất |
| `keystroke` | 20k | 31 | 51 | 0% | 0.400 | nhiều lớp so với độ dài stream |
| `shuttle` | 49k | 9 | 2 | 0% | 0.931 | nhanh; chỉ để đối chứng purity vs ARI |
| `electricity` | 45k | 8 | 2 | 0% | 0.579 | chỉ để liên tục với literature; xem cảnh báo |
| `kddcup98` | 95k | 56 | 2 | — | — | RFM, giữ từ repo gốc |

**INSECTS** (Souza et al. 2020) là bộ đáng chú ý nhất: nó được tạo ra chính vì
các benchmark quen thuộc đều không ghi rõ drift ở đâu, thuộc loại gì — nên khi
điểm tụt xuống không thể quy cho distribution shift thay vì underfitting hay
nhiễu. INSECTS đọc côn trùng bay bằng cảm biến quang, drift do thay đổi nhiệt độ
có kiểm soát, nên **loại và thời điểm drift đều biết trước**. 7 biến thể:
abrupt / gradual / incremental / incremental-reoccurring, mỗi loại balanced và
imbalanced.

Đo được: trên KDDCup99 hai bảng xếp hạng (toàn stream vs per-window) **ngược
nhau** (Spearman ρ = −0.8); trên INSECTS chúng **trùng khớp hoàn toàn** (ρ = +1).

`covtype` và `kdd99` đều đến theo khối cùng lớp. Đó không phải lý do bỏ chúng,
nhưng là lý do đọc `RunResult.window_label_diagnostics` trước khi đọc bất kỳ
điểm per-window nào.

## Thuật toán

Framework benchmark **mọi** clusterer trong `river.cluster`, không chỉ nhóm
distance-based. Bảy thuật toán chia làm ba nhóm về mặt *protocol*, và framework
xử lý cả ba thay vì giả định nhóm đầu:

| Thuật toán | Gán mẫu vào cụm? | Đầu vào | Ghi chú |
|---|---|---|---|
| `CluStream` | ✅ | vector số | micro-cluster + cluster feature |
| `DStream` | ✅ | vector số | gom cụm mật độ dựa trên lưới (grid density); cụm hình dạng bất kỳ |
| `DenStream` | ✅ | vector số | mật độ + suy giảm thời gian |
| `DBSTREAM` | ✅ | vector số | mật độ chia sẻ giữa micro-cluster |
| `STREAMKMeans` | ✅ | vector số | k-means theo chunk |
| `KMeans` | ✅ | vector số | k-means tăng dần |
| `ODAC` | ❌ | vector số | gom cụm **các biến**, không phải mẫu |
| `TextClust` | ✅ | **token counts** | so bằng tf-idf cosine |

Hai trường hợp đặc biệt và cách framework xử lý:

- **ODAC** dựng cây phân cấp trên *tập biến*; `predict_one` của nó raise
  `NotImplementedError` theo thiết kế. Không có nhãn per-sample nên ARI/NMI/AMI/
  purity **không định nghĩa được** — nó bị loại khỏi bảng chất lượng (đúng, không
  phải lỗi) và được đo bằng **bảng cấu trúc** (số cụm biến, số cụm active, chiều
  cao cây) cộng runtime/RAM. Cột `Macro K` để trống cho nó, vì `n_clusters` của
  ODAC đếm nhóm *biến* — khác đơn vị, không so được với cụm mẫu.
- **TextClust** ăn dictionary tần suất từ, không phải toạ độ. Nó cần
  preprocessor `bag_of_words` và một text stream; chuẩn hoá tần suất từ bằng
  `StandardScaler` là vô nghĩa. Nó cũng nhận vị trí stream qua `t` khi bật
  `real_time_fading` — adapter tự phát hiện qua signature và truyền vào.

```python
Setting("ODAC", params={"n_min": 100})                       # structure-only
Setting("TextClust", scaler="bag_of_words",
        scaler_params={"on": "body"})                        # text
```

`algorithms_for()` lọc theo năng lực: mặc định trả về nhóm so sánh được bằng
external metric; `partitions_samples=False` cho ODAC; `input_kind="tokens"` cho
TextClust.

## Preprocessing

Slot đứng trước clusterer nhận **mọi transformer unsupervised của river** — 25
bước, không riêng scaler. Benchmark clustering không có `y` lúc học nên thứ nào
cần target đều bị loại theo định nghĩa; phần còn lại lấp những lỗ hổng thật:

| Nhóm | Bước | Lỗ hổng nó lấp |
|---|---|---|
| Scaling | `standard` `minmax` `maxabs` `robust` `adaptive_standard` `normalizer` `binarizer` | mọi tham số bán kính (`epsilon`, `clustering_threshold`) đọc theo đơn vị scaler tạo ra |
| Missing values | `impute` `impute_previous` | stream thật có thiếu giá trị; river coi thiếu = **vắng key**, clusterer sẽ so hai điểm trên hai tập key khác nhau |
| Categorical | `one_hot_categorical` `one_hot` `ordinal` `feature_hasher` `gap_encoder` | không có thì phải **vứt** cột nominal — KDDCup99 mất 3 cột, CoverType mất 44 |
| Dimensionality | `sparse_projection` `gaussian_projection` `variance_threshold` | giảm chiều bảo toàn khoảng cách (Johnson–Lindenstrauss); cột hằng số không đóng góp gì vào khoảng cách mà còn **phá scaler** |
| Text | `bag_of_words` `tfidf` `lda` | TextClust ăn token counts; `lda` biến text thành vector dày để clusterer số dùng được |
| Kernel | `rbf_sampler` `random_trees_embedding` | random Fourier features cho phép phương pháp tâm-cụm tách được cụm không cầu |
| Routing | `select` `discard` `select_type` | chọn cột trước bước không tự phân biệt được kiểu |

**Cố ý không có**: `SelectKBest`, `PoissonInclusion`, `Target*` (cần target);
`PredClipper` (bọc regressor); `PolynomialExtender` (nhân số chiều lên và bóp
méo đúng cái khoảng cách mà clusterer dựa vào).

### Ghép chuỗi

Pipeline hữu ích hiếm khi chỉ một bước. Ghép bằng `|`, chạy trái sang phải:

```python
Setting("DBSTREAM", scaler="impute|one_hot_categorical|standard")
Setting("DBSTREAM", scaler="variance_threshold|sparse_projection|standard",
        scaler_params={"sparse_projection": {"n_components": 16}})
```

Một bước thì tham số phẳng; **chuỗi thì tham số phải lồng theo tên bước**, vì hai
bước trong cùng chuỗi dễ trùng tên tham số (`seed`). Truyền phẳng vào chuỗi sẽ
báo lỗi thay vì đoán.

### ⚠ Bẫy `one_hot`

`OneHotEncoder` của river mã hoá **mọi key** nó nhận. Trên bản ghi lẫn số và chữ,
nó biến cột số thành một indicator cho *mỗi giá trị từng thấy*:

```
one_hot             -> {'dur_1.0': 0, 'dur_3.0': 0, 'dur_5.0': 0, 'dur_7.0': 1, ...}
one_hot_categorical -> {'dur': 7.0, 'proto_tcp': 0, 'proto_udp': 1}
```

Với `duration` của KDDCup99 thì đó là hàng chục nghìn cột, mỗi cột một quan sát,
và mọi khoảng cách sụp về gần như nhau. Dùng `one_hot_categorical` — nó chỉ mã
hoá giá trị không phải số và cho số đi thẳng qua.

### KDDCup99 với cột nominal

Giờ có encoder rồi thì 3 cột nominal dùng được:

```python
ds = load_kddcup99(n_samples=50_000, include_nominal=True)   # 37 features
Setting("DBSTREAM", scaler="one_hot_categorical|standard")
```

Đo thử ở 20k mẫu: 34 cột liên tục cho ARI 0.0236, thêm 3 cột nominal cho 0.0105 —
**tệ hơn**. Lý do hợp lý là `service` có 60+ mức nên sinh 60+ chiều indicator, sau
chuẩn hoá chúng lấn át 34 chiều liên tục. Tức là có encoder không có nghĩa là cứ
bật lên; nó chỉ biến một câu hỏi không trả lời được thành một câu hỏi đo được.

Scaler nhận **tham số riêng** qua `Setting(..., scaler_params={...})`, và tham số đó
được ghi vào kết quả cùng với params của thuật toán. Điểm khác nhau thật sự giữa các
scaler là *nhớ bao nhiêu* — yếu tố quyết định trên stream không dừng:

| Scaler | Thống kê | Quên dữ liệu cũ? |
|---|---|---|
| `standard` | mean/var Welford; `window_size=N` → rolling | không, trừ khi đặt `window_size` |
| `minmax` / `maxabs` | min/max (hoặc rolling min/max) | không, trừ khi đặt `window_size` |
| `robust` | median + IQR (`q_inf`, `q_sup`) | không |
| `adaptive_standard` | EWMean + EWVar (`fading_factor`) | **có** |
| `normalizer` | chuẩn hoá L-p theo từng dòng, không có state | — |

```python
Setting("DBSTREAM", scaler="standard", scaler_params={"window_size": 10_000})
Setting("DBSTREAM", scaler="adaptive_standard", scaler_params={"fading_factor": 0.01})
```

Scaler ảnh hưởng rất lớn: KDD99 có đặc trưng dải giá trị cực rộng, `minmax` dễ ép
mọi điểm về gần nhau khiến model gộp còn 1 cluster. `standard` luỹ kế cũng có điểm
yếu riêng trên KDD99 — một outlier khổng lồ thổi phồng variance vĩnh viễn — nên đáng
đối chứng với biến thể có `window_size` hoặc với `adaptive_standard`. Bán kính
(`epsilon` của DenStream, `clustering_threshold` của DBSTREAM) phải khớp với thang
dữ liệu sau khi scale.

## Chạy

Một entry point duy nhất: `run_benchmark.py`. Mọi thứ điều khiển qua CLI —
framework **không chọn tham số hay dataset thay anh**, mặc định chỉ là chỗ khởi
động chạy được.

```bash
pip install -r requirements.txt   # river được import từ cây nguồn ../river

# mặc định: synthetic stream, nhóm thuật toán phân hoạch mẫu, không preprocessing
python run_benchmark.py

# tham số của anh, dạng JSON
python run_benchmark.py --dataset kdd99 --n-samples 50000     --algorithms DenStream,DBSTREAM     --params '{"DenStream": {"epsilon": 0.5, "mu": 2.5}}'     --scalers standard

# so sánh chuỗi preprocessing trên cùng một thuật toán
python run_benchmark.py --algorithms DBSTREAM     --scalers 'standard,standard_safe,variance_threshold|standard'

# text stream + clusterer ăn token
python run_benchmark.py --dataset sms_spam --algorithms TextClust     --scalers bag_of_words --scaler-params '{"bag_of_words": {"on": "body"}}'

# nhiều seed -> mean ± std; pin hash seed để số tái lập được (RIVER_BUGS.md BUG-2)
PYTHONHASHSEED=0 python run_benchmark.py --dataset insects --seeds 1,2,3
```

Kết quả ghi vào `benchmark_out/<dataset>/`: `report.md`, `results.json` (đầy đủ
quỹ đạo window + checkpoint), và các biểu đồ PNG.

`--help` liệt kê đủ cờ. Xem [STRUCTURE.md](STRUCTURE.md) để biết chức năng từng
module và cách thêm dataset của riêng anh.

Hai phép quét scalability (`sweep_samples`, `sweep_features`) dùng bằng code:

```python
from benchmark.experiments import Setting, sweep_features, scalability_table
sweep = sweep_features([Setting("DBSTREAM", scaler="standard")], dims=(2, 10, 50, 100))
print(scalability_table(sweep, axis_label="#features"))
```

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
`evaluation/` · `reports/` · `experiments/`. Xem `STRUCTURE.md` để biết cách dùng từng module, và `run_benchmark.py` để chạy thử có
giải thích từng bước, và `benchmark/README.md` cho chi tiết API.
