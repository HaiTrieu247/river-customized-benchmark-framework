# Kế hoạch thử chuẩn hóa theo giai đoạn trong conf_pipeline

Ngày: 24/10/2026. Trạng thái: đề xuất thiết kế và triển khai; chưa chạy thí nghiệm.
Các giá trị pilot bên dưới là điểm khởi đầu có thể cấu hình, không phải thông số tối ưu.

## 1. Câu hỏi và phạm vi

Kiểm tra liệu cập nhật scaler từ dữ liệu gần đây có cải thiện clustering khi phân
phối thay đổi, vượt lợi ích của riêng việc restart model hay không.

Giai đoạn đầu dùng lịch định kỳ, không dùng drift detector, không biết trước
drift points và không dùng labels để quyết định refresh. Chưa chạy hai ensembles
song song để chuyển tiếp: chi phí và quy tắc chọn giữa hai ensembles sẽ thành
can thiệp khác. Giữ raw baseline và không ghi đè kết quả paper hiện có.

## 2. Các đối chứng bắt buộc

| Policy đề xuất | Scaler | Model history | Vai trò |
|---|---|---|---|
| `raw_continuous` | Không scale | Giữ liên tục | Baseline representation |
| `prefix_continuous` | Fit prefix một lần, đóng băng | Giữ liên tục | Lợi ích chuẩn hóa ban đầu |
| `prefix_restart` | Cùng scaler prefix, đóng băng | Restart định kỳ | Đối chứng renewal |
| `stage_restart` | Fit lại trên buffer gần đây | Restart cùng lịch | Can thiệp cần kiểm tra |

Ba comparisons chính:

1. `prefix_continuous - raw_continuous`: thay representation ban đầu.
2. `prefix_restart - prefix_continuous`: renewal với representation giữ nguyên.
3. **`stage_restart - prefix_restart`: lợi ích thêm của scaler refresh khi đã có renewal.**

Không có policy refresh-scaler/giữ-model-cũ trong phép so chính: nó trộn hệ tọa độ.
Các differences là hiệu ứng trong thiết kế này, không phải phân rã nhân quả phổ quát.
Mọi policy dùng cùng stream order, evaluation windows và seed settings tương ứng.

## 3. Lịch xử lý chính xác, tránh nhìn tương lai

Đề xuất smoke/pilot đầu: h = 1.000; calibration ban đầu = 1 window;
buffer fit scaler = tối đa 5 windows raw; warm-start = window raw gần nhất;
refresh sau mỗi 10 windows đã xử lý kể từ calibration. Biên giai đoạn căn theo
window; chưa thay h hoặc ensemble size trong cùng experiment.

```text
B0: học scaler ban đầu từ x; transform B0; warm-start models; không chấm B0.
B1..B10: scaler S0 giữ cố định; mỗi điểm predict-then-learn trong models.
Cuối B10: hoàn tất scoring/selection, ghi kết quả B10 trước khi refresh.
         S1 = fit(x của 5 windows quá khứ gần nhất), chỉ ở stage_restart.
         Tạo model copies mới, warm-start bằng transform(B10, S1).
         prefix_restart làm cùng việc với S0.
B11: điểm mới được dự đoán bởi models mới, chưa từng học B11.
B20: refresh tiếp theo theo cùng quy tắc; không căn theo drift annotations.
```

Refresh diễn ra sau scoring của B_t và trước khi nhìn B_(t+1). Không fit trên
B_(t+1) rồi dùng để báo cáo dự đoán B_(t+1). Việc replay B_t là warm-start;
không tính những predictions lúc replay như quan sát test mới.

Mọi policy giữ buffer raw giống nhau để có thể so chi phí bộ nhớ công bằng;
tách thêm chi phí riêng thực hiện fit và warm-start. Buffer có giới hạn kích thước.

## 4. Scaler và representation

- Standardization theo từng numeric feature: mean/std population từ fit buffer.
  Feature có std = 0 dùng divisor = 1, ghi rõ; không tự xóa feature.
- Không clip, PCA, impute hoặc đổi feature set trong experiment đầu. NaN/Inf,
  schema không hợp lệ phải báo lỗi/diagnostic theo policy khai báo; không âm thầm
  thay bằng 0. Missing-data policy là experiment tiếp theo nếu dữ liệu đòi hỏi.
- Feature schema cố định trong pilot. Dynamic schema chưa hỗ trợ thì fail rõ.
- Fit/transform tuyệt đối không nhận y. Labels chỉ đi qua nhánh evaluation.
- Một fitted scaler dùng chung cho tất cả members/challengers trong job và cho
  các algorithms/regimes có cùng data/calibration/epoch policy; không fit riêng
  theo tuổi hoặc lịch sử từng member. Mỗi lần transform tạo bản sao an toàn.
- Model, silhouette, DBCV và Table A dùng cùng transformed representation.
  Không gắn thêm online scaler trong adapter để tránh transform hai lần.
- ARI/NMI dùng predicted labels và y giữ nguyên. CMM phụ thuộc geometry: ghi rõ
  representation; nếu dùng, thêm diagnostic ở anchor representation cố định,
  không diễn giải thay đổi CMM giữa hai geometries như cùng một thước đo.
- `silhouette_std` sẵn có là signal chuẩn hóa lại theo window, không phải policy
  preprocessing này; giữ raw silhouette trên model-space trong comparison đầu.

## 5. Giữ tham số và cơ chế tune có thể so sánh

Thực hiện theo hai bước để không trộn preprocessing với tuner restart.

### Bước A: một fixed configuration mỗi algorithm

Giữ constructor parameters giống nhau giữa hai restart policies và giữa các
epochs. Scaler thay đổi, numerical parameters không đổi. So đủ bốn policies.
Chọn configurations từ protocol khai báo trước, không chọn tốt nhất bằng labels
của evaluation suffix. Đây là mechanism check, không nhằm chứng minh best-static.

### Bước B: tích hợp confstream

- Đầu tiên chỉ dùng `confstream`, cùng E/C và search space giữa policies.
- Khi restart, rebuild models tại chính constructor parameters đang có; giữ
  candidate roles, sampler states và tuner RNG. Không đồng thời reset tìm kiếm
  về defaults. Xử lý cả active members và pending challengers theo cùng chính sách.
- Sau restart không mang score cũ sang cạnh tranh trong geometry mới. Incumbent
  cho B_(t+1) là copy mới của configuration đã chọn ở B_t; không chọn bằng labels
  hoặc score của B_(t+1). Sau B_(t+1) mới cập nhật scores theo protocol thường.
- Log logical member ID + instance ID/epoch ID để một configuration được rebuild
  không bị nhầm là cùng model state. Model seed policy giống nhau giữa hai controls.
- Để cô lập scaler refresh, giữ distance unit s của search space cố định từ
  initial calibration trong từng representation; không tự re-estimate s mỗi epoch
  hoặc remap params theo s mới. Những thay đổi này có thể thử riêng về sau.
- Raw và standardized initial s khác nhau là thành phần của initial representation
  condition; comparison trọng tâm stage_restart/prefix_restart có cùng initial s.
- Nếu thử re-tune từ defaults mỗi epoch, gọi thành policy khác và chạy cả hai
  restart controls tương ứng, không ghép với preserve-search policy ở trên.

`frozen_reinit` hiện có spawn copies mỗi window sau freeze; không tương đương
periodic restart này. Có thể reuse construction/warm-start primitives, không dùng
arm đó như đối chứng đã khớp lịch. Một fixed configuration restart định kỳ phải
được gọi fixed-parameter renewal, không gọi là continuous best-static reference.

## 6. Phần cần triển khai

| Nơi | Thay đổi dự kiến |
|---|---|
| `benchmark/preprocessing/` (module mới) | Frozen scaler snapshot, raw ring buffer và epoch controller, serializable state |
| `conf_pipeline/run.py` | Giữ raw buffer, transform trước khi giao window cho ensemble; refresh sau scoring; giữ lag semantics |
| `benchmark/tuning/confstream.py` | API rebuild model instances có giữ configs/search state; invalidate scores, map incumbent |
| `conf_pipeline/_common.py` | Khai báo policy/options và seed policy dùng lại |
| `conf_pipeline/manifest.py` | Fingerprint preprocessing, fit/warm buffer, refresh schedule, feature schema, metric geometry |
| `conf_pipeline/report.py` | Paired comparisons, boundary trajectories, cost breakdown và coverage |
| Tests mới | Timing của fit/restart, consistency, repeatability và backward compatibility |

Các options đề xuất: policy, calibration samples, refresh interval, fit-buffer
size, warm-start samples. Đây chưa phải CLI đang có. Tiếp tục dùng runner/pipeline
hiện hữu, không thêm top-level runner, không sửa sibling River.

## 7. Logging và kiểm tra trước khi chạy dài

Window/run logs thêm: preprocessing policy, epoch ID, scaler hash, fit sample
range, refresh reason, model-instance mapping, initial distance unit, fit/transform/
rebuild/warm-start durations, buffer memory, counts invalid/missing/constant features.
Lưu scaler snapshots một lần mỗi epoch, window rows chỉ tham chiếu snapshot.
Hash raw stream riêng với hash transformed representation; không để checksum dữ
liệu nguyên bản thay đổi chỉ vì policy khác.

Các kiểm tra bắt buộc:

1. Sửa suffix tương lai không làm thay scaler/predictions ở trước suffix.
2. Không score lại warm-start buffer; mọi policy có cùng evaluated timestamps.
3. Members/challengers nhận cùng transformed x; không transform hai lần.
4. Model mới không chứa state cũ; configurations/search state giữ đúng như khai báo.
5. Refresh không đổi dự đoán đã ghi của B_t, không chọn incumbent bằng B_(t+1).
6. Cùng seed/protocol, runtime gate off, lặp lại trajectory và scaler hashes.
7. Policy `none`/raw vẫn giữ hành vi cũ trong cùng code version.
8. Manifest từ chối trộn khác preprocessing hoặc buffer schedule trong một out-dir.
9. Chọn boundary không có raw points đủ warm-start phải theo quy tắc công khai,
   không silently drop test windows. Các buffers phải đủ initialization yêu cầu.

## 8. Các lượt chạy

### A. Engineering smoke

Một algorithm, synthetic ngắn và một real stream; đủ ít nhất hai refresh events.
Chạy bốn policies; lặp cùng seed để kiểm tra timing/geometry/repeatability.
E/C nhỏ có thể dùng cho smoke; quality smoke không phải evidence của paper.

### B. Mechanism pilot

Đề xuất subset để khóa trước khi chạy: DenStream, DBSTREAM, BIRCH và CluStream;
streams scale_drift, static_overlap, covtype, insects_abrupt_imbalanced.
CluStream được giữ để đo tác động ở algorithm đắt, dù có thể chạy sau smoke cost.
Đây là subset đề xuất, không thay đổi protocol datasets hiện có.

- Fixed-config step: 4 algorithms × 4 streams × 4 policies = 64 cells, chạy model
  repetitions chỉ khi algorithm có randomness; không lặp deterministic fixed runs.
- Online step: cùng subset × 4 policies × 3 tuner seeds = 192 jobs; model seed
  cố định, calibration seed cố định, dùng seed-split policy mới trong code.
- Giữ E=20,C=10 trong mechanism comparison với baseline hiện tại; nếu budget cần
  nhỏ hơn thì dùng cùng E/C cho cả bốn policies và ghi là subset-budget study.
- Không dùng runtime gate làm thay trajectory; record runtime và resource failures.
- Không chạy thêm signals/CMM đắt nếu chưa cần câu hỏi này; giữ metric settings
  giống nhau giữa policies. Đo cost pilot rồi mới dự toán toàn bộ.

### C. Confirmation và sensitivity

Khóa policy/schedule từ development pilot; xác nhận trên synthetic data seeds mới
và real sources chưa dùng để chọn. Không coi thêm variants cùng INSECTS source là
nhiều nguồn độc lập. Nếu mở rộng claim tới tám algorithms, cần evidence tương ứng.

Sau khi comparison chính xong mới thử intervals 5/10/20 windows và fit buffers
1/5/10 windows. Ghi chúng theo samples; không tối ưu lịch theo drift annotations.
Chưa làm full Cartesian product. Drift-triggered refresh chỉ là bước kế tiếp.

## 9. Cách quyết định thành công

Primary: paired difference tail ARI giữa stage_restart và prefix_restart, kèm
absolute effect và source-block uncertainty; NMI là kiểm tra bổ sung. Giữ primary
evaluation region hiện có (từ t0=20) giữa mọi policies, đồng thời báo cáo toàn bộ
post-calibration trajectory để không giấu chi phí khởi động/restart trước t0.

Secondary: degradation ở các windows ngay sau refresh, common-threshold drift
recovery, defined/failed coverage, end-to-end latency/memory, total compute và
incumbent processing time. Không bỏ cold-start windows ở stage_restart khỏi mean.
Không so trực tiếp silhouette values của hai epochs như cùng geometry.

Không quy định trước phải tăng ARI X%: khóa practical-effect/cost bands trước
confirmation, theo mục tiêu triển khai; công bố denominator và failures.

- Stage hơn prefix-restart với chi phí chấp nhận được: có evidence cho refresh.
- Chỉ restart giúp: ưu tiên renewal, không claim scaler refresh là nguyên nhân.
- Chỉ scale-drift giúp: giới hạn kết luận vào loại drift; xem shape/natural drift.
- Stage thua hoặc gây spikes: giữ prefix-frozen như baseline, báo cáo trade-off.

Đầu ra paper: sơ đồ timeline, bảng bốn policies, plot quanh refresh/drift và bảng
quality–cost; ghi preprocessing là một trục của evaluation bên cạnh tuning regime.

## 10. Thứ tự thực hiện

1. Khóa design/semantics và lưu manifest riêng.
2. Viết preprocessing controller + tests, nối fixed-config experiment.
3. Smoke; kiểm tra không leakage, không double transform và repeatability.
4. Fixed-config pilot; profile fit/warm-start và kiểm tra tín hiệu có đáng mở rộng.
5. Tích hợp preserve-search restart; online pilot cùng budgets/seed settings.
6. Đọc kết quả, khóa phương án, rồi confirmation/sensitivity và cập nhật paper.

Chưa khởi chạy 192 jobs từ tài liệu này. Chưa có kết quả cho thấy stage normalization
tốt hơn prefix; kế hoạch được thiết kế để kiểm tra cả khả năng có và không có lợi ích.
