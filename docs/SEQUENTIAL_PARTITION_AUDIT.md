# Audit phân hoạch tuần tự trong một cửa sổ (04/11/2026)

Tín hiệu chọn mô hình của bộ tune là silhouette cổ điển tính trên phân hoạch tuần tự: điểm thứ i của
cửa sổ được gán nhãn bởi mô hình đã học xong các điểm 0..i-1, rồi mới học điểm i (predict-before-learn).
ARI báo cáo (`lag_ari_win`) cũng đọc đúng phân hoạch đó. Phân hoạch này chỉ có nghĩa như ta đọc nó khi
một nhãn gọi tên cùng một cụm từ điểm đầu đến điểm cuối cửa sổ. Tài liệu này ghi lại ba điều đã kiểm:
nhãn có bị đánh số lại hay dùng lại trong một cửa sổ không, `predict_one` có đổi trạng thái mô hình
không, và nhiễu, `None`, sentinel, fallback được xử lý thế nào. Đây là bằng chứng chẩn đoán, không phải
kết quả của chiến dịch; không có phép relabel nào được áp vào dự đoán.

## 1. Cách đánh số nhãn trong mã River

Đọc trên cây `../river` ở commit `873fa6e0` (river 0.26.1), không sửa gì trong đó.

| Thuật toán | Nhãn trả về là gì | Nhãn có giữ nghĩa qua `learn_one` không |
|---|---|---|
| DenStream | chỉ số macro-cluster do DBSCAN trong `predict_one` đánh từ 0, theo thứ tự `p_micro_clusters` (`denstream.py:351`) | Không. Mỗi lần gọi đánh số lại; một p-micro-cluster đứng trước bị xoá thì mọi cụm sau nó lùi một số (test `test_denstream_renumbers_its_macro_clusters_when_an_earlier_p_micro_cluster_goes`) |
| BIRCH | khoá của `dict(enumerate(features))` sau mỗi lần phân cụm lại (`birch.py:392`) | Không. Đánh lại từ 0 mỗi khi phân cụm toàn cục cũ đi, tức sau mỗi lần học |
| DBSTREAM | khoá do `_generate_clusters_from_labels` dựng ở mỗi lần phân cụm lại (`dbstream.py:384-389`) | Không. Đánh lại ở mỗi lần phân cụm lại |
| CluStream | nhãn của một `KMeans` mới dựng trên tâm các micro-cluster sau mỗi `time_gap` điểm (`clustream.py:251-259`) | Không có cam kết. Mỗi `time_gap` (100 điểm, cố định trong không gian tìm kiếm) là một k-means mới, nhãn của nó không nối với nhãn lần trước |
| HPStream | khoá của `_clusters`; cụm mới nhận nhãn tự do nhỏ nhất (`_free_label`, `hpstream.py:620-646`) | Không. Nhãn của cụm vừa bị loại được trao thẳng cho cụm mới, theo chú thích trong mã |
| SOStream | khoá micro-cluster từ bộ đếm tăng dần `_next_id` (`sostream.py:193-195`) | Có: một nhãn không bao giờ được dùng lại. Gộp cụm làm nhãn cũ biến mất |
| RepStream | `vertex.cluster`, cấp từ bộ đếm tăng dần `_next_cluster` (`repstream.py:392-393, 469-472`) | Có: không dùng lại. Gộp cụm chuyển các đỉnh sang nhãn của cụm còn lại (`repstream.py:359`) |
| STREAMKMeans | khoá tâm của một `KMeans` dùng suốt run (`streamkmeans.py:107`) | Có: cùng một k-means tăng dần, tâm giữ khoá |

Như vậy năm thuật toán (DenStream, BIRCH, DBSTREAM, CluStream, HPStream) có cơ chế cụ thể trong mã
để một nhãn đổi nghĩa giữa hai điểm của cùng cửa sổ. Ba thuật toán còn lại có danh tính cụm bền, và ở
chúng một nhãn đổi phân hoạch chỉ khi cụm thật sự gộp hay tách.

## 2. `predict_one` có đổi trạng thái không

- **DenStream (river):** có. `predict_one` ghi lại `linear_sum`, `squared_sum` của các p-micro-cluster
  sống khi hai cái chung một nhãn DBSCAN (RIVER_BUGS.md BUG-7). Trong một lần thử nhanh (ba khối Gauss,
  `decaying_factor=0.25`, không lưu thành test), 49/50 lần gọi đổi `p_micro_clusters`. Arm `DenStream` của chiến dịch là bản river nguyên gốc, như
  `out/full`; bản sửa `DenStreamPatchedFast` là biến thể đặt tên riêng, không thay nó.
- **CluStream:** `predict_one` chỉ ghi bộ nhớ đệm `_center` của micro-cluster (`clustream.py:298-301`),
  không đổi hành vi.
- **BIRCH, DBSTREAM:** `predict_one` dựng lại phân cụm toàn cục khi nó đã cũ và lưu vào bộ nhớ đệm
  (`_clusters`, `_centers`); trạng thái học không đổi.
- Trên mọi ô đo ở mục 4, `n_predict_mutations = 0`: không lần dự đoán nào làm mô hình gán nhãn khác
  cho tập dò ngay sau đó. Đếm `n_state_writes` (trạng thái pickle đổi, trừ bộ nhớ đệm đã biết) là cận
  trên: CluStream 180/200 trên cả bốn luồng (bộ nhớ đệm `_center`), DenStream 183–200/200 trên ba luồng
  và 0 trên shuttle.

Hệ quả cho chẩn đoán: `stability` và silhouette snapshot dự đoán lại cửa sổ trên **một bản sao**
(`deepcopy`) của member. Bản gốc, dòng số của bộ tune và các bộ sinh số toàn cục của Python và NumPy
không bị đụng (test `test_diagnostics_isolate_the_global_generators_and_predict_on_a_copy`). Nhưng với
DenStream, chính bản sao bị `predict_one` ghi trong lúc dự đoán lại, nên snapshot cuối cửa sổ của nó
phụ thuộc thứ tự các điểm được dự đoán lại. Snapshot vì thế là một cách đọc chẩn đoán, không phải phân
hoạch "đúng" của mô hình tại cuối cửa sổ.

## 3. Nhiễu, `None`, sentinel và fallback

- **`None`:** adapter trả `None` khi mô hình không dự đoán được (`NotImplementedError`, từ đó trở đi
  cho cả run). Silhouette bỏ điểm đó ra khỏi phép tính; ARI cũng bỏ qua. Số điểm bị bỏ được ghi
  (`n_none`, `n_points_scored`).
- **Sentinel:** DenStream trả `-1` khi chưa có cụm (BUG-6). Mặc định nhãn này được giữ như một cụm
  (`drop_sentinel_predictions` tắt), cả trong silhouette lẫn ARI; số dự đoán sentinel được đếm
  (`n_sentinel`, và `n_member_windows_sentinel` trong dòng tóm tắt). Các tín hiệu chẩn đoán khác đọc
  nó theo `--signal-noise label`.
- **Fallback:** `ValueError`, `ZeroDivisionError`, `KeyError` trong `predict_one` được thay bằng nhãn 0
  và đếm (`n_fallback`, `n_member_windows_fallback`).
- **Nhãn nhiễu hợp lệ và điểm thật sự bị loại.** Sentinel `-1` và nhãn fallback `0` là nhãn: silhouette
  và ARI/NMI đều đọc điểm đó như thuộc một cụm. Điểm chỉ thật sự bị loại khi dự đoán là `None` (khỏi
  silhouette và ARI/NMI) hoặc khi nhãn thật là `None` (khỏi ARI/NMI; `runner.run_once` chỉ cập nhật độ đo
  ngoài khi cả hai khác `None`). CMM thì khác: nó đặt điểm sentinel ra ngoài mọi cụm. Chính sách này không
  đổi trong vòng này.
- **Tập điểm được chấm, ghi từ 04/11.** Mỗi member-window ghi `n_points_input`, `n_points_scored` và
  `support_sha` (vị trí các điểm silhouette đọc, `window.silhouette_support`), `n_points_external` và
  `external_support_sha` (vị trí các điểm ARI/NMI đọc, `window.external_support`). Dấu vân tay là SHA-256
  của mặt nạ vị trí kèm độ dài cửa sổ (`window.support_fingerprint`), giống nhau ở mọi tiến trình. Dòng cửa
  sổ ghi các trường đó cho incumbent và incumbent trễ (tiền tố `lag_`), cạnh `start`, `end`. Test
  `tests/test_point_support.py` kiểm rằng mặt nạ đúng là tập điểm ARI của `run_once` đọc (có `None`,
  sentinel, fallback, nhãn thật thiếu), rằng hai mặt nạ cùng số điểm khác vị trí cho dấu vân tay khác
  nhau, và smoke `conf_pipeline/out/mech_smoke_20261104_v7` cho thấy việc ghi không đổi dự đoán, điểm số
  hay quyết định nào (so với smoke `_v4` chạy trên code trước khi có các trường này: 22/22 job trùng).
- **Lý do điểm số của member:** mỗi member-window có `score_status`, tách các trường hợp mà trước đây
  cùng hiện thành điểm −1: `valid`; `member_error`; `infeasible_memory`, `infeasible_micro_clusters`,
  `infeasible_runtime`; `undefined_insufficient_points`, `undefined_single_label`,
  `undefined_one_label_per_point` (phân hoạch không có silhouette); `signal_undefined`. Silhouette
  thô trước mọi giới hạn nằm ở `silhouette_raw` và `silhouette_reason`. `score_status` cho giới hạn
  đứng trước phân hoạch không xác định, nên tính hợp lệ của phân hoạch không đọc từ nó: dòng tóm tắt
  của run đếm `member_silhouette_reason_counts` (lý do của mọi member-window, bất kể giới hạn) và
  `member_reason_feasibility_counts` (lý do kèm giới hạn đã loại, ví dụ `valid|memory`), và bảng
  coverage tính ba tỉ lệ riêng từ đó. Một job hết giờ, hết bộ nhớ
  hay lỗi có trạng thái riêng trong `runs.jsonl` (`timeout`, `oom`, `error`) và không bao giờ thành
  điểm chất lượng.

## 4. Đo trên cấu hình khởi đầu

`conf_pipeline.partition_audit` dựng một member như `run.py` dựng (mặc định của River, hoặc
`scaled_default`), cho nó học cửa sổ đầu, rồi phát lại từng cửa sổ sau theo predict-before-learn trên
một bản sao. Cứ 10 điểm, nó dự đoán lại cả cửa sổ bằng một bản sao khác và so hai lần gán nhãn liên
tiếp, không relabel gì:

- **đổi tên:** cùng phân hoạch dưới nhãn khác (ARI = 1); **đổi tên dùng lại nhãn:** trong đó có một
  nhãn có mặt ở cả hai lần nhưng gọi tập điểm khác. Đây là trường hợp phân hoạch tuần tự gộp hai cụm
  mà mô hình chưa từng gộp.
- **đổi phân hoạch:** ARI < 1; trong đó **có đánh số lại** khi phép ghép một-một tốt nhất (Hungarian)
  giữ được nhiều điểm hơn so khớp nhãn thô. Với SOStream và RepStream, vốn không dùng lại nhãn, con số
  này đến từ gộp và tách cụm, nên chỉ là cận trên của việc đổi tên.

Lệnh (h = 1 000, 2 cửa sổ sau cửa sổ khởi động, dò mỗi 10 điểm, data seed 42, model seed 201):

```bash
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.partition_audit \
    --folds static_overlap,scale_drift,electricity,shuttle --h 1000 --windows 2 --every 10 \
    --out result/partition_audit_2026-11-04
```

Kết quả `result/partition_audit_2026-11-04/audit.csv`, cấu hình mặc định, cộng trên 4 luồng × 2 cửa
sổ = 800 khoảng dò mỗi thuật toán:

| Thuật toán | đổi tên | dùng lại nhãn | đổi phân hoạch | có đánh số lại | ARI tuần tự–snapshot (TB) | số nhãn TB |
|---|---:|---:|---:|---:|---:|---:|
| BIRCH | 3 | 3 | 657 | 559 | 0,245 | 309 |
| CluStream | 11 | 7 | 205 | 40 | 0,164 | 3,6 |
| DBSTREAM | 8 | 4 | 365 | 120 | 0,559 | 6,0 |
| DenStream | 5 | 0 | 95 | 7 | 0,677 | 3,3 |
| HPStream | 4 | 0 | 792 | 684 | 0,071 | 10,0 |
| RepStream | 1 | 0 | 794 | 591 | 0,278 | 169 |
| SOStream | 9 | 0 | 589 | 199 | 0,345 | 120 |
| STREAMKMeans | 0 | 0 | 108 | 0 | 0,978 | 1,6 |

`scaled_default` cho cùng bảng với HPStream và RepStream (không có tham số khoảng cách) và với BIRCH
(ngưỡng 0 nhân s vẫn là 0). Ở các thuật toán còn lại: CluStream 8 đổi tên, 3 dùng lại nhãn; DBSTREAM 6 và
2; DenStream 4 và 0, với số nhãn trung bình 25 thay vì 3,3; SOStream và STREAMKMeans gần như không đổi. Theo luồng, HPStream có đánh số lại ở 184–199 trên 200 khoảng dò của electricity,
scale_drift và static_overlap; DBSTREAM gần như không đổi gì trên electricity và static_overlap nhưng
đổi phân hoạch ở cả 200 khoảng dò của shuttle.

## 5. Điều kết luận được và điều còn mở

- Có bằng chứng cụ thể (mã và đo) rằng trong một cửa sổ, nhãn tuần tự của BIRCH, CluStream và DBSTREAM
  đôi khi là cùng một phân hoạch dưới nhãn hoán vị, với nhãn được dùng lại (3, 7 và 4 lần trên 800
  khoảng dò). Đó là trường hợp phân hoạch tuần tự gộp hai cụm khác nhau dưới một nhãn.
- HPStream dùng lại nhãn theo thiết kế, và ở cấu hình mặc định phân hoạch của nó đổi ở gần mọi khoảng
  dò kèm đánh số lại; phân hoạch tuần tự gần như không tương quan với snapshot cuối cửa sổ (ARI 0,07).
  Cấu hình mặc định của HPStream vốn suy biến trên các luồng này, nên con số không nói gì về cấu hình
  đã tune.
- Stability thấp không được đọc là lỗi nhãn: RepStream và SOStream có ARI tuần tự–snapshot thấp mà
  không dùng lại nhãn nào; phân hoạch của chúng thay đổi thật.
- **Đã sửa cho HPStream, chưa sửa cho bốn thuật toán còn lại** (mục 6). Với DenStream, BIRCH, DBSTREAM,
  CluStream, River dựng lại các macro-cluster từ đầu và không giữ danh tính nào qua các lần dựng; giữ nhãn
  ổn định sẽ cần một quy tắc đặt tên cho gộp và tách, hoặc ghép cụm giữa các lần dựng, tức một heuristic
  relabel, mà vòng này không coi là lời giải. Riêng HPStream, `HPStreamStableLabels` cấp nhãn từ một bộ
  đếm tăng dần thay cho `_free_label`; nó đã được kiểm bằng test và không thay HPStream của chiến dịch.
- **Phạm vi của phép đo:** chỉ cấu hình khởi đầu, 4 luồng, 2 cửa sổ, dò mỗi 10 điểm (một sự kiện có
  thể gộp nhiều thay đổi). Chưa đo trên cấu hình mà bộ tune chọn, và chưa đo ảnh hưởng lên quyết định
  chọn. Chiến dịch ghi `stability`, `snap_label_agreement` và silhouette snapshot cho mọi member-window,
  nên mức độ của hiện tượng dưới cấu hình đã tune sẽ đọc được từ chính log của nó.
- **Điểm chặn:** việc diễn giải silhouette tuần tự và ARI của năm thuật toán trên, khi chạy chiến dịch
  mà không sửa nhãn, là quyết định nghiên cứu của người dùng. Code của chiến dịch không phụ thuộc vào
  quyết định này; cách đọc kết quả thì có. Mô tả cho upstream ở RIVER_BUGS.md GAP-6.

## 6. Vòng thứ hai (04/11/2026): ba trường hợp, counterexample theo thuật toán, và HPStream

**Ba trường hợp được tách riêng.**

1. *Nhãn đổi giữa hai cửa sổ.* Silhouette tuần tự và `lag_ari_win` đọc từng cửa sổ riêng, nên một cụm
   mang nhãn khác ở cửa sổ sau không làm sai số đọc nào của chiến dịch. Chỉ `lag_ari_cumulative` của báo
   cáo chung ghép nhãn qua mọi cửa sổ (và qua các lần đổi incumbent); nó không nhất quán về danh tính với
   bất kỳ thuật toán nào, và `report --contrasts` hay `selection --lobo` không đọc nó.
2. *Nhãn hoán vị hay dùng lại trong một cửa sổ.* Đây là trường hợp làm phân hoạch tuần tự ghép những điểm
   mô hình chưa từng gộp. Phép đọc chính xác, không ghép cặp nào, là `partition_audit.label_handoffs`: giữa
   hai lần dò, một nhãn chuyển sang các điểm rời hẳn với các điểm nó gọi tên trước đó, trong khi các điểm cũ
   vẫn ở chung dưới một nhãn khác. Gộp cụm, tách cụm hay nhãn mới không bao giờ là một handoff.
3. *Phân hoạch đổi thật* (học, gộp, tách, xoá, tạo cụm): đây là điều phân hoạch tuần tự cần ghi lại.

Một cụm dịch chuyển trọn vẹn sang các điểm khác giữa hai lần dò cũng cho một handoff. Vì vậy ở thuật toán có
danh tính cụm bền, handoff là cận trên của chuyển động, không phải lỗi nhãn; còn ở thuật toán không có danh
tính, nó là bằng chứng chứ không phải chứng minh. Stability thấp hay ghép Hungarian tốt hơn (`n_renumbering_events`)
không được dùng làm bằng chứng của trường hợp 2.

**Counterexample nhỏ trên chính lớp của River** (không sửa gì trong `../river`):

| Thuật toán | Test | Điều xảy ra trong một cửa sổ |
|---|---|---|
| HPStream | `test_river_hpstream_hands_the_evicted_label_to_the_newcomer_within_one_window` | ba điểm của khối ở 0 nhận nhãn 0; điểm đầu tiên ở 30 mở cụm mới, cụm của khối ở 0 (cũ nhất) bị loại và nhãn 0 trao cho cụm mới; hai điểm sau ở 30 cũng nhận nhãn 0 |
| DenStream | `test_denstream_prunes_an_earlier_p_micro_cluster_and_the_clusters_that_stay_change_label` | `learn_one` xoá p-micro-cluster ở 0; p-micro-cluster ở 10 và 20 là cùng đối tượng trước và sau, nhưng nhãn 1 chuyển từ cụm ở 10 sang cụm ở 20 |
| DBSTREAM | `test_dbstream_cleans_up_a_micro_cluster_and_the_macro_clusters_that_stay_change_label` | dọn một micro-cluster làm các macro-cluster còn lại lùi số; nhãn 1 và 2 chuyển sang cụm khác |
| BIRCH | `test_birch_under_rivers_defaults_renames_its_clusters_with_the_partition_unchanged` | với mặc định của River (`threshold = 0`, `leaf_capacity = 50`), phân hoạch của tập dò không đổi (ARI 1) mà một nhãn chuyển sang điểm khác |
| CluStream | `test_clustream_names_its_macro_clusters_afresh_at_every_offline_k_means` | ở một mốc `time_gap`, k-means mới giữ cụm ở 0 nguyên vẹn dưới nhãn 0, còn nhãn 2 của nó chuyển sang cụm ở 10 |
| SOStream | `test_sostream_never_gives_one_label_to_two_micro_clusters` | đối chứng: qua các lần gộp, một nhãn không bao giờ gọi tên đối tượng micro-cluster thứ hai |

**HPStream có cách sửa chính xác.** `HPStreamStableLabels` (`benchmark/algorithms/hpstream_stable_labels.py`)
chỉ thay `_free_label` bằng một bộ đếm tăng dần, bắt đầu từ `max_clusters`. Theo đọc mã, khoá của
`_clusters` chỉ là tên: mọi phép duyệt (`_closest`, `_compute_dimensions` với `sort` ổn định chỉ theo bán
kính, `min` theo `last_seen` khi loại, quét cụm hết chiều) đi theo thứ tự chèn, và cụm mới luôn được chèn
cuối ở cả hai cách. `tests/test_hpstream_stable_labels.py` kiểm điều đó từng điểm trên ba cấu hình (lành;
suy biến, gần như mọi điểm mở cụm và loại một cụm; có `radius_threshold`, cụm bị bỏ đói và xoá): cùng các
cụm với cùng tổng, trọng số, thời điểm và chiều theo cùng thứ tự, cùng cụm được dự đoán ở mọi điểm, và không
nhãn nào gọi tên hai đối tượng cụm, trong khi lớp của River có trao nhãn. Trên cấu hình khởi đầu của
protocol, mỗi nhãn của biến thể ứng với đúng một nhãn của River trong một cửa sổ: biến thể chỉ tách những
gì nhãn dùng lại đã gộp. HPStream không gộp hay tách cụm online; tạo, loại và bỏ đói đều được test phủ. Biến
thể nằm trong `_common.LABEL_VARIANTS` (không bao giờ xếp hạng chung trong các bảng OTB, không vào preset nào
của notebook; LOBO và các hiệu đọc nó dưới tên riêng). Theo quyết định của người dùng ngày 04/11, chiến dịch
cơ chế chạy nó thay cho HPStream (`decisions` trong `mechanism_pilot.json`);
`HPStream` của River giữ nguyên.

**Bốn thuật toán còn lại chưa có cách sửa chính xác.** DenStream, DBSTREAM, BIRCH và CluStream dựng lại
macro-cluster từ đầu; không có danh tính nào trong mô hình để giữ, và đặt tên cho chúng cần một quy tắc
cho gộp và tách. Đó là quyết định thiết kế, không phải sửa lỗi, nên chưa làm.

**Đo lại cùng các ô của mục 4** (`result/partition_audit_2026-11-04_handoffs/`, chỉ cấu hình `default`,
thêm cột `n_label_handoff_events` và biến thể HPStream; các cột cũ trùng tuyệt đối với lần đo đầu):

| Thuật toán | handoff (trên 800 khoảng dò) | điểm dò bị nhãn chuyển giao (tổng) | số nhãn TB |
|---|---:|---:|---:|
| BIRCH | 540 | 280 361 | 309 |
| HPStream | 288 | 36 539 | 10 |
| HPStreamStableLabels | 35 | 4 865 | 376 |
| DBSTREAM | 32 | 8 293 | 6,0 |
| CluStream | 16 | 7 908 | 3,6 |
| RepStream | 4 | 283 | 169 |
| SOStream | 4 | 757 | 120 |
| DenStream | 0 | 0 | 3,3 |
| STREAMKMeans | 0 | 0 | 1,6 |

Biến thể HPStream giảm handoff từ 288 xuống 35; 35 handoff còn lại (31 trên electricity) là cụm dịch chuyển
thật, vì biến thể không bao giờ dùng lại nhãn, như 4 của RepStream và 4 của SOStream. DenStream không có
handoff nào trên bốn luồng này ở cấu hình mặc định, dù cơ chế có thật (counterexample); phạm vi dưới cấu
hình đã tune chưa được đo. Lệnh:

```bash
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.partition_audit \
    --algos BIRCH,CluStream,DBSTREAM,DenStream,HPStream,HPStreamStableLabels,RepStream,SOStream,STREAMKMeans \
    --folds static_overlap,scale_drift,electricity,shuttle --arms default --h 1000 --windows 2 --every 10 \
    --out result/partition_audit_<ngày mới>
```

**Trạng thái đo.** `conf_pipeline.campaign` ghi trạng thái của từng thuật toán vào dry run và manifest
(`measurement_status`). DenStream, BIRCH, DBSTREAM và CluStream là `open`, nên chiến dịch không được đánh dấu đủ điều kiện làm
bằng chứng xác nhận paper. Không thuật toán nào bị bỏ khỏi thiết kế, và dry run cùng chẩn đoán vẫn chạy.
Với HPStream điều kiện đã được gỡ: người dùng quyết định 04/11 đọc nó qua `HPStreamStableLabels`, chạy thay
nó trong chiến dịch, nên còn bốn thuật toán `open`. Với bốn thuật toán đó, một quy tắc đặt tên do người dùng chọn,
đăng ký thành biến thể và có test, hoặc báo cáo kèm giới hạn.
