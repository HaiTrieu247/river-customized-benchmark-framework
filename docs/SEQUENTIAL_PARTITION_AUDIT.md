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

## 7. Vòng thứ ba (08/11/2026): danh tính cụm theo sự kiện, so sánh được theo độ trễ, đánh giá snapshot

Hai vòng trước đếm renaming và handoff giữa các lần dò trên luồng của protocol, và `fidelity` đã có các
kịch bản birth, deletion, merge, split, identity_stress với sự kiện biết trước. Còn thiếu bốn điều: hành vi
của nhãn được tách theo loại thay đổi của mô hình; một phép thử trực tiếp xem hai dự đoán cũ trong cùng cửa sổ
có so sánh được với nhau không; một phép đánh giá riêng cho phân hoạch mạch lạc mà một trạng thái mô hình tạo
ra; và một định nghĩa rõ cho chất lượng gán tuần tự cùng một cách giữ danh tính cụm không đọc nhãn đánh giá.
Phần lõi nằm ở `benchmark/tuning/cluster_identity.py`, driver ở `conf_pipeline/identity_audit.py`, phần trong
pipeline là cờ `run --snapshot-eval`. Không có quyết định nào của bộ tune đổi theo các phép đọc này.

### 7.1 Bốn phân hoạch của một cửa sổ

Với một member chạy predict-before-learn trên cửa sổ t, audit đọc bốn phân hoạch trên cùng các điểm:

- `seq`: nhãn tuần tự mà silhouette của bộ tune và `lag_ari_win` đọc; điểm i được gán bởi mô hình đã học
  xong các điểm 0..i-1.
- `tracked`: cùng các dự đoán đó, nhưng mỗi nhãn thô được thay bằng danh tính mà `IdentityTracker` mang theo
  (mục 7.4).
- `pre`: mọi điểm của cửa sổ được gán bởi đúng một trạng thái, trạng thái của mô hình khi cửa sổ bắt đầu. Đây
  là phân hoạch ngoài mẫu: mô hình chưa học điểm nào của cửa sổ.
- `post`: mọi điểm được gán bởi trạng thái sau khi mô hình đã học xong cửa sổ, tức phân hoạch trong mẫu.

`pre` và `post` không thể dùng lại nhãn vì chúng đến từ một trạng thái. Với DenStream của River, `predict_one`
ghi vào trạng thái (BUG-7), nên `snapshot_labels` khôi phục trạng thái trước mỗi điểm (`PREDICT_MUTATES`);
như vậy snapshot của DenStream giờ đúng là phân hoạch của một trạng thái, khác với snapshot ở mục 2.

### 7.2 Chuyển tiếp giữa hai lần dò và nguyên nhân của nó

Cứ `probe_every` điểm, audit gán lại cả cửa sổ bằng mô hình hiện tại và đọc chuyển tiếp từ lần dò trước theo
cách phân loại của MONIC (Spiliopoulou và cộng sự, KDD 2006). Một cụm trước *tách* khi ít nhất hai cụm sau mỗi
cụm giữ ít nhất `tau_split` điểm của nó và được tạo phần lớn (`tau`) từ nó; nếu không, nó *sống sót* vào cụm sau
giữ ít nhất `tau` điểm của nó, hoặc bị *hấp thụ* vào cụm đó khi một cụm trước khác cũng dồn `tau` điểm vào đấy;
còn lại là *biến mất*. Cụm sau không cụm trước nào dẫn tới là cụm *mới*. Mỗi cụm trước trao danh tính cho đúng một
người thừa kế. Kết cục của nhãn được đọc trên các cụm sau: *giữ nhãn* khi nhãn của nó là nhãn của một người nó
thừa kế, *đổi tên* khi nó có người thừa kế nhưng mang nhãn khác, và một nhãn có mặt ở cả hai lần mà cụm sau
mang nó không hậu duệ của cụm trước mang nó là nhãn *bị dùng lại*. Dưới một nhãn bị dùng lại, phân hoạch tuần
tự ghép hai cụm mà mô hình tách riêng ở cả hai thời điểm.

Mỗi khoảng dò có một `cause`: `rename_only` khi phân hoạch không đổi mà nhãn đổi (macro-cluster được dựng lại
và đánh số khác), `structural_micro` khi phân hoạch đổi và có micro-cluster sinh ra hay mất đi (đọc theo danh
tính đối tượng, `micro_objects`), `structural_macro_only` khi phân hoạch đổi mà các micro-cluster vẫn là những
đối tượng cũ, và `structural_unattributed` khi mô hình không cho đọc micro-cluster (STREAMKMeans). Trên kịch bản
của `fidelity`, mỗi khoảng dò còn mang các sự kiện thật nằm trong nó, nên bảng 1b cho biết điều gì xảy ra với
nhãn đúng lúc có birth, deletion, merge hay split.

Ở thuật toán có danh tính bền (SOStream, RepStream, STREAMKMeans, HPStreamStableLabels), một cụm dịch chuyển
trọn sang các điểm khác giữa hai lần dò cũng được đọc là nhãn bị dùng lại, nên với chúng con số này là cận trên.

### 7.3 Các dự đoán cũ trong một cửa sổ có so sánh được với nhau không

`pair_consistency` xét mọi cặp i < j của cửa sổ, chia theo độ trễ j − i (mặc định các khoảng [1, 10), [10,
100), [100, 1000), [1000, ∞)). Với một phân hoạch tham chiếu, *precision* là tỉ lệ các cặp cùng nhãn tuần tự mà
tham chiếu cũng đặt chung một cụm, *recall* là tỉ lệ các cặp tham chiếu đặt chung mà cũng cùng nhãn tuần tự.
Nếu một nhãn gọi tên cùng một cụm suốt cửa sổ thì cả hai đường phẳng theo độ trễ. Nhãn bị dùng lại kéo
precision xuống ở độ trễ dài, cụm bị đổi tên kéo recall xuống ở độ trễ dài. Audit đọc `seq` và `tracked` so với
`post` và so với lớp thật, và đọc `pre`, `post` so với lớp thật làm đối chứng: một phân hoạch của một trạng thái
không có lý do gì để suy giảm theo độ trễ, trừ khi chính các lớp thật dịch chuyển trong cửa sổ.

### 7.4 Chất lượng gán tuần tự: nó đo gì, và cách giữ danh tính cụm không cần nhãn

**Định nghĩa.** Chất lượng gán tuần tự của một cửa sổ (ARI hay silhouette của `seq`) đo mức khớp giữa các quyết
định mà mô hình đưa ra trực tuyến, mỗi quyết định bởi trạng thái nó có lúc đó, với lớp thật (hay với hình học
của dữ liệu), *với giả định rằng một nhãn thô gọi tên cùng một cụm từ điểm đầu đến điểm cuối cửa sổ*. Giả định
này đúng với SOStream, RepStream, STREAMKMeans và HPStreamStableLabels, và sai với DenStream, BIRCH, DBSTREAM,
CluStream và HPStream của River (mục 1 và 6). Khi giả định sai, con số trộn chất lượng gom cụm với cách mô hình
đặt tên: đổi tên tách một cụm thành hai nhãn và hạ điểm, dùng lại nhãn ghép hai cụm và làm điểm lệch theo hướng
không đoán trước được. Con số gộp cả run (`lag_ari_cumulative`, hay `cum_ari_seq` trong audit) còn ghép nhãn qua
mọi cửa sổ, nên nó chịu cả những lần đánh số lại giữa các cửa sổ; với các chế độ tune trực tuyến nó còn ghép nhãn
của những mô hình khác nhau sau mỗi lần đổi incumbent, và không cách giữ danh tính nào làm nó mạch lạc được.

**Cách giữ danh tính không dùng nhãn đánh giá** (`IdentityTracker`). Mỗi danh tính giữ vài *điểm neo*, là những
điểm gần nhất được gán cho nó (`anchors_per_id`, tổng tối đa `max_anchors`, ưu tiên danh tính được dùng gần
nhất). Sau mỗi điểm mô hình học, các điểm neo được mô hình gán lại, và chuyển tiếp từ danh tính sang nhãn thô
mới được đọc như mục 7.2. Một nhãn thô nhận danh tính của người thừa kế nó: trước hết là người thừa kế vốn đang
mang chính nhãn thô này (không có bằng chứng nào cho thấy tên của mô hình đã thôi gọi cụm đó), nếu không có thì
người thừa kế có nhiều điểm neo nhất ở đó, rồi đến danh tính cũ hơn. Cụm mới, và cụm con của một lần tách mà không
phải người thừa kế, nhận danh tính mới. Danh tính bị hấp thụ mà không kế tục thì bị cho nghỉ và không bao giờ được
trao lại. Một nhãn thô không xuất hiện trên điểm neo nào giữ danh tính cũ của nó, nếu danh tính đó chưa nghỉ và
chưa thuộc về nhãn thô khác. Nhãn tracked của một điểm là danh tính mà nhãn thô của nó mang lúc điểm được dự đoán.

Hai điểm i < j cùng một nhãn tracked nghĩa là: cụm chứa điểm i đã trao danh tính của nó, qua các lần sống sót,
hấp thụ hay là cụm con kế tục, cho cụm chứa điểm j, như chính các dự đoán của mô hình trên điểm neo cho thấy. Nhãn
tracked vì thế không đổi khi mô hình chỉ hoán vị tên (BIRCH, CluStream), nhận danh tính mới khi một tên cũ được
trao cho cụm không liên quan (HPStream), và giữ danh tính của cụm lớn hơn khi hai cụm gộp. Tracker chỉ đọc dự đoán
của mô hình trên các điểm của luồng; nó dự đoán trên mô hình đang chạy khi việc dự đoán không đổi được trạng thái,
và trên một bản sao khôi phục khi có thể đổi, nên không bao giờ đổi điều mô hình dự đoán (test
`test_tracking_and_probing_leave_every_protocol_model_predicting_as_it_would` trên cả mười mô hình).

**Giới hạn đã biết.** (1) Một cụm dịch chuyển trọn sang vùng khác được đọc là vẫn danh tính đó, vì mô hình vẫn
đặt các điểm neo cũ chung với điểm mới. (2) Một cụm không còn điểm neo nào (do trần `max_anchors`) giữ danh tính
theo tên thô cho tới khi được thấy lại; với mô hình có hàng trăm nhãn (BIRCH với `threshold = 0`, RepStream) điều
này xảy ra thường. (3) Ở mô hình có macro-cluster nhỏ sinh ra rồi mất đi liên tục (DBSTREAM ở mặc định), tracker
tạo nhiều danh tính ngắn hạn: đó là đúng phân hoạch tạm thời của mô hình, nhưng nó hạ recall so với nhãn thô, vốn
ghép các cụm tạm thời khác nhau dưới cùng một số. (4) `tau`, `tau_split`, `anchors_per_id`, `max_anchors` là giá
trị chạy được, không phải khuyến nghị. Độ tin cậy của tracker được kiểm ở nơi biết đáp án chính xác: trên cùng
một ô, HPStreamStableLabels là phân hoạch an toàn danh tính của chính mô hình HPStream, nên `validation.csv` so
nhãn thô và nhãn tracked của HPStream với nó, và so nhãn tracked của biến thể với chính nó (một tracker hoàn hảo
cho ARI 1 ở phép so sau).

### 7.5 Đánh giá snapshot trong pipeline: `run --snapshot-eval`

Với incumbent trễ của mỗi cửa sổ (member mà `lag_ari_win` đọc), cờ này ghi thêm vào dòng cửa sổ: `lag_snap_seq_*`
(chính nhãn tuần tự đó, chấm bằng cùng một phép tính batch như hai snapshot), `lag_snap_pre_*` (mọi điểm gán bởi
mô hình khi cửa sổ bắt đầu, ngoài mẫu) và `lag_snap_post_*` (sau khi đã học cửa sổ, trong mẫu). Với mỗi snapshot
có ARI, NMI, purity, silhouette cổ điển, số nhãn và ARI so với nhãn tuần tự; dòng tóm tắt có trung bình và trung
bình phần đuôi, cùng `snapshot_runtime_s`. Cửa sổ có ít hơn hai lớp thật không có số đo ngoài, như `lag_ari_win`.
Cờ chỉ đọc bản sao của member, không đổi quyết định nào (test so từng cửa sổ với lần chạy không cờ), và không
thuộc protocol trong manifest. `python -m conf_pipeline.identity_audit --read-run DIR` tóm tắt các trường này theo
thuật toán, nhóm luồng và chế độ tune vào `DIR/report/snapshot_eval/`.

Cách đọc đề xuất: `lag_snap_pre_ari_win` là chất lượng phân hoạch của mô hình mà người dùng nhận được ở đầu cửa sổ,
không phụ thuộc cách mô hình đặt tên trong cửa sổ; `lag_ari_win` là chất lượng gán tuần tự theo định nghĩa ở mục
7.4. Hai số lệch nhau nhiều ở một thuật toán đặt tên không bền là dấu hiệu con số tuần tự đang đo cách đặt tên.

### 7.6 Lệnh và chi phí đo được (08/11/2026)

Smoke trên Mac (Apple M3 Pro, 8 worker), 6 kịch bản × seed 1 × `default,scaled_default`, h = 500, 9 cửa sổ chấm:
96 ô, không ô nào lỗi, 1 506 s wall, khoảng 8 150 s CPU cộng trên các ô. Riêng DenStream của River chiếm phần lớn:
461 s ở `default` và 6 045 s ở `scaled_default` (ô dài nhất 1 328 s), vì snapshot phải khôi phục trạng thái
trước mỗi điểm. Sau smoke, lần refresh của tracker trên mô hình loại này đổi sang một bản sao mỗi lần refresh,
nên con số đó là cận trên. Chi phí trên luồng thật chưa đo xong (smoke dừng giữa chừng): BIRCH ở `threshold = 0`
có hàng trăm nhãn mỗi cửa sổ và là ô đắt nhất. `run --snapshot-eval` tốn dưới 1 s mỗi job trên smoke
`static_overlap` 8 000 điểm (DBSTREAM, DenStream, CluStream × default, confstream), và không đổi quyết định nào.

```bash
# kịch bản có sự kiện biết trước
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m conf_pipeline.identity_audit --out result/identity_audit_scenarios \
  --seeds 1,2,3 --configs default,scaled_default --workers 16
# luồng của protocol: chạy bản ngắn trước để đo, rồi bản đủ
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m conf_pipeline.identity_audit --out result/identity_audit_folds_w3 --scenarios "" \
  --folds <các fold> --fold-windows 3 --configs default,scaled_default --workers 16
```

### 7.7 Tín hiệu chọn của bộ tune giờ đọc snapshot cuối cửa sổ (08/11/2026)

Bài confStream (Carnein và cộng sự, mục 3) huấn luyện ensemble trên các điểm của cửa sổ rồi mới chấm
chất lượng gom cụm của từng cấu hình; bản MOA chấm trên micro-cluster cuối cửa sổ. Bài không bàn tới danh
tính nhãn, vì cách chấm đó không đọc nhãn tuần tự. Việc chấm silhouette trên nhãn tuần tự là lựa chọn riêng
của repo, và mục 7 cho thấy nó làm điểm chấm phụ thuộc cách đặt tên của mô hình. Từ ngày này
`EnsembleConfig.score_partition` mặc định là `snapshot`: score, `silhouette_raw`, `silhouette_reason`,
`support_sha` và các tín hiệu chẩn đoán theo member đọc phân hoạch của mô hình sau khi đã học cửa sổ
(`snapshot_labels`, DenStream của River được khôi phục trước mỗi điểm). Silhouette tuần tự vẫn được ghi
cạnh đó (`seq_silhouette`, `seq_silhouette_reason`), cùng các số đo ngoài của snapshot cho từng member
(`snap_ari_win`, `snap_nmi_win`, `snap_purity_win`, chỉ để báo cáo). Hai chỗ còn khác MOA, do API của River:
mọi điểm đều có nhãn của `predict_one` (MOA bỏ điểm nằm ngoài bán kính mọi micro-cluster), và silhouette
nằm trên [−1, 1] thay vì được co về [0, 1]. Quy tắc cũ là `--score-partition sequential`; một job thiếu
trường này (lập kế hoạch trước ngày này) được đọc là `sequential`, và manifest phân biệt hai quy tắc, nên
một out dir cũ không bị chạy tiếp bằng quy tắc mới.

**Số chính của báo cáo cũng đổi theo.** Dưới quy tắc `snapshot`, mỗi member còn được gán lại cả cửa sổ bằng
trạng thái nó có khi cửa sổ bắt đầu, và các số đo ngoài `ari_win`, `nmi_win`, `purity_win`, `cmm_win` của nó
đọc phân hoạch đó (ngoài mẫu, một trạng thái). Vì vậy `lag_ari_win` của dòng cửa sổ là ARI của snapshot đầu
cửa sổ của incumbent trễ, trùng với `lag_snap_pre_ari_win`, và mọi bảng của `report`, `robustness`,
`followup`, `selection`, mọi lựa chọn oracle của `sweep` (cờ `--score-partition` của nó mặc định cũng là
`snapshot`) đọc con số này mà không cần đổi gì. Số tuần tự vẫn được ghi cạnh đó dưới tên `seq_<metric>_win`,
`lag_seq_<metric>_win` và `lag_seq_ari_win_mean`, với nghĩa chất lượng gán trực tuyến ở mục 7.4; riêng CMM chỉ
được tính trên phân hoạch chính. Mỗi dòng mang `ari_basis` (`snapshot_pre` hoặc `sequential`; dòng cũ không có
trường này được đọc là `sequential`), và `report.current_runs` từ chối một out dir trộn hai cơ sở. Số gộp cả
luồng `lag_<metric>_cumulative` đã bị bỏ khỏi dòng tóm tắt và khỏi báo cáo: nó ghép nhãn của mọi cửa sổ và mọi
incumbent, vốn không gọi tên chung cụm nào.

### 7.8 Vùng đánh giá bắt đầu sau cửa sổ đóng băng hai cửa sổ; thứ tự của challenger (08/11/2026)

Theo cách đọc trễ, cửa sổ `t0` (cửa sổ đóng băng, 20) được đọc từ incumbent chọn sau `t0 − 1`, và cửa sổ
`t0 + 1` từ incumbent chọn sau `t0`: dưới `confstream` và `frozen_prefix` đó là cùng một mô hình. Cửa sổ đầu
tiên mà một quyết định sau khi đóng băng có thể chạm tới là `t0 + 2`. Từ ngày này phần đuôi của mọi arm bắt
đầu ở đó (`_common.POST_FREEZE_LAG = 2`, `_common.tail_start`): `tail_from` của dòng tóm tắt là 22 ở protocol,
dòng tóm tắt ghi thêm `tail_freeze_window`, lượt quét chọn oracle trên cùng vùng đó, và mọi so sánh
`confstream`–`frozen_prefix` là strictly post-freeze. Run trước ngày này tính đuôi từ chính cửa sổ đóng băng;
báo cáo đọc `tail_from` của từng run nên vẫn đọc đúng chúng, còn `followup post-freeze` lấy `t0` từ
`tail_freeze_window` khi có. KDD99 còn 23 trên 78 cửa sổ có hơn một lớp, INSECTS gradual balanced còn 2.

Thứ tự vào tuner của một challenger, đúng như mã chạy: tạo sau cửa sổ `t − 1` và warm-start trên cửa sổ đó chỉ
bằng lời gọi learn; chấm lần đầu ở cửa sổ `t`, mỗi điểm được dự đoán trước khi học; nhận vào hay thay thế sau
cửa sổ `t`; làm incumbent sớm nhất từ cửa sổ `t + 1`; không mô hình nào học một điểm hai lần, và snapshot chỉ dự
đoán trên bản sao. `tests/test_challenger_timing.py` kiểm từng điều này trên mô hình ghi lại mọi lời gọi. Theo
quy tắc snapshot, silhouette dùng để chọn được đọc sau khi mọi mô hình đã học cửa sổ `t` (trong mẫu, như
confStream), không phải điểm prequential; điểm ngoài của cửa sổ đọc trạng thái trước cửa sổ, nên ngoài mẫu.
