# Chiến dịch cơ chế (E1, E2, E3, E5): protocol thực tế và lệnh chạy

**Cập nhật 09/11/2026.** Chiến dịch có thêm ba nhánh E3 của paper: `frozen_ensemble` (giữ cả portfolio
lúc đóng băng, tiếp tục chọn giữa các model sẵn có, không tạo model mới), `portfolio_renewal` (giữ cấu hình
của portfolio, mỗi cửa sổ làm mới model của `e_test` cấu hình) và `random_after_prefix` (tiếp tục tune bằng
cấu hình rút đều trên không gian từ cùng trạng thái). Cùng với `frozen_prefix` và `confstream`, đó là năm thủ
tục mà phản biện yêu cầu; cả năm dùng chung phần đầu luồng tới cửa sổ đóng băng (chạy lại tất định, chứng
nhận bằng `followup prefix-identity`). Danh sách job thành 1 848 job chạy được (240 job cho mỗi nhánh mới) và
24 job ghi là bỏ qua. `report --contrasts` có thêm `portfolio_renewal − frozen_prefix` và
`random_after_prefix − frozen_prefix`, và với mỗi phép so ghi chi phí của hai bên sau đóng băng (`costs.csv`:
giây CPU mỗi run, tỉ lệ trung vị, số model được chấm mỗi cửa sổ), vì các thủ tục liên tục chấm E + C model mỗi
cửa sổ còn `frozen_prefix` chỉ một: hiệu chất lượng đi kèm hiệu chi phí đó, không phải ở ngân sách bằng nhau.
Cấu hình ghi `score_partition: snapshot` (silhouette chọn và mọi số `*_win` đọc snapshot,
`docs/SEQUENTIAL_PARTITION_AUDIT.md` mục 7.7), vùng đánh giá bắt đầu ở t ≥ 22 (mục 7.8), nên bốn vấn đề nhãn
GAP-6 không còn giới hạn các số đọc của chiến dịch (`measurement_status` ghi `resolved by the protocol`); ba
điều kiện chung của mục 8 vẫn mở. Dự toán (`cost --campaign`, chi phí đo của `out/full`, các nhánh mới đọc
qua `confstream`, hệ số 1,15 cho hai lần dự đoán snapshot): khoảng 471 giờ tiến trình, khoảng 59 giờ trên 8
worker. Các đoạn dưới đây mô tả phiên bản sáu arm ngày 04/11 và giữ nguyên làm lịch sử.

**Chỉ chạy E3 (09/11/2026).** `conf_pipeline/campaigns/e3_mechanism.json` là chiến dịch trên, giới hạn ở
năm thủ tục E3: `frozen_prefix`, `frozen_ensemble`, `portfolio_renewal`, `random_after_prefix`, `confstream`.
Luồng, nhóm thuật toán, tuner seed, model seed và protocol giữ nguyên, trừ một điểm: chẩn đoán `stability`
không được ghi (`signals: []`). Nó không bao giờ vào phép chọn và không phép so E3 nào đọc nó, nhưng chiếm
khoảng 5–20% thời gian mỗi job ở các smoke ngày 04/11. Vì vậy các run E3 không gộp được vào thư mục của
`mechanism_pilot.json`. Thư mục ra là `conf_pipeline/out/e3_mechanism`, gồm 1 200 job (240 mỗi arm), không
job nào bị bỏ qua. Danh sách job đúng bằng các job năm arm đó trong chiến dịch đầy đủ
(`tests/test_conf_campaign.py`).

Timeout là 86 400 giây. Job dài nhất dự kiến là CluStream trên kdd99 ở bốn arm liên tục: 10,5 giờ trên một lõi
vật lý (`cost --stat max`, hệ số 1,15 cho snapshot, 1,2 cho bỏ gate). Với 16 worker trên máy 8 lõi 16 luồng,
mỗi job chậm khoảng 1,5 lần, tức khoảng 15,7 giờ, quá 12 giờ của chiến dịch đầy đủ.

Dự toán bên dưới đọc từ chi phí đo của `out/full`. Ba nhánh mới được đọc qua `confstream`, và
HPStreamStableLabels qua HPStream.

| Máy, số worker | Giờ tiến trình (trung bình / chậm nhất) | Wall (trung bình / chậm nhất) |
|---|---:|---:|
| 8C/16T, 8 worker | 383 / 422 | 48 h / 53 h |
| 8C/16T, 16 worker (giả định chậm 1,5 lần mỗi job) | 575 / 633 | 36 h / 40 h |
| 32C/64T, 32 worker | 383 / 422 | 12 h / 13 h |

`report --contrasts` đọc bảy phép so E3:

- `frozen_ensemble`, `portfolio_renewal`, `random_after_prefix` và `confstream`, mỗi thủ tục so với `frozen_prefix`;
- `portfolio_renewal − frozen_ensemble`;
- `confstream − random_after_prefix`;
- `confstream − portfolio_renewal`.

Mỗi phép so có chi phí sau đóng băng của hai bên trong `costs.csv`. `followup prefix-identity` chứng nhận
phần đầu luồng chung; `followup post-freeze` kiểm thứ tự challenger trên log.

```bash
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.campaign --config conf_pipeline/campaigns/e3_mechanism.json
PYTHONHASHSEED=0 nohup .venv/bin/python -m conf_pipeline.campaign --config conf_pipeline/campaigns/e3_mechanism.json \
    --execute --workers 16 > conf_pipeline/out/e3_mechanism.log 2>&1 &
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.campaign --config conf_pipeline/campaigns/e3_mechanism.json --status
```

**Đối chứng cùng ngân sách (09/11/2026).** Phản biện đòi so mỗi thủ tục liên tục với mô hình đóng băng
"dưới ngân sách tương đương". `frozen_prefix` chỉ chấm một model mỗi cửa sổ sau đóng băng, còn các thủ tục
liên tục chấm E + C model. Arm `frozen_prefix_matched` (`_common.MATCHED_ARMS`) đóng băng một cấu hình ở
cùng cửa sổ, nhưng trước đóng băng tune với E và C nhân cùng một hệ số (`_common.matched_sizes`). Hệ số được
chọn sao cho tổng số model được chấm trên cả luồng bằng của `confstream`: E ở cửa sổ chấm đầu tiên, E + C ở
mỗi cửa sổ sau, và một model mỗi cửa sổ sau đóng băng (`_common.evaluations`).

Hệ số tính trong run, sau pha hiệu chỉnh, vì số cửa sổ được chấm phụ thuộc độ dài luồng và số cửa sổ hiệu
chỉnh. Trên luồng 100 cửa sổ, hệ số khoảng 4,9 (E' = 98, C' = 49). Run ghi `budget_factor`, `budget_e_size`,
`budget_e_test`, `budget_target_evaluations`, `budget_planned_evaluations` và `n_evaluated_total`; trên
smoke hai số cuối khớp đúng với `confstream` (68 = 68, `tests/test_budget_matched.py`).

Ngân sách đo bằng số model được chấm, không bằng giây CPU, vì giây CPU phụ thuộc máy, số worker và cấu
hình được chọn. Arm này khác các nhánh từ cửa sổ 0, nên không chung phần đầu luồng; `followup
prefix-identity` bỏ nó ra. `report --contrasts` đọc thêm năm phép so cùng ngân sách: bốn thủ tục liên tục,
mỗi thủ tục so với `frozen_prefix_matched`, và `frozen_prefix_matched − frozen_prefix`.

Đã thêm vào `e3_mechanism.json`: 1 440 job. Dự toán (`--stat max`, arm mới đọc qua `confstream`, là giả định):
520 giờ tiến trình; khoảng 65 h trên 8 worker, 49 h trên 16 worker (8C/16T), 16 h trên 32 worker.
Rủi ro chưa đo: 147 model trong một tiến trình. Với RepStream trên k_drift (51 MB mỗi model ở `out/full`),
tổng có thể vượt giới hạn 4 GiB mỗi tiến trình; job như vậy sẽ được ghi `oom` và đếm trong coverage.

Ngày 04/11/2026 (cập nhật cùng ngày sau vòng review thứ hai). Tài liệu này mô tả protocol mà code hiện tại
chạy, các kiểm tra đã làm và các lệnh để chạy tiếp. Chiến dịch chưa chạy; ở đây không có kết quả thực nghiệm
nào của nó. Test và dry run chạy qua không có nghĩa là chiến dịch đã sẵn sàng làm bằng chứng xác nhận cho
paper: mục 8 liệt kê những gì còn chặn. Cấu hình nằm ở
`conf_pipeline/campaigns/mechanism_pilot.json` (`.gitignore` có ngoại lệ `!conf_pipeline/campaigns/*.json`
để nó đi cùng code), lệnh chạy là `conf_pipeline.campaign`. Hash code (`code_sha256`) không phủ file JSON;
manifest của chiến dịch ghi `config_sha256` riêng.

So với thiết kế pilot ở `docs/MECHANISM_PILOT.md`, chiến dịch này giữ đúng 12 luồng đã khóa và sáu
arm `default`, `scaled_default`, `confstream`, `frozen_prefix`, `frozen_reinit`, `random_online`. Hai
arm E4 (`confstream_ari_oracle`, `confstream_ari_oracle_constrained`) không nằm trong danh sách job.
Thêm chúng sau này dưới cùng protocol là một lần mở rộng hợp lệ: thêm tên arm vào `arms` của cấu hình,
manifest ghi lần mở rộng vào lịch sử và giữ nguyên các kiểm tra khác.

## 1. Protocol

**Tín hiệu chọn (đổi ngày 08/11/2026).** Mỗi member vẫn gán nhãn từng điểm theo predict-before-learn
trong lúc học cửa sổ t. Cuối cửa sổ, silhouette cổ điển được tính trên cụm mà member có sau khi đã học
xong cửa sổ: mọi điểm của cửa sổ được gán bởi đúng trạng thái đó (`score_partition = snapshot`), như bài
confStream ("after every window, we evaluate the clustering quality for every configuration"). Trước
ngày này, silhouette được tính trên chính các nhãn tuần tự; quy tắc đó còn chạy được bằng
`run --score-partition sequential` (để tái lập `out/full`) và không trộn được với quy tắc mới trong một
out dir, vì manifest ghi nó trong cấu hình của arm. Incumbent chọn tại t chỉ được chấm ARI ở cửa sổ t+1:
`lag_ari_win` là ARI của snapshot của incumbent khi cửa sổ t+1 bắt đầu (một trạng thái, ngoài mẫu;
`ari_basis = snapshot_pre`), `lag_seq_ari_win` là ARI trên nhãn tuần tự của cùng cửa sổ
(`docs/SEQUENTIAL_PARTITION_AUDIT.md` mục 7.7). Không có số gộp cả luồng. Incumbent có thể giữ nguyên
hay đổi giữa các cửa sổ. `stability` được ghi như chẩn đoán và không bao giờ vào phép chọn.

**Giá trị khóa.** Các giá trị dưới đây được đọc từ hằng số của repo (`conf_pipeline/_common.py`);
`campaign.check_config` từ chối một cấu hình lệch khỏi chúng, nên chiến dịch không tự chỉnh protocol.

| Thành phần | Giá trị |
|---|---|
| Luồng | covtype, kdd99, electricity, shuttle, insects_abrupt_imbalanced, scale_drift, density_drift, k_drift, mixed_drift, static_overlap, incremental_drift, evolving |
| Cửa sổ h, ensemble E, số con mỗi cửa sổ C | 1 000, 20, 10 |
| Cửa sổ freeze t0 (`frozen_prefix`, `frozen_reinit`; cũng là điểm bắt đầu đuôi đánh giá) | 20 |
| Mức cắt mẫu | 100 000 |
| p_reset; λ | 0,01; λ = 1/round(0,10 · số cửa sổ của luồng), ghi ở từng run (`lam`) |
| Thuật toán | DenStream, HPStreamStableLabels (thay HPStream, xem dưới), DBSTREAM, STREAMKMeans, SOStream, BIRCH, CluStream, RepStream |
| Chính sách seed | `split`: tuner seed 101, 102, 103 cho DenStream, HPStreamStableLabels, DBSTREAM, STREAMKMeans, SOStream, BIRCH; 101 cho CluStream, RepStream; model seed 201 cho arm online; model seed cố định 201; calibration seed 0; data seed 42 |
| Runtime gate | tắt (`--no-runtime-gate`) |
| Phá hoà | `micro_clusters`, khoá theo thứ tự: điểm số, dùng được hay không, số micro-cluster (không biết thì tính 0), incumbent trước, rồi `member_id` nhỏ hơn. Member không dùng được: điểm số, incumbent trước, `member_id`. Runtime không vào khoá |
| Log | `--log-members`, `--signals stability` |
| Môi trường con | `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` ép cho mọi tiến trình con |

**Giới hạn còn áp dụng.** Đây là giới hạn vận hành, không phải tham số khoa học, và đều được ghi
trong manifest của chiến dịch:

- Mỗi member: 2 048 MiB bộ nhớ ước lượng và 10 000 micro-cluster. Vượt một trong hai thì member bị
  đánh dấu không khả thi (`score_status` = `infeasible_memory` hay `infeasible_micro_clusters`) và
  nhận điểm sàn −1, như ở `out/full`.
- Mỗi tiến trình con: giới hạn không gian địa chỉ 4 GiB (`RLIMIT_AS`). macOS từ chối giới hạn này,
  nên trên máy kiểm tra hôm nay nó không được áp; manifest ghi `enforcement.rlimit_as` của máy chạy thật.
- Mỗi job: 43 200 giây, lấy từ preset `full` của `conf_pipeline/make_notebook.py` (`TIMEOUT_S = 12 * 3600`),
  giới hạn mà `out/full` đã chạy dưới. Mặc định 21 600 giây của CLI `run` không được dùng. Job quá giờ
  được ghi `status = timeout` và không có điểm; nó vẫn đếm trong bảng coverage.

**Danh sách job.** `run.plan_jobs`, cũng là hàm `run` dùng để chạy, cho 1 128 job chạy và 24 job ghi là
bỏ qua: 996 + 132 job chạy, `scaled_default` của HPStreamStableLabels và RepStream (12 + 12) trùng `default` nên
không chạy. Theo arm: `confstream`, `frozen_prefix`, `frozen_reinit`, `random_online` mỗi arm 240,
`default` 96, `scaled_default` 72. Arm cố định của thuật toán tất định chạy một lần, không theo tuner seed;
của CluStream, HPStreamStableLabels, STREAMKMeans chạy một lần trên model seed 201.

**HPStream đọc qua biến thể (quyết định của người dùng, 04/11/2026).** Danh sách job chạy
`HPStreamStableLabels` ở chỗ của HPStream: cùng tham số, cùng không gian tìm kiếm, cùng cụm và cùng cụm
được dự đoán ở mọi điểm, chỉ khác là nhãn không bao giờ được dùng lại (RIVER_BUGS.md GAP-6, mục 8). Cấu hình
ghi quyết định này ở `decisions`. Chiến dịch không chạy cả hai: chúng là một mô hình, và LOBO cùng các trung
bình qua thuật toán sẽ đếm nó hai lần. Các bảng đọc nó dưới tên `HPStreamStableLabels`. `HPStream` của River
vẫn được đăng ký và chạy được như cũ; kết quả cũ (`out/full`) là của bản River.

## 2. Các arm đối chứng

- **E1 `scaled_default`.** Chỉ các tham số khoảng cách tuyệt đối được nhân với s của luồng; mọi tham số
  khác giữ mặc định River. DenStream `epsilon`, DBSTREAM `clustering_threshold`, SOStream
  `merge_threshold`, CluStream và STREAMKMeans `sigma` đổi. BIRCH có `threshold = 0`, nhân s vẫn là 0,
  nên run của nó trùng `default`. HPStreamStableLabels và RepStream không có tham số khoảng cách và không chạy.
  Mỗi run ghi `baseline_params`, `scaled_params`, `scaled_changed`, `scaled_unchanged`; dry run in
  bảng này (`e1_configs`) trước khi chạy gì.
- **E2 `frozen_reinit`.** Đi giống hệt `confstream` đến quyết định ở cửa sổ t0 (test
  `test_the_frozen_arms_follow_confstream_event_for_event_until_the_freeze_decision`). Ở t0 ensemble
  thu lại còn đúng incumbent, như `frozen_prefix`. Từ đó cấu hình bị khóa ở cấu hình của incumbent,
  nhưng mỗi cửa sổ vẫn sinh C bản sao mới của cấu hình đó, warm-start trên cửa sổ vừa qua; các bản sao
  lấp ensemble trở lại E member (sự kiện `filled`) rồi cạnh tranh bằng silhouette như con của bộ tune.
  `frozen_prefix` giữ một mô hình tiếp tục học và không sinh gì thêm. Vì vậy sau t0, `frozen_reinit` và
  `confstream` khác nhau ở hai chỗ cùng lúc: cấu hình (khóa hay còn đổi) và lịch sử của ensemble (các
  member của `frozen_reinit` là bản sao chỉ học từ cửa sổ sinh ra chúng, cạnh mô hình incumbent cũ;
  member của `confstream` là các cấu hình khác nhau với lịch sử dài ngắn khác nhau). Hiệu
  `confstream − frozen_reinit` là so sánh hai thủ tục, chưa phải hiệu ứng của riêng việc đổi cấu hình;
  `frozen_reinit − frozen_prefix` cũng vậy với việc làm mới mô hình.
- **E5 `random_online`.** Code hiện tại (`ConfStreamEnsemble.initialise` và nhánh `sampler == "uniform"`
  của `process_window`) giữ quần thể ban đầu chung với `confstream`: cùng cấu hình khởi đầu, cùng các
  con `initial_offspring` rút quanh nó, cùng dòng số của tuner seed (test
  `test_uniform_children_keep_the_initial_population_and_the_schedule_of_the_evolutionary_ones`). Chỉ các
  lần sinh sau đó đổi cách rút: mỗi con được rút đều trên toàn không gian tìm kiếm, không cha, ở độ rộng
  ban đầu, thay vì rút tiến hoá quanh một cha. Từ lần sinh đầu tiên đó hai arm tiêu dòng số khác nhau,
  nên cả việc chọn nạn nhân cũng tách ra. E, C, lịch sinh và lịch chấm giữ nguyên. Cùng lịch sinh và lịch
  chấm không có nghĩa là cùng chi phí tính toán: các cấu hình khác nhau tốn thời gian và bộ nhớ khác
  nhau, và không có giới hạn tính toán chung, nên đây không phải phép so cùng ngân sách tính toán.

**Đã implement, đã test, đã smoke, đã có kết quả.** Bốn mức này khác nhau và được ghi riêng:

| Phần | Implement | Test | Smoke | Kết quả chiến dịch |
|---|---|---|---|---|
| Sáu arm, lifecycle, `score_status` | có | có (`tests/test_confstream.py`, `tests/test_conf_pipeline.py`) | có, `static_easy`, 4 thuật toán (mục 4) | chưa |
| E1 `scaled_default` | có | có | có | chưa |
| E2 `frozen_reinit` | có | có (theo sự kiện đến t0, bản sao sau t0) | có | chưa |
| E5 `random_online` | có | có (quần thể ban đầu, lịch sinh) | có | chưa |
| LOBO theo bảng huấn luyện mong đợi (mục 3) | có | có (`tests/test_conf_selection.py`) | có, luồng thật (mục 4) | chưa |
| Ghi tập điểm được chấm và phân loại support (mục 3) | có | có (`tests/test_point_support.py`) | có (mục 4) | chưa |
| `HPStreamStableLabels` thay HPStream (mục 1, 8) | có | có (`tests/test_hpstream_stable_labels.py`) | có, trong ensemble và trên luồng thật (mục 4) | chưa |
| Trạng thái đo theo thuật toán trong dry run và manifest (mục 8) | có | có (`tests/test_conf_campaign.py`) | có (dry run) | — |

Mỗi job ghi lịch sử của ensemble vào `lifecycle/<job>.jsonl`, tách khỏi `members/`: mỗi lần sinh
(nguồn gốc `start`, `initial_offspring`, `evolutionary`, `uniform`, `reinit_copy`, cha, có reset không,
tham số, số mẫu warm-start, lỗi), mỗi lần thách đấu (nạn nhân, kết quả), mỗi quyết định (incumbent,
mức của khoá phá hoà đã quyết định, số member hoà ở đỉnh, số member được chấm) và lần freeze. Dòng
tóm tắt của run đếm `n_births`, `n_births_ok`, `births_by_origin`, `n_member_evaluations`,
`member_status_counts`, `decided_by_counts`. Số đếm là số thực tế; không có chỗ nào ép chúng bằng nhau
giữa các arm.

## 3. Đọc kết quả

- `report --contrasts` (thư mục `report/contrasts/`): năm hiệu `scaled_default − default`,
  `confstream − scaled_default`, `frozen_reinit − frozen_prefix`, `confstream − frozen_reinit`,
  `confstream − random_online`, mỗi hiệu tính theo replicate trên các cửa sổ đuôi mà cả hai arm cùng có
  số đọc. Hai arm online ghép theo cả tuner seed lẫn model seed (cùng tuner seed nhưng model seed khác là
  một replicate khác, được báo là thiếu ở bên kia); arm cố định lấy trung bình các run của nó và đặt cạnh
  mọi replicate của arm online. Trung bình lên (thuật toán, luồng), rồi (thuật toán, khối nguồn) với các biến thể của một
  nguồn lấy trung bình trước, rồi khối nguồn, rồi nhóm luồng. Tuner seed không thêm khối. ARI là chính,
  NMI là độ nhạy. Có hiệu liên tục, khoảng bootstrap theo khối nguồn khi có từ ba khối, số khối trên và
  dưới 0, tỉ lệ ô vượt +0,05 và dưới −0,05. `scaled_default − default` có thêm tập con chỉ gồm các ô
  mà cấu hình thật sự đổi. Bảng coverage theo arm: job mong đợi (từ danh sách job của chiến dịch), ok,
  lỗi, timeout, OOM, thiếu. Ba tỉ lệ tách riêng, vì một member có thể hỏng nhiều cách cùng lúc: phân
  hoạch không có silhouette (theo từng lý do, đọc từ `silhouette_reason` trước mọi giới hạn, mẫu số là
  các member-window có member chạy được), bị một giới hạn loại (bộ nhớ, micro-cluster, runtime; mẫu số
  là mọi member-window; kèm số member-window có phân hoạch hợp lệ mà vẫn bị loại và số có phân hoạch
  không xác định mà bị loại), và member lỗi (mẫu số là mọi member-window). `share_floor_score` là tỉ lệ
  nhận điểm sàn vì bất kỳ lý do nào, đúng điều mà phép chọn thấy. Run ghi trước khi có các bộ đếm này
  để trống các tỉ lệ cần chúng, không suy từ `score_status`. Thêm: sentinel, fallback; số đọc snapshot
  và stability; số lần sinh và lần chấm.
- **Support ở cấp điểm** (`report/contrasts/point_support.csv`). Hai arm được ghép theo cửa sổ chung,
  không theo điểm chung. Từ 04/11 mỗi dòng cửa sổ ghi, cho incumbent trễ, vị trí của cửa sổ (`start`,
  `end`), số điểm đưa vào (`lag_n_points_input`), số điểm silhouette đọc và dấu vân tay của vị trí các
  điểm đó (`lag_n_points_scored`, `lag_support_sha`), số điểm ARI/NMI đọc và dấu vân tay của chúng
  (`lag_n_points_external`, `lag_external_support_sha`; SHA-256 của mặt nạ vị trí, không dùng `hash`
  của Python); log member ghi các trường đó không có tiền tố. Mỗi cửa sổ chung của một cặp replicate
  được xếp vào một trong ba nhóm: `identical` (cả hai bên có dấu vân tay, trùng nhau, cùng vị trí cửa
  sổ), `differs` (khác dấu vân tay, khác vị trí cửa sổ, hoặc các run của một arm cố định đọc cửa sổ trên
  tập điểm khác nhau), `undetermined` (một bên không có bản ghi, tức log trước 04/11). Cùng số điểm không
  bao giờ được coi là cùng tập điểm. Bảng ghi số cửa sổ của từng nhóm theo hiệu và nhóm luồng, kèm
  `status`: `verified identical` chỉ khi mọi cửa sổ chung đều `identical`. Phân tích chính vẫn trên các
  cửa sổ chung. Tập con `point_identical_only` là phân tích bổ sung chỉ trên các cửa sổ đã xác nhận cùng
  tập điểm; phần bị loại và độ phủ (`share_point_identical`) nằm ngay cạnh. Log cũ cho `undetermined`, và
  kết quả trên chúng không được gọi là ghép theo điểm. Không có phép tính lại nào trên giao các điểm.
  Theo đọc code, tám thuật toán của chiến dịch không trả dự đoán `None` (không thuật toán nào ném
  `NotImplementedError`, `drop_sentinel_predictions` tắt), và đếm hôm nay trên 100 000 mẫu đầu của 12
  luồng không có điểm nào thiếu nhãn thật (bộ sinh tổng hợp có thể sinh điểm nhiễu với nhãn `None`,
  nhưng cấu hình của bảy luồng tổng hợp không bật nó). Vì thế dự kiến ARI ở cả hai bên đọc cả cửa sổ;
  dấu vân tay trong log mới là thứ xác nhận điều đó, không phải dự kiến này.
- `selection --lobo` (thư mục `report/lobo/`, `--metric nmi` cho `report/lobo_nmi/`): chọn thuật toán
  trên nguồn thật, giữ ngoài nguyên một khối nguồn. Bốn bộ chọn `lobo_default`, `lobo_scaled_default`,
  `lobo_target` (theo `confstream`) và chọn đều (kỳ vọng chính xác) dùng chung một bảng huấn luyện:
  cùng các khối, luồng, cửa sổ và tập thuật toán ứng viên, tất cả lấy từ các khối huấn luyện. Mỗi khối
  huấn luyện có trọng số bằng nhau (biến thể trong khối lấy trung bình trước). `scaled_default` của
  HPStreamStableLabels và RepStream được điền từ `default`, có ghi nguồn, trước khi dựng bảng. Hoà đúng (sai khác
  dưới 1e−12) về thuật toán có tên đứng trước theo thứ tự chữ; nguồn kiểm tra không được đọc khi chọn
  hay phá hoà. Chỉ số chính: Q_confstream,test(a_lobo_target) − Q_confstream,test(a_lobo_scaled_default),
  theo khối giữ ngoài rồi trung bình qua khối. Có thêm lựa chọn ở từng fold, điểm huấn luyện, regret,
  kết quả theo nguồn, phép bỏ từng nguồn ra khỏi toàn bộ phân tích (có bỏ Shuttle) và biến thể
  `suffix_eval` (chỉ nửa sau cửa sổ của nguồn giữ ngoài) làm độ nhạy. Một lựa chọn không có số đọc trên
  nguồn kiểm tra giữ nguyên lựa chọn và có regret không xác định, không bị thay.
- **Bảng huấn luyện mong đợi và điều kiện của phân tích chính.** Các khối, luồng, thuật toán và regime
  mà LOBO cần được đọc từ kế hoạch trước khi đọc kết quả nào: `--campaign-config` nếu đưa, nếu không
  thì manifest chiến dịch trong thư mục (danh sách job, từng job một), nếu không nữa thì manifest của
  `run` (theo ô thuật toán × luồng × arm). Không có kế hoạch nào thì phân tích chính không bao giờ đủ
  điều kiện. Một `--campaign-config` được đối chiếu với manifest của thư mục (hash cấu hình của manifest
  chiến dịch, protocol của manifest `run`, và thiết lập của từng arm: E, C, p_reset, tie-break của arm
  online, tín hiệu ghi, CMM, gate); khác protocol, hay thư mục không có manifest `run`, cũng cho
  `ineligible`. Một kế hoạch có cả `HPStreamStableLabels` lẫn `HPStream` là `ineligible`: đó là cùng một
  mô hình dưới nhãn khác, và bộ chọn `random` cũng như trung bình qua thuật toán sẽ đếm nó hai lần;
  `report --contrasts` ghi cảnh báo tương tự. Manifest `run` chỉ ghi hợp các seed của mọi job, không theo thuật toán, nên với kế hoạch
  theo ô, mỗi ô chờ mọi replicate mà cặp (thuật toán, arm) của nó đã chạy trên bất kỳ luồng thật nào
  của kế hoạch: một tuner seed thiếu ở một luồng là một job thiếu (một seed không chạy ở đâu cả thì
  không phân biệt được với một seed không được lên kế hoạch). Chỉ các job của kế hoạch được đọc, và một
  cửa sổ chỉ thuộc support của một ô khi mọi replicate mong đợi của ô đọc nó; cửa sổ chỉ một phần
  replicate đọc bị bỏ khỏi ô đó, không lấy trung bình trên phần còn lại, và được đếm
  (`n_windows_partial_replicates`). Phép bỏ từng nguồn ra (LOSO) chỉ được đọc khi phân tích chính đủ
  điều kiện; nếu không, các dòng LOSO ghi `not read: primary not eligible`, vì bỏ đúng khối có khoảng
  trống chính là một bảng thu hẹp dưới tên khác. Một khối giữ ngoài chỉ chọn khi bảng huấn luyện của nó đủ: mọi ô (thuật toán, luồng, regime)
  của các khối huấn luyện có job ok với số đọc trên đuôi, và mỗi luồng huấn luyện có ít nhất một cửa sổ
  mà mọi ứng viên đọc dưới mọi regime. Thiếu một điều trong đó thì khối đó không chọn gì (`pick` trống),
  `panel_status = ineligible`, và `missing` kể từng ô hay luồng thiếu cùng lý do (`not on file`, `error`,
  `timeout`, `oom` với số job, `ok job without a tail reading`, không có cửa sổ chung). Trung bình qua các
  khối chỉ được tính khi mọi khối mong đợi đủ điều kiện và có số đọc kiểm tra; nếu không, `mean` để trống
  và `lobo_status.csv` ghi `ineligible`, các khối mong đợi, khối có khoảng trống, khối không có số đọc
  kiểm tra, regime thiếu và danh sách thiếu. Job kiểm tra thiếu hay thất bại không đổi lựa chọn hay tập
  ứng viên; chất lượng kiểm tra của khối đó là `unavailable`, kèm lý do. Không có số nào được điền thay
  cho job lỗi, hết giờ hay hết bộ nhớ. `--reduced-panel` thêm, chỉ khi được yêu cầu, một phân tích độ nhạy
  trong `report/lobo/reduced_panel/`: loại thuật toán có khoảng trống cho mọi bộ chọn như nhau và bỏ luồng
  không có cửa sổ chung, ghi rõ đã loại gì; nhãn của nó là `sensitivity: reduced panel` và không thay
  phân tích chính.
- `selection` không có `--lobo` vẫn là phép đọc P1.1 đã công bố. Dưới chính sách `split`, biến thể giữ
  ngoài seed chỉ được đọc khi đưa `--select-seeds` và `--eval-seeds` tường minh; seed 1/2/3 cũ không được
  mặc định.

Với 5 khối nguồn thật, khoảng tin cậy rất thô và kiểm định Wilcoxon hai phía chính xác không xuống
dưới p = 0,0625. Không có ý nghĩa thống kê không được đọc là tương đương.

## 4. Đã kiểm tra (04/11/2026, macOS 27.2, Python 3.13.3, river 0.26.1 ở `873fa6e0`)

**Đã implement và có test.** Bộ test tập trung (lệnh ở mục 5) chạy qua. Các test bám vào yêu cầu:

- quyết định cuối cửa sổ t không đổi dự đoán hay điểm ở t và chỉ được chấm ở t+1
  (`test_a_decision_after_window_t_moves_nothing_scored_on_t_and_is_read_on_t_plus_1`);
- bật hay tắt chẩn đoán không đổi dự đoán, điểm, lần sinh, quyết định; chẩn đoán lỗi được ghi và không
  đổi gì khác; bộ sinh số toàn cục được cô lập (`tests/test_confstream.py`);
- LOBO không đọc nguồn kiểm tra khi chọn hay phá hoà, cân mỗi khối một lần, điền `scaled_default` trước
  khi dựng bảng, một khoảng trống huấn luyện loại thuật toán cho mọi bộ chọn như nhau
  (`tests/test_conf_selection.py`);
- seed `None`/NaN, model seed 201, tuner seed 101/102 được ghép đúng; arm cố định không bị đếm như
  replicate theo tuner seed; hai arm online chỉ ghép khi trùng cả tuner seed lẫn model seed, và hai model
  seed dưới một tuner seed là hai replicate (`tests/test_conf_report.py`, `tests/test_conf_selection.py`);
- phân hoạch không xác định, bị giới hạn loại và member lỗi được đếm riêng, kể cả member-window có
  phân hoạch hợp lệ mà bị loại bởi giới hạn bộ nhớ (`tests/test_conf_report.py`, `tests/test_conf_pipeline.py`);
- danh sách job 996 + 132 + 24, lệnh không mang cờ bỏ qua, resume bị từ chối khi đổi protocol, timeout,
  code, package hay bỏ job, thư mục `out/full` không bao giờ được ghi (`tests/test_conf_campaign.py`);
- log lifecycle không làm hỏng `report`, `screen`, `oracle_compare`.

Vòng thứ hai (cùng ngày) thêm:

- LOBO: một trong hai khối huấn luyện mất cửa sổ chung, thiếu cả một khối, thiếu một regime, một job
  huấn luyện lỗi, không có kế hoạch, kế hoạch của protocol khác, đều cho `ineligible` và không chọn gì;
  đổi giá trị hay nhãn của nguồn kiểm tra, hoặc job kiểm tra thiếu hay lỗi (`error`, `timeout`, `oom`,
  không có dòng), không đổi lựa chọn hay tập ứng viên; bảng đầy đủ cho kết quả như thiết kế; phân tích
  bảng thu hẹp chỉ có khi được yêu cầu và nằm riêng (`tests/test_conf_selection.py`). Trên code trước khi
  sửa, ca "mất cửa sổ chung" chọn từ một khối huấn luyện duy nhất và in trung bình qua 3 khối như phân tích
  chính; đã tái hiện trước khi sửa.
- Support ở cấp điểm: cùng số điểm khác mặt nạ là `differs`, cùng mặt nạ là `identical`, log cũ là
  `undetermined`, cửa sổ khác vị trí và run cố định không cùng tập điểm là `differs`; mặt nạ đúng là tập
  điểm ARI của `run_once` đọc với `None`, sentinel, fallback, nhãn thật thiếu; dấu vân tay giống nhau giữa
  các tiến trình (`tests/test_point_support.py`).
- Audit nhãn: counterexample cho năm thuật toán, `HPStreamStableLabels` trùng HPStream từng điểm
  (`tests/test_partition_audit.py`, `tests/test_hpstream_stable_labels.py`); trạng thái đo giữ đủ tám
  thuật toán và không bao giờ cho `eligible` khi còn vấn đề mở (`tests/test_conf_campaign.py`).
- Sau review độc lập: LOSO không được đọc khi phân tích chính không đủ điều kiện; kế hoạch theo ô bắt
  được tuner seed thiếu ở một luồng; cửa sổ chỉ một phần replicate đọc bị bỏ khỏi ô và được đếm; biến
  thể nhãn được đọc dưới tên riêng khi một cấu hình đặt nó vào danh sách job; lớp con mang vấn đề của lớp
  River, thuật toán chưa audit chặn; hai arm cố định có vị trí cửa sổ `<mixed>` không bao giờ là cùng tập
  điểm; một run bỏ hẳn việc ghi tập điểm (thay bằng hàm rỗng) cho cùng quỹ đạo.

**Smoke đã chạy** (`conf_pipeline/out/mech_smoke_20261104_v4/`, trên code trước vòng thứ hai; các lần chạy trước ở `_v2` và `_v3` cho cùng kết quả): CluStream, RepStream, DenStream,
HPStream trên `static_easy`, sáu arm, h = 100, E = 3, C = 2, t0 = 3, 1 200 mẫu (12 cửa sổ), không gate,
tie-break `micro_clusters`, ba thư mục chạy trong các tiến trình mới: tắt chẩn đoán, bật chẩn đoán, bật
chẩn đoán lần hai. E = 3, C = 2 để mỗi lần thách đấu có hai nạn nhân để rút, khác với E = 2. So bằng
`repeat --strict --require-members`: tắt–bật 24/24 job trùng tuyệt đối, bật–bật 24/24 (22 run và hai
`scaled_default` cùng bị bỏ qua), so cả hash dự đoán của từng member, tham số, điểm, lifecycle.

**Smoke của vòng thứ hai** (`conf_pipeline/out/mech_smoke_20261104_v7/`, cùng lệnh, trên code cuối của vòng
này; `_v5` và `_v6`, trên code trước các sửa sau review, cho cùng kết quả): tắt–bật 24/24, bật–bật 24/24, và
`_v4/on` (code cũ) với `_v7/on` (code cuối) 22/22 run trùng tuyệt đối (hai `scaled_default` bị bỏ qua ở cả
hai bên): việc ghi tập điểm không đổi dự đoán, điểm, tham
số, quyết định hay lifecycle. Đây là tái lập trên một máy; **chưa kiểm tra giữa hai máy**.

**Smoke đọc kết quả trên luồng thật** (`conf_pipeline/out/mech_report_smoke_20261104_v3/` trên code cuối; `/` và
`_v2` trước đó cho cùng bảng): DenStream,
HPStream, SOStream, STREAMKMeans trên năm khối nguồn thật và `static_overlap`, sáu arm, 3 000 mẫu, h = 200,
E = 3, C = 2, t0 = 3, tuner seed 101; 138 job ok, 6 `scaled_default` bỏ qua. `report --contrasts` cho năm
hiệu với support ở cấp điểm `verified identical` trên mọi cửa sổ chung (log mới); `selection --lobo` đọc
kế hoạch từ manifest của `run` và cho phân tích chính `eligible` trên 5/5 khối; `--metric nmi
--reduced-panel` ghi thêm `reduced_panel/`. Các con số của smoke này là kiểm tra máy móc với E, h nhỏ,
không phải kết quả. Đọc cùng thư mục với `--campaign-config` của chiến dịch (kế hoạch 8 thuật toán × 12
luồng, protocol h = 1 000) cho `ineligible` trên 5/5 khối, 80 ô thiếu và các ô protocol lệch; đọc
`_v4/on` (log cũ) cho support `undetermined` trên mọi cửa sổ.

**Smoke của biến thể HPStream** (sau quyết định đọc HPStream qua `HPStreamStableLabels`; cùng lệnh, HPStream thay
bằng biến thể): `conf_pipeline/out/mech_smoke_20261104_v8/` cho tắt–bật 24/24 và bật–bật 24/24 trùng tuyệt đối;
so với `_v7` (HPStream của River), 17/17 job của ba thuật toán kia trùng, còn sáu job HPStream và sáu job của
biến thể nằm mỗi bên một. Trên smoke này, ở arm `default` (cùng một mô hình, chỉ khác nhãn) ARI đuôi trễ là
0,042 với HPStream và 0,049 với biến thể; ở các arm online là khoảng 0,005–0,024 so với khoảng 0,10, vì silhouette
tuần tự trên nhãn không dùng lại chọn khác. Đây là kiểm tra cơ chế trên `static_easy` với E = 3, không phải
kết quả. `conf_pipeline/out/mech_report_smoke_20261104_v4/` (luồng thật, biến thể thay HPStream): 138 job ok,
LOBO đủ điều kiện trên 5/5 khối với biến thể là một ứng viên dưới tên riêng, mọi hiệu `verified identical`.

**Chi phí của chẩn đoán** (`conf_pipeline/out/mech_smoke_20261104_v2/overhead/`): tắt và bật chẩn đoán trên cả tám thuật toán cho 8/8 job trùng tuyệt đối; hệ số thời gian ở mục 6.

**Dry run:** `conf_pipeline.campaign --config conf_pipeline/campaigns/mechanism_pilot.json` in 1 128
job chạy và 24 job bỏ qua, các lệnh, trạng thái đo theo thuật toán (`paper confirmation NOT ELIGIBLE`), và
không ghi gì vào `conf_pipeline/out/mech_campaign`. `--help`
của `campaign`, `run`, `selection`, `report`, `repeat`, `cost`, `partition_audit` chạy được.

**Audit phân hoạch tuần tự:** `docs/SEQUENTIAL_PARTITION_AUDIT.md` và RIVER_BUGS.md GAP-6.

## 5. Lệnh

Kiểm tra (từ thư mục framework):

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q \
    tests/test_tuning_window.py tests/test_tuning_signals.py tests/test_confstream.py tests/test_conf_pipeline.py \
    tests/test_conf_seeds.py tests/test_conf_selection.py tests/test_conf_report.py tests/test_conf_manifest.py \
    tests/test_conf_repeat.py tests/test_conf_cost.py tests/test_conf_screen.py tests/test_conf_oracle_compare.py \
    tests/test_sentinel_predictions.py tests/test_denstream_patched.py tests/test_conf_campaign.py \
    tests/test_partition_audit.py tests/test_point_support.py tests/test_hpstream_stable_labels.py
```

Smoke tái lập (đặt `S` là một thư mục mới; chạy lại trên máy thứ hai cùng commit vào `S2` rồi so `S/on`
với `S2/on` để có kiểm tra giữa máy):

```bash
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S=conf_pipeline/out/mech_smoke_$(date +%Y%m%d_%H%M)
COMMON="--algos CluStream,RepStream,DenStream,HPStreamStableLabels --folds static_easy \
  --arms default,scaled_default,confstream,frozen_prefix,frozen_reinit,random_online \
  --seed-policy split --tuner-seeds 101 --model-seed 201 --fixed-model-seeds 201 --n-samples 1200 --h 100 \
  --e-size 3 --e-test 2 --freeze-after-window 3 --no-runtime-gate --tie-break micro_clusters --log-members \
  --workers 4 --timeout 900"
.venv/bin/python -m conf_pipeline.run $COMMON --out-dir $S/off
.venv/bin/python -m conf_pipeline.run $COMMON --signals stability --out-dir $S/on
.venv/bin/python -m conf_pipeline.run $COMMON --signals stability --out-dir $S/on_repeat
.venv/bin/python -m conf_pipeline.repeat --a $S/off --b $S/on --out $S/report/off_vs_on --strict --require-members
.venv/bin/python -m conf_pipeline.repeat --a $S/on --b $S/on_repeat --out $S/report/on_vs_on_repeat --strict --require-members
```

Trong zsh, `$COMMON` không tự tách thành từng đối số: chạy khối trên bằng `bash`, hoặc viết `${=COMMON}`.

Smoke đọc kết quả trên luồng thật (đặt `R` là một thư mục mới; vài phút trên 6 worker):

```bash
R=conf_pipeline/out/mech_report_smoke_$(date +%Y%m%d_%H%M)
.venv/bin/python -m conf_pipeline.run --algos DenStream,HPStreamStableLabels,SOStream,STREAMKMeans \
  --folds covtype,kdd99,electricity,shuttle,insects_abrupt_imbalanced,static_overlap \
  --arms default,scaled_default,confstream,frozen_prefix,frozen_reinit,random_online \
  --seed-policy split --tuner-seeds 101 --model-seed 201 --fixed-model-seeds 201 --n-samples 3000 --h 200 \
  --e-size 3 --e-test 2 --freeze-after-window 3 --no-runtime-gate --tie-break micro_clusters --log-members \
  --workers 6 --timeout 900 --out-dir $R
.venv/bin/python -m conf_pipeline.report --out-dir $R --contrasts
.venv/bin/python -m conf_pipeline.selection --out-dir $R --lobo
.venv/bin/python -m conf_pipeline.selection --out-dir $R --lobo --metric nmi --reduced-panel
```

Dry run, trạng thái và dự toán:

```bash
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.campaign --config conf_pipeline/campaigns/mechanism_pilot.json
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.campaign --config conf_pipeline/campaigns/mechanism_pilot.json \
    --write-inventory /tmp/mech_campaign_inventory.json
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.cost --from conf_pipeline/out/full \
    --campaign conf_pipeline/campaigns/mechanism_pilot.json \
    --proxy scaled_default=default,frozen_reinit=confstream,random_online=confstream \
    --arm-factor confstream=1.2,frozen_prefix=1.2,frozen_reinit=1.2,random_online=1.2 \
    --algo-proxy HPStreamStableLabels=HPStream \
    --algo-factor DenStream=1.50,CluStream=1.05,HPStreamStableLabels=1.08,RepStream=1.08,DBSTREAM=1.36,STREAMKMeans=1.19,SOStream=1.13,BIRCH=1.43 \
    --stat mean --workers 8 --out /tmp/mech_campaign_cost_mean.csv
```

Đổi `--stat mean` thành `--stat max` để đọc lần chạy chậm nhất của mỗi ô (mục 6).

Chạy đầy đủ (máy chủ, từ một bản sao code cố định; không sửa cây làm việc trong lúc chạy vì mỗi job
kiểm hash code với manifest):

```bash
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.campaign --config conf_pipeline/campaigns/mechanism_pilot.json \
    --execute --workers 8
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.campaign --config conf_pipeline/campaigns/mechanism_pilot.json --status
```

Chạy lại cùng lệnh `--execute` là resume: chỉ được nhận khi protocol, giới hạn vận hành, hash code và
khoá package trùng với manifest, và danh sách job mới chứa mọi job cũ. Không có cờ nào bỏ qua kiểm tra
này; `--force`, `--allow-code-change`, `--adopt-legacy` không bao giờ được chuyển cho `run`.

Đọc kết quả:

```bash
D=conf_pipeline/out/mech_campaign
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.report --out-dir $D --contrasts
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.selection --out-dir $D --lobo
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.selection --out-dir $D --lobo --metric nmi
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.selection --out-dir $D --selecting-arm scaled_default
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.report --out-dir $D
```

## 6. Dự toán chi phí

Đây là dự đoán, không phải đo. Nền là thời gian tiến trình của `out/full` (máy chủ Linux, có gate,
chính sách `shared`), đọc qua đúng danh sách 1 128 job của chiến dịch (`cost --campaign`), không phải tích
Descartes của arm và seed. Các giả định:

- Arm chưa từng chạy trong `out/full` đọc qua arm gần nhất: `scaled_default` qua `default`,
  `frozen_reinit` và `random_online` qua `confstream`.
- `HPStreamStableLabels` chưa từng chạy trong `out/full`; nó được đọc qua chi phí đo của HPStream
  (`--algo-proxy HPStreamStableLabels=HPStream`), và hệ số stability 1,08 đo trên HPStream được áp cho nó.
  Hai mô hình học như nhau, nhưng nhãn khác có thể đổi silhouette tuần tự và từ đó đổi lựa chọn của bộ tune,
  nên chi phí của các arm online có thể khác; đây là giả định.
- Bỏ gate nhân 1,2 cho bốn arm online: phần tăng đo được ở smoke ngày 23/10 (377 giây lên 449 giây,
  sáu thuật toán rẻ, 20 000 mẫu, `docs/MECHANISM_PILOT.md`). Hệ số này chưa được đo cho CluStream và
  RepStream, nên với hai thuật toán đó nó chỉ là giả định.
- Chẩn đoán `stability` (dự đoán lại cửa sổ trên bản sao và silhouette của snapshot) nhân theo thuật toán,
  đo hôm nay trên một cặp chạy ngắn, cùng máy, một worker, tắt rồi bật chẩn đoán
  (`static_overlap`, 6 000 mẫu, h = 500, E = 6, C = 3, t0 = 4, `confstream`, tuner seed 101):

  | Thuật toán | tắt (s) | bật (s) | hệ số | phần thời gian của tín hiệu khi bật (s) |
  |---|---:|---:|---:|---:|
  | DenStream | 2,8 | 4,1 | 1,50 | 1,4 |
  | CluStream | 107,0 | 111,8 | 1,05 | 3,4 |
  | HPStream | 9,5 | 10,3 | 1,08 | 1,0 |
  | RepStream | 25,9 | 28,0 | 1,08 | 1,7 |
  | DBSTREAM | 2,9 | 3,9 | 1,36 | 0,4 |
  | STREAMKMeans | 2,8 | 3,3 | 1,19 | 0,5 |
  | SOStream | 2,7 | 3,0 | 1,13 | 0,4 |
  | BIRCH | 26,6 | 38,0 | 1,43 | 11,2 |

  Hệ số được áp cho mọi arm của thuật toán đó, cả arm cố định (một member, cũng ghi stability). Ngoại suy
  sang h = 1 000, E = 20, C = 10 là giả định: số member và số điểm tăng như nhau ở hai bên, nhưng silhouette
  tốn O(h²) mỗi member-window và snapshot thêm một lần tính như vậy, nên ở h = 1 000 hệ số có thể lớn hơn.
  Cặp chạy này cũng cho dự đoán và quyết định trùng tuyệt đối (8/8 job, `repeat --strict`).

Kết quả (`/tmp/mech_campaign_cost_{mean,max}.csv` theo lệnh ở mục 5):

| Cách đọc ô đã đo | Giờ tiến trình | Wall 8 worker (xếp job dài trước) | Job dự đoán quá 43 200 s |
|---|---:|---:|---:|
| trung bình các run | 293 | 36,7 h | 0 |
| run chậm nhất | 322 | 40,3 h | 0 |

Theo thuật toán (trung bình): CluStream 94,6 giờ, RepStream 81,4, SOStream 27,5, DenStream 27,0, BIRCH 22,2,
STREAMKMeans 14,8, DBSTREAM 13,4, HPStreamStableLabels 12,4 (chi phí đo của HPStream, `--algo-proxy`). Job dài nhất là CluStream trên kdd99 ở ba arm
`confstream`, `frozen_reinit`, `random_online`: 8,0 giờ theo trung bình, 9,6 giờ theo run chậm nhất
(`out/full` ghi 7,59 giờ cho run chậm nhất của ô này). Biên tới 12 giờ là hệ số 1,25: nếu bỏ gate làm
CluStream trên kdd99 chậm hơn khoảng 1,5 lần thay vì 1,2, các job đó sẽ chạm timeout. Đó là rủi ro thực
thi chưa giải quyết; timeout không được nâng theo dự toán hay theo kết quả.

## 7. Mở rộng

- **Thêm tuner seed:** sửa `tuner_seeds` của nhóm (ví dụ 101, 102, 103 cho CluStream, RepStream) rồi
  `--execute` vào cùng thư mục; manifest ghi đây là lần mở rộng và chỉ chạy các job mới.
- **Thêm arm cùng protocol** (ví dụ hai oracle E4): thêm vào `arms`, như trên.
- **Data seed mới cho luồng tổng hợp, nguồn thật mới, điều kiện tiền xử lý khác:** là protocol khác.
  Chép cấu hình, đổi `name` và `out_dir` (một thư mục cho mỗi data seed hay điều kiện), và đổi giá trị
  cần đổi. `data_seed` đang khóa theo hằng số repo; một data seed khác cần bỏ nó khỏi
  `locked_to_repo_constants` trong bản chép, một quyết định ghi rõ trong cấu hình. Nguồn thật mới (ví dụ
  rialto, sensor) cần có trong `_common` và qua kiểm tra label-reach trước. Chưa tải dữ liệu nào.

## 8. Chưa giải quyết

- **Nhãn tuần tự đổi nghĩa trong một cửa sổ** ở BIRCH, CluStream, DBSTREAM, DenStream; ở HPStream đã giải
  quyết bằng biến thể chạy thay nó (`docs/SEQUENTIAL_PARTITION_AUDIT.md` mục 6, RIVER_BUGS.md GAP-6). Mỗi
  thuật toán có một counterexample nhỏ trên chính lớp của River (`tests/test_partition_audit.py`,
  `tests/test_hpstream_stable_labels.py`). Trạng thái đo theo thuật toán nằm trong
  `campaign.MEASUREMENT_ISSUES`; dry run in nó, manifest chiến dịch ghi nó (`measurement_status`), `--status`
  in lại. Với bốn thuật toán DenStream, BIRCH, DBSTREAM, CluStream trạng thái là `open` (HPStreamStableLabels
  là `resolved`), nên `paper_confirmation` vẫn là `not eligible`. Không thuật toán nào bị bỏ khỏi danh sách job, và dry run, chạy
  và chẩn đoán vẫn làm được.

  | Thuật toán | Cơ chế | Cách sửa chính xác trong framework | Điều kiện gỡ chặn |
  |---|---|---|---|
  | HPStream | `_free_label` trao nhãn của cụm vừa bị loại cho cụm mới | có: `HPStreamStableLabels`, cùng cụm và cùng dự đoán ở mọi điểm, nhãn không bao giờ dùng lại (test) | đã gỡ: người dùng quyết định 04/11 chạy biến thể thay HPStream; trạng thái `resolved` |
  | DenStream | đánh số lại macro-cluster mỗi lần `predict_one`; xoá p-micro-cluster đứng trước làm các cụm sau lùi số | chưa có: macro-cluster dựng lại từ đầu, không mang danh tính | một quy tắc đặt tên (gộp, tách) do người dùng chọn, đăng ký thành biến thể và có test; hoặc báo cáo kèm giới hạn |
  | BIRCH | nhãn là vị trí trong lá; một entry chèn trước đẩy mọi nhãn sau | chưa có: entry gộp nhau và cây được dựng lại | như trên, cho danh tính của entry |
  | DBSTREAM | đánh số lại ở mỗi lần phân cụm lại; dọn một micro-cluster làm số đổi | chưa có | như DenStream |
  | CluStream | một k-means mới mỗi `time_gap` điểm | chưa có: mỗi k-means là mới | như DenStream |
  | RepStream, SOStream, STREAMKMeans | nhãn từ bộ đếm tăng dần hoặc khoá bền | không cần | không có vấn đề đã biết |

  Một lớp con của framework mang vấn đề của lớp River nó mở rộng (DenStreamPatched, DenStreamPatchedFast
  như DenStream; CluStreamPyramidal như CluStream); `HPStreamStableLabels` là `resolved`; một thuật toán
  chưa được audit là `not audited` và chặn như một vấn đề mở. `HPStreamStableLabels` nằm trong
  `_common.LABEL_VARIANTS`: không xếp hạng trong các bảng OTB, không vào preset nào của notebook, và nếu
  một cấu hình chiến dịch đặt nó vào danh sách job thì LOBO và các hiệu đọc nó dưới tên riêng.

  Tách ba trường hợp: nhãn đổi **giữa hai cửa sổ** không ảnh hưởng silhouette tuần tự hay `lag_ari_win`,
  vì mỗi cửa sổ được đọc riêng (chỉ `lag_ari_cumulative` của báo cáo chung ghép nhãn qua các cửa sổ, và
  chiến dịch không đọc nó); nhãn **hoán vị hay dùng lại trong một cửa sổ** làm phân hoạch tuần tự ghép
  những điểm mô hình chưa từng gộp; phân hoạch **đổi thật** (học, gộp, tách) là điều phân hoạch tuần tự
  cần ghi lại. Stability thấp hay ghép Hungarian tốt hơn không được dùng làm bằng chứng của trường hợp
  thứ hai.
- **DenStream của River ghi trạng thái khi dự đoán** (BUG-7). Arm `DenStream` của chiến dịch vẫn là bản
  River nguyên gốc, như `out/full`.
- **Chưa kiểm tra tái lập giữa hai máy**, và chưa có smoke trên luồng thật.
- **Timeout 12 giờ chưa chắc đủ** khi bỏ gate và bật stability; xem mục 6. Job quá giờ được ghi là
  timeout, không được nâng giới hạn theo kết quả.
- **Giới hạn bộ nhớ của tiến trình con không được áp trên macOS**; trên Linux manifest ghi việc áp có
  thành công không.
