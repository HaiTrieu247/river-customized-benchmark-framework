# Quy tắc về seed trong conf_pipeline

Cập nhật 24/10/2026, sau khi tách seed (spec: `docs/superpowers/specs/2026-10-24-seed-split-design.md`).
Mục 1–8 mô tả code đang làm gì, đối chiếu với mã nguồn, và mỗi điểm có test trong
`tests/test_conf_seeds.py`. Mục 9 mô tả chính sách cũ, cần để đọc log làm trước 24/10. Mục 10 là
những câu hỏi còn mở.

## 1. Hai chính sách seed

Mỗi out dir chạy dưới đúng một chính sách. Manifest ghi chính sách đó (`protocol.seed_policy`,
schema 2), và từ chối mọi lệnh khác chính sách.

- **`split`**, mặc định cho mọi thư mục mới: bộ tune, mô hình và bước hiệu chuẩn đơn vị khoảng cách s
  mỗi thứ có seed riêng.
- **`shared`**: một job seed điều khiển cả ba, như mọi run làm trước 24/10 (`out/full`, `out/pilot`,
  `out/smoke`). Chỉ dùng để chạy cạnh hoặc tái lập các run đó: `--seed-policy shared --seeds ...`.
  Hai chỗ trong code vẫn chạy theo chính sách này:
  - notebook `otb_benchmark.ipynb`, vì nó dựng lại protocol của `out/full`;
  - các biến thể X6 của `variants run`, vì chúng so với `out/full`.

## 2. Các loại seed dưới chính sách `split`

| Seed | Cờ | Mặc định | Điều khiển | Không điều khiển |
|---|---|---|---|---|
| tuner seed | `run --tuner-seeds` | 101, 102, 103 | mọi lần rút của bộ tune: quần thể ban đầu, nạn nhân, cha, reset, đột biến | mô hình, dữ liệu, s |
| model seed của arm online | `run --model-seed`, `sweep --model-seed` | 201, một giá trị cho mỗi out dir | `seed=` truyền vào constructor của mọi member, chỉ ở CluStream, HPStream, STREAMKMeans | dòng số của bộ tune (xem chú thích dưới bảng) |
| model seed của arm cố định | `run --fixed-model-seeds` | bằng `--model-seed` | các lần chạy lặp của `default`, `scaled_default`, `oracle_static` ở 3 thuật toán nhận seed | |
| model seed của lượt quét | `sweep --model-seeds` | bằng `--model-seed` | các lần chạy lặp của mỗi điểm quét ở 3 thuật toán nhận seed | |
| calibration seed | hằng `_common.CALIBRATION_SEED` | 0 | lấy mẫu con khi ước lượng s; giống nhau ở mọi arm và mọi seed | |
| seed Sobol | `sweep --sobol-seed` | 1 | các điểm của thiết kế quét | lần chạy của từng điểm |
| data seed | `--data-seed`, một giá trị cho mỗi out dir | 42 | nội dung luồng tổng hợp | luồng thật; mọi thứ khác |
| seed phân tích | `robustness --boot-seed`; `label_reach` | 0 | bootstrap; chẩn đoán | mọi run |

Chú thích về độc lập: model seed không rút từ dòng số của bộ tune, nên cùng tuner seed thì quần thể ban đầu
giống hệt nhau. Nhưng bộ tune chọn nạn nhân và cha theo điểm của các member, mà điểm phụ thuộc đầu ra của mô
hình, nên ở thuật toán nhận seed, đổi model seed vẫn đổi những cấu hình được rút từ cửa sổ thứ hai trở đi.
Nguồn ngẫu nhiên tách nhau; quỹ đạo thì không.

Ngoài seed, có hai thiết lập cố định cần cho tái lập: `PYTHONHASHSEED=0` và một luồng cho mỗi thư viện
số. Cả hai được ép cho mọi tiến trình con. Hash seed cần vì KMeans của River phụ thuộc thứ tự băm
(RIVER_BUGS.md BUG-2).

Dải giá trị mới (101–103, 201) cố ý khác dải cũ (1–3), để nhìn là biết một dòng thuộc chính sách nào.
`--seeds` (job seed) và `sweep --seed` chỉ hợp lệ dưới `shared`; dưới `split` chúng bị từ chối kèm
hướng dẫn, không âm thầm đổi nghĩa.

## 3. Job nào được chạy

| Arm | CluStream, HPStream, STREAMKMeans | BIRCH, DBSTREAM, DenStream, RepStream, SOStream |
|---|---|---|
| tune online (`confstream`, `frozen_prefix`, `frozen_reinit`, `random_online`, arm tín hiệu, oracle) | một run mỗi tuner seed, mô hình trên model seed của out dir | một run mỗi tuner seed |
| cấu hình cố định (`default`, `scaled_default`, `oracle_static`) | một run mỗi model seed cố định | **một run** |
| điểm quét (`sweep`) | một run mỗi model seed của lượt quét | **một run** |

Một cấu hình cố định không rút gì từ bộ tune. Thuật toán tất định chạy lại sẽ cho y hệt, nên chỉ chạy
một lần (`run.job_seeds`).

## 4. Seed hoạt động thế nào trong một job

- **Bộ tune:** `np.random.default_rng(tuner_seed)`, rút theo thứ tự:
  1. 19 cấu hình con ban đầu quanh cấu hình khởi đầu: một lần rút chọn cha, rồi một lần rút cho mỗi tham số;
  2. mỗi cửa sổ: một nạn nhân cho mỗi cấu hình thử thách;
  3. mỗi cấu hình con mới: một cha, một lần rút xem có reset không, rồi các tham số.

  Một dòng số dùng chung cho cả quá trình tìm kiếm. Vì vậy cùng tuner seed nhưng khác E, C hay h là hai
  đường đi không so từng bước được.
- **Mô hình:** model seed được truyền vào mọi member của ensemble, thay cho tuner seed như trước. Mọi
  member dùng chung một model seed. Bên trong River:
  - CluStream truyền nó cho KMeans của pha offline;
  - STREAMKMeans chuyển nó cho các KMeans nó tạo;
  - HPStream dùng nó để chọn tâm ban đầu của k-means khởi tạo.
- **Hiệu chuẩn:** `np.random.default_rng(CALIBRATION_SEED)`. Mẫu con chỉ được rút khi các cửa sổ hiệu
  chuẩn có hơn 2 000 mẫu, tức khi s = 0 trên hai cửa sổ đầu. Vì nó không đụng tới dòng số của bộ tune,
  s giống nhau ở mọi arm và mọi tuner seed. Ở cả 1 944 run của `out/full`, s > 0 ngay cửa sổ đầu.
- **Lần rút phụ thuộc điểm số:** trọng số chọn nạn nhân và chọn cha lấy từ điểm của các member.
  - Còn runtime gate thì điểm phụ thuộc thời gian đo được, nên đường đi không tái lập. Smoke 23/10:
    7/12 job lệch, tail ARI lệch tới 0,40.
  - Không gate thì đường đi tất định. Cùng smoke: 12/12 job trùng tuyệt đối.
- **Các arm cùng tuner seed khởi đầu như nhau:**
  - `frozen_prefix` đi giống hệt `confstream` cho đến cửa sổ t0;
  - các arm tín hiệu và hai oracle tách ra ở lần chọn đầu tiên cho kết quả khác.

## 5. Dòng kết quả và khóa job

- Mỗi dòng (run, cửa sổ, member) ghi `seed_policy`, `tuner_seed`, `model_seed`, `calibration_seed`.
- Cột `seed` là **chỉ số lần lặp bên trong arm**:
  - tuner seed với arm tune online;
  - model seed với arm cố định và điểm quét của thuật toán nhận seed;
  - trống với arm cố định và điểm quét của thuật toán tất định. Member log của run này có tên `seedx`.
- Khóa job gồm cả model seed. Một dòng cũ không có trường này được hiểu như chính sách cũ đã làm: model
  seed bằng job seed ở thuật toán nhận seed, trống ở thuật toán tất định.
- Manifest ghi các tuner seed và model seed đã chạy (`seeds.tuner`, `seeds.model`). Đổi model seed của
  arm online, hay đổi chính sách, là một protocol khác và phải vào out dir mới.

## 6. Lượt quét và reference

- **Thiết kế Sobol:** `--sobol-seed` chọn thiết kế. Dãy Sobol của một seed ổn định theo tiền tố, nên tăng
  `--n-points` chỉ thêm điểm.
- **Chọn reference:** điểm của thuật toán nhận seed được chạy trên các model seed của lượt quét, và
  reference được chọn theo trung bình qua chúng (`_common.seed_means`). `oracle_static.json` ghi các
  model seed đó vào `select_seeds`.
- **Chạy lại reference:** `run --arms oracle_static` dưới `split` bị **từ chối** khi một model seed chạy
  lại trùng với `select_seeds`, vì reference không được đọc trên seed đã chọn ra nó. Thuật toán tất định
  chạy lại một lần, và bản chạy lại đó trùng điểm quét. Muốn bỏ độ lạc quan của một lần chọn trên nhiều
  ứng viên thì phải tách theo cửa sổ (A2 trong `robustness`): chọn trên cửa sổ chẵn, đọc trên cửa sổ lẻ.

## 7. Phân tích: một run làm một lần được đọc một lần

- **Ghép cặp với arm cố định.** Dưới `split`, seed của arm cố định không phải tuner seed, nên không ghép
  cặp nó với arm online theo số seed được. `report.replicate_view` lấy trung bình mỗi (thuật toán, luồng)
  của arm cố định qua các run thật, rồi đặt giá trị đó cạnh mọi tuner seed của arm online, kèm cờ
  `broadcast` và `n_runs` là số run thật. Hàm này được dùng ở:
  - bootstrap và các kiểm định của `robustness`, kể cả các bảng đọc theo cửa sổ;
  - các phép tách seed và tách thời gian của `selection`;
  - `variants.arm_means`.
- **Thư mục `shared`:** không đổi gì, nên các bảng đã công bố từ `out/full` giữ nguyên (`robustness_check`:
  362/362 số khớp).
- **Bảng tóm tắt** ghi `replicates` (`tuner seeds`, `model seeds`, `none: deterministic, run once`, hoặc
  `job seeds (tuner and model)` với log cũ) và `n_runs`.
- **Đơn vị thống kê** là luồng, hoặc khối nguồn (`_common.source_block`). Seed không bao giờ thêm đơn vị.
  Một arm cố định tất định góp đúng một giá trị, không có độ dao động theo seed. Khoảng tin cậy phản ánh
  sự thay đổi giữa các luồng; độ dao động theo tuner seed hay model seed chỉ đi vào qua các lần rút lại
  bên trong mỗi cặp.

## 8. Quy tắc hiện hành

1. Seed là số nguyên tường minh trên dòng lệnh. Manifest và mỗi dòng kết quả ghi từng loại seed.
2. Mọi tiến trình con chạy với `PYTHONHASHSEED=0` và một luồng cho mỗi thư viện số.
3. Mọi arm tune online trong một phép so dùng cùng tập tuner seed và cùng model seed.
4. Không đọc reference trên model seed đã chọn ra nó (code từ chối). Với thuật toán tất định, phải tách
   theo cửa sổ (A2).
5. Mỗi out dir một chính sách seed, một model seed cho arm online, một data seed, một seed Sobol. Manifest
   từ chối trộn.
6. Một cấu hình cố định của thuật toán tất định chạy một lần, và được đọc là một run.
7. Seed phân tích cố định và được ghi ra.
8. Tái lập tuyệt đối chỉ được hứa dưới protocol không gate, cùng hash code và cùng hash dữ liệu (manifest
   kiểm cả ba).
9. Log cũ giữ nhãn "bộ tune và mô hình dùng chung job seed" (`shared`). Không đổi tên chúng thành kết
   quả đã tách seed.

## 9. Chính sách cũ (`shared`): log trước 24/10

Một job seed s tạo `np.random.default_rng(s)` cho bộ tune và cho bước hiệu chuẩn, và cũng chính s được
truyền vào constructor của mô hình ở 3 thuật toán nhận seed. Mọi arm, kể cả arm cố định, chạy trên mọi
job seed. Với 5 thuật toán tất định, các run của arm cố định vì thế lặp lại y hệt; chạy lại reference trên
seed 2–4 cũng trùng seed 1 tuyệt đối. Một test (`tests/fixtures/seed_shared_golden.json`, chụp từ code
trước khi sửa) giữ cho chính sách này cho ra đúng như cũ.

## 10. Câu hỏi còn mở

Số liệu từ `out/full` (có gate):
- Bộ tune online ổn định về chất lượng qua seed: độ lệch chuẩn tail ARI có trung vị 0,004–0,009.
- Nhưng các seed dừng ở những bộ tham số rất khác nhau: tham số khoảng cách chênh trung vị khoảng 2 lần.
- `frozen_prefix` phụ thuộc seed nhiều hơn: 21–35% số ô có độ lệch chuẩn vượt 0,05.

Các câu hỏi:
1. **Seed mô hình riêng cho từng member:** hiện mọi member dùng chung một model seed.
2. **Nhiều model seed cho arm online trong một thư mục:** hiện mỗi thư mục chỉ một, vì khóa
   (thuật toán, luồng, seed, arm) của báo cáo sẽ trùng.
3. **Data seed mới cho luồng tổng hợp:** dùng để xác nhận sau khi chốt phương án.
4. **Luồng thật:** không có data seed. Xáo thứ tự phá drift; cắt đoạn khác thì đổi cả mức drift.
