# Tách seed của bộ tune, của mô hình và của hiệu chuẩn

Ngày 24/10/2026. Nguồn: đề xuất sáu việc của user (24/10). Spec này giữ nguyên đề xuất và chốt
những chỗ nó chưa nói (đánh dấu **[chốt]**). Phạm vi: chỉ sửa cách tạo, tách và ghi nhận seed;
không tăng số lần chạy, không đổi dữ liệu, không đổi thuật toán hay không gian tìm kiếm.

## 1. Mục tiêu và tiêu chí xong

- Bộ tune, mô hình và bước hiệu chuẩn đơn vị s mỗi thứ có nguồn ngẫu nhiên riêng, chỉnh được độc lập.
- Mỗi cấu hình cố định chạy bao nhiêu lần là đúng bấy nhiêu lần cần: thuật toán tất định chạy một lần.
- Mỗi dòng kết quả và manifest ghi rõ từng loại seed, số lần chạy thật, và cái gì tạo ra độ dao động.
- Log cũ giữ nguyên nhãn "bộ tune và mô hình dùng chung job seed"; chế độ cũ vẫn tái lập được.

Xong khi: có test cho thấy các nguồn ngẫu nhiên độc lập (mục 7), chế độ cũ cho kết quả trùng tuyệt
đối với code trước khi sửa, các báo cáo chạy được trên thư mục kiểu mới mà không đếm một lần chạy
thành nhiều quan sát, và `docs/SEED_RULES.md` mô tả chính sách mới.

## 2. Chính sách seed: hai phiên bản protocol

**[chốt]** Manifest có thêm `protocol.seed_policy`, và schema lên 2:

- `split` (mặc định cho mọi thư mục mới): tuner seed, model seed và calibration seed tách nhau.
- `shared`: hành vi cũ, một job seed điều khiển cả ba. Chỉ dùng để chạy cạnh hoặc tái lập `out/full`;
  cờ cũ `--seeds` chỉ hợp lệ trong chế độ này.

Dòng kết quả không có `seed_policy` (mọi log trước 24/10) được đọc là `shared`. Thư mục tạo từ trước
với manifest schema 1 không resume được bằng code mới (khác protocol); cần thư mục mới.

## 3. Các loại seed và cờ

| Seed | Cờ | Mặc định | Điều khiển |
|---|---|---|---|
| tuner seed | `run --tuner-seeds` | 101, 102, 103 | mọi lần rút của bộ tune (quần thể ban đầu, nạn nhân, cha, reset, đột biến) |
| model seed của arm online | `run --model-seed` | 201 | `seed=` truyền vào constructor của mọi member, chỉ ở 3 thuật toán nhận seed; một giá trị cho mỗi out dir |
| model seed của arm cố định | `run --fixed-model-seeds` | bằng `--model-seed` | các lần chạy lặp của cấu hình cố định ở 3 thuật toán nhận seed |
| calibration seed | hằng `_common.CALIBRATION_SEED` | 0 | lấy mẫu con khi ước lượng s; giống nhau ở mọi arm và mọi seed |
| Sobol seed | `sweep --sobol-seed` | 1 | các điểm của thiết kế |
| model seed của lượt quét | `sweep --model-seeds` | 201 | các lần chạy lặp của mỗi điểm quét ở 3 thuật toán nhận seed |
| data seed | `--data-seed` (giữ nguyên) | 42 | luồng tổng hợp |
| seed phân tích | giữ nguyên | 0 | bootstrap, chẩn đoán |

Giải thích các chỗ chốt:
- **[chốt]** Dải giá trị mới (101–103, 201) khác dải cũ (1–3), để nhìn là biết dòng nào thuộc chính sách nào.
- **[chốt]** Mỗi out dir một model seed cho arm online, như data seed. Nhiều model seed cho arm online sẽ
  làm khóa (thuật toán, luồng, seed, arm) trùng nhau trong mọi báo cáo; để sau.
- **[chốt]** `sweep --seed` đổi tên thành `--sobol-seed`. Tên cũ bị từ chối kèm hướng dẫn, không âm thầm
  đổi nghĩa.
- **[chốt]** Mọi member trong một ensemble vẫn nhận cùng model seed, như hiện nay. Seed riêng cho từng
  member là một thay đổi khác, chưa làm.

## 4. Job nào được chạy

Gọi T là tập tuner seed, M là model seed của arm online, F là các model seed của arm cố định.

| Arm | Thuật toán nhận seed (CluStream, HPStream, STREAMKMeans) | Thuật toán tất định (5 còn lại) |
|---|---|---|
| online (`confstream`, `frozen_prefix`, `frozen_reinit`, `random_online`, arm tín hiệu, oracle) | T × {M} | T (model seed không áp dụng) |
| cố định (`default`, `scaled_default`, `oracle_static`) | F, tuner seed không áp dụng | **một job** |
| `sweep` (mỗi điểm) | các model seed của lượt quét | **một job** |

**[chốt]** `oracle_static` trong chế độ `split`: lệnh bị từ chối khi các model seed chạy lại trùng với
model seed đã dùng để chọn reference (`select_seeds` trong `oracle_static.json`). Đây là quy tắc 4 của
SEED_RULES, nay được kiểm bằng code. Thuật toán tất định chạy lại một lần; bản chạy lại đó trùng điểm
quét, và báo cáo ghi rõ điều này.

## 5. Dòng kết quả và khóa job

- Mỗi dòng có thêm `seed_policy`, `tuner_seed`, `model_seed`, `calibration_seed`.
- **[chốt]** Cột `seed` giữ nghĩa *chỉ số lần lặp bên trong arm*, để mọi phép trung bình theo seed hiện có
  vẫn đúng: tuner seed với arm online; model seed với arm cố định của thuật toán nhận seed; trống với arm
  cố định của thuật toán tất định. Hệ quả: hai arm khác loại không còn ghép cặp được theo số `seed` (mục 6).
- Khóa job thêm `model_seed`. Dòng cũ không có trường này được hiểu là `model_seed = seed` (thuật toán nhận
  seed) hoặc trống (tất định), đúng như hành vi cũ, nên resume thư mục cũ trong chế độ `shared` không đổi.
- Tên file member log: `seed<seed>` như cũ; khi `seed` trống thì là `seedx`.

## 6. Báo cáo: không đếm một lần chạy thành nhiều quan sát

**[chốt]** Thêm một hàm đọc chung, `report.replicate_view`. Hàm này giữ nguyên arm online; với mỗi
(thuật toán, luồng) của arm cố định, nó lấy trung bình qua các lần chạy thật, rồi nhân bản giá trị đó
ra cho từng tuner seed của arm online cùng (thuật toán, luồng). Các dòng nhân bản có cờ `broadcast` và
`n_runs` là số lần chạy thật. Với thư mục `shared`, hàm không làm gì.

Nơi dùng hàm này, thay vì ghép theo số seed:
- `robustness.load_inputs`, cho bootstrap và các kiểm định;
- `selection.seed_values` và đọc theo cửa sổ, cho phép tách seed và tách thời gian;
- `variants.arm_means`.

Các bảng trung bình theo (thuật toán, luồng, arm) của `report` (`summary`, `gain`, `headroom`) không
cần nhân bản. `summary` có thêm hai cột: `replicates` (`tuner seeds`, `model seeds`, hoặc
`none: deterministic, run once`) và `n_runs`.

Bootstrap: đơn vị vẫn là luồng hoặc khối nguồn. Một thuật toán tất định ở arm cố định góp đúng một giá
trị, không có độ dao động theo seed. Báo cáo ghi rõ khoảng tin cậy phản ánh nguồn nào: luồng, tuner,
hay model.

## 7. Test

- Cùng tuner seed, khác model seed: quần thể ban đầu và mọi tham số được rút trùng nhau; đầu ra mô hình
  khác nhau ở thuật toán nhận seed, và trùng nhau ở thuật toán tất định.
- Khác tuner seed, cùng model seed: tham số được rút khác nhau.
- s giống nhau giữa các arm và các tuner seed, kể cả khi bước hiệu chuẩn phải lấy mẫu con.
- Chế độ `shared` cho kết quả trùng tuyệt đối với code trước khi sửa. Kiểm bằng một fixture chụp từ code
  hiện tại **trước** khi sửa: tham số và hash dự đoán theo từng cửa sổ.
- Danh sách job theo bảng mục 4; một thuật toán tất định chỉ có một job cố định.
- Khóa, manifest, chuyện từ chối `--seeds` trong `split`, và `oracle_static` trùng model seed.
- `replicate_view`: nhân bản đúng, `n_runs` đúng, không làm gì với thư mục `shared`; robustness, selection
  và variants chạy được trên thư mục kiểu mới.

## 8. Không làm trong spec này

Thêm số lần chạy; nhiều model seed cho arm online trong một thư mục; seed riêng cho từng member; data
seed mới; seed cho luồng thật.

## 9. Ảnh hưởng tới notebook E/C (`otb_ec_variants.ipynb`)

Notebook chuyển sang chính sách `split`: `--tuner-seeds` thay `--seeds 1`, và `--model-seed 201`.
**Đã chốt 24/10 (user):** notebook E/C chạy 1 tuner seed (101); thí nghiệm này không vào paper. Đổi được
ở bảng điều khiển của notebook. Spec được user duyệt cùng ngày; không viết file kế hoạch riêng (user không
cần), tiến độ ghi ở ledger.
