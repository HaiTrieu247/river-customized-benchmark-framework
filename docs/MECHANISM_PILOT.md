# Mechanism pilot (giai đoạn 3 của PAPER_IMPROVEMENT_PLAN): chờ chốt trước khi chạy

Ngày 23/10/2026. Giai đoạn 1 (audit/log-only) và 2 (engineering smoke) của kế hoạch đã xong;
code của E1, E2, E4, E5 đã có và đã chạy thử. Tài liệu này là đề xuất thiết kế, chưa phải
kết quả. Theo AGENTS.md, chọn tập luồng và seed là quyết định nghiên cứu của người dùng:
các lựa chọn dưới đây là placeholder để chốt, không phải khuyến nghị.

## 1. Những gì giai đoạn 1–2 đã cho thấy (đọc trước khi chốt)

- **Runtime gate làm run không tái lập được.** Smoke 12 job trên một laptop (`conf_pipeline/out/repeat_smoke/report/SUMMARY.md`;
  sáu thuật toán rẻ, shuttle và scale_drift, 20 000 mẫu, seed 1): bật gate, cùng seed, tải máy khác
  → 7/12 job lệch, cả 7 do gate, tail ARI lệch tới 0,40; tắt gate (`--no-runtime-gate`) → 12/12 job
  trùng tuyệt đối đến hash dự đoán của từng member. Chưa kiểm tra lặp lại giữa hai máy, và chưa có
  CluStream, RepStream: pilot trên máy chủ nên lặp lại vài job của smoke để biết điều này.
  Hệ quả: mọi so sánh mới nên chạy gate-free làm protocol chính, protocol có gate của `out/full`
  là sensitivity; hai chế độ không được gọi là tương đương.
- **Selection regret (P1.1, `out/full/report/selection/`).** Chọn thuật toán theo default rồi dùng
  dưới online tuning: luồng thật mất 0,139 tail ARI mỗi luồng (0,171 theo khối nguồn; trung vị 0,086),
  không tốt hơn chọn ngẫu nhiên (0,126); 9/11 luồng thật có mọi thuật toán nằm trong 0,05 của
  đỉnh dưới default. Drift tham số và tổng hợp khác: ≤ 0,01. Shuttle một mình mất 0,699.
- **Đơn vị thống kê và NMI (P1.2, `out/full/report/robustness.md`).** Capture G luồng thật 0,483
  (block CI [0,231; 0,600]); NMI (a) 0,422, NMI chọn lại (b) 0,349; selection share 0,827 → 0,600
  dưới (b). Trên 5 khối nguồn thật không kiểm định Wilcoxon nào xuống dưới p = 0,0625.
  362/362 số của `robustness_extra.txt` tính lại khớp.
- **Signal pilot (E3 bước 1).** Trên 12 run, own support = common support (564 cửa sổ) nên các
  tỉ số cũ so trực tiếp được; shuttle chiếm 61% mẫu số; chỉ `dbcv_std` dương trên cả ba luồng.
  Không có DBCV độc lập để đối chiếu, và DBCV phụ thuộc cách MST phá hoà.

## 2. Quyết định cần người dùng chốt

1. **Protocol chính — ĐÃ CHỐT 23/10: không gate** (`--no-runtime-gate`) cho mọi arm của pilot, kể cả
   baseline chạy lại trong cùng out dir. Protocol có gate của `out/full` là sensitivity. Mặc định
   của CLI vẫn có gate, để một lệnh chạy cạnh `out/full` vẫn so được với nó; lệnh pilot mang cờ.
2. **Tập luồng — ĐÃ CHỐT VÀ KHÓA 23/10, trước khi chạy job pilot nào: tập con theo nguồn, 12 luồng.**
   Mỗi khối nguồn độc lập (`_common.source_block`) một luồng, để mỗi nguồn và mỗi kiểu thay đổi có mặt:

   - thật: `covtype`, `kdd99`, `electricity`, `shuttle`, `insects_abrupt_imbalanced`
   - drift tham số: `scale_drift`, `density_drift`, `k_drift`, `mixed_drift`
   - tổng hợp khác: `static_overlap`, `incremental_drift`, `evolving`

   Đại diện của bảy biến thể INSECTS (một nguồn) được chọn bằng quy tắc đặt trước, không theo kết quả:
   biến thể có nhiều cửa sổ nhất dưới mức cắt 100 000 mẫu (nhiều cửa sổ đánh giá nhất), hoà thì lấy
   biến thể đứng trước trong `_common.REAL_FOLDS`: `insects_abrupt_imbalanced` (99 cửa sổ, hoà với
   `insects_gradual_imbalanced`). Mọi luồng trong danh sách đều được báo cáo, kể cả khi kết quả ngược
   giả thuyết; danh sách không đổi sau khi đã thấy kết quả. Sáu biến thể INSECTS còn lại, 8 luồng X7 và
   data seed mới của luồng tổng hợp không vào pilot. Luồng đã dùng trong signal pilot (covtype,
   insects_abrupt_balanced, shuttle) là development data cho E3, không được gọi là held-out về sau.
3. **Seed:** 1 seed (rẻ, đủ cho kiểm tra cơ chế) hay 3 seed (cùng đơn vị với `out/full`).
4. **E4 — ĐÃ CHỐT 23/10: cả hai oracle** (`confstream_ari_oracle` và bản `_constrained`), để giảm sai sót.
   Vai trò của chúng: bộ tune online tốt nhất mà cơ chế tìm kiếm đạt được khi có tín hiệu chọn hoàn hảo
   (một mốc so sánh trên, không triển khai được); ARI chỉ là tín hiệu lái. Vẫn chọn ở cửa sổ t và đọc ở
   t+1: trên luồng thật, đọc ngay cửa sổ đã chọn cao hơn trung bình 0,036 (0,154 so với 0,118), gần bằng
   cả khoảng cách do chọn sai (0,043), nên bỏ độ trễ sẽ cho oracle lợi thế mà không lựa chọn online nào có.
5. **Nơi chạy:** máy chủ Linux như `out/full`, số worker thực tế.
6. **CluStream và RepStream — ĐÃ CHỐT 23/10: có.** Xem chi phí ở mục 4: chúng chiếm phần lớn ngân sách,
   nên số seed (mục 3) quyết định độ dài pilot.

## 3. Lệnh (đã chạy `--dry-run`, guard chấp nhận)

Một out dir chung, mọi arm cùng protocol (manifest từ chối nếu lẫn protocol):

```bash
F=covtype,kdd99,electricity,shuttle,insects_abrupt_imbalanced,scale_drift,density_drift,k_drift,mixed_drift,static_overlap,incremental_drift,evolving
ARMS=default,scaled_default,confstream,frozen_prefix,frozen_reinit,confstream_ari_oracle,confstream_ari_oracle_constrained,random_online
# seeds (docs/SEED_RULES.md, chính sách split): số tuner seed chờ chốt mục 3; ví dụ phương án C:
# 3 tuner seed cho sáu thuật toán rẻ, 1 cho CluStream/RepStream; model seed 201 ở mọi arm online.
# Arm cố định của thuật toán tất định chạy một lần, không theo tuner seed.
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.run --algos DenStream,HPStream,DBSTREAM,STREAMKMeans,SOStream,BIRCH \
    --folds $F --arms $ARMS --tuner-seeds 101,102,103 --model-seed 201 --no-runtime-gate --log-members --workers 8 \
    --out-dir conf_pipeline/out/mech_pilot
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.run --algos CluStream,RepStream \
    --folds $F --arms $ARMS --tuner-seeds 101 --model-seed 201 --no-runtime-gate --log-members --workers 8 \
    --out-dir conf_pipeline/out/mech_pilot
```

`--log-members` cần cho E4 (labels' pick trong ensemble của silhouette) và cho screen. Mỗi job kiểm
hash code nó chạy với manifest: sửa code trong lúc pilot chạy làm lệnh dừng (không ghi dòng của code
khác), nên chạy từ một bản sao code cố định hoặc không sửa cây làm việc trong 33 giờ đó. Hai oracle
arm không vào bảng mặc định của `report`/`analyze`. Đọc kết quả:

```bash
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.selection --out-dir conf_pipeline/out/mech_pilot --selecting-arm scaled_default   # E1
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.selection --out-dir conf_pipeline/out/mech_pilot                                 # baseline mới
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.oracle_compare --out-dir conf_pipeline/out/mech_pilot                            # E4
PYTHONHASHSEED=0 .venv/bin/python -m conf_pipeline.report --out-dir conf_pipeline/out/mech_pilot                                    # E2/E5: frozen_reinit, random_online cạnh confstream
```

Capture G cần reference dưới cùng protocol: `sweep --no-runtime-gate --incumbents` vào cùng out dir
rồi `run --arms oracle_static --no-runtime-gate --fixed-model-seeds 202,203,204` (model seed khác seed đã
chọn reference; code từ chối nếu trùng; thuật toán tất định chạy lại một lần) (E7; chưa tính vào bảng dưới). Lưu ý:
`--incumbents` lấy incumbent của mọi arm online trong dir, kể cả `random_online`, nên tập ứng viên
của reference này rộng hơn của `out/full`.

## 4. Chi phí dự toán (`python -m conf_pipeline.cost --from conf_pipeline/out/full`)

Sáu thuật toán rẻ, 18 luồng, 8 worker. Arm chưa từng chạy đọc qua arm gần nhất đã đo
(`scaled_default`→`default`; `frozen_reinit`, hai oracle, `random_online`→`confstream`), và mọi arm
online nhân 1,2: phần tăng đo được khi bỏ gate trong smoke (377 s → 449 s, cùng chạy một mình).
Đây là giả định, không phải đo; wall time là thời gian tiến trình trên máy của `out/full`.

| Phần | 3 seed: CPU-h | 3 seed: wall 8 worker | 1 seed: CPU-h | 1 seed: wall |
|---|---:|---:|---:|---:|
| baseline `default` | 2,1 | 0,3 h | 0,7 | 0,1 h |
| baseline `confstream` + `frozen_prefix` (gate-free) | 64,8 | 8,1 h | 21,6 | 2,7 h |
| E1 `scaled_default` (5 thuật toán chạy; BIRCH 0·s = 0 nên chỉ 4 thật sự đổi) | 1,9 | 0,2 h | 0,6 | 0,1 h |
| E2 `frozen_reinit` | 49,0 | 6,1 h | 16,3 | 2,0 h |
| E4 hai oracle | 98,0 | 12,3 h | 32,7 | 4,1 h |
| E5 `random_online` | 49,0 | 6,1 h | 16,3 | 2,0 h |
| **Tổng** | **≈ 265** | **≈ 33 h** | **≈ 88** | **≈ 11 h** |

Với cả 8 thuật toán (mọi arm trên, một oracle bản ràng buộc, 18 luồng, 8 worker):

| Phương án seed | CPU-h | wall 8 worker |
|---|---:|---:|
| A. 3 seed mọi nơi | ≈ 1 383 | ≈ 173 h |
| B. 1 seed mọi nơi | ≈ 461 | ≈ 58 h |
| C. 3 seed cho 6 thuật toán rẻ, 1 seed cho CluStream/RepStream | ≈ 605 | ≈ 76 h |
| thêm oracle không ràng buộc (theo C) | +139 | +17 h |

Trên tập 12 luồng đã khóa, với cả hai oracle: C ≈ 415 CPU-h (≈ 52 h), A ≈ 948 CPU-h (≈ 118 h),
B ≈ 316 CPU-h (≈ 39 h); số seed (mục 3) chưa chốt.

CluStream + RepStream với 1 seed đã chiếm ≈ 387 CPU-h của C. Sau pilot, dự toán lại confirmation
bằng chi phí đo được của chính pilot (`--from conf_pipeline/out/mech_pilot`).
