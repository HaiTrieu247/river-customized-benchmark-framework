# Kế hoạch tăng chất lượng khoa học cho paper ECML PKDD

Ngày đánh giá: 23/10/2026. Phạm vi: code và artifacts hiện có trong framework,
đối chiếu `result/otb_full/paper_ecml/main.pdf` và supplementary.
Đây là kế hoạch nghiên cứu, chưa phải kết quả của các thí nghiệm đề xuất.

## 1. Hướng nghiên cứu cần củng cố

Đóng góp nên là: **các lựa chọn về tuning và evaluation làm thay đổi kết luận
benchmark stream clustering; protocol này đo được thay đổi, hậu quả lựa chọn,
và giới hạn của label-free tuning.**

Ba câu hỏi chính:

1. Ranking dưới defaults có chuyển được sang điều kiện có online tuning không,
   sau khi kiểm soát units và các outputs suy biến? Chọn theo ranking đó mất gì?
2. Sau drift, chất lượng tăng nhờ thay đổi parameters hay nhờ thay mới model?
3. Trong các ensembles quan sát được và khi đổi toàn bộ tín hiệu điều khiển,
   label-free selection hạn chế chất lượng đến đâu? Cải thiện signal có giúp không?

Không đặt mục tiêu giữ nguyên τ = 0.07, capture = 48% hay selection share = 83%.
Kết quả trái giả thuyết vẫn phải được báo cáo và dùng để sửa câu chuyện.
Một benchmark tốt không bắt buộc phải tạo thuật toán mới hoặc cải thiện mọi stream.

## 2. Những gì đã có: đừng xây lại hoặc chạy lại thiếu mục đích

| Hạng mục | Code/artifact đã đọc | Trạng thái quan sát được |
|---|---|---|
| Main experiment | `conf_pipeline/run.py`, `sweep.py`, `report.py`, `analyze.py` | `out/full` có 486 summaries/arm cho default, confstream, frozen_prefix, oracle_static; gồm 9 implementations, 18 streams và 3 seeds mỗi arm |
| Logged ensemble | `out/full/members/` | 1.458 member files; schema mẫu có ARI/NMI/purity và member metadata, không có predictions hoặc alternative signals |
| Robustness | `conf_pipeline/robustness.py` | Có bootstrap, metric agreement, even/odd reference, Wilcoxon–Holm, label reach diagnostics |
| Extra analyses | `result/otb_full/report/make_robustness_extra.py` | Có LOO, INSECTS block, weak-default filtering, tie replay và recovery chung ngưỡng; một số logic còn nằm trong result directory |
| Alternative signals | `benchmark/tuning/signals.py`, `conf_pipeline/screen.py` | Có standardised silhouette, DBCV, standardised DBCV, CH/DB components, rank aggregate, stability; selectable signals điều khiển cả incumbent, victims, parents |
| Signal pilot | `out/signals_pilot/report/screen.md` | 4 algorithms × 3 real streams × 1 seed = 12 runs; raw DBCV recovery 0.224, rank aggregate 0.218, standardised silhouette 0.005; chỉ là replay trên silhouette-driven ensembles |
| Random sampler | `run.py`, `out/pilot` | Có arm `random_online`; pilot có 16 runs của arm này; chưa có trong full |
| Fresh-model control | `ConfStreamEnsemble`, arm `frozen_reinit` | Có code; chưa thấy kết quả của arm trong các output directories hiện có |
| Reference nhiều seeds | `sweep.py --seeds`, `robustness.py --reference-dir` | Có code; chưa thấy output x5 trong workspace hiện tại |
| Protocol sensitivity | `conf_pipeline/variants.py` | Có h/E/C/t0 variants; chưa thấy output x6 trong workspace hiện tại |
| Additional drift streams | `_common.X7_FOLDS`, `drift_streams.py`, `param_drift.py` | Có loaders/generators; chưa thấy output x7 trong workspace hiện tại |
| Non-convex generators | `benchmark/datasets/synthetic/paper_benchmarks.py`, `repstream.py` | Có nguồn để tái sử dụng; chưa nối thành controlled shape-versus-scale experiment của conf_pipeline |
| CMM | `benchmark/evaluation/cmm.py`, fixtures/tests | Có MOA parity fixtures; full/pilot/signals_pilot/smoke summaries hiện chưa có CMM fields |
| Validation scaffolding | `benchmark/algorithms/checks.py`, `benchmark/experiments/papers/` | Có behavioural checks và reproduction scaffolding; không đồng nghĩa đã validate fidelity của cả bốn implementations tự cài |
| Provenance | `benchmark/reports/environment.py`, run/sweep | Có environment capture mới; full cũ không có environment.jsonl |

Kiểm tra đã chạy khi lập kế hoạch: 130 tests pass trong các files
`test_tuning_signals`, `test_conf_screen`, `test_conf_variants`, `test_cmm`,
`test_confstream`, `test_conf_sweep` (37,23 giây). Đây là kiểm tra phần mềm,
không xác nhận các kết luận khoa học hoặc DBCV parity với implementation độc lập.
Working tree đang có nhiều thay đổi; giữ nguyên chúng, không reset/ghi đè.

## 3. P0 — Khóa protocol và làm cho kết quả kiểm toán được

### P0.1. Manifest, version và resume trước mọi run mới

**Lý do:** `_job_key`/`_row_key` hiện không bao gồm h/E/C/t0, search-space version,
selection signals, CMM setting hay tie-break. Thay flags rồi resume vào directory
cũ có thể bỏ qua các jobs cần chạy lại.

**Làm thêm:**

- Tạo protocol manifest và fingerprint từ toàn bộ scientific configuration:
  algorithms/implementations, constructor defaults/pinned overrides, search spaces,
  stream identity/content hash/order/cap/generator seed, model/tuner seeds,
  window/warm-start/tail policies, signal/noise/feasibility/tie rules và metrics.
- Ghi framework commit + working-tree diff hash, River commit + diff hash, package
  versions/lock, machine/worker/thread settings. Không chỉ ghi `River 0.26.1`.
- Manifest mismatch phải báo rõ và dừng resume; xuất sang directory mới. Không
  sửa hoặc giả tạo metadata cho runs cũ; ghi `unknown` nếu không khôi phục được.
- Giữ `out/full` làm archived baseline. Mỗi experiment mới có out-dir riêng.
- Phân biệt tuner seed, model seed và data-generator seed; deterministic algorithm
  không cần lặp một fixed configuration chỉ để tạo ra ba giá trị giống nhau.

**Nơi mở rộng:** `conf_pipeline/_common.py`, `run.py`, `sweep.py`,
`benchmark/reports/environment.py`. Dùng orchestration hiện có, không thêm runner
top-level. Chỉ sửa framework; không chỉnh sibling `river/`.

**Hoàn thành khi:** một command khác protocol không thể âm thầm resume vào kết
quả cũ; mọi con số chính truy được về manifest và input files cụ thể.

### P0.2. Repeatability toàn quỹ đạo

- Chạy lặp cùng protocol/seed trong processes mới với `tie_break=micro_clusters`.
- So incumbent/member IDs, params, predictions nếu được lưu, quality trajectories,
  và feasibility reasons; runtime được phép khác.
- Kiểm tra riêng `apply_feasibility`: runtime factor 30 vẫn dùng thời gian thực,
  nên deterministic tie-break chưa đảm bảo deterministic trajectory.
- Nếu runtime gate gây divergence, expose chế độ scientific repeatability không
  dùng gate thời gian để quyết định lựa chọn; giữ resource limits an toàn, ghi rõ
  failures. So sánh với resource-constrained mode như sensitivity, không âm thầm
  đổi protocol hoặc gọi hai mode là tương đương.
- Signals/CMM logging có thể tăng tải máy; kiểm tra hiệu ứng lên feasibility và
  tách thời gian model, scoring, warm-start, logging và end-to-end wall time.

**Hoàn thành khi:** biết rõ nguồn divergence và ảnh hưởng; nếu còn khác biệt,
báo cáo độ biến thiên thực tế thay vì cam kết exact reproduction.

## 4. P1 — Phân tích thêm từ logs: hiệu quả cao, chưa cần tuning runs mới

### P1.1. Biến rank instability thành hậu quả chọn thuật toán

Với mỗi stream D, chọn `a_default = argmax_a A_default(a,D)`, rồi tính
`regret(D) = max_a A_online(a,D) - A_online(a_default,D)`.

- Báo cáo absolute regret, top-1 agreement, top-k overlap và rank τ; xử lý ties
  công khai, thêm sensitivity theo practical-effect band đã định trước.
- Tách real / parameter drift / other synthetic và source blocks.
- So với lựa chọn thuật toán ngẫu nhiên và một lựa chọn toàn cục được chọn trên
  các streams khác; không chọn baseline tốt nhất trên test stream rồi gọi deployable.
- Chỉ số cùng dữ liệu này là **descriptive selection regret**, chưa phải ước lượng
  out-of-sample. Thêm selection trên prefix/held-out seeds và đánh giá phần sau
  nếu logs đủ; synthetic generalisation cần data seeds mới, không chỉ tuner seeds.

**Output:** `selection_regret.csv`, figure τ bên cạnh absolute selection regret.
**Quyết định:** regret nhỏ thì giảm vai trò headline rank; regret lớn thì đây là
bằng chứng trực tiếp cho ý nghĩa thực tế của regime-aware evaluation.

### P1.2. Statistical unit và metric sensitivity

- Đưa LOO/INSECTS-block analyses từ result script thành API/report có tests và CLI
  trong pipeline hiện có; giữ compatibility với kết quả published.
- Family/block bootstrap: không resample các INSECTS variants như các nguồn độc
  lập. Synthetic repetitions nằm trong generator family, không tăng N bằng model seeds.
- Báo cáo paired effects/intervals theo source blocks. Dùng paired seeds thật sự
  tương ứng; không ghép reporting seed của reference với seed khác chỉ vì cùng chỉ số.
- Dùng per-member NMI đã log để làm lại conditional selection analysis theo NMI;
  phân biệt (a) đổi evaluation metric, giữ ARI-selected reference và (b) chọn lại
  reference theo NMI từ candidate set đã có. Không gọi (a) là NMI-optimal reference.
- Báo cáo denominator, absolute gain/loss và unstable/undefined ratios; không clip
  negative losses hoặc ratio vượt [0,1] để tạo kết luận đẹp.
- Giữ toàn bộ streams trong primary descriptive tables; diagnostics/filters dùng
  sensitivity được công bố, không loại streams vì kết quả không thuận giả thuyết.

**Output:** bảng kết luận × metric × source-block/reference construction; không
chỉ bảng correlation giữa metric rankings.

## 5. P1 — Những đối chứng cần chạy để bảo vệ đóng góp trung tâm

### E1. Scaled defaults: cô lập units khỏi tuning

**Cần thêm code:** arm hoặc configuration policy riêng, chạy như một fixed config,
dựa trên baseline `start_params`, chỉ thay các absolute-distance parameters đã
kiểm kê bằng default-value × s. Không dùng Sobol point thay cho control này.

- Giữ nguyên các parameters khác, stream, seeds, h/t0, warm-start và common
  implementation policies. Công bố baseline là framework-adjusted defaults ở
  đâu: `start_params` áp pinned values và clamp buffers, không hoàn toàn constructor defaults.
- `sigma` của inner KMeans phải lấy effective default khi constructor chuyển
  qua kwargs. BIRCH threshold = 0 thì 0 × s vẫn là 0: giữ và giải thích, không
  tự thêm epsilon để làm control tốt lên.
- Với algorithms không có distance parameter, đánh dấu unchanged; có thể reuse
  đúng baseline khi scientific protocol tương đương, nhưng không đếm thành run mới.
- Một scalar s không xử lý feature-wise scale hoặc translation. Prefix-fitted
  scaler cố định, nếu thêm, là control khác, fit trên calibration prefix dùng
  chung cho mọi regime; không dùng toàn stream và không dịch tọa độ model đang chạy.

**Đo:** τ, broken rate, ARI và selection regret ở E1 so với raw defaults và online.
**Hoàn thành khi:** phân biệt được phần kết quả do units/default pathologies và
phần còn lại do tuning. Nếu τ tăng mạnh, viết lại headline theo kết quả đó.

### E2. Fresh model versus new parameters

**Dùng code có:** `--arms frozen_reinit`; so với `frozen_prefix` và `confstream`.
Chạy cùng deterministic protocol mới, giữ E/C, freeze point, scoring và warm-start
như nhau; không chỉ đặt arm mới cạnh runtime-tie baseline cũ.

- Arm này spawn nhiều copies của một frozen config; cần báo cáo compute budget
  và model-seed policy, không gọi nó là restart đơn giản một lần mỗi drift.
- Đo same-segment ARI, common-threshold recovery, recovery depth, latency và cost.
- `frozen_reinit - frozen_prefix` mô tả lợi ích renewal dưới policy này;
  `confstream - frozen_reinit` đo lợi ích thêm của adaptive parameters trong
  comparison này. Không diễn giải thành universal causal partition.
- Nếu giữ claim về cạnh tranh giữa deployable regimes, bổ sung periodic/triggered
  retuning với budget rõ ràng; nếu chưa đủ nguồn lực, thu hẹp claim vào regimes đã xét.

**Output:** recovery table dùng một định nghĩa chính ngay main paper, figure ba arms.

### E3. Alternative signals: screening rồi xác nhận end-to-end

**Dùng code có:** `signals.py`, `screen.py`, signal arms của `run.py`.

1. Đưa pilot 12 runs hiện có thành bảng: pair coverage, absolute ARI gain, recovery
   ratio, signal-defined rate và chi phí. Không quảng bá 22.4% như kết quả full.
2. Mở rộng logging trên subset được định trước theo algorithm mechanisms và
   stream sources, không chọn vì pilot có gain lớn. Stability cần chạy mới.
3. Screening phải có paired common support và missing/fallback report; các signals
   được chấm trên tập windows khác nhau thì recovery ratios chưa so trực tiếp được.
4. Chọn ít candidate bằng development data, sau đó khóa và chạy signal arms trên
   confirmation streams/data seeds độc lập. Các streams đã dùng trong pilot là
   exploratory, không đổi tên thành held-out sau khi đã xem kết quả.
5. Đánh giá end-to-end quality và total cost: signal mới thay parents/victims và
   ensemble trajectory, nên replay gain không dự đoán chắc chắn deployment gain.

**Cần kiểm tra thêm:** DBCV implementation đối chiếu một implementation/reference
fixture độc lập; tiny unit tests hiện có chưa đủ xác nhận fidelity. Noise `-1`
hiện được `_partition` giữ như label thông thường; expose/document policy nhất quán
cho comparison, kèm sensitivity thay vì âm thầm sửa.

**Output:** một bảng chính gồm raw silhouette / standardised silhouette / candidate
đã khóa: ARI, NMI, absolute gain, conditional recovery và total overhead.
Nếu không signal nào giúp, báo cáo negative result cùng kiểm tra đủ mạnh.

### E4. Oracle-signal tuner: kiểm tra ảnh hưởng của signal lên cả search

**Cần thêm code:** diagnostic arm riêng; current label-free `SELECTABLE` không có
ARI oracle. Tránh trộn vào danh sách deployable regimes.

- Chỉ thay selection score bằng ARI của window vừa kết thúc cho cả incumbent,
  victim replacement và parent roulette; score trên window t chọn đầu ra t+1.
- Giữ sampling, model reset, budget, starts, caps và seed policy tương ứng.
- Không vô tình giữ silhouette-defined gate: oracle eligibility phải dùng
  definition riêng; nếu vẫn giữ gate để isolate effect, ghi là constrained oracle
  và thêm sensitivity cho single-label outputs.
- Khi true labels của window chỉ có một class, định trước missing/fallback policy
  nhất quán với primary metric; báo cáo ảnh hưởng, không tận dụng ARI suy biến.
- So oracle-driven trajectory, label-selected replay và online silhouette trên
  cùng pairs. Các signed differences là operational comparisons, không tự động
  trở thành non-negative hoặc causal signal/search losses.

**Output:** figure/bảng ba đường so sánh. Nếu không chạy E4, đổi framing thành
`conditional selection gap under silhouette-driven search`, bỏ claim phân tách
được nguyên nhân signal/search một cách tổng quát.

### E5. Random search control

**Dùng code có:** `--arms random_online`. Giữ model births, E/C và số evaluations
như confstream; báo cáo thêm wall budget vì configurations có cost khác nhau.
Chạy trước trên subset mechanism/source đã khóa, sau đó mở rộng khi có lý do.

**Đo:** paired quality differences, cost, coverage/search-space exploration;
τ giữa hai rankings không đủ kết luận hai search strategies tương đương.
Không cần chứng minh evolutionary search thắng để benchmark có giá trị.

## 6. P2 — Củng cố generalisation, reference và implementation fidelity

### E6. Structured stress tests và dữ liệu chưa dùng để chọn phương án

- Tái sử dụng generators non-convex hiện có, tạo thiết kế shape × scale có paired
  generating geometry/seeds, thêm density/noise khi cần phân biệt giả thuyết.
- Tách data-generator seeds khỏi tuner/model seeds. Khóa design trước, không
  sinh nhiều streams rồi giữ lại những streams ủng hộ claim.
- X7 loaders có sẵn: dùng để kiểm tra natural drift và drift severity/timing;
  không giả định sensor identity/Rialto class tự động là ground-truth clusters.
  Chạy label-reach/label diagnostics trước và báo cáo cả trường hợp kém phù hợp.
- Có thể chạy subset trước để phát hiện lỗi/cost, nhưng confirmation set phải
  giữ nguyên sau khi đã xem kết quả. Các variants cùng family không tính là
  nhiều nguồn độc lập.

### E7. Reference ổn định hơn, phạm vi rõ hơn

- Dùng `sweep --seeds` và `robustness --reference-dir` cho stochastic algorithms.
  Reuse candidate runs cũ chỉ khi manifest/protocol tương đương và provenance rõ.
- Chọn reference trên nhiều model seeds giảm seed-selection variance; không sửa
  optimism theo dữ liệu của deterministic algorithms.
- Giữ in-sample candidate-set oracle là upper comparator có giới hạn đã nêu;
  thêm forward split như deployable fixed-config baseline riêng. Không mong một
  configuration chọn trên early segment vẫn là oracle cho later drift segment.
- Kiểm tra 32/64/128 Sobol budgets trên cùng nested design nếu reference gap nhạy
  budget; không cần sweep toàn bộ ngay. Reporting seeds tách khỏi selecting seeds.
- Candidate-set oracle không phải upper bound của adaptive model, và sampled
  best-static không phải `every fixed configuration`.

### E8. Validation bốn implementations tự cài

- Lập bảng BIRCH/HPStream/RepStream/SOStream: paper specification, implementation
  deviation, preprocessing/distance/macro-clustering policy, expected invariants.
- Tái sử dụng behavioural checks nhưng tách khỏi algorithm fidelity.
- So một implementation độc lập tương ứng nếu có, hoặc tái hiện thí nghiệm gốc
  với đủ metadata; không gọi reproduction script tồn tại là reproduction thành công.
- `benchmark/experiments/papers/repstream.py` có normalization/distance và protocol
  riêng; không lấy số đó ghép vào conf_pipeline. Đọc lại các mô tả stale trước khi dùng.
- Đối chiếu bằng partition-invariant measures/invariants, không bắt cluster IDs
  giống nhau. Nếu không có implementation độc lập, nêu giới hạn và cân nhắc
  sensitivity chỉ trên algorithms có validation tốt hơn.

### E9. CMM và protocol constants

- CMM đã có code và parity tests, chưa có trong full logs. Không thể tính từ
  ARI/NMI summaries: cần points + per-member predictions hoặc rerun.
- CMM có thể saturate do class merging và sphere allowance. Đọc cùng
  `cmm_ca_separability`, `cmm_ca_model`, cluster counts và noise coverage.
  `cmm_strict` là variant khác, không gọi là số MOA hoặc mặc định tốt hơn ARI.
- Dùng `variants.py` cho h/E/C/t0; lưu ý e10 và c20 thay đồng thời E/C, không
  gọi là pure one-factor effects. h thay đổi cả số mẫu calibration và warm-start;
  so theo sample budget và window budget rõ ràng. Variant h/t0 cần reference
  tương ứng nếu báo G; đừng reuse oracle chỉ vì cùng algorithm/stream.
- Không để h/t0 làm một số streams chỉ còn vài windows rồi kết luận ngang trọng số.

## 7. Chiến lược chạy và budget

### Chọn mức chạy theo bằng chứng, không chạy mọi tổ hợp ngay

1. **Audit/log-only:** P0 manifest + P1 analyses; chưa chạy heavy experiment.
2. **Engineering smoke:** một vài pairs ngắn, xác nhận seeds/lag/tail/metrics và
   repeatability; không dùng quality ở smoke làm evidence cho paper full.
3. **Mechanism pilot:** six cheaper algorithms, subset sources/drift mechanisms đã
   khóa; E1–E5 tùy claim giữ lại. Giữ baseline cùng new protocol trong mỗi comparison.
4. **Confirmation:** mở rộng fixed designs/candidates sang data seeds hoặc sources
   chưa dùng để chọn; ba tuner seeds làm điểm khởi đầu, tăng theo uncertainty.
5. **Final artifact:** đồng bộ main/supplementary/report từ một source of truth.

CluStream và RepStream cần xuất hiện trong final evidence nếu claim bao phủ tám
algorithms, nhưng không nhất thiết chạy chúng qua mọi exploratory signal/variant.
Không suy rộng subset sáu algorithms thành kết quả cho cả tám.

### Mốc cost thực sự quan sát được

Tổng `wall_s` của từng job trong `out/full`, không gồm DenStreamPatchedFast:

| Arm | Six cheaper algorithms: 324 jobs | Tám algorithms: 432 jobs |
|---|---:|---:|
| default | 2,06 giờ | 5,17 giờ |
| confstream | 40,85 giờ | 265,37 giờ |
| frozen_prefix | 13,14 giờ | 83,06 giờ |

Đây là **tổng thời gian thực của các processes**, dùng làm baseline chi phí,
không phải hardware-independent CPU measurement. Full cũ thiếu environment record;
không cam kết một run mới có cùng cost hoặc wall time bằng tổng/8.
Signals pilot 12 runs có tổng wall time khoảng 1,50 giờ, nhưng không ngoại suy
tuyến tính cho mọi signal/stream. DBCV và stability cần đo riêng.

Sau pilot, dự toán mỗi experiment bằng tổng measured per-cell costs + overhead,
công bố job count và projected wall time với số workers thực tế. Không tự động
khởi chạy hàng trăm giờ từ kế hoạch này.

## 8. Các điều kiện đóng gói manuscript

- Abstract/Introduction/Conclusion phải dùng cùng phạm vi: defaults versus tuned,
  evaluated candidate set, compared deployable regimes, conditional selection gaps.
- Dùng một recovery definition chính; bảng cũ chỉ là sensitivity nếu giữ lại.
- Thêm literature theo câu hỏi, không chỉ tăng số citations: clustering benchmark
  bias, class-label validity, internal/incremental CVIs, clustering AutoML,
  dependence-aware comparisons. Kiểm tra nguồn gốc KDD99 và metadata references.
- Viết rõ insight mới so với benchmark/tuner đã có; thêm claim–evidence table.
- Kiểm tra artifact trong environment sạch; lưu raw logs, manifests, package lock,
  deterministic settings và figure-generation commands; mỗi headline truy được.
- Điền hardware TODO bằng thông tin có thể xác minh; thiếu thì ghi unknown.
- Bản PDF hiện 18 trang. Kiểm tra CfP của năm nộp; quy định 2026 là 16 trang gồm
  references, không tự coi quy định đó đã được xác nhận cho 2027.
- AI-use/ethics/anonymity theo đúng quy định năm/track; đây là checklist nộp bài,
  không dùng để suy đoán có vi phạm từ bản nháp.

## 9. Ưu tiên và điều kiện chuyển giai đoạn

| Thứ tự | Deliverable | Tiêu chí được nâng trực tiếp | Điều kiện chuyển bước |
|---|---|---|---|
| 1 | Manifest + repeatability audit | Reproducibility, soundness | Không trộn protocols, biết nguồn nondeterminism |
| 2 | Selection regret + source-block/metric robustness | Significance, statistics, validity | Biết ranking đổi có hậu quả thực tế đến đâu |
| 3 | E1 scaled defaults | Novelty, experimental validity | Phân biệt units với tuning; cập nhật headline |
| 4 | E2 renewal control | Soundness, causal interpretation | Định lượng fresh-model confound |
| 5 | E3 screen + confirmation | Insight, actionable value | Có positive hoặc negative result được kiểm chứng |
| 6 | E4 và E5 nếu giữ claims signal/search | Mechanistic depth, controls | Không dùng replay để kết luận toàn trajectory |
| 7 | E6–E9 theo residual uncertainty | Generalisation, fidelity, robustness | Phạm vi evidence khớp phạm vi kết luận |
| 8 | Rewrite + archival artifact | Clarity, calibrated claims | Không còn claim không có table/figure/analysis đỡ |

Không có phần trăm tăng điểm hoặc xác suất accept được bảo đảm. Kế hoạch hoàn
thành tốt khi các kết luận vẫn kiểm tra được dù kết quả thay đổi so với bản nháp.
