"""Write ``otb_benchmark.ipynb`` at the framework root: the one notebook that drives conf_pipeline.

The notebook is generated from the cell sources below, so it is reproducible
and testable (``tests/test_otb_notebook.py`` regenerates it and executes it in
the ``tiny`` preset). Edit the cells here, never the ``.ipynb``, then run from
the repo root::

    python -m conf_pipeline.make_notebook

Cell ids are fixed, so regenerating an unchanged notebook leaves the file
byte-identical. Markdown cells are in Vietnamese; code and identifiers in English.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat import v4

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "otb_benchmark.ipynb"

# --- cells, in notebook order ----------------------------------------------------------

TITLE = r"""
# Benchmark tune online cho phân cụm luồng (`conf_pipeline`)

Notebook này chạy trọn giao thức của `conf_pipeline` ở một chỗ. Mỗi bộ (thuật toán, luồng, seed) được chạy dưới ba chế độ tune, gọi là *arm*, trên cùng một driver: `default` giữ một cấu hình default của River suốt luồng, `confstream` là ensemble confStream tự chọn lại cấu hình theo silhouette sau mỗi cửa sổ, còn `frozen_prefix` tune như `confstream` rồi giữ nguyên cấu hình từ cửa sổ `FREEZE_AFTER_WINDOW`. Arm `random_online` (giống `confstream` nhưng rút cấu hình con đều trên không gian tìm kiếm) đã được bỏ khỏi giao thức ngày 12/10 vì chiếm khoảng một nửa ngân sách; driver vẫn chạy nó khi được gọi bằng `--arms random_online`, và báo cáo vẫn đọc được các run cũ của nó. Một lượt quét các cấu hình tĩnh, gồm các điểm Sobol, default của River và các cấu hình mà những arm online đã chọn làm incumbent, chọn ra oracle tĩnh cho từng cặp (thuật toán, luồng), và oracle đó được chạy lại như arm thứ năm `oracle_static` để làm trần. Các cell cuối dựng báo cáo, phân tích thứ hạng giữa các arm và ước tính thời gian cho giao thức đầy đủ.

**Cách dùng.** Mở notebook từ gốc `river-customized-benchmark-framework/` rồi chạy lần lượt từ trên xuống. Preset được đọc từ biến môi trường `OTB_PRESET`:

| Preset | Dùng ở đâu | Quy mô |
|---|---|---|
| `tiny` | máy Mac, kiểm tra đường ống trong vài phút | DenStream × `static_easy` × seed 1 × 2 arm, 900 mẫu |
| `pilot` (mặc định) | máy Mac hoặc server | DenStream, CluStream, HPStream, RepStream × 4 luồng × seed 1 × 3 arm, 100 000 mẫu |
| `smoke` | server Linux, vài giờ, ngay trước `full` | mọi thuật toán và luồng của `full` × seed 1 × `default` và `confstream`, 25 000 mẫu: bắt lỗi nạp dữ liệu, lỗi thuật toán và đo chi phí trên những ô `pilot` chưa chạy |
| `full` | server Linux | 8 thuật toán × 18 luồng × 3 seed × 3 arm, 100 000 mẫu, cộng `DenStreamPatchedFast` làm phiên bản cài đặt của DenStream cho bảng T7 |

Trên máy Mac có thể mở bằng `OTB_PRESET=tiny jupyter lab`. Trên server nên chạy trong `tmux`, ví dụ `OTB_PRESET=full jupyter nbconvert --to notebook --execute otb_benchmark.ipynb --output otb_full_run.ipynb --ExecutePreprocessor.timeout=-1`. Kết quả nằm ở `conf_pipeline/out/<preset>/` (đổi bằng `OTB_OUT_DIR`): `runs.jsonl`, `windows.jsonl`, `members/`, `sweep.jsonl`, `oracle_static.json` và thư mục `report/`.

**Mọi cell đều chạy lại được.** Job đã có dòng trong file kết quả được bỏ qua, nên khi dừng giữa chừng thì chỉ cần chạy lại cell để làm tiếp phần còn dở. Job lỗi không tự chạy lại; muốn thử lại chúng thì đặt `RETRY_FAILED = True` trong bảng điều khiển. Kết quả chạy trên server rồi `scp` về `conf_pipeline/out/<preset>/` được các cell báo cáo đọc ngay mà không phải chạy lại gì.
"""

SETUP = r'''
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Markdown, display

# The notebook lives at the framework root; conf_pipeline must be importable from here.
REPO = Path.cwd()
if not (REPO / "conf_pipeline").is_dir():
    raise SystemExit(f"Hãy mở notebook từ gốc river-customized-benchmark-framework (cwd={REPO})")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conf_pipeline import _common as C, analyze, report, robustness  # noqa: E402

# Every algorithm run is a child process with the hash seed pinned (CluStream replays only then).
ENV = {**os.environ, "PYTHONHASHSEED": C.PYTHONHASHSEED}
# report/analyze switch pyplot to Agg to write their PNGs; their cells switch back to this.
BACKEND = matplotlib.get_backend()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def run_cli(module: str, out_dir: Path, *args) -> int:
    """``python -m <module> --out-dir <out_dir> <args>`` from the repo root, streaming its output.

    Same contract as ``subprocess.run(..., check=False)``: a non-zero exit is
    printed, not raised. Interrupting the cell stops the driver and its jobs
    (their own process group: SIGTERM, then SIGKILL if it is still running
    30 s later) and removes the lock files the driver held, so the cell can
    simply be run again: finished jobs are on file and skipped.
    """
    cmd = [sys.executable, "-m", module, "--out-dir", str(out_dir), *map(str, args)]
    print("$ PYTHONHASHSEED=0 python -m", *cmd[2:], flush=True)
    # Unbuffered, so lines the CLI prints without flushing (its job count) show up at once.
    proc = subprocess.Popen(cmd, cwd=REPO, env={**ENV, "PYTHONUNBUFFERED": "1"}, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
        rc = proc.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        for lock in Path(out_dir).glob("*.lock"):
            if lock.read_text(encoding="utf-8").startswith(f"pid={proc.pid} "):
                lock.unlink()
        print("Đã dừng; chạy lại cell để làm tiếp các job còn dở.")
        raise
    if rc != 0:
        print(f"!! {module} thoát với mã {rc}; các cell sau vẫn đọc được những gì đã ghi.")
    return rc


def show(title: str, df: pd.DataFrame, cols: list[str] | None = None) -> None:
    """A bold title, then ``df`` (only the ``cols`` it has), or a note when it is empty."""
    display(Markdown(f"**{title}**"))
    if df is None or df.empty:
        print("(trống)")
        return
    display(df[[c for c in cols if c in df.columns]] if cols else df)
'''

PANEL_MD = r"""
## 1. Bảng điều khiển

Cell dưới chọn preset theo `OTB_PRESET` (mặc định `pilot`) và thư mục kết quả theo `OTB_OUT_DIR` (mặc định `conf_pipeline/out/<preset>`), dựng sẵn danh sách cờ cho ba lệnh CLI rồi in toàn bộ thiết lập đã chốt để đối chiếu trước khi chạy. Muốn đổi một giá trị cho riêng lần chạy này thì gán lại ở chỗ đánh dấu ngay dưới các cờ, ví dụ `WORKERS = 4`; khi đổi `N_SAMPLES`, `H`, `E_SIZE`, `E_TEST` hay `FREEZE_AFTER_WINDOW` thì nên đổi cả `OTB_OUT_DIR` để hai lần chạy không trộn vào cùng một file. `FREEZE_AFTER_WINDOW` là cửa sổ mà `frozen_prefix` thôi tune và cũng là nơi phần đuôi của mọi arm bắt đầu; `tiny` đặt nó bằng 1 vì luồng chỉ có 3 cửa sổ. Preset `full` thêm `DenStreamPatchedFast` vào `ALGOS`: đây là DenStream đã sửa BUG-7 và BUG-8 với pha offline nhanh (GAP-3), được chạy như mọi thuật toán khác nhưng không bao giờ được xếp hạng chung, chỉ so với DenStream ở bảng T7. `RUN_SWEEP` và `RUN_ORACLE` bật hoặc tắt hai bước quét và chạy lại oracle, `LOG_MEMBERS` ghi kết quả của từng thành viên ensemble (cần cho `R_proxy`, `R_search` và mốc hồi phục sau drift), còn `RETRY_FAILED = True` chạy lại các job đã lỗi.
"""

PANEL = r"""
PRESET = os.environ.get("OTB_PRESET", "pilot")
OUT_DIR = Path(os.environ.get("OTB_OUT_DIR", REPO / "conf_pipeline" / "out" / PRESET))

PRESETS = {
    "tiny": dict(ALGOS=["DenStream"], FOLDS=["static_easy"], SEEDS=[1], ARMS=["default", "confstream"],
                 N_SAMPLES=900, H=300, E_SIZE=2, E_TEST=1, FREEZE_AFTER_WINDOW=1, SWEEP_N_POINTS=2,
                 WORKERS=1, TIMEOUT_S=600, ORACLE_SEEDS=[1]),
    "pilot": dict(ALGOS=["DenStream", "CluStream", "HPStream", "RepStream"],
                  FOLDS=["covtype", "insects_abrupt_balanced", "scale_drift", "k_drift"],
                  SEEDS=[1], ARMS=list(C.ARMS), N_SAMPLES=C.N_SAMPLES_CAP, H=C.H, E_SIZE=C.E_SIZE,
                  E_TEST=C.E_TEST, FREEZE_AFTER_WINDOW=C.FREEZE_AFTER_WINDOW, SWEEP_N_POINTS=8,
                  WORKERS=8, TIMEOUT_S=6 * 3600, ORACLE_SEEDS=[1]),
    "smoke": dict(ALGOS=[*C.ALGOS, *C.IMPL_VARIANTS], FOLDS=list(C.FOLDS), SEEDS=[1],
                  ARMS=["default", "confstream"], N_SAMPLES=25_000, H=C.H, E_SIZE=C.E_SIZE,
                  E_TEST=C.E_TEST, FREEZE_AFTER_WINDOW=C.FREEZE_AFTER_WINDOW, SWEEP_N_POINTS=2,
                  WORKERS=8, TIMEOUT_S=2 * 3600, ORACLE_SEEDS=[1]),
    "full": dict(ALGOS=[*C.ALGOS, *C.IMPL_VARIANTS], FOLDS=list(C.FOLDS), SEEDS=list(C.SEEDS), ARMS=list(C.ARMS),
                 N_SAMPLES=C.N_SAMPLES_CAP, H=C.H, E_SIZE=C.E_SIZE, E_TEST=C.E_TEST,
                 FREEZE_AFTER_WINDOW=C.FREEZE_AFTER_WINDOW, SWEEP_N_POINTS=C.SWEEP_N_POINTS,
                 WORKERS=8, TIMEOUT_S=12 * 3600,
                 ORACLE_SEEDS=[2, 3, 4]),
}
if PRESET not in PRESETS:
    raise SystemExit(f"OTB_PRESET={PRESET!r}; chọn một trong {sorted(PRESETS)}")
globals().update(PRESETS[PRESET])
RUN_SWEEP = True
RUN_ORACLE = True
LOG_MEMBERS = True
RETRY_FAILED = False

# --- overrides for this run only go here, e.g. WORKERS = 4 ---

# --- derived: the flags of the three CLIs (nothing to edit below) ---
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMMON_ARGS = ["--algos", ",".join(ALGOS), "--folds", ",".join(FOLDS), "--n-samples", N_SAMPLES,
               "--h", H, "--workers", WORKERS, "--timeout", TIMEOUT_S]
# The protocol of out/full: one job seed drives the tuner and the models (docs/SEED_RULES.md).
COMMON_ARGS += ["--seed-policy", "shared"]
COMMON_ARGS += ["--retry-failed"] if RETRY_FAILED else []
DRIVER_ARGS = ["--e-size", E_SIZE, "--e-test", E_TEST, "--freeze-after-window", FREEZE_AFTER_WINDOW]
RUN_ARGS = [*COMMON_ARGS, *DRIVER_ARGS, "--seeds", ",".join(map(str, SEEDS)), "--arms", ",".join(ARMS)]
RUN_ARGS += ["--log-members"] if LOG_MEMBERS else []
SWEEP_ARGS = [*COMMON_ARGS, "--n-points", SWEEP_N_POINTS, "--seed", 1, "--incumbents",
              "--freeze-after-window", FREEZE_AFTER_WINDOW]
ORACLE_ARGS = [*COMMON_ARGS, *DRIVER_ARGS, "--seeds", ",".join(map(str, ORACLE_SEEDS)),
               "--arms", "oracle_static"]

print(f"{'PRESET':19s} = {PRESET}")
print(f"{'OUT_DIR':19s} = {OUT_DIR}")
for name in ("ALGOS", "FOLDS", "SEEDS", "ARMS", "N_SAMPLES", "H", "E_SIZE", "E_TEST",
             "FREEZE_AFTER_WINDOW", "SWEEP_N_POINTS", "ORACLE_SEEDS", "WORKERS", "TIMEOUT_S", "RUN_SWEEP",
             "RUN_ORACLE", "LOG_MEMBERS", "RETRY_FAILED"):
    print(f"{name:19s} = {globals()[name]}")
"""

DRY_RUN_MD = r"""
## 2. Chạy thử (dry run)

Cell dưới gọi `conf_pipeline.run` và `conf_pipeline.sweep` với `--dry-run`, nên không job nào thật sự chạy. Dòng đầu của `run` cho tổng số job, số job đã có trong `runs.jsonl` và số job còn phải chạy; dòng đầu của `sweep` là số điểm quét, tính cả điểm default của River và chưa trừ những điểm đã có trong `sweep.jsonl`. Vài job đầu được liệt kê bên dưới để đối chiếu với bảng điều khiển.
"""

DRY_RUN = r"""
SHOW = 5
for module, args in (("conf_pipeline.run", RUN_ARGS), ("conf_pipeline.sweep", SWEEP_ARGS)):
    cmd = [sys.executable, "-m", module, "--out-dir", str(OUT_DIR), *map(str, args), "--dry-run"]
    res = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True, check=False)
    lines = res.stdout.splitlines()
    print(f"[{module}] " + (lines[0] if lines else "(không có output)"))
    for line in lines[1:1 + SHOW]:
        print(line)
    if len(lines) > 1 + SHOW:
        print(f"  ... và {len(lines) - 1 - SHOW} dòng nữa")
    if res.returncode != 0:
        print(res.stderr[-2000:])
"""

RUN_MD = r"""
## 3. Chạy các arm

Cell dưới chạy mọi job (thuật toán, luồng, seed, arm) của preset bằng `conf_pipeline.run`; mỗi job là một tiến trình con có hash seed cố định, giới hạn bộ nhớ và giới hạn thời gian `TIMEOUT_S`, và có `WORKERS` job chạy song song. Mỗi dòng in ra là một job vừa xong, với trạng thái (`ok`, `error`, `timeout`, `oom`), số cửa sổ, ARI lagged trung bình trên toàn luồng và thời gian chạy. Kết quả được ghi dần vào `runs.jsonl` và `windows.jsonl` (cùng `members/` khi `LOG_MEMBERS` bật), nên dừng cell rồi chạy lại thì các job đã xong được bỏ qua.
"""

RUN = r"""
rc = run_cli("conf_pipeline.run", OUT_DIR, *RUN_ARGS)
"""

SWEEP_MD = r"""
## 4. Quét cấu hình tĩnh và chọn oracle

Khi `RUN_SWEEP` bật, cell dưới chạy `conf_pipeline.sweep`: với mỗi cặp (thuật toán, luồng), từng điểm Sobol của không gian tìm kiếm và điểm default của River được chạy một lần với seed 1 như một cấu hình tĩnh không bao giờ đổi. Cùng lượt đó, với mỗi run của `confstream` và `frozen_prefix` (và của `random_online` nếu đã chạy) có trong `windows.jsonl`, cấu hình giữ vai incumbent lâu nhất và cấu hình incumbent cuối cùng cũng được chạy lại như vậy, đúng từng tham số, nên trần không thấp hơn một cấu hình mà chính các arm online đã tìm ra; vì thế cell này phải chạy sau bước chạy các arm. Nhãn không tham gia vào việc chọn các ứng viên này, chỉ dùng để chọn điểm thắng như với mọi điểm khác. Cuối lượt quét, điểm có ARI lagged trung bình cao nhất trên phần đuôi (các cửa sổ từ `FREEZE_AFTER_WINDOW` trở đi, vùng mà mọi arm được so với nhau) được ghi vào `oracle_static.json`, còn luồng quá ngắn để có phần đuôi thì chọn theo toàn luồng; cột `select_metric` trong file cho biết đã chọn theo vùng nào; điểm lỗi vẫn nằm trong `sweep.jsonl` nhưng không bao giờ được chọn. Khi `RUN_SWEEP` tắt, cell không làm gì và `oracle_static.json` đang có được giữ nguyên.
"""

SWEEP = r"""
if RUN_SWEEP:
    rc = run_cli("conf_pipeline.sweep", OUT_DIR, *SWEEP_ARGS)
else:
    print("RUN_SWEEP = False: bỏ qua lượt quét; oracle_static.json giữ nguyên.")
"""

ORACLE_MD = r"""
## 5. Chạy lại oracle tĩnh

Khi `RUN_ORACLE` bật, cell dưới chạy `conf_pipeline.run --arms oracle_static` với các seed trong `ORACLE_SEEDS`: mỗi cặp (thuật toán, luồng) chạy lại đúng cấu hình mà `oracle_static.json` đã chọn, như một arm một thành viên không đổi. Ở preset `full` các seed này (2 đến 4) tách khỏi seed 1 của lượt quét, nên trần không được đo trên chính lần chạy đã chọn ra nó; ở `tiny` và `pilot` trần dùng lại seed 1. Cặp nào chưa có oracle được ghi là `skipped` kèm lý do, vì vậy cell này phải chạy sau bước quét.
"""

ORACLE = r"""
if RUN_ORACLE:
    rc = run_cli("conf_pipeline.run", OUT_DIR, *ORACLE_ARGS)
else:
    print("RUN_ORACLE = False: bỏ qua; headroom lấy điểm tốt nhất của lượt quét làm trần.")
"""

REPORT_MD = r"""
## 6. Báo cáo

Cell dưới gọi `report.write_report`, ghi `report.md` cùng các CSV và hình đường cong theo cửa sổ vào `OUT_DIR/report/`, rồi hiện các bảng chính ngay trong notebook. Mọi con số chất lượng là ARI *lagged*: cấu hình đương nhiệm sau cửa sổ t−1 được chấm trên cửa sổ t. Các bảng lần lượt là tóm tắt theo (thuật toán, luồng, arm), chênh lệch so với `default`, trần tĩnh và headroom, phân rã regret, tương quan giữa silhouette và ARI, thích nghi với drift và Table A; cách đọc từng bảng nằm ở phần ghi chú cuối notebook. Preset `tiny` chỉ có 3 cửa sổ nên đặt `FREEZE_AFTER_WINDOW = 1`, để phần đuôi gồm các cửa sổ từ 1 trở đi; con số của nó chỉ dùng để kiểm tra đường ống, không để đọc kết quả.
"""

REPORT = r"""
written = report.write_report(OUT_DIR)
plt.switch_backend(BACKEND)
print(f"Đã ghi {len(written)} file vào {OUT_DIR / 'report'}")

summary = report.summary_table(OUT_DIR)
show("Tóm tắt: trung bình qua seed", summary,
     ["algo", "fold", "arm", "n_seeds", "lag_ari_win_tail_mean", "lag_ari_win_tail_sd",
      "lag_ari_n_windows_tail", "incumbent_throughput_sps"])
show("Arm so với default", report.gain_table(OUT_DIR))
show("Trần tĩnh và headroom", report.headroom_table(OUT_DIR))
regret = report.regret_table(OUT_DIR)
show("Phân rã regret trên phần đuôi (region = tail)", regret[regret["region"] == "tail"])
proxy = report.proxy_table(OUT_DIR)
show("Silhouette so với ARI trên phần đuôi (region = tail)", proxy[proxy["region"] == "tail"] if not proxy.empty else proxy)
show("Thích nghi với drift", report.drift_table(OUT_DIR))
show("Table A: phân cụm lại chung trên micro-cluster", summary,
     ["algo", "fold", "arm", "n_seeds", *report.TABLE_A_FIELDS])
"""

ANALYSIS_MD = r"""
## 7. Phân tích giữa các arm

Cell dưới gọi `analyze.write_analysis`, ghi `analysis.md`, các CSV và hình F1–F4 vào `OUT_DIR/report/`, rồi hiện các bảng tính trên cùng độ đo `lag_ari_win_tail_mean`. Bảng đầu cho hạng trung bình của từng thuật toán dưới mỗi arm (hạng 1 là tốt nhất) kèm p của kiểm định Friedman và khoảng cách tới hạn `cd` của Nemenyi. Bảng thứ hai là ma trận Kendall τ giữa thứ hạng thuật toán của các arm, bảng thứ ba là cùng ma trận đó nhưng chỉ tính trên các thuật toán mà cả hai arm đều có số đo, và bảng thứ tư đếm theo arm các ô mà cấu hình không tạo được phân hoạch dùng được. Sau đó là số luồng mà `confstream` thắng, hoà hoặc thua `default` với mỗi thuật toán, bảng Pareto giữa chất lượng và thông lượng, và cuối cùng là các hình. Phiên bản cài đặt (`DenStreamPatchedFast` ở preset `full`) không có mặt trong các bảng xếp hạng này; bảng cuối so nó với DenStream theo từng arm (T7). Preset `tiny` chỉ có một thuật toán nên không có thứ hạng nào để so, vì vậy Kendall τ ngoài đường chéo, p của Friedman và `cd` để trống.
"""

ANALYSIS = r"""
written = analyze.write_analysis(OUT_DIR)
plt.switch_backend(BACKEND)
print(f"Đã ghi {len(written)} file; tóm tắt ở {OUT_DIR / 'report' / 'analysis.md'}")

summary = report.summary_table(OUT_DIR)
present = set(summary["arm"]) if "arm" in summary else set()
arms = [a for a in analyze.ARM_ORDER if a in present] + sorted(present - set(analyze.ARM_ORDER))
if not arms:
    print("Chưa có run nào để phân tích.")
else:
    rows = []
    for arm in arms:
        fn = analyze.friedman_nemenyi(analyze.rank_matrix(summary, arm))
        rows.append({"arm": arm, **fn["avg_rank"].to_dict(), "n_folds": fn["n_folds"],
                     "friedman_p": fn["pvalue"], "cd": fn["cd"]})
    show("Hạng trung bình theo arm (1 = tốt nhất) và CD Nemenyi", pd.DataFrame(rows).set_index("arm"))
    show("Kendall τ giữa thứ hạng thuật toán của các arm", analyze.kendall_matrix(summary, arms))
    show("Kendall τ, chỉ trên các thuật toán mà cả hai arm đều có số đo",
         analyze.kendall_matrix(summary, arms, complete_only=True))
    show("Ô hỏng theo arm (lag_undefined_share > 0,5)", analyze.degenerate_table(summary))
    if {"default", "confstream"} <= present:
        show("confstream so với default: thắng / hoà / thua theo luồng",
             analyze.win_tie_loss(summary, "confstream"))
    show("Pareto: ARI tail và thông lượng", analyze.pareto_table(summary))
    impl = analyze.impl_table(summary)
    if not impl.empty:
        show("T7: phiên bản cài đặt so với thuật toán gốc, theo arm và luồng", impl)
for path in written:
    if path.suffix == ".png":
        display(Markdown(f"`{path.name}`"))
        display(Image(filename=str(path)))
"""

ROBUSTNESS_MD = r"""
## 8. Độ vững của kết luận (A1–A5)

Cell dưới kiểm tra các kết luận ở trên phụ thuộc bao nhiêu vào độ đo, vào cấu hình tham chiếu và vào số seed, mà không chạy thêm lượt tune nào. Trước hết nó chạy `conf_pipeline.label_reach`: trên từng cửa sổ của từng luồng, k-means và Ward được cho biết số lớp thật, còn DBSCAN được chọn bán kính theo ARI, mỗi phương pháp chạy trên dữ liệu thô và trên dữ liệu chuẩn hoá bằng thống kê của chính cửa sổ đó. Kết quả là một mức tham khảo cho biết nhãn lớp đạt được tới đâu khi được biết trước số lớp. Đây không phải cận trên: các phương pháp này chỉ thấy một cửa sổ và (trừ DBSCAN) dùng đúng số cụm bằng số lớp, nên một thuật toán luồng giữ nhiều cụm hơn hoặc dùng lịch sử của các cửa sổ trước vẫn có thể vượt nó. Sau đó `robustness.write_robustness` đọc các log đã có và ghi `robustness.md` cùng các CSV trong `report/robustness/`. Các bảng gồm: tỉ lệ thu được và hai phần mất mát theo nhóm luồng kèm khoảng tin cậy bootstrap (lấy lại luồng trong từng nhóm, rồi lấy lại seed trong từng cặp); mức đồng thuận của thứ hạng thuật toán dưới ARI, NMI và purity; cấu hình tham chiếu chọn trên các cửa sổ chẵn của phần đuôi và đọc trên các cửa sổ lẻ; kiểm định Wilcoxon theo từng thuật toán có hiệu chỉnh Holm; số cửa sổ một lớp bị loại và thứ hạng khi bỏ KDD99; và bảng cho biết nhãn của luồng nào đủ dùng để chấm phân cụm. `N_BOOT` là số lần lấy mẫu lại. Hai ngưỡng của bảng cuối chỉ là giá trị tạm, đổi được bằng tham số của `write_robustness`. Preset `tiny` chạy trên một luồng không thuộc nhóm luồng nào của giao thức, nên cell bỏ qua.
"""

ROBUSTNESS = r"""
N_BOOT = 2000

unknown = [f for f in FOLDS if f not in C.FOLDS]
if unknown:
    print(f"Bỏ qua: các luồng {unknown} không thuộc nhóm luồng nào của giao thức.")
else:
    rc = run_cli("conf_pipeline.label_reach", OUT_DIR, "--folds", ",".join(FOLDS), "--n-samples", N_SAMPLES,
                 "--h", H, "--workers", WORKERS)
    written = robustness.write_robustness(OUT_DIR, n_boot=N_BOOT)
    print(f"Đã ghi {len(written)} file vào {OUT_DIR / 'report'}")
    display(Markdown((OUT_DIR / "report" / "robustness.md").read_text(encoding="utf-8")))
"""

BUDGET_MD = r"""
## 9. Ngân sách thời gian

Cell dưới đọc `runs.jsonl`, đếm job theo arm và trạng thái, rồi tính `wall_s` trung bình của các job `ok` theo (thuật toán, arm) và ngoại suy ra preset `full` (các thuật toán của nó × 18 luồng × 3 seed × 3 arm, cộng lượt quét, các incumbent chạy lại và lần chạy lại oracle), với giả định luôn có `WORKERS` job chạy song song. Thời gian của một job tăng gần tuyến tính theo số cửa sổ, nên preset chạy ít mẫu hơn mức cắt 100 000 (`smoke`: 25 000) được nhân lên theo tỉ lệ đó. Arm mà preset hiện tại không chạy (`frozen_prefix` ở `smoke`) được ước từ tỉ lệ chi phí của nó so với `confstream`, đo ở `conf_pipeline/out/pilot` nếu có, nếu không thì dùng tỉ lệ đã đo ở pilot ngày 01/10. Bảng theo thuật toán cho biết thuật toán nào chiếm phần lớn ngân sách, và bảng cuối liệt kê các job mà thời gian ngoại suy vượt `TIMEOUT_S` của `full`: những job đó sẽ chiếm một worker tới khi hết giờ rồi ghi `timeout`. Mọi con số ở đây là ước lượng thô; với `tiny` chúng chỉ để kiểm tra cách tính.
"""

BUDGET = r"""
#: Cost of an arm relative to confstream, measured in the pilot of 2026-10-01 (mean wall_s over
#: its 16 cells); used only when neither this preset nor conf_pipeline/out/pilot ran the arm.
PILOT_COST_RATIO = {"random_online": 1.646, "frozen_prefix": 0.366}

runs = pd.DataFrame([r for r in C.read_jsonl(OUT_DIR / "runs.jsonl") if r.get("kind") == "run"])
if runs.empty:
    print(f"Chưa có dòng nào trong {OUT_DIR / 'runs.jsonl'}.")
else:
    runs["wall_s"] = pd.to_numeric(runs.get("wall_s"), errors="coerce")
    ok = runs[runs["status"] == "ok"]
    show("Số job theo arm và trạng thái", runs.groupby(["arm", "status"]).size().unstack(fill_value=0))
    if not ok.empty:
        show("wall_s trung bình (giây) của job ok, theo (algo, arm)",
             ok.pivot_table(index="algo", columns="arm", values="wall_s", aggfunc="mean").round(1))

    full = PRESETS["full"]
    scale = C.N_SAMPLES_CAP / N_SAMPLES
    n_cells = len(full["ALGOS"]) * len(full["FOLDS"])

    ratio, ratio_source = dict(PILOT_COST_RATIO), "pilot 01/10"
    pilot_runs = REPO / "conf_pipeline" / "out" / "pilot" / "runs.jsonl"
    if pilot_runs.exists() and OUT_DIR.resolve() != pilot_runs.parent.resolve():
        p = pd.DataFrame([r for r in C.read_jsonl(pilot_runs) if r.get("status") == "ok"])
        if not p.empty and "confstream" in set(p["arm"]):
            pm = p.groupby("arm")["wall_s"].mean()
            ratio = {a: float(pm[a] / pm["confstream"]) for a in ratio if a in pm} or ratio
            ratio_source = str(pilot_runs)

    by_algo = ok.pivot_table(index="algo", columns="arm", values="wall_s", aggfunc="mean") * scale
    estimated = [a for a in ratio if a in full["ARMS"] and a not in by_algo and "confstream" in by_algo]
    for arm in estimated:
        by_algo[arm] = by_algo["confstream"] * ratio[arm]
    mean_wall = by_algo.mean()

    sweep_ok = [r for r in C.read_jsonl(OUT_DIR / "sweep.jsonl") if r.get("status") == "ok"]
    is_inc = [(r.get("config_id") or 0) >= C.INCUMBENT_ID_BASE for r in sweep_ok]
    sobol_wall = [r.get("wall_s") for r, inc in zip(sweep_ok, is_inc) if not inc]
    inc_wall = [r.get("wall_s") for r, inc in zip(sweep_ok, is_inc) if inc]
    cells_here = len({(r["algo"], r["fold"]) for r in sweep_ok}) or 1
    # Each online run (arm x seed) contributes at most two incumbents.
    online_runs_full = sum(a in ("confstream", "random_online", "frozen_prefix") for a in full["ARMS"]) * len(full["SEEDS"])
    online_runs_here = max(sum(a in ("confstream", "random_online", "frozen_prefix") for a in ARMS) * len(SEEDS), 1)
    inc_per_cell = len(inc_wall) / cells_here * online_runs_full / online_runs_here

    plan = [(arm, mean_wall.get(arm), n_cells * len(full["SEEDS"])) for arm in full["ARMS"]]
    plan.append(("oracle_static", mean_wall.get("oracle_static"), n_cells * len(full["ORACLE_SEEDS"])))
    plan.append(("sweep", pd.Series(sobol_wall, dtype=float).mean() * scale, n_cells * (full["SWEEP_N_POINTS"] + 1)))
    plan.append(("sweep_incumbents", pd.Series(inc_wall, dtype=float).mean() * scale, round(n_cells * inc_per_cell)))
    budget = pd.DataFrame(plan, columns=["step", "mean_wall_s", "n_jobs_full"]).set_index("step")
    budget["mean_wall_s"] = budget["mean_wall_s"].astype(float)
    budget["cpu_h"] = budget["mean_wall_s"] * budget["n_jobs_full"] / 3600
    budget["wall_h"] = budget["cpu_h"] / WORKERS
    show(f"Ngoại suy ra preset full (wall_h với {WORKERS} worker)", budget.round(2))

    protocol = budget.loc[list(full["ARMS"])]
    print(f"{len(full['ARMS'])} arm, {protocol['n_jobs_full'].sum()} job: ~{protocol['cpu_h'].sum():.1f} giờ CPU, "
          f"~{protocol['wall_h'].sum():.1f} giờ với {WORKERS} worker")
    print(f"Tính cả sweep, incumbent và oracle: ~{budget['cpu_h'].sum():.1f} giờ CPU, "
          f"~{budget['wall_h'].sum():.1f} giờ ({budget['wall_h'].sum() / 24:.1f} ngày) với {WORKERS} worker")

    arms_here = [a for a in full["ARMS"] if a in by_algo]
    per_algo = (by_algo[arms_here].sum(axis=1) * len(full["FOLDS"]) * len(full["SEEDS"]) / 3600)
    show(f"Giờ CPU của {len(full['ARMS'])} arm ở preset full, theo thuật toán", per_algo.rename("cpu_h").round(1).to_frame())

    projected = ok.assign(projected_s=ok["wall_s"] * scale)
    over = projected[projected["projected_s"] > full["TIMEOUT_S"]]
    show(f"Job có thể vượt TIMEOUT_S = {full['TIMEOUT_S']} s của full", over,
         ["algo", "fold", "arm", "wall_s", "projected_s"])

    missing = budget.index[budget["mean_wall_s"].isna()].tolist()
    if missing:
        print(f"Chưa có số đo cho {missing}: các dòng này không được cộng vào tổng.")
    if estimated:
        print(f"{estimated} không chạy ở preset này: ước bằng tỉ lệ chi phí so với confstream "
              f"{ {a: round(ratio[a], 2) for a in estimated} } ({ratio_source}).")
    if scale > 10:
        print(f"Preset này chạy {N_SAMPLES} mẫu, quá ít để ngoại suy; số ở đây chỉ để kiểm tra cách tính.")
    elif scale != 1:
        print(f"Preset này chạy {N_SAMPLES} mẫu: mọi thời gian đã nhân {scale:g} theo số cửa sổ.")
"""

NOTES = r"""
## 10. Ghi chú: cách đọc từng bảng

**Tóm tắt.** Mỗi dòng là một bộ (thuật toán, luồng, arm), lấy trung bình qua các seed đã chạy xong (`n_seeds`), và cột `_sd` là độ lệch chuẩn giữa các seed. Mọi cột chất lượng là ARI lagged, tức cấu hình đương nhiệm sau cửa sổ t−1 được chấm trên cửa sổ t, nên không con số nào là điểm của một cấu hình trên chính cửa sổ đã chọn ra nó. Cột có `tail` chỉ lấy các cửa sổ từ `FREEZE_AFTER_WINDOW` trở đi (20 ở `pilot` và `full`, 1 ở `tiny`), vùng mà mọi arm cùng có và `frozen_prefix` đã thôi tune, nên đây là con số chính để so các arm; cột không có `tail` lấy toàn luồng, gồm cả giai đoạn đầu khi ensemble còn đang dò. `incumbent_throughput_sps` là thông lượng (mẫu mỗi giây) của riêng cấu hình đương nhiệm, khác với chi phí của cả ensemble ở `ensemble_runtime_s`. Cửa sổ nào chỉ chứa một lớp thật thì không có ARI, NMI hay purity: ở đó ARI bằng 1 nếu thuật toán trả về đúng một nhóm và bằng 0 trong mọi trường hợp khác, còn purity luôn bằng 1, nên con số chỉ cho biết thuật toán có trả về một nhóm hay không. Các cửa sổ này bị bỏ khỏi mọi phép trung bình; `n_single_class_windows` đếm chúng, còn `lag_ari_n_windows` và `lag_ari_n_windows_tail` là số cửa sổ mà ARI lagged trên toàn luồng và trên đuôi thực sự dựa vào (kdd99 chỉ còn khoảng 23 trên 80 cửa sổ đuôi).

**Arm so với default.** `gain_tail` và `gain_all` là ARI lagged của arm trừ đi của `default`, trên phần đuôi và trên toàn luồng. Gain dương trên đuôi nhưng âm trên toàn luồng nghĩa là arm trả giá ở giai đoạn dò rồi mới vượt default. `throughput_ratio` là thông lượng của cấu hình đương nhiệm của arm chia cho của `default`, nên nhỏ hơn 1 nghĩa là cấu hình được tune chạy chậm hơn default.

**Trần tĩnh và headroom.** Trần (`ceiling`) là ARI lagged của oracle tĩnh: trung bình các lần chạy lại `oracle_static` khi có, nếu không thì điểm tốt nhất của lượt quét (`ceiling_source` cho biết nguồn, `metric` cho biết đo trên đuôi hay toàn luồng). Oracle được chọn theo phần đuôi, đúng vùng mà trần được đọc, để một cấu hình chỉ mạnh ở giai đoạn đầu không giành được trần. Một cặp chỉ được coi là tune được (`tunable`) khi `gain = ceiling − default` đạt ít nhất `MIN_CEILING_GAIN` = 0,05 và lớn hơn 1,645 lần sai số chuẩn của default qua seed; với một seed thì sai số chuẩn bằng 0 và cổng chỉ còn điều kiện 0,05. Cặp không qua cổng để trống mọi cột headroom và ghi lý do ở `note`, thay vì đưa ra một tỉ số của nhiễu. Với cặp tune được, `headroom_<arm> = (arm − default) / gain`, trong đó 0 là ngang default, 1 là đạt trần tĩnh, số âm là tệ hơn default và số lớn hơn 1 là tune online vượt cả cấu hình tĩnh tốt nhất; Cột `evolution_value = headroom_confstream − headroom_random_online`, phần đóng góp riêng của cơ chế tiến hoá so với rút cấu hình ngẫu nhiên, chỉ xuất hiện khi arm tuỳ chọn `random_online` đã được chạy.

**Phân rã regret.** Khoảng cách từ trần tới điểm người dùng thật sự nhận được, `A_static − a_lag`, được tách theo hai cách, lấy trung bình theo cửa sổ rồi theo seed. Cách chính cho việc chọn bằng nhãn chịu cùng độ trễ một cửa sổ như silhouette: `a_star_lag` là ARI trên cửa sổ t của thành viên mà nhãn chọn sau cửa sổ t−1, trong số các thành viên còn hoạt động ở cửa sổ t và khả thi ở cửa sổ t−1 (hoà thì giữ cấu hình đương nhiệm). `R_proxy_lag = a_star_lag − a_lag` là phần mất vì chọn bằng silhouette thay vì bằng nhãn, còn `R_search_lag = A_static − a_star_lag` là phần mà các cấu hình trong ensemble vẫn thua cấu hình tĩnh tốt nhất; số âm nghĩa là đổi cấu hình theo thời gian đã thắng cấu hình tĩnh tốt nhất. Hai phần này cộng lại đúng bằng khoảng cách, và hình F3 vẽ cách tách này. Cách thứ hai so với `a_star`, thành viên tốt nhất trên chính cửa sổ t, được chọn bằng nhãn của chính cửa sổ đó: `A_static − a_lag = R_lag + R_proxy + R_search`, với `R_lag = a_inc − a_lag` là cái giá của việc chậm một cửa sổ, `R_proxy = a_star − a_inc` và `R_search = A_static − a_star`. Khi các thành viên có ARI gần nhau, thành viên đứng đầu một cửa sổ thường chỉ đứng đầu nhờ may mắn của cửa sổ đó, điều mà không cách chọn trước nào có được, kể cả khi có nhãn; vì vậy cách thứ hai làm `R_proxy` lớn hơn thực tế và `R_search` nhỏ hơn thực tế, và chỉ nên đọc như một cận. Bảng trong notebook chỉ hiện vùng đuôi (`region = tail`), tức các cửa sổ từ `tail_from` trở đi, và ở vùng này `A_static` là ARI trên đuôi của oracle tĩnh; `regret.csv` và `report.md` có thêm dòng `region = all` cho toàn luồng, khi đó `A_static` là ARI toàn luồng của oracle. Cột `ceiling_source` cho biết trần lấy từ các lần chạy lại oracle hay từ lượt quét. Các cột trừ `R_lag` cần log thành viên (`LOG_MEMBERS = True`), còn `R_search` và `R_search_lag` cần thêm oracle tĩnh; thiếu thì cột để trống và `note` nói lý do.

**Silhouette so với ARI.** Trên mỗi cửa sổ, hệ số tương quan thứ hạng Spearman được tính giữa silhouette và ARI của các thành viên được chấm trên cửa sổ đó (khả thi, có silhouette xác định, tính cả thành viên đang thử; cần ít nhất ba thành viên và cả hai đại lượng không hằng), rồi lấy trung bình theo cửa sổ và theo seed. Hệ số bằng 1 khi silhouette xếp các cấu hình đúng thứ tự như ARI, và gần 0 khi hai thứ tự không liên quan. Cột `ari_spread` là khoảng chênh trung bình giữa ARI cao nhất và thấp nhất của các thành viên đó trong một cửa sổ. Khi khoảng chênh dưới 0,05, ARI gần như không phân biệt được các cấu hình, nên tương quan với nó chỉ là nhiễu; khi đó cột `spearman` để trống và `note` ghi lý do. Điều kiện này xét các thành viên trong cùng một cửa sổ chứ không xét trần tĩnh, vì một luồng mà không cấu hình cố định nào hơn default vẫn có thể có các thành viên khác nhau rõ ở từng cửa sổ. `R_proxy_lag` cho biết lựa chọn của silhouette làm mất bao nhiêu ARI, còn bảng này cho biết silhouette xếp toàn bộ các cấu hình đúng tới đâu.

**Thích nghi với drift.** Bảng chỉ có dòng cho các luồng tổng hợp có điểm drift (`scale_drift`, `density_drift`, `k_drift`, `mixed_drift`). `pre` là ARI lagged ở cửa sổ ngay trước drift, và `drop_depth` là `pre` trừ ARI lagged thấp nhất trong 10 cửa sổ kể từ drift, tức độ sâu của cú rơi. `recovery_windows` là số cửa sổ cho tới khi ARI lagged nằm trong khoảng 0,05 dưới mốc `ref` suốt 3 cửa sổ liên tiếp, và để trống nếu không hồi phục trước drift kế tiếp; `ref` là thành viên tốt nhất đang hoạt động khi có log thành viên và hơn một thành viên, nếu không thì là `pre`. `param_ratio` chỉ có ở `scale_drift` và `mixed_drift`: đó là tham số khoảng cách của cấu hình đương nhiệm sau khi hồi phục chia cho giá trị trước drift, nên một bộ tune bám được thang dữ liệu sẽ cho tỉ số gần hệ số nhân của thang.

**Table A.** Mỗi thuật toán luồng gộp micro-cluster thành cụm bằng pha offline riêng, nên chênh lệch ARI giữa hai thuật toán trộn lẫn pha online với pha offline. Table A đưa micro-cluster của cấu hình lagged của mọi thuật toán qua cùng một bộ phân cụm lại, gồm k-means với k bằng số lớp thật (`lag_ari_A_kmeans_*`) và DBSCAN với `eps = 0,5·s` (`lag_ari_A_dbscan_*`), nên phần chênh lệch còn lại giữa các thuật toán là của pha online. `lag_n_micro_centers_mean` là số micro-cluster, còn `lag_ssq_micro_mean` là tổng bình phương khoảng cách từ các điểm của cửa sổ tới tâm micro-cluster gần nhất.

**Thứ hạng và Kendall τ.** Trên mỗi luồng, các thuật toán được xếp hạng theo `lag_ari_win_tail_mean` dưới từng arm, và thuật toán không có số đo trên một luồng nhận hạng kém nhất. Một ô chỉ được tính khi mọi seed mà arm đó đã chạy đều xong (`all_seeds_ok`); ô thiếu seed bị coi như không có số đo, nên nhận hạng kém nhất và không vào Kendall τ, bảng thắng/hoà/thua hay bảng Pareto. Bảng hạng trung bình đi kèm p của kiểm định Friedman và CD Nemenyi: hai thuật toán có hạng trung bình lệch nhau hơn CD thì khác nhau ở mức α = 0,05. Kendall τ giữa hai arm là trung bình qua các luồng của τ-b giữa hai vector hạng, với τ = 1 là cùng thứ tự, 0 là không liên quan và −1 là đảo ngược. Đây là con số chính của bài báo: τ giữa `default` và các arm tune online thấp hơn hẳn 1 nghĩa là thứ hạng thuật toán mà một benchmark công bố phụ thuộc vào chế độ tune, còn τ gần 1 nghĩa là so sánh ở default là đủ. Có hai điều cần đọc kèm. Thứ nhất, vì thuật toán thiếu số đo nhận hạng kém nhất, một job lỗi hoặc quá giờ chỉ ở arm đắt sẽ kéo τ xuống dù chất lượng không đổi; `kendall_complete.csv` và hình `F2_kendall_complete.png` tính lại τ chỉ trên các thuật toán mà cả hai arm đều có số đo, nên chênh lệch giữa hai bảng chính là tác động đó. Thứ hai, `degenerate.csv` đếm theo arm các ô mà cấu hình lagged không tạo được phân hoạch dùng được ở hơn một nửa số cửa sổ (`lag_undefined_share > 0,5`). ARI gần 0 ở các ô này là do cấu hình hỏng chứ không phải do phân cụm kém, nên τ giữa `default` và các arm khác phải đọc cùng con số này.

**Thắng, hoà, thua.** Với mỗi thuật toán, trên các luồng mà cả `confstream` và `default` đều có số đo, `confstream` thắng khi chênh lệch lớn hơn dải nhiễu `max(sd_arm, sd_default)` giữa các seed, thua khi nhỏ hơn âm của dải đó, và hoà trong các trường hợp còn lại. Với một seed thì dải nhiễu bằng 0, nên mọi chênh lệch khác 0 đều thành thắng hoặc thua, và bảng này chỉ đáng đọc khi có từ hai seed trở lên. `worst_gap` là chênh lệch âm nhất.

**Pareto.** `ari` là trung bình qua các luồng của ARI lagged trên đuôi, còn `sps` là thông lượng trung bình của cấu hình đương nhiệm. `hv` được tính trên từng luồng: ARI và log10 thông lượng được chuẩn hoá min–max trên mọi cặp (thuật toán, arm) của luồng đó, điểm của cặp chiếm một diện tích trong hình vuông đơn vị, và `hv` là trung bình diện tích ấy qua các luồng. `hv` càng gần 1 thì cặp đó càng vừa tốt vừa nhanh so với các cặp khác trên cùng luồng.
"""

CELLS: list[tuple[str, str]] = [
    ("markdown", TITLE),
    ("code", SETUP),
    ("markdown", PANEL_MD),
    ("code", PANEL),
    ("markdown", DRY_RUN_MD),
    ("code", DRY_RUN),
    ("markdown", RUN_MD),
    ("code", RUN),
    ("markdown", SWEEP_MD),
    ("code", SWEEP),
    ("markdown", ORACLE_MD),
    ("code", ORACLE),
    ("markdown", REPORT_MD),
    ("code", REPORT),
    ("markdown", ANALYSIS_MD),
    ("code", ANALYSIS),
    ("markdown", ROBUSTNESS_MD),
    ("code", ROBUSTNESS),
    ("markdown", BUDGET_MD),
    ("code", BUDGET),
    ("markdown", NOTES),
]


def build() -> nbformat.NotebookNode:
    """The notebook, cells in order, with fixed ids and no outputs."""
    nb = v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for i, (kind, source) in enumerate(CELLS):
        new = v4.new_markdown_cell if kind == "markdown" else v4.new_code_cell
        nb.cells.append(new(source.strip("\n"), id=f"otb-{i:02d}"))
    nbformat.validate(nb)
    return nb


def main(path: Path = NOTEBOOK) -> None:
    nbformat.write(build(), path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
