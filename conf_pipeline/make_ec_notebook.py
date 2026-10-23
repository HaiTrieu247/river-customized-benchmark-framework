"""Write ``otb_ec_variants.ipynb`` at the framework root: can the online tuner run with fewer members
and challengers, at what cost in quality and for what gain in speed?

Generated from the cell sources below, like ``make_notebook`` (whose setup cell it reuses), so it is
reproducible and testable (``tests/test_ec_notebook.py`` regenerates it and executes it in the
``tiny`` preset). Edit the cells here, never the ``.ipynb``, then run from the repo root::

    python -m conf_pipeline.make_ec_notebook

Cell ids are fixed, so regenerating an unchanged notebook leaves the file byte-identical. Markdown
cells are in Vietnamese; code and identifiers in English.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat import v4

from .make_notebook import SETUP as BASE_SETUP

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "otb_ec_variants.ipynb"

# --- cells, in notebook order ----------------------------------------------------------

TITLE = r"""
# Có giảm được số member và số cấu hình thử của bộ tune online không?

Bộ tune online (`confstream`) giữ một ensemble gồm E cấu hình chạy song song trên luồng, và sau mỗi cửa sổ sinh thêm C cấu hình thử để thách đấu chúng. Giao thức hiện dùng E = 20 và C = 10, như bài confStream gốc. Mỗi cửa sổ vì vậy chạy 30 mô hình, và chi phí của bộ tune gần như tỉ lệ với con số đó. Notebook này hỏi: nếu giảm E và C thì chất lượng tụt bao nhiêu, và tốc độ tăng bao nhiêu?

Sáu bộ E/C được chạy, mỗi bộ vào một thư mục riêng dưới `OUT_ROOT`. Bộ mốc `ec20_10` là giao thức hiện tại, chạy lại tại đây chứ không lấy từ `out/full`, vì `out/full` chạy với runtime gate còn mọi bộ ở đây chạy không gate. Ba bộ `ec15_7`, `ec10_5`, `ec5_2` thu nhỏ ensemble nhưng giữ tỉ lệ khoảng hai member một cấu hình thử. Hai bộ `ec20_5`, `ec20_2` giữ 20 member và chỉ giảm số cấu hình thử. Mọi thứ khác giữ nguyên: 12 luồng đã khóa cho pilot, mức cắt 100 000 mẫu, cửa sổ 1 000 mẫu, phần đuôi từ cửa sổ 20. Seed theo chính sách `split` (`docs/SEED_RULES.md`): bộ tune chạy một tuner seed (101), mô hình dùng model seed 201 ở mọi bộ, nên các bộ chỉ khác nhau ở ngân sách của ensemble.

Cờ `ORACLES` ở bảng điều khiển chọn có chạy thêm hai bộ tune chẩn đoán chọn theo nhãn (E4) hay không: `confstream_ari_oracle` chấm mọi member bằng ARI của cửa sổ vừa xong, còn `confstream_ari_oracle_constrained` làm như vậy nhưng giữ điều kiện của silhouette. Mặc định là `both`, chạy cả hai cạnh `confstream` trong mọi bộ, để thấy việc giảm ngân sách làm mất bao nhiêu ở một bộ tune có tín hiệu chọn hoàn hảo. `none` thì chỉ chạy `confstream`.

Mỗi bộ được so với bộ mốc của chính arm đó, trên cùng (thuật toán, luồng, seed). Chất lượng là tail ARI lagged. Tốc độ là tỉ lệ thời gian của bộ mốc so với bộ đó, đo theo wall time và theo thời gian chạy mô hình, cộng số mẫu mỗi giây của cả bộ tune. Chỉ có một seed, nên một chênh lệch nhỏ hơn độ dao động theo seed của bộ tune online (khoảng 0,005 đến 0,01 ARI trong `out/full`) chưa đọc được thành kết luận.
"""

SETUP = BASE_SETUP + r'''

from conf_pipeline import cost, variants as V  # noqa: E402
'''

PANEL_MD = r"""
## Bảng điều khiển

Cell dưới chọn preset theo `OTB_EC_PRESET` (mặc định `full`) và thư mục kết quả theo `OTB_EC_OUT_DIR` (mặc định `conf_pipeline/out/ec_variants`). Preset `full` là thiết kế đã chốt: 8 thuật toán trên 12 luồng đã khóa, tuner seed 101 và model seed 201, 8 worker, giới hạn 24 giờ mỗi job. Lượt thứ nhất chạy sáu thuật toán rẻ cho mọi bộ, lượt thứ hai chạy CluStream và RepStream, nên kết quả đầu tiên có sau vài giờ. Preset `tiny` chỉ để kiểm tra đường ống trong vài phút, con số của nó không dùng để đọc kết quả.

`ORACLES` nhận `none`, `unconstrained`, `constrained` hoặc `both`; mỗi oracle tốn xấp xỉ một lần chạy `confstream`, nên `both` làm chi phí gần gấp ba. Các ngưỡng kết luận là giá trị tạm, đổi được ở đây. Một bộ được coi là giữ được chất lượng khi mức thay đổi trung bình không tệ hơn `-MAX_MEAN_LOSS` và không quá `MAX_SHARE_OVER_BAND` số cặp tụt quá `BAND`. `MIN_SPS` là tốc độ thấp nhất của cả bộ tune mà bài confStream gốc còn chấp nhận (CluStream trên powersupply, 15 mẫu mỗi giây).
"""

PANEL = r"""
from conf_pipeline import variants as V

PRESET = os.environ.get("OTB_EC_PRESET", "full")
OUT_ROOT = Path(os.environ.get("OTB_EC_OUT_DIR", REPO / "conf_pipeline" / "out" / "ec_variants"))

#: The streams locked for the mechanism pilot on 2026-10-23 (docs/MECHANISM_PILOT.md), one per source block.
LOCKED_FOLDS = ["covtype", "kdd99", "electricity", "shuttle", "insects_abrupt_imbalanced",
                "scale_drift", "density_drift", "k_drift", "mixed_drift", "static_overlap",
                "incremental_drift", "evolving"]
PRESETS = {
    "full": dict(ALGOS_FIRST=["DenStream", "HPStream", "DBSTREAM", "STREAMKMeans", "SOStream", "BIRCH"],
                 ALGOS_SECOND=["CluStream", "RepStream"], FOLDS=LOCKED_FOLDS, TUNER_SEEDS=[101], MODEL_SEED=201,
                 N_SAMPLES=C.N_SAMPLES_CAP, H=C.H, FREEZE_AFTER_WINDOW=C.FREEZE_AFTER_WINDOW,
                 WORKERS=8, TIMEOUT_S=V.EC_TIMEOUT_S),
    "tiny": dict(ALGOS_FIRST=["DenStream"], ALGOS_SECOND=[], FOLDS=["static_easy"], TUNER_SEEDS=[101], MODEL_SEED=201,
                 N_SAMPLES=900, H=300, FREEZE_AFTER_WINDOW=1, WORKERS=2, TIMEOUT_S=600),
}
if PRESET not in PRESETS:
    raise SystemExit(f"OTB_EC_PRESET={PRESET!r}; presets: {sorted(PRESETS)}")
globals().update(PRESETS[PRESET])
GRID = list(V.EC_GRID)
#: The oracle tuners run beside confstream in every setting: "none", "unconstrained", "constrained" or "both".
ORACLES = "both"
BAND = V.EC_BAND
MAX_MEAN_LOSS = V.EC_MAX_MEAN_LOSS
MAX_SHARE_OVER_BAND = V.EC_MAX_SHARE_OVER_BAND
MIN_SPS = V.EC_MIN_SPS
#: The projected cost reads the full run's measured costs, which had the runtime gate; a gate-free run
#: keeps its slow members. 1.2 is what the repeat smoke measured (an assumption here, not a measurement).
GATE_FREE_FACTOR = 1.2
RUN_FIRST = True
RUN_SECOND = True
# --- one-off overrides go here, e.g. WORKERS = 6 ---

# --- derived (nothing to edit below) ---
OUT_ROOT.mkdir(parents=True, exist_ok=True)
ARMS = V.ec_arms(ORACLES)


def ec_args(algos):
    return ["ec-run", "--grid", ",".join(GRID), "--algos", ",".join(algos), "--folds", ",".join(FOLDS),
            "--tuner-seeds", ",".join(map(str, TUNER_SEEDS)), "--model-seed", MODEL_SEED, "--n-samples", N_SAMPLES, "--h", H,
            "--freeze-after-window", FREEZE_AFTER_WINDOW, "--workers", WORKERS, "--timeout", TIMEOUT_S,
            "--oracles", ORACLES]


RUN_ARGS_FIRST = ec_args(ALGOS_FIRST)
RUN_ARGS_SECOND = ec_args(ALGOS_SECOND) if ALGOS_SECOND else None

for name in ("PRESET", "OUT_ROOT", "GRID", "ORACLES", "ARMS", "ALGOS_FIRST", "ALGOS_SECOND", "FOLDS", "TUNER_SEEDS", "MODEL_SEED", "N_SAMPLES", "H",
             "FREEZE_AFTER_WINDOW", "WORKERS", "TIMEOUT_S", "BAND", "MAX_MEAN_LOSS", "MAX_SHARE_OVER_BAND", "MIN_SPS",
             "RUN_FIRST", "RUN_SECOND"):
    print(f"{name:20s} = {globals()[name]}")
"""

DRY_RUN_MD = r"""
## Chạy thử và dự toán

Cell dưới in các lệnh sẽ chạy (`--dry-run`, không job nào chạy thật), rồi dự toán chi phí của từng bộ từ chi phí đo được của `conf_pipeline/out/full`. Dự toán giả định chi phí tỉ lệ với số mô hình chạy mỗi cửa sổ, (E + C) / 30 của bộ mốc, nhân thêm `GATE_FREE_FACTOR` vì không gate thì các member chậm không bị loại. Mỗi oracle được tính như một lần chạy `confstream`, vì nó chạy cùng số mô hình; đây cũng là giả định. Đây là giả định, không phải số đo. Riêng CluStream, chạy thử trên máy này cho thấy không gate chậm hơn nhiều so với hệ số đó.
"""

DRY_RUN = r"""
for args in (RUN_ARGS_FIRST, RUN_ARGS_SECOND):
    if args:
        run_cli("conf_pipeline.variants", OUT_ROOT, *args, "--dry-run")

SRC = REPO / "conf_pipeline" / "out" / "full"
if (SRC / "runs.jsonl").exists():
    rows = []
    for name in GRID:
        size = V.EC_GRID[name]
        factor = GATE_FREE_FACTOR * (size["e_size"] + size["e_test"]) / (C.E_SIZE + C.E_TEST)
        for part, algos in (("lượt 1", ALGOS_FIRST), ("lượt 2", ALGOS_SECOND)):
            if not algos:
                continue
            jobs = cost.plan(SRC, algos, FOLDS, list(ARMS), TUNER_SEEDS, N_SAMPLES, factor=factor,
                             proxy={arm: "confstream" for arm in ARMS if arm != "confstream"})
            s = cost.summarise(jobs, WORKERS)
            rows.append({"setting": name, "lượt": part, "jobs": s["n_jobs"], "chưa đo": s["n_unknown"],
                         "CPU-h": round(s["cpu_h"], 1), f"giờ trên {WORKERS} worker": round(s["makespan_h"], 1)})
    est = pd.DataFrame(rows)
    show("Dự toán theo bộ và theo lượt", est)
    print(f"Tổng: {est['CPU-h'].sum():.0f} CPU-h, khoảng {est['CPU-h'].sum() / WORKERS:.0f} giờ trên {WORKERS} worker")
else:
    print(f"Không có {SRC / 'runs.jsonl'}: bỏ qua dự toán.")
"""

RUN_FIRST_MD = r"""
## Lượt 1: sáu thuật toán rẻ, mọi bộ

Mỗi bộ chạy vào thư mục riêng của nó, lần lượt từng bộ, mỗi bộ dùng `WORKERS` job song song, mỗi job một nhân. Mỗi thư mục có manifest riêng, nên dừng cell rồi chạy lại thì các job đã xong được bỏ qua, và một lệnh khác protocol sẽ bị từ chối thay vì trộn kết quả.
"""

RUN_FIRST_CELL = r"""
if RUN_FIRST:
    run_cli("conf_pipeline.variants", OUT_ROOT, *RUN_ARGS_FIRST)
"""

RUN_SECOND_MD = r"""
## Lượt 2: CluStream và RepStream

Đây là phần lâu nhất. Một job CluStream của bộ mốc có thể chạy nhiều giờ, nên giới hạn mỗi job là `TIMEOUT_S`. Job nào vượt giới hạn được ghi trạng thái `timeout`, không có số chất lượng, và được đếm trong bảng kết luận như một job chưa xong. Đó cũng là một kết quả về tốc độ.
"""

RUN_SECOND_CELL = r"""
if RUN_SECOND and RUN_ARGS_SECOND:
    run_cli("conf_pipeline.variants", OUT_ROOT, *RUN_ARGS_SECOND)
"""

COMPARE_MD = r"""
## Kết quả

Cell dưới so mọi bộ với bộ mốc và ghi `report/ec/` (`runs.csv`, `pairs.csv`, `summary.csv`, `ec.md`). Nó đọc được cả khi mới có một phần kết quả, ví dụ sau lượt 1. `d_tail` là tail ARI của bộ đó trừ của bộ mốc trên cùng (thuật toán, luồng, seed). `speedup_wall` và `speedup_model` là thời gian của bộ mốc chia thời gian của bộ đó, lớn hơn 1 là nhanh hơn. `sps` là số mẫu mỗi giây của cả bộ tune. `tau_vs_base` là Kendall τ giữa thứ hạng các thuật toán của bộ đó và của bộ mốc, trung bình qua các luồng. `verdict` là `reducible` khi bộ đó vừa giữ được chất lượng theo các ngưỡng ở bảng điều khiển vừa nhanh hơn, `quality drops` khi chất lượng tụt quá ngưỡng, `no speed-up` khi không nhanh hơn, `undecided` khi chưa có cặp nào chạy xong ở cả hai phía.
"""

COMPARE = r"""
paths = V.write_ec(OUT_ROOT, GRID, band=BAND, max_mean_loss=MAX_MEAN_LOSS,
                   max_share_over_band=MAX_SHARE_OVER_BAND, min_sps=MIN_SPS)
runs, pairs, summary = (pd.read_csv(p) for p in paths[:3])
if runs.empty:
    print("Chưa có run nào.")
else:
    show("Số job theo bộ và trạng thái", runs.groupby(["setting", "status"]).size().unstack(fill_value=0))
cols = ["setting", "e_size", "e_test", "arm", "group", "n_pairs", "n_unfinished", "mean_base_tail", "mean_tail",
        "mean_d_tail", "min_d_tail", "share_loss_over_band", "tau_vs_base", "median_speedup_wall",
        "median_speedup_model", "median_sps", "share_sps_at_least_min", "verdict"]
show("Kết luận cho từng bộ, mọi luồng", summary[summary["group"] == "all"] if len(summary) else summary, cols)
show("Theo nhóm luồng", summary[summary["group"] != "all"] if len(summary) else summary, cols)
"""

PER_ALGO_MD = r"""
## Theo thuật toán

Với mỗi arm đã chạy, hai bảng dưới cho từng thuật toán: mức thay đổi tail ARI trung bình qua các luồng, và mức tăng tốc trung vị theo wall time. Một bộ có thể giảm được cho thuật toán này mà không giảm được cho thuật toán khác. Hình của mỗi arm đặt mỗi (bộ, thuật toán) thành một điểm: trục ngang là mức tăng tốc, trục dọc là mức thay đổi chất lượng. Điểm càng ở góc phải trên càng tốt; đường gạch là `-MAX_MEAN_LOSS`, đường chấm là `-BAND`.
"""

PER_ALGO = r"""
done = pairs.dropna(subset=["d_tail"]) if len(pairs) else pairs
if done.empty:
    print("Chưa có cặp nào chạy xong ở cả hai phía.")
for arm, by_arm in (done.groupby("arm", sort=False) if len(done) else []):
    show(f"{arm}: thay đổi tail ARI trung bình (thuật toán × bộ)",
         by_arm.pivot_table(index="algo", columns="setting", values="d_tail", aggfunc="mean").round(3))
    show(f"{arm}: tăng tốc trung vị theo wall time (thuật toán × bộ)",
         by_arm.pivot_table(index="algo", columns="setting", values="speedup_wall", aggfunc="median").round(2))
    per = by_arm.groupby(["setting", "algo"]).agg(d=("d_tail", "mean"), speed=("speedup_wall", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for setting, g in per.groupby("setting"):
        ax.scatter(g["speed"], g["d"], label=setting, s=36)
    ax.axhline(-MAX_MEAN_LOSS, ls="--", lw=1, color="0.4")
    ax.axhline(-BAND, ls=":", lw=1, color="0.4")
    ax.axvline(1.0, lw=1, color="0.6")
    ax.set_xscale("log")
    ax.set_title(arm)
    ax.set_xlabel("tăng tốc so với ec20_10 (wall time, trung vị)")
    ax.set_ylabel("thay đổi tail ARI so với ec20_10 (trung bình)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
"""

CELLS: list[tuple[str, str]] = [
    ("markdown", TITLE), ("code", SETUP),
    ("markdown", PANEL_MD), ("code", PANEL),
    ("markdown", DRY_RUN_MD), ("code", DRY_RUN),
    ("markdown", RUN_FIRST_MD), ("code", RUN_FIRST_CELL),
    ("markdown", RUN_SECOND_MD), ("code", RUN_SECOND_CELL),
    ("markdown", COMPARE_MD), ("code", COMPARE),
    ("markdown", PER_ALGO_MD), ("code", PER_ALGO),
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
        nb.cells.append(new(source.strip("\n"), id=f"ec-{i:02d}"))
    nbformat.validate(nb)
    return nb


def main(path: Path = NOTEBOOK) -> None:
    nbformat.write(build(), path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
