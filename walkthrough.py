"""walkthrough.py — Chạy thử toàn bộ framework benchmark, giải thích từng bước.

Mục tiêu framework: chạy NHIỀU thuật toán clustering (với nhiều setting: tham số
+ scaler) trên CÙNG một dataset, rồi so sánh hiệu năng (thời gian / RAM / số
cluster / throughput) và chất lượng (ARI/NMI/purity khi dataset có nhãn).

    Dataset -> [ thuật toán + params + scaler ] x N -> Runner -> Báo cáo
                                                          |
                        (runtime, RAM, số cluster, chất lượng)

Cách chạy:

    python walkthrough.py

Script in giải thích từng bước. Nếu có matplotlib, nó lưu vài biểu đồ vào
``walkthrough_out/``.
"""

from __future__ import annotations

import os
import sys

# Console Windows mặc định cp1252 -> không in được tiếng Việt. Ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from benchmark.datasets import StreamConfig, SyntheticStream
from benchmark.algorithms.registry import get_algorithm, ALGORITHMS, SCALERS
from benchmark.evaluation.runner import run_once
from benchmark.experiments import BenchmarkConfig, Setting, run_benchmark
from benchmark.reports.tables import efficiency_table, quality_table


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ===========================================================================
# BƯỚC 1 — DATASET: nguồn dữ liệu chung mà mọi thuật toán cùng chạy
# ===========================================================================
def step1_dataset():
    hr("BƯỚC 1 — Tạo Dataset (nguồn stream dùng chung)")
    print(
        "Dataset là nguồn phát (x, y) lặp lại được: mỗi thuật toán đều tiêu thụ\n"
        "CÙNG một chuỗi mẫu từ đầu, không xáo trộn. y là nhãn thật, chỉ dùng để\n"
        "chấm chất lượng — model chỉ thấy x.\n"
    )
    cfg = StreamConfig(n_samples=8000, n_features=2, n_clusters=5, cluster_std=1.0, seed=42)
    dataset = SyntheticStream(cfg)
    print(f"Dataset: {dataset.name}  (n_features={dataset.n_features}, có nhãn={dataset.has_labels})")

    print("\n5 mẫu đầu tiên (x = đặc trưng đưa cho model, y = nhãn cluster thật):")
    for i, (x, y) in enumerate(dataset.take(5)):
        print(f"   {i}: x={ {k: round(v,2) for k,v in x.items()} }  y={y}")
    print(
        "\n=> Đây là dataset synthetic có nhãn. Bạn cũng có thể dùng dataset thật:\n"
        "   from benchmark.datasets import load_kddcup99; ds = load_kddcup99(n_samples=50000)\n"
    )
    return dataset


# ===========================================================================
# BƯỚC 2 — MỘT LẦN CHẠY: giao thức online scale -> predict -> learn
# ===========================================================================
def step2_single_run(dataset):
    hr("BƯỚC 2 — Chạy MỘT thuật toán và đo tài nguyên")
    print(
        "Với mỗi mẫu: scale(x) -> predict_one(x) -> learn_one(x) (test-then-train).\n"
        "Runner đo runtime tích lũy, RAM, số micro-cluster tại các checkpoint.\n"
    )
    adapter = get_algorithm("DBSTREAM", scaler="standard")
    print(f"Thuật toán: {adapter.name} | scaler: {adapter.scaler_name}")
    print(f"Tham số: {adapter.params}\n")

    res = run_once(adapter, dataset, n_checkpoints=8)

    print("Diễn tiến tài nguyên theo số mẫu đã xử lý (checkpoints):")
    print(f"   {'n_seen':>7} | {'runtime(s)':>10} | {'RAM(MB)':>8} | {'#micro':>6}")
    for c in res.checkpoints:
        mem = f"{c.memory_bytes/1048576:.3f}" if c.memory_bytes else "-"
        mc = c.n_micro_clusters if c.n_micro_clusters is not None else "-"
        print(f"   {c.n_seen:>7} | {c.cumulative_runtime_s:>10.4f} | {mem:>8} | {str(mc):>6}")

    print("\nKết quả tổng hợp:")
    print("   efficiency:", res.efficiency.to_dict())
    print("   quality:   ", res.quality)
    print(
        "\n=> Bạn thấy runtime tăng dần theo số mẫu, RAM/số micro-cluster ổn định\n"
        "   hay phình to. Đây chính là thứ framework dùng để so sánh scalability.\n"
    )


# ===========================================================================
# BƯỚC 3 — SO SÁNH NHIỀU THUẬT TOÁN trên CÙNG dataset (§ mục tiêu chính)
# ===========================================================================
def step3_compare_algorithms():
    hr("BƯỚC 3 — So sánh NHIỀU thuật toán trên CÙNG dataset")
    print(
        "Định nghĩa nhiều Setting (thuật toán + params + scaler) rồi chạy tất cả\n"
        "trên cùng dataset. Mọi setting dùng chung 1 stream => so sánh công bằng.\n"
    )
    dataset = SyntheticStream(StreamConfig(n_samples=8000, n_clusters=5, seed=42))
    config = BenchmarkConfig(
        settings=[
            Setting("DBSTREAM", scaler="standard"),
            Setting("STREAMKMeans", params={"n_clusters": 5}, scaler="standard"),
            Setting("CluStream", params={"n_macro_clusters": 5}, scaler="standard"),
            Setting("KMeans", params={"n_clusters": 5}, scaler="standard"),
        ],
        seeds=(1,),
        n_checkpoints=10,
    )
    results = run_benchmark(dataset, config)

    print("## Hiệu năng (thời gian / RAM / số cluster)\n")
    print(efficiency_table(results))
    print("\n## Chất lượng (dataset có nhãn nên tính được)\n")
    print(quality_table(results))
    print(
        "\n=> Đây là câu trả lời chính: thuật toán nào nhanh nhất / tốn RAM ít nhất /\n"
        "   tạo bao nhiêu cluster / chất lượng ra sao — tất cả trên cùng dữ liệu.\n"
    )
    return results


# ===========================================================================
# BƯỚC 4 — NHIỀU SETTING của cùng 1 thuật toán (ví dụ: so sánh scaler)
# ===========================================================================
def step4_compare_settings():
    hr("BƯỚC 4 — So sánh nhiều SETTING của cùng thuật toán (vd: scaler)")
    print(
        "README nhấn mạnh: scaler ảnh hưởng rất lớn tới số cluster & chất lượng.\n"
        "Ở đây chạy DBSTREAM với 3 scaler khác nhau trên cùng dataset.\n"
    )
    dataset = SyntheticStream(StreamConfig(n_samples=5000, n_clusters=5, seed=42))
    config = BenchmarkConfig(
        settings=[
            Setting("DBSTREAM", scaler="none", label="DBSTREAM [none]"),
            Setting("DBSTREAM", scaler="standard", label="DBSTREAM [standard]"),
            Setting("DBSTREAM", scaler="minmax", label="DBSTREAM [minmax]"),
        ],
        n_checkpoints=6,
    )
    results = run_benchmark(dataset, config)
    print("## Hiệu năng\n")
    print(efficiency_table(results))
    print("\n## Chất lượng\n")
    print(quality_table(results))
    print(
        "\n=> Thường thấy: 'standard' cho chất lượng tốt nhất; 'minmax' có thể ép\n"
        "   mọi điểm về gần nhau -> gộp thành 1-2 cluster -> ARI tụt hẳn.\n"
    )
    return results


# ===========================================================================
# BƯỚC 5 — VẼ BIỂU ĐỒ (nếu có matplotlib)
# ===========================================================================
def step5_plots(results):
    hr("BƯỚC 5 — Xuất biểu đồ trực quan")
    try:
        from benchmark.reports import (
            plot_runtime_vs_samples,
            plot_memory_vs_samples,
            plot_clusters_vs_samples,
            plot_quality_bars,
        )
    except Exception as e:
        print(f"(Bỏ qua vẽ biểu đồ — không import được matplotlib: {e})")
        return

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walkthrough_out")
    os.makedirs(out_dir, exist_ok=True)
    plot_runtime_vs_samples(results).savefig(os.path.join(out_dir, "01_runtime.png"), dpi=90, bbox_inches="tight")
    plot_memory_vs_samples(results).savefig(os.path.join(out_dir, "02_memory.png"), dpi=90, bbox_inches="tight")
    plot_clusters_vs_samples(results).savefig(os.path.join(out_dir, "03_clusters.png"), dpi=90, bbox_inches="tight")
    plot_quality_bars(results).savefig(os.path.join(out_dir, "04_quality.png"), dpi=90, bbox_inches="tight")

    print("Đã lưu 4 biểu đồ vào:")
    print(f"   {out_dir}")
    print(
        "   01_runtime.png   : runtime tích lũy theo số mẫu (scalability thời gian)\n"
        "   02_memory.png    : RAM model theo số mẫu\n"
        "   03_clusters.png  : số micro-cluster theo số mẫu\n"
        "   04_quality.png   : ARI cuối cùng của từng thuật toán\n"
    )


def main():
    print(
        "\nWALKTHROUGH — FRAMEWORK BENCHMARK CLUSTERING (River)\n"
        "So sánh nhiều thuật toán / nhiều setting trên cùng dataset về hiệu năng."
    )
    print(f"\nThuật toán có sẵn: {list(ALGORITHMS)}")
    print(f"Scaler có sẵn:     {sorted(SCALERS)}")

    dataset = step1_dataset()
    step2_single_run(dataset)
    step3_compare_algorithms()
    results = step4_compare_settings()
    step5_plots(results)

    hr("XONG")
    print(
        "Tóm tắt luồng: Dataset -> nhiều Setting (algo+params+scaler)\n"
        "               -> Runner (scale/predict/learn, đo tài nguyên) -> Báo cáo.\n\n"
        "Thử trên terminal:\n"
        "   python -m benchmark synthetic --n-samples 20000 --n-clusters 5\n"
        "   python -m benchmark synthetic --algorithms DBSTREAM --scalers none,standard,minmax\n"
        "   python -m benchmark kdd99 --n-samples 50000 --scalers standard\n"
        "   python -m benchmark synthetic --seeds 1,2,3 --out results/\n"
    )


if __name__ == "__main__":
    main()
