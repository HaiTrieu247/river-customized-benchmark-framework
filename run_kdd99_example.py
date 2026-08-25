"""Ví dụ: so sánh 4 thuật toán clustering trên KDDCup99 (standard scaler).

Chạy 4 thuật toán — CluStream, DenStream, DBSTREAM, STREAMKMeans — trên CÙNG một
stream KDDCup99 (400.000 samples, 34 đặc trưng liên tục), cùng scaler `standard`,
rồi in bảng hiệu năng + chất lượng và (nếu có matplotlib) lưu biểu đồ.

Chạy:

    python run_kdd99_example.py

KDDCup99 được tự tải qua sklearn ở lần chạy đầu (có thể mất một lúc để tải).
"""

from __future__ import annotations

import os
import sys
import time

# Console Windows mặc định cp1252 -> không in được tiếng Việt. Ép UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from benchmark.datasets import load_kddcup99
from benchmark.experiments import BenchmarkConfig, Setting, run_setting
from benchmark.reports.tables import (
    efficiency_table,
    quality_table,
    stream_quality_table,
)

N_SAMPLES = 250_000
SCALER = "standard"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kdd99_example_out")


def main():
    print("=" * 72)
    print(f"So sánh 4 thuật toán trên KDDCup99 | scaler={SCALER} | n_samples={N_SAMPLES:,}")
    print("=" * 72)

    # 1) Dataset dùng chung — KDDCup99 10% (~494k mẫu), cắt còn N_SAMPLES.
    print("\n[1/3] Nạp dataset KDDCup99 (tự tải qua sklearn ở lần đầu)...")
    dataset = load_kddcup99(n_samples=N_SAMPLES, percent10=True)
    print(f"      dataset = {dataset.name} | n_features = {dataset.n_features} | có nhãn = {dataset.has_labels}")

    # 2) 4 setting: cùng scaler 'standard'. Tham số cluster count đặt ~ số lớp lớn.
    #    (KDDCup99 có nhiều lớp nhưng bị chi phối bởi 'normal.' và vài loại tấn công.)
    settings = [
        Setting("CluStream", params={"n_macro_clusters": 5, "max_micro_clusters": 100}, scaler=SCALER),
        Setting("DenStream", params={"epsilon": 0.5, "mu": 5.0, "beta": 0.5, "n_samples_init": 1000}, scaler=SCALER),
        Setting("DBSTREAM", params={"clustering_threshold": 1.0, "fading_factor": 0.05}, scaler=SCALER),
        Setting("STREAMKMeans", params={"n_clusters": 5, "chunk_size": 50}, scaler=SCALER),
    ]

    config = BenchmarkConfig(
        settings=settings,
        seeds=(1,),
        n_samples=N_SAMPLES,
        n_checkpoints=25,   # 25 điểm đo tài nguyên dọc stream
        window_size=1000,   # cửa sổ đánh giá chất lượng theo thời gian
    )

    # 3) Chạy từng thuật toán, in thời gian ngay khi mỗi cái xong.
    #    Mọi setting dùng chung đúng một stream (cùng dataset, cùng seed).
    print(f"\n[2/3] Chạy {len(settings)} thuật toán (mỗi cái stream {N_SAMPLES:,} mẫu)...")
    print("      (DenStream chậm hơn hẳn — bình thường với dữ liệu lớn.)\n")
    results = []
    for i, setting in enumerate(settings, 1):
        print(f"      [{i}/{len(settings)}] {setting.display()} ...", end="", flush=True)
        t0 = time.perf_counter()
        res = run_setting(setting, dataset, config, seed=config.seeds[0])
        dt = time.perf_counter() - t0
        ari = res.quality.get("ari")
        ari_s = f"{ari:.4f}" if ari is not None else "-"
        print(f" xong {dt:6.1f}s | {res.efficiency.throughput_sps:8.0f} sps | ARI={ari_s}", flush=True)
        results.append(res)

    print("\n[3/3] Kết quả\n")
    print("## Hiệu năng (thời gian / throughput / RAM / số cluster)\n")
    print(efficiency_table(results))
    print("\n## Chất lượng — toàn stream (ARI/NMI/AMI/purity)\n")
    print(quality_table(results))
    print("\n## Chất lượng — ARI theo thời gian (mean/median/final trên các window)\n")
    print(stream_quality_table(results, metric="ari"))

    # Biểu đồ (tùy chọn)
    try:
        from benchmark.reports import (
            plot_clusters_vs_samples,
            plot_memory_vs_samples,
            plot_quality_bars,
            plot_runtime_vs_samples,
        )
    except Exception as e:
        print(f"\n(Bỏ qua biểu đồ — matplotlib không khả dụng: {e})")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_runtime_vs_samples(results).savefig(os.path.join(OUT_DIR, "runtime_vs_samples.png"), dpi=90, bbox_inches="tight")
    plot_memory_vs_samples(results).savefig(os.path.join(OUT_DIR, "memory_vs_samples.png"), dpi=90, bbox_inches="tight")
    plot_clusters_vs_samples(results).savefig(os.path.join(OUT_DIR, "clusters_vs_samples.png"), dpi=90, bbox_inches="tight")
    plot_quality_bars(results, metric="ari").savefig(os.path.join(OUT_DIR, "ari_bars.png"), dpi=90, bbox_inches="tight")
    print(f"\nĐã lưu 4 biểu đồ vào: {OUT_DIR}")


if __name__ == "__main__":
    main()
