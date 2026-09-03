# Hướng Dẫn Tích Hợp & Thêm Thí Nghiệm Cho Bài Báo Mới (Paper Benchmark)

Tài liệu này hướng dẫn chi tiết từng bước cách tạo và mở rộng một bài báo/thực nghiệm so sánh mới trong hệ thống `river-customized-benchmark-framework` dựa trên kiến trúc **Pluggable Paper Benchmark**.

---

## 1. Tổng Quan Kiến Trúc

Hệ thống quản lý thực nghiệm bài báo khoa học được tổ chức tại:
```
river-customized-benchmark-framework/
├── run_paper_benchmark.py                     ← CLI runner trung tâm duy nhất cho MỌI bài báo
└── benchmark/experiments/papers/
    ├── base.py                                ← Lớp cơ sở BasePaperBenchmark & Paper Registry
    ├── dstream_vs_clustream.py                ← Thí nghiệm D-Stream KDD '07
    ├── <tên_bài_báo_mới>.py                   ← Thí nghiệm cho bài báo mới của bạn
    ├── __init__.py                            ← Export tự động đăng ký
    └── HOW_TO_ADD_PAPER_BENCHMARK.md          ← Tài liệu hướng dẫn này
```

Bất kỳ bài báo mới nào chỉ cần **kế thừa `BasePaperBenchmark`** và đăng ký bằng decorator `@register_paper`. Hệ thống sẽ tự động phát hiện và đưa vào giao diện dòng lệnh `run_paper_benchmark.py`.

---

## 2. Các Bước Thêm Thí Nghiệm Mới

### Bước 1: Tạo file mã nguồn mới
Tạo một file Python trong thư mục `benchmark/experiments/papers/`, ví dụ:
`benchmark/experiments/papers/denstream_sdm06.py`

### Bước 2: Kế thừa `BasePaperBenchmark` và điền metadata
Khai báo class kế thừa và cung cấp thông tin bài báo:
- `name`: Mã định danh viết thường, không dấu (ví dụ: `denstream_2006`, `dbstream_2016`).
- `title`: Tiêu đề bài báo khoa học.
- `authors`: Tác giả.
- `year`: Năm xuất bản.
- `venue`: Hội nghị/Tạp chí (ví dụ: `SIAM SDM`, `ACM KDD`, `IEEE TKDE`).
- `description`: Tóm tắt nội dung so sánh của bài báo.

### Bước 3: Đăng ký các Figure / Thí nghiệm trong `setup()`
Trong phương thức `setup()`, gọi `self.register_figure(...)` cho từng bài test:
```python
self.register_figure(
    figure_id="1",                          # Mã hình trong bài báo (chuỗi, ví dụ "1", "5,6", "7-10")
    title="Figure 1: Quality vs Horizon",   # Tên bài thí nghiệm
    description="So sánh Purity của DenStream vs CluStream theo kích thước horizon",
    runner=self.run_figure_1,               # Hàm thực thi thí nghiệm
)
```

### Bước 4: Viết các hàm thực thi (`runner`)
Hàm thực thi nhận 2 tham số chuẩn:
- `quick: bool`: Khi `True`, giảm bớt số lượng mẫu hoặc số lần lặp để chạy thử nhanh trong 30-60 giây.
- `output_dir: Path | None`: Thư mục lưu đồ thị (nếu `None`, tự động lấy `benchmark_out/papers/<name>/`).

### Bước 5: Đăng ký vào `benchmark/experiments/papers/__init__.py`
Thêm 1 dòng import class của bạn vào `__init__.py`:
```python
from .denstream_sdm06 import DenStream2006Benchmark
```

---

## 3. Mã Nguồn Mẫu (Boilerplate Template)

Dưới đây là khung code hoàn chỉnh để bạn copy và điền nội dung cho bài báo mới:

```python
"""
Reproduction of: "<Tên Bài Báo>"
Tác giả: <Tên Tác Giả> (<Hội nghị / Năm>)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo import được benchmark và river từ bất kỳ đâu
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RIVER_ROOT = PROJECT_ROOT.parent / "river"
for p in (PROJECT_ROOT, RIVER_ROOT):
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

import matplotlib.pyplot as plt
import numpy as np
from river import cluster, preprocessing

# Import dataset và metric từ benchmark framework
from benchmark.datasets import load_kddcup99, make_scenario
from benchmark.evaluation.clustering_quality import purity, ari
from .base import BasePaperBenchmark, register_paper


@register_paper
class MyNewPaperBenchmark(BasePaperBenchmark):
    """Lớp thực nghiệm cho bài báo mới."""

    def __init__(self):
        super().__init__(
            name="ten_bai_bao_2024",                    # ID duy nhất dùng trong CLI (--paper ten_bai_bao_2024)
            title="Title of the Research Paper",
            authors="Author 1, Author 2",
            year=2024,
            venue="ACM SIGKDD",
            description="Mô tả ngắn về nội dung so sánh trong bài báo."
        )
        self.setup()

    def setup(self) -> None:
        """Đăng ký toàn bộ các figure / bài test của bài báo tại đây."""
        self.register_figure(
            figure_id="1",
            title="Figure 1: Quality vs Stream Length",
            description="Đánh giá Purity và ARI khi chiều dài luồng tăng.",
            runner=self.run_figure_1,
        )
        self.register_figure(
            figure_id="2",
            title="Figure 2: Execution Time Comparison",
            description="So sánh tốc độ xử lý giữa thuật toán đề xuất và CluStream.",
            runner=self.run_figure_2,
        )

    # -----------------------------------------------------------------------
    # Triển khai từng bài thí nghiệm
    # -----------------------------------------------------------------------
    def run_figure_1(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = output_dir or self.get_output_dir()
        n_samples = 5_000 if quick else 30_000
        
        # 1. Nạp dữ liệu
        ds = load_kddcup99(n_samples=n_samples, map_to_5_classes=True)
        
        # 2. Khởi tạo các mô hình cần so sánh
        model_a = cluster.CluStream(n_macro_clusters=5, seed=42)
        # model_b = cluster.<ThuatToanMoi>(...)
        
        # 3. Chạy luồng và đo lường
        # ...
        
        # 4. Vẽ đồ thị
        fig, ax = plt.subplots(figsize=(8, 5))
        # ax.plot(...)
        ax.set_title("Figure 1: Quality vs Stream Length")
        ax.set_xlabel("Stream Size")
        ax.set_ylabel("Purity")
        
        plt.tight_layout()
        plot_path = out_dir / "fig1_quality_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path

    def run_figure_2(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = output_dir or self.get_output_dir()
        # Triển khai logic Figure 2...
        plot_path = out_dir / "fig2_time_comparison.png"
        return plot_path
```

---

## 4. Cách Chạy và Kiểm Tra Thí Nghiệm Mới

Sau khi lưu file và khai báo vào `benchmark/experiments/papers/__init__.py`:

### 1. Kiểm tra bài báo đã được đăng ký thành công chưa:
```powershell
& "F:\DACN\river\.venv\Scripts\python.exe" run_paper_benchmark.py --list
```
Bạn sẽ thấy bài báo mới xuất hiện trong danh sách kèm theo các Figure IDs của nó!

### 2. Chạy bài báo mới ở chế độ nhanh:
```powershell
& "F:\DACN\river\.venv\Scripts\python.exe" run_paper_benchmark.py --paper ten_bai_bao_2024 --quick
```

### 3. Chạy riêng từng Figure của bài báo mới:
```powershell
& "F:\DACN\river\.venv\Scripts\python.exe" run_paper_benchmark.py --paper ten_bai_bao_2024 --fig 1
```

---

## 5. Các Best Practices Khi Viết Thí Nghiệm
1. **Tuân thủ giao thức Stream (Test-then-Train)**: Luôn gọi `predict_one(x)` trước, sau đó mới gọi `learn_one(x)`.
2. **Chuẩn hóa dữ liệu tách biệt**:
   ```python
   scaler.learn_one(x)
   xs = scaler.transform_one(x)
   ```
   Không gộp `scaler.learn_one(x).transform_one(x)` vì River cập nhật in-place và trả về `None`.
3. **Hỗ trợ cờ `--quick`**: Giúp người dùng hoặc CI/CD có thể kiểm chứng toàn bộ code trong vòng dưới 1 phút mà không cần đợi chạy hàng trăm nghìn mẫu.
4. **Lưu đồ thị vào `out_dir`**: Hàm thực thi nên trả về đường dẫn `Path` của file ảnh đã lưu để runner tự động in ra màn hình.
