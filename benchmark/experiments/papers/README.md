# Thư Mục Thí Nghiệm Tái Hiện Bài Báo Khoa Học (Paper Reproductions)

Thư mục này chứa các kịch bản thực nghiệm tái hiện chính xác các bài báo khoa học nổi tiếng trong lĩnh vực streaming clustering:

## 1. D-Stream vs CluStream (`dstream_vs_clustream.py`)
- **Bài báo gốc**: *"Density-Based Clustering for Real-Time Stream Data"* (Yixin Chen & Li Tu, ACM SIGKDD 2007).
- **So sánh với**: CluStream (Aggarwal et al., VLDB 2003).

### Danh mục các Figures được tái hiện:
- **Figures 5 & 6**: Dữ liệu 30K gồm 4 cụm phi lồi (Non-Convex) + 5K điểm nhiễu ngoại lai. D-Stream gom cụm chính xác với $len = 0.05$ và loại bỏ ngoại lai.
- **Figures 7, 8, 9, 10**: Luồng động 85K gồm 4 lớp xuất hiện tuần tự + 10K điểm nhiễu tại các mốc thời gian $t_1 = 25, t_2 = 55, t_3 = 85$. Minh họa cơ chế xóa lưới rác lẻ tẻ (*sporadic grid removal*).
- **Figure 11**: Tỷ lệ gom cụm chính xác (Correct Rate) trên dữ liệu nhân tạo tại $t \in [25, 45, 65, 85]$ với $len \in [0.02, 0.04, 0.05]$.
- **Figure 12**: Tỷ lệ gom cụm chính xác trên KDD CUP-99 tại $t \in [100, 150, 200, 250, 300]$ với $len \in [0.025, 0.05]$.
- **Figure 13**: Độ chặt chẽ của cụm (Mean SSQ) trên KDD CUP-99 theo thang đo Logarit ($1.00E+00 \rightarrow 1.00E+14$).
- **Figure 14**: So sánh thời gian xử lý khi kích thước dữ liệu tăng từ 5K đến 30K trên KDD CUP-99.
- **Figure 15**: So sánh thời gian xử lý khi số chiều tăng từ 2 đến 40 chiều trên KDD CUP-99.

---

### Hướng dẫn chạy:
Môi trường Python: `F:\DACN\river\.venv\Scripts\python.exe`.

Mở PowerShell tại thư mục gốc `F:\DACN\river-customized-benchmark-framework`:

```powershell
# Chạy chế độ demo nhanh (khoảng 1 phút):
& "F:\DACN\river\.venv\Scripts\python.exe" -m benchmark.experiments.papers.dstream_vs_clustream --quick

# Chạy toàn bộ dữ liệu đầy đủ:
& "F:\DACN\river\.venv\Scripts\python.exe" -m benchmark.experiments.papers.dstream_vs_clustream --all

# Chạy riêng từng bài kiểm tra cụ thể:
& "F:\DACN\river\.venv\Scripts\python.exe" -m benchmark.experiments.papers.dstream_vs_clustream --fig 5,6
& "F:\DACN\river\.venv\Scripts\python.exe" -m benchmark.experiments.papers.dstream_vs_clustream --fig 7-10
& "F:\DACN\river\.venv\Scripts\python.exe" -m benchmark.experiments.papers.dstream_vs_clustream --fig 11,12
& "F:\DACN\river\.venv\Scripts\python.exe" -m benchmark.experiments.papers.dstream_vs_clustream --fig 13
& "F:\DACN\river\.venv\Scripts\python.exe" -m benchmark.experiments.papers.dstream_vs_clustream --fig 14,15
```

---

### Nơi xuất kết quả:
Toàn bộ đồ thị được lưu tự động tại:
📂 `benchmark_out/comparison_dstream_clustream/`
