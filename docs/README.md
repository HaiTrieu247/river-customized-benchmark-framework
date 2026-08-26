# Báo cáo kỹ thuật

`benchmark_framework_report.tex` — mô tả đầy đủ framework benchmark: giao thức
đánh giá, công thức của mọi độ đo, mã giả, và kết quả thực nghiệm.

## Biên dịch

File tự chứa (không cần ảnh ngoài — mọi đồ thị vẽ bằng `pgfplots`/`tikz` với dữ
liệu nhúng sẵn). Chạy **hai lần** để mục lục và tham chiếu chéo hiện đúng:

```bash
pdflatex benchmark_framework_report.tex
pdflatex benchmark_framework_report.tex
```

`xelatex` và `lualatex` cũng chạy được — phần preamble tự phát hiện engine và
chọn cách xử lý tiếng Việt tương ứng (`T5` + `inputenc` cho pdfLaTeX,
`fontspec` cho engine Unicode).

Không có TeX trên máy thì tải file lên [Overleaf](https://overleaf.com) rồi
chọn compiler nào cũng được; hoặc cài
[MiKTeX](https://miktex.org/download) (Windows) / TeX Live.

## Gói cần có

Đều nằm trong bản cài đầy đủ của TeX Live và MiKTeX:

```
iftex, inputenc, fontenc, lmodern, fontspec, geometry,
amsmath, amssymb, amsthm, booktabs, multirow, array,
graphicx, xcolor, tikz, pgfplots, algorithm, algpseudocode,
listings, caption, hyperref
```

MiKTeX sẽ tự hỏi để cài gói thiếu ở lần biên dịch đầu.

## Dữ liệu trong báo cáo

Mọi con số đến từ các lần chạy thật trên máy ghi ở Bảng 6 của báo cáo. Tái lập:

```bash
python run_benchmark_standard.py --dataset kdd99 --n-samples 50000 --seeds 1,2,3
python run_benchmark_standard.py --dataset kdd99 --n-samples 50000 --seeds 1 \
       --compare-scalers DBSTREAM
python run_benchmark_standard.py --dataset synthetic --n-samples 20000 --seeds 1,2
```

Kết quả thô nằm trong `benchmark_out/<dataset>/results.json`.
