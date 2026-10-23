"""Does ``robustness`` reproduce the numbers the paper drafts cite from the result script?

``result/otb_full/report/make_robustness_extra.py`` printed its checks to
``result/otb_full/report/robustness_extra.txt``; the drafts cite that file. The checks now live in
:mod:`conf_pipeline.robustness` as tested functions. This module recomputes them on an out dir,
drawing from one generator in the script's order (seed 0, 2000 replicates, as the script did),
reads the printed file, and compares every number at the precision it was printed with: a number
printed with d decimals matches when the recomputed one is within half a unit of its last place.
Nothing is adjusted to match; a mismatch is listed with both readings.

Sections of the script that were not ported (silhouette on single-class windows, -1 sentinels,
the tie-break replay, recovery after drift) are listed as not checked.

Run from the repo root (a few minutes: it reads the window and member logs)::

    python -m conf_pipeline.robustness_check --out-dir conf_pipeline/out/full \\
        --published result/otb_full/report/robustness_extra.txt
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import _common as C
from . import report
from . import robustness as R

PUBLISHED = C.REPO_ROOT / "result" / "otb_full" / "report" / "robustness_extra.txt"
#: What the script drew: one generator, seed 0, 2000 replicates per interval.
SCRIPT_SEED, SCRIPT_BOOT = 0, 2000
#: The script's section titles, in its order, and whether this module checks them.
SECTIONS: dict[str, bool] = {
    "tau giữa các chế độ: trung bình qua luồng, khoảng bootstrap 95 %": True,
    "tau khi gộp 7 luồng INSECTS thành một khối (12 khối)": True,
    "tau(default, ·) khi bỏ default hỏng hoặc gần 0": True,
    "thứ hạng theo từng điểm Sobol (cấu hình ngẫu nhiên đã nhân s, lượt quét seed 1)": True,
    "bỏ từng luồng thật (online, nhóm thật)": True,
    "từng luồng thật: phần của cải thiện khả dĩ và hai khoản mất": True,
    "bootstrap phần do bước chọn trên luồng thật (rút lại luồng)": True,
    "thuật toán tất định trong các cặp tunable": True,
    "STREAMKMeans online trên luồng thật: hai nguồn": True,
    "silhouette trên cửa sổ một lớp (online, mọi cửa sổ)": False,
    "nhãn -1 (sentinel) của incumbent trên cửa sổ đuôi": False,
    "phá hoà: quy tắc runtime đã chạy và quy tắc tất định, chỉ phần chọn": False,
    "hồi phục: trong 0,05 của mức best static từng đoạn, 3 cửa sổ liền (seed trung vị)": False,
    "hồi phục: trong 0,05 của ARI trước drift của chính chế độ đó": False,
    "ARI trung bình của đoạn trừ mức best static của đoạn": False,
}
#: The script's printed labels -> this module's names.
_FILTERS = {"bỏ default hỏng (undefined > 0,5)": R.WEAK_DEFAULT_FILTERS[0][0],
            "default ARI >= 0,01": R.WEAK_DEFAULT_FILTERS[1][0], "default ARI >= 0,05": R.WEAK_DEFAULT_FILTERS[2][0]}
_LOO = {"không bỏ": "none", "cả 7 INSECTS": R.LOO_EXTRA_DROPS[0][0]}
_STREAM_COLS = ["numerator", "denominator", "denominator_share", "n_tunable", "a_lag", "a_star_lag", "A_static",
                "selection", "search", "selection_share"]
_LOO_COLS = ["capture_confstream", "capture_frozen_prefix", "selection_loss", "search_loss", "selection_share", "gap"]
_DESCRIBE = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
_NUM = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


def sections(text: str) -> dict[str, list[str]]:
    """The printed file split at its ``======== title ========`` lines."""
    out, title = {}, None
    for line in text.splitlines():
        m = re.match(r"^=+ (.*) =+$", line.strip())
        if m:
            title = m.group(1)
            out[title] = []
        elif title is not None and line.strip():
            out[title].append(line)
    return out


def matches(printed: str, value) -> bool:
    """Whether ``value`` reads ``printed`` at the precision it was printed with."""
    if printed in ("True", "False"):
        return str(bool(value)) == printed
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return printed.lower() == "nan"
    if "e" in printed.lower():  # a p printed with one significant digit
        return f"{float(value):.1g}" == f"{float(printed):.1g}"
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return abs(float(value) - float(printed)) <= 0.5 * 10 ** -decimals + 1e-9


def recompute(out_dir: Path) -> dict:
    """The ported checks on ``out_dir``, in the script's order of generator draws."""
    inputs = R.load_inputs(out_dir)
    cells = R.cells_table(inputs)
    loss = R.loss_cells(inputs)
    summary = R._protocol(report.summary_table(out_dir))
    wide = R.arm_wides(cells)
    d = summary[summary["arm"] == "default"]
    undefined = d.pivot_table(index="fold", columns="algo", values="lag_undefined_share", aggfunc="mean")
    undefined = undefined.reindex(columns=wide["default"].columns)
    rng = np.random.default_rng(SCRIPT_SEED)
    tau = R.arm_tau_table(wide, rng, SCRIPT_BOOT)
    blk_tau, blk_ranks = R.block_rankings(wide)
    weak = R.weak_default_tau(wide, undefined, rng, SCRIPT_BOOT)
    sobol = R.sobol_rankings(report.load_sweep(out_dir), wide, rng, SCRIPT_BOOT)
    share = R.selection_share_interval(loss, rng, SCRIPT_BOOT)
    loo = R.leave_one_out(cells, loss, "fold", "real", R.LOO_EXTRA_DROPS)
    streams = R.per_stream_split(cells, loss)
    real = np.array([C.stream_group(f) == "real" for f in cells["fold"]])
    real_loss = loss[[C.stream_group(f) == "real" for f in loss["fold"]]]
    km = {"summary": float(cells.loc[real & (cells["algo"] == "STREAMKMeans"), "confstream"].mean()),
          "regret": float(real_loss.loc[real_loss["algo"] == "STREAMKMeans", "a_lag"].mean())}
    return {"tau": tau, "block_tau": blk_tau, "block_ranks": blk_ranks, "weak": weak, "sobol": sobol,
            "share": share, "loo": loo, "streams": streams, "seedless": R.seedless_pairs(inputs.tunable, summary),
            "streamkmeans": km}


def compare(text: str, got: dict) -> pd.DataFrame:
    """One row per printed number of a checked section: ``section, item, printed, computed,
    match``; and one row per section not checked, with ``match`` None."""
    secs = sections(text)
    rows = []

    def add(section, item, printed, value):
        rows.append({"section": section, "item": item, "printed": printed,
                     "computed": value if not isinstance(value, (np.floating, np.integer)) else value.item(),
                     "match": matches(printed, value)})

    titles = list(SECTIONS)
    tau = got["tau"].set_index(["arm_a", "arm_b", "scope"])
    for line in secs.get(titles[0], []):
        m = re.match(rf"^\s*(\w+) vs (\w+)\s+(\d+) luồng ({_NUM}) \[({_NUM}); ({_NUM})\] \| thật ({_NUM}) "
                     rf"\[({_NUM}); ({_NUM})\]", line)
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        add(titles[0], f"{a} vs {b}: n folds", m.group(3), tau.loc[(a, b, "all"), "n_folds"])
        for k, (scope, col) in enumerate([("all", "tau"), ("all", "low"), ("all", "high"),
                                          ("real", "tau"), ("real", "low"), ("real", "high")]):
            add(titles[0], f"{a} vs {b}: {scope} {col}", m.group(4 + k), tau.loc[(a, b, scope), col])
    bt = got["block_tau"].set_index(["arm_a", "arm_b"])
    br = got["block_ranks"].set_index(["arm", "scope"])
    current = None
    for line in secs.get(titles[1], []):
        m = re.match(rf"^\s*(\w+) vs (\w+)\s+({_NUM})\s*$", line)
        if m:
            add(titles[1], f"{m.group(1)} vs {m.group(2)}: block tau", m.group(3), bt.loc[(m.group(1), m.group(2)), "tau"])
            continue
        m = re.match(rf"^hạng trung bình (\w+) \((\d+) khối, Friedman p = ({_NUM})\): (\{{.*\}})", line)
        if m:
            current = m.group(1)
            add(titles[1], f"{current}: n blocks", m.group(2), br.loc[(current, "all"), "n_blocks"])
            add(titles[1], f"{current}: Friedman p", m.group(3), br.loc[(current, "all"), "friedman_p"])
            for algo, v in ast.literal_eval(m.group(4)).items():
                add(titles[1], f"{current}: rank {algo}", repr(v), br.loc[(current, "all"), f"rank_{algo}"])
            continue
        m = re.match(r"^\s*chỉ luồng thật \((\d+) khối\): (\{.*\})", line)
        if m and current:
            add(titles[1], f"{current} real: n blocks", m.group(1), br.loc[(current, "real"), "n_blocks"])
            for algo, v in ast.literal_eval(m.group(2)).items():
                add(titles[1], f"{current} real: rank {algo}", repr(v), br.loc[(current, "real"), f"rank_{algo}"])
    weak = got["weak"].set_index(["filter", "arm"])
    for line in secs.get(titles[2], []):
        m = re.match(rf"^(.*?)\s+vs (\w+)\s+({_NUM}) \[({_NUM}); ({_NUM})\] luồng=(\d+) thuật toán/luồng=({_NUM})", line)
        if not m:
            continue
        key = (_FILTERS[m.group(1).strip()], m.group(2))
        for k, col in enumerate(["tau", "low", "high", "n_folds", "algos_per_fold"]):
            add(titles[2], f"{key[0]} vs {key[1]}: {col}", m.group(3 + k), weak.loc[key, col])
    sob = got["sobol"]
    desc = sob["describe"]
    agg = sob["aggregate"].set_index(["aggregate", "arm"])
    for line in secs.get(titles[3], []):
        m = re.match(rf"^kiểm: tau\(điểm 0 của lượt quét, default\) = ({_NUM})", line)
        if m:
            add(titles[3], "point 0 vs default", m.group(1), sob["point_zero_tau"])
            continue
        parts = line.split()
        if parts and parts[0] in _DESCRIBE and len(parts) == 4:
            for col, tok in zip(["default", "confstream", "oracle_static"], parts[1:]):
                add(titles[3], f"describe {parts[0]} {col}", tok, desc.loc[parts[0], col])
            continue
        m = re.match(rf"^(trung bình|trung vị) 32 điểm\s+vs (\w+)\s+({_NUM}) \[({_NUM}); ({_NUM})\]", line)
        if m:
            key = ("mean" if m.group(1) == "trung bình" else "median", m.group(2))
            for k, col in enumerate(["tau", "low", "high"]):
                add(titles[3], f"{key[0]} of the points vs {key[1]}: {col}", m.group(3 + k), agg.loc[key, col])
    loo = got["loo"].set_index("dropped")
    for line in secs.get(titles[4], [])[1:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        label = " ".join(parts[:-6])
        key = _LOO.get(label, label)
        for col, tok in zip(_LOO_COLS, parts[-6:]):
            add(titles[4], f"{key}: {col}", tok, loo.loc[key, col] if key in loo.index else np.nan)
    streams = got["streams"].set_index("fold")
    for line in secs.get(titles[5], []):
        parts = line.split()
        if len(parts) == 11 and parts[0] in streams.index:
            for col, tok in zip(_STREAM_COLS, parts[1:]):
                add(titles[5], f"{parts[0]}: {col}", tok, streams.loc[parts[0], col])
    sh = got["share"]
    for line in secs.get(titles[6], []):
        m = re.match(rf"^phần chọn ({_NUM}) khoảng 95 % \[({_NUM}) ({_NUM})\] \| P\(mất do tìm <= 0\) = ({_NUM})", line)
        if m:
            for k, col in enumerate(["share", "low", "high", "p_search_le_0"]):
                add(titles[6], col, m.group(1 + k), sh[col])
    sl = got["seedless"]
    for line in secs.get(titles[7], []):
        m = re.match(r"^cặp tunable: (\d+) \| thuộc thuật toán tất định: (\d+)", line)
        if m:
            add(titles[7], "tunable pairs", m.group(1), sl["n_tunable"])
            add(titles[7], "tunable pairs of seedless algorithms", m.group(2), sl["n_tunable_seedless"])
        m = re.match(r"^sd qua seed 2-4 .*: (True|False)", line)
        if m:
            add(titles[7], "reference sd 0 on every seedless pair", m.group(1), sl["reference_sd_zero_on_seedless"])
    for line in secs.get(titles[8], []):
        m = re.match(rf"^summary .*: ({_NUM}) \| regret .*: ({_NUM})", line)
        if m:
            add(titles[8], "STREAMKMeans online, real, summary", m.group(1), got["streamkmeans"]["summary"])
            add(titles[8], "STREAMKMeans online, real, regret windows", m.group(2), got["streamkmeans"]["regret"])
    for title, checked in SECTIONS.items():
        if not checked and title in secs:
            rows.append({"section": title, "item": "(not ported)", "printed": None, "computed": None, "match": None})
    return pd.DataFrame(rows, columns=["section", "item", "printed", "computed", "match"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(C.OUT / "full"))
    parser.add_argument("--published", default=str(PUBLISHED))
    parser.add_argument("--csv", default=None, help="write the comparison here (default: "
                        "<out-dir>/report/robustness/robustness_extra_check.csv)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    table = compare(Path(args.published).read_text(encoding="utf-8"), recompute(out_dir))
    path = Path(args.csv) if args.csv else out_dir / C.REPORT_DIR.name / "robustness" / "robustness_extra_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    checked = table[table["match"].notna()]
    bad = checked[~checked["match"].astype(bool)]
    print(f"{len(checked)} printed numbers checked, {len(checked) - len(bad)} match, {len(bad)} do not; "
          f"{int(table['match'].isna().sum())} sections not ported. Written to {path}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        if len(bad):
            print(bad.to_string(index=False))
    raise SystemExit(1 if len(bad) else 0)


if __name__ == "__main__":
    main()
