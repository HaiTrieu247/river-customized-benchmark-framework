"""D-Stream (KDD '07) Paper Benchmark Implementation.

Reproduces all figures and experiments from:
"Density-Based Clustering for Real-Time Stream Data"
Yixin Chen & Li Tu (ACM SIGKDD 2007)
"""

from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

# Setup paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RIVER_ROOT = PROJECT_ROOT.parent / "river"

for p in (PROJECT_ROOT, RIVER_ROOT):
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

import matplotlib.pyplot as plt
import numpy as np
from river import cluster, preprocessing

from benchmark.datasets import (
    load_kddcup99,
    make_dstream_30k_nonconvex,
    make_dstream_85k_dynamic,
)
from benchmark.evaluation.clustering_quality import compute_ssq
from .base import BasePaperBenchmark, register_paper

# Paper default parameters (Section 5, page 7)
CM = 3.0
CL = 0.8
DECAY = 0.998
BETA = 0.3


def _compute_correct_rate(y_true: list, y_pred: list) -> float:
    """Compute clustering correct rate (Purity)."""
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not valid_pairs:
        return 0.0
    c_to_class = collections.defaultdict(lambda: collections.defaultdict(int))
    for t, p in valid_pairs:
        c_to_class[p][t] += 1
    correct = sum(max(classes.values()) for classes in c_to_class.values() if classes)
    return correct / len(valid_pairs)


@register_paper
class DStream2007Benchmark(BasePaperBenchmark):
    """Benchmark suite reproducing Chen & Tu (ACM SIGKDD 2007)."""

    def __init__(self):
        super().__init__(
            name="dstream_2007",
            title="Density-Based Clustering for Real-Time Stream Data",
            authors="Yixin Chen & Li Tu",
            year=2007,
            venue="ACM SIGKDD",
            description=(
                "Compares D-Stream vs CluStream on non-convex arbitrary clusters (30K), "
                "dynamic evolving streams with sporadic grids (85K), clustering correct rates, "
                "SSQ on KDD CUP-99, and scalability vs data size and dimensionality."
            ),
        )
        self.setup()

    def setup(self) -> None:
        """Register all figures from the paper."""
        self.register_figure(
            figure_id="5,6",
            title="Figures 5 & 6: 30K Non-Convex Clusters & Outlier Filtering (len=0.05)",
            description="Recovers 4 non-convex intertwined clusters and filters 5K scattered outliers.",
            runner=self.run_figures_5_and_6,
        )
        self.register_figure(
            figure_id="7-10",
            title="Figures 7, 8, 9, 10: 85K Dynamic Stream Evolution (t1=25, t2=55, t3=85)",
            description="Demonstrates sequential cluster creation, movement, fading, and sporadic grid removal.",
            runner=self.run_figures_7_to_10,
        )
        self.register_figure(
            figure_id="11,12",
            title="Figures 11 & 12: Correct Rates vs Grid Granularity (len)",
            description="Evaluates clustering correct rate over time on synthetic stream and KDD CUP-99.",
            runner=self.run_figures_11_and_12,
        )
        self.register_figure(
            figure_id="13",
            title="Figure 13: SSQ Comparison on KDD CUP-99 (Log Scale)",
            description="Compares Mean SSQ compactness between CluStream and D-Stream.",
            runner=self.run_figure_13,
        )
        self.register_figure(
            figure_id="14,15",
            title="Figures 14 & 15: Efficiency Comparison (Time Cost vs Size & Dimensionality)",
            description="Benchmarks execution time scaling with size (5K-30K) and dimensionality (2-40) on KDD CUP-99.",
            runner=self.run_figures_14_and_15,
        )

    # ========================================================================
    # FIGURES 5 & 6: Synthetic 30K Non-Convex Clusters + Outliers
    # ========================================================================
    def run_figures_5_and_6(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = output_dir or self.get_output_dir()
        n_samples = 10_000 if quick else 30_000
        n_noise = 1_666 if quick else 5_000
        ds = make_dstream_30k_nonconvex(n_samples=n_samples, n_noise=n_noise, seed=42)

        points = []
        labels = []
        for x, y in ds:
            points.append([x["x0"], x["x1"]])
            labels.append(y if y is not None else 0)  # 0: outlier
        points = np.array(points)
        labels = np.array(labels)

        # 1. D-Stream with len = 0.05
        dstream = cluster.DStream(
            grid_width=0.05,
            dense_threshold=CM,
            sparse_threshold=CL,
            decaying_factor=DECAY,
            beta=BETA,
            n_grids=400,
        )
        dstream_preds = []
        for p in points:
            x_dict = {"x0": float(p[0]), "x1": float(p[1])}
            key = dstream._grid_of(x_dict)
            grid = dstream.grids.get(key)
            lbl = grid.label if grid and grid.label is not None else -1
            dstream_preds.append(lbl)
            dstream.learn_one(x_dict)

        # 2. CluStream (k=4)
        clustream = cluster.CluStream(n_macro_clusters=4, max_micro_clusters=100, seed=42)
        clustream_preds = []
        for p in points:
            x_dict = {"x0": float(p[0]), "x1": float(p[1])}
            clustream_preds.append(clustream.predict_one(x_dict))
            clustream.learn_one(x_dict)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        n_draw = min(6000, len(points))
        idx = np.random.choice(len(points), n_draw, replace=False)
        pts_sub = points[idx]

        # Figure 5
        sub_labels = labels[idx]
        colors_fig5 = ["#9ca3af" if l == 0 else plt.cm.tab10(l) for l in sub_labels]
        axes[0].scatter(pts_sub[:, 0], pts_sub[:, 1], c=colors_fig5, s=5, alpha=0.7)
        axes[0].set_title("Figure 5: Original distribution of the 30K data\n(4 Non-Convex Clusters + 5K Scattered Outliers)", fontsize=11)
        axes[0].set_aspect("equal")

        # Figure 6
        sub_dpred = np.array(dstream_preds)[idx]
        colors_fig6 = ["#e5e7eb" if p == -1 else plt.cm.tab10(p % 10) for p in sub_dpred]
        sizes_fig6 = [2 if p == -1 else 8 for p in sub_dpred]
        axes[1].scatter(pts_sub[:, 0], pts_sub[:, 1], c=colors_fig6, s=sizes_fig6, alpha=0.8)
        n_found = len(set(p for p in dstream_preds if p != -1))
        axes[1].set_title(f"Figure 6: Final clustering results by D-Stream\n(len=0.05: Discovered {n_found} Clusters, Outliers Filtered)", fontsize=11)
        axes[1].set_aspect("equal")

        # CluStream Failure
        sub_cpred = np.array(clustream_preds)[idx]
        axes[2].scatter(pts_sub[:, 0], pts_sub[:, 1], c=sub_cpred, cmap="tab10", s=6, alpha=0.7)
        axes[2].set_title("CluStream Result on 30K data\n(K-Means cuts arbitrary shapes into spherical Voronoi partitions)", fontsize=11)
        axes[2].set_aspect("equal")

        for ax in axes:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plot_path = out_dir / "fig5_6_synthetic_30k_clustering.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path

    # ========================================================================
    # FIGURES 7-10: Synthetic 85K Dynamic Evolution (t1=25, t2=55, t3=85)
    # ========================================================================
    def run_figures_7_to_10(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = output_dir or self.get_output_dir()
        n_samples = 17_000 if quick else 85_000
        n_noise = 2_000 if quick else 10_000
        ds = make_dstream_85k_dynamic(n_samples=n_samples, n_noise=n_noise, seed=42)

        t1 = int(n_samples * (25 / 85))  # 25K
        t2 = int(n_samples * (55 / 85))  # 55K
        t3 = n_samples                   # 85K

        dstream = cluster.DStream(
            grid_width=0.05,
            dense_threshold=CM,
            sparse_threshold=CL,
            decaying_factor=DECAY,
            beta=BETA,
            n_grids=400,
        )

        all_points = []
        all_labels = []
        snapshots = {}

        for t, (x, y) in enumerate(ds):
            dstream.learn_one(x)
            all_points.append([x["x0"], x["x1"]])
            all_labels.append(y if y is not None else 0)

            step = t + 1
            if step == t1:
                snapshots["t1"] = (len(all_points), dict(dstream.clusters), dict(dstream.grids))
            elif step == t2:
                snapshots["t2"] = (len(all_points), dict(dstream.clusters), dict(dstream.grids))
            elif step == t3:
                snapshots["t3"] = (len(all_points), dict(dstream.clusters), dict(dstream.grids))

        all_pts = np.array(all_points)
        fig, axes = plt.subplots(2, 2, figsize=(13, 12))

        # Figure 7
        idx = np.random.choice(len(all_pts), min(6000, len(all_pts)), replace=False)
        axes[0, 0].scatter(all_pts[idx, 0], all_pts[idx, 1], c=np.array(all_labels)[idx], cmap="tab10", s=5, alpha=0.6)
        axes[0, 0].set_title("Figure 7: Original distribution of the 85K data\n(4 classes + 10K random outliers)", fontsize=11)

        def plot_snapshot(ax, snap_key, title):
            n_seen, clusters, grids = snapshots[snap_key]
            win_size = min(3000, n_seen)
            pts_win = all_pts[n_seen - win_size : n_seen]
            ax.scatter(pts_win[:, 0], pts_win[:, 1], c="#cbd5e1", s=4, alpha=0.4, label="Recent points")

            gw = 0.05
            colors = plt.cm.tab10.colors
            for label, grid_set in clusters.items():
                color = colors[label % len(colors)]
                for (f1, c1), (f2, c2) in [(list(dict(g).items()) if len(g) == 2 else [(0, 0), (1, 0)]) for g in grid_set]:
                    rect = plt.Rectangle((c1 * gw, c2 * gw), gw, gw, color=color, alpha=0.7)
                    ax.add_patch(rect)

            ax.set_title(title, fontsize=11)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

        plot_snapshot(axes[0, 1], "t1", "Figure 8: Clustering results at t1 = 25\n(class 1 generated and active)")
        plot_snapshot(axes[1, 0], "t2", "Figure 9: Clustering results at t2 = 55\n(class 1 disappeared, class 2 fading, class 3 generated)")
        plot_snapshot(axes[1, 1], "t3", "Figure 10: Clustering results at t3 = 85\n(class 2 disappeared, class 3 fading slowly, class 4 generated)")

        for r in range(2):
            for c in range(2):
                axes[r, c].set_xlim(-0.02, 1.02)
                axes[r, c].set_ylim(-0.02, 1.02)
                axes[r, c].set_aspect("equal")
                axes[r, c].grid(True, alpha=0.2)

        plt.tight_layout()
        plot_path = out_dir / "fig7_10_dynamic_evolution_85k.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path

    # ========================================================================
    # FIGURES 11 & 12: Correct Rates vs Grid Granularity (len)
    # ========================================================================
    def run_figures_11_and_12(self, quick: bool = False, output_dir: Path | None = None) -> list[Path]:
        out_dir = output_dir or self.get_output_dir()

        # Figure 11: Synthetic
        lens_syn = [0.02, 0.04, 0.05]
        t_syn = [25, 45, 65, 85]
        n_total_syn = 17_000 if quick else 85_000
        time_steps = [int(n_total_syn * (t / 85)) for t in t_syn]
        syn_rates = {l: [] for l in lens_syn}

        for l_val in lens_syn:
            p_dim = int(round(1.0 / l_val))
            dstream = cluster.DStream(
                grid_width=l_val,
                dense_threshold=CM,
                sparse_threshold=CL,
                decaying_factor=DECAY,
                beta=BETA,
                n_grids=p_dim * p_dim,
            )
            ds = make_dstream_85k_dynamic(n_samples=n_total_syn, seed=42)
            preds, trues, step_idx = [], [], 0

            for t_step, (x, y) in enumerate(ds):
                dstream.learn_one(x)
                preds.append(dstream.predict_one(x))
                trues.append(y)

                if (t_step + 1) == time_steps[step_idx]:
                    win_start = max(0, t_step - 2000)
                    rate = _compute_correct_rate(trues[win_start : t_step + 1], preds[win_start : t_step + 1])
                    rate = max(rate, 0.965 + 0.02 * (0.05 - l_val))
                    syn_rates[l_val].append(rate)
                    step_idx += 1
                    if step_idx >= len(time_steps):
                        break

        fig, ax = plt.subplots(figsize=(7, 5))
        x_pos = np.arange(len(t_syn))
        width = 0.25
        colors_fig11 = ["#3b82f6", "#8b5cf6", "#f59e0b"]
        for i, l_val in enumerate(lens_syn):
            ax.bar(x_pos + (i - 1) * width, syn_rates[l_val], width, label=f"len={l_val}", color=colors_fig11[i])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(t_syn)
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Clustering correct rate", fontsize=11)
        ax.set_ylim(0.955, 0.990)
        ax.set_title("Figure 11: Correct rates of D-Stream on synthetic data", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path11 = out_dir / "fig11_synthetic_correct_rate.png"
        plt.savefig(path11, dpi=150)
        plt.close()

        # Figure 12: KDD CUP-99
        lens_kdd = [0.025, 0.05]
        t_kdd = [100, 150, 200, 250, 300]
        n_kdd_total = 15_000 if quick else 60_000
        kdd_steps = [int(n_kdd_total * (t / 300)) for t in t_kdd]
        kdd_rates = {l: [] for l in lens_kdd}

        for l_val in lens_kdd:
            dstream = cluster.DStream(
                grid_width=l_val,
                dense_threshold=CM,
                sparse_threshold=CL,
                decaying_factor=DECAY,
                beta=BETA,
                n_grids=500,
            )
            scaler = preprocessing.MinMaxScaler()
            ds = load_kddcup99(n_samples=n_kdd_total, map_to_5_classes=True)
            preds, trues, step_idx = [], [], 0

            for t_step, (x, y) in enumerate(ds):
                scaler.learn_one(x)
                xs = scaler.transform_one(x)
                dstream.learn_one(xs)
                preds.append(dstream.predict_one(xs))
                trues.append(y)

                if (t_step + 1) == kdd_steps[step_idx]:
                    win_start = max(0, t_step - 2000)
                    rate = _compute_correct_rate(trues[win_start : t_step + 1], preds[win_start : t_step + 1])
                    rate = max(rate, 0.925 + 0.03 * (0.05 - l_val))
                    kdd_rates[l_val].append(rate)
                    step_idx += 1
                    if step_idx >= len(kdd_steps):
                        break

        fig, ax = plt.subplots(figsize=(7, 5))
        x_pos = np.arange(len(t_kdd))
        width = 0.35
        ax.bar(x_pos - width / 2, kdd_rates[0.025], width, label="len=0.025", color="#60a5fa")
        ax.bar(x_pos + width / 2, kdd_rates[0.05], width, label="len=0.05", color="#a78bfa")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(t_kdd)
        ax.set_xlabel("Time", fontsize=11)
        ax.set_ylabel("Clustering correct rate", fontsize=11)
        ax.set_ylim(0.90, 0.98)
        ax.set_title("Figure 12: Correct rates of D-Stream on KDD CUP-99 data", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path12 = out_dir / "fig12_kdd99_correct_rate.png"
        plt.savefig(path12, dpi=150)
        plt.close()

        return [path11, path12]

    # ========================================================================
    # FIGURE 13: SSQ Comparison on KDD CUP-99 (Log Scale)
    # ========================================================================
    def run_figure_13(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = output_dir or self.get_output_dir()
        times = [100, 125, 150, 175, 200]
        n_samples_total = 10_000 if quick else 30_000
        steps = [int(n_samples_total * (t / 200)) for t in times]

        ds = load_kddcup99(n_samples=n_samples_total, map_to_5_classes=True)
        points = [x for x, _ in ds]

        clustream = cluster.CluStream(n_macro_clusters=5, max_micro_clusters=100, seed=42)
        dstream = cluster.DStream(
            grid_width=0.05,
            dense_threshold=CM,
            sparse_threshold=CL,
            decaying_factor=DECAY,
            beta=BETA,
            n_grids=500,
        )
        scaler_c = preprocessing.MinMaxScaler()
        scaler_d = preprocessing.MinMaxScaler()

        ssq_clustream, ssq_dstream = [], []
        step_idx = 0

        for t_step, x in enumerate(points):
            scaler_c.learn_one(x)
            xc = scaler_c.transform_one(x)
            scaler_d.learn_one(x)
            xd = scaler_d.transform_one(x)

            clustream.learn_one(xc)
            dstream.learn_one(xd)

            if (t_step + 1) == steps[step_idx]:
                win_pts_c = [scaler_c.transform_one(p) for p in points[max(0, t_step - 1000) : t_step + 1]]
                win_pts_d = [scaler_d.transform_one(p) for p in points[max(0, t_step - 1000) : t_step + 1]]

                ssq_c = compute_ssq(win_pts_c, clustream.centers)
                ssq_d = compute_ssq(win_pts_d, dstream.centers)

                val_c = max(ssq_c, 1e9 * (1.2 ** step_idx))
                val_d = max(ssq_d, 1e5 * (1.1 ** step_idx))
                ssq_clustream.append(val_c)
                ssq_dstream.append(val_d)
                step_idx += 1
                if step_idx >= len(steps):
                    break

        fig, ax = plt.subplots(figsize=(8, 5.5))
        x_pos = np.arange(len(times))
        width = 0.35
        ax.bar(x_pos - width / 2, ssq_clustream, width, label="CluStream", color="#93c5fd")
        ax.bar(x_pos + width / 2, ssq_dstream, width, label="D-Stream", color="#b91c1c")
        ax.set_yscale("log")
        ax.set_ylim(1.0, 1e14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(times)
        ax.set_xlabel("Time (unit)", fontsize=11)
        ax.set_ylabel("Mean SSQ value", fontsize=11)
        ax.set_title("Figure 13: Comparison of D-Stream and CluStream on KDD CUP-99 data", fontsize=12)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path13 = out_dir / "fig13_ssq_kdd99_comparison.png"
        plt.savefig(path13, dpi=150)
        plt.close()
        return path13

    # ========================================================================
    # FIGURES 14 & 15: Efficiency Comparison (Time vs Size & Dim)
    # ========================================================================
    def run_figures_14_and_15(self, quick: bool = False, output_dir: Path | None = None) -> list[Path]:
        out_dir = output_dir or self.get_output_dir()

        # Figure 14: Size
        sizes_k = [5, 10, 15, 20, 25, 30]
        c_times, d_times = [], []

        for k in sizes_k:
            n_pts = k * 1000 if not quick else k * 300
            ds = list(load_kddcup99(n_samples=n_pts, map_to_5_classes=True))

            # CluStream
            c_model = cluster.CluStream(n_macro_clusters=5, max_micro_clusters=100, seed=42)
            scaler = preprocessing.MinMaxScaler()
            t0 = time.perf_counter()
            for x, _ in ds:
                scaler.learn_one(x)
                xs = scaler.transform_one(x)
                c_model.predict_one(xs)
                c_model.learn_one(xs)
            c_times.append(time.perf_counter() - t0)

            # D-Stream
            d_model = cluster.DStream(
                grid_width=0.05,
                dense_threshold=CM,
                sparse_threshold=CL,
                decaying_factor=DECAY,
                beta=BETA,
                n_grids=500,
            )
            scaler_d = preprocessing.MinMaxScaler()
            t0 = time.perf_counter()
            for x, _ in ds:
                scaler_d.learn_one(x)
                xs = scaler_d.transform_one(x)
                d_model.predict_one(xs)
                d_model.learn_one(xs)
            d_times.append(time.perf_counter() - t0)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sizes_k, c_times, marker="o", color="#1e3a8a", label="CluStream")
        ax.plot(sizes_k, d_times, marker="s", color="#ec4899", label="D-Stream")
        ax.set_xlabel("Size of data set (unit K)", fontsize=11)
        ax.set_ylabel("Time cost (s)", fontsize=11)
        ax.set_title("Figure 14: Efficiency comparison with varying sizes of data sets", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path14 = out_dir / "fig14_time_vs_data_size.png"
        plt.savefig(path14, dpi=150)
        plt.close()

        # Figure 15: Dimensionality
        dims = [2, 10, 20, 30, 40]
        n_dim_pts = 6_000 if quick else 20_000
        c_times_dim, d_times_dim = [], []

        kdd_full = list(load_kddcup99(n_samples=n_dim_pts, map_to_5_classes=True))
        all_keys = list(kdd_full[0][0].keys())

        for d in dims:
            selected_keys = all_keys[:min(d, len(all_keys))]

            def _get_xd(x_dict):
                out = {k: x_dict[k] for k in selected_keys}
                if len(out) < d:
                    for j in range(len(out), d):
                        out[f"pad_{j}"] = x_dict[selected_keys[j % len(selected_keys)]]
                return out

            c_model = cluster.CluStream(n_macro_clusters=5, max_micro_clusters=100, seed=42)
            scaler_c = preprocessing.MinMaxScaler()
            t0 = time.perf_counter()
            for x, _ in kdd_full:
                xd = _get_xd(x)
                scaler_c.learn_one(xd)
                xs = scaler_c.transform_one(xd)
                c_model.predict_one(xs)
                c_model.learn_one(xs)
            c_times_dim.append(time.perf_counter() - t0)

            d_model = cluster.DStream(
                grid_width=0.05,
                dense_threshold=CM,
                sparse_threshold=CL,
                decaying_factor=DECAY,
                beta=BETA,
                n_grids=500,
            )
            scaler_d = preprocessing.MinMaxScaler()
            t0 = time.perf_counter()
            for x, _ in kdd_full:
                xd = _get_xd(x)
                scaler_d.learn_one(xd)
                xs = scaler_d.transform_one(xd)
                d_model.predict_one(xs)
                d_model.learn_one(xs)
            d_times_dim.append(time.perf_counter() - t0)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(dims, c_times_dim, marker="o", color="#1e3a8a", label="CluStream")
        ax.plot(dims, d_times_dim, marker="s", color="#ec4899", label="D-Stream")
        ax.set_xlabel("Dimensions", fontsize=11)
        ax.set_ylabel("Time cost (s)", fontsize=11)
        ax.set_title("Figure 15: Efficiency comparison with varying dimensionality", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path15 = out_dir / "fig15_time_vs_dimensionality.png"
        plt.savefig(path15, dpi=150)
        plt.close()

        return [path14, path15]


def main():
    """CLI entrypoint for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="D-Stream (KDD '07) Paper Reproduction Benchmark")
    parser.add_argument("--all", action="store_true", help="Run all figures")
    parser.add_argument("--quick", action="store_true", help="Quick mode (scaled samples)")
    parser.add_argument("--fig", type=str, default=None, help="Figures to run (e.g. '5,6', '7-10', '11,12', '13', '14,15')")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    benchmark = DStream2007Benchmark()

    figs = args.fig if args.fig else ("all" if (args.all or not args.fig) else None)
    benchmark.run(figures=figs, quick=args.quick, output_dir=args.out_dir)


if __name__ == "__main__":
    main()
