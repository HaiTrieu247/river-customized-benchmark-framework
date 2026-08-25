"""Plots for the resource benchmark.

Uses a non-interactive matplotlib backend so it works headless. Each function
returns the matplotlib ``Figure`` so callers can save or show it.

* ``plot_runtime_vs_samples`` — cumulative runtime as the stream grows;
* ``plot_memory_vs_samples`` — model memory (MB) as the stream grows;
* ``plot_clusters_vs_samples`` — micro-cluster count over the stream;
* ``plot_quality_bars`` — final quality comparison (labeled datasets).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _checkpoint_xy(result, key):
    xs, ys = [], []
    for c in result.checkpoints:
        v = getattr(c, key)
        if v is not None:
            xs.append(c.n_seen)
            ys.append(v)
    return xs, ys


def plot_runtime_vs_samples(results, ax=None):
    created = ax is None
    fig, ax = (plt.subplots(figsize=(9, 5)) if created else (ax.figure, ax))
    for r in results:
        xs, ys = _checkpoint_xy(r, "cumulative_runtime_s")
        ax.plot(xs, ys, marker="o", ms=3, label=r.algorithm)
    ax.set_xlabel("samples seen")
    ax.set_ylabel("cumulative runtime (s)")
    ax.set_title("Execution time vs samples")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_memory_vs_samples(results, ax=None):
    created = ax is None
    fig, ax = (plt.subplots(figsize=(9, 5)) if created else (ax.figure, ax))
    for r in results:
        xs, ys = _checkpoint_xy(r, "memory_bytes")
        ys = [y / (1024 * 1024) for y in ys]
        ax.plot(xs, ys, marker="o", ms=3, label=r.algorithm)
    ax.set_xlabel("samples seen")
    ax.set_ylabel("model memory (MB)")
    ax.set_title("RAM vs samples")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_clusters_vs_samples(results, ax=None):
    created = ax is None
    fig, ax = (plt.subplots(figsize=(9, 5)) if created else (ax.figure, ax))
    for r in results:
        xs, ys = _checkpoint_xy(r, "n_micro_clusters")
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, label=r.algorithm)
    ax.set_xlabel("samples seen")
    ax.set_ylabel("# micro-clusters")
    ax.set_title("Micro-cluster count vs samples")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return fig


def plot_quality_bars(results, metric="ari", ax=None):
    created = ax is None
    fig, ax = (plt.subplots(figsize=(9, 5)) if created else (ax.figure, ax))
    labels = [r.algorithm for r in results]
    values = [r.quality.get(metric) or 0.0 for r in results]
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Final {metric.upper()} by setting")
    ax.grid(alpha=0.3, axis="y")
    return fig
