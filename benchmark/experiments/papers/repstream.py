"""Reproduction of Section 5 of "Incremental clustering of dynamic data streams using
connectivity based representative points" (Lühr & Lazarescu, Data & Knowledge
Engineering 68, 2009) -- the RepStream paper.

Every figure of Sec. 5 is reproduced: Fig. 13-18 (synthetic data and the DBSCAN
comparison), Fig. 19-23 (network intrusion and forest cover), Fig. 24-27 (parameter
sensitivity) and Fig. 28-31 (scale-up). Nothing outside Sec. 5 is plotted.

Protocol taken from the paper
-----------------------------
* Manhattan distance everywhere (Sec. 5.1, 5.2, 5.3).
* Normalisation on the fly by Eq. 22: each dimension is divided by the sum of that
  dimension over *the points currently in memory*, implemented as
  :class:`PaperNormalizer` over a window the size of the model's vertex budget.
* Cluster purity by Eq. 18 -- the **unweighted** mean of per-cluster purity, which is
  not the size-weighted purity most libraries ship -- measured over a horizon of the
  last ``h`` points, with ``h = 200`` and ``h = 1000``.
* Cluster entropy by Eq. 21, size-weighted, over the ``h = 1000`` horizon.
* Decay factor ``lambda = 0.99`` in every experiment (Sec. 5).
* ``k = 5``, ``alpha = 1.5`` on both real data sets; ``k = 4, alpha = 4.0`` on DS1 and
  ``k = 4, alpha = 3.0`` on DS2; DBSCAN with ``minPts = 5``.

Four places where this cannot be the paper's experiment
-------------------------------------------------------
1. **Memory budget.** The paper caps RepStream at 10 MiB. Measured with River's
   ``_raw_memory_usage``, this implementation spends ~4.2 KiB per vertex on the
   34-dimensional KDD data, so 10 MiB would be ~2400 vertices; the paper's C++ vertex
   is nearer 500 B, so *its* 10 MiB held ~20000. Neither number is reachable here at a
   sensible cost -- a full 494k-point stream costs O(|V|) per point -- so the real-data
   figures run at ``VERTEX_BUDGET`` vertices and every plot says so in its title. The
   budget is a knob, not a reproduction of the paper's constraint.
2. **HPStream.** Figures 20 and 23 compare against HPStream, which River does not
   implement. Those curves are absent, and the panels say so rather than leaving a
   silent gap.
3. **Competitor numbers.** The paper *read* the CluStream/DenStream/HPStream curves off
   previously published plots; it did not re-run them. Here CluStream and DenStream are
   re-run from River on the identical normalised stream, which is a different -- and
   reproducible -- comparison. Plot titles label them as re-runs.
4. **Scale-up figures.** Fig. 28-30 measure execution time against a parameter, not
   quality, so they run on the first ``SCALEUP_PREFIX`` points instead of the whole
   stream. The shape of each curve is what the figure shows; the absolute seconds are
   not comparable to the paper's 2007 hardware in any case.

The synthetic sets themselves are reconstructions; see
``benchmark/datasets/synthetic/repstream.py`` for what could and could not be recovered
from Fig. 13.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RIVER_ROOT = PROJECT_ROOT.parent / "river"
for _p in (PROJECT_ROOT, RIVER_ROOT):
    if str(_p) not in sys.path and _p.exists():
        sys.path.insert(0, str(_p))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

from benchmark.datasets import load_covtype, load_kddcup99
from benchmark.datasets.synthetic.repstream import (
    DS1_NAMES,
    DS2_NAMES,
    make_repstream_ds1,
    make_repstream_ds2,
)
from river import cluster

from .base import BasePaperBenchmark, register_paper

#: Vertex budget for the two real data sets. See point 1 of the module docstring.
VERTEX_BUDGET = 250
#: Micro-cluster budget for the CluStream comparison. Deliberately *not* matched to
#: ``VERTEX_BUDGET``: River's CluStream searches for the closest pair among all micro
#: clusters whenever it opens a new one, which is O(q^2) per insertion, so q = 600 does
#: not finish on a 500k-point stream. The comparison panels say so.
CLUSTREAM_MICRO = 100
#: Prefix length used by the execution-time figures (28, 29, 30).
SCALEUP_PREFIX = 30_000
#: Stream positions are sampled this often when tracing purity/entropy curves.
SAMPLE_EVERY = 2_500
#: Paper defaults, Sec. 5.
PAPER_K = 5
PAPER_ALPHA = 1.5
PAPER_DECAY = 0.99


# --- Eq. 22 -----------------------------------------------------------------
class PaperNormalizer:
    """On-the-fly normalisation of Eq. 22.

    ``p'[d] = p[d] / sum over the points in memory of that dimension``. "Points in
    memory" is read as a window the size of the model's vertex budget, which is the
    only concrete reading available: the paper ties the denominator to whatever the
    algorithm is currently holding.
    """

    def __init__(self, window: int):
        self.window = window
        self._recent: deque = deque(maxlen=window)
        self._sums: dict = defaultdict(float)

    def learn_transform(self, x: dict) -> dict:
        if len(self._recent) == self._recent.maxlen and self._recent:
            oldest = self._recent[0]
            for key, value in oldest.items():
                self._sums[key] -= value
        self._recent.append(x)
        for key, value in x.items():
            self._sums[key] += value
        out = {}
        for key, value in x.items():
            total = self._sums[key]
            out[key] = value / total if total else 0.0
        return out


# --- Eq. 18 and Eq. 21 ------------------------------------------------------
def paper_purity(pairs) -> float | None:
    """Total clustering purity, Eq. 18: mean over clusters, *not* weighted by size."""
    by_cluster: dict = defaultdict(Counter)
    for y_true, y_pred in pairs:
        by_cluster[y_pred][y_true] += 1
    if not by_cluster:
        return None
    return float(
        np.mean([max(counts.values()) / sum(counts.values()) for counts in by_cluster.values()])
    )


def paper_entropy(pairs) -> float | None:
    """Total cluster entropy, Eq. 21: per-cluster entropy weighted by cluster size."""
    by_cluster: dict = defaultdict(Counter)
    for y_true, y_pred in pairs:
        by_cluster[y_pred][y_true] += 1
    total = sum(sum(counts.values()) for counts in by_cluster.values())
    if not total:
        return None
    out = 0.0
    for counts in by_cluster.values():
        n = sum(counts.values())
        entropy = sum(abs(q * math.log(q)) for q in (v / n for v in counts.values()) if q > 0)
        out += n / total * entropy
    return float(out)


# --- stream runners ---------------------------------------------------------
def _safe_predict(model, x):
    """River clusterers can raise before their first cluster exists; score that as 0."""
    try:
        return model.predict_one(x)
    except (ValueError, ZeroDivisionError, KeyError, IndexError):
        return 0


def _trace_stream(model, rows, horizons, sample_every, budget, entropy_horizon=1000):
    """Run one model test-then-train and trace purity/entropy/elapsed time."""
    scaler = PaperNormalizer(window=budget)
    recent: deque = deque(maxlen=max(max(horizons), entropy_horizon))
    steps: list[int] = []
    purity = {h: [] for h in horizons}
    entropy: list[float] = []
    elapsed: list[float] = []
    started = time.perf_counter()
    for i, (x, y) in enumerate(rows, start=1):
        xs = scaler.learn_transform(x)
        prediction = _safe_predict(model, xs)
        model.learn_one(xs)
        recent.append((y, prediction))
        if i % sample_every == 0:
            window = list(recent)
            steps.append(i)
            for h in horizons:
                purity[h].append(paper_purity(window[-h:]))
            entropy.append(paper_entropy(window[-entropy_horizon:]))
            elapsed.append(time.perf_counter() - started)
    return {
        "steps": steps,
        "purity": {str(h): purity[h] for h in horizons},
        "entropy": entropy,
        "elapsed": elapsed,
        "total_seconds": time.perf_counter() - started,
        "n_clusters": int(getattr(model, "n_clusters", 0) or 0),
    }


def _denstream_epsilon(rows, budget, sample=2000):
    """Pick DenStream's radius from the data instead of guessing.

    Eq. 22 leaves every coordinate on a 1/|P| scale, so River's default epsilon is
    meaningless here. The median nearest-neighbour distance over a sample of the
    normalised stream is used instead, and printed so the choice is visible.
    """
    scaler = PaperNormalizer(window=budget)
    points = []
    for i, (x, _) in enumerate(rows):
        xs = scaler.learn_transform(x)
        if i >= budget:
            points.append([xs[k] for k in sorted(xs)])
        if len(points) >= sample:
            break
    matrix = np.asarray(points, dtype=float)
    distances = np.abs(matrix[:, None, :] - matrix[None, :, :]).sum(axis=2)
    np.fill_diagonal(distances, np.inf)
    epsilon = float(np.median(distances.min(axis=1)))
    print(f"      DenStream epsilon from data: {epsilon:.6f}")
    return epsilon


def _time_stream(model, rows, budget) -> float:
    """Wall-clock seconds to learn the whole prefix, no metrics in the loop."""
    scaler = PaperNormalizer(window=budget)
    started = time.perf_counter()
    for x, _ in rows:
        model.learn_one(scaler.learn_transform(x))
    return time.perf_counter() - started


def _repstream(k=PAPER_K, alpha=PAPER_ALPHA, budget=VERTEX_BUDGET):
    return cluster.RepStream(
        k=k,
        alpha=alpha,
        decay_rate=PAPER_DECAY,
        max_vertices=budget,
        distance_measure="manhattan",
    )


# --- data -------------------------------------------------------------------
class _Stream:
    """Re-iterable ``(x, y)`` stream backed by a dense matrix.

    Holding 500k points as a list of feature dicts costs one to two gigabytes, enough
    to make the machine swap and ruin the very execution-time figures of Sec. 5.5, so
    the values live in one array and each dict is rebuilt as it is handed out.
    """

    def __init__(self, matrix, labels, names):
        self.matrix = matrix
        self.labels = labels
        self.names = names

    def __iter__(self):
        names = self.names
        for row, y in zip(self.matrix, self.labels):
            yield dict(zip(names, row)), y

    def __len__(self):
        return len(self.labels)


def _materialise(dataset, drop_outlier=False):
    names = None
    values: list = []
    labels: list = []
    for x, y in dataset:
        if names is None:
            names = list(x)
        values.append([x[k] for k in names])
        labels.append(y)
    matrix = np.asarray(values, dtype=float)
    del values
    if drop_outlier:
        victim = int(np.abs(matrix).sum(axis=1).argmax())
        matrix = np.delete(matrix, victim, axis=0)
        labels.pop(victim)
    return _Stream(matrix, labels, names)


def _kdd_rows(n_samples=None):
    """KDD Cup-99, 10% subsample, 34 continuous features, raw attack labels.

    Sec. 5.2 keeps the 34 continuous attributes of the 494,021-record 10% set and drops
    a single outlier, which is how the paper arrives at 494,020 points. The dropped
    record is not identified, so the one with the largest feature norm goes.
    """
    dataset = load_kddcup99(n_samples=n_samples, map_to_5_classes=False)
    return _materialise(dataset, drop_outlier=n_samples is None)


def _covtype_rows(n_samples=None):
    """Forest cover, all 54 attributes, as Sec. 5.3 describes them."""
    return _materialise(load_covtype(n_samples=n_samples, quantitative_only=False))


# --- caching ----------------------------------------------------------------
def _cache_path(out_dir: Path, key: str) -> Path:
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{key}.json"


def _cached(out_dir: Path, key: str, compute):
    path = _cache_path(out_dir, key)
    if path.exists():
        print(f"    [cache] {key}")
        return json.loads(path.read_text())
    print(f"    [run  ] {key}")
    result = compute()
    path.write_text(json.dumps(result))
    return result


# --- plotting helpers -------------------------------------------------------
def _scatter(ax, points, labels, title):
    palette = plt.get_cmap("tab20")
    order = sorted(set(labels.tolist()))
    for i, c in enumerate(order):
        mask = labels == c
        colour = "0.75" if c < 0 else palette(i % 20)
        ax.scatter(points[mask, 0], points[mask, 1], s=1.2, color=colour, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def _purity_axes(ax, xlabel="Stream Time", ylabel="Purity", ylim=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        low, high = ylim
        ax.set_ylim(low, high + (high - low) * 0.02)
        ax.set_yticks(np.linspace(low, high, 6))
    ax.grid(alpha=0.25, linewidth=0.5)


def _save(fig, out_dir: Path, filename: str) -> Path:
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    -> {path}")
    return path


def _predict_all(model, rows):
    """Stream the set once, then relabel every point with the final model.

    No Eq. 22 normalisation here: Sec. 5.1 introduces it only for the real data sets in
    Sec. 5.2 and 5.3, and applying it to a 2-d set warps the very geometry the synthetic
    figures are meant to show.
    """
    points = [x for x, _ in rows]
    for x in points:
        model.learn_one(x)
    return np.array([_safe_predict(model, x) for x in points])


@register_paper
class RepStream2009Benchmark(BasePaperBenchmark):
    """Section 5 of the RepStream paper, figure by figure."""

    def __init__(self):
        super().__init__(
            name="repstream_2009",
            title=(
                "Incremental clustering of dynamic data streams using connectivity "
                "based representative points"
            ),
            authors="Sebastian Lühr, Mihai Lazarescu",
            year=2009,
            venue="Data & Knowledge Engineering 68",
            description=(
                "RepStream on two hand-crafted 2-d sets against DBSCAN, on KDD Cup-99 "
                "and Forest CoverType against CluStream/DenStream, plus the parameter "
                "sensitivity and scale-up sweeps of Sec. 5.4 and 5.5."
            ),
        )
        self.setup()

    def setup(self) -> None:
        register = [
            ("13", "Figure 13: synthetic data sets DS1 and DS2", "The two hand-crafted sets of Sec. 5.1, coloured by ground-truth shape.", self.run_figure_13),
            ("14", "Figure 14: RepStream on DS1, k=4 vs k=5 at alpha=4.0", "Higher connectivity removes the middle-triangle fragmentation but merges the two top-left triangles.", self.run_figure_14),
            ("15", "Figure 15: close-up of the fragmented triangle in DS1", "The sparse triangle that RepStream splits in two at k=4, alpha=4.0.", self.run_figure_15),
            ("16", "Figure 16: DBSCAN on DS1 at eps=15 and eps=16", "One global density cannot serve both the dense and the sparse shapes.", self.run_figure_16),
            ("17", "Figure 17: RepStream on DS2, k=4 vs k=5 at alpha=3.0", "The same connectivity trade-off on the second synthetic set.", self.run_figure_17),
            ("18", "Figure 18: DBSCAN on DS2 at eps=6, 6.5 and 7", "The window of workable eps values is half a unit wide.", self.run_figure_18),
            ("19", "Figure 19: purity and entropy on the network intrusion stream", "RepStream at k=5, alpha=1.5 over horizons of 200 and 1000 points.", self.run_figure_19),
            ("20", "Figure 20: RepStream vs CluStream on network intrusion", "HPStream is absent: River does not implement it.", self.run_figure_20),
            ("21", "Figure 21: RepStream vs DenStream vs CluStream on network intrusion", "Re-run from River rather than read off published plots.", self.run_figure_21),
            ("22", "Figure 22: purity and entropy on the forest cover stream", "RepStream at k=5, alpha=1.5 over horizons of 200 and 1000 points.", self.run_figure_22),
            ("23", "Figure 23: RepStream vs CluStream on forest cover", "HPStream is absent: River does not implement it.", self.run_figure_23),
            ("24", "Figure 24: connectivity sensitivity on network intrusion", "k = 3, 7, 9, 11 at alpha = 1.5, horizon 1000.", self.run_figure_24),
            ("25", "Figure 25: connectivity sensitivity on forest cover", "k = 3, 7, 9, 11 at alpha = 1.5, horizon 1000.", self.run_figure_25),
            ("26", "Figure 26: density scaler sensitivity on network intrusion", "alpha = 3.0, 4.5, 6.0 at k = 5, horizon 1000.", self.run_figure_26),
            ("27", "Figure 27: density scaler sensitivity on forest cover", "alpha = 3.0, 4.5, 6.0 at k = 5, horizon 1000.", self.run_figure_27),
            ("28", "Figure 28: execution time against k", "k = 3..11 at alpha = 1.5 on both data sets.", self.run_figure_28),
            ("29", "Figure 29: execution time against alpha", "alpha = 1..6 at k = 5 on both data sets.", self.run_figure_29),
            ("30", "Figure 30: execution time against memory", "Vertex budget swept over the range the paper sweeps in MiB.", self.run_figure_30),
            ("31", "Figure 31: execution time against stream length", "Cumulative time along both full streams; singularities flatten the KDD curve.", self.run_figure_31),
        ]
        for figure_id, title, description, runner in register:
            self.register_figure(figure_id, title, description, runner)

    # -- Sec. 5.1: synthetic data -------------------------------------------
    def run_figure_13(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = self.get_output_dir(output_dir)
        ds1, ds2 = make_repstream_ds1(), make_repstream_ds2()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        _scatter(axes[0], ds1.points, ds1.labels, f"(a) DS1 — {len(ds1.points)} points, {len(DS1_NAMES)} clusters")
        _scatter(axes[1], ds2.points, ds2.labels, f"(b) DS2 — {len(ds2.points)} points, {len(DS2_NAMES)} clusters")
        fig.suptitle("Fig. 13. Synthetic data sets (reconstructed from the paper's figure)", fontsize=10)
        return _save(fig, out_dir, "fig13_synthetic_datasets.png")

    def _repstream_on_synthetic(self, dataset, k, alpha):
        rows = list(dataset)
        model = _repstream(k=k, alpha=alpha, budget=len(rows))
        predicted = _predict_all(model, rows)
        return dataset.points, predicted, model

    def run_figure_14(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = self.get_output_dir(output_dir)
        ds1 = make_repstream_ds1()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, k, tag in ((axes[0], 4, "a"), (axes[1], 5, "b")):
            points, predicted, model = self._repstream_on_synthetic(ds1, k, 4.0)
            _scatter(ax, points, predicted, f"({tag}) k = {k}, alpha = 4.0 — {model.n_clusters} clusters")
        fig.suptitle("Fig. 14. RepStream clustering of DS1", fontsize=10)
        return _save(fig, out_dir, "fig14_repstream_ds1_connectivity.png")

    def run_figure_15(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = self.get_output_dir(output_dir)
        ds1 = make_repstream_ds1()
        points, predicted, _ = self._repstream_on_synthetic(ds1, 4, 4.0)
        counts = {
            i: len({c for c in predicted[ds1.labels == i].tolist()})
            for i in range(len(DS1_NAMES))
        }
        target = max(counts, key=lambda i: (counts[i], -i))
        mask = ds1.labels == target
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        _scatter(ax, points[mask], predicted[mask], "")
        ax.set_title(
            f"Fig. 15. Close-up of the most fragmented shape in DS1: {DS1_NAMES[target]}\n"
            f"k = 4, alpha = 4.0 — RepStream splits it into {counts[target]} clusters",
            fontsize=9,
        )
        return _save(fig, out_dir, "fig15_ds1_fragmented_triangle.png")

    def _dbscan_panel(self, ax, points, eps, tag):
        labels = DBSCAN(eps=eps, min_samples=5, metric="manhattan").fit(points).labels_
        n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
        noise = float((labels == -1).mean())
        _scatter(ax, points, labels, f"({tag}) eps = {eps} — {n_clusters} clusters, {noise:.1%} unclustered")
        return labels

    def run_figure_16(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = self.get_output_dir(output_dir)
        ds1 = make_repstream_ds1()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, eps, tag in ((axes[0], 15.0, "a"), (axes[1], 16.0, "b")):
            self._dbscan_panel(ax, ds1.points, eps, tag)
        fig.suptitle("Fig. 16. DBSCAN clustering of DS1 (minPts = 5, Manhattan)", fontsize=10)
        return _save(fig, out_dir, "fig16_dbscan_ds1.png")

    def run_figure_17(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = self.get_output_dir(output_dir)
        ds2 = make_repstream_ds2()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, k, tag in ((axes[0], 4, "a"), (axes[1], 5, "b")):
            points, predicted, model = self._repstream_on_synthetic(ds2, k, 3.0)
            _scatter(ax, points, predicted, f"({tag}) k = {k}, alpha = 3.0 — {model.n_clusters} clusters")
        fig.suptitle("Fig. 17. RepStream clustering of DS2", fontsize=10)
        return _save(fig, out_dir, "fig17_repstream_ds2_connectivity.png")

    def run_figure_18(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = self.get_output_dir(output_dir)
        ds2 = make_repstream_ds2()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, eps, tag in ((axes[0], 6.0, "a"), (axes[1], 6.5, "b"), (axes[2], 7.0, "c")):
            self._dbscan_panel(ax, ds2.points, eps, tag)
        fig.suptitle("Fig. 18. DBSCAN clustering of DS2 (minPts = 5, Manhattan)", fontsize=10)
        return _save(fig, out_dir, "fig18_dbscan_ds2.png")

    # -- Sec. 5.2 / 5.3: real data ------------------------------------------
    def _real_trace(self, out_dir, dataset, k, alpha, quick):
        limit = 20_000 if quick else None
        key = f"{dataset}_k{k}_a{alpha}_v{VERTEX_BUDGET}" + ("_quick" if quick else "")

        def compute():
            rows = _kdd_rows(limit) if dataset == "kdd" else _covtype_rows(limit)
            model = _repstream(k=k, alpha=alpha)
            return _trace_stream(
                model, rows, horizons=(200, 1000), sample_every=SAMPLE_EVERY if not quick else 500,
                budget=VERTEX_BUDGET,
            )

        return _cached(out_dir, key, compute)

    def _competitor_trace(self, out_dir, dataset, name, quick):
        limit = 20_000 if quick else None
        key = f"{dataset}_{name}_v{VERTEX_BUDGET}" + ("_quick" if quick else "")

        def compute():
            rows = _kdd_rows(limit) if dataset == "kdd" else _covtype_rows(limit)
            n_classes = len({y for _, y in rows})
            if name == "clustream":
                model = cluster.CluStream(
                    n_macro_clusters=n_classes,
                    max_micro_clusters=CLUSTREAM_MICRO,
                    seed=42,
                )
            else:
                model = cluster.DenStream(
                    decaying_factor=0.25,
                    beta=0.5,
                    mu=2.5,
                    epsilon=_denstream_epsilon(rows, VERTEX_BUDGET),
                )
            return _trace_stream(
                model, rows, horizons=(200, 1000), sample_every=SAMPLE_EVERY if not quick else 500,
                budget=VERTEX_BUDGET,
            )

        return _cached(out_dir, key, compute)

    def _quality_figure(self, out_dir, dataset, filename, caption, purity_ylim, entropy_ylim, quick):
        trace = self._real_trace(out_dir, dataset, PAPER_K, PAPER_ALPHA, quick)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
        for ax, h, tag in ((axes[0], "200", "a"), (axes[1], "1000", "b")):
            ax.plot(trace["steps"], trace["purity"][h], linewidth=0.8)
            _purity_axes(ax, ylim=purity_ylim)
            ax.set_title(f"({tag}) horizon h = {h}", fontsize=9)
        axes[2].plot(trace["steps"], trace["entropy"], linewidth=0.8, color="tab:red")
        _purity_axes(axes[2], ylabel="Entropy", ylim=entropy_ylim)
        axes[2].set_title("(c) cluster entropy, h = 1000", fontsize=9)
        fig.suptitle(f"{caption}  |  RepStream k={PAPER_K}, alpha={PAPER_ALPHA}, {VERTEX_BUDGET}-vertex budget", fontsize=10)
        return _save(fig, out_dir, filename)

    def run_figure_19(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._quality_figure(
            self.get_output_dir(output_dir), "kdd", "fig19_kdd_purity_entropy.png",
            "Fig. 19. Network intrusion stream", (0.90, 1.0), (0.0, 0.1), quick,
        )

    def run_figure_22(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._quality_figure(
            self.get_output_dir(output_dir), "covtype", "fig22_covtype_purity_entropy.png",
            "Fig. 22. Forest cover stream", (0.70, 1.0), (0.0, 1.0), quick,
        )

    def _comparison_figure(self, out_dir, dataset, competitors, filename, caption, ylim, quick):
        traces = {"RepStream": self._real_trace(out_dir, dataset, PAPER_K, PAPER_ALPHA, quick)}
        for name in competitors:
            traces[name] = self._competitor_trace(out_dir, dataset, name.lower(), quick)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        for ax, h, tag in ((axes[0], "200", "a"), (axes[1], "1000", "b")):
            for name, trace in traces.items():
                ax.plot(trace["steps"], trace["purity"][h], linewidth=0.8, label=name)
            _purity_axes(ax, ylim=ylim)
            ax.set_title(f"({tag}) horizon h = {h}", fontsize=9)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"{caption}  |  competitors re-run from River, not the paper's cited values. "
            f"HPStream: not in River. RepStream {VERTEX_BUDGET} vertices vs CluStream "
            f"{CLUSTREAM_MICRO} micro-clusters — budgets are not matched",
            fontsize=8,
        )
        return _save(fig, out_dir, filename)

    def run_figure_20(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._comparison_figure(
            self.get_output_dir(output_dir), "kdd", ["CluStream"],
            "fig20_kdd_vs_clustream.png", "Fig. 20. Network intrusion", (0.70, 1.0), quick,
        )

    def run_figure_21(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._comparison_figure(
            self.get_output_dir(output_dir), "kdd", ["DenStream", "CluStream"],
            "fig21_kdd_vs_denstream_clustream.png", "Fig. 21. Network intrusion", (0.70, 1.0), quick,
        )

    def run_figure_23(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._comparison_figure(
            self.get_output_dir(output_dir), "covtype", ["CluStream"],
            "fig23_covtype_vs_clustream.png", "Fig. 23. Forest cover", (0.50, 1.0), quick,
        )

    # -- Sec. 5.4: parameter sensitivity ------------------------------------
    def _sensitivity_figure(self, out_dir, dataset, sweep, filename, caption, ylim, quick, vary):
        fig, axes = plt.subplots(1, len(sweep), figsize=(3.6 * len(sweep), 3.8), squeeze=False)
        for ax, value, tag in zip(axes[0], sweep, "abcdef"):
            k = value if vary == "k" else PAPER_K
            alpha = value if vary == "alpha" else PAPER_ALPHA
            trace = self._real_trace(out_dir, dataset, k, alpha, quick)
            ax.plot(trace["steps"], trace["purity"]["1000"], linewidth=0.8)
            _purity_axes(ax, ylim=ylim)
            label = f"k = {value}" if vary == "k" else f"alpha = {value}"
            ax.set_title(f"({tag}) {label}", fontsize=9)
        fig.suptitle(f"{caption}  |  horizon h = 1000, {VERTEX_BUDGET}-vertex budget", fontsize=10)
        return _save(fig, out_dir, filename)

    def run_figure_24(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._sensitivity_figure(
            self.get_output_dir(output_dir), "kdd", [3, 7, 9, 11],
            "fig24_kdd_connectivity_sensitivity.png",
            "Fig. 24. Connectivity sensitivity, network intrusion", (0.70, 1.0), quick, "k",
        )

    def run_figure_25(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._sensitivity_figure(
            self.get_output_dir(output_dir), "covtype", [3, 7, 9, 11],
            "fig25_covtype_connectivity_sensitivity.png",
            "Fig. 25. Connectivity sensitivity, forest cover", (0.70, 1.0), quick, "k",
        )

    def run_figure_26(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._sensitivity_figure(
            self.get_output_dir(output_dir), "kdd", [3.0, 4.5, 6.0],
            "fig26_kdd_alpha_sensitivity.png",
            "Fig. 26. Density scaler sensitivity, network intrusion", (0.86, 1.0), quick, "alpha",
        )

    def run_figure_27(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._sensitivity_figure(
            self.get_output_dir(output_dir), "covtype", [3.0, 4.5, 6.0],
            "fig27_covtype_alpha_sensitivity.png",
            "Fig. 27. Density scaler sensitivity, forest cover", (0.70, 1.0), quick, "alpha",
        )

    # -- Sec. 5.5: scale-up --------------------------------------------------
    def _scaleup(self, out_dir, dataset, key, values, build, quick):
        prefix = 5_000 if quick else SCALEUP_PREFIX
        cache_key = f"scaleup_{dataset}_{key}_n{prefix}"

        def compute():
            rows = _kdd_rows(prefix) if dataset == "kdd" else _covtype_rows(prefix)
            seconds = []
            for value in values:
                model = build(value)
                window = getattr(model, "max_vertices", VERTEX_BUDGET)
                seconds.append(_time_stream(model, rows, window))
            return {"values": list(values), "seconds": seconds}

        return _cached(out_dir, cache_key, compute)

    def _scaleup_figure(self, out_dir, key, values, build, filename, caption, xlabel, quick):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for ax, dataset, tag, label in (
            (axes[0], "kdd", "a", "network intrusion"),
            (axes[1], "covtype", "b", "forest canopy"),
        ):
            result = self._scaleup(out_dir, dataset, key, values, build, quick)
            ax.plot(result["values"], result["seconds"], marker="o", linewidth=0.9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Execution Time (seconds)")
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.set_title(f"({tag}) {label}", fontsize=9)
        prefix = 5_000 if quick else SCALEUP_PREFIX
        fig.suptitle(f"{caption}  |  first {prefix:,} points of each stream", fontsize=10)
        return _save(fig, out_dir, filename)

    def run_figure_28(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._scaleup_figure(
            self.get_output_dir(output_dir), "k", list(range(3, 12)),
            lambda v: _repstream(k=v), "fig28_time_vs_k.png",
            "Fig. 28. Execution time against k-nearest neighbours", "K-Nearest Neighbours", quick,
        )

    def run_figure_29(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        return self._scaleup_figure(
            self.get_output_dir(output_dir), "alpha", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
            lambda v: _repstream(alpha=v), "fig29_time_vs_alpha.png",
            "Fig. 29. Execution time against the density scaler", "Alpha", quick,
        )

    def run_figure_30(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        budgets = [240, 360, 480, 600, 720, 840, 960, 1080, 1200]
        return self._scaleup_figure(
            self.get_output_dir(output_dir), "memory", budgets,
            lambda v: _repstream(budget=v), "fig30_time_vs_memory.png",
            "Fig. 30. Execution time against the vertex budget", "Vertex budget", quick,
        )

    def run_figure_31(self, quick: bool = False, output_dir: Path | None = None) -> Path:
        out_dir = self.get_output_dir(output_dir)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for ax, dataset, tag, label in (
            (axes[0], "kdd", "a", "network intrusion"),
            (axes[1], "covtype", "b", "forest canopy"),
        ):
            trace = self._real_trace(out_dir, dataset, PAPER_K, PAPER_ALPHA, quick)
            ax.plot(trace["steps"], trace["elapsed"], linewidth=0.9)
            ax.set_xlabel("Stream Length")
            ax.set_ylabel("Execution Time (seconds)")
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.set_title(f"({tag}) {label}", fontsize=9)
        fig.suptitle(f"Fig. 31. Execution time against stream length  |  k={PAPER_K}, alpha={PAPER_ALPHA}, {VERTEX_BUDGET}-vertex budget", fontsize=10)
        return _save(fig, out_dir, "fig31_time_vs_stream_length.png")
