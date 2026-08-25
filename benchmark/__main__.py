"""Command-line entry point.

Examples::

    # Compare all default algorithms on a synthetic labeled dataset
    python -m benchmark synthetic --n-samples 20000

    # Compare on KDDCup99 (34 continuous features, auto-fetched)
    python -m benchmark kdd99 --n-samples 50000 --scalers standard

    # Repeat over seeds and aggregate to mean +/- std
    python -m benchmark synthetic --seeds 1,2,3

    # Write plots + JSON to a directory
    python -m benchmark synthetic --out results/
"""

from __future__ import annotations

import argparse
import json
import os

from .algorithms.registry import ALGORITHMS, SCALERS
from .datasets import SyntheticStream, StreamConfig, load_kddcup98, load_kddcup99
from .experiments import (
    BenchmarkConfig,
    Setting,
    run_benchmark,
    run_benchmark_repeated,
)
from .experiments.scalability import scalability_table, sweep_features, sweep_samples
from .reports.tables import (
    efficiency_table,
    efficiency_table_agg,
    quality_table,
    quality_table_agg,
    stream_quality_table,
    stream_quality_table_agg,
)

DEFAULT_ALGOS = ("CluStream", "DenStream", "DBSTREAM", "STREAMKMeans", "KMeans")


def _build_settings(algos, scalers) -> list[Setting]:
    settings = []
    for alg in algos:
        for sc in scalers:
            settings.append(Setting(algorithm=alg, scaler=sc))
    return settings


def _make_dataset(args):
    if args.command == "synthetic":
        cfg = StreamConfig(
            n_samples=args.n_samples or 20_000,
            n_features=args.n_features,
            n_clusters=args.n_clusters,
            cluster_std=args.cluster_std,
            noise=args.noise,
            seed=args.data_seed,
        )
        return SyntheticStream(cfg)
    if args.command == "kdd99":
        return load_kddcup99(n_samples=args.n_samples, percent10=not args.full)
    if args.command == "kdd98":
        return load_kddcup98(path=args.path, n_samples=args.n_samples)
    raise ValueError(args.command)


def _emit_plots(results, out_dir):
    try:
        from .reports import (
            plot_clusters_vs_samples,
            plot_memory_vs_samples,
            plot_quality_bars,
            plot_runtime_vs_samples,
        )
    except Exception as e:
        print(f"(skipping plots — matplotlib unavailable: {e})")
        return
    os.makedirs(out_dir, exist_ok=True)
    plot_runtime_vs_samples(results).savefig(os.path.join(out_dir, "runtime_vs_samples.png"), dpi=90, bbox_inches="tight")
    plot_memory_vs_samples(results).savefig(os.path.join(out_dir, "memory_vs_samples.png"), dpi=90, bbox_inches="tight")
    plot_clusters_vs_samples(results).savefig(os.path.join(out_dir, "clusters_vs_samples.png"), dpi=90, bbox_inches="tight")
    if any(r.quality for r in results):
        plot_quality_bars(results).savefig(os.path.join(out_dir, "quality_bars.png"), dpi=90, bbox_inches="tight")
    print(f"wrote plots to {out_dir}")


def _run(args):
    dataset = _make_dataset(args)
    algos = args.algorithms.split(",") if args.algorithms else list(DEFAULT_ALGOS)
    scalers = args.scalers.split(",") if args.scalers else ["none"]
    seeds = tuple(int(s) for s in args.seeds.split(","))

    internal = tuple(args.internal.split(",")) if args.internal else ()
    config = BenchmarkConfig(
        settings=_build_settings(algos, scalers),
        seeds=seeds,
        n_samples=args.n_samples,
        n_checkpoints=args.checkpoints,
        window_size=args.window,
        internal_metrics=internal,
    )
    primary = config.external_metrics[0]

    print(f"# Benchmark on dataset: {dataset.name}")
    print(f"  settings: {[s.display() for s in config.settings]}")
    print(f"  seeds: {seeds}  |  window_size: {args.window}\n")

    if len(seeds) > 1:
        digest = run_benchmark_repeated(dataset, config)
        print("## Efficiency (mean ± std)\n")
        print(efficiency_table_agg(digest))
        if dataset.has_labels:
            print("\n## Quality — whole stream (mean ± std)\n")
            print(quality_table_agg(digest))
            print(f"\n## Quality — {primary.upper()} over stream (mean ± std)\n")
            print(stream_quality_table_agg(digest, metric=primary))
        results = [b["runs"][0] for b in digest.values()]  # first-seed runs for plots
        payload = {label: b["aggregate"] for label, b in digest.items()}
    else:
        results = run_benchmark(dataset, config)
        print("## Efficiency\n")
        print(efficiency_table(results))
        if dataset.has_labels:
            print("\n## Quality — whole stream\n")
            print(quality_table(results))
            print(f"\n## Quality — {primary.upper()} over stream (window mean/median/final)\n")
            print(stream_quality_table(results, metric=primary))
        if internal:
            print(f"\n## Internal quality — {internal[0]} over stream\n")
            print(stream_quality_table(results, metric=internal[0]))
        payload = [r.to_dict() for r in results]

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        json_path = os.path.join(args.out, "results.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {json_path}")
        _emit_plots(results, args.out)


def _add_common(p):
    p.add_argument("--algorithms", default=None,
                   help=f"comma-separated; default all of {list(DEFAULT_ALGOS)}")
    p.add_argument("--scalers", default=None,
                   help=f"comma-separated online scalers; options: {sorted(SCALERS)}")
    p.add_argument("--seeds", default="1", help="comma-separated seeds (>1 => aggregate)")
    p.add_argument("--n-samples", dest="n_samples", type=int, default=None)
    p.add_argument("--checkpoints", type=int, default=20,
                   help="number of efficiency measurements across the stream")
    p.add_argument("--window", type=int, default=1000,
                   help="quality evaluation window size (time-aware evaluation)")
    p.add_argument("--internal", default=None,
                   help="comma-separated internal metrics: silhouette,davies_bouldin,calinski_harabasz")
    p.add_argument("--out", default=None, help="directory for JSON + plots")
    p.set_defaults(func=_run)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchmark", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synthetic", help="synthetic labeled Gaussian-blob dataset")
    s.add_argument("--n-features", dest="n_features", type=int, default=2)
    s.add_argument("--n-clusters", dest="n_clusters", type=int, default=5)
    s.add_argument("--cluster-std", dest="cluster_std", type=float, default=1.0)
    s.add_argument("--noise", type=float, default=0.0)
    s.add_argument("--data-seed", dest="data_seed", type=int, default=42)
    _add_common(s)

    k99 = sub.add_parser("kdd99", help="KDDCup99 (34 continuous features, auto-fetched)")
    k99.add_argument("--full", action="store_true", help="use full set instead of 10%%")
    _add_common(k99)

    k98 = sub.add_parser("kdd98", help="KDDCup98 (from local cup98LRN.txt)")
    k98.add_argument("--path", required=True, help="path to cup98LRN.txt")
    _add_common(k98)

    sc = sub.add_parser("scalability", help="sweep runtime/memory/quality vs samples or features")
    sc.add_argument("--axis", choices=("samples", "features"), default="samples")
    sc.add_argument("--values", default=None,
                    help="comma-separated axis values (default: 10000,50000,100000 or 2,10,50,100)")
    sc.add_argument("--algorithms", default=None)
    sc.add_argument("--scalers", default=None)
    sc.add_argument("--n-clusters", dest="n_clusters", type=int, default=5)
    sc.add_argument("--n-features", dest="n_features", type=int, default=2)
    sc.add_argument("--n-samples", dest="n_samples", type=int, default=20_000)
    sc.add_argument("--metric", default="ari")
    sc.add_argument("--seed", type=int, default=1)
    sc.set_defaults(func=_run_scalability)

    return p


def _run_scalability(args):
    algos = args.algorithms.split(",") if args.algorithms else list(DEFAULT_ALGOS)
    scalers = args.scalers.split(",") if args.scalers else ["standard"]
    settings = _build_settings(algos, scalers)

    if args.axis == "samples":
        sizes = ([int(v) for v in args.values.split(",")] if args.values
                 else (10_000, 50_000, 100_000))
        sweep = sweep_samples(settings, sizes=sizes, n_clusters=args.n_clusters,
                              n_features=args.n_features, seed=args.seed)
        label = "n_samples"
    else:
        dims = ([int(v) for v in args.values.split(",")] if args.values
                else (2, 10, 50, 100))
        sweep = sweep_features(settings, dims=dims, n_samples=args.n_samples,
                               n_clusters=args.n_clusters, seed=args.seed)
        label = "n_features"

    print(f"# Scalability sweep vs {label}\n")
    print(scalability_table(sweep, axis_label=label, metric=args.metric))


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
