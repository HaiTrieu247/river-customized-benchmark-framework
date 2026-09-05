"""Basic benchmark runner for river.cluster.

One script, driven entirely from the command line. It does not choose
hyperparameters or datasets for you — every value below is a placeholder you are
expected to override. What it does is run whatever you ask for under a
consistent protocol and report it honestly.

What the protocol guarantees
----------------------------
* every setting consumes the **same stream**, in the same record order;
* each sample is predicted before it is learned from (test-then-train);
* quality is tracked both cumulatively and **per evaluation window**, because
  averaging the cumulative curve is not a time average;
* resources are sampled at checkpoints, so scalability is visible;
* randomised algorithms are repeated over seeds and reported as mean +/- std,
  while deterministic ones run once and say so;
* the environment is recorded, including ``PYTHONHASHSEED`` (see RIVER_BUGS.md).

Examples
--------
    # four clusterers on a synthetic stream
    python run_benchmark.py --dataset synthetic --algorithms DBSTREAM,STREAMKMeans

    # your own parameters, as JSON
    python run_benchmark.py --dataset kdd99 --n-samples 50000 \
        --algorithms DenStream \
        --params '{"DenStream": {"epsilon": 0.5, "mu": 2.5}}'

    # compare preprocessing chains for one algorithm
    python run_benchmark.py --dataset kdd99 --algorithms DBSTREAM \
        --scalers standard,standard_safe,adaptive_standard

    # reproducible numbers (see RIVER_BUGS.md BUG-2)
    PYTHONHASHSEED=0 python run_benchmark.py --dataset synthetic --seeds 1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Windows consoles default to cp1252 and cannot print the report. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from benchmark.algorithms.registry import (
    ALGORITHMS,
    PREPROCESSORS,
    accepts_seed,
    algorithms_for,
    get_spec,
)
from benchmark.datasets import (
    INSECTS_VARIANTS,
    SCENARIOS,
    load_covtype,
    load_electricity,
    load_insects,
    load_kddcup98,
    load_kddcup99,
    load_keystroke,
    load_shuttle,
    load_sms_spam,
    make_scenario,
)
from benchmark.evaluation.aggregation import aggregate_runs
from benchmark.experiments import BenchmarkConfig, Setting, run_setting
from benchmark.reports import (
    capture_environment,
    efficiency_table,
    efficiency_table_agg,
    format_environment,
    quality_table,
    quality_table_agg,
    stream_quality_table,
    stream_quality_table_agg,
    structure_table,
    window_diagnostics_table,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _require(value, flag: str):
    """Fail with the flag name rather than a stack trace deep in a loader."""
    if not value:
        raise SystemExit(f"this dataset needs {flag}")
    return value


def _check_keyed(mapping: dict, names: list[str], flag: str, kind: str) -> None:
    """Reject a JSON map whose keys are not names actually being run."""
    stray = sorted(set(mapping) - set(names))
    if stray:
        raise SystemExit(
            f"{flag} is keyed by {kind} name; {stray} match nothing being run "
            f"({sorted(set(names))}). Example: "
            f'{flag} \'{{"{names[0] if names else "NAME"}": {{...}}}}\''
        )
    for name, value in mapping.items():
        if not isinstance(value, dict):
            raise SystemExit(
                f"{flag}: params for {name!r} must be an object, "
                f"got {type(value).__name__}"
            )

#: Datasets the framework can load. Add your own by writing a loader that returns
#: a ``benchmark.datasets.Dataset`` and registering it here.
DATASETS = {
    "synthetic": lambda a: make_scenario(
        a.scenario, n_samples=a.n_samples, n_features=a.n_features,
        n_clusters=a.n_clusters, seed=a.data_seed,
    ),
    "kdd99": lambda a: load_kddcup99(
        n_samples=a.n_samples, include_nominal=a.include_nominal
    ),
    # Needs a local cup98LRN.txt; there is no automatic download for it.
    "kdd98": lambda a: load_kddcup98(_require(a.kdd98_path, "--kdd98-path"),
                                     n_samples=a.n_samples),
    "covtype": lambda a: load_covtype(n_samples=a.n_samples),
    "insects": lambda a: load_insects(a.insects_variant, n_samples=a.n_samples),
    "keystroke": lambda a: load_keystroke(n_samples=a.n_samples),
    "shuttle": lambda a: load_shuttle(n_samples=a.n_samples),
    "electricity": lambda a: load_electricity(n_samples=a.n_samples),
    "sms_spam": lambda a: load_sms_spam(n_samples=a.n_samples),
}


def build_settings(args) -> list[Setting]:
    """Turn the command line into a list of settings.

    ``--params`` and ``--scaler-params`` are JSON maps keyed by algorithm and by
    preprocessor name respectively. Anything not named there falls back to
    River's own defaults, which is the honest starting point: the framework has
    no opinion about what your parameters should be.
    """
    algorithms = (
        [a.strip() for a in args.algorithms.split(",")]
        if args.algorithms
        else list(algorithms_for())
    )
    scalers = [s.strip() for s in args.scalers.split(",")] if args.scalers else ["none"]
    params = json.loads(args.params) if args.params else {}
    scaler_params = json.loads(args.scaler_params) if args.scaler_params else {}

    unknown = [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        raise SystemExit(f"unknown algorithm(s) {unknown}; available: {sorted(ALGORITHMS)}")
    steps = [step for spec in scalers for step in spec.split("|")]
    unknown = [s for s in steps if s not in PREPROCESSORS]
    if unknown:
        raise SystemExit(f"unknown preprocessor(s) {unknown}; available: {sorted(PREPROCESSORS)}")

    # Both JSON maps are keyed by name. Getting that wrong -- passing
    # `{"on": "body"}` instead of `{"bag_of_words": {"on": "body"}}` -- otherwise
    # fails silently: the params are simply never applied, and the step receives
    # its defaults and dies much later inside river.
    _check_keyed(params, algorithms, "--params", "algorithm")
    _check_keyed(scaler_params, steps, "--scaler-params", "preprocessing step")

    return [
        Setting(
            algorithm=name,
            params=dict(params.get(name, {})),
            scaler=scaler,
            scaler_params=dict(scaler_params.get(scaler, {})),
        )
        for name in algorithms
        for scaler in scalers
    ]


def seeds_for(setting: Setting, seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Repeat seeds only for algorithms that actually have randomness.

    Repeating a deterministic algorithm on the same stream returns the identical
    number, so it only costs time. The report states each setting's real run
    count so a standard deviation of zero is not mistaken for stability.
    """
    return seeds if accepts_seed(setting.algorithm) else seeds[:1]


def run_all(settings, dataset, config, seeds) -> dict:
    digest: dict[str, dict] = {}
    total = sum(len(seeds_for(s, seeds)) for s in settings)
    done = 0
    for setting in settings:
        runs = []
        for seed in seeds_for(setting, seeds):
            done += 1
            print(f"  [{done}/{total}] {setting.display()} seed={seed} ...", end="", flush=True)
            t0 = time.perf_counter()
            result = run_setting(setting, dataset, config, seed=seed)
            elapsed = time.perf_counter() - t0
            ari = result.quality.get("ari")
            score = f"ARI={ari:.4f}" if ari is not None else "no per-sample partition"
            print(
                f" done {elapsed:7.1f}s | {result.efficiency.throughput_sps:8.0f} sps | {score}",
                flush=True,
            )
            runs.append(result)
        digest[setting.display()] = {"runs": runs, "aggregate": aggregate_runs(runs)}
    return digest


def build_report(digest, dataset, config, seeds, env) -> str:
    results = [block["runs"][0] for block in digest.values()]
    multi = any(len(block["runs"]) > 1 for block in digest.values())
    # Algorithms that assign samples to clusters can be scored on external
    # quality; the rest (ODAC clusters variables) only on structure. Mixing them
    # into one quality table would print blanks that read like zeros.
    scored = [r for r in results if r.partitions_samples]
    structural = [r for r in results if not r.partitions_samples]
    scored_digest = {
        label: block for label, block in digest.items()
        if block["runs"][0].partitions_samples
    }
    out: list[str] = [f"# river.cluster benchmark — {dataset.name}"]
    out.append(
        f"\n- n_samples: {results[0].n_samples:,}"
        f"\n- n_features: {dataset.n_features}"
        f"\n- window_size: {config.window_size}"
        f"\n- checkpoints: {config.n_checkpoints}"
        f"\n- seeds: {list(seeds)}"
        f"\n- labelled: {dataset.has_labels}"
    )

    def section(title, body):
        out.append(f"\n## {title}\n\n{body}")

    section("Environment", format_environment(env))
    if env.get("python_hash_seed") == "random (unpinned)":
        out.append(
            "\n> ⚠ `PYTHONHASHSEED` is unpinned. River's KMeans gives different"
            " results per process even with a fixed `seed` (RIVER_BUGS.md BUG-2),"
            " so these numbers are not reproducible as they stand. Re-run with"
            " `PYTHONHASHSEED=0` for figures you intend to publish."
        )

    provenance = {s.display(): get_spec(s.algorithm).provenance for s in config.settings}
    section(
        "Configuration",
        "\n".join(
            f"- **{r.algorithm}** — params: `{r.params}` | preprocessing: "
            f"`{r.scaler}` params: `{r.scaler_params or '{}'}`"
            f" | provenance: {provenance.get(r.algorithm, 'unknown')}"
            for r in results
        ),
    )
    out.append(
        "\n> Parameters not named on the command line are River's own defaults."
        " The framework does not tune them."
    )

    section("Efficiency", efficiency_table_agg(digest) if multi else efficiency_table(results))
    out.append(
        "\n> `Model MB` is the model (+preprocessing) footprint via River's"
        " `_raw_memory_usage`, the comparable figure across algorithms. `RSS MB`"
        " is whole-process memory, for context. Peak memory is a max over"
        " checkpoints, so it is a lower bound for models that prune."
    )

    if structural:
        section("Structure — algorithms that do not partition samples", structure_table(structural))
        out.append(
            "\n> ODAC clusters **variables**, not samples; its `predict_one`"
            " raises `NotImplementedError` by design. External metrics are"
            " undefined for it, so it is absent from the quality tables below —"
            " that is correct, not a failure."
        )

    if dataset.has_labels and scored:
        section(
            "Quality — whole stream",
            quality_table_agg(scored_digest) if multi else quality_table(scored),
        )
        section("Window diagnostics — read BEFORE the time-aware tables",
                window_diagnostics_table(scored))
        diag = next((r.window_label_diagnostics for r in scored if r.window_label_diagnostics), None)
        if diag and diag["single_class_window_share"] >= 0.3:
            out.append(
                f"\n> ⚠ **{diag['single_class_window_share']:.0%} of windows hold a single"
                " class.** On such a window ARI degenerates into a \"did the model lump"
                " the whole window together?\" test: 1.0 for lumping, 0.0 for any split,"
                " regardless of clustering quality. An under-segmenting model therefore"
                " scores well for the wrong reason. Read the whole-stream table as the"
                " primary result and treat the time-aware tables as trend only."
            )
        for metric in config.external_metrics:
            section(
                f"Quality over time — {metric.upper()} per window",
                stream_quality_table_agg(scored_digest, metric=metric) if multi
                else stream_quality_table(scored, metric=metric),
            )

    if multi:
        counts = ", ".join(f"{label}: {len(b['runs'])}" for label, b in digest.items())
        out.append(
            f"\n> Real run counts — {counts}. Deterministic algorithms run once,"
            " so their standard deviation is zero by definition, not by stability."
        )
    return "\n".join(out) + "\n"


def emit_plots(results, out_dir) -> list[str]:
    try:
        from benchmark.reports import (
            plot_clusters_vs_samples,
            plot_memory_vs_samples,
            plot_quality_vs_throughput,
            plot_runtime_vs_samples,
            plot_throughput_vs_samples,
            plot_window_quality_curve,
        )
    except Exception as exc:
        print(f"\n(skipping plots — matplotlib unavailable: {exc})")
        return []

    os.makedirs(out_dir, exist_ok=True)
    written = []

    def save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        written.append(path)

    save(plot_runtime_vs_samples(results), "runtime_vs_samples.png")
    save(plot_memory_vs_samples(results), "memory_vs_samples.png")
    save(plot_clusters_vs_samples(results), "clusters_vs_samples.png")
    save(plot_throughput_vs_samples(results), "throughput_vs_samples.png")
    if results[0].quality.get("ari") is not None:
        save(plot_window_quality_curve(results, metric="ari"), "ari_per_window.png")
        save(plot_quality_vs_throughput(results, metric="ari"), "ari_vs_throughput.png")
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    d = p.add_argument_group("dataset")
    d.add_argument("--dataset", choices=sorted(DATASETS), default="synthetic")
    d.add_argument("--n-samples", dest="n_samples", type=int, default=20_000)
    d.add_argument("--scenario", choices=sorted(SCENARIOS), default="static_easy",
                   help="synthetic only")
    d.add_argument("--n-features", dest="n_features", type=int, default=10,
                   help="synthetic only")
    d.add_argument("--n-clusters", dest="n_clusters", type=int, default=5,
                   help="synthetic only")
    d.add_argument("--data-seed", dest="data_seed", type=int, default=42,
                   help="synthetic only")
    d.add_argument("--insects-variant", dest="insects_variant",
                   choices=sorted(INSECTS_VARIANTS), default="abrupt_balanced")
    d.add_argument("--include-nominal", dest="include_nominal", action="store_true",
                   help="kdd99 only: keep the 3 nominal columns (needs a categorical encoder)")
    d.add_argument("--kdd98-path", dest="kdd98_path", default=None, metavar="PATH",
                   help="kdd98 only: path to a local cup98LRN.txt (no auto-download)")

    m = p.add_argument_group("what to run")
    m.add_argument("--algorithms", default=None,
                   help=f"comma-separated; default {list(algorithms_for())}. "
                        f"all: {sorted(ALGORITHMS)}")
    m.add_argument("--scalers", default=None,
                   help=f"comma-separated preprocessing specs, chain with '|'. "
                        f"steps: {sorted(PREPROCESSORS)}")
    m.add_argument("--params", default=None, metavar="JSON",
                   help='per-algorithm params, e.g. \'{"DenStream": {"epsilon": 0.5}}\'')
    m.add_argument("--scaler-params", dest="scaler_params", default=None, metavar="JSON",
                   help='per-preprocessor params, e.g. \'{"bag_of_words": {"on": "body"}}\'')

    e = p.add_argument_group("protocol")
    e.add_argument("--seeds", default="1", help="comma-separated")
    e.add_argument("--window", type=int, default=1000, help="evaluation window size")
    e.add_argument("--checkpoints", type=int, default=20, help="resource sampling points")
    e.add_argument("--metrics", default="ari,nmi,ami,purity", help="external metrics")
    e.add_argument("--internal", action="store_true",
                   help="also compute river's streaming silhouette (slower)")
    e.add_argument("--out", default=None, help="output directory (default: benchmark_out/<dataset>)")
    return p


def main():
    args = build_parser().parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    settings = build_settings(args)
    out_dir = args.out or os.path.join(HERE, "benchmark_out", args.dataset)

    print("=" * 78)
    print(f"river.cluster benchmark | dataset={args.dataset} | "
          f"n_samples={args.n_samples:,} | seeds={list(seeds)}")
    print("=" * 78)

    env = capture_environment()
    dataset = DATASETS[args.dataset](args)
    print(f"\ndataset: {dataset.name} | n_features={dataset.n_features} | "
          f"labelled={dataset.has_labels}")

    config = BenchmarkConfig(
        settings=settings,
        seeds=seeds,
        n_samples=args.n_samples,
        n_checkpoints=args.checkpoints,
        window_size=args.window,
        external_metrics=tuple(m.strip() for m in args.metrics.split(",") if m.strip()),
        internal_metrics=("silhouette",) if args.internal else (),
    )

    n_runs = sum(len(seeds_for(s, seeds)) for s in settings)
    print(f"\nrunning {len(settings)} setting(s) / {n_runs} run(s)\n")
    digest = run_all(settings, dataset, config, seeds)

    report = build_report(digest, dataset, config, seeds, env)
    print("\n" + report)

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    payload = {
        "environment": env,
        "protocol": {
            "dataset": dataset.name,
            "n_samples": args.n_samples,
            "window_size": args.window,
            "n_checkpoints": args.checkpoints,
            "seeds": list(seeds),
        },
        "settings": {
            label: {
                "runs": [r.to_dict() for r in block["runs"]],
                "aggregate": block["aggregate"],
            }
            for label, block in digest.items()
        },
    }
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    plots = emit_plots([b["runs"][0] for b in digest.values()], out_dir)
    print(f"\nwrote:\n  {report_path}\n  {json_path}")
    for path in plots:
        print(f"  {path}")


if __name__ == "__main__":
    main()
