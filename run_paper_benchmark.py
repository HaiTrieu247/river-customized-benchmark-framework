"""Unified CLI Runner for Research Paper Benchmark Reproductions.

Allows running, listing, and reproducing experiments from any registered
data stream clustering research paper.

Examples:
    # List all available paper benchmarks and their figures
    python run_paper_benchmark.py --list

    # Run D-Stream (KDD '07) benchmark in quick mode
    python run_paper_benchmark.py --paper dstream_2007 --quick

    # Run full paper benchmark for D-Stream
    python run_paper_benchmark.py --paper dstream_2007 --all

    # Run specific figures from D-Stream (e.g. Figures 5 & 6)
    python run_paper_benchmark.py --paper dstream_2007 --fig 5,6

    # Run Figures 7-10 (Dynamic evolution)
    python run_paper_benchmark.py --paper dstream_2007 --fig 7-10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.experiments.papers import (
    get_paper,
    list_paper_names,
    list_papers,
)


def print_available_papers():
    """Pretty-print all registered paper benchmarks."""
    papers = list_papers()
    print("\n" + "=" * 80)
    print(f"REGISTERED STREAM CLUSTERING PAPER BENCHMARKS ({len(papers)} available):")
    print("=" * 80)

    for p in papers:
        print(f"\n[PAPER ID: {p.name}]")
        print(f"  Title  : {p.title}")
        print(f"  Authors: {p.authors} ({p.venue} {p.year})")
        if p.description:
            print(f"  Summary: {p.description}")
        print("  Available Figures / Experiments:")
        for fig in p.list_figures():
            print(f"    - [{fig.figure_id}] {fig.title}")
            print(f"        {fig.description}")

    print("\n" + "=" * 80)
    print("Run with: python run_paper_benchmark.py --paper <ID> [--quick | --all | --fig <ID>]")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Research Paper Stream Clustering Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_paper_benchmark.py --list
  python run_paper_benchmark.py --paper dstream_2007 --quick
  python run_paper_benchmark.py --paper dstream_2007 --fig 5,6
  python run_paper_benchmark.py --paper dstream_2007 --all
""",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered papers and their available figures/experiments.",
    )
    parser.add_argument(
        "--paper",
        type=str,
        default=None,
        help="Paper identifier to run (e.g. 'dstream_2007'). Run --list to see all available.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all figures in the selected paper.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode (fewer samples for fast validation).",
    )
    parser.add_argument(
        "--fig",
        type=str,
        default=None,
        help="Comma-separated figure IDs to run, e.g. '5,6' or '7-10' or '13'.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Custom directory to save output plots and reports.",
    )

    args = parser.parse_args()

    if args.list:
        print_available_papers()
        return

    available = list_paper_names()
    if not available:
        print("Error: No paper benchmarks registered in benchmark/experiments/papers.")
        sys.exit(1)

    paper_name = args.paper
    if not paper_name:
        if len(available) == 1:
            paper_name = available[0]
            print(f"Defaulting to single registered paper: '{paper_name}'")
        else:
            print("Error: Please specify --paper <name>. Available papers:")
            for name in available:
                print(f"  - {name}")
            sys.exit(1)

    try:
        benchmark = get_paper(paper_name)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    figs = args.fig if args.fig else ("all" if (args.all or not args.fig) else None)
    benchmark.run(figures=figs, quick=args.quick, output_dir=args.out_dir)


if __name__ == "__main__":
    main()
