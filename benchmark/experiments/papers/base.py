"""Base Framework for Stream Clustering Paper Benchmark Reproductions.

Provides:
- `BasePaperBenchmark`: Abstract base class that any paper reproduction inherits from.
- `PaperFigureSpec`: Specification of a runnable figure/experiment from a paper.
- Paper Registry: `register_paper`, `get_paper`, `list_papers` for automatic discovery.
"""

from __future__ import annotations

import abc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Setup paths to ensure river and benchmark imports work
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RIVER_ROOT = PROJECT_ROOT.parent / "river"

for p in (PROJECT_ROOT, RIVER_ROOT):
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))


@dataclass
class PaperFigureSpec:
    """Specification of an experiment / figure in a paper reproduction."""

    figure_id: str
    title: str
    description: str
    runner: Callable[..., Any]


@dataclass
class BasePaperBenchmark(abc.ABC):
    """Abstract base class for reproducing any stream clustering research paper.

    Subclasses implement `setup()` to register their figures and experiments.
    Adding a new paper reproduction is as simple as creating a subclass and
    registering it with `@register_paper`.
    """

    name: str
    title: str
    authors: str
    year: int
    venue: str
    description: str = ""

    figures: dict[str, PaperFigureSpec] = field(default_factory=dict)

    def register_figure(
        self,
        figure_id: str,
        title: str,
        description: str,
        runner: Callable[..., Any],
    ) -> None:
        """Register a runnable figure / experiment."""
        self.figures[figure_id] = PaperFigureSpec(
            figure_id=figure_id,
            title=title,
            description=description,
            runner=runner,
        )

    def get_output_dir(self, custom_dir: Path | str | None = None) -> Path:
        """Get output directory for generated artifacts and plots."""
        if custom_dir:
            out = Path(custom_dir)
        else:
            out = PROJECT_ROOT / "benchmark_out" / "papers" / self.name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def list_figures(self) -> list[PaperFigureSpec]:
        """Return all registered figures in order."""
        return list(self.figures.values())

    @abc.abstractmethod
    def setup(self) -> None:
        """Register figures and experiment configurations in this method."""
        pass

    def run(
        self,
        figures: list[str] | str | None = None,
        quick: bool = False,
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Execute selected figures or all figures in this benchmark."""
        out_dir = self.get_output_dir(output_dir)

        # Parse selected figures
        if figures is None or figures == "all":
            selected_ids = list(self.figures.keys())
        elif isinstance(figures, str):
            selected_ids = [f.strip() for f in figures.split(",") if f.strip()]
        else:
            selected_ids = list(figures)

        print("\n" + "=" * 75)
        print(f"RUNNING PAPER BENCHMARK: {self.name.upper()}")
        print(f"Title: {self.title} ({self.authors}, {self.venue} {self.year})")
        print(f"Selected Figures: {selected_ids} | Quick Mode: {quick}")
        print(f"Output Directory: {out_dir.resolve()}")
        print("=" * 75)

        results = {}
        t_total_start = time.perf_counter()

        for fig_id in selected_ids:
            spec = self.figures.get(fig_id)
            if spec is None:
                # Match partial, e.g. "5" in "5,6"
                matched = [s for s in self.figures.values() if fig_id in s.figure_id]
                if matched:
                    spec = matched[0]

            if spec is None:
                print(f"Warning: Figure '{fig_id}' not found in paper '{self.name}'. Skipping.")
                continue

            print(f"\n>>> Running {spec.title}...")
            t0 = time.perf_counter()
            try:
                res = spec.runner(quick=quick, output_dir=out_dir)
                elapsed = time.perf_counter() - t0
                print(f">>> Completed {spec.title} in {elapsed:.2f}s")
                results[spec.figure_id] = {"status": "SUCCESS", "time_s": elapsed, "result": res}
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f">>> FAILED {spec.title} in {elapsed:.2f}s: {e}")
                results[spec.figure_id] = {"status": "FAILED", "time_s": elapsed, "error": str(e)}

        total_elapsed = time.perf_counter() - t_total_start
        print("\n" + "=" * 75)
        print(f"BENCHMARK COMPLETED IN {total_elapsed:.2f}s")
        print(f"Generated artifacts in {out_dir.resolve()}:")
        for f in sorted(out_dir.glob("*.png")):
            print(f"  - {f.name}")
        print("=" * 75)

        return results


# ---------------------------------------------------------------------------
# Global Paper Registry
# ---------------------------------------------------------------------------
_PAPERS: dict[str, BasePaperBenchmark] = {}


def register_paper(benchmark_cls_or_instance: type[BasePaperBenchmark] | BasePaperBenchmark) -> Any:
    """Decorator or function to register a paper benchmark."""
    if isinstance(benchmark_cls_or_instance, type):
        inst = benchmark_cls_or_instance()
    else:
        inst = benchmark_cls_or_instance
    _PAPERS[inst.name] = inst
    return benchmark_cls_or_instance


def get_paper(name: str) -> BasePaperBenchmark:
    """Retrieve a registered paper benchmark by name."""
    if name not in _PAPERS:
        avail = ", ".join(list_paper_names())
        raise KeyError(f"Paper '{name}' not found. Available papers: [{avail}]")
    return _PAPERS[name]


def list_paper_names() -> list[str]:
    """List all registered paper benchmark identifiers."""
    return list(_PAPERS.keys())


def list_papers() -> list[BasePaperBenchmark]:
    """List all registered paper benchmark instances."""
    return list(_PAPERS.values())
