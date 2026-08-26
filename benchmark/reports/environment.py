"""Capture the machine + library versions a benchmark ran on.

The evaluation protocol requires runtime and memory numbers to be interpretable,
which means recording where they were produced: CPU, RAM, OS, Python and the
versions of river and its numeric stack. Without this a table of throughputs is
not comparable to anything measured elsewhere.
"""

from __future__ import annotations

import platform
import sys

_LIBRARIES = ("river", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib")


def _version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def capture_environment() -> dict:
    """Return a JSON-serialisable record of the current environment."""
    import os

    env: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        # Recorded because river's KMeans is sensitive to it -- see RIVER_BUGS.md
        # BUG-2. Without a pinned value the same seed gives different clusterings
        # from one process to the next, so a result is only interpretable
        # alongside the hash seed it was produced under.
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random (unpinned)"),
    }

    try:
        import psutil

        env["cpu_count_physical"] = psutil.cpu_count(logical=False)
        env["cpu_count_logical"] = psutil.cpu_count(logical=True)
        env["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:  # pragma: no cover - psutil is optional
        import os

        env["cpu_count_logical"] = os.cpu_count()

    # river is imported from the sibling source tree, so record where from: a
    # local checkout and a pip install can behave differently.
    try:
        import river

        env["river_version"] = getattr(river, "__version__", None)
        env["river_path"] = river.__file__
    except Exception:  # pragma: no cover
        pass

    env["libraries"] = {name: _version(name) for name in _LIBRARIES}
    return env


def format_environment(env: dict) -> str:
    """Render :func:`capture_environment` output as a markdown table."""
    from .tables import format_table

    rows = []
    for key in (
        "os", "platform", "machine", "processor", "cpu_count_physical",
        "cpu_count_logical", "ram_total_gb", "python", "python_implementation",
        "python_hash_seed", "river_version", "river_path",
    ):
        if env.get(key) is not None:
            rows.append([key, env[key]])
    for name, ver in (env.get("libraries") or {}).items():
        if ver is not None:
            rows.append([f"lib:{name}", ver])
    return format_table(["Key", "Value"], rows)
