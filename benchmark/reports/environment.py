"""Capture the machine + library versions a benchmark ran on.

The evaluation protocol requires runtime and memory numbers to be interpretable,
which means recording where they were produced: CPU, RAM, OS, Python and the
versions of river and its numeric stack. Without this a table of throughputs is
not comparable to anything measured elsewhere.
"""

from __future__ import annotations

import pathlib
import platform
import subprocess
import sys

_LIBRARIES = ("river", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib")


def _version(dist: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _git_head(repo: "pathlib.Path") -> str | None:
    """Commit of the river tree the numbers were produced from.

    Once local algorithms are added to the sibling checkout, "baseline river" is
    no longer plain upstream, and a result is only interpretable next to the
    commit it came from.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = head.stdout.strip()
        if dirty.returncode == 0 and dirty.stdout.strip():
            return f"{commit} (dirty)"
        return commit
    except (OSError, subprocess.SubprocessError):
        return None


def _git(repo: "pathlib.Path", *args: str) -> bytes | None:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_provenance(repo: "pathlib.Path", paths: tuple[str, ...] = ()) -> dict:
    """The commit of ``repo`` and a hash of what differs from it under ``paths`` (the whole
    tree when empty): the tracked diff against HEAD and every untracked, non-ignored file.

    A commit alone does not say what produced a number when the tree is dirty, and a
    ``(dirty)`` flag does not tell two dirty trees apart; the hash does. ``changed_files``
    lists the paths behind it, so a reader can see whether a change could have touched the
    results. A directory that is not a repository, or a machine without git, reads
    ``"unknown"``: never guessed.
    """
    repo = pathlib.Path(repo)
    head = _git(repo, "rev-parse", "HEAD")
    if head is None:
        return {"repo": str(repo), "commit": "unknown", "dirty": None, "diff_sha256": None,
                "changed_files": [], "paths": list(paths), "code_sha256": code_hash(repo, paths)}
    import hashlib

    scope = ["--", *paths] if paths else []
    diff = _git(repo, "diff", "HEAD", "--binary", *scope) or b""
    tracked = (_git(repo, "diff", "HEAD", "--name-only", *scope) or b"").decode().split()
    untracked = [p for p in (_git(repo, "ls-files", "--others", "--exclude-standard", "-z", *scope)
                             or b"").decode().split("\0") if p]
    digest = hashlib.sha256(diff)
    for rel in sorted(untracked):
        digest.update(rel.encode() + b"\0")
        try:
            digest.update((repo / rel).read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    dirty = bool(diff or untracked)
    return {
        "repo": str(repo), "commit": head.decode().strip(), "dirty": dirty,
        "diff_sha256": digest.hexdigest() if dirty else None,
        "changed_files": sorted(set(tracked) | set(untracked)), "paths": list(paths),
        "code_sha256": code_hash(repo, paths),
    }


def code_hash(repo: "pathlib.Path", paths: tuple[str, ...] = (), suffix: str = ".py") -> str | None:
    """SHA-256 over the content of every ``suffix`` file under ``paths`` of ``repo`` (the whole
    tree when empty), tracked or untracked but not ignored, in path order.

    What the code *is*, apart from how it was committed: a commit of an unchanged tree, or an
    edit to a document beside the code, leaves it as it was. Outside a git repository the
    files are found by walking ``paths``; None when there is nothing to read.
    """
    import hashlib

    repo = pathlib.Path(repo)
    listed = _git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z", *(["--", *paths] if paths else []))
    if listed is not None:
        files = sorted({p for p in listed.decode().split("\0") if p.endswith(suffix)})
    else:
        roots = [repo / p for p in paths] or [repo]
        files = sorted(str(f.relative_to(repo)) for r in roots if r.exists() for f in r.rglob(f"*{suffix}")
                       if "__pycache__" not in f.parts)
    if not files:
        return None
    digest = hashlib.sha256()
    for rel in files:
        digest.update(rel.encode() + b"\0")
        try:
            digest.update((repo / rel).read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def package_lock() -> dict:
    """Every installed distribution as ``name==version``, sorted, and the hash of that list:
    a lock of the environment as it was, not as a requirements file says it should be."""
    import hashlib
    from importlib.metadata import distributions

    pkgs = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"]},
                  key=str.lower)
    return {"n": len(pkgs), "sha256": hashlib.sha256("\n".join(pkgs).encode()).hexdigest(), "packages": pkgs}


def cpu_model() -> str | None:
    """The CPU's marketing name, which ``platform.processor()`` does not give on Linux
    (it reads ``x86_64`` or nothing): ``/proc/cpuinfo`` there, ``sysctl`` on macOS."""
    system = platform.system()
    try:
        if system == "Linux":
            for line in pathlib.Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                key, _, value = line.partition(":")
                if key.strip() in ("model name", "Model", "Hardware") and value.strip():
                    return value.strip()
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5, check=False)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or None


def ram_total_gb() -> float | None:
    """Physical memory without psutil: ``/proc/meminfo`` on Linux, ``sysctl hw.memsize`` on macOS."""
    try:
        if platform.system() == "Linux":
            for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 2)
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                 timeout=5, check=False)
            if out.returncode == 0 and out.stdout.strip():
                return round(int(out.stdout.strip()) / 1024**3, 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def capture_environment() -> dict:
    """Return a JSON-serialisable record of the current environment."""
    import os

    env: dict = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_model": cpu_model(),
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
        env["ram_total_gb"] = ram_total_gb()

    # river is imported from the sibling source tree, so record where from: a
    # local checkout and a pip install can behave differently.
    try:
        import river

        env["river_version"] = getattr(river, "__version__", None)
        env["river_path"] = river.__file__
        env["river_git_head"] = _git_head(pathlib.Path(river.__file__).parent.parent)
    except Exception:  # pragma: no cover
        pass

    env["libraries"] = {name: _version(name) for name in _LIBRARIES}
    return env


def format_environment(env: dict) -> str:
    """Render :func:`capture_environment` output as a markdown table."""
    from .tables import format_table

    rows = []
    for key in (
        "os", "platform", "machine", "processor", "cpu_model", "cpu_count_physical",
        "cpu_count_logical", "ram_total_gb", "python", "python_implementation",
        "python_hash_seed", "river_version", "river_git_head", "river_path",
    ):
        if env.get(key) is not None:
            rows.append([key, env[key]])
    for name, ver in (env.get("libraries") or {}).items():
        if ver is not None:
            rows.append([f"lib:{name}", ver])
    return format_table(["Key", "Value"], rows)
