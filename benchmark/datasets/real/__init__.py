"""Real-world dataset loaders (KDDCup99, KDDCup98)."""

from __future__ import annotations

from .kdd import load_kddcup98, load_kddcup99

__all__ = ["load_kddcup99", "load_kddcup98"]
