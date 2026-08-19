"""
Project source root.

Exports :data:`PROJECT_ROOT` — the absolute path to the repository root —
so callers never have to reason about ``__file__.parents[N]``. The value
is the parent of this package's directory (repo/src/), regardless of how
deeply nested the caller is under ``src/``.
"""

from pathlib import Path

#: Absolute path to the repository root (parent of ``src/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

__all__ = ["PROJECT_ROOT"]
