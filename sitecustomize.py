"""
sitecustomize.py

Python automatically imports `sitecustomize` (if available on sys.path) during
startup. When running commands from the repo root, this file is discoverable
via the current working directory and ensures the `src/` folder is importable.

Goal:
- Keep existing commands working after moving packages into `src/`, e.g.:
  `python -m pipeline.run_full_pipeline_batch ...`
"""

from __future__ import annotations

import sys
from pathlib import Path


def _add_src_to_syspath() -> None:
    """
    Prepend `<repo_root>/src` to sys.path if present.

    This preserves the original import style (pipeline/predict/train/...)
    even after adopting a `src/` layout.
    """
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_add_src_to_syspath()


