"""
Compatibility shim package.

The real implementation lives in `src/pipeline/`.

This shim keeps existing imports and CLI invocations working, e.g.:
  - `import pipeline`
  - `python -m pipeline.run_full_pipeline_batch ...`
"""

from __future__ import annotations

from pathlib import Path

# Make submodules (pipeline.*) resolvable from src/pipeline without requiring PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PKG = _REPO_ROOT / "src" / "pipeline"
if _SRC_PKG.is_dir():
    __path__.append(str(_SRC_PKG))  # type: ignore[name-defined]


