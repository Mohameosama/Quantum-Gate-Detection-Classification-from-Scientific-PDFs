"""
Compatibility shim package.

The real implementation lives in `src/predict/`.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PKG = _REPO_ROOT / "src" / "predict"
if _SRC_PKG.is_dir():
    __path__.append(str(_SRC_PKG))  # type: ignore[name-defined]


