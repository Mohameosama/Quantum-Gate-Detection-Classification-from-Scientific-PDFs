"""
config.py

Central configuration for the quantum circuit figure pipeline.

Why this file exists:
- Avoid scattering thresholds across multiple scripts
- Make tuning reproducible and safer
- Keep defaults conservative to reduce false positives

How to use:
- Import values directly:
    from config import CVConfig, OCRConfig, ParserConfig

- Or create config objects and pass them into your classes if you later
  decide to support dependency injection (recommended for long-term maintainability).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------
# CV (computer vision) configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CVConfig:
    """
    Parameters used by CV-based circuit detection.

    Notes:
    - min/max wires are important to avoid plots/histograms being detected as circuits.
    - wire_snap_tolerance controls how strict we are when validating that control points
      lie on a wire.
    """

    # Wire detection
    min_wire_length: int = 100
    min_wires: int = 2
    max_wires: int = 20

    # Gate detection (rectangle contours)
    min_gate_area: int = 200
    max_gate_area: int = 20000

    # Control circle detection
    min_control_radius: int = 3
    max_control_radius: int = 15

    # Control must be near a wire y-level
    wire_snap_tolerance: int = 8

    # CV validity threshold (used inside validation scoring)
    min_valid_confidence: float = 55.0


# ---------------------------------------------------------------------
# OCR configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class OCRConfig:
    """
    OCR configuration for gate-label detection.

    Notes:
    - OCR quality is sensitive to preprocessing, scaling, and Tesseract psm/oem.
    """

    # Scale factor used before OCR (2 is a common good compromise)
    scale: int = 2

    # Tesseract settings
    tesseract_oem: int = 3
    tesseract_psm: int = 6

    # Context length limit passed to NLP-ish heuristics
    context_limit: int = 2000


# ---------------------------------------------------------------------
# Parser-level configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ParserConfig:
    """
    High-level parser thresholds and behavior.

    Notes:
    - cv_min_confidence is the early exit threshold: if CV is too low, don't do OCR.
    - confidence boosts are intentionally small to avoid false certainty.
    """

    # Early exit if CV confidence is below this
    cv_min_confidence: float = 30.0

    # Confidence boosts (small and interpretable)
    boost_if_gates_detected: float = 10.0
    boost_if_patterns_detected: float = 10.0

    # Semantic validation limits
    max_controls_without_strong_evidence: int = 60


# ---------------------------------------------------------------------
# Output / processing defaults
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessingConfig:
    """
    Defaults used by scripts that extract/process images.
    """

    # Skip very small images (logos/icons)
    min_extracted_image_size: int = 50

    # Default temp folder for extracted images
    default_output_dir: str = "temp_images"


# ---------------------------------------------------------------------
# Default config instances (import these directly)
# ---------------------------------------------------------------------

CV_DEFAULT = CVConfig()
OCR_DEFAULT = OCRConfig()
PARSER_DEFAULT = ParserConfig()
PROCESSING_DEFAULT = ProcessingConfig()

