"""
Cross-module evaluation and robustness experiments.

Currently ships :mod:`robustness_experiments` — a small harness that
compares how the three transparency signals implemented in this project
(C2PA provenance, Adobe TrustMark, SHA-256 byte-exact) behave under a
handful of common image transformations. It answers "how does each
scheme degrade" for the schemes we actually support; it deliberately
says nothing about SynthID or other schemes without a detector.

Per-module evaluation notebooks continue to live alongside their
modules (``metadata_extraction_eval.ipynb`` etc.). This package hosts
cross-module reasoning: shared transformations, shared metric
definitions, and one CLI that runs the whole comparison end-to-end.
"""

from .transformations import (
    SMOKE_TRANSFORMATIONS,
    TRANSFORMATIONS,
    Transformation,
    transformations_config,
)
from .metrics import (
    RESULT_COLUMNS,
    classify_outcome,
    row_coverage,
    summarise,
)

__all__ = [
    "TRANSFORMATIONS",
    "SMOKE_TRANSFORMATIONS",
    "Transformation",
    "transformations_config",
    "RESULT_COLUMNS",
    "classify_outcome",
    "row_coverage",
    "summarise",
]
