"""
Watermark detection module.

Scheme-specific detectors for machine-readable watermarks named in the
EU AI Act Art. 50 / Draft Code of Practice transparency layers.

Today it ships one detector — :class:`TrustMarkDetector`, wrapping
Adobe's official ``trustmark`` library — plus the typed result models
consumed by the integration pipeline. Additional scheme wrappers
(SynthID, Stable Signature, …) would live alongside as separate
detectors, each surfacing their own scoped result rather than a
"universal watermark" verdict.

Importing this package never triggers a model download and never
requires the ``trustmark`` library to be installed: the heavy import
lives inside :meth:`TrustMarkDetector.analyse` and every failure path
returns a typed :class:`TrustMarkResult` with
``status=DETECTOR_UNAVAILABLE`` rather than raising.
"""

from .models import (
    DEFAULT_VARIANT,
    SCHEME_NAME,
    SCOPE_STATEMENT,
    SUPPORTED_VARIANTS,
    TrustMarkResult,
    TrustMarkStatus,
)
from .trustmark_detector import ImageInput, TrustMarkDetector

__all__ = [
    "SCHEME_NAME",
    "SUPPORTED_VARIANTS",
    "DEFAULT_VARIANT",
    "SCOPE_STATEMENT",
    "TrustMarkStatus",
    "TrustMarkResult",
    "TrustMarkDetector",
    "ImageInput",
]
