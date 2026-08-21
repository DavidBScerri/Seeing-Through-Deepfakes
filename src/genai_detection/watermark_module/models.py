"""
Typed output models for the watermark module.

This module is scheme-specific: it wraps Adobe's `trustmark` library and
speaks only about TrustMark-family watermarks. Detection of other
watermarking schemes (Google DeepMind SynthID, Meta Stable Signature,
proprietary vendor watermarks, generic invisible-watermark schemes, …)
is deliberately NOT the job of this module and must be reported as
`unsupported`, never as `not_detected`.

Design rules the models enforce:

1. A negative TrustMark result means only "no supported TrustMark
   watermark was decoded". It never means "no watermark exists",
   "not AI-generated", or "real".
2. Detector unavailability (library missing, model weights unloadable)
   is reported separately from a genuine "no watermark" finding, so a
   consumer can never mistake one for the other.
3. The payload is only surfaced when TrustMark reports a positive
   detection with a schema version; empty-string or garbage bits from a
   negative decode are never exposed.
4. Every result carries a scope statement so downstream UI cannot
   accidentally rebrand a TrustMark-negative result as
   "watermark-free".

These mirror the split already used by the metadata module's
`ProvenanceResult` / `ProvenanceStatus` (see
``src/genai_detection/metadata_module/models.py``).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Scheme + variant metadata
# ---------------------------------------------------------------------------

#: Human-facing name of the watermarking scheme this module understands.
SCHEME_NAME = "Adobe TrustMark"

#: Model variants exposed by the official `trustmark` Python API
#: (see ``TrustMark.__init__``'s ``model_type`` argument). Q is the
#: library's default and the variant used unless the caller asks for
#: another. P is the higher-visual-quality variant referenced by C2PA
#: interoperability discussions. B and C are documented in the upstream
#: source but are secondary.
#:
#: This tuple is deliberately what the library actually implements —
#: expanding it to advertise variants the library does not support
#: would be lying to the caller.
SUPPORTED_VARIANTS: tuple[str, ...] = ("Q", "P", "B", "C")

#: Default variant chosen when the caller does not specify one — matches
#: the upstream library's own default so `TrustMarkDetector()` mirrors
#: `TrustMark()` behaviour.
DEFAULT_VARIANT = "Q"

#: Fixed scope statement attached to every result so downstream
#: consumers cannot accidentally rebrand a negative TrustMark result as
#: "no watermark present" or "not AI-generated".
SCOPE_STATEMENT = (
    "This result concerns Adobe TrustMark watermarks only. A negative "
    "result means no supported TrustMark watermark was decoded — it does "
    "NOT prove the image is unwatermarked, not AI-generated, or real. "
    "Other watermarking schemes (e.g. Google DeepMind SynthID, Meta "
    "Stable Signature, vendor-proprietary schemes) are not covered."
)


class TrustMarkStatus(str, Enum):
    """
    Outcome of running the TrustMark detector on one image.

    The states deliberately split three failure modes the caller must be
    able to tell apart:

    * ``DETECTED`` / ``NOT_DETECTED`` — the detector ran successfully;
      the answer is "yes, a TrustMark watermark was decoded" or "no,
      none was decoded from this image with the requested variant".
    * ``UNSUPPORTED`` — the input describes a scheme we cannot handle
      (a variant the library does not implement, an unsupported input
      type, or a caller asking about SynthID/etc. through this module).
    * ``DETECTOR_UNAVAILABLE`` — the ``trustmark`` library is not
      installed, or its model weights could not be downloaded/loaded.
      This is NOT "no watermark" — the question was not answered.
    * ``ERROR`` — an unexpected exception was raised while decoding.
    """

    DETECTED = "detected"
    """A TrustMark watermark was decoded and its schema is recognised."""

    NOT_DETECTED = "not_detected"
    """The detector ran but did not decode a supported TrustMark
    watermark from the image. Means only that — see :data:`SCOPE_STATEMENT`."""

    UNSUPPORTED = "unsupported"
    """The requested variant, input type, or scheme is not supported by
    this detector (e.g. asking about SynthID, or requesting a variant
    the upstream library does not implement)."""

    DETECTOR_UNAVAILABLE = "detector_unavailable"
    """The ``trustmark`` library or its model weights could not be
    loaded, so no assertion about the presence of a watermark can be
    made. Not to be confused with :attr:`NOT_DETECTED`."""

    ERROR = "error"
    """The detector raised an unexpected exception while processing the
    image. See :attr:`TrustMarkResult.error_details` for diagnostics."""


class TrustMarkResult(BaseModel):
    """
    Structured output of one TrustMark detection attempt.

    Every field is populated even when the detector is unavailable — a
    caller can render a stable UI card off this schema without special-
    casing the failure modes.
    """

    scheme: str = SCHEME_NAME
    """Name of the watermarking scheme this result speaks about.
    Always :data:`SCHEME_NAME` — the module does not fabricate
    cross-scheme verdicts."""

    supported_variants: list[str] = Field(default_factory=lambda: list(SUPPORTED_VARIANTS))
    """TrustMark variants this detector wraps, in library terms.
    Reflects what ``trustmark.TrustMark`` actually accepts today."""

    variant_used: str | None = None
    """Variant the detector attempted for this call. ``None`` when the
    detector never ran (unavailable, unsupported)."""

    status: TrustMarkStatus = TrustMarkStatus.DETECTOR_UNAVAILABLE
    """Overall outcome — see :class:`TrustMarkStatus`."""

    detected: bool = False
    """Convenience mirror of ``status == DETECTED``. False for every
    non-``DETECTED`` state, including UNAVAILABLE and ERROR — a UI must
    never treat this as "no watermark exists"."""

    schema_version: int | None = None
    """Integer schema/version id returned by TrustMark's decoder
    (BCH_SUPER=0, BCH_5=1, BCH_4=2, BCH_3=3). Only populated on a
    positive detection; ``None`` otherwise."""

    payload: str | None = None
    """Decoded payload string. Only populated on a positive detection —
    never fabricated from a negative decode. Callers deciding whether to
    render this in ordinary UI should treat it as potentially long or
    binary-looking."""

    rationale: str = ""
    """Short human-readable summary suitable for UI display."""

    error_details: str | None = None
    """Extended diagnostic string for research use — populated when
    ``status`` is ``DETECTOR_UNAVAILABLE`` or ``ERROR``. Never
    populated for the DETECTED / NOT_DETECTED / UNSUPPORTED steady
    states."""

    processing_time_seconds: float = 0.0
    """Wall-clock time spent inside the detector (model load excluded
    when the model is already cached). Populated even on failure so
    startup vs steady-state cost stays visible."""

    scope_statement: str = SCOPE_STATEMENT
    """Fixed disclaimer about what this result does and does not
    imply. Attached to every result so downstream UI cannot accidentally
    lose the caveat."""


__all__ = [
    "SCHEME_NAME",
    "SUPPORTED_VARIANTS",
    "DEFAULT_VARIANT",
    "SCOPE_STATEMENT",
    "TrustMarkStatus",
    "TrustMarkResult",
]
