"""
Typed output models for the metadata module — kept separate from
``metadata_extraction.py`` so the C2PA provenance validator can consume
the same enums without a circular import.

The models here deliberately split three concerns that the earlier
heuristic scorer conflated:

1. **Raw EXIF/XMP indicators** — surfaced as booleans on
   :class:`FeatureSet` (``has_c2pa_marker``, provider keyword hits, …).
   These are unverified textual/binary matches; they do NOT prove AI
   generation.
2. **Cryptographically validated C2PA provenance** — surfaced as a
   :class:`ProvenanceResult` produced by
   ``provenance_validation.validate_provenance``. Trust and origin come
   from the manifest's cryptographic validation, not from string sniffing.
3. **Substantive truth** — never asserted by this module. Even a valid
   AI-origin manifest only proves the signer said the image was AI; it
   cannot prove the image really is (or is not) AI-generated.

Missing metadata remains inconclusive — never "real". A parser or
validator being unavailable is reported separately from a genuine
"no manifest" finding, so downstream consumers cannot mistake one for
the other.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProvenanceStatus(str, Enum):
    """
    Outcome of running the C2PA validator on an image.

    Distinguishes real absence from tool unavailability so a missing
    validator can never be reported as "no C2PA found".
    """

    VALID = "valid"
    """A C2PA manifest is present and its cryptographic validation passed."""

    INVALID_OR_TAMPERED = "invalid_or_tampered"
    """
    A manifest was found but its cryptographic validation failed —
    either the payload was altered after signing, an assertion hash
    mismatched, or the signature did not verify.
    """

    UNTRUSTED_SIGNER = "untrusted_signer"
    """
    Validation succeeded structurally but the signer's certificate is
    not in the configured trust list. The claim is present and internally
    consistent, but its authority cannot be established.
    """

    ABSENT = "absent"
    """The image was scanned and no C2PA manifest was found."""

    UNSUPPORTED_FORMAT = "unsupported_format"
    """
    The validator ran but the file format is not one it knows how to
    parse for C2PA data. Distinct from ABSENT.
    """

    VALIDATOR_UNAVAILABLE = "validator_unavailable"
    """
    The C2PA validation library is not installed or failed to load, so
    no assertion about the presence or validity of provenance can be
    made. This is NOT "no manifest" and must not be reported as such.
    """

    ERROR = "error"
    """The validator raised an unexpected error while processing the file."""


class OriginClaim(str, Enum):
    """
    High-level interpretation of the origin the C2PA manifest claims.

    Only meaningful when :class:`ProvenanceStatus` is ``VALID`` (and, if
    trust matters, when the signer is trusted).
    """

    AI_GENERATED = "ai_generated"
    """The manifest asserts the asset is AI-generated (e.g. IPTC
    ``trainedAlgorithmicMedia`` / ``compositeSynthetic``)."""

    AI_MODIFIED = "ai_modified"
    """The manifest asserts the asset was algorithmically/AI modified
    (e.g. ``algorithmicallyEnhanced``,
    ``compositeWithTrainedAlgorithmicMedia``)."""

    CAMERA_OR_HUMAN_ORIGIN = "camera_or_human_origin"
    """The manifest asserts a camera capture or human-only edits
    (e.g. ``digitalCapture``, ``negativeFilm``, ``humanEdits``)."""

    UNSPECIFIED = "unspecified"
    """A manifest exists but does not carry a digital-source-type
    assertion that lets us classify the origin."""

    CONFLICTING = "conflicting"
    """Assertions in the manifest disagree about origin — reported so a
    consumer does not silently pick one."""


class ProvenanceResult(BaseModel):
    """
    Structured output of the C2PA provenance-validation step.

    Every field carries the caveat that it is only as trustworthy as the
    signer chain — this model exposes *what the manifest claims*, not
    substantive truth.
    """

    status: ProvenanceStatus = ProvenanceStatus.VALIDATOR_UNAVAILABLE
    """Overall status of the validation attempt."""

    manifest_found: bool = False
    """True iff at least one C2PA manifest was recovered from the asset."""

    validation_passed: bool | None = None
    """
    Cryptographic validation result.

    - ``True`` — manifest present and validation passed.
    - ``False`` — manifest present but validation failed
      (tampered, hash mismatch, bad signature).
    - ``None`` — no manifest, or validator unavailable / errored.
    """

    signer_trusted: bool | None = None
    """
    Whether the signer's certificate is in the configured trust list.
    ``None`` when trust could not be evaluated (no manifest, validator
    unavailable, error). Untrusted valid signatures produce ``False``.
    """

    validation_state: str | None = None
    """Raw validation-state string as reported by the C2PA library
    (e.g. ``"Valid"``, ``"Invalid"``, ``"Trusted"``, ``"Untrusted"``).
    Kept verbatim for debugging."""

    validation_errors: list[str] = Field(default_factory=list)
    """Human-readable validation error/warning codes surfaced by the C2PA
    library. Empty on a clean validation."""

    origin_claim: OriginClaim = OriginClaim.UNSPECIFIED
    """Origin interpretation derived from ``c2pa.actions`` +
    ``digitalSourceType`` assertions in the active manifest."""

    has_ai_generation_assertion: bool = False
    """Explicit AI-generation assertion present
    (e.g. ``trainedAlgorithmicMedia``, ``compositeSynthetic``,
    ``algorithmicMedia``)."""

    has_ai_manipulation_assertion: bool = False
    """Explicit AI-manipulation/enhancement assertion present
    (e.g. ``algorithmicallyEnhanced``,
    ``compositeWithTrainedAlgorithmicMedia``)."""

    claim_generator: str | None = None
    """Descriptive ``claim_generator`` string from the active manifest
    (e.g. ``"Adobe_Photoshop/25.0"``). Software identifier only — the
    presence of a value is NOT evidence of AI generation."""

    software_agents: list[str] = Field(default_factory=list)
    """``createdSoftwareAgent`` / ``softwareAgent`` values collected
    from action assertions. Descriptive provenance only."""

    actions: list[str] = Field(default_factory=list)
    """C2PA action labels found in the active manifest
    (e.g. ``c2pa.created``, ``c2pa.edited``, ``c2pa.opened``)."""

    digital_source_types: list[str] = Field(default_factory=list)
    """Full digital-source-type URIs found in the active manifest,
    preserved for auditability."""

    rationale: str = ""
    """One-line human-readable summary suitable for UI display."""

    raw: dict[str, Any] | None = None
    """
    Small subset of the raw manifest data kept for debugging — only
    included when it is safe to serialise (basic types). ``None`` when
    the validator could not read a manifest.
    """


__all__ = [
    "ProvenanceStatus",
    "OriginClaim",
    "ProvenanceResult",
]
