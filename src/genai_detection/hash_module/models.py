"""
Typed output models for the SHA-256 hash module.

The hash module is byte-exact, not perceptual: a SHA-256 digest can only
identify an exact byte-for-byte match with a previously recorded file.
It cannot decide, on its own, whether an image is AI-generated — only
the ``origin_label`` on the matched registry record carries that
meaning, and only when a trusted record was registered ahead of time.

Design rules the models enforce:

1. A no-match result means only "no exact byte-identical file has been
   registered". It never means "not AI-generated", "not manipulated",
   or "real". Any recompression, metadata edit, or single-bit change
   normally alters the digest.
2. Detector-availability failures (missing registry file, unreadable
   registry, malformed JSON, schema violations) are reported separately
   from a genuine "no match" finding, so a consumer can never mistake
   one for the other.
3. A registry record may contain only the digest and small
   descriptive metadata — never image bytes, thumbnails, embeddings,
   or perceptual hashes. See :class:`HashRecord` for the closed schema.
4. Every result carries a scope statement so downstream UI cannot
   accidentally rebrand a no-match result as "not AI-generated".

Mirrors the split already used by
:mod:`src.genai_detection.watermark_module.models` and
:mod:`src.genai_detection.metadata_module.models` (provenance).
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Scheme metadata
# ---------------------------------------------------------------------------

#: Human-facing name of the identification scheme this module implements.
SCHEME_NAME = "SHA-256 byte-exact hash"

#: Fixed scope statement attached to every result so downstream consumers
#: cannot accidentally rebrand a no-match result as "no watermark", "not
#: AI-generated", or "real".
SCOPE_STATEMENT = (
    "This result concerns SHA-256 byte-exact matches only. A digest is a "
    "fingerprint of the exact file bytes — any recompression, metadata "
    "edit, format conversion, or single-bit change normally produces a "
    "different digest. A no-match result therefore means only that no "
    "byte-identical file has been registered; it does NOT prove the image "
    "is real, unmodified, or human-authored. A match's AI-related "
    "interpretation depends entirely on the trusted record it was matched "
    "against — the digest itself carries no such meaning."
)


#: Digest length used everywhere (SHA-256, hex-encoded).
DIGEST_LENGTH = 64

#: Regex matching a canonical SHA-256 digest: exactly 64 lowercase hex chars.
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def is_valid_sha256_hex(digest: str) -> bool:
    """Return True iff ``digest`` is exactly 64 lowercase hex characters.

    The lowercase requirement is a canonical-form choice, not an
    algorithmic one: every entrypoint in this module normalises to lower
    case before comparison so lookup keys never depend on case.
    """
    return isinstance(digest, str) and bool(_SHA256_HEX_RE.match(digest))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OriginLabel(str, Enum):
    """
    Closed vocabulary of origin claims a registry record may carry.

    Deliberately small: this module never re-derives AI-ness from the
    digest, so the label is the only thing that gives a match its
    meaning. Any expansion must be additive — never rename existing
    values, or historical registries stop validating.
    """

    AI_GENERATED = "ai_generated"
    """The registered file was produced by a generative model."""

    AI_MODIFIED = "ai_modified"
    """The registered file was substantially edited by an AI tool."""

    CAMERA_OR_HUMAN = "camera_or_human"
    """The registered file has a documented camera-capture or purely
    human authoring origin."""

    UNKNOWN = "unknown"
    """The registered file's origin was not established when it was
    recorded. Use this instead of guessing."""


class HashLookupStatus(str, Enum):
    """
    Outcome of one hash-lookup attempt.

    The states deliberately split the failure modes the caller must be
    able to tell apart:

    * ``EXACT_MATCH`` — the digest matched a registered record.
    * ``NO_MATCH`` — the registry loaded fine, but the digest is not
      registered. Inconclusive — see :data:`SCOPE_STATEMENT`.
    * ``REGISTRY_UNAVAILABLE`` — no registry file is configured, or the
      configured file does not exist / cannot be read. NOT ``NO_MATCH``.
    * ``INVALID_REGISTRY`` — the file exists but is malformed JSON or
      violates the schema. NOT ``NO_MATCH``.
    * ``ERROR`` — an unexpected exception was raised during lookup.
    """

    EXACT_MATCH = "exact_match"
    NO_MATCH = "no_match"
    REGISTRY_UNAVAILABLE = "registry_unavailable"
    INVALID_REGISTRY = "invalid_registry"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Records + results
# ---------------------------------------------------------------------------


class HashRecord(BaseModel):
    """
    One entry in the hash registry.

    A record contains ONLY a digest and small descriptive metadata —
    never image bytes, thumbnails, embeddings, or perceptual hashes.
    The schema is closed (``extra="forbid"``) so a caller cannot smuggle
    binary payload fields into the registry.
    """

    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(..., description="64-character lowercase hex SHA-256 digest of the registered file's exact bytes.")
    origin_label: OriginLabel = Field(..., description="Closed-vocabulary origin claim for the registered file.")
    provider: str | None = Field(default=None, description="Optional short string naming the provider (e.g. 'openai', 'midjourney').")
    model: str | None = Field(default=None, description="Optional short string naming the model or version.")
    source_reference: str | None = Field(default=None, description="Optional non-sensitive reference (e.g. dataset id, public URL, ticket number). Do NOT paste secrets here.")
    notes: str | None = Field(default=None, description="Optional short free-text notes. No binary data.")

    @field_validator("sha256")
    @classmethod
    def _digest_must_be_canonical_sha256(cls, v: str) -> str:
        if not is_valid_sha256_hex(v):
            raise ValueError(
                f"invalid sha256 digest {v!r}: must be exactly 64 lowercase hex characters"
            )
        return v

    @field_validator("provider", "model", "source_reference", "notes")
    @classmethod
    def _strings_must_be_short(cls, v: str | None) -> str | None:
        # Small guard so a caller cannot try to store base64-encoded
        # image bytes in the "notes" field. Anything genuinely
        # descriptive fits comfortably under this cap.
        if v is None:
            return v
        if len(v) > 512:
            raise ValueError(
                "hash-registry text fields are capped at 512 characters; "
                "the registry stores no image content."
            )
        return v


class HashLookupResult(BaseModel):
    """
    Structured output of one hash lookup.

    Every field is populated even when the registry is unavailable — a
    caller can render a stable UI card off this schema without special-
    casing the failure modes.
    """

    scheme: str = SCHEME_NAME
    sha256: str
    """The digest that was looked up (always the canonical lowercase form)."""

    status: HashLookupStatus
    match: HashRecord | None = None
    """The matched record on ``EXACT_MATCH``; ``None`` for every other status.
    Never fabricated from partial data."""

    registry_available: bool = False
    """True iff a registry file was successfully loaded and validated."""

    registry_path: str | None = None
    """Filesystem path of the registry that was consulted, when known.
    ``None`` when no registry is configured."""

    registry_entry_count: int | None = None
    """Number of validated records in the loaded registry, when known.
    ``None`` when the registry was not loaded."""

    rationale: str = ""
    """Short human-readable summary suitable for UI display."""

    error_details: str | None = None
    """Extended diagnostic string for research use — populated when
    ``status`` is ``REGISTRY_UNAVAILABLE``, ``INVALID_REGISTRY``, or
    ``ERROR``. Never populated for the steady-state EXACT_MATCH / NO_MATCH
    outcomes."""

    scope_statement: str = SCOPE_STATEMENT
    """Fixed disclaimer about what this result does and does not imply.
    Attached to every result so downstream UI cannot lose the caveat."""


__all__ = [
    "SCHEME_NAME",
    "SCOPE_STATEMENT",
    "DIGEST_LENGTH",
    "is_valid_sha256_hex",
    "OriginLabel",
    "HashLookupStatus",
    "HashRecord",
    "HashLookupResult",
]
