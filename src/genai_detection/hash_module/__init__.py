"""
SHA-256 byte-exact hash module.

Ships two independent pieces:

* Streaming SHA-256 hashing (:mod:`~.sha256_hasher`) — computes
  canonical 64-char lowercase hex digests from bytes or files without
  loading whole files into memory.
* A tiny, text-only registry (:mod:`~.registry`) that lets a
  deployment record digests it recognises, along with a small
  descriptive metadata block. **No image bytes, thumbnails, embeddings
  or perceptual hashes are ever stored.**

Scope discipline
----------------
A SHA-256 digest cannot decide, on its own, whether an image is
AI-generated. It can only identify an exact byte-for-byte match against
a previously registered file. Any recompression, metadata edit, or
single-bit change normally changes the digest. See
:data:`.models.SCOPE_STATEMENT` — every result carries that statement
so downstream UI cannot rebrand a no-match result as
"not AI-generated" or "real".

This module deliberately does not feed into the fusion formula in this
iteration. Adding a fusion weight for it would require separate
evaluation and David's sign-off — see the "Rules" section of the
project's CLAUDE.md.
"""

from .models import (
    DIGEST_LENGTH,
    HashLookupResult,
    HashLookupStatus,
    HashRecord,
    OriginLabel,
    SCHEME_NAME,
    SCOPE_STATEMENT,
    is_valid_sha256_hex,
)
from .registry import (
    DEFAULT_RUNTIME_REGISTRY_PATH,
    EXAMPLE_REGISTRY_PATH,
    HashRegistry,
    HashRegistryConflictError,
    HashRegistryError,
    HashRegistryInvalidError,
    RUNTIME_REGISTRY_ENV,
    invalid_result,
    load_registry,
    unavailable_result,
)
from .sha256_hasher import (
    DEFAULT_CHUNK_SIZE,
    normalise_digest,
    sha256_bytes,
    sha256_file,
)

__all__ = [
    # Scheme + scope
    "SCHEME_NAME",
    "SCOPE_STATEMENT",
    "DIGEST_LENGTH",
    "is_valid_sha256_hex",
    # Models
    "OriginLabel",
    "HashLookupStatus",
    "HashRecord",
    "HashLookupResult",
    # Hashing
    "DEFAULT_CHUNK_SIZE",
    "sha256_bytes",
    "sha256_file",
    "normalise_digest",
    # Registry
    "RUNTIME_REGISTRY_ENV",
    "EXAMPLE_REGISTRY_PATH",
    "DEFAULT_RUNTIME_REGISTRY_PATH",
    "HashRegistry",
    "HashRegistryError",
    "HashRegistryInvalidError",
    "HashRegistryConflictError",
    "load_registry",
    "unavailable_result",
    "invalid_result",
]
