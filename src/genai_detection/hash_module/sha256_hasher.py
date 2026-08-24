"""
SHA-256 hashing primitives for the hash module.

Byte-exact, streaming-friendly, and deliberately narrow in scope. This
file computes and validates SHA-256 digests — nothing here decides
whether an image is AI-generated. That interpretation belongs to a
matched :class:`~.models.HashRecord` on the registry, and only when a
trusted record was registered ahead of time. See
:data:`~.models.SCOPE_STATEMENT` for the full caveat.

Design rules the module enforces:

1. Hashing is streamed: large files are read in bounded chunks so the
   host process never loads the whole file into memory. This keeps the
   web app and the CLI usable on very large uploads.
2. Every digest returned by this module is in canonical form —
   64 lowercase hexadecimal characters (see
   :func:`~.models.is_valid_sha256_hex`). Comparison sites elsewhere in
   the module can therefore assume canonical form without re-normalising.
3. Byte-hashing and streamed-file-hashing MUST produce the same digest
   for the same input; the tests pin that invariant with the standard
   SHA-256 vectors.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import DIGEST_LENGTH, is_valid_sha256_hex


# Default read-chunk size when hashing files. 1 MiB is large enough that
# syscall overhead is negligible for realistic uploads yet small enough
# that peak memory stays bounded on the demo laptop.
DEFAULT_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """
    Return the canonical SHA-256 hex digest of ``data``.

    The result is exactly :data:`~.models.DIGEST_LENGTH` (64) lowercase
    hex characters. Raises :class:`TypeError` if ``data`` is not a
    ``bytes``-like object — passing a string here is almost always a
    caller bug (which encoding? which normalisation?), so it is
    refused loudly.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"sha256_bytes expects bytes-like input, got {type(data).__name__}"
        )
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(
    path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """
    Return the canonical SHA-256 hex digest of the file at ``path``,
    reading the file in bounded chunks so peak memory stays low.

    Parameters
    ----------
    path:
        Filesystem path to the file to hash.
    chunk_size:
        Bytes to read per iteration. Must be a positive integer.
        Defaults to :data:`DEFAULT_CHUNK_SIZE` (1 MiB).

    Raises
    ------
    FileNotFoundError:
        The path does not exist.
    IsADirectoryError:
        The path exists but is a directory.
    ValueError:
        ``chunk_size`` is not a positive integer.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size!r}"
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"expected a file, got a directory: {p}")

    hasher = hashlib.sha256()
    # Streamed read — never load the full file into memory. `iter(...,
    # b"")` yields until read() returns an empty bytes object at EOF.
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalise_digest(digest: str) -> str:
    """
    Return the canonical (lowercase) form of a SHA-256 hex digest.

    Raises :class:`ValueError` if the input, after lower-casing, is not
    exactly 64 hex characters. Use this at the boundary of any code that
    accepts a digest from a user, a file, or an environment variable —
    once past this function, every internal comparison can assume
    canonical form.
    """
    if not isinstance(digest, str):
        raise ValueError(
            f"digest must be a string, got {type(digest).__name__}"
        )
    lowered = digest.strip().lower()
    if not is_valid_sha256_hex(lowered):
        raise ValueError(
            f"invalid sha256 digest {digest!r}: must be exactly "
            f"{DIGEST_LENGTH} hexadecimal characters"
        )
    return lowered


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "sha256_bytes",
    "sha256_file",
    "normalise_digest",
]
