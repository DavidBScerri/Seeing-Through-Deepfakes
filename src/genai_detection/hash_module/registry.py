"""
SHA-256 hash registry — a tiny, text-only lookup table.

A registry is a JSON document holding one flat list of
:class:`~.models.HashRecord` entries. Each record is nothing more than a
64-character SHA-256 digest and a small descriptive metadata block —
never image bytes, thumbnails, embeddings, or perceptual hashes. The
whole point of that closed schema is that a live registry stays a *tiny
text file* even if it grows to thousands of entries; it must not become
a dataset or consume significant storage. See
:class:`~.models.HashRecord` and the CLAUDE.md "hash_module" rules.

Design rules enforced here:

* Loads validate every record. On a parse or schema failure the loader
  returns a structured error rather than raising — the caller can
  distinguish "no registry configured" from "registry present but
  malformed", which is what the UI needs to render an honest card.
* Writes are atomic: we serialise to a sibling ``*.tmp`` file, fsync,
  then ``os.replace`` on top of the target. A crash mid-write leaves
  the previous valid registry untouched.
* We never overwrite a valid on-disk registry after a validation
  failure. Callers must first fix the file (or start from a fresh path)
  before a write can succeed.
* Duplicates for the same digest are rejected on load and on register.
  Two records with the same digest but conflicting labels/providers are
  a data-quality bug — silently keeping either one would corrupt the
  meaning of an ``EXACT_MATCH`` result.
* The runtime registry path comes from an environment variable
  (:data:`RUNTIME_REGISTRY_ENV`) or an explicit constructor argument;
  the committed example lives at :data:`EXAMPLE_REGISTRY_PATH` and is
  only used as a schema reference, never as a live registry.

Exact-match semantics are deliberately narrow: a match tells you a
byte-identical file was previously registered with a given label — it
does NOT infer AI-ness on its own. See :data:`~.models.SCOPE_STATEMENT`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from src import PROJECT_ROOT

from .models import (
    HashLookupResult,
    HashLookupStatus,
    HashRecord,
    SCHEME_NAME,
    SCOPE_STATEMENT,
)
from .sha256_hasher import normalise_digest


#: Environment variable a deployment can set to point at its runtime
#: registry file. Empty / unset means "no runtime registry configured"
#: and the hash module reports ``REGISTRY_UNAVAILABLE`` for every lookup.
RUNTIME_REGISTRY_ENV = "SEEING_THROUGH_DEEPFAKES_HASH_REGISTRY"

#: Committed example registry — schema reference only. Kept tiny (one
#: illustrative record) so it can safely be shipped in the repo.
EXAMPLE_REGISTRY_PATH: Path = (
    Path(__file__).resolve().parent / "registry.example.json"
)

#: Default runtime location when neither the env var nor an explicit
#: path is provided. Lives under the repo-level ``data/`` directory
#: (already gitignored — see ``.gitignore``), so a user's registry is
#: never accidentally committed even if they forget to set the env var.
DEFAULT_RUNTIME_REGISTRY_PATH: Path = (
    PROJECT_ROOT / "data" / "hash_registry" / "registry.json"
)


# Small sanity cap so a malformed / attacker-supplied "registry" file
# cannot be a 500 MB blob quietly loaded into memory. Real registries
# are tiny text and comfortably fit inside this cap; anything larger is
# almost certainly not a hash registry.
_MAX_REGISTRY_BYTES = 10 * 1024 * 1024  # 10 MiB


@dataclass(frozen=True)
class _LoadedRegistry:
    """Internal wrapper around a validated in-memory registry."""

    records: dict[str, HashRecord]  # digest -> record (canonical digest key)
    path: Path


def _resolve_path(explicit: str | Path | None = None) -> Path | None:
    """
    Pick the registry path with an explicit override > env var > default
    precedence. Returns ``None`` only when no source has provided a
    non-empty value AND the default cannot be used (which never happens
    for the current default, but stays a possibility for future changes).
    """
    if explicit is not None:
        return Path(explicit)
    env_value = os.environ.get(RUNTIME_REGISTRY_ENV, "").strip()
    if env_value:
        return Path(env_value)
    return DEFAULT_RUNTIME_REGISTRY_PATH


class HashRegistryError(RuntimeError):
    """Base class for expected registry failures the loader converts to
    structured statuses. Kept lightweight — the real reporting channel
    is :class:`HashLookupResult`, not raised exceptions."""


class HashRegistryInvalidError(HashRegistryError):
    """Registry file exists but is malformed or violates the schema."""


class HashRegistryConflictError(HashRegistryError):
    """A digest already exists in the registry with a different record."""


def _read_registry_file(path: Path) -> list[dict]:
    """
    Read and JSON-decode the registry file.

    Raises :class:`FileNotFoundError` when the file does not exist,
    :class:`HashRegistryInvalidError` on any other structural problem
    (too large, not JSON, not a list of objects). Individual record
    validation happens later in :func:`_records_from_raw`.
    """
    if not path.exists():
        raise FileNotFoundError(f"registry file not found: {path}")
    if path.is_dir():
        raise HashRegistryInvalidError(
            f"registry path is a directory, expected a JSON file: {path}"
        )
    size = path.stat().st_size
    if size > _MAX_REGISTRY_BYTES:
        raise HashRegistryInvalidError(
            f"registry file is {size} bytes — refusing to load anything "
            f"over {_MAX_REGISTRY_BYTES} bytes. Real registries are tiny "
            f"text; this is almost certainly not a hash registry."
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HashRegistryInvalidError(
            f"registry file unreadable ({type(exc).__name__}): {exc}"
        ) from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HashRegistryInvalidError(
            f"registry file is not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"
        ) from exc

    # Accept either a bare list (canonical form) or a small envelope
    # object with a "records" key — the example file uses the envelope
    # so it can carry a human-readable note. Anything else is rejected.
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and isinstance(raw.get("records"), list):
        records = raw["records"]
    else:
        raise HashRegistryInvalidError(
            "registry root must be a JSON list of records, or an object "
            "with a 'records' list."
        )

    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise HashRegistryInvalidError(
                f"registry record at index {i} is not a JSON object."
            )
    return records


def _records_from_raw(raw_records: list[dict]) -> dict[str, HashRecord]:
    """
    Validate every raw record and index them by their canonical digest.

    Enforces uniqueness: a digest may appear at most once. Two records
    for the same digest are rejected on load — see the class-level rule
    on "conflicting duplicate records" in the docstring.
    """
    out: dict[str, HashRecord] = {}
    for i, raw in enumerate(raw_records):
        try:
            record = HashRecord.model_validate(raw)
        except ValidationError as exc:
            raise HashRegistryInvalidError(
                f"registry record at index {i} failed schema validation: {exc.errors()}"
            ) from exc
        key = record.sha256  # already lowercase (see HashRecord validator)
        existing = out.get(key)
        if existing is not None:
            # Same digest twice. Even identical duplicates are refused —
            # if the source of truth allows two rows for one digest, the
            # source is wrong.
            raise HashRegistryInvalidError(
                f"duplicate registry entry for digest {key}: index {i} "
                f"conflicts with an earlier record."
            )
        out[key] = record
    return out


def _load_registry_from_path(path: Path) -> _LoadedRegistry:
    """Load, JSON-decode and validate the registry at ``path``."""
    raw = _read_registry_file(path)
    records = _records_from_raw(raw)
    return _LoadedRegistry(records=records, path=path)


def _atomic_write_registry(path: Path, records: dict[str, HashRecord]) -> None:
    """
    Write ``records`` to ``path`` atomically.

    Uses a temporary file in the target directory (so ``os.replace`` is
    a same-filesystem move) and fsync's before the replace to make the
    swap survive a power failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "scheme": SCHEME_NAME,
        "notes": (
            "Text-only hash registry. Each entry is a SHA-256 digest and "
            "small descriptive metadata — no image bytes, thumbnails, "
            "embeddings, or perceptual hashes are stored."
        ),
        "records": [
            r.model_dump(exclude_none=True) for r in records.values()
        ],
    }
    payload = json.dumps(envelope, indent=2, sort_keys=True).encode("utf-8")

    # tempfile.NamedTemporaryFile in the target directory guarantees
    # os.replace stays on the same filesystem.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=".registry.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Never leave a half-written .tmp behind on failure.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class HashRegistry:
    """
    Loaded view of a hash registry.

    The registry is loaded eagerly in the constructor so schema errors
    surface immediately — the alternative (lazy on first lookup) would
    let a website start up cleanly then fail on the first upload, which
    is the wrong error location.

    Parameters
    ----------
    path:
        Explicit registry path override. When ``None`` the loader falls
        back to the :data:`RUNTIME_REGISTRY_ENV` environment variable,
        then to :data:`DEFAULT_RUNTIME_REGISTRY_PATH`. See
        :func:`load_registry` for a factory that returns a typed error
        object instead of raising.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        resolved = _resolve_path(path)
        if resolved is None:
            raise HashRegistryError("no registry path could be resolved")
        loaded = _load_registry_from_path(resolved)
        self._records: dict[str, HashRecord] = dict(loaded.records)
        self.path: Path = loaded.path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, digest: str) -> bool:
        try:
            key = normalise_digest(digest)
        except ValueError:
            return False
        return key in self._records

    def get(self, digest: str) -> HashRecord | None:
        """Return the record for ``digest`` or ``None`` if not present.

        Raises :class:`ValueError` on a malformed digest — that's a
        caller bug, not a "no match" outcome, and must not be silently
        conflated with one.
        """
        return self._records.get(normalise_digest(digest))

    def all_records(self) -> list[HashRecord]:
        """Return a copy of every stored record (order not guaranteed)."""
        return list(self._records.values())

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def register(self, record: HashRecord, *, allow_replace: bool = False) -> None:
        """
        Add ``record`` to the registry and atomically persist it to
        disk.

        By default, a second call for the same digest raises
        :class:`HashRegistryConflictError` unless the new record is
        bit-for-bit identical to the stored one (idempotent re-register
        is allowed). Pass ``allow_replace=True`` to overwrite an
        existing record intentionally — the CLI does not expose this,
        so callers must reach for it deliberately.
        """
        key = record.sha256  # already canonical
        existing = self._records.get(key)
        if existing is not None:
            if existing == record:
                return  # Idempotent no-op — same digest, same metadata.
            if not allow_replace:
                raise HashRegistryConflictError(
                    f"digest {key} is already registered with a different "
                    f"record; refusing to overwrite. Pass allow_replace=True "
                    f"to intentionally replace."
                )

        new_records = dict(self._records)
        new_records[key] = record
        _atomic_write_registry(self.path, new_records)
        self._records = new_records

    def remove(self, digest: str) -> bool:
        """
        Remove the record for ``digest`` if present. Returns True when a
        record was removed, False when the digest was not registered.
        """
        key = normalise_digest(digest)
        if key not in self._records:
            return False
        new_records = dict(self._records)
        del new_records[key]
        _atomic_write_registry(self.path, new_records)
        self._records = new_records
        return True

    # ------------------------------------------------------------------
    # Structured lookup — the shape the web app consumes
    # ------------------------------------------------------------------

    def lookup(self, digest: str) -> HashLookupResult:
        """
        Return a :class:`HashLookupResult` describing whether ``digest``
        matches a registered record.

        Always returns; caller-side errors (malformed digest) surface as
        ``status=ERROR``. Registry availability is baked in via
        ``registry_available=True`` here — a caller with no registry at
        all should build the failure result via
        :func:`unavailable_result` / :func:`invalid_result` instead.
        """
        try:
            key = normalise_digest(digest)
        except ValueError as exc:
            return HashLookupResult(
                scheme=SCHEME_NAME,
                sha256=str(digest),
                status=HashLookupStatus.ERROR,
                match=None,
                registry_available=True,
                registry_path=str(self.path),
                registry_entry_count=len(self._records),
                rationale="Malformed SHA-256 digest supplied to lookup.",
                error_details=str(exc),
                scope_statement=SCOPE_STATEMENT,
            )

        match = self._records.get(key)
        if match is not None:
            return HashLookupResult(
                scheme=SCHEME_NAME,
                sha256=key,
                status=HashLookupStatus.EXACT_MATCH,
                match=match,
                registry_available=True,
                registry_path=str(self.path),
                registry_entry_count=len(self._records),
                rationale=(
                    f"Exact byte-for-byte SHA-256 match against a registered "
                    f"record with origin label {match.origin_label.value!r}. "
                    "This match is only as trustworthy as the record it was "
                    "matched against."
                ),
                error_details=None,
                scope_statement=SCOPE_STATEMENT,
            )

        return HashLookupResult(
            scheme=SCHEME_NAME,
            sha256=key,
            status=HashLookupStatus.NO_MATCH,
            match=None,
            registry_available=True,
            registry_path=str(self.path),
            registry_entry_count=len(self._records),
            rationale=(
                "No byte-identical file has been registered. This is "
                "INCONCLUSIVE — recompression, metadata edits or any byte "
                "alteration would produce a different digest, and most "
                "images have never been registered in the first place."
            ),
            error_details=None,
            scope_statement=SCOPE_STATEMENT,
        )


# ---------------------------------------------------------------------------
# Factory + failure-mode helpers
# ---------------------------------------------------------------------------


def load_registry(path: str | Path | None = None) -> HashRegistry | HashLookupResult:
    """
    Attempt to load the registry, returning either a live
    :class:`HashRegistry` or a :class:`HashLookupResult` describing the
    failure mode.

    This is the entry point the web app / bulk pipelines should use —
    it never raises for the "no registry configured" or "malformed
    registry" cases, which are the two expected steady-state failures.
    The returned failure result carries a placeholder empty digest so
    the caller only has to fill in :attr:`~HashLookupResult.sha256`
    before returning it.
    """
    resolved = _resolve_path(path)
    if resolved is None:
        return unavailable_result(
            path=None,
            reason=(
                f"No registry path was provided and the "
                f"{RUNTIME_REGISTRY_ENV} environment variable is unset."
            ),
        )
    try:
        return HashRegistry(resolved)
    except FileNotFoundError:
        return unavailable_result(
            path=resolved,
            reason=f"Registry file does not exist at {resolved}.",
        )
    except HashRegistryInvalidError as exc:
        return invalid_result(path=resolved, reason=str(exc))
    except HashRegistryError as exc:  # pragma: no cover - defensive
        return unavailable_result(path=resolved, reason=str(exc))


def unavailable_result(
    *,
    path: Path | None,
    reason: str,
    digest: str = "",
) -> HashLookupResult:
    """
    Build the canonical ``REGISTRY_UNAVAILABLE`` result.

    ``digest`` is optional — leave it blank when the caller has not yet
    computed one (e.g. reporting failure at server startup). The rest
    of the shape stays identical to a successful lookup result so the
    web UI can render one card template unconditionally.
    """
    return HashLookupResult(
        scheme=SCHEME_NAME,
        sha256=digest,
        status=HashLookupStatus.REGISTRY_UNAVAILABLE,
        match=None,
        registry_available=False,
        registry_path=str(path) if path is not None else None,
        registry_entry_count=None,
        rationale=(
            "Hash registry is unavailable — no exact-match check could be "
            "performed. This is NOT the same as 'no match'; nothing was "
            "compared."
        ),
        error_details=reason,
        scope_statement=SCOPE_STATEMENT,
    )


def invalid_result(
    *,
    path: Path,
    reason: str,
    digest: str = "",
) -> HashLookupResult:
    """Build the canonical ``INVALID_REGISTRY`` result."""
    return HashLookupResult(
        scheme=SCHEME_NAME,
        sha256=digest,
        status=HashLookupStatus.INVALID_REGISTRY,
        match=None,
        registry_available=False,
        registry_path=str(path),
        registry_entry_count=None,
        rationale=(
            "Hash registry file was found but could not be validated — "
            "no exact-match check was performed. This is NOT the same as "
            "'no match'."
        ),
        error_details=reason,
        scope_statement=SCOPE_STATEMENT,
    )


__all__ = [
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
