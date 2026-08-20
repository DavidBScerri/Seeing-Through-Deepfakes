"""
C2PA provenance validation via the official ``c2pa-python`` library.

This module is intentionally the *only* place in the metadata module
that treats C2PA content as authoritative. Anywhere else (see
``metadata_extraction.build_features``), a C2PA marker in the raw
EXIF/XMP is a heuristic hint only.

Semantics (see ``models.py`` docstrings for the full contract):

* A missing manifest → :attr:`ProvenanceStatus.ABSENT` (never "real").
* The C2PA library not being importable →
  :attr:`ProvenanceStatus.VALIDATOR_UNAVAILABLE` (never confused with
  "no C2PA found").
* A parseable manifest whose cryptographic validation failed →
  :attr:`ProvenanceStatus.INVALID_OR_TAMPERED`.
* A parseable manifest whose signer isn't in the trust list →
  :attr:`ProvenanceStatus.UNTRUSTED_SIGNER`.
* A ``c2pa.created`` action carrying an IPTC AI ``digitalSourceType`` →
  :attr:`OriginClaim.AI_GENERATED` / :attr:`OriginClaim.AI_MODIFIED`.

The library-specific parts are isolated behind a small facade
(:func:`_open_reader`, :func:`_extract_manifest_data`) so the tests can
mock the C2PA calls without pulling the native binary into unit runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import OriginClaim, ProvenanceResult, ProvenanceStatus

# ─── Library import (soft) ─────────────────────────────────────────────────
# The library ships a native binary; importing it can fail on some
# platforms. Any import error must degrade to VALIDATOR_UNAVAILABLE
# instead of raising through the metadata module.

try:
    import c2pa as _c2pa  # type: ignore[import-not-found]
    _C2PA_AVAILABLE = True
    _C2PA_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - environment-dependent
    _c2pa = None  # type: ignore[assignment]
    _C2PA_AVAILABLE = False
    _C2PA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ─── IPTC digital-source-type mapping ──────────────────────────────────────
# The C2PA spec anchors "was this AI?" on the IPTC newscodes URI written
# into a `c2pa.created` (or similar) action's `digitalSourceType`. We
# accept both the full URI and the short suffix used in some manifests.
#
# References:
#   * https://cv.iptc.org/newscodes/digitalsourcetype/
#   * https://c2pa.org/specifications/specifications/2.1/specs/C2PA_Specification.html
#
# Anything not listed here (or an empty digitalSourceType) is treated as
# unspecified — never silently classified.

_IPTC_PREFIX = "http://cv.iptc.org/newscodes/digitalsourcetype/"

_AI_GENERATED_TYPES: set[str] = {
    "trainedAlgorithmicMedia",
    "compositeSynthetic",
    "algorithmicMedia",
    "dataDrivenMedia",
}

_AI_MODIFIED_TYPES: set[str] = {
    "algorithmicallyEnhanced",
    "compositeWithTrainedAlgorithmicMedia",
}

_CAMERA_HUMAN_TYPES: set[str] = {
    "digitalCapture",
    "computationalCapture",
    "negativeFilm",
    "positiveFilm",
    "humanEdits",
    "print",
    "screenCapture",  # arguable; screenshots aren't AI, treat as camera-like
}


def _short_source_type(value: str) -> str:
    """Return the trailing token of an IPTC digitalSourceType URI, or the
    input unchanged if it doesn't carry the standard prefix."""
    if not isinstance(value, str):
        return ""
    if value.startswith(_IPTC_PREFIX):
        return value[len(_IPTC_PREFIX):]
    # Some manifests write a bare short token, or a differently-cased URL.
    if "/" in value:
        return value.rsplit("/", 1)[-1]
    return value


def _classify_source_type(value: str) -> OriginClaim | None:
    short = _short_source_type(value)
    if short in _AI_GENERATED_TYPES:
        return OriginClaim.AI_GENERATED
    if short in _AI_MODIFIED_TYPES:
        return OriginClaim.AI_MODIFIED
    if short in _CAMERA_HUMAN_TYPES:
        return OriginClaim.CAMERA_OR_HUMAN_ORIGIN
    return None


# ─── Manifest walk helpers ─────────────────────────────────────────────────

def _iter_actions(manifest: dict[str, Any]):
    """Yield every ``action`` dict across all c2pa action assertions in a
    manifest (both v1 ``c2pa.actions`` and v2 ``c2pa.actions.v2``)."""

    for assertion in manifest.get("assertions", []) or []:
        label = assertion.get("label", "") if isinstance(assertion, dict) else ""
        if not isinstance(label, str) or "c2pa.actions" not in label:
            continue
        data = assertion.get("data") if isinstance(assertion, dict) else None
        if not isinstance(data, dict):
            continue
        actions = data.get("actions")
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, dict):
                yield action


def _extract_manifest_data(manifest: dict[str, Any]) -> dict[str, Any]:
    """Pull the human-relevant fields out of a single C2PA manifest.

    Returns a plain dict with:
        actions:                list[str]
        digital_source_types:   list[str]  (full URIs, deduped, ordered)
        software_agents:        list[str]
        claim_generator:        str | None
        origin_claim:           OriginClaim
    """

    actions: list[str] = []
    digital_source_types: list[str] = []
    software_agents: list[str] = []
    origins: set[OriginClaim] = set()

    for action in _iter_actions(manifest):
        label = action.get("action")
        if isinstance(label, str) and label:
            actions.append(label)

        dst = action.get("digitalSourceType")
        if isinstance(dst, str) and dst:
            if dst not in digital_source_types:
                digital_source_types.append(dst)
            classified = _classify_source_type(dst)
            if classified is not None:
                origins.add(classified)

        # softwareAgent may appear as a plain string or as a nested
        # ClaimGeneratorInfo-shaped dict ({"name": ..., "version": ...}).
        agent = action.get("softwareAgent")
        if isinstance(agent, str) and agent.strip():
            software_agents.append(agent.strip())
        elif isinstance(agent, dict):
            name = agent.get("name")
            version = agent.get("version")
            if isinstance(name, str) and name.strip():
                software_agents.append(
                    f"{name.strip()} {version.strip()}".strip()
                    if isinstance(version, str) and version.strip()
                    else name.strip()
                )

    # Manifest-level claim_generator / claim_generator_info.
    claim_generator: str | None = None
    raw_gen = manifest.get("claim_generator")
    if isinstance(raw_gen, str) and raw_gen.strip():
        claim_generator = raw_gen.strip()
    else:
        info = manifest.get("claim_generator_info")
        if isinstance(info, list) and info:
            first = info[0]
            if isinstance(first, dict):
                name = first.get("name")
                version = first.get("version")
                if isinstance(name, str) and name.strip():
                    claim_generator = (
                        f"{name.strip()} {version.strip()}".strip()
                        if isinstance(version, str) and version.strip()
                        else name.strip()
                    )

    # Deduplicate software agents while preserving order.
    seen: set[str] = set()
    software_agents = [a for a in software_agents if not (a in seen or seen.add(a))]
    actions = [a for a in actions if a]

    # Aggregate origin. Any AI signal wins; conflicting AI + camera → conflicting.
    if not origins:
        origin = OriginClaim.UNSPECIFIED
    elif len(origins) == 1:
        origin = next(iter(origins))
    else:
        has_ai = OriginClaim.AI_GENERATED in origins or OriginClaim.AI_MODIFIED in origins
        has_camera = OriginClaim.CAMERA_OR_HUMAN_ORIGIN in origins
        if has_ai and has_camera:
            origin = OriginClaim.CONFLICTING
        elif OriginClaim.AI_GENERATED in origins:
            origin = OriginClaim.AI_GENERATED
        elif OriginClaim.AI_MODIFIED in origins:
            origin = OriginClaim.AI_MODIFIED
        else:
            origin = OriginClaim.CAMERA_OR_HUMAN_ORIGIN

    return {
        "actions": actions,
        "digital_source_types": digital_source_types,
        "software_agents": software_agents,
        "claim_generator": claim_generator,
        "origin_claim": origin,
    }


# ─── Validation-state / trust interpretation ───────────────────────────────

def _interpret_validation(state: str | None, errors: list[str]) -> tuple[bool | None, bool | None]:
    """
    Translate the C2PA library's validation-state string into
    (validation_passed, signer_trusted) booleans.

    The library's ``get_validation_state()`` currently returns strings
    like ``"Valid"``, ``"Invalid"``, ``"Trusted"``, ``"Untrusted"``,
    ``"OtherError"``. Both fields are ``None`` when the state cannot be
    interpreted.
    """

    if state is None:
        # No state reported but we did open a manifest — treat as
        # unknown validation. Errors, if any, still surface.
        return (None if not errors else False, None)

    normalised = state.strip().lower()

    if normalised in {"valid", "trusted"}:
        return True, True
    if normalised == "untrusted":
        # Cryptographic validation passed, but the signer's cert is not
        # in the configured trust list.
        return True, False
    if normalised in {"invalid", "othererror", "error"}:
        return False, None

    # Unknown state — surface errors if any, otherwise leave unknown.
    if errors:
        return False, None
    return None, None


def _collect_validation_errors(reader) -> list[str]:
    """Best-effort extraction of validation error/warning codes from a
    ``c2pa.Reader``. Never raises — the reader may not expose the
    method on older lib versions."""

    errors: list[str] = []
    try:
        results = reader.get_validation_results()
    except Exception:
        results = None

    def _harvest(items):
        if not isinstance(items, list):
            return
        for entry in items:
            if isinstance(entry, dict):
                code = entry.get("code") or entry.get("kind")
                explanation = entry.get("explanation") or entry.get("message")
                if code and explanation:
                    errors.append(f"{code}: {explanation}")
                elif code:
                    errors.append(str(code))
                elif explanation:
                    errors.append(str(explanation))

    if isinstance(results, dict):
        # activeManifest / ingredientDeltas shapes both surface a `failure`
        # / `informational` list per manifest — walk defensively.
        for key, value in results.items():
            if isinstance(value, dict):
                for sub in ("failure", "informational", "success"):
                    if sub == "success":
                        continue
                    _harvest(value.get(sub))
            elif isinstance(value, list):
                _harvest(value)

    # Older lib versions expose validation_status directly on the JSON.
    try:
        raw = json.loads(reader.json() or "{}")
    except Exception:
        raw = {}
    _harvest(raw.get("validation_status"))
    return errors


# ─── Public entrypoint ─────────────────────────────────────────────────────

def _open_reader(image_path: Path):
    """
    Thin wrapper around ``c2pa.Reader(str(path))`` so unit tests can
    monkey-patch the reader factory.
    """
    return _c2pa.Reader(str(image_path))  # type: ignore[union-attr]


def _rationale(status: ProvenanceStatus, origin: OriginClaim, signer_trusted: bool | None) -> str:
    if status is ProvenanceStatus.VALIDATOR_UNAVAILABLE:
        return "Validator unavailable — the C2PA library is not installed or failed to load. No conclusion about provenance can be drawn."
    if status is ProvenanceStatus.ABSENT:
        return "No C2PA manifest detected. Absence of provenance is inconclusive — it does not mean the image is authentic."
    if status is ProvenanceStatus.UNSUPPORTED_FORMAT:
        return "The C2PA validator does not support this file format. Provenance could not be evaluated."
    if status is ProvenanceStatus.ERROR:
        return "The C2PA validator raised an unexpected error while reading the file."
    if status is ProvenanceStatus.INVALID_OR_TAMPERED:
        return "A C2PA manifest was found but its cryptographic validation failed — the asset or its manifest appears altered."
    if status is ProvenanceStatus.UNTRUSTED_SIGNER:
        base = "A C2PA manifest was found and its signature verified, but the signer is not in the configured trust list."
        if origin is OriginClaim.AI_GENERATED:
            return base + " The manifest claims AI generation."
        if origin is OriginClaim.AI_MODIFIED:
            return base + " The manifest claims AI modification."
        if origin is OriginClaim.CAMERA_OR_HUMAN_ORIGIN:
            return base + " The manifest claims camera capture or human-only editing."
        return base
    # VALID
    trust_hint = "" if signer_trusted else " (signer not in trust list)"
    if origin is OriginClaim.AI_GENERATED:
        return f"Valid C2PA manifest with an explicit AI-generation claim{trust_hint}."
    if origin is OriginClaim.AI_MODIFIED:
        return f"Valid C2PA manifest with an explicit AI-modification claim{trust_hint}."
    if origin is OriginClaim.CAMERA_OR_HUMAN_ORIGIN:
        return f"Valid C2PA manifest claiming camera capture or human-only editing{trust_hint}."
    if origin is OriginClaim.CONFLICTING:
        return f"Valid C2PA manifest, but its origin assertions conflict{trust_hint}."
    return f"Valid C2PA manifest without an AI-generation claim{trust_hint}."


def validate_provenance(image_path: str | Path) -> ProvenanceResult:
    """
    Run the official C2PA validator against ``image_path``.

    Never raises: every failure mode maps to a :class:`ProvenanceStatus`
    value. Callers should read ``status`` before interpreting the other
    fields.
    """

    path = Path(image_path)

    if not _C2PA_AVAILABLE:
        return ProvenanceResult(
            status=ProvenanceStatus.VALIDATOR_UNAVAILABLE,
            rationale=_rationale(ProvenanceStatus.VALIDATOR_UNAVAILABLE, OriginClaim.UNSPECIFIED, None),
            validation_errors=[_C2PA_IMPORT_ERROR] if _C2PA_IMPORT_ERROR else [],
        )

    if not path.exists():
        return ProvenanceResult(
            status=ProvenanceStatus.ERROR,
            rationale="File not found while attempting provenance validation.",
            validation_errors=[f"file not found: {path}"],
        )

    try:
        reader = _open_reader(path)
    except Exception as exc:
        return _classify_reader_open_error(exc)

    try:
        return _read_provenance(reader)
    finally:
        # c2pa.Reader is a ManagedResource; close it if the context
        # protocol isn't being used.
        close = getattr(reader, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _classify_reader_open_error(exc: Exception) -> ProvenanceResult:
    """Map an exception raised while opening ``c2pa.Reader`` to a
    ``ProvenanceStatus`` — ``ManifestNotFound`` is the common "clean"
    absence case, everything else is escalated."""

    name = type(exc).__name__
    message = str(exc)

    # The library raises typed subclasses of C2paError, e.g.
    # ``_C2paManifestNotFound``. Match by name so tests can raise any
    # exception with that name without importing the native module.
    if "ManifestNotFound" in name or "no JUMBF data" in message.lower():
        return ProvenanceResult(
            status=ProvenanceStatus.ABSENT,
            manifest_found=False,
            rationale=_rationale(ProvenanceStatus.ABSENT, OriginClaim.UNSPECIFIED, None),
        )

    if "NotSupported" in name or "unsupported" in message.lower():
        return ProvenanceResult(
            status=ProvenanceStatus.UNSUPPORTED_FORMAT,
            rationale=_rationale(ProvenanceStatus.UNSUPPORTED_FORMAT, OriginClaim.UNSPECIFIED, None),
            validation_errors=[f"{name}: {message}"],
        )

    if "Verify" in name or "Signature" in name:
        return ProvenanceResult(
            status=ProvenanceStatus.INVALID_OR_TAMPERED,
            manifest_found=True,
            validation_passed=False,
            rationale=_rationale(ProvenanceStatus.INVALID_OR_TAMPERED, OriginClaim.UNSPECIFIED, None),
            validation_errors=[f"{name}: {message}"],
        )

    return ProvenanceResult(
        status=ProvenanceStatus.ERROR,
        rationale=_rationale(ProvenanceStatus.ERROR, OriginClaim.UNSPECIFIED, None),
        validation_errors=[f"{name}: {message}"],
    )


def _read_provenance(reader) -> ProvenanceResult:
    """
    Given an already-opened ``c2pa.Reader``, extract the validation
    state, active-manifest data, and origin classification.

    Split out from :func:`validate_provenance` so tests can call it
    with a fake reader.
    """

    validation_state: str | None = None
    try:
        validation_state = reader.get_validation_state()
    except Exception:
        validation_state = None

    errors = _collect_validation_errors(reader)

    active: dict[str, Any] | None
    try:
        active = reader.get_active_manifest()
        if not isinstance(active, dict):
            active = None
    except Exception:
        active = None

    manifest_found = active is not None

    manifest_data: dict[str, Any]
    if active is not None:
        manifest_data = _extract_manifest_data(active)
    else:
        manifest_data = {
            "actions": [],
            "digital_source_types": [],
            "software_agents": [],
            "claim_generator": None,
            "origin_claim": OriginClaim.UNSPECIFIED,
        }

    validation_passed, signer_trusted = _interpret_validation(validation_state, errors)

    # Status resolution:
    #   errors + manifest ⇒ invalid_or_tampered
    #   validation_state == "Untrusted" ⇒ untrusted_signer
    #   no manifest ⇒ absent (shouldn't reach here — open error caught it,
    #     but the library sometimes returns an empty manifest store on
    #     odd assets)
    #   otherwise valid
    if not manifest_found:
        status = ProvenanceStatus.ABSENT
    elif validation_passed is False:
        status = ProvenanceStatus.INVALID_OR_TAMPERED
    elif signer_trusted is False and validation_passed is True:
        status = ProvenanceStatus.UNTRUSTED_SIGNER
    else:
        status = ProvenanceStatus.VALID

    origin: OriginClaim = manifest_data["origin_claim"]

    has_ai_gen = origin is OriginClaim.AI_GENERATED
    has_ai_mod = origin is OriginClaim.AI_MODIFIED
    # A conflicting origin can still expose the underlying assertions.
    if origin is OriginClaim.CONFLICTING:
        for dst in manifest_data["digital_source_types"]:
            short = _short_source_type(dst)
            if short in _AI_GENERATED_TYPES:
                has_ai_gen = True
            elif short in _AI_MODIFIED_TYPES:
                has_ai_mod = True

    raw: dict[str, Any] | None = None
    if manifest_found:
        raw = {
            "validation_state": validation_state,
            "claim_generator": manifest_data["claim_generator"],
            "actions": manifest_data["actions"],
            "digital_source_types": manifest_data["digital_source_types"],
        }

    return ProvenanceResult(
        status=status,
        manifest_found=manifest_found,
        validation_passed=validation_passed if manifest_found else None,
        signer_trusted=signer_trusted if manifest_found else None,
        validation_state=validation_state,
        validation_errors=errors,
        origin_claim=origin,
        has_ai_generation_assertion=has_ai_gen,
        has_ai_manipulation_assertion=has_ai_mod,
        claim_generator=manifest_data["claim_generator"],
        software_agents=manifest_data["software_agents"],
        actions=manifest_data["actions"],
        digital_source_types=manifest_data["digital_source_types"],
        rationale=_rationale(status, origin, signer_trusted),
        raw=raw,
    )


__all__ = [
    "validate_provenance",
]
