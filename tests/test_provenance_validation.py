"""
Tests for the C2PA provenance validator
(src/genai_detection/metadata_module/provenance_validation.py).

The strategy is to mock ``c2pa.Reader`` (or the module-level
``_open_reader`` factory) so every ProvenanceStatus path is exercised
without needing the native C2PA binary or signed sample assets. This
keeps the suite deterministic and fast.

Optional integration test: if the environment provides an official,
small, legally-reusable C2PA sample image via
``C2PA_TEST_ASSET_PATH``, one end-to-end validation is run against
the real library. The test is skipped when the variable is unset or
the file is missing — no large binary is checked in.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.genai_detection.metadata_module import (
    OriginClaim,
    ProvenanceResult,
    ProvenanceStatus,
    validate_provenance,
)
from src.genai_detection.metadata_module import provenance_validation as pv


# ─── Fake exception classes ────────────────────────────────────────────────
# The real library raises typed subclasses of ``C2paError`` whose names
# encode the failure mode. The validator matches by NAME, so bare
# stand-ins work here without importing the native module.

class _FakeManifestNotFound(Exception):
    pass
_FakeManifestNotFound.__name__ = "_C2paManifestNotFound"


class _FakeNotSupported(Exception):
    pass
_FakeNotSupported.__name__ = "_C2paNotSupported"


class _FakeSignatureError(Exception):
    pass
_FakeSignatureError.__name__ = "_C2paSignature"


# ─── Fake Reader ────────────────────────────────────────────────────────────

class _FakeReader:
    """Minimal duck-type for c2pa.Reader — only the methods the
    validator calls need to exist."""

    def __init__(self, *, active=None, state=None, results=None, raw_json=None):
        self._active = active
        self._state = state
        self._results = results or {}
        self._raw_json = raw_json or "{}"
        self.closed = False

    def get_active_manifest(self):
        return self._active

    def get_validation_state(self):
        return self._state

    def get_validation_results(self):
        return self._results

    def json(self):
        return self._raw_json

    def close(self):
        self.closed = True


@pytest.fixture
def fake_image(tmp_path: Path) -> Path:
    """A trivial byte-file — the validator only needs the path to exist,
    the reader factory is mocked."""
    path = tmp_path / "asset.png"
    path.write_bytes(b"not really a png")
    return path


# ─── Status coverage ───────────────────────────────────────────────────────

class TestValidatorUnavailable:
    def test_returns_validator_unavailable_when_lib_missing(self, fake_image, monkeypatch):
        # Simulate the library not being importable.
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", False)
        monkeypatch.setattr(pv, "_C2PA_IMPORT_ERROR", "ImportError: no module named c2pa")
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALIDATOR_UNAVAILABLE
        assert result.manifest_found is False
        # Rationale must NOT read as "no C2PA found" — that's a separate
        # status (ABSENT).
        assert "not installed" in result.rationale.lower() or "unavailable" in result.rationale.lower()
        assert "no manifest" not in result.rationale.lower()

    def test_unavailable_is_distinct_from_absent(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", False)
        monkeypatch.setattr(pv, "_C2PA_IMPORT_ERROR", None)
        unavailable = validate_provenance(fake_image)

        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        monkeypatch.setattr(pv, "_open_reader", lambda p: (_ for _ in ()).throw(_FakeManifestNotFound("no JUMBF data found")))
        absent = validate_provenance(fake_image)

        assert unavailable.status is ProvenanceStatus.VALIDATOR_UNAVAILABLE
        assert absent.status is ProvenanceStatus.ABSENT
        assert unavailable.status is not absent.status


class TestAbsent:
    def test_manifest_not_found_becomes_absent(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        monkeypatch.setattr(
            pv, "_open_reader",
            lambda p: (_ for _ in ()).throw(_FakeManifestNotFound("no JUMBF data found")),
        )
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.ABSENT
        assert result.manifest_found is False
        # Absence must never READ AS an assertion of authenticity —
        # the word "authentic" may appear inside "does not mean the
        # image is authentic", which is fine, but a bare "authentic"
        # is not.
        r = result.rationale.lower()
        assert "does not mean" in r or "inconclusive" in r
        assert "likely real" not in r


class TestUnsupportedFormat:
    def test_not_supported_error_maps_to_unsupported_format(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        monkeypatch.setattr(
            pv, "_open_reader",
            lambda p: (_ for _ in ()).throw(_FakeNotSupported("format not supported")),
        )
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.UNSUPPORTED_FORMAT


class TestInvalidOrTampered:
    def test_signature_error_at_open_time(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        monkeypatch.setattr(
            pv, "_open_reader",
            lambda p: (_ for _ in ()).throw(_FakeSignatureError("signature verification failed")),
        )
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.INVALID_OR_TAMPERED
        assert result.validation_passed is False

    def test_invalid_state_after_open(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "claim_generator": "Adobe_Photoshop/25.0",
            "assertions": [],
        }
        reader = _FakeReader(active=active, state="Invalid")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.INVALID_OR_TAMPERED
        assert result.manifest_found is True
        assert result.validation_passed is False


class TestUntrustedSigner:
    def test_untrusted_state_maps_to_untrusted_signer(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "claim_generator": "Some Tool/1.0",
            "assertions": [
                {
                    "label": "c2pa.actions.v2",
                    "data": {"actions": [
                        {"action": "c2pa.created",
                         "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"},
                    ]},
                },
            ],
        }
        reader = _FakeReader(active=active, state="Untrusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.UNTRUSTED_SIGNER
        assert result.signer_trusted is False
        assert result.validation_passed is True


class TestValidAiOrigin:
    def test_ai_generated_origin_detected(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "claim_generator": "OpenAI/dall-e-3",
            "claim_generator_info": [{"name": "OpenAI", "version": "dall-e-3"}],
            "assertions": [
                {
                    "label": "c2pa.actions.v2",
                    "data": {"actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                            "softwareAgent": {"name": "OpenAI", "version": "dall-e-3"},
                        },
                    ]},
                },
            ],
        }
        reader = _FakeReader(active=active, state="Valid")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.AI_GENERATED
        assert result.has_ai_generation_assertion is True
        assert result.has_ai_manipulation_assertion is False
        # Manifest-level ``claim_generator`` is a bare string — we keep
        # it verbatim rather than reformatting.
        assert result.claim_generator == "OpenAI/dall-e-3"
        assert "OpenAI dall-e-3" in result.software_agents
        assert "c2pa.created" in result.actions
        assert "trainedAlgorithmicMedia" in result.digital_source_types[0]
        assert "ai-generation" in result.rationale.lower()

    def test_ai_modified_origin_detected(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {"actions": [
                        {"action": "c2pa.color_adjustments",
                         "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced"},
                    ]},
                },
            ],
        }
        reader = _FakeReader(active=active, state="Valid")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.AI_MODIFIED
        assert result.has_ai_manipulation_assertion is True
        assert "ai-modification" in result.rationale.lower()


class TestValidNonAiOrigin:
    def test_camera_capture_is_not_an_ai_claim(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "claim_generator": "Sony_A7R/1.0",
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {"actions": [
                        {"action": "c2pa.created",
                         "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"},
                    ]},
                },
            ],
        }
        reader = _FakeReader(active=active, state="Valid")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.CAMERA_OR_HUMAN_ORIGIN
        assert result.has_ai_generation_assertion is False
        assert result.has_ai_manipulation_assertion is False
        assert "camera capture" in result.rationale.lower() or "human-only" in result.rationale.lower()

    def test_valid_manifest_without_source_type_is_unspecified(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "assertions": [
                {"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.opened"}]}},
            ],
        }
        reader = _FakeReader(active=active, state="Valid")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.UNSPECIFIED
        assert "without an ai-generation claim" in result.rationale.lower()


class TestConflictingOrigin:
    def test_ai_and_camera_assertions_conflict(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "assertions": [
                {
                    "label": "c2pa.actions.v2",
                    "data": {"actions": [
                        {"action": "c2pa.created",
                         "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"},
                        {"action": "c2pa.opened",
                         "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"},
                    ]},
                },
            ],
        }
        reader = _FakeReader(active=active, state="Valid")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.CONFLICTING
        # Both AI and camera assertions must still be surfaced.
        assert result.has_ai_generation_assertion is True


class TestErrorHandling:
    def test_unexpected_exception_maps_to_error(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        monkeypatch.setattr(
            pv, "_open_reader",
            lambda p: (_ for _ in ()).throw(RuntimeError("something else broke")),
        )
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.ERROR
        assert result.validation_errors  # at least one entry

    def test_file_not_found_returns_error_status(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        missing = tmp_path / "does-not-exist.png"
        result = validate_provenance(missing)
        assert result.status is ProvenanceStatus.ERROR


class TestValidationErrorHarvesting:
    def test_failure_entries_surfaced(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {"assertions": []}
        results = {
            "activeManifest": {
                "failure": [
                    {"code": "assertion.hashedURI.mismatch", "explanation": "hash did not match"},
                ],
                "success": [{"code": "signingCredential.trusted"}],
            }
        }
        reader = _FakeReader(active=active, state="Invalid", results=results)
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.INVALID_OR_TAMPERED
        assert any("hashedURI.mismatch" in e for e in result.validation_errors)


# ─── Analyse_image integration ─────────────────────────────────────────────

class TestAnalyseImageIntegration:
    def test_analyse_image_populates_provenance(self, fake_image, monkeypatch):
        """``analyse_image`` must always return a ProvenanceResult, even
        when ExifTool is missing — the two pipelines are independent."""
        from src.genai_detection.metadata_module import analyse_image
        from src.genai_detection.metadata_module import metadata_extraction as me

        # ExifTool may not be installed in CI; short-circuit it.
        monkeypatch.setattr(me, "run_exiftool", lambda p: (_ for _ in ()).throw(RuntimeError("no exiftool")))
        # Force validator to a valid AI result via the module-level factory.
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "assertions": [
                {"label": "c2pa.actions.v2", "data": {"actions": [
                    {"action": "c2pa.created",
                     "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"},
                ]}},
            ],
        }
        monkeypatch.setattr(pv, "_open_reader", lambda p: _FakeReader(active=active, state="Valid"))

        result = analyse_image(str(fake_image))
        assert result.provenance.status is ProvenanceStatus.VALID
        assert result.provenance.origin_claim is OriginClaim.AI_GENERATED
        # And critically, missing EXIF keeps the heuristic P(AI) at 0.5:
        assert result.ai_probability == pytest.approx(0.50)


# ─── Optional real-library integration ─────────────────────────────────────

@pytest.mark.integration
def test_real_library_end_to_end_on_optional_asset():
    """
    End-to-end sanity check against the real ``c2pa-python`` library on
    an OFFICIAL C2PA sample asset provided via ``C2PA_TEST_ASSET_PATH``.

    Skipped when the environment variable isn't set or the file is
    missing — no large binary is committed. Attribution for the asset
    is the responsibility of whoever supplies the path (see
    https://github.com/contentauth/c2pa-python for the public sample
    library and its license).
    """
    asset = os.environ.get("C2PA_TEST_ASSET_PATH")
    if not asset:
        pytest.skip("C2PA_TEST_ASSET_PATH not set — skipping real-library test.")
    path = Path(asset)
    if not path.exists():
        pytest.skip(f"asset not found at {path}")

    result = validate_provenance(path)
    assert isinstance(result, ProvenanceResult)
    # The asset is either valid or, at worst, absent — never
    # validator_unavailable if we got this far.
    assert result.status is not ProvenanceStatus.VALIDATOR_UNAVAILABLE
