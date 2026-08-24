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
the file is missing — no large binary is committed. Attribution for
the asset is the responsibility of whoever supplies the path (see
https://github.com/contentauth/c2pa-python for the public sample
library and its license).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.genai_detection.metadata_module import (
    OriginClaim,
    ProvenanceResult,
    ProvenanceStatus,
    validate_provenance,
)
from src.genai_detection.metadata_module import provenance_validation as pv


# ─── Fake exception classes ────────────────────────────────────────────────

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
    path = tmp_path / "asset.png"
    path.write_bytes(b"not really a png")
    return path


# ─── Reusable assertion / action builders ──────────────────────────────────

def _actions_manifest(*, dst: str | None = None, action: str = "c2pa.created",
                       claim_generator: str | None = None) -> dict:
    actions = [{"action": action}]
    if dst is not None:
        actions[0]["digitalSourceType"] = dst
    manifest: dict = {
        "assertions": [
            {"label": "c2pa.actions.v2", "data": {"actions": actions}},
        ],
    }
    if claim_generator is not None:
        manifest["claim_generator"] = claim_generator
    return manifest


# ─── Validator availability ────────────────────────────────────────────────

class TestValidatorUnavailable:
    def test_returns_validator_unavailable_when_lib_missing(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", False)
        monkeypatch.setattr(pv, "_C2PA_IMPORT_ERROR", "ImportError: no module named c2pa")
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALIDATOR_UNAVAILABLE
        assert result.manifest_found is False
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
        assert result.validation_passed is None
        assert result.signer_trusted is None
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


# ─── Validation-state semantics: Invalid / Valid / Trusted / Untrusted ─────

class TestInvalid:
    def test_signature_error_at_open_time(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        monkeypatch.setattr(
            pv, "_open_reader",
            lambda p: (_ for _ in ()).throw(_FakeSignatureError("signature verification failed")),
        )
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.INVALID_OR_TAMPERED
        assert result.validation_passed is False
        assert result.signer_trusted is None

    def test_invalid_state_after_open(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        reader = _FakeReader(active=_actions_manifest(dst=None), state="Invalid")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.INVALID_OR_TAMPERED
        assert result.manifest_found is True
        assert result.validation_passed is False
        assert result.signer_trusted is None


class TestValidVsTrusted:
    """The critical semantic split: `Valid` = crypto-valid but signer not
    established; `Trusted` = valid AND signer chains to a trust anchor.
    Older library versions also produce `Untrusted` — same meaning as
    `Valid`."""

    def test_valid_state_maps_to_untrusted_signer(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = _actions_manifest(
            dst="http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
        )
        reader = _FakeReader(active=active, state="Valid")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.UNTRUSTED_SIGNER, (
            "'Valid' means crypto-valid without trust — must NOT map to VALID"
        )
        assert result.validation_passed is True
        assert result.signer_trusted is False
        r = result.rationale.lower()
        assert "trust" in r
        # Must not read as a validation failure.
        assert "validation failed" not in r
        assert "appears altered" not in r

    def test_untrusted_legacy_state_matches_valid(self, fake_image, monkeypatch):
        """Older library versions surface `Untrusted` for what newer
        versions call `Valid`. Both must yield the same result."""
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = _actions_manifest(
            dst="http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
        )
        reader = _FakeReader(active=active, state="Untrusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.UNTRUSTED_SIGNER
        assert result.validation_passed is True
        assert result.signer_trusted is False

    def test_trusted_state_maps_to_valid_with_signer_trusted(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = _actions_manifest(
            dst="http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
        )
        reader = _FakeReader(active=active, state="Trusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.validation_passed is True
        assert result.signer_trusted is True


class TestUnknownValidationState:
    def test_none_state_becomes_error_not_valid(self, fake_image, monkeypatch):
        """A manifest with no reported validation state must NOT be
        classified as valid — Python truthiness on ``None`` must not
        become 'signer not in trust list'."""
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        reader = _FakeReader(active=_actions_manifest(), state=None)
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.ERROR
        assert result.validation_passed is None
        assert result.signer_trusted is None
        # And critically, rationale must not read as a valid manifest.
        assert "cannot be reported as valid" in result.rationale.lower() or \
               "not be determined" in result.rationale.lower() or \
               "no recognised" in result.rationale.lower()

    def test_unknown_state_string_becomes_error(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        reader = _FakeReader(active=_actions_manifest(), state="Purple")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.ERROR
        assert result.validation_passed is None
        assert result.signer_trusted is None


# ─── AI / non-AI origin (with Trusted state so status stays VALID) ─────────

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
        reader = _FakeReader(active=active, state="Trusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.signer_trusted is True
        assert result.origin_claim is OriginClaim.AI_GENERATED
        assert result.has_ai_generation_assertion is True
        assert result.has_ai_manipulation_assertion is False
        assert result.claim_generator == "OpenAI/dall-e-3"
        assert "OpenAI dall-e-3" in result.software_agents
        assert "c2pa.created" in result.actions
        assert "trainedAlgorithmicMedia" in result.digital_source_types[0]
        assert "ai-generation" in result.rationale.lower()

    def test_ai_modified_origin_detected(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = _actions_manifest(
            action="c2pa.edited",
            dst="http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
        )
        reader = _FakeReader(active=active, state="Trusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.AI_MODIFIED
        assert result.has_ai_manipulation_assertion is True
        assert "ai-modification" in result.rationale.lower()


class TestValidNonAiOrigin:
    @pytest.mark.parametrize("short_type", sorted(pv._CAMERA_HUMAN_TYPES))
    def test_camera_or_human_types_do_not_assert_ai(self, short_type, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = _actions_manifest(
            dst=f"http://cv.iptc.org/newscodes/digitalsourcetype/{short_type}",
        )
        reader = _FakeReader(active=active, state="Trusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.CAMERA_OR_HUMAN_ORIGIN
        assert result.has_ai_generation_assertion is False
        assert result.has_ai_manipulation_assertion is False

    def test_valid_manifest_without_source_type_is_unspecified(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "assertions": [
                {"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.opened"}]}},
            ],
        }
        reader = _FakeReader(active=active, state="Trusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.UNSPECIFIED
        assert "without an ai-generation claim" in result.rationale.lower()


class TestAmbiguousIptcTypes:
    """Values that DESCRIBE how media was produced/rendered but do not
    necessarily involve trained/generative AI. Recording them is fine;
    they must never set the AI-generation / AI-manipulation assertions."""

    @pytest.mark.parametrize(
        "short_type",
        sorted(pv._AMBIGUOUS_TYPES),
    )
    def test_ambiguous_type_stays_unspecified(self, short_type, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = _actions_manifest(
            dst=f"http://cv.iptc.org/newscodes/digitalsourcetype/{short_type}",
        )
        reader = _FakeReader(active=active, state="Trusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.origin_claim is OriginClaim.UNSPECIFIED, (
            f"Ambiguous IPTC type '{short_type}' must not classify as AI-generated / AI-modified."
        )
        assert result.has_ai_generation_assertion is False
        assert result.has_ai_manipulation_assertion is False
        # But the raw value is still preserved for auditability.
        assert any(short_type in d for d in result.digital_source_types)

    def test_unknown_type_stays_unspecified(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = _actions_manifest(
            dst="http://cv.iptc.org/newscodes/digitalsourcetype/somebrandnewthing",
        )
        reader = _FakeReader(active=active, state="Trusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.origin_claim is OriginClaim.UNSPECIFIED
        assert result.has_ai_generation_assertion is False
        assert result.has_ai_manipulation_assertion is False


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
        reader = _FakeReader(active=active, state="Trusted")
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.origin_claim is OriginClaim.CONFLICTING
        assert result.has_ai_generation_assertion is True


# ─── Warnings / failures split ─────────────────────────────────────────────

class TestFailuresVsWarnings:
    def test_failure_entries_surfaced_and_invalidate(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        results = {
            "activeManifest": {
                "failure": [
                    {"code": "assertion.hashedURI.mismatch", "explanation": "hash did not match"},
                ],
                "success": [{"code": "signingCredential.trusted"}],
            }
        }
        reader = _FakeReader(active=_actions_manifest(), state="Trusted", results=results)
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        # A hard failure MUST invalidate even a "Trusted" state.
        assert result.status is ProvenanceStatus.INVALID_OR_TAMPERED
        assert any("hashedURI.mismatch" in e for e in result.validation_errors)

    def test_informational_warnings_do_not_invalidate(self, fake_image, monkeypatch):
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        results = {
            "activeManifest": {
                "failure": [],
                "informational": [
                    {"code": "manifest.warning", "explanation": "an informational note"},
                ],
            }
        }
        reader = _FakeReader(active=_actions_manifest(), state="Trusted", results=results)
        monkeypatch.setattr(pv, "_open_reader", lambda p: reader)
        result = validate_provenance(fake_image)
        assert result.status is ProvenanceStatus.VALID
        assert result.validation_errors == []
        assert any("manifest.warning" in w for w in result.validation_warnings)


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


# ─── Analyse_image integration ─────────────────────────────────────────────

class TestAnalyseImageIntegration:
    def test_analyse_image_populates_provenance(self, fake_image, monkeypatch):
        """``analyse_image`` must always return a ProvenanceResult, even
        when ExifTool is missing — the two pipelines are independent."""
        from src.genai_detection.metadata_module import analyse_image
        from src.genai_detection.metadata_module import metadata_extraction as me

        monkeypatch.setattr(me, "run_exiftool", lambda p: (_ for _ in ()).throw(RuntimeError("no exiftool")))
        monkeypatch.setattr(pv, "_C2PA_AVAILABLE", True)
        active = {
            "assertions": [
                {"label": "c2pa.actions.v2", "data": {"actions": [
                    {"action": "c2pa.created",
                     "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"},
                ]}},
            ],
        }
        monkeypatch.setattr(pv, "_open_reader", lambda p: _FakeReader(active=active, state="Trusted"))

        result = analyse_image(str(fake_image))
        assert result.provenance.status is ProvenanceStatus.VALID
        assert result.provenance.signer_trusted is True
        assert result.provenance.origin_claim is OriginClaim.AI_GENERATED
        # And critically, missing EXIF keeps the heuristic P(AI) at 0.5:
        assert result.ai_probability == pytest.approx(0.50)


# ─── Optional real-library integration ─────────────────────────────────────

@pytest.mark.integration
def test_real_library_end_to_end_on_optional_asset():
    asset = os.environ.get("C2PA_TEST_ASSET_PATH")
    if not asset:
        pytest.skip("C2PA_TEST_ASSET_PATH not set — skipping real-library test.")
    path = Path(asset)
    if not path.exists():
        pytest.skip(f"asset not found at {path}")

    result = validate_provenance(path)
    assert isinstance(result, ProvenanceResult)
    assert result.status is not ProvenanceStatus.VALIDATOR_UNAVAILABLE
