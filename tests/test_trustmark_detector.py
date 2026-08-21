"""
Tests for the Adobe TrustMark detector
(src/genai_detection/watermark_module/).

Covers every status of :class:`TrustMarkStatus` — DETECTED,
NOT_DETECTED, DETECTOR_UNAVAILABLE (library missing + model-load
failure), UNSUPPORTED (variant + input type), and ERROR — plus the
non-obvious contract points the rest of the pipeline relies on:

  * Importing ``src.genai_detection.watermark_module`` never triggers a
    model download and never requires the ``trustmark`` package to be
    installed.
  * The heavy import lives inside :meth:`TrustMarkDetector.analyse`;
    constructing the detector is cheap.
  * A loaded ``TrustMark`` instance is memoised per variant so repeated
    website requests do not reload the model.
  * Colour conversion happens without resizing / recompressing the
    original — TrustMark's decoder is called on the untouched pixels.
  * Website startup does not require any watermark models to load.

Model-dependent behaviour goes through the ``@pytest.mark.integration``
opt-in below so the default suite stays usable on a box without the
``trustmark`` package or its downloaded weights.
"""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from src.genai_detection.watermark_module import (
    DEFAULT_VARIANT,
    SCHEME_NAME,
    SCOPE_STATEMENT,
    SUPPORTED_VARIANTS,
    TrustMarkDetector,
    TrustMarkResult,
    TrustMarkStatus,
)
from src.genai_detection.watermark_module import trustmark_detector as td_module


# ---------------------------------------------------------------------------
# Fake trustmark library — installed into sys.modules for the tests that
# need to exercise a "successful load" without pulling ~hundreds of MB of
# real model weights.
# ---------------------------------------------------------------------------


class _FakeTrustMark:
    """Enough of ``trustmark.TrustMark`` to exercise decode paths."""

    #: Overridden per-test to control what .decode() returns.
    decode_return: tuple[str, bool, int] = ("", False, -1)

    #: Records constructor calls across tests so we can assert lazy load
    #: + per-variant caching behaviour.
    init_calls: list[dict] = []

    #: When True, __init__ raises to simulate a failed model download.
    raise_on_init: bool = False

    def __init__(self, model_type="Q", device="", verbose=False, **kwargs):
        _FakeTrustMark.init_calls.append(
            {"model_type": model_type, "device": device, "verbose": verbose}
        )
        if _FakeTrustMark.raise_on_init:
            raise RuntimeError("simulated weight download failure")
        self.model_type = model_type
        # Track calls per instance so tests can prove the cached
        # instance was reused instead of a fresh one being built.
        self.decode_calls = 0

    def decode(self, image):
        self.decode_calls += 1
        return _FakeTrustMark.decode_return


@pytest.fixture
def fake_trustmark(monkeypatch):
    """
    Install a fake ``trustmark`` module into sys.modules for the
    duration of one test, resetting the class-level call log so
    assertions are isolated.
    """
    _FakeTrustMark.init_calls = []
    _FakeTrustMark.raise_on_init = False
    _FakeTrustMark.decode_return = ("", False, -1)

    fake_mod = types.ModuleType("trustmark")
    fake_mod.TrustMark = _FakeTrustMark  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trustmark", fake_mod)

    yield _FakeTrustMark


def _tiny_png_bytes(colour=(200, 40, 40), size=(16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Package-level invariants
# ---------------------------------------------------------------------------


class TestModuleImportContract:
    def test_import_does_not_require_trustmark_library(self, monkeypatch):
        """Importing the module must not attempt ``import trustmark``.

        Simulated by removing it from sys.modules and blocking a re-
        import: a fresh module import must still succeed."""
        # Sanity check: reimporting the package with a poisoned finder
        # should still work, because the real import happens inside
        # .analyse(), not at module top.
        monkeypatch.setitem(sys.modules, "trustmark", None)  # None disables import
        # Force reimport
        for name in list(sys.modules):
            if name.startswith("src.genai_detection.watermark_module"):
                monkeypatch.delitem(sys.modules, name, raising=False)
        import importlib

        pkg = importlib.import_module("src.genai_detection.watermark_module")
        assert pkg.TrustMarkDetector is not None
        assert pkg.TrustMarkStatus.DETECTOR_UNAVAILABLE.value == "detector_unavailable"

    def test_default_variant_is_supported(self):
        assert DEFAULT_VARIANT in SUPPORTED_VARIANTS

    def test_scheme_name_and_scope_are_fixed_strings(self):
        assert SCHEME_NAME == "Adobe TrustMark"
        assert "TrustMark" in SCOPE_STATEMENT
        assert "does NOT" in SCOPE_STATEMENT or "not prove" in SCOPE_STATEMENT.lower()


# ---------------------------------------------------------------------------
# Constructor / config
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_construction_never_touches_trustmark(self, monkeypatch):
        """Instantiating the detector must not trigger the import.

        We arrange sys.modules so importing trustmark would raise, and
        confirm construction still succeeds."""
        monkeypatch.setitem(sys.modules, "trustmark", None)
        det = TrustMarkDetector()
        assert det.default_variant == DEFAULT_VARIANT

    def test_unsupported_default_variant_rejected(self):
        with pytest.raises(ValueError, match="Unsupported default TrustMark variant"):
            TrustMarkDetector(default_variant="Z")


# ---------------------------------------------------------------------------
# Detected / not-detected / unsupported / error steady states
# ---------------------------------------------------------------------------


class TestDetected:
    def test_positive_decode_returns_detected_result(self, fake_trustmark):
        fake_trustmark.decode_return = ("hello-world", True, 1)
        det = TrustMarkDetector()
        result = det.analyse(_tiny_png_bytes())

        assert isinstance(result, TrustMarkResult)
        assert result.status == TrustMarkStatus.DETECTED
        assert result.detected is True
        assert result.variant_used == DEFAULT_VARIANT
        assert result.schema_version == 1
        assert result.payload == "hello-world"
        assert result.scheme == SCHEME_NAME
        assert result.scope_statement == SCOPE_STATEMENT
        assert result.processing_time_seconds >= 0.0
        assert result.error_details is None


class TestNotDetected:
    def test_negative_decode_reports_only_no_supported_watermark(self, fake_trustmark):
        fake_trustmark.decode_return = ("", False, -1)
        det = TrustMarkDetector()
        result = det.analyse(_tiny_png_bytes())

        assert result.status == TrustMarkStatus.NOT_DETECTED
        assert result.detected is False
        assert result.schema_version is None
        assert result.payload is None
        # Rationale must not overstate the finding.
        assert "does not mean" in result.rationale.lower()
        assert "unwatermarked" in result.rationale.lower() or "real" in result.rationale.lower()


class TestUnsupported:
    def test_unknown_variant_returns_unsupported_without_loading_model(self, fake_trustmark):
        det = TrustMarkDetector()
        result = det.analyse(_tiny_png_bytes(), variant="Z")
        assert result.status == TrustMarkStatus.UNSUPPORTED
        # Never loaded the fake model because the variant was rejected first.
        assert fake_trustmark.init_calls == []
        assert result.detected is False
        assert result.variant_used is None

    def test_unsupported_input_type_reports_unsupported(self, fake_trustmark):
        det = TrustMarkDetector()
        result = det.analyse(12345)  # type: ignore[arg-type]
        assert result.status == TrustMarkStatus.UNSUPPORTED
        assert "type" in result.rationale.lower()


class TestDetectorUnavailable:
    def test_missing_library_reports_detector_unavailable(self, monkeypatch):
        """No ``trustmark`` module installed → DETECTOR_UNAVAILABLE, not
        NOT_DETECTED, and never a raised exception."""
        # Make `import trustmark` raise ImportError.
        monkeypatch.setitem(sys.modules, "trustmark", None)

        det = TrustMarkDetector()
        result = det.analyse(_tiny_png_bytes())
        assert result.status == TrustMarkStatus.DETECTOR_UNAVAILABLE
        assert result.detected is False
        assert result.error_details is not None
        assert "not importable" in result.error_details.lower()
        # Second call fails fast without re-importing — the reason is memoised.
        result2 = det.analyse(_tiny_png_bytes())
        assert result2.status == TrustMarkStatus.DETECTOR_UNAVAILABLE

    def test_failed_model_init_reports_detector_unavailable(self, fake_trustmark):
        fake_trustmark.raise_on_init = True
        det = TrustMarkDetector()
        result = det.analyse(_tiny_png_bytes())
        assert result.status == TrustMarkStatus.DETECTOR_UNAVAILABLE
        assert "simulated weight download failure" in (result.error_details or "")

        # Once marked unavailable, subsequent requests must NOT retry
        # the failing init (network thrash guard).
        before = len(fake_trustmark.init_calls)
        det.analyse(_tiny_png_bytes())
        assert len(fake_trustmark.init_calls) == before


class TestError:
    def test_decoder_exception_becomes_error_result(self, fake_trustmark):
        # Build the detector with a fake that decodes fine, then swap in
        # a decode that blows up mid-call.
        det = TrustMarkDetector()

        class _BoomModel(_FakeTrustMark):
            def decode(self, image):
                raise RuntimeError("boom")

        # Manually seed the cache to bypass the fake sys.modules path.
        det._model_cache[DEFAULT_VARIANT] = _BoomModel()
        result = det.analyse(_tiny_png_bytes())
        assert result.status == TrustMarkStatus.ERROR
        assert "boom" in (result.error_details or "")


# ---------------------------------------------------------------------------
# Lazy loading + caching
# ---------------------------------------------------------------------------


class TestLazyLoadingAndCache:
    def test_construction_does_not_load_model(self, fake_trustmark):
        TrustMarkDetector()
        assert fake_trustmark.init_calls == [], "constructor loaded a model — must be lazy"

    def test_first_analysis_loads_second_reuses_cached_instance(self, fake_trustmark):
        det = TrustMarkDetector()
        det.analyse(_tiny_png_bytes())
        det.analyse(_tiny_png_bytes())
        # Exactly one TrustMark() init for the default variant across
        # multiple analyse() calls.
        assert len(fake_trustmark.init_calls) == 1
        assert fake_trustmark.init_calls[0]["model_type"] == DEFAULT_VARIANT

    def test_different_variants_load_independently(self, fake_trustmark):
        det = TrustMarkDetector()
        det.analyse(_tiny_png_bytes(), variant="Q")
        det.analyse(_tiny_png_bytes(), variant="P")
        det.analyse(_tiny_png_bytes(), variant="Q")  # cached
        variants_loaded = [c["model_type"] for c in fake_trustmark.init_calls]
        assert variants_loaded == ["Q", "P"]


# ---------------------------------------------------------------------------
# Colour-mode + input-shape contract
# ---------------------------------------------------------------------------


class TestInputHandling:
    def test_accepts_pil_image(self, fake_trustmark):
        fake_trustmark.decode_return = ("wm", True, 0)
        det = TrustMarkDetector()
        img = Image.new("RGB", (32, 32), (10, 20, 30))
        result = det.analyse(img)
        assert result.status == TrustMarkStatus.DETECTED

    def test_accepts_file_path(self, fake_trustmark, tmp_path):
        fake_trustmark.decode_return = ("wm", True, 0)
        p = tmp_path / "img.png"
        p.write_bytes(_tiny_png_bytes())
        det = TrustMarkDetector()
        result = det.analyse(str(p))
        assert result.status == TrustMarkStatus.DETECTED
        result2 = det.analyse(p)
        assert result2.status == TrustMarkStatus.DETECTED

    def test_converts_rgba_to_rgb_without_resizing(self, fake_trustmark):
        """The decoder must see the original pixels — colour-mode
        conversion is allowed, but the image must not be resized or
        recompressed before decoding."""
        received: list[Image.Image] = []

        class _CaptureModel(_FakeTrustMark):
            def decode(self_inner, image):
                received.append(image)
                return ("", False, -1)

        det = TrustMarkDetector()
        det._model_cache[DEFAULT_VARIANT] = _CaptureModel()

        rgba = Image.new("RGBA", (37, 29), (10, 20, 30, 128))
        det.analyse(rgba)

        assert len(received) == 1
        seen = received[0]
        assert seen.mode == "RGB"
        # Same width/height — no silent resize.
        assert seen.size == (37, 29)


# ---------------------------------------------------------------------------
# Website startup + optional integration test
# ---------------------------------------------------------------------------


class TestWebsiteStartupDoesNotDownloadModels:
    def test_constructing_pipeline_server_does_not_load_watermark_model(self, fake_trustmark):
        """The website builds its server object with detectors already
        constructed. Attaching a TrustMarkDetector to the server must
        not, on its own, load a model — that has to wait for the first
        analysis request."""
        # app.py imports the deepfake pipeline which needs torch. On a
        # torch-less test box, importorskip so this test is not counted
        # as a failure of the watermark contract.
        pytest.importorskip("torch")
        from src.genai_detection.integration_pipeline import app as app_mod

        # Minimal stubs — nothing here should touch TrustMark either.
        class _StubVisual:
            def predict(self, image):
                return {"prediction": "Real", "confidence": 0.9, "all_scores": {"Real": 0.9}}

        class _StubDeepfake:
            face_detector = None

        server, _url = app_mod.create_server(_StubVisual(), _StubDeepfake(), watermark_detector=TrustMarkDetector())
        try:
            assert fake_trustmark.init_calls == [], "server construction loaded a TrustMark model"
        finally:
            server.server_close()


@pytest.mark.integration
class TestTrustMarkIntegration:
    """
    Opt-in end-to-end test: encode a known payload with the real
    ``trustmark`` library, run the detector, confirm round-trip; then
    confirm an unwatermarked control is NOT reported as detected.

    Requires ``pip install trustmark`` and network access to fetch the
    model weights on first run. Skips (rather than fails) when either
    is unavailable so the default suite stays clean.

    Run with: ``pytest -m integration tests/test_trustmark_detector.py``
    """

    def test_round_trip_and_negative_control(self):
        pytest.importorskip("trustmark")
        try:
            from trustmark import TrustMark  # type: ignore
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"trustmark not usable: {exc}")

        try:
            tm = TrustMark(model_type=DEFAULT_VARIANT, verbose=False)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"TrustMark model weights unavailable: {exc}")

        # Encode a payload into a solid-colour test image.
        cover = Image.new("RGB", (256, 256), (128, 128, 128))
        secret = "abcdef"
        stego = tm.encode(cover, secret)

        det = TrustMarkDetector()
        wm_result = det.analyse(stego)
        assert wm_result.status == TrustMarkStatus.DETECTED
        assert wm_result.payload is not None
        assert secret in wm_result.payload

        control = Image.new("RGB", (256, 256), (200, 40, 40))
        control_result = det.analyse(control)
        assert control_result.status == TrustMarkStatus.NOT_DETECTED
