"""
End-to-end smoke test for the integration-pipeline web app
(src/genai_detection/integration_pipeline/app.py).

Spins up the real stdlib HTTPServer on a random port with **stub** visual
and deepfake classifiers so the test needs no HuggingFace downloads and
no ExifTool binary. Verifies:

  1. GET /            → the frontend HTML is served from web/index.html.
  2. POST /api/analyse with a tiny valid PNG → a JSON payload with the
     shape the frontend relies on (verdict, fusion, metadata, visual).
  3. The server thread stops cleanly and no temp files leak from
     run_analysis_pipeline's exiftool step.
"""

from __future__ import annotations

import io
import json
import tempfile
import threading
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

from src import PROJECT_ROOT
from src.genai_detection.integration_pipeline import app as app_mod
from src.genai_detection.hash_module import (
    HashRecord,
    HashRegistry,
    OriginLabel,
    sha256_bytes,
)


# ─── Stub classifiers ───────────────────────────────────────────────────────
# Both stubs return realistic-shaped payloads so run_analysis_pipeline
# exercises the real code paths without loading a 300 MB ViT or the YuNet
# ONNX file.

class _StubFaceDetector:
    """Enough of cv2.FaceDetectorYN's API for the saliency helpers we monkeypatch."""

    def setInputSize(self, size):  # noqa: N802 (mirroring OpenCV camelCase)
        return None

    def detect(self, img_bgr):
        # No faces — matches the small solid-colour test image.
        return None, None


class _StubVisualClassifier:
    def predict(self, image):
        return {
            "prediction": "Real",
            "confidence": 0.9,
            "all_scores": {"Real": 0.9, "AI-generated": 0.1},
        }


class _StubDeepfakeClassifier:
    def __init__(self):
        self.face_detector = _StubFaceDetector()

    def predict_face(self, image):
        return {"label": "No Face", "confidence": 0.05, "bbox": None}

    def predict(self, image):
        return {
            "visual_classification": None,
            "deepfake_analysis": {
                "is_deepfake": False,
                "has_face": False,
                "has_place": False,
                "face_analysis": {"label": "No Face", "confidence": 0.05, "bbox": None},
                "landmark_analysis": {"label": "None", "confidence": 0.0},
            },
        }


class _StubTrustMarkDetector:
    """
    Test double for the watermark-module detector — returns a
    NOT_DETECTED result without touching the real trustmark library.
    Used to exercise the watermark section of the /api/analyse response
    without requiring the ~hundreds of MB of TrustMark weights.
    """

    def analyse(self, image):
        from src.genai_detection.watermark_module import (
            SCOPE_STATEMENT,
            SUPPORTED_VARIANTS,
            TrustMarkResult,
            TrustMarkStatus,
        )

        return TrustMarkResult(
            supported_variants=list(SUPPORTED_VARIANTS),
            variant_used="Q",
            status=TrustMarkStatus.NOT_DETECTED,
            detected=False,
            rationale="stub: no watermark decoded — used for smoke tests.",
            processing_time_seconds=0.001,
            scope_statement=SCOPE_STATEMENT,
        )


def _tiny_png_bytes(colour=(200, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def stub_server(monkeypatch, tmp_path):
    """Bind app_mod.create_server on 127.0.0.1:<free>, tear it down after the test."""

    # Sidestep the real occlusion-saliency call: it wants a valid YuNet
    # detector and iterates a whole image. The stub keeps the shape the
    # caller expects (a HxW float array) but does zero real work.
    import numpy as np

    monkeypatch.setattr(
        app_mod,
        "compute_occlusion_saliency",
        lambda detector, image, patch_size, stride: np.zeros(image.size[::-1], dtype=np.float32),
    )
    monkeypatch.setattr(
        app_mod,
        "_overlay_heatmap",
        lambda img_np, sal, alpha=0.5, cmap="hot": img_np.astype(np.float32) / 255.0,
    )

    # Temporary hash registry with one preloaded record whose digest
    # matches the tiny PNG the smoke test uploads — exercises the
    # EXACT_MATCH branch without touching any user-visible location.
    registry_path = tmp_path / "hash_registry.json"
    registry_path.write_text('{"records": []}')
    registry = HashRegistry(registry_path)
    registry.register(
        HashRecord(
            sha256=sha256_bytes(_tiny_png_bytes()),
            origin_label=OriginLabel.AI_GENERATED,
            provider="smoketest",
            model="v0",
            notes="preloaded for the website smoke test",
        )
    )

    server, url = app_mod.create_server(
        _StubVisualClassifier(),
        _StubDeepfakeClassifier(),
        watermark_detector=_StubTrustMarkDetector(),
        hash_registry=registry,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "smoke-test server thread did not stop"


def test_get_index_serves_frontend_html(stub_server):
    resp = urllib.request.urlopen(f"{stub_server}/", timeout=5)
    body = resp.read()
    assert resp.status == 200
    assert resp.headers.get("Content-Type", "").startswith("text/html")
    # Sanity-check the served bytes match the checked-in frontend.
    expected = (PROJECT_ROOT / "web" / "index.html").read_bytes()
    assert body == expected


def test_post_analyse_returns_json_with_pipeline_shape(stub_server):
    boundary = "----smoketest-boundary-42"
    file_bytes = _tiny_png_bytes()
    body = (
        f"--{boundary}\r\n".encode()
        + b'Content-Disposition: form-data; name="file"; filename="tiny.png"\r\n'
        + b"Content-Type: image/png\r\n\r\n"
        + file_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        f"{stub_server}/api/analyse",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = urllib.request.urlopen(req, timeout=10)
    assert resp.status == 200
    payload = json.loads(resp.read())

    # Shape checks — the frontend depends on these keys existing.
    for key in ("verdict", "verdict_type", "fusion", "metadata", "visual", "watermark", "hash"):
        assert key in payload, f"missing {key} in analyse response"
    assert isinstance(payload["fusion"]["probability"], float)
    assert isinstance(payload["fusion"]["is_ai"], bool)
    assert 0.0 <= payload["metadata"]["probability"] <= 1.0
    assert 0.0 <= payload["visual"]["probability"] <= 1.0
    # No face in the stub, so cropped_visual is None and no deepfake stage ran.
    assert payload.get("cropped_visual") is None
    assert payload.get("deepfake") is None

    # Watermark section — the frontend depends on the full shape of the
    # scheme-specific TrustMark card, including the scope statement so
    # a negative result cannot be rebranded as "real" / "unwatermarked".
    wm = payload["watermark"]
    assert wm["scheme"] == "Adobe TrustMark"
    assert wm["status"] == "not_detected"
    assert wm["detected"] is False
    assert wm["schema_version"] is None
    assert wm["payload_preview"] is None
    assert wm["payload_truncated"] is False
    assert wm["variant_used"] == "Q"
    assert isinstance(wm["supported_variants"], list) and "Q" in wm["supported_variants"]
    assert "TrustMark" in wm["scope_statement"]

    # Hash section — the preloaded registry record matches the tiny PNG
    # the smoke test uploads, so we should see the exact-match branch
    # end-to-end (digest computed on the ORIGINAL upload bytes, not on
    # a re-encoded PIL image).
    hs = payload["hash"]
    assert hs["scheme"] == "SHA-256 byte-exact hash"
    assert hs["status"] == "exact_match"
    assert hs["sha256"] == sha256_bytes(_tiny_png_bytes())
    assert hs["registry_available"] is True
    assert hs["match"] is not None
    assert hs["match"]["origin_label"] == "ai_generated"
    assert hs["match"]["provider"] == "smoketest"
    assert "SHA-256" in hs["scope_statement"]


def test_hash_registry_failure_states_are_preserved(monkeypatch, tmp_path):
    """Task 5 regression: an invalid registry sentinel from
    load_registry() must reach /api/analyse as ``invalid_registry`` and
    a missing one as ``registry_unavailable`` — the two must never be
    collapsed. The digest of the uploaded bytes is always populated
    regardless of the registry failure mode."""
    import numpy as np
    from src.genai_detection.hash_module import (
        HashLookupStatus,
        invalid_result,
        unavailable_result,
    )

    monkeypatch.setattr(
        app_mod,
        "compute_occlusion_saliency",
        lambda detector, image, patch_size, stride: np.zeros(image.size[::-1], dtype=np.float32),
    )
    monkeypatch.setattr(
        app_mod,
        "_overlay_heatmap",
        lambda img_np, sal, alpha=0.5, cmap="hot": img_np.astype(np.float32) / 255.0,
    )

    file_bytes = _tiny_png_bytes()
    expected_digest = sha256_bytes(file_bytes)

    # 1) Invalid-registry sentinel — must survive intact through the
    # response as INVALID_REGISTRY, with the operator-facing diagnostic
    # preserved.
    bad_path = tmp_path / "broken.json"
    bad_path.write_text("{not valid")
    invalid_sentinel = invalid_result(path=bad_path, reason="registry file is not valid JSON: at line 1")

    result = app_mod.run_analysis_pipeline(
        file_bytes,
        {},
        _StubVisualClassifier(),
        _StubDeepfakeClassifier(),
        filename="tiny.png",
        hash_registry=invalid_sentinel,
    )
    assert result["hash"]["status"] == HashLookupStatus.INVALID_REGISTRY.value
    assert result["hash"]["sha256"] == expected_digest
    assert result["hash"]["registry_path"] == str(bad_path)
    assert result["hash"]["error_details"]  # diagnostic preserved

    # 2) Unavailable-registry sentinel — same digest surfaces, status
    # stays REGISTRY_UNAVAILABLE. Never a NO_MATCH.
    unavail_sentinel = unavailable_result(path=None, reason="env var unset")
    result = app_mod.run_analysis_pipeline(
        file_bytes,
        {},
        _StubVisualClassifier(),
        _StubDeepfakeClassifier(),
        filename="tiny.png",
        hash_registry=unavail_sentinel,
    )
    assert result["hash"]["status"] == HashLookupStatus.REGISTRY_UNAVAILABLE.value
    assert result["hash"]["sha256"] == expected_digest
    assert result["hash"]["match"] is None


def test_valid_empty_registry_produces_no_match(monkeypatch, tmp_path):
    """A valid but empty registry is NOT unavailable — it must produce
    a proper NO_MATCH result so the operator can distinguish 'nothing
    registered yet' from 'no registry file'."""
    import numpy as np
    from src.genai_detection.hash_module import HashLookupStatus, HashRegistry

    monkeypatch.setattr(
        app_mod,
        "compute_occlusion_saliency",
        lambda detector, image, patch_size, stride: np.zeros(image.size[::-1], dtype=np.float32),
    )
    monkeypatch.setattr(
        app_mod,
        "_overlay_heatmap",
        lambda img_np, sal, alpha=0.5, cmap="hot": img_np.astype(np.float32) / 255.0,
    )

    registry_path = tmp_path / "empty.json"
    registry_path.write_text('{"records": []}')
    registry = HashRegistry(registry_path)

    file_bytes = _tiny_png_bytes()
    result = app_mod.run_analysis_pipeline(
        file_bytes,
        {},
        _StubVisualClassifier(),
        _StubDeepfakeClassifier(),
        filename="tiny.png",
        hash_registry=registry,
    )
    assert result["hash"]["status"] == HashLookupStatus.NO_MATCH.value
    assert result["hash"]["registry_available"] is True
    assert result["hash"]["sha256"] == sha256_bytes(file_bytes)


def test_negative_verdict_is_inconclusive_not_likely_real(monkeypatch):
    """Task 4 regression: the pipeline must NOT emit 'Likely Real' when
    the fused decision is below threshold; it must be flagged
    inconclusive and never rebrand `1 - P(AI)` as "real confidence"."""
    import numpy as np

    monkeypatch.setattr(
        app_mod,
        "compute_occlusion_saliency",
        lambda detector, image, patch_size, stride: np.zeros(image.size[::-1], dtype=np.float32),
    )
    monkeypatch.setattr(
        app_mod,
        "_overlay_heatmap",
        lambda img_np, sal, alpha=0.5, cmap="hot": img_np.astype(np.float32) / 255.0,
    )

    file_bytes = _tiny_png_bytes()
    result = app_mod.run_analysis_pipeline(
        file_bytes, {}, _StubVisualClassifier(), _StubDeepfakeClassifier(), filename="tiny.png"
    )
    assert result["verdict_type"] == "inconclusive"
    assert "Likely Real" not in result["verdict"]
    assert "inconclusive" in result["verdict"].lower()
    assert "real confidence" not in result["verdict"].lower()


def test_pipeline_cleans_up_exiftool_tempfile(monkeypatch):
    """run_analysis_pipeline unlinks its ExifTool temp file even when the
    metadata module raises (ExifTool not installed on this box, malformed
    input, etc.). Regression guard for the analyse_image try/finally."""

    tmpdir = Path(tempfile.mkdtemp(prefix="smoketest-tmp-"))
    monkeypatch.setenv("TMPDIR", str(tmpdir))

    import numpy as np
    monkeypatch.setattr(
        app_mod,
        "compute_occlusion_saliency",
        lambda detector, image, patch_size, stride: np.zeros(image.size[::-1], dtype=np.float32),
    )
    monkeypatch.setattr(
        app_mod,
        "_overlay_heatmap",
        lambda img_np, sal, alpha=0.5, cmap="hot": img_np.astype(np.float32) / 255.0,
    )

    file_bytes = _tiny_png_bytes()
    app_mod.run_analysis_pipeline(
        file_bytes, {}, _StubVisualClassifier(), _StubDeepfakeClassifier(), filename="tiny.png"
    )

    leftovers = [p for p in tmpdir.iterdir() if p.is_file()]
    assert leftovers == [], f"exiftool temp files leaked: {leftovers}"
