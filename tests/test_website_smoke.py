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
def stub_server(monkeypatch):
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

    server, url = app_mod.create_server(
        _StubVisualClassifier(),
        _StubDeepfakeClassifier(),
        watermark_detector=_StubTrustMarkDetector(),
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
    for key in ("verdict", "verdict_type", "fusion", "metadata", "visual", "watermark"):
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
