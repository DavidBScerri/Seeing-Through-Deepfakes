import os
import sys
import json
import io
import tempfile
import socket
import webbrowser
import threading
import base64
import atexit
import traceback
from email.parser import BytesParser
from email.policy import default as _email_policy
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np
from PIL import Image

# Bootstrap the project root so `python src/.../app.py` works as well as
# `python -m src.genai_detection.integration_pipeline.app`. The canonical
# PROJECT_ROOT helper lives in src/__init__.py; import it via a short
# explicit walk so this file can run before `src` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import PROJECT_ROOT
from src.genai_detection.metadata_module import analyse_image, AnalysisResult
from src.genai_detection.integration_pipeline import config
from src.genai_detection.integration_pipeline.fusion import (
    get_fusion_strategy,
    extract_visual_ai_probability,
    crop_face_region,
)
from src.genai_detection.watermark_module import TrustMarkDetector, TrustMarkStatus
from src.deepfake_detection.gradcam_face_analysis import compute_occlusion_saliency, _overlay_heatmap

# Frontend (index.html) lives outside src/ under web/ so the backend
# package doesn't ship browser assets — see README "Project layout".
_WEB_DIR = PROJECT_ROOT / "web"

_active_server = None
_server_thread = None


class PipelineRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress request logging noise

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            
            html_path = _WEB_DIR / "index.html"
            if html_path.exists():
                self.wfile.write(html_path.read_bytes())
            else:
                self.wfile.write(b"<h1>index.html not found</h1>")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _send_json_error(self, status, message):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/analyse":
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self._send_json_error(400, "Content-Type must be multipart/form-data")
                return

            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self._send_json_error(400, "Missing or empty request body")
                return
            if content_length > config.MAX_UPLOAD_BYTES:
                self._send_json_error(413, f"Upload exceeds {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
                return
            body = self.rfile.read(content_length)

            try:
                file_data, filename, params = parse_multipart_form(body, content_type)
            except Exception:
                traceback.print_exc()
                self._send_json_error(400, "Malformed multipart/form-data body")
                return

            if file_data is None:
                self._send_json_error(400, "No file uploaded")
                return

            try:
                visual_classifier = self.server.visual_classifier
                deepfake_classifier = self.server.deepfake_classifier
                # Watermark detector is scheme-specific (Adobe TrustMark).
                # Deliberately kept off the fusion path — see run_analysis_pipeline
                # docstring and CLAUDE.md's fusion-formula rule.
                watermark_detector = getattr(self.server, "watermark_detector", None)

                result = run_analysis_pipeline(
                    file_data,
                    params,
                    visual_classifier,
                    deepfake_classifier,
                    filename=filename,
                    watermark_detector=watermark_detector,
                )
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


_ALLOWED_UPLOAD_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif", ".heic"}


def parse_multipart_form(body: bytes, content_type: str) -> tuple[bytes | None, str | None, dict[str, str]]:
    """
    Parses a multipart/form-data body using the stdlib email parser
    (boundary-safe for binary payloads, unlike naive splitting).

    Returns:
        (file_data, filename, params) — file_data/filename are None when
        no file part is present.
    """
    msg = BytesParser(policy=_email_policy).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8", errors="ignore")
        + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )

    file_data = None
    filename = None
    params: dict[str, str] = {}

    if not msg.is_multipart():
        return None, None, params

    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name is None:
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        # get_filename() is None for plain fields; an empty string still
        # marks a file part (some clients send filename=""), so compare
        # against None rather than truthiness.
        part_filename = part.get_filename()
        if part_filename is not None:
            file_data = payload
            filename = part_filename
        else:
            params[str(name)] = payload.decode("utf-8", errors="ignore")

    return file_data, filename, params


#: Truncation length for TrustMark payloads exposed in the ordinary UI
#: response. The full payload stays available to research consumers by
#: rebuilding the TrustMarkResult from the detector directly — the web
#: card only needs a short preview.
_WATERMARK_PAYLOAD_PREVIEW_LEN = 32


def _build_watermark_payload(watermark_detector, file_data: bytes) -> dict[str, object]:
    """
    Runs the (optional) TrustMark detector against the uploaded bytes
    and shapes its :class:`TrustMarkResult` into the JSON dict the
    frontend renders.

    Missing detector, missing library, missing weights and unexpected
    exceptions all resolve to a stable-shape dict — the analysis
    request must still succeed when TrustMark is not installed or its
    model is unavailable.

    The payload is truncated to a short preview so an unusually long
    or binary-looking decoded string does not bloat the JSON response;
    the full decoded string stays available to research consumers via
    the detector's return value directly.
    """
    from src.genai_detection.watermark_module import (
        SCHEME_NAME as _WM_SCHEME,
        SCOPE_STATEMENT as _WM_SCOPE,
        SUPPORTED_VARIANTS as _WM_VARIANTS,
        TrustMarkResult,
        TrustMarkStatus as _WMStatus,
    )

    if watermark_detector is None:
        # No detector attached — never invent a "not detected" result.
        result = TrustMarkResult(
            status=_WMStatus.DETECTOR_UNAVAILABLE,
            rationale=(
                "TrustMark detector is not attached to this server build. "
                "Watermark evidence card is unavailable."
            ),
            error_details="watermark_detector is None on the server object.",
        )
    else:
        try:
            result = watermark_detector.analyse(file_data)
        except Exception as exc:
            traceback.print_exc()
            result = TrustMarkResult(
                status=_WMStatus.ERROR,
                rationale="TrustMark detector raised unexpectedly.",
                error_details=f"{type(exc).__name__}: {exc}",
            )

    payload_preview: str | None = None
    payload_truncated = False
    if result.payload is not None:
        if len(result.payload) > _WATERMARK_PAYLOAD_PREVIEW_LEN:
            payload_preview = result.payload[:_WATERMARK_PAYLOAD_PREVIEW_LEN] + "…"
            payload_truncated = True
        else:
            payload_preview = result.payload

    return {
        "scheme": result.scheme,
        "supported_variants": result.supported_variants,
        "variant_used": result.variant_used,
        "status": result.status.value,
        "detected": result.detected,
        "schema_version": result.schema_version,
        # Only ever a short preview — the full payload stays behind the
        # detector's Python return value so ordinary UI never renders
        # something long or binary-looking.
        "payload_preview": payload_preview,
        "payload_truncated": payload_truncated,
        "rationale": result.rationale,
        "error_details": result.error_details,
        "processing_time_seconds": result.processing_time_seconds,
        "scope_statement": result.scope_statement,
    }


def _temp_suffix_for(filename: str | None) -> str:
    """ExifTool's handling is file-type dependent, so keep the uploaded extension when it's a known image type."""
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in _ALLOWED_UPLOAD_SUFFIXES:
            return suffix
    return ".png"


def run_analysis_pipeline(
    file_data,
    params,
    visual_classifier,
    deepfake_classifier,
    filename=None,
    watermark_detector=None,
):
    """
    Runs the full evidence pipeline against one uploaded image and
    returns the JSON payload the frontend expects.

    ``watermark_detector`` is optional. When supplied it produces a
    scheme-specific TrustMark evidence card exposed under the
    ``watermark`` key of the response — the result is deliberately NOT
    fed into fusion, and adding a fusion weight for it would need
    separate evaluation and David's sign-off (see CLAUDE.md).
    When absent, the response still carries a ``watermark`` block with
    ``status=detector_unavailable`` so the UI can render a stable card
    on every build.
    """
    pil_image = Image.open(io.BytesIO(file_data))
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Metadata analysis (needs file path for exiftool)
    with tempfile.NamedTemporaryFile(suffix=_temp_suffix_for(filename), delete=False) as tmp:
        tmp.write(file_data)
        tmp.flush()
        tmp_path = tmp.name

    try:
        meta_result = analyse_image(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Visual classifier (whole image)
    visual_result = visual_classifier.predict(pil_image)
    visual_ai_prob = extract_visual_ai_probability(visual_result)

    # Global YuNet Saliency Map for whole image
    try:
        # Use large patch/stride for whole image speed
        patch_size = max(16, int(min(pil_image.size) / 16))
        stride = max(8, int(patch_size / 2))
        saliency_map = compute_occlusion_saliency(
            deepfake_classifier.face_detector, 
            pil_image, 
            patch_size=patch_size, 
            stride=stride
        )
        
        # Overlay and convert to base64
        img_np = np.array(pil_image)
        overlay_rgb = _overlay_heatmap(img_np, saliency_map, alpha=0.5, cmap="hot")
        overlay_uint8 = (overlay_rgb * 255).astype(np.uint8)
        
        buffered = io.BytesIO()
        Image.fromarray(overlay_uint8).save(buffered, format="JPEG", quality=85)
        visual_gradcam_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        traceback.print_exc()
        print(f"[YuNet Saliency] Whole-image heatmap failed: {e}")
        visual_gradcam_b64 = None

    # Face crop detection & classification
    face_res = deepfake_classifier.predict_face(pil_image)
    bbox = face_res.get("bbox")
    
    cropped_visual_result = None
    cropped_visual_ai_prob = None
    cropped_face_b64 = None
    cropped_gradcam_b64 = None
    
    if bbox is not None:
        cropped_face = crop_face_region(pil_image, bbox, padding=config.FACE_PADDING)
        
        buffered = io.BytesIO()
        cropped_face.save(buffered, format="JPEG")
        cropped_face_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        cropped_visual_result = visual_classifier.predict(cropped_face)
        cropped_visual_ai_prob = extract_visual_ai_probability(cropped_visual_result)

        try:
            # For cropped face, use a smaller patch size
            patch_size = max(4, int(min(cropped_face.size) / 8))
            stride = max(2, int(patch_size / 2))
            crop_saliency = compute_occlusion_saliency(
                deepfake_classifier.face_detector,
                cropped_face,
                patch_size=patch_size,
                stride=stride
            )
            # Normalize crop saliency
            if crop_saliency.max() > 0:
                crop_saliency = crop_saliency / crop_saliency.max()
                
            crop_img_np = np.array(cropped_face)
            crop_overlay_rgb = _overlay_heatmap(crop_img_np, crop_saliency, alpha=0.5, cmap="jet")
            crop_overlay_uint8 = (crop_overlay_rgb * 255).astype(np.uint8)
            
            buffered2 = io.BytesIO()
            Image.fromarray(crop_overlay_uint8).save(buffered2, format="JPEG", quality=85)
            cropped_gradcam_b64 = base64.b64encode(buffered2.getvalue()).decode("utf-8")
        except Exception as e:
            traceback.print_exc()
            print(f"[YuNet Saliency] Cropped-face heatmap failed: {e}")
            cropped_gradcam_b64 = None

    # Decision fusion
    strategy_name = config.FUSION_STRATEGY
    if strategy_name == "weighted_average":
        strategy = get_fusion_strategy(
            "weighted_average",
            w_meta=config.W_META,
            w_visual=config.W_VISUAL,
            decision_threshold=config.WA_DECISION_THRESHOLD,
            meta_accuracy=config.META_ACCURACY,
            visual_accuracy=config.VISUAL_ACCURACY,
        )
    elif strategy_name == "conservative_threshold":
        strategy = get_fusion_strategy(
            "conservative_threshold",
            meta_threshold=config.CT_META_THRESHOLD,
            visual_threshold=config.CT_VISUAL_THRESHOLD,
        )
    elif strategy_name == "bayesian":
        strategy = get_fusion_strategy(
            "bayesian",
            prior=config.BAYES_PRIOR,
            decision_threshold=config.BAYES_THRESHOLD,
        )
    else:
        raise ValueError(f"Unknown fusion strategy '{strategy_name}'")

    fusion_result = strategy.fuse(
        metadata_ai_prob=meta_result.ai_probability,
        visual_ai_prob=visual_ai_prob,
        cropped_visual_ai_prob=cropped_visual_ai_prob,
    )

    # Conditional deepfake analysis
    deepfake_result_data = None
    if fusion_result.is_ai:
        deepfake_result = deepfake_classifier.predict(pil_image)
        da = deepfake_result.get("deepfake_analysis")
        
        is_deepfake = da.get("is_deepfake", False) if da else False
        has_face = da.get("has_face", False) if da else False
        has_place = da.get("has_place", False) if da else False

        if is_deepfake:
            verdict = "Potential Deepfake"
            verdict_type = "deepfake"
        else:
            verdict = "Likely AI Generated"
            verdict_type = "ai_generated"

        deepfake_result_data = {
            "has_face": has_face,
            "has_place": has_place,
            "face_analysis": da.get("face_analysis") if da else None,
            "landmark_analysis": da.get("landmark_analysis") if da else None,
        }
    else:
        verdict = f"Likely Real (confidence: {1 - fusion_result.ai_probability:.2%})"
        verdict_type = "real"

    meta_features = meta_result.features
    meta_features_dict = {
        "has_make": meta_features.has_make,
        "has_model": meta_features.has_model,
        "has_lens_model": meta_features.has_lens_model,
        "has_makernote": meta_features.has_makernote,
        "has_gps": meta_features.has_gps,
        # Raw C2PA marker only — cryptographic validation lives on
        # `provenance` below. The old `has_c2pa` alias is kept so
        # older frontend builds keep rendering while they migrate.
        "has_c2pa_marker": meta_features.has_c2pa_marker,
        "has_c2pa": meta_features.has_c2pa_marker,
        "has_ai_claim": meta_features.has_ai_claim,
        "has_camera_claim": meta_features.has_camera_claim,
        "has_edit_claim": meta_features.has_edit_claim,
        "unverified_ai_provider_hints": meta_features.unverified_ai_provider_hints,
        "suspicious_only_software_tags": meta_features.suspicious_only_software_tags,
        "suspicious_perfect_timestamp": meta_features.suspicious_perfect_timestamp,
    }

    # Watermark evidence (Adobe TrustMark, scheme-specific). Runs on
    # the original bytes we already have, so no re-decoding into a
    # possibly resized PIL image touches the pixels TrustMark reads.
    # NOT fed into fusion — see run_analysis_pipeline docstring.
    watermark_dict = _build_watermark_payload(watermark_detector, file_data)

    # Provenance is a SEPARATE evidence object — the fusion formula in
    # this task is intentionally unchanged, so this block reports the
    # C2PA validator's findings without folding them into P(AI)_m.
    provenance = meta_result.provenance
    provenance_dict = {
        "status": provenance.status.value,
        "manifest_found": provenance.manifest_found,
        "validation_passed": provenance.validation_passed,
        "signer_trusted": provenance.signer_trusted,
        "validation_state": provenance.validation_state,
        "validation_errors": provenance.validation_errors,
        "origin_claim": provenance.origin_claim.value,
        "has_ai_generation_assertion": provenance.has_ai_generation_assertion,
        "has_ai_manipulation_assertion": provenance.has_ai_manipulation_assertion,
        "claim_generator": provenance.claim_generator,
        "software_agents": provenance.software_agents,
        "actions": provenance.actions,
        "digital_source_types": provenance.digital_source_types,
        "rationale": provenance.rationale,
    }

    return {
        "verdict": verdict,
        "verdict_type": verdict_type,
        "fusion": {
            "probability": fusion_result.ai_probability,
            "is_ai": fusion_result.is_ai,
            "strategy_name": fusion_result.formula_name,
            "explanation": fusion_result.explanation,
        },
        "metadata": {
            "probability": meta_result.ai_probability,
            "decision": meta_result.decision,
            "rationale": meta_result.rationale,
            "features": meta_features_dict,
        },
        "provenance": provenance_dict,
        "watermark": watermark_dict,
        "visual": {
            "probability": visual_ai_prob,
            "prediction": visual_result["prediction"],
            "confidence": visual_result["confidence"],
            "all_scores": visual_result["all_scores"],
            "gradcam_b64": visual_gradcam_b64,
        },
        "cropped_visual": {
            "probability": cropped_visual_ai_prob,
            "prediction": cropped_visual_result["prediction"] if cropped_visual_result else None,
            "confidence": cropped_visual_result["confidence"] if cropped_visual_result else None,
            "all_scores": cropped_visual_result["all_scores"] if cropped_visual_result else None,
            "cropped_face_b64": cropped_face_b64,
            "gradcam_b64": cropped_gradcam_b64 if cropped_visual_result else None,
        } if cropped_visual_result else None,
        "deepfake": deepfake_result_data,
    }


def find_free_port():
    for port in range(5000, 6000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except socket.error:
                continue
    raise RuntimeError("Could not find an available port in the 5000-6000 range.")


class PipelineHTTPServer(HTTPServer):
    def __init__(
        self,
        server_address,
        RequestHandlerClass,
        visual_classifier,
        deepfake_classifier,
        watermark_detector=None,
    ):
        self.visual_classifier = visual_classifier
        self.deepfake_classifier = deepfake_classifier
        # Optional and cheap to attach — the detector loads its model
        # only on the first watermark analysis, never at startup.
        self.watermark_detector = watermark_detector
        super().__init__(server_address, RequestHandlerClass)


def create_server(visual_classifier, deepfake_classifier, watermark_detector=None):
    """Binds a PipelineHTTPServer on a free localhost port and returns (server, url)."""
    port = find_free_port()
    server = PipelineHTTPServer(
        ('127.0.0.1', port),
        PipelineRequestHandler,
        visual_classifier,
        deepfake_classifier,
        watermark_detector=watermark_detector,
    )
    return server, f"http://127.0.0.1:{port}"


def start_server_thread(visual_classifier, deepfake_classifier, watermark_detector=None):
    global _active_server, _server_thread

    if _active_server is not None:
        print("Stopping existing web server...")
        _active_server.shutdown()
        _active_server.server_close()
        _active_server = None
        if _server_thread is not None:
            _server_thread.join()
            _server_thread = None

    server, url = create_server(visual_classifier, deepfake_classifier, watermark_detector=watermark_detector)
    _active_server = server

    def serve():
        server.serve_forever()

    _server_thread = threading.Thread(target=serve, daemon=True)
    _server_thread.start()

    print(f"\nSeeing through Deepfakes web interface is live!")
    print(f"URL: {url}")
    print("Opening browser window automatically...")
    webbrowser.open(url)
    return url


@atexit.register
def stop_server():
    global _active_server
    if _active_server is not None:
        print("Shutting down active web server...")
        _active_server.shutdown()
        _active_server.server_close()
        _active_server = None


if __name__ == "__main__":
    print("Starting in Standalone mode. Initializing models...")
    from src.genai_detection.visual_module.visual_classifier import CommunityForensicsClassifier, COMMFOR_MODEL_384
    from src.deepfake_detection.deepfake_classifier import DeepfakeClassifier

    # Canonical visual backbone (David, 2026-08-12): OwensLab/commfor-model-384,
    # official Community Forensics weights (Park & Owens, arXiv:2411.04125),
    # used out of the box with no fine-tuning delta. Chosen over the run_01
    # fine-tuned ViT for generalisation to unseen generators — see
    # run_07_visual_backend_comparison_eval_results.json: pooled AUC 0.994 vs
    # 0.761 on the 21-generator external Community Forensics eval set (the
    # fine-tuned model wins only in-distribution, AUC 0.921 vs 0.822, which is
    # not the property that matters long-term). VISUAL_ACCURACY in config.py
    # was updated to match (see that file's docstring).
    deepfake_index_path = PROJECT_ROOT / "src" / "deepfake_detection" / "models" / "landmarks_index.faiss"
    deepfake_meta_path  = PROJECT_ROOT / "src" / "deepfake_detection" / "models" / "landmarks_metadata.json"

    print(f"Loading visual classifier ({COMMFOR_MODEL_384})...")
    visual_model = CommunityForensicsClassifier(repo_id=COMMFOR_MODEL_384)

    print("Loading deepfake classifier...")
    deepfake_model = DeepfakeClassifier(
        index_path=str(deepfake_index_path),
        metadata_path=str(deepfake_meta_path),
    )

    # Attach the TrustMark watermark detector. Construction is cheap:
    # no model weights load until the first watermark analysis request,
    # so a missing/unreachable trustmark package cannot block startup.
    watermark_model = TrustMarkDetector()

    server, url = create_server(visual_model, deepfake_model, watermark_detector=watermark_model)
    print(f"\nStandalone server running at {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping standalone server...")
        server.shutdown()
        server.server_close()
