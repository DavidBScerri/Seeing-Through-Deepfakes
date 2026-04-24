from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try importing image libraries for heuristics
try:
    from PIL import Image
    import numpy as np
    HAS_PIL_NP = True
except ImportError:
    HAS_PIL_NP = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class WatermarkResult:
    """Result of a watermark detection operation."""
    detector_name: str
    status: Literal["detected", "not_detected", "unsupported", "error"]
    confidence: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class WatermarkDetector:
    """Base adapter interface for watermark detectors."""
    
    def __init__(self, name: str):
        self.name = name

    def detect(self, image_path_or_bytes: Union[str, Path, bytes]) -> WatermarkResult:
        """
        Detects watermarks in the provided image.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the detect method.")


class SynthIDAdapter(WatermarkDetector):
    """
    Placeholder adapter for Google DeepMind's SynthID.
    As the backend is proprietary and not publicly available,
    this currently returns 'unsupported'.
    """
    def __init__(self):
        super().__init__("SynthID")

    def detect(self, image_path_or_bytes: Union[str, Path, bytes]) -> WatermarkResult:
        return WatermarkResult(
            detector_name=self.name,
            status="unsupported",
            details={"message": "SynthID API backend is not configured or publicly available."}
        )


class C2PAWatermarkAdapter(WatermarkDetector):
    """
    Placeholder adapter for C2PA or provenance-based signals
    that might be embedded as invisible watermarks.
    
    Note: Do not duplicate standard C2PA metadata extraction here. 
    This is specifically for watermark-based robust signals if supported.
    """
    def __init__(self):
        super().__init__("C2PA_Watermark")

    def detect(self, image_path_or_bytes: Union[str, Path, bytes]) -> WatermarkResult:
        return WatermarkResult(
            detector_name=self.name,
            status="unsupported",
            details={"message": "C2PA robust watermark signal extraction is not configured."}
        )


class InvisibleWatermarkAdapter(WatermarkDetector):
    """
    Generic placeholder for robust/invisible watermark detectors
    (e.g., IMAGINE, Stable Signature, or other frequency-domain watermarks).
    """
    def __init__(self):
        super().__init__("Invisible_Watermark")

    def detect(self, image_path_or_bytes: Union[str, Path, bytes]) -> WatermarkResult:
        return WatermarkResult(
            detector_name=self.name,
            status="unsupported",
            details={"message": "No invisible watermark backend configured."}
        )


class VisibleWatermarkHeuristic(WatermarkDetector):
    """
    A simple heuristic to detect potential visible watermarks.
    - Uses an optional OCR check (disabled by default).
    - Uses basic edge/text-like region detection.
    
    WARNING: Do not rely on this for final AI detection. It is only a heuristic.
    """
    def __init__(self, use_ocr: bool = False):
        super().__init__("Visible_Watermark_Heuristic")
        self.use_ocr = use_ocr

    def detect(self, image_path_or_bytes: Union[str, Path, bytes]) -> WatermarkResult:
        details = {}
        confidence = 0.0
        status = "not_detected"

        # If bytes are passed, we would normally decode them.
        # For simplicity in this placeholder, if it's a path, we try to load it.
        if isinstance(image_path_or_bytes, (str, Path)):
            image_path = str(image_path_or_bytes)
            
            # --- 1. Optional OCR Placeholder ---
            if self.use_ocr:
                try:
                    import pytesseract
                    if HAS_PIL_NP:
                        img = Image.open(image_path)
                        text = pytesseract.image_to_string(img)
                        details["ocr_text_found"] = bool(text.strip())
                        if details["ocr_text_found"]:
                            details["ocr_sample"] = text.strip()[:50]
                            confidence += 0.3
                except Exception as e:
                    details["ocr_error"] = str(e)
            else:
                details["ocr_skipped"] = True
                
            # --- 2. Edge / Text-like region heuristic ---
            if HAS_CV2:
                try:
                    # Simple mock edge detection to find high frequency regions
                    # often associated with overlaid text/logos
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        edges = cv2.Canny(img, 100, 200)
                        edge_density = np.sum(edges > 0) / edges.size
                        details["edge_density"] = edge_density
                        
                        # Very naive heuristic: if edge density is in a specific range,
                        # it might contain dense logo/text. (Purely for demonstration)
                        if 0.05 < edge_density < 0.20:
                            confidence += 0.2
                            details["heuristic_note"] = "Edge density suggests possible overlaid graphic."
                except Exception as e:
                    details["cv2_error"] = str(e)
            else:
                details["heuristic_skipped"] = "OpenCV not installed"

        # Aggregate mock confidence
        if confidence > 0.4:
            status = "detected"
        
        return WatermarkResult(
            detector_name=self.name,
            status=status,
            confidence=confidence,
            details=details
        )


class WatermarkAnalyzer:
    """
    Coordinator class to run all configured watermark detectors
    and return an aggregated set of results.
    """
    def __init__(self, include_visible_heuristic: bool = True, use_ocr: bool = False):
        self.detectors: list[WatermarkDetector] = [
            SynthIDAdapter(),
            C2PAWatermarkAdapter(),
            InvisibleWatermarkAdapter(),
        ]
        if include_visible_heuristic:
            self.detectors.append(VisibleWatermarkHeuristic(use_ocr=use_ocr))

    def analyze(self, image_path_or_bytes: Union[str, Path, bytes]) -> list[WatermarkResult]:
        """
        Runs the provided image through all configured detectors.
        """
        results = []
        for detector in self.detectors:
            try:
                result = detector.detect(image_path_or_bytes)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running detector {detector.name}: {e}")
                results.append(WatermarkResult(
                    detector_name=detector.name,
                    status="error",
                    details={"error": str(e)}
                ))
        return results
