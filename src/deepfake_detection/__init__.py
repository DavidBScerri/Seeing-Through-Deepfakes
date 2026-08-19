"""
Deepfake detection subsystem.

Sits downstream of the generative-AI fusion verdict (``src.genai_detection``).
Only runs on images the fusion stage flags as AI-generated
(proportionality gate) — YuNet face detection + DINOv2/FAISS landmark
retrieval + occlusion saliency for explainability.
"""

from .deepfake_classifier import DeepfakeClassifier

__all__ = [
    "DeepfakeClassifier",
]
