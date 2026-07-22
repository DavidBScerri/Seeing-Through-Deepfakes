import os
import torch
from transformers import AutoImageProcessor
import numpy as np
import cv2
import faiss
import json
from PIL import Image


# Pinned to a specific opencv_zoo commit and checksum so a moved/changed
# upstream file cannot silently alter face-detection behaviour between runs.
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def _sha256_of(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _get_yunet_model_path(model_path=None):
    """
    Returns the path to the YuNet ONNX model, downloading it (with checksum
    verification) if missing. An existing file is also verified on every
    call: a mismatch is reported loudly but the file is NOT deleted or
    replaced — committed model artefacts must never be overwritten
    automatically (see CLAUDE.md rules).
    """
    if model_path is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(module_dir, "models", "face_detection_yunet_2023mar.onnx")

    if not os.path.exists(model_path):
        import urllib.request
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Downloading YuNet model to {model_path} …")
        tmp_path = model_path + ".download"
        try:
            urllib.request.urlretrieve(_YUNET_URL, tmp_path)
            digest = _sha256_of(tmp_path)
            if digest != _YUNET_SHA256:
                raise RuntimeError(
                    f"YuNet download failed checksum verification: expected {_YUNET_SHA256}, got {digest}."
                )
            os.replace(tmp_path, model_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        print("Download complete (checksum verified).")
    else:
        digest = _sha256_of(model_path)
        if digest != _YUNET_SHA256:
            print(
                f"WARNING: YuNet model at {model_path} does not match the pinned checksum "
                f"(expected {_YUNET_SHA256}, got {digest}). Face-detection results may not "
                f"be reproducible against the reported figures. If this is unintentional, "
                f"delete the file so a verified copy is re-downloaded."
            )

    return model_path


class DeepfakeClassifier:
    def __init__(self,
                 landmark_model_name="facebook/dinov2-base",
                 index_path="models/landmarks_index.faiss",
                 metadata_path="models/landmarks_metadata.json"):
        """
        Initializes the Deepfake Classifier with sub-models for face detection
        and landmark retrieval.

        Args:
            landmark_model_name:  HuggingFace model ID for landmark embeddings (DINOv2).
            index_path:           Path to the FAISS index file.
            metadata_path:        Path to the landmark metadata JSON file.
        """
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print("Loading Face Detection model: YuNet (OpenCV FaceDetectorYN)")
        yunet_path = _get_yunet_model_path()
        # Initialise with a placeholder input size; updated per-image in predict_face
        self.face_detector = cv2.FaceDetectorYN.create(
            model=yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )

        print(f"Loading Landmark Retrieval model: {landmark_model_name}")
        self.landmark_index = LandmarkIndex(
            model_name=landmark_model_name,
            index_path=index_path,
            metadata_path=metadata_path,
            device=self.device
        )

    def predict_face(self, image):
        """
        Detects whether a face is present in the image using YuNet.

        Args:
            image: A PIL Image object.

        Returns:
            dict with keys ``label``, ``confidence``, and ``bbox``.
            Confidence is clamped to [0.0001, 0.9999] to avoid
            degenerate 0.0 or 1.0 values.
        """
        # Convert PIL → OpenCV BGR
        img_rgb = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # Update input size to match this image
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(img_bgr)

        if faces is None or len(faces) == 0:
            return {"label": "No Face", "confidence": 0.0001, "bbox": None}

        # YuNet output: each row is [x, y, w, h, ..., score] — score at index 14
        scores = faces[:, 14]
        idx = int(np.argmax(scores))
        face_certainty = float(np.clip(scores[idx], 0.0001, 0.9999))

        if face_certainty < 0.75:
            return {"label": "No Face", "confidence": round(face_certainty, 4), "bbox": None}

        # Convert (x, y, w, h) → (x1, y1, x2, y2)
        fx, fy, fw, fh = faces[idx][:4]
        bbox = [float(fx), float(fy), float(fx + fw), float(fy + fh)]
        return {"label": "Face Detected", "confidence": round(face_certainty, 4), "bbox": bbox}


    def predict_landmark(self, image, top_k=10, similarity_threshold=0.5):
        """
        Identifies landmarks using DINOv2 embeddings and FAISS retrieval.

        Args:
            image:                A PIL Image object.
            top_k:                Number of nearest neighbors to retrieve.
            similarity_threshold: Minimum similarity score to consider a match.

        Returns:
            dict with keys ``label``, ``confidence``, and optionally ``top_matches``.
        """
        return self.landmark_index.search(image, top_k=top_k, similarity_threshold=similarity_threshold)

    def predict(self, image):
        """
        Runs the full deepfake analysis (face + landmark sub-models).

        The proportionality gate lives with the caller: this module only
        runs on images already flagged AI-positive by decision fusion
        (``fusion_result.is_ai`` in app.py / the notebooks), so no internal
        visual-classifier gating happens here.

        Args:
            image: A PIL Image object.

        Returns:
            dict with keys ``visual_classification`` (always None, kept for
            backward compatibility) and ``deepfake_analysis``.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        face_res = self.predict_face(image)
        landmark_res = self.predict_landmark(image)

        has_face = face_res["confidence"] >= 0.75
        has_landmark = landmark_res["label"] not in ["Unknown", "None", "N/A"] and landmark_res.get("confidence", 0.0) >= 0.50

        return {
            "visual_classification": None,
            "deepfake_analysis": {
                "is_deepfake": has_face or has_landmark,
                "has_face": has_face,
                "has_place": has_landmark,
                "face_analysis": face_res,
                "landmark_analysis": landmark_res,
            },
        }


class LandmarkIndex:
    def __init__(self,
                 model_name="facebook/dinov2-base",
                 index_path="models/landmarks_index.faiss",
                 metadata_path="models/landmarks_metadata.json",
                 device=None):
        from transformers import AutoModel
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.load()
        else:
            print(f"Warning: Landmark index not found at {index_path}. "
                  "Please run the initialization script to build the FAISS index.")

    def load(self):
        print(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print("Landmark index loaded successfully.")

    def search(self, image, top_k=10, similarity_threshold=0.5):
        """Searches the FAISS index for the closest landmarks."""
        if self.index is None:
            return {"label": "N/A", "confidence": 0.0, "message": "Index not loaded."}

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            embedding_np = embedding.cpu().numpy().astype('float32')

        distances, indices = self.index.search(embedding_np, top_k)

        hits = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            label_idx = self.metadata["labels"][idx]
            label_name = self.metadata["class_names"][label_idx]
            if label_name not in hits:
                hits[label_name] = []
            hits[label_name].append(float(dist))

        if not hits:
            return {"label": "None", "confidence": 0.0}

        best_label = None
        max_avg_sim = -1.0
        for label, sims in hits.items():
            avg_sim = sum(sims) / len(sims)
            if avg_sim > max_avg_sim:
                max_avg_sim = avg_sim
                best_label = label

        if max_avg_sim < similarity_threshold:
            return {"label": "Unknown", "confidence": round(max_avg_sim, 4)}

        return {
            "label": best_label,
            "confidence": round(max_avg_sim, 4),
            "matches_count": len(hits[best_label]),
            "all_matches": hits
        }
