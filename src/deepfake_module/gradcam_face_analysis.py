"""
Occlusion-based Saliency Analysis for YuNet Face Detection
==========================================================

Investigates why YuNet sometimes assigns high face confidence to images
without faces, using model-agnostic visualization approaches:

1. **Occlusion Sensitivity Map**: A sliding-window approach that occludes
   patches of the image and measures how the maximum face detection
   confidence drops. This produces a spatial heatmap showing which pixels
   are most responsible for triggering a face detection.

2. **Detection Region Saliency**: For each detected face proposal, performs
   fine-grained occlusion within the bounding box to highlight the specific
   features driving the confidence score.

Usage:
    Standalone:   python gradcam_face_analysis.py
    Notebook:     %run gradcam_face_analysis.py
    Or import:    from gradcam_face_analysis import run_analysis
"""

from __future__ import annotations

import os
import glob
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# ─── Configuration ─────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _overlay_heatmap(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> np.ndarray:
    """Alpha-blend a heatmap (H×W, arbitrary range) onto an RGB image."""
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    colormap = plt.get_cmap(cmap)
    heatmap_rgb = colormap(heatmap_norm)[:, :, :3]     # drop alpha channel

    img_float = image_np.astype(np.float32) / 255.0
    blended = (1 - alpha) * img_float + alpha * heatmap_rgb
    return np.clip(blended, 0, 1)


# ─── Analysis Functions ────────────────────────────────────────────────────────

def _get_max_face_score(face_detector: cv2.FaceDetectorYN, img_bgr: np.ndarray) -> float:
    """Runs YuNet and returns the maximum face confidence score, or 0.0001 if none."""
    h, w = img_bgr.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(img_bgr)
    
    if faces is None or len(faces) == 0:
        return 0.0001
        
    scores = faces[:, 14]
    return float(np.clip(np.max(scores), 0.0001, 0.9999))


def compute_occlusion_saliency(
    face_detector: cv2.FaceDetectorYN, 
    image: Image.Image,
    patch_size: int = 32,
    stride: int = 16
) -> np.ndarray:
    """
    Input-space saliency map via Occlusion Sensitivity.
    
    Systematically occludes patches of the image (with mean color) and measures
    the drop in maximum face confidence.
    
    Returns:
        (H, W) numpy array of sensitivity values (larger means more important).
    """
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    baseline_score = _get_max_face_score(face_detector, img_bgr)
    
    # If no face is detected initially, the whole map is essentially 0
    if baseline_score < 0.01:
        return np.zeros((h, w), dtype=np.float32)
        
    mean_color = np.mean(img_bgr, axis=(0, 1)).astype(np.uint8)
    
    saliency = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    # Calculate grid
    y_steps = max(1, math.ceil((h - patch_size) / stride) + 1)
    x_steps = max(1, math.ceil((w - patch_size) / stride) + 1)
    
    for y_idx in range(y_steps):
        for x_idx in range(x_steps):
            y_start = min(y_idx * stride, h - patch_size if h > patch_size else 0)
            x_start = min(x_idx * stride, w - patch_size if w > patch_size else 0)
            y_end = min(y_start + patch_size, h)
            x_end = min(x_start + patch_size, w)
            
            # Create occluded image
            img_occ = img_bgr.copy()
            img_occ[y_start:y_end, x_start:x_end] = mean_color
            
            # Score
            occ_score = _get_max_face_score(face_detector, img_occ)
            
            # Drop in confidence (importance of this patch)
            drop = max(0.0, baseline_score - occ_score)
            
            saliency[y_start:y_end, x_start:x_end] += drop
            counts[y_start:y_end, x_start:x_end] += 1.0
            
    counts[counts == 0] = 1.0
    saliency_map = saliency / counts
    return saliency_map


def compute_detection_region_analysis(
    face_detector: cv2.FaceDetectorYN,
    image: Image.Image,
    detection_boxes: np.ndarray | None = None,
    detection_probs: np.ndarray | None = None,
) -> list[dict]:
    """
    Fine-grained occlusion sensitivity within detected face regions.
    
    For each face proposal, performs a denser occlusion scan just within
    that bounding box to see which specific features drive the confidence.
    
    Returns:
        List of dicts, each with:
            box             (x1, y1, x2, y2)
            confidence      Detection confidence
            saliency_crop   Raw sensitivity array for the crop
            saliency_full   (H, W) map mapped to original image
            crop_image      uint8 array of the region
    """
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # Get proposals if not provided
    if detection_boxes is None or len(detection_boxes) == 0:
        face_detector.setInputSize((w, h))
        # Lower threshold temporarily to find proposals
        old_thresh = face_detector.getScoreThreshold()
        face_detector.setScoreThreshold(0.3)
        _, faces = face_detector.detect(img_bgr)
        face_detector.setScoreThreshold(old_thresh)
        
        if faces is not None and len(faces) > 0:
            detection_boxes = []
            detection_probs = []
            for face in faces:
                fx, fy, fw, fh = face[:4]
                detection_boxes.append([fx, fy, fx + fw, fy + fh])
                detection_probs.append(face[14])
            detection_boxes = np.array(detection_boxes)
            detection_probs = np.array(detection_probs)

    if detection_boxes is None or len(detection_boxes) == 0:
        return []

    results: list[dict] = []
    
    # Analyze each detected box
    for i in range(len(detection_boxes)):
        box = detection_boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Clamp to bounds and add some padding to the analysis region
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        
        ax1, ay1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        ax2, ay2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        if ax2 - ax1 < 10 or ay2 - ay1 < 10:
            continue
            
        crop_rgb = img_rgb[ay1:ay2, ax1:ax2].copy()
        
        # Fine-grained occlusion on this region
        patch_size = max(4, int(min(ax2 - ax1, ay2 - ay1) / 8))
        stride = max(2, int(patch_size / 2))
        
        crop_pil = Image.fromarray(crop_rgb)
        
        # We need to evaluate the whole image but with occlusion only in the crop
        baseline_score = float(detection_probs[i])
        
        mean_color = np.mean(img_bgr[ay1:ay2, ax1:ax2], axis=(0, 1)).astype(np.uint8)
        
        saliency_crop = np.zeros((ay2 - ay1, ax2 - ax1), dtype=np.float32)
        counts = np.zeros((ay2 - ay1, ax2 - ax1), dtype=np.float32)
        
        y_steps = max(1, math.ceil(((ay2 - ay1) - patch_size) / stride) + 1)
        x_steps = max(1, math.ceil(((ax2 - ax1) - patch_size) / stride) + 1)
        
        for y_idx in range(y_steps):
            for x_idx in range(x_steps):
                y_start = min(y_idx * stride, (ay2 - ay1) - patch_size if (ay2 - ay1) > patch_size else 0)
                x_start = min(x_idx * stride, (ax2 - ax1) - patch_size if (ax2 - ax1) > patch_size else 0)
                y_end = min(y_start + patch_size, ay2 - ay1)
                x_end = min(x_start + patch_size, ax2 - ax1)
                
                # Occlude just this patch in the full image
                img_occ = img_bgr.copy()
                img_occ[ay1+y_start : ay1+y_end, ax1+x_start : ax1+x_end] = mean_color
                
                occ_score = _get_max_face_score(face_detector, img_occ)
                drop = max(0.0, baseline_score - occ_score)
                
                saliency_crop[y_start:y_end, x_start:x_end] += drop
                counts[y_start:y_end, x_start:x_end] += 1.0
                
        counts[counts == 0] = 1.0
        saliency_crop = saliency_crop / counts
        
        # Map back to full
        cam_full = np.zeros((h, w), dtype=np.float32)
        cam_full[ay1:ay2, ax1:ax2] = saliency_crop
        
        # Normalize the crop saliency for visualization
        if saliency_crop.max() > 0:
            saliency_crop = saliency_crop / saliency_crop.max()

        results.append({
            "box": (x1, y1, x2, y2),
            "confidence": baseline_score,
            "saliency_crop": saliency_crop,
            "saliency_full": cam_full,
            "crop_image": crop_rgb,
        })

    return results


# ─── Visualisation ─────────────────────────────────────────────────────────────

def analyse_and_visualise(
    face_detector: cv2.FaceDetectorYN,
    image_path: str,
    output_dir: str,
) -> dict:
    """
    Run occlusion sensitivity analysis on a single image and save a
    visualisation figure.
    """
    fname = os.path.basename(image_path)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # ── Standard face detection ──
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(img_bgr)
    
    has_det = faces is not None and len(faces) > 0

    boxes = []
    probs = []
    face_conf = 0.0001
    
    if has_det:
        for face in faces:
            fx, fy, fw, fh = face[:4]
            boxes.append([fx, fy, fx + fw, fy + fh])
            probs.append(float(np.clip(face[14], 0.0001, 0.9999)))
        boxes = np.array(boxes)
        probs = np.array(probs)
        face_conf = float(np.max(probs))

    # ── Analyses ──
    print("  ├─ Global Occlusion Saliency …")
    # Use larger patches for global saliency to speed it up
    patch_size = max(16, int(min(w, h) / 16))
    stride = max(8, int(patch_size / 2))
    saliency_map = compute_occlusion_saliency(face_detector, image, patch_size=patch_size, stride=stride)

    print("  └─ Face Region Saliency …")
    gc_results = compute_detection_region_analysis(face_detector, image, boxes, probs)

    # ── Build figure ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"{fname}\nYuNet Face Confidence: {face_conf:.4f}",
        fontsize=14, fontweight="bold", y=1.05,
    )

    # Panel 1 — Global Saliency
    ax = axes[0]
    ax.imshow(_overlay_heatmap(image_np, saliency_map, alpha=0.5, cmap="hot"))
    ax.set_title(
        "Global Occlusion Saliency\n"
        "(Which pixels influence face score most)",
        fontsize=11,
    )
    ax.axis("off")

    # Panel 2 — Region Saliency
    ax = axes[1]
    if gc_results:
        combined_gcam = np.zeros((h, w), dtype=np.float32)
        for gc in gc_results:
            combined_gcam = np.maximum(combined_gcam, gc["saliency_full"])
            
        ax.imshow(_overlay_heatmap(image_np, combined_gcam, alpha=0.5, cmap="jet"))
        n_props = len(gc_results)
        max_c = max(gc["confidence"] for gc in gc_results)
        ax.set_title(
            f"Face Region Features ({n_props} proposal(s), max conf: {max_c:.3f})\n"
            "(What features drive face confidence)",
            fontsize=11,
        )
        # Draw proposal boxes
        for gc in gc_results:
            bx = gc["box"]
            ax.add_patch(
                Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1],
                           linewidth=1.5, edgecolor="cyan",
                           facecolor="none", linestyle="--")
            )
    else:
        ax.imshow(image_np)
        ax.text(
            0.5, 0.5,
            "No face proposals\n(even with lowered thresholds)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="white",
            bbox=dict(facecolor="red", alpha=0.7, pad=8),
        )
        ax.set_title("Face Region Features\n(No proposals available)", fontsize=11)
    ax.axis("off")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"saliency_{stem}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✅  Saved → {out_path}")

    return {
        "filename": fname,
        "face_confidence": face_conf,
        "has_detection": has_det,
        "saliency_max": float(saliency_map.max()) if saliency_map.size > 0 else 0.0,
        "num_proposals": len(gc_results),
    }


def _get_yunet_model_path() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(module_dir, "models", "face_detection_yunet_2023mar.onnx")


# ─── Entry Points ──────────────────────────────────────────────────────────────

def run_analysis(
    sample_dir: str | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    """
    Run the analysis on all images in *sample_dir* and save
    visualisation figures to *output_dir*.
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()

    if sample_dir is None:
        sample_dir = os.path.join(base, "..", "..", "data", "sample_images")
    if output_dir is None:
        output_dir = os.path.join(base, "outputs", "saliency_analysis")

    sample_dir = os.path.abspath(sample_dir)
    output_dir = os.path.abspath(output_dir)

    print("=" * 70)
    print("  Occlusion Saliency Analysis for YuNet Face Detection")
    print("=" * 70)

    print("\n🔧  Loading YuNet …")
    yunet_path = _get_yunet_model_path()
    if not os.path.exists(yunet_path):
        print(f"Error: YuNet model not found at {yunet_path}")
        print("Please run deepfake_classifier.py first to download it.")
        return []
        
    face_detector = cv2.FaceDetectorYN.create(
        model=yunet_path,
        config="",
        input_size=(320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
        top_k=5000,
    )

    image_paths: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(sample_dir, ext)))
    image_paths = sorted(image_paths)

    print(f"📂  {len(image_paths)} images in {sample_dir}")
    print(f"💾  Output → {output_dir}\n")

    results: list[dict] = []
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        print(f"📷  {fname}")
        res = analyse_and_visualise(face_detector, img_path, output_dir)
        results.append(res)

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"\n  {'Image':<45} {'Face Conf':>10} {'Proposals':>10}")
    print("  " + "-" * 65)
    for r in results:
        marker = "⚠️" if r["face_confidence"] >= 0.5 and "-face" in r["filename"] else "  "
        print(
            f"{marker} {r['filename']:<45} "
            f"{r['face_confidence']:>10.4f} "
            f"{r['num_proposals']:>10}"
        )

    print(f"\n✅  All visualisations saved to: {output_dir}")
    return results


def main():
    """CLI entry point."""
    run_analysis()


if __name__ == "__main__":
    main()
