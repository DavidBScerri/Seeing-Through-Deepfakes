"""
Occlusion-based Saliency Analysis for Landmark Retrieval
========================================================

Investigates which pixels influence the DINOv2 + FAISS landmark 
recognition confidence, using a sliding-window occlusion sensitivity map.

Usage:
    Standalone:   python gradcam_landmark_analysis.py
    Notebook:     %run gradcam_landmark_analysis.py
    Or import:    from gradcam_landmark_analysis import run_analysis
"""

from __future__ import annotations

import os
import sys
import glob
import math
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Bootstrap the project root so the package import below works both when
# imported as src.deepfake_module.gradcam_landmark_analysis and when run
# standalone / via %run from inside src/deepfake_module (same pattern as app.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.deepfake_module.deepfake_classifier import LandmarkIndex

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

def _get_landmark_score(landmark_index: LandmarkIndex, image: Image.Image, target_label: str, top_k: int = 50) -> float:
    """Runs landmark inference and returns the confidence score for the target label."""
    res = landmark_index.search(image, top_k=top_k, similarity_threshold=0.0)
    
    if res["label"] in ["None", "N/A"]:
        return 0.0
        
    all_matches = res.get("all_matches", {})
    if target_label in all_matches:
        sims = all_matches[target_label]
        return sum(sims) / len(sims)
        
    return 0.0


def compute_occlusion_saliency(
    landmark_index: LandmarkIndex, 
    image: Image.Image,
    target_label: str,
    baseline_score: float,
    patch_size: int = 64,
    stride: int = 32
) -> np.ndarray:
    """
    Input-space saliency map via Occlusion Sensitivity.
    
    Systematically occludes patches of the image (with mean color) and measures
    the drop in landmark confidence for the target_label.
    """
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
        
    mean_color = np.mean(img_rgb, axis=(0, 1)).astype(np.uint8)
    
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
            img_occ = img_rgb.copy()
            img_occ[y_start:y_end, x_start:x_end] = mean_color
            img_occ_pil = Image.fromarray(img_occ)
            
            # Score
            occ_score = _get_landmark_score(landmark_index, img_occ_pil, target_label)
            
            # Drop in confidence (importance of this patch)
            drop = max(0.0, baseline_score - occ_score)
            
            saliency[y_start:y_end, x_start:x_end] += drop
            counts[y_start:y_end, x_start:x_end] += 1.0
            
    counts[counts == 0] = 1.0
    saliency_map = saliency / counts
    return saliency_map


# ─── Visualisation ─────────────────────────────────────────────────────────────

def analyse_and_visualise(
    landmark_index: LandmarkIndex,
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
    h, w = image_np.shape[:2]

    # ── Standard landmark detection ──
    baseline_res = landmark_index.search(image, top_k=10, similarity_threshold=0.5)
    
    label = baseline_res.get("label", "Unknown")
    confidence = baseline_res.get("confidence", 0.0)
    has_det = label not in ["Unknown", "None", "N/A"] and confidence >= 0.50

    if not has_det:
        saliency_map = np.zeros((h, w), dtype=np.float32)
    else:
        print("  ├─ Global Occlusion Saliency …")
        # Use relatively large patches for landmark saliency to speed it up
        patch_size = max(32, int(min(w, h) / 8))
        stride = max(16, int(patch_size / 2))
        saliency_map = compute_occlusion_saliency(
            landmark_index, image, target_label=label, baseline_score=confidence, 
            patch_size=patch_size, stride=stride
        )

    # ── Build figure ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    title_label = label if len(label) < 40 else label[:37] + "..."
    fig.suptitle(
        f"{fname}\nLandmark: {title_label} | Conf: {confidence:.4f}",
        fontsize=14, fontweight="bold", y=1.05,
    )

    # Panel 1 — Original Image
    ax = axes[0]
    ax.imshow(image_np)
    ax.set_title("Original Image", fontsize=11)
    ax.axis("off")

    # Panel 2 — Global Saliency
    ax = axes[1]
    if has_det:
        ax.imshow(_overlay_heatmap(image_np, saliency_map, alpha=0.5, cmap="hot"))
        ax.set_title(
            "Landmark Occlusion Saliency\n"
            "(Which pixels influence the landmark score most)",
            fontsize=11,
        )
    else:
        ax.imshow(image_np)
        ax.text(
            0.5, 0.5,
            "No high-confidence\nlandmark detected",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="white",
            bbox=dict(facecolor="red", alpha=0.7, pad=8),
        )
        ax.set_title("Landmark Saliency (No detection)", fontsize=11)
    ax.axis("off")

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"landmark_saliency_{stem}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✅  Saved → {out_path}")

    return {
        "filename": fname,
        "landmark_label": label,
        "landmark_confidence": confidence,
        "has_detection": has_det,
        "saliency_max": float(saliency_map.max()) if saliency_map.size > 0 else 0.0,
    }


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
        output_dir = os.path.join(base, "outputs", "landmark_saliency_analysis")

    sample_dir = os.path.abspath(sample_dir)
    output_dir = os.path.abspath(output_dir)
    
    models_dir = os.path.join(base, "models")
    index_path = os.path.join(models_dir, "landmarks_index.faiss")
    metadata_path = os.path.join(models_dir, "landmarks_metadata.json")

    print("=" * 70)
    print("  Occlusion Saliency Analysis for Landmark Retrieval")
    print("=" * 70)

    print("\n🔧  Loading LandmarkIndex …")
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print(f"Error: Landmark index or metadata not found in {models_dir}")
        print("Please run initialise_index.py first.")
        return []
        
    landmark_index = LandmarkIndex(
        model_name="facebook/dinov2-base",
        index_path=index_path,
        metadata_path=metadata_path
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
        res = analyse_and_visualise(landmark_index, img_path, output_dir)
        results.append(res)

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    # Using format with 35 chars for name and label to make it fit well
    print(f"\n  {'Image':<30} {'Landmark Conf':>15}  {'Label':<35}")
    print("  " + "-" * 82)
    for r in results:
        marker = "⚠️" if r["has_detection"] else "  "
        lbl = r['landmark_label']
        lbl_trunc = lbl if len(lbl) < 33 else lbl[:30] + "..."
        fname = r['filename']
        fname_trunc = fname if len(fname) < 28 else fname[:25] + "..."
        print(
            f"{marker} {fname_trunc:<30} "
            f"{r['landmark_confidence']:>15.4f}  "
            f"{lbl_trunc:<35}"
        )

    print(f"\n✅  All visualisations saved to: {output_dir}")
    return results


def main():
    """CLI entry point."""
    run_analysis()


if __name__ == "__main__":
    main()
