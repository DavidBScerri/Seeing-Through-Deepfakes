"""
GradCAM & Saliency Analysis for MTCNN Face Detection
=====================================================

Investigates why MTCNN sometimes assigns high face confidence to images
without faces, using three complementary visualization approaches:

1. **P-Net Face Probability Heatmap**: Direct spatial output from MTCNN's
   first-stage detector showing which regions look "face-like" at multiple
   scales.

2. **Input Saliency Map**: Pixel-level gradient magnitude showing which
   parts of the input most influence the face detection score.

3. **O-Net GradCAM**: Gradient-weighted Class Activation Maps on MTCNN's
   final-stage network, showing what features in the detected region
   contribute to face confidence.

Usage:
    Standalone:   python gradcam_face_analysis.py
    Notebook:     %run gradcam_face_analysis.py
    Or import:    from gradcam_face_analysis import run_analysis
"""

from __future__ import annotations

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from PIL import Image
from facenet_pytorch import MTCNN

# ─── Configuration ─────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


# ─── Helper Functions ──────────────────────────────────────────────────────────

def _pil_to_nchw(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to NCHW float tensor (0-255 range)."""
    img_np = np.array(image).astype(np.float32)      # H, W, C
    return torch.tensor(img_np).unsqueeze(0).permute(0, 3, 1, 2)  # 1, C, H, W


def _normalize_mtcnn(tensor: torch.Tensor) -> torch.Tensor:
    """Apply MTCNN's standard normalization: (x - 127.5) / 128."""
    return (tensor - 127.5) * 0.0078125


def _resize_2d(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a 2-D float array using bilinear interpolation (via torch)."""
    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)   # 1, 1, H, W
    t = F.interpolate(t, size=(target_h, target_w),
                      mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


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

def compute_pnet_heatmap(mtcnn: MTCNN, image: Image.Image) -> np.ndarray:
    """
    Multi-scale face probability heatmap from P-Net.

    P-Net scans the image at every scale in MTCNN's image pyramid and
    outputs a spatial map of face probabilities.  We resize each map
    back to the original image size and average across scales.

    Returns:
        (H, W) numpy array of face probabilities.
    """
    w, h = image.size
    img_nchw = _pil_to_nchw(image)                    # 1, 3, H, W  (0-255)

    # Replicate MTCNN's image pyramid
    min_face = mtcnn.min_face_size
    factor = mtcnn.factor
    min_len = min(h, w)
    m = 12.0 / min_face
    min_len_scaled = min_len * m

    scales: list[float] = []
    while min_len_scaled >= 12:
        scales.append(m)
        min_len_scaled *= factor
        m *= factor

    combined = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    pnet = mtcnn.pnet
    pnet.eval()

    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        if hs < 12 or ws < 12:
            continue

        im_data = F.interpolate(img_nchw, size=(hs, ws),
                                mode="bilinear", align_corners=False)
        im_data = _normalize_mtcnn(im_data)

        with torch.no_grad():
            _, probs = pnet(im_data.to("cpu"))

        # probs: (1, 2, H', W') — channel 1 = P(face)
        face_prob = probs[0, 1].cpu().numpy()
        face_prob_resized = _resize_2d(face_prob, w, h)

        combined += face_prob_resized
        count += 1.0

    count[count == 0] = 1.0
    return combined / count


def compute_input_saliency(mtcnn: MTCNN, image: Image.Image) -> np.ndarray:
    """
    Input-space saliency map via P-Net back-propagation.

    Computes the gradient of the maximum face probability with respect
    to the input pixels.  The gradient magnitude (max across colour
    channels) highlights which pixels most influence the face score.

    Returns:
        (H, W) numpy array of gradient magnitudes.
    """
    w, h = image.size
    img_nchw = _pil_to_nchw(image)                    # 1, 3, H, W

    # Use the first (largest) scale from MTCNN's pyramid
    m = 12.0 / mtcnn.min_face_size
    hs, ws = int(np.ceil(h * m)), int(np.ceil(w * m))

    im_data = F.interpolate(img_nchw, size=(hs, ws),
                            mode="bilinear", align_corners=False)
    im_data = _normalize_mtcnn(im_data)
    im_data = im_data.detach().requires_grad_(True)

    pnet = mtcnn.pnet
    pnet.eval()

    # Forward (with gradient tracking)
    _, probs = pnet(im_data)
    face_score = probs[:, 1].max()
    face_score.backward()

    grad = im_data.grad.data.abs()                     # 1, 3, H', W'
    saliency = grad.squeeze(0).max(dim=0)[0].cpu().numpy()
    return _resize_2d(saliency, w, h)


def compute_onet_gradcam(
    mtcnn: MTCNN,
    image: Image.Image,
    detection_boxes: np.ndarray | None = None,
    detection_probs: np.ndarray | None = None,
) -> list[dict]:
    """
    GradCAM on MTCNN's O-Net for detected face proposals.

    For each face proposal, crops and resizes to 48×48, runs through
    O-Net, and computes a gradient-weighted class activation map on
    the last convolutional activation (`prelu4`, spatial size 3×3).

    If no proposals are available at the normal thresholds, a
    secondary MTCNN with lowered thresholds is used so we can still
    visualise what almost-passed regions look like.

    Returns:
        List of dicts, each with:
            box             (x1, y1, x2, y2) in original image coords
            confidence      O-Net face probability for this crop
            gradcam_crop    (3, 3) raw GradCAM array
            gradcam_full    (H, W) GradCAM mapped to original image
            crop_image      (48, 48, 3) uint8 array of the resized crop
    """
    w, h = image.size

    # Get proposals — fall back to lower thresholds if nothing found
    if detection_boxes is None or len(detection_boxes) == 0:
        try:
            low_mtcnn = MTCNN(
                keep_all=True, device="cpu",
                thresholds=[0.3, 0.4, 0.4],
                min_face_size=15,
            )
            detection_boxes, detection_probs = low_mtcnn.detect(image)
        except Exception:
            pass

    if detection_boxes is None or len(detection_boxes) == 0:
        return []

    onet = mtcnn.onet
    onet.eval()
    results: list[dict] = []

    for i in range(len(detection_boxes)):
        box = detection_boxes[i]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        # Crop → 48×48
        crop = image.crop((x1, y1, x2, y2))
        crop_48 = crop.resize((48, 48), Image.BILINEAR)
        crop_np = np.array(crop_48)

        crop_tensor = _pil_to_nchw(crop_48)            # 1, 3, 48, 48
        crop_tensor = _normalize_mtcnn(crop_tensor)
        crop_tensor = crop_tensor.detach().requires_grad_(True)

        # ── Hook storage ──
        act_store: dict = {}
        grad_store: dict = {}

        def _fwd(module, inp, out, _store=act_store):
            _store["feat"] = out

        def _bwd(module, grad_in, grad_out, _store=grad_store):
            _store["feat"] = grad_out[0]

        # Target: prelu4 — last conv activation in O-Net
        handle_fwd = onet.prelu4.register_forward_hook(_fwd)
        handle_bwd = onet.prelu4.register_full_backward_hook(_bwd)

        try:
            # ONet.forward returns (bbox_reg, landmarks, face_prob)
            _, _, face_prob = onet(crop_tensor)
            face_score = face_prob[0, 1]               # P(face)

            onet.zero_grad()
            if crop_tensor.grad is not None:
                crop_tensor.grad.zero_()
            face_score.backward()

            # GradCAM weights: global-average-pool of gradients per channel
            g = grad_store["feat"]                     # 1, 128, 3, 3
            a = act_store["feat"]                      # 1, 128, 3, 3
            weights = g.mean(dim=(2, 3), keepdim=True) # 1, 128, 1, 1
            cam = (weights * a).sum(dim=1, keepdim=True)
            cam = F.relu(cam)                          # only positive
            cam_np = cam.squeeze().detach().cpu().numpy()   # 3×3

            if cam_np.max() > 0:
                cam_np = cam_np / cam_np.max()

            # Map GradCAM back onto the full image
            cam_full = np.zeros((h, w), dtype=np.float32)
            cam_box = _resize_2d(cam_np, x2 - x1, y2 - y1)
            cam_full[y1:y2, x1:x2] = np.maximum(
                cam_full[y1:y2, x1:x2], cam_box
            )

            results.append({
                "box": (x1, y1, x2, y2),
                "confidence": float(face_score.item()),
                "gradcam_crop": cam_np,
                "gradcam_full": cam_full,
                "crop_image": crop_np,
            })

        finally:
            handle_fwd.remove()
            handle_bwd.remove()

    return results


# ─── Visualisation ─────────────────────────────────────────────────────────────

def analyse_and_visualise(
    mtcnn: MTCNN,
    image_path: str,
    output_dir: str,
) -> dict:
    """
    Run the full three-method analysis on a single image and save a
    2×2 visualisation figure.

    Panels:
        Top-left:     Original image with detection bounding boxes.
        Top-right:    P-Net face probability heatmap overlay.
        Bottom-left:  Input saliency map overlay.
        Bottom-right: O-Net GradCAM overlay (mapped back to full image).
    """
    fname = os.path.basename(image_path)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    w, h = image.size

    # ── Standard face detection ──
    boxes, probs = mtcnn.detect(image)
    has_det = boxes is not None and len(boxes) > 0

    if has_det:
        best = int(np.argmax(probs))
        face_conf = float(np.clip(probs[best], 0.0001, 0.9999))
    else:
        face_conf = 0.0001

    # ── Analyses ──
    print("  ├─ P-Net heatmap …")
    pnet_heatmap = compute_pnet_heatmap(mtcnn, image)

    print("  ├─ Input saliency …")
    saliency_map = compute_input_saliency(mtcnn, image)

    print("  └─ O-Net GradCAM …")
    gc_results = compute_onet_gradcam(mtcnn, image, boxes, probs)

    # ── Build figure ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"{fname}\nFace Confidence: {face_conf:.4f}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Panel 1 — Original + bounding boxes
    ax = axes[0, 0]
    ax.imshow(image_np)
    if has_det:
        for bx, pr in zip(boxes, probs):
            x1, y1, x2, y2 = bx
            colour = "lime" if pr >= 0.9 else ("orange" if pr >= 0.5 else "red")
            ax.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1,
                           linewidth=2, edgecolor=colour, facecolor="none")
            )
            ax.text(x1, y1 - 5, f"{pr:.3f}", color=colour,
                    fontsize=10, fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.6, pad=1))
    ax.set_title("Original + Face Detection", fontsize=12)
    ax.axis("off")

    # Panel 2 — P-Net heatmap
    ax = axes[0, 1]
    ax.imshow(_overlay_heatmap(image_np, pnet_heatmap, alpha=0.5))
    ax.set_title(
        "P-Net Face Probability Heatmap\n"
        "(Where MTCNN sees face-like patterns)",
        fontsize=11,
    )
    ax.axis("off")

    # Panel 3 — Input saliency
    ax = axes[1, 0]
    ax.imshow(_overlay_heatmap(image_np, saliency_map, alpha=0.5, cmap="hot"))
    ax.set_title(
        "Input Saliency Map\n"
        "(Which pixels influence face score most)",
        fontsize=11,
    )
    ax.axis("off")

    # Panel 4 — O-Net GradCAM
    ax = axes[1, 1]
    if gc_results:
        combined_gcam = np.zeros((h, w), dtype=np.float32)
        for gc in gc_results:
            combined_gcam = np.maximum(combined_gcam, gc["gradcam_full"])
        ax.imshow(_overlay_heatmap(image_np, combined_gcam, alpha=0.5))
        n_props = len(gc_results)
        max_c = max(gc["confidence"] for gc in gc_results)
        ax.set_title(
            f"O-Net GradCAM ({n_props} proposal(s), max conf: {max_c:.3f})\n"
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
        ax.set_title("O-Net GradCAM\n(No proposals available)", fontsize=11)
    ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(fname)[0]
    out_path = os.path.join(output_dir, f"gradcam_{stem}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✅  Saved → {out_path}")

    return {
        "filename": fname,
        "face_confidence": face_conf,
        "has_detection": has_det,
        "pnet_max": float(pnet_heatmap.max()),
        "num_proposals": len(gc_results),
    }


# ─── Entry Points ──────────────────────────────────────────────────────────────

def run_analysis(
    sample_dir: str | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    """
    Run the GradCAM analysis on all images in *sample_dir* and save
    visualisation figures to *output_dir*.

    Can be called from a notebook cell:
        from gradcam_face_analysis import run_analysis
        results = run_analysis()
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()

    if sample_dir is None:
        sample_dir = os.path.join(base, "..", "..", "data", "sample_images")
    if output_dir is None:
        output_dir = os.path.join(base, "outputs", "gradcam_analysis")

    sample_dir = os.path.abspath(sample_dir)
    output_dir = os.path.abspath(output_dir)

    print("=" * 70)
    print("  GradCAM & Saliency Analysis for MTCNN Face Detection")
    print("=" * 70)

    print("\n🔧  Loading MTCNN …")
    mtcnn = MTCNN(keep_all=True, device="cpu")

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
        res = analyse_and_visualise(mtcnn, img_path, output_dir)
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
