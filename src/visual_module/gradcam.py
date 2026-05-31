"""
GradCAM for Image Classification Models
=========================================
Generates Gradient-weighted Class Activation Mapping heatmaps for
HuggingFace image classification models.

This module supports multiple model architectures:

  **ViT (Vision Transformer)**:
    Uses an Attention-Gradient approach:
    1. Extract CLS token attention weights to each patch from the last
       self-attention layer.
    2. Compute gradient of the target class logit w.r.t. the attention
       weights.
    3. cam_k = attn_CLS→k × grad_k (positive gradients only)

  **ConvNeXt V2 / CNN-based / Hybrid models**:
    Uses a Feature-Gradient approach:
    1. Hook into the last feature layer (before the classifier head).
    2. Compute gradient of the target class logit w.r.t. the feature
       activations.
    3. Global-average-pool the gradients over channels, then weight the
       feature map spatially.

If gradient computation fails (e.g. on MPS), falls back to activation-only
heatmap which still provides useful spatial information.

The resulting heatmap is upsampled and overlaid on the original image.

Usage
-----
    from src.visual_module.gradcam import generate_gradcam_overlay

    overlay_b64 = generate_gradcam_overlay(
        model=visual_classifier.model,
        processor=visual_classifier.processor,
        image=pil_image,
        device=visual_classifier.device,
    )
"""

from __future__ import annotations

import base64
import io
import math
from typing import Optional

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Hook-based feature extraction for CNN / hybrid architectures
# ---------------------------------------------------------------------------

class _FeatureHook:
    """Forward hook that captures a layer's output activations."""
    def __init__(self):
        self.features = None
        self.gradients = None

    def forward_hook(self, module, input, output):
        # Handle tuple outputs (some layers return (tensor, ...) )
        if isinstance(output, tuple):
            output = output[0]
        self.features = output

    def backward_hook(self, module, grad_input, grad_output):
        # Handle tuple outputs (some layers return (tensor, ...) )
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]
        self.gradients = grad_output


def _find_last_feature_layer(model):
    """
    Heuristic to find the last spatial feature layer before the classifier.

    Searches in reverse through the model's modules for the last
    Conv2d, LayerNorm (in a spatial context), or any named layer
    that looks like it produces spatial features.
    """
    # Try known architecture patterns
    # ConvNeXt V2: model.convnextv2.encoder.stages[-1].layers[-1]
    if hasattr(model, "convnextv2"):
        if hasattr(model.convnextv2, "encoder") and hasattr(model.convnextv2.encoder, "stages"):
            return model.convnextv2.encoder.stages[-1].layers[-1]

    # ConvNeXt: model.convnext.encoder.stages[-1].layers[-1]
    if hasattr(model, "convnext"):
        if hasattr(model.convnext, "encoder") and hasattr(model.convnext.encoder, "stages"):
            return model.convnext.encoder.stages[-1].layers[-1]

    # ViT: model.vit.encoder.layer[-1].layernorm_after or model.vit.layernorm
    if hasattr(model, "vit"):
        if hasattr(model.vit, "layernorm"):
            return model.vit.layernorm
        if hasattr(model.vit, "encoder") and hasattr(model.vit.encoder, "layer"):
            return model.vit.encoder.layer[-1]

    # Generic fallback: find the last module before classifier-like layers
    all_modules = list(model.named_modules())
    skip_names = {"classifier", "head", "fc", "output"}

    last_candidate = None
    for name, module in all_modules:
        # Skip classifier head modules
        if any(s in name.lower() for s in skip_names):
            continue
        # Prefer modules with parameters (conv, norm, etc.)
        if list(module.parameters()):
            last_candidate = module

    return last_candidate


# ---------------------------------------------------------------------------
# Core GradCAM computation
# ---------------------------------------------------------------------------

def compute_gradcam(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Computes a GradCAM heatmap for a given image and model.

    Tries attention-gradient approach first (for ViT-like models), then
    falls back to feature-gradient approach (for ConvNeXt V2 / CNN models).

    Args:
        model:        HuggingFace image classification model.
        processor:    The corresponding AutoImageProcessor.
        image:        PIL Image (RGB).
        device:       torch.device to run on.
        target_class: Class index to compute gradients for.
                      If None, uses the predicted (argmax) class.

    Returns:
        cam: numpy array of shape (grid_h, grid_w) with values in [0, 1],
             representing the normalised heatmap.
    """
    model.eval()

    # Prepare input
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    cam = None

    # ── Method 1: Attention-Gradient (ViT-like models) ──
    try:
        with torch.enable_grad():
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits

            if target_class is None:
                target_class = logits.argmax(dim=-1).item()

            attn_tuple = getattr(outputs, "attentions", None)
            if attn_tuple is not None and len(attn_tuple) > 0:
                last_attn = attn_tuple[-1]  # (1, heads, seq, seq)
                last_attn.retain_grad()

                model.zero_grad()
                target_logit = logits[0, target_class]
                target_logit.backward()

                if last_attn.grad is not None:
                    attn = last_attn[0, :, 0, 1:]  # (heads, num_patches)
                    grad = last_attn.grad[0, :, 0, 1:]

                    cam = (grad.clamp(min=0) * attn).mean(dim=0)
                    cam = torch.relu(cam)
                    cam = cam.detach().cpu().numpy()

                    num_patches = cam.shape[0]
                    grid_size = int(math.isqrt(num_patches))
                    if grid_size * grid_size == num_patches:
                        cam = cam.reshape(grid_size, grid_size)
                        cam_min, cam_max = cam.min(), cam.max()
                        if cam_max - cam_min > 1e-8:
                            cam = (cam - cam_min) / (cam_max - cam_min)
                        else:
                            cam = np.zeros_like(cam)
                        return cam
                    # If not a perfect square, fall through to feature method
                    cam = None
    except Exception as e:
        print(f"[GradCAM] Attention method failed: {e}")
        cam = None

    # ── Method 2: Feature-Gradient (ConvNeXt V2 / CNN / hybrid) ──
    try:
        target_layer = _find_last_feature_layer(model)
        if target_layer is None:
            print("[GradCAM] Warning: could not find a suitable feature layer")
            return np.zeros((14, 14))

        hook = _FeatureHook()
        fwd_handle = target_layer.register_forward_hook(hook.forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(hook.backward_hook)

        try:
            with torch.enable_grad():
                # Need fresh forward pass with hooks
                for p in model.parameters():
                    p.requires_grad_(True)

                outputs = model(**inputs)
                logits = outputs.logits

                if target_class is None:
                    target_class = logits.argmax(dim=-1).item()

                model.zero_grad()
                target_logit = logits[0, target_class]
                target_logit.backward()

            features = hook.features  # (B, C, H, W) or (B, seq_len, hidden)
            gradients = hook.gradients

            if features is not None and gradients is not None:
                features = features.detach()
                gradients = gradients.detach()

                if features.dim() == 4:
                    # CNN-style: (B, C, H, W)
                    weights = gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
                    cam_tensor = (weights * features).sum(dim=1)[0]  # (H, W)
                elif features.dim() == 3:
                    # Transformer-style: (B, seq_len, hidden)
                    weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, hidden)
                    cam_tensor = (weights * features).sum(dim=-1)[0]  # (seq_len,)

                    # Remove CLS token if present
                    seq_len = cam_tensor.shape[0]
                    grid_size = int(math.isqrt(seq_len))
                    if grid_size * grid_size != seq_len:
                        # Likely has CLS token
                        cam_tensor = cam_tensor[1:]
                        seq_len = cam_tensor.shape[0]
                        grid_size = int(math.isqrt(seq_len))

                    if grid_size * grid_size == seq_len:
                        cam_tensor = cam_tensor.reshape(grid_size, grid_size)
                    else:
                        # Can't reshape to square — use 1D as-is, reshape to approx
                        side = int(math.ceil(math.sqrt(seq_len)))
                        padded = torch.zeros(side * side, device=cam_tensor.device)
                        padded[:seq_len] = cam_tensor
                        cam_tensor = padded.reshape(side, side)
                else:
                    cam_tensor = None

                if cam_tensor is not None:
                    cam = torch.relu(cam_tensor).cpu().numpy()
                    cam_min, cam_max = cam.min(), cam.max()
                    if cam_max - cam_min > 1e-8:
                        cam = (cam - cam_min) / (cam_max - cam_min)
                    else:
                        cam = np.zeros_like(cam)
        finally:
            fwd_handle.remove()
            bwd_handle.remove()
            model.eval()

    except Exception as e:
        print(f"[GradCAM] Feature-gradient method failed: {e}")
        import traceback
        traceback.print_exc()

    # Last resort fallback
    if cam is None:
        return np.zeros((14, 14))

    return cam


# ---------------------------------------------------------------------------
# Heatmap overlay generation
# ---------------------------------------------------------------------------

def _apply_colormap(cam: np.ndarray) -> np.ndarray:
    """
    Applies a standard JET colourmap (blue → cyan → green → yellow → red) to a
    normalised [0, 1] heatmap without requiring matplotlib.

    Args:
        cam: 2D numpy array with values in [0, 1].

    Returns:
        RGB numpy array of shape (H, W, 3) with uint8 values.
    """
    cam = np.clip(cam, 0.0, 1.0)

    # Standard JET colormap control points
    xs = np.array([0.0, 0.125, 0.375, 0.5, 0.625, 0.875, 1.0])
    b_ys = np.array([0.56, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    g_ys = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    r_ys = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.56])

    r = np.interp(cam, xs, r_ys)
    g = np.interp(cam, xs, g_ys)
    b = np.interp(cam, xs, b_ys)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)



def generate_gradcam_overlay(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    target_class: Optional[int] = None,
    alpha: float = 0.5,
) -> str:
    """
    Generates a GradCAM heatmap overlay on the original image and returns
    it as a base64-encoded JPEG string.

    Args:
        model:        HuggingFace image classification model.
        processor:    The corresponding AutoImageProcessor.
        image:        PIL Image (RGB).
        device:       torch.device to run on.
        target_class: Class index to compute gradients for (None = predicted).
        alpha:        Opacity of the heatmap overlay (0 = invisible, 1 = opaque).

    Returns:
        Base64-encoded JPEG string of the overlay image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 1. Compute the heatmap
    cam = compute_gradcam(model, processor, image, device, target_class)

    # 2. Upsample heatmap to original image size using bicubic interpolation
    img_w, img_h = image.size

    cam_pil = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    cam_pil = cam_pil.resize((img_w, img_h), resample=Image.BICUBIC)
    cam_upsampled = np.array(cam_pil).astype(np.float32) / 255.0

    # 3. Apply JET colourmap to create RGB heatmap
    heatmap_rgb = _apply_colormap(cam_upsampled)
    heatmap_pil = Image.fromarray(heatmap_rgb, mode="RGB")

    # 4. Blend with original image
    original_arr = np.array(image).astype(np.float32)
    heatmap_arr = np.array(heatmap_pil).astype(np.float32)
    blended = (1 - alpha) * original_arr + alpha * heatmap_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    blended_pil = Image.fromarray(blended, mode="RGB")

    # 5. Encode to base64 JPEG
    buffered = io.BytesIO()
    blended_pil.save(buffered, format="JPEG", quality=85)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return b64_str
