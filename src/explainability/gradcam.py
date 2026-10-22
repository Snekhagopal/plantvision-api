"""Grad-CAM for the MobileNetV2 classifier (adapted from src/training/gradcam_demo.py).

The heatmap shows which image regions most increased the score of the chosen
class. It does not show that the prediction is correct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from src.utils.preprocessing import leaf_mask

logger = logging.getLogger(__name__)


@dataclass
class GradCamResult:
    overlay: Image.Image | None
    heatmap: np.ndarray | None
    error: str | None = None
    leaf_attention: float | None = None   # share of heatmap weight on leaf pixels

    @property
    def ok(self) -> bool:
        return self.overlay is not None


def leaf_attention(heat: np.ndarray, image: Image.Image) -> float | None:
    """Fraction of Grad-CAM weight on leaf-coloured pixels (None if no leaf found)."""
    mask = leaf_mask(np.asarray(image.convert("RGB"), dtype=np.uint8))
    total = float(heat.sum())
    if not mask.any() or total <= 0:
        return None
    return float(heat[mask].sum() / total)


def mobilenet_target_layer(model: nn.Module) -> nn.Module:
    """Last conv block (Conv 1x1 -> 1280 channels, 4x4 at 128 px input)."""
    return model.features[-1]


def compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, class_index: int,
                    original: Image.Image, target_layer: nn.Module | None = None) -> GradCamResult:
    """Return an overlay at the original image's resolution, or an error message.

    Never raises: any failure is logged and reported in ``GradCamResult.error``
    so the rest of the results page still renders.
    """
    try:
        model.eval()
        layer = target_layer if target_layer is not None else mobilenet_target_layer(model)
        # The feature extractor's weights are frozen (requires_grad=False), so
        # the input must require grad or the target layer gets no gradients.
        x = input_tensor.detach().clone().requires_grad_(True)
        with torch.enable_grad(), GradCAM(model=model, target_layers=[layer]) as cam:
            heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(class_index))])[0]
        if not np.isfinite(heat).all():
            raise ValueError("Grad-CAM produced non-finite values")

        display = original.convert("RGB").copy()
        display.thumbnail((768, 768))
        heat_full = np.asarray(
            Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255)).resize(display.size, Image.BILINEAR),
            dtype=np.float32) / 255.0
        rgb = np.asarray(display, dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb, heat_full, use_rgb=True, image_weight=0.55)
        return GradCamResult(Image.fromarray(overlay), heat, leaf_attention=leaf_attention(heat_full, display))
    except Exception as exc:
        logger.exception("Grad-CAM failed")
        return GradCamResult(None, None, f"The heatmap could not be generated ({type(exc).__name__}).")
