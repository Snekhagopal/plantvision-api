"""Checkpoint loading for the trained MobileNetV2 disease classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models.mobilenet import create_mobilenet

REQUIRED_KEYS = {"model_state_dict", "class_to_idx"}


class ModelLoadError(RuntimeError):
    """Raised when the checkpoint is missing or incompatible."""


@dataclass
class LoadedModel:
    model: nn.Module
    class_names: list[str]   # index -> training-time label
    image_size: int
    mean: list[float]
    std: list[float]


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    """Invert the training-time mapping, verifying indices are 0..N-1."""
    indices = sorted(class_to_idx.values())
    if indices != list(range(len(class_to_idx))):
        raise ModelLoadError("class_to_idx indices are not a contiguous 0..N-1 range")
    names = [""] * len(class_to_idx)
    for name, index in class_to_idx.items():
        names[index] = name
    return names


def load_mobilenet_checkpoint(path: Path, device: str | torch.device = "cpu") -> LoadedModel:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise ModelLoadError(f"Model checkpoint not found or empty: {path.name}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # corrupt file, pickle error, version mismatch
        raise ModelLoadError(f"Could not read checkpoint {path.name}: {exc}") from exc

    missing = REQUIRED_KEYS - set(checkpoint)
    if missing:
        raise ModelLoadError(f"Checkpoint {path.name} is missing keys: {sorted(missing)}")

    class_names = class_names_from_mapping(checkpoint["class_to_idx"])
    # pretrained=False: the checkpoint holds the full network, so no
    # ImageNet download is needed.
    model = create_mobilenet(num_classes=len(class_names), pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"Checkpoint weights do not match MobileNetV2 architecture: {exc}") from exc
    model.to(device).eval()

    return LoadedModel(
        model=model,
        class_names=class_names,
        image_size=int(checkpoint.get("image_size", 128)),
        mean=list(checkpoint.get("normalization_mean", [0.485, 0.456, 0.406])),
        std=list(checkpoint.get("normalization_std", [0.229, 0.224, 0.225])),
    )


def pooled_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate 1280-d MobileNetV2 features (same path as model.forward)."""
    f = model.features(x)
    f = F.adaptive_avg_pool2d(f, (1, 1))
    return torch.flatten(f, 1)


def forward_with_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    feats = pooled_features(model, x)
    logits = model.classifier(feats)  # Dropout is inactive in eval mode
    return logits, feats
