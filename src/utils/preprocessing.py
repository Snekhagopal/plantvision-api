"""Upload validation, image-quality checks and model preprocessing."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from src import config


class ImageValidationError(ValueError):
    """Raised for uploads that cannot be analysed. Message is user-safe."""


def load_image(data: bytes, filename: str | None = None) -> Image.Image:
    """Validate raw upload bytes and return an RGB PIL image.

    Checks extension, byte size, decodability, actual format, pixel count and
    minimum dimensions. EXIF orientation from phone cameras is applied.
    """
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"The file is larger than {config.MAX_UPLOAD_MB} MB. Please upload a smaller photo."
        )
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in config.ALLOWED_EXTENSIONS:
            raise ImageValidationError("Unsupported file type. Please upload a JPG, JPEG, PNG or WebP image.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            fmt = probe.format
            width, height = probe.size
            if width * height > config.MAX_IMAGE_PIXELS:
                raise ImageValidationError("The image resolution is too large. Please resize it and try again.")
            probe.verify()  # detects truncated/corrupt files
        image = Image.open(io.BytesIO(data))
        image.load()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(
            "The file could not be read as an image. It may be corrupt or not a JPG, PNG or WebP image."
        ) from exc

    if fmt not in config.ALLOWED_PIL_FORMATS:
        raise ImageValidationError("Unsupported image format. Please upload a JPG, JPEG, PNG or WebP image.")
    if min(width, height) < config.MIN_IMAGE_SIDE:
        raise ImageValidationError(
            f"The image is too small ({width}x{height}). Please upload a photo at least "
            f"{config.MIN_IMAGE_SIDE} pixels on each side."
        )

    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        # Composite transparency onto white rather than black.
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


@dataclass
class ImageQuality:
    brightness: float           # mean HSV value in [0, 1]
    blur_variance: float        # variance of the Laplacian (higher = sharper)
    vegetation_fraction: float  # share of green/yellow/brown plant-like pixels
    issues: list[str] = field(default_factory=list)

    @property
    def is_low_quality(self) -> bool:
        return bool(self.issues)


def vegetation_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels whose colour is plausible for leaf tissue.

    Counts saturated green-to-yellow hues plus darker, saturated
    brown/orange hues (diseased tissue). This is a coarse colour heuristic,
    not a plant detector: green objects pass and white/purple leaves fail.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # OpenCV hue range is 0-179
    h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    green_yellow = (h >= 18) & (h <= 95) & (s > 0.18) & (v > 0.12)
    brown = (h >= 5) & (h < 18) & (s > 0.25) & (v > 0.12) & (v < 0.75)
    return float((green_yellow | brown).mean())


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of the main leaf-coloured regions, holes filled.

    Used to check whether Grad-CAM attention falls on the leaf or the background.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green_yellow = (h >= 18) & (h <= 95) & (s > 40) & (v > 30)
    brown = (h >= 5) & (h < 18) & (s > 60) & (v > 30) & (v < 200)
    mask = ((green_yellow | brown) * 255).astype(np.uint8)
    k = max(3, rgb.shape[1] // 64) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    big = [c for c in contours if cv2.contourArea(c) >= 0.01 * mask.size]
    cv2.drawContours(filled, big, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def assess_quality(image: Image.Image, thresholds: dict) -> ImageQuality:
    """Compute simple, explainable quality signals on a 256 px copy."""
    small = image.copy()
    small.thumbnail((256, 256))
    rgb = np.asarray(small, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2].mean() / 255.0)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    veg = vegetation_fraction(rgb)

    issues = []
    if brightness < thresholds["brightness_min"]:
        issues.append("The photo is very dark. Retake it in good, even daylight.")
    elif brightness > thresholds["brightness_max"]:
        issues.append("The photo is very bright or washed out. Avoid direct glare or flash.")
    if blur < thresholds["blur_min_variance"]:
        issues.append("The photo looks blurry. Hold the camera steady and focus on the leaf.")
    return ImageQuality(brightness, blur, veg, issues)


def build_transform(image_size: int, mean, std) -> transforms.Compose:
    """Exactly the evaluation-time transform used in training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])
