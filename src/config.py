"""Central configuration: file paths, upload limits and decision thresholds.

Thresholds learned from data (temperature, confidence threshold, OOD
threshold, image-quality cut-offs) are written to ``models/calibration.json``
by ``python -m src.evaluation.calibrate`` using the *validation* split only.
The defaults below are used when that file is missing.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

MOBILENET_CHECKPOINT = MODELS_DIR / "mobilenet_best.pth"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
OOD_STATS_PATH = MODELS_DIR / "ood_stats.npz"
HISTORY_DB_PATH = DATA_DIR / "history.db"

# Locally cached torchvision ImageNet weights, reused for the generic
# visible-insect screen. Never downloaded at runtime.
TORCH_HUB_CHECKPOINTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
IMAGENET_MOBILENET_GLOB = "mobilenet_v2-*.pth"

# Zero-shot pest screen (loaded from the local Hugging Face cache only).
PEST_CLIP_MODEL = "openai/clip-vit-large-patch14"

# Upload validation
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "MPO", "WEBP"}  # MPO = multi-picture JPEG from phones
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MIN_IMAGE_SIDE = 64
MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard

TOP_K = 3

# Fallback decision thresholds (overridden by models/calibration.json).
DEFAULT_THRESHOLDS = {
    "temperature": 1.0,
    "confidence_threshold": 0.80,
    "ood_threshold": None,          # None -> OOD detector disabled
    "plant_mass_threshold": 0.50,
    "vegetation_min_fraction": 0.05,
    "blur_min_variance": 20.0,
    "brightness_min": 0.10,
    "brightness_max": 0.95,
    "pest_screen_threshold": 0.90,            # CLIP screen
    "pest_screen_threshold_imagenet": 0.37,   # fallback ImageNet screen
    # Below this share of Grad-CAM weight on the leaf, "likely" becomes "uncertain".
    # Chosen on 400 validation images: 3.3% flagged, 69% accurate vs 95% unflagged.
    "gradcam_min_leaf_attention": 0.30,
    # Zero-shot symptom check: below this top probability the pattern is shown as "not clear".
    "symptom_clear_threshold": 0.60,
    # Otherwise, report the top two patterns when together they reach this probability.
    "symptom_pair_threshold": 0.80,
}
