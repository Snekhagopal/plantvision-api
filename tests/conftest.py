import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src import config
from src.inference.predictor import ModelOutput
from src.plants.catalog import PLANTVILLAGE_LABELS, PlantCatalog

CHECKPOINT_AVAILABLE = config.MOBILENET_CHECKPOINT.exists() and config.MOBILENET_CHECKPOINT.stat().st_size > 0
requires_checkpoint = pytest.mark.skipif(not CHECKPOINT_AVAILABLE, reason="models/mobilenet_best.pth not available")


def to_bytes(image: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, fmt)
    return buf.getvalue()


def make_leaf_image(size=256, seed=0) -> Image.Image:
    """Synthetic textured green leaf with brown spots on a gray background."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (150, 150, 150))
    draw = ImageDraw.Draw(img)
    draw.ellipse([size * 0.1, size * 0.2, size * 0.9, size * 0.8], fill=(60, 140, 50))
    for _ in range(8):
        x, y = rng.integers(int(size * 0.25), int(size * 0.75), 2)
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(110, 70, 30))
    arr = np.asarray(img).astype(np.int16) + rng.integers(-25, 25, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


@pytest.fixture
def catalog():
    return PlantCatalog(PLANTVILLAGE_LABELS)


@pytest.fixture
def thresholds():
    t = dict(config.DEFAULT_THRESHOLDS)
    t.update({"confidence_threshold": 0.8, "ood_threshold": 50.0, "pest_screen_threshold": 0.3})
    return t


@pytest.fixture
def leaf_bytes():
    return to_bytes(make_leaf_image())


@pytest.fixture
def gray_bytes():
    rng = np.random.default_rng(1)
    arr = np.clip(128 + rng.normal(0, 30, (256, 256, 3)), 0, 255).astype(np.uint8)
    return to_bytes(Image.fromarray(arr))


class FakePredictor:
    """Returns a fixed probability vector; no neural network involved."""

    def __init__(self, catalog, thresholds, probs=None, ood_score=10.0, pest_score=0.0, raises=False,
                 symptom=None):
        self.catalog = catalog
        self.class_names = catalog.class_names
        self.thresholds = thresholds
        self.probs = probs
        self.ood_score = ood_score
        self.pest_score = pest_score
        self.raises = raises
        self.symptom = symptom
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        if self.raises:
            raise RuntimeError("boom")
        probs = np.asarray(self.probs, dtype=float)
        order = np.argsort(probs)[::-1][:3]
        return ModelOutput(probs, probs, [(self.class_names[i], float(probs[i])) for i in order],
                           ood_score=self.ood_score, pest_score=self.pest_score,
                           pest_top=[("leaf beetle", self.pest_score)], symptom=self.symptom)


def probs_for(catalog, weights: dict[str, float]) -> np.ndarray:
    """Probability vector with given mass on labels and the rest spread evenly."""
    n = len(catalog.class_names)
    rest = 1.0 - sum(weights.values())
    p = np.full(n, rest / (n - len(weights)))
    for label, w in weights.items():
        p[catalog.class_names.index(label)] = w
    return p


@pytest.fixture
def fake_factory(catalog, thresholds):
    def make(weights=None, **kw):
        probs = probs_for(catalog, weights or {"Tomato___Early_blight": 0.95})
        return FakePredictor(catalog, thresholds, probs, **kw)
    return make
