"""End-to-end service tests: mocks for unit paths, real model for smoke paths."""

import pytest
import torch

from src.database.history import HistoryError, HistoryStore
from src.explainability.gradcam import GradCamResult
from src.inference.analysis import Status
from src.models.mobilenet import create_mobilenet
from src.service import run_full_analysis
from tests.conftest import make_leaf_image, requires_checkpoint, to_bytes


class GradcamCapableFake:
    """Wraps FakePredictor with a real (random) network for Grad-CAM."""

    def __init__(self, fake):
        self._fake = fake
        self.catalog, self.thresholds, self.class_names = fake.catalog, fake.thresholds, fake.class_names
        torch.manual_seed(0)
        self.model = create_mobilenet(38, pretrained=False).eval()

    def predict(self, image):
        return self._fake.predict(image)

    def preprocess(self, image):
        return torch.randn(1, 3, 128, 128)


@pytest.fixture
def history(tmp_path):
    return HistoryStore(tmp_path / "h.db")


def test_full_flow_saves_history_and_gradcam(fake_factory, catalog, thresholds, leaf_bytes, history):
    predictor = GradcamCapableFake(fake_factory({"Tomato___Early_blight": 0.95}))
    b = run_full_analysis(leaf_bytes, "leaf.jpg", "Tomato", None, predictor, catalog, thresholds, history)
    assert b.result.status == Status.LIKELY
    assert b.gradcam is not None and b.gradcam.ok
    saved = history.get(b.history_id)
    assert saved.condition_name == "Tomato early blight"
    assert saved.recommendations["sections"]


def test_gradcam_skipped_for_unsupported_plant(fake_factory, catalog, thresholds, leaf_bytes, history):
    predictor = GradcamCapableFake(fake_factory())
    b = run_full_analysis(leaf_bytes, "leaf.jpg", "Rose", None, predictor, catalog, thresholds, history)
    assert b.result.status == Status.UNSUPPORTED_PLANT
    assert b.gradcam is None
    assert history.get(b.history_id).confidence is None


def test_gradcam_failure_does_not_break_results(fake_factory, catalog, thresholds, leaf_bytes, monkeypatch):
    predictor = GradcamCapableFake(fake_factory())
    monkeypatch.setattr("src.service.compute_gradcam",
                        lambda *a, **k: GradCamResult(None, None, "The heatmap could not be generated (X)."))
    b = run_full_analysis(leaf_bytes, "leaf.jpg", "Tomato", None, predictor, catalog, thresholds)
    assert b.result.status == Status.LIKELY and not b.gradcam.ok
    assert b.recommendation.sections


def test_gradcam_preparation_exception_is_contained(fake_factory, catalog, thresholds, leaf_bytes):
    predictor = GradcamCapableFake(fake_factory())
    predictor.preprocess = lambda image: (_ for _ in ()).throw(RuntimeError("bad"))
    b = run_full_analysis(leaf_bytes, "leaf.jpg", "Tomato", None, predictor, catalog, thresholds)
    assert b.gradcam is not None and not b.gradcam.ok


def test_invalid_input_not_saved(fake_factory, catalog, thresholds, history):
    b = run_full_analysis(b"junk", "x.jpg", "Tomato", None, fake_factory(), catalog, thresholds, history)
    assert b.result.status == Status.INVALID_INPUT
    assert b.history_id is None and history.recent() == []


def test_history_failure_is_reported_not_raised(fake_factory, catalog, thresholds, leaf_bytes):
    class BrokenStore:
        def add(self, **kw):
            raise HistoryError("A history database operation failed.")
    b = run_full_analysis(leaf_bytes, "leaf.jpg", "Tomato", None, fake_factory(), catalog, thresholds, BrokenStore())
    assert b.history_error and b.result.status == Status.LIKELY


def test_unexpected_exception_is_contained(catalog, thresholds, leaf_bytes, monkeypatch):
    monkeypatch.setattr("src.service.analyze", lambda *a, **k: 1 / 0)
    b = run_full_analysis(leaf_bytes, "leaf.jpg", "Tomato", None, None, catalog, thresholds)
    assert b.result.status == Status.ANALYSIS_ERROR


# ---- real-model smoke tests ----

@pytest.fixture(scope="module")
def real_predictor():
    from src.inference.predictor import PlantDiseasePredictor
    return PlantDiseasePredictor()


@requires_checkpoint
@pytest.mark.parametrize("plant,data", [
    ("Tomato", to_bytes(make_leaf_image())),
    ("Peace lily", to_bytes(make_leaf_image())),
    ("Tomato", to_bytes(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (300, 300), (20, 40, 200)))),
    ("Tomato", b"\x89PNG\r\n\x1a\n corrupted"),
])
def test_real_model_never_crashes_on_odd_inputs(real_predictor, plant, data):
    b = run_full_analysis(data, "upload.png", plant, None, real_predictor,
                          real_predictor.catalog, real_predictor.thresholds)
    assert b.result.status in Status
    assert b.recommendation.summary


@requires_checkpoint
def test_real_model_blue_square_is_not_diagnosed(real_predictor):
    from PIL import Image
    data = to_bytes(Image.new("RGB", (300, 300), (20, 40, 200)))
    b = run_full_analysis(data, "x.jpg", "Tomato", None, real_predictor, real_predictor.catalog,
                          real_predictor.thresholds)
    assert not b.result.shows_candidate
