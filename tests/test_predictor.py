"""Checkpoint loading, class mapping and real inference."""

import numpy as np
import pytest
import torch
from PIL import Image

from src import config
from src.inference.model import ModelLoadError, class_names_from_mapping, load_mobilenet_checkpoint
from src.inference.predictor import PlantDiseasePredictor, load_thresholds
from src.plants.catalog import PLANTVILLAGE_LABELS
from tests.conftest import make_leaf_image, requires_checkpoint


def test_class_mapping_inversion():
    assert class_names_from_mapping({"b": 1, "a": 0, "c": 2}) == ["a", "b", "c"]


def test_class_mapping_rejects_gaps():
    with pytest.raises(ModelLoadError):
        class_names_from_mapping({"a": 0, "b": 2})


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_mobilenet_checkpoint(tmp_path / "nope.pth")


def test_incompatible_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {"foo": torch.zeros(1)}, "class_to_idx": {"a": 0}}, path)
    with pytest.raises(ModelLoadError, match="do not match"):
        load_mobilenet_checkpoint(path)


def test_checkpoint_missing_keys_raises(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"weights": 1}, path)
    with pytest.raises(ModelLoadError, match="missing keys"):
        load_mobilenet_checkpoint(path)


def test_load_thresholds_falls_back_to_defaults(tmp_path):
    assert load_thresholds(tmp_path / "missing.json") == config.DEFAULT_THRESHOLDS


@requires_checkpoint
def test_real_checkpoint_loads_with_training_mapping():
    loaded = load_mobilenet_checkpoint(config.MOBILENET_CHECKPOINT)
    assert len(loaded.class_names) == 38
    assert loaded.class_names == PLANTVILLAGE_LABELS
    assert loaded.image_size == 128
    assert np.allclose(loaded.mean, [0.485, 0.456, 0.406])
    assert not loaded.model.training


@pytest.fixture(scope="module")
def predictor():
    if not config.MOBILENET_CHECKPOINT.exists():
        pytest.skip("checkpoint not available")
    return PlantDiseasePredictor()


@requires_checkpoint
def test_valid_image_inference(predictor):
    out = predictor.predict(make_leaf_image())
    assert out.probabilities.shape == (38,)
    assert np.isclose(out.probabilities.sum(), 1.0)
    scores = [p for _, p in out.top_k]
    assert scores == sorted(scores, reverse=True)
    assert out.top_k[0][0] == PLANTVILLAGE_LABELS[int(np.argmax(out.probabilities))]


@requires_checkpoint
def test_output_index_maps_to_correct_class_name(predictor, monkeypatch):
    """Force the classifier to favour index 29 and check the label returned."""
    target = 29
    layer = predictor.model.classifier[1]
    with torch.no_grad():
        saved_w, saved_b = layer.weight.clone(), layer.bias.clone()
        layer.weight.zero_()
        layer.bias.zero_()
        layer.bias[target] = 10.0
    try:
        out = predictor.predict(Image.new("RGB", (128, 128), "green"))
    finally:
        with torch.no_grad():
            layer.weight.copy_(saved_w)
            layer.bias.copy_(saved_b)
    assert out.top_k[0][0] == PLANTVILLAGE_LABELS[target] == "Tomato___Early_blight"


@requires_checkpoint
def test_inference_is_deterministic(predictor):
    img = make_leaf_image(seed=3)
    a, b = predictor.predict(img), predictor.predict(img)
    assert np.allclose(a.probabilities, b.probabilities)


@requires_checkpoint
def test_temperature_scaling_preserves_ranking(predictor):
    out = predictor.predict(make_leaf_image(seed=4))
    assert np.argmax(out.probabilities) == np.argmax(out.raw_probabilities)
