import numpy as np
import pytest
import torch
from torch import nn

from src.explainability.gradcam import compute_gradcam, mobilenet_target_layer
from src.models.mobilenet import create_mobilenet
from tests.conftest import make_leaf_image


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return create_mobilenet(num_classes=38, pretrained=False).eval()


def test_gradcam_produces_overlay_at_display_resolution(tiny_model):
    image = make_leaf_image(size=300)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():  # Grad-CAM must re-enable gradients internally
        r = compute_gradcam(tiny_model, x, 5, image)
    assert r.ok, r.error
    assert r.overlay.size == image.size
    assert np.isfinite(r.heatmap).all()
    assert r.heatmap.min() >= 0 and r.heatmap.max() <= 1


def test_gradcam_target_layer_is_last_conv_block(tiny_model):
    layer = mobilenet_target_layer(tiny_model)
    assert any(isinstance(m, nn.Conv2d) and m.out_channels == 1280 for m in layer.modules())


def test_gradcam_invalid_class_returns_error(tiny_model):
    r = compute_gradcam(tiny_model, torch.randn(1, 3, 128, 128), 999, make_leaf_image())
    assert not r.ok
    assert r.error and "could not be generated" in r.error


def test_gradcam_bad_input_returns_error(tiny_model):
    r = compute_gradcam(tiny_model, torch.randn(1, 5, 128, 128), 0, make_leaf_image())
    assert not r.ok


def test_gradcam_does_not_leave_hooks(tiny_model):
    compute_gradcam(tiny_model, torch.randn(1, 3, 128, 128), 1, make_leaf_image())
    layer = mobilenet_target_layer(tiny_model)
    assert not layer._forward_hooks and not layer._backward_hooks
