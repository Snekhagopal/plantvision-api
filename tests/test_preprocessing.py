import io

import numpy as np
import pytest
from PIL import Image

from src import config
from src.utils.preprocessing import ImageValidationError, assess_quality, build_transform, load_image
from tests.conftest import make_leaf_image, to_bytes


def test_valid_jpeg_loads_as_rgb(leaf_bytes):
    image = load_image(leaf_bytes, "leaf.jpg")
    assert image.mode == "RGB" and image.size == (256, 256)


def test_png_with_transparency_is_flattened():
    rgba = Image.new("RGBA", (100, 100), (0, 200, 0, 0))
    image = load_image(to_bytes(rgba, "PNG"), "leaf.png")
    assert image.mode == "RGB"
    assert image.getpixel((5, 5)) == (255, 255, 255)


def test_webp_loads_as_rgb():
    image = load_image(to_bytes(make_leaf_image(), "WEBP"), "leaf.webp")
    assert image.mode == "RGB" and image.size == (256, 256)


def test_webp_with_transparency_is_flattened():
    rgba = Image.new("RGBA", (100, 100), (0, 200, 0, 0))
    buf = io.BytesIO()
    rgba.save(buf, "WEBP", lossless=True)
    image = load_image(buf.getvalue(), "leaf.webp")
    assert image.mode == "RGB" and image.getpixel((5, 5)) == (255, 255, 255)


def test_truncated_webp_rejected():
    data = to_bytes(make_leaf_image(), "WEBP")
    with pytest.raises(ImageValidationError):
        load_image(data[: len(data) // 3], "leaf.webp")


def test_gif_renamed_to_webp_rejected():
    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        load_image(to_bytes(make_leaf_image(), "GIF"), "leaf.webp")


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\xff\xd8\xff\xe0" + b"\x00" * 50])
def test_corrupt_or_empty_files_rejected(data):
    with pytest.raises(ImageValidationError):
        load_image(data, "leaf.jpg")


def test_truncated_jpeg_rejected(leaf_bytes):
    with pytest.raises(ImageValidationError):
        load_image(leaf_bytes[: len(leaf_bytes) // 3], "leaf.jpg")


def test_wrong_extension_rejected(leaf_bytes):
    with pytest.raises(ImageValidationError, match="Unsupported file type"):
        load_image(leaf_bytes, "leaf.gif")


def test_disallowed_actual_format_rejected():
    with pytest.raises(ImageValidationError):
        load_image(to_bytes(make_leaf_image(), "BMP"), "leaf.png")


def test_oversized_file_rejected(monkeypatch, leaf_bytes):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ImageValidationError, match="larger than"):
        load_image(leaf_bytes, "leaf.jpg")


def test_tiny_image_rejected():
    with pytest.raises(ImageValidationError, match="too small"):
        load_image(to_bytes(Image.new("RGB", (20, 20), "green")), "leaf.jpg")


def test_quality_flags_dark_and_blurry_images():
    dark = Image.new("RGB", (200, 200), (5, 8, 5))
    q = assess_quality(dark, config.DEFAULT_THRESHOLDS)
    assert q.is_low_quality
    assert any("dark" in i for i in q.issues)
    assert any("blurry" in i for i in q.issues)


def test_quality_accepts_textured_leaf():
    q = assess_quality(make_leaf_image(), config.DEFAULT_THRESHOLDS)
    assert not q.is_low_quality
    assert q.vegetation_fraction > 0.3


def test_transform_output_shape_and_normalisation():
    t = build_transform(128, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    x = t(Image.new("RGB", (300, 200), (124, 116, 104)))  # ~ImageNet mean colour
    assert x.shape == (3, 128, 128)
    assert np.allclose(x.numpy(), 0, atol=0.03)
