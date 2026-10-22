"""Decision logic of the analysis pipeline, using a mocked predictor."""

import pytest

from src.inference.analysis import Status, analyze
from src.recommendations.context import CareContext


def run(fake, catalog, thresholds, data, plant="Tomato", ctx=None, filename="leaf.jpg"):
    return analyze(data, filename, plant, fake, catalog, thresholds, ctx)


def test_supported_plant_confident_prediction(fake_factory, catalog, thresholds, leaf_bytes):
    r = run(fake_factory({"Tomato___Early_blight": 0.95}), catalog, thresholds, leaf_bytes)
    assert r.status == Status.LIKELY
    assert r.predicted_label == "Tomato___Early_blight"
    assert r.condition.name == "Tomato early blight"
    assert r.confidence == pytest.approx(0.95)
    assert r.gradcam_allowed


def test_supported_plant_below_threshold_is_uncertain(fake_factory, catalog, thresholds, leaf_bytes):
    r = run(fake_factory({"Tomato___Early_blight": 0.55, "Tomato___Target_Spot": 0.40}), catalog, thresholds, leaf_bytes)
    assert r.status == Status.UNCERTAIN
    assert r.predicted_label == "Tomato___Early_blight"
    assert r.alternatives[0][0] == "Tomato___Target_Spot"


@pytest.mark.parametrize("threshold,expected", [(0.70, Status.LIKELY), (0.75, Status.LIKELY), (0.76, Status.UNCERTAIN)])
def test_confidence_threshold_boundary(fake_factory, catalog, thresholds, leaf_bytes, threshold, expected):
    thresholds["confidence_threshold"] = threshold
    r = run(fake_factory({"Tomato___Septoria_leaf_spot": 0.75}), catalog, thresholds, leaf_bytes)
    assert r.status == expected


def test_missing_plant_name(fake_factory, catalog, thresholds, leaf_bytes):
    fake = fake_factory()
    for name in ["", "   ", None]:
        r = run(fake, catalog, thresholds, leaf_bytes, plant=name)
        assert r.status == Status.INVALID_INPUT
        assert "plant name" in r.message
    assert fake.calls == 0


def test_missing_image(fake_factory, catalog, thresholds):
    r = run(fake_factory(), catalog, thresholds, None)
    assert r.status == Status.INVALID_INPUT


def test_corrupt_image(fake_factory, catalog, thresholds):
    fake = fake_factory()
    r = run(fake, catalog, thresholds, b"garbage bytes that are not an image")
    assert r.status == Status.INVALID_INPUT
    assert fake.calls == 0


def test_unsupported_plant_gets_no_disease_label(fake_factory, catalog, thresholds, leaf_bytes):
    r = run(fake_factory({"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 0.9}),
            catalog, thresholds, leaf_bytes, plant="Peace lily")
    assert r.status == Status.UNSUPPORTED_PLANT
    assert r.predicted_label is None and r.condition is None and r.confidence is None
    assert not r.gradcam_allowed
    assert "Peace lily" in r.message


def test_species_mismatch_is_flagged(fake_factory, catalog, thresholds, leaf_bytes):
    r = run(fake_factory({"Apple___Black_rot": 0.9}), catalog, thresholds, leaf_bytes, plant="Tomato")
    assert r.status == Status.SPECIES_MISMATCH
    assert r.predicted_label is None
    assert "Apple" in r.message and not r.gradcam_allowed


def test_restricted_prediction_when_other_species_top_but_plant_mass_high(fake_factory, catalog, thresholds, leaf_bytes):
    weights = {"Potato___Late_blight": 0.30, "Tomato___Late_blight": 0.29, "Tomato___Early_blight": 0.28}
    r = run(fake_factory(weights), catalog, thresholds, leaf_bytes, plant="Tomato")
    assert r.status == Status.UNCERTAIN  # top-1 is a potato class, so never LIKELY
    assert r.predicted_label == "Tomato___Late_blight"


def test_out_of_distribution_image_flagged(fake_factory, catalog, thresholds, leaf_bytes):
    r = run(fake_factory({"Tomato___Early_blight": 0.99}, ood_score=500.0), catalog, thresholds, leaf_bytes)
    assert r.status == Status.UNFAMILIAR
    assert r.ood_flag and r.predicted_label is None and not r.gradcam_allowed


def test_ood_disabled_when_threshold_missing(fake_factory, catalog, thresholds, leaf_bytes):
    thresholds["ood_threshold"] = None
    r = run(fake_factory(ood_score=500.0), catalog, thresholds, leaf_bytes)
    assert r.status == Status.LIKELY


def test_possible_non_plant_image(fake_factory, catalog, thresholds, gray_bytes):
    r = run(fake_factory({"Tomato___Early_blight": 0.99}), catalog, thresholds, gray_bytes)
    assert r.status == Status.POSSIBLE_NON_PLANT
    assert r.predicted_label is None


def test_low_quality_image_is_tentative(fake_factory, catalog, thresholds, leaf_bytes):
    thresholds["blur_min_variance"] = 1e9  # force "blurry"
    r = run(fake_factory({"Tomato___Early_blight": 0.99}), catalog, thresholds, leaf_bytes)
    assert r.status == Status.LOW_QUALITY
    assert any("blurry" in w for w in r.warnings)


def test_single_class_species_never_likely(fake_factory, catalog, thresholds, leaf_bytes):
    r = run(fake_factory({"Orange___Haunglongbing_(Citrus_greening)": 0.99}), catalog, thresholds, leaf_bytes,
            plant="Orange")
    assert r.status == Status.UNCERTAIN
    assert any("never saw healthy" in w for w in r.warnings)


def test_inference_error_does_not_crash(fake_factory, catalog, thresholds, leaf_bytes):
    r = run(fake_factory(raises=True), catalog, thresholds, leaf_bytes)
    assert r.status == Status.ANALYSIS_ERROR
    assert "boom" not in r.message  # internal details are not exposed


def test_no_predictor_supported_plant(catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "leaf.jpg", "Tomato", None, catalog, thresholds)
    assert r.status == Status.MODEL_UNAVAILABLE


def test_no_predictor_unsupported_plant(catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "leaf.jpg", "Basil", None, catalog, thresholds)
    assert r.status == Status.UNSUPPORTED_PLANT


def test_pest_flags_from_image_and_user(fake_factory, catalog, thresholds, leaf_bytes):
    r = run(fake_factory(pest_score=0.8), catalog, thresholds, leaf_bytes,
            ctx=CareContext(insects_visible="Yes"))
    assert r.pest_flag
    assert r.pest_flag_sources == ["image screen", "user report"]
    r2 = run(fake_factory(pest_score=0.01), catalog, thresholds, leaf_bytes)
    assert not r2.pest_flag
