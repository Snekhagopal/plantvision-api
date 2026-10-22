import re

import pytest

from src.inference.analysis import Status, analyze
from src.plants.catalog import PLANTVILLAGE_LABELS
from src.recommendations.assessments import assess_nutrients, assess_watering
from src.recommendations.context import CareContext
from src.recommendations.engine import build_recommendations


def all_text(rec) -> str:
    return " ".join([rec.summary] + [i for s in rec.sections for i in s.items])


def analyse(fake, catalog, thresholds, data, plant="Tomato", ctx=None):
    return analyze(data, "leaf.jpg", plant, fake, catalog, thresholds, ctx)


@pytest.mark.parametrize("label", [l for l in PLANTVILLAGE_LABELS if not l.startswith(("Orange", "Squash", "Blueberry", "Raspberry", "Soybean"))])
def test_recommendations_for_every_supported_condition(fake_factory, catalog, thresholds, leaf_bytes, label):
    plant = catalog.species_of(label).display_name
    r = analyse(fake_factory({label: 0.97}), catalog, thresholds, leaf_bytes, plant=plant)
    assert r.status == Status.LIKELY
    rec = build_recommendations(r)
    keys = [s.key for s in rec.sections]
    for required in ("actions", "inspect", "watering", "pests", "prevention", "expert", "limitations"):
        assert required in keys
    assert "not a confirmed diagnosis" in rec.summary
    if not label.endswith("healthy"):
        assert "disease" in keys


def test_late_blight_escalates(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyse(fake_factory({"Tomato___Late_blight": 0.97}), catalog, thresholds, leaf_bytes)
    rec = build_recommendations(r)
    assert "extension service" in rec.section("expert").items[0]


def test_uncertain_prediction_advises_against_treatment(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyse(fake_factory({"Tomato___Early_blight": 0.5}), catalog, thresholds, leaf_bytes)
    rec = build_recommendations(r)
    assert "Do not treat" in rec.section("actions").items[0]
    assert "uncertain" in rec.summary


def test_no_specific_products_or_doses_recommended(fake_factory, catalog, thresholds, leaf_bytes):
    """Treatment categories are allowed (approved policy), but never named actives, brands or doses."""
    text = ""
    for label in PLANTVILLAGE_LABELS:
        plant = catalog.species_of(label).display_name
        r = analyse(fake_factory({label: 0.97}), catalog, thresholds, leaf_bytes, plant=plant)
        text += all_text(build_recommendations(r, CareContext(insects_visible="Yes", symptoms=["Yellowing leaves"])))
    lowered = text.lower()
    for banned in ("mancozeb", "chlorothalonil", "imidacloprid", "glyphosate", "spray with", "apply fungicide",
                   "apply insecticide", "apply pesticide"):
        assert banned not in lowered
    assert not re.search(r"\d+(\.\d+)?\s?(ml|g|mg|kg|l|oz|tbsp|tsp|%|ppm)\b", lowered)


def test_unsupported_plant_recommendations_are_general(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyse(fake_factory(), catalog, thresholds, leaf_bytes, plant="Basil")
    rec = build_recommendations(r)
    assert rec.section("disease") is None
    assert "cannot reliably diagnose" in rec.summary


# ---- watering must not overstate what an image proves ----

OVERCLAIMS = re.compile(r"(definitely|certainly|clearly)\b.*\b(water|dry|overwater)|needs water\b|is underwatered\b|is overwatered\b", re.I)


def test_watering_with_no_answers_is_insufficient():
    a = assess_watering(CareContext())
    assert a.level == "insufficient_information"
    assert a.questions


@pytest.mark.parametrize("ctx,level", [
    (CareContext(soil_moisture="Bone dry", symptoms=["Wilting or drooping"]), "possible_underwatering"),
    (CareContext(soil_moisture="Wet / soggy", drainage="Container has no drainage hole"), "possible_overwatering"),
    (CareContext(soil_moisture="Bone dry", drainage="Drains slowly / water pools"), "mixed"),
    (CareContext(soil_moisture="Evenly moist", drainage="Drains well"), "no_clear_signal"),
])
def test_watering_levels_depend_on_answers(ctx, level):
    a = assess_watering(ctx)
    assert a.level == level
    for text in [a.headline] + a.reasons + a.advice:
        assert not OVERCLAIMS.search(text), text


def test_photo_alone_never_produces_watering_claim(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyse(fake_factory({"Tomato___healthy": 0.99}), catalog, thresholds, leaf_bytes)
    rec = build_recommendations(r)
    assert rec.watering.level == "insufficient_information"
    items = rec.section("watering").items
    assert "cannot show whether a plant needs water" in items[0]
    assert not any(OVERCLAIMS.search(i) for i in items[1:])  # items[0] is the caveat itself


def test_nutrients_only_when_colour_symptoms_reported():
    assert assess_nutrients(CareContext()) is None
    a = assess_nutrients(CareContext(symptoms=["Yellowing leaves"], affected_leaves="Older / lower leaves first"))
    assert a is not None
    text = " ".join(a.reasons + a.advice).lower()
    assert "soil test" in text and "consistent with" in text
    assert "deficiency confirmed" not in text


def test_pest_section_flags_but_does_not_identify(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyse(fake_factory(pest_score=0.9), catalog, thresholds, leaf_bytes)
    rec = build_recommendations(r)
    text = " ".join(rec.section("pests").items)
    assert "not a pest identification" in text
    assert "close-up" in text


def test_invalid_input_has_no_sections(catalog, thresholds):
    r = analyze(None, None, "", None, catalog, thresholds)
    rec = build_recommendations(r)
    assert rec.sections == []
