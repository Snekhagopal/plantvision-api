"""Evidence-gated treatment advice, whole-plant photos, symptoms on supported crops, and the report."""

import json

import pytest

from src.inference.analysis import LEAF_CLOSE_UP, WHOLE_PLANT, Status, analyze
from src.inference.symptom_screen import SymptomResult
from src.plants.catalog import PLANTVILLAGE_LABELS
from src.recommendations.context import CareContext
from src.recommendations.engine import build_recommendations
from src.recommendations.treatments import SAFETY
from src.report import DETECTED, HEALTHY, NOT_ASSESSED, NOT_DETECTED, PROBLEM, UNCERTAIN, UNCLEAR, build_report
from src.service import run_full_analysis
from tests.conftest import requires_checkpoint

CHEMICAL_WORDS = ("fungicide", "insecticid", "pesticide", "copper", "sulfur", "bicarbonate", "horticultural oil",
                  "thuringiensis")


def symptom(key="spots", p=0.9, clear=True, second=None):
    return SymptomResult(key, p, {key: p}, clear=clear, second=second)


def run(fake_factory, catalog, thresholds, leaf_bytes, plant="Tomato", weights=None, ctx=None,
        photo_type=LEAF_CLOSE_UP, pest_top=None, **kw):
    predictor = fake_factory(weights, **kw)
    if pest_top is not None:
        predict = predictor.predict
        def with_hits(image):
            out = predict(image)
            out.pest_top = pest_top
            return out
        predictor.predict = with_hits
    r = analyze(leaf_bytes, "x.jpg", plant, predictor, catalog, thresholds, ctx, photo_type)
    return r, build_recommendations(r, ctx)


def chemical_text(rec) -> str:
    return " ".join(i for s in rec.sections if s.key != "expert" for i in s.items).lower()


# ---- treatment gating ----

@pytest.mark.parametrize("label", [l for l in PLANTVILLAGE_LABELS if not l.endswith("___healthy")])
def test_likely_disease_gets_treatment_category_with_safety(label, fake_factory, catalog, thresholds, leaf_bytes):
    plant = catalog.species_of(label).display_name
    if len(catalog.species_of(label).labels) == 1:
        pytest.skip("single-class species are never 'likely'")
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, plant, {label: 0.97})
    assert r.status == Status.LIKELY
    section = rec.section("treatment")
    assert section is not None and section.title == "Treatment options"
    assert all(item in section.items for item in SAFETY)


def test_healthy_result_gets_no_treatment(fake_factory, catalog, thresholds, leaf_bytes):
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___healthy": 0.97})
    assert r.status == Status.LIKELY and rec.section("treatment") is None


def test_uncertain_disease_gets_only_conditional_treatment(fake_factory, catalog, thresholds, leaf_bytes):
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___Early_blight": 0.5})
    assert r.status == Status.UNCERTAIN
    section = rec.section("treatment")
    assert section.title.endswith("only after confirming the cause")
    assert section.items[0].startswith("If a local expert confirms tomato early blight")
    assert "Do not treat" in rec.section("actions").items[0]   # existing behaviour kept


@pytest.mark.parametrize("kw,plant", [
    ({"ood_score": 99.0}, "Tomato"),                                        # unfamiliar photo
    ({"weights": {"Apple___Apple_scab": 0.95}}, "Tomato"),                  # species mismatch
])
def test_unreliable_results_get_no_treatment(kw, plant, fake_factory, catalog, thresholds, leaf_bytes):
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, plant, **kw)
    assert r.status in (Status.UNFAMILIAR, Status.SPECIES_MISMATCH)
    assert rec.section("treatment") is None
    assert not any(w in chemical_text(rec) for w in CHEMICAL_WORDS)


def test_non_plant_photo_gets_no_treatment(fake_factory, catalog, thresholds, gray_bytes):
    r = analyze(gray_bytes, "x.jpg", "Tomato", fake_factory(), catalog, thresholds)
    rec = build_recommendations(r)
    assert r.status == Status.POSSIBLE_NON_PLANT and rec.section("treatment") is None


def test_yellowing_or_dryness_alone_never_triggers_products(fake_factory, catalog, thresholds, leaf_bytes):
    ctx = CareContext(symptoms=["Yellowing leaves", "Brown or crispy leaf edges", "Wilting or drooping"],
                      soil_moisture="Bone dry")
    for plant, kw in (("Hibiscus", {}), ("Hibiscus", {"symptom": symptom("mottling")}),
                      ("Tomato", {"weights": {"Tomato___healthy": 0.97}})):
        r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, plant, ctx=ctx, **kw)
        assert rec.section("treatment") is None, (plant, kw)
        assert not any(w in chemical_text(rec) for w in CHEMICAL_WORDS), (plant, kw)
        nutrients = " ".join(rec.section("nutrients").items)
        assert "Do not add fertiliser based on leaf colour alone" in nutrients
        assert "If a soil test confirms a deficiency" in nutrients


@pytest.mark.parametrize("key,expect", [("spots", None), ("mottling", None), ("healthy", None),
                                        ("mildew", "powdery mildew"), ("rust", "rust")])
def test_symptom_check_treatment_is_conditional_and_limited(key, expect, fake_factory, catalog, thresholds,
                                                           leaf_bytes):
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, "Rose", symptom=symptom(key))
    section = rec.section("treatment")
    if expect is None:
        assert section is None
    else:
        assert section.title.endswith("only after confirming the cause")
        assert section.items[0].startswith(f"If a nursery or extension service confirms")
        assert expect in section.items[0]


def test_pest_treatment_follows_photo_appearance(fake_factory, catalog, thresholds, leaf_bytes):
    hits = [("small sap-sucking insects (e.g. whiteflies, aphids, mealybugs, scale)", 0.8)]
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, "Hibiscus", pest_score=0.9, pest_top=hits)
    items = rec.section("treatment").items
    assert items[0].startswith("Identify the pest first")
    assert "insecticidal soap or horticultural oil" in items[1]
    assert not any("thuringiensis" in i for i in items)


def test_eggs_get_no_spray(fake_factory, catalog, thresholds, leaf_bytes):
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, "Hibiscus", pest_score=0.9,
                 pest_top=[("insect eggs", 0.7)])
    assert "No spray is needed" in rec.section("treatment").items[1]


def test_user_reported_insects_get_options_after_identification(fake_factory, catalog, thresholds, leaf_bytes):
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, "Hibiscus", ctx=CareContext(insects_visible="Yes"))
    items = rec.section("treatment").items
    assert items[0].startswith("Identify the pest first") and len(items) > len(SAFETY) + 2


def test_no_cure_conditions_do_not_suggest_sprays(fake_factory, catalog, thresholds, leaf_bytes):
    for label in ("Tomato___Tomato_mosaic_virus", "Grape___Esca_(Black_Measles)"):
        plant = catalog.species_of(label).display_name
        r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, plant, {label: 0.97})
        option = rec.section("treatment").items[0]
        assert "no chemical cure" in option.lower() or "no effective curative spray" in option.lower()


def test_citrus_greening_has_no_treatment_and_is_reported():
    from src.plants.conditions import get_condition
    from src.recommendations.treatments import disease_treatment
    label = "Orange___Haunglongbing_(Citrus_greening)"
    text = disease_treatment(label, get_condition(label), "Orange")
    assert "no cure or chemical treatment" in text and "report" in text and "copper" not in text.lower()


def test_squash_mildew_warns_about_sulfur_sensitivity():
    from src.plants.conditions import get_condition
    from src.recommendations.treatments import disease_treatment
    text = disease_treatment("Squash___Powdery_mildew", get_condition("Squash___Powdery_mildew"), "Squash")
    assert "sensitive to sulfur" in text


# ---- whole-plant photos ----

def test_whole_plant_photo_is_never_likely(fake_factory, catalog, thresholds, leaf_bytes):
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___Early_blight": 0.97},
                 photo_type=WHOLE_PLANT)
    assert r.status == Status.UNCERTAIN and r.photo_type == WHOLE_PLANT
    assert "whole-plant photo" in r.message and "close-up" in r.warnings[0]
    assert "upload a close-up of one affected leaf" in rec.section("actions").items[0]
    assert rec.section("treatment") is None                       # no disease treatment, not even conditional
    report = build_report(r)
    assert report.disease.state == UNCERTAIN
    assert any("close-up" in d for d in report.disease.details)


def test_leaf_photo_behaviour_unchanged_by_default(fake_factory, catalog, thresholds, leaf_bytes):
    r, _ = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___Early_blight": 0.97})
    assert r.status == Status.LIKELY and r.photo_type == LEAF_CLOSE_UP and not r.warnings


def test_unknown_photo_type_falls_back_to_leaf(fake_factory, catalog, thresholds, leaf_bytes):
    r, _ = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___Early_blight": 0.97},
               photo_type="selfie")
    assert r.photo_type == LEAF_CLOSE_UP


# ---- symptoms on supported crops stay separate from the disease prediction ----

def test_supported_crop_shows_symptoms_separately(fake_factory, catalog, thresholds, leaf_bytes):
    r, rec = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___healthy": 0.97},
                 symptom=symptom("spots"))
    assert r.status == Status.LIKELY and not r.symptom_is_primary
    assert "Healthy tomato leaf" in rec.summary and "Visible pattern" not in rec.summary
    items = rec.section("symptoms").items
    assert "**Leaf spots or blight**" in items[0]
    assert any("separate from the disease model" in i for i in items)
    assert any("disagree" in i for i in items)


# ---- report ----

def test_report_for_likely_disease(fake_factory, catalog, thresholds, leaf_bytes):
    r, _ = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___Early_blight": 0.97},
               symptom=symptom("spots"))
    rep = build_report(r)
    assert rep.health.state == PROBLEM and rep.disease.state == DETECTED
    assert rep.disease.headline == "Tomato early blight" and rep.disease.score == pytest.approx(0.97)
    assert rep.symptoms.state == DETECTED and rep.pests.state == NOT_DETECTED
    json.dumps(rep.as_dict())                                      # API-ready


def test_report_for_healthy_supported_crop(fake_factory, catalog, thresholds, leaf_bytes):
    r, _ = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___healthy": 0.97},
               symptom=symptom("healthy"))
    rep = build_report(r)
    assert rep.health.state == HEALTHY and rep.disease.state == NOT_DETECTED and rep.symptoms.state == NOT_DETECTED


def test_report_for_unsupported_plant(fake_factory, catalog, thresholds, leaf_bytes):
    ctx = CareContext(symptoms=["Wilting or drooping"])
    r, _ = run(fake_factory, catalog, thresholds, leaf_bytes, "Hibiscus", ctx=ctx,
               symptom=symptom("spots", 0.48, clear=False, second="mottling"))
    rep = build_report(r, ctx)
    assert rep.disease.state == NOT_ASSESSED and rep.symptoms.state == DETECTED
    assert rep.symptoms.headline == "Leaf spots or blight + Mosaic, mottling or yellowing"
    assert any("wilting or drooping" in d for d in rep.symptoms.details)
    assert rep.health.state == PROBLEM


def test_report_pests_make_health_a_problem(fake_factory, catalog, thresholds, leaf_bytes):
    r, _ = run(fake_factory, catalog, thresholds, leaf_bytes, "Tomato", {"Tomato___healthy": 0.97}, pest_score=0.9)
    rep = build_report(r)
    assert rep.pests.state == DETECTED and rep.health.state == PROBLEM


def test_report_for_non_plant_photo(fake_factory, catalog, thresholds, gray_bytes):
    r = analyze(gray_bytes, "x.jpg", "Tomato", fake_factory(), catalog, thresholds)
    rep = build_report(r)
    assert rep.health.state == UNCLEAR
    assert rep.disease.state == NOT_ASSESSED and rep.pests.state == NOT_ASSESSED


def test_service_bundle_carries_report(fake_factory, catalog, thresholds, leaf_bytes):
    b = run_full_analysis(leaf_bytes, "x.jpg", "Tomato", None, fake_factory(), catalog, thresholds,
                          with_gradcam=False, photo_type=WHOLE_PLANT)
    assert b.report is not None and b.report.photo_type == WHOLE_PLANT


# ---- real model ----

@requires_checkpoint
def test_real_model_whole_plant_photo_is_not_likely():
    from pathlib import Path
    from src.inference.predictor import PlantDiseasePredictor
    from src.plants.catalog import PlantCatalog
    leaf = Path(__file__).resolve().parents[1] / "data" / "raw" / "train" / "Apple___Apple_scab" / "image_0000416.jpg"
    if not leaf.exists():
        pytest.skip("PlantVillage image not available")
    predictor = PlantDiseasePredictor()
    catalog = PlantCatalog(PLANTVILLAGE_LABELS)
    as_leaf = analyze(leaf.read_bytes(), leaf.name, "Apple", predictor, catalog, predictor.thresholds)
    as_whole = analyze(leaf.read_bytes(), leaf.name, "Apple", predictor, catalog, predictor.thresholds,
                       photo_type=WHOLE_PLANT)
    assert as_leaf.status == Status.LIKELY
    assert as_whole.status == Status.UNCERTAIN
    assert as_whole.predicted_label == as_leaf.predicted_label          # same prediction, only the status is capped
    assert as_whole.confidence == pytest.approx(as_leaf.confidence)
