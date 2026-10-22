"""General symptom check for plants without a disease model, and output for every listed plant."""

import numpy as np
import pytest

from src import config
from src.inference.analysis import Status, analyze
from src.inference.symptom_screen import SYMPTOM_PROMPTS, SYMPTOMS, SymptomResult, SymptomScreen, symptom_of
from src.plants.care_profiles import profile_names
from src.plants.catalog import OTHER_OPTION, PLANTVILLAGE_LABELS, PlantCatalog
from src.recommendations.engine import build_recommendations
from src.recommendations.symptoms import SYMPTOM_ADVICE
from tests.conftest import probs_for

CHEMICALS = ("fungicide", "pesticide", "insecticide", "copper", "sulfur", "neem", "ml/", "g/l")


def symptom(key="spots", p=0.9, clear=True):
    return SymptomResult(key, p, {key: p}, clear=clear)


def all_listed_plants(catalog):
    """Exactly the options offered by the app's plant selector."""
    return sorted(set(catalog.display_names()) | set(profile_names()), key=str.lower) + [OTHER_OPTION]


# ---- label grouping and screen ----

def test_every_label_except_spider_mites_has_a_symptom_group():
    groups = {label: symptom_of(label) for label in PLANTVILLAGE_LABELS}
    assert [l for l, g in groups.items() if g is None] == ["Tomato___Spider_mites Two-spotted_spider_mite"]
    assert set(groups.values()) - {None} == set(SYMPTOMS)
    assert set(SYMPTOM_ADVICE) == set(SYMPTOMS)


class FakeClip:
    """Each pattern's prompts embed to that pattern's own axis."""

    def embed_texts(self, texts):
        import torch
        owner = {t: i for i, k in enumerate(SYMPTOMS) for t in SYMPTOM_PROMPTS[k]}
        return torch.eye(len(SYMPTOMS))[[owner[t] for t in texts]]


def test_screen_prediction_and_clear_threshold():
    screen = SymptomScreen(FakeClip(), threshold=0.6)
    assert set(SYMPTOM_PROMPTS) == set(SYMPTOMS)
    r = screen.predict(np.array([0, 1.0, 0, 0, 0]))
    assert r.key == "spots" and r.clear and abs(sum(r.probabilities.values()) - 1) < 1e-9
    unclear = screen.predict(np.array([0.004, 0.005, 0, 0, 0]))
    assert unclear.key == "spots" and not unclear.clear and unclear.keys == []


def test_two_close_patterns_are_reported_together():
    screen = SymptomScreen(FakeClip(), threshold=0.6, pair_threshold=0.75)
    r = screen.predict(np.array([0, 0.05, 0, 0, 0.0495]))   # spots ~ mottling, others far behind
    assert not r.clear and r.mixed and r.keys == ["spots", "mottling"]


def test_two_pattern_result_gives_advice_for_both(fake_factory, catalog, thresholds, leaf_bytes):
    mixed = SymptomResult("spots", 0.48, {"spots": 0.48, "mottling": 0.47}, clear=False, second="mottling")
    r = analyze(leaf_bytes, "x.jpg", "Hibiscus", fake_factory(symptom=mixed), catalog, thresholds)
    rec = build_recommendations(r)
    assert "signs of both **Leaf spots or blight** and **Mosaic, mottling or yellowing**" in rec.summary
    actions = rec.section("actions").items
    assert SYMPTOM_ADVICE["spots"]["actions"][0] in actions and SYMPTOM_ADVICE["mottling"]["actions"][0] in actions


def test_other_option_is_called_plant_in_summary(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", OTHER_OPTION, fake_factory(symptom=symptom()), catalog, thresholds)
    assert "your plant leaf" in build_recommendations(r).summary


# ---- unsupported plants get a symptom result ----

def test_unsupported_plant_reports_symptom_pattern(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "Hibiscus", fake_factory(symptom=symptom("spots")), catalog, thresholds)
    assert r.status == Status.UNSUPPORTED_PLANT and r.symptom.key == "spots"
    assert r.status_label.startswith("General symptom check")
    assert not r.shows_candidate                     # never a PlantVillage disease label
    rec = build_recommendations(r)
    assert "**Leaf spots or blight**" in rec.summary and "cannot be named" in rec.summary
    assert SYMPTOM_ADVICE["spots"]["actions"][0] in rec.section("actions").items
    assert any("general symptom check" in l for l in rec.section("limitations").items)
    assert rec.section("care") is not None           # care guide still included


def test_unclear_symptom_pattern_gives_no_specific_advice(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "Hibiscus", fake_factory(symptom=symptom("rust", 0.4, clear=False)),
                catalog, thresholds)
    rec = build_recommendations(r)
    assert "no clear pattern" in rec.summary
    assert SYMPTOM_ADVICE["rust"]["actions"][0] not in rec.section("actions").items


def test_pests_override_symptom_summary(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "Hibiscus", fake_factory(symptom=symptom("spots"), pest_score=0.95),
                catalog, thresholds)
    rec = build_recommendations(r)
    assert rec.summary.startswith("**Possible insects or eggs")
    assert "symptom check is unreliable" in rec.summary


def test_without_clip_unsupported_plant_falls_back_to_care_guide(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "Hibiscus", fake_factory(), catalog, thresholds)
    assert r.status == Status.UNSUPPORTED_PLANT and r.symptom is None
    assert "not trained on 'Hibiscus'" in build_recommendations(r).summary


@pytest.mark.parametrize("key", list(SYMPTOMS))
def test_symptom_advice_is_conservative(key):
    text = " ".join(i for items in SYMPTOM_ADVICE[key].values() for i in items).lower()
    assert not any(c in text for c in CHEMICALS)


# ---- every plant in the selector gets an output ----

def test_every_listed_plant_gets_an_output(fake_factory, catalog, thresholds, leaf_bytes):
    plants = all_listed_plants(catalog)
    assert len(plants) > 140
    for plant in plants:
        match = catalog.match(plant)
        if match.supported:
            # Blueberry, raspberry and soybean only have a healthy class.
            label = next((l for l in match.species.labels if not l.endswith("___healthy")),
                         match.species.labels[0])
            predictor = fake_factory({label: 0.95}, symptom=symptom())
        else:
            predictor = fake_factory(symptom=symptom("mottling"))
        r = analyze(leaf_bytes, "x.jpg", plant, predictor, catalog, thresholds)
        rec = build_recommendations(r)
        if match.supported:
            assert r.shows_candidate and r.predicted_label == label, plant
            # Crop-specific disease model stays the primary result; the symptom check is shown separately.
            assert not r.symptom_is_primary and rec.section("symptoms") is not None, plant
        else:
            assert r.status == Status.UNSUPPORTED_PLANT and r.symptom.key == "mottling", plant
            assert "**Mosaic, mottling or yellowing**" in rec.summary, plant
        assert rec.summary and rec.section("actions").items, plant


# ---- real models ----

@pytest.mark.skipif(not config.MOBILENET_CHECKPOINT.exists(), reason="disease checkpoint not available")
def test_real_symptom_check_on_unseen_plant_photo():
    from pathlib import Path
    from src.inference.predictor import PlantDiseasePredictor
    predictor = PlantDiseasePredictor()
    if predictor.symptom_screen is None:
        pytest.skip("CLIP not cached")
    photo = Path(__file__).parent / "fixtures" / "hibiscus_leaf_spots.jpg"
    r = analyze(photo.read_bytes(), photo.name, "Hibiscus", predictor, PlantCatalog(PLANTVILLAGE_LABELS),
                predictor.thresholds)
    assert r.status == Status.UNSUPPORTED_PLANT
    # The leaf shows dead brown spots on blotchy yellowing: spots and/or mottling, never "healthy".
    assert r.symptom is not None and r.symptom.keys
    assert set(r.symptom.keys) <= {"spots", "mottling"}
