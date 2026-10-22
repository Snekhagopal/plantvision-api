"""Pest-screen integration, Grad-CAM attention check and plant care guides."""

import pytest

from src.explainability.gradcam import GradCamResult
from src.inference.analysis import Status, analyze
from src.plants.care_profiles import GROUPS, PROFILES, find_profile, profile_names
from src.plants.catalog import normalize_name
from src.recommendations.context import CareContext
from src.recommendations.engine import build_recommendations
from src.service import apply_attention_check
from tests.conftest import requires_checkpoint
from pathlib import Path

PEST_PHOTO = Path(__file__).parent / "fixtures" / "whitefly_tomato_leaf.jpg"


# ---- pests make disease predictions unreliable ----

def test_visible_pests_downgrade_confident_disease(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "Tomato", fake_factory({"Tomato___Late_blight": 0.95}, pest_score=0.95),
                catalog, thresholds)
    assert r.status == Status.UNCERTAIN
    assert "insects or eggs" in r.message
    rec = build_recommendations(r)
    assert "Possible insects or eggs" in rec.summary
    assert rec.sections[1].key == "pests"           # right after immediate actions
    assert "insects or eggs" in rec.section("actions").items[0]


def test_spider_mite_prediction_consistent_with_pest_flag(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "Tomato",
                fake_factory({"Tomato___Spider_mites Two-spotted_spider_mite": 0.95}, pest_score=0.95),
                catalog, thresholds)
    assert r.status == Status.LIKELY


def test_user_reported_insects_do_not_downgrade(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "Tomato", fake_factory({"Tomato___Late_blight": 0.95}), catalog, thresholds,
                CareContext(insects_visible="Yes"))
    assert r.status == Status.LIKELY and r.pest_flag


def test_pests_flagged_for_unsupported_plants(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "Hibiscus", fake_factory(pest_score=0.95), catalog, thresholds)
    assert r.status == Status.UNSUPPORTED_PLANT and r.pest_flag
    rec = build_recommendations(r)
    assert rec.section("pests") and rec.section("care")


# ---- Grad-CAM attention check ----

def make_likely(fake_factory, catalog, thresholds, leaf_bytes):
    return analyze(leaf_bytes, "x.jpg", "Tomato", fake_factory(), catalog, thresholds)


@pytest.mark.parametrize("attention,expected", [(0.1, Status.UNCERTAIN), (0.29, Status.UNCERTAIN),
                                                (0.31, Status.LIKELY), (None, Status.LIKELY)])
def test_attention_check(fake_factory, catalog, thresholds, leaf_bytes, attention, expected):
    thresholds["gradcam_min_leaf_attention"] = 0.30
    r = make_likely(fake_factory, catalog, thresholds, leaf_bytes)
    apply_attention_check(r, GradCamResult(object(), None, leaf_attention=attention), thresholds)
    assert r.status == expected
    if expected == Status.UNCERTAIN:
        assert any("background" in w for w in r.warnings)


def test_attention_check_ignores_failed_gradcam(fake_factory, catalog, thresholds, leaf_bytes):
    r = make_likely(fake_factory, catalog, thresholds, leaf_bytes)
    apply_attention_check(r, GradCamResult(None, None, "err"), thresholds)
    assert r.status == Status.LIKELY


# ---- plant list and care guides ----

def test_over_100_plants_with_valid_groups():
    assert len(PROFILES) >= 100
    assert all(p.group in GROUPS for p in PROFILES)
    assert len(profile_names()) == len(set(profile_names()))


def test_aliases_do_not_collide():
    seen = {}
    for p in PROFILES:
        for key in (p.name, *p.aliases):
            k = normalize_name(key)
            assert seen.setdefault(k, p.name) == p.name, f"{key} used by {seen[k]} and {p.name}"


def test_every_supported_species_has_a_profile(catalog):
    for name in catalog.display_names():
        assert find_profile(name) is not None, name


def test_profile_aliases_never_point_to_a_different_supported_species(catalog):
    for p in PROFILES:
        for key in (p.name, *p.aliases):
            m = catalog.match(key)
            assert not m.supported or m.species.display_name == p.name, key


@pytest.mark.parametrize("name,expected", [("money plant", "Pothos"), ("Tulsi", "Holy basil"),
                                           ("brinjal", "Eggplant"), ("bhindi", "Okra"), ("karela", "Bitter gourd"),
                                           ("snake plant", "Snake plant"), ("Rose", "Rose")])
def test_common_names_resolve(name, expected):
    assert find_profile(name).name == expected


def test_unsupported_plant_gets_care_guide_not_disease(fake_factory, catalog, thresholds, leaf_bytes):
    r = analyze(leaf_bytes, "x.jpg", "money plant", fake_factory(), catalog, thresholds)
    assert r.status == Status.UNSUPPORTED_PLANT
    rec = build_recommendations(r)
    care = rec.section("care")
    assert care and care.title.startswith("Care guide: Pothos")
    assert any("not a diagnosis" in i for i in care.items)
    assert rec.section("disease") is None
    assert "care guide" in rec.summary


def test_supported_plant_also_gets_care_guide(fake_factory, catalog, thresholds, leaf_bytes):
    rec = build_recommendations(analyze(leaf_bytes, "x.jpg", "Tomato", fake_factory(), catalog, thresholds))
    assert any("blossom-end rot" in i for i in rec.section("care").items)


def test_care_guides_contain_no_chemicals_or_doses():
    import re
    text = " ".join(f"{p.note} {g.watering} {g.light} {g.pests} {g.problems}"
                    for p in PROFILES for g in [p.care]).lower()
    for banned in ("fungicide", "insecticide", "pesticide", "spray with", "copper", "neem oil"):
        assert banned not in text
    assert not re.search(r"\d+\s?(ml|mg|g|oz|tbsp|tsp)\b", text)


# ---- real CLIP screen on the user's whitefly photo ----

@pytest.fixture(scope="module")
def clip_screen():
    from src.inference.pest_screen import ClipPestScreen
    screen = ClipPestScreen.from_cache()
    if screen is None:
        pytest.skip("CLIP model not cached")
    return screen


def test_clip_flags_user_whitefly_photo(clip_screen):
    from PIL import Image
    from src.inference.predictor import load_thresholds
    result = clip_screen.score(Image.open(PEST_PHOTO))
    assert result.method == "clip"
    assert result.score >= load_thresholds()["pest_screen_threshold"]
    assert "insects" in result.top[0][0] or "eggs" in result.top[0][0]


@requires_checkpoint
def test_user_photo_end_to_end_no_longer_confident_late_blight(clip_screen):
    from src.inference.predictor import PlantDiseasePredictor
    from src.service import run_full_analysis
    p = PlantDiseasePredictor()
    b = run_full_analysis(PEST_PHOTO.read_bytes(), "leaf.jpg", "Tomato", None, p, p.catalog, p.thresholds)
    assert b.result.status != Status.LIKELY
    assert b.result.pest_flag and "image screen" in b.result.pest_flag_sources
    assert b.recommendation.sections[1].key == "pests"

