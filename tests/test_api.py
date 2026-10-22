"""FastAPI backend: every endpoint, all 10 result types, uploads, errors, history and concurrency."""

import io
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from src import config
from src.database.history import HistoryStore
from src.inference.symptom_screen import SymptomResult
from src.recommendations.treatments import SAFETY
from tests.conftest import make_leaf_image, requires_checkpoint, to_bytes


def client_for(predictor, tmp_path=None, history=True):
    store = HistoryStore(tmp_path / "history.db") if (history and tmp_path is not None) else None
    return TestClient(create_app(predictor=predictor, history=store), raise_server_exceptions=False)


def post(client, image=None, plant="Tomato", filename="leaf.jpg", **fields):
    files = {"file": (filename, image, "image/jpeg")} if image is not None else None
    return client.post("/api/analyze", data={"plant_name": plant, **fields}, files=files)


def assert_error(resp, status, error):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error", "message"} and body["error"] == error
    return body["message"]


@pytest.fixture
def leaf():
    return to_bytes(make_leaf_image())


# ---- health, meta ----

def test_health(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        body = c.get("/api/health").json()
    assert body == {"status": "ok", "model_loaded": True, "pest_screen": None, "symptom_check": False,
                    "history_available": True}


def test_health_without_model_or_history():
    with client_for(None) as c:
        body = c.get("/api/health").json()
    assert body["model_loaded"] is False and body["history_available"] is False


def test_meta_lists_every_selector_plant_and_questionnaire(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        m = c.get("/api/meta").json()
    names = {p["name"]: p["disease_check"] for p in m["plants"]}
    assert len(names) > 140 and names["Tomato"] is True and names["Hibiscus"] is False
    assert m["plants"][-1]["name"] == m["other_option"] == "Other / Not listed"
    assert m["photo_types"] == {"leaf": "Leaf close-up", "whole_plant": "Whole plant"}
    assert len(m["questions"]) == 10 and m["questions"][0]["multiple"] is True
    assert m["max_upload_mb"] == 10 and m["allowed_extensions"] == ["jpeg", "jpg", "png", "webp"]
    assert len(m["disease_check_crops"]) == 14


# ---- analyze: all 10 result types ----

def symptom(key="spots"):
    return SymptomResult(key, 0.9, {key: 0.9}, clear=True)


def test_analyze_likely_match_full_response(fake_factory, tmp_path, leaf):
    with client_for(fake_factory({"Tomato___Early_blight": 0.97}, symptom=symptom()), tmp_path) as c:
        r = post(c, leaf)
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "likely_match" and b["photo_type"] == "leaf"
    assert b["prediction"]["label"] == "Tomato___Early_blight" and b["prediction"]["score"] == pytest.approx(0.97)
    assert b["report"]["disease"]["state"] == "detected" and b["report"]["health"]["state"] == "problem"
    assert b["plant"] == {"input": "Tomato", "matched_species": "Tomato", "disease_check": True}
    assert b["images"]["photo"].startswith("data:image/jpeg;base64,")
    treatment = next(s for s in b["recommendations"] if s["key"] == "treatment")
    assert all(item in treatment["items"] for item in SAFETY)
    # Symptom observation stays separate from the disease prediction.
    assert b["symptom"]["is_primary"] is False and b["symptom"]["key"] == "spots"
    assert "Visible pattern" not in b["summary"]
    assert isinstance(b["history_id"], int)


CASES = {
    "likely_match": dict(weights={"Tomato___Early_blight": 0.97}),
    "uncertain": dict(weights={"Tomato___Early_blight": 0.5}),
    "low_quality": dict(weights={"Tomato___Early_blight": 0.97}, blurry=True),
    "species_mismatch": dict(weights={"Apple___Apple_scab": 0.95}),
    "unfamiliar_image": dict(ood_score=99.0),
    "possible_non_plant": dict(gray=True),
    "unsupported_plant": dict(plant="Hibiscus", symptom=symptom("mottling")),
    "model_unavailable": dict(no_model=True),
    "analysis_error": dict(raises=True),
}


@pytest.mark.parametrize("status", list(CASES))
def test_analyze_result_types(status, fake_factory, tmp_path, leaf, gray_bytes):
    case = dict(CASES[status])
    plant = case.pop("plant", "Tomato")
    image = gray_bytes if case.pop("gray", False) else leaf
    blurry = case.pop("blurry", False)
    predictor = None if case.pop("no_model", False) else fake_factory(case.pop("weights", None), **case)
    if blurry:
        predictor.thresholds["blur_min_variance"] = 1e9
    with client_for(predictor, tmp_path) as c:
        r = post(c, image, plant)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["status"] == status
    assert b["summary"] and b["recommendations"] and set(b["report"]) == {"health", "symptoms", "pests", "disease"}
    if status == "unsupported_plant":
        assert b["symptom"]["is_primary"] is True and b["report"]["disease"]["state"] == "not_assessed"
    if status == "analysis_error":
        assert "boom" not in r.text                                   # internal error text never leaks


def test_invalid_input_is_the_tenth_result_type(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        msg = assert_error(post(c, b"not an image"), 422, "invalid_input")
    assert "could not be read as an image" in msg


# ---- uploads and validation ----

def test_missing_photo(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        assert "upload a photo" in assert_error(post(c, None), 422, "invalid_input")


def test_missing_plant_name(fake_factory, tmp_path, leaf):
    with client_for(fake_factory(), tmp_path) as c:
        assert "plant name" in assert_error(post(c, leaf, plant="  "), 422, "invalid_input")


def test_unsupported_file_type(fake_factory, tmp_path, leaf):
    with client_for(fake_factory(), tmp_path) as c:
        assert "JPG, JPEG, PNG or WebP" in assert_error(post(c, leaf, filename="leaf.gif"), 422, "invalid_input")


def test_webp_upload_is_accepted(fake_factory, tmp_path):
    webp = to_bytes(make_leaf_image(), "WEBP")
    with client_for(fake_factory({"Tomato___Early_blight": 0.97}), tmp_path) as c:
        r = post(c, webp, filename="leaf.webp")
    assert r.status_code == 200 and r.json()["status"] == "likely_match"


def test_png_upload_is_accepted(fake_factory, tmp_path):
    png = to_bytes(make_leaf_image(), "PNG")
    with client_for(fake_factory({"Tomato___Early_blight": 0.97}), tmp_path) as c:
        assert post(c, png, filename="leaf.png").status_code == 200


def test_file_just_over_limit_is_rejected(fake_factory, tmp_path):
    big = b"\xff" * (config.MAX_UPLOAD_BYTES + 10)
    with client_for(fake_factory(), tmp_path) as c:
        assert "larger than 10 MB" in assert_error(post(c, big), 413, "upload_too_large")


def test_huge_request_rejected_before_reading(fake_factory, tmp_path):
    huge = b"\xff" * (config.MAX_UPLOAD_BYTES + 2 * 1024 * 1024)
    with client_for(fake_factory(), tmp_path) as c:
        assert_error(post(c, huge), 413, "upload_too_large")


def test_invalid_photo_type(fake_factory, tmp_path, leaf):
    with client_for(fake_factory(), tmp_path) as c:
        assert_error(post(c, leaf, photo_type="selfie"), 422, "invalid_photo_type")


@pytest.mark.parametrize("context", ["{not json", json.dumps({"soil_moisture": "Flooded"}),
                                     json.dumps({"symptoms": ["Glowing leaves"]}), json.dumps({"colour": "x"})])
def test_invalid_questionnaire(context, fake_factory, tmp_path, leaf):
    with client_for(fake_factory(), tmp_path) as c:
        assert_error(post(c, leaf, context=context), 422, "invalid_context")


def test_questionnaire_answers_reach_recommendations(fake_factory, tmp_path, leaf):
    ctx = json.dumps({"symptoms": ["Yellowing leaves"], "soil_moisture": "Wet / soggy", "insects_visible": "Yes"})
    with client_for(fake_factory({"Tomato___healthy": 0.97}), tmp_path) as c:
        b = post(c, leaf, context=ctx).json()
    keys = [s["key"] for s in b["recommendations"]]
    assert "nutrients" in keys and b["pests"]["sources"] == ["user report"]
    assert b["report"]["pests"]["state"] == "detected"
    watering = next(s for s in b["recommendations"] if s["key"] == "watering")
    assert any("overwatering" in i for i in watering["items"])


# ---- approved rules ----

def test_whole_plant_photo_is_at_most_uncertain(fake_factory, tmp_path, leaf):
    with client_for(fake_factory({"Tomato___Early_blight": 0.99}), tmp_path) as c:
        b = post(c, leaf, photo_type="whole_plant").json()
    assert b["status"] == "uncertain" and b["photo_type"] == "whole_plant"
    assert b["report"]["disease"]["state"] == "uncertain"
    assert any("close-up" in w for w in b["warnings"])
    assert "treatment" not in [s["key"] for s in b["recommendations"]]


def test_no_treatment_from_yellowing_alone(fake_factory, tmp_path, leaf):
    ctx = json.dumps({"symptoms": ["Yellowing leaves", "Brown or crispy leaf edges"]})
    with client_for(fake_factory(symptom=symptom("mottling")), tmp_path) as c:
        b = post(c, leaf, plant="Hibiscus", context=ctx).json()
    assert "treatment" not in [s["key"] for s in b["recommendations"]]


# ---- history ----

def test_history_lifecycle(fake_factory, tmp_path, leaf):
    with client_for(fake_factory({"Tomato___Early_blight": 0.97}), tmp_path) as c:
        first = post(c, leaf).json()["history_id"]
        second = post(c, leaf).json()["history_id"]
        items = c.get("/api/history").json()
        assert [i["id"] for i in items] == [second, first]
        assert items[0]["condition_name"] == "Tomato early blight" and items[0]["confidence"] == pytest.approx(0.97)
        detail = c.get(f"/api/history/{first}").json()
        assert detail["recommendations"]["sections"] and detail["predicted_label"] == "Tomato___Early_blight"
        assert c.get("/api/history?limit=1").json()[0]["id"] == second
        assert c.delete(f"/api/history/{first}").json() == {"deleted": 1}
        assert_error(c.get(f"/api/history/{first}"), 404, "not_found")
        assert_error(c.delete(f"/api/history/{first}"), 404, "not_found")
        assert c.delete("/api/history").json() == {"deleted": 1}
        assert c.get("/api/history").json() == []


def test_invalid_input_is_not_saved(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        post(c, b"junk")
        assert c.get("/api/history").json() == []


def test_history_limit_is_validated(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        assert_error(c.get("/api/history?limit=0"), 422, "invalid_request")


def test_history_unavailable(fake_factory, leaf):
    with client_for(fake_factory({"Tomato___Early_blight": 0.97}), history=False) as c:
        assert_error(c.get("/api/history"), 503, "unavailable")
        assert_error(c.delete("/api/history"), 503, "unavailable")
        b = post(c, leaf).json()                       # analysis still works without history
    assert b["status"] == "likely_match" and b["history_id"] is None


# ---- plant guide, about ----

def test_plant_guide_supported_crop(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        g = c.get("/api/plants/Tomato").json()
    assert g["disease_check"] is True and len(g["conditions"]) == 10 and g["care"]["watering"]
    blight = next(x for x in g["conditions"] if x["label"] == "Tomato___Late_blight")
    assert blight["symptoms"] and blight["prevention"]


def test_plant_guide_care_only_plant_and_alias(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        g = c.get("/api/plants/gudhal").json()                    # alias of Hibiscus
        assert g["name"] == "Hibiscus" and g["disease_check"] is False and g["conditions"] == []
        assert "whiteflies" in g["care"]["note"]
        assert_error(c.get("/api/plants/Unobtainium"), 404, "not_found")


def test_condition_library_covers_every_class(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        lib = c.get("/api/conditions").json()
    assert len(lib) == 38 and {x["crop"] for x in lib} >= {"Tomato", "Apple", "Corn (maize)"}
    scab = next(x for x in lib if x["label"] == "Apple___Apple_scab")
    assert scab["name"] == "Apple scab" and scab["category"] == "Fungal disease" and scab["symptoms"]


def test_care_groups_list_every_profile_once(fake_factory, tmp_path):
    from src.plants.care_profiles import PROFILES
    with client_for(fake_factory(), tmp_path) as c:
        groups = c.get("/api/care-groups").json()
    plants = [p for g in groups for p in g["plants"]]
    assert sorted(plants) == sorted(p.name for p in PROFILES)
    assert all(g["watering"] and g["light"] for g in groups)


def test_prediction_carries_reference_symptoms(fake_factory, tmp_path, leaf):
    with client_for(fake_factory({"Tomato___Early_blight": 0.97}), tmp_path) as c:
        p = post(c, leaf).json()["prediction"]
    assert p["symptoms"] and p["prevention"]


def test_about_reports_recorded_metrics_only(fake_factory, tmp_path):
    cal = json.loads(config.CALIBRATION_PATH.read_text())
    with client_for(fake_factory(), tmp_path) as c:
        a = c.get("/api/about").json()
    assert a["model"]["architecture"] == "MobileNetV2" and a["model"]["classes"] == 38 and a["model"]["crops"] == 14
    assert a["disease_test"]["accuracy"] == cal["test_report"]["accuracy"]
    assert a["pest_screen"]["ip102_hit_rate"] == cal["pest_screen_ip102_eval"]["test_hit_rate"]
    assert a["symptom_check"]["accuracy_when_clear"] == cal["symptom_screen_report"]["test_accuracy_when_clear"]
    assert a["limitations"]


# ---- error handling ----

def test_unexpected_errors_hide_internals(fake_factory, tmp_path, leaf, monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("secret at /Users/someone/plant-health-detector/src/service.py")
    monkeypatch.setattr("api.main.run_full_analysis", explode)
    with client_for(fake_factory(), tmp_path) as c:
        r = post(c, leaf)
    assert_error(r, 500, "internal_error")
    assert "/Users" not in r.text and "secret" not in r.text and "Traceback" not in r.text


def test_unknown_route_and_method(fake_factory, tmp_path):
    with client_for(fake_factory(), tmp_path) as c:
        assert_error(c.get("/api/nope"), 404, "not_found")
        assert_error(c.put("/api/history"), 405, "method_not_allowed")


# ---- concurrency ----

def test_analyses_run_one_at_a_time(fake_factory, tmp_path, leaf):
    predictor = fake_factory({"Tomato___Early_blight": 0.97})
    active, peak, guard = [0], [0], threading.Lock()
    original = predictor.predict

    def slow_predict(image):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with guard:
            active[0] -= 1
        return original(image)

    predictor.predict = slow_predict
    with client_for(predictor, tmp_path) as c:
        results = []
        threads = [threading.Thread(target=lambda: results.append(post(c, leaf).status_code)) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert results == [200, 200, 200] and peak[0] == 1


# ---- real model: parity with the service used by Streamlit ----

@pytest.fixture(scope="module")
def real_predictor():
    from src.inference.predictor import PlantDiseasePredictor
    return PlantDiseasePredictor()


@requires_checkpoint
@pytest.mark.parametrize("image,plant,photo_type", [
    ("tests/fixtures/hibiscus_leaf_spots.jpg", "Hibiscus", "leaf"),
    ("tests/fixtures/whitefly_tomato_leaf.jpg", "Tomato", "leaf"),
    ("data/raw/train/Apple___Apple_scab/image_0000416.jpg", "Apple", "leaf"),
    ("data/raw/train/Apple___Apple_scab/image_0000416.jpg", "Apple", "whole_plant"),
])
def test_real_model_matches_service(image, plant, photo_type, real_predictor, tmp_path):
    from src.service import run_full_analysis
    path = config.ROOT / image
    if not path.exists():
        pytest.skip(f"{image} not available")
    data = path.read_bytes()
    direct = run_full_analysis(data, path.name, plant, None, real_predictor, real_predictor.catalog,
                               real_predictor.thresholds, photo_type=photo_type)
    with client_for(real_predictor, tmp_path) as c:
        r = post(c, data, plant, filename=path.name, photo_type=photo_type)
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == direct.result.status.value and b["summary"] == direct.recommendation.summary
    assert [s["title"] for s in b["recommendations"]] == [s.title for s in direct.recommendation.sections]
    if direct.result.shows_candidate:
        assert b["prediction"]["label"] == direct.result.predicted_label
        assert b["prediction"]["score"] == pytest.approx(direct.result.confidence)
        assert b["images"]["gradcam"].startswith("data:image/jpeg;base64,")
        Image.open(io.BytesIO(__import__("base64").b64decode(b["images"]["gradcam"].split(",", 1)[1])))
    if photo_type == "whole_plant":
        assert b["status"] != "likely_match"
