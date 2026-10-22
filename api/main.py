"""FastAPI backend for the AI Plant Health Assistant.

Run:  uvicorn api.main:app --port 8000   (Docker / Hugging Face Spaces: port 7860)

A thin HTTP layer over src/service.run_full_analysis(): every prediction,
reliability check and recommendation comes from the existing pipeline. The
model is loaded once at startup, and analyses run one at a time because
Grad-CAM temporarily attaches hooks to the shared model. History is a
local SQLite file (data/history.db).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from api.schemas import (
    QUESTIONS, AboutResponse, AnalyzeResponse, CareContextIn, CareGroupOut, CareOut, ConditionOut,
    DeletedResponse, ErrorResponse, HealthResponse, HistoryDetail, HistoryItem, LibraryCondition, MetaResponse,
    PlantGuideResponse, PlantOption, QuestionOut,
)
from api.serialize import to_response
from src import config
from src.database.history import HistoryError, HistoryStore
from src.inference.analysis import LEAF_CLOSE_UP, PHOTO_TYPES, Status
from src.inference.model import ModelLoadError
from src.inference.predictor import PlantDiseasePredictor, load_thresholds
from src.plants.care_profiles import GROUPS, PROFILES, find_profile, profile_names
from src.plants.catalog import OTHER_OPTION, PLANTVILLAGE_LABELS, PlantCatalog
from src.plants.conditions import get_condition
from src.recommendations.engine import GENERAL_LIMITATIONS
from src.service import run_full_analysis

logger = logging.getLogger("plant_health_api")

UPLOAD_TOO_LARGE = f"The file is larger than {config.MAX_UPLOAD_MB} MB. Please upload a smaller photo."
FORM_OVERHEAD_BYTES = 512 * 1024   # multipart boundaries and the small text fields
_DEFAULT = object()


def _error(status_code: int, error: str, message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(error=error, message=message).model_dump(), status_code=status_code)


def _load_predictor():
    try:
        return PlantDiseasePredictor(), None
    except ModelLoadError as exc:
        logger.error("Model load failed: %s", exc)
    except Exception:
        logger.exception("Unexpected model load failure")
    return None, "The disease model could not be loaded. Only general guidance is available."


def _load_history() -> HistoryStore | None:
    try:
        return HistoryStore(config.HISTORY_DB_PATH)
    except HistoryError as exc:
        logger.error("History unavailable: %s", exc)
        return None


def create_app(predictor=_DEFAULT, history=_DEFAULT) -> FastAPI:
    """Build the app. Tests inject a predictor (or None) and a history store (or None)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is _DEFAULT:
            app.state.predictor, app.state.model_error = await anyio.to_thread.run_sync(_load_predictor)
        else:
            app.state.predictor = predictor
            app.state.model_error = None if predictor is not None else "The disease model is not loaded."
        p = app.state.predictor
        app.state.catalog = p.catalog if p is not None else PlantCatalog(PLANTVILLAGE_LABELS)
        app.state.thresholds = p.thresholds if p is not None else load_thresholds()
        app.state.history = _load_history() if history is _DEFAULT else history
        app.state.lock = threading.Lock()
        yield

    app = FastAPI(title="AI Plant Health Assistant API", version="1.0.0", lifespan=lifespan,
                  responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})

    origins = os.environ.get("PLANT_API_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

    # ---- errors: one shape, no tracebacks, paths or internals ----

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        code = {404: "not_found", 405: "method_not_allowed", 413: "upload_too_large",
                503: "unavailable"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        fields = sorted({str(e["loc"][-1]) for e in exc.errors() if e.get("loc")})
        return _error(422, "invalid_request", "Invalid or missing fields: " + ", ".join(fields))

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception):
        logger.exception("Unhandled API error")
        return _error(500, "internal_error", "Something went wrong on the server. Please try again.")

    @app.middleware("http")
    async def limit_upload_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > config.MAX_UPLOAD_BYTES + FORM_OVERHEAD_BYTES:
            return _error(413, "upload_too_large", UPLOAD_TOO_LARGE)
        return await call_next(request)

    def history_store(request: Request) -> HistoryStore:
        store = request.app.state.history
        if store is None:
            raise HTTPException(503, "History is unavailable because the local database could not be opened.")
        return store

    # ---- endpoints ----

    @app.get("/api/health", response_model=HealthResponse)
    def health(request: Request):
        p = request.app.state.predictor
        return HealthResponse(
            status="ok", model_loaded=p is not None,
            pest_screen=getattr(getattr(p, "pest_screen", None), "method", None),
            symptom_check=getattr(p, "symptom_screen", None) is not None,
            history_available=request.app.state.history is not None)

    @app.get("/api/meta", response_model=MetaResponse)
    def meta(request: Request):
        catalog: PlantCatalog = request.app.state.catalog
        names = sorted(set(catalog.display_names()) | set(profile_names()), key=str.lower)
        return MetaResponse(
            plants=[PlantOption(name=n, disease_check=catalog.match(n).supported) for n in names]
                   + [PlantOption(name=OTHER_OPTION, disease_check=False)],
            other_option=OTHER_OPTION, photo_types=PHOTO_TYPES,
            questions=[QuestionOut(field=f, label=l, options=o, multiple=f == "symptoms") for f, l, o in QUESTIONS],
            max_upload_mb=config.MAX_UPLOAD_MB, allowed_extensions=sorted(config.ALLOWED_EXTENSIONS),
            disease_check_crops=catalog.display_names(), model_loaded=request.app.state.predictor is not None)

    @app.post("/api/analyze", response_model=AnalyzeResponse,
              responses={413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
    async def analyze(request: Request, file: UploadFile | None = File(None), plant_name: str = Form(""),
                      photo_type: str = Form(LEAF_CLOSE_UP), context: str = Form("")):
        if photo_type not in PHOTO_TYPES:
            return _error(422, "invalid_photo_type", f"photo_type must be one of: {', '.join(PHOTO_TYPES)}.")
        try:
            ctx = CareContextIn.model_validate(json.loads(context) if context.strip() else {}).to_context()
        except (ValueError, ValidationError):
            return _error(422, "invalid_context", "The questionnaire answers are not valid options.")

        data, filename = None, None
        if file is not None:
            data = await file.read(config.MAX_UPLOAD_BYTES + 1)
            filename = file.filename
            if len(data) > config.MAX_UPLOAD_BYTES:
                return _error(413, "upload_too_large", UPLOAD_TOO_LARGE)

        state = request.app.state

        def run():
            with state.lock:   # Grad-CAM hooks on the shared model are not safe to run concurrently
                return run_full_analysis(data, filename, plant_name, ctx, state.predictor, state.catalog,
                                         state.thresholds, state.history, photo_type=photo_type)

        bundle = await anyio.to_thread.run_sync(run)
        if bundle.result.status == Status.INVALID_INPUT:
            return _error(422, "invalid_input", bundle.result.message)
        return to_response(bundle, state.catalog)

    @app.get("/api/history", response_model=list[HistoryItem])
    def history_list(request: Request, limit: int = Query(100, ge=1, le=200)):
        try:
            records = history_store(request).recent(limit)
        except HistoryError as exc:
            raise HTTPException(503, str(exc))
        return [HistoryItem(**{k: getattr(r, k) for k in HistoryItem.model_fields}) for r in records]

    @app.get("/api/history/{record_id}", response_model=HistoryDetail)
    def history_get(request: Request, record_id: int):
        try:
            record = history_store(request).get(record_id)
        except HistoryError as exc:
            raise HTTPException(503, str(exc))
        if record is None:
            raise HTTPException(404, "No saved analysis with this id.")
        return HistoryDetail(**{k: getattr(record, k) for k in HistoryDetail.model_fields})

    @app.delete("/api/history/{record_id}", response_model=DeletedResponse)
    def history_delete(request: Request, record_id: int):
        try:
            deleted = history_store(request).delete(record_id)
        except HistoryError as exc:
            raise HTTPException(503, str(exc))
        if not deleted:
            raise HTTPException(404, "No saved analysis with this id.")
        return DeletedResponse(deleted=1)

    @app.delete("/api/history", response_model=DeletedResponse)
    def history_clear(request: Request):
        try:
            return DeletedResponse(deleted=history_store(request).clear())
        except HistoryError as exc:
            raise HTTPException(503, str(exc))

    @app.get("/api/plants/{name}", response_model=PlantGuideResponse)
    def plant_guide(request: Request, name: str):
        catalog: PlantCatalog = request.app.state.catalog
        profile, species = find_profile(name), catalog.match(name).species
        if profile is None and species is None:
            raise HTTPException(404, "No care guide or disease check for this plant name.")
        conditions = [_condition(label) for label in (species.labels if species else []) if get_condition(label)]
        care = None
        if profile is not None:
            g = profile.care
            care = CareOut(group=g.label, watering=g.watering, light=g.light, pests=g.pests, problems=g.problems,
                           note=profile.note)
        return PlantGuideResponse(name=profile.name if profile else species.display_name,
                                  disease_check=species is not None, care=care, conditions=conditions,
                                  coverage_warning=species.coverage_warning if species else None)

    @app.get("/api/conditions", response_model=list[LibraryCondition])
    def conditions(request: Request):
        """Reference information for every class the disease model knows (Disease Library)."""
        catalog: PlantCatalog = request.app.state.catalog
        return [LibraryCondition(crop=catalog.species_of(label).display_name, **_condition(label).model_dump())
                for label in catalog.class_names if get_condition(label)]

    @app.get("/api/care-groups", response_model=list[CareGroupOut])
    def care_groups():
        """General care guidance per plant group, with the plants in each group (Care Tips)."""
        return [CareGroupOut(key=key, label=g.label, watering=g.watering, light=g.light, pests=g.pests,
                             problems=g.problems,
                             plants=sorted(p.name for p in PROFILES if p.group == key))
                for key, g in GROUPS.items()]

    @app.get("/api/about", response_model=AboutResponse)
    def about(request: Request):
        state = request.app.state
        try:
            with open(config.CALIBRATION_PATH, encoding="utf-8") as f:
                cal = json.load(f)
        except (OSError, ValueError):
            cal = {}
        p = state.predictor
        model = {"architecture": "MobileNetV2", "classes": len(state.catalog.class_names),
                 "crops": len(state.catalog.species), "image_size": getattr(p, "image_size", None),
                 "temperature": state.thresholds.get("temperature"), "loaded": p is not None}
        return AboutResponse(model=model, disease_test=_disease_test(cal), pest_screen=_pest_metrics(cal),
                             symptom_check=_symptom_metrics(cal), limitations=GENERAL_LIMITATIONS)

    return app


def _condition(label: str) -> ConditionOut:
    c = get_condition(label)
    return ConditionOut(label=label, name=c.name, category=c.category, cause=c.cause, symptoms=c.symptoms,
                        inspect=c.inspect, prevention=c.prevention, escalate=c.escalate)


# ---- measured metrics, copied from calibration.json (never hand-typed) ----

def _disease_test(cal: dict) -> dict | None:
    t = cal.get("test_report")
    if not t:
        return None
    sel = t.get("selective", {})
    return {"images": t.get("n"), "accuracy": t.get("accuracy"), "accepted_share": sel.get("coverage"),
            "accuracy_when_accepted": sel.get("accuracy_on_accepted"), "unfamiliar_flag_rate": t.get("ood_flag_rate"),
            "dataset": "PlantVillage test split (lab photos of single leaves)"}


def _pest_metrics(cal: dict) -> dict | None:
    e, r = cal.get("pest_screen_ip102_eval"), cal.get("pest_screen_report", {})
    if not e:
        return None
    return {"ip102_test_images": e.get("test_images"), "ip102_hit_rate": e.get("test_hit_rate"),
            "pest_free_flag_rate": r.get("pest_free_flag_rate", e.get("pest_free_flag_rate"))}


def _symptom_metrics(cal: dict) -> dict | None:
    s = cal.get("symptom_screen_report")
    if not s:
        return None
    return {"test_images": s.get("test_images"), "share_clear": s.get("test_share_clear"),
            "accuracy_when_clear": s.get("test_accuracy_when_clear"),
            "share_two_patterns": s.get("test_share_two_patterns"),
            "true_pattern_in_pair": s.get("test_true_pattern_in_pair"),
            "dataset": "PlantVillage test photos (lab photos of single leaves)"}


app = create_app()
