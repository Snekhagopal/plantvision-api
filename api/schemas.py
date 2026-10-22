"""Pydantic request/response schemas for the Plant Health API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.recommendations import context as q

PhotoType = Literal["leaf", "whole_plant"]


# ---- requests ----

class CareContextIn(BaseModel):
    """Optional questionnaire answers; every value must be one of the offered options."""

    model_config = {"extra": "forbid"}

    symptoms: list[str] = Field(default_factory=list)
    soil_moisture: str = q.UNKNOWN
    watering_frequency: str = q.UNKNOWN
    drainage: str = q.UNKNOWN
    location: str = q.UNKNOWN
    sunlight: str = q.UNKNOWN
    weather: str = q.UNKNOWN
    fertilizer: str = q.UNKNOWN
    insects_visible: str = q.UNKNOWN
    affected_leaves: str = q.UNKNOWN

    @field_validator("symptoms")
    @classmethod
    def _known_symptoms(cls, v: list[str]) -> list[str]:
        unknown = [s for s in v if s not in q.SYMPTOM_OPTIONS]
        if unknown:
            raise ValueError(f"unknown symptom(s): {unknown}")
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def _known_answers(self) -> "CareContextIn":
        for name, _label, options in QUESTIONS:
            if name != "symptoms" and getattr(self, name) not in options:
                raise ValueError(f"invalid value for {name}")
        return self

    def to_context(self) -> q.CareContext:
        return q.CareContext(**self.model_dump())


# (field, label shown in the UI, allowed options) - same wording as the Streamlit questionnaire.
QUESTIONS: list[tuple[str, str, list[str]]] = [
    ("symptoms", "Symptoms you can see", q.SYMPTOM_OPTIONS),
    ("soil_moisture", "Soil moisture 3-5 cm deep", q.SOIL_MOISTURE),
    ("watering_frequency", "Watering frequency", q.WATERING_FREQUENCY),
    ("drainage", "Drainage", q.DRAINAGE),
    ("location", "Where is the plant growing?", q.LOCATION),
    ("affected_leaves", "Which leaves are affected?", q.AFFECTED_LEAVES),
    ("sunlight", "Sunlight", q.SUNLIGHT),
    ("weather", "Recent weather", q.WEATHER),
    ("fertilizer", "Last fertiliser", q.FERTILIZER),
    ("insects_visible", "Insects visible on the plant?", q.INSECTS),
]


# ---- analysis response ----

class AssessmentOut(BaseModel):
    key: str
    title: str
    state: str
    headline: str
    details: list[str]
    score: float | None = None


class ReportOut(BaseModel):
    health: AssessmentOut
    symptoms: AssessmentOut
    pests: AssessmentOut
    disease: AssessmentOut


class PlantOut(BaseModel):
    input: str
    matched_species: str | None
    disease_check: bool


class ScoredLabel(BaseModel):
    label: str
    name: str
    species: str
    score: float


class PredictionOut(BaseModel):
    """The disease model's candidate. Present only for statuses that may show one."""
    label: str
    name: str
    category: str | None
    cause: str | None
    score: float
    alternatives: list[ScoredLabel]
    symptoms: list[str] = Field(default_factory=list, description="Typical signs (reference info, not from the photo)")
    prevention: list[str] = Field(default_factory=list)


class SymptomOut(BaseModel):
    """Photo symptom check - a visible pattern, never a disease name."""
    key: str
    name: str
    description: str
    score: float
    clear: bool
    patterns: list[str]          # 1 if clear, 2 if mixed, 0 if no clear pattern
    is_primary: bool             # main photo result (plants without a disease model)
    scores: dict[str, float]


class PestOut(BaseModel):
    flagged: bool
    sources: list[str]
    score: float | None
    method: str | None
    appearance: list[tuple[str, float]]


class SectionOut(BaseModel):
    key: str
    title: str
    items: list[str]


class QualityOut(BaseModel):
    brightness: float
    blur_variance: float
    vegetation_fraction: float
    issues: list[str]


class ImagesOut(BaseModel):
    photo: str | None = Field(None, description="JPEG data URL of the (EXIF-rotated) photo")
    gradcam: str | None = Field(None, description="JPEG data URL of the Grad-CAM overlay")
    gradcam_note: str | None = None
    leaf_attention: float | None = None


class ModelDetailsOut(BaseModel):
    top_k: list[ScoredLabel]
    plant_mass: float | None
    unfamiliarity_score: float | None
    unfamiliar: bool
    model_top_label: str | None


class AnalyzeResponse(BaseModel):
    status: str
    status_label: str
    message: str
    summary: str = Field(description="May contain **bold** markers; render as text, never as HTML")
    photo_type: PhotoType
    plant: PlantOut
    report: ReportOut
    prediction: PredictionOut | None
    symptom: SymptomOut | None
    pests: PestOut
    recommendations: list[SectionOut]
    quality: QualityOut | None
    images: ImagesOut
    model_details: ModelDetailsOut | None
    warnings: list[str]
    history_id: int | None
    history_error: str | None


class ErrorResponse(BaseModel):
    error: str
    message: str


# ---- other endpoints ----

class PlantOption(BaseModel):
    name: str
    disease_check: bool


class QuestionOut(BaseModel):
    field: str
    label: str
    options: list[str]
    multiple: bool


class MetaResponse(BaseModel):
    plants: list[PlantOption]
    other_option: str
    photo_types: dict[str, str]
    questions: list[QuestionOut]
    max_upload_mb: int
    allowed_extensions: list[str]
    disease_check_crops: list[str]
    model_loaded: bool


class HistoryItem(BaseModel):
    id: int
    created_at: str
    plant_name: str
    status: str
    status_label: str
    condition_name: str | None
    confidence: float | None


class HistoryDetail(HistoryItem):
    predicted_label: str | None
    summary: str
    recommendations: dict


class DeletedResponse(BaseModel):
    deleted: int


class ConditionOut(BaseModel):
    label: str
    name: str
    category: str
    cause: str
    symptoms: list[str]
    inspect: list[str]
    prevention: list[str]
    escalate: str


class LibraryCondition(ConditionOut):
    crop: str


class CareGroupOut(BaseModel):
    key: str
    label: str
    watering: str
    light: str
    pests: str
    problems: str
    plants: list[str]


class CareOut(BaseModel):
    group: str
    watering: str
    light: str
    pests: str
    problems: str
    note: str


class PlantGuideResponse(BaseModel):
    name: str
    disease_check: bool
    care: CareOut | None
    conditions: list[ConditionOut]
    coverage_warning: str | None


class AboutResponse(BaseModel):
    model: dict
    disease_test: dict | None
    pest_screen: dict | None
    symptom_check: dict | None
    limitations: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pest_screen: str | None
    symptom_check: bool
    history_available: bool
