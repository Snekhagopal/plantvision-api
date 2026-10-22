"""Plant-health report: the four assessments (health, symptoms, pests, disease).

A presentation-neutral view of one AnalysisResult, shared by the Streamlit UI
and the API. It only re-describes evidence the pipeline already produced; it
never computes a new prediction. Symptom observations are kept separate from
the disease model's prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.inference.analysis import CANDIDATE_STATUSES, WHOLE_PLANT, AnalysisResult, Status
from src.inference.symptom_screen import SYMPTOMS
from src.recommendations.context import CareContext

# Assessment states
HEALTHY, PROBLEM, UNCLEAR = "healthy", "problem", "unclear"                  # overall health only
DETECTED, NOT_DETECTED, UNCERTAIN, NOT_ASSESSED = "detected", "not_detected", "uncertain", "not_assessed"

NO_PHOTO_RESULT = {Status.INVALID_INPUT, Status.ANALYSIS_ERROR, Status.POSSIBLE_NON_PLANT}


@dataclass
class Assessment:
    key: str          # "health" | "symptoms" | "pests" | "disease"
    title: str
    state: str
    headline: str
    details: list[str] = field(default_factory=list)
    score: float | None = None   # model score where one exists (never a probability of being right)


@dataclass
class PlantHealthReport:
    status: str
    status_label: str
    photo_type: str
    health: Assessment
    symptoms: Assessment
    pests: Assessment
    disease: Assessment

    @property
    def assessments(self) -> list[Assessment]:
        return [self.health, self.symptoms, self.pests, self.disease]

    def as_dict(self) -> dict:
        return asdict(self)


def build_report(result: AnalysisResult, ctx: CareContext | None = None) -> PlantHealthReport:
    ctx = ctx or CareContext()
    symptoms, pests, disease = _symptoms(result, ctx), _pests(result), _disease(result)
    return PlantHealthReport(result.status.value, result.status_label, result.photo_type,
                             _health(result, pests, disease), symptoms, pests, disease)


def _health(result: AnalysisResult, pests: Assessment, disease: Assessment) -> Assessment:
    title = "Overall health"
    if result.status in NO_PHOTO_RESULT or (result.status == Status.MODEL_UNAVAILABLE and result.symptom is None):
        return Assessment("health", title, UNCLEAR, "Can't tell from this photo", [result.message])
    if pests.state == DETECTED:
        return Assessment("health", title, PROBLEM, "Signs of a problem: possible pests",
                          ["See the Pests assessment."])
    if result.species is not None:   # crops with a disease model: rely on the validated model
        if disease.state == DETECTED:
            return Assessment("health", title, PROBLEM, "Signs of a problem: possible disease",
                              ["See the Disease assessment."])
        if disease.state == NOT_DETECTED:
            return Assessment("health", title, HEALTHY, "Looks healthy",
                              ["The disease model matched a healthy leaf. One leaf does not clear the whole "
                               "plant, so keep checking other leaves."])
        return Assessment("health", title, UNCLEAR, "Can't tell reliably from this photo", [disease.headline])
    s = result.symptom
    if s is not None and s.keys == ["healthy"]:
        return Assessment("health", title, HEALTHY, "No obvious symptoms",
                          ["The symptom check found no obvious disease pattern. It cannot rule out problems "
                           "that are not visible in the photo."])
    if s is not None and s.keys:
        return Assessment("health", title, PROBLEM, "Signs of a problem: visible symptoms",
                          ["See the Visible symptoms assessment."])
    return Assessment("health", title, UNCLEAR, "Can't tell reliably from this photo",
                      ["Use the inspection steps and questionnaire to narrow it down."])


def _symptoms(result: AnalysisResult, ctx: CareContext) -> Assessment:
    title = "Visible symptoms"
    reported = [f"You reported: {', '.join(s.lower() for s in ctx.symptoms)}."] if ctx.symptoms else []
    s = result.symptom
    if s is None:
        state = DETECTED if ctx.symptoms else NOT_ASSESSED
        headline = "Reported by you" if ctx.symptoms else "Photo symptom check not available"
        return Assessment("symptoms", title, state, headline, reported)
    note = ["Describes what the leaf looks like; it does not name a disease."]
    if s.clear and s.key == "healthy":
        return Assessment("symptoms", title, NOT_DETECTED, "No obvious symptoms in the photo",
                          reported + note, s.probability)
    if s.clear:
        return Assessment("symptoms", title, DETECTED, s.name, [s.description] + reported + note, s.probability)
    if s.mixed:
        names = [SYMPTOMS[k][0] for k in s.keys]
        return Assessment("symptoms", title, DETECTED, " + ".join(names),
                          [SYMPTOMS[k][1] for k in s.keys] + reported + note, s.probability)
    return Assessment("symptoms", title, UNCERTAIN, "No clear pattern in the photo",
                      [f"Closest: {s.name.lower()}."] + reported + note, s.probability)


def _pests(result: AnalysisResult) -> Assessment:
    title = "Pests"
    out = result.model_output
    image, user = "image screen" in result.pest_flag_sources, "user report" in result.pest_flag_sources
    if image:
        hits = out.pest_top if out else []
        details = [f"Looks most like {hits[0][0]}." if hits else "Possible insects or eggs on the leaf.",
                   "This is not a species identification - confirm by looking closely."]
        if user:
            details.append("You also reported visible insects.")
        return Assessment("pests", title, DETECTED, "Possible insects or eggs in the photo", details,
                          out.pest_score if out else None)
    if user:
        return Assessment("pests", title, DETECTED, "Insects reported by you",
                          ["The photo check did not flag insects; identify them with a close-up or hand lens."])
    if out is not None and out.pest_score is not None and result.status not in NO_PHOTO_RESULT:
        return Assessment("pests", title, NOT_DETECTED, "No insects flagged in the photo",
                          ["The check can miss tiny pests such as mites and scale; inspect leaf undersides."],
                          out.pest_score)
    return Assessment("pests", title, NOT_ASSESSED, "Pest check not available for this photo")


def _disease(result: AnalysisResult) -> Assessment:
    title = "Disease"
    if result.status == Status.UNSUPPORTED_PLANT:
        return Assessment("disease", title, NOT_ASSESSED, "No disease model for this plant",
                          ["Disease checks cover 14 crops. See Visible symptoms for what the photo shows."])
    if result.status in CANDIDATE_STATUSES and result.condition is not None:
        cond = result.condition
        if result.status == Status.LIKELY:
            state = NOT_DETECTED if cond.is_healthy else DETECTED
            headline = cond.name if not cond.is_healthy else f"No disease detected ({cond.name.lower()})"
        else:
            state, headline = UNCERTAIN, f"Possibly {cond.name.lower()} (uncertain)"
        details = [result.message]
        if result.photo_type == WHOLE_PLANT:
            details.append("Upload a close-up of one affected leaf for a reliable disease check.")
        return Assessment("disease", title, state, headline, details, result.confidence)
    if result.status in (Status.SPECIES_MISMATCH, Status.UNFAMILIAR):
        return Assessment("disease", title, UNCERTAIN, "No reliable disease result", [result.message])
    return Assessment("disease", title, NOT_ASSESSED, "Disease check not run", [result.message] if result.message else [])
