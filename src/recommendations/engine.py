"""Build structured, conservative care recommendations from an AnalysisResult."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src import config
from src.inference.analysis import WHOLE_PLANT, AnalysisResult, Status
from src.inference.symptom_screen import SYMPTOMS, SymptomResult
from src.plants.care_profiles import PlantProfile, find_profile
from src.plants.catalog import OTHER_OPTION
from src.plants.conditions import CATEGORY_SANITATION, HEALTHY, OOMYCETE
from src.recommendations.assessments import (
    PHOTO_CAVEAT_WATER, Assessment, assess_nutrients, assess_pests, assess_watering,
)
from src.recommendations.context import CareContext
from src.recommendations.symptoms import SYMPTOM_ADVICE
from src.recommendations.treatments import treatment_advice

GENERAL_LIMITATIONS = [
    "Image-based predictions are not definitive diagnoses. Confirm with a local agricultural extension "
    "service, plant clinic or plant pathologist before taking costly or irreversible action.",
    "The model was trained on PlantVillage images (single leaves photographed on plain backgrounds); accuracy "
    "on real garden or field photos is unmeasured and likely lower than the reported test accuracy.",
    "Confidence scores are temperature-calibrated on validation data, but they are not the probability that "
    "the diagnosis is correct for your plant.",
]

EXPERT_TRIGGERS = [
    "Symptoms spread quickly or affect many plants.",
    "The plant is valuable, a commercial crop, or a fruit tree.",
    "You are considering any pesticide, fungicide or other chemical treatment.",
    "You suspect a regulated or quarantine disease (e.g. citrus greening, late blight).",
]

SYMPTOM_SEPARATE_NOTE = (
    "This photo check describes what the leaf looks like. It is separate from the disease model's result above "
    "and does not name a disease."
)

WHOLE_PLANT_ACTION = (
    "For a disease check, upload a close-up of one affected leaf: the disease model was trained on leaf "
    "close-ups, so whole-plant photos cannot give a reliable disease result."
)

SYMPTOM_LIMITATION = (
    "The general symptom check describes what the leaf looks like (e.g. spots, mildew, mottling), not which "
    "disease causes it. When it reports a clear pattern, it was right {acc} of the time on lab photos of "
    "leaves; real garden photos are harder."
)


def _symptom_accuracy() -> str:
    """Accuracy of clear symptom results, as recorded by src/evaluation/evaluate_symptoms.py."""
    try:
        with open(config.CALIBRATION_PATH, encoding="utf-8") as f:
            acc = json.load(f)["symptom_screen_report"]["test_accuracy_when_clear"]
        return f"{acc:.0%}"
    except (OSError, ValueError, KeyError, TypeError):
        return "an unmeasured share"


SYMPTOM_ACCURACY = _symptom_accuracy()

GENERAL_INSPECTION = [
    "Photograph one affected leaf close-up, in daylight, against a plain background.",
    "Check both leaf surfaces, stems, new growth and fruit, and note whether damage is spreading.",
    "Check soil moisture 3-5 cm deep and look at roots if the plant is potted.",
    "Note when symptoms started and any recent changes (weather, location, repotting, fertiliser, sprays).",
]


@dataclass
class Section:
    key: str
    title: str
    items: list[str]


@dataclass
class Recommendation:
    summary: str
    sections: list[Section] = field(default_factory=list)
    watering: Assessment | None = None
    nutrients: Assessment | None = None
    pests: Assessment | None = None

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)

    def as_dict(self) -> dict:
        return {"summary": self.summary, "sections": [{"title": s.title, "items": s.items} for s in self.sections]}


def _assessment_items(a: Assessment) -> list[str]:
    return [a.headline + "."] + a.reasons + a.advice


def build_recommendations(result: AnalysisResult, ctx: CareContext | None = None) -> Recommendation:
    ctx = ctx or CareContext()
    if result.status == Status.INVALID_INPUT:
        return Recommendation(summary=result.message)

    profile = plant_profile(result)
    pests_in_photo = "image screen" in result.pest_flag_sources
    rec = Recommendation(summary=_summary(result, profile, pests_in_photo))
    actions: list[str] = []
    inspect: list[str] = []
    disease: list[str] = []
    prevention: list[str] = []
    expert = list(EXPERT_TRIGGERS)

    cond = result.condition if result.shows_candidate else None
    if cond is not None:
        inspect.extend(f"Look for: {s}" for s in cond.symptoms)
        inspect.extend(cond.inspect)
        prevention.extend(cond.prevention)
        if cond.escalate:
            expert.insert(0, cond.escalate)
        if cond.category == HEALTHY:
            actions.extend(CATEGORY_SANITATION[HEALTHY])
        elif result.status == Status.LIKELY:
            actions.append(f"Isolate or mark the plant and compare it with the typical signs of {cond.name} below.")
            disease.append(f"Category: {cond.category.lower()}" + (f" ({cond.cause})." if cond.cause else "."))
            disease.extend(CATEGORY_SANITATION[cond.category])
            if cond.category == OOMYCETE:
                actions.insert(0, "Check the plant and its neighbours again within 1-2 days - this disease can move fast.")
        else:
            actions.append(f"Do not treat for {cond.name} yet - the prediction is uncertain. Confirm with the "
                           "inspection steps first.")
            disease.append("Low-risk hygiene that is sensible whatever the cause: remove badly damaged leaves, "
                           "avoid wetting foliage, and disinfect tools between plants.")
    else:
        if result.status == Status.SPECIES_MISMATCH:
            actions.append("Double-check the plant name you selected, then re-upload a clear photo of a single leaf.")
        elif result.status in (Status.UNFAMILIAR, Status.POSSIBLE_NON_PLANT):
            actions.append("Retake the photo: one leaf, filling most of the frame, in daylight, on a plain background.")
        elif result.symptom_is_primary:
            for key in result.symptom.keys:
                advice = SYMPTOM_ADVICE[key]
                actions.extend(advice["actions"])
                inspect.extend(advice["inspect"])
                prevention.extend(advice["prevention"])
            if not result.symptom.keys:
                actions.append("The symptom check found no clear pattern; use the inspection checklist below.")
            actions.append("A local nursery or extension service can identify the exact cause.")
        elif result.status == Status.UNSUPPORTED_PLANT:
            actions.append("Use the inspection checklist below; a local nursery or extension service can identify "
                           "problems on plants this model does not cover.")
        actions.append("Avoid treating with any product until the cause is identified.")
        inspect.extend(GENERAL_INSPECTION)
        prevention.extend(["Remove fallen, diseased leaves from around the plant.",
                           "Water at the base of the plant and avoid wetting foliage late in the day.",
                           "Monitor weekly and keep a dated photo record to see if symptoms spread."])

    rec.watering = assess_watering(ctx)
    rec.nutrients = assess_nutrients(ctx)
    pest_image = "image screen" in result.pest_flag_sources
    pest_user = "user report" in result.pest_flag_sources
    hits = result.model_output.pest_top if (pest_image and result.model_output) else None
    rec.pests = assess_pests(ctx, pest_image, pest_user, hits)

    if result.quality and result.quality.issues:
        actions.extend(result.quality.issues)
    if result.photo_type == WHOLE_PLANT and result.species is not None:
        actions.insert(0, WHOLE_PLANT_ACTION)
    if pests_in_photo:
        actions.insert(0, "Possible insects or eggs were detected in the photo. Check both leaf surfaces with a "
                          "magnifying glass and isolate the plant from others until you know what they are.")

    rec.sections.append(Section("actions", "Recommended immediate actions", actions))
    if pests_in_photo and rec.pests:
        rec.sections.append(Section("pests", "Pests (detected in photo)", _assessment_items(rec.pests)))
    if result.symptom is not None and not result.symptom_is_primary:
        rec.sections.append(Section("symptoms", "Visible symptoms (photo check)",
                                    _symptom_items(result.symptom, result)))
    rec.sections.append(Section("inspect", "What to inspect next", inspect))
    if profile:
        rec.sections.append(Section("care", f"Care guide: {profile.name} ({profile.care.label.lower()})",
                                    care_items(profile)))
    if disease:
        rec.sections.append(Section("disease", "Disease management (low-risk steps)", disease))
    treatment = treatment_advice(result)
    if treatment:
        rec.sections.append(Section("treatment", treatment.title, treatment.items))

    watering_items = [PHOTO_CAVEAT_WATER] + _assessment_items(rec.watering)
    rec.sections.append(Section("watering", "Watering", watering_items))
    if rec.nutrients:
        rec.sections.append(Section("nutrients", "Nutrients", _assessment_items(rec.nutrients)))
    if rec.pests and not pests_in_photo:
        rec.sections.append(Section("pests", "Pests", _assessment_items(rec.pests)))
    elif not rec.pests:
        rec.sections.append(Section("pests", "Pests", [
            "No pests were flagged. Note: this app has no dedicated insect detector; check leaf undersides "
            "and new shoots yourself."]))

    rec.sections.append(Section("prevention", "Prevention and monitoring", prevention))
    rec.sections.append(Section("expert", "When to consult a local expert", expert))

    questions = [q for a in (rec.watering, rec.nutrients) if a for q in a.questions]
    if result.status in (Status.UNCERTAIN, Status.UNFAMILIAR, Status.LOW_QUALITY, Status.UNSUPPORTED_PLANT) \
            and not ctx.symptoms:
        questions.append("Which symptoms do you see (spots, yellowing, wilting, holes, powdery coating, webbing)?")
    if questions:
        rec.sections.append(Section("questions", "Information that would help", questions))

    limitations = list(result.warnings) + GENERAL_LIMITATIONS
    if result.symptom is not None:
        limitations.insert(len(result.warnings), SYMPTOM_LIMITATION.format(acc=SYMPTOM_ACCURACY))
    rec.sections.append(Section("limitations", "Limitations and uncertainty", limitations))
    return rec


def symptom_phrase(s: SymptomResult) -> str:
    """One-line description of a symptom-check result (bold markers for the UI)."""
    if s.clear:
        return f"**{s.name}** ({s.description[0].lower()}{s.description[1:].rstrip('.')})."
    if s.mixed:
        return (f"signs of both **{s.name}** and **{SYMPTOMS[s.second][0]}**. The check could not separate them; "
                "the leaf may show both.")
    return f"**no clear pattern** (closest: {s.name.lower()})."


def _symptom_items(s: SymptomResult, result: AnalysisResult) -> list[str]:
    first = (f"Visible pattern: {symptom_phrase(s)}" if s.keys
             else f"No clear visible pattern (closest: {s.name.lower()}).")
    items = [first, SYMPTOM_SEPARATE_NOTE]
    disease_says_healthy = (result.status == Status.LIKELY and result.condition is not None
                            and result.condition.is_healthy)
    if disease_says_healthy and any(k != "healthy" for k in s.keys):
        items.append("The two checks disagree. Inspect the leaf closely and compare it with the signs listed "
                     "below; if in doubt, ask a local expert.")
    items.extend(i for k in s.keys if k != "healthy" for i in SYMPTOM_ADVICE[k]["inspect"])
    return items


def plant_profile(result: AnalysisResult) -> PlantProfile | None:
    profile = find_profile(result.plant_name)
    if profile is None and result.species is not None:
        profile = find_profile(result.species.display_name)
    return profile


def care_items(profile: PlantProfile) -> list[str]:
    care = profile.care
    items = [profile.note] if profile.note else []
    items += [f"Watering: {care.watering}", f"Light: {care.light}",
              f"Common pests to check for: {care.pests}", f"Common problems: {care.problems}",
              "This is general guidance for this type of plant, not a diagnosis from your photo."]
    return items


def _summary(result: AnalysisResult, profile: PlantProfile | None = None, pests_in_photo: bool = False) -> str:
    plant = "plant" if result.plant_name == OTHER_OPTION else result.plant_name
    pest_note = (" **Possible insects or eggs were detected on the leaf** - see the Pests section."
                 if pests_in_photo else "")
    if pests_in_photo and result.status != Status.LIKELY:
        hits = result.model_output.pest_top if result.model_output else []
        looks_like = f" It looks most like {hits[0][0]}." if hits else ""
        disease = (f" The disease model's suggestion ({result.condition.name}) is unreliable for "
                   "pest-infested leaves." if result.shows_candidate and result.condition else "")
        if result.symptom_is_primary:
            disease = " The general symptom check is unreliable for pest-infested leaves."
        return f"**Possible insects or eggs were detected on your {plant} leaf.**{looks_like}{disease}"
    if result.shows_candidate and result.condition:
        name = result.condition.name
        if result.status == Status.LIKELY:
            return (f"The model's most likely match for your {plant} is **{name}**. This is a prediction, not a "
                    f"confirmed diagnosis.{pest_note}")
        return (f"Possible condition for your {plant}: **{name}** (uncertain). Please confirm with the checks "
                f"below.{pest_note}")
    text = result.message
    if result.symptom_is_primary:
        s = result.symptom
        text = (f"Visible pattern on your {plant} leaf: {symptom_phrase(s)} " if s.keys else
                f"The symptom check found **no clear pattern** on your {plant} leaf (closest: {s.name.lower()}). ")
        text +=("This is a general symptom check: there is no plant-specific disease model for this plant, "
                 "so the exact disease cannot be named.")
    if result.status == Status.UNSUPPORTED_PLANT and profile is not None:
        text += f" A care guide for {profile.care.label.lower()}s is included below."
    return text + pest_note
