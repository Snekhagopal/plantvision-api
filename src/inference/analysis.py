"""End-to-end analysis: validation -> quality -> plant compatibility -> model -> status.

The pipeline never turns a model output directly into a diagnosis. Each
result carries a ``Status`` that says how much (if anything) the model output
can be trusted, and the UI/recommendation engine phrase advice accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np
from PIL import Image

from src.inference.predictor import ModelOutput
from src.inference.symptom_screen import SymptomResult
from src.plants.catalog import PlantCatalog, PlantMatch, SpeciesInfo
from src.plants.conditions import ConditionInfo, get_condition
from src.recommendations.context import CareContext
from src.utils.preprocessing import ImageQuality, ImageValidationError, assess_quality, load_image

logger = logging.getLogger(__name__)


class Status(str, Enum):
    LIKELY = "likely_match"
    UNCERTAIN = "uncertain"
    LOW_QUALITY = "low_quality"
    SPECIES_MISMATCH = "species_mismatch"
    UNFAMILIAR = "unfamiliar_image"
    POSSIBLE_NON_PLANT = "possible_non_plant"
    UNSUPPORTED_PLANT = "unsupported_plant"
    INVALID_INPUT = "invalid_input"
    MODEL_UNAVAILABLE = "model_unavailable"
    ANALYSIS_ERROR = "analysis_error"


STATUS_LABELS = {
    Status.LIKELY: "Likely match (model prediction, not a confirmed diagnosis)",
    Status.UNCERTAIN: "Uncertain prediction",
    Status.LOW_QUALITY: "Low image quality - tentative only",
    Status.SPECIES_MISMATCH: "Image may not match the selected plant",
    Status.UNFAMILIAR: "Unfamiliar image - outside what the model knows",
    Status.POSSIBLE_NON_PLANT: "This may not be a plant photo",
    Status.UNSUPPORTED_PLANT: "Plant not supported by the disease model",
    Status.INVALID_INPUT: "Invalid input",
    Status.MODEL_UNAVAILABLE: "Disease model unavailable",
    Status.ANALYSIS_ERROR: "Analysis failed",
}

# Statuses for which a disease candidate may be shown (with caveats) and
# Grad-CAM may be computed.
CANDIDATE_STATUSES = {Status.LIKELY, Status.UNCERTAIN, Status.LOW_QUALITY}

# What the photo shows, as stated by the user. The disease model was trained and
# calibrated on close-ups of single leaves only.
LEAF_CLOSE_UP = "leaf"
WHOLE_PLANT = "whole_plant"
PHOTO_TYPES = {LEAF_CLOSE_UP: "Leaf close-up", WHOLE_PLANT: "Whole plant"}

WHOLE_PLANT_NOTE = (
    "This is a whole-plant photo, but the disease model was trained only on close-ups of single leaves, so its "
    "result cannot be treated as reliable here. For a disease check, upload a close-up of one affected leaf."
)


class Predictor(Protocol):
    catalog: PlantCatalog
    thresholds: dict

    def predict(self, image: Image.Image) -> ModelOutput: ...


@dataclass
class AnalysisResult:
    plant_name: str
    status: Status
    message: str
    species: SpeciesInfo | None = None
    image: Image.Image | None = None
    quality: ImageQuality | None = None
    model_output: ModelOutput | None = None
    predicted_label: str | None = None
    condition: ConditionInfo | None = None
    confidence: float | None = None         # calibrated softmax score of predicted_label
    plant_mass: float | None = None         # total score on the selected plant's classes
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    model_top_label: str | None = None      # unrestricted top-1 label
    ood_flag: bool = False
    pest_flag: bool = False
    pest_flag_sources: list[str] = field(default_factory=list)
    symptom: SymptomResult | None = None    # visible symptom pattern (photo check), kept separate from disease
    photo_type: str = LEAF_CLOSE_UP
    warnings: list[str] = field(default_factory=list)

    @property
    def symptom_is_primary(self) -> bool:
        """The symptom check is the main photo result (plants without a disease model)."""
        return self.status == Status.UNSUPPORTED_PLANT and self.symptom is not None

    @property
    def status_label(self) -> str:
        if self.symptom_is_primary:
            return "General symptom check (no plant-specific disease model)"
        return STATUS_LABELS[self.status]

    @property
    def shows_candidate(self) -> bool:
        return self.status in CANDIDATE_STATUSES and self.predicted_label is not None

    @property
    def gradcam_allowed(self) -> bool:
        return self.shows_candidate and self.image is not None


def validate_plant_name(plant_name: str | None) -> str:
    name = (plant_name or "").strip()
    if not name:
        raise ValueError("Please enter or select the plant name - it is required.")
    if len(name) > 80:
        raise ValueError("The plant name is too long (max 80 characters).")
    return name


def _apply_pest_flags(result: AnalysisResult, output: ModelOutput | None, context: CareContext, thresholds: dict):
    if output is not None and output.pest_score is not None and output.pest_score >= thresholds["pest_screen_threshold"]:
        result.pest_flag = True
        result.pest_flag_sources.append("image screen")
    if context.insects_visible == "Yes":
        result.pest_flag = True
        result.pest_flag_sources.append("user report")


def analyze(image_bytes: bytes | None, filename: str | None, plant_name: str | None,
            predictor: Predictor | None, catalog: PlantCatalog, thresholds: dict,
            context: CareContext | None = None, photo_type: str = LEAF_CLOSE_UP) -> AnalysisResult:
    context = context or CareContext()
    if photo_type not in PHOTO_TYPES:
        photo_type = LEAF_CLOSE_UP

    try:
        plant_name = validate_plant_name(plant_name)
    except ValueError as exc:
        return AnalysisResult(plant_name or "", Status.INVALID_INPUT, str(exc))
    if not image_bytes:
        return AnalysisResult(plant_name, Status.INVALID_INPUT, "Please upload a photo of the plant or leaf.")

    try:
        image = load_image(image_bytes, filename)
    except ImageValidationError as exc:
        return AnalysisResult(plant_name, Status.INVALID_INPUT, str(exc))

    match: PlantMatch = catalog.match(plant_name)
    quality = assess_quality(image, thresholds)
    result = AnalysisResult(plant_name, Status.UNCERTAIN, "", species=match.species, image=image, quality=quality,
                            photo_type=photo_type)
    result.warnings.extend(quality.issues)

    output = None
    if predictor is not None:
        try:
            output = predictor.predict(image)
        except Exception:
            logger.exception("Model inference failed")
            if match.supported:
                result.status = Status.ANALYSIS_ERROR
                result.message = "The image could not be analysed due to an internal error. Please try another photo."
                return result
    result.model_output = output
    result.symptom = output.symptom if output is not None else None
    _apply_pest_flags(result, output, context, thresholds)

    if quality.vegetation_fraction < thresholds["vegetation_min_fraction"]:
        result.status = Status.POSSIBLE_NON_PLANT
        result.message = (
            "Very little leaf-coloured area was found in this photo. It may not show a plant, or the leaf "
            "may be too small in the frame. Please upload a close-up of the affected leaf."
        )
        result.symptom = None   # no plant to describe
        return result

    if not match.supported:
        result.status = Status.UNSUPPORTED_PLANT
        if result.symptom is not None:
            result.message = (
                f"There is no plant-specific disease model for '{plant_name}', so a general symptom check was "
                "used. It describes the visible pattern on the leaf but cannot name the exact disease."
            )
        else:
            result.message = (
                f"The disease classifier was not trained on '{plant_name}', so it cannot reliably diagnose this "
                "plant. The guidance below is based only on general plant-care principles and your answers."
            )
        return result

    species = match.species
    if species.coverage_warning:
        result.warnings.append(species.coverage_warning)

    if output is None:
        result.status = Status.MODEL_UNAVAILABLE
        result.message = "The disease model could not be loaded, so only general guidance is available."
        return result

    _classify(result, output, catalog, species, thresholds)
    if photo_type == WHOLE_PLANT:
        result.warnings.insert(0, WHOLE_PLANT_NOTE)
        if result.status == Status.LIKELY:   # not validated on whole-plant photos: never "likely"
            result.status = Status.UNCERTAIN
            result.message = WHOLE_PLANT_NOTE
    return result


def _classify(result: AnalysisResult, output: ModelOutput, catalog: PlantCatalog,
              species: SpeciesInfo, thresholds: dict) -> None:
    probs = output.probabilities
    names = catalog.class_names
    idx = catalog.label_indices(species)
    plant_mass = float(probs[idx].sum())
    best = int(idx[int(np.argmax(probs[idx]))])
    top = int(np.argmax(probs))

    result.plant_mass = plant_mass
    result.model_top_label = names[top]
    result.ood_flag = (output.ood_score is not None and thresholds.get("ood_threshold") is not None
                       and output.ood_score > thresholds["ood_threshold"])

    top_species = catalog.species_of(names[top])
    if top_species.key != species.key and plant_mass < thresholds["plant_mass_threshold"]:
        result.status = Status.SPECIES_MISMATCH
        result.message = (
            f"You selected {species.display_name}, but the image looks more like {top_species.display_name} "
            f"to the model ({plant_mass:.0%} of its score falls on {species.display_name} classes). Please check "
            "the plant name, or upload a clearer close-up of a single leaf."
        )
        return

    if result.ood_flag:
        result.status = Status.UNFAMILIAR
        result.message = (
            "This photo looks unlike the images the model was trained on (single leaves on plain backgrounds). "
            "Its prediction is not reliable here. Try a close-up of one leaf against a plain background, "
            "or use the inspection guidance below."
        )
        result.alternatives = [(l, p) for l, p in output.top_k]
        return

    result.predicted_label = names[best]
    result.condition = get_condition(names[best])
    result.confidence = float(probs[best])
    result.alternatives = [(names[i], float(probs[i])) for i in sorted(idx, key=lambda i: -probs[i]) if i != best][:2]

    confident = result.confidence >= thresholds["confidence_threshold"] and best == top
    single_class_species = len(species.labels) == 1
    # The disease model never saw pest-infested leaves (except spider-mite damage),
    # so a visible-pest flag makes any other disease label unreliable.
    pests_conflict = "image screen" in result.pest_flag_sources and "Spider_mites" not in names[best]

    if result.quality and result.quality.is_low_quality:
        result.status = Status.LOW_QUALITY
        result.message = "The photo quality is limited, so this result is tentative. A clearer photo is recommended."
    elif pests_conflict:
        result.status = Status.UNCERTAIN
        result.message = (
            "The photo appears to show insects or eggs on the leaf. The disease model was not trained on "
            "pest-infested leaves, so its disease suggestion is unreliable here - check for pests first."
        )
    elif confident and not single_class_species:
        result.status = Status.LIKELY
        result.message = (
            "The model's prediction is consistent with your selected plant and above the confidence threshold. "
            "This is a likely match, not a confirmed diagnosis."
        )
    else:
        result.status = Status.UNCERTAIN
        if single_class_species:
            result.message = (
                f"The model only has one class for {species.display_name}, so it cannot compare alternatives. "
                "Treat this as a hint to investigate, not a diagnosis."
            )
        else:
            result.message = (
                "The model is not confident enough to single out one condition. Use the inspection steps below "
                "to narrow it down."
            )
