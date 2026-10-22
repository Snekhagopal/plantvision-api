"""Application service layer: runs one analysis end to end, without Streamlit.

Keeping this outside app.py lets tests exercise the full workflow (including
error paths) with mocked predictors and temporary databases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.database.history import HistoryError, HistoryStore
from src.explainability.gradcam import GradCamResult, compute_gradcam
from src.inference.analysis import LEAF_CLOSE_UP, AnalysisResult, Status, analyze
from src.inference.symptom_screen import SYMPTOMS
from src.plants.catalog import PlantCatalog
from src.recommendations.context import CareContext
from src.recommendations.engine import Recommendation, build_recommendations
from src.report import PlantHealthReport, build_report

logger = logging.getLogger(__name__)


@dataclass
class AnalysisBundle:
    result: AnalysisResult
    recommendation: Recommendation
    gradcam: GradCamResult | None = None
    report: PlantHealthReport | None = None
    history_id: int | None = None
    history_error: str | None = None


def gradcam_for(result: AnalysisResult, predictor) -> GradCamResult | None:
    """Grad-CAM only for results that show a disease candidate."""
    if not result.gradcam_allowed or predictor is None:
        return None
    try:
        class_index = predictor.class_names.index(result.predicted_label)
        tensor = predictor.preprocess(result.image)
    except Exception as exc:
        logger.exception("Could not prepare Grad-CAM input")
        return GradCamResult(None, None, f"The heatmap could not be generated ({type(exc).__name__}).")
    return compute_gradcam(predictor.model, tensor, class_index, result.image)


def apply_attention_check(result: AnalysisResult, gradcam: GradCamResult, thresholds: dict) -> None:
    """Downgrade a 'likely' result when the model mostly looked at the background."""
    limit = thresholds.get("gradcam_min_leaf_attention")
    if (limit is None or not gradcam.ok or gradcam.leaf_attention is None
            or gradcam.leaf_attention >= limit):
        return
    note = (f"Only {gradcam.leaf_attention:.0%} of the heatmap falls on the leaf - the model relied mostly on "
            "the background, so this prediction is less trustworthy.")
    result.warnings.append(note)
    if result.status == Status.LIKELY:
        result.status = Status.UNCERTAIN
        result.message = note + " Treat it as a hint to investigate."


def run_full_analysis(image_bytes: bytes | None, filename: str | None, plant_name: str | None,
                      context: CareContext | None, predictor, catalog: PlantCatalog, thresholds: dict,
                      history: HistoryStore | None = None, with_gradcam: bool = True,
                      photo_type: str = LEAF_CLOSE_UP) -> AnalysisBundle:
    context = context or CareContext()
    try:
        result = analyze(image_bytes, filename, plant_name, predictor, catalog, thresholds, context, photo_type)
    except Exception:  # last-resort guard: never crash the UI
        logger.exception("Unexpected analysis failure")
        result = AnalysisResult((plant_name or "").strip(), Status.ANALYSIS_ERROR,
                                "Something went wrong while analysing this image. Please try another photo.")
    gradcam = gradcam_for(result, predictor) if with_gradcam else None
    if gradcam is not None:
        apply_attention_check(result, gradcam, thresholds)
    recommendation = build_recommendations(result, context)
    bundle = AnalysisBundle(result, recommendation, gradcam, build_report(result, context))

    if history is not None and result.status != Status.INVALID_INPUT:
        try:
            bundle.history_id = history.add(
                plant_name=result.plant_name,
                status=result.status.value,
                status_label=result.status_label,
                summary=recommendation.summary,
                recommendations=recommendation.as_dict(),
                predicted_label=result.predicted_label if result.shows_candidate else None,
                condition_name=(result.condition.name if (result.shows_candidate and result.condition)
                                else "Symptom pattern: " + " + ".join(SYMPTOMS[k][0] for k in result.symptom.keys)
                                if (result.symptom_is_primary and result.symptom.keys) else None),
                confidence=result.confidence if result.shows_candidate else None,
            )
        except HistoryError as exc:
            bundle.history_error = str(exc)
    return bundle
