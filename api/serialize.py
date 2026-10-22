"""Convert a service AnalysisBundle (PIL images, numpy values, dataclasses) into API schemas."""

from __future__ import annotations

import base64
import io

from PIL import Image

from api.schemas import (
    AnalyzeResponse, AssessmentOut, ImagesOut, ModelDetailsOut, PestOut, PlantOut, PredictionOut, QualityOut,
    ReportOut, ScoredLabel, SectionOut, SymptomOut,
)
from src.plants.catalog import PlantCatalog
from src.plants.conditions import get_condition
from src.report import build_report
from src.service import AnalysisBundle

PHOTO_MAX_SIDE = 1024


def data_url(image: Image.Image | None, max_side: int = PHOTO_MAX_SIDE) -> str | None:
    if image is None:
        return None
    img = image.convert("RGB").copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _scored(label: str, score, catalog: PlantCatalog) -> ScoredLabel:
    cond = get_condition(label)
    return ScoredLabel(label=label, name=cond.name if cond else catalog.condition_display(label),
                       species=catalog.species_of(label).display_name, score=float(score))


def to_response(bundle: AnalysisBundle, catalog: PlantCatalog) -> AnalyzeResponse:
    r, rec = bundle.result, bundle.recommendation
    report = bundle.report or build_report(r)
    out = r.model_output

    prediction = None
    if r.shows_candidate:
        cond = r.condition
        prediction = PredictionOut(
            label=r.predicted_label, name=cond.name if cond else catalog.condition_display(r.predicted_label),
            category=cond.category if cond else None, cause=(cond.cause or None) if cond else None,
            score=float(r.confidence), alternatives=[_scored(l, p, catalog) for l, p in r.alternatives],
            symptoms=list(cond.symptoms) if cond else [], prevention=list(cond.prevention) if cond else [])

    symptom = None
    if r.symptom is not None:
        s = r.symptom
        symptom = SymptomOut(key=s.key, name=s.name, description=s.description, score=float(s.probability),
                             clear=s.clear, patterns=s.keys, is_primary=r.symptom_is_primary,
                             scores={k: float(v) for k, v in s.probabilities.items()})

    pests = PestOut(flagged=r.pest_flag, sources=list(r.pest_flag_sources),
                    score=float(out.pest_score) if out and out.pest_score is not None else None,
                    method=out.pest_method if out else None,
                    appearance=[(n, float(v)) for n, v in out.pest_top] if out and r.pest_flag else [])

    model_details = None
    if out is not None and r.species is not None:
        model_details = ModelDetailsOut(
            top_k=[_scored(l, p, catalog) for l, p in out.top_k],
            plant_mass=float(r.plant_mass) if r.plant_mass is not None else None,
            unfamiliarity_score=float(out.ood_score) if out.ood_score is not None else None,
            unfamiliar=r.ood_flag, model_top_label=r.model_top_label)

    gc = bundle.gradcam
    if gc is not None and gc.ok:
        gradcam_note = None
    elif gc is not None:
        gradcam_note = gc.error or "The heatmap could not be generated."
    elif r.symptom_is_primary:
        gradcam_note = "Grad-CAM is only available for the 14 crops with a plant-specific disease model."
    else:
        gradcam_note = ("Grad-CAM is only shown when the model produced a usable prediction for a supported plant, "
                        "so it is not presented as evidence for an unreliable result.")
    images = ImagesOut(photo=data_url(r.image), gradcam=data_url(gc.overlay) if gc is not None and gc.ok else None,
                       gradcam_note=gradcam_note,
                       leaf_attention=float(gc.leaf_attention) if gc is not None and gc.leaf_attention is not None
                       else None)

    q = r.quality
    return AnalyzeResponse(
        status=r.status.value, status_label=r.status_label, message=r.message, summary=rec.summary,
        photo_type=r.photo_type,
        plant=PlantOut(input=r.plant_name, matched_species=r.species.display_name if r.species else None,
                       disease_check=r.species is not None),
        report=ReportOut(**{a.key: AssessmentOut(**a.__dict__) for a in report.assessments}),
        prediction=prediction, symptom=symptom, pests=pests,
        recommendations=[SectionOut(key=s.key, title=s.title, items=list(s.items)) for s in rec.sections],
        quality=QualityOut(brightness=float(q.brightness), blur_variance=float(q.blur_variance),
                           vegetation_fraction=float(q.vegetation_fraction), issues=list(q.issues)) if q else None,
        images=images, model_details=model_details, warnings=list(r.warnings),
        history_id=bundle.history_id, history_error=bundle.history_error,
    )
