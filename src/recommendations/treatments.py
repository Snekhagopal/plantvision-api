"""Evidence-gated treatment categories.

This is the only module that mentions pesticide, fungicide or similar product
categories. Rules (approved project policy):

- A treatment category is shown as an option only for a *likely* disease
  match (which already passed every reliability check) or for pests that the
  user is told to confirm first.
- Uncertain disease results, and the symptom check for mildew or rust, get
  categories only phrased as "if a local expert confirms ...".
- Nothing is suggested for whole-plant photos' disease results, unfamiliar or
  mismatched photos, leaf spots or yellowing seen by the symptom check, or
  yellowing / dryness reported in the questionnaire.
- Never product brands, never doses or rates; always the safety block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.inference.analysis import WHOLE_PLANT, AnalysisResult, Status
from src.plants.conditions import BACTERIAL, FUNGAL, OOMYCETE, PEST, VIRAL, ConditionInfo

SAFETY = [
    "Safety: read and follow the product label and local regulations - the label is the legal guide to use.",
    "Check that the product is registered for this plant (and for edible crops, observe the pre-harvest "
    "interval on the label).",
    "Wear the protective equipment the label lists, and keep children and pets away until it has dried.",
    "Do not spray open flowers, in midday heat or before rain: this protects pollinators and avoids leaf burn.",
    "Try the non-chemical steps in this report first where practical; a local extension service or nursery "
    "can advise on registered products in your area.",
]

_FUNGAL_DEFAULT = (
    "A fungicide labelled for {name} on {crop}. Fungicides protect healthy leaves; they do not cure spots "
    "that are already there, so combine them with the hygiene steps above."
)

# Per-label overrides of the category defaults.
_BY_LABEL = {
    "Cherry_(including_sour)___Powdery_mildew": (
        "Products labelled for powdery mildew on cherry, e.g. potassium bicarbonate-, sulfur- or "
        "horticultural-oil-based. Check the label for crop safety and spacing between different products."
    ),
    "Squash___Powdery_mildew": (
        "Products labelled for powdery mildew on squash, e.g. potassium bicarbonate-, sulfur- or "
        "horticultural-oil-based. Check the label: some squash and other cucurbit varieties are sensitive to "
        "sulfur."
    ),
    "Grape___Esca_(Black_Measles)": (
        "There is no effective curative spray for Esca. Management relies on pruning in dry weather, removing "
        "dead wood and protecting large pruning wounds; ask a local viticulture adviser."
    ),
    "Orange___Haunglongbing_(Citrus_greening)": (
        "There is no cure or chemical treatment for citrus greening. It is regulated in many regions: report "
        "suspected cases to your local agricultural authority, which also advises on psyllid control."
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "There is no chemical cure for plant viruses. This virus is spread by whiteflies, so controlling "
        "whiteflies (see Pests) and removing infected plants limits spread."
    ),
    "Tomato___Tomato_mosaic_virus": (
        "There is no chemical cure for plant viruses. This virus spreads on hands and tools: remove infected "
        "plants and disinfect tools; sprays do not help."
    ),
}

_BY_CATEGORY = {
    FUNGAL: _FUNGAL_DEFAULT,
    BACTERIAL: ("Copper-based products labelled for {name} on {crop} can slow the spread; they do not cure "
                "infected leaves, and bacteria can become copper-tolerant, so rely on the hygiene steps too."),
    OOMYCETE: ("Protectant fungicides labelled for late blight on {crop} can protect healthy plants; heavily "
               "infected plants usually need to be removed and bagged. Contact your local extension service "
               "promptly - late blight spreads fast and is notifiable in some areas."),
    VIRAL: "There is no chemical cure for plant viruses; remove infected plants and control insects that spread them.",
    PEST: ("Rinse leaf undersides with water first. If mites persist, insecticidal soap or horticultural oil "
           "labelled for mites can help. Avoid broad-spectrum insecticides, which kill the mites' natural "
           "predators and can make outbreaks worse."),
}

# Pest-screen appearance labels (src/inference/pest_screen.py CLIP_PROMPTS) -> category advice.
_PEST_BY_APPEARANCE = {
    "small sap-sucking insects (e.g. whiteflies, aphids, mealybugs, scale)": (
        "For soft-bodied insects such as aphids, whiteflies or mealybugs: insecticidal soap or horticultural oil "
        "labelled for them. Both work only on direct contact, so cover leaf undersides."
    ),
    "insect eggs": (
        "For eggs: wipe or prune them off. No spray is needed until you know which insect laid them."
    ),
    "spider mites / webbing": _BY_CATEGORY[PEST],
    "larger insects (e.g. caterpillars, beetles)": (
        "For larger insects: hand-pick them. If they are confirmed caterpillars, products based on Bacillus "
        "thuringiensis (Bt) labelled for caterpillars target them specifically."
    ),
}

_SYMPTOM_CONDITIONAL = {
    "mildew": ("If a nursery or extension service confirms powdery mildew: products labelled for powdery mildew "
               "on this plant (e.g. potassium bicarbonate-, sulfur- or horticultural-oil-based) are an option."),
    "rust": ("If a nursery or extension service confirms a rust disease: a fungicide labelled for rust on this "
             "plant is an option, together with removing affected leaves."),
}


@dataclass
class TreatmentAdvice:
    confirmed_basis: bool                 # True when backed by a likely disease match
    options: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return "Treatment options" if self.confirmed_basis else "Treatment options - only after confirming the cause"

    @property
    def items(self) -> list[str]:
        return self.options + SAFETY


def disease_treatment(label: str, condition: ConditionInfo, crop: str) -> str | None:
    template = _BY_LABEL.get(label) or _BY_CATEGORY.get(condition.category)
    return template.format(name=condition.name.lower(), crop=crop.lower()) if template else None


def treatment_advice(result: AnalysisResult) -> TreatmentAdvice | None:
    likely, conditional = [], []
    pests_in_photo = "image screen" in result.pest_flag_sources

    cond = result.condition if result.shows_candidate else None
    if cond is not None and not cond.is_healthy and result.species is not None:
        text = disease_treatment(result.predicted_label, cond, result.species.display_name)
        if text and result.status == Status.LIKELY:
            likely.append(text)
        elif (text and result.status in (Status.UNCERTAIN, Status.LOW_QUALITY)
              and result.photo_type != WHOLE_PLANT and not pests_in_photo):
            conditional.append(f"If a local expert confirms {cond.name.lower()}: {text}")

    if result.symptom_is_primary:
        conditional.extend(_SYMPTOM_CONDITIONAL[k] for k in result.symptom.keys if k in _SYMPTOM_CONDITIONAL)

    if result.pest_flag:
        hits = result.model_output.pest_top if (pests_in_photo and result.model_output) else []
        appearance = hits[0][0] if hits else None
        pest_options = ([_PEST_BY_APPEARANCE[appearance]] if appearance in _PEST_BY_APPEARANCE
                        else [_PEST_BY_APPEARANCE[a] for a in _PEST_BY_APPEARANCE if a != "insect eggs"])
        conditional.append("Identify the pest first (close-up photo or hand lens); many insects are harmless or "
                           "beneficial. Once identified:")
        conditional.extend(pest_options)

    if not (likely or conditional):
        return None
    # A likely spider-mite match and a mite-like pest hit give the same advice once.
    return TreatmentAdvice(bool(likely), list(dict.fromkeys(likely + conditional)))
