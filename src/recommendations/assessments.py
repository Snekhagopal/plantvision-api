"""Questionnaire-based assessments for watering, nutrients and pests.

These never rely on the photo alone. A leaf photo cannot show soil moisture,
root health or soil chemistry, so every output is phrased as a possibility to
check, and the absence of answers yields "not enough information".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.recommendations.context import CareContext

PHOTO_CAVEAT_WATER = (
    "A photo cannot show whether a plant needs water - wilting and yellowing can come from both too little "
    "and too much water, from root damage, heat or disease."
)


@dataclass
class Assessment:
    level: str                       # short machine-readable outcome
    headline: str
    reasons: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)


def assess_watering(ctx: CareContext) -> Assessment:
    under, over = [], []
    soil = ctx.soil_moisture

    if soil == "Bone dry":
        under.append("You reported the soil is bone dry.")
    if soil == "Wet / soggy":
        over.append("You reported the soil is wet or soggy.")
    if ctx.drainage in ("Drains slowly / water pools", "Container has no drainage hole"):
        over.append(f"Drainage: {ctx.drainage.lower()}.")
    if ctx.watering_frequency == "Daily or more" and soil in ("Evenly moist", "Wet / soggy"):
        over.append("Watering daily while the soil is still moist can keep roots waterlogged.")
    if ctx.watering_frequency in ("Less than weekly", "Irregular") and soil in ("Bone dry", "Slightly dry"):
        under.append(f"Watering is {ctx.watering_frequency.lower()} and the soil is dry.")
    if ctx.has("Wilting or drooping"):
        if soil in ("Bone dry", "Slightly dry"):
            under.append("Wilting together with dry soil.")
        elif soil == "Wet / soggy":
            over.append("Wilting despite wet soil can indicate root problems (e.g. root rot).")
    if ctx.has("Soft or mushy stems / roots"):
        over.append("Soft or mushy stems/roots are often associated with excess moisture or rot.")
    if ctx.has("Brown or crispy leaf edges") and soil in ("Bone dry", "Slightly dry"):
        under.append("Crispy leaf edges with dry soil.")
    if ctx.weather == "Hot and dry" and soil in ("Bone dry", "Slightly dry"):
        under.append("Recent hot, dry weather increases water demand.")

    questions = []
    if soil == "Not sure / skip":
        questions.append("Push a finger 3-5 cm into the soil: is it dry, moist or soggy?")
    if ctx.drainage == "Not sure / skip":
        questions.append("Does water drain freely, or does it pool on the surface / in the saucer?")
    if ctx.watering_frequency == "Not sure / skip":
        questions.append("How often is the plant watered, and how much?")

    check = "Before watering, check the soil 3-5 cm deep; water only if it feels dry at that depth."
    if under and over:
        return Assessment("mixed", "Your answers point both ways - check the roots and soil directly",
                          under + over,
                          [check, "Gently inspect roots if possible: healthy roots are firm and pale; "
                                  "dark, mushy, foul-smelling roots suggest overwatering/root rot."], questions)
    if over:
        return Assessment("possible_overwatering", "Your answers suggest possible overwatering or poor drainage",
                          over,
                          ["Let the top few centimetres of soil dry before watering again.",
                           "Make sure containers have drainage holes and empty standing water from saucers.",
                           "Check roots for dark, mushy tissue if the plant keeps declining.", check], questions)
    if under:
        return Assessment("possible_underwatering", "Your answers suggest the plant may be short of water",
                          under,
                          [check, "If dry, water slowly and deeply until water drains, rather than frequent sips.",
                           "Mulch outdoor beds to reduce evaporation."], questions)
    if not ctx.answered():
        return Assessment("insufficient_information", "Not enough information to assess watering",
                          [], [check], questions)
    return Assessment("no_clear_signal", "No clear watering problem indicated by your answers",
                      [], [check], questions)


def assess_nutrients(ctx: CareContext) -> Assessment | None:
    """Only produced when the user reports colour symptoms."""
    colour = [s for s in ("Yellowing leaves", "Yellowing between veins", "Pale new growth") if ctx.has(s)]
    if not colour:
        return None
    reasons = [f"You reported: {', '.join(s.lower() for s in colour)}."]
    advice = [
        "Yellowing has many possible causes - watering problems, root damage, low light, disease, pests, "
        "natural ageing of lower leaves and nutrient shortages can all look similar.",
        "Do not add fertiliser based on leaf colour alone. A soil test (or tissue test) is the reliable way to "
        "confirm a deficiency; local extension services often offer these.",
        "If a soil test confirms a deficiency, choose a fertiliser that supplies that nutrient and follow the "
        "label rate - extra fertiliser does not help and can burn roots.",
    ]
    if ctx.affected_leaves == "Older / lower leaves first":
        reasons.append("Symptoms start on older leaves - consistent with mobile nutrients (e.g. nitrogen, "
                       "magnesium, potassium) but also with normal ageing or overwatering.")
    elif ctx.affected_leaves == "New / upper leaves first":
        reasons.append("Symptoms start on new growth - consistent with less mobile nutrients (e.g. iron), "
                       "often linked to high soil pH, but also with some viruses and pests.")
    if ctx.fertilizer == "Within the last 2 weeks" and ctx.has("Brown or crispy leaf edges"):
        reasons.append("Recent fertiliser plus browning edges could indicate fertiliser (salt) burn.")
        advice.append("Avoid further fertilising until the cause is clear; for containers, watering thoroughly "
                      "so excess drains out can help flush salts.")
    questions = []
    if ctx.affected_leaves == "Not sure / skip":
        questions.append("Are the oldest (lower) leaves or the newest (upper) leaves affected first?")
    if ctx.fertilizer == "Not sure / skip":
        questions.append("When was the plant last fertilised, and with what?")
    return Assessment("possible_nutrient_issue", "Colour changes reported - nutrient status unconfirmed",
                      reasons, advice, questions)


def assess_pests(ctx: CareContext, image_flag: bool, user_flag: bool,
                 image_hits: list[tuple[str, float]] | None = None) -> Assessment | None:
    signs = [s for s in ("Holes or chewed leaves", "Fine webbing", "Sticky residue on leaves") if ctx.has(s)]
    if not (image_flag or user_flag or signs):
        return None
    reasons = []
    if image_flag:
        hint = f" It most resembles: {', '.join(n for n, _ in image_hits or [])}." if image_hits else ""
        reasons.append("An automatic image screen found what looks like insects or eggs on the leaf." + hint +
                       " This is not a pest identification - confirm by looking closely.")
    if user_flag:
        reasons.append("You reported visible insects.")
    if signs:
        reasons.append(f"Reported signs: {', '.join(s.lower() for s in signs)}.")
    advice = [
        "Take a sharp close-up of the insect and of both leaf surfaces; a hand lens helps for tiny pests.",
        "Check leaf undersides, new shoots and stems - most sap-feeding pests hide there.",
        "Remove visible pests by hand or rinse them off with a strong spray of plain water.",
        "Isolate affected potted plants from others.",
        "Yellow sticky cards can help monitor flying pests such as whiteflies and fungus gnats.",
        "Do not apply insecticides until the pest is identified: many insects are harmless or beneficial "
        "(e.g. ladybirds, lacewings). A local extension service or nursery can identify samples.",
    ]
    return Assessment("possible_pest", "Possible pest activity - identification needed", reasons, advice)
