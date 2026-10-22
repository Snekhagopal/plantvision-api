"""Optional questionnaire answers supplied by the user."""

from __future__ import annotations

from dataclasses import dataclass, field

UNKNOWN = "Not sure / skip"

SYMPTOM_OPTIONS = [
    "Yellowing leaves",
    "Yellowing between veins",
    "Pale new growth",
    "Brown or crispy leaf edges",
    "Wilting or drooping",
    "Spots or lesions",
    "Holes or chewed leaves",
    "Curling or distorted leaves",
    "White powdery coating",
    "Fine webbing",
    "Sticky residue on leaves",
    "Leaf drop",
    "Soft or mushy stems / roots",
]

SOIL_MOISTURE = [UNKNOWN, "Bone dry", "Slightly dry", "Evenly moist", "Wet / soggy"]
WATERING_FREQUENCY = [UNKNOWN, "Daily or more", "Every 2-3 days", "About weekly", "Less than weekly", "Irregular"]
DRAINAGE = [UNKNOWN, "Drains well", "Drains slowly / water pools", "Container has no drainage hole"]
LOCATION = [UNKNOWN, "Container / pot", "Garden bed or field", "Indoors"]
SUNLIGHT = [UNKNOWN, "Full sun (6+ hours)", "Partial sun", "Shade / low light"]
WEATHER = [UNKNOWN, "Hot and dry", "Cool, wet or humid", "Mild / normal"]
FERTILIZER = [UNKNOWN, "Not in the last 2 months", "Within the last 2 months", "Within the last 2 weeks"]
INSECTS = [UNKNOWN, "No", "Yes"]
AFFECTED_LEAVES = [UNKNOWN, "Older / lower leaves first", "New / upper leaves first", "Whole plant evenly"]


@dataclass
class CareContext:
    symptoms: list[str] = field(default_factory=list)
    soil_moisture: str = UNKNOWN
    watering_frequency: str = UNKNOWN
    drainage: str = UNKNOWN
    location: str = UNKNOWN
    sunlight: str = UNKNOWN
    weather: str = UNKNOWN
    fertilizer: str = UNKNOWN
    insects_visible: str = UNKNOWN
    affected_leaves: str = UNKNOWN

    def has(self, symptom: str) -> bool:
        return symptom in self.symptoms

    def answered(self) -> dict[str, str]:
        """Non-default answers, for display and for the history record."""
        out = {k: v for k, v in self.__dict__.items() if k != "symptoms" and v != UNKNOWN}
        if self.symptoms:
            out["symptoms"] = ", ".join(self.symptoms)
        return out
