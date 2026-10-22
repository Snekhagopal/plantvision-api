"""Plant catalog derived from the classifier's training labels.

PlantVillage labels look like ``"Tomato___Early_blight"``: the part before the
triple underscore is the host species, the rest is the condition. The catalog
groups the 38 classes by species so the app can check the user's plant name
against what the model was actually trained on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

OTHER_OPTION = "Other / Not listed"

# Display name for each PlantVillage species prefix.
SPECIES_DISPLAY = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry (incl. sour cherry)",
    "Corn_(maize)": "Corn (maize)",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper,_bell": "Bell pepper",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}

# Extra names users commonly type. Keys are normalised (see normalize_name).
# Only unambiguous synonyms are listed; e.g. "pepper" alone could be chili or
# black pepper, so it is deliberately absent.
ALIASES = {
    "maize": "Corn_(maize)",
    "corn": "Corn_(maize)",
    "sweet corn": "Corn_(maize)",
    "cherry": "Cherry_(including_sour)",
    "sour cherry": "Cherry_(including_sour)",
    "sweet cherry": "Cherry_(including_sour)",
    "bell pepper": "Pepper,_bell",
    "sweet pepper": "Pepper,_bell",
    "capsicum": "Pepper,_bell",
    "pepper bell": "Pepper,_bell",
    "orange": "Orange",
    "sweet orange": "Orange",
    "orange tree": "Orange",
    "tomatoes": "Tomato",
    "potatoes": "Potato",
    "apple tree": "Apple",
    "peach tree": "Peach",
    "grapevine": "Grape",
    "grape vine": "Grape",
    "grapes": "Grape",
    "soy": "Soybean",
    "soya": "Soybean",
    "soybeans": "Soybean",
    "strawberries": "Strawberry",
    "blueberries": "Blueberry",
    "raspberries": "Raspberry",
    "pumpkin": None,  # explicitly ambiguous: not the same species set as "Squash"
}


def normalize_name(name: str) -> str:
    """Lower-case, strip punctuation/underscores and collapse whitespace."""
    text = re.sub(r"[_(),./\-]+", " ", str(name).lower())
    return re.sub(r"\s+", " ", text).strip()


def split_label(label: str) -> tuple[str, str]:
    """Split a PlantVillage label into (species_key, condition_key)."""
    species, _, condition = label.partition("___")
    return species, condition


def prettify_condition(condition_key: str) -> str:
    text = condition_key.replace("_", " ").strip()
    return re.sub(r"\s+", " ", text)


@dataclass
class SpeciesInfo:
    key: str
    display_name: str
    labels: list[str] = field(default_factory=list)

    @property
    def has_healthy_class(self) -> bool:
        return any(split_label(l)[1].lower() == "healthy" for l in self.labels)

    @property
    def disease_labels(self) -> list[str]:
        return [l for l in self.labels if split_label(l)[1].lower() != "healthy"]

    @property
    def coverage_warning(self) -> str | None:
        """Explain gaps in what the model saw for this species, if any."""
        if not self.disease_labels:
            return (
                f"For {self.display_name}, the model was trained on healthy leaves only. "
                "It cannot recognise any disease of this plant; a 'healthy' output only "
                "means the photo resembles healthy training images."
            )
        if not self.has_healthy_class:
            return (
                f"For {self.display_name}, the model never saw healthy leaves - only "
                f"{len(self.disease_labels)} disease class(es). It may label a healthy "
                "leaf as diseased, so treat this result with extra caution."
            )
        if len(self.disease_labels) == 1:
            return (
                f"For {self.display_name}, the model knows only one disease. Other "
                "diseases of this plant will be missed or mislabelled."
            )
        return None


@dataclass
class PlantMatch:
    """Result of matching the user's plant name to the catalog."""
    user_input: str
    species: SpeciesInfo | None

    @property
    def supported(self) -> bool:
        return self.species is not None


class PlantCatalog:
    def __init__(self, class_names: list[str]):
        if not class_names:
            raise ValueError("class_names must not be empty")
        self.class_names = list(class_names)
        self.species: dict[str, SpeciesInfo] = {}
        for label in self.class_names:
            key, _ = split_label(label)
            info = self.species.setdefault(
                key, SpeciesInfo(key, SPECIES_DISPLAY.get(key, prettify_condition(key)))
            )
            info.labels.append(label)

        self._lookup: dict[str, str] = {}
        for key, info in self.species.items():
            self._lookup[normalize_name(key)] = key
            self._lookup[normalize_name(info.display_name)] = key
        for alias, key in ALIASES.items():
            if key is None or key in self.species:
                self._lookup[normalize_name(alias)] = key

    def display_names(self) -> list[str]:
        return sorted(info.display_name for info in self.species.values())

    def selector_options(self) -> list[str]:
        return self.display_names() + [OTHER_OPTION]

    def match(self, plant_name: str) -> PlantMatch:
        """Map free text to a supported species, or report it as unsupported.

        Matching is exact on normalised names/aliases only. Fuzzy matching is
        intentionally avoided: silently turning "pepper mint" into "bell
        pepper" would attach the wrong disease classes to the user's plant.
        """
        norm = normalize_name(plant_name)
        key = self._lookup.get(norm)
        return PlantMatch(plant_name, self.species.get(key) if key else None)

    def species_of(self, label: str) -> SpeciesInfo:
        return self.species[split_label(label)[0]]

    def label_indices(self, species: SpeciesInfo) -> list[int]:
        return [self.class_names.index(l) for l in species.labels]

    def condition_display(self, label: str) -> str:
        return prettify_condition(split_label(label)[1])


# Training-time label order (index -> label) of models/mobilenet_best.pth.
# Used only to populate the plant selector when the checkpoint is unavailable;
# tests verify it matches the checkpoint's class_to_idx.
PLANTVILLAGE_LABELS = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]
