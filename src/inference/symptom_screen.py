"""Plant-agnostic symptom screen for plants without a disease model.

The 38-class disease model can only answer with PlantVillage labels such as
"Tomato late blight", which is meaningless for e.g. a hibiscus. This screen
instead reports the *visible symptom pattern* (no obvious symptoms, leaf
spots / blight, powdery mildew, rust, mosaic / yellowing) by comparing the
CLIP ViT-L/14 image embedding with several text descriptions per pattern
(zero-shot; the embedding is shared with the pest screen).

Zero-shot was chosen over a probe trained on PlantVillage: in leave-one-
species-out tests a trained probe learned "squash -> mildew"-style species
shortcuts and recognised 0 % of mildew, rust and mosaic on unseen plants. See
src/evaluation/evaluate_symptoms.py. It is not a diagnosis and never names a
species-specific disease.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


# key -> (display name, what the pattern looks like)
SYMPTOMS = {
    "healthy": ("No obvious disease symptoms",
                "Leaf colour and surface look like healthy leaves in the training data."),
    "spots": ("Leaf spots or blight",
              "Brown, black or water-soaked spots, lesions or dead patches, often with yellow halos."),
    "mildew": ("Powdery mildew-like coating",
               "A white or grey powdery coating on the leaf surface."),
    "rust": ("Rust-like pustules",
             "Small orange, rusty-brown or yellow raised spots or pustules."),
    "mottling": ("Mosaic, mottling or yellowing",
                 "Patchy light/dark green mosaic, mottled or blotchy yellowing, or curled, distorted leaves, "
                 "as seen with viruses, nutrient problems or root stress."),
}

_SPOTS = ["Apple___Apple_scab", "Apple___Black_rot", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
          "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
          "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Peach___Bacterial_spot", "Pepper,_bell___Bacterial_spot",
          "Potato___Early_blight", "Potato___Late_blight", "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot",
          "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
          "Tomato___Target_Spot"]
_MILDEW = ["Cherry_(including_sour)___Powdery_mildew", "Squash___Powdery_mildew"]
_RUST = ["Apple___Cedar_apple_rust", "Corn_(maize)___Common_rust_"]
_MOTTLING = ["Orange___Haunglongbing_(Citrus_greening)", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
             "Tomato___Tomato_mosaic_virus"]


def symptom_of(label: str) -> str | None:
    """Symptom group of a PlantVillage label; None for spider mites (handled by the pest screen)."""
    if label.endswith("___healthy"):
        return "healthy"
    for key, labels in (("spots", _SPOTS), ("mildew", _MILDEW), ("rust", _RUST), ("mottling", _MOTTLING)):
        if label in labels:
            return key
    return None


@dataclass
class SymptomResult:
    key: str                                   # most likely symptom group
    probability: float
    probabilities: dict[str, float] = field(default_factory=dict)
    clear: bool = True                         # probability >= threshold
    second: str | None = None                  # runner-up, set when two patterns together dominate

    @property
    def mixed(self) -> bool:
        return not self.clear and self.second is not None

    @property
    def keys(self) -> list[str]:
        """Patterns worth acting on: one if clear, two if mixed, none otherwise."""
        return [self.key] if self.clear else [self.key, self.second] if self.mixed else []

    @property
    def name(self) -> str:
        return SYMPTOMS[self.key][0]

    @property
    def description(self) -> str:
        return SYMPTOMS[self.key][1]


# Several phrasings per pattern, averaged, are more robust than a single prompt.
SYMPTOM_PROMPTS = {
    "healthy": ["a photo of a healthy green plant leaf",
                "a photo of a clean, uniformly green leaf with no spots",
                "a close-up of a healthy leaf"],
    "spots": ["a photo of a plant leaf with brown disease spots",
              "a photo of a leaf with dark lesions and yellow halos",
              "a photo of a leaf with blight and dead brown patches",
              "a photo of a leaf with many small black spots"],
    "mildew": ["a photo of a plant leaf covered in white powdery mildew",
               "a photo of a leaf with white powdery fungus patches",
               "a photo of a leaf dusted with white powder"],
    "rust": ["a photo of a plant leaf with orange rust pustules",
             "a photo of a leaf with many small orange-brown raised spots",
             "a photo of a leaf with rusty orange spots"],
    "mottling": ["a photo of a plant leaf with yellow mosaic mottling",
                 "a photo of a leaf with blotchy yellow and green patches",
                 "a photo of a curled, yellowed, distorted leaf",
                 "a photo of a leaf with yellow veins and mottled colour"],
}
LOGIT_SCALE = 100.0  # CLIP's own logit scale


class SymptomScreen:
    """Zero-shot symptom pattern from a CLIP image embedding."""

    def __init__(self, clip_screen, threshold: float, pair_threshold: float = 1.0):
        self.classes = list(SYMPTOMS)
        texts = [t for c in self.classes for t in SYMPTOM_PROMPTS[c]]
        self.owner = np.array([c for c in self.classes for _ in SYMPTOM_PROMPTS[c]])
        self.text_embeddings = clip_screen.embed_texts(texts).numpy()
        self.threshold = threshold
        self.pair_threshold = pair_threshold   # top-2 probability needed to report two patterns

    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        sim = LOGIT_SCALE * np.asarray(embeddings, dtype=np.float64) @ self.text_embeddings.T
        z = np.stack([sim[:, self.owner == c].mean(1) for c in self.classes], axis=1)
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def predict(self, embedding: torch.Tensor | np.ndarray) -> SymptomResult:
        p = self.probabilities(np.asarray(embedding).reshape(1, -1))[0]
        best, runner_up = (int(i) for i in np.argsort(-p)[:2])
        probs = {c: float(v) for c, v in zip(self.classes, p)}
        clear = float(p[best]) >= self.threshold
        second = (self.classes[runner_up]
                  if not clear and float(p[best] + p[runner_up]) >= self.pair_threshold else None)
        return SymptomResult(self.classes[best], float(p[best]), probs, clear=clear, second=second)
