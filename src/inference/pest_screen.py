"""Visible-pest screening.

Primary method: zero-shot CLIP ViT-L/14 (openai/clip-vit-large-patch14). The
image is compared with text descriptions of pest-infested and pest-free
leaves; the score is the total probability of the pest descriptions. CLIP
was not trained for this task. The threshold comes from validation images (a
~1% false-alarm rate on pest-free PlantVillage leaves). On IP102 field photos
it flags 83% of pest images (src/evaluation/evaluate_pest.py); tiny mites and
scale insects are missed most. It can suggest a *type* of pest appearance but
cannot identify species.

Fallback (if CLIP is not in the local Hugging Face cache): the ImageNet
MobileNetV2 head reused on the disease model's frozen features, which only
recognises large, whole insects (beetles, caterpillars, spiders...).

Neither model is ever downloaded at app runtime. Run
``python -m src.inference.download_models`` once to fetch CLIP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import MobileNet_V2_Weights

from src import config

logger = logging.getLogger(__name__)

# (description, is_pest, short label shown to users). Order matters only for display.
CLIP_PROMPTS = [
    ("a close-up photo of a plant leaf infested with small insects such as whiteflies, aphids, "
     "mealybugs or scale insects", True, "small sap-sucking insects (e.g. whiteflies, aphids, mealybugs, scale)"),
    ("a close-up photo of a plant leaf with insect eggs on it", True, "insect eggs"),
    ("a close-up photo of a plant leaf with spider mites and fine webbing", True, "spider mites / webbing"),
    ("a close-up photo of a caterpillar, beetle or other insect on a plant leaf", True,
     "larger insects (e.g. caterpillars, beetles)"),
    ("a close-up photo of a plant leaf with fungal or bacterial disease spots", False, "disease spots"),
    ("a close-up photo of a plant leaf covered in white powdery mildew", False, "powdery mildew"),
    ("a close-up photo of a plant leaf with yellowing or dead brown patches", False, "yellowing / dead patches"),
    ("a close-up photo of a healthy green plant leaf", False, "healthy leaf"),
]

# ImageNet-1k indices for arthropods and molluscs plausibly seen on plants (fallback only).
ARTHROPOD_CLASS_INDICES = list(range(70, 80)) + [113, 114] + list(range(300, 327))


@dataclass
class PestScreenResult:
    score: float                                  # total probability of pest descriptions
    method: str                                   # "clip" or "imagenet"
    top: list[tuple[str, float]] = field(default_factory=list)  # most likely pest appearances


class ClipPestScreen:
    method = "clip"

    def __init__(self, model, processor):
        self.model = model.eval()
        self.processor = processor
        self.texts = [p for p, _, _ in CLIP_PROMPTS]
        self.is_pest = torch.tensor([pest for _, pest, _ in CLIP_PROMPTS])
        self.labels = [label for _, _, label in CLIP_PROMPTS]
        self.text_embeddings = self.embed_texts(self.texts)

    @classmethod
    def from_cache(cls, model_id: str = config.PEST_CLIP_MODEL) -> "ClipPestScreen | None":
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            logger.warning("transformers not installed; CLIP pest screen disabled")
            return None
        try:
            model = CLIPModel.from_pretrained(model_id, local_files_only=True)
            processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        except (OSError, ValueError) as exc:
            logger.warning("CLIP model %s not in local cache (%s); run "
                           "`python -m src.inference.download_models`", model_id, exc)
            return None
        return cls(model, processor)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return nn.functional.normalize(self.model.get_text_features(**inputs).pooler_output, dim=-1)

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> torch.Tensor:
        """L2-normalised CLIP image embeddings, shared with the symptom screen."""
        inputs = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        return nn.functional.normalize(self.model.get_image_features(**inputs).pooler_output, dim=-1)

    def probabilities_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.model.logit_scale.exp().detach() * embeddings @ self.text_embeddings.T
        return logits.softmax(dim=-1)

    def probabilities(self, images: list[Image.Image]) -> torch.Tensor:
        return self.probabilities_from_embeddings(self.embed(images))

    def score(self, image: Image.Image) -> PestScreenResult:
        return self.score_embedding(self.embed([image])[0])

    def score_embedding(self, embedding: torch.Tensor) -> PestScreenResult:
        probs = self.probabilities_from_embeddings(embedding.unsqueeze(0))[0]
        pest_probs = probs[self.is_pest]
        pest_labels = [l for l, p in zip(self.labels, self.is_pest) if p]
        order = torch.argsort(pest_probs, descending=True)
        top = [(pest_labels[i], float(pest_probs[i])) for i in order[:2]]
        return PestScreenResult(float(pest_probs.sum()), self.method, top)


class ImageNetPestScreen:
    """Fallback: ImageNet head on the disease model's frozen MobileNetV2 features."""

    method = "imagenet"

    def __init__(self, head: nn.Linear, categories: list[str], feature_fn, transform):
        self.head = head.eval()
        self.categories = categories
        self.feature_fn = feature_fn
        self.transform = transform

    @classmethod
    def from_cache(cls, feature_fn, transform) -> "ImageNetPestScreen | None":
        matches = sorted(config.TORCH_HUB_CHECKPOINTS.glob(config.IMAGENET_MOBILENET_GLOB))
        if not matches:
            return None
        state = torch.load(Path(matches[0]), map_location="cpu", weights_only=True)
        head = nn.Linear(1280, 1000)
        head.load_state_dict({"weight": state["classifier.1.weight"], "bias": state["classifier.1.bias"]})
        return cls(head, MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"], feature_fn, transform)

    @torch.no_grad()
    def score(self, image: Image.Image) -> PestScreenResult:
        feats = self.feature_fn(self.transform(image.convert("RGB")).unsqueeze(0))
        probs = torch.softmax(self.head(feats), dim=1)[0][ARTHROPOD_CLASS_INDICES]
        values, order = probs.topk(2)
        top = [(self.categories[ARTHROPOD_CLASS_INDICES[i]], float(v)) for v, i in zip(values, order)]
        return PestScreenResult(float(probs.sum()), self.method, top)
