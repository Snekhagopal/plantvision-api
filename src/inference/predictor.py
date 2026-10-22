"""MobileNetV2 inference: calibrated class scores, OOD score and pest screen."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src import config
from src.inference.model import forward_with_features, load_mobilenet_checkpoint, pooled_features
from src.inference.ood import MahalanobisScorer
from src.inference.pest_screen import ClipPestScreen, ImageNetPestScreen
from src.inference.symptom_screen import SymptomResult, SymptomScreen
from src.plants.catalog import PlantCatalog
from src.utils.preprocessing import build_transform

logger = logging.getLogger(__name__)

PEST_SCREEN_SIZE = 224  # ImageNet head works best at its native resolution


def load_thresholds(path: Path = config.CALIBRATION_PATH) -> dict:
    thresholds = dict(config.DEFAULT_THRESHOLDS)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        thresholds.update({k: v for k, v in data.get("thresholds", {}).items() if k in thresholds})
    except FileNotFoundError:
        logger.warning("Calibration file %s not found; using default thresholds", path)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read calibration file %s (%s); using defaults", path, exc)
    return thresholds


@dataclass
class ModelOutput:
    probabilities: np.ndarray          # temperature-scaled softmax over all classes
    raw_probabilities: np.ndarray      # plain softmax (T = 1)
    top_k: list[tuple[str, float]]     # (label, calibrated probability)
    ood_score: float | None = None
    pest_score: float | None = None
    pest_top: list[tuple[str, float]] = field(default_factory=list)
    pest_method: str | None = None     # "clip" or "imagenet"
    symptom: SymptomResult | None = None  # plant-agnostic symptom pattern (needs CLIP)


class PlantDiseasePredictor:
    """Loads the checkpoint once; ``predict`` is safe to call repeatedly."""

    def __init__(self, checkpoint_path: Path = config.MOBILENET_CHECKPOINT,
                 calibration_path: Path = config.CALIBRATION_PATH,
                 ood_stats_path: Path = config.OOD_STATS_PATH,
                 device: str = "cpu", enable_pest_screen: bool = True):
        self.device = torch.device(device)
        loaded = load_mobilenet_checkpoint(checkpoint_path, self.device)
        self.model = loaded.model
        self.class_names = loaded.class_names
        self.image_size = loaded.image_size
        self.transform = build_transform(loaded.image_size, loaded.mean, loaded.std)
        self.pest_transform = build_transform(PEST_SCREEN_SIZE, loaded.mean, loaded.std)
        self.catalog = PlantCatalog(self.class_names)
        self.thresholds = load_thresholds(calibration_path)

        self.ood_scorer = None
        if Path(ood_stats_path).exists() and self.thresholds.get("ood_threshold") is not None:
            try:
                self.ood_scorer = MahalanobisScorer.load(ood_stats_path)
            except Exception as exc:
                logger.warning("OOD statistics could not be loaded: %s", exc)

        self.pest_screen = None
        if enable_pest_screen:
            self.pest_screen = self._load_pest_screen()
            if self.pest_screen is not None and self.pest_screen.method == "imagenet":
                # Each screen has its own score scale and threshold.
                self.thresholds["pest_screen_threshold"] = self.thresholds["pest_screen_threshold_imagenet"]
        # The symptom screen works on CLIP embeddings, so it is only available with the CLIP screen.
        self.symptom_screen = None
        if self.pest_screen is not None and self.pest_screen.method == "clip":
            try:
                self.symptom_screen = SymptomScreen(self.pest_screen, self.thresholds["symptom_clear_threshold"],
                                                    self.thresholds["symptom_pair_threshold"])
            except Exception as exc:
                logger.warning("Symptom screen unavailable: %s", exc)

    def _load_pest_screen(self):
        try:
            screen = ClipPestScreen.from_cache()
            if screen is not None:
                return screen
            return ImageNetPestScreen.from_cache(lambda x: pooled_features(self.model, x.to(self.device)),
                                                 self.pest_transform)
        except Exception as exc:
            logger.warning("Pest screen unavailable: %s", exc)
            return None

    @property
    def temperature(self) -> float:
        return float(self.thresholds.get("temperature", 1.0))

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = config.TOP_K) -> ModelOutput:
        self.model.eval()
        logits, feats = forward_with_features(self.model, self.preprocess(image))
        logits = logits[0].double().cpu()
        raw = torch.softmax(logits, dim=0).numpy()
        probs = torch.softmax(logits / self.temperature, dim=0).numpy()
        order = np.argsort(probs)[::-1][:top_k]
        output = ModelOutput(
            probabilities=probs,
            raw_probabilities=raw,
            top_k=[(self.class_names[i], float(probs[i])) for i in order],
        )
        if self.ood_scorer is not None:
            output.ood_score = float(self.ood_scorer.score(feats.cpu().numpy())[0])
        if self.pest_screen is not None:
            try:
                if self.pest_screen.method == "clip":
                    embedding = self.pest_screen.embed([image])[0]   # one CLIP pass for both screens
                    pest = self.pest_screen.score_embedding(embedding)
                    if self.symptom_screen is not None:
                        output.symptom = self.symptom_screen.predict(embedding.numpy())
                else:
                    pest = self.pest_screen.score(image)
                output.pest_score, output.pest_top, output.pest_method = pest.score, pest.top, pest.method
            except Exception:
                logger.exception("Pest/symptom screen failed; continuing without it")
        return output
