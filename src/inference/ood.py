"""Out-of-distribution (OOD) scoring with class-conditional Mahalanobis distance.

Following Lee et al. (2018), we fit one Gaussian per class on the classifier's
penultimate features (shared covariance) using the *training* split, and score
an image by its distance to the nearest class mean. The rejection threshold is
a percentile of validation-set scores, so by construction about
(100 - percentile)% of ordinary PlantVillage images are also flagged.

Limitations: the detector only knows what PlantVillage leaves look like. A
field photo of a supported plant (cluttered background, different lighting)
can be flagged as unfamiliar, and an unfamiliar leaf that happens to resemble
the training images can pass. Its true detection rate on real OOD photos has
not been measured because no labelled OOD set is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MahalanobisScorer:
    def __init__(self, class_means: np.ndarray, precision: np.ndarray):
        self.class_means = np.asarray(class_means, dtype=np.float64)
        self.precision = np.asarray(precision, dtype=np.float64)

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, num_classes: int,
            shrinkage: float = 1e-3) -> "MahalanobisScorer":
        features = np.asarray(features, dtype=np.float64)
        means = np.stack([features[labels == c].mean(axis=0) for c in range(num_classes)])
        centered = features - means[labels]
        cov = centered.T @ centered / len(features)
        # Shrink towards a scaled identity for a stable inverse.
        cov += shrinkage * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        return cls(means, np.linalg.inv(cov))

    def score(self, features: np.ndarray) -> np.ndarray:
        """Minimum Mahalanobis distance to any class mean (higher = more unusual)."""
        f = np.atleast_2d(np.asarray(features, dtype=np.float64))
        # (f - mu)^T P (f - mu) = f^T P f - 2 f^T P mu + mu^T P mu, computed with
        # matrix products instead of materialising an (N, C, D) difference array.
        fp = f @ self.precision                                   # (N, D)
        ff = np.einsum("nd,nd->n", fp, f)                        # (N,)
        mm = np.einsum("cd,cd->c", self.class_means @ self.precision, self.class_means)
        d2 = ff[:, None] - 2.0 * fp @ self.class_means.T + mm[None, :]
        return np.sqrt(np.maximum(d2.min(axis=1), 0.0))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, class_means=self.class_means.astype(np.float32),
                            precision=self.precision.astype(np.float32))

    @classmethod
    def load(cls, path: Path) -> "MahalanobisScorer":
        with np.load(path) as data:
            return cls(data["class_means"], data["precision"])
