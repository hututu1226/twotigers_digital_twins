from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class SpectralCompressor:
    output_dim: int
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    components: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None

    def fit(self, values: np.ndarray, seed: int = 2026) -> "SpectralCompressor":
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) < 2:
            raise ValueError("SpectralCompressor requires a 2D matrix with at least two rows")
        self.mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = matrix - self.mean
        self.scale = np.maximum(centered.std(axis=0, dtype=np.float64), 1e-3).astype(np.float32)
        normalized = centered / self.scale
        fitted_dim = min(int(self.output_dim), len(matrix) - 1, matrix.shape[1])
        pca = PCA(
            n_components=fitted_dim,
            svd_solver="randomized" if fitted_dim < min(normalized.shape) else "full",
            random_state=int(seed),
            whiten=False,
        )
        pca.fit(normalized)
        components = np.zeros((int(self.output_dim), matrix.shape[1]), dtype=np.float32)
        components[:fitted_dim] = pca.components_.astype(np.float32)
        ratio = np.zeros(int(self.output_dim), dtype=np.float32)
        ratio[:fitted_dim] = pca.explained_variance_ratio_.astype(np.float32)
        self.components = components
        self.explained_variance_ratio = ratio
        return self

    def _check(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mean is None or self.scale is None or self.components is None:
            raise RuntimeError("SpectralCompressor has not been fitted")
        return self.mean, self.scale, self.components

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean, scale, components = self._check()
        matrix = np.asarray(values, dtype=np.float32)
        return ((matrix - mean) / scale) @ components.T

    def inverse_transform(self, latent: np.ndarray) -> np.ndarray:
        mean, scale, components = self._check()
        values = np.asarray(latent, dtype=np.float32) @ components
        return values * scale + mean

    def state_dict(self) -> dict[str, np.ndarray | int]:
        mean, scale, components = self._check()
        return {
            "output_dim": int(self.output_dim),
            "mean": mean,
            "scale": scale,
            "components": components,
            "explained_variance_ratio": np.asarray(
                self.explained_variance_ratio, dtype=np.float32
            ),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SpectralCompressor":
        instance = cls(int(state["output_dim"]))
        instance.mean = np.asarray(state["mean"], dtype=np.float32)
        instance.scale = np.asarray(state["scale"], dtype=np.float32)
        instance.components = np.asarray(state["components"], dtype=np.float32)
        instance.explained_variance_ratio = np.asarray(
            state["explained_variance_ratio"], dtype=np.float32
        )
        return instance
