from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch


KERNEL_LENGTHS = {
    "rq10": 10.0,
    "rq20": 20.0,
    "matern20": 20.0,
}


def _kernel(
    query_positions: torch.Tensor,
    train_positions: torch.Tensor,
    query_features: torch.Tensor,
    train_features: torch.Tensor,
    name: str,
    feature_length: float,
) -> torch.Tensor:
    if name not in KERNEL_LENGTHS:
        raise ValueError(f"Unknown GP kernel: {name}")
    spatial = torch.cdist(query_positions[:, :2].float(), train_positions[:, :2].float())
    radius = spatial / float(KERNEL_LENGTHS[name])
    if name.startswith("rq"):
        spatial_kernel = (1.0 + 0.5 * radius.square()).reciprocal()
    else:
        root5 = math.sqrt(5.0)
        spatial_kernel = (
            1.0 + root5 * radius + 5.0 / 3.0 * radius.square()
        ) * torch.exp(-root5 * radius)
    feature_distance = torch.cdist(query_features.float(), train_features.float())
    feature_distance = feature_distance / math.sqrt(max(query_features.shape[1], 1))
    feature_kernel = torch.exp(-0.5 * (feature_distance / float(feature_length)).square())
    return spatial_kernel * (0.5 + 0.5 * feature_kernel)


@dataclass
class SharedMultiOutputGP:
    kernel_name: str
    noise: float = 0.01
    feature_length: float = 1.0
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    target_mean: np.ndarray | None = None
    target_std: np.ndarray | None = None
    train_positions: np.ndarray | None = None
    train_features: np.ndarray | None = None
    alpha: np.ndarray | None = None
    jitter_used: float = 0.0

    def fit(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        device: torch.device | str = "cpu",
    ) -> "SharedMultiOutputGP":
        positions = np.asarray(positions, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape [samples,3]")
        if features.ndim != 2 or targets.ndim != 2:
            raise ValueError("features and targets must be 2D")
        if not (len(positions) == len(features) == len(targets)):
            raise ValueError("positions, features, and targets must have equal rows")
        if len(positions) < 2:
            raise ValueError("GP requires at least two training samples")
        self.feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_std = np.maximum(
            features.std(axis=0, dtype=np.float64), 1e-4
        ).astype(np.float32)
        normalized_features = np.clip(
            (features - self.feature_mean) / self.feature_std, -8.0, 8.0
        )
        self.target_mean = targets.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.target_std = np.maximum(
            targets.std(axis=0, dtype=np.float64), 1e-4
        ).astype(np.float32)
        normalized_targets = (targets - self.target_mean) / self.target_std
        target_device = torch.device(device)
        position_tensor = torch.as_tensor(positions, device=target_device)
        feature_tensor = torch.as_tensor(normalized_features, device=target_device)
        target_tensor = torch.as_tensor(normalized_targets, device=target_device)
        covariance = _kernel(
            position_tensor,
            position_tensor,
            feature_tensor,
            feature_tensor,
            self.kernel_name,
            self.feature_length,
        )
        identity = torch.eye(len(covariance), device=target_device, dtype=covariance.dtype)
        jitter = max(float(self.noise), 1e-6)
        factor = None
        for _ in range(8):
            factor, info = torch.linalg.cholesky_ex(covariance + jitter * identity)
            if int(info.max().item()) == 0:
                break
            jitter *= 10.0
        if factor is None or int(info.max().item()) != 0:
            raise RuntimeError(f"GP covariance remained non-positive definite at jitter={jitter}")
        solved = torch.cholesky_solve(target_tensor, factor)
        self.train_positions = positions.astype(np.float32)
        self.train_features = normalized_features.astype(np.float32)
        self.alpha = solved.detach().cpu().numpy().astype(np.float32)
        self.jitter_used = float(jitter)
        return self

    def _check(self) -> tuple[np.ndarray, ...]:
        values = (
            self.feature_mean,
            self.feature_std,
            self.target_mean,
            self.target_std,
            self.train_positions,
            self.train_features,
            self.alpha,
        )
        if any(value is None for value in values):
            raise RuntimeError("GP has not been fitted")
        return tuple(np.asarray(value) for value in values)

    @torch.no_grad()
    def predict(
        self,
        positions: np.ndarray,
        features: np.ndarray,
        device: torch.device | str = "cpu",
        batch_size: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        (
            feature_mean,
            feature_std,
            target_mean,
            target_std,
            train_positions,
            train_features,
            alpha,
        ) = self._check()
        positions = np.asarray(positions, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)
        normalized_features = np.clip((features - feature_mean) / feature_std, -8.0, 8.0)
        target_device = torch.device(device)
        train_position_tensor = torch.as_tensor(train_positions, device=target_device)
        train_feature_tensor = torch.as_tensor(train_features, device=target_device)
        alpha_tensor = torch.as_tensor(alpha, device=target_device)
        output = np.empty((len(positions), alpha.shape[1]), dtype=np.float32)
        uncertainty = np.empty(len(positions), dtype=np.float32)
        for start in range(0, len(positions), int(batch_size)):
            stop = min(start + int(batch_size), len(positions))
            query_positions = torch.as_tensor(positions[start:stop], device=target_device)
            query_features = torch.as_tensor(normalized_features[start:stop], device=target_device)
            cross = _kernel(
                query_positions,
                train_position_tensor,
                query_features,
                train_feature_tensor,
                self.kernel_name,
                self.feature_length,
            )
            predicted = cross @ alpha_tensor
            output[start:stop] = predicted.cpu().numpy() * target_std + target_mean
            uncertainty[start:stop] = (1.0 - cross.square().sum(dim=1) / max(len(train_positions), 1)).clamp(0.0, 1.0).cpu().numpy()
        return output, uncertainty

    def state_dict(self) -> dict:
        self._check()
        return {
            "kernel_name": self.kernel_name,
            "noise": float(self.noise),
            "feature_length": float(self.feature_length),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "train_positions": self.train_positions,
            "train_features": self.train_features,
            "alpha": self.alpha,
            "jitter_used": float(self.jitter_used),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SharedMultiOutputGP":
        model = cls(
            kernel_name=str(state["kernel_name"]),
            noise=float(state["noise"]),
            feature_length=float(state["feature_length"]),
        )
        for name in (
            "feature_mean",
            "feature_std",
            "target_mean",
            "target_std",
            "train_positions",
            "train_features",
            "alpha",
        ):
            setattr(model, name, np.asarray(state[name], dtype=np.float32))
        model.jitter_used = float(state.get("jitter_used", model.noise))
        return model


def convex_grid_weights(scores: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Choose non-negative three-model weights from per-sample losses."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must have shape [samples,models]")
    model_count = values.shape[1]
    if model_count == 1:
        return np.ones(1, dtype=np.float32)
    if model_count != 3:
        inverse = 1.0 / np.maximum(values.mean(axis=0), 1e-8)
        return (inverse / inverse.sum()).astype(np.float32)
    best_loss = float("inf")
    best = np.full(3, 1.0 / 3.0, dtype=np.float64)
    grid = np.arange(0.0, 1.0 + step * 0.5, step)
    for first in grid:
        for second in grid:
            third = 1.0 - first - second
            if third < -1e-9:
                continue
            weights = np.asarray([first, second, max(third, 0.0)])
            loss = float(np.mean(values @ weights))
            if loss < best_loss:
                best_loss = loss
                best = weights
    return (best / best.sum()).astype(np.float32)


def convex_cosine_weights(
    log_predictions: np.ndarray,
    log_targets: np.ndarray,
    scale: float,
    step: float = 0.05,
) -> np.ndarray:
    """Select weights from the cosine score of the actual power-domain ensemble."""
    predictions = np.asarray(log_predictions, dtype=np.float32)
    targets = np.asarray(log_targets, dtype=np.float32)
    if predictions.ndim != 3 or targets.ndim != 2:
        raise ValueError("Expected predictions [models,samples,features] and 2D targets")
    if predictions.shape[1:] != targets.shape:
        raise ValueError("Prediction and target shapes do not match")
    model_count = predictions.shape[0]
    if model_count == 1:
        return np.ones(1, dtype=np.float32)
    if model_count != 3:
        raise ValueError("Cosine grid search currently requires exactly three models")
    if scale <= 0.0:
        raise ValueError("Power log scale must be positive")

    prediction_power = np.expm1(np.clip(predictions, 0.0, 20.0)) / float(scale)
    target_power = np.expm1(np.clip(targets, 0.0, 20.0)) / float(scale)
    target_norm = np.linalg.norm(target_power, axis=1)
    valid = target_norm > 1e-12
    if not np.any(valid):
        return np.full(model_count, 1.0 / model_count, dtype=np.float32)

    prediction_power = prediction_power[:, valid]
    target_power = target_power[valid]
    target_norm = target_norm[valid]
    target_dot = np.einsum(
        "msd,sd->sm", prediction_power, target_power, optimize=True
    )
    prediction_gram = np.einsum(
        "msd,nsd->smn", prediction_power, prediction_power, optimize=True
    )
    best_score = -float("inf")
    best = np.full(3, 1.0 / 3.0, dtype=np.float64)
    grid = np.arange(0.0, 1.0 + step * 0.5, step)
    for first in grid:
        for second in grid:
            third = 1.0 - first - second
            if third < -1e-9:
                continue
            weights = np.asarray([first, second, max(third, 0.0)], dtype=np.float64)
            numerator = target_dot @ weights
            prediction_norm = np.sqrt(
                np.maximum(
                    np.einsum(
                        "i,sij,j->s",
                        weights,
                        prediction_gram,
                        weights,
                        optimize=True,
                    ),
                    1e-30,
                )
            )
            cosine = numerator / np.maximum(prediction_norm * target_norm, 1e-30)
            score = float(np.mean(np.clip(cosine, 0.0, 1.0)))
            if score > best_score:
                best_score = score
                best = weights
    return (best / best.sum()).astype(np.float32)


def convex_mse_weights(
    predictions: np.ndarray, targets: np.ndarray, step: float = 0.05
) -> np.ndarray:
    """Select convex weights using variance-normalized multi-output MSE."""
    values = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    if values.ndim != 3 or target.ndim != 2 or values.shape[1:] != target.shape:
        raise ValueError("Expected predictions [models,samples,features] matching targets")
    model_count = values.shape[0]
    if model_count == 1:
        return np.ones(1, dtype=np.float32)
    if model_count != 3:
        raise ValueError("MSE grid search currently requires exactly three models")
    scale = np.maximum(target.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
    errors = (values - target[None]) / scale[None, None, :]
    gram = np.einsum("msd,nsd->mn", errors, errors, optimize=True)
    gram /= max(target.size, 1)
    best_loss = float("inf")
    best = np.full(3, 1.0 / 3.0, dtype=np.float64)
    grid = np.arange(0.0, 1.0 + step * 0.5, step)
    for first in grid:
        for second in grid:
            third = 1.0 - first - second
            if third < -1e-9:
                continue
            weights = np.asarray([first, second, max(third, 0.0)], dtype=np.float64)
            loss = float(weights @ gram @ weights)
            if loss < best_loss:
                best_loss = loss
                best = weights
    return (best / best.sum()).astype(np.float32)


def ensemble_predictions(predictions: Iterable[np.ndarray], weights: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float32) for value in predictions]
    if not arrays:
        raise ValueError("No predictions were provided")
    if len(arrays) != len(weights):
        raise ValueError("Prediction and weight counts differ")
    output = np.zeros_like(arrays[0], dtype=np.float32)
    for weight, value in zip(weights, arrays, strict=True):
        output += float(weight) * value
    return output


def ensemble_log_power_predictions(
    predictions: Iterable[np.ndarray], weights: np.ndarray, scale: float
) -> np.ndarray:
    """Blend non-negative physical powers, then return the configured log encoding."""
    arrays = [np.asarray(value, dtype=np.float32) for value in predictions]
    if not arrays or len(arrays) != len(weights):
        raise ValueError("Prediction and weight counts differ")
    if scale <= 0.0:
        raise ValueError("Power log scale must be positive")
    output_power = np.zeros_like(arrays[0], dtype=np.float32)
    for weight, value in zip(weights, arrays, strict=True):
        power = np.expm1(np.clip(value, 0.0, 20.0)) / float(scale)
        output_power += float(weight) * power
    return np.log1p(float(scale) * np.maximum(output_power, 0.0)).astype(np.float32)
