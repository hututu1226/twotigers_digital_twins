from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE


def local_expert_settings(section: dict) -> list[tuple[str, int, float]]:
    settings: list[tuple[str, int, float]] = []
    names: set[str] = set()
    for raw in section.get("local_spectral_experts", []):
        name = str(raw["name"])
        neighbors = int(raw.get("neighbors", 8))
        distance_power = float(raw.get("distance_power", 1.0))
        if name in names:
            raise ValueError(f"Duplicate local spectral expert: {name}")
        if neighbors < 1:
            raise ValueError("Local spectral expert neighbors must be positive")
        if distance_power <= 0.0:
            raise ValueError("Local spectral expert distance_power must be positive")
        names.add(name)
        settings.append((name, neighbors, distance_power))
    return settings


def _decode_log_power(values: np.ndarray, scale: float) -> np.ndarray:
    return np.expm1(np.clip(values.astype(np.float32), 0.0, 20.0)) / float(scale)


def _encode_log_power(values: np.ndarray, scale: float) -> np.ndarray:
    return np.log1p(float(scale) * np.maximum(values, 0.0)).astype(np.float32)


def local_spectral_prediction(
    support_positions: np.ndarray,
    query_positions: np.ndarray,
    support_pas_log: np.ndarray,
    support_pdp_log: np.ndarray,
    support_ue_log_energy: np.ndarray,
    support_log_power: np.ndarray,
    *,
    neighbors: int,
    distance_power: float,
    minimum_distance: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_positions = np.asarray(support_positions, dtype=np.float32)
    query_positions = np.asarray(query_positions, dtype=np.float32)
    if len(support_positions) < 1:
        raise ValueError("Local spectral prediction requires at least one support sample")
    count = min(int(neighbors), len(support_positions))
    distance, local = cKDTree(support_positions[:, :2]).query(
        query_positions[:, :2], k=count
    )
    distance = np.asarray(distance, dtype=np.float64).reshape(len(query_positions), count)
    local = np.asarray(local, dtype=np.int64).reshape(len(query_positions), count)
    exact = distance <= 1e-8
    inverse = 1.0 / np.maximum(distance, float(minimum_distance)) ** float(
        distance_power
    )
    weights = inverse / np.maximum(inverse.sum(axis=1, keepdims=True), 1e-12)
    exact_rows = exact.any(axis=1)
    if np.any(exact_rows):
        exact_weights = exact[exact_rows].astype(np.float64)
        exact_weights /= exact_weights.sum(axis=1, keepdims=True)
        weights[exact_rows] = exact_weights

    pas_power = _decode_log_power(support_pas_log[local], PAS_LOG_SCALE)
    pdp_power = _decode_log_power(support_pdp_log[local], PDP_LOG_SCALE)
    pas = _encode_log_power(
        np.einsum("ij,ijk->ik", weights, pas_power, optimize=True), PAS_LOG_SCALE
    )
    pdp = _encode_log_power(
        np.einsum("ij,ijk->ik", weights, pdp_power, optimize=True), PDP_LOG_SCALE
    )
    ue = np.einsum(
        "ij,ijk->ik", weights, support_ue_log_energy[local], optimize=True
    ).astype(np.float32)
    power = np.einsum(
        "ij,ij->i", weights, support_log_power[local], optimize=True
    ).astype(np.float32)
    uncertainty = (distance[:, 0] / (distance[:, 0] + 10.0)).astype(np.float32)
    return pas, pdp, ue, power, uncertainty
