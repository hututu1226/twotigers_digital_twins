from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def same_cell_neighbors(
    positions: np.ndarray,
    cells: np.ndarray,
    support: np.ndarray,
    queries: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find same-cell spatial neighbors without allowing unsupported cells."""
    if int(count) < 1:
        raise ValueError("count must be positive")
    positions = np.asarray(positions, dtype=np.float32)
    cells = np.asarray(cells, dtype=np.int64)
    support = np.asarray(support, dtype=np.int64)
    queries = np.asarray(queries, dtype=np.int64)
    neighbors = np.empty((len(queries), int(count)), dtype=np.int64)
    distances = np.empty((len(queries), int(count)), dtype=np.float32)
    for cell in np.unique(cells[queries]):
        rows = np.flatnonzero(cells[queries] == cell)
        local_support = support[cells[support] == cell]
        if not len(local_support):
            raise ValueError(f"Cell {int(cell)} has no support observations")
        actual = min(int(count), len(local_support))
        local_distance, local_index = cKDTree(
            positions[local_support, :2]
        ).query(positions[queries[rows], :2], k=actual)
        local_distance = np.asarray(local_distance, dtype=np.float32).reshape(
            len(rows), actual
        )
        local_index = np.asarray(local_index, dtype=np.int64).reshape(
            len(rows), actual
        )
        selected = local_support[local_index]
        if actual < int(count):
            padding = int(count) - actual
            selected = np.concatenate(
                [selected, np.repeat(selected[:, -1:], padding, axis=1)], axis=1
            )
            local_distance = np.concatenate(
                [
                    local_distance,
                    np.repeat(local_distance[:, -1:], padding, axis=1),
                ],
                axis=1,
            )
        neighbors[rows] = selected
        distances[rows] = local_distance
    return neighbors, distances


def transfer_log_power_residual(
    query_base: np.ndarray,
    neighbor_base: np.ndarray,
    neighbor_target: np.ndarray,
    distances: np.ndarray,
    *,
    count: int,
    strength: float,
    distance_floor: float = 1.0,
    maximum_log_power: float = 20.0,
) -> np.ndarray:
    """Transfer a local weighted Teacher residual while retaining the query map."""
    query_base = np.asarray(query_base, dtype=np.float32)
    neighbor_base = np.asarray(neighbor_base, dtype=np.float32)
    neighbor_target = np.asarray(neighbor_target, dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32)
    if neighbor_base.shape != neighbor_target.shape or neighbor_base.ndim < 3:
        raise ValueError("neighbor maps must share [B,K,...]")
    if len(query_base) != len(neighbor_base) or distances.shape[:2] != neighbor_base.shape[:2]:
        raise ValueError("query, neighbor and distance batch shapes are inconsistent")
    active = min(int(count), int(neighbor_base.shape[1]))
    if active < 1:
        raise ValueError("count must select at least one neighbor")
    weights = 1.0 / (
        distances[:, :active].astype(np.float64) + float(distance_floor)
    )
    weights /= weights.sum(axis=1, keepdims=True).clip(min=1e-12)
    residual = neighbor_target[:, :active] - neighbor_base[:, :active]
    weight_shape = (len(weights), active) + (1,) * (residual.ndim - 2)
    correction = np.sum(
        residual.astype(np.float32) * weights.reshape(weight_shape).astype(np.float32),
        axis=1,
        dtype=np.float32,
    )
    return np.clip(
        query_base + float(strength) * correction,
        0.0,
        float(maximum_log_power),
    ).astype(np.float32)
