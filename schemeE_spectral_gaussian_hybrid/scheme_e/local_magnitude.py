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
    shifts: np.ndarray | None = None,
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
    if shifts is not None:
        shifts = np.asarray(shifts, dtype=np.int64)
        if shifts.shape != (len(residual), neighbor_base.shape[1], 3):
            raise ValueError("shifts must be [B,K,3]")
        aligned = np.empty_like(residual)
        for batch in range(len(residual)):
            for neighbor in range(active):
                aligned[batch, neighbor] = np.roll(
                    residual[batch, neighbor],
                    tuple(int(value) for value in shifts[batch, neighbor]),
                    axis=(1, 2, 3),
                )
        residual = aligned
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


def estimate_magnitude_profile_shifts(
    query_base: np.ndarray,
    neighbor_base: np.ndarray,
    *,
    scale: float = 4.0,
    maximum_vertical_shift: int = 2,
    maximum_horizontal_shift: int = 4,
    maximum_delay_shift: int = 12,
) -> np.ndarray:
    """Estimate circular angle/delay shifts from known Teacher magnitude profiles."""
    query = np.asarray(query_base, dtype=np.float32)
    neighbors = np.asarray(neighbor_base, dtype=np.float32)
    if query.ndim != 5 or neighbors.ndim != 6:
        raise ValueError("expected query [B,C,V,H,S] and neighbors [B,K,C,V,H,S]")
    if len(query) != len(neighbors) or query.shape[1:] != neighbors.shape[2:]:
        raise ValueError("query and neighbor map shapes are inconsistent")
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    query_power = np.expm1(np.clip(query, 0.0, 20.0)) / float(scale)
    neighbor_power = np.expm1(np.clip(neighbors, 0.0, 20.0)) / float(scale)
    query_angle = query_power.sum(axis=(1, 4))
    neighbor_angle = neighbor_power.sum(axis=(2, 5))
    query_delay = query_power.sum(axis=(1, 2, 3))
    neighbor_delay = neighbor_power.sum(axis=(2, 3, 4))

    def normalized(value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        centered = value - value.mean(axis=axes, keepdims=True)
        norm = np.sqrt(np.square(centered, dtype=np.float64).sum(axis=axes, keepdims=True))
        return centered / np.maximum(norm, 1e-12)

    query_angle = normalized(query_angle, (1, 2))
    neighbor_angle = normalized(neighbor_angle, (2, 3))
    angle_correlation = np.fft.ifftn(
        np.fft.fftn(query_angle[:, None], axes=(-2, -1))
        * np.conj(np.fft.fftn(neighbor_angle, axes=(-2, -1))),
        axes=(-2, -1),
    ).real
    vertical, horizontal = query.shape[2:4]
    signed_vertical = np.where(
        np.arange(vertical) <= vertical // 2,
        np.arange(vertical),
        np.arange(vertical) - vertical,
    )
    signed_horizontal = np.where(
        np.arange(horizontal) <= horizontal // 2,
        np.arange(horizontal),
        np.arange(horizontal) - horizontal,
    )
    allowed_angle = (
        np.abs(signed_vertical[:, None]) <= int(maximum_vertical_shift)
    ) & (
        np.abs(signed_horizontal[None, :]) <= int(maximum_horizontal_shift)
    )
    angle_correlation = np.where(allowed_angle[None, None], angle_correlation, -np.inf)
    angle_index = angle_correlation.reshape(len(query), len(neighbors[0]), -1).argmax(axis=2)
    vertical_index, horizontal_index = np.unravel_index(
        angle_index, (vertical, horizontal)
    )

    query_delay = normalized(query_delay, (1,))
    neighbor_delay = normalized(neighbor_delay, (2,))
    delay_correlation = np.fft.ifft(
        np.fft.fft(query_delay[:, None], axis=-1)
        * np.conj(np.fft.fft(neighbor_delay, axis=-1)),
        axis=-1,
    ).real
    delay = query.shape[-1]
    signed_delay = np.where(
        np.arange(delay) <= delay // 2,
        np.arange(delay),
        np.arange(delay) - delay,
    )
    delay_correlation = np.where(
        (np.abs(signed_delay) <= int(maximum_delay_shift))[None, None],
        delay_correlation,
        -np.inf,
    )
    delay_index = delay_correlation.argmax(axis=2)
    return np.stack(
        [
            signed_vertical[vertical_index],
            signed_horizontal[horizontal_index],
            signed_delay[delay_index],
        ],
        axis=2,
    ).astype(np.int16)
