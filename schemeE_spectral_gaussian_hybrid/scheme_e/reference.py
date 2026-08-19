from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_reference_candidates(
    target_positions: np.ndarray,
    target_cells: np.ndarray,
    observed_positions: np.ndarray,
    observed_cells: np.ndarray,
    observed_outage: np.ndarray,
    top_k: int = 64,
    target_global_indices: np.ndarray | None = None,
    observed_global_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    target_positions = np.asarray(target_positions, dtype=np.float64)
    observed_positions = np.asarray(observed_positions, dtype=np.float64)
    target_cells = np.asarray(target_cells, dtype=np.int64)
    observed_cells = np.asarray(observed_cells, dtype=np.int64)
    observed_outage = np.asarray(observed_outage, dtype=np.bool_)
    candidates = np.full((len(target_positions), top_k), -1, dtype=np.int64)
    distances = np.full((len(target_positions), top_k), np.inf, dtype=np.float32)
    target_global = (
        np.asarray(target_global_indices, dtype=np.int64)
        if target_global_indices is not None
        else None
    )
    observed_global = (
        np.asarray(observed_global_indices, dtype=np.int64)
        if observed_global_indices is not None
        else None
    )
    for cell in np.unique(target_cells):
        target_local = np.flatnonzero(target_cells == cell)
        observed_local = np.flatnonzero((observed_cells == cell) & ~observed_outage)
        if not len(observed_local):
            raise RuntimeError(f"Cell {cell} has no non-outage reference channel")
        tree = cKDTree(observed_positions[observed_local, :2])
        query_k = min(len(observed_local), top_k + (1 if target_global is not None else 0))
        query_distance, query_index = tree.query(
            target_positions[target_local, :2], k=query_k
        )
        if query_k == 1:
            query_distance = np.asarray(query_distance).reshape(-1, 1)
            query_index = np.asarray(query_index).reshape(-1, 1)
        else:
            query_distance = np.atleast_2d(query_distance)
            query_index = np.atleast_2d(query_index)
        if len(target_local) == 1 and query_distance.shape[0] != 1:
            query_distance = query_distance.T
            query_index = query_index.T
        for row, target_index in enumerate(target_local):
            mapped = observed_local[np.asarray(query_index[row]).reshape(-1)]
            mapped_distance = np.asarray(query_distance[row]).reshape(-1)
            if target_global is not None and observed_global is not None:
                keep = observed_global[mapped] != target_global[target_index]
                mapped = mapped[keep]
                mapped_distance = mapped_distance[keep]
            count = min(top_k, len(mapped))
            candidates[target_index, :count] = mapped[:count]
            distances[target_index, :count] = mapped_distance[:count]
            if count < top_k and count:
                candidates[target_index, count:] = mapped[count - 1]
                distances[target_index, count:] = mapped_distance[count - 1]
    if np.any(candidates[:, 0] < 0):
        raise RuntimeError("At least one target has no reference candidate")
    return candidates, distances


def sample_references(
    candidates: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
    guard_min_meters: float,
    guard_max_meters: float,
) -> np.ndarray:
    output = np.empty(len(candidates), dtype=np.int64)
    for row in range(len(candidates)):
        lower = float(rng.uniform(guard_min_meters, guard_max_meters))
        valid = np.flatnonzero(np.isfinite(distances[row]) & (distances[row] >= lower))
        if not len(valid):
            valid = np.flatnonzero(np.isfinite(distances[row]))
        rank = int(rng.choice(valid[: min(len(valid), 12)]))
        output[row] = candidates[row, rank]
    return output
