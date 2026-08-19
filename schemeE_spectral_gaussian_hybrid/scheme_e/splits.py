from __future__ import annotations

import numpy as np


def spatial_block_folds(
    positions: np.ndarray,
    cell_ids: np.ndarray,
    fold_count: int = 8,
    tile_meters: float = 24.0,
    seed: int = 2026,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if tile_meters <= 0:
        raise ValueError("tile_meters must be positive")
    output = np.full(len(positions), -1, dtype=np.int16)
    rng = np.random.default_rng(int(seed))
    for cell_id in np.unique(cell_ids):
        indices = np.flatnonzero(cell_ids == cell_id)
        xy = positions[indices, :2]
        minimum = xy.min(axis=0)
        tiles = np.floor((xy - minimum) / float(tile_meters)).astype(np.int64)
        unique_tiles, inverse, counts = np.unique(
            tiles, axis=0, return_inverse=True, return_counts=True
        )
        order = np.arange(len(unique_tiles))
        rng.shuffle(order)
        order = order[np.argsort(-counts[order], kind="stable")]
        fold_sizes = np.zeros(fold_count, dtype=np.int64)
        tile_fold = np.empty(len(unique_tiles), dtype=np.int16)
        for tile_index in order:
            candidates = np.flatnonzero(fold_sizes == fold_sizes.min())
            selected = int(rng.choice(candidates))
            tile_fold[tile_index] = selected
            fold_sizes[selected] += counts[tile_index]
        output[indices] = tile_fold[inverse]
    if np.any(output < 0):
        raise RuntimeError("Some samples were not assigned to a spatial fold")
    for fold in range(fold_count):
        if not np.any(output == fold):
            raise RuntimeError(f"Spatial fold {fold} is empty")
    return output


def fold_summary(folds: np.ndarray, cell_ids: np.ndarray) -> list[dict[str, object]]:
    folds = np.asarray(folds)
    cell_ids = np.asarray(cell_ids)
    result: list[dict[str, object]] = []
    for fold in sorted(np.unique(folds).tolist()):
        mask = folds == fold
        result.append(
            {
                "fold": int(fold),
                "samples": int(mask.sum()),
                "cell_counts": {
                    str(int(cell)): int(np.sum(mask & (cell_ids == cell)))
                    for cell in np.unique(cell_ids)
                },
            }
        )
    return result
