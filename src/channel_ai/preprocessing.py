from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import save_json
from .map_context import build_bev, build_map_tokens, read_ply_vertices


def load_setup(data_root: str | Path) -> dict[str, Any]:
    path = Path(data_root) / "Round2_Setup.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_two_cell_labels(positions: np.ndarray, base_stations: np.ndarray) -> tuple[np.ndarray, dict]:
    sample_count = len(positions)
    best = None
    for axis in (0, 1):
        order = np.argsort(positions[:, axis])
        values = positions[order, axis]
        gaps = np.diff(values)
        minimum_side = max(2, int(0.2 * sample_count))
        valid = np.arange(len(gaps))
        valid = valid[(valid + 1 >= minimum_side) & (sample_count - valid - 1 >= minimum_side)]
        local_index = valid[np.argmax(gaps[valid])]
        score = float(gaps[local_index] / max(np.ptp(values), 1e-6))
        candidate = (score, axis, float((values[local_index] + values[local_index + 1]) / 2.0))
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    score, axis, threshold = best
    low = positions[:, axis] <= threshold
    centroids = np.stack([positions[low].mean(axis=0), positions[~low].mean(axis=0)])
    cost_identity = np.linalg.norm(centroids[0] - base_stations[0]) + np.linalg.norm(centroids[1] - base_stations[1])
    cost_swapped = np.linalg.norm(centroids[0] - base_stations[1]) + np.linalg.norm(centroids[1] - base_stations[0])
    if cost_identity <= cost_swapped:
        labels = np.where(low, 0, 1)
        low_cell = 0
    else:
        labels = np.where(low, 1, 0)
        low_cell = 1
    details = {
        "axis": int(axis),
        "threshold": threshold,
        "relative_gap": score,
        "low_side_cell": low_cell,
        "counts": np.bincount(labels, minlength=2).tolist(),
    }
    return labels.astype(np.int64), details


def _kmeans(points: np.ndarray, cluster_count: int, seed: int, iterations: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, cluster_count):
        distance = np.min(
            np.stack([np.sum((points - center) ** 2, axis=1) for center in centers]), axis=0
        )
        centers.append(points[int(np.argmax(distance))])
    centers_array = np.asarray(centers, dtype=np.float64)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(iterations):
        distance = np.sum((points[:, None, :] - centers_array[None, :, :]) ** 2, axis=-1)
        next_labels = distance.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster in range(cluster_count):
            members = points[labels == cluster]
            if len(members):
                centers_array[cluster] = members.mean(axis=0)
    return labels


def spatial_block_split(
    positions: np.ndarray,
    cell_labels: np.ndarray,
    validation_fraction: float,
    blocks_per_cell: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    validation_parts = []
    for cell in (0, 1):
        cell_indices = np.flatnonzero(cell_labels == cell)
        clusters = _kmeans(
            positions[cell_indices, :2], min(blocks_per_cell, len(cell_indices)), seed + cell
        )
        target = max(1, int(round(validation_fraction * len(cell_indices))))
        selected = []
        cluster_order = rng.permutation(int(clusters.max()) + 1)
        for cluster in cluster_order:
            selected.extend(cell_indices[clusters == cluster].tolist())
            if len(selected) >= target:
                break
        validation_parts.append(np.asarray(selected, dtype=np.int64))
    validation = np.unique(np.concatenate(validation_parts))
    training = np.setdiff1d(np.arange(len(positions)), validation, assume_unique=True)
    return training.astype(np.int64), validation.astype(np.int64)


def channel_statistics(
    channel_path: str | Path, cell_labels: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channels = np.load(channel_path, mmap_mode="r")
    power = np.empty(len(channels), dtype=np.float64)
    outage = np.empty(len(channels), dtype=np.bool_)
    for start in range(0, len(channels), chunk_size):
        stop = min(start + chunk_size, len(channels))
        block = np.asarray(channels[start:stop])
        magnitude_square = np.square(block.real) + np.square(block.imag)
        power[start:stop] = magnitude_square.mean(axis=(1, 2, 3), dtype=np.float64)
        outage[start:stop] = np.all(block == 0, axis=(1, 2, 3))
    log_power = np.log10(np.maximum(power, 1e-30)).astype(np.float32)
    mean = np.zeros(2, dtype=np.float32)
    std = np.ones(2, dtype=np.float32)
    for cell in (0, 1):
        valid = (cell_labels == cell) & ~outage
        mean[cell] = float(log_power[valid].mean())
        std[cell] = max(float(log_power[valid].std()), 0.1)
        log_power[(cell_labels == cell) & outage] = mean[cell]
    return power, outage.astype(np.float32), log_power, np.stack([mean, std])


def run_preprocessing(
    data_root: str | Path,
    output_dir: str | Path,
    resolution: float,
    link_samples: int,
    local_grid: int,
    local_radius: float,
    validation_fraction: float,
    blocks_per_cell: int,
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup = load_setup(root)
    train_positions = np.load(root / "Round2_Train_Pos.npy")
    test_positions = np.load(root / "Round2_Test_Pos.npy")
    base_stations = np.asarray(setup["X"], dtype=np.float32)
    labels, label_details = infer_two_cell_labels(train_positions, base_stations)
    train_indices, validation_indices = spatial_block_split(
        train_positions, labels, validation_fraction, blocks_per_cell, seed
    )
    power, outage, log_power, power_stats = channel_statistics(
        root / "Round2_Train_Channel.npy", labels, chunk_size
    )

    vertices = read_ply_vertices(root / "Round2_Map.ply")
    bev = build_bev(vertices, resolution)
    train_tokens = build_map_tokens(
        bev, train_positions, base_stations, link_samples, local_grid, local_radius
    )
    test_tokens = build_map_tokens(
        bev, test_positions, base_stations, link_samples, local_grid, local_radius
    )
    np.save(output / "train_map_tokens.npy", train_tokens)
    np.save(output / "test_map_tokens.npy", test_tokens)
    np.savez_compressed(
        output / "metadata.npz",
        cell_labels=labels,
        outage_labels=outage,
        power=power,
        log_power=log_power,
        power_mean=power_stats[0],
        power_std=power_stats[1],
        train_indices=train_indices,
        validation_indices=validation_indices,
        position_center=train_positions.mean(axis=0).astype(np.float32),
        position_scale=np.maximum(train_positions.std(axis=0), 1.0).astype(np.float32),
    )
    manifest = {
        "version": 1,
        "data_root": str(root),
        "setup": setup,
        "cell_label_inference": label_details,
        "training_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "outage_count": int(outage.sum()),
        "map": {
            "resolution": resolution,
            "bev_shape": list(bev.features.shape),
            "minimum_xy": bev.minimum_xy.tolist(),
            "feature_names": [
                "log_density", "max_height", "height_lt_3m", "height_3_10m",
                "height_10_25m", "height_ge_25m",
            ],
            "link_samples": link_samples,
            "local_grid": local_grid,
            "local_radius": local_radius,
            "token_count": int(train_tokens.shape[2]),
            "token_feature_dim": int(train_tokens.shape[3]),
        },
        "source_files": {
            path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in [
                root / "Round2_Setup.json", root / "Round2_Map.ply",
                root / "Round2_Train_Pos.npy", root / "Round2_Train_Channel.npy",
                root / "Round2_Test_Pos.npy",
            ]
        },
    }
    save_json(output / "manifest.json", manifest)
    return manifest
