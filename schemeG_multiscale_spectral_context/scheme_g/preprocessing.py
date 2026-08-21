from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import save_json
from .rf_geometry import (
    build_rf_gaussians,
    extract_geometry_features,
    feature_names,
    read_ascii_ply_mesh,
)
from .spatial_grid import (
    assign_cells,
    build_bev_features,
    build_geometry_maps,
    grid_collision_statistics,
    infer_boresight,
    infer_two_cell_rule,
    load_setup,
    make_grid_spec,
    nearest_neighbor_summary,
    test_like_validation_masks,
    test_support_summary,
    validation_support_summary,
    wrap_degrees,
)


def _geometry_features(
    data_root: Path,
    artifact_dir: Path,
    section: dict,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_cells: np.ndarray,
    test_cells: np.ndarray,
    stations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Reuse compatible Scheme E geometry, otherwise rebuild it from the mesh."""
    if not bool(section.get("rf_geometry_enabled", True)):
        vertices, _ = read_ascii_ply_mesh(data_root / "Round2_Map.ply")
        return (
            np.empty((len(train_positions), 0), dtype=np.float32),
            np.empty((len(test_positions), 0), dtype=np.float32),
            vertices.astype(np.float32),
            0,
        )
    reuse_path = section.get("reuse_geometry_metadata")
    if reuse_path:
        source_path = Path(reuse_path)
        if source_path.is_file():
            with np.load(source_path) as source:
                required = {
                    "train_positions",
                    "test_positions",
                    "train_geometry_features",
                    "test_geometry_features",
                }
                if required.issubset(source.files):
                    same_train = np.array_equal(
                        source["train_positions"], train_positions
                    )
                    same_test = np.array_equal(source["test_positions"], test_positions)
                    if same_train and same_test:
                        train = source["train_geometry_features"].astype(np.float32)
                        test = source["test_geometry_features"].astype(np.float32)
                        if train.shape[1] == len(feature_names()):
                            vertices, _ = read_ascii_ply_mesh(
                                data_root / "Round2_Map.ply"
                            )
                            return train, test, vertices.astype(np.float32), 0

    vertices, faces = read_ascii_ply_mesh(data_root / "Round2_Map.ply")
    field = build_rf_gaussians(
        vertices,
        faces,
        int(section.get("rf_gaussian_count", 10_070)),
    )
    field.save(artifact_dir / "rf_gaussians.npz")
    train = extract_geometry_features(
        train_positions,
        train_cells,
        stations,
        field,
        corridor_samples=int(section.get("rf_corridor_samples", 24)),
        fresnel_radius_meters=float(section.get("fresnel_radius_meters", 2.0)),
    )
    test = extract_geometry_features(
        test_positions,
        test_cells,
        stations,
        field,
        corridor_samples=int(section.get("rf_corridor_samples", 24)),
        fresnel_radius_meters=float(section.get("fresnel_radius_meters", 2.0)),
    )
    return train, test, vertices.astype(np.float32), len(field.centers)


def _normalize_geometry_by_cell(
    train: np.ndarray,
    test: np.ndarray,
    train_cells: np.ndarray,
    test_cells: np.ndarray,
    cell_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.empty((cell_count, train.shape[1]), dtype=np.float32)
    standard_deviation = np.empty_like(mean)
    normalized_train = np.empty_like(train, dtype=np.float32)
    normalized_test = np.empty_like(test, dtype=np.float32)
    for cell_id in range(cell_count):
        selected = train_cells == cell_id
        mean[cell_id] = train[selected].mean(axis=0, dtype=np.float64)
        standard_deviation[cell_id] = np.maximum(
            train[selected].std(axis=0, dtype=np.float64), 1e-4
        )
        normalized_train[selected] = np.clip(
            (train[selected] - mean[cell_id]) / standard_deviation[cell_id],
            -8.0,
            8.0,
        )
        test_selected = test_cells == cell_id
        normalized_test[test_selected] = np.clip(
            (test[test_selected] - mean[cell_id]) / standard_deviation[cell_id],
            -8.0,
            8.0,
        )
    return normalized_train, normalized_test, mean, standard_deviation


def _channel_power(
    path: Path, sample_count: int, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    channels = np.load(path, mmap_mode="r")
    if len(channels) != sample_count:
        raise ValueError(
            f"Channel count {len(channels)} does not match positions {sample_count}"
        )
    power = np.empty(sample_count, dtype=np.float64)
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        block = np.asarray(channels[start:stop])
        power[start:stop] = np.mean(
            np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
    outage = power <= 1e-30
    log_power = np.zeros(sample_count, dtype=np.float32)
    log_power[~outage] = np.log10(power[~outage]).astype(np.float32)
    return log_power, outage


def _cross_collision_summary(
    train_rows: np.ndarray,
    train_columns: np.ndarray,
    test_rows: np.ndarray,
    test_columns: np.ndarray,
    train_cells: np.ndarray,
    test_cells: np.ndarray,
) -> list[dict]:
    output: list[dict] = []
    for cell_id in np.unique(train_cells):
        train_keys = set(
            zip(
                train_rows[train_cells == cell_id].tolist(),
                train_columns[train_cells == cell_id].tolist(),
            )
        )
        test_pairs = list(
            zip(
                test_rows[test_cells == cell_id].tolist(),
                test_columns[test_cells == cell_id].tolist(),
            )
        )
        output.append(
            {
                "cell_id": int(cell_id),
                "test_samples": len(test_pairs),
                "test_unique_cells": len(set(test_pairs)),
                "test_samples_sharing_train_cell": int(
                    sum(pair in train_keys for pair in test_pairs)
                ),
            }
        )
    return output


def preprocess_dataset(config: dict, force: bool = False) -> dict:
    started = time.perf_counter()
    data_root = Path(config["data"]["root"])
    section = config["preprocessing"]
    artifact_dir = Path(section["artifact_dir"])
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} already exists; pass --force only when you intend to rebuild it"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    setup = load_setup(data_root)
    train_positions = np.load(data_root / "Round2_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(data_root / "Round2_Test_Pos.npy").astype(np.float32)
    if len(train_positions) != int(setup["P_Train"]) or len(test_positions) != int(
        setup["P_Test"]
    ):
        raise ValueError("Position arrays do not match Round2_Setup.json")

    cell_rule = infer_two_cell_rule(train_positions)
    split_axis = int(cell_rule["axis"])
    station_order = np.argsort(np.asarray(setup["X"], dtype=np.float32)[:, split_axis])
    cell_rule["lower_cell"] = int(station_order[0])
    cell_rule["upper_cell"] = int(station_order[-1])
    train_cells = assign_cells(train_positions, cell_rule)
    test_cells = assign_cells(test_positions, cell_rule)
    cell_rule["train_minimum_margin"] = float(
        np.min(np.abs(train_positions[:, split_axis] - cell_rule["threshold"]))
    )
    cell_rule["test_minimum_margin"] = float(
        np.min(np.abs(test_positions[:, split_axis] - cell_rule["threshold"]))
    )
    cell_rule["train_ranges"] = [
        [
            float(train_positions[train_cells == cell_id, split_axis].min()),
            float(train_positions[train_cells == cell_id, split_axis].max()),
        ]
        for cell_id in range(int(setup["Q"]))
    ]
    cell_rule["test_ranges"] = [
        [
            float(test_positions[test_cells == cell_id, split_axis].min()),
            float(test_positions[test_cells == cell_id, split_axis].max()),
        ]
        for cell_id in range(int(setup["Q"]))
    ]
    cell_count = int(setup["Q"])
    if set(np.unique(train_cells)) != set(range(cell_count)):
        raise ValueError(
            "Automatic serving-cell assignment did not produce every configured cell"
        )

    validation_masks = test_like_validation_masks(
        train_positions,
        train_cells,
        int(section["fold_count"]),
        float(section["validation_tile_meters"]),
        float(section["validation_hole_meters"]),
    )
    log_power, outage = _channel_power(
        data_root / "Round2_Train_Channel.npy",
        len(train_positions),
        int(section.get("channel_chunk_size", 8)),
    )
    stations = np.asarray(setup["X"], dtype=np.float32)
    (
        train_geometry,
        test_geometry,
        point_cloud,
        gaussian_count,
    ) = _geometry_features(
        data_root,
        artifact_dir,
        section,
        train_positions,
        test_positions,
        train_cells,
        test_cells,
        stations,
    )
    (
        train_geometry,
        test_geometry,
        geometry_mean,
        geometry_std,
    ) = _normalize_geometry_by_cell(
        train_geometry,
        test_geometry,
        train_cells,
        test_cells,
        cell_count,
    )

    context_rows = np.empty(len(train_positions), dtype=np.int32)
    context_columns = np.empty(len(train_positions), dtype=np.int32)
    context_offsets = np.empty((len(train_positions), 2), dtype=np.float32)
    test_context_rows = np.empty(len(test_positions), dtype=np.int32)
    test_context_columns = np.empty(len(test_positions), dtype=np.int32)
    test_context_offsets = np.empty((len(test_positions), 2), dtype=np.float32)
    grid_entries: list[dict] = []
    for cell_id in range(cell_count):
        train_mask = train_cells == cell_id
        test_mask = test_cells == cell_id
        all_positions = np.concatenate(
            [train_positions[train_mask], test_positions[test_mask]], axis=0
        )
        base_station = np.asarray(setup["X"], dtype=np.float32)[cell_id]
        context_spec = make_grid_spec(
            all_positions,
            float(section["context_grid_resolution"]),
            float(section["grid_margin"]),
        )
        environment_spec = make_grid_spec(
            np.concatenate([all_positions, base_station[None, :]], axis=0),
            float(section["environment_grid_resolution"]),
            float(section["grid_margin"]),
        )
        context_rows[train_mask], context_columns[train_mask] = context_spec.indices(
            train_positions[train_mask, :2]
        )
        test_context_rows[test_mask], test_context_columns[test_mask] = (
            context_spec.indices(test_positions[test_mask, :2])
        )
        context_offsets[train_mask] = context_spec.offsets(
            train_positions[train_mask, :2]
        )
        test_context_offsets[test_mask] = context_spec.offsets(
            test_positions[test_mask, :2]
        )

        boresight = infer_boresight(train_positions[train_mask], base_station)
        geometry = build_geometry_maps(
            context_spec,
            base_station,
            boresight,
            float(section["sector_half_angle_degrees"]),
            float(section["maximum_distance"]),
            cell_id,
            cell_count,
        )
        geometry["valid"][0, context_rows[train_mask], context_columns[train_mask]] = (
            1.0
        )
        geometry["valid"][
            0, test_context_rows[test_mask], test_context_columns[test_mask]
        ] = 1.0
        context_bev = build_bev_features(point_cloud, context_spec)
        environment_bev = build_bev_features(point_cloud, environment_spec)
        context_path = artifact_dir / f"context_static_cell_{cell_id}.npz"
        environment_path = artifact_dir / f"environment_static_cell_{cell_id}.npz"
        np.savez_compressed(context_path, bev=context_bev, **geometry)
        np.savez_compressed(environment_path, bev=environment_bev)

        delta = train_positions[train_mask, :2] - base_station[:2]
        angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        grid_entries.append(
            {
                "cell_id": cell_id,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "boresight_degrees": boresight,
                "observed_relative_angle_limit": float(
                    np.max(np.abs(wrap_degrees(angles - boresight)))
                ),
                "context_spec": context_spec.to_dict(),
                "environment_spec": environment_spec.to_dict(),
                "context_static_path": context_path.name,
                "environment_static_path": environment_path.name,
            }
        )

    np.savez_compressed(
        artifact_dir / "metadata.npz",
        train_positions=train_positions,
        test_positions=test_positions,
        train_cells=train_cells,
        test_cells=test_cells,
        context_rows=context_rows,
        context_columns=context_columns,
        context_offsets=context_offsets,
        test_context_rows=test_context_rows,
        test_context_columns=test_context_columns,
        test_context_offsets=test_context_offsets,
        validation_masks=validation_masks,
        log_power=log_power,
        outage=outage,
        train_geometry_features=train_geometry,
        test_geometry_features=test_geometry,
        geometry_mean=geometry_mean,
        geometry_std=geometry_std,
    )
    manifest = {
        "version": 3,
        "setup": setup,
        "data_root": str(data_root),
        "metadata_path": "metadata.npz",
        "cell_rule": cell_rule,
        "grids": grid_entries,
        "fold_count": int(section["fold_count"]),
        "train_cell_counts": [
            int(np.sum(train_cells == index)) for index in range(cell_count)
        ],
        "test_cell_counts": [
            int(np.sum(test_cells == index)) for index in range(cell_count)
        ],
        "outage_count": int(outage.sum()),
        "outage_by_cell": [
            int(np.sum(outage & (train_cells == index))) for index in range(cell_count)
        ],
        "context_grid_collisions": grid_collision_statistics(
            context_rows, context_columns, train_cells
        ),
        "context_test_collisions": _cross_collision_summary(
            context_rows,
            context_columns,
            test_context_rows,
            test_context_columns,
            train_cells,
            test_cells,
        ),
        "validation_support": validation_support_summary(
            train_positions, train_cells, validation_masks
        ),
        "nearest_neighbor_meters": nearest_neighbor_summary(
            train_positions, train_cells
        ),
        "test_support": test_support_summary(
            train_positions,
            test_positions,
            train_cells,
            test_cells,
            float(section.get("test_component_link_meters", 6.0)),
        ),
        "point_cloud_vertices": int(len(point_cloud)),
        "rf_gaussian_count": int(gaussian_count),
        "rf_geometry_channels": int(train_geometry.shape[1]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(manifest_path, manifest)
    return manifest
