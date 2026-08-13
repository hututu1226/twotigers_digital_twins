from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import save_json
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
    read_ascii_ply_xyz,
    spatial_fold_ids,
    test_support_summary,
    wrap_degrees,
)


def _channel_power(path: Path, sample_count: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    channels = np.load(path, mmap_mode="r")
    if len(channels) != sample_count:
        raise ValueError(f"Channel count {len(channels)} does not match positions {sample_count}")
    power = np.empty(sample_count, dtype=np.float64)
    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        block = np.asarray(channels[start:stop])
        power[start:stop] = np.mean(np.abs(block) ** 2, axis=(1, 2, 3), dtype=np.float64)
    outage = power <= 1e-30
    log_power = np.zeros(sample_count, dtype=np.float32)
    log_power[~outage] = np.log10(power[~outage]).astype(np.float32)
    return log_power, outage


def preprocess_dataset(config: dict, force: bool = False) -> dict:
    started = time.perf_counter()
    data_root = Path(config["data"]["root"])
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} already exists; pass --force only when you intend to rebuild it"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    setup = load_setup(data_root)
    train_positions = np.load(data_root / "Round2_Train_Pos.npy").astype(np.float32)
    test_positions = np.load(data_root / "Round2_Test_Pos.npy").astype(np.float32)
    if len(train_positions) != int(setup["P_Train"]):
        raise ValueError("Round2_Train_Pos.npy does not match P_Train")
    if len(test_positions) != int(setup["P_Test"]):
        raise ValueError("Round2_Test_Pos.npy does not match P_Test")

    cell_rule = infer_two_cell_rule(train_positions)
    split_axis = int(cell_rule["axis"])
    station_order = np.argsort(np.asarray(setup["X"], dtype=np.float32)[:, split_axis])
    cell_rule["lower_cell"] = int(station_order[0])
    cell_rule["upper_cell"] = int(station_order[-1])
    train_cells = assign_cells(train_positions, cell_rule)
    test_cells = assign_cells(test_positions, cell_rule)
    cell_count = int(setup["Q"])
    if set(np.unique(train_cells)) != set(range(cell_count)):
        raise ValueError("Automatic serving-cell rule did not produce every configured cell")

    preprocessing = config["preprocessing"]
    fold_ids = spatial_fold_ids(
        train_positions,
        train_cells,
        int(preprocessing["fold_count"]),
        int(config["seed"]),
    )
    log_power, outage = _channel_power(
        data_root / "Round2_Train_Channel.npy",
        len(train_positions),
        int(preprocessing.get("channel_chunk_size", 8)),
    )
    point_cloud = read_ascii_ply_xyz(data_root / "Round2_Map.ply")

    train_rows = np.empty(len(train_positions), dtype=np.int32)
    train_columns = np.empty(len(train_positions), dtype=np.int32)
    test_rows = np.empty(len(test_positions), dtype=np.int32)
    test_columns = np.empty(len(test_positions), dtype=np.int32)
    grid_entries: list[dict] = []
    boresights: list[float] = []
    for cell_id in range(cell_count):
        train_mask = train_cells == cell_id
        test_mask = test_cells == cell_id
        all_positions = np.concatenate([train_positions[train_mask], test_positions[test_mask]], axis=0)
        spec = make_grid_spec(
            all_positions,
            float(preprocessing["grid_resolution"]),
            float(preprocessing["grid_margin"]),
        )
        train_rows[train_mask], train_columns[train_mask] = spec.indices(train_positions[train_mask, :2])
        test_rows[test_mask], test_columns[test_mask] = spec.indices(test_positions[test_mask, :2])
        boresight = infer_boresight(train_positions[train_mask], np.asarray(setup["X"])[cell_id])
        boresights.append(boresight)
        geometry = build_geometry_maps(
            spec=spec,
            base_station=np.asarray(setup["X"])[cell_id],
            boresight_degrees=boresight,
            sector_half_angle_degrees=float(preprocessing["sector_half_angle_degrees"]),
            maximum_distance=float(preprocessing["maximum_distance"]),
            cell_id=cell_id,
            cell_count=cell_count,
        )
        # Grid-center quantization can move a boundary UE slightly outside the analytic sector.
        # Cells containing an assigned train/test UE are valid by construction.
        geometry["valid"][0, train_rows[train_mask], train_columns[train_mask]] = 1.0
        geometry["valid"][0, test_rows[test_mask], test_columns[test_mask]] = 1.0
        bev = build_bev_features(point_cloud, spec)
        train_valid = geometry["valid"][0, train_rows[train_mask], train_columns[train_mask]]
        test_valid = geometry["valid"][0, test_rows[test_mask], test_columns[test_mask]]
        if not np.all(train_valid) or not np.all(test_valid):
            raise ValueError(
                f"Cell {cell_id} has assigned samples outside valid_mask; increase the sector or distance limit"
            )
        static_path = artifact_dir / f"static_cell_{cell_id}.npz"
        np.savez_compressed(static_path, bev=bev, **geometry)
        delta = train_positions[train_mask, :2] - np.asarray(setup["X"], dtype=np.float32)[cell_id, :2]
        angles = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
        grid_entries.append(
            {
                "cell_id": cell_id,
                "spec": spec.to_dict(),
                "boresight_degrees": boresight,
                "observed_relative_angle_limit": float(np.max(np.abs(wrap_degrees(angles - boresight)))),
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "static_path": static_path.name,
            }
        )

    np.savez_compressed(
        artifact_dir / "metadata.npz",
        train_positions=train_positions,
        test_positions=test_positions,
        train_cells=train_cells,
        test_cells=test_cells,
        train_rows=train_rows,
        train_columns=train_columns,
        test_rows=test_rows,
        test_columns=test_columns,
        fold_ids=fold_ids,
        log_power=log_power,
        outage=outage,
    )
    collision_stats = grid_collision_statistics(train_rows, train_columns, train_cells)
    outage_by_cell = [int(np.sum(outage & (train_cells == cell_id))) for cell_id in range(cell_count)]
    manifest = {
        "version": 1,
        "setup": setup,
        "data_root": str(data_root),
        "metadata_path": "metadata.npz",
        "cell_rule": cell_rule,
        "grids": grid_entries,
        "fold_count": int(preprocessing["fold_count"]),
        "train_cell_counts": [int(np.sum(train_cells == index)) for index in range(cell_count)],
        "test_cell_counts": [int(np.sum(test_cells == index)) for index in range(cell_count)],
        "outage_count": int(outage.sum()),
        "outage_by_cell": outage_by_cell,
        "grid_collisions": collision_stats,
        "nearest_neighbor_meters": nearest_neighbor_summary(train_positions, train_cells),
        "test_support": test_support_summary(
            train_positions,
            test_positions,
            train_cells,
            test_cells,
            float(preprocessing.get("test_component_link_meters", 6.0)),
        ),
        "point_cloud_vertices": int(len(point_cloud)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(manifest_path, manifest)
    return manifest
