from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .spatial_grid import GridSpec


def load_manifest(config: dict) -> dict:
    path = Path(config["preprocessing"]["artifact_dir"]) / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(config: dict) -> dict[str, np.ndarray]:
    path = Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz"
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def split_indices(metadata: dict[str, np.ndarray], config: dict) -> tuple[np.ndarray, np.ndarray]:
    fold = config["split"].get("validation_fold")
    all_indices = np.arange(len(metadata["fold_ids"]), dtype=np.int64)
    if fold is None:
        return all_indices, np.empty(0, dtype=np.int64)
    validation = all_indices[metadata["fold_ids"] == int(fold)]
    training = all_indices[metadata["fold_ids"] != int(fold)]
    if not len(validation) or not len(training):
        raise ValueError(f"validation_fold={fold} produced an empty split")
    return training, validation


def balanced_limit(
    indices: np.ndarray,
    limit: int | None,
    group_arrays: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if not limit or limit <= 0 or len(selected) <= int(limit):
        return selected
    groups: dict[tuple[int, ...], list[int]] = {}
    for index in selected:
        key = tuple(int(values[index]) for values in group_arrays)
        groups.setdefault(key, []).append(int(index))
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    output: list[int] = []
    keys = sorted(groups)
    cursor = 0
    while len(output) < int(limit):
        added = False
        for key in keys:
            values = groups[key]
            if cursor < len(values):
                output.append(values[cursor])
                added = True
                if len(output) == int(limit):
                    break
        if not added:
            break
        cursor += 1
    return np.asarray(sorted(output), dtype=np.int64)


class ChannelDataset(Dataset):
    def __init__(
        self,
        channel_path: str | Path,
        indices: np.ndarray,
        limit: int | None = None,
    ) -> None:
        self.channels = np.load(channel_path, mmap_mode="r")
        selected = np.asarray(indices, dtype=np.int64)
        self.indices = selected if not limit or limit <= 0 else selected[: int(limit)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        channel = torch.from_numpy(np.array(self.channels[index], copy=True))
        return {"index": torch.tensor(index, dtype=torch.long), "channel": channel}


class SpatialRepository:
    """Materialize per-cell observed maps from encoded training samples."""

    def __init__(self, config: dict, observed_indices: np.ndarray) -> None:
        self.config = config
        self.manifest = load_manifest(config)
        self.metadata = load_metadata(config)
        with np.load(config["encoding"]["output_path"]) as source:
            self.encoded = {key: source[key] for key in source.files}
        self.observed_indices = np.asarray(observed_indices, dtype=np.int64)
        if "available" in self.encoded:
            self.observed_indices = self.observed_indices[
                self.encoded["available"][self.observed_indices]
            ]
        self.latent_dim = int(self.encoded["latent"].shape[1])
        self.cell_count = int(self.manifest["setup"]["Q"])
        self.specs = [GridSpec.from_dict(entry["spec"]) for entry in self.manifest["grids"]]
        self.cells: list[dict[str, np.ndarray]] = []
        self._materialize()

    @property
    def input_channels(self) -> int:
        return self.latent_dim + 13

    def normalized_latent(self, indices: np.ndarray) -> np.ndarray:
        values = self.encoded["latent"][indices]
        return (values - self.encoded["latent_mean"]) / self.encoded["latent_std"]

    def normalized_power(self, indices: np.ndarray) -> np.ndarray:
        cells = self.metadata["train_cells"][indices]
        mean = self.encoded["power_mean"][cells]
        standard_deviation = self.encoded["power_std"][cells]
        values = (self.metadata["log_power"][indices] - mean) / standard_deviation
        values = values.astype(np.float32)
        values[self.metadata["outage"][indices]] = 0.0
        return values

    def _materialize(self) -> None:
        latent_z = self.normalized_latent(np.arange(len(self.metadata["train_cells"])))
        latent_z[self.metadata["outage"]] = 0.0
        power_z = self.normalized_power(np.arange(len(self.metadata["train_cells"])))
        for cell_id, (spec, grid_entry) in enumerate(zip(self.specs, self.manifest["grids"])):
            static_path = Path(grid_entry["static_path"])
            if not static_path.is_absolute():
                static_path = Path(self.config["preprocessing"]["artifact_dir"]) / static_path
            with np.load(static_path) as source:
                static = {key: source[key].astype(np.float32) for key in source.files}
            size = spec.height * spec.width
            latent_sum = np.zeros((self.latent_dim, size), dtype=np.float32)
            power_sum = np.zeros(size, dtype=np.float32)
            count = np.zeros(size, dtype=np.float32)
            mask = self.metadata["train_cells"][self.observed_indices] == cell_id
            indices = self.observed_indices[mask]
            flat = (
                self.metadata["train_rows"][indices].astype(np.int64) * spec.width
                + self.metadata["train_columns"][indices].astype(np.int64)
            )
            if len(indices):
                np.add.at(count, flat, 1.0)
                np.add.at(power_sum, flat, power_z[indices])
                for dimension in range(self.latent_dim):
                    np.add.at(latent_sum[dimension], flat, latent_z[indices, dimension])
            occupied = count > 0
            latent_sum[:, occupied] /= count[occupied]
            power_sum[occupied] /= count[occupied]
            latent_map = latent_sum.reshape(self.latent_dim, spec.height, spec.width)
            power_map = power_sum.reshape(1, spec.height, spec.width)
            observed_map = occupied.reshape(1, spec.height, spec.width).astype(np.float32)
            fixed = np.concatenate(
                [
                    static["bev"],
                    static["valid"],
                    static["distance"],
                    static["relative_angle"],
                    static["identity"],
                ],
                axis=0,
            )
            self.cells.append(
                {
                    "latent": latent_map,
                    "power": power_map,
                    "observed": observed_map,
                    "fixed": fixed,
                    "indices": indices,
                }
            )

    def full_input(self, cell_id: int, hidden_rows: np.ndarray, hidden_columns: np.ndarray) -> np.ndarray:
        cell = self.cells[cell_id]
        latent = cell["latent"].copy()
        power = cell["power"].copy()
        observed = cell["observed"].copy()
        if len(hidden_rows):
            latent[:, hidden_rows, hidden_columns] = 0.0
            power[:, hidden_rows, hidden_columns] = 0.0
            observed[:, hidden_rows, hidden_columns] = 0.0
        return np.concatenate([latent, power, cell["fixed"][:6], observed, cell["fixed"][6:]], axis=0)


class DynamicHoleDataset(Dataset):
    def __init__(self, config: dict, repository: SpatialRepository, training_indices: np.ndarray) -> None:
        self.config = config
        self.repository = repository
        self.metadata = repository.metadata
        self.training_indices = np.asarray(training_indices, dtype=np.int64)
        self.channels = np.load(Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r")
        spatial = config["spatial"]
        self.crop_size = int(spatial["crop_size"])
        self.hole_min = float(spatial["hole_min_meters"])
        self.hole_max = float(spatial["hole_max_meters"])
        self.minimum_targets = int(spatial.get("minimum_targets", 2))
        self.maximum_targets = int(spatial.get("maximum_targets", 64))
        self.crops_per_epoch = int(spatial["crops_per_epoch"])
        self.seed = int(config["seed"])
        self.epoch = 0
        self.indices_by_cell = []
        for cell_id in range(repository.cell_count):
            candidates = self.training_indices[
                self.metadata["train_cells"][self.training_indices] == cell_id
            ]
            rows = self.metadata["train_rows"][candidates]
            columns = self.metadata["train_columns"][candidates]
            valid = repository.cells[cell_id]["fixed"][6, rows, columns] > 0.5
            candidates = candidates[valid]
            if not len(candidates):
                raise ValueError(f"Cell {cell_id} has no valid training candidates")
            self.indices_by_cell.append(candidates)
        for spec in repository.specs:
            if spec.height < self.crop_size or spec.width < self.crop_size:
                raise ValueError(
                    f"crop_size={self.crop_size} exceeds grid {spec.height}x{spec.width}; "
                    "lower crop_size or grid_resolution"
                )
            maximum_hole_cells = int(np.ceil(self.hole_max / spec.resolution))
            if maximum_hole_cells > self.crop_size:
                raise ValueError(
                    f"hole_max_meters={self.hole_max} exceeds crop_size={self.crop_size} "
                    f"at resolution={spec.resolution}"
                )
        if self.minimum_targets < 1 or self.maximum_targets < self.minimum_targets:
            raise ValueError("Target limits must satisfy 1 <= minimum_targets <= maximum_targets")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.crops_per_epoch

    def _sample_hole(self, rng: np.random.Generator, cell_id: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        spec = self.repository.specs[cell_id]
        candidates = self.indices_by_cell[cell_id]
        rows = self.metadata["train_rows"][candidates]
        columns = self.metadata["train_columns"][candidates]
        selected = np.empty(0, dtype=np.int64)
        bounds = (0, 1, 0, 1)
        for _ in range(64):
            anchor = int(candidates[rng.integers(len(candidates))])
            center_row = int(self.metadata["train_rows"][anchor])
            center_column = int(self.metadata["train_columns"][anchor])
            hole_height = max(1, int(round(rng.uniform(self.hole_min, self.hole_max) / spec.resolution)))
            hole_width = max(1, int(round(rng.uniform(self.hole_min, self.hole_max) / spec.resolution)))
            row_start = max(0, center_row - hole_height // 2)
            column_start = max(0, center_column - hole_width // 2)
            row_stop = min(spec.height, row_start + hole_height)
            column_stop = min(spec.width, column_start + hole_width)
            in_hole = (
                (rows >= row_start)
                & (rows < row_stop)
                & (columns >= column_start)
                & (columns < column_stop)
            )
            selected = candidates[in_hole]
            bounds = row_start, row_stop, column_start, column_stop
            if len(selected) >= self.minimum_targets:
                break
        if not len(selected):
            selected = np.asarray([anchor], dtype=np.int64)
        if len(selected) > self.maximum_targets:
            selected = rng.choice(selected, size=self.maximum_targets, replace=False)
        return np.sort(selected), bounds

    def __getitem__(self, item: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(item) * 9_973)
        cell_id = int(rng.integers(self.repository.cell_count))
        targets, (hole_row_start, hole_row_stop, hole_column_start, hole_column_stop) = self._sample_hole(
            rng, cell_id
        )
        spec = self.repository.specs[cell_id]
        center_row = (hole_row_start + hole_row_stop) // 2
        center_column = (hole_column_start + hole_column_stop) // 2
        crop_row = int(np.clip(center_row - self.crop_size // 2, 0, spec.height - self.crop_size))
        crop_column = int(np.clip(center_column - self.crop_size // 2, 0, spec.width - self.crop_size))
        row_slice = slice(crop_row, crop_row + self.crop_size)
        column_slice = slice(crop_column, crop_column + self.crop_size)
        cell = self.repository.cells[cell_id]
        latent = cell["latent"][:, row_slice, column_slice].copy()
        power = cell["power"][:, row_slice, column_slice].copy()
        observed = cell["observed"][:, row_slice, column_slice].copy()
        local_hole_rows = slice(hole_row_start - crop_row, hole_row_stop - crop_row)
        local_hole_columns = slice(hole_column_start - crop_column, hole_column_stop - crop_column)
        latent[:, local_hole_rows, local_hole_columns] = 0.0
        power[:, local_hole_rows, local_hole_columns] = 0.0
        observed[:, local_hole_rows, local_hole_columns] = 0.0
        model_input = np.concatenate(
            [
                latent,
                power,
                cell["fixed"][:6, row_slice, column_slice],
                observed,
                cell["fixed"][6:, row_slice, column_slice],
            ],
            axis=0,
        )
        local_rows = self.metadata["train_rows"][targets] - crop_row
        local_columns = self.metadata["train_columns"][targets] - crop_column
        return {
            "input": torch.from_numpy(model_input),
            "cell_id": torch.tensor(cell_id, dtype=torch.long),
            "target_indices": torch.from_numpy(targets.copy()),
            "target_rows": torch.from_numpy(local_rows.astype(np.int64)),
            "target_columns": torch.from_numpy(local_columns.astype(np.int64)),
            "target_latent": torch.from_numpy(self.repository.normalized_latent(targets).astype(np.float32)),
            "target_power": torch.from_numpy(self.repository.normalized_power(targets)),
            "target_outage": torch.from_numpy(self.metadata["outage"][targets].astype(np.float32)),
            "target_channel": torch.from_numpy(np.array(self.channels[targets], copy=True)),
        }


def collate_dynamic_holes(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {
        "input": torch.stack([item["input"] for item in items]),
        "cell_id": torch.stack([item["cell_id"] for item in items]),
    }
    counts = [len(item["target_indices"]) for item in items]
    output["target_batch"] = torch.repeat_interleave(
        torch.arange(len(items), dtype=torch.long), torch.tensor(counts, dtype=torch.long)
    )
    for key in (
        "target_indices",
        "target_rows",
        "target_columns",
        "target_latent",
        "target_power",
        "target_outage",
        "target_channel",
    ):
        output[key] = torch.cat([item[key] for item in items], dim=0)
    return output
