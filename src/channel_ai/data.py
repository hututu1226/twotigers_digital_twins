from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_manifest(artifact_dir: str | Path) -> dict:
    with (Path(artifact_dir) / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(artifact_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(artifact_dir) / "metadata.npz") as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def _limited_indices(
    indices: np.ndarray,
    cell_labels: np.ndarray,
    outage_labels: np.ndarray,
    limit: int | None,
    seed: int,
) -> np.ndarray:
    if limit is None or limit <= 0 or limit >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cell in (0, 1):
        for outage in (0, 1):
            candidates = indices[
                (cell_labels[indices] == cell) & (outage_labels[indices].astype(int) == outage)
            ]
            if len(candidates) and len(chosen) < limit:
                chosen.append(int(rng.choice(candidates)))
    remaining = np.setdiff1d(indices, np.asarray(chosen, dtype=np.int64), assume_unique=False)
    fill = rng.choice(remaining, size=limit - len(chosen), replace=False)
    result = np.asarray(chosen + fill.tolist(), dtype=np.int64)
    rng.shuffle(result)
    return result


class ChannelTrainingDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        artifact_dir: str | Path,
        split: str,
        limit: int | None,
        seed: int,
    ) -> None:
        self.data_root = Path(data_root)
        self.artifact_dir = Path(artifact_dir)
        metadata = load_metadata(self.artifact_dir)
        key = "train_indices" if split == "train" else "validation_indices"
        self.indices = _limited_indices(
            metadata[key], metadata["cell_labels"], metadata["outage_labels"], limit, seed
        )
        self.positions = np.load(self.data_root / "Round2_Train_Pos.npy", mmap_mode="r")
        self.map_tokens = np.load(self.artifact_dir / "train_map_tokens.npy", mmap_mode="r")
        self.cell_labels = metadata["cell_labels"]
        self.outage_labels = metadata["outage_labels"]
        self.log_power = metadata["log_power"]
        self.power_mean = metadata["power_mean"]
        self.power_std = metadata["power_std"]
        self._channels = None

    def _channel_array(self) -> np.ndarray:
        if self._channels is None:
            self._channels = np.load(
                self.data_root / "Round2_Train_Channel.npy", mmap_mode="r"
            )
        return self._channels

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_channels"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        cell = int(self.cell_labels[index])
        power_z = (float(self.log_power[index]) - float(self.power_mean[cell])) / float(
            self.power_std[cell]
        )
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "position": torch.from_numpy(np.array(self.positions[index], dtype=np.float32, copy=True)),
            "map_tokens": torch.from_numpy(np.array(self.map_tokens[index], dtype=np.float32, copy=True)),
            "channel": torch.from_numpy(np.array(self._channel_array()[index], copy=True)),
            "cell": torch.tensor(cell, dtype=torch.long),
            "outage": torch.tensor(float(self.outage_labels[index]), dtype=torch.float32),
            "log_power": torch.tensor(float(self.log_power[index]), dtype=torch.float32),
            "power_z": torch.tensor(power_z, dtype=torch.float32),
        }


class ChannelInferenceDataset(Dataset):
    def __init__(self, data_root: str | Path, artifact_dir: str | Path, limit: int | None = None) -> None:
        self.positions = np.load(Path(data_root) / "Round2_Test_Pos.npy", mmap_mode="r")
        self.map_tokens = np.load(Path(artifact_dir) / "test_map_tokens.npy", mmap_mode="r")
        self.length = len(self.positions) if not limit or limit <= 0 else min(limit, len(self.positions))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        return {
            "index": torch.tensor(item, dtype=torch.long),
            "position": torch.from_numpy(np.array(self.positions[item], dtype=np.float32, copy=True)),
            "map_tokens": torch.from_numpy(np.array(self.map_tokens[item], dtype=np.float32, copy=True)),
        }


def training_loaders(config: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
    data = config["data"]
    seed = int(config.get("seed", 2026))
    train_dataset = ChannelTrainingDataset(
        data["root"], data["artifacts"], "train", data.get("train_limit"), seed
    )
    validation_dataset = ChannelTrainingDataset(
        data["root"], data["artifacts"], "validation", data.get("validation_limit"), seed + 1
    )
    common = {
        "batch_size": int(data["batch_size"]),
        "num_workers": int(data.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, validation_loader

