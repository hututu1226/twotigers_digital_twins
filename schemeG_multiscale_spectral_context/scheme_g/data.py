from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(config: dict) -> dict:
    path = Path(config["preprocessing"]["artifact_dir"]) / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(config: dict) -> dict[str, np.ndarray]:
    path = Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz"
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def split_indices(
    metadata: dict[str, np.ndarray], config: dict
) -> tuple[np.ndarray, np.ndarray]:
    fold = config["split"].get("validation_fold")
    indices = np.arange(len(metadata["train_cells"]), dtype=np.int64)
    if fold is None:
        return indices, np.empty(0, dtype=np.int64)
    masks = metadata["validation_masks"]
    if not 0 <= int(fold) < len(masks):
        raise ValueError(f"validation_fold={fold} is outside [0, {len(masks)})")
    validation_mask = masks[int(fold)].astype(bool)
    training = indices[~validation_mask]
    validation = indices[validation_mask]
    if not len(training) or not len(validation):
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
            if cursor < len(groups[key]):
                output.append(groups[key][cursor])
                added = True
                if len(output) == int(limit):
                    break
        if not added:
            break
        cursor += 1
    return np.asarray(sorted(output), dtype=np.int64)


class ChannelDataset(Dataset):
    def __init__(self, channel_path: str | Path, indices: np.ndarray) -> None:
        self.channels = np.load(channel_path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "channel": torch.from_numpy(np.array(self.channels[index], copy=True)),
        }
