from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .angle_delay import channel_to_shape_target
from .autoencoder_training import load_autoencoder_checkpoint
from .config import autocast_context, choose_device, save_json, worker_count
from .data import ChannelDataset, balanced_limit, load_metadata, split_indices


@torch.no_grad()
def encode_training_set(config: dict, checkpoint_path: str | Path | None = None) -> dict:
    started = time.perf_counter()
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    configured_checkpoint = checkpoint_path or config["encoding"]["autoencoder_checkpoint"]
    model, shape, checkpoint = load_autoencoder_checkpoint(config, configured_checkpoint, device)
    model.eval()
    metadata = load_metadata(config)
    sample_count = len(metadata["train_cells"])
    indices = np.arange(sample_count, dtype=np.int64)
    indices = balanced_limit(
        indices,
        config["runtime"].get("encoding_limit"),
        [metadata["train_cells"], metadata["fold_ids"]],
        int(config["seed"]) + 2,
    )
    dataset = ChannelDataset(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
        indices,
        None,
    )
    workers = worker_count(int(config["runtime"].get("workers", -1)))
    loader = DataLoader(
        dataset,
        batch_size=int(config["encoding"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    latent_dim = int(config["autoencoder"]["latent_dim"])
    latents = np.zeros((sample_count, latent_dim), dtype=np.float32)
    for batch in loader:
        sample_indices = batch["index"].numpy()
        channel = batch["channel"].to(device, non_blocking=True)
        target_shape, _, outage = channel_to_shape_target(channel, shape)
        with autocast_context(device, amp):
            values = model.encode(target_shape)
        values = values.float().cpu().numpy()
        values[outage.cpu().numpy()] = 0.0
        latents[sample_indices] = values

    training_indices, _ = split_indices(metadata, config)
    available = np.zeros(sample_count, dtype=bool)
    available[dataset.indices] = True
    if len(dataset) < sample_count:
        training_indices = training_indices[available[training_indices]]
    nonzero_training = training_indices[~metadata["outage"][training_indices]]
    if not len(nonzero_training):
        raise ValueError("No non-outage samples are available for latent statistics")
    latent_mean = latents[nonzero_training].mean(axis=0).astype(np.float32)
    latent_std = latents[nonzero_training].std(axis=0).astype(np.float32)
    latent_std = np.maximum(latent_std, float(config["encoding"].get("minimum_latent_std", 1e-3)))
    cell_count = int(metadata["train_cells"].max()) + 1
    power_mean = np.zeros(cell_count, dtype=np.float32)
    power_std = np.ones(cell_count, dtype=np.float32)
    for cell_id in range(cell_count):
        selected = nonzero_training[metadata["train_cells"][nonzero_training] == cell_id]
        power_mean[cell_id] = metadata["log_power"][selected].mean()
        power_std[cell_id] = max(
            float(metadata["log_power"][selected].std()),
            float(config["encoding"].get("minimum_power_std", 0.1)),
        )
    output_path = Path(config["encoding"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        latent=latents,
        latent_mean=latent_mean,
        latent_std=latent_std,
        power_mean=power_mean,
        power_std=power_std,
        available=available,
    )
    summary = {
        "output_path": str(output_path),
        "autoencoder_checkpoint": str(configured_checkpoint),
        "autoencoder_epoch": int(checkpoint.get("epoch", -1)),
        "encoded_samples": len(dataset),
        "statistics_samples": int(len(nonzero_training)),
        "latent_dim": latent_dim,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output_path.with_suffix(".json"), summary)
    return summary
