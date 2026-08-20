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
def encode_training_set(
    config: dict, checkpoint_path: str | Path | None = None
) -> dict:
    started = time.perf_counter()
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    configured_checkpoint = (
        checkpoint_path or config["encoding"]["autoencoder_checkpoint"]
    )
    model, shape, checkpoint = load_autoencoder_checkpoint(
        config, configured_checkpoint, device
    )
    metadata = load_metadata(config)
    sample_count = len(metadata["train_cells"])
    indices = balanced_limit(
        np.arange(sample_count, dtype=np.int64),
        config["runtime"].get("encoding_limit"),
        [metadata["train_cells"]],
        int(config["seed"]) + 2,
    )
    dataset = ChannelDataset(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", indices
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
    spectrum = np.zeros((sample_count, model.spectrum_latent_dim), dtype=np.float32)
    phase = np.zeros((sample_count, model.phase_latent_dim), dtype=np.float32)
    for batch in loader:
        sample_indices = batch["index"].numpy()
        channel = batch["channel"].to(device, non_blocking=True)
        target_shape, _, outage = channel_to_shape_target(channel, shape)
        with autocast_context(device, amp):
            spectrum_values, phase_values = model.encode(target_shape)
        spectrum_values = spectrum_values.float().cpu().numpy()
        phase_values = phase_values.float().cpu().numpy()
        outage_values = outage.cpu().numpy()
        spectrum_values[outage_values] = 0.0
        phase_values[outage_values] = 0.0
        spectrum[sample_indices] = spectrum_values
        phase[sample_indices] = phase_values

    training_indices, _ = split_indices(metadata, config)
    available = np.zeros(sample_count, dtype=bool)
    available[dataset.indices] = True
    training_indices = training_indices[available[training_indices]]
    nonzero_training = training_indices[~metadata["outage"][training_indices]]
    if not len(nonzero_training):
        raise ValueError("No non-outage samples are available for latent statistics")
    minimum_std = float(config["encoding"].get("minimum_latent_std", 1e-3))
    cell_count = int(metadata["train_cells"].max()) + 1
    spectrum_mean = np.zeros((cell_count, spectrum.shape[1]), dtype=np.float32)
    spectrum_std = np.ones((cell_count, spectrum.shape[1]), dtype=np.float32)
    phase_mean = np.zeros((cell_count, phase.shape[1]), dtype=np.float32)
    phase_std = np.ones((cell_count, phase.shape[1]), dtype=np.float32)
    power_mean = np.zeros(cell_count, dtype=np.float32)
    power_std = np.ones(cell_count, dtype=np.float32)
    for cell_id in range(cell_count):
        selected = nonzero_training[
            metadata["train_cells"][nonzero_training] == cell_id
        ]
        if not len(selected):
            raise ValueError(
                f"Cell {cell_id} has no non-outage samples for latent statistics"
            )
        spectrum_mean[cell_id] = spectrum[selected].mean(axis=0)
        spectrum_std[cell_id] = np.maximum(spectrum[selected].std(axis=0), minimum_std)
        phase_mean[cell_id] = phase[selected].mean(axis=0)
        phase_std[cell_id] = np.maximum(phase[selected].std(axis=0), minimum_std)
        power_mean[cell_id] = metadata["log_power"][selected].mean()
        power_std[cell_id] = max(
            float(metadata["log_power"][selected].std()),
            float(config["encoding"].get("minimum_power_std", 0.1)),
        )

    storage_dtype = (
        np.float16
        if config["encoding"].get("storage_dtype") == "float16"
        else np.float32
    )
    output_path = Path(config["encoding"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        spectrum_latent=spectrum.astype(storage_dtype),
        phase_latent=phase.astype(storage_dtype),
        spectrum_shape=np.asarray(model.spectrum_shape.tensor_shape, dtype=np.int16),
        phase_shape=np.asarray(model.phase_shape.tensor_shape, dtype=np.int16),
        spectrum_mean=spectrum_mean,
        spectrum_std=spectrum_std,
        phase_mean=phase_mean,
        phase_std=phase_std,
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
        "spectrum_latent_dim": model.spectrum_latent_dim,
        "phase_latent_dim": model.phase_latent_dim,
        "spectrum_shape": list(model.spectrum_shape.tensor_shape),
        "phase_shape": list(model.phase_shape.tensor_shape),
        "total_latent_dim": model.total_latent_dim,
        "latent_statistics": "per_cell",
        "storage_dtype": np.dtype(storage_dtype).name,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output_path.with_suffix(".json"), summary)
    return summary
