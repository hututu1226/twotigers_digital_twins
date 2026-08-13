from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .angle_delay import shape_to_channel
from .config import choose_device, save_json
from .data import SpatialRepository, balanced_limit, load_metadata, split_indices
from .spatial_training import load_spatial_checkpoint, predict_grid_points


@torch.no_grad()
def generate_test_channels(
    config: dict,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    outage_threshold: float | None = None,
) -> dict:
    started = time.perf_counter()
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    training_indices, _ = split_indices(metadata, config)
    repository = SpatialRepository(config, training_indices)
    checkpoint_path = checkpoint_path or config["inference"]["spatial_checkpoint"]
    model, autoencoder, shape, checkpoint = load_spatial_checkpoint(
        config, checkpoint_path, repository, device
    )
    limit = config["runtime"].get("test_limit")
    test_count = len(metadata["test_cells"])
    selected = balanced_limit(
        np.arange(test_count, dtype=np.int64),
        limit,
        [metadata["test_cells"]],
        int(config["seed"]) + 3,
    )
    selected_count = len(selected)
    cells = metadata["test_cells"][selected]
    latent_z, power_z, outage_probability = predict_grid_points(
        model,
        repository,
        cells,
        metadata["test_rows"][selected],
        metadata["test_columns"][selected],
        device,
        amp,
    )
    outage_threshold = float(
        config["inference"].get("outage_threshold", 0.5)
        if outage_threshold is None
        else outage_threshold
    )
    if not 0.0 < outage_threshold < 1.0:
        raise ValueError("outage_threshold must lie in the open interval (0, 1)")
    predicted_outage = outage_probability >= outage_threshold
    output = np.zeros((selected_count, *shape.raw_shape), dtype=np.complex64)
    latent_mean = torch.from_numpy(repository.encoded["latent_mean"]).to(device)
    latent_std = torch.from_numpy(repository.encoded["latent_std"]).to(device)
    power_mean = torch.from_numpy(repository.encoded["power_mean"]).to(device)
    power_std = torch.from_numpy(repository.encoded["power_std"]).to(device)
    decode_batch_size = int(config["inference"].get("decode_batch_size", 8))
    for start in range(0, selected_count, decode_batch_size):
        stop = min(start + decode_batch_size, selected_count)
        latent = torch.from_numpy(latent_z[start:stop]).to(device) * latent_std + latent_mean
        cell_tensor = torch.from_numpy(cells[start:stop]).to(device)
        normalized_power = torch.from_numpy(power_z[start:stop]).to(device)
        log_power = normalized_power * power_std[cell_tensor] + power_mean[cell_tensor]
        prediction_shape = autoencoder.decode(latent)
        channel = shape_to_channel(prediction_shape, log_power, shape)
        outage_tensor = torch.from_numpy(predicted_outage[start:stop]).to(device)
        channel = channel.masked_fill(outage_tensor[:, None, None, None], 0.0)
        output[start:stop] = channel.cpu().numpy().astype(np.complex64)
    target_path = Path(output_path or config["inference"]["output_path"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, output)
    summary = {
        "output_path": str(target_path),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "outage_threshold": outage_threshold,
        "predicted_outages": int(predicted_outage.sum()),
        "cell_counts": [int(np.sum(cells == cell_id)) for cell_id in range(repository.cell_count)],
        "selected_test_indices": selected.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(target_path.with_suffix(".json"), summary)
    return summary
