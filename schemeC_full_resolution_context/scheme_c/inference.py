from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .angle_delay import shape_to_channel
from .config import choose_device, save_json
from .context_data import ContextRepository
from .context_training import load_context_checkpoint, predict_indices
from .data import balanced_limit, load_metadata, split_indices


@torch.no_grad()
def generate_test_channels(
    config: dict,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    outage_threshold: float | None = None,
) -> dict:
    """Predict the requested test subset and stream decoded channels into an NPY file."""
    started = time.perf_counter()
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    training_indices, _ = split_indices(metadata, config)
    repository = ContextRepository(config, training_indices)
    selected = balanced_limit(
        np.arange(len(metadata["test_cells"]), dtype=np.int64),
        config["runtime"].get("test_limit"),
        [metadata["test_cells"]],
        int(config["seed"]) + 5,
    )
    checkpoint_path = checkpoint_path or config["inference"]["context_checkpoint"]
    model, autoencoder, shape, checkpoint = load_context_checkpoint(
        config, checkpoint_path, repository, device
    )
    outputs = predict_indices(
        model,
        repository,
        selected,
        device,
        amp,
        test=True,
    )
    threshold = float(
        config["inference"].get("outage_threshold", 0.999)
        if outage_threshold is None
        else outage_threshold
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("outage_threshold must lie in the open interval (0, 1)")
    predicted_outage = outputs["outage_probability"] >= threshold
    cells = metadata["test_cells"][selected]
    target_path = Path(output_path or config["inference"]["output_path"])
    if target_path.suffix.lower() != ".npy":
        raise ValueError("Inference output_path must end in .npy")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(selected), *shape.raw_shape),
    )
    decode_batch_size = int(config["inference"].get("decode_batch_size", 8))
    spectrum_mean = torch.from_numpy(repository.encoded["spectrum_mean"]).to(device)
    spectrum_std = torch.from_numpy(repository.encoded["spectrum_std"]).to(device)
    phase_mean = torch.from_numpy(repository.encoded["phase_mean"]).to(device)
    phase_std = torch.from_numpy(repository.encoded["phase_std"]).to(device)
    power_mean = torch.from_numpy(repository.encoded["power_mean"]).to(device)
    power_std = torch.from_numpy(repository.encoded["power_std"]).to(device)
    for start in range(0, len(selected), decode_batch_size):
        stop = min(start + decode_batch_size, len(selected))
        cell_tensor = torch.from_numpy(cells[start:stop]).to(device=device, dtype=torch.long)
        local_spectrum_mean = (
            spectrum_mean[cell_tensor] if spectrum_mean.ndim == 2 else spectrum_mean
        )
        local_spectrum_std = (
            spectrum_std[cell_tensor] if spectrum_std.ndim == 2 else spectrum_std
        )
        local_phase_mean = phase_mean[cell_tensor] if phase_mean.ndim == 2 else phase_mean
        local_phase_std = phase_std[cell_tensor] if phase_std.ndim == 2 else phase_std
        spectrum = (
            torch.from_numpy(outputs["spectrum"][start:stop]).to(device)
            * local_spectrum_std
            + local_spectrum_mean
        )
        phase = (
            torch.from_numpy(outputs["phase"][start:stop]).to(device)
            * local_phase_std
            + local_phase_mean
        )
        normalized_power = torch.from_numpy(outputs["power"][start:stop]).to(device)
        log_power = normalized_power * power_std[cell_tensor] + power_mean[cell_tensor]
        prediction_shape = autoencoder.decode(spectrum, phase)
        channel = shape_to_channel(prediction_shape, log_power, shape)
        outage_tensor = torch.from_numpy(predicted_outage[start:stop]).to(device)
        channel = channel.masked_fill(outage_tensor[:, None, None, None], 0.0)
        output[start:stop] = channel.cpu().numpy().astype(np.complex64, copy=False)
    output.flush()
    del output
    summary = {
        "output_path": str(target_path),
        "shape": [len(selected), *shape.raw_shape],
        "dtype": "complex64",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "outage_threshold": threshold,
        "predicted_outages": int(predicted_outage.sum()),
        "cell_counts": [
            int(np.sum(cells == cell_id)) for cell_id in range(repository.cell_count)
        ],
        "selected_test_indices": selected.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(target_path.with_suffix(".json"), summary)
    return summary
