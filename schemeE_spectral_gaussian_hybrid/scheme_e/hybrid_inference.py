from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .config import choose_device, save_json
from .hybrid_training import load_hybrid_checkpoint
from .reference import build_reference_candidates


@torch.no_grad()
def generate_test_channels(config: dict) -> dict[str, object]:
    started = time.perf_counter()
    section = config["inference"]
    device = choose_device(str(config["runtime"].get("device", "auto")))
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(config["spectral_teacher"]["test_output_path"]) as source:
        priors = {name: source[name] for name in source.files}
    channels = np.load(Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r")
    checkpoint_path = Path(section["checkpoint"])
    model, shape, checkpoint = load_hybrid_checkpoint(config, checkpoint_path, device)
    test_count = len(metadata["test_positions"])
    limit = int(config["runtime"].get("test_limit", 0) or 0)
    if limit:
        test_count = min(test_count, limit)
    test_indices = np.arange(test_count, dtype=np.int64)
    observed_indices = np.arange(len(metadata["train_positions"]), dtype=np.int64)
    candidates, distances = build_reference_candidates(
        metadata["test_positions"][test_indices],
        metadata["test_cells"][test_indices],
        metadata["train_positions"],
        metadata["train_cells"],
        metadata["outage"],
        top_k=1,
    )
    references = candidates[:, 0]
    geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    geometry_std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    output_path = Path(section["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(test_count, *shape.raw_shape),
    )
    batch_size = int(section.get("batch_size", 4))
    configured_threshold = section.get("outage_threshold")
    threshold = (
        float(configured_threshold)
        if configured_threshold is not None
        else float(np.asarray(priors["outage_threshold"]).item())
    )
    projection_iterations = int(section.get("projection_iterations", model.projection_iterations))
    model.eval()
    for start in range(0, test_count, batch_size):
        stop = min(start + batch_size, test_count)
        indices = test_indices[start:stop]
        reference = torch.as_tensor(np.asarray(channels[references[start:stop]]), device=device)
        geometry = np.clip(
            (metadata["test_geometry_features"][indices] - geometry_mean) / geometry_std,
            -8.0,
            8.0,
        )
        inputs = {
            "pas_log": torch.as_tensor(priors["pas_log"][indices].astype(np.float32), device=device),
            "pdp_log": torch.as_tensor(priors["pdp_log"][indices].astype(np.float32), device=device),
            "ue_log_energy": torch.as_tensor(priors["ue_log_energy"][indices], device=device),
            "log_power": torch.as_tensor(priors["log_power"][indices], device=device),
            "uncertainty": torch.as_tensor(priors["uncertainty"][indices], device=device),
            "outage_probability": torch.as_tensor(priors["outage_probability"][indices], device=device),
            "geometry": torch.as_tensor(geometry, device=device),
        }
        result = model(
            reference,
            projection_iterations=projection_iterations,
            **inputs,
        )["channel"]
        predicted_outage = inputs["outage_probability"] >= threshold
        result = result.masked_fill(predicted_outage[:, None, None, None], 0.0)
        output[start:stop] = result.cpu().numpy().astype(np.complex64)
        output.flush()
    finite = bool(np.isfinite(output).all())
    if not finite:
        raise FloatingPointError("Scheme E inference produced NaN or Inf")
    report = {
        "stage": "scheme_e_test_inference",
        "output_path": str(output_path),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "samples": int(test_count),
        "predicted_outages": int(np.sum(priors["outage_probability"][:test_count] >= threshold)),
        "outage_threshold": threshold,
        "projection_iterations": projection_iterations,
        "reference_distance_meters": {
            "minimum": float(np.min(distances[:, 0])),
            "median": float(np.median(distances[:, 0])),
            "maximum": float(np.max(distances[:, 0])),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(section["report_path"], report)
    return report
