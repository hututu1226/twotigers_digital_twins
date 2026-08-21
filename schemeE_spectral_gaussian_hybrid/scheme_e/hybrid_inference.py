from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .carrier_transport import CarrierFit, select_transport_candidates
from .config import choose_device, save_json
from .hybrid_training import (
    _normalized_geometry,
    _prior_batch,
    _reference_context_batch,
    _transport_batch,
    load_hybrid_checkpoint,
)
from .power_safety import apply_outage_policy
from .reference import build_reference_candidates
from .reference_context import select_reference_candidates


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
    with np.load(config["spectral"]["target_path"]) as source:
        spectral_targets = {name: source[name] for name in source.files}
    channels = np.load(Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r")
    checkpoint_path = Path(section["checkpoint"])
    model, shape, checkpoint = load_hybrid_checkpoint(config, checkpoint_path, device)
    test_count = len(metadata["test_positions"])
    limit = int(config["runtime"].get("test_limit", 0) or 0)
    if limit:
        test_count = min(test_count, limit)
    test_indices = np.arange(test_count, dtype=np.int64)
    reference_strategy = section.get(
        "reference_strategy", {"name": "nearest", "top_k": 1}
    )
    if isinstance(reference_strategy, str):
        reference_strategy = {"name": reference_strategy, "top_k": 1}
    transport_config = section.get("transport_seed", config["hybrid"].get("transport_seed", {}))
    carrier_payload = checkpoint.get("carrier_fit")
    carrier_fit = None
    if carrier_payload is not None:
        carrier_fit = CarrierFit(
            np.asarray(carrier_payload["wave_numbers"], dtype=np.float64),
            np.asarray(carrier_payload["qualities"], dtype=np.float64),
            np.asarray(carrier_payload["pair_counts"], dtype=np.int64),
        )
    transport_count = int(transport_config.get("count", 8)) if carrier_fit else 1
    candidates, distances = build_reference_candidates(
        metadata["test_positions"][test_indices],
        metadata["test_cells"][test_indices],
        metadata["train_positions"],
        metadata["train_cells"],
        metadata["outage"],
        top_k=max(1, transport_count, int(reference_strategy.get("top_k", 1))),
    )
    geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    geometry_std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    if str(reference_strategy.get("name", "nearest")) == "nearest":
        references = candidates[:, 0]
    else:
        references = select_reference_candidates(
            candidates,
            distances,
            _normalized_geometry(
                metadata["test_geometry_features"],
                test_indices,
                geometry_mean,
                geometry_std,
            ),
            np.clip(
                (metadata["train_geometry_features"] - geometry_mean) / geometry_std,
                -8.0,
                8.0,
            ),
            priors["pas_log"][test_indices].astype(np.float32),
            priors["pdp_log"][test_indices].astype(np.float32),
            spectral_targets["pas_log"].astype(np.float32),
            spectral_targets["pdp_log"].astype(np.float32),
            reference_strategy,
        )
    transport_indices = None
    transport_distances = None
    if carrier_fit is not None:
        transport_indices, transport_distances = select_transport_candidates(
            candidates, distances, transport_count
        )
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
    threshold = float(configured_threshold) if configured_threshold is not None else float(
        np.asarray(priors["outage_threshold"]).reshape(-1)[0]
    )
    thresholds = np.asarray(
        section.get("outage_threshold_by_cell", [threshold]), dtype=np.float32
    ).reshape(-1)
    soft_strengths = np.asarray(
        section.get("soft_outage_strength_by_cell", [0.0]), dtype=np.float32
    ).reshape(-1)
    power_bounds = checkpoint.get("power_bounds")
    if power_bounds is not None:
        power_bounds = np.asarray(power_bounds, dtype=np.float32)
    projection_iterations = int(section.get("projection_iterations", model.projection_iterations))
    model.eval()
    for start in range(0, test_count, batch_size):
        stop = min(start + batch_size, test_count)
        indices = test_indices[start:stop]
        reference = torch.as_tensor(np.asarray(channels[references[start:stop]]), device=device)
        reference_context = None
        if model.condition_encoder.reference_dim:
            reference_context = _reference_context_batch(
                metadata,
                priors,
                spectral_targets,
                indices,
                references[start:stop],
                geometry_mean,
                geometry_std,
                target_is_test=True,
            )
        inputs = _prior_batch(
            priors,
            metadata,
            indices,
            geometry_mean,
            geometry_std,
            device,
            power_bounds=power_bounds,
            reference_context=reference_context,
            cells_key="test_cells",
            geometry_key="test_geometry_features",
        )
        transport_channel = None
        if carrier_fit is not None:
            if transport_indices is None or transport_distances is None:
                raise AssertionError("transport inference candidates are missing")
            transport_channel, transport_context = _transport_batch(
                channels,
                metadata,
                indices,
                transport_indices[start:stop],
                transport_distances[start:stop],
                carrier_fit,
                device,
                target_is_test=True,
                distance_power=float(transport_config.get("distance_power", 2.0)),
            )
            inputs["transport_context"] = transport_context
        result = model(
            reference,
            transport_channel=transport_channel,
            projection_iterations=projection_iterations,
            **inputs,
        )["channel"]
        cells = metadata["test_cells"][indices].astype(np.int64)
        threshold_batch = torch.as_tensor(
            thresholds[np.minimum(cells, len(thresholds) - 1)], device=device
        )
        strength_batch = torch.as_tensor(
            soft_strengths[np.minimum(cells, len(soft_strengths) - 1)], device=device
        )
        result = apply_outage_policy(
            result,
            inputs["outage_probability"],
            threshold_batch,
            strength_batch,
        )
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
        "predicted_outages": int(
            np.sum(
                priors["outage_probability"][:test_count]
                >= thresholds[
                    np.minimum(metadata["test_cells"][:test_count], len(thresholds) - 1)
                ]
            )
        ),
        "outage_threshold": threshold,
        "outage_threshold_by_cell": thresholds.tolist(),
        "soft_outage_strength_by_cell": soft_strengths.tolist(),
        "projection_iterations": projection_iterations,
        "reference_strategy": reference_strategy,
        "reference_distance_meters": {
            "minimum": float(np.min(distances[:, 0])),
            "median": float(np.median(distances[:, 0])),
            "maximum": float(np.max(distances[:, 0])),
        },
        "carrier_fit": None if carrier_fit is None else carrier_fit.to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(section["report_path"], report)
    return report
