from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as functional

from .angle_delay import channel_to_shape_target
from .autoencoder import FactorizedResidualAutoencoder
from .autoencoder_training import load_autoencoder_checkpoint
from .carrier_transport import (
    TRANSPORT_CONTEXT_DIM,
    CarrierFit,
    build_transport_seed,
    fit_carrier_transport,
    select_transport_candidates,
)
from .config import (
    autocast_context,
    choose_device,
    count_parameters,
    make_grad_scaler,
    save_json,
    seed_everything,
)
from .hybrid_model import SpectralGaussianHybrid
from .losses import metric_aligned_channel_losses, weighted_sum
from .metrics import ChannelMetricAccumulator
from .power_safety import apply_outage_policy, compute_power_bounds
from .projection import relaxed_output_projection
from .reference import build_reference_candidates, sample_references
from .reference_context import (
    REFERENCE_CONTEXT_DIM,
    build_reference_context,
    sample_test_matched_references,
    select_reference_candidates,
)


def _load_repository(
    config: dict, prior_path: str | Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(prior_path) as source:
        priors = {name: source[name] for name in source.files}
    with np.load(config["spectral"]["target_path"]) as source:
        spectral_targets = {name: source[name] for name in source.files}
    return metadata, priors, spectral_targets


def _balanced_limit(indices: np.ndarray, metadata: dict[str, np.ndarray], limit: int, seed: int) -> np.ndarray:
    if not limit or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(int(seed))
    groups: dict[int, list[int]] = {}
    for index in indices:
        groups.setdefault(int(metadata["train_cells"][index]), []).append(int(index))
    selected: list[int] = []
    while len(selected) < limit:
        progressed = False
        for cell in sorted(groups):
            values = groups[cell]
            if values:
                selected.append(values.pop(int(rng.integers(0, len(values)))))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _validation_mask(
    metadata: dict[str, np.ndarray],
    available_count: int,
    validation_fold: object,
    final: bool,
) -> np.ndarray:
    if final:
        return np.zeros(available_count, dtype=bool)
    if validation_fold is None:
        raise ValueError("validation_fold cannot be null during Fold training")
    fold = int(validation_fold)
    return metadata["validation_masks"][fold, :available_count].astype(bool)


def _build_model(
    config: dict,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
    section_override: dict | None = None,
) -> tuple[SpectralGaussianHybrid, object]:
    section = {**config["hybrid"], **(section_override or {})}
    autoencoder, shape, ae_checkpoint = load_autoencoder_checkpoint(
        config, section["autoencoder_checkpoint"], device
    )
    if not isinstance(autoencoder, FactorizedResidualAutoencoder):
        raise TypeError("Scheme E requires the factorized_residual_v4 autoencoder")
    model = SpectralGaussianHybrid(
        autoencoder,
        shape,
        proxy_count=int(config["spectral"].get("proxy_count", 24)),
        geometry_dim=71,
        condition_width=int(section.get("condition_width", 192)),
        spectrum_blocks=int(section.get("spectrum_blocks", 4)),
        detail_blocks=int(section.get("detail_blocks", 6)),
        maximum_spectrum_residual=float(section.get("maximum_spectrum_residual", 1.0)),
        maximum_detail_residual=float(section.get("maximum_detail_residual", 1.0)),
        projection_iterations=int(section.get("projection_iterations", 4)),
        projection_minimum_scale=float(section.get("projection_minimum_scale", 0.25)),
        projection_maximum_scale=float(section.get("projection_maximum_scale", 4.0)),
        train_decoder=bool(section.get("train_decoder", False)),
        reference_dim=(
            REFERENCE_CONTEXT_DIM if bool(section.get("reference_aware", False)) else 0
        ),
        transport_dim=(
            TRANSPORT_CONTEXT_DIM
            if bool(section.get("transport_seed", {}).get("enabled", False))
            else 0
        ),
        station_count=(2 if bool(section.get("station_embedding", False)) else 0),
        maximum_power_delta=float(section.get("maximum_power_delta", 0.5)),
        preserve_spectral_positions=bool(
            section.get("preserve_spectral_positions", False)
        ),
        structured_spectral_field=bool(
            section.get("structured_spectral_field", False)
        ),
    ).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint["model"]
        if bool(section.get("allow_partial_initial_checkpoint", False)):
            current = model.state_dict()
            compatible = {
                name: value
                for name, value in state.items()
                if name in current and current[name].shape == value.shape
            }
            if not compatible:
                raise RuntimeError("partial initial checkpoint has no compatible weights")
            model.load_state_dict(compatible, strict=False)
            print(
                "Loaded partial initial checkpoint: "
                f"{len(compatible)}/{len(current)} tensors from {checkpoint_path}"
            )
        else:
            model.load_state_dict(state)
    else:
        checkpoint = {"autoencoder_checkpoint": ae_checkpoint.get("epoch")}
    return model, shape


def _geometry_stats(features: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features[indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(features[indices].std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
    return mean, std


def _prior_batch(
    priors: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    indices: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    device: torch.device,
    power_bounds: np.ndarray | None = None,
    reference_context: np.ndarray | None = None,
    cells_key: str = "train_cells",
    geometry_key: str = "train_geometry_features",
) -> dict[str, torch.Tensor]:
    cells = metadata[cells_key][indices].astype(np.int64)
    output = {
        "pas_log": torch.as_tensor(priors["pas_log"][indices].astype(np.float32), device=device),
        "pdp_log": torch.as_tensor(priors["pdp_log"][indices].astype(np.float32), device=device),
        "ue_log_energy": torch.as_tensor(priors["ue_log_energy"][indices], device=device),
        "log_power": torch.as_tensor(priors["log_power"][indices], device=device),
        "uncertainty": torch.as_tensor(priors["uncertainty"][indices], device=device),
        "outage_probability": torch.as_tensor(priors["outage_probability"][indices], device=device),
        "geometry": torch.as_tensor(
            np.clip(
                (metadata[geometry_key][indices] - geometry_mean) / geometry_std,
                -8.0,
                8.0,
            ),
            device=device,
        ),
        "cell_ids": torch.as_tensor(cells, device=device),
    }
    if power_bounds is not None:
        selected_bounds = np.asarray(power_bounds, dtype=np.float32)[cells]
        output["power_lower"] = torch.as_tensor(selected_bounds[:, 0], device=device)
        output["power_upper"] = torch.as_tensor(selected_bounds[:, 1], device=device)
    if reference_context is not None:
        output["reference_context"] = torch.as_tensor(reference_context, device=device)
    return output


def _normalized_geometry(
    features: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return np.clip((features[indices] - mean) / std, -8.0, 8.0).astype(np.float32)


def _station_positions(metadata: dict[str, np.ndarray]) -> np.ndarray:
    cells = metadata["train_cells"].astype(np.int64)
    geometry = metadata["train_geometry_features"]
    output = []
    for cell in range(int(cells.max()) + 1):
        rows = geometry[cells == cell, 3:6]
        if not len(rows):
            raise RuntimeError(f"Cell {cell} has no geometry rows")
        output.append(np.median(rows, axis=0))
    return np.asarray(output, dtype=np.float32)


def _load_or_fit_transport(
    section: dict,
    metadata: dict[str, np.ndarray],
    channels: np.ndarray,
    observed_indices: np.ndarray,
    seed: int,
) -> CarrierFit | None:
    transport = section.get("transport_seed", {})
    if not bool(transport.get("enabled", False)):
        return None
    fit_path = Path(transport["fit_path"])
    if fit_path.is_file():
        payload = json.loads(fit_path.read_text(encoding="utf-8"))
        return CarrierFit(
            np.asarray(payload["wave_numbers"], dtype=np.float64),
            np.asarray(payload["qualities"], dtype=np.float64),
            np.asarray(payload["pair_counts"], dtype=np.int64),
        )
    fit = fit_carrier_transport(
        metadata["train_positions"],
        metadata["train_cells"],
        metadata["outage"],
        channels,
        observed_indices,
        _station_positions(metadata),
        seed=int(transport.get("fit_seed", seed)) + 811,
        maximum_targets_per_cell=int(transport.get("fit_targets_per_cell", 256)),
        neighbors=int(transport.get("fit_neighbors", 4)),
        prior_wave_number=float(transport.get("prior_wave_number", -140.33)),
        search_radius=float(transport.get("search_radius", 12.0)),
    )
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(fit_path, fit.to_dict())
    return fit


def _transport_batch(
    channels: np.ndarray,
    metadata: dict[str, np.ndarray],
    target_indices: np.ndarray,
    transport_indices: np.ndarray,
    transport_distances: np.ndarray,
    carrier_fit: CarrierFit,
    device: torch.device,
    *,
    target_is_test: bool = False,
    distance_power: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_key = "test_positions" if target_is_test else "train_positions"
    cells_key = "test_cells" if target_is_test else "train_cells"
    references = torch.as_tensor(
        np.asarray(channels[transport_indices], dtype=np.complex64), device=device
    )
    seed, context = build_transport_seed(
        references,
        torch.as_tensor(metadata[position_key][target_indices], device=device),
        torch.as_tensor(metadata["train_positions"][transport_indices], device=device),
        torch.as_tensor(metadata[cells_key][target_indices], device=device),
        torch.as_tensor(transport_distances, device=device),
        torch.as_tensor(_station_positions(metadata), device=device),
        torch.as_tensor(carrier_fit.wave_numbers, dtype=torch.float32, device=device),
        torch.as_tensor(carrier_fit.qualities, dtype=torch.float32, device=device),
        distance_power=float(distance_power),
    )
    return seed, context


def _reference_context_batch(
    metadata: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    spectral_targets: dict[str, np.ndarray],
    target_indices: np.ndarray,
    reference_indices: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    target_is_test: bool = False,
) -> np.ndarray:
    position_key = "test_positions" if target_is_test else "train_positions"
    geometry_key = "test_geometry_features" if target_is_test else "train_geometry_features"
    target_geometry = _normalized_geometry(
        metadata[geometry_key], target_indices, geometry_mean, geometry_std
    )
    reference_geometry = _normalized_geometry(
        metadata["train_geometry_features"], reference_indices, geometry_mean, geometry_std
    )
    return build_reference_context(
        metadata[position_key][target_indices],
        metadata["train_positions"][reference_indices],
        target_geometry,
        reference_geometry,
        priors["pas_log"][target_indices].astype(np.float32),
        priors["pdp_log"][target_indices].astype(np.float32),
        spectral_targets["pas_log"][reference_indices].astype(np.float32),
        spectral_targets["pdp_log"][reference_indices].astype(np.float32),
        priors["log_power"][target_indices],
        metadata["log_power"][reference_indices],
        priors["uncertainty"][target_indices],
    )


@torch.no_grad()
def evaluate_hybrid(
    model: SpectralGaussianHybrid,
    shape: object,
    channels: np.ndarray,
    metadata: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    target_indices: np.ndarray,
    observed_indices: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    device: torch.device,
    batch_size: int,
    outage_threshold: float,
    projection_iterations: int | None = None,
    spectral_targets: dict[str, np.ndarray] | None = None,
    power_bounds: np.ndarray | None = None,
    reference_strategy: dict[str, float | int] | None = None,
    outage_policy: dict[str, object] | None = None,
    carrier_fit: CarrierFit | None = None,
    transport_config: dict[str, object] | None = None,
    output_projection: dict[str, object] | None = None,
) -> dict[str, float | int]:
    observed_outage = metadata["outage"][observed_indices].astype(bool)
    transport_count = (
        int((transport_config or {}).get("count", 8)) if carrier_fit is not None else 1
    )
    candidates, distances = build_reference_candidates(
        metadata["train_positions"][target_indices],
        metadata["train_cells"][target_indices],
        metadata["train_positions"][observed_indices],
        metadata["train_cells"][observed_indices],
        observed_outage,
        top_k=max(
            1,
            transport_count,
            int((reference_strategy or {}).get("top_k", 1)),
        ),
        target_global_indices=target_indices,
        observed_global_indices=observed_indices,
    )
    candidate_globals = observed_indices[candidates]
    if reference_strategy and str(reference_strategy.get("name", "nearest")) != "nearest":
        if spectral_targets is None:
            raise ValueError("spectral_targets are required for spectral reference selection")
        target_geometry = _normalized_geometry(
            metadata["train_geometry_features"], target_indices, geometry_mean, geometry_std
        )
        reference_geometry = (
            metadata["train_geometry_features"] - geometry_mean
        ) / geometry_std
        references = select_reference_candidates(
            candidate_globals,
            distances,
            target_geometry,
            np.clip(reference_geometry, -8.0, 8.0),
            priors["pas_log"][target_indices].astype(np.float32),
            priors["pdp_log"][target_indices].astype(np.float32),
            spectral_targets["pas_log"].astype(np.float32),
            spectral_targets["pdp_log"].astype(np.float32),
            reference_strategy,
        )
    else:
        references = candidate_globals[:, 0]
    transport_globals = None
    transport_distances = None
    if carrier_fit is not None:
        transport_local, transport_distances = select_transport_candidates(
            candidates, distances, transport_count
        )
        transport_globals = observed_indices[transport_local]
    target_cells = metadata["train_cells"][target_indices].astype(np.int64)
    threshold_values = np.asarray(
        (outage_policy or {}).get("threshold_by_cell", [outage_threshold]),
        dtype=np.float32,
    ).reshape(-1)
    strength_values = np.asarray(
        (outage_policy or {}).get("soft_strength_by_cell", [0.0]),
        dtype=np.float32,
    ).reshape(-1)
    target_thresholds = threshold_values[
        np.minimum(target_cells, len(threshold_values) - 1)
    ]
    target_strengths = strength_values[
        np.minimum(target_cells, len(strength_values) - 1)
    ]
    output_projection = output_projection or {}
    output_projection_iterations = int(output_projection.get("iterations", 0))
    output_projection_strengths = np.asarray(
        output_projection.get("strength_by_cell", [0.0]), dtype=np.float32
    ).reshape(-1)
    target_output_projection_strengths = output_projection_strengths[
        np.minimum(target_cells, len(output_projection_strengths) - 1)
    ]
    accumulator = ChannelMetricAccumulator(shape)
    gate_sums = np.zeros((2, 2), dtype=np.float64)
    gate_counts = np.zeros(2, dtype=np.int64)
    model.eval()
    for start in range(0, len(target_indices), int(batch_size)):
        stop = min(start + int(batch_size), len(target_indices))
        indices = target_indices[start:stop]
        reference = torch.as_tensor(np.asarray(channels[references[start:stop]]), device=device)
        target = torch.as_tensor(np.asarray(channels[indices]), device=device)
        reference_context = None
        if model.condition_encoder.reference_dim:
            if spectral_targets is None:
                raise ValueError("spectral_targets are required by the reference-aware model")
            reference_context = _reference_context_batch(
                metadata,
                priors,
                spectral_targets,
                indices,
                references[start:stop],
                geometry_mean,
                geometry_std,
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
        )
        transport_channel = None
        if carrier_fit is not None:
            if transport_globals is None or transport_distances is None:
                raise AssertionError("transport candidates were not initialized")
            transport_channel, transport_context = _transport_batch(
                channels,
                metadata,
                indices,
                transport_globals[start:stop],
                transport_distances[start:stop],
                carrier_fit,
                device,
                distance_power=float((transport_config or {}).get("distance_power", 2.0)),
            )
            inputs["transport_context"] = transport_context
        outputs = model(
            reference,
            transport_channel=transport_channel,
            projection_iterations=projection_iterations,
            **inputs,
        )
        predicted_channel = outputs["channel"]
        if output_projection_iterations > 0:
            projection_power = (
                inputs["log_power"]
                if str(output_projection.get("power_source", "model")) == "input"
                else outputs["power"]
            )
            predicted_channel = relaxed_output_projection(
                predicted_channel,
                inputs["pas_log"],
                inputs["pdp_log"],
                inputs["ue_log_energy"],
                projection_power,
                shape,
                iterations=output_projection_iterations,
                proxy_count=model.proxy_count,
                strength=torch.as_tensor(
                    target_output_projection_strengths[start:stop], device=device
                ),
                minimum_scale=float(output_projection.get("minimum_scale", 0.5)),
                maximum_scale=float(output_projection.get("maximum_scale", 2.0)),
            )
        if outputs["spectrum_transport_gate"] is not None:
            batch_cells = target_cells[start:stop]
            spectrum_gate = outputs["spectrum_transport_gate"].mean(
                dim=(1, 2, 3, 4)
            )
            detail_gate = outputs["detail_transport_gate"].mean(dim=(1, 2, 3, 4))
            for cell in np.unique(batch_cells):
                selected = torch.as_tensor(batch_cells == cell, device=device)
                gate_sums[int(cell), 0] += float(spectrum_gate[selected].sum().cpu())
                gate_sums[int(cell), 1] += float(detail_gate[selected].sum().cpu())
                gate_counts[int(cell)] += int(np.sum(batch_cells == cell))
        threshold_batch = torch.as_tensor(
            target_thresholds[start:stop], device=device
        )
        strength_batch = torch.as_tensor(
            target_strengths[start:stop], device=device
        )
        predicted = apply_outage_policy(
            predicted_channel,
            inputs["outage_probability"],
            threshold_batch,
            strength_batch,
        )
        true_outage = torch.as_tensor(metadata["outage"][indices], device=device)
        accumulator.update(predicted, target, true_outage)
    result = accumulator.compute()
    result.update(
        {
            "samples": int(len(target_indices)),
            "projection_iterations": int(
                model.projection_iterations if projection_iterations is None else projection_iterations
            ),
            "predicted_outages": int(
                np.sum(
                    priors["outage_probability"][target_indices]
                    >= target_thresholds
                )
            ),
            "reference_strategy": str((reference_strategy or {}).get("name", "nearest")),
            "output_projection_iterations": output_projection_iterations,
            "output_projection_strength_by_cell": output_projection_strengths.tolist(),
            "output_projection_power_source": str(
                output_projection.get("power_source", "model")
            ),
        }
    )
    if carrier_fit is not None:
        result["transport_wave_numbers"] = carrier_fit.wave_numbers.astype(float).tolist()
        result["transport_fit_qualities"] = carrier_fit.qualities.astype(float).tolist()
        result["transport_spectrum_gate_by_cell"] = [
            float(gate_sums[cell, 0] / max(gate_counts[cell], 1))
            for cell in range(len(gate_counts))
        ]
        result["transport_detail_gate_by_cell"] = [
            float(gate_sums[cell, 1] / max(gate_counts[cell], 1))
            for cell in range(len(gate_counts))
        ]
    return result


def train_hybrid(config: dict, final: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(int(config["seed"]))
    section = (
        {**config["hybrid"], **config["hybrid_final"]}
        if final
        else config["hybrid"]
    )
    prior_path = config["spectral_teacher"]["oof_output_path"]
    metadata, priors, spectral_targets = _load_repository(config, prior_path)
    channels = np.load(Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r")
    available_count = min(len(channels), len(priors["available"]))
    available = priors["available"][:available_count].astype(bool)
    nonzero = ~metadata["outage"][:available_count].astype(bool)
    validation_mask = _validation_mask(
        metadata,
        available_count,
        config["split"].get("validation_fold", 0),
        final,
    )
    all_indices = np.arange(available_count, dtype=np.int64)
    if final:
        training_indices = all_indices[available & nonzero]
        validation_indices = np.empty(0, dtype=np.int64)
        observed_indices = all_indices[available]
    else:
        training_indices = all_indices[available & nonzero & ~validation_mask]
        validation_indices = all_indices[available & validation_mask]
        observed_indices = all_indices[available & ~validation_mask]
    training_indices = _balanced_limit(
        training_indices,
        metadata,
        int(config["runtime"].get("hybrid_train_limit", 0) or 0),
        int(config["seed"]) + 31,
    )
    validation_indices = _balanced_limit(
        validation_indices,
        metadata,
        int(config["runtime"].get("hybrid_validation_limit", 0) or 0),
        int(config["seed"]) + 37,
    )
    if len(training_indices) < 2:
        raise RuntimeError("Hybrid training requires at least two non-outage samples")
    device = choose_device(str(config["runtime"].get("device", "auto")))
    initial = section.get("initial_checkpoint")
    model, shape = _build_model(
        config,
        device,
        initial if initial and Path(initial).is_file() else None,
        section_override=section if final else None,
    )
    output_dir = Path(section["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    transport_config = section.get("transport_seed", {})
    carrier_fit = _load_or_fit_transport(
        section,
        metadata,
        channels,
        observed_indices,
        int(config["seed"]),
    )
    geometry_mean, geometry_std = _geometry_stats(
        metadata["train_geometry_features"], observed_indices
    )
    power_quantiles = section.get("power_bound_quantiles")
    power_bounds = None
    if power_quantiles is not None:
        power_bounds = compute_power_bounds(
            metadata["log_power"],
            metadata["outage"],
            metadata["train_cells"],
            observed_indices,
            float(power_quantiles[0]),
            float(power_quantiles[1]),
        )
    candidates, distances = build_reference_candidates(
        metadata["train_positions"][training_indices],
        metadata["train_cells"][training_indices],
        metadata["train_positions"][observed_indices],
        metadata["train_cells"][observed_indices],
        metadata["outage"][observed_indices],
        top_k=int(section.get("reference_candidates", 64)),
        target_global_indices=training_indices,
        observed_global_indices=observed_indices,
    )
    distance_profiles: dict[int, np.ndarray] = {}
    if str(section.get("reference_sampling", "guarded")) == "test_matched":
        if len(validation_indices):
            profile_positions = metadata["train_positions"][validation_indices]
            profile_cells = metadata["train_cells"][validation_indices]
        else:
            profile_positions = metadata["test_positions"]
            profile_cells = metadata["test_cells"]
        _, profile_distances = build_reference_candidates(
            profile_positions,
            profile_cells,
            metadata["train_positions"][observed_indices],
            metadata["train_cells"][observed_indices],
            metadata["outage"][observed_indices],
            top_k=1,
        )
        for cell in np.unique(profile_cells):
            distance_profiles[int(cell)] = profile_distances[
                profile_cells == cell, 0
            ].astype(np.float32)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = float(section["learning_rate"])
    decoder_scale = float(section.get("decoder_learning_rate_scale", 1.0))
    decoder_parameters = [
        parameter
        for parameter in model.autoencoder.decoder.parameters()
        if parameter.requires_grad
    ]
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    primary_parameters = [
        parameter for parameter in parameters if id(parameter) not in decoder_ids
    ]
    parameter_groups: list[dict[str, object]] = []
    if primary_parameters:
        parameter_groups.append({"params": primary_parameters, "lr": learning_rate})
    if decoder_parameters:
        parameter_groups.append(
            {"params": decoder_parameters, "lr": learning_rate * decoder_scale}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        weight_decay=float(section.get("weight_decay", 1e-4)),
    )
    scheduler_name = str(section.get("scheduler", "plateau")).lower()
    if scheduler_name == "plateau":
        scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(section.get("scheduler_factor", 0.5)),
                patience=int(section.get("scheduler_patience_validations", 8)),
                min_lr=float(section.get("minimum_learning_rate", 2e-6)),
            )
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(section["epochs"]), 1),
            eta_min=float(section.get("minimum_learning_rate", 2e-6)),
        )
    else:
        raise ValueError(f"Unsupported Scheme E scheduler: {scheduler_name}")
    amp = bool(config["runtime"].get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp)
    history_path = output_dir / "history.jsonl"
    resume = bool(section.get("resume", False))
    if not resume:
        history_path.unlink(missing_ok=True)
        for name in ("best.pt", "last.pt", "summary.json"):
            (output_dir / name).unlink(missing_ok=True)
    epochs = int(section["epochs"])
    steps = int(section["steps_per_epoch"])
    batch_size = int(section["batch_size"])
    rng = np.random.default_rng(int(config["seed"]) + (101 if final else 43))
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    start_epoch = 1
    weights = section.get("loss_weights", {"score": 1.0})
    outage_threshold = float(np.asarray(priors["outage_threshold"]).item())
    resume_path = output_dir / "last.pt"
    if resume and not resume_path.is_file() and (output_dir / "best.pt").is_file():
        resume_path = output_dir / "best.pt"
    if resume and resume_path.is_file():
        resumed = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        if "optimizer" in resumed:
            optimizer.load_state_dict(resumed["optimizer"])
        if "scaler" in resumed:
            scaler.load_state_dict(resumed["scaler"])
        if "scheduler" in resumed:
            scheduler.load_state_dict(resumed["scheduler"])
        if "rng_state" in resumed:
            rng.bit_generator.state = resumed["rng_state"]
        start_epoch = int(resumed.get("epoch", 0)) + 1
        if "best_score" in resumed:
            best_score = float(resumed["best_score"])
        elif "score" in resumed.get("metrics", {}):
            best_score = float(resumed["metrics"]["score"])
        elif "train_total" in resumed.get("metrics", {}):
            best_score = -float(resumed["metrics"]["train_total"])
        best_epoch = int(resumed.get("best_epoch", resumed.get("epoch", 0)))
        stale = int(resumed.get("stale", 0))
        print(f"SchemeE resume={resume_path} next_epoch={start_epoch}", flush=True)

    def save_last(epoch: int, metrics: dict[str, float | int]) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": rng.bit_generator.state,
                "epoch": epoch,
                "metrics": metrics,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale": stale,
                "geometry_mean": geometry_mean,
                "geometry_std": geometry_std,
                "outage_threshold": outage_threshold,
                "power_bounds": power_bounds,
                "carrier_fit": None if carrier_fit is None else carrier_fit.to_dict(),
                "config": config,
            },
            output_dir / "last.pt",
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        sums: dict[str, float] = {}
        for _ in range(steps):
            local = rng.integers(0, len(training_indices), size=batch_size)
            target_indices = training_indices[local]
            if str(section.get("reference_sampling", "guarded")) == "test_matched":
                selected_references = sample_test_matched_references(
                    candidates[local],
                    distances[local],
                    metadata["train_cells"][target_indices],
                    distance_profiles,
                    rng,
                )
            else:
                selected_references = sample_references(
                    candidates[local],
                    distances[local],
                    rng,
                    float(section.get("reference_guard_min_meters", 3.0)),
                    float(section.get("reference_guard_max_meters", 8.0)),
                )
            reference_indices = observed_indices[selected_references]
            transport_indices = None
            selected_transport_distances = None
            if carrier_fit is not None:
                batch_candidates = candidates[local]
                batch_distances = distances[local]
                selected_mask = batch_candidates == selected_references[:, None]
                if not np.all(np.any(selected_mask, axis=1)):
                    raise RuntimeError("Selected reference is missing from its candidate row")
                selected_ranks = np.argmax(selected_mask, axis=1)
                minimum_distances = batch_distances[
                    np.arange(len(batch_distances)), selected_ranks
                ]
                transport_local, selected_transport_distances = (
                    select_transport_candidates(
                        batch_candidates,
                        batch_distances,
                        int(transport_config.get("count", 8)),
                        minimum_distances,
                    )
                )
                transport_indices = observed_indices[transport_local]
            reference = torch.as_tensor(np.asarray(channels[reference_indices]), device=device)
            target = torch.as_tensor(np.asarray(channels[target_indices]), device=device)
            reference_context = None
            if model.condition_encoder.reference_dim:
                reference_context = _reference_context_batch(
                    metadata,
                    priors,
                    spectral_targets,
                    target_indices,
                    reference_indices,
                    geometry_mean,
                    geometry_std,
                )
            inputs = _prior_batch(
                priors,
                metadata,
                target_indices,
                geometry_mean,
                geometry_std,
                device,
                power_bounds=power_bounds,
                reference_context=reference_context,
            )
            transport_channel = None
            if carrier_fit is not None:
                if transport_indices is None or selected_transport_distances is None:
                    raise AssertionError("transport training candidates are missing")
                transport_channel, transport_context = _transport_batch(
                    channels,
                    metadata,
                    target_indices,
                    transport_indices,
                    selected_transport_distances,
                    carrier_fit,
                    device,
                    distance_power=float(transport_config.get("distance_power", 2.0)),
                )
                inputs["transport_context"] = transport_context
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                outputs = model(
                    reference,
                    transport_channel=transport_channel,
                    **inputs,
                )
                terms = metric_aligned_channel_losses(outputs["channel"], target, shape)
                target_shape, target_power, _ = channel_to_shape_target(target, shape)
                with torch.no_grad():
                    target_spectrum, target_detail = model.autoencoder.encode(target_shape)
                terms["spectrum_latent"] = functional.smooth_l1_loss(
                    outputs["spectrum"].flatten(1).float(), target_spectrum.float()
                )
                terms["detail_latent"] = functional.smooth_l1_loss(
                    outputs["detail"].flatten(1).float(), target_detail.float()
                )
                terms["detail_correlation"] = 1.0 - functional.cosine_similarity(
                    outputs["detail"].flatten(1).float(), target_detail.float(), dim=1, eps=1e-8
                ).mean()
                terms["power"] = functional.smooth_l1_loss(outputs["power"].float(), target_power.float())
                terms["residual"] = 0.5 * (
                    outputs["spectrum_residual"].float().square().mean()
                    + outputs["detail_residual"].float().square().mean()
                )
                if outputs["spectrum_transport_gate"] is not None:
                    terms["seed_spectrum"] = functional.smooth_l1_loss(
                        outputs["base_spectrum"].flatten(1).float(),
                        target_spectrum.float(),
                    )
                    terms["seed_detail"] = functional.smooth_l1_loss(
                        outputs["base_detail"].flatten(1).float(),
                        target_detail.float(),
                    )
                    terms["transport_gate"] = 0.5 * (
                        outputs["spectrum_transport_gate"].float().mean()
                        + outputs["detail_transport_gate"].float().mean()
                    )
                total = weighted_sum(terms, weights)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite Scheme E loss at epoch={epoch}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, float(section.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach().cpu()) / steps
            sums["total"] = sums.get("total", 0.0) + float(total.detach().cpu()) / steps
        validation: dict[str, float | int] = {}
        interval = int(section.get("validation_interval", 5))
        if len(validation_indices) and (epoch == 1 or epoch % interval == 0 or epoch == epochs):
            validation = evaluate_hybrid(
                model,
                shape,
                channels,
                metadata,
                priors,
                validation_indices,
                observed_indices,
                geometry_mean,
                geometry_std,
                device,
                int(section.get("validation_batch_size", batch_size)),
                outage_threshold,
                spectral_targets=spectral_targets,
                power_bounds=power_bounds,
                reference_strategy={"name": "nearest", "top_k": 1},
                carrier_fit=carrier_fit,
                transport_config=transport_config,
            )
            score = float(validation["score"])
            if score > best_score + float(section.get("minimum_delta", 1e-4)):
                best_score = score
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "metrics": validation,
                        "geometry_mean": geometry_mean,
                        "geometry_std": geometry_std,
                        "outage_threshold": outage_threshold,
                        "power_bounds": power_bounds,
                        "carrier_fit": (
                            None if carrier_fit is None else carrier_fit.to_dict()
                        ),
                        "config": config,
                    },
                    output_dir / "best.pt",
                )
            else:
                stale += interval
        elif not len(validation_indices):
            best_epoch = epoch
            best_score = -float(sums["total"])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if validation:
                scheduler.step(float(validation["score"]))
        else:
            scheduler.step()
        checkpoint_interval = int(section.get("checkpoint_interval", interval))
        if epoch == epochs or (checkpoint_interval and epoch % checkpoint_interval == 0):
            save_last(epoch, validation or {"train_total": sums["total"]})
        record = {
            "epoch": epoch,
            "train": sums,
            "validation": validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "decoder_learning_rate": (
                float(optimizer.param_groups[-1]["lr"])
                if decoder_parameters
                else None
            ),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"SchemeE epoch={epoch}/{epochs} train={sums['total']:.6f} "
            f"score={validation.get('score', float('nan')):.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"seconds={record['elapsed_seconds']:.2f}",
            flush=True,
        )
        patience = int(section.get("early_stopping_patience", 0))
        if len(validation_indices) and patience and stale >= patience:
            break
        maximum_hours = float(section.get("maximum_training_hours", 0.0))
        if maximum_hours and time.perf_counter() - started >= maximum_hours * 3600.0:
            break
    if not len(validation_indices):
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": best_epoch,
                "metrics": {"train_total": -best_score},
                "geometry_mean": geometry_mean,
                "geometry_std": geometry_std,
                "outage_threshold": outage_threshold,
                "power_bounds": power_bounds,
                "carrier_fit": None if carrier_fit is None else carrier_fit.to_dict(),
                "config": config,
            },
            output_dir / "best.pt",
        )
    if not (output_dir / "best.pt").is_file():
        raise RuntimeError("Scheme E training did not produce best.pt")
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    projection_reports: list[dict[str, float | int]] = []
    if len(validation_indices):
        reference_strategies = section.get(
            "reference_strategies", [{"name": "nearest", "top_k": 1}]
        )
        for strategy in reference_strategies:
            for iterations in section.get("projection_candidates", [0, 2, 4, 8]):
                projection_reports.append(
                    evaluate_hybrid(
                        model,
                        shape,
                        channels,
                        metadata,
                        priors,
                        validation_indices,
                        observed_indices,
                        geometry_mean,
                        geometry_std,
                        device,
                        int(section.get("validation_batch_size", batch_size)),
                        outage_threshold,
                        int(iterations),
                        spectral_targets=spectral_targets,
                        power_bounds=power_bounds,
                        reference_strategy=dict(strategy),
                        carrier_fit=carrier_fit,
                        transport_config=transport_config,
                    )
                )
        selected_projection = max(projection_reports, key=lambda item: float(item["score"]))
    else:
        selected_projection = {
            "projection_iterations": int(section.get("projection_iterations", 4)),
            "reference_strategy": str(section.get("reference_strategy", "nearest")),
        }
    summary = {
        "stage": "hybrid_final" if final else "hybrid_fold0",
        "architecture": (
            "spectral_gaussian_structured_field_v4"
            if bool(section.get("structured_spectral_field", False))
            else "spectral_gaussian_dual_seed_transport_v3"
            if carrier_fit is not None
            else
            "spectral_gaussian_reference_aware_v2"
            if bool(section.get("reference_aware", False))
            else "spectral_gaussian_power_safe_v2"
            if power_bounds is not None
            else "spectral_gaussian_full_resolution_adapter_v1"
        ),
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "training_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "best_epoch": int(checkpoint["epoch"]),
        "best_metrics": checkpoint.get("metrics", {}),
        "projection_candidates": projection_reports,
        "selected_projection_iterations": int(selected_projection["projection_iterations"]),
        "selected_reference_strategy": str(
            selected_projection.get("reference_strategy", "nearest")
        ),
        "power_bounds": None if power_bounds is None else power_bounds.tolist(),
        "carrier_fit": None if carrier_fit is None else carrier_fit.to_dict(),
        "checkpoint": str(output_dir / "best.pt"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def load_hybrid_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[SpectralGaussianHybrid, object, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, shape = _build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, shape, checkpoint
