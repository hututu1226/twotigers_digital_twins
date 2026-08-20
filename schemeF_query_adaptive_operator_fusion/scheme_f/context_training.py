from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .angle_delay import ChannelShape, channel_to_shape_target, shape_to_channel
from .autoencoder import (
    FactorizedResidualAutoencoder,
    MetricHighFidelityAutoencoder,
    StructuredAngleDelayAutoencoder,
)
from .autoencoder_training import load_autoencoder_checkpoint
from .config import (
    append_jsonl,
    autocast_context,
    choose_device,
    count_parameters,
    make_grad_scaler,
    save_json,
    seed_everything,
)
from .context_data import ContextRepository
from .context_model import FullResolutionContextField, channel_chart
from .data import balanced_limit, load_manifest, load_metadata, split_indices
from .losses import joint_power_loss, metric_aligned_channel_losses, weighted_sum
from .metrics import ChannelMetricAccumulator
from .projection import soft_shape_projection


Autoencoder = (
    StructuredAngleDelayAutoencoder
    | MetricHighFidelityAutoencoder
    | FactorizedResidualAutoencoder
)


def build_context_model(
    config: dict, repository: ContextRepository
) -> FullResolutionContextField:
    section = config["context"]
    return FullResolutionContextField(
        repository.spectrum_shape,
        repository.phase_shape,
        repository.cell_count,
        repository.static_context_channels,
        repository.query_numeric_channels,
        map_token_channels=int(section["map_token_channels"]),
        map_hidden_channels=int(section["map_hidden_channels"]),
        context_base_channels=int(section["base_channels"]),
        context_feature_channels=int(section["context_feature_channels"]),
        environment_base_channels=int(section.get("environment_base_channels", 32)),
        environment_feature_channels=int(section["environment_feature_channels"]),
        environment_blocks=int(section.get("environment_blocks", 2)),
        corridor_width=int(section.get("corridor_width", 96)),
        corridor_heads=int(section.get("corridor_heads", 4)),
        corridor_layers=int(section.get("corridor_layers", 2)),
        corridor_maximum_samples=int(section.get("corridor_maximum_samples", 32)),
        station_embedding_channels=int(section["station_embedding_channels"]),
        fourier_bands=int(section["fourier_bands"]),
        global_width=int(section["global_width"]),
        global_blocks=int(section["global_blocks"]),
        router_width=int(section.get("router_width", 128)),
        router_top_k=int(section.get("router_top_k", 96)),
        router_temperature=float(section.get("router_temperature_initial", 1.5)),
        router_uniform_mix=float(section.get("router_uniform_mix", 0.15)),
        router_dropout=float(section.get("router_dropout", 0.1)),
        route_bias_scale=float(section.get("route_bias_scale", 0.15)),
        chart_dimensions=int(section.get("chart_dimensions", 64)),
        chart_weight=float(section.get("chart_weight", 1.0)),
        pair_width=int(section.get("pair_width", 128)),
        spectrum_token_channels=int(section["spectrum_token_channels"]),
        detail_token_channels=int(section["detail_token_channels"]),
        attention_heads=int(section["attention_heads"]),
        attention_chunk_size=int(section["attention_chunk_size"]),
        refinement_blocks=int(section["refinement_blocks"]),
        axial_blocks=int(section.get("axial_blocks", 2)),
        operator_blocks=int(section.get("operator_blocks", 2)),
        operator_modes=tuple(
            int(value) for value in section.get("operator_modes", [4, 6, 8])
        ),
        token_top_k=int(section.get("token_top_k", 2)),
        regional_warp=bool(section.get("regional_warp", True)),
        detail_phase_rotation=bool(section.get("detail_phase_rotation", True)),
        spectrum_maximum_warp=tuple(
            float(value)
            for value in section.get("spectrum_maximum_warp", [0.75, 1.5, 3.0])
        ),
        detail_maximum_warp=tuple(
            float(value)
            for value in section.get("detail_maximum_warp", [1.5, 3.0, 6.0])
        ),
        spectrum_maximum_residual=float(section.get("spectrum_maximum_residual", 1.0)),
        detail_maximum_residual=float(section.get("detail_maximum_residual", 1.0)),
        maximum_power_residual=float(section.get("maximum_power_residual", 1.0)),
        maximum_power_z=float(section.get("maximum_power_z", 5.0)),
        power_width=int(section.get("power_width", 192)),
        dropout=float(section.get("dropout", 0.05)),
        gradient_checkpointing=bool(section.get("gradient_checkpointing", True)),
    )


def _tensor(value: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value)).to(device)
    return tensor if dtype is None else tensor.to(dtype=dtype)


def build_device_cache(
    repository: ContextRepository, device: torch.device
) -> dict[str, object] | None:
    section = repository.config["context"]
    if device.type != "cuda" or not bool(section.get("cache_latents_on_device", True)):
        return None
    dtype_name = str(section.get("latent_cache_dtype", "float32")).lower()
    latent_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in latent_dtypes:
        raise ValueError(f"Unsupported latent_cache_dtype={dtype_name}")
    latent_dtype = latent_dtypes[dtype_name]
    spectrum = torch.as_tensor(
        repository.spectrum_z, device=device, dtype=latent_dtype
    ).reshape(-1, *repository.spectrum_shape)
    phase = torch.as_tensor(
        repository.phase_z, device=device, dtype=latent_dtype
    ).reshape(-1, *repository.phase_shape)
    power = torch.as_tensor(repository.power_z, device=device, dtype=torch.float32)
    outage = torch.as_tensor(
        repository.metadata["outage"], device=device, dtype=torch.float32
    )
    return {
        "spectrum": spectrum,
        "phase": phase,
        "power": power,
        "outage": outage,
        "context_static": [
            torch.as_tensor(value, device=device) for value in repository.context_static
        ],
        "environment_bev": [
            torch.as_tensor(value, device=device)
            for value in repository.environment_bev
        ],
    }


def _cached_values(
    repository: ContextRepository,
    indices: np.ndarray,
    device: torch.device,
    device_cache: dict[str, object] | None,
) -> dict[str, torch.Tensor]:
    if device_cache is None:
        values = repository.structured_latents(indices)
        return {key: _tensor(value, device) for key, value in values.items()}
    selected = torch.as_tensor(indices, device=device, dtype=torch.long)
    return {
        key: device_cache[key].index_select(0, selected)
        for key in ("spectrum", "phase", "power", "outage")
    }


def _model_inputs(
    repository: ContextRepository,
    cell_id: int,
    context_indices: np.ndarray,
    query_indices: np.ndarray | None,
    device: torch.device,
    test: bool = False,
    device_cache: dict[str, object] | None = None,
) -> dict[str, torch.Tensor]:
    query = repository.query_features(cell_id, query_indices, test=test)
    observed_query = repository.query_features(
        cell_id, context_indices, test=False, include_corridor=False
    )
    observed = _cached_values(repository, context_indices, device, device_cache)
    context_static = (
        device_cache["context_static"][cell_id]
        if device_cache is not None
        else _tensor(repository.context_static[cell_id], device)
    )
    environment_bev = (
        device_cache["environment_bev"][cell_id]
        if device_cache is not None
        else _tensor(repository.environment_bev[cell_id], device)
    )
    return {
        "observed_spectrum": observed["spectrum"],
        "observed_phase": observed["phase"],
        "observed_power": observed["power"],
        "observed_outage": observed["outage"],
        "point_features": _tensor(repository.point_features(context_indices), device),
        "point_flat_indices": _tensor(
            repository.flat_indices(cell_id, context_indices), device, torch.long
        ),
        "context_static": context_static,
        "environment_bev": environment_bev,
        "observed_context_coordinates": _tensor(
            observed_query["context_coordinates"], device
        ),
        "observed_environment_coordinates": _tensor(
            observed_query["environment_coordinates"], device
        ),
        "observed_numeric": _tensor(observed_query["numeric"], device),
        "observed_relative_xy": _tensor(observed_query["relative_xy"], device),
        "query_context_coordinates": _tensor(query["context_coordinates"], device),
        "query_environment_coordinates": _tensor(
            query["environment_coordinates"], device
        ),
        "query_corridor_coordinates": _tensor(query["corridor_coordinates"], device),
        "query_numeric": _tensor(query["numeric"], device),
        "query_relative_xy": _tensor(query["relative_xy"], device),
    }


def _decode_predictions(
    outputs: dict[str, torch.Tensor],
    cell_id: int,
    repository: ContextRepository,
    autoencoder: Autoencoder,
    shape: ChannelShape,
) -> torch.Tensor:
    device = outputs["spectrum"].device

    def latent_stat(name: str) -> torch.Tensor:
        value = torch.as_tensor(repository.encoded[name], device=device)
        return value[int(cell_id)] if value.ndim == 2 else value

    spectrum_mean = latent_stat("spectrum_mean")
    spectrum_std = latent_stat("spectrum_std")
    phase_mean = latent_stat("phase_mean")
    phase_std = latent_stat("phase_std")
    power_mean = torch.as_tensor(repository.encoded["power_mean"], device=device)
    power_std = torch.as_tensor(repository.encoded["power_std"], device=device)
    spectrum = outputs["spectrum"].float() * spectrum_std + spectrum_mean
    phase = outputs["phase"].float() * phase_std + phase_mean
    log_power = (
        outputs["power"].float() * power_std[int(cell_id)] + power_mean[int(cell_id)]
    )
    prediction_shape = autoencoder.decode(spectrum, phase)
    return shape_to_channel(prediction_shape, log_power, shape)


def apply_outage_policy(
    channel: torch.Tensor,
    probability: np.ndarray | torch.Tensor,
    threshold: float,
    soft_strength: float,
) -> torch.Tensor:
    """Attenuate uncertain links before applying a conservative exact-zero decision."""
    if not 0.0 <= soft_strength <= 1.0:
        raise ValueError("soft outage strength must lie in [0, 1]")
    probabilities = torch.as_tensor(
        probability, device=channel.device, dtype=torch.float32
    )
    scale = (1.0 - float(soft_strength) * probabilities).clamp_min(0.0).sqrt()
    result = channel * scale[:, None, None, None].to(channel.dtype)
    return result.masked_fill(
        (probabilities >= float(threshold))[:, None, None, None], 0.0
    )


@torch.no_grad()
def predict_indices(
    model: FullResolutionContextField,
    repository: ContextRepository,
    target_indices: np.ndarray | None,
    device: torch.device,
    amp: bool,
    test: bool = False,
    device_cache: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    if device_cache is None:
        device_cache = build_device_cache(repository, device)
    if test:
        all_count = len(repository.metadata["test_cells"])
        selected_indices = (
            np.arange(all_count, dtype=np.int64)
            if target_indices is None
            else np.asarray(target_indices, dtype=np.int64)
        )
        count = len(selected_indices)
        cell_ids = repository.metadata["test_cells"][selected_indices]
    else:
        if target_indices is None:
            raise ValueError("Training/validation prediction requires target_indices")
        selected_indices = np.asarray(target_indices, dtype=np.int64)
        count = len(selected_indices)
        cell_ids = repository.metadata["train_cells"][selected_indices]
    spectrum = np.empty((count, repository.spectrum_latent_dim), dtype=np.float32)
    phase = np.empty((count, repository.phase_latent_dim), dtype=np.float32)
    power = np.empty(count, dtype=np.float32)
    power_q10 = np.empty(count, dtype=np.float32)
    power_q90 = np.empty(count, dtype=np.float32)
    outage_probability = np.empty(count, dtype=np.float32)
    diagnostics = {
        name: np.empty(count, dtype=np.float32)
        for name in (
            "router_entropy",
            "router_top1_mass",
            "router_effective_neighbors",
            "router_distance",
            "spectrum_warp",
            "detail_warp",
            "warp_saturation",
            "router_temperature",
            "spectrum_residual_rms",
            "phase_residual_rms",
            "spectrum_token_effective_neighbors",
            "detail_token_effective_neighbors",
            "spectrum_token_top1_mass",
            "detail_token_top1_mass",
            "power_effective_neighbors",
        )
    }
    for cell_id in range(repository.cell_count):
        local = np.flatnonzero(cell_ids == cell_id)
        if not len(local):
            continue
        context_indices = repository.context_indices(cell_id)
        query_batch_size = max(
            1, int(repository.config["context"].get("inference_query_batch_size", 16))
        )
        for start in range(0, len(local), query_batch_size):
            stop = min(start + query_batch_size, len(local))
            local_batch = local[start:stop]
            queries = selected_indices[local_batch]
            inputs = _model_inputs(
                repository,
                cell_id,
                context_indices,
                queries,
                device,
                test=test,
                device_cache=device_cache,
            )
            with autocast_context(device, amp):
                outputs = model(cell_id=cell_id, **inputs)
            spectrum[local_batch] = outputs["spectrum"].float().cpu().numpy()
            phase[local_batch] = outputs["phase"].float().cpu().numpy()
            power[local_batch] = outputs["power"].float().cpu().numpy()
            power_q10[local_batch] = outputs["power_q10"].float().cpu().numpy()
            power_q90[local_batch] = outputs["power_q90"].float().cpu().numpy()
            outage_probability[local_batch] = (
                torch.sigmoid(outputs["outage_logit"]).float().cpu().numpy()
            )
            for name, values in diagnostics.items():
                values[local_batch] = outputs[name].float().cpu().numpy()
    return {
        "spectrum": spectrum,
        "phase": phase,
        "power": power,
        "power_q10": power_q10,
        "power_q90": power_q90,
        "outage_probability": outage_probability,
        **diagnostics,
    }


@torch.no_grad()
def evaluate_context_model(
    model: FullResolutionContextField,
    autoencoder: Autoencoder,
    repository: ContextRepository,
    target_indices: np.ndarray,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
    outage_threshold: float,
    decode_batch_size: int,
    device_cache: dict[str, object] | None = None,
    soft_outage_strength: float | None = None,
    spectral_prior_alpha: float | None = None,
) -> dict[str, float]:
    return evaluate_context_thresholds(
        model,
        autoencoder,
        repository,
        target_indices,
        shape,
        device,
        amp,
        [outage_threshold],
        decode_batch_size,
        device_cache,
        soft_outage_strength,
        spectral_prior_alpha,
    )[0]


@torch.no_grad()
def evaluate_context_thresholds(
    model: FullResolutionContextField,
    autoencoder: Autoencoder,
    repository: ContextRepository,
    target_indices: np.ndarray,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
    outage_thresholds: list[float],
    decode_batch_size: int,
    device_cache: dict[str, object] | None = None,
    soft_outage_strength: float | None = None,
    spectral_prior_alpha: float | None = None,
    prediction_outputs: dict[str, np.ndarray] | None = None,
) -> list[dict[str, float]]:
    thresholds = [float(value) for value in outage_thresholds]
    if not thresholds:
        raise ValueError("At least one outage threshold is required")
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Every outage threshold must lie in the open interval (0, 1)")
    outputs = prediction_outputs
    if outputs is None:
        outputs = predict_indices(
            model,
            repository,
            target_indices,
            device,
            amp,
            device_cache=device_cache,
        )
    soft_strength = float(
        repository.config["context"].get("soft_outage_strength", 0.0)
        if soft_outage_strength is None
        else soft_outage_strength
    )
    prior_alpha = float(
        repository.config["context"].get("spectral_prior_alpha", 0.0)
        if spectral_prior_alpha is None
        else spectral_prior_alpha
    )
    metadata = repository.metadata
    cell_ids = metadata["train_cells"][target_indices]
    true_outage = metadata["outage"][target_indices]
    target = repository.target_values(target_indices)
    nonzero = ~true_outage
    spectrum_mse = (
        float(
            np.mean((outputs["spectrum"][nonzero] - target["spectrum"][nonzero]) ** 2)
        )
        if np.any(nonzero)
        else 0.0
    )
    phase_mse = (
        float(np.mean((outputs["phase"][nonzero] - target["phase"][nonzero]) ** 2))
        if np.any(nonzero)
        else 0.0
    )
    power_error = outputs["power"][nonzero] - target["power"][nonzero]
    log_power_error = power_error * repository.encoded["power_std"][cell_ids[nonzero]]
    absolute_log_power_error = np.abs(log_power_error)
    accumulators = [ChannelMetricAccumulator(shape) for _ in thresholds]
    channels = np.load(
        Path(repository.config["data"]["root"]) / "Round2_Train_Channel.npy",
        mmap_mode="r",
    )
    for start in range(0, len(target_indices), decode_batch_size):
        stop = min(start + decode_batch_size, len(target_indices))
        selected = slice(start, stop)
        predictions = {
            key: torch.from_numpy(outputs[key][selected]).to(device)
            for key in ("spectrum", "phase", "power")
        }
        batch_cells = cell_ids[selected]
        prediction_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        for cell_id in np.unique(batch_cells):
            local = np.flatnonzero(batch_cells == cell_id)
            local_tensor = torch.from_numpy(local).to(device=device, dtype=torch.long)
            local_outputs = {
                key: value.index_select(0, local_tensor)
                for key, value in predictions.items()
            }
            decoded = _decode_predictions(
                local_outputs, int(cell_id), repository, autoencoder, shape
            )
            prediction_parts.append((local_tensor, decoded))
        prediction = torch.empty(
            (stop - start, *shape.raw_shape), dtype=torch.complex64, device=device
        )
        for local, decoded in prediction_parts:
            prediction.index_copy_(0, local, decoded)
        prior = repository.spectral_prior_values(target_indices[selected], test=False)
        if prior_alpha > 0.0:
            if prior is None:
                raise ValueError(
                    "spectral_prior_alpha is positive but priors are unavailable"
                )
            prediction = soft_shape_projection(
                prediction,
                torch.from_numpy(prior["pas_log"]).to(device),
                torch.from_numpy(prior["pdp_log"]).to(device),
                shape,
                prior_alpha,
                proxy_count=int(
                    repository.config["context"]
                    .get("spectral_prior", {})
                    .get("proxy_count", 24)
                ),
                iterations=int(
                    repository.config["context"].get("spectral_prior_iterations", 1)
                ),
                minimum_scale=float(
                    repository.config["context"].get(
                        "spectral_prior_minimum_scale", 0.75
                    )
                ),
                maximum_scale=float(
                    repository.config["context"].get(
                        "spectral_prior_maximum_scale", 1.35
                    )
                ),
            )
        target_channel = torch.from_numpy(
            np.array(channels[target_indices[selected]], copy=True)
        ).to(device)
        true_outage_tensor = torch.from_numpy(true_outage[selected]).to(device)
        probabilities = outputs["outage_probability"][selected]
        for threshold, metrics in zip(thresholds, accumulators):
            masked_prediction = apply_outage_policy(
                prediction,
                probabilities,
                threshold,
                soft_strength,
            )
            metrics.update(masked_prediction, target_channel, true_outage_tensor)

    reports: list[dict[str, float]] = []
    for threshold, metrics in zip(thresholds, accumulators):
        predicted_outage = outputs["outage_probability"] >= threshold
        true_positive = int(np.sum(predicted_outage & true_outage))
        false_positive = int(np.sum(predicted_outage & ~true_outage))
        false_negative = int(np.sum(~predicted_outage & true_outage))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        result = metrics.compute()
        result.update(
            {
                "outage_threshold": threshold,
                "soft_outage_strength": soft_strength,
                "spectral_prior_alpha": prior_alpha,
                "spectrum_latent_mse_z": spectrum_mse,
                "phase_latent_mse_z": phase_mse,
                "power_mae_z": (
                    float(np.mean(np.abs(power_error))) if len(power_error) else 0.0
                ),
                "power_rmse_z": (
                    float(np.sqrt(np.mean(power_error**2))) if len(power_error) else 0.0
                ),
                "power_mae_log10": (
                    float(np.mean(absolute_log_power_error))
                    if len(log_power_error)
                    else 0.0
                ),
                "power_p90_log10": (
                    float(np.percentile(absolute_log_power_error, 90))
                    if len(log_power_error)
                    else 0.0
                ),
                "power_p99_log10": (
                    float(np.percentile(absolute_log_power_error, 99))
                    if len(log_power_error)
                    else 0.0
                ),
                "outage_accuracy": float(np.mean(predicted_outage == true_outage)),
                "outage_precision": precision,
                "outage_recall": recall,
                "outage_f1": 2.0 * precision * recall / max(precision + recall, 1e-30),
                "predicted_outages": int(predicted_outage.sum()),
                "router_entropy": float(outputs["router_entropy"].mean()),
                "router_top1_mass": float(outputs["router_top1_mass"].mean()),
                "router_effective_neighbors": float(
                    outputs["router_effective_neighbors"].mean()
                ),
                "router_distance_normalized": float(outputs["router_distance"].mean()),
                "spectrum_warp_bins": float(outputs["spectrum_warp"].mean()),
                "detail_warp_bins": float(outputs["detail_warp"].mean()),
                "warp_saturation": float(outputs["warp_saturation"].mean()),
                "router_temperature": float(outputs["router_temperature"].mean()),
                "spectrum_residual_rms": float(outputs["spectrum_residual_rms"].mean()),
                "phase_residual_rms": float(outputs["phase_residual_rms"].mean()),
                "spectrum_token_effective_neighbors": float(
                    outputs["spectrum_token_effective_neighbors"].mean()
                ),
                "detail_token_effective_neighbors": float(
                    outputs["detail_token_effective_neighbors"].mean()
                ),
                "spectrum_token_top1_mass": float(
                    outputs["spectrum_token_top1_mass"].mean()
                ),
                "detail_token_top1_mass": float(
                    outputs["detail_token_top1_mass"].mean()
                ),
                "power_effective_neighbors": float(
                    outputs["power_effective_neighbors"].mean()
                ),
                "samples": int(len(target_indices)),
            }
        )
        reports.append(result)
    return reports


def _save_checkpoint(
    path: Path,
    model: FullResolutionContextField,
    autoencoder: Autoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    config: dict,
    epoch: int,
    metrics: dict,
    best_score: float,
    epochs_without_improvement: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "autoencoder": autoencoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "router_temperature": float(model.router.temperature),
            "training_stage": "query_adaptive_operator_fusion_v1",
        },
        path,
    )


def train_context_model(config: dict, resume: bool = False) -> dict:
    seed_everything(int(config["seed"]))
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    training_indices, validation_indices = split_indices(metadata, config)
    with np.load(config["encoding"]["output_path"]) as encoded_source:
        available = (
            encoded_source["available"].astype(bool)
            if "available" in encoded_source.files
            else np.ones(len(metadata["train_cells"]), dtype=bool)
        )
    training_indices = training_indices[available[training_indices]]
    validation_indices = validation_indices[available[validation_indices]]
    runtime = config["runtime"]
    training_indices = balanced_limit(
        training_indices,
        runtime.get("context_train_limit", runtime.get("train_limit")),
        [metadata["train_cells"]],
        int(config["seed"]) + 3,
    )
    validation_indices = balanced_limit(
        validation_indices,
        runtime.get("context_validation_limit", runtime.get("validation_limit")),
        [metadata["train_cells"]],
        int(config["seed"]) + 4,
    )
    repository = ContextRepository(config, training_indices)
    if any(not len(indices) for indices in repository.indices_by_cell):
        raise ValueError(
            "The encoded/limited context set must contain at least one sample from every cell"
        )
    model = build_context_model(config, repository).to(device)
    autoencoder_path = (
        config["context"].get("autoencoder_checkpoint")
        or config["encoding"]["autoencoder_checkpoint"]
    )
    autoencoder, loaded_shape, _ = load_autoencoder_checkpoint(
        config, autoencoder_path, device
    )
    if loaded_shape != shape:
        raise ValueError("Autoencoder and preprocessing channel shapes differ")

    section = config["context"]
    autoencoder.requires_grad_(False)
    train_decoder = bool(section.get("train_decoder", True))
    if train_decoder:
        autoencoder.decoder.requires_grad_(True)
    model_parameters = list(model.parameters())
    decoder_parameters = list(autoencoder.decoder.parameters()) if train_decoder else []
    parameters = model_parameters + decoder_parameters
    parameter_groups: list[dict] = [
        {"params": model_parameters, "lr": float(section["learning_rate"])}
    ]
    if decoder_parameters:
        parameter_groups.append(
            {
                "params": decoder_parameters,
                "lr": float(
                    section.get(
                        "decoder_learning_rate",
                        float(section["learning_rate"]) * 0.02,
                    )
                ),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(section.get("weight_decay", 1e-4)),
    )
    epochs = int(section["epochs"])
    scheduler_name = str(section.get("scheduler", "plateau")).lower()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(section.get("plateau_factor", 0.5)),
            patience=int(section.get("plateau_patience", 4)),
            threshold=float(section.get("plateau_threshold", 1e-4)),
            threshold_mode="abs",
            min_lr=float(section.get("minimum_learning_rate", 1e-6)),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(section.get("minimum_learning_rate", 1e-6)),
        )
    else:
        raise ValueError(f"Unsupported context scheduler: {scheduler_name}")
    scaler = make_grad_scaler(device, amp)
    output_dir = Path(section["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)
    history_path = output_dir / "history.jsonl"
    if not resume:
        history_path.unlink(missing_ok=True)
        for name in ("best.pt", "last.pt", "final.pt"):
            (output_dir / name).unlink(missing_ok=True)

    start_epoch = 0
    best_score = -math.inf
    epochs_without_improvement = 0
    resumed_metrics: dict = {}
    if resume:
        checkpoint_path = output_dir / "last.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot resume because {checkpoint_path} does not exist"
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        autoencoder.load_state_dict(checkpoint["autoencoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -math.inf))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        resumed_metrics = checkpoint.get("metrics", {})

    weights = section["loss_weights"]
    steps_per_epoch = int(section["steps_per_epoch"])
    accumulation = int(section.get("gradient_accumulation", 1))
    validation_interval = int(section.get("validation_interval", 5))
    patience = int(section.get("early_stopping_patience", 0))
    minimum_delta = float(section.get("minimum_delta", 1e-4))
    outage_thresholds = [
        float(value)
        for value in section.get(
            "validation_outage_thresholds",
            [section.get("outage_threshold", 0.999)],
        )
    ]
    decode_batch_size = int(section.get("validation_decode_batch_size", 8))
    nonoutage_count = int((~metadata["outage"][training_indices]).sum())
    outage_count = int(metadata["outage"][training_indices].sum())
    positive_weight = float(section.get("outage_positive_weight", 1.0))
    if positive_weight <= 0:
        positive_weight = nonoutage_count / max(outage_count, 1)
    positive_weight_tensor = torch.tensor(positive_weight, device=device)
    channel_array = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    device_cache = build_device_cache(repository, device)
    final_metrics = resumed_metrics
    last_epoch = start_epoch - 1
    maximum_training_hours = float(section.get("maximum_training_hours", 0.0))
    target_score = float(section.get("target_score", 0.7))
    stop_at_target = bool(section.get("stop_at_target", False))
    training_started = time.perf_counter()
    stop_reason = "maximum_epochs"
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        started = time.perf_counter()
        model.train()
        temperature_initial = float(section.get("router_temperature_initial", 1.8))
        temperature_final = float(section.get("router_temperature_final", 0.9))
        anneal_epochs = max(
            int(section.get("router_temperature_anneal_epochs", epochs)), 1
        )
        anneal_progress = min(epoch / max(anneal_epochs - 1, 1), 1.0)
        model.router.temperature = (
            temperature_initial
            + (temperature_final - temperature_initial) * anneal_progress
        )
        autoencoder.decoder.train(train_decoder)
        optimizer.zero_grad(set_to_none=True)
        sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        rng = np.random.default_rng(int(config["seed"]) + epoch * 1000003)
        for step in range(steps_per_epoch):
            cell_id = int(rng.integers(repository.cell_count))
            mask_sample = repository.sample_spatial_mask(
                rng,
                cell_id,
                float(section["hole_min_meters"]),
                float(section["hole_max_meters"]),
                int(section["minimum_targets"]),
                int(section["maximum_targets"]),
                float(section.get("outage_anchor_probability", 0.25)),
                float(section.get("test_template_probability", 0.65)),
                float(section.get("template_radius_meters", 3.0)),
                float(section.get("observation_guard_min_meters", 3.5)),
                float(section.get("observation_guard_max_meters", 8.5)),
            )
            targets = mask_sample.targets
            context_indices = repository.context_indices(cell_id, mask_sample.hidden)
            maximum_observations = int(section.get("maximum_observations", 0))
            if maximum_observations > 0 and len(context_indices) > maximum_observations:
                context_indices = np.sort(
                    rng.choice(
                        context_indices, size=maximum_observations, replace=False
                    ).astype(np.int64)
                )
            inputs = _model_inputs(
                repository,
                cell_id,
                context_indices,
                targets,
                device,
                test=False,
                device_cache=device_cache,
            )
            target_values = repository.target_values(targets)
            target_channel = torch.from_numpy(
                np.array(channel_array[targets], copy=True)
            ).to(device)
            target_shape, _, _ = channel_to_shape_target(target_channel, shape)
            target_spectrum = _tensor(target_values["spectrum"], device)
            target_phase = _tensor(target_values["phase"], device)
            target_power = _tensor(target_values["power"], device)
            target_outage = _tensor(target_values["outage"], device)
            with autocast_context(device, amp):
                outputs = model(cell_id=cell_id, **inputs)
                terms: dict[str, torch.Tensor] = {
                    "outage": functional.binary_cross_entropy_with_logits(
                        outputs["outage_logit"],
                        target_outage,
                        pos_weight=positive_weight_tensor,
                    )
                }
                nonzero = target_outage < 0.5
                if torch.any(nonzero):
                    terms["spectrum_latent"] = functional.mse_loss(
                        outputs["spectrum"][nonzero].float(), target_spectrum[nonzero]
                    )
                    terms["phase_latent"] = functional.smooth_l1_loss(
                        outputs["phase"][nonzero].float(), target_phase[nonzero]
                    )
                    terms["detail_correlation"] = (
                        1.0
                        - functional.cosine_similarity(
                            outputs["phase"][nonzero].float().flatten(1),
                            target_phase[nonzero].float().flatten(1),
                            dim=1,
                            eps=1e-8,
                        ).mean()
                    )
                    terms["power"] = functional.smooth_l1_loss(
                        outputs["power"][nonzero].float(), target_power[nonzero]
                    )
                    q10_error = (
                        target_power[nonzero] - outputs["power_q10"][nonzero].float()
                    )
                    q90_error = (
                        target_power[nonzero] - outputs["power_q90"][nonzero].float()
                    )
                    terms["power_quantile"] = (
                        torch.maximum(0.1 * q10_error, -0.9 * q10_error).mean()
                        + torch.maximum(0.9 * q90_error, -0.1 * q90_error).mean()
                    )
                    target_chart = channel_chart(
                        target_spectrum[nonzero],
                        target_phase[nonzero],
                        outputs["query_chart"].shape[1],
                    ).detach()
                    terms["chart"] = (
                        1.0
                        - functional.cosine_similarity(
                            outputs["query_chart"][nonzero].float(),
                            target_chart,
                            dim=1,
                            eps=1e-6,
                        ).mean()
                    )
                    terms["base_spectrum_latent"] = functional.smooth_l1_loss(
                        outputs["spectrum_base"][nonzero].float(),
                        target_spectrum[nonzero],
                    )
                    terms["base_phase_latent"] = functional.smooth_l1_loss(
                        outputs["phase_base"][nonzero].float(), target_phase[nonzero]
                    )
                    terms["base_power"] = functional.smooth_l1_loss(
                        outputs["power_base"][nonzero].float(), target_power[nonzero]
                    )
                    selected_outputs = {
                        key: value[nonzero]
                        for key, value in outputs.items()
                        if key in ("spectrum", "phase", "power")
                    }
                    prediction = _decode_predictions(
                        selected_outputs, cell_id, repository, autoencoder, shape
                    )
                    terms.update(
                        metric_aligned_channel_losses(
                            prediction, target_channel[nonzero], shape
                        )
                    )
                    predicted_shape, _, _ = channel_to_shape_target(prediction, shape)
                    terms["joint_power"] = joint_power_loss(
                        predicted_shape, target_shape[nonzero], shape
                    )
                    if train_decoder:
                        teacher_outputs = {
                            "spectrum": target_spectrum[nonzero],
                            "phase": target_phase[nonzero],
                            "power": target_power[nonzero],
                        }
                        teacher_prediction = _decode_predictions(
                            teacher_outputs,
                            cell_id,
                            repository,
                            autoencoder,
                            shape,
                        )
                        teacher_losses = metric_aligned_channel_losses(
                            teacher_prediction, target_channel[nonzero], shape
                        )
                        terms["decoder_teacher"] = teacher_losses["score"]
                    terms["residual_regularization"] = 0.5 * (
                        outputs["spectrum_residual"][nonzero].float().square().mean()
                        + outputs["phase_residual"][nonzero].float().square().mean()
                    )
                minimum_neighbors = float(
                    section.get("router_minimum_effective_neighbors", 8.0)
                )
                terms["router_diversity"] = (
                    (
                        functional.relu(
                            minimum_neighbors
                            - outputs["router_effective_neighbors"].float()
                        )
                        / max(minimum_neighbors, 1.0)
                    )
                    .square()
                    .mean()
                )
                minimum_token_neighbors = float(
                    section.get("token_minimum_effective_neighbors", 1.15)
                )
                token_effective = 0.5 * (
                    outputs["spectrum_token_effective_neighbors"].float()
                    + outputs["detail_token_effective_neighbors"].float()
                )
                terms["token_diversity"] = (
                    functional.relu(minimum_token_neighbors - token_effective)
                    .square()
                    .mean()
                )
                terms["warp_saturation"] = outputs["warp_saturation"].float().mean()
                total = weighted_sum(terms, weights) / accumulation
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite context loss at epoch={epoch + 1}, step={step + 1}"
                )
            scaler.scale(total).backward()
            if (step + 1) % accumulation == 0 or step + 1 == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(section.get("gradient_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for name, value in terms.items():
                sums[name] += float(value.detach().cpu())
            sums["mask_hidden"] += float(len(mask_sample.hidden))
            sums["mask_targets"] += float(len(mask_sample.targets))
            sums["mask_guard_meters"] += mask_sample.guard_meters
            sums["router_entropy"] += float(
                outputs["router_entropy"].detach().mean().cpu()
            )
            sums["router_top1_mass"] += float(
                outputs["router_top1_mass"].detach().mean().cpu()
            )
            sums["router_effective_neighbors"] += float(
                outputs["router_effective_neighbors"].detach().mean().cpu()
            )
            sums["router_distance"] += float(
                outputs["router_distance"].detach().mean().cpu()
            )
            sums["spectrum_warp"] += float(
                outputs["spectrum_warp"].detach().mean().cpu()
            )
            sums["detail_warp"] += float(outputs["detail_warp"].detach().mean().cpu())
            sums["warp_saturation"] += float(
                outputs["warp_saturation"].detach().mean().cpu()
            )
            sums["spectrum_residual_rms"] += float(
                outputs["spectrum_residual_rms"].detach().mean().cpu()
            )
            sums["phase_residual_rms"] += float(
                outputs["phase_residual_rms"].detach().mean().cpu()
            )
            for name in (
                "spectrum_token_effective_neighbors",
                "detail_token_effective_neighbors",
                "spectrum_token_top1_mass",
                "detail_token_top1_mass",
                "power_effective_neighbors",
            ):
                sums[name] += float(outputs[name].detach().mean().cpu())
            sums["router_temperature"] += float(model.router.temperature)
            sums["total"] += float(total.detach().cpu()) * accumulation
            batches += 1
        train_metrics = {name: value / max(batches, 1) for name, value in sums.items()}
        should_validate = len(validation_indices) and (
            (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        )
        validation_reports = (
            evaluate_context_thresholds(
                model,
                autoencoder,
                repository,
                validation_indices,
                shape,
                device,
                amp,
                outage_thresholds,
                decode_batch_size,
                device_cache,
            )
            if should_validate
            else []
        )
        validation = (
            max(validation_reports, key=lambda report: float(report["score"]))
            if validation_reports
            else {}
        )
        if validation:
            validation["outage_threshold_candidates"] = len(outage_thresholds)
        final_metrics = validation or train_metrics
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "decoder_learning_rate": (
                optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 1
                else 0.0
            ),
            "train": train_metrics,
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - started,
        }
        append_jsonl(history_path, record)
        score = float(validation.get("score", -math.inf))
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if validation:
                scheduler.step(score)
        else:
            scheduler.step()
        improved = bool(validation) and score > best_score + minimum_delta
        if improved:
            best_score = score
            epochs_without_improvement = 0
            _save_checkpoint(
                output_dir / "best.pt",
                model,
                autoencoder,
                optimizer,
                scheduler,
                scaler,
                config,
                epoch,
                validation,
                best_score,
                0,
            )
        elif validation:
            epochs_without_improvement += validation_interval
        _save_checkpoint(
            output_dir / "last.pt",
            model,
            autoencoder,
            optimizer,
            scheduler,
            scaler,
            config,
            epoch,
            final_metrics,
            best_score,
            epochs_without_improvement,
        )
        stage = "Context"
        print(
            f"{stage} epoch={epoch + 1}/{epochs} train={train_metrics.get('total', 0.0):.6f} "
            f"score={validation.get('score', float('nan')):.6f} "
            f"threshold={validation.get('outage_threshold', float('nan')):.4f} "
            f"seconds={record['elapsed_seconds']:.2f}",
            flush=True,
        )
        if patience > 0 and validation and epochs_without_improvement >= patience:
            stop_reason = "early_stopping"
            print(f"{stage} early stopping at epoch {epoch + 1}", flush=True)
            break
        if stop_at_target and validation and best_score >= target_score:
            stop_reason = "target_reached"
            print(
                f"{stage} target reached: {best_score:.6f} >= {target_score:.6f}",
                flush=True,
            )
            break
        elapsed_training_seconds = time.perf_counter() - training_started
        if (
            maximum_training_hours > 0.0
            and elapsed_training_seconds >= maximum_training_hours * 3600.0
        ):
            stop_reason = "runtime_limit"
            print(
                f"{stage} runtime limit reached after "
                f"{elapsed_training_seconds / 3600.0:.2f} hours",
                flush=True,
            )
            break

    _save_checkpoint(
        output_dir / "final.pt",
        model,
        autoencoder,
        optimizer,
        scheduler,
        scaler,
        config,
        last_epoch,
        final_metrics,
        best_score,
        epochs_without_improvement,
    )
    summary = {
        "device": str(device),
        "training_stage": "query_adaptive_operator_fusion_v1",
        "architecture": "query_adaptive_operator_fusion_v1",
        "decoder_trained_end_to_end": train_decoder,
        "context_parameters": count_parameters(model),
        "autoencoder_parameters": count_parameters(autoencoder),
        "decoder_trainable_parameters": sum(
            parameter.numel() for parameter in decoder_parameters
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "training_samples": int(len(repository.observed_indices)),
        "validation_samples": int(len(validation_indices)),
        "last_epoch": last_epoch,
        "best_score": None if best_score == -math.inf else best_score,
        "target_score": target_score,
        "target_reached": best_score >= target_score,
        "stop_reason": stop_reason,
        "training_elapsed_seconds": time.perf_counter() - training_started,
        "latent_cache_on_device": device_cache is not None,
        "latent_cache_dtype": str(section.get("latent_cache_dtype", "float32")),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def load_context_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    repository: ContextRepository,
    device: torch.device,
) -> tuple[FullResolutionContextField, Autoencoder, ChannelShape, dict]:
    model = build_context_model(config, repository).to(device)
    autoencoder_path = (
        config["context"].get("autoencoder_checkpoint")
        or config["encoding"]["autoencoder_checkpoint"]
    )
    autoencoder, shape, _ = load_autoencoder_checkpoint(
        config, autoencoder_path, device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if "router_temperature" in checkpoint:
        model.router.temperature = float(checkpoint["router_temperature"])
    if "autoencoder" in checkpoint:
        autoencoder.load_state_dict(checkpoint["autoencoder"])
    model.eval()
    autoencoder.eval()
    return model, autoencoder, shape, checkpoint
