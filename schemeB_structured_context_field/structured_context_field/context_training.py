from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .angle_delay import ChannelShape, channel_to_shape_target, shape_to_channel
from .autoencoder import StructuredAngleDelayAutoencoder
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
from .context_model import StructuredContextField
from .data import balanced_limit, load_manifest, load_metadata, split_indices
from .losses import joint_power_loss, metric_aligned_channel_losses, weighted_sum
from .metrics import ChannelMetricAccumulator


def build_context_model(config: dict, repository: ContextRepository) -> StructuredContextField:
    section = config["context"]
    return StructuredContextField(
        repository.spectrum_latent_dim,
        repository.phase_latent_dim,
        repository.cell_count,
        repository.static_context_channels,
        repository.query_numeric_channels,
        token_channels=int(section["token_channels"]),
        token_hidden_channels=int(section["token_hidden_channels"]),
        context_base_channels=int(section["base_channels"]),
        context_feature_channels=int(section["context_feature_channels"]),
        environment_feature_channels=int(section["environment_feature_channels"]),
        station_embedding_channels=int(section["station_embedding_channels"]),
        fourier_bands=int(section["fourier_bands"]),
        query_width=int(section["query_width"]),
        query_blocks=int(section["query_blocks"]),
        adapter_width=int(section["adapter_width"]),
        dropout=float(section.get("dropout", 0.05)),
    )


def _tensor(value: np.ndarray, device: torch.device, dtype=None) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(value)).to(device)
    return tensor if dtype is None else tensor.to(dtype=dtype)


def _model_inputs(
    repository: ContextRepository,
    cell_id: int,
    context_indices: np.ndarray,
    query_indices: np.ndarray | None,
    device: torch.device,
    test: bool = False,
) -> dict[str, torch.Tensor]:
    query = repository.query_features(cell_id, query_indices, test=test)
    return {
        "point_features": _tensor(repository.point_features(context_indices), device),
        "point_flat_indices": _tensor(
            repository.flat_indices(cell_id, context_indices), device, torch.long
        ),
        "context_static": _tensor(repository.context_static[cell_id], device),
        "environment_bev": _tensor(repository.environment_bev[cell_id], device),
        "query_context_coordinates": _tensor(query["context_coordinates"], device),
        "query_environment_coordinates": _tensor(
            query["environment_coordinates"], device
        ),
        "query_numeric": _tensor(query["numeric"], device),
        "query_relative_xy": _tensor(query["relative_xy"], device),
    }


def _decode_predictions(
    outputs: dict[str, torch.Tensor],
    cell_id: int,
    repository: ContextRepository,
    autoencoder: StructuredAngleDelayAutoencoder,
    shape: ChannelShape,
) -> torch.Tensor:
    device = outputs["spectrum"].device
    spectrum_mean = torch.as_tensor(repository.encoded["spectrum_mean"], device=device)
    spectrum_std = torch.as_tensor(repository.encoded["spectrum_std"], device=device)
    phase_mean = torch.as_tensor(repository.encoded["phase_mean"], device=device)
    phase_std = torch.as_tensor(repository.encoded["phase_std"], device=device)
    power_mean = torch.as_tensor(repository.encoded["power_mean"], device=device)
    power_std = torch.as_tensor(repository.encoded["power_std"], device=device)
    spectrum = outputs["spectrum"].float() * spectrum_std + spectrum_mean
    phase = outputs["phase"].float() * phase_std + phase_mean
    log_power = outputs["power"].float() * power_std[int(cell_id)] + power_mean[int(cell_id)]
    prediction_shape = autoencoder.decode(spectrum, phase)
    return shape_to_channel(prediction_shape, log_power, shape)


@torch.no_grad()
def predict_indices(
    model: StructuredContextField,
    repository: ContextRepository,
    target_indices: np.ndarray | None,
    device: torch.device,
    amp: bool,
    test: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
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
    outage_probability = np.empty(count, dtype=np.float32)
    for cell_id in range(repository.cell_count):
        local = np.flatnonzero(cell_ids == cell_id)
        if not len(local):
            continue
        queries = selected_indices[local]
        context_indices = repository.context_indices(cell_id)
        inputs = _model_inputs(
            repository, cell_id, context_indices, queries, device, test=test
        )
        with autocast_context(device, amp):
            outputs = model(cell_id=cell_id, **inputs)
        spectrum[local] = outputs["spectrum"].float().cpu().numpy()
        phase[local] = outputs["phase"].float().cpu().numpy()
        power[local] = outputs["power"].float().cpu().numpy()
        outage_probability[local] = torch.sigmoid(outputs["outage_logit"]).float().cpu().numpy()
    return {
        "spectrum": spectrum,
        "phase": phase,
        "power": power,
        "outage_probability": outage_probability,
    }


@torch.no_grad()
def evaluate_context_model(
    model: StructuredContextField,
    autoencoder: StructuredAngleDelayAutoencoder,
    repository: ContextRepository,
    target_indices: np.ndarray,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
    outage_threshold: float,
    decode_batch_size: int,
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
    )[0]


@torch.no_grad()
def evaluate_context_thresholds(
    model: StructuredContextField,
    autoencoder: StructuredAngleDelayAutoencoder,
    repository: ContextRepository,
    target_indices: np.ndarray,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
    outage_thresholds: list[float],
    decode_batch_size: int,
) -> list[dict[str, float]]:
    thresholds = [float(value) for value in outage_thresholds]
    if not thresholds:
        raise ValueError("At least one outage threshold is required")
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Every outage threshold must lie in the open interval (0, 1)")
    outputs = predict_indices(model, repository, target_indices, device, amp)
    metadata = repository.metadata
    cell_ids = metadata["train_cells"][target_indices]
    true_outage = metadata["outage"][target_indices]
    target = repository.target_values(target_indices)
    nonzero = ~true_outage
    spectrum_mse = (
        float(np.mean((outputs["spectrum"][nonzero] - target["spectrum"][nonzero]) ** 2))
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
        target_channel = torch.from_numpy(
            np.array(channels[target_indices[selected]], copy=True)
        ).to(device)
        true_outage_tensor = torch.from_numpy(true_outage[selected]).to(device)
        probabilities = outputs["outage_probability"][selected]
        for threshold, metrics in zip(thresholds, accumulators):
            outage_tensor = torch.from_numpy(probabilities >= threshold).to(device)
            masked_prediction = prediction.masked_fill(
                outage_tensor[:, None, None, None], 0.0
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
                "spectrum_latent_mse_z": spectrum_mse,
                "phase_latent_mse_z": phase_mse,
                "power_mae_z": (
                    float(np.mean(np.abs(power_error))) if len(power_error) else 0.0
                ),
                "power_rmse_z": (
                    float(np.sqrt(np.mean(power_error**2))) if len(power_error) else 0.0
                ),
                "power_mae_log10": (
                    float(np.mean(np.abs(log_power_error)))
                    if len(log_power_error)
                    else 0.0
                ),
                "outage_accuracy": float(np.mean(predicted_outage == true_outage)),
                "outage_precision": precision,
                "outage_recall": recall,
                "outage_f1": 2.0 * precision * recall / max(precision + recall, 1e-30),
                "predicted_outages": int(predicted_outage.sum()),
                "samples": int(len(target_indices)),
            }
        )
        reports.append(result)
    return reports


def _save_checkpoint(
    path: Path,
    model: StructuredContextField,
    autoencoder: StructuredAngleDelayAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    config: dict,
    epoch: int,
    metrics: dict,
    best_score: float,
    epochs_without_improvement: int,
    joint: bool,
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
            "joint": joint,
        },
        path,
    )


def train_context_model(config: dict, resume: bool = False, joint: bool = False) -> dict:
    seed_everything(int(config["seed"]) + (10000 if joint else 0))
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

    section = config["joint"] if joint else config["context"]
    if joint:
        source_path = Path(section["context_checkpoint"])
        source = torch.load(source_path, map_location=device, weights_only=False)
        model.load_state_dict(source["model"])
        if "autoencoder" in source:
            autoencoder.load_state_dict(source["autoencoder"])
    autoencoder.requires_grad_(False)
    if joint:
        autoencoder.decoder.requires_grad_(True)
    parameters = list(model.parameters())
    if joint:
        parameters.extend(parameter for parameter in autoencoder.decoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(section["learning_rate"]),
        weight_decay=float(section.get("weight_decay", 1e-4)),
    )
    epochs = int(section["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(section.get("minimum_learning_rate", 1e-6)),
    )
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
            raise FileNotFoundError(f"Cannot resume because {checkpoint_path} does not exist")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        autoencoder.load_state_dict(checkpoint["autoencoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -math.inf))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        resumed_metrics = checkpoint.get("metrics", {})

    weights = section["loss_weights"]
    steps_per_epoch = int(section["steps_per_epoch"])
    accumulation = int(section.get("gradient_accumulation", 1))
    validation_interval = int(section.get("validation_interval", 5))
    patience = int(section.get("early_stopping_patience", 0))
    minimum_delta = float(section.get("minimum_delta", 1e-4))
    outage_threshold = float(section.get("outage_threshold", 0.999))
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
    final_metrics = resumed_metrics
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        started = time.perf_counter()
        model.train()
        autoencoder.decoder.train(joint)
        optimizer.zero_grad(set_to_none=True)
        sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        rng = np.random.default_rng(int(config["seed"]) + epoch * 1000003 + (91 if joint else 0))
        for step in range(steps_per_epoch):
            cell_id = int(rng.integers(repository.cell_count))
            targets = repository.sample_hole(
                rng,
                cell_id,
                float(section["hole_min_meters"]),
                float(section["hole_max_meters"]),
                int(section["minimum_targets"]),
                int(section["maximum_targets"]),
            )
            context_indices = repository.context_indices(cell_id, targets)
            inputs = _model_inputs(
                repository, cell_id, context_indices, targets, device, test=False
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
                    terms["power"] = functional.smooth_l1_loss(
                        outputs["power"][nonzero].float(), target_power[nonzero]
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
            sums["total"] += float(total.detach().cpu()) * accumulation
            batches += 1
        scheduler.step()
        train_metrics = {name: value / max(batches, 1) for name, value in sums.items()}
        should_validate = len(validation_indices) and (
            (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        )
        validation = (
            evaluate_context_model(
                model,
                autoencoder,
                repository,
                validation_indices,
                shape,
                device,
                amp,
                outage_threshold,
                decode_batch_size,
            )
            if should_validate
            else {}
        )
        final_metrics = validation or train_metrics
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - started,
        }
        append_jsonl(history_path, record)
        score = float(validation.get("score", -math.inf))
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
                joint,
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
            joint,
        )
        stage = "Joint" if joint else "Context"
        print(
            f"{stage} epoch={epoch + 1}/{epochs} train={train_metrics.get('total', 0.0):.6f} "
            f"score={validation.get('score', float('nan')):.6f} "
            f"seconds={record['elapsed_seconds']:.2f}",
            flush=True,
        )
        if patience > 0 and validation and epochs_without_improvement >= patience:
            print(f"{stage} early stopping at epoch {epoch + 1}", flush=True)
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
        joint,
    )
    summary = {
        "device": str(device),
        "joint": joint,
        "context_parameters": count_parameters(model),
        "autoencoder_parameters": count_parameters(autoencoder),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "training_samples": int(len(repository.observed_indices)),
        "validation_samples": int(len(validation_indices)),
        "last_epoch": last_epoch,
        "best_score": None if best_score == -math.inf else best_score,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def load_context_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    repository: ContextRepository,
    device: torch.device,
) -> tuple[StructuredContextField, StructuredAngleDelayAutoencoder, ChannelShape, dict]:
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
    if "autoencoder" in checkpoint:
        autoencoder.load_state_dict(checkpoint["autoencoder"])
    model.eval()
    autoencoder.eval()
    return model, autoencoder, shape, checkpoint
