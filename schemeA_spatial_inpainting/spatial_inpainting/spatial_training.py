from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .angle_delay import ChannelShape, shape_to_channel
from .autoencoder import AngleDelayAutoencoder
from .autoencoder_training import load_autoencoder_checkpoint
from .config import (
    append_jsonl,
    autocast_context,
    choose_device,
    count_parameters,
    make_grad_scaler,
    save_json,
    seed_everything,
    worker_count,
)
from .data import (
    DynamicHoleDataset,
    SpatialRepository,
    collate_dynamic_holes,
    load_manifest,
    load_metadata,
    split_indices,
)
from .metrics import ChannelMetricAccumulator, nmse, pas_accuracy, pdp_accuracy
from .unet import SpatialUNet, pad_to_multiple, unpad


def build_spatial_model(config: dict, input_channels: int) -> SpatialUNet:
    section = config["spatial"]
    return SpatialUNet(
        input_channels=input_channels,
        latent_dim=int(config["autoencoder"]["latent_dim"]),
        base_channels=int(section["base_channels"]),
        dropout=float(section.get("dropout", 0.05)),
    )


def _gather(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch_index = batch["target_batch"]
    rows = batch["target_rows"]
    columns = batch["target_columns"]
    return {
        name: value[batch_index, :, rows, columns]
        for name, value in outputs.items()
    }


def _spectral_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: ChannelShape,
) -> dict[str, torch.Tensor]:
    channel_nmse = nmse(prediction, target)
    return {
        "pas": 1.0 - pas_accuracy(prediction, target, shape),
        "pdp": 1.0 - pdp_accuracy(prediction, target),
        "nmse": torch.log1p(channel_nmse),
    }


def _weighted_loss(terms: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    total = next(iter(terms.values())).new_zeros(())
    for name, value in terms.items():
        total = total + float(weights.get(name, 0.0)) * value
    return total


def _decode_targets(
    predicted_latent_z: torch.Tensor,
    predicted_power_z: torch.Tensor,
    target_cells: torch.Tensor,
    repository: SpatialRepository,
    autoencoder: AngleDelayAutoencoder,
    shape: ChannelShape,
) -> torch.Tensor:
    latent_mean = torch.as_tensor(repository.encoded["latent_mean"], device=predicted_latent_z.device)
    latent_std = torch.as_tensor(repository.encoded["latent_std"], device=predicted_latent_z.device)
    power_mean = torch.as_tensor(repository.encoded["power_mean"], device=predicted_power_z.device)
    power_std = torch.as_tensor(repository.encoded["power_std"], device=predicted_power_z.device)
    latent = predicted_latent_z.float() * latent_std + latent_mean
    log_power = predicted_power_z.float().squeeze(1) * power_std[target_cells] + power_mean[target_cells]
    predicted_shape = autoencoder.decode(latent)
    return shape_to_channel(predicted_shape, log_power, shape)


@torch.no_grad()
def predict_grid_points(
    model: SpatialUNet,
    repository: SpatialRepository,
    cell_ids: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    count = len(cell_ids)
    latent = np.empty((count, repository.latent_dim), dtype=np.float32)
    power = np.empty(count, dtype=np.float32)
    outage_probability = np.empty(count, dtype=np.float32)
    for cell_id in range(repository.cell_count):
        selected = np.flatnonzero(cell_ids == cell_id)
        if not len(selected):
            continue
        # Hide exact target pairs, not the Cartesian product of rows and columns.
        model_input = repository.full_input(cell_id, rows[selected], columns[selected])
        value = torch.from_numpy(model_input).unsqueeze(0).to(device)
        padded, original_shape = pad_to_multiple(value)
        with autocast_context(device, amp):
            outputs = model(padded)
        outputs = {name: unpad(result, original_shape) for name, result in outputs.items()}
        target_rows = torch.from_numpy(rows[selected]).to(device)
        target_columns = torch.from_numpy(columns[selected]).to(device)
        latent[selected] = outputs["latent"][0, :, target_rows, target_columns].T.float().cpu().numpy()
        power[selected] = outputs["power"][0, 0, target_rows, target_columns].float().cpu().numpy()
        outage_probability[selected] = torch.sigmoid(
            outputs["outage_logit"][0, 0, target_rows, target_columns]
        ).float().cpu().numpy()
    return latent, power, outage_probability


@torch.no_grad()
def evaluate_spatial_model(
    model: SpatialUNet,
    autoencoder: AngleDelayAutoencoder,
    repository: SpatialRepository,
    target_indices: np.ndarray,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
    outage_threshold: float,
    decode_batch_size: int,
) -> dict[str, float]:
    metadata = repository.metadata
    cell_ids = metadata["train_cells"][target_indices]
    rows = metadata["train_rows"][target_indices]
    columns = metadata["train_columns"][target_indices]
    latent_z, power_z, outage_probability = predict_grid_points(
        model, repository, cell_ids, rows, columns, device, amp
    )
    true_outage = metadata["outage"][target_indices]
    predicted_outage = outage_probability >= outage_threshold
    target_latent_z = repository.normalized_latent(target_indices)
    target_power_z = repository.normalized_power(target_indices)
    nonzero = ~true_outage
    latent_mse = float(np.mean((latent_z[nonzero] - target_latent_z[nonzero]) ** 2)) if np.any(nonzero) else 0.0
    power_error = power_z[nonzero] - target_power_z[nonzero]
    power_mae = float(np.mean(np.abs(power_error))) if len(power_error) else 0.0
    power_rmse = float(np.sqrt(np.mean(power_error**2))) if len(power_error) else 0.0
    cell_power_std = repository.encoded["power_std"][cell_ids[nonzero]]
    log_power_error = power_error * cell_power_std
    log_power_mae = float(np.mean(np.abs(log_power_error))) if len(log_power_error) else 0.0
    log_power_rmse = (
        float(np.sqrt(np.mean(log_power_error**2))) if len(log_power_error) else 0.0
    )
    true_positive = int(np.sum(predicted_outage & true_outage))
    false_positive = int(np.sum(predicted_outage & ~true_outage))
    false_negative = int(np.sum(~predicted_outage & true_outage))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    outage_f1 = 2.0 * precision * recall / max(precision + recall, 1e-30)

    metrics = ChannelMetricAccumulator(shape)
    channels = np.load(Path(repository.config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r")
    for start in range(0, len(target_indices), decode_batch_size):
        stop = min(start + decode_batch_size, len(target_indices))
        selected = slice(start, stop)
        latent_tensor = torch.from_numpy(latent_z[selected]).to(device)
        power_tensor = torch.from_numpy(power_z[selected]).to(device)
        cell_tensor = torch.from_numpy(cell_ids[selected]).to(device)
        prediction = _decode_targets(
            latent_tensor, power_tensor[:, None], cell_tensor, repository, autoencoder, shape
        )
        outage_tensor = torch.from_numpy(predicted_outage[selected]).to(device)
        prediction = prediction.masked_fill(outage_tensor[:, None, None, None], 0.0)
        target = torch.from_numpy(np.array(channels[target_indices[selected]], copy=True)).to(device)
        true_outage_tensor = torch.from_numpy(true_outage[selected]).to(device)
        metrics.update(prediction, target, true_outage_tensor)
    result = metrics.compute()
    result.update(
        {
            "latent_mse": latent_mse,
            "power_mae_z": power_mae,
            "power_rmse_z": power_rmse,
            "power_mae_log10": log_power_mae,
            "power_rmse_log10": log_power_rmse,
            "outage_accuracy": float(np.mean(predicted_outage == true_outage)),
            "outage_f1": outage_f1,
            "outage_precision": precision,
            "outage_recall": recall,
            "samples": int(len(target_indices)),
        }
    )
    return result


@torch.no_grad()
def scan_outage_thresholds(
    model: SpatialUNet,
    autoencoder: AngleDelayAutoencoder,
    repository: SpatialRepository,
    target_indices: np.ndarray,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
    thresholds: list[float],
    decode_batch_size: int,
) -> list[dict[str, float]]:
    """Evaluate many outage thresholds while sharing U-Net and AE inference."""
    metadata = repository.metadata
    cell_ids = metadata["train_cells"][target_indices]
    rows = metadata["train_rows"][target_indices]
    columns = metadata["train_columns"][target_indices]
    latent_z, power_z, outage_probability = predict_grid_points(
        model, repository, cell_ids, rows, columns, device, amp
    )
    true_outage = metadata["outage"][target_indices]
    target_latent_z = repository.normalized_latent(target_indices)
    target_power_z = repository.normalized_power(target_indices)
    nonzero = ~true_outage
    latent_mse = (
        float(np.mean((latent_z[nonzero] - target_latent_z[nonzero]) ** 2))
        if np.any(nonzero)
        else 0.0
    )
    power_error = power_z[nonzero] - target_power_z[nonzero]
    power_mae = float(np.mean(np.abs(power_error))) if len(power_error) else 0.0
    power_rmse = float(np.sqrt(np.mean(power_error**2))) if len(power_error) else 0.0
    cell_power_std = repository.encoded["power_std"][cell_ids[nonzero]]
    log_power_error = power_error * cell_power_std
    log_power_mae = float(np.mean(np.abs(log_power_error))) if len(log_power_error) else 0.0
    log_power_rmse = (
        float(np.sqrt(np.mean(log_power_error**2))) if len(log_power_error) else 0.0
    )

    predicted_outages = [outage_probability >= threshold for threshold in thresholds]
    accumulators = [ChannelMetricAccumulator(shape) for _ in thresholds]
    channels = np.load(
        Path(repository.config["data"]["root"]) / "Round2_Train_Channel.npy",
        mmap_mode="r",
    )
    for start in range(0, len(target_indices), decode_batch_size):
        stop = min(start + decode_batch_size, len(target_indices))
        selected = slice(start, stop)
        base_prediction = _decode_targets(
            torch.from_numpy(latent_z[selected]).to(device),
            torch.from_numpy(power_z[selected, None]).to(device),
            torch.from_numpy(cell_ids[selected]).to(device),
            repository,
            autoencoder,
            shape,
        )
        target = torch.from_numpy(np.array(channels[target_indices[selected]], copy=True)).to(device)
        true_outage_tensor = torch.from_numpy(true_outage[selected]).to(device)
        for accumulator, predicted_outage in zip(accumulators, predicted_outages):
            predicted_outage_tensor = torch.from_numpy(predicted_outage[selected]).to(device)
            prediction = base_prediction.masked_fill(
                predicted_outage_tensor[:, None, None, None], 0.0
            )
            accumulator.update(prediction, target, true_outage_tensor)

    results: list[dict[str, float]] = []
    for threshold, predicted_outage, accumulator in zip(
        thresholds, predicted_outages, accumulators
    ):
        true_positive = int(np.sum(predicted_outage & true_outage))
        false_positive = int(np.sum(predicted_outage & ~true_outage))
        false_negative = int(np.sum(~predicted_outage & true_outage))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        outage_f1 = 2.0 * precision * recall / max(precision + recall, 1e-30)
        result = accumulator.compute()
        result.update(
            {
                "threshold": float(threshold),
                "latent_mse": latent_mse,
                "power_mae_z": power_mae,
                "power_rmse_z": power_rmse,
                "power_mae_log10": log_power_mae,
                "power_rmse_log10": log_power_rmse,
                "outage_accuracy": float(np.mean(predicted_outage == true_outage)),
                "outage_f1": outage_f1,
                "outage_precision": precision,
                "outage_recall": recall,
                "predicted_outages": int(predicted_outage.sum()),
                "samples": int(len(target_indices)),
            }
        )
        results.append(result)
    return results


def _save_checkpoint(
    path: Path,
    model: SpatialUNet,
    autoencoder: AngleDelayAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    config: dict,
    epoch: int,
    metrics: dict,
    best_score: float,
    epochs_without_improvement: int = 0,
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
        },
        path,
    )


def train_spatial_model(config: dict, resume: bool = False) -> dict:
    seed_everything(int(config["seed"]))
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    training_indices, validation_indices = split_indices(metadata, config)
    repository = SpatialRepository(config, training_indices)
    training_indices = repository.observed_indices
    available = repository.encoded.get(
        "available", np.ones(len(metadata["train_cells"]), dtype=bool)
    )
    validation_indices = validation_indices[available[validation_indices]]
    dataset = DynamicHoleDataset(config, repository, training_indices)
    workers = worker_count(int(config["runtime"].get("spatial_workers", 0)))
    loader = DataLoader(
        dataset,
        batch_size=int(config["spatial"]["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=collate_dynamic_holes,
    )
    model = build_spatial_model(config, repository.input_channels).to(device)
    autoencoder, loaded_shape, _ = load_autoencoder_checkpoint(
        config, config["encoding"]["autoencoder_checkpoint"], device
    )
    if loaded_shape != shape:
        raise ValueError("Autoencoder and preprocessing channel shapes do not match")
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    section = config["spatial"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        for checkpoint_name in ("best.pt", "last.pt", "final.pt"):
            (output_dir / checkpoint_name).unlink(missing_ok=True)
    start_epoch = 0
    best_score = -math.inf
    epochs_without_improvement = 0
    resumed_metrics: dict = {}
    if resume:
        last_checkpoint = output_dir / "last.pt"
        if not last_checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume because {last_checkpoint} does not exist")
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "autoencoder" in checkpoint:
            autoencoder.load_state_dict(checkpoint["autoencoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -math.inf))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        resumed_metrics = checkpoint.get("metrics", {})

    weights = section["loss_weights"]
    accumulation = int(section.get("gradient_accumulation", 1))
    validation_interval = int(section.get("validation_interval", 5))
    patience = int(section.get("early_stopping_patience", 0))
    minimum_delta = float(section.get("minimum_delta", 1e-4))
    outage_threshold = float(section.get("outage_threshold", 0.5))
    outage_count = int(metadata["outage"][training_indices].sum())
    nonoutage_count = int(len(training_indices) - outage_count)
    configured_positive_weight = float(section.get("outage_positive_weight", 0.0))
    outage_positive_weight = (
        configured_positive_weight
        if configured_positive_weight > 0
        else nonoutage_count / max(outage_count, 1)
    )
    outage_positive_weight_tensor = torch.tensor(outage_positive_weight, device=device)
    decode_batch_size = int(section.get("validation_decode_batch_size", 8))
    final_metrics: dict = resumed_metrics
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        dataset.set_epoch(epoch)
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for batch_index, raw_batch in enumerate(loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items()}
            with autocast_context(device, amp):
                outputs = model(batch["input"])
                gathered = _gather(outputs, batch)
                outage_target = batch["target_outage"]
                terms: dict[str, torch.Tensor] = {
                    "outage": functional.binary_cross_entropy_with_logits(
                        gathered["outage_logit"].squeeze(1),
                        outage_target,
                        pos_weight=outage_positive_weight_tensor,
                    )
                }
                nonzero = outage_target < 0.5
                if torch.any(nonzero):
                    terms["latent"] = functional.mse_loss(
                        gathered["latent"][nonzero].float(), batch["target_latent"][nonzero]
                    )
                    terms["power"] = functional.smooth_l1_loss(
                        gathered["power"][nonzero].float().squeeze(1), batch["target_power"][nonzero]
                    )
                    target_cells = batch["cell_id"][batch["target_batch"]][nonzero]
                    prediction = _decode_targets(
                        gathered["latent"][nonzero],
                        gathered["power"][nonzero],
                        target_cells,
                        repository,
                        autoencoder,
                        shape,
                    )
                    terms.update(
                        _spectral_losses(prediction, batch["target_channel"][nonzero], shape)
                    )
                total = _weighted_loss(terms, weights) / accumulation
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite spatial loss at epoch={epoch + 1}, batch={batch_index + 1}: "
                    f"{float(total.detach().cpu())}"
                )
            scaler.scale(total).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(section.get("gradient_clip", 1.0)))
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
            evaluate_spatial_model(
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
        print(
            f"Spatial epoch={epoch + 1}/{epochs} train={train_metrics.get('total', 0.0):.6f} "
            f"score={validation.get('score', float('nan')):.6f} "
            f"seconds={record['elapsed_seconds']:.2f}",
            flush=True,
        )
        if patience > 0 and validation and epochs_without_improvement >= patience:
            print(f"Spatial early stopping at epoch {epoch + 1}", flush=True)
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
        "spatial_parameters": count_parameters(model),
        "autoencoder_parameters": count_parameters(autoencoder),
        "input_channels": repository.input_channels,
        "outage_positive_weight": outage_positive_weight,
        "training_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "last_epoch": last_epoch,
        "best_score": None if best_score == -math.inf else best_score,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def load_spatial_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    repository: SpatialRepository,
    device: torch.device,
) -> tuple[SpatialUNet, AngleDelayAutoencoder, ChannelShape, dict]:
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    model = build_spatial_model(config, repository.input_channels).to(device)
    autoencoder, _, _ = load_autoencoder_checkpoint(
        config, config["encoding"]["autoencoder_checkpoint"], device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if "autoencoder" in checkpoint:
        autoencoder.load_state_dict(checkpoint["autoencoder"])
    model.eval()
    autoencoder.eval()
    return model, autoencoder, shape, checkpoint
