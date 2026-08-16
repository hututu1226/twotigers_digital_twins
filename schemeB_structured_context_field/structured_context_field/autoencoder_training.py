from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .angle_delay import ChannelShape, channel_to_shape_target, shape_to_channel
from .autoencoder import StructuredAngleDelayAutoencoder
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
from .data import ChannelDataset, balanced_limit, load_manifest, load_metadata, split_indices
from .losses import (
    energy_weighted_angle_delay_mse,
    joint_power_loss,
    metric_aligned_channel_losses,
    weighted_sum,
)
from .metrics import ChannelMetricAccumulator


def build_autoencoder(config: dict, shape: ChannelShape) -> StructuredAngleDelayAutoencoder:
    section = config["autoencoder"]
    return StructuredAngleDelayAutoencoder(
        shape,
        spectrum_stem_channels=int(section["spectrum_stem_channels"]),
        phase_stem_channels=int(section["phase_stem_channels"]),
        spectrum_latent_channels=int(section["spectrum_latent_channels"]),
        phase_latent_channels=int(section["phase_latent_channels"]),
        spectrum_log_scale=float(section.get("spectrum_log_scale", 4.0)),
    )


def load_autoencoder_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[StructuredAngleDelayAutoencoder, ChannelShape, dict]:
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    model = build_autoencoder(config, shape).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    return model, shape, checkpoint


@torch.no_grad()
def evaluate_autoencoder(
    model: StructuredAngleDelayAutoencoder,
    loader: DataLoader,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    full_metrics = ChannelMetricAccumulator(shape)
    spectrum_metrics = ChannelMetricAccumulator(shape)
    square_error = 0.0
    elements = 0
    samples = 0
    for batch in loader:
        channel = batch["channel"].to(device, non_blocking=True)
        target_shape, log_power, outage = channel_to_shape_target(channel, shape)
        with autocast_context(device, amp):
            prediction_shape, spectrum, phase = model(target_shape)
            spectrum_shape = model.decode(spectrum, None)
        prediction = shape_to_channel(prediction_shape, log_power, shape, outage)
        spectrum_prediction = shape_to_channel(spectrum_shape, log_power, shape, outage)
        full_metrics.update(prediction, channel, outage)
        spectrum_metrics.update(spectrum_prediction, channel, outage)
        difference = prediction_shape.float() - target_shape.float()
        square_error += float(difference.square().sum().cpu())
        elements += difference.numel()
        samples += len(channel)
    result = full_metrics.compute()
    spectrum_result = spectrum_metrics.compute()
    result.update(
        {
            "angle_delay_mse": square_error / max(elements, 1),
            "spectrum_only_pas": spectrum_result["pas"],
            "spectrum_only_pdp": spectrum_result["pdp"],
            "spectrum_only_nmse": spectrum_result["nmse"],
            "spectrum_only_score": spectrum_result["score"],
            "samples": samples,
        }
    )
    return result


def _save_checkpoint(
    path: Path,
    model: StructuredAngleDelayAutoencoder,
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
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "spectrum_latent_dim": model.spectrum_latent_dim,
            "phase_latent_dim": model.phase_latent_dim,
        },
        path,
    )


def train_autoencoder(config: dict, resume: bool = False) -> dict:
    seed_everything(int(config["seed"]))
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    training_indices, validation_indices = split_indices(metadata, config)
    training_indices = training_indices[~metadata["outage"][training_indices]]
    validation_indices = validation_indices[~metadata["outage"][validation_indices]]
    runtime = config["runtime"]
    training_indices = balanced_limit(
        training_indices,
        runtime.get("train_limit"),
        [metadata["train_cells"]],
        int(config["seed"]),
    )
    validation_indices = balanced_limit(
        validation_indices,
        runtime.get("validation_limit"),
        [metadata["train_cells"]],
        int(config["seed"]) + 1,
    )
    channel_path = Path(config["data"]["root"]) / "Round2_Train_Channel.npy"
    train_dataset = ChannelDataset(channel_path, training_indices)
    validation_dataset = ChannelDataset(channel_path, validation_indices)
    section = config["autoencoder"]
    workers = worker_count(int(runtime.get("workers", -1)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(section["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(section.get("validation_batch_size", section["batch_size"])),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model = build_autoencoder(config, shape).to(device)
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
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -math.inf))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        resumed_metrics = checkpoint.get("metrics", {})

    weights = section["loss_weights"]
    accumulation = int(section.get("gradient_accumulation", 1))
    validation_interval = int(section.get("validation_interval", 1))
    patience = int(section.get("early_stopping_patience", 0))
    minimum_delta = float(section.get("minimum_delta", 1e-4))
    final_metrics = resumed_metrics
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            channel = batch["channel"].to(device, non_blocking=True)
            target_shape, log_power, outage = channel_to_shape_target(channel, shape)
            with autocast_context(device, amp):
                prediction_shape, spectrum, phase = model(target_shape)
                prediction = shape_to_channel(prediction_shape, log_power, shape)
                terms = {
                    "angle_delay": energy_weighted_angle_delay_mse(
                        prediction_shape,
                        target_shape,
                        shape,
                        float(section.get("energy_emphasis", 2.0)),
                        float(section.get("maximum_energy_weight", 16.0)),
                    ),
                    "joint_power": joint_power_loss(prediction_shape, target_shape, shape),
                    **metric_aligned_channel_losses(prediction, channel, shape),
                }
                spectrum_shape = model.decode(spectrum, None)
                spectrum_prediction = shape_to_channel(spectrum_shape, log_power, shape)
                spectrum_terms = metric_aligned_channel_losses(
                    spectrum_prediction, channel, shape
                )
                terms.update(
                    {
                        "spectrum_pas": spectrum_terms["pas"],
                        "spectrum_pdp": spectrum_terms["pdp"],
                        "spectrum_joint_power": joint_power_loss(
                            spectrum_shape, target_shape, shape
                        ),
                    }
                )
                total = weighted_sum(terms, weights) / accumulation
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite AE loss at epoch={epoch + 1}, batch={batch_index + 1}"
                )
            scaler.scale(total).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(section.get("gradient_clip", 1.0))
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
        should_validate = len(validation_dataset) and (
            (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        )
        validation = (
            evaluate_autoencoder(model, validation_loader, shape, device, amp)
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
            f"AE epoch={epoch + 1}/{epochs} train={train_metrics.get('total', 0.0):.6f} "
            f"score={validation.get('score', float('nan')):.6f} "
            f"seconds={record['elapsed_seconds']:.2f}",
            flush=True,
        )
        if patience > 0 and validation and epochs_without_improvement >= patience:
            print(f"AE early stopping at epoch {epoch + 1}", flush=True)
            break

    _save_checkpoint(
        output_dir / "final.pt",
        model,
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
        "parameters": count_parameters(model),
        "spectrum_latent_dim": model.spectrum_latent_dim,
        "phase_latent_dim": model.phase_latent_dim,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "last_epoch": last_epoch,
        "best_score": None if best_score == -math.inf else best_score,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary
