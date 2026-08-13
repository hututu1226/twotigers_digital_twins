from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .angle_delay import ChannelShape, channel_to_shape_target, shape_to_channel
from .autoencoder import AngleDelayAutoencoder
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
from .metrics import ChannelMetricAccumulator, nmse, pas_accuracy, pdp_accuracy


def build_autoencoder(config: dict, shape: ChannelShape) -> AngleDelayAutoencoder:
    section = config["autoencoder"]
    return AngleDelayAutoencoder(
        shape=shape,
        base_channels=int(section["base_channels"]),
        latent_dim=int(section["latent_dim"]),
    )


def load_autoencoder_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[AngleDelayAutoencoder, ChannelShape, dict]:
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    model = build_autoencoder(config, shape).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    return model, shape, checkpoint


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


@torch.no_grad()
def evaluate_autoencoder(
    model: AngleDelayAutoencoder,
    loader: DataLoader,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    metrics = ChannelMetricAccumulator(shape)
    shape_square_error = 0.0
    shape_elements = 0
    sample_count = 0
    for batch in loader:
        channel = batch["channel"].to(device, non_blocking=True)
        target_shape, log_power, outage = channel_to_shape_target(channel, shape)
        with autocast_context(device, amp):
            prediction_shape, _ = model(target_shape)
        prediction = shape_to_channel(prediction_shape.float(), log_power, shape, outage)
        metrics.update(prediction, channel, outage)
        nonzero = ~outage
        if torch.any(nonzero):
            difference = prediction_shape[nonzero].float() - target_shape[nonzero]
            shape_square_error += float(difference.square().sum().cpu())
            shape_elements += difference.numel()
        sample_count += len(channel)
    result = metrics.compute()
    result.update(
        {
            "angle_delay_mse": shape_square_error / max(shape_elements, 1),
            "samples": sample_count,
        }
    )
    return result


def _save_checkpoint(
    path: Path,
    model: AngleDelayAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    config: dict,
    epoch: int,
    metrics: dict,
    best_score: float = -math.inf,
    epochs_without_improvement: int = 0,
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
        },
        path,
    )


def train_autoencoder(config: dict, resume: bool = False) -> dict:
    seed_everything(int(config["seed"]))
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    manifest = load_manifest(config)
    metadata = load_metadata(config)
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
    data_root = Path(config["data"]["root"])
    channel_path = data_root / "Round2_Train_Channel.npy"
    train_dataset = ChannelDataset(channel_path, training_indices)
    validation_dataset = ChannelDataset(channel_path, validation_indices)
    workers = worker_count(int(runtime.get("workers", -1)))
    section = config["autoencoder"]
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
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", checkpoint.get("metrics", {}).get("score", -math.inf)))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        resumed_metrics = checkpoint.get("metrics", {})

    weights = section["loss_weights"]
    accumulation = int(section.get("gradient_accumulation", 1))
    patience = int(section.get("early_stopping_patience", 0))
    minimum_delta = float(section.get("minimum_delta", 1e-4))
    validation_interval = int(section.get("validation_interval", 1))
    final_metrics: dict = resumed_metrics
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
            if torch.all(outage):
                continue
            with autocast_context(device, amp):
                prediction_shape, _ = model(target_shape)
                shape_loss = functional.mse_loss(prediction_shape.float(), target_shape)
                prediction = shape_to_channel(prediction_shape.float(), log_power, shape)
                spectral = _spectral_losses(prediction, channel, shape)
                terms = {"angle_delay": shape_loss, **spectral}
                total = _weighted_loss(terms, weights) / accumulation
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite AE loss at epoch={epoch + 1}, batch={batch_index + 1}: "
                    f"{float(total.detach().cpu())}"
                )
            scaler.scale(total).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
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
        should_validate = len(validation_dataset) and (
            (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        )
        validation = (
            evaluate_autoencoder(model, validation_loader, shape, device, amp) if should_validate else {}
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
        checkpoint_metrics = validation or train_metrics
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
            checkpoint_metrics,
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
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "last_epoch": last_epoch,
        "best_score": None if best_score == -math.inf else best_score,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary
