from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import training_loaders
from .metrics import composite_score, nmse, pas_accuracy, pdp_accuracy
from .models import Scheme1Model, Scheme2Model, build_model
from .transforms import (
    ChannelShape,
    angle_delay_to_channel,
    channel_to_angle_delay,
    normalize_angle_delay,
    scaled_angle_delay,
)
from .utils import append_jsonl, autocast_context, choose_device, count_parameters, seed_everything


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _target_representation(
    channel: torch.Tensor, shape: ChannelShape
) -> tuple[torch.Tensor, torch.Tensor]:
    angle_delay = channel_to_angle_delay(channel, shape)
    return normalize_angle_delay(angle_delay), angle_delay


def _spectral_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: ChannelShape,
) -> dict[str, torch.Tensor]:
    channel_nmse = nmse(prediction, target)
    pas = pas_accuracy(prediction, target, shape)
    pdp = pdp_accuracy(prediction, target)
    return {
        "nmse": channel_nmse,
        "pas_loss": 1.0 - pas,
        "pdp_loss": 1.0 - pdp,
        "pas": pas,
        "pdp": pdp,
    }


def _weighted_total(terms: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    total = next(iter(terms.values())).new_tensor(0.0)
    for name, value in terms.items():
        if name in weights:
            total = total + float(weights[name]) * value
    return total


def training_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    shape: ChannelShape,
    scheme: str,
    stage: str,
    weights: dict[str, float],
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    batch = _move(batch, device)
    channel = batch["channel"]
    target_shape, _ = _target_representation(channel, shape)
    nonzero = batch["outage"] < 0.5

    with autocast_context(device, amp):
        if scheme == "scheme1" and stage == "autoencoder":
            if not torch.any(nonzero):
                return None, {}
            predicted_shape, _ = model.autoencoder(target_shape)
            predicted_shape = predicted_shape.float()
            shape_loss = F.mse_loss(predicted_shape[nonzero], target_shape[nonzero])
            predicted_ad = scaled_angle_delay(predicted_shape[nonzero], batch["log_power"][nonzero])
            predicted_channel = angle_delay_to_channel(predicted_ad.float(), shape)
            spectral = _spectral_terms(predicted_channel, channel[nonzero], shape)
            terms = {
                "ad": shape_loss,
                "pas": spectral["pas_loss"],
                "pdp": spectral["pdp_loss"],
                "nmse": torch.log1p(spectral["nmse"]),
            }
        else:
            if scheme == "scheme1":
                outputs = model(
                    batch["position"], batch["map_tokens"], batch["cell"], target_shape
                )
            else:
                outputs = model(batch["position"], batch["map_tokens"], batch["cell"])
            terms = {
                "bs": F.cross_entropy(outputs["gate_logits"], batch["cell"]),
                "outage": F.binary_cross_entropy_with_logits(
                    outputs["selected_outage_logits"], batch["outage"]
                ),
            }
            if torch.any(nonzero):
                terms["power"] = F.smooth_l1_loss(
                    outputs["selected_power_z"][nonzero], batch["power_z"][nonzero]
                )
                predicted_shape = outputs["predicted_shape"].float()
                terms["ad"] = F.mse_loss(predicted_shape[nonzero], target_shape[nonzero])
                if scheme == "scheme1":
                    terms["latent"] = F.mse_loss(
                        outputs["predicted_latent"][nonzero], outputs["target_latent"][nonzero]
                    )
                predicted_ad = scaled_angle_delay(
                    predicted_shape[nonzero], outputs["log_power"][nonzero].float()
                )
                predicted_channel = angle_delay_to_channel(predicted_ad, shape)
                spectral = _spectral_terms(predicted_channel, channel[nonzero], shape)
                terms.update(
                    pas=spectral["pas_loss"],
                    pdp=spectral["pdp_loss"],
                    nmse=torch.log1p(spectral["nmse"]),
                )
            if scheme == "scheme1" and stage == "joint" and torch.any(nonzero):
                reconstructed_shape, _ = model.autoencoder(target_shape[nonzero])
                terms["reconstruction"] = F.mse_loss(
                    reconstructed_shape.float(), target_shape[nonzero]
                )
        total = _weighted_total(terms, weights)
    scalars = {name: float(value.detach().cpu()) for name, value in terms.items()}
    scalars["total"] = float(total.detach().cpu())
    return total, scalars


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    shape: ChannelShape,
    scheme: str,
    stage: str,
    device: torch.device,
    outage_threshold: float,
) -> dict[str, float]:
    model.eval()
    pas_sum = 0.0
    pdp_sum = 0.0
    nonzero_count = 0
    nmse_numerator = 0.0
    nmse_denominator = 0.0
    gate_correct = 0
    outage_correct = 0
    sample_count = 0
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        target = batch["channel"]
        nonzero = batch["outage"] < 0.5
        if scheme == "scheme1" and stage == "autoencoder":
            target_shape, _ = _target_representation(target, shape)
            predicted_shape, _ = model.autoencoder(target_shape)
            prediction = angle_delay_to_channel(
                scaled_angle_delay(predicted_shape.float(), batch["log_power"]), shape
            )
            prediction = prediction.masked_fill((~nonzero)[:, None, None, None], 0.0)
        else:
            generated = model.generate(batch["position"], batch["map_tokens"], outage_threshold)
            prediction = generated["channel"]
            gate_correct += int((generated["route"] == batch["cell"]).sum().item())
            predicted_outage = generated["outage_probability"] >= outage_threshold
            outage_correct += int((predicted_outage == (~nonzero)).sum().item())
        if torch.any(nonzero):
            count = int(nonzero.sum().item())
            pas_sum += float(pas_accuracy(prediction[nonzero], target[nonzero], shape).cpu()) * count
            pdp_sum += float(pdp_accuracy(prediction[nonzero], target[nonzero]).cpu()) * count
            nonzero_count += count
        nmse_numerator += float((prediction - target).abs().square().sum().cpu())
        nmse_denominator += float(target.abs().square().sum().cpu())
        sample_count += len(target)
    pas = pas_sum / max(nonzero_count, 1)
    pdp = pdp_sum / max(nonzero_count, 1)
    channel_nmse = nmse_numerator / max(nmse_denominator, 1e-30)
    score = 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + channel_nmse)
    metrics = {
        "pas": pas,
        "pdp": pdp,
        "nmse": channel_nmse,
        "score": score,
        "samples": sample_count,
    }
    if not (scheme == "scheme1" and stage == "autoencoder"):
        metrics["gate_accuracy"] = gate_correct / max(sample_count, 1)
        metrics["outage_accuracy"] = outage_correct / max(sample_count, 1)
    return metrics


def _checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    phase_index: int,
    phase_name: str,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "phase_index": phase_index,
            "phase_name": phase_name,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def train_from_config(config: dict, device_override: str | None = None, resume: str | None = None) -> Path:
    seed_everything(int(config.get("seed", 2026)))
    training_config = config["training"]
    device = choose_device(device_override or training_config.get("device", "auto"))
    amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    model, shape = build_model(config)
    model.to(device)
    train_loader, validation_loader = training_loaders(config, device)
    print(
        f"scheme={config['scheme']} device={device} parameters={count_parameters(model):,} "
        f"train_samples={len(train_loader.dataset)} val_samples={len(validation_loader.dataset)}"
    )

    resume_state = None
    if resume:
        resume_state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
        print(f"Loaded checkpoint: {resume}")

    phases = training_config["phases"]
    history_path = output_dir / "history.jsonl"
    best_final_score = -math.inf
    start_phase = int(resume_state["phase_index"]) if resume_state else 0
    for phase_index, phase in enumerate(phases):
        if phase_index < start_phase:
            continue
        stage = phase["name"]
        model.configure_stage(stage)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(phase["learning_rate"]),
            weight_decay=float(phase.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(phase["epochs"]))
        )
        start_epoch = 0
        if resume_state and phase_index == start_phase and resume_state["phase_name"] == stage:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            start_epoch = int(resume_state["epoch"]) + 1
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        best_phase_score = -math.inf
        epochs_without_improvement = 0
        patience = int(phase.get("early_stopping_patience", 0))
        minimum_delta = float(phase.get("early_stopping_min_delta", 0.0))
        print(f"Starting phase={stage} epochs={phase['epochs']} from_epoch={start_epoch}")
        for epoch in range(start_epoch, int(phase["epochs"])):
            model.train()
            started = time.perf_counter()
            aggregate: dict[str, float] = {}
            steps = 0
            for raw_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, scalars = training_step(
                    model,
                    raw_batch,
                    shape,
                    config["scheme"],
                    stage,
                    config["loss_weights"],
                    device,
                    amp,
                )
                if loss is None:
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(phase.get("gradient_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                for key, value in scalars.items():
                    aggregate[key] = aggregate.get(key, 0.0) + value
                steps += 1
            scheduler.step()
            train_metrics = {key: value / max(steps, 1) for key, value in aggregate.items()}
            validation = evaluate_model(
                model,
                validation_loader,
                shape,
                config["scheme"],
                stage,
                device,
                float(training_config.get("outage_threshold", 0.5)),
            )
            elapsed = time.perf_counter() - started
            record: dict[str, Any] = {
                "phase": stage,
                "phase_index": phase_index,
                "epoch": epoch,
                "seconds": elapsed,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation,
            }
            append_jsonl(history_path, record)
            print(
                f"phase={stage} epoch={epoch + 1}/{phase['epochs']} "
                f"loss={train_metrics.get('total', float('nan')):.5f} "
                f"score={validation['score']:.5f} seconds={elapsed:.2f}"
            )
            _checkpoint(
                output_dir / "last.pt", model, optimizer, scheduler, config,
                phase_index, stage, epoch, validation
            )
            if validation["score"] > best_phase_score + minimum_delta:
                best_phase_score = validation["score"]
                epochs_without_improvement = 0
                _checkpoint(
                    output_dir / f"best_{stage}.pt", model, optimizer, scheduler, config,
                    phase_index, stage, epoch, validation
                )
            else:
                epochs_without_improvement += 1
            if phase_index == len(phases) - 1 and validation["score"] > best_final_score:
                best_final_score = validation["score"]
                _checkpoint(
                    output_dir / "best.pt", model, optimizer, scheduler, config,
                    phase_index, stage, epoch, validation
                )
            if patience > 0 and epochs_without_improvement >= patience:
                print(
                    f"Early stopping phase={stage}: no score improvement greater than "
                    f"{minimum_delta} for {patience} epochs"
                )
                break
        resume_state = None
    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "config": config}, final_path)
    print(f"Training complete: {final_path}")
    return final_path
