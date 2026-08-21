from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .angle_delay import (
    ChannelShape,
    channel_to_shape_target,
    normalize_angle_delay,
    shape_to_channel,
)
from .autoencoder import (
    FactorizedResidualAutoencoder,
    MetricHighFidelityAutoencoder,
    StructuredAngleDelayAutoencoder,
)
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
    ChannelDataset,
    balanced_limit,
    load_manifest,
    load_metadata,
    split_indices,
)
from .losses import (
    complex_coherence_loss,
    energy_weighted_angle_delay_mse,
    energy_weighted_complex_direction_loss,
    joint_power_loss,
    metric_aligned_channel_losses,
    weighted_sum,
)
from .metrics import ChannelMetricAccumulator


Autoencoder = (
    StructuredAngleDelayAutoencoder
    | MetricHighFidelityAutoencoder
    | FactorizedResidualAutoencoder
)


def build_autoencoder(config: dict, shape: ChannelShape) -> Autoencoder:
    section = config["autoencoder"]
    architecture = section.get("architecture", "structured_v2")
    if architecture == "factorized_residual_v4":
        return FactorizedResidualAutoencoder(
            shape,
            spectrum_stem_channels=int(section["spectrum_stem_channels"]),
            phase_stem_channels=int(section["phase_stem_channels"]),
            spectrum_latent_channels=int(section["spectrum_latent_channels"]),
            phase_latent_channels=int(section["phase_latent_channels"]),
            residual_blocks=int(section.get("residual_blocks", 3)),
            spectrum_log_scale=float(section.get("spectrum_log_scale", 4.0)),
            detail_hidden_channels=int(section.get("detail_hidden_channels", 64)),
            spectrum_decoder_channels=int(section.get("spectrum_decoder_channels", 64)),
            detail_decoder_channels=int(section.get("detail_decoder_channels", 64)),
            detail_gain=float(section.get("detail_gain", 2.0)),
            maximum_log_power=float(section.get("maximum_log_power", 12.0)),
            envelope_floor=float(section.get("envelope_floor", 1e-4)),
        )
    if architecture == "metric_high_fidelity_v3":
        return MetricHighFidelityAutoencoder(
            shape,
            spectrum_stem_channels=int(section["spectrum_stem_channels"]),
            phase_stem_channels=int(section["phase_stem_channels"]),
            spectrum_latent_channels=int(section["spectrum_latent_channels"]),
            phase_latent_channels=int(section["phase_latent_channels"]),
            residual_blocks=int(section.get("residual_blocks", 2)),
            spectrum_log_scale=float(section.get("spectrum_log_scale", 4.0)),
        )
    if architecture != "structured_v2":
        raise ValueError(f"Unsupported autoencoder architecture: {architecture}")
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
) -> tuple[
    Autoencoder,
    ChannelShape,
    dict,
]:
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    model = build_autoencoder(config, shape).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    return model, shape, checkpoint


@torch.no_grad()
def evaluate_autoencoder(
    model: Autoencoder,
    loader: DataLoader,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
    detail_scale: float = 1.0,
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
            if isinstance(model, FactorizedResidualAutoencoder):
                prediction_shape, spectrum, phase = model(target_shape, detail_scale)
            else:
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
            "detail_gain_over_coarse": result["score"] - spectrum_result["score"],
            "samples": samples,
        }
    )
    return result


@torch.no_grad()
def evaluate_autoencoder_ablations(
    model: Autoencoder,
    loader: DataLoader,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
) -> dict[str, object]:
    model.eval()
    accumulators = {
        "full": ChannelMetricAccumulator(shape),
        "coarse_only": ChannelMetricAccumulator(shape),
        "shuffled_detail": ChannelMetricAccumulator(shape),
    }
    samples = 0
    for batch in loader:
        channel = batch["channel"].to(device, non_blocking=True)
        target_shape, log_power, outage = channel_to_shape_target(channel, shape)
        with autocast_context(device, amp):
            spectrum, detail = model.encode(target_shape)
            full_shape = model.decode(spectrum, detail)
            coarse_shape = model.decode(spectrum, None)
            shuffled_detail = detail.roll(1, dims=0) if len(detail) > 1 else -detail
            shuffled_shape = model.decode(spectrum, shuffled_detail)
        predictions = {
            "full": shape_to_channel(full_shape, log_power, shape, outage),
            "coarse_only": shape_to_channel(coarse_shape, log_power, shape, outage),
            "shuffled_detail": shape_to_channel(
                shuffled_shape, log_power, shape, outage
            ),
        }
        for name, prediction in predictions.items():
            accumulators[name].update(prediction, channel, outage)
        samples += len(channel)
    metrics = {name: value.compute() for name, value in accumulators.items()}
    full_score = float(metrics["full"]["score"])
    coarse_score = float(metrics["coarse_only"]["score"])
    shuffled_score = float(metrics["shuffled_detail"]["score"])
    return {
        "samples": samples,
        "metrics": metrics,
        "detail_gain": full_score - coarse_score,
        "shuffle_drop": full_score - shuffled_score,
    }


def _save_checkpoint(
    path: Path,
    model: Autoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    config: dict,
    epoch: int,
    metrics: dict,
    best_score: float,
    epochs_without_improvement: int,
    best_spectrum_score: float = -math.inf,
    early_stopping_score: float = -math.inf,
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
            "best_spectrum_score": best_spectrum_score,
            "early_stopping_score": early_stopping_score,
            "epochs_without_improvement": epochs_without_improvement,
            "spectrum_latent_dim": model.spectrum_latent_dim,
            "phase_latent_dim": model.phase_latent_dim,
        },
        path,
    )


def autoencoder_training_stage(section: dict, epoch: int) -> str:
    coarse_epochs = max(0, int(section.get("coarse_pretrain_epochs", 0)))
    detail_epochs = max(0, int(section.get("detail_pretrain_epochs", 0)))
    if epoch < coarse_epochs:
        return "coarse"
    if epoch < coarse_epochs + detail_epochs:
        return "detail"
    return "joint"


def _spectrum_power_terms(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {
        "spectrum_power": functional.smooth_l1_loss(prediction, target),
        "spectrum_angle_marginal": functional.smooth_l1_loss(
            prediction.mean(dim=-1), target.mean(dim=-1)
        ),
        "spectrum_delay_marginal": functional.smooth_l1_loss(
            prediction.mean(dim=(2, 3)), target.mean(dim=(2, 3))
        ),
    }


def factorized_autoencoder_batch(
    model: FactorizedResidualAutoencoder,
    target_shape: torch.Tensor,
    log_power: torch.Tensor,
    channel: torch.Tensor,
    shape: ChannelShape,
    section: dict,
    stage: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    spectrum_map = model.spectrum_input(target_shape)
    spectrum_tensor = model.spectrum_encoder(spectrum_map)
    spectrum = spectrum_tensor.flatten(1)
    if stage == "coarse":
        components = model.decode_components(spectrum, None, detail_scale=0.0)
    else:
        detail = model.phase_encoder(target_shape).flatten(1)
        components = model.decode_components(spectrum, detail, detail_scale=1.0)
    prediction_shape = components["prediction"]
    terms = _spectrum_power_terms(components["spectrum_log_power"], spectrum_map)
    if stage != "coarse":
        prediction = shape_to_channel(prediction_shape, log_power, shape)
        terms.update(
            {
                "angle_delay": energy_weighted_angle_delay_mse(
                    prediction_shape,
                    target_shape,
                    shape,
                    float(section.get("energy_emphasis", 2.0)),
                    float(section.get("maximum_energy_weight", 16.0)),
                ),
                "joint_power": joint_power_loss(prediction_shape, target_shape, shape),
                "coherence": complex_coherence_loss(prediction_shape, target_shape),
                "complex_direction": energy_weighted_complex_direction_loss(
                    prediction_shape, target_shape, shape
                ),
                **metric_aligned_channel_losses(prediction, channel, shape),
            }
        )
        residual_target = normalize_angle_delay(
            target_shape - components["coarse"].detach() / model.decoder.detail_gain
        )
        terms["detail_angle_delay"] = energy_weighted_angle_delay_mse(
            components["detail_residual"],
            residual_target,
            shape,
            float(section.get("energy_emphasis", 2.0)),
            float(section.get("maximum_energy_weight", 16.0)),
        )
        terms["detail_coherence"] = complex_coherence_loss(
            components["detail_residual"], residual_target
        )
    return prediction_shape, terms


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
    scheduler_name = str(section.get("scheduler", "cosine")).lower()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(section.get("plateau_factor", 0.5)),
            patience=int(section.get("plateau_patience", 6)),
            threshold=float(section.get("plateau_threshold", 1e-4)),
            threshold_mode=str(section.get("plateau_threshold_mode", "abs")),
            min_lr=float(section.get("minimum_learning_rate", 1e-6)),
        )
    elif scheduler_name == "cosine":
        scheduler_epochs = epochs
        if section.get("architecture") == "factorized_residual_v4":
            scheduler_epochs -= max(0, int(section.get("coarse_pretrain_epochs", 0)))
            scheduler_epochs -= max(0, int(section.get("detail_pretrain_epochs", 0)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(scheduler_epochs, 1),
            eta_min=float(section.get("minimum_learning_rate", 1e-6)),
        )
    else:
        raise ValueError(f"Unsupported autoencoder scheduler: {scheduler_name}")
    scaler = make_grad_scaler(device, amp)
    output_dir = Path(section["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "resolved_config.json", config)
    history_path = output_dir / "history.jsonl"
    if not resume:
        history_path.unlink(missing_ok=True)
        for name in ("best.pt", "best_spectrum.pt", "last.pt", "final.pt"):
            (output_dir / name).unlink(missing_ok=True)

    start_epoch = 0
    best_score = -math.inf
    best_spectrum_score = -math.inf
    early_stopping_score = -math.inf
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
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -math.inf))
        best_spectrum_score = float(checkpoint.get("best_spectrum_score", -math.inf))
        early_stopping_score = float(checkpoint.get("early_stopping_score", best_score))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        resumed_metrics = checkpoint.get("metrics", {})

    weights = section["loss_weights"]
    coarse_weights = section.get("coarse_loss_weights", weights)
    detail_weights = section.get("detail_loss_weights", weights)
    factorized = isinstance(model, FactorizedResidualAutoencoder)
    accumulation = int(section.get("gradient_accumulation", 1))
    validation_interval = int(section.get("validation_interval", 1))
    patience = int(section.get("early_stopping_patience", 0))
    minimum_delta = float(section.get("minimum_delta", 1e-4))
    final_metrics = resumed_metrics
    last_epoch = start_epoch - 1
    detail_warmup_epochs = int(section.get("detail_warmup_epochs", 0))
    detail_ramp_epochs = max(1, int(section.get("detail_ramp_epochs", 1)))
    target_score = float(section.get("target_score", 0.8))
    stop_at_target = bool(section.get("stop_at_target", False))
    maximum_training_hours = float(section.get("maximum_training_hours", 0.0))
    training_started = time.perf_counter()
    stop_reason = "maximum_epochs"
    previous_stage: str | None = None
    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        started = time.perf_counter()
        stage = autoencoder_training_stage(section, epoch) if factorized else "legacy"
        if factorized:
            model.set_trainable_stage(stage)
        if stage != previous_stage:
            print(f"AE training stage -> {stage} (epoch {epoch + 1})", flush=True)
            previous_stage = stage
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: defaultdict[str, float] = defaultdict(float)
        batches = 0
        detail_scale = (
            0.0
            if stage == "coarse"
            else 1.0
            if factorized
            else min(
                1.0,
                max(0.0, (epoch - detail_warmup_epochs + 1) / detail_ramp_epochs),
            )
        )
        stage_weights = (
            coarse_weights
            if stage == "coarse"
            else detail_weights
            if stage == "detail"
            else weights
        )
        for batch_index, batch in enumerate(train_loader):
            channel = batch["channel"].to(device, non_blocking=True)
            target_shape, log_power, outage = channel_to_shape_target(channel, shape)
            with autocast_context(device, amp):
                if factorized:
                    prediction_shape, terms = factorized_autoencoder_batch(
                        model,
                        target_shape,
                        log_power,
                        channel,
                        shape,
                        section,
                        stage,
                    )
                else:
                    spectrum, phase = model.encode(target_shape)
                    spectrum_shape = model.decode(spectrum, None)
                    prediction_shape = (
                        spectrum_shape
                        if detail_scale <= 0.0
                        else model.decode(spectrum, phase * detail_scale)
                    )
                    prediction = shape_to_channel(prediction_shape, log_power, shape)
                    terms = {
                        "angle_delay": energy_weighted_angle_delay_mse(
                            prediction_shape,
                            target_shape,
                            shape,
                            float(section.get("energy_emphasis", 2.0)),
                            float(section.get("maximum_energy_weight", 16.0)),
                        ),
                        "joint_power": joint_power_loss(
                            prediction_shape, target_shape, shape
                        ),
                        "coherence": complex_coherence_loss(
                            prediction_shape, target_shape
                        ),
                        **metric_aligned_channel_losses(prediction, channel, shape),
                    }
                    spectrum_prediction = shape_to_channel(
                        spectrum_shape, log_power, shape
                    )
                    spectrum_terms = metric_aligned_channel_losses(
                        spectrum_prediction, channel, shape
                    )
                    terms.update(
                        {
                            "spectrum_angle_delay": energy_weighted_angle_delay_mse(
                                spectrum_shape,
                                target_shape,
                                shape,
                                float(section.get("energy_emphasis", 2.0)),
                                float(section.get("maximum_energy_weight", 16.0)),
                            ),
                            "spectrum_coherence": complex_coherence_loss(
                                spectrum_shape, target_shape
                            ),
                            "spectrum_pas": spectrum_terms["pas"],
                            "spectrum_pdp": spectrum_terms["pdp"],
                            "spectrum_nmse": spectrum_terms["nmse"],
                            "spectrum_joint_power": joint_power_loss(
                                spectrum_shape, target_shape, shape
                            ),
                        }
                    )
                total = weighted_sum(terms, stage_weights) / accumulation
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite AE loss at epoch={epoch + 1}, batch={batch_index + 1}"
                )
            scaler.scale(total).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
                train_loader
            ):
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
        train_metrics = {name: value / max(batches, 1) for name, value in sums.items()}
        should_validate = len(validation_dataset) and (
            (epoch + 1) % validation_interval == 0 or epoch + 1 == epochs
        )
        validation = (
            evaluate_autoencoder(
                model,
                validation_loader,
                shape,
                device,
                amp,
                detail_scale=detail_scale,
            )
            if should_validate
            else {}
        )
        selection_metric = str(section.get("selection_metric", "score"))
        selection_value = float(validation.get(selection_metric, -math.inf))
        scheduler_active = not factorized or stage == "joint"
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if validation and scheduler_active:
                scheduler.step(selection_value)
        elif scheduler_active:
            scheduler.step()
        final_metrics = validation or train_metrics
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - started,
            "detail_scale": detail_scale,
            "training_stage": stage,
        }
        append_jsonl(history_path, record)
        score = selection_value
        spectrum_score = float(validation.get("spectrum_only_score", -math.inf))
        improved = bool(validation) and score > best_score
        meaningful_improvement = (
            bool(validation) and score > early_stopping_score + minimum_delta
        )
        if improved:
            best_score = score
        if meaningful_improvement:
            early_stopping_score = score
            epochs_without_improvement = 0
        elif validation and (not factorized or stage == "joint"):
            epochs_without_improvement += validation_interval
        spectrum_improved = bool(validation) and spectrum_score > best_spectrum_score
        if spectrum_improved:
            best_spectrum_score = spectrum_score
        if improved:
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
                epochs_without_improvement,
                best_spectrum_score,
                early_stopping_score,
            )
        if spectrum_improved:
            _save_checkpoint(
                output_dir / "best_spectrum.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                epoch,
                validation,
                best_score,
                epochs_without_improvement,
                best_spectrum_score,
                early_stopping_score,
            )
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
            best_spectrum_score,
            early_stopping_score,
        )
        print(
            f"AE epoch={epoch + 1}/{epochs} train={train_metrics.get('total', 0.0):.6f} "
            f"score={validation.get('score', float('nan')):.6f} "
            f"spectrum={validation.get('spectrum_only_score', float('nan')):.6f} "
            f"stage={stage} detail={detail_scale:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"seconds={record['elapsed_seconds']:.2f}",
            flush=True,
        )
        if (
            stop_at_target
            and validation
            and best_score >= target_score
            and (not factorized or stage == "joint")
        ):
            stop_reason = "target_reached"
            print(
                f"AE target reached: best score {best_score:.6f} >= {target_score:.6f}",
                flush=True,
            )
            break
        if (
            patience > 0
            and validation
            and epochs_without_improvement >= patience
            and (not factorized or stage == "joint")
        ):
            stop_reason = "early_stopping"
            print(f"AE early stopping at epoch {epoch + 1}", flush=True)
            break
        elapsed_training_seconds = time.perf_counter() - training_started
        if (
            maximum_training_hours > 0.0
            and elapsed_training_seconds >= maximum_training_hours * 3600.0
        ):
            stop_reason = "runtime_limit"
            print(
                f"AE runtime limit reached after {elapsed_training_seconds / 3600.0:.2f} hours",
                flush=True,
            )
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
        best_spectrum_score,
        early_stopping_score,
    )
    summary = {
        "device": str(device),
        "architecture": str(section.get("architecture", "structured_v2")),
        "parameters": count_parameters(model),
        "spectrum_latent_dim": model.spectrum_latent_dim,
        "phase_latent_dim": model.phase_latent_dim,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "last_epoch": last_epoch,
        "best_score": None if best_score == -math.inf else best_score,
        "best_spectrum_score": (
            None if best_spectrum_score == -math.inf else best_spectrum_score
        ),
        "early_stopping_score": (
            None if early_stopping_score == -math.inf else early_stopping_score
        ),
        "target_score": target_score,
        "target_reached": best_score >= target_score,
        "stop_reason": stop_reason,
        "training_elapsed_seconds": time.perf_counter() - training_started,
        "selection_metric": str(section.get("selection_metric", "score")),
        "coarse_pretrain_epochs": int(section.get("coarse_pretrain_epochs", 0)),
        "detail_pretrain_epochs": int(section.get("detail_pretrain_epochs", 0)),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "summary.json", summary)
    return summary
