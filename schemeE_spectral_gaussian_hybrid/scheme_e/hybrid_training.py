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
from .reference import build_reference_candidates, sample_references


def _load_repository(config: dict, prior_path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(prior_path) as source:
        priors = {name: source[name] for name in source.files}
    return metadata, priors


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
    ).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
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
) -> dict[str, torch.Tensor]:
    return {
        "pas_log": torch.as_tensor(priors["pas_log"][indices].astype(np.float32), device=device),
        "pdp_log": torch.as_tensor(priors["pdp_log"][indices].astype(np.float32), device=device),
        "ue_log_energy": torch.as_tensor(priors["ue_log_energy"][indices], device=device),
        "log_power": torch.as_tensor(priors["log_power"][indices], device=device),
        "uncertainty": torch.as_tensor(priors["uncertainty"][indices], device=device),
        "outage_probability": torch.as_tensor(priors["outage_probability"][indices], device=device),
        "geometry": torch.as_tensor(
            np.clip(
                (metadata["train_geometry_features"][indices] - geometry_mean) / geometry_std,
                -8.0,
                8.0,
            ),
            device=device,
        ),
    }


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
) -> dict[str, float | int]:
    observed_outage = metadata["outage"][observed_indices].astype(bool)
    candidates, _ = build_reference_candidates(
        metadata["train_positions"][target_indices],
        metadata["train_cells"][target_indices],
        metadata["train_positions"][observed_indices],
        metadata["train_cells"][observed_indices],
        observed_outage,
        top_k=1,
        target_global_indices=target_indices,
        observed_global_indices=observed_indices,
    )
    references = observed_indices[candidates[:, 0]]
    accumulator = ChannelMetricAccumulator(shape)
    model.eval()
    for start in range(0, len(target_indices), int(batch_size)):
        stop = min(start + int(batch_size), len(target_indices))
        indices = target_indices[start:stop]
        reference = torch.as_tensor(np.asarray(channels[references[start:stop]]), device=device)
        target = torch.as_tensor(np.asarray(channels[indices]), device=device)
        inputs = _prior_batch(priors, metadata, indices, geometry_mean, geometry_std, device)
        outputs = model(reference, projection_iterations=projection_iterations, **inputs)
        predicted = outputs["channel"]
        predicted_outage = inputs["outage_probability"] >= float(outage_threshold)
        predicted = predicted.masked_fill(predicted_outage[:, None, None, None], 0.0)
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
                np.sum(priors["outage_probability"][target_indices] >= outage_threshold)
            ),
        }
    )
    return result


def train_hybrid(config: dict, final: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(int(config["seed"]))
    section = config["hybrid_final"] if final else config["hybrid"]
    prior_path = config["spectral_teacher"]["oof_output_path"]
    metadata, priors = _load_repository(config, prior_path)
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
    geometry_mean, geometry_std = _geometry_stats(
        metadata["train_geometry_features"], observed_indices
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
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(section["learning_rate"]),
        weight_decay=float(section.get("weight_decay", 1e-4)),
    )
    amp = bool(config["runtime"].get("amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp)
    output_dir = Path(section["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
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
                "rng_state": rng.bit_generator.state,
                "epoch": epoch,
                "metrics": metrics,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale": stale,
                "geometry_mean": geometry_mean,
                "geometry_std": geometry_std,
                "outage_threshold": outage_threshold,
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
            selected_references = sample_references(
                candidates[local],
                distances[local],
                rng,
                float(section.get("reference_guard_min_meters", 3.0)),
                float(section.get("reference_guard_max_meters", 8.0)),
            )
            reference_indices = observed_indices[selected_references]
            reference = torch.as_tensor(np.asarray(channels[reference_indices]), device=device)
            target = torch.as_tensor(np.asarray(channels[target_indices]), device=device)
            inputs = _prior_batch(
                priors, metadata, target_indices, geometry_mean, geometry_std, device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                outputs = model(reference, **inputs)
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
                        "config": config,
                    },
                    output_dir / "best.pt",
                )
            else:
                stale += interval
        elif not len(validation_indices):
            best_epoch = epoch
            best_score = -float(sums["total"])
        checkpoint_interval = int(section.get("checkpoint_interval", interval))
        if epoch == epochs or (checkpoint_interval and epoch % checkpoint_interval == 0):
            save_last(epoch, validation or {"train_total": sums["total"]})
        record = {
            "epoch": epoch,
            "train": sums,
            "validation": validation,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"SchemeE epoch={epoch}/{epochs} train={sums['total']:.6f} "
            f"score={validation.get('score', float('nan')):.6f} "
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
                )
            )
        selected_projection = max(projection_reports, key=lambda item: float(item["score"]))
    else:
        selected_projection = {"projection_iterations": int(section.get("projection_iterations", 4))}
    summary = {
        "stage": "hybrid_final" if final else "hybrid_fold0",
        "architecture": "spectral_gaussian_full_resolution_adapter_v1",
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "training_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "best_epoch": int(checkpoint["epoch"]),
        "best_metrics": checkpoint.get("metrics", {}),
        "projection_candidates": projection_reports,
        "selected_projection_iterations": int(selected_projection["projection_iterations"]),
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
