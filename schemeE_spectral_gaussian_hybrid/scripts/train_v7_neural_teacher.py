from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import _bootstrap  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as functional

from diagnose_local_teacher_bank import _hybrid_context, _power_cosine
from scheme_e.config import choose_device, load_config, save_json, seed_everything
from scheme_e.hybrid_training import evaluate_hybrid
from scheme_e.neural_spectral_refiner import SpectralNeighborRefiner
from scheme_e.spectral_compression import SpectralCompressor
from scheme_e.spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE


def _tree_neighbors(
    positions: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    count: int,
) -> np.ndarray:
    if not len(support) or not len(query):
        raise ValueError("neighbor lookup requires non-empty support and query sets")
    actual = min(int(count), len(support))
    _, local = cKDTree(positions[support, :2]).query(
        positions[query, :2], k=actual
    )
    local = np.asarray(local, dtype=np.int64).reshape(len(query), actual)
    selected = support[local]
    if actual < int(count):
        selected = np.concatenate(
            [
                selected,
                np.repeat(selected[:, -1:], int(count) - actual, axis=1),
            ],
            axis=1,
        )
    return selected


def _leakage_free_neighbors(
    metadata: dict[str, np.ndarray],
    training: np.ndarray,
    validation: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = metadata["train_positions"]
    folds = metadata["spectral_folds"]
    train_neighbors = np.empty((len(training), int(count)), dtype=np.int64)
    for fold in sorted(np.unique(folds[training]).tolist()):
        local = np.flatnonzero(folds[training] == fold)
        support = training[folds[training] != fold]
        train_neighbors[local] = _tree_neighbors(
            positions, support, training[local], count
        )
    validation_neighbors = _tree_neighbors(
        positions, training, validation, count
    )
    return train_neighbors, validation_neighbors


def _normalization(
    metadata: dict[str, np.ndarray], training: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_mean = metadata["train_positions"][training, :2].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    position_std = np.maximum(
        metadata["train_positions"][training, :2].std(axis=0, dtype=np.float64),
        1.0,
    ).astype(np.float32)
    geometry_mean = metadata["train_geometry_features"][training].mean(
        axis=0, dtype=np.float64
    ).astype(np.float32)
    geometry_std = np.maximum(
        metadata["train_geometry_features"][training].std(
            axis=0, dtype=np.float64
        ),
        1e-3,
    ).astype(np.float32)
    return position_mean, position_std, geometry_mean, geometry_std


def _feature_arrays(
    metadata: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    position_mean: np.ndarray,
    position_std: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = metadata["train_positions"][:, :2].astype(np.float32)
    position_z = (positions - position_mean) / position_std
    geometry_z = (
        metadata["train_geometry_features"].astype(np.float32) - geometry_mean
    ) / geometry_std
    relative_ue = (
        priors["ue_log_energy"].astype(np.float32)
        - priors["log_power"].astype(np.float32)[:, None]
    )
    query_features = np.concatenate(
        [
            position_z,
            geometry_z,
            relative_ue,
            priors["log_power"].astype(np.float32)[:, None],
            priors["uncertainty"].astype(np.float32)[:, None],
            priors["outage_probability"].astype(np.float32)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    return query_features, position_z, geometry_z


def _neighbor_features(
    target_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    position_z: np.ndarray,
    geometry_z: np.ndarray,
    targets: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
) -> np.ndarray:
    relative_position = position_z[neighbor_indices] - position_z[target_indices, None]
    distance = np.linalg.norm(relative_position, axis=2, keepdims=True)
    direction = relative_position / np.maximum(distance, 1e-6)
    geometry_delta = geometry_z[neighbor_indices] - geometry_z[target_indices, None]
    neighbor_relative_ue = (
        targets["ue_log_energy"][neighbor_indices].astype(np.float32)
        - targets["log_power"][neighbor_indices].astype(np.float32)[..., None]
    )
    relative_power = (
        targets["log_power"][neighbor_indices].astype(np.float32)
        - priors["log_power"][target_indices].astype(np.float32)[:, None]
    )[..., None]
    return np.concatenate(
        [
            geometry_delta,
            relative_position,
            distance,
            direction,
            neighbor_relative_ue,
            relative_power,
        ],
        axis=2,
    ).astype(np.float32)


def _corrected_power(
    base_log: torch.Tensor,
    residual: torch.Tensor,
    scale: torch.Tensor,
    components: torch.Tensor,
    log_scale: float,
) -> torch.Tensor:
    log_value = base_log.float() + (
        residual.float() @ components.float()
    ) * scale.float()
    return torch.expm1(log_value.clamp(0.0, 20.0)) / float(log_scale)


def _cosine(power: torch.Tensor, target_power: torch.Tensor) -> torch.Tensor:
    return functional.cosine_similarity(
        power.float(), target_power.float(), dim=1, eps=1e-8
    ).clamp(0.0, 1.0)


def _model_metrics(
    model: SpectralNeighborRefiner,
    arrays: dict[str, np.ndarray],
    pas_compressor: SpectralCompressor,
    pdp_compressor: SpectralCompressor,
    pas_dim: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    pas_scale = torch.as_tensor(pas_compressor.scale, device=device)
    pas_components = torch.as_tensor(pas_compressor.components, device=device)
    pdp_scale = torch.as_tensor(pdp_compressor.scale, device=device)
    pdp_components = torch.as_tensor(pdp_compressor.components, device=device)
    predicted: list[np.ndarray] = []
    pas_sum = 0.0
    pdp_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(arrays["base"]), int(batch_size)):
            stop = min(start + int(batch_size), len(arrays["base"]))
            outputs = model(
                torch.as_tensor(arrays["base"][start:stop], device=device),
                torch.as_tensor(arrays["query"][start:stop], device=device),
                torch.as_tensor(arrays["neighbor_latent"][start:stop], device=device),
                torch.as_tensor(arrays["neighbor_features"][start:stop], device=device),
            )
            latent = outputs["latent"].float()
            residual = outputs["residual"].float()
            predicted.append(residual.cpu().numpy())
            pas_power = _corrected_power(
                torch.as_tensor(
                    arrays["base_pas_log"][start:stop], device=device
                ),
                residual[:, :pas_dim],
                pas_scale,
                pas_components,
                PAS_LOG_SCALE,
            )
            pdp_power = _corrected_power(
                torch.as_tensor(
                    arrays["base_pdp_log"][start:stop], device=device
                ),
                residual[:, pas_dim:],
                pdp_scale,
                pdp_components,
                PDP_LOG_SCALE,
            )
            target_pas = torch.expm1(
                torch.as_tensor(
                    arrays["pas_log"][start:stop], device=device
                ).float().clamp(0.0, 20.0)
            ) / PAS_LOG_SCALE
            target_pdp = torch.expm1(
                torch.as_tensor(
                    arrays["pdp_log"][start:stop], device=device
                ).float().clamp(0.0, 20.0)
            ) / PDP_LOG_SCALE
            pas_sum += float(_cosine(pas_power, target_pas).sum().cpu())
            pdp_sum += float(_cosine(pdp_power, target_pdp).sum().cpu())
    count = max(len(arrays["base"]), 1)
    return {
        "pas": pas_sum / count,
        "pdp": pdp_sum / count,
        "score": 0.55 * pas_sum / count + 0.45 * pdp_sum / count,
    }, np.concatenate(predicted, axis=0)


def _cell_arrays(
    indices: np.ndarray,
    neighbors: np.ndarray,
    base_latent: np.ndarray,
    true_latent: np.ndarray,
    query_features: np.ndarray,
    position_z: np.ndarray,
    geometry_z: np.ndarray,
    metadata: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "base": base_latent[indices].astype(np.float32),
        "query": query_features[indices].astype(np.float32),
        "neighbor_latent": true_latent[neighbors].astype(np.float32),
        "neighbor_features": _neighbor_features(
            indices,
            neighbors,
            position_z,
            geometry_z,
            targets,
            priors,
        ),
        "target": true_latent[indices].astype(np.float32),
        "base_pas_log": priors["pas_log"][indices].astype(np.float32),
        "base_pdp_log": priors["pdp_log"][indices].astype(np.float32),
        "pas_log": targets["pas_log"][indices].astype(np.float32),
        "pdp_log": targets["pdp_log"][indices].astype(np.float32),
    }


def _train_cell(
    cell: int,
    training: np.ndarray,
    validation: np.ndarray,
    metadata: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, object], np.ndarray, SpectralCompressor, SpectralCompressor]:
    pas_compressor = SpectralCompressor(int(args.pas_dim)).fit(
        targets["pas_log"][training].astype(np.float32), int(args.seed) + cell * 17
    )
    pdp_compressor = SpectralCompressor(int(args.pdp_dim)).fit(
        targets["pdp_log"][training].astype(np.float32), int(args.seed) + cell * 19
    )
    count = len(metadata["train_cells"])
    true_latent = np.zeros((count, int(args.pas_dim) + int(args.pdp_dim)), dtype=np.float32)
    base_latent = np.zeros_like(true_latent)
    cell_indices = np.flatnonzero(metadata["train_cells"] == cell)
    true_latent[cell_indices, : int(args.pas_dim)] = pas_compressor.transform(
        targets["pas_log"][cell_indices].astype(np.float32)
    )
    true_latent[cell_indices, int(args.pas_dim) :] = pdp_compressor.transform(
        targets["pdp_log"][cell_indices].astype(np.float32)
    )
    base_latent[cell_indices, : int(args.pas_dim)] = pas_compressor.transform(
        priors["pas_log"][cell_indices].astype(np.float32)
    )
    base_latent[cell_indices, int(args.pas_dim) :] = pdp_compressor.transform(
        priors["pdp_log"][cell_indices].astype(np.float32)
    )
    position_mean, position_std, geometry_mean, geometry_std = _normalization(
        metadata, training
    )
    query_features, position_z, geometry_z = _feature_arrays(
        metadata,
        targets,
        priors,
        position_mean,
        position_std,
        geometry_mean,
        geometry_std,
    )
    train_neighbors, validation_neighbors = _leakage_free_neighbors(
        metadata, training, validation, int(args.neighbors)
    )
    train_arrays = _cell_arrays(
        training,
        train_neighbors,
        base_latent,
        true_latent,
        query_features,
        position_z,
        geometry_z,
        metadata,
        targets,
        priors,
    )
    validation_arrays = _cell_arrays(
        validation,
        validation_neighbors,
        base_latent,
        true_latent,
        query_features,
        position_z,
        geometry_z,
        metadata,
        targets,
        priors,
    )
    model = SpectralNeighborRefiner(
        latent_dim=int(args.pas_dim) + int(args.pdp_dim),
        pas_dim=int(args.pas_dim),
        query_feature_dim=train_arrays["query"].shape[1],
        neighbor_feature_dim=train_arrays["neighbor_features"].shape[2],
        width=int(args.width),
        layers=int(args.layers),
        heads=int(args.heads),
        dropout=float(args.dropout),
        maximum_residual=float(args.maximum_residual),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6,
        min_lr=float(args.minimum_learning_rate),
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    pas_scale = torch.as_tensor(pas_compressor.scale, device=device)
    pas_components = torch.as_tensor(pas_compressor.components, device=device)
    pdp_scale = torch.as_tensor(pdp_compressor.scale, device=device)
    pdp_components = torch.as_tensor(pdp_compressor.components, device=device)
    rng = np.random.default_rng(int(args.seed) + cell * 101)
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = rng.permutation(len(training))
        total_sum = 0.0
        for start in range(0, len(order), int(args.batch_size)):
            local = order[start : start + int(args.batch_size)]
            base = torch.as_tensor(train_arrays["base"][local], device=device)
            query = torch.as_tensor(train_arrays["query"][local], device=device)
            neighbor_latent = torch.as_tensor(
                train_arrays["neighbor_latent"][local], device=device
            )
            neighbor_features = torch.as_tensor(
                train_arrays["neighbor_features"][local], device=device
            )
            target_latent = torch.as_tensor(
                train_arrays["target"][local], device=device
            )
            target_pas = torch.expm1(
                torch.as_tensor(train_arrays["pas_log"][local], device=device)
                .float()
                .clamp(0.0, 20.0)
            ) / PAS_LOG_SCALE
            target_pdp = torch.expm1(
                torch.as_tensor(train_arrays["pdp_log"][local], device=device)
                .float()
                .clamp(0.0, 20.0)
            ) / PDP_LOG_SCALE
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                output = model(base, query, neighbor_latent, neighbor_features)
                latent = output["latent"]
            pas_power = _corrected_power(
                torch.as_tensor(
                    train_arrays["base_pas_log"][local], device=device
                ),
                output["residual"][:, : int(args.pas_dim)],
                pas_scale,
                pas_components,
                PAS_LOG_SCALE,
            )
            pdp_power = _corrected_power(
                torch.as_tensor(
                    train_arrays["base_pdp_log"][local], device=device
                ),
                output["residual"][:, int(args.pas_dim) :],
                pdp_scale,
                pdp_components,
                PDP_LOG_SCALE,
            )
            pas_loss = 1.0 - _cosine(pas_power, target_pas).mean()
            pdp_loss = 1.0 - _cosine(pdp_power, target_pdp).mean()
            latent_loss = functional.smooth_l1_loss(
                latent.float(), target_latent.float()
            )
            residual_loss = output["residual"].float().square().mean()
            loss = (
                0.55 * pas_loss
                + 0.45 * pdp_loss
                + float(args.latent_weight) * latent_loss
                + float(args.residual_weight) * residual_loss
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_sum += float(loss.detach().cpu()) * len(local)
        validation_metrics: dict[str, float] = {}
        if epoch == 1 or epoch % int(args.validation_interval) == 0:
            validation_metrics, _ = _model_metrics(
                model,
                validation_arrays,
                pas_compressor,
                pdp_compressor,
                int(args.pas_dim),
                device,
                int(args.validation_batch_size),
            )
            score = float(validation_metrics["score"])
            scheduler.step(score)
            if score > best_score + 1e-5:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale += int(args.validation_interval)
        record = {
            "epoch": epoch,
            "train": total_sum / max(len(training), 1),
            "validation": validation_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if validation_metrics:
            print(
                "NeuralTeacher cell=%d epoch=%d/%d train=%.6f pas=%.6f pdp=%.6f score=%.6f lr=%.2e"
                % (
                    cell,
                    epoch,
                    int(args.epochs),
                    record["train"],
                    validation_metrics["pas"],
                    validation_metrics["pdp"],
                    validation_metrics["score"],
                    record["learning_rate"],
                ),
                flush=True,
            )
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise RuntimeError("Neural spectral teacher did not produce a checkpoint")
    model.load_state_dict(best_state)
    best_metrics, validation_prediction = _model_metrics(
        model,
        validation_arrays,
        pas_compressor,
        pdp_compressor,
        int(args.pas_dim),
        device,
        int(args.validation_batch_size),
    )
    base_metrics = {
        "pas": float(
            _power_cosine(
                priors["pas_log"][validation],
                targets["pas_log"][validation],
                PAS_LOG_SCALE,
            ).mean()
        ),
        "pdp": float(
            _power_cosine(
                priors["pdp_log"][validation],
                targets["pdp_log"][validation],
                PDP_LOG_SCALE,
            ).mean()
        ),
    }
    checkpoint_path = output_dir / f"cell{cell}_best.pt"
    torch.save(
        {
            "model": best_state,
            "cell": cell,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "base_metrics": base_metrics,
            "pas_compressor": pas_compressor.state_dict(),
            "pdp_compressor": pdp_compressor.state_dict(),
            "position_mean": position_mean,
            "position_std": position_std,
            "geometry_mean": geometry_mean,
            "geometry_std": geometry_std,
            "query_feature_dim": train_arrays["query"].shape[1],
            "neighbor_feature_dim": train_arrays["neighbor_features"].shape[2],
            "settings": vars(args),
        },
        checkpoint_path,
    )
    save_json(output_dir / f"cell{cell}_history.json", history)
    report = {
        "cell": cell,
        "training_samples": int(len(training)),
        "validation_samples": int(len(validation)),
        "best_epoch": best_epoch,
        "base": base_metrics,
        "best": best_metrics,
        "checkpoint": str(checkpoint_path),
    }
    return report, validation_prediction, pas_compressor, pdp_compressor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a leakage-free neural residual spectral teacher on Fold0"
    )
    parser.add_argument("--config", default="configs/v5_local_teacher.json")
    parser.add_argument("--policy", default="reports/generated/v5_fold0_policy.json")
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--pas-dim", type=int, default=128)
    parser.add_argument("--pdp-dim", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--maximum-residual", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-weight", type=float, default=0.04)
    parser.add_argument("--residual-weight", type=float, default=0.001)
    parser.add_argument("--output-dir", default="artifacts/v7/fold0/neural_teacher")
    parser.add_argument(
        "--output-prior", default="artifacts/v7/fold0/neural_teacher_priors.npz"
    )
    parser.add_argument(
        "--output", default="reports/generated/v7_neural_teacher.json"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    seed_everything(int(args.seed))
    config = load_config(args.config)
    device = choose_device(str(config["runtime"].get("device", "auto")))
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(config["spectral"]["target_path"]) as source:
        targets = {name: source[name] for name in source.files}
    with np.load(config["spectral_teacher"]["oof_output_path"]) as source:
        priors = {name: np.array(source[name], copy=True) for name in source.files}
    fold = int(config["split"]["validation_fold"])
    count = min(len(metadata["train_cells"]), len(priors["available"]))
    indices = np.arange(count, dtype=np.int64)
    available = priors["available"][:count].astype(bool)
    validation_mask = metadata["validation_masks"][fold][:count].astype(bool)
    observed = indices[available & ~validation_mask]
    validation = indices[available & validation_mask]
    outages = targets["outage"][:count].astype(bool)
    refined = deepcopy(priors)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for cell in sorted(np.unique(metadata["train_cells"][:count]).tolist()):
        training_cell = observed[
            (metadata["train_cells"][observed] == cell) & ~outages[observed]
        ]
        validation_cell = validation[
            (metadata["train_cells"][validation] == cell) & ~outages[validation]
        ]
        report, prediction, pas_compressor, pdp_compressor = _train_cell(
            int(cell),
            training_cell,
            validation_cell,
            metadata,
            targets,
            priors,
            args,
            device,
            output_dir,
        )
        pas_delta = (
            prediction[:, : int(args.pas_dim)] @ pas_compressor.components
        ) * pas_compressor.scale
        pdp_delta = (
            prediction[:, int(args.pas_dim) :] @ pdp_compressor.components
        ) * pdp_compressor.scale
        refined["pas_log"][validation_cell] = np.clip(
            priors["pas_log"][validation_cell].astype(np.float32) + pas_delta,
            0.0,
            20.0,
        ).astype(refined["pas_log"].dtype)
        refined["pdp_log"][validation_cell] = np.clip(
            priors["pdp_log"][validation_cell].astype(np.float32) + pdp_delta,
            0.0,
            20.0,
        ).astype(refined["pdp_log"].dtype)
        reports.append(report)
    output_prior = Path(args.output_prior)
    output_prior.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_prior, **refined)

    nonzero_validation = validation[~outages[validation]]
    teacher = {
        "base_pas": float(
            _power_cosine(
                priors["pas_log"][nonzero_validation],
                targets["pas_log"][nonzero_validation],
                PAS_LOG_SCALE,
            ).mean()
        ),
        "base_pdp": float(
            _power_cosine(
                priors["pdp_log"][nonzero_validation],
                targets["pdp_log"][nonzero_validation],
                PDP_LOG_SCALE,
            ).mean()
        ),
        "refined_pas": float(
            _power_cosine(
                refined["pas_log"][nonzero_validation],
                targets["pas_log"][nonzero_validation],
                PAS_LOG_SCALE,
            ).mean()
        ),
        "refined_pdp": float(
            _power_cosine(
                refined["pdp_log"][nonzero_validation],
                targets["pdp_log"][nonzero_validation],
                PDP_LOG_SCALE,
            ).mean()
        ),
    }
    base_context = _hybrid_context(
        config,
        metadata,
        targets,
        priors,
        validation,
        observed,
        Path(args.policy),
    )
    refined_context = dict(base_context)
    refined_context["priors"] = refined
    base_channel = evaluate_hybrid(**base_context)
    refined_channel = evaluate_hybrid(**refined_context)
    refined_projected = evaluate_hybrid(
        **refined_context,
        output_projection={
            "iterations": 2,
            "strength_by_cell": [0.5, 0.5],
            "minimum_scale": 0.5,
            "maximum_scale": 2.0,
            "channel_source": "model",
        },
    )
    report = {
        "status": "PASS",
        "stage": "v7_neural_residual_teacher_fold0",
        "leakage_free_validation": True,
        "config": args.config,
        "teacher": teacher,
        "cells": reports,
        "channel": {
            "base": base_channel,
            "refined": refined_channel,
            "refined_projected": refined_projected,
        },
        "output_prior": str(output_prior),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
