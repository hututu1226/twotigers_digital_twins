from __future__ import annotations

import json
import hashlib
import pickle
from pathlib import Path
import time

import numpy as np

from .config import choose_device, save_json, seed_everything
from .gp import (
    SharedMultiOutputGP,
    convex_cosine_weights,
    convex_mse_weights,
    ensemble_log_power_predictions,
    ensemble_predictions,
)
from .local_spectral import local_expert_settings, local_spectral_prediction
from .outage import OutageEnsemble, binary_metrics
from .power_safety import apply_power_calibration, fit_power_calibration
from .spectral_compression import SpectralCompressor
from .spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE


def _kernel_settings(section: dict) -> list[tuple[str, float]]:
    names = [str(value) for value in section.get("kernels", ["rq10", "rq20", "matern20"])]
    mixes = section.get("kernel_feature_mixes")
    if mixes is None:
        mixes = [float(section.get("feature_mix", 0.5))] * len(names)
    if len(mixes) != len(names):
        raise ValueError("kernel_feature_mixes must match kernels")
    return [(name, float(mix)) for name, mix in zip(names, mixes, strict=True)]


def _load_arrays(config: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    target_path = Path(config["spectral"]["target_path"])
    with np.load(target_path) as source:
        targets = {name: source[name] for name in source.files}
    count = len(targets["outage"])
    for name in ("train_positions", "train_cells", "spectral_folds", "train_geometry_features"):
        metadata[name] = metadata[name][:count]
    return metadata, targets


def _balanced_indices(metadata: dict[str, np.ndarray], targets: dict[str, np.ndarray], limit: int) -> np.ndarray:
    count = len(targets["outage"])
    indices = np.arange(count, dtype=np.int64)
    if not limit or limit >= count:
        return indices
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index in indices:
        key = (
            int(metadata["train_cells"][index]),
            int(metadata["spectral_folds"][index]),
            int(targets["outage"][index]),
        )
        groups.setdefault(key, []).append(int(index))
    rng = np.random.default_rng(2026)
    selected: list[int] = []
    while len(selected) < limit:
        progressed = False
        for key in sorted(groups):
            values = groups[key]
            if values:
                choice = int(rng.integers(0, len(values)))
                selected.append(values.pop(choice))
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _cosine_loss(prediction_log: np.ndarray, target_log: np.ndarray, scale: float) -> np.ndarray:
    prediction = np.expm1(np.clip(prediction_log.astype(np.float64), 0.0, 20.0)) / scale
    target = np.expm1(np.clip(target_log.astype(np.float64), 0.0, 20.0)) / scale
    prediction_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    denominator = np.maximum(prediction_norm * target_norm, 1e-30)
    cosine = np.sum(prediction * target, axis=1) / denominator
    return 1.0 - np.clip(cosine, 0.0, 1.0)


def _make_compressors(config: dict) -> tuple[SpectralCompressor, SpectralCompressor]:
    section = config["spectral_teacher"]
    return (
        SpectralCompressor(int(section.get("pas_latent_dim", 96))),
        SpectralCompressor(int(section.get("pdp_latent_dim", 48))),
    )


def _target_matrix(
    targets: dict[str, np.ndarray],
    indices: np.ndarray,
    pas_compressor: SpectralCompressor,
    pdp_compressor: SpectralCompressor,
) -> np.ndarray:
    pas_latent = pas_compressor.transform(targets["pas_log"][indices].astype(np.float32))
    pdp_latent = pdp_compressor.transform(targets["pdp_log"][indices].astype(np.float32))
    return np.concatenate(
        [
            pas_latent,
            pdp_latent,
            targets["ue_log_energy"][indices].astype(np.float32),
            targets["log_power"][indices, None].astype(np.float32),
        ],
        axis=1,
    )


def _decode_prediction(
    prediction: np.ndarray,
    pas_compressor: SpectralCompressor,
    pdp_compressor: SpectralCompressor,
    pas_dim: int,
    pdp_dim: int,
    ue_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pas = pas_compressor.inverse_transform(prediction[:, :pas_dim])
    pdp = pdp_compressor.inverse_transform(prediction[:, pas_dim : pas_dim + pdp_dim])
    start = pas_dim + pdp_dim
    ue = prediction[:, start : start + ue_dim]
    power = prediction[:, start + ue_dim]
    return pas, pdp, ue, power


def train_oof_teacher(config: dict) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(int(config["seed"]))
    metadata, targets = _load_arrays(config)
    section = config["spectral_teacher"]
    device = choose_device(str(config["runtime"].get("device", "auto")))
    limit = int(config["runtime"].get("spectral_train_limit", 0) or 0)
    selected = _balanced_indices(metadata, targets, limit)
    selected_mask = np.zeros(len(targets["outage"]), dtype=np.bool_)
    selected_mask[selected] = True
    kernel_settings = _kernel_settings(section)
    if not kernel_settings:
        raise ValueError("At least one spectral GP kernel is required")
    kernels = tuple(name for name, _ in kernel_settings)
    local_settings = local_expert_settings(section)
    expert_names = kernels + tuple(name for name, _, _ in local_settings)
    expert_count = len(expert_names)
    pas_width = targets["pas_log"].shape[1]
    pdp_width = targets["pdp_log"].shape[1]
    ue_dim = targets["ue_log_energy"].shape[1]
    kernel_pas = np.zeros((expert_count, len(targets["outage"]), pas_width), dtype=np.float32)
    kernel_pdp = np.zeros((expert_count, len(targets["outage"]), pdp_width), dtype=np.float32)
    kernel_ue = np.zeros((expert_count, len(targets["outage"]), ue_dim), dtype=np.float32)
    kernel_power = np.zeros((expert_count, len(targets["outage"])), dtype=np.float32)
    kernel_uncertainty = np.zeros((expert_count, len(targets["outage"])), dtype=np.float32)
    available = np.zeros(len(targets["outage"]), dtype=np.bool_)
    outage_probability = np.zeros(len(targets["outage"]), dtype=np.float32)
    fold_records: list[dict[str, object]] = []
    signature = hashlib.sha256(
        json.dumps(
            {
                "seed": int(config["seed"]),
                "selected": hashlib.sha256(selected.tobytes()).hexdigest(),
                "kernels": kernels,
                "kernel_feature_mixes": [mix for _, mix in kernel_settings],
                "local_spectral_experts": local_settings,
                "pas_latent_dim": int(section.get("pas_latent_dim", 96)),
                "pdp_latent_dim": int(section.get("pdp_latent_dim", 48)),
                "gp_noise": float(section.get("gp_noise", 0.01)),
                "feature_length": float(section.get("feature_length", 1.0)),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    progress_dir = Path(section["oof_output_path"]).parent / "progress" / signature
    progress_dir.mkdir(parents=True, exist_ok=True)
    folds = sorted(np.unique(metadata["spectral_folds"][selected]).tolist())
    cells = sorted(np.unique(metadata["train_cells"][selected]).tolist())
    for fold in folds:
        for cell in cells:
            validation = selected[
                (metadata["spectral_folds"][selected] == fold)
                & (metadata["train_cells"][selected] == cell)
            ]
            training = selected[
                (metadata["spectral_folds"][selected] != fold)
                & (metadata["train_cells"][selected] == cell)
            ]
            spectral_training = training[~targets["outage"][training].astype(bool)]
            if len(validation) == 0 or len(spectral_training) < 2:
                continue
            progress_path = progress_dir / f"fold{int(fold)}_cell{int(cell)}.npz"
            if progress_path.is_file():
                with np.load(progress_path) as saved:
                    saved_indices = saved["indices"].astype(np.int64)
                    if np.array_equal(saved_indices, validation):
                        kernel_pas[:, validation] = saved["pas"]
                        kernel_pdp[:, validation] = saved["pdp"]
                        kernel_ue[:, validation] = saved["ue"]
                        kernel_power[:, validation] = saved["power"]
                        kernel_uncertainty[:, validation] = saved["uncertainty"]
                        outage_probability[validation] = saved["outage_probability"]
                        available[validation] = True
                        fold_records.append(
                            {
                                "fold": int(fold), "cell": int(cell),
                                "training": int(len(training)),
                                "spectral_training": int(len(spectral_training)),
                                "validation": int(len(validation)), "resumed": True,
                            }
                        )
                        continue
            pas_compressor, pdp_compressor = _make_compressors(config)
            pas_compressor.fit(
                targets["pas_log"][spectral_training].astype(np.float32),
                int(config["seed"]) + int(fold) * 17 + int(cell),
            )
            pdp_compressor.fit(
                targets["pdp_log"][spectral_training].astype(np.float32),
                int(config["seed"]) + int(fold) * 19 + int(cell),
            )
            target_matrix = _target_matrix(
                targets, spectral_training, pas_compressor, pdp_compressor
            )
            for kernel_index, (kernel_name, feature_mix) in enumerate(kernel_settings):
                model = SharedMultiOutputGP(
                    str(kernel_name),
                    noise=float(section.get("gp_noise", 0.01)),
                    feature_length=float(section.get("feature_length", 1.0)),
                    feature_mix=float(feature_mix),
                ).fit(
                    metadata["train_positions"][spectral_training],
                    metadata["train_geometry_features"][spectral_training],
                    target_matrix,
                    device,
                )
                prediction, uncertainty = model.predict(
                    metadata["train_positions"][validation],
                    metadata["train_geometry_features"][validation],
                    device,
                    int(section.get("prediction_batch_size", 128)),
                )
                pas, pdp, ue, power = _decode_prediction(
                    prediction,
                    pas_compressor,
                    pdp_compressor,
                    int(section.get("pas_latent_dim", 96)),
                    int(section.get("pdp_latent_dim", 48)),
                    ue_dim,
                )
                kernel_pas[kernel_index, validation] = pas
                kernel_pdp[kernel_index, validation] = pdp
                kernel_ue[kernel_index, validation] = ue
                kernel_power[kernel_index, validation] = power
                kernel_uncertainty[kernel_index, validation] = uncertainty
            for local_offset, (_, neighbors, distance_power) in enumerate(
                local_settings, start=len(kernels)
            ):
                pas, pdp, ue, power, uncertainty = local_spectral_prediction(
                    metadata["train_positions"][spectral_training],
                    metadata["train_positions"][validation],
                    targets["pas_log"][spectral_training],
                    targets["pdp_log"][spectral_training],
                    targets["ue_log_energy"][spectral_training],
                    targets["log_power"][spectral_training],
                    neighbors=neighbors,
                    distance_power=distance_power,
                )
                kernel_pas[local_offset, validation] = pas
                kernel_pdp[local_offset, validation] = pdp
                kernel_ue[local_offset, validation] = ue
                kernel_power[local_offset, validation] = power
                kernel_uncertainty[local_offset, validation] = uncertainty
            training_labels = targets["outage"][training].astype(np.int64)
            if len(np.unique(training_labels)) >= 2:
                classifier = OutageEnsemble(
                    seed=int(config["seed"]) + int(fold) * 23 + int(cell),
                    positive_weight=float(section.get("outage_positive_weight", 4.0)),
                    false_kill_cost=float(section.get("false_kill_cost", 0.56)),
                ).fit(metadata["train_geometry_features"][training], training_labels)
                outage_probability[validation] = classifier.predict_proba(
                    metadata["train_geometry_features"][validation]
                )
            else:
                outage_probability[validation] = float(training_labels.mean())
            available[validation] = True
            np.savez_compressed(
                progress_path,
                indices=validation,
                pas=kernel_pas[:, validation].astype(np.float16),
                pdp=kernel_pdp[:, validation].astype(np.float16),
                ue=kernel_ue[:, validation],
                power=kernel_power[:, validation],
                uncertainty=kernel_uncertainty[:, validation],
                outage_probability=outage_probability[validation],
            )
            fold_records.append(
                {
                    "fold": int(fold),
                    "cell": int(cell),
                    "training": int(len(training)),
                    "spectral_training": int(len(spectral_training)),
                    "validation": int(len(validation)),
                }
            )
    valid_nonzero = available & ~targets["outage"].astype(bool)
    if not np.any(valid_nonzero):
        raise RuntimeError("OOF spectral teacher produced no nonzero validation predictions")
    cell_count = int(np.max(metadata["train_cells"])) + 1
    pas_weights = np.zeros((cell_count, expert_count), dtype=np.float32)
    pdp_weights = np.zeros((cell_count, expert_count), dtype=np.float32)
    auxiliary_weights_by_cell = np.zeros((cell_count, expert_count), dtype=np.float32)
    ensemble_pas = np.zeros((len(targets["outage"]), pas_width), dtype=np.float32)
    ensemble_pdp = np.zeros((len(targets["outage"]), pdp_width), dtype=np.float32)
    ensemble_ue = np.zeros((len(targets["outage"]), ue_dim), dtype=np.float32)
    ensemble_power = np.zeros(len(targets["outage"]), dtype=np.float32)
    ensemble_uncertainty = np.zeros(len(targets["outage"]), dtype=np.float32)
    cell_metrics: list[dict[str, object]] = []
    for cell in range(cell_count):
        indices = np.flatnonzero(valid_nonzero & (metadata["train_cells"] == cell))
        pas_weights[cell] = convex_cosine_weights(
            kernel_pas[:, indices],
            targets["pas_log"][indices],
            PAS_LOG_SCALE,
            float(section.get("weight_grid_step", 0.05)),
        )
        pdp_weights[cell] = convex_cosine_weights(
            kernel_pdp[:, indices],
            targets["pdp_log"][indices],
            PDP_LOG_SCALE,
            float(section.get("weight_grid_step", 0.05)),
        )
        auxiliary_predictions = np.concatenate(
            [kernel_ue[:, indices], kernel_power[:, indices, None]], axis=2
        )
        auxiliary_targets = np.concatenate(
            [targets["ue_log_energy"][indices], targets["log_power"][indices, None]],
            axis=1,
        )
        auxiliary_weights_by_cell[cell] = convex_mse_weights(
            auxiliary_predictions,
            auxiliary_targets,
            float(section.get("weight_grid_step", 0.05)),
        )
        all_indices = np.flatnonzero(available & (metadata["train_cells"] == cell))
        ensemble_pas[all_indices] = ensemble_log_power_predictions(
            [kernel_pas[kernel, all_indices] for kernel in range(expert_count)],
            pas_weights[cell],
            PAS_LOG_SCALE,
        )
        ensemble_pdp[all_indices] = ensemble_log_power_predictions(
            [kernel_pdp[kernel, all_indices] for kernel in range(expert_count)],
            pdp_weights[cell],
            PDP_LOG_SCALE,
        )
        auxiliary_weights = auxiliary_weights_by_cell[cell]
        ensemble_ue[all_indices] = ensemble_predictions(
            [kernel_ue[kernel, all_indices] for kernel in range(expert_count)],
            auxiliary_weights,
        )
        ensemble_power[all_indices] = ensemble_predictions(
            [kernel_power[kernel, all_indices, None] for kernel in range(expert_count)],
            auxiliary_weights,
        )[:, 0]
        ensemble_uncertainty[all_indices] = ensemble_predictions(
            [kernel_uncertainty[kernel, all_indices, None] for kernel in range(expert_count)],
            auxiliary_weights,
        )[:, 0]
        cell_metrics.append(
            {
                "cell": cell,
                "samples": int(len(indices)),
                "pas_accuracy": float(1.0 - _cosine_loss(ensemble_pas[indices], targets["pas_log"][indices], PAS_LOG_SCALE).mean()),
                "pdp_accuracy": float(1.0 - _cosine_loss(ensemble_pdp[indices], targets["pdp_log"][indices], PDP_LOG_SCALE).mean()),
                "pas_weights": pas_weights[cell].tolist(),
                "pdp_weights": pdp_weights[cell].tolist(),
                "auxiliary_weights": auxiliary_weights.tolist(),
            }
        )
    raw_power_mae = float(
        np.mean(
            np.abs(
                ensemble_power[valid_nonzero]
                - targets["log_power"][valid_nonzero]
            )
        )
    )
    calibration_section = section.get("power_calibration", {})
    if bool(calibration_section.get("enabled", False)):
        slope_bounds = tuple(
            float(value)
            for value in calibration_section.get("slope_bounds", [0.6, 1.4])
        )
        power_calibration = fit_power_calibration(
            ensemble_power,
            targets["log_power"],
            metadata["train_cells"],
            np.flatnonzero(valid_nonzero),
            slope_bounds=slope_bounds,
        )
        calibrated_power, calibrated_ue = apply_power_calibration(
            ensemble_power[available],
            ensemble_ue[available],
            metadata["train_cells"][available],
            power_calibration,
        )
        ensemble_power[available] = calibrated_power
        ensemble_ue[available] = calibrated_ue
    else:
        power_calibration = np.column_stack(
            [
                np.zeros(cell_count, dtype=np.float32),
                np.zeros(cell_count, dtype=np.float32),
                np.ones(cell_count, dtype=np.float32),
            ]
        )
    outage_calibrator = OutageEnsemble(
        false_kill_cost=float(section.get("false_kill_cost", 0.56))
    )
    threshold = outage_calibrator.calibrate_threshold(
        outage_probability[available], targets["outage"][available]
    )
    thresholds_by_cell = np.full(cell_count, threshold, dtype=np.float32)
    for cell in range(cell_count):
        cell_indices = np.flatnonzero(
            available & (metadata["train_cells"] == cell)
        )
        if len(cell_indices):
            thresholds_by_cell[cell] = OutageEnsemble(
                false_kill_cost=float(section.get("false_kill_cost", 0.56))
            ).calibrate_threshold(
                outage_probability[cell_indices], targets["outage"][cell_indices]
            )
            cell_metrics[cell]["outage"] = binary_metrics(
                outage_probability[cell_indices],
                targets["outage"][cell_indices],
                float(thresholds_by_cell[cell]),
            )
    output_path = Path(section["oof_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pas_log=ensemble_pas.astype(np.float16),
        pdp_log=ensemble_pdp.astype(np.float16),
        ue_log_energy=ensemble_ue,
        log_power=ensemble_power,
        outage_probability=outage_probability,
        uncertainty=ensemble_uncertainty,
        available=available,
        pas_weights=pas_weights,
        pdp_weights=pdp_weights,
        auxiliary_weights=auxiliary_weights_by_cell,
        outage_threshold=np.asarray(threshold, dtype=np.float32),
        outage_threshold_by_cell=thresholds_by_cell,
        power_calibration=power_calibration,
    )
    report = {
        "stage": "spectral_teacher_oof",
        "kernels": list(kernels),
        "kernel_feature_mixes": [mix for _, mix in kernel_settings],
        "experts": list(expert_names),
        "local_spectral_experts": [
            {"name": name, "neighbors": neighbors, "distance_power": distance_power}
            for name, neighbors, distance_power in local_settings
        ],
        "selected_samples": int(len(selected)),
        "available_predictions": int(available.sum()),
        "nonzero_predictions": int(valid_nonzero.sum()),
        "pas_accuracy": float(1.0 - _cosine_loss(ensemble_pas[valid_nonzero], targets["pas_log"][valid_nonzero], PAS_LOG_SCALE).mean()),
        "pdp_accuracy": float(1.0 - _cosine_loss(ensemble_pdp[valid_nonzero], targets["pdp_log"][valid_nonzero], PDP_LOG_SCALE).mean()),
        "power_mae_log10": float(np.mean(np.abs(ensemble_power[valid_nonzero] - targets["log_power"][valid_nonzero]))),
        "raw_power_mae_log10": raw_power_mae,
        "outage": binary_metrics(outage_probability[available], targets["outage"][available], threshold),
        "outage_threshold_by_cell": thresholds_by_cell.tolist(),
        "power_calibration": power_calibration.tolist(),
        "cells": cell_metrics,
        "folds": fold_records,
        "output_path": str(output_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(section["oof_report_path"], report)
    return report


def train_final_teacher(config: dict) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(int(config["seed"]))
    metadata, targets = _load_arrays(config)
    section = config["spectral_teacher"]
    device = choose_device(str(config["runtime"].get("device", "auto")))
    kernel_settings = _kernel_settings(section)
    kernels = tuple(name for name, _ in kernel_settings)
    local_settings = local_expert_settings(section)
    expert_names = kernels + tuple(name for name, _, _ in local_settings)
    with np.load(section["oof_output_path"]) as source:
        pas_weights = source["pas_weights"].astype(np.float32)
        pdp_weights = source["pdp_weights"].astype(np.float32)
        if "auxiliary_weights" in source.files:
            auxiliary_weights_by_cell = source["auxiliary_weights"].astype(np.float32)
        else:
            auxiliary_weights_by_cell = 0.5 * (pas_weights + pdp_weights)
            auxiliary_weights_by_cell /= auxiliary_weights_by_cell.sum(axis=1, keepdims=True)
        outage_threshold = float(np.asarray(source["outage_threshold"]).item())
        outage_threshold_by_cell = (
            source["outage_threshold_by_cell"].astype(np.float32)
            if "outage_threshold_by_cell" in source.files
            else np.full(len(pas_weights), outage_threshold, dtype=np.float32)
        )
        power_calibration = (
            source["power_calibration"].astype(np.float32)
            if "power_calibration" in source.files
            else np.column_stack(
                [
                    np.zeros(len(pas_weights), dtype=np.float32),
                    np.zeros(len(pas_weights), dtype=np.float32),
                    np.ones(len(pas_weights), dtype=np.float32),
                ]
            )
        )
    test_positions = metadata["test_positions"].astype(np.float32)
    test_cells = metadata["test_cells"].astype(np.int64)
    test_features = metadata["test_geometry_features"].astype(np.float32)
    pas_width = targets["pas_log"].shape[1]
    pdp_width = targets["pdp_log"].shape[1]
    ue_dim = targets["ue_log_energy"].shape[1]
    test_pas = np.zeros((len(test_positions), pas_width), dtype=np.float32)
    test_pdp = np.zeros((len(test_positions), pdp_width), dtype=np.float32)
    test_ue = np.zeros((len(test_positions), ue_dim), dtype=np.float32)
    test_power = np.zeros(len(test_positions), dtype=np.float32)
    test_uncertainty = np.zeros(len(test_positions), dtype=np.float32)
    test_outage_probability = np.zeros(len(test_positions), dtype=np.float32)
    state: dict[str, object] = {
        "kernels": list(kernels),
        "kernel_feature_mixes": [mix for _, mix in kernel_settings],
        "experts": list(expert_names),
        "local_spectral_experts": local_settings,
        "pas_weights": pas_weights,
        "pdp_weights": pdp_weights,
        "auxiliary_weights": auxiliary_weights_by_cell,
        "outage_threshold": outage_threshold,
        "outage_threshold_by_cell": outage_threshold_by_cell,
        "power_calibration": power_calibration,
        "cells": {},
    }
    cell_records: list[dict[str, object]] = []
    final_signature = hashlib.sha256(
        json.dumps(
            {
                "seed": int(config["seed"]),
                "kernels": kernels,
                "kernel_feature_mixes": [mix for _, mix in kernel_settings],
                "local_spectral_experts": local_settings,
                "pas_weights": pas_weights.tolist(),
                "pdp_weights": pdp_weights.tolist(),
                "auxiliary_weights": auxiliary_weights_by_cell.tolist(),
                "pas_latent_dim": int(section.get("pas_latent_dim", 96)),
                "pdp_latent_dim": int(section.get("pdp_latent_dim", 48)),
                "gp_noise": float(section.get("gp_noise", 0.01)),
                "feature_length": float(section.get("feature_length", 1.0)),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    progress_dir = Path(section["test_output_path"]).parent / "progress" / final_signature
    progress_dir.mkdir(parents=True, exist_ok=True)
    for cell in sorted(np.unique(metadata["train_cells"]).tolist()):
        training = np.flatnonzero(metadata["train_cells"] == cell)
        spectral_training = training[~targets["outage"][training].astype(bool)]
        testing = np.flatnonzero(test_cells == cell)
        progress_path = progress_dir / f"cell{int(cell)}.pkl"
        if progress_path.is_file():
            with progress_path.open("rb") as handle:
                cached = pickle.load(handle)
            if np.array_equal(np.asarray(cached["testing"]), testing):
                test_pas[testing] = cached["pas"]
                test_pdp[testing] = cached["pdp"]
                test_ue[testing] = cached["ue"]
                test_power[testing] = cached["power"]
                test_uncertainty[testing] = cached["uncertainty"]
                test_outage_probability[testing] = cached["outage_probability"]
                state["cells"][int(cell)] = cached["state"]
                cell_records.append(
                    {
                        "cell": int(cell), "training": int(len(training)),
                        "spectral_training": int(len(spectral_training)),
                        "test": int(len(testing)), "resumed": True,
                    }
                )
                continue
        pas_compressor, pdp_compressor = _make_compressors(config)
        pas_compressor.fit(targets["pas_log"][spectral_training].astype(np.float32), int(config["seed"]) + int(cell))
        pdp_compressor.fit(targets["pdp_log"][spectral_training].astype(np.float32), int(config["seed"]) + 7 + int(cell))
        target_matrix = _target_matrix(targets, spectral_training, pas_compressor, pdp_compressor)
        predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        gp_states: list[dict] = []
        for kernel_name, feature_mix in kernel_settings:
            model = SharedMultiOutputGP(
                str(kernel_name),
                noise=float(section.get("gp_noise", 0.01)),
                feature_length=float(section.get("feature_length", 1.0)),
                feature_mix=float(feature_mix),
            ).fit(
                metadata["train_positions"][spectral_training],
                metadata["train_geometry_features"][spectral_training],
                target_matrix,
                device,
            )
            raw, uncertainty = model.predict(
                test_positions[testing],
                test_features[testing],
                device,
                int(section.get("prediction_batch_size", 128)),
            )
            decoded = _decode_prediction(
                raw,
                pas_compressor,
                pdp_compressor,
                int(section.get("pas_latent_dim", 96)),
                int(section.get("pdp_latent_dim", 48)),
                ue_dim,
            )
            predictions.append((*decoded, uncertainty))
            gp_states.append(model.state_dict())
        for _, neighbors, distance_power in local_settings:
            predictions.append(
                local_spectral_prediction(
                    metadata["train_positions"][spectral_training],
                    test_positions[testing],
                    targets["pas_log"][spectral_training],
                    targets["pdp_log"][spectral_training],
                    targets["ue_log_energy"][spectral_training],
                    targets["log_power"][spectral_training],
                    neighbors=neighbors,
                    distance_power=distance_power,
                )
            )
        test_pas[testing] = ensemble_log_power_predictions(
            [value[0] for value in predictions], pas_weights[cell], PAS_LOG_SCALE
        )
        test_pdp[testing] = ensemble_log_power_predictions(
            [value[1] for value in predictions], pdp_weights[cell], PDP_LOG_SCALE
        )
        auxiliary_weights = auxiliary_weights_by_cell[cell]
        test_ue[testing] = ensemble_predictions([value[2] for value in predictions], auxiliary_weights)
        test_power[testing] = ensemble_predictions([value[3][:, None] for value in predictions], auxiliary_weights)[:, 0]
        test_uncertainty[testing] = ensemble_predictions([value[4][:, None] for value in predictions], auxiliary_weights)[:, 0]
        classifier = OutageEnsemble(
            seed=int(config["seed"]) + int(cell),
            positive_weight=float(section.get("outage_positive_weight", 4.0)),
            false_kill_cost=float(section.get("false_kill_cost", 0.56)),
        ).fit(metadata["train_geometry_features"][training], targets["outage"][training].astype(np.int64))
        classifier.threshold = float(outage_threshold_by_cell[int(cell)])
        test_outage_probability[testing] = classifier.predict_proba(test_features[testing])
        state["cells"][int(cell)] = {
            "pas_compressor": pas_compressor.state_dict(),
            "pdp_compressor": pdp_compressor.state_dict(),
            "gps": gp_states,
            "outage": classifier.state_dict(),
        }
        with progress_path.open("wb") as handle:
            pickle.dump(
                {
                    "testing": testing,
                    "pas": test_pas[testing], "pdp": test_pdp[testing],
                    "ue": test_ue[testing], "power": test_power[testing],
                    "uncertainty": test_uncertainty[testing],
                    "outage_probability": test_outage_probability[testing],
                    "state": state["cells"][int(cell)],
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        cell_records.append(
            {
                "cell": int(cell),
                "training": int(len(training)),
                "spectral_training": int(len(spectral_training)),
                "test": int(len(testing)),
            }
        )
    test_power, test_ue = apply_power_calibration(
        test_power,
        test_ue,
        test_cells,
        power_calibration,
    )
    prior_path = Path(section["test_output_path"])
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prior_path,
        pas_log=test_pas.astype(np.float16),
        pdp_log=test_pdp.astype(np.float16),
        ue_log_energy=test_ue,
        log_power=test_power,
        outage_probability=test_outage_probability,
        uncertainty=test_uncertainty,
        outage_threshold=np.asarray(outage_threshold, dtype=np.float32),
        outage_threshold_by_cell=outage_threshold_by_cell,
        power_calibration=power_calibration,
    )
    model_path = Path(section["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "stage": "spectral_teacher_final",
        "experts": list(expert_names),
        "local_spectral_experts": [
            {"name": name, "neighbors": neighbors, "distance_power": distance_power}
            for name, neighbors, distance_power in local_settings
        ],
        "cells": cell_records,
        "test_samples": int(len(test_positions)),
        "predicted_outages": int(
            np.sum(
                test_outage_probability
                >= outage_threshold_by_cell[test_cells]
            )
        ),
        "outage_threshold": outage_threshold,
        "outage_threshold_by_cell": outage_threshold_by_cell.tolist(),
        "power_calibration": power_calibration.tolist(),
        "model_path": str(model_path),
        "test_output_path": str(prior_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(section["final_report_path"], report)
    return report
