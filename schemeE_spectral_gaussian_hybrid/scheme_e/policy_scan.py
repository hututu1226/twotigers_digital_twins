from __future__ import annotations

import itertools
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as functional

from .carrier_transport import CarrierFit, select_transport_candidates
from .config import choose_device, save_json
from .hybrid_training import (
    _normalized_geometry,
    _prior_batch,
    _reference_context_batch,
    _transport_batch,
    load_hybrid_checkpoint,
)
from .metrics import pas_spectrum, pdp_spectrum, official_score
from .reference import build_reference_candidates
from .reference_context import select_reference_candidates


def _row_cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    prediction = prediction / prediction.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    target_norm = target.norm(dim=-1, keepdim=True)
    target = target / target_norm.clamp_min(1e-30)
    cosine = functional.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    return cosine.clamp(0.0, 1.0).masked_fill(target_norm[..., 0] <= 1e-30, 0.0)


def _sample_spectral_scores(
    prediction: torch.Tensor, target: torch.Tensor, shape: object
) -> tuple[torch.Tensor, torch.Tensor]:
    pas = _row_cosine(pas_spectrum(prediction, shape), pas_spectrum(target, shape))
    pdp = _row_cosine(pdp_spectrum(prediction), pdp_spectrum(target))
    return pas.flatten(1).mean(1), pdp.flatten(1).mean(1)


@torch.no_grad()
def collect_validation_statistics(
    config: dict,
    checkpoint_path: str | Path,
    projection_iterations: int,
    reference_strategy: dict[str, object],
) -> dict[str, np.ndarray]:
    device = choose_device(str(config["runtime"].get("device", "auto")))
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(config["spectral_teacher"]["oof_output_path"]) as source:
        priors = {name: source[name] for name in source.files}
    with np.load(config["spectral"]["target_path"]) as source:
        spectral_targets = {name: source[name] for name in source.files}
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    fold = int(config["split"]["validation_fold"])
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    validation = np.flatnonzero(validation_mask)
    observed = np.flatnonzero(~validation_mask)
    model, shape, checkpoint = load_hybrid_checkpoint(
        config, checkpoint_path, device
    )
    carrier_payload = checkpoint.get("carrier_fit")
    carrier_fit = None
    if carrier_payload is not None:
        carrier_fit = CarrierFit(
            np.asarray(carrier_payload["wave_numbers"], dtype=np.float64),
            np.asarray(carrier_payload["qualities"], dtype=np.float64),
            np.asarray(carrier_payload["pair_counts"], dtype=np.int64),
        )
    transport_config = config["hybrid"].get("transport_seed", {})
    transport_count = int(transport_config.get("count", 8)) if carrier_fit else 1
    candidates, distances = build_reference_candidates(
        metadata["train_positions"][validation],
        metadata["train_cells"][validation],
        metadata["train_positions"][observed],
        metadata["train_cells"][observed],
        metadata["outage"][observed],
        top_k=max(1, transport_count, int(reference_strategy.get("top_k", 1))),
        target_global_indices=validation,
        observed_global_indices=observed,
    )
    candidate_globals = observed[candidates]
    geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    geometry_std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    if str(reference_strategy.get("name", "nearest")) == "nearest":
        references = candidate_globals[:, 0]
    else:
        references = select_reference_candidates(
            candidate_globals,
            distances,
            _normalized_geometry(
                metadata["train_geometry_features"],
                validation,
                geometry_mean,
                geometry_std,
            ),
            np.clip(
                (metadata["train_geometry_features"] - geometry_mean) / geometry_std,
                -8.0,
                8.0,
            ),
            priors["pas_log"][validation].astype(np.float32),
            priors["pdp_log"][validation].astype(np.float32),
            spectral_targets["pas_log"].astype(np.float32),
            spectral_targets["pdp_log"].astype(np.float32),
            reference_strategy,
        )
    transport_indices = None
    transport_distances = None
    if carrier_fit is not None:
        transport_local, transport_distances = select_transport_candidates(
            candidates, distances, transport_count
        )
        transport_indices = observed[transport_local]
    power_bounds = checkpoint.get("power_bounds")
    if power_bounds is not None:
        power_bounds = np.asarray(power_bounds, dtype=np.float32)
    result = {
        "indices": validation.astype(np.int64),
        "cells": metadata["train_cells"][validation].astype(np.int64),
        "true_outage": metadata["outage"][validation].astype(np.bool_),
        "outage_probability": priors["outage_probability"][validation].astype(np.float32),
        "pas": np.zeros(len(validation), dtype=np.float64),
        "pdp": np.zeros(len(validation), dtype=np.float64),
        "prediction_energy": np.zeros(len(validation), dtype=np.float64),
        "target_energy": np.zeros(len(validation), dtype=np.float64),
        "cross_real": np.zeros(len(validation), dtype=np.float64),
    }
    model.eval()
    batch_size = int(config["hybrid"].get("validation_batch_size", 2))
    for start in range(0, len(validation), batch_size):
        stop = min(start + batch_size, len(validation))
        indices = validation[start:stop]
        reference_indices = references[start:stop]
        reference = torch.as_tensor(
            np.asarray(channels[reference_indices]), device=device
        )
        target = torch.as_tensor(np.asarray(channels[indices]), device=device)
        reference_context = None
        if model.condition_encoder.reference_dim:
            reference_context = _reference_context_batch(
                metadata,
                priors,
                spectral_targets,
                indices,
                reference_indices,
                geometry_mean,
                geometry_std,
            )
        inputs = _prior_batch(
            priors,
            metadata,
            indices,
            geometry_mean,
            geometry_std,
            device,
            power_bounds=power_bounds,
            reference_context=reference_context,
        )
        transport_channel = None
        if carrier_fit is not None:
            if transport_indices is None or transport_distances is None:
                raise AssertionError("transport policy candidates are missing")
            transport_channel, transport_context = _transport_batch(
                channels,
                metadata,
                indices,
                transport_indices[start:stop],
                transport_distances[start:stop],
                carrier_fit,
                device,
                distance_power=float(transport_config.get("distance_power", 2.0)),
            )
            inputs["transport_context"] = transport_context
        prediction = model(
            reference,
            transport_channel=transport_channel,
            projection_iterations=int(projection_iterations),
            **inputs,
        )["channel"]
        pas, pdp = _sample_spectral_scores(prediction, target, shape)
        result["pas"][start:stop] = pas.cpu().numpy()
        result["pdp"][start:stop] = pdp.cpu().numpy()
        result["prediction_energy"][start:stop] = (
            prediction.abs().square().sum(dim=(1, 2, 3)).double().cpu().numpy()
        )
        result["target_energy"][start:stop] = (
            target.abs().square().sum(dim=(1, 2, 3)).double().cpu().numpy()
        )
        result["cross_real"][start:stop] = (
            (prediction * target.conj()).real.sum(dim=(1, 2, 3)).double().cpu().numpy()
        )
    return result


def _score_policy(
    stats: dict[str, np.ndarray],
    thresholds: np.ndarray,
    strengths: np.ndarray,
) -> dict[str, float | int]:
    probability = stats["outage_probability"].astype(np.float64)
    cells = stats["cells"].astype(np.int64)
    threshold = thresholds[cells]
    strength = strengths[cells]
    hard = probability >= threshold
    power_scale = np.power(np.maximum(1.0 - probability, 1e-4), strength)
    amplitude = np.sqrt(power_scale)
    amplitude[hard] = 0.0
    true_nonzero = ~stats["true_outage"].astype(bool)
    pas = float(np.mean(np.where(hard[true_nonzero], 0.0, stats["pas"][true_nonzero])))
    pdp = float(np.mean(np.where(hard[true_nonzero], 0.0, stats["pdp"][true_nonzero])))
    numerator = np.sum(
        amplitude * amplitude * stats["prediction_energy"]
        + stats["target_energy"]
        - 2.0 * amplitude * stats["cross_real"]
    )
    denominator = max(float(np.sum(stats["target_energy"])), 1e-30)
    channel_nmse = float(max(numerator, 0.0) / denominator)
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": channel_nmse,
        "score": official_score(pas, pdp, channel_nmse),
        "predicted_outages": int(hard.sum()),
    }


def scan_outage_policy(
    config: dict,
    checkpoint_path: str | Path,
    projection_iterations: int,
    reference_strategy: dict[str, object],
    output_path: str | Path,
) -> dict[str, object]:
    started = time.perf_counter()
    stats = collect_validation_statistics(
        config, checkpoint_path, projection_iterations, reference_strategy
    )
    section = config.get("policy_scan", {})
    threshold_candidates = sorted(
        set(float(value) for value in section.get(
            "threshold_candidates",
            [0.2, 0.3, 0.4, 0.46, 0.55, 0.65, 0.75, 0.85, 0.92, 0.97, 0.99, 0.999],
        ))
    )
    strength_candidates = sorted(
        set(float(value) for value in section.get(
            "soft_strength_candidates", [0.0, 0.5, 1.0, 2.0, 4.0]
        ))
    )
    cell_count = int(stats["cells"].max()) + 1
    single_cell: list[list[tuple[float, float]]] = []
    for _ in range(cell_count):
        single_cell.append(list(itertools.product(threshold_candidates, strength_candidates)))
    best_metrics: dict[str, float | int] | None = None
    best_thresholds: np.ndarray | None = None
    best_strengths: np.ndarray | None = None
    for combination in itertools.product(*single_cell):
        thresholds = np.asarray([value[0] for value in combination], dtype=np.float64)
        strengths = np.asarray([value[1] for value in combination], dtype=np.float64)
        metrics = _score_policy(stats, thresholds, strengths)
        ordering = (
            float(metrics["score"]),
            float(thresholds.mean()),
            -float(strengths.mean()),
        )
        current = (
            -float("inf"), -float("inf"), -float("inf")
        ) if best_metrics is None else (
            float(best_metrics["score"]),
            float(best_thresholds.mean()),
            -float(best_strengths.mean()),
        )
        if ordering > current:
            best_metrics = metrics
            best_thresholds = thresholds
            best_strengths = strengths
    assert best_metrics is not None and best_thresholds is not None and best_strengths is not None
    baseline_threshold = float(
        np.asarray(
            np.load(config["spectral_teacher"]["oof_output_path"])["outage_threshold"]
        ).reshape(-1)[0]
    )
    baseline = _score_policy(
        stats,
        np.full(cell_count, baseline_threshold, dtype=np.float64),
        np.zeros(cell_count, dtype=np.float64),
    )
    report = {
        "status": "PASS",
        "samples": int(len(stats["indices"])),
        "projection_iterations": int(projection_iterations),
        "reference_strategy": reference_strategy,
        "baseline": baseline,
        "selected": best_metrics,
        "score_gain": float(best_metrics["score"] - baseline["score"]),
        "outage_threshold_by_cell": best_thresholds.tolist(),
        "soft_outage_strength_by_cell": best_strengths.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output_path, report)
    return report
