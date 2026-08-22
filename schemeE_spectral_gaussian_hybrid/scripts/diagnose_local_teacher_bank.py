from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from scheme_e.carrier_transport import CarrierFit
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.gp import (
    convex_cosine_weights,
    convex_mse_weights,
    ensemble_log_power_predictions,
    ensemble_predictions,
)
from scheme_e.hybrid_training import evaluate_hybrid, load_hybrid_checkpoint
from scheme_e.local_spectral import local_spectral_prediction
from scheme_e.spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE


def _parse_experts(value: str) -> list[tuple[str, int, float]]:
    result: list[tuple[str, int, float]] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        neighbors_text, power_text = raw.split(":", maxsplit=1)
        neighbors = int(neighbors_text)
        distance_power = float(power_text)
        if neighbors < 1 or distance_power <= 0.0:
            raise ValueError("Expert settings must use positive neighbors and power")
        result.append((f"idw{neighbors}_p{distance_power:g}", neighbors, distance_power))
    if not result:
        raise ValueError("At least one local expert is required")
    return result


def _power_cosine(
    prediction_log: np.ndarray,
    target_log: np.ndarray,
    scale: float,
) -> np.ndarray:
    prediction = np.expm1(np.clip(prediction_log.astype(np.float32), 0.0, 20.0))
    target = np.expm1(np.clip(target_log.astype(np.float32), 0.0, 20.0))
    prediction /= float(scale)
    target /= float(scale)
    numerator = np.einsum("ij,ij->i", prediction, target, optimize=True)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    return np.clip(numerator / np.maximum(denominator, 1e-30), 0.0, 1.0)


def _gate_features(
    metadata: dict[str, np.ndarray],
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    scale: float,
) -> np.ndarray:
    count = predictions.shape[1]
    expert_count = predictions.shape[0]
    base_power = np.expm1(
        np.clip(predictions[0].astype(np.float32), 0.0, 20.0)
    ) / float(scale)
    base_norm = np.linalg.norm(base_power, axis=1)
    agreement = np.empty((count, expert_count), dtype=np.float32)
    concentration = np.empty_like(agreement)
    peak_fraction = np.empty_like(agreement)
    for expert in range(expert_count):
        power = np.expm1(
            np.clip(predictions[expert].astype(np.float32), 0.0, 20.0)
        ) / float(scale)
        norm = np.linalg.norm(power, axis=1)
        agreement[:, expert] = np.clip(
            np.einsum("ij,ij->i", power, base_power, optimize=True)
            / np.maximum(norm * base_norm, 1e-30),
            0.0,
            1.0,
        )
        total = power.sum(axis=1)
        concentration[:, expert] = norm / np.maximum(total, 1e-30)
        peak_fraction[:, expert] = power.max(axis=1) / np.maximum(total, 1e-30)
    cells = metadata["train_cells"][:count].astype(np.int64)
    cell_count = int(cells.max()) + 1
    one_hot = np.eye(cell_count, dtype=np.float32)[cells]
    features = np.concatenate(
        [
            metadata["train_positions"][:count, :2].astype(np.float32),
            metadata["train_geometry_features"][:count].astype(np.float32),
            one_hot,
            uncertainties[:, :count].T.astype(np.float32),
            agreement,
            concentration,
            peak_fraction,
        ],
        axis=1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def _expert_scores(
    predictions: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    scale: float,
) -> np.ndarray:
    return np.stack(
        [
            _power_cosine(value[indices], targets[indices], scale)
            for value in predictions
        ],
        axis=1,
    ).astype(np.float32)


def _softmax_scores(scores: np.ndarray, temperature: float) -> np.ndarray:
    centered = scores / float(temperature)
    centered -= centered.max(axis=1, keepdims=True)
    weights = np.exp(np.clip(centered, -40.0, 0.0))
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)


def _weighted_log_predictions(
    predictions: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    scale: float,
    batch_size: int = 128,
) -> np.ndarray:
    output = np.empty((len(indices), predictions.shape[2]), dtype=np.float32)
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        local = indices[start:stop]
        power = np.expm1(
            np.clip(predictions[:, local].astype(np.float32), 0.0, 20.0)
        ) / float(scale)
        mixed = np.einsum(
            "be,ebd->bd", weights[start:stop], power, optimize=True
        )
        output[start:stop] = np.log1p(float(scale) * np.maximum(mixed, 0.0))
    return output


def _adaptive_gate(
    metadata: dict[str, np.ndarray],
    targets: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    training_indices: np.ndarray,
    base_weights_by_cell: np.ndarray,
    scale: float,
    seed: int,
) -> tuple[np.ndarray, ExtraTreesRegressor, dict[str, object]]:
    features = _gate_features(metadata, predictions, uncertainties, scale)
    scores = _expert_scores(predictions, targets, training_indices, scale)
    folds = metadata["spectral_folds"][training_indices]
    holdout_fold = int(sorted(np.unique(folds).tolist())[-1])
    fit_mask = folds != holdout_fold
    holdout_mask = ~fit_mask
    fit_indices = training_indices[fit_mask]
    holdout_indices = training_indices[holdout_mask]
    development = ExtraTreesRegressor(
        n_estimators=192,
        max_depth=12,
        min_samples_leaf=8,
        max_features=0.75,
        n_jobs=-1,
        random_state=int(seed),
    )
    development.fit(features[fit_indices], scores[fit_mask])
    predicted_scores = development.predict(features[holdout_indices]).astype(np.float32)
    cells = metadata["train_cells"][: len(features)].astype(np.int64)
    alpha_grid = (0.25, 0.5, 0.75, 1.0)
    temperature_grid = (0.01, 0.02, 0.05, 0.1, 0.2)
    selections: list[dict[str, float | int]] = []
    for cell in range(len(base_weights_by_cell)):
        local_mask = cells[holdout_indices] == cell
        local_indices = holdout_indices[local_mask]
        if not len(local_indices):
            selections.append(
                {
                    "cell": cell,
                    "holdout_samples": 0,
                    "alpha": 0.0,
                    "temperature": 0.1,
                    "score": 0.0,
                }
            )
            continue
        fixed = np.repeat(
            base_weights_by_cell[cell][None], len(local_indices), axis=0
        )
        baseline_prediction = _weighted_log_predictions(
            predictions, local_indices, fixed, scale
        )
        best_score = float(
            _power_cosine(baseline_prediction, targets[local_indices], scale).mean()
        )
        best_alpha = 0.0
        best_temperature = 0.1
        for alpha in alpha_grid:
            for temperature in temperature_grid:
                adaptive = _softmax_scores(
                    predicted_scores[local_mask], temperature
                )
                weights = (1.0 - alpha) * fixed + alpha * adaptive
                prediction = _weighted_log_predictions(
                    predictions, local_indices, weights, scale
                )
                score = float(
                    _power_cosine(prediction, targets[local_indices], scale).mean()
                )
                if score > best_score:
                    best_score = score
                    best_alpha = float(alpha)
                    best_temperature = float(temperature)
        selections.append(
            {
                "cell": cell,
                "holdout_samples": int(len(local_indices)),
                "alpha": best_alpha,
                "temperature": best_temperature,
                "score": best_score,
            }
        )

    final_model = ExtraTreesRegressor(
        n_estimators=256,
        max_depth=12,
        min_samples_leaf=8,
        max_features=0.75,
        n_jobs=-1,
        random_state=int(seed) + 1,
    )
    final_model.fit(features[training_indices], scores)
    final_scores = final_model.predict(features).astype(np.float32)
    all_weights = np.empty(
        (len(features), predictions.shape[0]), dtype=np.float32
    )
    for selection in selections:
        cell = int(selection["cell"])
        mask = cells == cell
        fixed = np.repeat(base_weights_by_cell[cell][None], int(mask.sum()), axis=0)
        adaptive = _softmax_scores(
            final_scores[mask], float(selection["temperature"])
        )
        alpha = float(selection["alpha"])
        all_weights[mask] = (1.0 - alpha) * fixed + alpha * adaptive
    return all_weights, final_model, {
        "holdout_fold": holdout_fold,
        "feature_width": int(features.shape[1]),
        "selections": selections,
    }


def _local_bank(
    metadata: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    observed: np.ndarray,
    validation: np.ndarray,
    settings: list[tuple[str, int, float]],
) -> dict[str, list[np.ndarray]]:
    count = len(metadata["train_cells"])
    cells = metadata["train_cells"][:count]
    folds = metadata["spectral_folds"][:count]
    outages = targets["outage"][:count].astype(bool)
    result: dict[str, list[np.ndarray]] = {
        "pas_log": [],
        "pdp_log": [],
        "ue_log_energy": [],
        "log_power": [],
        "uncertainty": [],
    }
    for name, neighbors, distance_power in settings:
        pas = np.zeros((count, targets["pas_log"].shape[1]), dtype=np.float16)
        pdp = np.zeros((count, targets["pdp_log"].shape[1]), dtype=np.float16)
        ue = np.zeros((count, targets["ue_log_energy"].shape[1]), dtype=np.float32)
        power = np.zeros(count, dtype=np.float32)
        uncertainty = np.ones(count, dtype=np.float32)
        for cell in sorted(np.unique(cells).tolist()):
            cell_observed = observed[cells[observed] == cell]
            spectral_support = cell_observed[~outages[cell_observed]]
            for fold in sorted(np.unique(folds[cell_observed]).tolist()):
                query = cell_observed[folds[cell_observed] == fold]
                support = spectral_support[folds[spectral_support] != fold]
                if not len(query):
                    continue
                values = local_spectral_prediction(
                    metadata["train_positions"][support],
                    metadata["train_positions"][query],
                    targets["pas_log"][support],
                    targets["pdp_log"][support],
                    targets["ue_log_energy"][support],
                    targets["log_power"][support],
                    neighbors=neighbors,
                    distance_power=distance_power,
                )
                pas[query], pdp[query], ue[query], power[query], uncertainty[query] = values
            query = validation[cells[validation] == cell]
            if len(query):
                values = local_spectral_prediction(
                    metadata["train_positions"][spectral_support],
                    metadata["train_positions"][query],
                    targets["pas_log"][spectral_support],
                    targets["pdp_log"][spectral_support],
                    targets["ue_log_energy"][spectral_support],
                    targets["log_power"][spectral_support],
                    neighbors=neighbors,
                    distance_power=distance_power,
                )
                pas[query], pdp[query], ue[query], power[query], uncertainty[query] = values
        result["pas_log"].append(pas)
        result["pdp_log"].append(pdp)
        result["ue_log_energy"].append(ue)
        result["log_power"].append(power)
        result["uncertainty"].append(uncertainty)
        print(f"local teacher expert ready: {name}", flush=True)
    return result


def _hybrid_context(
    config: dict,
    metadata: dict[str, np.ndarray],
    spectral_targets: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    validation: np.ndarray,
    observed: np.ndarray,
    policy_path: Path,
) -> dict[str, object]:
    section = config["hybrid"]
    device = choose_device(str(config["runtime"].get("device", "auto")))
    checkpoint_path = Path(section["output_dir"]) / "best.pt"
    model, shape, checkpoint = load_hybrid_checkpoint(config, checkpoint_path, device)
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    summary = json.loads(
        (Path(section["output_dir"]) / "summary.json").read_text(encoding="utf-8")
    )
    strategy_name = str(summary.get("selected_reference_strategy", "nearest"))
    reference_strategy = {"name": "nearest", "top_k": 1}
    for candidate in section.get("reference_strategies", []):
        if str(candidate.get("name")) == strategy_name:
            reference_strategy = dict(candidate)
            break
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    outage_policy = {
        "threshold_by_cell": policy["outage_threshold_by_cell"],
        "soft_strength_by_cell": policy["soft_outage_strength_by_cell"],
    }
    carrier_payload = checkpoint.get("carrier_fit")
    carrier_fit = None
    if carrier_payload is not None:
        carrier_fit = CarrierFit(
            np.asarray(carrier_payload["wave_numbers"], dtype=np.float64),
            np.asarray(carrier_payload["qualities"], dtype=np.float64),
            np.asarray(carrier_payload["pair_counts"], dtype=np.int64),
        )
    power_bounds = checkpoint.get("power_bounds")
    if power_bounds is not None:
        power_bounds = np.asarray(power_bounds, dtype=np.float32)
    return {
        "model": model,
        "shape": shape,
        "channels": channels,
        "metadata": metadata,
        "priors": priors,
        "target_indices": validation,
        "observed_indices": observed,
        "geometry_mean": np.asarray(checkpoint["geometry_mean"], dtype=np.float32),
        "geometry_std": np.asarray(checkpoint["geometry_std"], dtype=np.float32),
        "device": device,
        "batch_size": int(section.get("validation_batch_size", 4)),
        "outage_threshold": float(checkpoint["outage_threshold"]),
        "projection_iterations": int(summary["selected_projection_iterations"]),
        "spectral_targets": spectral_targets,
        "power_bounds": power_bounds,
        "reference_strategy": reference_strategy,
        "outage_policy": outage_policy,
        "carrier_fit": carrier_fit,
        "transport_config": section.get("transport_seed", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test a leakage-free expanded bank of local spectral teachers"
    )
    parser.add_argument("--config", default="configs/v5_local_teacher.json")
    parser.add_argument("--policy", default="reports/generated/v5_fold0_policy.json")
    parser.add_argument(
        "--experts", default="1:1,2:1,4:1,8:1,16:1,4:2,8:2,16:2"
    )
    parser.add_argument(
        "--output-prior", default="artifacts/v6/fold0/local_bank_priors.npz"
    )
    parser.add_argument(
        "--adaptive-output-prior",
        default="artifacts/v6/fold0/adaptive_local_bank_priors.npz",
    )
    parser.add_argument(
        "--gate-model", default="artifacts/v6/fold0/adaptive_local_gate.pkl"
    )
    parser.add_argument(
        "--output", default="reports/generated/v6_local_teacher_bank.json"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(config["spectral"]["target_path"]) as source:
        targets = {name: source[name] for name in source.files}
    with np.load(config["spectral_teacher"]["oof_output_path"]) as source:
        base = {name: np.array(source[name], copy=True) for name in source.files}

    fold = int(config["split"]["validation_fold"])
    count = min(len(metadata["train_cells"]), len(base["available"]))
    indices = np.arange(count, dtype=np.int64)
    available = base["available"][:count].astype(bool)
    validation_mask = metadata["validation_masks"][fold][:count].astype(bool)
    validation = indices[available & validation_mask]
    observed = indices[available & ~validation_mask]
    nonzero_observed = observed[~targets["outage"][observed].astype(bool)]
    nonzero_validation = validation[~targets["outage"][validation].astype(bool)]
    settings = _parse_experts(args.experts)
    local = _local_bank(metadata, targets, observed, validation, settings)
    names = ["gp_local_base", *[name for name, _, _ in settings]]

    pas_predictions = np.stack(
        [base["pas_log"][:count], *local["pas_log"]], axis=0
    )
    pdp_predictions = np.stack(
        [base["pdp_log"][:count], *local["pdp_log"]], axis=0
    )
    ue_predictions = np.stack(
        [base["ue_log_energy"][:count], *local["ue_log_energy"]], axis=0
    )
    power_predictions = np.stack(
        [base["log_power"][:count], *local["log_power"]], axis=0
    )
    uncertainty_predictions = np.stack(
        [base["uncertainty"][:count], *local["uncertainty"]], axis=0
    )

    improved = {name: np.array(value, copy=True) for name, value in base.items()}
    cell_count = int(np.max(metadata["train_cells"][:count])) + 1
    pas_weights = np.zeros((cell_count, len(names)), dtype=np.float32)
    pdp_weights = np.zeros_like(pas_weights)
    auxiliary_weights = np.zeros_like(pas_weights)
    cell_reports: list[dict[str, object]] = []
    for cell in range(cell_count):
        train_cell = nonzero_observed[
            metadata["train_cells"][nonzero_observed] == cell
        ]
        apply_cell = indices[metadata["train_cells"][:count] == cell]
        validation_cell = nonzero_validation[
            metadata["train_cells"][nonzero_validation] == cell
        ]
        pas_weights[cell] = convex_cosine_weights(
            pas_predictions[:, train_cell],
            targets["pas_log"][train_cell],
            PAS_LOG_SCALE,
        )
        pdp_weights[cell] = convex_cosine_weights(
            pdp_predictions[:, train_cell],
            targets["pdp_log"][train_cell],
            PDP_LOG_SCALE,
        )
        auxiliary_weights[cell] = convex_mse_weights(
            np.concatenate(
                [
                    ue_predictions[:, train_cell],
                    power_predictions[:, train_cell, None],
                ],
                axis=2,
            ),
            np.concatenate(
                [
                    targets["ue_log_energy"][train_cell],
                    targets["log_power"][train_cell, None],
                ],
                axis=1,
            ),
        )
        improved["pas_log"][apply_cell] = ensemble_log_power_predictions(
            [value[apply_cell] for value in pas_predictions],
            pas_weights[cell],
            PAS_LOG_SCALE,
        ).astype(improved["pas_log"].dtype)
        improved["pdp_log"][apply_cell] = ensemble_log_power_predictions(
            [value[apply_cell] for value in pdp_predictions],
            pdp_weights[cell],
            PDP_LOG_SCALE,
        ).astype(improved["pdp_log"].dtype)
        improved["ue_log_energy"][apply_cell] = ensemble_predictions(
            [value[apply_cell] for value in ue_predictions], auxiliary_weights[cell]
        )
        improved["log_power"][apply_cell] = ensemble_predictions(
            [value[apply_cell, None] for value in power_predictions],
            auxiliary_weights[cell],
        )[:, 0]
        improved["uncertainty"][apply_cell] = ensemble_predictions(
            [value[apply_cell, None] for value in uncertainty_predictions],
            auxiliary_weights[cell],
        )[:, 0]
        cell_reports.append(
            {
                "cell": cell,
                "validation_samples": int(len(validation_cell)),
                "pas_accuracy": float(
                    _power_cosine(
                        improved["pas_log"][validation_cell],
                        targets["pas_log"][validation_cell],
                        PAS_LOG_SCALE,
                    ).mean()
                ),
                "pdp_accuracy": float(
                    _power_cosine(
                        improved["pdp_log"][validation_cell],
                        targets["pdp_log"][validation_cell],
                        PDP_LOG_SCALE,
                    ).mean()
                ),
                "pas_weights": pas_weights[cell].tolist(),
                "pdp_weights": pdp_weights[cell].tolist(),
                "auxiliary_weights": auxiliary_weights[cell].tolist(),
            }
        )

    improved["local_bank_names"] = np.asarray(names)
    improved["local_bank_pas_weights"] = pas_weights
    improved["local_bank_pdp_weights"] = pdp_weights
    improved["local_bank_auxiliary_weights"] = auxiliary_weights
    output_prior = Path(args.output_prior)
    output_prior.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_prior, **improved)

    pas_gate_weights, pas_gate, pas_gate_report = _adaptive_gate(
        metadata,
        targets["pas_log"],
        pas_predictions,
        uncertainty_predictions,
        nonzero_observed,
        pas_weights,
        PAS_LOG_SCALE,
        int(config["seed"]) + 601,
    )
    pdp_gate_weights, pdp_gate, pdp_gate_report = _adaptive_gate(
        metadata,
        targets["pdp_log"],
        pdp_predictions,
        uncertainty_predictions,
        nonzero_observed,
        pdp_weights,
        PDP_LOG_SCALE,
        int(config["seed"]) + 607,
    )
    adaptive = {name: np.array(value, copy=True) for name, value in base.items()}
    adaptive["pas_log"][:count] = _weighted_log_predictions(
        pas_predictions,
        indices,
        pas_gate_weights,
        PAS_LOG_SCALE,
    ).astype(adaptive["pas_log"].dtype)
    adaptive["pdp_log"][:count] = _weighted_log_predictions(
        pdp_predictions,
        indices,
        pdp_gate_weights,
        PDP_LOG_SCALE,
    ).astype(adaptive["pdp_log"].dtype)
    adaptive["local_bank_names"] = np.asarray(names)
    adaptive["adaptive_pas_alpha_by_cell"] = np.asarray(
        [value["alpha"] for value in pas_gate_report["selections"]],
        dtype=np.float32,
    )
    adaptive["adaptive_pas_temperature_by_cell"] = np.asarray(
        [value["temperature"] for value in pas_gate_report["selections"]],
        dtype=np.float32,
    )
    adaptive["adaptive_pdp_alpha_by_cell"] = np.asarray(
        [value["alpha"] for value in pdp_gate_report["selections"]],
        dtype=np.float32,
    )
    adaptive["adaptive_pdp_temperature_by_cell"] = np.asarray(
        [value["temperature"] for value in pdp_gate_report["selections"]],
        dtype=np.float32,
    )
    adaptive_output_prior = Path(args.adaptive_output_prior)
    adaptive_output_prior.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(adaptive_output_prior, **adaptive)
    gate_model_path = Path(args.gate_model)
    gate_model_path.parent.mkdir(parents=True, exist_ok=True)
    with gate_model_path.open("wb") as handle:
        pickle.dump(
            {
                "names": names,
                "pas_model": pas_gate,
                "pdp_model": pdp_gate,
                "pas_report": pas_gate_report,
                "pdp_report": pdp_gate_report,
            },
            handle,
        )

    base_pas = _power_cosine(
        base["pas_log"][nonzero_validation],
        targets["pas_log"][nonzero_validation],
        PAS_LOG_SCALE,
    )
    base_pdp = _power_cosine(
        base["pdp_log"][nonzero_validation],
        targets["pdp_log"][nonzero_validation],
        PDP_LOG_SCALE,
    )
    improved_pas = _power_cosine(
        improved["pas_log"][nonzero_validation],
        targets["pas_log"][nonzero_validation],
        PAS_LOG_SCALE,
    )
    improved_pdp = _power_cosine(
        improved["pdp_log"][nonzero_validation],
        targets["pdp_log"][nonzero_validation],
        PDP_LOG_SCALE,
    )
    adaptive_pas = _power_cosine(
        adaptive["pas_log"][nonzero_validation],
        targets["pas_log"][nonzero_validation],
        PAS_LOG_SCALE,
    )
    adaptive_pdp = _power_cosine(
        adaptive["pdp_log"][nonzero_validation],
        targets["pdp_log"][nonzero_validation],
        PDP_LOG_SCALE,
    )
    pas_oracle = np.max(
        np.stack(
            [
                _power_cosine(
                    value[nonzero_validation],
                    targets["pas_log"][nonzero_validation],
                    PAS_LOG_SCALE,
                )
                for value in pas_predictions
            ],
            axis=1,
        ),
        axis=1,
    )
    pdp_oracle = np.max(
        np.stack(
            [
                _power_cosine(
                    value[nonzero_validation],
                    targets["pdp_log"][nonzero_validation],
                    PDP_LOG_SCALE,
                )
                for value in pdp_predictions
            ],
            axis=1,
        ),
        axis=1,
    )

    baseline_context = _hybrid_context(
        config,
        metadata,
        targets,
        base,
        validation,
        observed,
        Path(args.policy),
    )
    improved_context = dict(baseline_context)
    improved_context["priors"] = improved
    adaptive_context = dict(baseline_context)
    adaptive_context["priors"] = adaptive
    baseline_channel = evaluate_hybrid(**baseline_context)
    improved_channel = evaluate_hybrid(**improved_context)
    improved_projected = evaluate_hybrid(
        **improved_context,
        output_projection={
            "iterations": 2,
            "strength_by_cell": [0.5] * cell_count,
            "minimum_scale": 0.5,
            "maximum_scale": 2.0,
            "channel_source": "model",
        },
    )
    adaptive_channel = evaluate_hybrid(**adaptive_context)
    adaptive_projected = evaluate_hybrid(
        **adaptive_context,
        output_projection={
            "iterations": 2,
            "strength_by_cell": [0.5] * cell_count,
            "minimum_scale": 0.5,
            "maximum_scale": 2.0,
            "channel_source": "model",
        },
    )
    report = {
        "status": "PASS",
        "config": args.config,
        "experts": names,
        "leakage_free": True,
        "training_samples": int(len(nonzero_observed)),
        "validation_samples": int(len(nonzero_validation)),
        "teacher": {
            "base_pas": float(base_pas.mean()),
            "base_pdp": float(base_pdp.mean()),
            "improved_pas": float(improved_pas.mean()),
            "improved_pdp": float(improved_pdp.mean()),
            "adaptive_pas": float(adaptive_pas.mean()),
            "adaptive_pdp": float(adaptive_pdp.mean()),
            "oracle_pas": float(pas_oracle.mean()),
            "oracle_pdp": float(pdp_oracle.mean()),
        },
        "adaptive_gate": {
            "pas": pas_gate_report,
            "pdp": pdp_gate_report,
            "model_path": str(gate_model_path),
        },
        "cells": cell_reports,
        "channel": {
            "base": baseline_channel,
            "improved": improved_channel,
            "improved_projected": improved_projected,
            "adaptive": adaptive_channel,
            "adaptive_projected": adaptive_projected,
        },
        "output_prior": str(output_prior),
        "adaptive_output_prior": str(adaptive_output_prior),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
