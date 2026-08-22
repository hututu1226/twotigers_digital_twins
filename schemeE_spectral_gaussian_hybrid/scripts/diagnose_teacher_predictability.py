from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import _bootstrap  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree

from scheme_e.config import load_config, save_json
from scheme_e.gp import convex_cosine_weights, ensemble_log_power_predictions
from scheme_e.spectral_compression import SpectralCompressor
from scheme_e.spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: source[name] for name in source.files}


def _linear(log_values: np.ndarray, scale: float) -> np.ndarray:
    return np.expm1(np.clip(log_values.astype(np.float32), 0.0, 20.0)) / float(scale)


def _log(linear_values: np.ndarray, scale: float) -> np.ndarray:
    return np.log1p(float(scale) * np.maximum(linear_values, 0.0)).astype(np.float32)


def _row_cosine(
    prediction_log: np.ndarray, target_log: np.ndarray, scale: float
) -> np.ndarray:
    prediction = _linear(prediction_log, scale).astype(np.float64)
    target = _linear(target_log, scale).astype(np.float64)
    numerator = np.einsum("ij,ij->i", prediction, target, optimize=True)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1)
    return np.clip(numerator / np.maximum(denominator, 1e-30), 0.0, 1.0)


def _neighbor_predictions(
    metadata: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    validation: np.ndarray,
    observed_nonzero: np.ndarray,
    *,
    neighbors: int,
    distance_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pas = np.empty((len(validation), targets["pas_log"].shape[1]), dtype=np.float32)
    pdp = np.empty((len(validation), targets["pdp_log"].shape[1]), dtype=np.float32)
    nearest_distance = np.empty(len(validation), dtype=np.float32)
    for cell in np.unique(metadata["train_cells"]):
        target_rows = np.flatnonzero(metadata["train_cells"][validation] == cell)
        support = observed_nonzero[metadata["train_cells"][observed_nonzero] == cell]
        count = min(int(neighbors), len(support))
        distance, local = cKDTree(metadata["train_positions"][support, :2]).query(
            metadata["train_positions"][validation[target_rows], :2], k=count
        )
        distance = np.asarray(distance, dtype=np.float64).reshape(len(target_rows), count)
        local = np.asarray(local, dtype=np.int64).reshape(len(target_rows), count)
        source = support[local]
        nearest_distance[target_rows] = distance[:, 0]
        if count == 1:
            weights = np.ones((len(target_rows), 1), dtype=np.float64)
        else:
            weights = 1.0 / np.maximum(distance, 0.25) ** float(distance_power)
            weights /= weights.sum(axis=1, keepdims=True)
        pas_linear = _linear(targets["pas_log"][source], PAS_LOG_SCALE)
        pdp_linear = _linear(targets["pdp_log"][source], PDP_LOG_SCALE)
        pas[target_rows] = _log(
            np.einsum("ij,ijk->ik", weights, pas_linear, optimize=True), PAS_LOG_SCALE
        )
        pdp[target_rows] = _log(
            np.einsum("ij,ijk->ik", weights, pdp_linear, optimize=True), PDP_LOG_SCALE
        )
    return pas, pdp, nearest_distance


def _metrics(
    pas: np.ndarray,
    pdp: np.ndarray,
    targets: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, float]:
    return {
        "pas_accuracy": float(
            _row_cosine(pas, targets["pas_log"][indices], PAS_LOG_SCALE).mean()
        ),
        "pdp_accuracy": float(
            _row_cosine(pdp, targets["pdp_log"][indices], PDP_LOG_SCALE).mean()
        ),
    }


def _compression_ceiling(
    values: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    dimension: int,
    scale: float,
    seed: int,
) -> dict[str, float]:
    compressor = SpectralCompressor(dimension).fit(
        values[training].astype(np.float32), seed
    )
    reconstruction = compressor.inverse_transform(
        compressor.transform(values[validation].astype(np.float32))
    )
    return {
        "dimension": int(dimension),
        "accuracy": float(
            _row_cosine(reconstruction, values[validation], scale).mean()
        ),
        "explained_variance": float(compressor.explained_variance_ratio.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare strict GP priors with local spectral experts and PCA ceilings"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--output", default="reports/generated/v4_teacher_predictability.json"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    metadata = _load_npz(artifact_dir / "metadata.npz")
    targets = _load_npz(config["spectral"]["target_path"])
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    count = min(len(targets["outage"]), len(priors["available"]))
    fold = int(config["split"]["validation_fold"])
    validation_mask = metadata["validation_masks"][fold][:count].astype(bool)
    available = priors["available"][:count].astype(bool)
    nonzero = ~targets["outage"][:count].astype(bool)
    validation = np.flatnonzero(validation_mask & available & nonzero)
    observed_nonzero = np.flatnonzero(~validation_mask & available & nonzero)
    experts: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "strict_gp": (
            priors["pas_log"][validation].astype(np.float32),
            priors["pdp_log"][validation].astype(np.float32),
        )
    }
    distance = None
    for name, neighbors, power in (
        ("nearest", 1, 1.0),
        ("idw4_p1", 4, 1.0),
        ("idw4_p2", 4, 2.0),
        ("idw8_p1", 8, 1.0),
        ("idw8_p2", 8, 2.0),
    ):
        pas, pdp, current_distance = _neighbor_predictions(
            metadata,
            targets,
            validation,
            observed_nonzero,
            neighbors=neighbors,
            distance_power=power,
        )
        experts[name] = (pas, pdp)
        if distance is None:
            distance = current_distance

    expert_metrics = {
        name: _metrics(pas, pdp, targets, validation)
        for name, (pas, pdp) in experts.items()
    }
    names = list(experts)
    pas_stack = np.stack([experts[name][0] for name in names])
    pdp_stack = np.stack([experts[name][1] for name in names])
    pas_weights = convex_cosine_weights(
        pas_stack, targets["pas_log"][validation], PAS_LOG_SCALE, 0.05
    )
    pdp_weights = convex_cosine_weights(
        pdp_stack, targets["pdp_log"][validation], PDP_LOG_SCALE, 0.05
    )
    ensemble_pas = ensemble_log_power_predictions(
        [value for value in pas_stack], pas_weights, PAS_LOG_SCALE
    )
    ensemble_pdp = ensemble_log_power_predictions(
        [value for value in pdp_stack], pdp_weights, PDP_LOG_SCALE
    )
    ensemble = _metrics(ensemble_pas, ensemble_pdp, targets, validation)
    ensemble["pas_weights"] = {
        name: float(weight) for name, weight in zip(names, pas_weights, strict=True)
    }
    ensemble["pdp_weights"] = {
        name: float(weight) for name, weight in zip(names, pdp_weights, strict=True)
    }

    by_cell: dict[str, object] = {}
    compression: dict[str, object] = {}
    for cell in np.unique(metadata["train_cells"][:count]):
        selected = metadata["train_cells"][validation] == cell
        cell_indices = validation[selected]
        training = observed_nonzero[metadata["train_cells"][observed_nonzero] == cell]
        by_cell[str(int(cell))] = {
            name: _metrics(pas[selected], pdp[selected], targets, cell_indices)
            for name, (pas, pdp) in experts.items()
        }
        compression[str(int(cell))] = {
            "pas": [
                _compression_ceiling(
                    targets["pas_log"], training, cell_indices, dimension,
                    PAS_LOG_SCALE, int(config["seed"]) + int(cell) + dimension,
                )
                for dimension in (128, 256)
            ],
            "pdp": [
                _compression_ceiling(
                    targets["pdp_log"], training, cell_indices, dimension,
                    PDP_LOG_SCALE, int(config["seed"]) + 7 + int(cell) + dimension,
                )
                for dimension in (64, 128)
            ],
        }

    pas_per_expert = np.stack(
        [
            _row_cosine(experts[name][0], targets["pas_log"][validation], PAS_LOG_SCALE)
            for name in names
        ],
        axis=1,
    )
    pdp_per_expert = np.stack(
        [
            _row_cosine(experts[name][1], targets["pdp_log"][validation], PDP_LOG_SCALE)
            for name in names
        ],
        axis=1,
    )
    report = {
        "status": "PASS",
        "samples": int(len(validation)),
        "experts": expert_metrics,
        "global_convex_ensemble": ensemble,
        "per_sample_oracle_selector": {
            "pas_accuracy": float(pas_per_expert.max(axis=1).mean()),
            "pdp_accuracy": float(pdp_per_expert.max(axis=1).mean()),
        },
        "by_cell": by_cell,
        "compression_ceiling": compression,
        "nearest_distance_meters": {
            "median": float(np.median(distance)),
            "p90": float(np.percentile(distance, 90)),
            "maximum": float(np.max(distance)),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
