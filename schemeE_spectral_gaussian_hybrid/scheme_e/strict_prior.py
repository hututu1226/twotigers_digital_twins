from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from .config import choose_device, save_json, seed_everything
from .gp import SharedMultiOutputGP, ensemble_log_power_predictions, ensemble_predictions
from .local_spectral import local_expert_settings, local_spectral_prediction
from .outage import OutageEnsemble, binary_metrics
from .power_safety import apply_power_calibration
from .spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE
from .spectral_teacher import (
    _cosine_loss,
    _decode_prediction,
    _kernel_settings,
    _load_arrays,
    _make_compressors,
    _target_matrix,
    train_oof_teacher,
)


def build_strict_fold_prior(
    config: dict,
    fold: int,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(int(config["seed"]) + 1009 * int(fold))
    device = choose_device(str(config["runtime"].get("device", "auto")))
    metadata, targets = _load_arrays(config)
    validation_mask = metadata["validation_masks"][int(fold)].astype(bool)
    training_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    work_dir = Path(output_path).parent / "strict_visible_oof" / f"fold{int(fold)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = work_dir / "metadata.npz"
    target_path = work_dir / "channel_targets.npz"
    np.savez_compressed(
        metadata_path,
        **{
            name: metadata[name][training_indices]
            for name in (
                "train_positions",
                "train_cells",
                "spectral_folds",
                "train_geometry_features",
            )
        },
    )
    np.savez_compressed(
        target_path,
        **{
            name: value[training_indices]
            if value.ndim and len(value) == len(validation_mask)
            else value
            for name, value in targets.items()
        },
    )
    visible_config = deepcopy(config)
    visible_config["preprocessing"]["artifact_dir"] = str(work_dir)
    visible_config["spectral"]["target_path"] = str(target_path)
    visible_config["runtime"]["spectral_train_limit"] = 0
    visible_config["spectral_teacher"]["oof_output_path"] = str(
        work_dir / "oof_priors.npz"
    )
    visible_config["spectral_teacher"]["oof_report_path"] = str(
        work_dir / "oof_report.json"
    )
    visible_report = train_oof_teacher(visible_config)
    with np.load(visible_config["spectral_teacher"]["oof_output_path"]) as source:
        visible_prior = {name: np.array(source[name], copy=True) for name in source.files}
    if not bool(np.asarray(visible_prior["available"]).all()):
        raise RuntimeError("Visible-training OOF prior is incomplete")

    count = len(validation_mask)
    pas = np.zeros((count, targets["pas_log"].shape[1]), dtype=np.float32)
    pdp = np.zeros((count, targets["pdp_log"].shape[1]), dtype=np.float32)
    ue = np.zeros((count, targets["ue_log_energy"].shape[1]), dtype=np.float32)
    power = np.zeros(count, dtype=np.float32)
    uncertainty = np.zeros(count, dtype=np.float32)
    outage_probability = np.zeros(count, dtype=np.float32)
    available = np.zeros(count, dtype=np.bool_)
    for name, destination in (
        ("pas_log", pas),
        ("pdp_log", pdp),
        ("ue_log_energy", ue),
        ("log_power", power),
        ("uncertainty", uncertainty),
        ("outage_probability", outage_probability),
    ):
        destination[training_indices] = visible_prior[name]
    available[training_indices] = True

    section = config["spectral_teacher"]
    kernel_settings = _kernel_settings(section)
    local_settings = local_expert_settings(section)
    pas_dim = int(section.get("pas_latent_dim", 96))
    pdp_dim = int(section.get("pdp_latent_dim", 48))
    ue_dim = int(targets["ue_log_energy"].shape[1])
    records: list[dict[str, object]] = []
    for cell in sorted(np.unique(metadata["train_cells"]).tolist()):
        cell_training = training_indices[
            metadata["train_cells"][training_indices] == cell
        ]
        spectral_training = cell_training[
            ~targets["outage"][cell_training].astype(bool)
        ]
        cell_validation = validation_indices[
            metadata["train_cells"][validation_indices] == cell
        ]
        pas_compressor, pdp_compressor = _make_compressors(config)
        pas_compressor.fit(
            targets["pas_log"][spectral_training].astype(np.float32),
            int(config["seed"]) + int(cell),
        )
        pdp_compressor.fit(
            targets["pdp_log"][spectral_training].astype(np.float32),
            int(config["seed"]) + 17 + int(cell),
        )
        target_matrix = _target_matrix(
            targets, spectral_training, pas_compressor, pdp_compressor
        )
        predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for kernel_name, feature_mix in kernel_settings:
            model = SharedMultiOutputGP(
                kernel_name,
                noise=float(section.get("gp_noise", 0.01)),
                feature_length=float(section.get("feature_length", 1.0)),
                feature_mix=float(feature_mix),
            ).fit(
                metadata["train_positions"][spectral_training],
                metadata["train_geometry_features"][spectral_training],
                target_matrix,
                device,
            )
            raw, model_uncertainty = model.predict(
                metadata["train_positions"][cell_validation],
                metadata["train_geometry_features"][cell_validation],
                device,
                int(section.get("prediction_batch_size", 128)),
            )
            decoded = _decode_prediction(
                raw, pas_compressor, pdp_compressor, pas_dim, pdp_dim, ue_dim
            )
            predictions.append((*decoded, model_uncertainty))
        for _, neighbors, distance_power in local_settings:
            predictions.append(
                local_spectral_prediction(
                    metadata["train_positions"][spectral_training],
                    metadata["train_positions"][cell_validation],
                    targets["pas_log"][spectral_training],
                    targets["pdp_log"][spectral_training],
                    targets["ue_log_energy"][spectral_training],
                    targets["log_power"][spectral_training],
                    neighbors=neighbors,
                    distance_power=distance_power,
                )
            )

        pas_weights = visible_prior["pas_weights"][int(cell)].astype(np.float32)
        pdp_weights = visible_prior["pdp_weights"][int(cell)].astype(np.float32)
        auxiliary_weights = visible_prior["auxiliary_weights"][int(cell)].astype(
            np.float32
        )
        pas[cell_validation] = ensemble_log_power_predictions(
            [value[0] for value in predictions], pas_weights, PAS_LOG_SCALE
        )
        pdp[cell_validation] = ensemble_log_power_predictions(
            [value[1] for value in predictions], pdp_weights, PDP_LOG_SCALE
        )
        ue[cell_validation] = ensemble_predictions(
            [value[2] for value in predictions], auxiliary_weights
        )
        power[cell_validation] = ensemble_predictions(
            [value[3][:, None] for value in predictions], auxiliary_weights
        )[:, 0]
        uncertainty[cell_validation] = ensemble_predictions(
            [value[4][:, None] for value in predictions], auxiliary_weights
        )[:, 0]
        classifier = OutageEnsemble(
            seed=int(config["seed"]) + int(cell),
            positive_weight=float(section.get("outage_positive_weight", 4.0)),
            false_kill_cost=float(section.get("false_kill_cost", 0.56)),
        ).fit(
            metadata["train_geometry_features"][cell_training],
            targets["outage"][cell_training].astype(np.int64),
        )
        outage_probability[cell_validation] = classifier.predict_proba(
            metadata["train_geometry_features"][cell_validation]
        )
        if "power_calibration" in visible_prior:
            calibrated_power, calibrated_ue = apply_power_calibration(
                power[cell_validation],
                ue[cell_validation],
                metadata["train_cells"][cell_validation],
                visible_prior["power_calibration"],
            )
            power[cell_validation] = calibrated_power
            ue[cell_validation] = calibrated_ue
        available[cell_validation] = True
        nonzero = cell_validation[~targets["outage"][cell_validation].astype(bool)]
        records.append(
            {
                "cell": int(cell),
                "training": int(len(cell_training)),
                "validation": int(len(cell_validation)),
                "pas_accuracy": float(
                    1.0
                    - _cosine_loss(
                        pas[nonzero], targets["pas_log"][nonzero], PAS_LOG_SCALE
                    ).mean()
                ),
                "pdp_accuracy": float(
                    1.0
                    - _cosine_loss(
                        pdp[nonzero], targets["pdp_log"][nonzero], PDP_LOG_SCALE
                    ).mean()
                ),
            }
        )

    if not bool(available.all()):
        raise RuntimeError("Strict Fold prior did not cover every training sample")
    thresholds_by_cell = np.asarray(
        visible_prior.get(
            "outage_threshold_by_cell",
            np.full(
                len(visible_prior["pas_weights"]),
                float(np.asarray(visible_prior["outage_threshold"]).reshape(-1)[0]),
                dtype=np.float32,
            ),
        ),
        dtype=np.float32,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        pas_log=pas.astype(np.float16),
        pdp_log=pdp.astype(np.float16),
        ue_log_energy=ue,
        log_power=power,
        outage_probability=outage_probability,
        uncertainty=uncertainty,
        available=available,
        pas_weights=visible_prior["pas_weights"],
        pdp_weights=visible_prior["pdp_weights"],
        auxiliary_weights=visible_prior["auxiliary_weights"],
        outage_threshold=visible_prior["outage_threshold"],
        outage_threshold_by_cell=thresholds_by_cell,
        power_calibration=visible_prior.get(
            "power_calibration",
            np.column_stack(
                [
                    np.zeros(len(thresholds_by_cell), dtype=np.float32),
                    np.zeros(len(thresholds_by_cell), dtype=np.float32),
                    np.ones(len(thresholds_by_cell), dtype=np.float32),
                ]
            ),
        ),
    )
    nonzero_validation = validation_indices[
        ~targets["outage"][validation_indices].astype(bool)
    ]
    threshold = float(np.asarray(visible_prior["outage_threshold"]).reshape(-1)[0])
    report = {
        "stage": "scheme_e_v2_strict_fold_prior",
        "fold": int(fold),
        "leakage_free_validation": True,
        "experts": [name for name, _ in kernel_settings]
        + [name for name, _, _ in local_settings],
        "training_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "training_oof_pas_accuracy": float(visible_report["pas_accuracy"]),
        "training_oof_pdp_accuracy": float(visible_report["pdp_accuracy"]),
        "pas_accuracy": float(
            1.0
            - _cosine_loss(
                pas[nonzero_validation],
                targets["pas_log"][nonzero_validation],
                PAS_LOG_SCALE,
            ).mean()
        ),
        "pdp_accuracy": float(
            1.0
            - _cosine_loss(
                pdp[nonzero_validation],
                targets["pdp_log"][nonzero_validation],
                PDP_LOG_SCALE,
            ).mean()
        ),
        "power_mae_log10": float(
            np.mean(
                np.abs(
                    power[nonzero_validation]
                    - targets["log_power"][nonzero_validation]
                )
            )
        ),
        "power_calibration": visible_prior.get(
            "power_calibration", np.empty((0, 3), dtype=np.float32)
        ).tolist(),
        "outage": binary_metrics(
            outage_probability[validation_indices],
            targets["outage"][validation_indices],
            threshold,
        ),
        "cells": records,
        "output_path": str(destination),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(report_path, report)
    return report
