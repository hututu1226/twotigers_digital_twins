from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np


REPO_ROOT = _bootstrap.PROJECT_ROOT.parent
SCHEME_E_ROOT = REPO_ROOT / "schemeE_spectral_gaussian_hybrid"
if str(SCHEME_E_ROOT) not in sys.path:
    sys.path.insert(0, str(SCHEME_E_ROOT))

from scheme_e.config import choose_device, load_config, save_json, seed_everything  # noqa: E402
from scheme_e.gp import (  # noqa: E402
    SharedMultiOutputGP,
    ensemble_log_power_predictions,
    ensemble_predictions,
)
from scheme_e.outage import OutageEnsemble, binary_metrics  # noqa: E402
from scheme_e.spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE  # noqa: E402
from scheme_e.spectral_teacher import (  # noqa: E402
    _cosine_loss,
    _decode_prediction,
    _load_arrays,
    _make_compressors,
    _target_matrix,
    train_oof_teacher,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Fold0 prior without using any Fold0 validation labels"
    )
    parser.add_argument(
        "--scheme-e-config",
        default=str(SCHEME_E_ROOT / "configs" / "fold0_5090.json"),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(
            _bootstrap.PROJECT_ROOT
            / "artifacts"
            / "fold0"
            / "spectral_teacher"
            / "strict_priors.npz"
        ),
    )
    parser.add_argument(
        "--report",
        default=str(
            _bootstrap.PROJECT_ROOT
            / "artifacts"
            / "fold0"
            / "spectral_teacher"
            / "strict_prior_report.json"
        ),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.scheme_e_config)
    seed_everything(int(config["seed"]) + 1009 * int(args.fold))
    device = choose_device(str(config["runtime"].get("device", "auto")))
    metadata, targets = _load_arrays(config)
    validation_mask = metadata["validation_masks"][int(args.fold)].astype(bool)
    if len(validation_mask) != len(targets["outage"]):
        raise ValueError("validation mask and spectral targets have different lengths")
    training_mask = ~validation_mask
    training_indices = np.flatnonzero(training_mask)
    signature = hashlib.sha256(training_indices.tobytes()).hexdigest()[:12]
    work_dir = (
        Path(args.output).parent
        / "strict_visible_oof"
        / f"fold{int(args.fold)}_{signature}"
    )
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
    strict_config = deepcopy(config)
    strict_config["preprocessing"]["artifact_dir"] = str(work_dir)
    strict_config["spectral"]["target_path"] = str(target_path)
    strict_config["runtime"]["spectral_train_limit"] = 0
    strict_config["spectral_teacher"]["oof_output_path"] = str(
        work_dir / "oof_priors.npz"
    )
    strict_config["spectral_teacher"]["oof_report_path"] = str(
        work_dir / "oof_report.json"
    )
    strict_oof_report = train_oof_teacher(strict_config)
    with np.load(strict_config["spectral_teacher"]["oof_output_path"]) as source:
        prior = {name: np.array(source[name], copy=True) for name in source.files}
    if not bool(np.asarray(prior["available"]).all()):
        raise RuntimeError("Strict visible-training OOF prior is incomplete")

    section = config["spectral_teacher"]
    kernels = tuple(section.get("kernels", ["rq10", "rq20", "matern20"]))
    pas_dim = int(section.get("pas_latent_dim", 96))
    pdp_dim = int(section.get("pdp_latent_dim", 48))
    ue_dim = int(targets["ue_log_energy"].shape[1])
    records: list[dict[str, object]] = []
    count = len(validation_mask)
    strict_pas = np.zeros((count, targets["pas_log"].shape[1]), dtype=np.float32)
    strict_pdp = np.zeros((count, targets["pdp_log"].shape[1]), dtype=np.float32)
    strict_ue = np.zeros((count, targets["ue_log_energy"].shape[1]), dtype=np.float32)
    strict_power = np.zeros(count, dtype=np.float32)
    strict_uncertainty = np.zeros(count, dtype=np.float32)
    strict_outage = np.zeros(count, dtype=np.float32)
    available = np.zeros(count, dtype=bool)
    strict_pas[training_indices] = np.asarray(prior["pas_log"], dtype=np.float32)
    strict_pdp[training_indices] = np.asarray(prior["pdp_log"], dtype=np.float32)
    strict_ue[training_indices] = np.asarray(prior["ue_log_energy"], dtype=np.float32)
    strict_power[training_indices] = np.asarray(prior["log_power"], dtype=np.float32)
    strict_uncertainty[training_indices] = np.asarray(
        prior["uncertainty"], dtype=np.float32
    )
    strict_outage[training_indices] = np.asarray(
        prior["outage_probability"], dtype=np.float32
    )
    available[training_indices] = True

    for cell_id in sorted(np.unique(metadata["train_cells"]).tolist()):
        cell = metadata["train_cells"] == int(cell_id)
        train_indices = np.flatnonzero(training_mask & cell)
        spectral_training = train_indices[
            ~targets["outage"][train_indices].astype(bool)
        ]
        validation_indices = np.flatnonzero(validation_mask & cell)
        if len(spectral_training) < 2 or not len(validation_indices):
            raise RuntimeError(f"Fold {args.fold} cell {cell_id} has insufficient data")

        pas_compressor, pdp_compressor = _make_compressors(config)
        pas_compressor.fit(
            targets["pas_log"][spectral_training].astype(np.float32),
            int(config["seed"]) + int(cell_id),
        )
        pdp_compressor.fit(
            targets["pdp_log"][spectral_training].astype(np.float32),
            int(config["seed"]) + 17 + int(cell_id),
        )
        target_matrix = _target_matrix(
            targets, spectral_training, pas_compressor, pdp_compressor
        )
        kernel_pas: list[np.ndarray] = []
        kernel_pdp: list[np.ndarray] = []
        kernel_ue: list[np.ndarray] = []
        kernel_power: list[np.ndarray] = []
        kernel_uncertainty: list[np.ndarray] = []
        for kernel_name in kernels:
            model = SharedMultiOutputGP(
                str(kernel_name),
                noise=float(section.get("gp_noise", 0.01)),
                feature_length=float(section.get("feature_length", 1.0)),
            ).fit(
                metadata["train_positions"][spectral_training],
                metadata["train_geometry_features"][spectral_training],
                target_matrix,
                device,
            )
            prediction, uncertainty = model.predict(
                metadata["train_positions"][validation_indices],
                metadata["train_geometry_features"][validation_indices],
                device,
                int(section.get("prediction_batch_size", 128)),
            )
            pas, pdp, ue, power = _decode_prediction(
                prediction,
                pas_compressor,
                pdp_compressor,
                pas_dim,
                pdp_dim,
                ue_dim,
            )
            kernel_pas.append(pas)
            kernel_pdp.append(pdp)
            kernel_ue.append(ue)
            kernel_power.append(power[:, None])
            kernel_uncertainty.append(uncertainty[:, None])

        pas_weights = np.asarray(prior["pas_weights"][int(cell_id)], dtype=np.float32)
        pdp_weights = np.asarray(prior["pdp_weights"][int(cell_id)], dtype=np.float32)
        auxiliary_weights = np.asarray(
            prior["auxiliary_weights"][int(cell_id)], dtype=np.float32
        )
        strict_pas[validation_indices] = ensemble_log_power_predictions(
            kernel_pas, pas_weights, PAS_LOG_SCALE
        )
        strict_pdp[validation_indices] = ensemble_log_power_predictions(
            kernel_pdp, pdp_weights, PDP_LOG_SCALE
        )
        strict_ue[validation_indices] = ensemble_predictions(
            kernel_ue, auxiliary_weights
        )
        strict_power[validation_indices] = ensemble_predictions(
            kernel_power, auxiliary_weights
        )[:, 0]
        strict_uncertainty[validation_indices] = ensemble_predictions(
            kernel_uncertainty, auxiliary_weights
        )[:, 0]

        labels = targets["outage"][train_indices].astype(np.int64)
        classifier = OutageEnsemble(
            seed=int(config["seed"]) + int(cell_id),
            positive_weight=float(section.get("outage_positive_weight", 4.0)),
            false_kill_cost=float(section.get("false_kill_cost", 0.56)),
        ).fit(metadata["train_geometry_features"][train_indices], labels)
        strict_outage[validation_indices] = classifier.predict_proba(
            metadata["train_geometry_features"][validation_indices]
        )
        available[validation_indices] = True
        nonzero = validation_indices[
            ~targets["outage"][validation_indices].astype(bool)
        ]
        records.append(
            {
                "cell": int(cell_id),
                "training": int(len(train_indices)),
                "spectral_training": int(len(spectral_training)),
                "validation": int(len(validation_indices)),
                "pas_accuracy": float(
                    1.0
                    - _cosine_loss(
                        strict_pas[nonzero],
                        targets["pas_log"][nonzero],
                        PAS_LOG_SCALE,
                    ).mean()
                ),
                "pdp_accuracy": float(
                    1.0
                    - _cosine_loss(
                        strict_pdp[nonzero],
                        targets["pdp_log"][nonzero],
                        PDP_LOG_SCALE,
                    ).mean()
                ),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pas_log=strict_pas.astype(np.float16),
        pdp_log=strict_pdp.astype(np.float16),
        ue_log_energy=strict_ue,
        log_power=strict_power,
        outage_probability=strict_outage,
        uncertainty=strict_uncertainty,
        available=available,
        pas_weights=prior["pas_weights"],
        pdp_weights=prior["pdp_weights"],
        auxiliary_weights=prior["auxiliary_weights"],
        outage_threshold=prior["outage_threshold"],
    )
    validation = np.flatnonzero(validation_mask)
    nonzero = validation[~targets["outage"][validation].astype(bool)]
    threshold = float(np.asarray(prior["outage_threshold"]).reshape(-1)[0])
    report = {
        "stage": "strict_fold_spectral_prior",
        "fold": int(args.fold),
        "leakage_free_validation": True,
        "training_features_exclude_validation_labels": True,
        "training_samples": int(training_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "training_oof_pas_accuracy": float(strict_oof_report["pas_accuracy"]),
        "training_oof_pdp_accuracy": float(strict_oof_report["pdp_accuracy"]),
        "pas_accuracy": float(
            1.0
            - _cosine_loss(
                strict_pas[nonzero], targets["pas_log"][nonzero], PAS_LOG_SCALE
            ).mean()
        ),
        "pdp_accuracy": float(
            1.0
            - _cosine_loss(
                strict_pdp[nonzero], targets["pdp_log"][nonzero], PDP_LOG_SCALE
            ).mean()
        ),
        "power_mae_log10": float(
            np.mean(np.abs(strict_power[nonzero] - targets["log_power"][nonzero]))
        ),
        "outage": binary_metrics(
            strict_outage[validation], targets["outage"][validation], threshold
        ),
        "cells": records,
        "output_path": str(output_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
