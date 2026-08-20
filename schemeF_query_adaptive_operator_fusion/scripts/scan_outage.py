from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_f.config import choose_device, load_config, save_json
from scheme_f.context_data import ContextRepository
from scheme_f.context_training import (
    evaluate_context_thresholds,
    load_context_checkpoint,
    predict_indices,
)
from scheme_f.data import balanced_limit, load_metadata, split_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan conservative outage thresholds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 0.99, 0.999],
    )
    parser.add_argument(
        "--soft-strengths",
        nargs="*",
        type=float,
        default=[0.0, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--spectral-prior-alphas",
        nargs="*",
        type=float,
        default=[0.0, 0.15, 0.3],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(config["runtime"]["device"])
    metadata = load_metadata(config)
    training, validation = split_indices(metadata, config)
    if not len(validation):
        raise ValueError("Outage scanning requires a validation fold")
    with np.load(config["encoding"]["output_path"]) as source:
        available = (
            source["available"].astype(bool)
            if "available" in source.files
            else np.ones(len(metadata["train_cells"]), dtype=bool)
        )
    training = training[available[training]]
    validation = validation[available[validation]]
    training = balanced_limit(
        training,
        config["runtime"].get(
            "context_train_limit", config["runtime"].get("train_limit")
        ),
        [metadata["train_cells"]],
        int(config["seed"]) + 3,
    )
    validation = balanced_limit(
        validation,
        config["runtime"].get(
            "context_validation_limit", config["runtime"].get("validation_limit")
        ),
        [metadata["train_cells"]],
        int(config["seed"]) + 4,
    )
    repository = ContextRepository(config, training)
    model, autoencoder, shape, checkpoint = load_context_checkpoint(
        config, args.checkpoint, repository, device
    )
    amp = bool(config["runtime"].get("amp", True))
    predictions = predict_indices(model, repository, validation, device, amp)
    reports = []
    for prior_alpha in args.spectral_prior_alphas:
        for soft_strength in args.soft_strengths:
            metrics_by_threshold = evaluate_context_thresholds(
                model,
                autoencoder,
                repository,
                validation,
                shape,
                device,
                amp,
                args.thresholds,
                int(config["context"].get("validation_decode_batch_size", 8)),
                soft_outage_strength=soft_strength,
                spectral_prior_alpha=prior_alpha,
                prediction_outputs=predictions,
            )
            for threshold, metrics in zip(args.thresholds, metrics_by_threshold):
                reports.append(
                    {
                        "threshold": threshold,
                        "soft_strength": soft_strength,
                        "spectral_prior_alpha": prior_alpha,
                        **metrics,
                    }
                )
                print(
                    f"threshold={threshold:.6f} soft={soft_strength:.2f} "
                    f"prior={prior_alpha:.2f} score={metrics['score']:.6f}",
                    flush=True,
                )
    # Prefer the conservative (higher) threshold when validation scores tie.
    best = max(reports, key=lambda report: (report["score"], report["threshold"]))
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_threshold": float(best["threshold"]),
        "best_soft_strength": float(best["soft_strength"]),
        "best_spectral_prior_alpha": float(best["spectral_prior_alpha"]),
        "best_score": float(best["score"]),
        "reports": reports,
    }
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
