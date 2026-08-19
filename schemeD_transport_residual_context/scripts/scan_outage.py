from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_d.config import choose_device, load_config, save_json
from scheme_d.context_data import ContextRepository
from scheme_d.context_training import (
    evaluate_context_thresholds,
    load_context_checkpoint,
)
from scheme_d.data import balanced_limit, load_metadata, split_indices


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
    metrics_by_threshold = evaluate_context_thresholds(
        model,
        autoencoder,
        repository,
        validation,
        shape,
        device,
        bool(config["runtime"].get("amp", True)),
        args.thresholds,
        int(config["context"].get("validation_decode_batch_size", 8)),
    )
    reports = []
    for threshold, metrics in zip(args.thresholds, metrics_by_threshold):
        reports.append({"threshold": threshold, **metrics})
        print(f"threshold={threshold:.6f} score={metrics['score']:.6f}", flush=True)
    # Prefer the conservative (higher) threshold when validation scores tie.
    best = max(reports, key=lambda report: (report["score"], report["threshold"]))
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_threshold": float(best["threshold"]),
        "best_score": float(best["score"]),
        "reports": reports,
    }
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
