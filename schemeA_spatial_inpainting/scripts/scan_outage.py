from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from spatial_inpainting.config import choose_device, load_config, save_json
from spatial_inpainting.data import SpatialRepository, load_metadata, split_indices
from spatial_inpainting.spatial_training import (
    load_spatial_checkpoint,
    scan_outage_thresholds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan outage thresholds on the fixed spatial fold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--output")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if not thresholds or any(value <= 0.0 or value >= 1.0 for value in thresholds):
        raise ValueError("Every threshold must be in the open interval (0, 1)")
    config = load_config(args.config)
    device = choose_device(config["runtime"]["device"])
    metadata = load_metadata(config)
    training_indices, validation_indices = split_indices(metadata, config)
    if not len(validation_indices):
        raise ValueError("Threshold scanning requires a config with a validation fold")
    repository = SpatialRepository(config, training_indices)
    available = repository.encoded.get("available")
    if available is not None:
        validation_indices = validation_indices[available[validation_indices]]
    model, autoencoder, shape, checkpoint = load_spatial_checkpoint(
        config, args.checkpoint, repository, device
    )
    results = scan_outage_thresholds(
        model,
        autoencoder,
        repository,
        validation_indices,
        shape,
        device,
        bool(config["runtime"].get("amp", True)),
        thresholds,
        int(config["spatial"].get("validation_decode_batch_size", 8)),
    )
    best = max(results, key=lambda item: item["score"])
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_threshold": best["threshold"],
        "best_score": best["score"],
        "results": results,
    }
    if args.output:
        save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
