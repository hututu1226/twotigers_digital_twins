from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from torch.utils.data import DataLoader

from structured_context_field.autoencoder_training import (
    evaluate_autoencoder,
    load_autoencoder_checkpoint,
)
from structured_context_field.config import choose_device, load_config, save_json, worker_count
from structured_context_field.context_data import ContextRepository
from structured_context_field.context_training import (
    evaluate_context_model,
    load_context_checkpoint,
)
from structured_context_field.data import (
    ChannelDataset,
    balanced_limit,
    load_metadata,
    split_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Scheme B on its fixed spatial fold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("autoencoder", "context", "joint"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outage-threshold", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    training_indices, validation_indices = split_indices(metadata, config)
    if not len(validation_indices):
        raise ValueError("This config has no validation fold; evaluate with fold0_4090.json")
    if args.stage == "autoencoder":
        validation_indices = validation_indices[~metadata["outage"][validation_indices]]
        validation_indices = balanced_limit(
            validation_indices,
            config["runtime"].get("validation_limit"),
            [metadata["train_cells"]],
            int(config["seed"]) + 1,
        )
        dataset = ChannelDataset(
            Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
            validation_indices,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["autoencoder"].get("validation_batch_size", 8)),
            shuffle=False,
            num_workers=worker_count(int(config["runtime"].get("workers", -1))),
            pin_memory=device.type == "cuda",
        )
        model, shape, _ = load_autoencoder_checkpoint(config, args.checkpoint, device)
        result = evaluate_autoencoder(model, loader, shape, device, amp)
    else:
        with np.load(config["encoding"]["output_path"]) as source:
            available = (
                source["available"].astype(bool)
                if "available" in source.files
                else np.ones(len(metadata["train_cells"]), dtype=bool)
            )
        training_indices = training_indices[available[training_indices]]
        validation_indices = validation_indices[available[validation_indices]]
        training_indices = balanced_limit(
            training_indices,
            config["runtime"].get(
                "context_train_limit", config["runtime"].get("train_limit")
            ),
            [metadata["train_cells"]],
            int(config["seed"]) + 3,
        )
        validation_indices = balanced_limit(
            validation_indices,
            config["runtime"].get(
                "context_validation_limit", config["runtime"].get("validation_limit")
            ),
            [metadata["train_cells"]],
            int(config["seed"]) + 4,
        )
        repository = ContextRepository(config, training_indices)
        model, autoencoder, shape, _ = load_context_checkpoint(
            config, args.checkpoint, repository, device
        )
        section = config[args.stage]
        threshold = float(
            args.outage_threshold
            if args.outage_threshold is not None
            else section.get("outage_threshold", 0.999)
        )
        result = evaluate_context_model(
            model,
            autoencoder,
            repository,
            validation_indices,
            shape,
            device,
            amp,
            threshold,
            int(section.get("validation_decode_batch_size", 8)),
        )
    report = {
        "stage": args.stage,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "metrics": result,
    }
    if args.output:
        save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
