from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from torch.utils.data import DataLoader

from scheme_c.autoencoder_training import (
    evaluate_autoencoder_ablations,
    load_autoencoder_checkpoint,
)
from scheme_c.config import choose_device, load_config, save_json, worker_count
from scheme_c.data import (
    ChannelDataset,
    balanced_limit,
    load_metadata,
    split_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether the AE really uses its detail latent"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    _, validation_indices = split_indices(metadata, config)
    if not len(validation_indices):
        raise ValueError("AE ablation requires a validation fold")
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
    model, shape, checkpoint = load_autoencoder_checkpoint(
        config, args.checkpoint, device
    )
    result = evaluate_autoencoder_ablations(model, loader, shape, device, amp)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        **result,
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
