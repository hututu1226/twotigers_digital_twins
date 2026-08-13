from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from torch.utils.data import DataLoader

from spatial_inpainting.autoencoder_training import evaluate_autoencoder, load_autoencoder_checkpoint
from spatial_inpainting.config import choose_device, load_config, worker_count
from spatial_inpainting.data import (
    ChannelDataset,
    SpatialRepository,
    load_metadata,
    split_indices,
)
from spatial_inpainting.spatial_training import evaluate_spatial_model, load_spatial_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Scheme A checkpoint on the fixed spatial fold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("autoencoder", "spatial"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outage-threshold", type=float)
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    training_indices, validation_indices = split_indices(metadata, config)
    if not len(validation_indices):
        raise ValueError("This config has no validation fold; use fold0_4090.json for evaluation")
    if args.stage == "autoencoder":
        validation_indices = validation_indices[~metadata["outage"][validation_indices]]
        dataset = ChannelDataset(
            f"{config['data']['root']}/Round2_Train_Channel.npy", validation_indices
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
        repository = SpatialRepository(config, training_indices)
        available = repository.encoded.get("available")
        if available is not None:
            validation_indices = validation_indices[available[validation_indices]]
        model, autoencoder, shape, _ = load_spatial_checkpoint(
            config, args.checkpoint, repository, device
        )
        threshold = (
            args.outage_threshold
            if args.outage_threshold is not None
            else float(config["inference"].get("outage_threshold", 0.5))
        )
        result = evaluate_spatial_model(
            model,
            autoencoder,
            repository,
            validation_indices,
            shape,
            device,
            amp,
            threshold,
            int(config["spatial"].get("validation_decode_batch_size", 8)),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

