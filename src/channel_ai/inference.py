from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ChannelInferenceDataset
from .models import build_model
from .utils import autocast_context, choose_device


@torch.no_grad()
def run_inference(
    config: dict,
    checkpoint_path: str | Path,
    output_path: str | Path,
    device_name: str,
    limit: int | None = None,
) -> Path:
    device = choose_device(device_name)
    model, shape = build_model(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    dataset = ChannelInferenceDataset(
        config["data"]["root"], config["data"]["artifacts"], limit=limit
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["inference"].get("batch_size", config["data"]["batch_size"])),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.complex64, shape=(len(dataset), *shape.raw_shape)
    )
    offset = 0
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    for batch in loader:
        positions = batch["position"].to(device, non_blocking=True)
        map_tokens = batch["map_tokens"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            generated = model.generate(
                positions,
                map_tokens,
                float(config["training"].get("outage_threshold", 0.5)),
            )
        channel = generated["channel"].to(torch.complex64).cpu().numpy()
        target[offset : offset + len(channel)] = channel
        offset += len(channel)
    target.flush()
    print(f"Saved {target.shape} {target.dtype} to {output.resolve()}")
    return output

