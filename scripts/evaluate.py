from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from channel_ai.config import load_config
from channel_ai.data import training_loaders
from channel_ai.models import build_model
from channel_ai.training import evaluate_model
from channel_ai.utils import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the spatial validation split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stage", default="joint")
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    model, shape = build_model(config)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    _, loader = training_loaders(config, device)
    metrics = evaluate_model(
        model,
        loader,
        shape,
        config["scheme"],
        args.stage,
        device,
        float(config["training"].get("outage_threshold", 0.5)),
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

