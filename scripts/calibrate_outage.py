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
    parser = argparse.ArgumentParser(description="Scan outage thresholds on the spatial validation set")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7")
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    model, shape = build_model(config)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    _, validation_loader = training_loaders(config, device)
    best = None
    for threshold in [float(value) for value in args.thresholds.split(",")]:
        metrics = evaluate_model(
            model,
            validation_loader,
            shape,
            config["scheme"],
            "joint",
            device,
            threshold,
        )
        print(
            f"threshold={threshold:.3f} score={metrics['score']:.6f} "
            f"outage_accuracy={metrics.get('outage_accuracy', float('nan')):.6f} "
            f"nmse={metrics['nmse']:.6f}"
        )
        if best is None or metrics["score"] > best[1]["score"]:
            best = (threshold, metrics)
    assert best is not None
    print(f"best_threshold={best[0]:.3f} best_score={best[1]['score']:.6f}")


if __name__ == "__main__":
    main()

