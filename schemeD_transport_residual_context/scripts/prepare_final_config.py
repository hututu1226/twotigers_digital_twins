from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Scheme D full-data epochs and outage threshold")
    parser.add_argument("--template", default="configs/final_5090.json")
    parser.add_argument("--context-checkpoint", default="artifacts/fold0/context/best.pt")
    parser.add_argument("--outage-report", default="artifacts/fold0/context/outage_scan.json")
    parser.add_argument("--output", default="configs/final_selected.json")
    parser.add_argument("--epoch-multiplier", type=float, default=1.1)
    args = parser.parse_args()
    if args.epoch_multiplier <= 0:
        raise ValueError("epoch-multiplier must be positive")
    checkpoint = torch.load(args.context_checkpoint, map_location="cpu", weights_only=False)
    fold_epochs = int(checkpoint["epoch"]) + 1
    final_epochs = max(1, round(fold_epochs * args.epoch_multiplier))
    config = _read(Path(args.template))
    config["context"]["epochs"] = final_epochs
    config["context"]["router_temperature_anneal_epochs"] = min(
        int(config["context"].get("router_temperature_anneal_epochs", final_epochs)),
        final_epochs,
    )
    threshold = float(_read(Path(args.outage_report))["best_threshold"])
    config["inference"]["outage_threshold"] = threshold
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output": str(destination.resolve()),
        "fold0_best_epoch": fold_epochs,
        "final_epochs": final_epochs,
        "outage_threshold": threshold,
        "autoencoder_policy": "reuse fixed Scheme C AE score 0.9491",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
