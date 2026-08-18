from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trained_epochs(path: str, multiplier: float) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return max(1, round((int(checkpoint["epoch"]) + 1) * multiplier))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer Fold0-selected epochs and outage threshold to all-data config"
    )
    parser.add_argument("--template", default="configs/final_5090.json")
    parser.add_argument("--ae-checkpoint", default="artifacts/fold0/autoencoder/best.pt")
    parser.add_argument("--context-checkpoint", default="artifacts/fold0/context/best.pt")
    parser.add_argument("--outage-report", default="artifacts/fold0/context/outage_scan.json")
    parser.add_argument(
        "--ae-gate", default="artifacts/fold0/autoencoder/quality_gate.json"
    )
    parser.add_argument("--output", default="configs/final_selected.json")
    parser.add_argument("--epoch-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    if args.epoch_multiplier <= 0:
        raise ValueError("epoch-multiplier must be positive")
    gate = load_json(Path(args.ae_gate))
    if gate.get("status") != "PASS" or not gate.get("context_training_allowed"):
        raise RuntimeError(
            "Fold0 AE quality gate did not pass; refusing to prepare all-data training"
        )
    config = load_json(Path(args.template))
    selected_epochs = {
        "autoencoder": trained_epochs(args.ae_checkpoint, args.epoch_multiplier),
        "context": trained_epochs(args.context_checkpoint, args.epoch_multiplier),
    }
    for stage, epochs in selected_epochs.items():
        config[stage]["epochs"] = epochs
    report_path = Path(args.outage_report)
    threshold = float(config["inference"].get("outage_threshold", 0.999))
    if report_path.exists():
        threshold = float(load_json(report_path)["best_threshold"])
        config["inference"]["outage_threshold"] = threshold
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "selected_epochs": selected_epochs,
                "outage_threshold": threshold,
                "autoencoder_quality_gate": gate["status"],
                "split": config["split"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
