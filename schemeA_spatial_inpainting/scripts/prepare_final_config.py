from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an all-data config from fold0 best epochs and outage scan"
    )
    parser.add_argument("--template", default="configs/final_4090.json")
    parser.add_argument("--ae-checkpoint", default="artifacts/fold0/autoencoder/best.pt")
    parser.add_argument("--spatial-checkpoint", default="artifacts/fold0/spatial/best.pt")
    parser.add_argument("--outage-report", default="artifacts/fold0/spatial/outage_scan.json")
    parser.add_argument("--output", default="configs/final_selected.json")
    parser.add_argument("--epoch-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    if args.epoch_multiplier <= 0:
        raise ValueError("epoch-multiplier must be positive")
    template_path = Path(args.template)
    config = load_json(template_path)
    ae = torch.load(args.ae_checkpoint, map_location="cpu", weights_only=False)
    spatial = torch.load(args.spatial_checkpoint, map_location="cpu", weights_only=False)
    outage_report = load_json(Path(args.outage_report))
    ae_epochs = max(1, round((int(ae["epoch"]) + 1) * args.epoch_multiplier))
    spatial_epochs = max(1, round((int(spatial["epoch"]) + 1) * args.epoch_multiplier))
    threshold = float(outage_report["best_threshold"])
    config["autoencoder"]["epochs"] = ae_epochs
    config["spatial"]["epochs"] = spatial_epochs
    config["inference"]["outage_threshold"] = threshold
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "autoencoder_epochs": ae_epochs,
                "spatial_epochs": spatial_epochs,
                "outage_threshold": threshold,
                "split": config["split"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
