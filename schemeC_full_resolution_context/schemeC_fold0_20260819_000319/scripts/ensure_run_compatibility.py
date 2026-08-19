from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from scheme_c.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive an old run before loading an incompatible AE architecture"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", choices=("fold0", "final"), required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    current_architecture = str(
        config["autoencoder"].get("architecture", "structured_v2")
    )
    run_root = Path(config["autoencoder"]["output_dir"]).parent
    if run_root.name != args.run:
        raise ValueError(
            f"Config AE output belongs to {run_root.name}, not requested run {args.run}"
        )
    checkpoint_path = run_root / "autoencoder" / "last.pt"
    if not checkpoint_path.is_file():
        print(f"No previous {args.run} AE checkpoint; compatibility check passed")
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    old_section = old_config.get("autoencoder", {})
    current_section = config["autoencoder"]
    old_architecture = str(old_section.get("architecture", "structured_v2"))
    architecture_fields = (
        "architecture",
        "spectrum_stem_channels",
        "phase_stem_channels",
        "spectrum_latent_channels",
        "phase_latent_channels",
        "residual_blocks",
        "detail_hidden_channels",
        "spectrum_decoder_channels",
        "detail_decoder_channels",
    )
    changed_fields = [
        name
        for name in architecture_fields
        if old_section.get(name) != current_section.get(name)
    ]
    if not changed_fields:
        print(
            f"Existing {args.run} checkpoint architecture is compatible: "
            f"{current_architecture}"
        )
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_old_name = old_architecture.replace("/", "_").replace("\\", "_")
    destination = run_root.with_name(
        f"{run_root.name}_archived_{safe_old_name}_{stamp}"
    )
    run_root.rename(destination)
    print(
        f"Archived incompatible {old_architecture} artifacts to {destination}; "
        f"starting {current_architecture} from scratch. Changed fields: "
        f"{', '.join(changed_fields)}"
    )


if __name__ == "__main__":
    main()
