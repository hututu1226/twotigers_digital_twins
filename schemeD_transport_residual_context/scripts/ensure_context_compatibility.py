from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from scheme_d.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive an incompatible Context checkpoint before training V2"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    section = config["context"]
    output_dir = Path(section["output_dir"])
    checkpoint_path = output_dir / "last.pt"
    if not checkpoint_path.is_file():
        print("No previous Context checkpoint; compatibility check passed")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_section = checkpoint.get("config", {}).get("context", {})
    old_architecture = str(old_section.get("architecture", "full_resolution_context_v1"))
    current_architecture = str(section["architecture"])
    fields = (
        "architecture",
        "global_width",
        "router_width",
        "router_top_k",
        "pair_width",
        "spectrum_token_channels",
        "detail_token_channels",
        "refinement_blocks",
        "axial_blocks",
    )
    changed = [name for name in fields if old_section.get(name) != section.get(name)]
    if not changed:
        print(f"Existing Context checkpoint is compatible: {current_architecture}")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = old_architecture.replace("/", "_").replace("\\", "_")
    destination = output_dir.with_name(
        f"{output_dir.name}_archived_{safe_name}_{stamp}"
    )
    output_dir.rename(destination)
    print(
        f"Archived incompatible {old_architecture} Context to {destination}. "
        f"Changed fields: {', '.join(changed)}"
    )


if __name__ == "__main__":
    main()
