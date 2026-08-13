from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from spatial_inpainting.config import load_config
from spatial_inpainting.spatial_training import train_spatial_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dynamic-hole spatial U-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train_spatial_model(load_config(args.config), resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

