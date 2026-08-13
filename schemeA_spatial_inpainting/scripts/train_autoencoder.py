from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from spatial_inpainting.autoencoder_training import train_autoencoder
from spatial_inpainting.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the angle-delay autoencoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train_autoencoder(load_config(args.config), resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

