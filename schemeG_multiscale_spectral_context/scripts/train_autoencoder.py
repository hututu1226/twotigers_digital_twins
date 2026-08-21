from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from scheme_g.autoencoder_training import train_autoencoder
from scheme_g.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the structured angle-delay AE v2"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(train_autoencoder(load_config(args.config), args.resume), indent=2)
    )


if __name__ == "__main__":
    main()
