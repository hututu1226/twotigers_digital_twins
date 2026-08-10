from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from channel_ai.config import load_config
from channel_ai.training import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scheme 1 or scheme 2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train_from_config(load_config(args.config), args.device, args.resume)


if __name__ == "__main__":
    main()

