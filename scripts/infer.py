from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from channel_ai.config import load_config
from channel_ai.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Round2_Test_Channel.npy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/Round2_Test_Channel.npy")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only; omit for submission")
    args = parser.parse_args()
    run_inference(
        load_config(args.config), args.checkpoint, args.output, args.device, args.limit
    )


if __name__ == "__main__":
    main()

