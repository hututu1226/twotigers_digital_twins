from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from spatial_inpainting.config import load_config
from spatial_inpainting.encoding import encode_training_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode channels into frozen AE latents")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    result = encode_training_set(load_config(args.config), args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

