from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from spatial_inpainting.config import load_config
from spatial_inpainting.preprocessing import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Scheme A spatial grids and metadata")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = preprocess_dataset(load_config(args.config), force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

