from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from scheme_f.config import load_config
from scheme_f.preprocessing import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Scheme F dual-resolution artifacts"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = preprocess_dataset(load_config(args.config), force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
