from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from structured_context_field.config import load_config
from structured_context_field.encoding import encode_training_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode all training channels with AE v2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    result = encode_training_set(load_config(args.config), args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
