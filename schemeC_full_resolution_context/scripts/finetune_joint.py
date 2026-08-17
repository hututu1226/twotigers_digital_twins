from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from scheme_c.config import load_config
from scheme_c.context_training import train_context_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Jointly fine-tune context field and AE decoder")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = train_context_model(load_config(args.config), resume=args.resume, joint=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
