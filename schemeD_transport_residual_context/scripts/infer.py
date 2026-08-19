from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from scheme_d.config import load_config
from scheme_d.inference import generate_test_channels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Round2 test channels with Scheme D")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--outage-threshold", type=float)
    args = parser.parse_args()
    result = generate_test_channels(
        load_config(args.config),
        args.checkpoint,
        args.output,
        args.outage_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
