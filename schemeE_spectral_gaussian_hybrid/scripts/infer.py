from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from scheme_e.config import load_config
from scheme_e.hybrid_inference import generate_test_channels


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Scheme E Round2 test channels")
    parser.add_argument("--config", default=str(_bootstrap.PROJECT_ROOT / "configs" / "final_selected.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["runtime"]["device"] = args.device
        config["runtime"]["amp"] = args.device != "cpu"
    print(json.dumps(generate_test_channels(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
