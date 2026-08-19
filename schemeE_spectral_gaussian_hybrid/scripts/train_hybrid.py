from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from scheme_e.config import load_config
from scheme_e.hybrid_training import train_hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Scheme E full-resolution AE hybrid")
    parser.add_argument("--config", default=str(_bootstrap.PROJECT_ROOT / "configs" / "fold0_5090.json"))
    parser.add_argument("--stage", choices=("fold0", "final"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["runtime"]["device"] = args.device
        config["runtime"]["amp"] = args.device != "cpu"
    if args.resume:
        section = "hybrid_final" if args.stage == "final" else "hybrid"
        config[section]["resume"] = True
    result = train_hybrid(config, final=args.stage == "final")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
