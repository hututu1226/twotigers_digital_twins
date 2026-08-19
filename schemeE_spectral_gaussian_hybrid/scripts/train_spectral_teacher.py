from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from scheme_e.config import load_config
from scheme_e.spectral_teacher import train_final_teacher, train_oof_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Scheme E spatial OOF or final spectral teacher")
    parser.add_argument("--config", default=str(_bootstrap.PROJECT_ROOT / "configs" / "fold0_5090.json"))
    parser.add_argument("--mode", choices=("oof", "final"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["runtime"]["device"] = args.device
    result = train_oof_teacher(config) if args.mode == "oof" else train_final_teacher(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
