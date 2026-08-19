from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from scheme_e.config import load_config
from scheme_e.preprocessing import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess positions, mesh, RF Gaussians, and 71D geometry")
    parser.add_argument("--config", default=str(_bootstrap.PROJECT_ROOT / "configs" / "fold0_5090.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = preprocess_dataset(load_config(args.config), force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
