from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from scheme_e.config import load_config
from scheme_e.strict_prior import build_strict_fold_prior


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free Scheme E-v2 Fold prior")
    parser.add_argument("--config", default="configs/v2_5090.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/v2/fold0/spectral_teacher/strict_priors.npz"
    )
    parser.add_argument(
        "--report", default="artifacts/v2/fold0/spectral_teacher/strict_report.json"
    )
    args = parser.parse_args()
    report = build_strict_fold_prior(
        load_config(args.config), args.fold, args.output, args.report
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
