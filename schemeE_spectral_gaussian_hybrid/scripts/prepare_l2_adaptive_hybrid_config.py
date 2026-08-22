from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from scheme_e.adaptive_experiment import adaptive_hybrid_config
from scheme_e.config import load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the evidence-driven adaptive-Teacher Hybrid fine-tune"
    )
    parser.add_argument("--base", default="configs/v4_attempt1_structured.json")
    parser.add_argument(
        "--adaptive-prior",
        default="artifacts/v6/fold0/adaptive_local_bank_priors.npz",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default="artifacts/v4/fold0_attempt1/hybrid/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/scheme_e_065/l2_001_adaptive_hybrid/hybrid",
    )
    parser.add_argument(
        "--output", default="configs/l2_001_adaptive_hybrid.json"
    )
    args = parser.parse_args()
    config = adaptive_hybrid_config(
        load_config(args.base),
        adaptive_prior=args.adaptive_prior,
        initial_checkpoint=args.initial_checkpoint,
        output_dir=args.output_dir,
    )
    save_json(args.output, config)
    print(
        json.dumps(
            {
                "output": args.output,
                "adaptive_prior": config["spectral_teacher"]["oof_output_path"],
                "initial_checkpoint": config["hybrid"]["initial_checkpoint"],
                "hybrid_output_dir": config["hybrid"]["output_dir"],
                "learning_rate": config["hybrid"]["learning_rate"],
                "epochs": config["hybrid"]["epochs"],
                "maximum_training_hours": config["hybrid"]["maximum_training_hours"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
