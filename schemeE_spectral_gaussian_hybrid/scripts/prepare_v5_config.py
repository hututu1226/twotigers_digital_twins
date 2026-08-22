from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import _bootstrap  # noqa: F401

from scheme_e.config import load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Scheme E-v5 local teacher config")
    parser.add_argument("--base", default="configs/v4_attempt1_structured.json")
    parser.add_argument("--output", default="configs/v5_local_teacher.json")
    args = parser.parse_args()

    config = deepcopy(load_config(args.base))
    config["seed"] = 2171
    config["spectral_teacher"].update(
        {
            "local_spectral_experts": [
                {"name": "idw8_p1", "neighbors": 8, "distance_power": 1.0},
                {"name": "idw8_p2", "neighbors": 8, "distance_power": 2.0},
            ],
            "oof_output_path": "artifacts/v5/fold0/spectral_teacher/strict_priors.npz",
            "oof_report_path": "artifacts/v5/fold0/spectral_teacher/strict_oof_report.json",
            "test_output_path": "artifacts/v5/final/spectral_teacher/test_priors.npz",
            "model_path": "artifacts/v5/final/spectral_teacher/model.pkl",
            "final_report_path": "artifacts/v5/final/spectral_teacher/final_report.json",
        }
    )
    config["hybrid"].update(
        {
            "output_dir": "artifacts/v5/fold0/hybrid",
            "initial_checkpoint": "artifacts/v4/fold0_attempt1/hybrid/best.pt",
            "allow_partial_initial_checkpoint": False,
            "epochs": 700,
            "learning_rate": 1.0e-4,
            "scheduler_patience_validations": 6,
            "early_stopping_patience": 90,
            "maximum_training_hours": 3.0,
        }
    )
    config["inference"].update(
        {
            "checkpoint": "artifacts/v5/final/hybrid/best.pt",
            "output_path": "outputs/v5/Round2_Test_Channel.npy",
            "report_path": "reports/generated/v5_final_inference.json",
        }
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_json(destination, config)
    print(destination)


if __name__ == "__main__":
    main()
