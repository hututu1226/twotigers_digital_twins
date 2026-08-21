from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


def main() -> None:
    config = json.loads(Path("configs/smoke.json").read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["spectral_teacher"].update(
        {
            "power_calibration": {
                "enabled": True,
                "slope_bounds": [0.6, 1.4],
            },
            "oof_output_path": "artifacts/smoke_v2/spectral_teacher/oof_priors.npz",
            "oof_report_path": "artifacts/smoke_v2/spectral_teacher/oof_report.json",
            "test_output_path": "artifacts/smoke_v2/spectral_teacher/test_priors.npz",
            "model_path": "artifacts/smoke_v2/spectral_teacher/model.pkl",
            "final_report_path": "artifacts/smoke_v2/spectral_teacher/final_report.json",
        }
    )
    config["hybrid"].update(
        {
            "reference_aware": True,
            "station_embedding": True,
            "reference_sampling": "test_matched",
            "reference_strategies": [
                {"name": "nearest", "top_k": 1},
                {
                    "name": "spectral_local",
                    "top_k": 2,
                    "distance_weight": 1.0,
                    "pas_weight": 1.0,
                    "pdp_weight": 1.0,
                    "geometry_weight": 0.1,
                },
            ],
            "power_bound_quantiles": [0.01, 0.99],
            "maximum_power_delta": 0.25,
            "projection_candidates": [1],
            "output_dir": "artifacts/smoke_v2/hybrid",
        }
    )
    config["hybrid_final"].update(
        {
            "reference_sampling": "test_matched",
            "initial_checkpoint": "artifacts/smoke_v2/hybrid/best.pt",
            "output_dir": "artifacts/smoke_v2/final_hybrid",
        }
    )
    config["inference"].update(
        {
            "checkpoint": "artifacts/smoke_v2/final_hybrid/best.pt",
            "output_path": "outputs/smoke_v2/Round2_Test_Channel.npy",
            "report_path": "reports/generated/smoke_v2_inference.json",
            "reference_strategy": {"name": "spectral_local", "top_k": 2, "distance_weight": 1.0, "pas_weight": 1.0, "pdp_weight": 1.0, "geometry_weight": 0.1},
            "outage_threshold_by_cell": [0.99, 0.99],
            "soft_outage_strength_by_cell": [0.5, 0.5],
        }
    )
    destination = Path("configs/v2_smoke_generated.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
