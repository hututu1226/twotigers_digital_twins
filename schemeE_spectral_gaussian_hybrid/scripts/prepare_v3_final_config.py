from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full-data Scheme E-v3 config")
    parser.add_argument(
        "--selection", default="reports/generated/v3_attempt_selection.json"
    )
    parser.add_argument("--output", default="configs/v3_final_selected.json")
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    attempt = int(selection["selected_attempt"])
    config = json.loads(
        Path(selection["selected_config"]).read_text(encoding="utf-8")
    )
    summary = json.loads(
        (Path(config["hybrid"]["output_dir"]) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads(
        Path(f"reports/generated/v3_attempt{attempt}_policy.json").read_text(
            encoding="utf-8"
        )
    )
    config = deepcopy(config)
    config["split"]["validation_fold"] = None
    config["spectral_teacher"].update(
        {
            "oof_output_path": "artifacts/v2/final/spectral_teacher/oof_priors.npz",
            "oof_report_path": "artifacts/v2/final/spectral_teacher/oof_report.json",
            "test_output_path": "artifacts/v2/final/spectral_teacher/test_priors.npz",
            "model_path": "artifacts/v2/final/spectral_teacher/model.pkl",
            "final_report_path": "artifacts/v2/final/spectral_teacher/final_report.json",
        }
    )
    best_epoch = int(summary["best_epoch"])
    config["hybrid_final"]["epochs"] = max(
        1, min(900, int(math.ceil(best_epoch * 1.25)))
    )
    config["hybrid_final"].update(
        {
            "output_dir": "artifacts/v3/final/hybrid",
            "train_decoder": bool(config["hybrid"].get("train_decoder", False)),
            "decoder_learning_rate_scale": float(
                config["hybrid"].get("decoder_learning_rate_scale", 1.0)
            ),
        }
    )
    config["hybrid_final"].pop("initial_checkpoint", None)
    projection = int(summary["selected_projection_iterations"])
    strategy_name = str(summary.get("selected_reference_strategy", "nearest"))
    strategy = next(
        (
            dict(value)
            for value in config["hybrid"].get("reference_strategies", [])
            if str(value.get("name")) == strategy_name
        ),
        {"name": "nearest", "top_k": 1},
    )
    config["hybrid_final"]["projection_iterations"] = projection
    config["inference"].update(
        {
            "checkpoint": "artifacts/v3/final/hybrid/best.pt",
            "output_path": "outputs/v3/Round2_Test_Channel.npy",
            "report_path": "reports/generated/v3_final_inference.json",
            "projection_iterations": projection,
            "reference_strategy": strategy,
            "outage_threshold_by_cell": policy["outage_threshold_by_cell"],
            "soft_outage_strength_by_cell": policy[
                "soft_outage_strength_by_cell"
            ],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_attempt": attempt,
                "selected_fold_score": selection["selected_score"],
                "final_epochs": config["hybrid_final"]["epochs"],
                "projection_iterations": projection,
                "reference_strategy": strategy,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
