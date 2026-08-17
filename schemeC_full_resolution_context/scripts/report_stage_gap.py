from __future__ import annotations

import argparse
import json
from pathlib import Path


def metrics(path: str) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return report.get("metrics", report)


def ceiling_gap(ceiling: dict, prediction: dict) -> dict:
    return {
        "pas_loss": float(ceiling["pas"]) - float(prediction["pas"]),
        "pdp_loss": float(ceiling["pdp"]) - float(prediction["pdp"]),
        "nmse_increase": float(prediction["nmse"]) - float(ceiling["nmse"]),
        "score_loss": float(ceiling["score"]) - float(prediction["score"]),
    }


def joint_gain(context: dict, joint: dict) -> dict:
    return {
        "pas_gain": float(joint["pas"]) - float(context["pas"]),
        "pdp_gain": float(joint["pdp"]) - float(context["pdp"]),
        "nmse_reduction": float(context["nmse"]) - float(joint["nmse"]),
        "score_gain": float(joint["score"]) - float(context["score"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate representation ceiling from spatial-prediction loss"
    )
    parser.add_argument(
        "--autoencoder", default="artifacts/fold0/autoencoder/evaluation.json"
    )
    parser.add_argument("--context", default="artifacts/fold0/context/evaluation.json")
    parser.add_argument("--joint", default="artifacts/fold0/joint/evaluation.json")
    parser.add_argument("--output", default="artifacts/fold0/stage_gap.json")
    args = parser.parse_args()
    autoencoder = metrics(args.autoencoder)
    context = metrics(args.context)
    joint = metrics(args.joint)
    result = {
        "autoencoder_ceiling": autoencoder,
        "context_before_joint": context,
        "context_after_joint": joint,
        "ceiling_to_joint_gap": ceiling_gap(autoencoder, joint),
        "joint_finetune_gain": joint_gain(context, joint),
        "interpretation": {
            "low_autoencoder_score": "representation bottleneck; improve AE before context model",
            "large_ceiling_to_joint_gap": "context/query prediction bottleneck",
            "small_joint_finetune_gain": "joint fine-tuning adds little and can be shortened"
        }
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
