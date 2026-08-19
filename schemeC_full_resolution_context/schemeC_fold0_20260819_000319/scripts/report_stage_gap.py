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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate representation ceiling from spatial-prediction loss"
    )
    parser.add_argument(
        "--autoencoder", default="artifacts/fold0/autoencoder/evaluation.json"
    )
    parser.add_argument("--context", default="artifacts/fold0/context/evaluation.json")
    parser.add_argument("--output", default="artifacts/fold0/stage_gap.json")
    args = parser.parse_args()
    autoencoder = metrics(args.autoencoder)
    context = metrics(args.context)
    result = {
        "autoencoder_ceiling": autoencoder,
        "context_v2": context,
        "ceiling_to_context_gap": ceiling_gap(autoencoder, context),
        "interpretation": {
            "low_autoencoder_score": "representation bottleneck; improve AE before context model",
            "large_ceiling_to_context_gap": "context/query prediction bottleneck",
            "decoder_training": "the decoder is optimized inside the single Context V2 run"
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
