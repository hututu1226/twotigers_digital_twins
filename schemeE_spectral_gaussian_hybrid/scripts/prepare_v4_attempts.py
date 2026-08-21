from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


def _configure(base: dict, output_dir: str) -> dict:
    config = deepcopy(base)
    config["hybrid"].update(
        {
            "preserve_spectral_positions": True,
            "structured_spectral_field": True,
            "maximum_spectrum_residual": 0.75,
            "maximum_detail_residual": 0.75,
            "epochs": 900,
            "early_stopping_patience": 120,
            "maximum_training_hours": 3.0,
            "output_dir": output_dir,
        }
    )
    config["hybrid"]["loss_weights"].update(
        {
            "score": 1.2,
            "spectrum_latent": 0.08,
            "detail_latent": 0.03,
            "detail_correlation": 0.06,
            "power": 0.18,
            "residual": 0.003,
            "seed_spectrum": 0.06,
            "seed_detail": 0.04,
        }
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Scheme E-v4 Fold0 attempts")
    parser.add_argument("--base", default="configs/v3_5090.json")
    parser.add_argument("--output-dir", default="configs")
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structured = _configure(base, "artifacts/v4/fold0_attempt1/hybrid")
    structured["seed"] = 2141
    structured["hybrid"].update(
        {
            "train_decoder": False,
            "learning_rate": 0.0002,
        }
    )

    decoder = _configure(base, "artifacts/v4/fold0_attempt2/hybrid")
    decoder["seed"] = 2153
    decoder["hybrid"].update(
        {
            "initial_checkpoint": "artifacts/v4/fold0_attempt1/hybrid/best.pt",
            "train_decoder": True,
            "decoder_learning_rate_scale": 0.03,
            "learning_rate": 0.0001,
        }
    )

    warm_structured = deepcopy(base)
    warm_structured["seed"] = 2161
    warm_structured["hybrid"].update(
        {
            "preserve_spectral_positions": False,
            "structured_spectral_field": True,
            "initial_checkpoint": "artifacts/v3/fold0_attempt3/hybrid/best.pt",
            "allow_partial_initial_checkpoint": True,
            "train_decoder": False,
            "learning_rate": 0.00008,
            "maximum_spectrum_residual": 0.5,
            "maximum_detail_residual": 0.5,
            "epochs": 700,
            "early_stopping_patience": 100,
            "maximum_training_hours": 2.5,
            "output_dir": "artifacts/v4/fold0_attempt3/hybrid",
        }
    )

    paths = []
    for name, config in (
        ("v4_attempt1_structured.json", structured),
        ("v4_attempt2_decoder.json", decoder),
        ("v4_attempt3_warm_structured.json", warm_structured),
    ):
        path = output_dir / name
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(str(path))
    print(json.dumps({"status": "PASS", "attempts": paths}, indent=2))


if __name__ == "__main__":
    main()
