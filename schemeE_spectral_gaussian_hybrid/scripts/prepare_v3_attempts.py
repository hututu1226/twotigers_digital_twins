from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Scheme E-v3 Fold attempts")
    parser.add_argument("--base", default="configs/v3_5090.json")
    parser.add_argument("--output-dir", default="configs")
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conservative = deepcopy(base)
    conservative["seed"] = 2081
    conservative["hybrid"].update(
        {
            "maximum_spectrum_residual": 0.5,
            "maximum_detail_residual": 0.5,
            "train_decoder": False,
            "output_dir": "artifacts/v3/fold0_attempt1/hybrid",
        }
    )
    conservative["hybrid"]["loss_weights"].update(
        {"seed_spectrum": 0.15, "seed_detail": 0.08}
    )

    flexible = deepcopy(base)
    flexible["seed"] = 2093
    flexible["hybrid"].update(
        {
            "train_decoder": False,
            "output_dir": "artifacts/v3/fold0_attempt2/hybrid",
        }
    )

    decoder = deepcopy(base)
    decoder["seed"] = 2111
    decoder["hybrid"].update(
        {
            "train_decoder": True,
            "decoder_learning_rate_scale": 0.03,
            "learning_rate": 0.00012,
            "maximum_spectrum_residual": 0.5,
            "maximum_detail_residual": 0.5,
            "output_dir": "artifacts/v3/fold0_attempt3/hybrid",
        }
    )

    attempts = [
        ("v3_attempt1_conservative.json", conservative),
        ("v3_attempt2_flexible.json", flexible),
        ("v3_attempt3_decoder.json", decoder),
    ]
    written = []
    for name, config in attempts:
        path = output_dir / name
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(str(path))
    print(json.dumps({"status": "PASS", "attempts": written}, indent=2))


if __name__ == "__main__":
    main()
