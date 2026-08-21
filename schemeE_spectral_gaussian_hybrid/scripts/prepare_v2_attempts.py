from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare three bounded Scheme E-v2 Fold attempts")
    parser.add_argument("--base", default="configs/v2_5090.json")
    parser.add_argument("--output-dir", default="configs")
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[tuple[str, dict]] = []
    safe = deepcopy(base)
    safe["seed"] = 2026
    safe["hybrid"].update(
        {
            "reference_aware": False,
            "station_embedding": False,
            "train_decoder": False,
            "reference_strategies": [{"name": "nearest", "top_k": 1}],
            "output_dir": "artifacts/v2/fold0_attempt1/hybrid",
        }
    )
    attempts.append(("v2_attempt1_safe.json", safe))

    reference = deepcopy(base)
    reference["seed"] = 2039
    reference["hybrid"]["output_dir"] = "artifacts/v2/fold0_attempt2/hybrid"
    attempts.append(("v2_attempt2_reference.json", reference))

    decoder = deepcopy(base)
    decoder["seed"] = 2053
    decoder["hybrid"].update(
        {
            "train_decoder": True,
            "decoder_learning_rate_scale": 0.03,
            "learning_rate": 0.00012,
            "maximum_spectrum_residual": 0.5,
            "maximum_detail_residual": 0.5,
            "output_dir": "artifacts/v2/fold0_attempt3/hybrid",
        }
    )
    attempts.append(("v2_attempt3_decoder.json", decoder))

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
