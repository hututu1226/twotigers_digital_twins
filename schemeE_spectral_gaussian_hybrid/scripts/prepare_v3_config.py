from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Scheme E-v3 dual-seed config")
    parser.add_argument("--base", default="configs/v2_5090.json")
    parser.add_argument("--output", default="configs/v3_5090.json")
    args = parser.parse_args()
    config = deepcopy(json.loads(Path(args.base).read_text(encoding="utf-8")))
    transport_fold = {
        "enabled": True,
        "count": 8,
        "distance_power": 2.0,
        "prior_wave_number": -140.33,
        "search_radius": 12.0,
        "fit_targets_per_cell": 256,
        "fit_neighbors": 4,
        "fit_seed": 2026,
        "fit_path": "artifacts/v3/fold0/carrier_fit.json",
    }
    transport_final = {
        **transport_fold,
        "fit_path": "artifacts/v3/final/carrier_fit.json",
    }
    config["hybrid"].update(
        {
            "transport_seed": transport_fold,
            "output_dir": "artifacts/v3/fold0_attempt2/hybrid",
        }
    )
    config["hybrid"]["loss_weights"].update(
        {"seed_spectrum": 0.08, "seed_detail": 0.05, "transport_gate": 0.0}
    )
    config["hybrid_final"].update(
        {
            "transport_seed": transport_final,
            "output_dir": "artifacts/v3/final/hybrid",
        }
    )
    config["hybrid_final"]["loss_weights"].update(
        {"seed_spectrum": 0.08, "seed_detail": 0.05, "transport_gate": 0.0}
    )
    config["inference"].update(
        {
            "transport_seed": {"count": 8, "distance_power": 2.0},
            "checkpoint": "artifacts/v3/final/hybrid/best.pt",
            "output_path": "outputs/v3/Round2_Test_Channel.npy",
            "report_path": "reports/generated/v3_final_inference.json",
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
