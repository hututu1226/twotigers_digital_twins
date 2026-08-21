from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


def main() -> None:
    config = deepcopy(json.loads(Path("configs/smoke.json").read_text(encoding="utf-8")))
    fold_transport = {
        "enabled": True,
        "count": 2,
        "distance_power": 2.0,
        "prior_wave_number": -140.33,
        "search_radius": 4.0,
        "fit_targets_per_cell": 4,
        "fit_neighbors": 2,
        "fit_seed": 2026,
        "fit_path": "artifacts/smoke_v3/fold0/carrier_fit.json",
    }
    final_transport = {
        **fold_transport,
        "fit_path": "artifacts/smoke_v3/final/carrier_fit.json",
    }
    config["hybrid"].update(
        {
            "transport_seed": fold_transport,
            "output_dir": "artifacts/smoke_v3/hybrid",
        }
    )
    config["hybrid"]["loss_weights"].update(
        {"seed_spectrum": 0.01, "seed_detail": 0.01}
    )
    config["hybrid_final"].update(
        {
            "initial_checkpoint": "artifacts/smoke_v3/hybrid/best.pt",
            "transport_seed": final_transport,
            "output_dir": "artifacts/smoke_v3/final_hybrid",
        }
    )
    config["hybrid_final"]["loss_weights"].update(
        {"seed_spectrum": 0.01, "seed_detail": 0.01}
    )
    config["inference"].update(
        {
            "transport_seed": {"count": 2, "distance_power": 2.0},
            "checkpoint": "artifacts/smoke_v3/final_hybrid/best.pt",
            "output_path": "outputs/smoke_v3/Round2_Test_Channel.npy",
            "report_path": "reports/generated/smoke_v3_inference.json",
        }
    )
    output = Path("configs/v3_smoke_generated.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
