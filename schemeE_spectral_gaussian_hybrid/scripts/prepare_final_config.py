from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Scheme E final epochs and projection from Fold0 evidence")
    parser.add_argument("--base", default=str(_bootstrap.PROJECT_ROOT / "configs" / "fold0_5090.json"))
    parser.add_argument("--summary", default=str(_bootstrap.PROJECT_ROOT / "artifacts" / "fold0" / "hybrid" / "summary.json"))
    parser.add_argument("--output", default=str(_bootstrap.PROJECT_ROOT / "configs" / "final_selected.json"))
    args = parser.parse_args()
    base_path = Path(args.base)
    summary_path = Path(args.summary)
    config = json.loads(base_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_epoch = max(int(summary["best_epoch"]), 1)
    projection = int(summary["selected_projection_iterations"])
    config["split"]["validation_fold"] = None
    config["hybrid_final"]["epochs"] = best_epoch
    config["hybrid_final"].pop("initial_checkpoint", None)
    config["hybrid_final"]["projection_iterations"] = projection
    config["inference"]["projection_iterations"] = projection
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "selected_epochs": best_epoch,
                "selected_projection_iterations": projection,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
