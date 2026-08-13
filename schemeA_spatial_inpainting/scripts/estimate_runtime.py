from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_estimate(name: str, history_path: Path, configured_epochs: int, recent: int) -> dict:
    records = read_history(history_path)
    if not records:
        return {"stage": name, "status": "no history"}
    durations = [float(record["elapsed_seconds"]) for record in records[-recent:]]
    average = statistics.mean(durations)
    completed = int(records[-1]["epoch"]) + 1
    remaining = max(configured_epochs - completed, 0)
    return {
        "stage": name,
        "completed_epochs": completed,
        "configured_epochs": configured_epochs,
        "recent_epochs_used": len(durations),
        "average_epoch_seconds": average,
        "estimated_remaining_minutes": average * remaining / 60.0,
        "estimated_full_stage_minutes": average * configured_epochs / 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate remaining training time from JSONL logs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--recent", type=int, default=5)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    project_root = config_path.parent.parent
    stages = []
    for name in ("autoencoder", "spatial"):
        output_dir = Path(config[name]["output_dir"])
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        stages.append(
            stage_estimate(
                name,
                output_dir / "history.jsonl",
                int(config[name]["epochs"]),
                args.recent,
            )
        )
    print(json.dumps(stages, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

