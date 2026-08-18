from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stage_estimate(name: str, path: Path, epochs: int, recent: int) -> dict:
    records = read_history(path)
    if not records:
        return {"stage": name, "status": "no history"}
    durations = [float(record["elapsed_seconds"]) for record in records[-recent:]]
    average = statistics.mean(durations)
    completed = int(records[-1]["epoch"]) + 1
    remaining = max(int(epochs) - completed, 0)
    return {
        "stage": name,
        "completed_epochs": completed,
        "configured_epochs": int(epochs),
        "recent_epochs_used": len(durations),
        "average_epoch_seconds": average,
        "estimated_remaining_minutes": average * remaining / 60.0,
        "estimated_full_stage_minutes": average * int(epochs) / 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Scheme C time from real JSONL logs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--recent", type=int, default=5)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    results = []
    for name in ("autoencoder", "context"):
        if name not in config:
            continue
        output = Path(config[name]["output_dir"])
        output = output if output.is_absolute() else root / output
        results.append(
            stage_estimate(
                name,
                output / "history.jsonl",
                int(config[name]["epochs"]),
                max(1, args.recent),
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
