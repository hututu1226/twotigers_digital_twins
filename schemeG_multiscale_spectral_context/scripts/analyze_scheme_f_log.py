"""Summarize Scheme F context-training runs from an AutoDL launcher log."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


LINE = re.compile(
    r"Context epoch=(?P<epoch>\d+)/(?P<maximum>\d+) "
    r"train=(?P<train>[0-9.eE+-]+) score=(?P<score>[0-9.eE+-]+|nan) "
    r"threshold=(?P<threshold>[0-9.eE+-]+|nan) seconds=(?P<seconds>[0-9.eE+-]+)"
)


def summarize(records: list[dict[str, float | int]]) -> dict[str, object]:
    validation = [row for row in records if math.isfinite(float(row["score"]))]
    best = max(validation, key=lambda row: float(row["score"])) if validation else None
    milestones = []
    for stop in range(50, int(records[-1]["epoch"]) + 50, 50):
        window = [row for row in validation if int(row["epoch"]) <= stop]
        if window:
            current = max(window, key=lambda row: float(row["score"]))
            milestones.append(
                {
                    "through_epoch": stop,
                    "best_epoch": int(current["epoch"]),
                    "best_score": float(current["score"]),
                }
            )
    tail = records[-min(25, len(records)) :]
    return {
        "maximum_epochs": int(records[0]["maximum"]),
        "last_epoch": int(records[-1]["epoch"]),
        "elapsed_seconds": float(sum(float(row["seconds"]) for row in records)),
        "first_25_train_loss": float(
            sum(float(row["train"]) for row in records[:25]) / min(25, len(records))
        ),
        "last_25_train_loss": float(
            sum(float(row["train"]) for row in tail) / len(tail)
        ),
        "best_validation": best,
        "validation_points": len(validation),
        "best_score_milestones": milestones,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs: list[list[dict[str, float | int]]] = []
    current: list[dict[str, float | int]] = []
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.search(line)
        if not match:
            continue
        epoch = int(match.group("epoch"))
        if current and epoch <= int(current[-1]["epoch"]):
            runs.append(current)
            current = []
        current.append(
            {
                "epoch": epoch,
                "maximum": int(match.group("maximum")),
                "train": float(match.group("train")),
                "score": float(match.group("score")),
                "threshold": float(match.group("threshold")),
                "seconds": float(match.group("seconds")),
            }
        )
    if current:
        runs.append(current)
    report = {
        "log": str(args.log.resolve()),
        "runs": [summarize(records) for records in runs],
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
