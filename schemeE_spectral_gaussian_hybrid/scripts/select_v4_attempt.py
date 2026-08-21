from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTEMPTS = {
    1: "configs/v4_attempt1_structured.json",
    2: "configs/v4_attempt2_decoder.json",
    3: "configs/v4_attempt3_warm_structured.json",
}


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best Scheme E-v4 attempt")
    parser.add_argument("--output", default="configs/v4_fold_best.json")
    parser.add_argument("--report", default="reports/generated/v4_attempt_selection.json")
    args = parser.parse_args()
    candidates = []
    skipped = []
    for attempt, config_path in ATTEMPTS.items():
        config = _read(config_path)
        required = {
            "summary": Path(config["hybrid"]["output_dir"]) / "summary.json",
            "policy": Path(f"reports/generated/v4_attempt{attempt}_policy.json"),
            "projection": Path(
                f"reports/generated/v4_attempt{attempt}_output_projection.json"
            ),
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            skipped.append({"attempt": attempt, "missing": missing})
            continue
        summary = _read(required["summary"])
        policy = _read(required["policy"])
        projection = _read(required["projection"])
        policy_score = float(policy["selected"]["score"])
        projection_score = float(projection["selected"]["score"])
        selected_projection = projection["selected"]
        candidates.append(
            {
                "attempt": attempt,
                "config_path": config_path,
                "config": config,
                "summary": summary,
                "policy": policy,
                "projection": projection,
                "score": max(policy_score, projection_score),
                "output_projection": {
                    "iterations": int(
                        selected_projection.get("output_projection_iterations", 0)
                    ),
                    "strength_by_cell": selected_projection.get(
                        "output_projection_strength_by_cell", [0.0, 0.0]
                    ),
                    "minimum_scale": 0.5,
                    "maximum_scale": 2.0,
                },
            }
        )
    if not candidates:
        raise RuntimeError("No completed Scheme E-v4 attempt is available")
    selected = max(candidates, key=lambda value: value["score"])
    selected_config = selected["config"]
    selected_config["inference"]["output_projection"] = selected[
        "output_projection"
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selected_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "selected_attempt": int(selected["attempt"]),
        "selected_score": float(selected["score"]),
        "selected_config": str(output),
        "selected_output_projection": selected["output_projection"],
        "recommended_for_submission": bool(selected["score"] >= 0.65),
        "skipped_attempts": skipped,
        "candidates": [
            {
                "attempt": int(value["attempt"]),
                "config": value["config_path"],
                "raw_score": float(value["summary"]["best_metrics"]["score"]),
                "policy_score": float(value["policy"]["selected"]["score"]),
                "projection_score": float(value["projection"]["selected"]["score"]),
                "best_epoch": int(value["summary"]["best_epoch"]),
            }
            for value in candidates
        ],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
