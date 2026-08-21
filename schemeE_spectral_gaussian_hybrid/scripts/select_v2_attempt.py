from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best Scheme E-v2 Fold attempt")
    parser.add_argument("--output", default="configs/v2_fold_best.json")
    parser.add_argument("--report", default="reports/generated/v2_attempt_selection.json")
    args = parser.parse_args()
    candidates = []
    for attempt in (1, 2, 3):
        config_path = Path(
            {
                1: "configs/v2_attempt1_safe.json",
                2: "configs/v2_attempt2_reference.json",
                3: "configs/v2_attempt3_decoder.json",
            }[attempt]
        )
        config = _read(config_path)
        summary_path = Path(config["hybrid"]["output_dir"]) / "summary.json"
        policy_path = Path(f"reports/generated/v2_attempt{attempt}_policy.json")
        summary = _read(summary_path)
        policy = _read(policy_path)
        candidates.append(
            {
                "attempt": attempt,
                "config_path": str(config_path),
                "config": config,
                "summary": summary,
                "policy": policy,
                "score": float(policy["selected"]["score"]),
            }
        )
    selected = max(candidates, key=lambda value: value["score"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selected["config"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "PASS",
        "selected_attempt": int(selected["attempt"]),
        "selected_score": float(selected["score"]),
        "selected_config": str(output),
        "recommended_for_submission": bool(selected["score"] >= 0.65),
        "candidates": [
            {
                "attempt": int(value["attempt"]),
                "config": value["config_path"],
                "raw_score": float(value["summary"]["best_metrics"]["score"]),
                "policy_score": float(value["score"]),
                "best_epoch": int(value["summary"]["best_epoch"]),
                "policy_gain": float(value["policy"]["score_gain"]),
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
