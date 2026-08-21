from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTEMPTS = {
    1: "configs/v3_attempt1_conservative.json",
    2: "configs/v3_attempt2_flexible.json",
    3: "configs/v3_attempt3_decoder.json",
}


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best Scheme E-v3 Fold attempt")
    parser.add_argument("--output", default="configs/v3_fold_best.json")
    parser.add_argument("--report", default="reports/generated/v3_attempt_selection.json")
    args = parser.parse_args()
    candidates = []
    for attempt, config_path in ATTEMPTS.items():
        config = _read(config_path)
        summary = _read(Path(config["hybrid"]["output_dir"]) / "summary.json")
        policy = _read(f"reports/generated/v3_attempt{attempt}_policy.json")
        candidates.append(
            {
                "attempt": attempt,
                "config_path": config_path,
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
                "transport_spectrum_gate_by_cell": value["summary"][
                    "best_metrics"
                ].get("transport_spectrum_gate_by_cell"),
                "transport_detail_gate_by_cell": value["summary"][
                    "best_metrics"
                ].get("transport_detail_gate_by_cell"),
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
