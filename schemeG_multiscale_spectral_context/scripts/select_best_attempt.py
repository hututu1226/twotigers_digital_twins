from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from scheme_g.reporting import evaluation_metrics


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the better of at most two Scheme G folds"
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-config", default="configs/fold0_best.json")
    parser.add_argument("--canonical-dir", default="artifacts/fold0/context")
    parser.add_argument(
        "--report", default="reports/generated/fold_attempt_selection.json"
    )
    args = parser.parse_args()
    candidates: list[dict] = []
    for config_path in args.configs:
        config = read(Path(config_path))
        directory = Path(config["context"]["output_dir"])
        evaluation_path = directory / "evaluation.json"
        if not evaluation_path.is_file():
            continue
        metrics = evaluation_metrics(read(evaluation_path))
        candidates.append(
            {
                "config_path": config_path,
                "config": config,
                "directory": directory,
                "metrics": metrics,
            }
        )
    if not candidates:
        raise FileNotFoundError("No completed Scheme G attempt evaluation was found")
    best = max(candidates, key=lambda item: float(item["metrics"]["score"]))
    canonical = Path(args.canonical_dir)
    for name in (
        "best.pt",
        "last.pt",
        "summary.json",
        "evaluation.json",
        "outage_scan.json",
    ):
        source = best["directory"] / name
        if source.is_file():
            copy_or_link(source, canonical / name)
    config = best["config"]
    config["context"]["output_dir"] = str(canonical)
    config["inference"]["context_checkpoint"] = str(canonical / "best.pt")
    output_config = Path(args.output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    score = float(best["metrics"]["score"])
    report = {
        "status": "PASS",
        "selected": best["config_path"],
        "selected_metrics": best["metrics"],
        "recommended_for_submission": score >= 0.65,
        "warning": None
        if score >= 0.65
        else "Fold0 score is below 0.65; final output is diagnostic only.",
        "candidates": [
            {"config": item["config_path"], "metrics": item["metrics"]}
            for item in candidates
        ],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
