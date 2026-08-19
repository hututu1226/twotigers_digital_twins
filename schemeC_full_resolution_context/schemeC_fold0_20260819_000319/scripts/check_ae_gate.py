from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from scheme_c.config import load_config, save_json


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stop Scheme C before Context when the AE evidence is insufficient"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    gate = config["autoencoder"].get("quality_gates", {})
    enabled = bool(gate.get("enabled", True))
    evaluation = read_json(args.evaluation)
    ablation = read_json(args.ablation)
    score = float(evaluation["metrics"]["score"])
    detail_gain = float(ablation["detail_gain"])
    shuffle_drop = float(ablation["shuffle_drop"])
    thresholds = {
        "minimum_score": float(gate.get("minimum_score", 0.75)),
        "minimum_detail_gain": float(gate.get("minimum_detail_gain", 0.10)),
        "minimum_shuffle_drop": float(gate.get("minimum_shuffle_drop", 0.05)),
    }
    checks = {
        "score": score >= thresholds["minimum_score"],
        "detail_gain": detail_gain >= thresholds["minimum_detail_gain"],
        "shuffle_drop": shuffle_drop >= thresholds["minimum_shuffle_drop"],
    }
    passed = all(checks.values())
    report = {
        "status": "SKIPPED" if not enabled else "PASS" if passed else "FAIL",
        "enabled": enabled,
        "measurements": {
            "score": score,
            "detail_gain": detail_gain,
            "shuffle_drop": shuffle_drop,
        },
        "thresholds": thresholds,
        "checks": checks,
        "context_training_allowed": bool(not enabled or passed),
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if enabled and not passed:
        failed = ", ".join(name for name, value in checks.items() if not value)
        raise SystemExit(
            f"AE quality gate failed ({failed}); Context training was not started"
        )


if __name__ == "__main__":
    main()
