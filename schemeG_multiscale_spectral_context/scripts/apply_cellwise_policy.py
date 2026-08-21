from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from scheme_g.config import save_json
from scheme_g.reporting import evaluation_metrics


def apply_cellwise_policy(evaluation: dict, scan: dict) -> dict:
    combined = scan.get("best_cellwise_combined")
    if not isinstance(combined, dict):
        return evaluation
    metrics = evaluation_metrics(evaluation)
    global_score = float(metrics["score"])
    cellwise_score = float(combined["score"])
    if cellwise_score + 1e-9 < global_score:
        raise ValueError(
            "Cell-specific policy is worse than the global policy; scan is inconsistent"
        )
    metrics.update(
        {
            "pas": float(combined["pas"]),
            "pdp": float(combined["pdp"]),
            "nmse": float(combined["nmse"]),
            "score": cellwise_score,
            "global_policy_score": global_score,
            "cellwise_gain_vs_global": cellwise_score - global_score,
            "outage_threshold_by_cell": scan["best_threshold_by_cell"],
            "soft_outage_strength_by_cell": scan["best_soft_strength_by_cell"],
            "spectral_prior_alpha_by_cell": scan["best_spectral_prior_alpha_by_cell"],
        }
    )
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote the exact BS0/BS1 policy score into an evaluation report"
    )
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--scan", required=True)
    args = parser.parse_args()
    evaluation_path = Path(args.evaluation)
    scan_path = Path(args.scan)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    result = apply_cellwise_policy(evaluation, scan)
    save_json(evaluation_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
