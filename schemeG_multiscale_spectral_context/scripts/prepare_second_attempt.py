from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
from scheme_g.reporting import evaluation_metrics


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the single allowed Scheme G repair attempt"
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--skip-at-score", type=float, default=0.65)
    args = parser.parse_args()
    config = read(args.base)
    metrics = evaluation_metrics(read(args.evaluation))
    score = float(metrics["score"])
    if not math.isfinite(score):
        raise ValueError("Attempt 1 score is not finite")
    run_second = score < args.skip_at_score
    actions: list[str] = []
    context = config["context"]
    context["output_dir"] = "artifacts/fold0_attempt2/context"
    config["inference"]["context_checkpoint"] = (
        "artifacts/fold0_attempt2/context/best.pt"
    )
    config["seed"] = int(config["seed"]) + 17
    context["learning_rate"] = min(float(context["learning_rate"]), 0.00015)
    context["steps_per_epoch"] = max(int(context["steps_per_epoch"]), 64)
    context["maximum_training_hours"] = 3.0

    if float(metrics.get("nmse", 0.0)) > 0.9:
        context["loss_weights"]["power"] = 0.38
        context["loss_weights"]["power_quantile"] = 0.15
        context["maximum_power_z"] = 4.0
        context["maximum_power_residual"] = 0.9
        actions.append("tighten PowerCNP and increase power/quantile supervision")
    if float(metrics.get("pas", 1.0)) + 0.12 < float(metrics.get("pdp", 0.0)):
        context["chart_weight"] = 0.25
        context["router_uniform_mix"] = 0.15
        context["loss_weights"]["chart"] = 0.16
        context["loss_weights"]["spectrum_latent"] = 0.28
        actions.append(
            "strengthen gridded Spectrum supervision without letting chart similarity override distance"
        )
    token_effective = float(metrics.get("detail_token_effective_neighbors", 2.0))
    if token_effective < 1.2:
        context["router_temperature_initial"] = 2.0
        context["router_temperature_final"] = 1.2
        context["loss_weights"]["token_diversity"] = 0.05
        actions.append(
            "prevent per-token Top4 attention from collapsing to Top1 too early"
        )
    if float(metrics.get("detail_warp_bins", 1.0)) < 0.03:
        context["detail_maximum_warp"] = [0.5, 1.0, 2.0]
        context["loss_weights"]["warp_saturation"] = 0.002
        actions.append("reduce unused Detail warp range and saturation penalty")
    router_distance = float(metrics.get("router_distance_meters", 0.0))
    if router_distance > 20.0:
        context["router_candidate_top_k"] = 64
        context["router_distance_penalty"] = 3.0
        actions.append(
            "constrain Spectrum anchors because their mean distance exceeded 20 m"
        )
    detail_distance = float(metrics.get("detail_router_distance_meters", 0.0))
    if detail_distance > 12.0:
        context["detail_router_candidate_top_k"] = 16
        context["detail_router_distance_penalty"] = 4.0
        actions.append(
            "constrain Detail anchors because their mean distance exceeded 12 m"
        )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    decision = {
        "status": "PASS",
        "attempt1_score": score,
        "skip_at_score": args.skip_at_score,
        "run_second_attempt": run_second,
        "actions": actions,
        "output": str(destination),
    }
    target = Path(args.decision)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
