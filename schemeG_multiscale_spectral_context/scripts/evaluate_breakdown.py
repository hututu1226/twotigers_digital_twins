from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_g.config import choose_device, load_config, save_json
from scheme_g.context_data import ContextRepository
from scheme_g.context_training import evaluate_context_model, load_context_checkpoint
from scheme_g.data import load_metadata, split_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Scheme G separately for each base station"
    )
    parser.add_argument("--config", default="configs/fold0_best.json")
    parser.add_argument("--checkpoint", default="artifacts/fold0/context/best.pt")
    parser.add_argument("--policy", default="artifacts/fold0/context/outage_scan.json")
    parser.add_argument("--output", default="reports/generated/fold0_breakdown.json")
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = load_metadata(config)
    training, validation = split_indices(metadata, config)
    with np.load(config["encoding"]["output_path"]) as source:
        available = source["available"].astype(bool)
    training = training[available[training]]
    validation = validation[available[validation]]
    repository = ContextRepository(config, training)
    device = choose_device(config["runtime"]["device"])
    model, autoencoder, shape, checkpoint = load_context_checkpoint(
        config, args.checkpoint, repository, device
    )
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    thresholds = policy.get(
        "best_threshold_by_cell",
        [policy["best_threshold"]] * repository.cell_count,
    )
    soft_strengths = policy.get(
        "best_soft_strength_by_cell",
        [policy.get("best_soft_strength", 0.0)] * repository.cell_count,
    )
    prior_alphas = policy.get(
        "best_spectral_prior_alpha_by_cell",
        [policy.get("best_spectral_prior_alpha", 0.0)] * repository.cell_count,
    )
    combined = policy.get("best_cellwise_combined")
    groups: dict[str, dict] = {
        "all": {key: combined[key] for key in ("pas", "pdp", "nmse", "score")}
        if isinstance(combined, dict)
        else {}
    }
    for cell_id in range(repository.cell_count):
        indices = validation[metadata["train_cells"][validation] == cell_id]
        name = f"bs{cell_id}"
        groups[name] = evaluate_context_model(
            model,
            autoencoder,
            repository,
            indices,
            shape,
            device,
            bool(config["runtime"].get("amp", True)),
            float(thresholds[cell_id]),
            int(config["context"].get("validation_decode_batch_size", 8)),
            soft_outage_strength=float(soft_strengths[cell_id]),
            spectral_prior_alpha=float(prior_alphas[cell_id]),
        )
        groups[name]["samples"] = int(len(indices))
        groups[name]["selected_policy"] = {
            "outage_threshold": float(thresholds[cell_id]),
            "soft_outage_strength": float(soft_strengths[cell_id]),
            "spectral_prior_alpha": float(prior_alphas[cell_id]),
        }
    if not groups["all"]:
        nonzero = sum(
            int(groups[f"bs{cell}"]["metric_nonzero_count"])
            for cell in range(repository.cell_count)
        )
        groups["all"] = {
            "pas": sum(
                float(groups[f"bs{cell}"]["pas"])
                * int(groups[f"bs{cell}"]["metric_nonzero_count"])
                for cell in range(repository.cell_count)
            )
            / max(nonzero, 1),
            "pdp": sum(
                float(groups[f"bs{cell}"]["pdp"])
                * int(groups[f"bs{cell}"]["metric_nonzero_count"])
                for cell in range(repository.cell_count)
            )
            / max(nonzero, 1),
            "samples": int(len(validation)),
        }
        numerator = sum(
            float(groups[f"bs{cell}"]["nmse_numerator"])
            for cell in range(repository.cell_count)
        )
        denominator = sum(
            float(groups[f"bs{cell}"]["nmse_denominator"])
            for cell in range(repository.cell_count)
        )
        groups["all"]["nmse"] = numerator / max(denominator, 1e-30)
        groups["all"]["score"] = (
            0.4 * groups["all"]["pas"]
            + 0.4 * groups["all"]["pdp"]
            + 0.2 / (1.0 + groups["all"]["nmse"])
        )
    groups["all"]["samples"] = int(len(validation))
    finite = all(
        math.isfinite(float(values[key]))
        for values in groups.values()
        for key in ("pas", "pdp", "nmse", "score")
    )
    maximum_cell_nmse = max(
        float(values["nmse"])
        for name, values in groups.items()
        if name.startswith("bs")
    )
    report = {
        "status": "PASS" if finite and maximum_cell_nmse <= 10.0 else "FAILED",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "policy": {
            "outage_threshold_by_cell": thresholds,
            "soft_outage_strength_by_cell": soft_strengths,
            "spectral_prior_alpha_by_cell": prior_alphas,
        },
        "groups": groups,
        "maximum_cell_nmse": maximum_cell_nmse,
        "submission_warning": maximum_cell_nmse > 3.0,
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Per-base-station validation is non-finite or NMSE exceeds 10")


if __name__ == "__main__":
    main()
