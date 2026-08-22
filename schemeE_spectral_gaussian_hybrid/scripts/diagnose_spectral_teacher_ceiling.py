from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import _bootstrap  # noqa: F401
import numpy as np

from scheme_e.carrier_transport import CarrierFit
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.hybrid_training import evaluate_hybrid, load_hybrid_checkpoint


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: source[name] for name in source.files}


def _oracle_priors(
    priors: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    spectral: bool,
    power: bool,
) -> dict[str, np.ndarray]:
    result = {name: np.array(value, copy=True) for name, value in priors.items()}
    if spectral:
        result["pas_log"] = targets["pas_log"].astype(result["pas_log"].dtype)
        result["pdp_log"] = targets["pdp_log"].astype(result["pdp_log"].dtype)
    if power:
        result["ue_log_energy"] = targets["ue_log_energy"].astype(
            result["ue_log_energy"].dtype
        )
        result["log_power"] = targets["log_power"].astype(result["log_power"].dtype)
        result["outage_probability"] = targets["outage"].astype(
            result["outage_probability"].dtype
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the perfect coarse spectral-teacher ceiling on Fold0"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--policy", default="reports/generated/v4_attempt1_policy.json"
    )
    parser.add_argument(
        "--output", default="reports/generated/v4_spectral_teacher_ceiling.json"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    section = config["hybrid"]
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    metadata = _load_npz(artifact_dir / "metadata.npz")
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    targets = _load_npz(config["spectral"]["target_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )

    fold = int(config["split"]["validation_fold"])
    count = min(len(metadata["train_cells"]), len(priors["available"]))
    indices = np.arange(count, dtype=np.int64)
    available = priors["available"][:count].astype(bool)
    validation_mask = metadata["validation_masks"][fold][:count].astype(bool)
    validation = indices[available & validation_mask]
    observed = indices[available & ~validation_mask]
    device = choose_device(str(config["runtime"].get("device", "auto")))
    checkpoint_path = Path(section["output_dir"]) / "best.pt"
    model, shape, checkpoint = load_hybrid_checkpoint(config, checkpoint_path, device)
    summary = json.loads(
        (Path(section["output_dir"]) / "summary.json").read_text(encoding="utf-8")
    )
    reference_strategy = {"name": "nearest", "top_k": 1}
    selected_name = str(summary.get("selected_reference_strategy", "nearest"))
    for candidate in section.get("reference_strategies", []):
        if str(candidate.get("name")) == selected_name:
            reference_strategy = dict(candidate)
            break
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    outage_policy = {
        "threshold_by_cell": policy["outage_threshold_by_cell"],
        "soft_strength_by_cell": policy["soft_outage_strength_by_cell"],
    }
    carrier_payload = checkpoint.get("carrier_fit")
    carrier_fit = None
    if carrier_payload is not None:
        carrier_fit = CarrierFit(
            np.asarray(carrier_payload["wave_numbers"], dtype=np.float64),
            np.asarray(carrier_payload["qualities"], dtype=np.float64),
            np.asarray(carrier_payload["pair_counts"], dtype=np.int64),
        )
    power_bounds = checkpoint.get("power_bounds")
    if power_bounds is not None:
        power_bounds = np.asarray(power_bounds, dtype=np.float32)

    selected_projection = deepcopy(config.get("inference", {}).get("output_projection", {}))
    if not selected_projection:
        selected_projection = {
            "iterations": 0,
            "strength_by_cell": [0.0, 0.0],
        }
    common = dict(
        model=model,
        shape=shape,
        channels=channels,
        metadata=metadata,
        target_indices=validation,
        observed_indices=observed,
        geometry_mean=np.asarray(checkpoint["geometry_mean"], dtype=np.float32),
        geometry_std=np.asarray(checkpoint["geometry_std"], dtype=np.float32),
        device=device,
        batch_size=int(section.get("validation_batch_size", 4)),
        outage_threshold=float(checkpoint["outage_threshold"]),
        projection_iterations=int(summary["selected_projection_iterations"]),
        spectral_targets=targets,
        reference_strategy=reference_strategy,
        outage_policy=outage_policy,
        carrier_fit=carrier_fit,
        transport_config=section.get("transport_seed", {}),
    )

    scenarios = {
        "learned_teacher": {
            "priors": priors,
            "power_bounds": power_bounds,
            "output_projection": selected_projection,
        },
        "oracle_coarse_spectral_only": {
            "priors": _oracle_priors(priors, targets, spectral=True, power=False),
            "power_bounds": power_bounds,
            "output_projection": selected_projection,
        },
        "oracle_power_and_outage_only": {
            "priors": _oracle_priors(priors, targets, spectral=False, power=True),
            "power_bounds": None,
            "output_projection": {
                **selected_projection,
                "power_source": "input",
            },
        },
        "oracle_all_coarse_selected_projection": {
            "priors": _oracle_priors(priors, targets, spectral=True, power=True),
            "power_bounds": None,
            "output_projection": {
                **selected_projection,
                "power_source": "input",
            },
        },
        "oracle_all_coarse_full_projection": {
            "priors": _oracle_priors(priors, targets, spectral=True, power=True),
            "power_bounds": None,
            "output_projection": {
                "iterations": 4,
                "strength_by_cell": [1.0, 1.0],
                "minimum_scale": 0.25,
                "maximum_scale": 4.0,
                "power_source": "input",
            },
        },
    }
    reports: dict[str, dict[str, float | int]] = {}
    for name, values in scenarios.items():
        metrics = evaluate_hybrid(**common, **values)
        reports[name] = metrics
        print(
            "%s score=%.6f pas=%.6f pdp=%.6f nmse=%.6f"
            % (name, metrics["score"], metrics["pas"], metrics["pdp"], metrics["nmse"]),
            flush=True,
        )

    learned = float(reports["learned_teacher"]["score"])
    coarse = float(reports["oracle_all_coarse_full_projection"]["score"])
    report = {
        "status": "PASS",
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "samples": int(len(validation)),
        "scenarios": reports,
        "oracle_coarse_gain": coarse - learned,
        "coarse_teacher_can_cross_065": bool(coarse >= 0.65),
        "interpretation": (
            "teacher_prediction_is_the_primary_bottleneck"
            if coarse >= 0.65
            else "coarse_targets_or_downstream_representation_are_the_primary_bottleneck"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
