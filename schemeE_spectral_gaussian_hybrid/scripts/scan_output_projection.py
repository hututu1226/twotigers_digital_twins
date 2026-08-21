from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import _bootstrap  # noqa: F401
import numpy as np

from scheme_e.carrier_transport import CarrierFit
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.hybrid_training import evaluate_hybrid, load_hybrid_checkpoint


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan cell-specific post-decode PAS/PDP projection"
    )
    parser.add_argument("--config", default="configs/v3_attempt3_decoder.json")
    parser.add_argument(
        "--policy", default="reports/generated/v3_attempt3_policy.json"
    )
    parser.add_argument("--iterations", default="1,2")
    parser.add_argument("--strengths", default="0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--output", default="reports/generated/v3_output_projection_scan.json"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    section = config["hybrid"]
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(config["spectral_teacher"]["oof_output_path"]) as source:
        priors = {name: source[name] for name in source.files}
    with np.load(config["spectral"]["target_path"]) as source:
        spectral_targets = {name: source[name] for name in source.files}

    fold = int(config["split"]["validation_fold"])
    count = min(len(metadata["train_cells"]), len(priors["available"]))
    indices = np.arange(count, dtype=np.int64)
    available = priors["available"][:count].astype(bool)
    validation_mask = metadata["validation_masks"][fold][:count].astype(bool)
    validation = indices[available & validation_mask]
    observed = indices[available & ~validation_mask]
    device = choose_device(str(config["runtime"].get("device", "auto")))
    checkpoint_path = Path(section["output_dir"]) / "best.pt"
    model, shape, checkpoint = load_hybrid_checkpoint(
        config, checkpoint_path, device
    )
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    summary = json.loads(
        (Path(section["output_dir"]) / "summary.json").read_text(encoding="utf-8")
    )
    strategy_name = str(summary.get("selected_reference_strategy", "nearest"))
    reference_strategy = {"name": "nearest", "top_k": 1}
    for candidate in section.get("reference_strategies", []):
        if str(candidate.get("name")) == strategy_name:
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

    common = dict(
        model=model,
        shape=shape,
        channels=channels,
        metadata=metadata,
        priors=priors,
        target_indices=validation,
        observed_indices=observed,
        geometry_mean=np.asarray(checkpoint["geometry_mean"], dtype=np.float32),
        geometry_std=np.asarray(checkpoint["geometry_std"], dtype=np.float32),
        device=device,
        batch_size=int(section.get("validation_batch_size", 4)),
        outage_threshold=float(checkpoint["outage_threshold"]),
        projection_iterations=int(summary["selected_projection_iterations"]),
        spectral_targets=spectral_targets,
        power_bounds=power_bounds,
        reference_strategy=reference_strategy,
        outage_policy=outage_policy,
        carrier_fit=carrier_fit,
        transport_config=section.get("transport_seed", {}),
    )
    baseline = evaluate_hybrid(**common)
    print(f"output projection baseline score={baseline['score']:.6f}", flush=True)
    reports: list[dict[str, object]] = []
    strengths = _float_list(args.strengths)
    for iterations, pair in itertools.product(
        _int_list(args.iterations), itertools.product(strengths, repeat=2)
    ):
        values = evaluate_hybrid(
            **common,
            output_projection={
                "iterations": iterations,
                "strength_by_cell": list(pair),
                "minimum_scale": 0.5,
                "maximum_scale": 2.0,
            },
        )
        reports.append(values)
        print(
            "output projection iter=%d strengths=%s score=%.6f pas=%.6f pdp=%.6f nmse=%.6f"
            % (
                iterations,
                list(pair),
                values["score"],
                values["pas"],
                values["pdp"],
                values["nmse"],
            ),
            flush=True,
        )
    selected = max([baseline, *reports], key=lambda item: float(item["score"]))
    report = {
        "status": "PASS",
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "baseline": baseline,
        "selected": selected,
        "score_gain": float(selected["score"] - baseline["score"]),
        "candidates": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
