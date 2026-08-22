from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import _bootstrap  # noqa: F401
import numpy as np

from diagnose_carrier_quality_fallback import (
    _git_head,
    _load_json,
    _load_npz,
    _reference_strategy,
)
from scheme_e.carrier_transport import CarrierFit, quality_gated_carrier_fit
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.hybrid_training import evaluate_hybrid, load_hybrid_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether low carrier-fit quality should use one transport neighbor"
        )
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--policy", default="reports/generated/v4_attempt1_policy.json"
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_018_TRANSPORT_COUNT_GATE.json",
    )
    parser.add_argument("--prior-wave-number", type=float, default=-140.33)
    parser.add_argument("--minimum-quality", type=float, default=0.5)
    parser.add_argument("--low-quality-count", type=int, default=1)
    parser.add_argument("--minimum-cell-gain", type=float, default=0.003)
    parser.add_argument("--expected-baseline", type=float, default=0.6315811)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()

    if int(args.low_quality_count) != 1:
        raise ValueError("This fixed probe only permits low-quality-count=1")

    config = load_config(args.config)
    config["runtime"]["device"] = args.device
    device = choose_device(args.device)
    metadata = _load_npz(Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz")
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    spectral_targets = _load_npz(config["spectral"]["target_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
        mmap_mode="r",
    )

    fold = int(config["split"]["validation_fold"])
    available = priors["available"].astype(bool)
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    validation = np.flatnonzero(available & validation_mask)
    observed = np.flatnonzero(available & ~validation_mask)
    if not len(validation) or not len(observed):
        raise RuntimeError("Strict Fold0 split is empty")

    checkpoint_path = Path(config["hybrid"]["output_dir"]) / "best.pt"
    model, shape, checkpoint = load_hybrid_checkpoint(config, checkpoint_path, device)
    summary = _load_json(checkpoint_path.parent / "summary.json")
    carrier_payload = checkpoint.get("carrier_fit")
    if carrier_payload is None:
        raise RuntimeError("The selected checkpoint does not contain a carrier fit")
    original_fit = CarrierFit(
        np.asarray(carrier_payload["wave_numbers"], dtype=np.float64),
        np.asarray(carrier_payload["qualities"], dtype=np.float64),
        np.asarray(carrier_payload["pair_counts"], dtype=np.int64),
    )
    gated_fit = quality_gated_carrier_fit(
        original_fit,
        prior_wave_number=float(args.prior_wave_number),
        minimum_quality=float(args.minimum_quality),
    )
    low_quality_cells = np.flatnonzero(
        np.asarray(original_fit.qualities) < float(args.minimum_quality)
    ).astype(np.int64)
    if not len(low_quality_cells):
        raise RuntimeError("No low-quality cells were found for the fixed probe")

    policy = _load_json(args.policy)
    outage_policy = {
        "threshold_by_cell": policy["outage_threshold_by_cell"],
        "soft_strength_by_cell": policy["soft_outage_strength_by_cell"],
    }
    base_transport = dict(config["hybrid"].get("transport_seed", {}))
    probe_transport = dict(base_transport)
    probe_transport["count"] = int(args.low_quality_count)
    common = {
        "model": model,
        "shape": shape,
        "channels": channels,
        "metadata": metadata,
        "priors": priors,
        "observed_indices": observed,
        "geometry_mean": np.asarray(checkpoint["geometry_mean"], dtype=np.float32),
        "geometry_std": np.asarray(checkpoint["geometry_std"], dtype=np.float32),
        "device": device,
        "batch_size": int(config["hybrid"].get("validation_batch_size", 4)),
        "outage_threshold": float(checkpoint["outage_threshold"]),
        "projection_iterations": int(summary["selected_projection_iterations"]),
        "spectral_targets": spectral_targets,
        "power_bounds": (
            None
            if checkpoint.get("power_bounds") is None
            else np.asarray(checkpoint["power_bounds"], dtype=np.float32)
        ),
        "reference_strategy": _reference_strategy(config, summary),
        "outage_policy": outage_policy,
        "carrier_fit": gated_fit,
        "output_projection": config.get("inference", {}).get(
            "output_projection", {}
        ),
    }

    baseline = evaluate_hybrid(
        **common,
        target_indices=validation,
        transport_config=base_transport,
    )
    baseline_delta = abs(float(baseline["score"]) - float(args.expected_baseline))
    if baseline_delta > float(args.baseline_tolerance):
        failure = {
            "status": "FAIL_BASELINE_REPRODUCTION",
            "expected_baseline": float(args.expected_baseline),
            "observed_baseline": baseline,
            "absolute_delta": baseline_delta,
            "tolerance": float(args.baseline_tolerance),
        }
        save_json(args.report, failure)
        raise RuntimeError(json.dumps(failure, ensure_ascii=False))

    validation_cells = metadata["train_cells"][validation].astype(np.int64)
    cell_results: dict[str, dict[str, object]] = {}
    gains = []
    for cell in low_quality_cells.tolist():
        selected = validation[validation_cells == int(cell)]
        if not len(selected):
            continue
        current = evaluate_hybrid(
            **common,
            target_indices=selected,
            transport_config=base_transport,
        )
        single = evaluate_hybrid(
            **common,
            target_indices=selected,
            transport_config=probe_transport,
        )
        gain = float(single["score"]) - float(current["score"])
        gains.append(gain)
        cell_results[str(int(cell))] = {
            "current": current,
            "single_neighbor": single,
            "gain": gain,
        }

    promote = bool(gains) and min(gains) >= float(args.minimum_cell_gain)
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-018",
        "hypothesis": (
            "Low carrier-fit quality makes a multi-neighbor complex transport seed "
            "cancel, so the affected cell should use one transport neighbor."
        ),
        "git_commit": _git_head(),
        "leakage_control": {
            "quality_gate": "fixed by L0-014 from Fold0-train fit quality",
            "probe_count": "fixed at one before this experiment; no count scan",
            "fold0_targets": "metrics only",
        },
        "settings": {
            "current_transport_count": int(base_transport.get("count", 8)),
            "probe_transport_count": int(args.low_quality_count),
            "minimum_quality": float(args.minimum_quality),
            "minimum_cell_gain": float(args.minimum_cell_gain),
            "low_quality_cells": low_quality_cells.tolist(),
            "carrier_fit": gated_fit.to_dict(),
        },
        "strict_fold0": {
            "authoritative_baseline": baseline,
            "low_quality_cell_probe": cell_results,
            "mixed_candidate_not_yet_materialized": True,
        },
        "decision": "PROMOTE" if promote else "DROP",
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
