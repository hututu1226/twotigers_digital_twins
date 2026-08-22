from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np

from scheme_e.carrier_transport import CarrierFit, quality_gated_carrier_fit
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.hybrid_training import evaluate_hybrid, load_hybrid_checkpoint


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _reference_strategy(config: dict, summary: dict) -> dict[str, object]:
    selected = str(summary.get("selected_reference_strategy", "nearest"))
    for candidate in config["hybrid"].get("reference_strategies", []):
        if str(candidate.get("name")) == selected:
            return dict(candidate)
    return {"name": "nearest", "top_k": 1}


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_bootstrap.PROJECT_ROOT.parent,
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Fold0 test of a quality-gated carrier fit"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--policy", default="reports/generated/v4_attempt1_policy.json"
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_014_CARRIER_QUALITY_FALLBACK.json",
    )
    parser.add_argument("--prior-wave-number", type=float, default=-140.33)
    parser.add_argument("--minimum-quality", type=float, default=0.5)
    parser.add_argument("--expected-baseline", type=float, default=0.627089141574626)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()

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
    candidate_fit = quality_gated_carrier_fit(
        original_fit,
        prior_wave_number=float(args.prior_wave_number),
        minimum_quality=float(args.minimum_quality),
    )

    policy = _load_json(args.policy)
    outage_policy = {
        "threshold_by_cell": policy["outage_threshold_by_cell"],
        "soft_strength_by_cell": policy["soft_outage_strength_by_cell"],
    }
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
        "transport_config": config["hybrid"].get("transport_seed", {}),
        "output_projection": config.get("inference", {}).get("output_projection", {}),
    }

    baseline = evaluate_hybrid(
        **common,
        target_indices=validation,
        carrier_fit=original_fit,
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

    candidate = evaluate_hybrid(
        **common,
        target_indices=validation,
        carrier_fit=candidate_fit,
    )
    by_cell: dict[str, dict[str, object]] = {}
    validation_cells = metadata["train_cells"][validation].astype(np.int64)
    for cell in sorted(np.unique(validation_cells).tolist()):
        selected = validation[validation_cells == int(cell)]
        by_cell[str(int(cell))] = {
            "baseline": evaluate_hybrid(
                **common,
                target_indices=selected,
                carrier_fit=original_fit,
            ),
            "quality_gated": evaluate_hybrid(
                **common,
                target_indices=selected,
                carrier_fit=candidate_fit,
            ),
        }

    gain = float(candidate["score"]) - float(baseline["score"])
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-014",
        "hypothesis": (
            "A low-coherence per-cell carrier fit should fall back to the stable "
            "Round1 global carrier prior instead of moving the phase seed to an alias."
        ),
        "git_commit": _git_head(),
        "leakage_control": {
            "carrier_fit": "Fold0-train observations stored in the checkpoint",
            "fallback_rule": "fixed before Fold0 evaluation from fit quality and Round1 prior",
            "fold0_targets": "metrics only; never used to select the fallback",
        },
        "samples": {
            "observed": int(len(observed)),
            "validation": int(len(validation)),
        },
        "rule": {
            "prior_wave_number": float(args.prior_wave_number),
            "minimum_quality": float(args.minimum_quality),
            "original_fit": original_fit.to_dict(),
            "quality_gated_fit": candidate_fit.to_dict(),
        },
        "strict_fold0": {
            "baseline": baseline,
            "quality_gated": candidate,
            "gain": gain,
            "by_cell": by_cell,
        },
        "decision": "KEEP" if gain > 5e-4 else "DROP",
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
