from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from scheme_e.angle_delay import channel_to_shape_target, shape_to_channel
from scheme_e.complex_residual import (
    angle_delay_log_power,
    replace_angle_delay_log_power,
)
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
    target_informed_expert_oracle,
)
from scheme_e.hybrid_training import load_hybrid_checkpoint
from scheme_e.local_magnitude import (
    estimate_magnitude_profile_shifts,
    same_cell_neighbors,
    transfer_log_power_residual,
)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


@torch.no_grad()
def _evaluate_saved_prediction(
    path: str | Path,
    validation: np.ndarray,
    metadata: dict[str, np.ndarray],
    channels: np.ndarray,
    shape: object,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    prediction = np.load(path, mmap_mode="r")
    if len(prediction) != len(validation):
        raise ValueError("Saved Fold0 prediction row count is inconsistent")
    parts = []
    for start in range(0, len(validation), int(batch_size)):
        stop = min(start + int(batch_size), len(validation))
        selected = validation[start:stop]
        parts.append(
            sample_metric_batch(
                torch.as_tensor(
                    np.array(prediction[start:stop], copy=True), device=device
                ),
                torch.as_tensor(
                    np.array(channels[selected], copy=True), device=device
                ),
                shape,
                torch.as_tensor(
                    metadata["outage"][selected].astype(bool), device=device
                ),
            )
        )
    arrays = concatenate_metric_batches(parts)
    return aggregate_sample_metrics(arrays), arrays


def _by_cell(
    arrays: dict[str, np.ndarray],
    validation_cells: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    return {
        str(int(cell)): aggregate_sample_metrics(
            {
                name: values[validation_cells == cell]
                for name, values in arrays.items()
            }
        )
        for cell in np.unique(validation_cells)
    }


@torch.no_grad()
def _evaluate_composition(
    validation: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    base_cache: np.ndarray,
    target_cache: np.ndarray,
    baseline_prediction: np.ndarray,
    channels: np.ndarray,
    metadata: dict[str, np.ndarray],
    shape: object,
    device: torch.device,
    args: argparse.Namespace,
    *,
    strength: float,
    output_path: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, np.ndarray], dict[str, float]]:
    output = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.complex64,
            shape=(len(validation), *shape.raw_shape),
        )
    parts = []
    absolute_correction = 0.0
    correction_elements = 0
    for start in range(0, len(validation), int(args.batch_size)):
        stop = min(start + int(args.batch_size), len(validation))
        selected = validation[start:stop]
        local_neighbors = neighbors[start:stop]
        query_base = np.array(base_cache[selected], copy=True, dtype=np.float32)
        neighbor_base = np.array(
            base_cache[local_neighbors], copy=True, dtype=np.float32
        )
        neighbor_target = np.array(
            target_cache[local_neighbors], copy=True, dtype=np.float32
        )
        shifts = (
            estimate_magnitude_profile_shifts(
                query_base,
                neighbor_base,
                scale=float(args.log_power_scale),
            )
            if float(strength) != 0.0
            else None
        )
        transferred = transfer_log_power_residual(
            query_base,
            neighbor_base,
            neighbor_target,
            distances[start:stop],
            count=int(args.count),
            strength=float(strength),
            shifts=shifts,
        )
        correction = transferred - query_base
        absolute_correction += float(np.abs(correction).sum(dtype=np.float64))
        correction_elements += correction.size

        baseline_channel = torch.as_tensor(
            np.array(baseline_prediction[start:stop], copy=True), device=device
        )
        baseline_shape, global_log_power, baseline_outage = channel_to_shape_target(
            baseline_channel, shape
        )
        baseline_magnitude = angle_delay_log_power(
            baseline_shape, shape, float(args.log_power_scale)
        )
        candidate_magnitude = (
            baseline_magnitude
            + torch.as_tensor(correction, device=device).reshape_as(baseline_magnitude)
        ).clamp(0.0, 20.0)
        candidate_shape = replace_angle_delay_log_power(
            baseline_shape,
            candidate_magnitude,
            shape,
            float(args.log_power_scale),
        )
        prediction = shape_to_channel(
            candidate_shape,
            global_log_power,
            shape,
            baseline_outage,
        )
        target = torch.as_tensor(
            np.array(channels[selected], copy=True), device=device
        )
        true_outage = torch.as_tensor(
            metadata["outage"][selected].astype(bool), device=device
        )
        parts.append(sample_metric_batch(prediction, target, shape, true_outage))
        if output is not None:
            output[start:stop] = prediction.cpu().numpy().astype(np.complex64)
        if start == 0 or stop == len(validation) or stop % 200 == 0:
            print(f"quality-gated magnitude composition {stop}/{len(validation)}", flush=True)
    if output is not None:
        output.flush()
        del output
    arrays = concatenate_metric_batches(parts)
    return (
        aggregate_sample_metrics(arrays),
        arrays,
        {
            "mean_absolute_log_power_correction": absolute_correction
            / max(correction_elements, 1),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose the fixed aligned local magnitude residual with the "
            "quality-gated V4 phase without a parameter scan"
        )
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--map-cache",
        default="artifacts/scheme_e_065/fullres_log_power_cache",
    )
    parser.add_argument(
        "--baseline-prediction",
        default="../research/scheme_e_065/FOLD0_QUALITY_GATED_PREDICTION.npy",
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--log-power-scale", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--expected-baseline", type=float, default=0.631581)
    parser.add_argument("--baseline-tolerance", type=float, default=0.0001)
    parser.add_argument("--minimum-direct-gain", type=float, default=0.003)
    parser.add_argument("--minimum-oracle-score", type=float, default=0.660)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        default="artifacts/scheme_e_065/l0_022_quality_gated_magnitude",
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_022_QUALITY_GATED_MAGNITUDE.json",
    )
    args = parser.parse_args()
    if int(args.count) < 1:
        raise ValueError("count must be positive")
    if not 0.0 <= float(args.strength) <= 1.0:
        raise ValueError("strength must be in [0,1]")

    started = time.perf_counter()
    device = choose_device(args.device)
    config = load_config(args.config)
    metadata = _load_npz(
        Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz"
    )
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
        mmap_mode="r",
    )
    checkpoint_path = Path(config["hybrid"]["output_dir"]) / "best.pt"
    _, shape, _ = load_hybrid_checkpoint(config, checkpoint_path, device)
    base_cache = np.load(
        Path(args.map_cache) / "teacher_log_power.npy", mmap_mode="r"
    )
    target_cache = np.load(
        Path(args.map_cache) / "target_log_power.npy", mmap_mode="r"
    )
    baseline_prediction = np.load(args.baseline_prediction, mmap_mode="r")

    fold = int(config["split"]["validation_fold"])
    available = priors["available"].astype(bool)
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    observed = np.flatnonzero(available & ~validation_mask)
    validation = np.flatnonzero(available & validation_mask)
    support = observed[~metadata["outage"][observed].astype(bool)]
    if len(baseline_prediction) != len(validation):
        raise ValueError("Quality-gated prediction does not match strict Fold0")
    neighbors, distances = same_cell_neighbors(
        metadata["train_positions"],
        metadata["train_cells"],
        support,
        validation,
        int(args.count),
    )

    baseline_metrics, baseline_arrays = _evaluate_saved_prediction(
        args.baseline_prediction,
        validation,
        metadata,
        channels,
        shape,
        device,
        int(args.batch_size),
    )
    baseline_error = abs(
        float(baseline_metrics["score"]) - float(args.expected_baseline)
    )
    if baseline_error > float(args.baseline_tolerance):
        raise RuntimeError(
            "Quality-gated baseline reproduction failed: "
            f"observed={float(baseline_metrics['score']):.6f}"
        )
    identity_metrics, _, _ = _evaluate_composition(
        validation,
        neighbors,
        distances,
        base_cache,
        target_cache,
        baseline_prediction,
        channels,
        metadata,
        shape,
        device,
        args,
        strength=0.0,
    )
    identity_error = abs(
        float(identity_metrics["score"]) - float(baseline_metrics["score"])
    )
    if identity_error > 2e-5:
        raise RuntimeError(
            "Zero magnitude correction does not reproduce the deployment baseline: "
            f"absolute score error={identity_error:.8f}"
        )

    candidate_metrics, candidate_arrays, diagnostics = _evaluate_composition(
        validation,
        neighbors,
        distances,
        base_cache,
        target_cache,
        baseline_prediction,
        channels,
        metadata,
        shape,
        device,
        args,
        strength=float(args.strength),
    )
    oracle = target_informed_expert_oracle(
        {"quality_gated_v4": baseline_arrays, "magnitude_composition": candidate_arrays}
    )
    oracle.pop("selection", None)
    gain = float(candidate_metrics["score"]) - float(baseline_metrics["score"])
    oracle_score = float(oracle["metrics"]["score"])
    if gain >= float(args.minimum_direct_gain):
        decision = "PROMOTE"
    elif oracle_score >= float(args.minimum_oracle_score):
        decision = "PROMOTE_GATE_PROBE"
    else:
        decision = "DROP"

    output_path = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if gain > 0.0 or oracle_score >= float(args.minimum_oracle_score):
        output_path = output_dir / "Fold0_Quality_Gated_Magnitude.npy"
        _evaluate_composition(
            validation,
            neighbors,
            distances,
            base_cache,
            target_cache,
            baseline_prediction,
            channels,
            metadata,
            shape,
            device,
            args,
            strength=float(args.strength),
            output_path=output_path,
        )
    validation_cells = metadata["train_cells"][validation].astype(np.int64)
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-022",
        "diagnostic_only_oracle": True,
        "hypothesis": (
            "The fixed aligned local magnitude residual becomes useful when added "
            "to quality-gated V4 magnitude while preserving V4 carrier phase."
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_bootstrap.PROJECT_ROOT.parent,
            text=True,
        ).strip(),
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "leakage_control": {
            "count_and_strength": "fixed by the earlier Fold0-train inner split",
            "neighbors_and_residuals": "Fold0-train observations only",
            "alignment": "query/neighbor OOF Teacher magnitude only",
            "fold0_targets": "canonical evaluation and diagnostic oracle only",
        },
        "settings": {
            "count": int(args.count),
            "strength": float(args.strength),
            "aligned": True,
            "log_power_scale": float(args.log_power_scale),
        },
        "samples": {
            "support": int(len(support)),
            "validation": int(len(validation)),
        },
        "distance_summary": {
            "nearest_mean": float(distances[:, 0].mean()),
            "nearest_p90": float(np.quantile(distances[:, 0], 0.9)),
        },
        "strict_fold0": {
            "quality_gated_v4": baseline_metrics,
            "identity_reproduction": identity_metrics,
            "identity_score_error": identity_error,
            "magnitude_composition": candidate_metrics,
            "gain": gain,
            "diagnostics": diagnostics,
            "by_cell": {
                "quality_gated_v4": _by_cell(baseline_arrays, validation_cells),
                "magnitude_composition": _by_cell(candidate_arrays, validation_cells),
            },
            "target_informed_two_expert_oracle": oracle,
            "target_informed_oracle_score": oracle_score,
            "prediction": None if output_path is None else str(output_path),
        },
        "promotion_rule": {
            "minimum_direct_gain": float(args.minimum_direct_gain),
            "minimum_oracle_score": float(args.minimum_oracle_score),
        },
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    save_json(report_path, report)
    report_path.with_suffix(".md").write_text(
        "# L0-022 Quality-Gated Magnitude Composition\n\n"
        "Fold0 is offline validation, not the official online score.\n\n"
        f"- Quality-gated V4: `{float(baseline_metrics['score']):.6f}`\n"
        f"- Fixed magnitude composition: `{float(candidate_metrics['score']):.6f}`\n"
        f"- Direct gain: `{gain:+.6f}`\n"
        f"- Diagnostic two-expert oracle: `{oracle_score:.6f}`\n"
        f"- Decision: `{decision}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
