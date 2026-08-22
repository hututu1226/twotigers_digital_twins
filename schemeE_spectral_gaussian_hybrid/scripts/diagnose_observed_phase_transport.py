from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from diagnose_angle_delay_component_oracles import (
    _channel_to_complex_angle_delay,
    _complex_angle_delay_to_channel,
)
from scheme_e.angle_delay import ChannelShape
from scheme_e.carrier_transport import (
    CarrierFit,
    quality_gated_carrier_fit,
    select_transport_candidates,
)
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
    target_informed_expert_oracle,
)
from scheme_e.hybrid_training import _station_positions
from scheme_e.reference import build_reference_candidates


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_bootstrap.PROJECT_ROOT.parent,
        text=True,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe carrier-aligned observed angle-delay phase transfer"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--prediction",
        default="../research/scheme_e_065/FOLD0_QUALITY_GATED_PREDICTION.npy",
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_021_OBSERVED_PHASE_TRANSPORT.json",
    )
    parser.add_argument("--prior-wave-number", type=float, default=-140.33)
    parser.add_argument("--minimum-quality", type=float, default=0.5)
    parser.add_argument("--minimum-direct-gain", type=float, default=0.003)
    parser.add_argument("--minimum-oracle-score", type=float, default=0.655)
    parser.add_argument("--expected-baseline", type=float, default=0.6315811)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()

    config = load_config(args.config)
    device = choose_device(args.device)
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    metadata = _load_npz(artifact_dir / "metadata.npz")
    shape = ChannelShape.from_setup(_load_json(artifact_dir / "manifest.json")["setup"])
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
        mmap_mode="r",
    )
    prediction = np.load(args.prediction, mmap_mode="r")
    fold = int(config["split"]["validation_fold"])
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    available = priors["available"].astype(bool)
    validation = np.flatnonzero(available & validation_mask)
    observed = np.flatnonzero(available & ~validation_mask)
    if len(prediction) != len(validation):
        raise ValueError("Saved prediction does not match strict Fold0 rows")

    checkpoint_path = Path(config["hybrid"]["output_dir"]) / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = checkpoint.get("carrier_fit")
    if payload is None:
        raise RuntimeError("Selected checkpoint has no carrier fit")
    original_fit = CarrierFit(
        np.asarray(payload["wave_numbers"], dtype=np.float64),
        np.asarray(payload["qualities"], dtype=np.float64),
        np.asarray(payload["pair_counts"], dtype=np.int64),
    )
    carrier_fit = quality_gated_carrier_fit(
        original_fit,
        prior_wave_number=float(args.prior_wave_number),
        minimum_quality=float(args.minimum_quality),
    )
    transport = config["hybrid"].get("transport_seed", {})
    count = int(transport.get("count", 8))
    distance_power = float(transport.get("distance_power", 2.0))
    candidates, distances = build_reference_candidates(
        metadata["train_positions"][validation],
        metadata["train_cells"][validation],
        metadata["train_positions"][observed],
        metadata["train_cells"][observed],
        metadata["outage"][observed].astype(bool),
        top_k=max(count, 1),
        target_global_indices=validation,
        observed_global_indices=observed,
    )
    selected_local, selected_distances = select_transport_candidates(
        candidates, distances, count
    )
    selected_global = observed[selected_local]
    stations = _station_positions(metadata)

    stage_names = [
        "baseline",
        "coherent_neighbor_phase",
        *[f"neighbor_phase_{index}" for index in range(count)],
    ]
    stages: dict[str, list] = {name: [] for name in stage_names}
    for start in range(0, len(validation), int(args.batch_size)):
        stop = min(start + int(args.batch_size), len(validation))
        target_indices = validation[start:stop]
        reference_indices = selected_global[start:stop]
        predicted = torch.as_tensor(
            np.array(prediction[start:stop], copy=True), device=device
        )
        target = torch.as_tensor(
            np.array(channels[target_indices], copy=True), device=device
        )
        references = torch.as_tensor(
            np.array(channels[reference_indices], copy=True), device=device
        )
        target_positions = torch.as_tensor(
            metadata["train_positions"][target_indices], device=device
        ).float()
        source_positions = torch.as_tensor(
            metadata["train_positions"][reference_indices], device=device
        ).float()
        cells = torch.as_tensor(
            metadata["train_cells"][target_indices], device=device
        ).long()
        station_positions = torch.as_tensor(stations, device=device).float()
        target_station = station_positions[cells]
        target_range = torch.linalg.vector_norm(target_positions - target_station, dim=1)
        source_range = torch.linalg.vector_norm(
            source_positions - target_station[:, None, :], dim=2
        )
        slopes = torch.as_tensor(
            carrier_fit.wave_numbers, device=device, dtype=torch.float32
        )[cells]
        carrier_phase = slopes[:, None] * (target_range[:, None] - source_range)
        aligned = references * torch.polar(
            torch.ones_like(carrier_phase), carrier_phase
        )[:, :, None, None, None]

        safe_distances = torch.as_tensor(
            selected_distances[start:stop], device=device
        ).float().clamp_min(1e-3)
        weights = safe_distances.pow(-distance_power)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        coherent = torch.sum(aligned * weights[:, :, None, None, None], dim=1)

        predicted_ad = _channel_to_complex_angle_delay(predicted, shape)
        aligned_ad = _channel_to_complex_angle_delay(
            aligned.flatten(0, 1), shape
        ).reshape(len(predicted), count, shape.m_p, shape.n, shape.m_v, shape.m_h, shape.s)
        coherent_ad = _channel_to_complex_angle_delay(coherent, shape)
        predicted_magnitude = predicted_ad.abs()
        neighbor_phase = torch.polar(
            torch.ones_like(aligned_ad.abs()), torch.angle(aligned_ad)
        )
        coherent_phase = torch.polar(
            torch.ones_like(coherent_ad.abs()), torch.angle(coherent_ad)
        )
        neighbor_channels = _complex_angle_delay_to_channel(
            (predicted_magnitude[:, None] * neighbor_phase).flatten(0, 1), shape
        ).reshape(len(predicted), count, shape.m, shape.n, shape.s)
        coherent_channel = _complex_angle_delay_to_channel(
            predicted_magnitude * coherent_phase, shape
        )
        outage = metadata["outage"][target_indices].astype(bool)
        stages["baseline"].append(sample_metric_batch(predicted, target, shape, outage))
        stages["coherent_neighbor_phase"].append(
            sample_metric_batch(coherent_channel, target, shape, outage)
        )
        for index in range(count):
            stages[f"neighbor_phase_{index}"].append(
                sample_metric_batch(
                    neighbor_channels[:, index], target, shape, outage
                )
            )

    arrays = {
        name: concatenate_metric_batches(values) for name, values in stages.items()
    }
    metrics = {
        name: aggregate_sample_metrics(values) for name, values in arrays.items()
    }
    baseline_score = float(metrics["baseline"]["score"])
    baseline_delta = abs(baseline_score - float(args.expected_baseline))
    if baseline_delta > float(args.baseline_tolerance):
        failure = {
            "status": "FAIL_BASELINE_REPRODUCTION",
            "expected": float(args.expected_baseline),
            "observed": metrics["baseline"],
            "absolute_delta": baseline_delta,
        }
        save_json(args.report, failure)
        raise RuntimeError(json.dumps(failure, ensure_ascii=False))
    direct_names = ["coherent_neighbor_phase", "neighbor_phase_0"]
    direct_gains = {
        name: float(metrics[name]["score"]) - baseline_score for name in direct_names
    }
    oracle = target_informed_expert_oracle(arrays)
    oracle_gain = float(oracle["metrics"]["score"]) - baseline_score
    promote = (
        max(direct_gains.values()) >= float(args.minimum_direct_gain)
        or float(oracle["metrics"]["score"]) >= float(args.minimum_oracle_score)
    )
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-021",
        "diagnostic_only_oracle": True,
        "hypothesis": (
            "Carrier-aligned observed neighbors retain transferable pathwise phase "
            "that can support a query-conditioned phase attention model."
        ),
        "git_commit": _git_head(),
        "leakage_control": {
            "neighbors": "strict Fold0-train observations only",
            "carrier_fit": "L0-014 train-only quality gate",
            "fold0_targets": "metrics and target-informed expert oracle only",
        },
        "settings": {
            "neighbors": count,
            "distance_power": distance_power,
            "carrier_fit": carrier_fit.to_dict(),
            "minimum_direct_gain": float(args.minimum_direct_gain),
            "minimum_oracle_score": float(args.minimum_oracle_score),
        },
        "strict_fold0": {
            "baseline": metrics["baseline"],
            "nearest_phase": metrics["neighbor_phase_0"],
            "coherent_phase": metrics["coherent_neighbor_phase"],
            "direct_gains": direct_gains,
            "all_candidate_metrics": metrics,
            "diagnostic_oracle": {
                **{key: value for key, value in oracle.items() if key != "selection"},
                "gain": oracle_gain,
            },
        },
        "decision": "PROMOTE_PHASE_NEURAL_PROBE" if promote else "DROP",
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
