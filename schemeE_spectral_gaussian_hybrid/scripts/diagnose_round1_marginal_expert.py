from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from audit_scheme_e_065 import (
    _collect_variant,
    _outage_policy,
    _projection_settings,
)
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
from scheme_e.hybrid_training import _transport_batch, load_hybrid_checkpoint
from scheme_e.marginal_projection import alternating_marginal_projection
from scheme_e.metrics import ChannelMetricAccumulator
from scheme_e.power_safety import apply_outage_policy
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


def _metric_subset(
    values: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, float | int]:
    return aggregate_sample_metrics(
        {name: np.asarray(value)[mask] for name, value in values.items()}
    )


@torch.no_grad()
def _collect_marginal_expert(
    *,
    config: dict,
    metadata: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    channels: np.ndarray,
    validation: np.ndarray,
    observed: np.ndarray,
    carrier_fit: CarrierFit,
    policy: dict[str, object],
    device: torch.device,
    output_path: Path,
    neighbors: int,
    distance_power: float,
    iterations: int,
) -> dict[str, object]:
    _, shape, _ = load_hybrid_checkpoint(
        config,
        Path(config["hybrid"]["output_dir"]) / "best.pt",
        device,
    )
    candidates, distances = build_reference_candidates(
        metadata["train_positions"][validation],
        metadata["train_cells"][validation],
        metadata["train_positions"][observed],
        metadata["train_cells"][observed],
        metadata["outage"][observed].astype(bool),
        top_k=int(neighbors),
        target_global_indices=validation,
        observed_global_indices=observed,
    )
    selected, selected_distances = select_transport_candidates(
        candidates, distances, int(neighbors)
    )
    source_globals = observed[selected]
    cells = metadata["train_cells"][validation].astype(np.int64)
    thresholds = np.asarray(policy["threshold_by_cell"], dtype=np.float32)
    strengths = np.asarray(policy["strength_by_cell"], dtype=np.float32)
    target_thresholds = thresholds[np.minimum(cells, len(thresholds) - 1)]
    target_strengths = strengths[np.minimum(cells, len(strengths) - 1)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.complex64,
        shape=(len(validation), *shape.raw_shape),
    )
    batches = []
    legacy = ChannelMetricAccumulator(shape)
    batch_size = int(config["hybrid"].get("validation_batch_size", 4))
    started = time.perf_counter()
    for start in range(0, len(validation), batch_size):
        stop = min(start + batch_size, len(validation))
        indices = validation[start:stop]
        seed, _ = _transport_batch(
            channels,
            metadata,
            indices,
            source_globals[start:stop],
            selected_distances[start:stop],
            carrier_fit,
            device,
            distance_power=float(distance_power),
        )
        projected = alternating_marginal_projection(
            seed,
            torch.as_tensor(
                priors["pas_log"][indices].astype(np.float32), device=device
            ),
            torch.as_tensor(
                priors["pdp_log"][indices].astype(np.float32), device=device
            ),
            torch.as_tensor(
                priors["ue_log_energy"][indices].astype(np.float32), device=device
            ),
            shape,
            iterations=int(iterations),
            proxy_count=int(config["spectral"].get("proxy_count", 24)),
            minimum_scale=0.25,
            maximum_scale=4.0,
        )
        prediction = apply_outage_policy(
            projected,
            torch.as_tensor(
                priors["outage_probability"][indices].astype(np.float32),
                device=device,
            ),
            torch.as_tensor(target_thresholds[start:stop], device=device),
            torch.as_tensor(target_strengths[start:stop], device=device),
        )
        target = torch.as_tensor(np.asarray(channels[indices]), device=device)
        true_outage = torch.as_tensor(metadata["outage"][indices], device=device)
        batch = sample_metric_batch(prediction, target, shape, true_outage)
        batches.append(batch)
        legacy.update(prediction, target, true_outage)
        output[start:stop] = prediction.cpu().numpy().astype(np.complex64)
    output.flush()
    del output
    arrays = concatenate_metric_batches(batches)
    return {
        "arrays": arrays,
        "canonical": aggregate_sample_metrics(arrays),
        "legacy": legacy.compute(),
        "prediction_path": str(output_path),
        "nearest_distance_mean": float(selected_distances[:, 0].mean()),
        "nearest_distance_p90": float(np.quantile(selected_distances[:, 0], 0.9)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Round1-style marginal projection expert diagnostic"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--policy", default="reports/generated/v4_attempt1_policy.json"
    )
    parser.add_argument(
        "--projection-report",
        default="reports/generated/v4_attempt1_output_projection.json",
    )
    parser.add_argument("--prior-wave-number", type=float, default=-140.33)
    parser.add_argument("--minimum-quality", type=float, default=0.5)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--distance-power", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--expected-baseline", type=float, default=0.631581059599534)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--baseline-prediction",
        default="../research/scheme_e_065/FOLD0_QUALITY_GATED_PREDICTION.npy",
    )
    parser.add_argument(
        "--candidate-prediction",
        default="../research/scheme_e_065/FOLD0_ROUND1_MARGINAL_PREDICTION.npy",
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_015_ROUND1_MARGINAL_EXPERT.json",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()

    config = load_config(args.config)
    config["runtime"]["device"] = args.device
    device = choose_device(args.device)
    metadata = _load_npz(Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz")
    targets = _load_npz(config["spectral"]["target_path"])
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
        mmap_mode="r",
    )
    available = priors["available"].astype(bool)
    fold = int(config["split"]["validation_fold"])
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    validation = np.flatnonzero(available & validation_mask)
    observed = np.flatnonzero(available & ~validation_mask)

    checkpoint_path = Path(config["hybrid"]["output_dir"]) / "best.pt"
    _, _, checkpoint = load_hybrid_checkpoint(config, checkpoint_path, device)
    payload = checkpoint["carrier_fit"]
    original_fit = CarrierFit(
        np.asarray(payload["wave_numbers"], dtype=np.float64),
        np.asarray(payload["qualities"], dtype=np.float64),
        np.asarray(payload["pair_counts"], dtype=np.int64),
    )
    gated_fit = quality_gated_carrier_fit(
        original_fit,
        prior_wave_number=float(args.prior_wave_number),
        minimum_quality=float(args.minimum_quality),
    )
    policy = _outage_policy(args.policy)
    output_projection = _projection_settings(config, args.projection_report)

    baseline = _collect_variant(
        name="quality_gated_v4",
        config=config,
        priors=priors,
        metadata=metadata,
        targets=targets,
        channels=channels,
        validation=validation,
        observed=observed,
        checkpoint_path=checkpoint_path,
        policy=policy,
        output_projection=output_projection,
        device=device,
        save_path=Path(args.baseline_prediction),
        compare_path=None,
        include_scale_oracles=False,
        carrier_fit_override=gated_fit,
    )
    baseline_metrics = baseline["canonical"]["final"]
    baseline_delta = abs(
        float(baseline_metrics["score"]) - float(args.expected_baseline)
    )
    if baseline_delta > float(args.baseline_tolerance):
        failure = {
            "status": "FAIL_BASELINE_REPRODUCTION",
            "expected": float(args.expected_baseline),
            "observed": baseline_metrics,
            "absolute_delta": baseline_delta,
        }
        save_json(args.report, failure)
        raise RuntimeError(json.dumps(failure, ensure_ascii=False))

    candidate = _collect_marginal_expert(
        config=config,
        metadata=metadata,
        priors=priors,
        channels=channels,
        validation=validation,
        observed=observed,
        carrier_fit=gated_fit,
        policy=policy,
        device=device,
        output_path=Path(args.candidate_prediction),
        neighbors=int(args.neighbors),
        distance_power=float(args.distance_power),
        iterations=int(args.iterations),
    )
    oracle = target_informed_expert_oracle(
        {
            "quality_gated_v4": baseline["arrays"]["final"],
            "round1_marginal": candidate["arrays"],
        }
    )
    cells = metadata["train_cells"][validation].astype(np.int64)
    by_cell = {
        str(int(cell)): {
            "baseline": _metric_subset(
                baseline["arrays"]["final"], cells == int(cell)
            ),
            "candidate": _metric_subset(candidate["arrays"], cells == int(cell)),
        }
        for cell in sorted(np.unique(cells).tolist())
    }
    candidate_gain = float(candidate["canonical"]["score"]) - float(
        baseline_metrics["score"]
    )
    oracle_gain = float(oracle["metrics"]["score"]) - float(
        baseline_metrics["score"]
    )
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-015",
        "hypothesis": (
            "The Round1 horizontal/vertical marginal projection is structurally "
            "different from V4 joint 2-D PAS generation and can add a complementary expert."
        ),
        "git_commit": _git_head(),
        "leakage_control": {
            "parameters": "fixed from the Round1 submitted pipeline",
            "carrier_rule": "fixed by L0-014 before this experiment",
            "teacher": "strict Fold0 OOF priors",
            "fold0_targets": "metrics and explicitly diagnostic oracle only",
        },
        "settings": {
            "neighbors": int(args.neighbors),
            "distance_power": float(args.distance_power),
            "iterations": int(args.iterations),
            "carrier_fit": gated_fit.to_dict(),
        },
        "strict_fold0": {
            "baseline": baseline_metrics,
            "candidate": candidate["canonical"],
            "candidate_gain": candidate_gain,
            "by_cell": by_cell,
            "diagnostic_oracle": {
                **{key: value for key, value in oracle.items() if key != "selection"},
                "gain": oracle_gain,
            },
        },
        "decision": (
            "PROMOTE_TO_ROUTER"
            if float(oracle["metrics"]["score"]) > 0.66
            else "KEEP_AS_EXPERT"
            if float(oracle["metrics"]["score"]) > 0.65
            else "DROP"
        ),
        "artifacts": {
            "baseline_prediction": args.baseline_prediction,
            "candidate_prediction": args.candidate_prediction,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
