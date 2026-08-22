from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from scheme_e.angle_delay import ChannelShape
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
)


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


def _channel_to_complex_angle_delay(
    channel: torch.Tensor, shape: ChannelShape
) -> torch.Tensor:
    array = channel.reshape(
        -1, shape.m_p, shape.m_v, shape.m_h, shape.n, shape.s
    )
    beam = torch.fft.fft2(array, dim=(2, 3), norm="ortho")
    angle_delay = torch.fft.ifft(beam, dim=-1, norm="ortho")
    return angle_delay.permute(0, 1, 4, 2, 3, 5).contiguous()


def _complex_angle_delay_to_channel(
    angle_delay: torch.Tensor, shape: ChannelShape
) -> torch.Tensor:
    frequency_beam = torch.fft.fft(
        angle_delay.permute(0, 1, 3, 4, 2, 5).contiguous(),
        dim=-1,
        norm="ortho",
    )
    array = torch.fft.ifft2(frequency_beam, dim=(2, 3), norm="ortho")
    return array.reshape(-1, shape.m, shape.n, shape.s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Angle-delay magnitude and phase component oracles"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--prediction",
        default="../research/scheme_e_065/FOLD0_QUALITY_GATED_PREDICTION.npy",
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_020_ANGLE_DELAY_COMPONENT_ORACLES.json",
    )
    parser.add_argument("--expected-baseline", type=float, default=0.6315811)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4)
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
    validation = np.flatnonzero(
        priors["available"].astype(bool)
        & metadata["validation_masks"][fold].astype(bool)
    )
    if len(prediction) != len(validation):
        raise ValueError("Saved prediction does not match strict Fold0 rows")

    stages: dict[str, list] = {
        "baseline": [],
        "oracle_target_phase": [],
        "oracle_target_magnitude": [],
    }
    for start in range(0, len(validation), int(args.batch_size)):
        stop = min(start + int(args.batch_size), len(validation))
        selected = validation[start:stop]
        predicted = torch.as_tensor(
            np.array(prediction[start:stop], copy=True), device=device
        )
        target = torch.as_tensor(np.array(channels[selected], copy=True), device=device)
        predicted_ad = _channel_to_complex_angle_delay(predicted, shape)
        target_ad = _channel_to_complex_angle_delay(target, shape)
        target_phase = torch.polar(torch.ones_like(target_ad.abs()), torch.angle(target_ad))
        predicted_phase = torch.polar(
            torch.ones_like(predicted_ad.abs()), torch.angle(predicted_ad)
        )
        target_phase_channel = _complex_angle_delay_to_channel(
            predicted_ad.abs() * target_phase, shape
        )
        target_magnitude_channel = _complex_angle_delay_to_channel(
            target_ad.abs() * predicted_phase, shape
        )
        outage = metadata["outage"][selected].astype(bool)
        for name, value in {
            "baseline": predicted,
            "oracle_target_phase": target_phase_channel,
            "oracle_target_magnitude": target_magnitude_channel,
        }.items():
            stages[name].append(sample_metric_batch(value, target, shape, outage))

    arrays = {
        name: concatenate_metric_batches(values) for name, values in stages.items()
    }
    metrics = {
        name: aggregate_sample_metrics(values) for name, values in arrays.items()
    }
    baseline_delta = abs(
        float(metrics["baseline"]["score"]) - float(args.expected_baseline)
    )
    if baseline_delta > float(args.baseline_tolerance):
        failure = {
            "status": "FAIL_BASELINE_REPRODUCTION",
            "expected": float(args.expected_baseline),
            "observed": metrics["baseline"],
            "absolute_delta": baseline_delta,
        }
        save_json(args.report, failure)
        raise RuntimeError(json.dumps(failure, ensure_ascii=False))

    baseline_score = float(metrics["baseline"]["score"])
    gains = {
        name: float(value["score"]) - baseline_score
        for name, value in metrics.items()
        if name != "baseline"
    }
    best_name = max(gains, key=lambda name: float(metrics[name]["score"]))
    cells = metadata["train_cells"][validation].astype(np.int64)
    by_cell = {
        str(int(cell)): {
            name: aggregate_sample_metrics(
                {
                    field: np.asarray(value)[cells == int(cell)]
                    for field, value in arrays[name].items()
                }
            )
            for name in arrays
        }
        for cell in sorted(np.unique(cells).tolist())
    }
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-020",
        "diagnostic_only": True,
        "hypothesis": (
            "Separating angle-delay magnitude from per-bin phase identifies the "
            "component with enough remaining ceiling for the next neural probe."
        ),
        "git_commit": _git_head(),
        "leakage_control": {
            "prediction": "saved strict Fold0 output",
            "fold0_targets": "component-swap oracle and metrics only; not deployable",
        },
        "strict_fold0": {
            "metrics": metrics,
            "gains_vs_baseline": gains,
            "best_oracle": best_name,
            "by_cell": by_cell,
        },
        "decision": (
            "PROMOTE_COMPONENT_PROBE"
            if float(metrics[best_name]["score"]) >= 0.67
            else "DROP"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
