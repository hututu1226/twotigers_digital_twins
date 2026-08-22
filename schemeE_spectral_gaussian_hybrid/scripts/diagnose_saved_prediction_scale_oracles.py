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
    scale_oracle_predictions,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical scale oracles for a saved strict Fold0 prediction"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--prediction",
        default="../research/scheme_e_065/FOLD0_QUALITY_GATED_PREDICTION.npy",
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_019_QUALITY_GATED_SCALE_ORACLES.json",
    )
    parser.add_argument("--expected-baseline", type=float, default=0.6315811)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-4)
    parser.add_argument("--target-score", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()

    config = load_config(args.config)
    device = choose_device(args.device)
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    metadata = _load_npz(artifact_dir / "metadata.npz")
    manifest = _load_json(artifact_dir / "manifest.json")
    shape = ChannelShape.from_setup(manifest["setup"])
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
        mmap_mode="r",
    )
    prediction = np.load(args.prediction, mmap_mode="r")

    fold = int(config["split"]["validation_fold"])
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    validation = np.flatnonzero(priors["available"].astype(bool) & validation_mask)
    if len(prediction) != len(validation):
        raise ValueError(
            f"Prediction rows {len(prediction)} do not match Fold0 rows {len(validation)}"
        )

    batches: dict[str, list] = {
        "baseline": [],
        "oracle_real_scale": [],
        "oracle_complex_scale": [],
        "oracle_power_scale": [],
    }
    for start in range(0, len(validation), int(args.batch_size)):
        stop = min(start + int(args.batch_size), len(validation))
        selected = validation[start:stop]
        predicted = torch.as_tensor(
            np.array(prediction[start:stop], copy=True), device=device
        )
        target = torch.as_tensor(np.array(channels[selected], copy=True), device=device)
        outage = metadata["outage"][selected].astype(bool)
        batches["baseline"].append(
            sample_metric_batch(predicted, target, shape, outage)
        )
        for name, value in scale_oracle_predictions(predicted, target).items():
            batches[name].append(sample_metric_batch(value, target, shape, outage))

    arrays = {
        name: concatenate_metric_batches(values) for name, values in batches.items()
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
    best_score = float(metrics[best_name]["score"])
    cells = metadata["train_cells"][validation].astype(np.int64)
    by_cell = {
        str(int(cell)): {
            name: aggregate_sample_metrics(
                {field: np.asarray(value)[cells == int(cell)] for field, value in arrays[name].items()}
            )
            for name in arrays
        }
        for cell in sorted(np.unique(cells).tolist())
    }
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-019",
        "diagnostic_only": True,
        "hypothesis": (
            "The quality-gated baseline may still lose enough score to per-sample "
            "amplitude or complex scale to justify an OOF scale model."
        ),
        "git_commit": _git_head(),
        "leakage_control": {
            "prediction": "saved strict Fold0 output",
            "fold0_targets": "oracle scales and metrics only; not deployable",
        },
        "strict_fold0": {
            "metrics": metrics,
            "gains_vs_baseline": gains,
            "best_oracle": best_name,
            "best_oracle_score": best_score,
            "by_cell": by_cell,
        },
        "decision": "PROMOTE" if best_score >= float(args.target_score) else "DROP",
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
