from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from scheme_e.angle_delay import channel_to_shape_target, shape_to_channel
from scheme_e.complex_residual import (
    angle_delay_log_power,
    decode_low_rank_coefficients,
    project_low_rank_coefficients,
    replace_angle_delay_log_power,
)
from scheme_e.config import choose_device, load_config, save_json, seed_everything
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
    target_informed_expert_oracle,
)
from scheme_e.gp import SharedMultiOutputGP
from scheme_e.hybrid_training import load_hybrid_checkpoint


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _load_cache(path: Path, prefix: str) -> dict[str, np.ndarray]:
    return {
        name: np.load(path / f"{prefix}_{name}.npy", mmap_mode="r")
        for name in ("spectrum", "detail", "log_power", "outage")
    }


def _gp_features(
    metadata: dict[str, np.ndarray], priors: dict[str, np.ndarray]
) -> np.ndarray:
    relative_ue = (
        priors["ue_log_energy"].astype(np.float32)
        - priors["log_power"].astype(np.float32)[:, None]
    )
    values = np.concatenate(
        [
            metadata["train_geometry_features"].astype(np.float32),
            relative_ue,
            priors["log_power"].astype(np.float32)[:, None],
            priors["uncertainty"].astype(np.float32)[:, None],
            priors["outage_probability"].astype(np.float32)[:, None],
        ],
        axis=1,
    )
    return np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0).astype(
        np.float32
    )


@torch.no_grad()
def _decode_seed_shape(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    autoencoder: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    spectrum = torch.as_tensor(
        np.asarray(cache["spectrum"][indices], dtype=np.float32), device=device
    )
    detail = torch.as_tensor(
        np.asarray(cache["detail"][indices], dtype=np.float32), device=device
    )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        decoded = autoencoder.decode(spectrum, detail)
    return decoded.float()


@torch.no_grad()
def _residual_matrix(
    channels: np.ndarray,
    indices: np.ndarray,
    teacher_cache: dict[str, np.ndarray],
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    batch_size: int,
    log_power_scale: float,
) -> torch.Tensor:
    dimensions = int(shape.m_p * shape.n * shape.m_v * shape.m_h * shape.s)
    output = torch.empty(
        (len(indices), dimensions), dtype=torch.float32, device=device
    )
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        selected = indices[start:stop]
        target_channel = torch.as_tensor(
            np.array(channels[selected], copy=True), device=device
        )
        target_shape, _, _ = channel_to_shape_target(target_channel, shape)
        seed_shape = _decode_seed_shape(
            teacher_cache, selected, autoencoder, device
        )
        target_log = angle_delay_log_power(
            target_shape, shape, float(log_power_scale)
        )
        seed_log = angle_delay_log_power(seed_shape, shape, float(log_power_scale))
        output[start:stop] = (target_log - seed_log).flatten(1)
    return output


@torch.no_grad()
def _fit_basis(
    residual: torch.Tensor,
    rank: int,
    oversample: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    mean = residual.mean(dim=0)
    residual.sub_(mean)
    total_energy = float(residual.square().sum(dtype=torch.float64).cpu())
    q = min(int(rank) + int(oversample), len(residual) - 1, residual.shape[1])
    if q < int(rank):
        raise ValueError(f"Only {q} components are available for rank {rank}")
    torch.manual_seed(int(seed))
    if residual.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    _, singular, vectors = torch.pca_lowrank(
        residual, q=q, center=False, niter=int(iterations)
    )
    components = vectors[:, : int(rank)].T.contiguous()
    explained = float(
        singular[: int(rank)].double().square().sum().cpu()
        / max(total_energy, 1e-30)
    )
    return {
        "mean": mean.cpu(),
        "components": components.cpu(),
        "singular_values": singular[: int(rank)].cpu(),
        "explained": explained,
        "samples": int(len(residual)),
        "dimensions": int(residual.shape[1]),
        "pca_q": int(q),
    }


@torch.no_grad()
def _fit_inner_bases(
    channels: np.ndarray,
    indices: np.ndarray,
    teacher_cache: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    rank: int,
    batch_size: int,
    log_power_scale: float,
    oversample: int,
    iterations: int,
    seed: int,
) -> dict[int, dict[str, object]]:
    bases: dict[int, dict[str, object]] = {}
    for cell in sorted(np.unique(metadata["train_cells"]).tolist()):
        selected = indices[metadata["train_cells"][indices] == int(cell)]
        print(
            f"inner basis cell={int(cell)} samples={len(selected)} rank={rank}",
            flush=True,
        )
        residual = _residual_matrix(
            channels,
            selected,
            teacher_cache,
            autoencoder,
            shape,
            device,
            int(batch_size),
            float(log_power_scale),
        )
        basis = _fit_basis(
            residual,
            int(rank),
            int(oversample),
            int(iterations),
            int(seed) + int(cell),
        )
        del residual
        torch.cuda.empty_cache()
        bases[int(cell)] = basis
        print(
            f"inner basis cell={int(cell)} explained={basis['explained']:.6f}",
            flush=True,
        )
    return bases


@torch.no_grad()
def _coefficients(
    channels: np.ndarray,
    indices: np.ndarray,
    teacher_cache: dict[str, np.ndarray],
    bases: dict[int, dict[str, object]],
    metadata: dict[str, np.ndarray],
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    rank: int,
    batch_size: int,
    log_power_scale: float,
) -> np.ndarray:
    output = np.zeros((len(metadata["train_cells"]), int(rank)), dtype=np.float32)
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        selected = indices[start:stop]
        target_channel = torch.as_tensor(
            np.array(channels[selected], copy=True), device=device
        )
        target_shape, _, _ = channel_to_shape_target(target_channel, shape)
        seed_shape = _decode_seed_shape(
            teacher_cache, selected, autoencoder, device
        )
        residual = (
            angle_delay_log_power(target_shape, shape, float(log_power_scale))
            - angle_delay_log_power(seed_shape, shape, float(log_power_scale))
        ).flatten(1)
        cells = metadata["train_cells"][selected].astype(np.int64)
        batch_output = torch.empty(
            (len(selected), int(rank)), dtype=torch.float32, device=device
        )
        for cell in np.unique(cells):
            rows = np.flatnonzero(cells == int(cell))
            basis = bases[int(cell)]
            batch_output[rows] = project_low_rank_coefficients(
                residual[rows],
                basis["mean"],
                basis["components"],
                int(rank),
            )
        output[selected] = batch_output.cpu().numpy()
    return output


def _fit_gp_ensemble(
    metadata: dict[str, np.ndarray],
    features: np.ndarray,
    coefficients: np.ndarray,
    training: np.ndarray,
    queries: np.ndarray,
    kernels: list[str],
    noise: float,
    feature_length: float,
    feature_mix: float,
    device: torch.device,
    prediction_batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, object]]]:
    predictions = {
        kernel: np.zeros_like(coefficients, dtype=np.float32) for kernel in kernels
    }
    states: list[dict[str, object]] = []
    for cell in sorted(np.unique(metadata["train_cells"]).tolist()):
        cell_training = training[
            metadata["train_cells"][training] == int(cell)
        ]
        cell_queries = queries[metadata["train_cells"][queries] == int(cell)]
        for kernel in kernels:
            print(
                f"GP cell={int(cell)} kernel={kernel} train={len(cell_training)} "
                f"query={len(cell_queries)}",
                flush=True,
            )
            model = SharedMultiOutputGP(
                kernel,
                noise=float(noise),
                feature_length=float(feature_length),
                feature_mix=float(feature_mix),
            ).fit(
                metadata["train_positions"][cell_training],
                features[cell_training],
                coefficients[cell_training],
                device,
            )
            values, uncertainty = model.predict(
                metadata["train_positions"][cell_queries],
                features[cell_queries],
                device,
                int(prediction_batch_size),
            )
            predictions[kernel][cell_queries] = values
            states.append(
                {
                    "cell": int(cell),
                    "kernel": kernel,
                    "training_samples": int(len(cell_training)),
                    "query_samples": int(len(cell_queries)),
                    "mean_uncertainty": float(uncertainty.mean()),
                    "state": model.state_dict(),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    ensemble = np.mean(
        np.stack([predictions[name] for name in kernels], axis=0), axis=0
    ).astype(np.float32)
    return ensemble, predictions, states


def _coefficient_diagnostics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    prediction64 = prediction.astype(np.float64)
    target64 = target.astype(np.float64)
    error = float(np.square(prediction64 - target64).sum())
    zero_error = float(np.square(target64).sum())
    centered = target64 - target64.mean(axis=0, keepdims=True)
    centered_error = float(np.square(centered).sum())
    flat_prediction = prediction64.reshape(-1)
    flat_target = target64.reshape(-1)
    correlation = (
        float(np.corrcoef(flat_prediction, flat_target)[0, 1])
        if np.std(flat_prediction) > 0.0 and np.std(flat_target) > 0.0
        else 0.0
    )
    return {
        "mse": error / max(target64.size, 1),
        "skill_vs_zero": 1.0 - error / max(zero_error, 1e-30),
        "r2_vs_holdout_mean": 1.0 - error / max(centered_error, 1e-30),
        "pearson_flat": correlation,
    }


@torch.no_grad()
def _evaluate_teacher(
    indices: np.ndarray,
    predicted_coefficients: np.ndarray,
    teacher_cache: dict[str, np.ndarray],
    bases: dict[int, dict[str, object]],
    metadata: dict[str, np.ndarray],
    channels: np.ndarray,
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    rank: int,
    alpha: float,
    batch_size: int,
    log_power_scale: float,
    output_path: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    output = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.complex64,
            shape=(len(indices), *shape.raw_shape),
        )
    parts = []
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        selected = indices[start:stop]
        base_shape = _decode_seed_shape(
            teacher_cache, selected, autoencoder, device
        )
        base_log = angle_delay_log_power(
            base_shape, shape, float(log_power_scale)
        )
        cells = metadata["train_cells"][selected].astype(np.int64)
        correction = torch.zeros_like(base_log).flatten(1)
        for cell in np.unique(cells):
            rows = np.flatnonzero(cells == int(cell))
            basis = bases[int(cell)]
            coefficient_tensor = torch.as_tensor(
                predicted_coefficients[selected[rows]], device=device
            )
            correction[rows] = decode_low_rank_coefficients(
                coefficient_tensor,
                basis["mean"],
                basis["components"][: int(rank)],
            )
        corrected_log = base_log + float(alpha) * correction.reshape_as(base_log)
        corrected_shape = replace_angle_delay_log_power(
            base_shape,
            corrected_log,
            shape,
            float(log_power_scale),
        )
        source_outage = torch.as_tensor(
            np.asarray(teacher_cache["outage"][selected], dtype=bool), device=device
        )
        prediction = shape_to_channel(
            corrected_shape,
            torch.as_tensor(
                np.asarray(teacher_cache["log_power"][selected], dtype=np.float32),
                device=device,
            ),
            shape,
            source_outage,
        )
        target = torch.as_tensor(
            np.array(channels[selected], copy=True), device=device
        )
        target_outage = torch.as_tensor(
            metadata["outage"][selected].astype(bool), device=device
        )
        parts.append(sample_metric_batch(prediction, target, shape, target_outage))
        if output is not None:
            output[start:stop] = prediction.cpu().numpy().astype(np.complex64)
    if output is not None:
        output.flush()
        del output
    arrays = concatenate_metric_batches(parts)
    return aggregate_sample_metrics(arrays), arrays


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
        raise ValueError(
            f"Saved prediction has {len(prediction)} rows; expected {len(validation)}"
        )
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


def _basis_summary(
    bases: dict[int, dict[str, object]], rank: int
) -> dict[str, object]:
    output = {}
    for cell, value in bases.items():
        explained = value.get("explained")
        if explained is None and "explained_cumulative" in value:
            explained = value["explained_cumulative"][int(rank) - 1]
        output[str(cell)] = {
            "samples": int(value.get("samples", value.get("training_samples", 0))),
            "dimensions": int(value["dimensions"]),
            "components": int(len(value["components"])),
            "rank_explained": float(explained) if explained is not None else None,
        }
    return output


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    inner = report["inner"]
    strict = report.get("strict_fold0")
    rows = []
    for name, metrics in inner["candidate_metrics"].items():
        rows.append(
            f"| inner | {name} | {metrics['pas']:.6f} | {metrics['pdp']:.6f} | "
            f"{metrics['nmse']:.6f} | {metrics['score']:.6f} |"
        )
    if strict:
        for name, metrics in strict["candidate_metrics"].items():
            rows.append(
                f"| strict Fold0 | {name} | {metrics['pas']:.6f} | "
                f"{metrics['pdp']:.6f} | {metrics['nmse']:.6f} | "
                f"{metrics['score']:.6f} |"
            )
    text = f"""# L1-003 Magnitude GP Probe

Fold0 is offline validation, not the official online score.

## Hypothesis

A per-BS, three-kernel shared multi-output GP can predict the eight coefficients
of a train-only full-resolution log-power residual basis across spatial holes.

## Leakage boundary

- Inner basis and GP: Fold0-train inner-training targets only.
- Strict basis and GP: Fold0-train targets and OOF Teacher predictions only.
- Fold0 targets: canonical evaluation and diagnostic oracle only.

| Split | Candidate | PAS | PDP | NMSE | Score |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Decision: `{report['decision']}`.
Elapsed: `{float(report['elapsed_seconds']):.2f}` seconds.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict inner-spatial probe for rank-8 magnitude residual GPs"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--cache-dir", default="../research/scheme_e_065/residual_rank"
    )
    parser.add_argument(
        "--baseline-prediction",
        default="../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy",
    )
    parser.add_argument(
        "--strict-basis",
        default=(
            "artifacts/scheme_e_065/l0_011_magnitude_residual/"
            "train_only_log_power_basis.pt"
        ),
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--kernels", default="rq10,rq20,matern20")
    parser.add_argument("--alphas", default="0.5,1.0")
    parser.add_argument("--gp-noise", type=float, default=0.01)
    parser.add_argument("--feature-length", type=float, default=1.0)
    parser.add_argument("--feature-mix", type=float, default=0.5)
    parser.add_argument("--log-power-scale", type=float, default=4.0)
    parser.add_argument("--minimum-inner-gain", type=float, default=0.004)
    parser.add_argument("--pca-oversample", type=int, default=12)
    parser.add_argument("--pca-iterations", type=int, default=3)
    parser.add_argument("--residual-batch-size", type=int, default=8)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2681)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", default="artifacts/scheme_e_065/l1_003_magnitude_gp"
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L1_003_MAGNITUDE_GP_PROBE.json",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    seed_everything(int(args.seed))
    rank = int(args.rank)
    if rank < 1:
        raise ValueError("rank must be positive")
    kernels = [value.strip() for value in args.kernels.split(",") if value.strip()]
    if not kernels:
        raise ValueError("At least one GP kernel is required")
    alphas = sorted({float(value) for value in args.alphas.split(",")})
    if not alphas or min(alphas) <= 0.0:
        raise ValueError("Probe alphas must be positive")

    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Full-resolution inner PCA requires CUDA")
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
    hybrid, shape, _ = load_hybrid_checkpoint(config, checkpoint_path, device)
    autoencoder = hybrid.autoencoder.eval()
    teacher_cache = _load_cache(Path(args.cache_dir), "teacher_seed")
    features = _gp_features(metadata, priors)

    fold = int(config["split"]["validation_fold"])
    available = priors["available"].astype(bool)
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    observed = np.flatnonzero(available & ~validation_mask)
    validation = np.flatnonzero(available & validation_mask)
    nonoutage_observed = observed[~metadata["outage"][observed].astype(bool)]
    holdout_fold = int(np.max(metadata["spectral_folds"][nonoutage_observed]))
    inner_training = nonoutage_observed[
        metadata["spectral_folds"][nonoutage_observed] != holdout_fold
    ]
    inner_validation_nonoutage = nonoutage_observed[
        metadata["spectral_folds"][nonoutage_observed] == holdout_fold
    ]
    inner_validation = observed[
        metadata["spectral_folds"][observed] == holdout_fold
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "inner_split_indices.npz",
        training=inner_training,
        validation_nonoutage=inner_validation_nonoutage,
        validation_all=inner_validation,
        strict_training=nonoutage_observed,
        strict_validation=validation,
    )

    print(
        f"L1-003 rank={rank} kernels={kernels} inner_train={len(inner_training)} "
        f"inner_val={len(inner_validation)} strict_val={len(validation)}",
        flush=True,
    )
    inner_bases = _fit_inner_bases(
        channels,
        inner_training,
        teacher_cache,
        metadata,
        autoencoder,
        shape,
        device,
        rank,
        int(args.residual_batch_size),
        float(args.log_power_scale),
        int(args.pca_oversample),
        int(args.pca_iterations),
        int(args.seed),
    )
    torch.save(
        {
            "bases": inner_bases,
            "rank": rank,
            "leakage_boundary": "Fold0-train inner-training targets only",
        },
        output_dir / "inner_train_only_log_power_basis.pt",
    )
    inner_coefficients = _coefficients(
        channels,
        nonoutage_observed,
        teacher_cache,
        inner_bases,
        metadata,
        autoencoder,
        shape,
        device,
        rank,
        int(args.residual_batch_size),
        float(args.log_power_scale),
    )
    inner_prediction, inner_kernel_predictions, inner_states = _fit_gp_ensemble(
        metadata,
        features,
        inner_coefficients,
        inner_training,
        inner_validation,
        kernels,
        float(args.gp_noise),
        float(args.feature_length),
        float(args.feature_mix),
        device,
        int(args.prediction_batch_size),
    )
    inner_metrics: dict[str, dict[str, float | int]] = {}
    inner_arrays: dict[str, dict[str, np.ndarray]] = {}
    baseline_metrics, baseline_arrays = _evaluate_teacher(
        inner_validation,
        inner_prediction,
        teacher_cache,
        inner_bases,
        metadata,
        channels,
        autoencoder,
        shape,
        device,
        rank,
        0.0,
        int(args.decode_batch_size),
        float(args.log_power_scale),
    )
    inner_metrics["teacher_base"] = baseline_metrics
    inner_arrays["teacher_base"] = baseline_arrays
    for alpha in alphas:
        name = f"gp_uniform_alpha_{alpha}"
        metrics, arrays = _evaluate_teacher(
            inner_validation,
            inner_prediction,
            teacher_cache,
            inner_bases,
            metadata,
            channels,
            autoencoder,
            shape,
            device,
            rank,
            alpha,
            int(args.decode_batch_size),
            float(args.log_power_scale),
        )
        inner_metrics[name] = metrics
        inner_arrays[name] = arrays
    selected_inner_name, selected_inner_metrics = max(
        (
            (name, value)
            for name, value in inner_metrics.items()
            if name != "teacher_base"
        ),
        key=lambda item: float(item[1]["score"]),
    )
    selected_alpha = float(selected_inner_name.rsplit("_", 1)[1])
    inner_gain = float(selected_inner_metrics["score"]) - float(
        baseline_metrics["score"]
    )
    inner_passed = inner_gain >= float(args.minimum_inner_gain)
    coefficient_diagnostics = {
        "uniform": _coefficient_diagnostics(
            inner_prediction[inner_validation_nonoutage],
            inner_coefficients[inner_validation_nonoutage],
        ),
        **{
            kernel: _coefficient_diagnostics(
                prediction[inner_validation_nonoutage],
                inner_coefficients[inner_validation_nonoutage],
            )
            for kernel, prediction in inner_kernel_predictions.items()
        },
    }
    np.savez_compressed(
        output_dir / "Inner_Per_Sample_Metrics.npz",
        **{
            f"{name}__{field}": values
            for name, arrays in inner_arrays.items()
            for field, values in arrays.items()
        },
    )

    report: dict[str, object] = {
        "status": "INNER_PASS" if inner_passed else "INNER_FAIL",
        "hypothesis": (
            "A per-BS equal ensemble of RQ10, RQ20 and Matern20 shared "
            "multi-output GPs can predict rank-8 full-resolution log-power "
            "residual coefficients across spatial holes."
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_bootstrap.PROJECT_ROOT.parent,
            text=True,
        ).strip(),
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "baseline_prediction": args.baseline_prediction,
        "rank": rank,
        "kernels": kernels,
        "fixed_equal_kernel_weights": [1.0 / len(kernels)] * len(kernels),
        "settings": vars(args),
        "leakage_control": {
            "inner_basis": "Fold0-train inner-training targets only",
            "inner_gp": "Fold0-train inner-training coefficients only",
            "strict_basis": "Fold0-train OOF Teacher residuals only",
            "strict_gp": "Fold0-train OOF Teacher residual coefficients only",
            "fold0_target_usage": "canonical evaluation and diagnostic oracle only",
        },
        "split": {
            "fold": fold,
            "holdout_spectral_fold": holdout_fold,
            "inner_training": int(len(inner_training)),
            "inner_validation": int(len(inner_validation)),
            "strict_training": int(len(nonoutage_observed)),
            "strict_validation": int(len(validation)),
        },
        "inner_basis": _basis_summary(inner_bases, rank),
        "inner": {
            "candidate_metrics": inner_metrics,
            "selected_candidate": selected_inner_name,
            "selected_alpha": selected_alpha,
            "baseline_score": float(baseline_metrics["score"]),
            "selected_score": float(selected_inner_metrics["score"]),
            "gain": inner_gain,
            "minimum_gain": float(args.minimum_inner_gain),
            "passed": inner_passed,
            "coefficient_diagnostics": coefficient_diagnostics,
        },
        "strict_fold0": None,
        "decision": "PENDING_STRICT" if inner_passed else "DROP",
    }
    if not inner_passed:
        report["elapsed_seconds"] = time.perf_counter() - started
        with (output_dir / "gp_states.pkl").open("wb") as handle:
            pickle.dump({"inner": inner_states}, handle)
        save_json(report_path, report)
        _write_markdown(report_path.with_suffix(".md"), report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    strict_payload = torch.load(
        args.strict_basis, map_location="cpu", weights_only=False
    )
    if str(strict_payload.get("representation")) != "log_power":
        raise ValueError("Strict residual basis is not the log_power representation")
    strict_bases = {int(cell): value for cell, value in strict_payload["bases"].items()}
    for cell, basis in strict_bases.items():
        if len(basis["components"]) < rank:
            raise ValueError(f"Cell {cell} strict basis has fewer than {rank} components")
    full_coefficients = _coefficients(
        channels,
        nonoutage_observed,
        teacher_cache,
        strict_bases,
        metadata,
        autoencoder,
        shape,
        device,
        rank,
        int(args.residual_batch_size),
        float(args.log_power_scale),
    )
    strict_prediction, strict_kernel_predictions, strict_states = _fit_gp_ensemble(
        metadata,
        features,
        full_coefficients,
        nonoutage_observed,
        validation,
        kernels,
        float(args.gp_noise),
        float(args.feature_length),
        float(args.feature_mix),
        device,
        int(args.prediction_batch_size),
    )
    np.save(output_dir / "Fold0_Predicted_Coefficients.npy", strict_prediction)
    v4_metrics, v4_arrays = _evaluate_saved_prediction(
        args.baseline_prediction,
        validation,
        metadata,
        channels,
        shape,
        device,
        int(args.decode_batch_size),
    )
    teacher_metrics, teacher_arrays = _evaluate_teacher(
        validation,
        strict_prediction,
        teacher_cache,
        strict_bases,
        metadata,
        channels,
        autoencoder,
        shape,
        device,
        rank,
        0.0,
        int(args.decode_batch_size),
        float(args.log_power_scale),
    )
    candidate_metrics, candidate_arrays = _evaluate_teacher(
        validation,
        strict_prediction,
        teacher_cache,
        strict_bases,
        metadata,
        channels,
        autoencoder,
        shape,
        device,
        rank,
        selected_alpha,
        int(args.decode_batch_size),
        float(args.log_power_scale),
    )
    strict_candidates = {
        "v4_baseline": v4_arrays,
        "teacher_base": teacher_arrays,
        f"magnitude_gp_alpha_{selected_alpha}": candidate_arrays,
    }
    strict_candidate_metrics = {
        "v4_baseline": v4_metrics,
        "teacher_base": teacher_metrics,
        f"magnitude_gp_alpha_{selected_alpha}": candidate_metrics,
    }
    oracle = target_informed_expert_oracle(
        {
            "v4_baseline": v4_arrays,
            f"magnitude_gp_alpha_{selected_alpha}": candidate_arrays,
        }
    )
    oracle.pop("selection")
    baseline_score = float(v4_metrics["score"])
    candidate_score = float(candidate_metrics["score"])
    delta = candidate_score - baseline_score
    oracle_gain = float(oracle["metrics"]["score"]) - baseline_score
    if delta >= 0.004:
        decision = "PROMOTE"
    elif delta >= 0.001:
        decision = "MODIFY_ONCE"
    elif oracle_gain >= 0.010:
        decision = "KEEP_AS_EXPERT"
    else:
        decision = "DROP"
    selected_name = (
        f"magnitude_gp_alpha_{selected_alpha}" if delta > 0.0 else "v4_baseline"
    )
    selected_arrays = strict_candidates[selected_name]
    prediction_path: str | None = None
    if delta > 0.0:
        path = output_dir / "Fold0_Magnitude_GP_Prediction.npy"
        _evaluate_teacher(
            validation,
            strict_prediction,
            teacher_cache,
            strict_bases,
            metadata,
            channels,
            autoencoder,
            shape,
            device,
            rank,
            selected_alpha,
            int(args.decode_batch_size),
            float(args.log_power_scale),
            path,
        )
        prediction_path = str(path)
    np.savez_compressed(
        output_dir / "Fold0_Per_Sample_Metrics.npz", **selected_arrays
    )
    with (output_dir / "gp_states.pkl").open("wb") as handle:
        pickle.dump(
            {
                "inner": inner_states,
                "strict": strict_states,
                "rank": rank,
                "selected_alpha": selected_alpha,
                "kernels": kernels,
                "strict_basis": args.strict_basis,
            },
            handle,
        )
    strict_nonoutage = validation[
        ~metadata["outage"][validation].astype(bool)
    ]
    strict_target_coefficients = _coefficients(
        channels,
        strict_nonoutage,
        teacher_cache,
        strict_bases,
        metadata,
        autoencoder,
        shape,
        device,
        rank,
        int(args.residual_batch_size),
        float(args.log_power_scale),
    )
    report.update(
        {
            "status": "PASS",
            "strict_basis": _basis_summary(strict_bases, rank),
            "strict_fold0": {
                "candidate_metrics": strict_candidate_metrics,
                "selected_result": selected_name,
                "candidate_delta_vs_v4": delta,
                "target_informed_two_expert_oracle": oracle,
                "target_informed_oracle_gain": oracle_gain,
                "prediction": prediction_path,
                "coefficient_diagnostics": {
                    "uniform": _coefficient_diagnostics(
                        strict_prediction[strict_nonoutage],
                        strict_target_coefficients[strict_nonoutage],
                    )
                },
                "milestones": {
                    "M1_0635": candidate_score >= 0.635,
                    "M2_0642": candidate_score >= 0.642,
                    "M3_0650": candidate_score >= 0.650,
                },
            },
            "decision": decision,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    save_json(report_path, report)
    _write_markdown(report_path.with_suffix(".md"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
