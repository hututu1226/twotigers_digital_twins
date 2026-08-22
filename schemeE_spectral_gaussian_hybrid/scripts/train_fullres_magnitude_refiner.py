from __future__ import annotations

import argparse
from copy import deepcopy
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
from scheme_e.config import choose_device, load_config, save_json, seed_everything
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
    target_informed_expert_oracle,
)
from scheme_e.hybrid_training import load_hybrid_checkpoint
from scheme_e.magnitude_refiner import (
    FullResolutionMagnitudeRefiner,
    energy_weighted_log_power_loss,
    magnitude_marginal_cosine_loss,
)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _load_cache(path: Path, prefix: str) -> dict[str, np.ndarray]:
    return {
        name: np.load(path / f"{prefix}_{name}.npy", mmap_mode="r")
        for name in ("spectrum", "detail", "log_power", "outage")
    }


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
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        decoded = autoencoder.decode(spectrum, detail)
    return decoded.float()


@torch.no_grad()
def _ensure_log_power_cache(
    output_dir: Path,
    channels: np.ndarray,
    teacher_cache: dict[str, np.ndarray],
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    batch_size: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = output_dir / "teacher_log_power.npy"
    target_path = output_dir / "target_log_power.npy"
    manifest_path = output_dir / "manifest.json"
    expected_shape = (
        len(channels),
        int(shape.m_p * shape.n),
        int(shape.m_v),
        int(shape.m_h),
        int(shape.s),
    )
    if base_path.is_file() and target_path.is_file() and manifest_path.is_file():
        base = np.load(base_path, mmap_mode="r")
        target = np.load(target_path, mmap_mode="r")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            tuple(base.shape) == expected_shape
            and tuple(target.shape) == expected_shape
            and float(manifest.get("log_power_scale", -1.0)) == float(scale)
        ):
            print(f"reusing full-resolution cache at {output_dir}", flush=True)
            return base, target, manifest_path
        del base, target
        raise RuntimeError(
            "Existing full-resolution cache is incompatible; use a new cache directory"
        )

    base = np.lib.format.open_memmap(
        base_path, mode="w+", dtype=np.float16, shape=expected_shape
    )
    target = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float16, shape=expected_shape
    )
    indices = np.arange(len(channels), dtype=np.int64)
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
        base[start:stop] = (
            angle_delay_log_power(seed_shape, shape, float(scale))
            .reshape(stop - start, *expected_shape[1:])
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        target[start:stop] = (
            angle_delay_log_power(target_shape, shape, float(scale))
            .reshape(stop - start, *expected_shape[1:])
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        if start == 0 or stop == len(indices) or stop % 400 == 0:
            print(f"full-resolution cache {stop}/{len(indices)}", flush=True)
    base.flush()
    target.flush()
    del base, target
    manifest = {
        "shape": list(expected_shape),
        "dtype": "float16",
        "log_power_scale": float(scale),
        "source": "OOF Teacher decoded by frozen AE and raw training targets",
    }
    save_json(manifest_path, manifest)
    return (
        np.load(base_path, mmap_mode="r"),
        np.load(target_path, mmap_mode="r"),
        manifest_path,
    )


def _geometry_statistics(
    geometry: np.ndarray, training: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = geometry[training].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        geometry[training].std(axis=0, dtype=np.float64), 1e-3
    ).astype(np.float32)
    return mean, std


def _normalized_geometry(
    geometry: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    values = np.clip((geometry.astype(np.float32) - mean) / std, -8.0, 8.0)
    return np.nan_to_num(values, nan=0.0, posinf=8.0, neginf=-8.0).astype(
        np.float32
    )


def _build_model(
    shape: object,
    geometry_dim: int,
    cell_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> FullResolutionMagnitudeRefiner:
    return FullResolutionMagnitudeRefiner(
        input_channels=int(shape.m_p * shape.n),
        geometry_dim=int(geometry_dim),
        cell_count=int(cell_count),
        width=int(args.width),
        blocks=int(args.blocks),
        dropout=float(args.dropout),
        maximum_residual=float(args.maximum_residual),
        log_power_scale=float(args.log_power_scale),
    ).to(device)


def _train_epoch(
    model: FullResolutionMagnitudeRefiner,
    optimizer: torch.optim.Optimizer,
    base_cache: np.ndarray,
    target_cache: np.ndarray,
    geometry: np.ndarray,
    cells: np.ndarray,
    training: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    args: argparse.Namespace,
    frequency_groups: int,
) -> float:
    model.train()
    order = rng.permutation(training)
    loss_sum = 0.0
    delay = int(base_cache.shape[-1])
    crop = min(int(args.delay_crop), delay)
    for start in range(0, len(order), int(args.batch_size)):
        selected = order[start : start + int(args.batch_size)]
        delay_start = int(rng.integers(0, delay - crop + 1))
        delay_stop = delay_start + crop
        base = torch.as_tensor(
            np.array(base_cache[selected, ..., delay_start:delay_stop], copy=True),
            device=device,
            dtype=torch.float32,
        )
        target = torch.as_tensor(
            np.array(target_cache[selected, ..., delay_start:delay_stop], copy=True),
            device=device,
            dtype=torch.float32,
        )
        geometry_tensor = torch.as_tensor(geometry[selected], device=device)
        cell_tensor = torch.as_tensor(cells[selected], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(
                base,
                geometry_tensor,
                cell_tensor,
                delay_start=delay_start,
                total_delay=delay,
            )["log_power"]
            loss = energy_weighted_log_power_loss(
                prediction,
                target,
                float(args.log_power_scale),
                float(args.energy_emphasis),
                float(args.maximum_energy_weight),
            )
            if float(args.marginal_weight) > 0.0:
                loss = loss + float(args.marginal_weight) * magnitude_marginal_cosine_loss(
                    prediction,
                    target,
                    float(args.log_power_scale),
                    int(frequency_groups),
                )
        if not torch.isfinite(loss):
            raise RuntimeError("Magnitude refiner produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
        optimizer.step()
        loss_sum += float(loss.detach().cpu()) * len(selected)
    return loss_sum / max(len(order), 1)


@torch.no_grad()
def _evaluate_model(
    model: FullResolutionMagnitudeRefiner,
    indices: np.ndarray,
    base_cache: np.ndarray,
    geometry: np.ndarray,
    metadata: dict[str, np.ndarray],
    channels: np.ndarray,
    teacher_cache: dict[str, np.ndarray],
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    args: argparse.Namespace,
    output_path: Path | None = None,
) -> tuple[dict[str, float | int], dict[str, np.ndarray], float]:
    model.eval()
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
    correction_sum = 0.0
    correction_elements = 0
    for start in range(0, len(indices), int(args.validation_batch_size)):
        stop = min(start + int(args.validation_batch_size), len(indices))
        selected = indices[start:stop]
        base_log = torch.as_tensor(
            np.array(base_cache[selected], copy=True),
            device=device,
            dtype=torch.float32,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            result = model(
                base_log,
                torch.as_tensor(geometry[selected], device=device),
                torch.as_tensor(metadata["train_cells"][selected], device=device),
                total_delay=int(base_cache.shape[-1]),
            )
        predicted_log = result["log_power"].float()
        correction = result["correction"].float()
        correction_sum += float(correction.abs().sum().cpu())
        correction_elements += correction.numel()
        seed_shape = _decode_seed_shape(
            teacher_cache, selected, autoencoder, device
        )
        predicted_shape = replace_angle_delay_log_power(
            seed_shape,
            predicted_log.reshape(
                len(selected),
                shape.m_p,
                shape.n,
                shape.m_v,
                shape.m_h,
                shape.s,
            ),
            shape,
            float(args.log_power_scale),
        )
        source_outage = torch.as_tensor(
            np.asarray(teacher_cache["outage"][selected], dtype=bool), device=device
        )
        prediction = shape_to_channel(
            predicted_shape,
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
    return (
        aggregate_sample_metrics(arrays),
        arrays,
        correction_sum / max(correction_elements, 1),
    )


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


def _train_with_validation(
    model: FullResolutionMagnitudeRefiner,
    training: np.ndarray,
    validation: np.ndarray,
    base_cache: np.ndarray,
    target_cache: np.ndarray,
    geometry: np.ndarray,
    metadata: dict[str, np.ndarray],
    channels: np.ndarray,
    teacher_cache: dict[str, np.ndarray],
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, object]], dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
        eta_min=float(args.minimum_learning_rate),
    )
    rng = np.random.default_rng(int(args.seed))
    baseline_metrics, _, _ = _evaluate_model(
        model,
        validation,
        base_cache,
        geometry,
        metadata,
        channels,
        teacher_cache,
        autoencoder,
        shape,
        device,
        args,
    )
    best_score = float(baseline_metrics["score"])
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        started = time.perf_counter()
        train_loss = _train_epoch(
            model,
            optimizer,
            base_cache,
            target_cache,
            geometry,
            metadata["train_cells"],
            training,
            device,
            rng,
            args,
            int(shape.n),
        )
        scheduler.step()
        metrics = None
        mean_correction = None
        if epoch == 1 or epoch % int(args.validation_interval) == 0:
            metrics, _, mean_correction = _evaluate_model(
                model,
                validation,
                base_cache,
                geometry,
                metadata,
                channels,
                teacher_cache,
                autoencoder,
                shape,
                device,
                args,
            )
            score = float(metrics["score"])
            if score > best_score + float(args.minimum_delta):
                best_score = score
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "metrics": metrics,
            "mean_absolute_correction": mean_correction,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        score_text = "nan" if metrics is None else f"{float(metrics['score']):.6f}"
        print(
            f"Refiner epoch={epoch}/{args.epochs} train={train_loss:.6f} "
            f"score={score_text} best={best_score:.6f}@{best_epoch} "
            f"lr={record['learning_rate']:.2e} seconds={record['seconds']:.2f}",
            flush=True,
        )
        if metrics is not None and stale >= int(args.patience):
            print(f"Refiner early stop at epoch={epoch}", flush=True)
            break
    return best_state, best_epoch, history, baseline_metrics


def _train_fixed_epochs(
    model: FullResolutionMagnitudeRefiner,
    training: np.ndarray,
    base_cache: np.ndarray,
    target_cache: np.ndarray,
    geometry: np.ndarray,
    cells: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
    frequency_groups: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
        eta_min=float(args.minimum_learning_rate),
    )
    rng = np.random.default_rng(int(args.seed) + 1000)
    history = []
    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        loss = _train_epoch(
            model,
            optimizer,
            base_cache,
            target_cache,
            geometry,
            cells,
            training,
            device,
            rng,
            args,
            int(frequency_groups),
        )
        scheduler.step()
        record = {
            "epoch": float(epoch),
            "train_loss": loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"Full refiner epoch={epoch}/{epochs} train={loss:.6f} "
            f"lr={record['learning_rate']:.2e} seconds={record['seconds']:.2f}",
            flush=True,
        )
    return history


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    inner = report["inner"]
    strict = report.get("strict_fold0")
    marginal_weight = float(report["settings"].get("marginal_weight", 0.0))
    experiment_name = (
        "L1-005 Metric-Aligned Full-Resolution Magnitude Refiner"
        if marginal_weight > 0.0
        else "L1-004 Full-Resolution Magnitude Refiner"
    )
    rows = [
        (
            "inner baseline",
            inner["baseline_metrics"],
        ),
        (
            "inner best",
            inner["best_metrics"],
        ),
    ]
    if strict:
        rows.extend(
            [
                ("strict V4 baseline", strict["v4_baseline"]),
                ("strict refiner", strict["refiner"]),
            ]
        )
    table = "\n".join(
        f"| {name} | {metrics['pas']:.6f} | {metrics['pdp']:.6f} | "
        f"{metrics['nmse']:.6f} | {metrics['score']:.6f} |"
        for name, metrics in rows
    )
    path.write_text(
        f"""# {experiment_name}

Fold0 is offline validation, not the official online score.

The model keeps the complete `[Mp*N,Mv,Mh,S]` log-power grid and predicts a
bounded local 3D convolutional correction. Fold0 targets are never used for
training or early stopping.

| Split | PAS | PDP | NMSE | Score |
|---|---:|---:|---:|---:|
{table}

Decision: `{report['decision']}`.
Elapsed: `{float(report['elapsed_seconds']):.2f}` seconds.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inner-spatial probe for a full-resolution magnitude refiner"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--latent-cache", default="../research/scheme_e_065/residual_rank"
    )
    parser.add_argument(
        "--map-cache",
        default="artifacts/scheme_e_065/fullres_log_power_cache",
    )
    parser.add_argument(
        "--baseline-prediction",
        default="../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--maximum-residual", type=float, default=4.0)
    parser.add_argument("--log-power-scale", type=float, default=4.0)
    parser.add_argument("--energy-emphasis", type=float, default=2.0)
    parser.add_argument("--maximum-energy-weight", type=float, default=12.0)
    parser.add_argument("--marginal-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--minimum-delta", type=float, default=0.0002)
    parser.add_argument("--minimum-inner-gain", type=float, default=0.004)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--delay-crop", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2691)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", default="artifacts/scheme_e_065/l1_004_fullres_refiner"
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L1_004_FULLRES_REFINER.json",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    seed_everything(int(args.seed))
    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The full-resolution refiner probe requires CUDA")
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
    teacher_cache = _load_cache(Path(args.latent_cache), "teacher_seed")
    base_cache, target_cache, cache_manifest = _ensure_log_power_cache(
        Path(args.map_cache),
        channels,
        teacher_cache,
        autoencoder,
        shape,
        device,
        int(args.cache_batch_size),
        float(args.log_power_scale),
    )

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
    inner_validation = observed[
        metadata["spectral_folds"][observed] == holdout_fold
    ]
    raw_geometry = metadata["train_geometry_features"].astype(np.float32)
    inner_mean, inner_std = _geometry_statistics(raw_geometry, inner_training)
    inner_geometry = _normalized_geometry(raw_geometry, inner_mean, inner_std)
    cell_count = int(metadata["train_cells"].max()) + 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "inner_split_indices.npz",
        training=inner_training,
        validation=inner_validation,
        strict_training=nonoutage_observed,
        strict_validation=validation,
    )
    experiment_id = "L1-005" if float(args.marginal_weight) > 0.0 else "L1-004"
    print(
        f"{experiment_id} inner_train={len(inner_training)} inner_val={len(inner_validation)} "
        f"grid={tuple(base_cache.shape[1:])}",
        flush=True,
    )
    model = _build_model(
        shape, raw_geometry.shape[1], cell_count, args, device
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_state, best_epoch, history, inner_baseline = _train_with_validation(
        model,
        inner_training,
        inner_validation,
        base_cache,
        target_cache,
        inner_geometry,
        metadata,
        channels,
        teacher_cache,
        autoencoder,
        shape,
        device,
        args,
    )
    model.load_state_dict(best_state)
    inner_best, inner_arrays, mean_correction = _evaluate_model(
        model,
        inner_validation,
        base_cache,
        inner_geometry,
        metadata,
        channels,
        teacher_cache,
        autoencoder,
        shape,
        device,
        args,
    )
    inner_gain = float(inner_best["score"]) - float(inner_baseline["score"])
    inner_passed = inner_gain >= float(args.minimum_inner_gain) and best_epoch > 0
    torch.save(
        {
            "model": best_state,
            "best_epoch": int(best_epoch),
            "history": history,
            "geometry_mean": inner_mean,
            "geometry_std": inner_std,
            "settings": vars(args),
            "leakage_boundary": "Fold0-train inner-training only",
        },
        output_dir / "inner_best.pt",
    )
    np.savez_compressed(
        output_dir / "Inner_Per_Sample_Metrics.npz", **inner_arrays
    )
    report: dict[str, object] = {
        "status": "INNER_PASS" if inner_passed else "INNER_FAIL",
        "hypothesis": (
            "Angle/delay marginal cosine supervision preserves the local CNN's "
            "PAS and NMSE gains while preventing PDP regression."
            if float(args.marginal_weight) > 0.0
            else "A geometry-conditioned local 3D CNN can refine the complete OOF "
            "Teacher log-power grid across spatial holes without PCA coordinates."
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_bootstrap.PROJECT_ROOT.parent,
            text=True,
        ).strip(),
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "cache_manifest": str(cache_manifest),
        "parameters": int(parameter_count),
        "settings": vars(args),
        "leakage_control": {
            "inner_training": "Fold0-train inner-training OOF Teacher/target pairs",
            "early_stopping": "Fold0-train inner spatial validation",
            "strict_training": "Fold0-train OOF Teacher/target pairs only",
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
        "inner": {
            "baseline_metrics": inner_baseline,
            "best_metrics": inner_best,
            "best_epoch": int(best_epoch),
            "gain": inner_gain,
            "minimum_gain": float(args.minimum_inner_gain),
            "mean_absolute_correction": float(mean_correction),
            "passed": inner_passed,
        },
        "strict_fold0": None,
        "decision": (
            "PENDING_STRICT"
            if inner_passed
            else "MODIFY_ONCE"
            if inner_gain >= 0.001 and float(args.marginal_weight) <= 0.0
            else "DROP"
        ),
    }
    if not inner_passed:
        report["elapsed_seconds"] = time.perf_counter() - started
        save_json(report_path, report)
        _write_markdown(report_path.with_suffix(".md"), report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    full_mean, full_std = _geometry_statistics(raw_geometry, nonoutage_observed)
    full_geometry = _normalized_geometry(raw_geometry, full_mean, full_std)
    seed_everything(int(args.seed) + 1000)
    full_model = _build_model(
        shape, raw_geometry.shape[1], cell_count, args, device
    )
    full_history = _train_fixed_epochs(
        full_model,
        nonoutage_observed,
        base_cache,
        target_cache,
        full_geometry,
        metadata["train_cells"],
        device,
        args,
        int(best_epoch),
        int(shape.n),
    )
    full_checkpoint = output_dir / "strict_full_train.pt"
    torch.save(
        {
            "model": full_model.state_dict(),
            "epochs": int(best_epoch),
            "history": full_history,
            "geometry_mean": full_mean,
            "geometry_std": full_std,
            "settings": vars(args),
            "leakage_boundary": "Fold0-train OOF Teacher/target pairs only",
        },
        full_checkpoint,
    )
    candidate_metrics, candidate_arrays, strict_correction = _evaluate_model(
        full_model,
        validation,
        base_cache,
        full_geometry,
        metadata,
        channels,
        teacher_cache,
        autoencoder,
        shape,
        device,
        args,
    )
    v4_metrics, v4_arrays = _evaluate_saved_prediction(
        args.baseline_prediction,
        validation,
        metadata,
        channels,
        shape,
        device,
        int(args.validation_batch_size),
    )
    oracle = target_informed_expert_oracle(
        {"v4_baseline": v4_arrays, "fullres_refiner": candidate_arrays}
    )
    oracle.pop("selection")
    delta = float(candidate_metrics["score"]) - float(v4_metrics["score"])
    oracle_gain = float(oracle["metrics"]["score"]) - float(v4_metrics["score"])
    if delta >= 0.004:
        decision = "PROMOTE"
    elif delta >= 0.001:
        decision = "MODIFY_ONCE"
    elif oracle_gain >= 0.010:
        decision = "KEEP_AS_EXPERT"
    else:
        decision = "DROP"
    prediction_path = None
    selected_arrays = v4_arrays
    if delta > 0.0:
        prediction = output_dir / "Fold0_Fullres_Refiner_Prediction.npy"
        _evaluate_model(
            full_model,
            validation,
            base_cache,
            full_geometry,
            metadata,
            channels,
            teacher_cache,
            autoencoder,
            shape,
            device,
            args,
            prediction,
        )
        prediction_path = str(prediction)
        selected_arrays = candidate_arrays
    np.savez_compressed(
        output_dir / "Fold0_Per_Sample_Metrics.npz", **selected_arrays
    )
    report.update(
        {
            "status": "PASS",
            "strict_fold0": {
                "v4_baseline": v4_metrics,
                "refiner": candidate_metrics,
                "delta": delta,
                "mean_absolute_correction": float(strict_correction),
                "target_informed_two_expert_oracle": oracle,
                "target_informed_oracle_gain": oracle_gain,
                "checkpoint": str(full_checkpoint),
                "prediction": prediction_path,
                "milestones": {
                    "M1_0635": float(candidate_metrics["score"]) >= 0.635,
                    "M2_0642": float(candidate_metrics["score"]) >= 0.642,
                    "M3_0650": float(candidate_metrics["score"]) >= 0.650,
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
