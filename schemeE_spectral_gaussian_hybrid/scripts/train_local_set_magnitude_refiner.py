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
from scheme_e.local_magnitude import (
    estimate_magnitude_profile_shifts,
    same_cell_neighbors,
)
from scheme_e.local_set_magnitude import QueryConditionedLocalSetMagnitudeRefiner
from scheme_e.magnitude_refiner import energy_weighted_log_power_loss
from train_fullres_magnitude_refiner import (
    _decode_seed_shape,
    _ensure_log_power_cache,
    _evaluate_saved_prediction,
    _geometry_statistics,
    _load_cache,
    _load_npz,
    _normalized_geometry,
)


RELATIVE_FEATURE_DIM = 6


def _same_cell_neighbors_excluding_self(
    positions: np.ndarray,
    cells: np.ndarray,
    support: np.ndarray,
    queries: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidates, distances = same_cell_neighbors(
        positions,
        cells,
        support,
        queries,
        int(count) + 1,
    )
    selected = np.empty((len(queries), int(count)), dtype=np.int64)
    selected_distances = np.empty((len(queries), int(count)), dtype=np.float32)
    for row, query in enumerate(np.asarray(queries, dtype=np.int64)):
        valid = np.flatnonzero(candidates[row] != int(query))
        if not len(valid):
            valid = np.arange(len(candidates[row]), dtype=np.int64)
        chosen = valid[: int(count)]
        if len(chosen) < int(count):
            chosen = np.pad(chosen, (0, int(count) - len(chosen)), mode="edge")
        selected[row] = candidates[row, chosen]
        selected_distances[row] = distances[row, chosen]
    return selected, selected_distances


def _relative_features(
    positions: np.ndarray,
    queries: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    query = np.asarray(positions[queries], dtype=np.float32)
    source = np.asarray(positions[neighbors], dtype=np.float32)
    if query.shape[1] < 3:
        query = np.pad(query, ((0, 0), (0, 3 - query.shape[1])))
        source = np.pad(source, ((0, 0), (0, 0), (0, 3 - source.shape[2])))
    query = query[:, :3]
    source = source[:, :, :3]
    delta = (query[:, None] - source) / 20.0
    distance = np.asarray(distances, dtype=np.float32)
    return np.concatenate(
        [
            delta,
            (distance / 20.0)[..., None],
            (np.log1p(distance) / 3.0)[..., None],
            np.exp(-distance / 20.0)[..., None],
        ],
        axis=2,
    ).astype(np.float32)


def _precompute_shifts(
    queries: np.ndarray,
    neighbors: np.ndarray,
    base_cache: np.ndarray,
    batch_size: int,
    scale: float,
) -> np.ndarray:
    shifts = np.empty((len(queries), neighbors.shape[1], 3), dtype=np.int16)
    for start in range(0, len(queries), int(batch_size)):
        stop = min(start + int(batch_size), len(queries))
        shifts[start:stop] = estimate_magnitude_profile_shifts(
            np.array(base_cache[queries[start:stop]], copy=True),
            np.array(base_cache[neighbors[start:stop]], copy=True),
            scale=float(scale),
        )
        if start == 0 or stop == len(queries) or stop % 400 == 0:
            print(f"local-set shifts {stop}/{len(queries)}", flush=True)
    return shifts


def _aligned_neighbor_maps(
    base_cache: np.ndarray,
    target_cache: np.ndarray,
    query_base: np.ndarray,
    neighbors: np.ndarray,
    shifts: np.ndarray,
    delay_start: int,
    delay_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor_base = np.array(base_cache[neighbors], copy=True, dtype=np.float32)
    neighbor_target = np.array(target_cache[neighbors], copy=True, dtype=np.float32)
    residual = neighbor_target - neighbor_base
    aligned_base = np.empty_like(neighbor_base)
    aligned_residual = np.empty_like(residual)
    for batch in range(len(neighbors)):
        for neighbor in range(neighbors.shape[1]):
            shift = tuple(int(value) for value in shifts[batch, neighbor])
            aligned_base[batch, neighbor] = np.roll(
                neighbor_base[batch, neighbor], shift, axis=(1, 2, 3)
            )
            aligned_residual[batch, neighbor] = np.roll(
                residual[batch, neighbor], shift, axis=(1, 2, 3)
            )
    base_crop = query_base[..., int(delay_start) : int(delay_stop)]
    return (
        aligned_residual[..., int(delay_start) : int(delay_stop)],
        aligned_base[..., int(delay_start) : int(delay_stop)] - base_crop[:, None],
    )


def _build_model(
    shape: object,
    geometry_dim: int,
    cell_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> QueryConditionedLocalSetMagnitudeRefiner:
    return QueryConditionedLocalSetMagnitudeRefiner(
        input_channels=int(shape.m_p * shape.n),
        geometry_dim=int(geometry_dim),
        relative_dim=RELATIVE_FEATURE_DIM,
        cell_count=int(cell_count),
        width=int(args.width),
        blocks=int(args.blocks),
        dropout=float(args.dropout),
        maximum_residual=float(args.maximum_residual),
    ).to(device)


def _train_epoch(
    model: QueryConditionedLocalSetMagnitudeRefiner,
    optimizer: torch.optim.Optimizer,
    queries: np.ndarray,
    neighbors: np.ndarray,
    shifts: np.ndarray,
    relative: np.ndarray,
    base_cache: np.ndarray,
    target_cache: np.ndarray,
    geometry: np.ndarray,
    cells: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> float:
    model.train()
    order = rng.permutation(len(queries))
    delay = int(base_cache.shape[-1])
    crop = min(int(args.delay_crop), delay)
    loss_sum = 0.0
    for start in range(0, len(order), int(args.batch_size)):
        rows = order[start : start + int(args.batch_size)]
        selected = queries[rows]
        delay_start = int(rng.integers(0, delay - crop + 1))
        delay_stop = delay_start + crop
        query_base_full = np.array(base_cache[selected], copy=True, dtype=np.float32)
        neighbor_residual, neighbor_delta = _aligned_neighbor_maps(
            base_cache,
            target_cache,
            query_base_full,
            neighbors[rows],
            shifts[rows],
            delay_start,
            delay_stop,
        )
        base = torch.as_tensor(
            query_base_full[..., delay_start:delay_stop], device=device
        )
        target = torch.as_tensor(
            np.array(
                target_cache[selected, ..., delay_start:delay_stop],
                copy=True,
                dtype=np.float32,
            ),
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(
                base,
                torch.as_tensor(neighbor_residual, device=device),
                torch.as_tensor(neighbor_delta, device=device),
                torch.as_tensor(relative[rows], device=device),
                torch.as_tensor(geometry[selected], device=device),
                torch.as_tensor(cells[selected], device=device),
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
        if not torch.isfinite(loss):
            raise RuntimeError("Local-set magnitude model produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
        optimizer.step()
        loss_sum += float(loss.detach().cpu()) * len(rows)
    return loss_sum / max(len(order), 1)


@torch.no_grad()
def _evaluate_model(
    model: QueryConditionedLocalSetMagnitudeRefiner,
    queries: np.ndarray,
    neighbors: np.ndarray,
    shifts: np.ndarray,
    relative: np.ndarray,
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
    output_path: Path | None = None,
    deployment_baseline: np.ndarray | None = None,
) -> tuple[dict[str, float | int], dict[str, np.ndarray], dict[str, float]]:
    model.eval()
    if deployment_baseline is not None and len(deployment_baseline) != len(queries):
        raise ValueError("Deployment baseline row count does not match queries")
    output = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.complex64,
            shape=(len(queries), *shape.raw_shape),
        )
    parts = []
    correction_sum = 0.0
    correction_elements = 0
    entropy_sum = 0.0
    entropy_elements = 0
    effective_sum = 0.0
    for start in range(0, len(queries), int(args.validation_batch_size)):
        stop = min(start + int(args.validation_batch_size), len(queries))
        selected = queries[start:stop]
        query_base = np.array(base_cache[selected], copy=True, dtype=np.float32)
        neighbor_residual, neighbor_delta = _aligned_neighbor_maps(
            base_cache,
            target_cache,
            query_base,
            neighbors[start:stop],
            shifts[start:stop],
            0,
            int(base_cache.shape[-1]),
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            result = model(
                torch.as_tensor(query_base, device=device),
                torch.as_tensor(neighbor_residual, device=device),
                torch.as_tensor(neighbor_delta, device=device),
                torch.as_tensor(relative[start:stop], device=device),
                torch.as_tensor(geometry[selected], device=device),
                torch.as_tensor(metadata["train_cells"][selected], device=device),
                total_delay=int(base_cache.shape[-1]),
            )
        correction = result["correction"].float()
        attention = result["attention"].float().clamp_min(1e-12)
        correction_sum += float(correction.abs().sum().cpu())
        correction_elements += correction.numel()
        entropy = -torch.sum(attention * torch.log(attention), dim=1)
        entropy_sum += float(entropy.sum().cpu())
        entropy_elements += entropy.numel()
        effective_sum += float(
            (1.0 / attention.square().sum(dim=1).clamp_min(1e-12)).mean().cpu()
        ) * len(selected)

        if deployment_baseline is None:
            seed_shape = _decode_seed_shape(
                teacher_cache, selected, autoencoder, device
            )
            predicted_log = result["log_power"].float()
            output_log_power = torch.as_tensor(
                np.asarray(teacher_cache["log_power"][selected], dtype=np.float32),
                device=device,
            )
            output_outage = torch.as_tensor(
                np.asarray(teacher_cache["outage"][selected], dtype=bool),
                device=device,
            )
        else:
            baseline_channel = torch.as_tensor(
                np.array(deployment_baseline[start:stop], copy=True), device=device
            )
            seed_shape, output_log_power, output_outage = channel_to_shape_target(
                baseline_channel, shape
            )
            baseline_log = angle_delay_log_power(
                seed_shape, shape, float(args.log_power_scale)
            ).reshape_as(correction)
            predicted_log = (baseline_log + correction).clamp(0.0, 20.0)
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
        prediction = shape_to_channel(
            predicted_shape,
            output_log_power,
            shape,
            output_outage,
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
        {
            "mean_absolute_correction": correction_sum
            / max(correction_elements, 1),
            "mean_attention_entropy": entropy_sum / max(entropy_elements, 1),
            "mean_effective_neighbors": effective_sum / max(len(queries), 1),
        },
    )


def _train_with_validation(
    model: QueryConditionedLocalSetMagnitudeRefiner,
    training: np.ndarray,
    training_neighbors: np.ndarray,
    training_shifts: np.ndarray,
    training_relative: np.ndarray,
    validation: np.ndarray,
    validation_neighbors: np.ndarray,
    validation_shifts: np.ndarray,
    validation_relative: np.ndarray,
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
    baseline, _, _ = _evaluate_model(
        model,
        validation,
        validation_neighbors,
        validation_shifts,
        validation_relative,
        base_cache,
        target_cache,
        geometry,
        metadata,
        channels,
        teacher_cache,
        autoencoder,
        shape,
        device,
        args,
    )
    best_score = float(baseline["score"])
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.perf_counter()
        train_loss = _train_epoch(
            model,
            optimizer,
            training,
            training_neighbors,
            training_shifts,
            training_relative,
            base_cache,
            target_cache,
            geometry,
            metadata["train_cells"],
            device,
            rng,
            args,
        )
        scheduler.step()
        metrics = None
        diagnostics = None
        if epoch == 1 or epoch % int(args.validation_interval) == 0:
            metrics, _, diagnostics = _evaluate_model(
                model,
                validation,
                validation_neighbors,
                validation_shifts,
                validation_relative,
                base_cache,
                target_cache,
                geometry,
                metadata,
                channels,
                teacher_cache,
                autoencoder,
                shape,
                device,
                args,
            )
            if float(metrics["score"]) > best_score + float(args.minimum_delta):
                best_score = float(metrics["score"])
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        score_text = "nan" if metrics is None else f"{float(metrics['score']):.6f}"
        print(
            f"LocalSet epoch={epoch}/{args.epochs} train={train_loss:.6f} "
            f"score={score_text} best={best_score:.6f}@{best_epoch} "
            f"lr={record['learning_rate']:.2e} seconds={record['seconds']:.2f}",
            flush=True,
        )
        if metrics is not None and stale >= int(args.patience):
            print(f"LocalSet early stop at epoch={epoch}", flush=True)
            break
    return best_state, best_epoch, history, baseline


def _train_fixed_epochs(
    model: QueryConditionedLocalSetMagnitudeRefiner,
    training: np.ndarray,
    neighbors: np.ndarray,
    shifts: np.ndarray,
    relative: np.ndarray,
    base_cache: np.ndarray,
    target_cache: np.ndarray,
    geometry: np.ndarray,
    cells: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    epochs: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs), 1),
        eta_min=float(args.minimum_learning_rate),
    )
    rng = np.random.default_rng(int(args.seed) + 1000)
    history = []
    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        loss = _train_epoch(
            model,
            optimizer,
            training,
            neighbors,
            shifts,
            relative,
            base_cache,
            target_cache,
            geometry,
            cells,
            device,
            rng,
            args,
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"LocalSet full epoch={epoch}/{epochs} train={loss:.6f} "
            f"seconds={history[-1]['seconds']:.2f}",
            flush=True,
        )
    return history


def _prepare_neighbor_data(
    label: str,
    positions: np.ndarray,
    cells: np.ndarray,
    support: np.ndarray,
    queries: np.ndarray,
    base_cache: np.ndarray,
    args: argparse.Namespace,
    *,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(
        f"{label}: queries={len(queries)} support={len(support)} "
        f"neighbors={args.neighbors}",
        flush=True,
    )
    if exclude_self:
        neighbors, distances = _same_cell_neighbors_excluding_self(
            positions, cells, support, queries, int(args.neighbors)
        )
    else:
        neighbors, distances = same_cell_neighbors(
            positions, cells, support, queries, int(args.neighbors)
        )
    relative = _relative_features(positions, queries, neighbors, distances)
    shifts = _precompute_shifts(
        queries,
        neighbors,
        base_cache,
        int(args.shift_batch_size),
        float(args.log_power_scale),
    )
    return neighbors, distances, relative, shifts


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    inner = report["inner"]
    rows = [
        ("inner identity", inner["baseline_metrics"]),
        ("inner best", inner["best_metrics"]),
    ]
    strict = report.get("strict_fold0")
    if strict:
        rows.extend(
            [
                ("strict quality-gated V4", strict["v4_baseline"]),
                ("strict local-set correction", strict["candidate"]),
            ]
        )
    table = "\n".join(
        f"| {name} | {float(metrics['pas']):.6f} | "
        f"{float(metrics['pdp']):.6f} | {float(metrics['nmse']):.6f} | "
        f"{float(metrics['score']):.6f} |"
        for name, metrics in rows
    )
    path.write_text(
        f"""# L1-006 Query-Conditioned Local-Set Magnitude Probe

Fold0 is offline validation, not the official online score.

The model keeps the complete `[Mp*N,Mv,Mh,S]` log-power grid. Four same-cell
observations provide full-resolution OOF Teacher residual maps. A shared local
3D encoder and query-conditioned per-voxel attention fuse those maps into one
bounded magnitude correction; there is no PCA or global latent bottleneck.

The strict candidate adds only that correction to the existing quality-gated
V4 prediction, so a zero correction reproduces the current deployable baseline
and preserves its carrier phase.

| Split | PAS | PDP | NMSE | Score |
|---|---:|---:|---:|---:|
{table}

Inner gate: Score gain >= `{float(inner['minimum_score_gain']):.6f}`, PDP drop
<= `{float(inner['maximum_pdp_drop']):.6f}`, and best epoch > 0.

Decision: `{report['decision']}`.
Elapsed: `{float(report['elapsed_seconds']):.2f}` seconds.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-controlled inner probe for query-conditioned local-set "
            "full-resolution magnitude correction"
        )
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
        default="../research/scheme_e_065/FOLD0_QUALITY_GATED_PREDICTION.npy",
    )
    parser.add_argument("--neighbors", type=int, default=4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.02)
    parser.add_argument("--maximum-residual", type=float, default=4.0)
    parser.add_argument("--log-power-scale", type=float, default=4.0)
    parser.add_argument("--energy-emphasis", type=float, default=2.0)
    parser.add_argument("--maximum-energy-weight", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--minimum-delta", type=float, default=0.0002)
    parser.add_argument("--minimum-inner-gain", type=float, default=0.004)
    parser.add_argument("--maximum-inner-pdp-drop", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--shift-batch-size", type=int, default=8)
    parser.add_argument("--delay-crop", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2721)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        default="artifacts/scheme_e_065/l1_006_local_set_magnitude",
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L1_006_LOCAL_SET_MAGNITUDE.json",
    )
    args = parser.parse_args()
    if int(args.neighbors) < 1:
        raise ValueError("neighbors must be positive")

    started = time.perf_counter()
    seed_everything(int(args.seed))
    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("L1-006 requires CUDA for the full-resolution grid")

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
    if not len(inner_training) or not len(inner_validation) or not len(validation):
        raise RuntimeError("L1-006 received an empty leakage-control split")

    positions = metadata["train_positions"].astype(np.float32)
    cells = metadata["train_cells"].astype(np.int64)
    raw_geometry = metadata["train_geometry_features"].astype(np.float32)
    inner_mean, inner_std = _geometry_statistics(raw_geometry, inner_training)
    inner_geometry = _normalized_geometry(raw_geometry, inner_mean, inner_std)
    cell_count = int(cells.max()) + 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"L1-006 inner_train={len(inner_training)} inner_val={len(inner_validation)} "
        f"strict_val={len(validation)} grid={tuple(base_cache.shape[1:])}",
        flush=True,
    )
    (
        inner_training_neighbors,
        inner_training_distances,
        inner_training_relative,
        inner_training_shifts,
    ) = _prepare_neighbor_data(
        "inner training neighbors",
        positions,
        cells,
        inner_training,
        inner_training,
        base_cache,
        args,
        exclude_self=True,
    )
    (
        inner_validation_neighbors,
        inner_validation_distances,
        inner_validation_relative,
        inner_validation_shifts,
    ) = _prepare_neighbor_data(
        "inner validation neighbors",
        positions,
        cells,
        inner_training,
        inner_validation,
        base_cache,
        args,
        exclude_self=False,
    )
    np.savez_compressed(
        output_dir / "inner_neighbor_data.npz",
        training=inner_training,
        validation=inner_validation,
        training_neighbors=inner_training_neighbors,
        training_distances=inner_training_distances,
        training_shifts=inner_training_shifts,
        validation_neighbors=inner_validation_neighbors,
        validation_distances=inner_validation_distances,
        validation_shifts=inner_validation_shifts,
    )

    model = _build_model(
        shape, raw_geometry.shape[1], cell_count, args, device
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_state, best_epoch, history, inner_baseline = _train_with_validation(
        model,
        inner_training,
        inner_training_neighbors,
        inner_training_shifts,
        inner_training_relative,
        inner_validation,
        inner_validation_neighbors,
        inner_validation_shifts,
        inner_validation_relative,
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
    inner_best, inner_arrays, inner_diagnostics = _evaluate_model(
        model,
        inner_validation,
        inner_validation_neighbors,
        inner_validation_shifts,
        inner_validation_relative,
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
    inner_gain = float(inner_best["score"]) - float(inner_baseline["score"])
    inner_pdp_delta = float(inner_best["pdp"]) - float(inner_baseline["pdp"])
    inner_passed = (
        best_epoch > 0
        and inner_gain >= float(args.minimum_inner_gain)
        and inner_pdp_delta >= -float(args.maximum_inner_pdp_drop)
    )
    inner_checkpoint = output_dir / "inner_best.pt"
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
        inner_checkpoint,
    )
    np.savez_compressed(
        output_dir / "Inner_Per_Sample_Metrics.npz", **inner_arrays
    )

    report: dict[str, object] = {
        "status": "INNER_PASS" if inner_passed else "INNER_FAIL",
        "experiment_id": "L1-006",
        "hypothesis": (
            "Query-conditioned per-voxel attention can select and transport useful "
            "full-resolution magnitude residuals from a local same-cell set."
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
            "inner_validation": "held-out spatial spectral fold inside Fold0-train",
            "strict_training": "Fold0-train OOF Teacher/target pairs only",
            "fold0_target_usage": "canonical metrics and diagnostic oracle only",
            "neighbor_inputs": "other observed same-cell channels only",
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
            "score_gain": inner_gain,
            "pdp_delta": inner_pdp_delta,
            "minimum_score_gain": float(args.minimum_inner_gain),
            "maximum_pdp_drop": float(args.maximum_inner_pdp_drop),
            "diagnostics": inner_diagnostics,
            "passed": inner_passed,
            "checkpoint": str(inner_checkpoint),
        },
        "strict_fold0": None,
        "decision": "PENDING_STRICT" if inner_passed else "DROP",
    }
    if not inner_passed:
        report["elapsed_seconds"] = time.perf_counter() - started
        save_json(report_path, report)
        _write_markdown(report_path.with_suffix(".md"), report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    model.to("cpu")
    del model, best_state
    torch.cuda.empty_cache()
    full_mean, full_std = _geometry_statistics(raw_geometry, nonoutage_observed)
    full_geometry = _normalized_geometry(raw_geometry, full_mean, full_std)
    (
        full_training_neighbors,
        full_training_distances,
        full_training_relative,
        full_training_shifts,
    ) = _prepare_neighbor_data(
        "strict training neighbors",
        positions,
        cells,
        nonoutage_observed,
        nonoutage_observed,
        base_cache,
        args,
        exclude_self=True,
    )
    (
        strict_validation_neighbors,
        strict_validation_distances,
        strict_validation_relative,
        strict_validation_shifts,
    ) = _prepare_neighbor_data(
        "strict validation neighbors",
        positions,
        cells,
        nonoutage_observed,
        validation,
        base_cache,
        args,
        exclude_self=False,
    )
    np.savez_compressed(
        output_dir / "strict_neighbor_data.npz",
        training=nonoutage_observed,
        validation=validation,
        training_neighbors=full_training_neighbors,
        training_distances=full_training_distances,
        training_shifts=full_training_shifts,
        validation_neighbors=strict_validation_neighbors,
        validation_distances=strict_validation_distances,
        validation_shifts=strict_validation_shifts,
    )

    deployment_baseline = np.load(args.baseline_prediction, mmap_mode="r")
    v4_metrics, v4_arrays = _evaluate_saved_prediction(
        args.baseline_prediction,
        validation,
        metadata,
        channels,
        shape,
        device,
        int(args.validation_batch_size),
    )
    seed_everything(int(args.seed) + 1000)
    full_model = _build_model(
        shape, raw_geometry.shape[1], cell_count, args, device
    )
    identity_metrics, _, _ = _evaluate_model(
        full_model,
        validation,
        strict_validation_neighbors,
        strict_validation_shifts,
        strict_validation_relative,
        base_cache,
        target_cache,
        full_geometry,
        metadata,
        channels,
        teacher_cache,
        autoencoder,
        shape,
        device,
        args,
        deployment_baseline=deployment_baseline,
    )
    identity_error = abs(float(identity_metrics["score"]) - float(v4_metrics["score"]))
    if identity_error > 2e-5:
        raise RuntimeError(
            "Strict residual integration does not reproduce the quality-gated "
            f"baseline: absolute score error={identity_error:.8f}"
        )

    full_history = _train_fixed_epochs(
        full_model,
        nonoutage_observed,
        full_training_neighbors,
        full_training_shifts,
        full_training_relative,
        base_cache,
        target_cache,
        full_geometry,
        cells,
        device,
        args,
        int(best_epoch),
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
    candidate_metrics, candidate_arrays, strict_diagnostics = _evaluate_model(
        full_model,
        validation,
        strict_validation_neighbors,
        strict_validation_shifts,
        strict_validation_relative,
        base_cache,
        target_cache,
        full_geometry,
        metadata,
        channels,
        teacher_cache,
        autoencoder,
        shape,
        device,
        args,
        deployment_baseline=deployment_baseline,
    )
    oracle = target_informed_expert_oracle(
        {"quality_gated_v4": v4_arrays, "local_set_magnitude": candidate_arrays}
    )
    oracle.pop("selection", None)
    delta = float(candidate_metrics["score"]) - float(v4_metrics["score"])
    oracle_gain = float(oracle["metrics"]["score"]) - float(v4_metrics["score"])
    if float(candidate_metrics["score"]) >= 0.650 and delta > 0.0:
        decision = "PROMOTE_TARGET"
    elif delta >= 0.004:
        decision = "PROMOTE"
    elif oracle_gain >= 0.010:
        decision = "KEEP_AS_EXPERT"
    else:
        decision = "DROP"

    prediction_path = None
    if delta > 0.0:
        prediction_path = output_dir / "Fold0_Local_Set_Magnitude_Prediction.npy"
        _evaluate_model(
            full_model,
            validation,
            strict_validation_neighbors,
            strict_validation_shifts,
            strict_validation_relative,
            base_cache,
            target_cache,
            full_geometry,
            metadata,
            channels,
            teacher_cache,
            autoencoder,
            shape,
            device,
            args,
            output_path=prediction_path,
            deployment_baseline=deployment_baseline,
        )
    np.savez_compressed(
        output_dir / "Fold0_Candidate_Per_Sample_Metrics.npz", **candidate_arrays
    )
    report.update(
        {
            "status": "STRICT_COMPLETE",
            "strict_fold0": {
                "v4_baseline": v4_metrics,
                "identity_reproduction": identity_metrics,
                "identity_score_error": identity_error,
                "candidate": candidate_metrics,
                "delta": delta,
                "diagnostics": strict_diagnostics,
                "target_informed_two_expert_oracle": oracle,
                "target_informed_oracle_gain": oracle_gain,
                "checkpoint": str(full_checkpoint),
                "prediction": None if prediction_path is None else str(prediction_path),
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
