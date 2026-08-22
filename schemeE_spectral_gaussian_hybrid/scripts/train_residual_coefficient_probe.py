from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import _bootstrap  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as functional

from scheme_e.angle_delay import shape_to_channel
from scheme_e.config import choose_device, load_config, save_json, seed_everything
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
    target_informed_expert_oracle,
)
from scheme_e.hybrid_training import load_hybrid_checkpoint
from scheme_e.power_safety import apply_outage_policy
from scheme_e.residual_set_model import ResidualCoefficientSetEncoder


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _load_cache(path: Path, prefix: str) -> dict[str, np.ndarray]:
    return {
        name: np.load(path / f"{prefix}_{name}.npy", mmap_mode="r")
        for name in ("spectrum", "detail", "log_power", "outage")
    }


def _query_features(
    metadata: dict[str, np.ndarray], priors: dict[str, np.ndarray]
) -> np.ndarray:
    relative_ue = (
        priors["ue_log_energy"].astype(np.float32)
        - priors["log_power"].astype(np.float32)[:, None]
    )
    cells = metadata["train_cells"].astype(np.int64)
    cell_count = int(cells.max()) + 1
    return np.concatenate(
        [
            metadata["train_positions"][:, :2].astype(np.float32),
            metadata["train_geometry_features"].astype(np.float32),
            relative_ue,
            priors["log_power"].astype(np.float32)[:, None],
            priors["uncertainty"].astype(np.float32)[:, None],
            priors["outage_probability"].astype(np.float32)[:, None],
            np.eye(cell_count, dtype=np.float32)[cells],
        ],
        axis=1,
    ).astype(np.float32)


def _coefficients(
    target_cache: dict[str, np.ndarray],
    seed_cache: dict[str, np.ndarray],
    bases: dict[str, dict[str, object]],
    metadata: dict[str, np.ndarray],
    indices: np.ndarray,
    rank: int,
) -> np.ndarray:
    output = np.zeros((len(metadata["train_cells"]), rank), dtype=np.float32)
    for cell in np.unique(metadata["train_cells"]):
        selected = indices[metadata["train_cells"][indices] == cell]
        basis = bases[f"cell{int(cell)}_spectrum"]
        mean = np.asarray(basis["mean"], dtype=np.float32)
        components = np.asarray(basis["components"], dtype=np.float32)[:, :rank]
        residual = (
            target_cache["spectrum"][selected].astype(np.float32)
            - seed_cache["spectrum"][selected].astype(np.float32)
            - mean
        )
        output[selected] = residual @ components
    return output


def _fit_residual_bases(
    target_cache: dict[str, np.ndarray],
    seed_cache: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    training: np.ndarray,
    rank: int,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    """Fit a probe basis without exposing its inner holdout targets."""
    bases: dict[str, dict[str, object]] = {}
    for cell in np.unique(metadata["train_cells"]):
        selected = training[metadata["train_cells"][training] == cell]
        residual = torch.as_tensor(
            target_cache["spectrum"][selected].astype(np.float32)
            - seed_cache["spectrum"][selected].astype(np.float32),
            device=device,
        )
        mean = residual.mean(dim=0)
        centered = residual - mean
        actual_rank = min(int(rank), len(selected) - 1, centered.shape[1])
        if actual_rank < int(rank):
            raise ValueError(
                f"Cell {int(cell)} has only {len(selected)} samples for rank {rank}"
            )
        _, singular, components = torch.pca_lowrank(
            centered, q=actual_rank, center=False, niter=4
        )
        bases[f"cell{int(cell)}_spectrum"] = {
            "mean": mean.cpu(),
            "components": components.cpu(),
            "singular_values": singular.cpu(),
            "samples": int(len(selected)),
            "dimensions": int(centered.shape[1]),
            "source": "inner_training_only",
        }
    return bases


def _tree_neighbors(
    positions: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    count: int,
) -> np.ndarray:
    actual = min(int(count), len(support))
    if actual < 1 or not len(query):
        raise ValueError("Neighbor lookup requires non-empty query and support")
    _, local = cKDTree(positions[support, :2]).query(
        positions[query, :2], k=actual
    )
    local = np.asarray(local, dtype=np.int64).reshape(len(query), actual)
    neighbors = support[local]
    if actual < int(count):
        neighbors = np.concatenate(
            [neighbors, np.repeat(neighbors[:, -1:], int(count) - actual, axis=1)],
            axis=1,
        )
    return neighbors


def _fold_excluded_neighbors(
    metadata: dict[str, np.ndarray], indices: np.ndarray, count: int
) -> np.ndarray:
    folds = metadata["spectral_folds"]
    output = np.empty((len(indices), int(count)), dtype=np.int64)
    for fold in np.unique(folds[indices]):
        rows = np.flatnonzero(folds[indices] == fold)
        support = indices[folds[indices] != fold]
        output[rows] = _tree_neighbors(
            metadata["train_positions"], support, indices[rows], count
        )
    return output


def _neighbor_coefficient_prediction(
    metadata: dict[str, np.ndarray],
    support: np.ndarray,
    queries: np.ndarray,
    coefficients: np.ndarray,
    count: int,
) -> np.ndarray:
    output = np.zeros((len(metadata["train_cells"]), coefficients.shape[1]), dtype=np.float32)
    for cell in np.unique(metadata["train_cells"]):
        cell_support = support[metadata["train_cells"][support] == cell]
        cell_queries = queries[metadata["train_cells"][queries] == cell]
        neighbors = _tree_neighbors(
            metadata["train_positions"], cell_support, cell_queries, count
        )
        output[cell_queries] = coefficients[neighbors].mean(axis=1)
    return output


def _coefficient_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    difference_energy = float(
        np.square(prediction.astype(np.float64) - target.astype(np.float64)).sum()
    )
    zero_energy = float(np.square(target.astype(np.float64)).sum())
    centered_target = target.astype(np.float64) - target.mean(axis=0, keepdims=True)
    variance = float(np.square(centered_target).sum())
    correlation = float(
        np.corrcoef(prediction.reshape(-1), target.reshape(-1))[0, 1]
    )
    return {
        "mse": difference_energy / max(target.size, 1),
        "skill_vs_zero": 1.0 - difference_energy / max(zero_energy, 1e-30),
        "r2_vs_holdout_mean": 1.0 - difference_energy / max(variance, 1e-30),
        "pearson_flat": correlation,
    }


def _fit_statistics(
    query_raw: np.ndarray,
    seed_spectrum: np.ndarray,
    coefficients: np.ndarray,
    metadata: dict[str, np.ndarray],
    training: np.ndarray,
) -> dict[str, np.ndarray]:
    def stats(values: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(values.std(axis=0, dtype=np.float64), floor).astype(np.float32)
        return mean, std

    query_mean, query_std = stats(query_raw[training], 1e-3)
    seed_mean, seed_std = stats(seed_spectrum[training].astype(np.float32), 1e-3)
    coefficient_mean, coefficient_std = stats(coefficients[training], 1e-3)
    position_mean, position_std = stats(
        metadata["train_positions"][training, :2].astype(np.float32), 1.0
    )
    geometry_mean, geometry_std = stats(
        metadata["train_geometry_features"][training].astype(np.float32), 1e-3
    )
    return {
        "query_mean": query_mean,
        "query_std": query_std,
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "coefficient_mean": coefficient_mean,
        "coefficient_std": coefficient_std,
        "position_mean": position_mean,
        "position_std": position_std,
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
    }


def _model_arrays(
    queries: np.ndarray,
    neighbors: np.ndarray,
    query_raw: np.ndarray,
    seed_cache: dict[str, np.ndarray],
    coefficients: np.ndarray,
    metadata: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    statistics: dict[str, np.ndarray],
    spectrum_shape: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    position_z = (
        metadata["train_positions"][:, :2].astype(np.float32)
        - statistics["position_mean"]
    ) / statistics["position_std"]
    geometry_z = (
        metadata["train_geometry_features"].astype(np.float32)
        - statistics["geometry_mean"]
    ) / statistics["geometry_std"]
    coefficient_z = (
        coefficients - statistics["coefficient_mean"]
    ) / statistics["coefficient_std"]
    relative_position = position_z[neighbors] - position_z[queries, None]
    physical_delta = (
        metadata["train_positions"][neighbors, :2]
        - metadata["train_positions"][queries, None, :2]
    ).astype(np.float32)
    physical_distance = np.linalg.norm(physical_delta, axis=2, keepdims=True)
    normalized_distance = np.linalg.norm(relative_position, axis=2, keepdims=True)
    direction = physical_delta / np.maximum(physical_distance, 1e-3)
    geometry_delta = geometry_z[neighbors] - geometry_z[queries, None]
    power_delta = (
        metadata["log_power"][neighbors].astype(np.float32)
        - priors["log_power"][queries].astype(np.float32)[:, None]
    )[..., None]
    neighbor_features = np.concatenate(
        [
            coefficient_z[neighbors],
            relative_position,
            normalized_distance,
            direction,
            geometry_delta,
            power_delta,
        ],
        axis=2,
    ).astype(np.float32)
    seed = (
        seed_cache["spectrum"][queries].astype(np.float32)
        - statistics["seed_mean"]
    ) / statistics["seed_std"]
    return {
        "indices": queries,
        "spectrum": seed.reshape(len(queries), *spectrum_shape).astype(np.float32),
        "query": (
            (query_raw[queries] - statistics["query_mean"])
            / statistics["query_std"]
        ).astype(np.float32),
        "neighbor": neighbor_features,
        "distance": (physical_distance / 10.0).astype(np.float32),
        "target": coefficient_z[queries].astype(np.float32),
    }


def _build_model(
    arrays: dict[str, np.ndarray],
    spectrum_shape: tuple[int, int, int, int],
    rank: int,
    width: int,
    dropout: float,
    device: torch.device,
) -> ResidualCoefficientSetEncoder:
    return ResidualCoefficientSetEncoder(
        spectrum_shape=spectrum_shape,
        query_dim=arrays["query"].shape[1],
        neighbor_dim=arrays["neighbor"].shape[2],
        coefficient_dim=rank,
        width=width,
        dropout=dropout,
    ).to(device)


@torch.no_grad()
def _predict(
    model: ResidualCoefficientSetEncoder,
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    model.eval()
    predictions = []
    effective = []
    for start in range(0, len(arrays["indices"]), batch_size):
        stop = min(start + batch_size, len(arrays["indices"]))
        output = model(
            torch.as_tensor(arrays["spectrum"][start:stop], device=device),
            torch.as_tensor(arrays["query"][start:stop], device=device),
            torch.as_tensor(arrays["neighbor"][start:stop], device=device),
            torch.as_tensor(arrays["distance"][start:stop], device=device),
        )
        predictions.append(output["coefficients"].float().cpu().numpy())
        effective.append(output["effective_neighbors"].float().cpu().numpy())
    return np.concatenate(predictions), float(np.concatenate(effective).mean())


def _train(
    model: ResidualCoefficientSetEncoder,
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray] | None,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    validation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    label: str,
) -> tuple[dict[str, torch.Tensor], int, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(training["indices"]))
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                output = model(
                    torch.as_tensor(training["spectrum"][rows], device=device),
                    torch.as_tensor(training["query"][rows], device=device),
                    torch.as_tensor(training["neighbor"][rows], device=device),
                    torch.as_tensor(training["distance"][rows], device=device),
                )
                target = torch.as_tensor(training["target"][rows], device=device)
                loss = functional.smooth_l1_loss(
                    output["coefficients"].float(), target.float(), beta=0.5
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu()) * len(rows)
        validation_loss = float("nan")
        effective = float("nan")
        if validation is not None and (epoch == 1 or epoch % 5 == 0):
            predicted, effective = _predict(
                model, validation, device, validation_batch_size
            )
            validation_loss = float(
                np.mean(functional.smooth_l1_loss(
                    torch.from_numpy(predicted),
                    torch.from_numpy(validation["target"]),
                    beta=0.5,
                    reduction="none",
                ).numpy())
            )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                stale += 5
        elif validation is None and epoch == epochs:
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        record = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(len(order), 1),
            "validation_loss": validation_loss,
            "effective_neighbors": effective,
        }
        history.append(record)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{label} epoch={epoch}/{epochs} train={record['train_loss']:.6f} "
                f"validation={validation_loss:.6f} effective_neighbors={effective:.2f}",
                flush=True,
            )
        if validation is not None and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{label} did not produce a checkpoint")
    return best_state, best_epoch, history


def _restore_coefficients(
    normalized: np.ndarray, statistics: dict[str, np.ndarray]
) -> np.ndarray:
    return (
        normalized * statistics["coefficient_std"]
        + statistics["coefficient_mean"]
    ).astype(np.float32)


@torch.no_grad()
def _decode_metrics(
    indices: np.ndarray,
    predicted_coefficients: np.ndarray,
    seed_cache: dict[str, np.ndarray],
    bases: dict[str, dict[str, object]],
    metadata: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    channels: np.ndarray,
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    batch_size: int,
    alpha: float,
    outage_thresholds: np.ndarray,
    outage_strengths: np.ndarray,
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
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        current = indices[start:stop]
        cells = metadata["train_cells"][current].astype(np.int64)
        spectrum = torch.as_tensor(
            seed_cache["spectrum"][current].astype(np.float32), device=device
        )
        for cell in np.unique(cells):
            selected = np.flatnonzero(cells == cell)
            basis = bases[f"cell{int(cell)}_spectrum"]
            mean = basis["mean"].to(device)
            components = basis["components"][:, : predicted_coefficients.shape[1]].to(
                device
            )
            coefficients = torch.as_tensor(
                predicted_coefficients[start:stop][selected], device=device
            )
            correction = mean + coefficients @ components.T
            spectrum[selected] = spectrum[selected] + float(alpha) * correction
        detail = torch.as_tensor(
            seed_cache["detail"][current].astype(np.float32), device=device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            decoded = autoencoder.decode(spectrum, detail)
        prediction = shape_to_channel(
            decoded.float(),
            torch.as_tensor(seed_cache["log_power"][current], device=device),
            shape,
            torch.as_tensor(seed_cache["outage"][current], device=device),
        )
        prediction = apply_outage_policy(
            prediction,
            torch.as_tensor(priors["outage_probability"][current], device=device),
            torch.as_tensor(outage_thresholds[cells], device=device),
            torch.as_tensor(outage_strengths[cells], device=device),
        )
        target = torch.as_tensor(np.array(channels[current], copy=True), device=device)
        true_outage = torch.as_tensor(metadata["outage"][current], device=device)
        parts.append(sample_metric_batch(prediction, target, shape, true_outage))
        if output is not None:
            output[start:stop] = prediction.cpu().numpy().astype(np.complex64)
    if output is not None:
        output.flush()
        del output
    arrays = concatenate_metric_batches(parts)
    return aggregate_sample_metrics(arrays), arrays


def _baseline_arrays(
    path: str | Path,
    validation: np.ndarray,
    metadata: dict[str, np.ndarray],
    channels: np.ndarray,
    shape: object,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    prediction = np.load(path, mmap_mode="r")
    parts = []
    for start in range(0, len(validation), batch_size):
        stop = min(start + batch_size, len(validation))
        parts.append(
            sample_metric_batch(
                torch.as_tensor(
                    np.array(prediction[start:stop], copy=True), device=device
                ),
                torch.as_tensor(
                    np.array(channels[validation[start:stop]], copy=True), device=device
                ),
                shape,
                torch.as_tensor(metadata["outage"][validation[start:stop]], device=device),
            )
        )
    return concatenate_metric_batches(parts)


def _policy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        np.asarray(value["outage_threshold_by_cell"], dtype=np.float32),
        np.asarray(value["soft_outage_strength_by_cell"], dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1 probe for a query-conditioned spectrum residual coefficient model"
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
        "--policy", default="reports/generated/v4_attempt1_policy.json"
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--validation-batch-size", type=int, default=128)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-inner-gain", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=2651)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", default="artifacts/scheme_e_065/l1_residual_probe"
    )
    parser.add_argument(
        "--report", default="../research/scheme_e_065/L1_RESIDUAL_PROBE.json"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    seed_everything(int(args.seed))
    config = load_config(args.config)
    device = choose_device(args.device)
    metadata = _load_npz(Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz")
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    checkpoint_path = Path(config["hybrid"]["output_dir"]) / "best.pt"
    hybrid, shape, _ = load_hybrid_checkpoint(config, checkpoint_path, device)
    autoencoder = hybrid.autoencoder
    spectrum_shape = tuple(autoencoder.spectrum_shape.tensor_shape)
    cache_dir = Path(args.cache_dir)
    target_cache = _load_cache(cache_dir, "target")
    seed_cache = _load_cache(cache_dir, "teacher_seed")
    bases = torch.load(
        cache_dir / "train_only_basis.pt", map_location="cpu", weights_only=False
    )
    fold = int(config["split"]["validation_fold"])
    available = priors["available"].astype(bool)
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    observed = np.flatnonzero(available & ~validation_mask)
    validation = np.flatnonzero(available & validation_mask)
    nonoutage_observed = observed[~metadata["outage"][observed].astype(bool)]
    query_raw = _query_features(metadata, priors)
    outage_thresholds, outage_strengths = _policy(args.policy)
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
    inner_bases = _fit_residual_bases(
        target_cache,
        seed_cache,
        metadata,
        inner_training,
        int(args.rank),
        device,
    )
    inner_coefficients = _coefficients(
        target_cache,
        seed_cache,
        inner_bases,
        metadata,
        nonoutage_observed,
        int(args.rank),
    )
    inner_predictions = np.zeros((len(metadata["train_cells"]), int(args.rank)), dtype=np.float32)
    inner_reports = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(inner_bases, output_dir / "inner_train_only_basis.pt")
    np.savez_compressed(
        output_dir / "inner_split_indices.npz",
        training=inner_training,
        validation_nonoutage=inner_validation_nonoutage,
        validation_all=inner_validation,
    )
    best_epochs: dict[int, int] = {}

    for cell in np.unique(metadata["train_cells"]):
        cell_train = inner_training[metadata["train_cells"][inner_training] == cell]
        cell_validation = inner_validation_nonoutage[
            metadata["train_cells"][inner_validation_nonoutage] == cell
        ]
        cell_validation_all = inner_validation[
            metadata["train_cells"][inner_validation] == cell
        ]
        statistics = _fit_statistics(
            query_raw,
            seed_cache["spectrum"],
            inner_coefficients,
            metadata,
            cell_train,
        )
        train_neighbors = _fold_excluded_neighbors(
            metadata, cell_train, int(args.neighbors)
        )
        validation_neighbors = _tree_neighbors(
            metadata["train_positions"],
            cell_train,
            cell_validation,
            int(args.neighbors),
        )
        validation_all_neighbors = _tree_neighbors(
            metadata["train_positions"],
            cell_train,
            cell_validation_all,
            int(args.neighbors),
        )
        training_arrays = _model_arrays(
            cell_train,
            train_neighbors,
            query_raw,
            seed_cache,
            inner_coefficients,
            metadata,
            priors,
            statistics,
            spectrum_shape,
        )
        validation_arrays = _model_arrays(
            cell_validation,
            validation_neighbors,
            query_raw,
            seed_cache,
            inner_coefficients,
            metadata,
            priors,
            statistics,
            spectrum_shape,
        )
        model = _build_model(
            training_arrays,
            spectrum_shape,
            int(args.rank),
            int(args.width),
            float(args.dropout),
            device,
        )
        state, best_epoch, history = _train(
            model,
            training_arrays,
            validation_arrays,
            device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            validation_batch_size=int(args.validation_batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            seed=int(args.seed) + int(cell),
            label=f"inner-cell{int(cell)}",
        )
        model.load_state_dict(state)
        validation_all_arrays = _model_arrays(
            cell_validation_all,
            validation_all_neighbors,
            query_raw,
            seed_cache,
            inner_coefficients,
            metadata,
            priors,
            statistics,
            spectrum_shape,
        )
        normalized_prediction, effective = _predict(
            model,
            validation_all_arrays,
            device,
            int(args.validation_batch_size),
        )
        inner_predictions[cell_validation_all] = _restore_coefficients(
            normalized_prediction, statistics
        )
        best_epochs[int(cell)] = int(best_epoch)
        torch.save(
            {
                "model": state,
                "statistics": statistics,
                "best_epoch": best_epoch,
                "history": history,
                "cell": int(cell),
                "settings": vars(args),
            },
            output_dir / f"inner_cell{int(cell)}.pt",
        )
        inner_reports.append(
            {
                "cell": int(cell),
                "training_samples": int(len(cell_train)),
                "validation_samples": int(len(cell_validation)),
                "best_epoch": int(best_epoch),
                "effective_neighbors": effective,
            }
        )

    alpha_metrics = {}
    diagnostic_arrays: dict[str, dict[str, np.ndarray]] = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        metrics, arrays = _decode_metrics(
            inner_validation,
            inner_predictions[inner_validation],
            seed_cache,
            inner_bases,
            metadata,
            priors,
            channels,
            autoencoder,
            shape,
            device,
            int(args.decode_batch_size),
            alpha,
            outage_thresholds,
            outage_strengths,
        )
        alpha_metrics[str(alpha)] = metrics
        diagnostic_arrays[f"model_alpha_{alpha}"] = arrays
    spatial_predictions = {
        "nearest1": _neighbor_coefficient_prediction(
            metadata,
            inner_training,
            inner_validation,
            inner_coefficients,
            1,
        ),
        "mean16": _neighbor_coefficient_prediction(
            metadata,
            inner_training,
            inner_validation,
            inner_coefficients,
            int(args.neighbors),
        ),
    }
    spatial_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, prediction in spatial_predictions.items():
        spatial_metrics[name] = {}
        for alpha in (0.25, 0.5, 1.0):
            metrics, arrays = _decode_metrics(
                inner_validation,
                prediction[inner_validation],
                seed_cache,
                inner_bases,
                metadata,
                priors,
                channels,
                autoencoder,
                shape,
                device,
                int(args.decode_batch_size),
                alpha,
                outage_thresholds,
                outage_strengths,
            )
            spatial_metrics[name][str(alpha)] = metrics
            diagnostic_arrays[f"{name}_alpha_{alpha}"] = arrays
    inner_oracle = target_informed_expert_oracle(diagnostic_arrays)
    inner_oracle.pop("selection")
    nonoutage_rows = inner_validation_nonoutage
    coefficient_diagnostics = {
        "model": _coefficient_diagnostics(
            inner_predictions[nonoutage_rows],
            inner_coefficients[nonoutage_rows],
        ),
        "nearest1": _coefficient_diagnostics(
            spatial_predictions["nearest1"][nonoutage_rows],
            inner_coefficients[nonoutage_rows],
        ),
        "mean16": _coefficient_diagnostics(
            spatial_predictions["mean16"][nonoutage_rows],
            inner_coefficients[nonoutage_rows],
        ),
    }
    selected_alpha, selected_inner = max(
        alpha_metrics.items(), key=lambda item: float(item[1]["score"])
    )
    inner_baseline = alpha_metrics["0.0"]
    inner_gain = float(selected_inner["score"]) - float(inner_baseline["score"])
    inner_oracle_gain = (
        float(inner_oracle["metrics"]["score"]) - float(inner_baseline["score"])
    )
    average_probe_passed = (
        inner_gain >= float(args.minimum_inner_gain) and float(selected_alpha) > 0
    )
    expert_probe_passed = inner_oracle_gain >= 0.010
    if average_probe_passed:
        inner_status = "INNER_AVERAGE_PASS"
    elif expert_probe_passed:
        inner_status = "INNER_EXPERT_PASS"
    else:
        inner_status = "DROP"
    report: dict[str, object] = {
        "status": inner_status,
        "hypothesis": "A local-set model can predict rank-16 spectrum residual coefficients.",
        "fold": fold,
        "holdout_spectral_fold": holdout_fold,
        "inner_basis": "fit from inner_training only",
        "rank": int(args.rank),
        "neighbors": int(args.neighbors),
        "inner": {
            "cells": inner_reports,
            "alpha_scan": alpha_metrics,
            "selected_alpha": float(selected_alpha),
            "baseline_score": float(inner_baseline["score"]),
            "selected_score": float(selected_inner["score"]),
            "gain": inner_gain,
            "minimum_gain": float(args.minimum_inner_gain),
            "spatial_diagnostic_metrics": spatial_metrics,
            "coefficient_diagnostics": coefficient_diagnostics,
            "target_informed_candidate_oracle": inner_oracle,
            "target_informed_candidate_oracle_gain": inner_oracle_gain,
            "average_probe_passed": average_probe_passed,
            "expert_probe_passed": expert_probe_passed,
        },
        "strict_fold0": None,
        "decision": (
            "PENDING_FULL_PROBE"
            if average_probe_passed
            else "PENDING_STRICT_EXPERT_ORACLE"
            if expert_probe_passed
            else "DROP"
        ),
    }
    if not (average_probe_passed or expert_probe_passed):
        report["elapsed_seconds"] = time.perf_counter() - started
        save_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    strict_predictions = np.zeros(
        (len(metadata["train_cells"]), int(args.rank)), dtype=np.float32
    )
    full_coefficients = _coefficients(
        target_cache,
        seed_cache,
        bases,
        metadata,
        nonoutage_observed,
        int(args.rank),
    )
    full_reports = []
    for cell in np.unique(metadata["train_cells"]):
        cell_train = nonoutage_observed[
            metadata["train_cells"][nonoutage_observed] == cell
        ]
        cell_validation = validation[metadata["train_cells"][validation] == cell]
        statistics = _fit_statistics(
            query_raw,
            seed_cache["spectrum"],
            full_coefficients,
            metadata,
            cell_train,
        )
        train_neighbors = _fold_excluded_neighbors(
            metadata, cell_train, int(args.neighbors)
        )
        validation_neighbors = _tree_neighbors(
            metadata["train_positions"],
            cell_train,
            cell_validation,
            int(args.neighbors),
        )
        training_arrays = _model_arrays(
            cell_train,
            train_neighbors,
            query_raw,
            seed_cache,
            full_coefficients,
            metadata,
            priors,
            statistics,
            spectrum_shape,
        )
        validation_arrays = _model_arrays(
            cell_validation,
            validation_neighbors,
            query_raw,
            seed_cache,
            full_coefficients,
            metadata,
            priors,
            statistics,
            spectrum_shape,
        )
        seed_everything(int(args.seed) + 100 + int(cell))
        model = _build_model(
            training_arrays,
            spectrum_shape,
            int(args.rank),
            int(args.width),
            float(args.dropout),
            device,
        )
        state, _, history = _train(
            model,
            training_arrays,
            None,
            device,
            epochs=best_epochs[int(cell)],
            batch_size=int(args.batch_size),
            validation_batch_size=int(args.validation_batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            seed=int(args.seed) + 100 + int(cell),
            label=f"full-cell{int(cell)}",
        )
        model.load_state_dict(state)
        normalized_prediction, effective = _predict(
            model,
            validation_arrays,
            device,
            int(args.validation_batch_size),
        )
        strict_predictions[cell_validation] = _restore_coefficients(
            normalized_prediction, statistics
        )
        checkpoint = output_dir / f"full_cell{int(cell)}.pt"
        torch.save(
            {
                "model": state,
                "statistics": statistics,
                "epochs": best_epochs[int(cell)],
                "history": history,
                "cell": int(cell),
                "settings": vars(args),
                "basis_path": str(cache_dir / "train_only_basis.pt"),
            },
            checkpoint,
        )
        full_reports.append(
            {
                "cell": int(cell),
                "training_samples": int(len(cell_train)),
                "epochs": best_epochs[int(cell)],
                "effective_neighbors": effective,
                "checkpoint": str(checkpoint),
            }
        )

    baseline_arrays = _baseline_arrays(
        args.baseline_prediction,
        validation,
        metadata,
        channels,
        shape,
        device,
        int(args.decode_batch_size),
    )
    baseline_metrics = aggregate_sample_metrics(baseline_arrays)
    strict_candidate_arrays = {"baseline": baseline_arrays}
    strict_candidate_metrics = {"baseline": baseline_metrics}
    strict_candidate_inputs: dict[str, tuple[np.ndarray, float]] = {}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        name = f"model_alpha_{alpha}"
        metrics, arrays = _decode_metrics(
            validation,
            strict_predictions[validation],
            seed_cache,
            bases,
            metadata,
            priors,
            channels,
            autoencoder,
            shape,
            device,
            int(args.decode_batch_size),
            alpha,
            outage_thresholds,
            outage_strengths,
        )
        strict_candidate_metrics[name] = metrics
        strict_candidate_arrays[name] = arrays
        strict_candidate_inputs[name] = (strict_predictions[validation], alpha)
    strict_spatial_predictions = {
        "nearest1": _neighbor_coefficient_prediction(
            metadata,
            nonoutage_observed,
            validation,
            full_coefficients,
            1,
        ),
        "mean16": _neighbor_coefficient_prediction(
            metadata,
            nonoutage_observed,
            validation,
            full_coefficients,
            int(args.neighbors),
        ),
    }
    for source, prediction in strict_spatial_predictions.items():
        for alpha in (0.25, 0.5, 1.0):
            name = f"{source}_alpha_{alpha}"
            metrics, arrays = _decode_metrics(
                validation,
                prediction[validation],
                seed_cache,
                bases,
                metadata,
                priors,
                channels,
                autoencoder,
                shape,
                device,
                int(args.decode_batch_size),
                alpha,
                outage_thresholds,
                outage_strengths,
            )
            strict_candidate_metrics[name] = metrics
            strict_candidate_arrays[name] = arrays
            strict_candidate_inputs[name] = (prediction[validation], alpha)
    expert_oracle = target_informed_expert_oracle(
        strict_candidate_arrays
    )
    expert_oracle.pop("selection")
    best_single_name, best_single_metrics = max(
        strict_candidate_metrics.items(),
        key=lambda item: float(item[1]["score"]),
    )
    best_single_arrays = strict_candidate_arrays[best_single_name]
    baseline_score = float(baseline_metrics["score"])
    delta = float(best_single_metrics["score"]) - baseline_score
    oracle_gain = float(expert_oracle["metrics"]["score"]) - baseline_score
    strict_output: str | None = str(args.baseline_prediction)
    if best_single_name != "baseline":
        output_path = output_dir / "Fold0_Residual_Prediction.npy"
        coefficients_for_output, alpha_for_output = strict_candidate_inputs[
            best_single_name
        ]
        _decode_metrics(
            validation,
            coefficients_for_output,
            seed_cache,
            bases,
            metadata,
            priors,
            channels,
            autoencoder,
            shape,
            device,
            int(args.decode_batch_size),
            alpha_for_output,
            outage_thresholds,
            outage_strengths,
            output_path,
        )
        strict_output = str(output_path)
    if delta >= 0.004:
        decision = "PROMOTE"
    elif oracle_gain >= 0.010:
        decision = "KEEP_AS_EXPERT"
    elif delta >= 0.001:
        decision = "MODIFY_ONCE"
    else:
        decision = "DROP"
    np.savez_compressed(
        output_dir / "Fold0_Per_Sample_Metrics.npz", **best_single_arrays
    )
    report.update(
        {
            "status": "PASS",
            "full_cells": full_reports,
            "strict_fold0": {
                "best_single_candidate": best_single_name,
                "metrics": best_single_metrics,
                "candidate_metrics": strict_candidate_metrics,
                "authoritative_baseline": baseline_score,
                "delta": delta,
                "prediction": strict_output,
                "expert_oracle": expert_oracle,
                "expert_oracle_gain": oracle_gain,
            },
            "decision": decision,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
