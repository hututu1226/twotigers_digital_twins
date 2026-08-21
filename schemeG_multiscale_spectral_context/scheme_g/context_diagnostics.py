from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .angle_delay import ChannelShape
from .config import save_json
from .context_data import ContextRepository
from .context_model import FullResolutionContextField
from .context_training import (
    Autoencoder,
    _decode_predictions,
    build_device_cache,
    predict_indices,
)
from .metrics import ChannelMetricAccumulator


PREDICTION_KEYS = ("spectrum", "phase", "power", "outage_probability")


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            name: 0.0
            for name in ("minimum", "p25", "median", "p75", "p95", "maximum", "mean")
        }
    points = np.percentile(array, [0, 25, 50, 75, 95, 100])
    return {
        **{
            name: float(value)
            for name, value in zip(
                ("minimum", "p25", "median", "p75", "p95", "maximum"),
                points,
            )
        },
        "mean": float(array.mean()),
    }


def nearest_observed_indices(
    positions: np.ndarray,
    cell_ids: np.ndarray,
    observed_indices: np.ndarray,
    target_indices: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed_indices, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    count = max(1, int(neighbors))
    selected = np.empty((len(targets), count), dtype=np.int64)
    distances = np.empty((len(targets), count), dtype=np.float32)
    xy = np.asarray(positions, dtype=np.float32)[:, :2]
    cells = np.asarray(cell_ids)
    for cell_id in np.unique(cells[targets]):
        local_targets = np.flatnonzero(cells[targets] == cell_id)
        references = observed[cells[observed] == cell_id]
        if len(references) < count:
            raise ValueError(
                f"Cell {int(cell_id)} has {len(references)} observations, fewer than k={count}"
            )
        square = (
            (xy[targets[local_targets], None, :] - xy[references][None, :, :]) ** 2
        ).sum(axis=2)
        candidates = np.argpartition(square, kth=count - 1, axis=1)[:, :count]
        candidate_square = np.take_along_axis(square, candidates, axis=1)
        order = np.argsort(candidate_square, axis=1)
        candidates = np.take_along_axis(candidates, order, axis=1)
        candidate_square = np.take_along_axis(candidate_square, order, axis=1)
        selected[local_targets] = references[candidates]
        distances[local_targets] = np.sqrt(candidate_square).astype(np.float32)
    return selected, distances


def hybrid_predictions(
    predicted: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    replacements: tuple[str, ...] = (),
    oracle_outage: bool = False,
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(predicted[key]) for key in PREDICTION_KEYS}
    for key in replacements:
        if key not in ("spectrum", "phase", "power"):
            raise ValueError(f"Unsupported oracle component: {key}")
        result[key] = np.asarray(target[key])
    if oracle_outage:
        result["outage_probability"] = np.asarray(target["outage"], dtype=np.float32)
    return result


def routing_summary(outputs: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    names = (
        "router_entropy",
        "router_top1_mass",
        "router_effective_neighbors",
        "router_distance",
        "spectrum_warp",
        "detail_warp",
    )
    return {
        name: percentile_summary(outputs[name]) for name in names if name in outputs
    }


@torch.no_grad()
def evaluate_prediction_arrays(
    outputs: dict[str, np.ndarray],
    autoencoder: Autoencoder,
    repository: ContextRepository,
    target_indices: np.ndarray,
    shape: ChannelShape,
    target_channels: np.ndarray,
    device: torch.device,
    outage_threshold: float,
    decode_batch_size: int,
) -> dict[str, float]:
    indices = np.asarray(target_indices, dtype=np.int64)
    metadata = repository.metadata
    cell_ids = metadata["train_cells"][indices]
    true_outage = metadata["outage"][indices]
    target = repository.target_values(indices)
    nonzero = ~true_outage
    spectrum_mse = (
        float(
            np.mean((outputs["spectrum"][nonzero] - target["spectrum"][nonzero]) ** 2)
        )
        if np.any(nonzero)
        else 0.0
    )
    phase_mse = (
        float(np.mean((outputs["phase"][nonzero] - target["phase"][nonzero]) ** 2))
        if np.any(nonzero)
        else 0.0
    )
    power_error = outputs["power"][nonzero] - target["power"][nonzero]
    log_power_error = power_error * repository.encoded["power_std"][cell_ids[nonzero]]
    metrics = ChannelMetricAccumulator(shape)
    batch_size = max(1, int(decode_batch_size))
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        selected = slice(start, stop)
        predictions = {
            key: torch.from_numpy(np.asarray(outputs[key][selected])).to(device)
            for key in ("spectrum", "phase", "power")
        }
        batch_cells = cell_ids[selected]
        prediction = torch.empty(
            (stop - start, *shape.raw_shape),
            dtype=torch.complex64,
            device=device,
        )
        for cell_id in np.unique(batch_cells):
            local = np.flatnonzero(batch_cells == cell_id)
            local_tensor = torch.from_numpy(local).to(device=device, dtype=torch.long)
            local_outputs = {
                key: value.index_select(0, local_tensor)
                for key, value in predictions.items()
            }
            decoded = _decode_predictions(
                local_outputs,
                int(cell_id),
                repository,
                autoencoder,
                shape,
            )
            prediction.index_copy_(0, local_tensor, decoded)
        predicted_outage = outputs["outage_probability"][selected] >= outage_threshold
        outage_tensor = torch.from_numpy(predicted_outage).to(device)
        prediction = prediction.masked_fill(outage_tensor[:, None, None, None], 0.0)
        target_channel = torch.from_numpy(
            np.ascontiguousarray(target_channels[selected])
        ).to(device)
        true_outage_tensor = torch.from_numpy(true_outage[selected]).to(device)
        metrics.update(prediction, target_channel, true_outage_tensor)

    predicted_outage = outputs["outage_probability"] >= outage_threshold
    true_positive = int(np.sum(predicted_outage & true_outage))
    false_positive = int(np.sum(predicted_outage & ~true_outage))
    false_negative = int(np.sum(~predicted_outage & true_outage))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    report = metrics.compute()
    report.update(
        {
            "outage_threshold": float(outage_threshold),
            "spectrum_latent_mse_z": spectrum_mse,
            "phase_latent_mse_z": phase_mse,
            "power_mae_z": float(np.mean(np.abs(power_error)))
            if len(power_error)
            else 0.0,
            "power_rmse_z": float(np.sqrt(np.mean(power_error**2)))
            if len(power_error)
            else 0.0,
            "power_mae_log10": (
                float(np.mean(np.abs(log_power_error))) if len(log_power_error) else 0.0
            ),
            "outage_accuracy": float(np.mean(predicted_outage == true_outage)),
            "outage_precision": precision,
            "outage_recall": recall,
            "outage_f1": 2.0 * precision * recall / max(precision + recall, 1e-30),
            "predicted_outages": int(predicted_outage.sum()),
            "samples": int(len(indices)),
        }
    )
    return report


@torch.no_grad()
def evaluate_channel_copy(
    nearest_indices: np.ndarray,
    target_indices: np.ndarray,
    true_outage: np.ndarray,
    shape: ChannelShape,
    channels: np.ndarray,
    target_channels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    metrics = ChannelMetricAccumulator(shape)
    nearest = np.asarray(nearest_indices, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    size = max(1, int(batch_size))
    for start in range(0, len(targets), size):
        stop = min(start + size, len(targets))
        prediction = torch.from_numpy(
            np.array(channels[nearest[start:stop]], copy=True)
        ).to(device)
        target = torch.from_numpy(np.ascontiguousarray(target_channels[start:stop])).to(
            device
        )
        outage = torch.from_numpy(np.asarray(true_outage[start:stop])).to(device)
        metrics.update(prediction, target, outage)
    report = metrics.compute()
    report["samples"] = int(len(targets))
    return report


def write_markdown_summary(path: str | Path, report: dict) -> None:
    baseline = report["baseline"]["metrics"]
    lines = [
        "# Scheme G Context Diagnostic Report",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Baseline",
        "",
        "| PAS | PDP | NMSE | Score |",
        "|---:|---:|---:|---:|",
        (
            f"| {baseline['pas']:.6f} | {baseline['pdp']:.6f} | "
            f"{baseline['nmse']:.6f} | {baseline['score']:.6f} |"
        ),
        "",
        "## Oracle component replacement",
        "",
        "| Variant | Score | Delta | PAS | PDP | NMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["oracle_replacements"].items():
        metrics = values["metrics"]
        lines.append(
            f"| {name} | {metrics['score']:.6f} | {values['score_delta']:+.6f} | "
            f"{metrics['pas']:.6f} | {metrics['pdp']:.6f} | {metrics['nmse']:.6f} |"
        )
    signals = report["decision_signals"]
    lines.extend(
        [
            "",
            "## Spatial and power baselines",
            "",
            "| Variant | Score | Delta |",
            "|---|---:|---:|",
        ]
    )
    for name, values in report["spatial_baselines"].items():
        metrics = values["metrics"]
        lines.append(
            f"| {name} | {metrics['score']:.6f} | {values['score_delta']:+.6f} |"
        )
    if report.get("counterfactuals"):
        lines.extend(
            [
                "",
                "## Router and warp counterfactuals",
                "",
                "| Variant | Score | Delta |",
                "|---|---:|---:|",
            ]
        )
        for name, values in report["counterfactuals"].items():
            metrics = values["metrics"]
            lines.append(
                f"| {name} | {metrics['score']:.6f} | {values['score_delta']:+.6f} |"
            )
    lines.extend(
        [
            "",
            "## Decision signals",
            "",
            f"- Primary single-component oracle: `{signals['primary_oracle_component']}`.",
            f"- Largest single-component gain: `{signals['primary_oracle_gain']:+.6f}`.",
            f"- Nearest-latent score: `{signals['nearest_latent_score']:.6f}`.",
            f"- Decoder oracle score: `{signals['decoder_oracle_score']:.6f}`.",
            (
                "- Counterfactual inference was not retrained; treat its score "
                "deltas as diagnostic evidence, not final model estimates."
            ),
            "",
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostics(
    config: dict,
    checkpoint_path: str | Path,
    repository: ContextRepository,
    model: FullResolutionContextField,
    autoencoder: Autoencoder,
    shape: ChannelShape,
    checkpoint: dict,
    validation_indices: np.ndarray,
    device: torch.device,
    amp: bool,
    output_dir: str | Path,
    outage_threshold: float,
    decode_batch_size: int,
    include_counterfactuals: bool = True,
) -> dict:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
        mmap_mode="r",
    )
    print(
        "Loading the validation channels once to avoid repeated disk reads...",
        flush=True,
    )
    target_channels = np.array(channels[validation_indices], copy=True)
    report: dict = {
        "status": "RUNNING",
        "config": str(config.get("_config_path", "")),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "device": str(device),
        "validation_samples": int(len(validation_indices)),
        "outage_threshold": float(outage_threshold),
        "notes": [
            "Oracle variants replace only the named predicted component with validation truth.",
            "Nearest-neighbor variants are diagnostic rulers, not proposed final algorithms.",
            "Counterfactual model settings are inference-only and were not retrained.",
        ],
    }
    save_json(report_path, report)

    device_cache = build_device_cache(repository, device)
    model.set_diagnostic_ablation(disable_warp=False, route_bias_scale=1.0)
    print("[1/4] Predicting the trained Context baseline...", flush=True)
    predicted = predict_indices(
        model,
        repository,
        validation_indices,
        device,
        amp,
        device_cache=device_cache,
    )
    target = repository.target_values(validation_indices)
    baseline_metrics = evaluate_prediction_arrays(
        predicted,
        autoencoder,
        repository,
        validation_indices,
        shape,
        target_channels,
        device,
        outage_threshold,
        decode_batch_size,
    )
    baseline_score = float(baseline_metrics["score"])
    report["baseline"] = {
        "metrics": baseline_metrics,
        "routing": routing_summary(predicted),
    }
    save_json(report_path, report)

    print(
        "[2/4] Replacing Spectrum, Detail, Power, and outage with truth...", flush=True
    )
    oracle_definitions = {
        "oracle_spectrum": (("spectrum",), False),
        "oracle_detail": (("phase",), False),
        "oracle_power": (("power",), False),
        "oracle_spectrum_detail": (("spectrum", "phase"), False),
        "oracle_spectrum_power": (("spectrum", "power"), False),
        "oracle_detail_power": (("phase", "power"), False),
        "oracle_all_components": (("spectrum", "phase", "power"), False),
        "oracle_all_including_outage": (("spectrum", "phase", "power"), True),
    }
    oracle_reports: dict[str, dict] = {}
    for name, (components, oracle_outage) in oracle_definitions.items():
        print(f"  evaluating {name}", flush=True)
        values = hybrid_predictions(predicted, target, components, oracle_outage)
        metrics = evaluate_prediction_arrays(
            values,
            autoencoder,
            repository,
            validation_indices,
            shape,
            target_channels,
            device,
            outage_threshold,
            decode_batch_size,
        )
        oracle_reports[name] = {
            "replaced": list(components) + (["outage"] if oracle_outage else []),
            "metrics": metrics,
            "score_delta": float(metrics["score"] - baseline_score),
        }
        report["oracle_replacements"] = oracle_reports
        save_json(report_path, report)

    print(
        "[3/4] Measuring nearest-latent and low-dimensional power baselines...",
        flush=True,
    )
    metadata = repository.metadata
    nearest, nearest_distance = nearest_observed_indices(
        metadata["train_positions"],
        metadata["train_cells"],
        repository.observed_indices,
        validation_indices,
        neighbors=1,
    )
    nearest_flat = nearest[:, 0]
    nearest_outputs = {
        "spectrum": repository.spectrum_z[nearest_flat],
        "phase": repository.phase_z[nearest_flat],
        "power": repository.power_z[nearest_flat],
        "outage_probability": metadata["outage"][nearest_flat].astype(np.float32),
    }
    spatial_reports: dict[str, dict] = {}

    def add_spatial(name: str, values: dict[str, np.ndarray]) -> None:
        metrics = evaluate_prediction_arrays(
            values,
            autoencoder,
            repository,
            validation_indices,
            shape,
            target_channels,
            device,
            outage_threshold,
            decode_batch_size,
        )
        spatial_reports[name] = {
            "metrics": metrics,
            "score_delta": float(metrics["score"] - baseline_score),
        }

    add_spatial("nearest_latent", nearest_outputs)
    add_spatial(
        "nearest_latent_oracle_power",
        hybrid_predictions(nearest_outputs, target, ("power",), False),
    )
    direct_metrics = evaluate_channel_copy(
        nearest_flat,
        validation_indices,
        metadata["outage"][validation_indices],
        shape,
        channels,
        target_channels,
        device,
        decode_batch_size,
    )
    spatial_reports["nearest_channel_copy"] = {
        "metrics": direct_metrics,
        "score_delta": float(direct_metrics["score"] - baseline_score),
    }

    nonoutage_observed = repository.observed_indices[
        ~metadata["outage"][repository.observed_indices]
    ]
    validation_cells = np.unique(metadata["train_cells"][validation_indices])
    power_neighbor_count = min(
        8,
        *(
            int(np.sum(metadata["train_cells"][nonoutage_observed] == cell_id))
            for cell_id in validation_cells
        ),
    )
    if power_neighbor_count < 1:
        raise ValueError(
            "Every validation cell needs at least one non-outage observation"
        )
    power_neighbors, power_distance = nearest_observed_indices(
        metadata["train_positions"],
        metadata["train_cells"],
        nonoutage_observed,
        validation_indices,
        neighbors=power_neighbor_count,
    )
    nearest_power = repository.power_z[power_neighbors[:, 0]]
    inverse_square = 1.0 / np.maximum(power_distance, 0.25) ** 2
    inverse_square /= inverse_square.sum(axis=1, keepdims=True)
    idw_power = (repository.power_z[power_neighbors] * inverse_square).sum(axis=1)
    nearest_power_outputs = hybrid_predictions(predicted, target)
    nearest_power_outputs["power"] = nearest_power.astype(np.float32)
    add_spatial("context_latent_nearest_power", nearest_power_outputs)
    idw_power_outputs = hybrid_predictions(predicted, target)
    idw_power_outputs["power"] = idw_power.astype(np.float32)
    add_spatial("context_latent_idw8_power", idw_power_outputs)
    report["spatial_baselines"] = spatial_reports
    report["nearest_distance_meters"] = percentile_summary(nearest_distance[:, 0])
    report["power_neighbors_used"] = int(power_neighbor_count)
    report["power_neighbor_distance_meters"] = percentile_summary(power_distance)
    save_json(report_path, report)

    counterfactual_reports: dict[str, dict] = {}
    if include_counterfactuals:
        print(
            "[4/4] Running no-Warp and weaker Router-prior counterfactuals...",
            flush=True,
        )
        definitions = {
            "no_warp": (True, 1.0),
            "weak_router_prior_0_25": (False, 0.25),
            "no_router_prior": (False, 0.0),
        }
        try:
            for name, (disable_warp, route_bias_scale) in definitions.items():
                print(f"  evaluating {name}", flush=True)
                model.set_diagnostic_ablation(
                    disable_warp=disable_warp,
                    route_bias_scale=route_bias_scale,
                )
                values = predict_indices(
                    model,
                    repository,
                    validation_indices,
                    device,
                    amp,
                    device_cache=device_cache,
                )
                metrics = evaluate_prediction_arrays(
                    values,
                    autoencoder,
                    repository,
                    validation_indices,
                    shape,
                    target_channels,
                    device,
                    outage_threshold,
                    decode_batch_size,
                )
                counterfactual_reports[name] = {
                    "settings": {
                        "disable_warp": disable_warp,
                        "route_bias_scale": route_bias_scale,
                    },
                    "metrics": metrics,
                    "routing": routing_summary(values),
                    "score_delta": float(metrics["score"] - baseline_score),
                }
                report["counterfactuals"] = counterfactual_reports
                save_json(report_path, report)
        finally:
            model.set_diagnostic_ablation(disable_warp=False, route_bias_scale=1.0)
    else:
        print("[4/4] Counterfactuals skipped by request.", flush=True)
        report["counterfactuals"] = {}

    single_oracles = {
        name: values["score_delta"]
        for name, values in oracle_reports.items()
        if name in ("oracle_spectrum", "oracle_detail", "oracle_power")
    }
    primary_oracle = max(single_oracles, key=single_oracles.get)
    report["decision_signals"] = {
        "primary_oracle_component": primary_oracle,
        "primary_oracle_gain": float(single_oracles[primary_oracle]),
        "single_component_gains": single_oracles,
        "nearest_latent_score": float(
            spatial_reports["nearest_latent"]["metrics"]["score"]
        ),
        "nearest_channel_copy_score": float(
            spatial_reports["nearest_channel_copy"]["metrics"]["score"]
        ),
        "decoder_oracle_score": float(
            oracle_reports["oracle_all_including_outage"]["metrics"]["score"]
        ),
        "counterfactual_score_deltas": {
            name: values["score_delta"]
            for name, values in counterfactual_reports.items()
        },
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "PASS"
    save_json(report_path, report)
    write_markdown_summary(output / "SUMMARY.md", report)
    return report
