from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from scheme_e.angle_delay import shape_to_channel
from scheme_e.complex_residual import replace_angle_delay_log_power
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


def _load_latent_cache(path: Path) -> dict[str, np.ndarray]:
    return {
        name: np.load(path / f"teacher_seed_{name}.npy", mmap_mode="r")
        for name in ("spectrum", "detail", "log_power", "outage")
    }


@torch.no_grad()
def _decode_seed(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    autoencoder: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        return autoencoder.decode(
            torch.as_tensor(
                np.asarray(cache["spectrum"][indices], dtype=np.float32),
                device=device,
            ),
            torch.as_tensor(
                np.asarray(cache["detail"][indices], dtype=np.float32),
                device=device,
            ),
        ).float()


@torch.no_grad()
def _evaluate_transfer(
    indices: np.ndarray,
    neighbors: np.ndarray,
    distances: np.ndarray,
    count: int,
    strength: float,
    base_cache: np.ndarray,
    target_cache: np.ndarray,
    latent_cache: dict[str, np.ndarray],
    channels: np.ndarray,
    metadata: dict[str, np.ndarray],
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    batch_size: int,
    log_power_scale: float,
    aligned: bool,
) -> dict[str, np.ndarray]:
    batches = []
    for start in range(0, len(indices), int(batch_size)):
        stop = min(start + int(batch_size), len(indices))
        selected = indices[start:stop]
        local_neighbors = neighbors[start:stop]
        query_base = np.array(base_cache[selected], copy=True)
        neighbor_base = np.array(base_cache[local_neighbors], copy=True)
        shifts = (
            estimate_magnitude_profile_shifts(
                query_base,
                neighbor_base,
                scale=float(log_power_scale),
            )
            if bool(aligned) and float(strength) != 0.0
            else None
        )
        predicted_log = transfer_log_power_residual(
            query_base,
            neighbor_base,
            np.array(target_cache[local_neighbors], copy=True),
            distances[start:stop],
            count=int(count),
            strength=float(strength),
            shifts=shifts,
        )
        seed = _decode_seed(latent_cache, selected, autoencoder, device)
        predicted_shape = replace_angle_delay_log_power(
            seed,
            torch.as_tensor(predicted_log, device=device).reshape(
                len(selected),
                shape.m_p,
                shape.n,
                shape.m_v,
                shape.m_h,
                shape.s,
            ),
            shape,
            float(log_power_scale),
        )
        prediction = shape_to_channel(
            predicted_shape,
            torch.as_tensor(
                np.asarray(latent_cache["log_power"][selected], dtype=np.float32),
                device=device,
            ),
            shape,
            torch.as_tensor(
                np.asarray(latent_cache["outage"][selected], dtype=bool),
                device=device,
            ),
        )
        target = torch.as_tensor(
            np.array(channels[selected], copy=True), device=device
        )
        batches.append(
            sample_metric_batch(
                prediction,
                target,
                shape,
                metadata["outage"][selected].astype(bool),
            )
        )
    return concatenate_metric_batches(batches)


@torch.no_grad()
def _evaluate_saved_prediction(
    path: str | Path,
    validation: np.ndarray,
    channels: np.ndarray,
    metadata: dict[str, np.ndarray],
    shape: object,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    prediction = np.load(path, mmap_mode="r")
    if len(prediction) != len(validation):
        raise ValueError("Fold0 baseline row count is inconsistent")
    batches = []
    for start in range(0, len(validation), int(batch_size)):
        stop = min(start + int(batch_size), len(validation))
        selected = validation[start:stop]
        batches.append(
            sample_metric_batch(
                torch.as_tensor(
                    np.array(prediction[start:stop], copy=True), device=device
                ),
                torch.as_tensor(
                    np.array(channels[selected], copy=True), device=device
                ),
                shape,
                metadata["outage"][selected].astype(bool),
            )
        )
    return concatenate_metric_batches(batches)


def _merge_by_cell(
    arrays: dict[str, dict[str, np.ndarray]],
    names_by_cell: dict[int, str],
    cells: np.ndarray,
) -> dict[str, np.ndarray]:
    output = {}
    for field in next(iter(arrays.values())):
        merged = np.empty_like(next(iter(arrays.values()))[field])
        for cell, name in names_by_cell.items():
            selected = cells == int(cell)
            merged[selected] = arrays[name][field][selected]
        output[field] = merged
    return output


def _candidate_name(count: int, strength: float) -> str:
    return f"k{int(count)}_a{float(strength):g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe local full-resolution magnitude residual diagnostic"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--latent-cache", default="../research/scheme_e_065/residual_rank"
    )
    parser.add_argument(
        "--map-cache", default="artifacts/scheme_e_065/fullres_log_power_cache"
    )
    parser.add_argument(
        "--baseline-prediction",
        default="../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy",
    )
    parser.add_argument(
        "--report", default="../research/scheme_e_065/L0_012_LOCAL_MAGNITUDE_TRANSFER.json"
    )
    parser.add_argument(
        "--extra-expert-prediction",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional saved Fold0 prediction to include in the diagnostic oracle. "
            "May be supplied more than once."
        ),
    )
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--strengths", type=float, nargs="+", default=[0.25, 0.5, 1.0]
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-power-scale", type=float, default=4.0)
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--experiment-id", default="L0-012")
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    args = parser.parse_args()
    started = time.perf_counter()
    device = choose_device(args.device)
    config = load_config(args.config)
    metadata = _load_npz(
        Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz"
    )
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    checkpoint = Path(config["hybrid"]["output_dir"]) / "best.pt"
    hybrid, shape, _ = load_hybrid_checkpoint(config, checkpoint, device)
    autoencoder = hybrid.autoencoder.eval()
    latent_cache = _load_latent_cache(Path(args.latent_cache))
    base_cache = np.load(
        Path(args.map_cache) / "teacher_log_power.npy", mmap_mode="r"
    )
    target_cache = np.load(
        Path(args.map_cache) / "target_log_power.npy", mmap_mode="r"
    )

    fold = int(config["split"]["validation_fold"])
    available = priors["available"].astype(bool)
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    observed = np.flatnonzero(available & ~validation_mask)
    validation = np.flatnonzero(available & validation_mask)
    observed_nonoutage = observed[~metadata["outage"][observed].astype(bool)]
    holdout_fold = int(np.max(metadata["spectral_folds"][observed_nonoutage]))
    inner_support = observed_nonoutage[
        metadata["spectral_folds"][observed_nonoutage] != holdout_fold
    ]
    inner_validation = observed[
        metadata["spectral_folds"][observed] == holdout_fold
    ]
    maximum_count = max(int(value) for value in args.counts)
    inner_neighbors, inner_distances = same_cell_neighbors(
        metadata["train_positions"],
        metadata["train_cells"],
        inner_support,
        inner_validation,
        maximum_count,
    )
    strict_neighbors, strict_distances = same_cell_neighbors(
        metadata["train_positions"],
        metadata["train_cells"],
        observed_nonoutage,
        validation,
        maximum_count,
    )

    specs = [(1, 0.0)] + [
        (int(count), float(strength))
        for count in args.counts
        for strength in args.strengths
    ]
    inner_arrays = {}
    for count, strength in specs:
        name = "teacher_seed" if strength == 0.0 else _candidate_name(count, strength)
        inner_arrays[name] = _evaluate_transfer(
            inner_validation,
            inner_neighbors,
            inner_distances,
            count,
            strength,
            base_cache,
            target_cache,
            latent_cache,
            channels,
            metadata,
            autoencoder,
            shape,
            device,
            int(args.batch_size),
            float(args.log_power_scale),
            bool(args.aligned),
        )
        metrics = aggregate_sample_metrics(inner_arrays[name])
        print(
            f"inner {name}: score={metrics['score']:.6f} pas={metrics['pas']:.6f} "
            f"pdp={metrics['pdp']:.6f} nmse={metrics['nmse']:.6f}",
            flush=True,
        )

    candidate_names = [name for name in inner_arrays if name != "teacher_seed"]
    selected_global = max(
        candidate_names,
        key=lambda name: float(aggregate_sample_metrics(inner_arrays[name])["score"]),
    )
    inner_cells = metadata["train_cells"][inner_validation].astype(np.int64)
    selected_per_cell = {
        int(cell): max(
            candidate_names,
            key=lambda name: float(
                aggregate_sample_metrics(
                    {field: value[inner_cells == cell] for field, value in inner_arrays[name].items()}
                )["score"]
            ),
        )
        for cell in np.unique(inner_cells)
    }
    selected_names = sorted(set([selected_global, *selected_per_cell.values()]))
    strict_arrays = {}
    for name in ["teacher_seed", *selected_names]:
        count, strength = (1, 0.0) if name == "teacher_seed" else (
            int(name.split("_")[0][1:]),
            float(name.split("_")[1][1:]),
        )
        strict_arrays[name] = _evaluate_transfer(
            validation,
            strict_neighbors,
            strict_distances,
            count,
            strength,
            base_cache,
            target_cache,
            latent_cache,
            channels,
            metadata,
            autoencoder,
            shape,
            device,
            int(args.batch_size),
            float(args.log_power_scale),
            bool(args.aligned),
        )
    v4_arrays = _evaluate_saved_prediction(
        args.baseline_prediction,
        validation,
        channels,
        metadata,
        shape,
        device,
        int(args.batch_size),
    )
    extra_experts = {}
    extra_expert_paths = {}
    for specification in args.extra_expert_prediction:
        if "=" not in specification:
            raise ValueError(
                "--extra-expert-prediction must use the NAME=PATH format"
            )
        name, path = specification.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(
                "--extra-expert-prediction must contain a non-empty name and path"
            )
        if name in {"v4", "local_transfer"} or name in extra_experts:
            raise ValueError(f"Duplicate diagnostic expert name: {name}")
        extra_experts[name] = _evaluate_saved_prediction(
            path,
            validation,
            channels,
            metadata,
            shape,
            device,
            int(args.batch_size),
        )
        extra_expert_paths[name] = path
    strict_cells = metadata["train_cells"][validation].astype(np.int64)
    per_cell_arrays = _merge_by_cell(strict_arrays, selected_per_cell, strict_cells)
    strict_global_metrics = aggregate_sample_metrics(strict_arrays[selected_global])
    strict_per_cell_metrics = aggregate_sample_metrics(per_cell_arrays)
    v4_metrics = aggregate_sample_metrics(v4_arrays)
    deployable_name, deployable_metrics, deployable_arrays = max(
        [
            ("inner_global", strict_global_metrics, strict_arrays[selected_global]),
            ("inner_per_cell", strict_per_cell_metrics, per_cell_arrays),
        ],
        key=lambda item: float(item[1]["score"]),
    )
    oracle = target_informed_expert_oracle(
        {
            "v4": v4_arrays,
            "local_transfer": deployable_arrays,
            **extra_experts,
        }
    )
    strict_gain = float(deployable_metrics["score"]) - float(v4_metrics["score"])
    oracle_gain = float(oracle["metrics"]["score"]) - float(v4_metrics["score"])
    decision = (
        "PROMOTE"
        if strict_gain >= 0.004
        else "KEEP_AS_EXPERT"
        if oracle_gain >= 0.010
        else "DROP"
    )
    report = {
        "status": "COMPLETED",
        "diagnostic_only_oracle": True,
        "hypothesis": (
            "Teacher-profile alignment makes nearby full-resolution OOF magnitude "
            "residuals transferable without destroying angular structure."
            if bool(args.aligned)
            else "Full-resolution OOF Teacher magnitude residuals are locally transferable "
            "within each base station and can expose useful neighborhood context."
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_bootstrap.PROJECT_ROOT.parent, text=True
        ).strip(),
        "split": {
            "fold": fold,
            "holdout_spectral_fold": holdout_fold,
            "inner_support": int(len(inner_support)),
            "inner_validation": int(len(inner_validation)),
            "strict_support": int(len(observed_nonoutage)),
            "strict_validation": int(len(validation)),
        },
        "leakage_control": {
            "inner_selection": "Fold0-train inner spatial validation only",
            "strict_prediction": "Fold0 validation uses Fold0-train support only",
            "fold0_target": "evaluation and target-informed oracle only",
        },
        "extra_expert_predictions": extra_expert_paths,
        "distance_summary": {
            "inner_nearest_mean": float(inner_distances[:, 0].mean()),
            "inner_nearest_p90": float(np.quantile(inner_distances[:, 0], 0.9)),
            "strict_nearest_mean": float(strict_distances[:, 0].mean()),
            "strict_nearest_p90": float(np.quantile(strict_distances[:, 0], 0.9)),
        },
        "alignment": {
            "enabled": bool(args.aligned),
            "maximum_vertical_shift": 2,
            "maximum_horizontal_shift": 4,
            "maximum_delay_shift": 12,
            "source": "query and neighbor OOF Teacher magnitudes only",
        },
        "inner_metrics": {
            name: aggregate_sample_metrics(values) for name, values in inner_arrays.items()
        },
        "selection": {
            "global": selected_global,
            "per_cell": {str(cell): name for cell, name in selected_per_cell.items()},
        },
        "strict_fold0": {
            "v4_baseline": v4_metrics,
            "teacher_seed": aggregate_sample_metrics(strict_arrays["teacher_seed"]),
            "selected_global": strict_global_metrics,
            "selected_per_cell": strict_per_cell_metrics,
            "deployable_selection": deployable_name,
            "deployable_metrics": deployable_metrics,
            "gain_vs_v4": strict_gain,
            "v4_plus_local_oracle": {
                **{key: value for key, value in oracle.items() if key != "selection"},
                "gain_vs_v4": oracle_gain,
            },
        },
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    save_json(report_path, report)
    report_path.with_suffix(".md").write_text(
        f"# {args.experiment_id} Local Magnitude Transfer\n\n"
        "Fold0 is offline validation, not the official online score.\n\n"
        f"- Inner-selected global strategy: `{selected_global}`\n"
        f"- Inner-selected per-cell strategies: `{report['selection']['per_cell']}`\n"
        f"- Strict V4 baseline: `{float(v4_metrics['score']):.6f}`\n"
        f"- Strict deployable candidate: `{float(deployable_metrics['score']):.6f}`\n"
        f"- Diagnostic two-expert oracle: `{float(oracle['metrics']['score']):.6f}`\n"
        f"- Decision: `{decision}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
