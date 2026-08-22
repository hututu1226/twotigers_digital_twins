from __future__ import annotations

import argparse
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
    reconstruct_low_rank_residual,
    replace_angle_delay_log_power,
    split_complex_correction,
)
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
)
from scheme_e.hybrid_training import load_hybrid_checkpoint


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _load_latent_cache(path: Path, prefix: str) -> dict[str, np.ndarray]:
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
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        decoded = autoencoder.decode(spectrum, detail)
    return decoded.float()


@torch.no_grad()
def _training_residual_matrix(
    channels: np.ndarray,
    indices: np.ndarray,
    teacher_cache: dict[str, np.ndarray],
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
    batch_size: int,
    representation: str,
    log_power_scale: float,
) -> torch.Tensor:
    dimensions = (
        int(np.prod(shape.ad_shape))
        if representation == "complex"
        else int(shape.m_p * shape.n * shape.m_v * shape.m_h * shape.s)
    )
    residual = torch.empty(
        (len(indices), dimensions), dtype=torch.float32, device=device
    )
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        selected = indices[start:stop]
        target_channel = torch.as_tensor(
            np.array(channels[selected], copy=True), device=device
        )
        target_shape, _, _ = channel_to_shape_target(target_channel, shape)
        seed_shape = _decode_seed_shape(
            teacher_cache, selected, autoencoder, device
        )
        if representation == "complex":
            target_value = target_shape
            seed_value = seed_shape
        else:
            target_value = angle_delay_log_power(
                target_shape, shape, log_power_scale
            )
            seed_value = angle_delay_log_power(
                seed_shape, shape, log_power_scale
            )
        residual[start:stop] = (target_value - seed_value).flatten(1)
    return residual


@torch.no_grad()
def _fit_cell_basis(
    residual: torch.Tensor,
    maximum_rank: int,
    oversample: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    mean = residual.mean(dim=0)
    residual.sub_(mean)
    total_energy = float(residual.square().sum(dtype=torch.float64).cpu())
    q = min(
        int(maximum_rank) + int(oversample),
        len(residual) - 1,
        residual.shape[1],
    )
    if q < int(maximum_rank):
        raise ValueError(
            f"Only {q} PCA components are available for rank {maximum_rank}"
        )
    torch.manual_seed(int(seed))
    if residual.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    _, singular, vectors = torch.pca_lowrank(
        residual, q=q, center=False, niter=int(iterations)
    )
    components = vectors[:, : int(maximum_rank)].T.contiguous()
    explained = torch.cumsum(
        singular[: int(maximum_rank)].double().square(), dim=0
    ) / max(total_energy, 1e-30)
    return {
        "mean": mean.cpu(),
        "components": components.cpu(),
        "singular_values": singular[: int(maximum_rank)].cpu(),
        "explained_cumulative": explained.cpu(),
        "training_samples": int(len(residual)),
        "dimensions": int(residual.shape[1]),
        "pca_q": int(q),
        "pca_iterations": int(iterations),
    }


@torch.no_grad()
def _teacher_source_batch(
    cache: dict[str, np.ndarray],
    global_indices: np.ndarray,
    autoencoder: torch.nn.Module,
    shape: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    angle_delay = _decode_seed_shape(cache, global_indices, autoencoder, device)
    log_power = torch.as_tensor(
        np.asarray(cache["log_power"][global_indices], dtype=np.float32),
        device=device,
    )
    outage = torch.as_tensor(
        np.asarray(cache["outage"][global_indices], dtype=bool), device=device
    )
    channel = shape_to_channel(angle_delay, log_power, shape, outage)
    return channel, angle_delay, log_power, outage


@torch.no_grad()
def _prediction_source_batch(
    prediction: np.ndarray,
    rows: np.ndarray,
    shape: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    channel = torch.as_tensor(np.array(prediction[rows], copy=True), device=device)
    angle_delay, log_power, outage = channel_to_shape_target(channel, shape)
    return channel, angle_delay, log_power, outage


@torch.no_grad()
def _evaluate_source(
    source_name: str,
    prediction: np.ndarray | None,
    teacher_cache: dict[str, np.ndarray],
    validation: np.ndarray,
    metadata: dict[str, np.ndarray],
    channels: np.ndarray,
    bases: dict[int, dict[str, object]],
    autoencoder: torch.nn.Module,
    shape: object,
    ranks: list[int],
    device: torch.device,
    batch_size: int,
    representation: str,
    log_power_scale: float,
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, np.ndarray]]]:
    metric_parts: dict[str, list[object]] = {"none": []}
    modes = (
        ("complex", "magnitude", "phase")
        if representation == "complex"
        else ("magnitude",)
    )
    for rank in ranks:
        for mode in modes:
            metric_parts[f"rank{rank}_{mode}"] = []
    for start in range(0, len(validation), batch_size):
        stop = min(start + batch_size, len(validation))
        global_indices = validation[start:stop]
        rows = np.arange(start, stop, dtype=np.int64)
        if source_name == "teacher_seed":
            base_channel, base_shape, log_power, source_outage = _teacher_source_batch(
                teacher_cache,
                global_indices,
                autoencoder,
                shape,
                device,
            )
        else:
            if prediction is None:
                raise ValueError("A saved prediction is required for baseline_final")
            base_channel, base_shape, log_power, source_outage = _prediction_source_batch(
                prediction, rows, shape, device
            )
        target_channel = torch.as_tensor(
            np.array(channels[global_indices], copy=True), device=device
        )
        target_shape, _, target_outage = channel_to_shape_target(target_channel, shape)
        metric_parts["none"].append(
            sample_metric_batch(base_channel, target_channel, shape, target_outage)
        )
        cells = metadata["train_cells"][global_indices].astype(np.int64)
        active = ~target_outage
        if representation == "complex":
            target_value = target_shape
            base_value = base_shape
        else:
            target_value = angle_delay_log_power(
                target_shape, shape, log_power_scale
            )
            base_value = angle_delay_log_power(
                base_shape, shape, log_power_scale
            )
        residual = (target_value - base_value).flatten(1)
        for rank in ranks:
            correction = torch.zeros_like(residual)
            for cell in np.unique(cells):
                selected = torch.as_tensor(cells == int(cell), device=device)
                basis = bases[int(cell)]
                correction[selected] = reconstruct_low_rank_residual(
                    residual[selected],
                    basis["mean"],
                    basis["components"],
                    int(rank),
                )
            correction = correction.masked_fill(~active[:, None], 0.0)
            if representation == "complex":
                corrected_shape = base_shape + correction.reshape_as(base_shape)
                variants = split_complex_correction(
                    base_shape, corrected_shape, shape
                )
            else:
                corrected_log_power = (
                    base_value + correction.reshape_as(base_value)
                )
                variants = {
                    "magnitude": replace_angle_delay_log_power(
                        base_shape,
                        corrected_log_power,
                        shape,
                        log_power_scale,
                    )
                }
            for mode, candidate_shape in variants.items():
                candidate_channel = shape_to_channel(
                    candidate_shape, log_power, shape, source_outage
                )
                metric_parts[f"rank{rank}_{mode}"].append(
                    sample_metric_batch(
                        candidate_channel, target_channel, shape, target_outage
                    )
                )
    arrays = {
        name: concatenate_metric_batches(parts)
        for name, parts in metric_parts.items()
    }
    metrics = {
        name: aggregate_sample_metrics(values) for name, values in arrays.items()
    }
    return metrics, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-only complex angle-delay residual oracle diagnostic"
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
        "--output-dir",
        default="artifacts/scheme_e_065/l0_010_complex_residual",
    )
    parser.add_argument(
        "--report", default="../research/scheme_e_065/L0_010_COMPLEX_ORACLE.json"
    )
    parser.add_argument(
        "--representation",
        choices=("complex", "log_power"),
        default="complex",
    )
    parser.add_argument("--log-power-scale", type=float, default=4.0)
    parser.add_argument("--ranks", default="0,8,16,32,64")
    parser.add_argument("--pca-oversample", type=int, default=12)
    parser.add_argument("--pca-iterations", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=2671)
    args = parser.parse_args()
    started = time.perf_counter()
    ranks = sorted({int(value) for value in args.ranks.split(",")})
    if not ranks or ranks[0] < 0:
        raise ValueError("Ranks must be non-negative")
    if float(args.log_power_scale) <= 0.0:
        raise ValueError("log-power-scale must be positive")
    maximum_rank = max(ranks)
    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This high-dimensional PCA diagnostic requires CUDA")
    config = load_config(args.config)
    metadata = _load_npz(Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz")
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    checkpoint_path = Path(config["hybrid"]["output_dir"]) / "best.pt"
    model, shape, _ = load_hybrid_checkpoint(config, checkpoint_path, device)
    model.autoencoder.eval()
    cache_dir = Path(args.cache_dir)
    teacher_cache = _load_latent_cache(cache_dir, "teacher_seed")
    fold = int(config["split"]["validation_fold"])
    available = priors["available"].astype(bool)
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    validation = np.flatnonzero(available & validation_mask)
    observed = np.flatnonzero(available & ~validation_mask)
    nonoutage_observed = observed[~metadata["outage"][observed].astype(bool)]
    prediction = np.load(args.baseline_prediction, mmap_mode="r")
    if len(prediction) != len(validation):
        raise ValueError(
            f"Prediction has {len(prediction)} rows; Fold0 requires {len(validation)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bases: dict[int, dict[str, object]] = {}
    basis_report = {}
    for cell in np.unique(metadata["train_cells"]):
        indices = nonoutage_observed[
            metadata["train_cells"][nonoutage_observed] == int(cell)
        ]
        print(
            f"building complex residual matrix cell={int(cell)} samples={len(indices)}",
            flush=True,
        )
        residual = _training_residual_matrix(
            channels,
            indices,
            teacher_cache,
            model.autoencoder,
            shape,
            device,
            int(args.batch_size),
            str(args.representation),
            float(args.log_power_scale),
        )
        basis = _fit_cell_basis(
            residual,
            maximum_rank,
            int(args.pca_oversample),
            int(args.pca_iterations),
            int(args.seed) + int(cell),
        )
        del residual
        torch.cuda.empty_cache()
        bases[int(cell)] = basis
        basis_report[str(int(cell))] = {
            "training_samples": int(basis["training_samples"]),
            "dimensions": int(basis["dimensions"]),
            "pca_q": int(basis["pca_q"]),
            "explained_cumulative": {
                str(rank): float(basis["explained_cumulative"][rank - 1])
                if rank > 0
                else 0.0
                for rank in ranks
            },
        }
        print(
            f"cell={int(cell)} rank{maximum_rank}_explained="
            f"{basis_report[str(int(cell))]['explained_cumulative'][str(maximum_rank)]:.6f}",
            flush=True,
        )
    basis_path = output_dir / f"train_only_{args.representation}_basis.pt"
    torch.save(
        {
            "bases": bases,
            "shape": shape.__dict__,
            "settings": vars(args),
            "leakage_boundary": "Fold0-train OOF teacher seed residuals only",
            "representation": str(args.representation),
        },
        basis_path,
    )
    evaluation_bases = {
        cell: {
            **basis,
            "mean": basis["mean"].to(device),
            "components": basis["components"].to(device),
        }
        for cell, basis in bases.items()
    }

    evaluations = {}
    per_sample = {}
    for source_name, source_prediction in (
        ("teacher_seed", None),
        ("baseline_final", prediction),
    ):
        print(f"evaluating source={source_name}", flush=True)
        metrics, arrays = _evaluate_source(
            source_name,
            source_prediction,
            teacher_cache,
            validation,
            metadata,
            channels,
            evaluation_bases,
            model.autoencoder,
            shape,
            ranks,
            device,
            int(args.batch_size),
            str(args.representation),
            float(args.log_power_scale),
        )
        evaluations[source_name] = metrics
        for candidate_name, values in arrays.items():
            for field, value in values.items():
                per_sample[f"{source_name}__{candidate_name}__{field}"] = value
        best_name, best_metrics = max(
            metrics.items(), key=lambda item: float(item[1]["score"])
        )
        print(
            f"source={source_name} best={best_name} "
            f"score={float(best_metrics['score']):.6f}",
            flush=True,
        )
    np.savez_compressed(
        output_dir / "Fold0_Per_Sample_Oracle_Metrics.npz", **per_sample
    )

    flat = [
        (source, name, metrics)
        for source, candidates in evaluations.items()
        for name, metrics in candidates.items()
        if name != "none"
    ]
    best = max(flat, key=lambda item: float(item[2]["score"]))
    baseline_metrics = evaluations["baseline_final"]["none"]
    report = {
        "status": "PASS",
        "diagnostic_only": True,
        "deployable": False,
        "hypothesis": (
            "A train-only low-rank angle-delay residual basis in the selected "
            "representation has a strict Fold0 oracle ceiling above 0.65."
        ),
        "representation": str(args.representation),
        "log_power_scale": float(args.log_power_scale),
        "leakage_control": {
            "basis_fit": "Fold0-train non-outage residuals only",
            "basis_seed": "OOF teacher latent decoded by the frozen AE",
            "fold0_target_usage": "oracle coefficients and evaluation only",
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_bootstrap.PROJECT_ROOT.parent, text=True
        ).strip(),
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "baseline_prediction": args.baseline_prediction,
        "basis_path": str(basis_path),
        "training_samples": int(len(nonoutage_observed)),
        "validation_samples": int(len(validation)),
        "ranks": ranks,
        "basis": basis_report,
        "evaluations": evaluations,
        "authoritative_baseline": baseline_metrics,
        "best_oracle": {
            "source": best[0],
            "candidate": best[1],
            "metrics": best[2],
            "delta": float(best[2]["score"]) - float(baseline_metrics["score"]),
        },
        "crosses_065": bool(float(best[2]["score"]) >= 0.65),
        "decision": (
            "PROMOTE_TO_L1" if float(best[2]["score"]) >= 0.65 else "DROP"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(args.report, report)
    rows = []
    for source, candidates in evaluations.items():
        for name, metrics in candidates.items():
            rows.append(
                f"| {source} | {name} | {metrics['pas']:.6f} | "
                f"{metrics['pdp']:.6f} | {metrics['nmse']:.6f} | "
                f"{metrics['score']:.6f} |"
            )
    markdown = f"""# Complex Residual Oracle

`DIAGNOSTIC ONLY - NOT DEPLOYABLE`

The PCA basis uses only Fold0-train OOF teacher residuals. Fold0 targets are used
only to calculate oracle coefficients and final metrics.

| Source | Candidate | PAS | PDP | NMSE | Score |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Best oracle: `{best[0]} / {best[1]}` with Score `{float(best[2]['score']):.6f}`.
Decision: `{report['decision']}`.
"""
    Path(args.report).with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report["best_oracle"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
