from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np
import torch

from scheme_e.angle_delay import channel_to_shape_target, shape_to_channel
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
)
from scheme_e.hybrid_training import _normalized_geometry, load_hybrid_checkpoint
from scheme_e.projection import alternating_spectral_projection
from scheme_e.reference import build_reference_candidates
from scheme_e.reference_context import select_reference_candidates


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_strategy(config: dict, summary: dict) -> dict:
    selected = str(summary.get("selected_reference_strategy", "nearest"))
    for strategy in config["hybrid"].get("reference_strategies", []):
        if str(strategy.get("name")) == selected:
            return dict(strategy)
    return {"name": "nearest", "top_k": 1}


def _select_references(
    target_indices: np.ndarray,
    support_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    spectral_targets: dict[str, np.ndarray],
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    strategy: dict,
) -> np.ndarray:
    candidate_count = max(16, int(strategy.get("top_k", 1)))
    local, distances = build_reference_candidates(
        metadata["train_positions"][target_indices],
        metadata["train_cells"][target_indices],
        metadata["train_positions"][support_indices],
        metadata["train_cells"][support_indices],
        metadata["outage"][support_indices],
        top_k=candidate_count,
        target_global_indices=target_indices,
        observed_global_indices=support_indices,
    )
    candidates = support_indices[local]
    if str(strategy.get("name", "nearest")) == "nearest":
        return candidates[:, 0]
    normalized = np.clip(
        (metadata["train_geometry_features"] - geometry_mean) / geometry_std,
        -8.0,
        8.0,
    ).astype(np.float32)
    return select_reference_candidates(
        candidates,
        distances,
        _normalized_geometry(
            metadata["train_geometry_features"],
            target_indices,
            geometry_mean,
            geometry_std,
        ),
        normalized,
        priors["pas_log"][target_indices].astype(np.float32),
        priors["pdp_log"][target_indices].astype(np.float32),
        spectral_targets["pas_log"].astype(np.float32),
        spectral_targets["pdp_log"].astype(np.float32),
        strategy,
    )


def _latent_memmaps(
    output_dir: Path,
    prefix: str,
    count: int,
    spectrum_dim: int,
    detail_dim: int,
) -> dict[str, np.memmap]:
    return {
        "spectrum": np.lib.format.open_memmap(
            output_dir / f"{prefix}_spectrum.npy",
            mode="w+",
            dtype=np.float16,
            shape=(count, spectrum_dim),
        ),
        "detail": np.lib.format.open_memmap(
            output_dir / f"{prefix}_detail.npy",
            mode="w+",
            dtype=np.float16,
            shape=(count, detail_dim),
        ),
        "log_power": np.lib.format.open_memmap(
            output_dir / f"{prefix}_log_power.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count,),
        ),
    }


@torch.no_grad()
def _encode_targets(
    channels: np.ndarray,
    indices: np.ndarray,
    cache: dict[str, np.memmap],
    model: torch.nn.Module,
    shape: object,
    device: torch.device,
    batch_size: int,
) -> None:
    model.autoencoder.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        channel = torch.as_tensor(
            np.array(channels[batch_indices], copy=True), device=device
        )
        target_shape, log_power, _ = channel_to_shape_target(channel, shape)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            spectrum, detail = model.autoencoder.encode(target_shape)
        cache["spectrum"][batch_indices] = (
            spectrum.float().cpu().numpy().astype(np.float16)
        )
        cache["detail"][batch_indices] = detail.float().cpu().numpy().astype(np.float16)
        cache["log_power"][batch_indices] = log_power.float().cpu().numpy()
        if start % max(batch_size * 32, 1) == 0:
            print(f"target latent {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
    for value in cache.values():
        value.flush()


def _safe_ue_log(
    priors: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    indices: np.ndarray,
    power_bounds: np.ndarray | None,
    device: torch.device,
) -> torch.Tensor:
    ue = torch.as_tensor(priors["ue_log_energy"][indices], device=device).float()
    if power_bounds is None:
        return ue
    cells = metadata["train_cells"][indices].astype(np.int64)
    bounds = np.asarray(power_bounds, dtype=np.float32)[cells]
    lower = torch.as_tensor(bounds[:, 0], device=device)
    upper = torch.as_tensor(bounds[:, 1], device=device)
    original = torch.as_tensor(priors["log_power"][indices], device=device).float()
    safe_power = torch.maximum(torch.minimum(original, upper), lower)
    relative = (ue - original[:, None]).clamp(-1.5, 1.5)
    return torch.maximum(
        torch.minimum(safe_power[:, None] + relative, upper[:, None] + 1.5),
        lower[:, None] - 1.5,
    )


@torch.no_grad()
def _encode_teacher_seed(
    channels: np.ndarray,
    target_indices: np.ndarray,
    references: np.ndarray,
    cache: dict[str, np.memmap],
    metadata: dict[str, np.ndarray],
    priors: dict[str, np.ndarray],
    model: torch.nn.Module,
    shape: object,
    power_bounds: np.ndarray | None,
    projection_iterations: int,
    device: torch.device,
    batch_size: int,
) -> None:
    model.autoencoder.eval()
    for start in range(0, len(target_indices), batch_size):
        indices = target_indices[start : start + batch_size]
        reference_indices = references[start : start + batch_size]
        reference = torch.as_tensor(
            np.array(channels[reference_indices], copy=True), device=device
        )
        pas_log = torch.as_tensor(
            priors["pas_log"][indices].astype(np.float32), device=device
        )
        pdp_log = torch.as_tensor(
            priors["pdp_log"][indices].astype(np.float32), device=device
        )
        projected = alternating_spectral_projection(
            reference,
            pas_log,
            pdp_log,
            _safe_ue_log(priors, metadata, indices, power_bounds, device),
            shape,
            iterations=projection_iterations,
            proxy_count=model.proxy_count,
            minimum_scale=model.projection_minimum_scale,
            maximum_scale=model.projection_maximum_scale,
        )
        projected_shape, log_power, _ = channel_to_shape_target(projected, shape)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            spectrum, detail = model.autoencoder.encode(projected_shape)
        cache["spectrum"][indices] = spectrum.float().cpu().numpy().astype(np.float16)
        cache["detail"][indices] = detail.float().cpu().numpy().astype(np.float16)
        cache["log_power"][indices] = log_power.float().cpu().numpy()
        if start % max(batch_size * 32, 1) == 0:
            print(
                f"teacher seed latent {min(start + batch_size, len(target_indices))}/{len(target_indices)}",
                flush=True,
            )
    for value in cache.values():
        value.flush()


@torch.no_grad()
def _encode_prediction(
    prediction: np.ndarray,
    cache: dict[str, np.memmap],
    model: torch.nn.Module,
    shape: object,
    device: torch.device,
    batch_size: int,
) -> None:
    rows = np.arange(len(prediction), dtype=np.int64)
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        channel = torch.as_tensor(np.array(prediction[local], copy=True), device=device)
        target_shape, log_power, _ = channel_to_shape_target(channel, shape)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            spectrum, detail = model.autoencoder.encode(target_shape)
        cache["spectrum"][local] = spectrum.float().cpu().numpy().astype(np.float16)
        cache["detail"][local] = detail.float().cpu().numpy().astype(np.float16)
        cache["log_power"][local] = log_power.float().cpu().numpy()
    for value in cache.values():
        value.flush()


def _fit_basis(
    residual: np.ndarray,
    maximum_rank: int,
    device: torch.device,
) -> dict[str, object]:
    values = torch.as_tensor(residual, dtype=torch.float32, device=device)
    mean = values.mean(dim=0)
    centered = values - mean
    rank = min(int(maximum_rank), len(values) - 1, values.shape[1])
    _, singular, components = torch.pca_lowrank(
        centered, q=rank, center=False, niter=4
    )
    total = centered.square().sum().double().clamp_min(1e-30)
    explained = torch.cumsum(singular.double().square(), dim=0) / total
    return {
        "mean": mean.cpu(),
        "components": components.cpu(),
        "singular_values": singular.cpu(),
        "explained_cumulative": explained.cpu(),
        "samples": int(len(values)),
        "dimensions": int(values.shape[1]),
    }


def _correct_branch(
    base: torch.Tensor,
    target: torch.Tensor,
    cells: torch.Tensor,
    active: torch.Tensor,
    bases: dict[str, dict[str, object]],
    branch: str,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    output = base.clone()
    for cell in torch.unique(cells).tolist():
        selected = active & (cells == int(cell))
        if not torch.any(selected):
            continue
        basis = bases[f"cell{int(cell)}_{branch}"]
        mean = basis["mean"].to(device)
        residual = target[selected].float() - base[selected].float()
        reconstructed = mean.expand_as(residual)
        if rank > 0:
            components = basis["components"][:, :rank].to(device)
            centered = residual - mean
            reconstructed = reconstructed + (centered @ components) @ components.T
        output[selected] = base[selected].float() + reconstructed
    return output


def _validation_explained_by_rank(
    target: np.ndarray,
    base: np.ndarray,
    cells: np.ndarray,
    active: np.ndarray,
    bases: dict[str, dict[str, object]],
    branch: str,
    ranks: list[int],
    device: torch.device,
) -> dict[str, float]:
    error = {rank: 0.0 for rank in ranks}
    total = 0.0
    for cell in np.unique(cells):
        selected = active & (cells == cell)
        residual = torch.as_tensor(
            target[selected].astype(np.float32) - base[selected].astype(np.float32),
            device=device,
        )
        basis = bases[f"cell{int(cell)}_{branch}"]
        mean = basis["mean"].to(device)
        components = basis["components"][:, : max(ranks)].to(device)
        centered = residual - mean
        centered_energy = float(centered.double().square().sum().cpu())
        total += float(residual.double().square().sum().cpu())
        coefficients = centered @ components if max(ranks) > 0 else None
        for rank in ranks:
            captured = (
                float(coefficients[:, :rank].double().square().sum().cpu())
                if rank > 0 and coefficients is not None
                else 0.0
            )
            error[rank] += max(centered_energy - captured, 0.0)
    return {
        str(rank): 1.0 - error[rank] / max(total, 1e-30) for rank in ranks
    }


@torch.no_grad()
def _evaluate_oracle(
    candidate: dict[str, np.ndarray],
    candidate_rows: np.ndarray,
    target_cache: dict[str, np.ndarray],
    validation: np.ndarray,
    metadata: dict[str, np.ndarray],
    channels: np.ndarray,
    bases: dict[str, dict[str, object]],
    model: torch.nn.Module,
    shape: object,
    device: torch.device,
    batch_size: int,
    rank: int,
    mode: str,
) -> dict[str, float | int]:
    parts = []
    cells_all = metadata["train_cells"][validation].astype(np.int64)
    outage_all = metadata["outage"][validation].astype(bool)
    for start in range(0, len(validation), batch_size):
        stop = min(start + batch_size, len(validation))
        global_indices = validation[start:stop]
        rows = candidate_rows[start:stop]
        cells = torch.as_tensor(cells_all[start:stop], device=device)
        outage = torch.as_tensor(outage_all[start:stop], device=device)
        active = ~outage
        base_spectrum = torch.as_tensor(
            np.asarray(candidate["spectrum"][rows], dtype=np.float32), device=device
        )
        base_detail = torch.as_tensor(
            np.asarray(candidate["detail"][rows], dtype=np.float32), device=device
        )
        target_spectrum = torch.as_tensor(
            np.asarray(target_cache["spectrum"][global_indices], dtype=np.float32),
            device=device,
        )
        target_detail = torch.as_tensor(
            np.asarray(target_cache["detail"][global_indices], dtype=np.float32),
            device=device,
        )
        spectrum = base_spectrum
        detail = base_detail
        if mode in {"spectrum", "both"}:
            spectrum = _correct_branch(
                base_spectrum,
                target_spectrum,
                cells,
                active,
                bases,
                "spectrum",
                rank,
                device,
            )
        if mode in {"detail", "both"}:
            detail = _correct_branch(
                base_detail,
                target_detail,
                cells,
                active,
                bases,
                "detail",
                rank,
                device,
            )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            decoded = model.autoencoder.decode(spectrum, detail)
        prediction = shape_to_channel(
            decoded.float(),
            torch.as_tensor(candidate["log_power"][rows], device=device),
            shape,
        )
        target_channel = torch.as_tensor(
            np.array(channels[global_indices], copy=True), device=device
        )
        parts.append(sample_metric_batch(prediction, target_channel, shape, outage))
    return aggregate_sample_metrics(concatenate_metric_batches(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-only PCA residual basis and diagnostic Fold0 rank oracle"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--prediction",
        default="../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy",
    )
    parser.add_argument(
        "--output-dir", default="../research/scheme_e_065/residual_rank"
    )
    parser.add_argument("--ranks", default="0,8,16,32,64,128")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    ranks = sorted({int(value) for value in args.ranks.split(",")})
    if not ranks or ranks[0] < 0:
        raise ValueError("Ranks must be non-negative")
    maximum_rank = max(ranks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["runtime"]["device"] = args.device
    device = choose_device(args.device)
    torch.manual_seed(int(config["seed"]) + 650)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]) + 650)
    metadata = _load_npz(Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz")
    priors = _load_npz(config["spectral_teacher"]["oof_output_path"])
    spectral_targets = _load_npz(config["spectral"]["target_path"])
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    checkpoint_path = Path(config["hybrid"]["output_dir"]) / "best.pt"
    model, shape, checkpoint = load_hybrid_checkpoint(config, checkpoint_path, device)
    summary = json.loads(
        (checkpoint_path.parent / "summary.json").read_text(encoding="utf-8")
    )
    strategy = _reference_strategy(config, summary)
    fold = int(config["split"]["validation_fold"])
    available = priors["available"].astype(bool)
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    validation = np.flatnonzero(available & validation_mask)
    observed = np.flatnonzero(available & ~validation_mask)
    nonoutage_observed = observed[~metadata["outage"][observed].astype(bool)]
    all_indices = np.flatnonzero(available)

    target_cache = _latent_memmaps(
        output_dir,
        "target",
        len(available),
        model.autoencoder.spectrum_latent_dim,
        model.autoencoder.phase_latent_dim,
    )
    _encode_targets(
        channels,
        all_indices,
        target_cache,
        model,
        shape,
        device,
        int(args.batch_size),
    )

    teacher_cache = _latent_memmaps(
        output_dir,
        "teacher_seed",
        len(available),
        model.autoencoder.spectrum_latent_dim,
        model.autoencoder.phase_latent_dim,
    )
    geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    geometry_std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    power_bounds = checkpoint.get("power_bounds")
    if power_bounds is not None:
        power_bounds = np.asarray(power_bounds, dtype=np.float32)
    for target_indices in (observed, validation):
        references = _select_references(
            target_indices,
            observed,
            metadata,
            priors,
            spectral_targets,
            geometry_mean,
            geometry_std,
            strategy,
        )
        _encode_teacher_seed(
            channels,
            target_indices,
            references,
            teacher_cache,
            metadata,
            priors,
            model,
            shape,
            power_bounds,
            int(summary["selected_projection_iterations"]),
            device,
            int(args.batch_size),
        )

    bases: dict[str, dict[str, object]] = {}
    basis_report: dict[str, object] = {}
    for cell in np.unique(metadata["train_cells"]):
        training = nonoutage_observed[
            metadata["train_cells"][nonoutage_observed] == cell
        ]
        for branch in ("spectrum", "detail"):
            residual = (
                target_cache[branch][training].astype(np.float32)
                - teacher_cache[branch][training].astype(np.float32)
            )
            key = f"cell{int(cell)}_{branch}"
            bases[key] = _fit_basis(residual, maximum_rank, device)
            cumulative = np.asarray(bases[key]["explained_cumulative"])
            basis_report[key] = {
                "samples": int(len(training)),
                "dimensions": int(residual.shape[1]),
                "explained_variance": {
                    str(rank): float(cumulative[rank - 1]) if rank > 0 else 0.0
                    for rank in ranks
                },
            }
            print(f"basis ready: {key}", flush=True)
            del residual
            if device.type == "cuda":
                torch.cuda.empty_cache()
    torch.save(bases, output_dir / "train_only_basis.pt")

    prediction = np.load(args.prediction, mmap_mode="r")
    if len(prediction) != len(validation):
        raise ValueError("Saved baseline prediction does not match strict Fold0")
    baseline_cache = _latent_memmaps(
        output_dir,
        "baseline_final",
        len(validation),
        model.autoencoder.spectrum_latent_dim,
        model.autoencoder.phase_latent_dim,
    )
    _encode_prediction(
        prediction,
        baseline_cache,
        model,
        shape,
        device,
        int(args.batch_size),
    )

    cells = metadata["train_cells"][validation].astype(np.int64)
    active = ~metadata["outage"][validation].astype(bool)
    candidates = {
        "teacher_seed": (teacher_cache, validation),
        "baseline_final": (baseline_cache, np.arange(len(validation), dtype=np.int64)),
    }
    evaluations: dict[str, dict[str, dict[str, float | int]]] = {}
    validation_explained: dict[str, dict[str, dict[str, float]]] = {}
    for candidate_name, (candidate, rows) in candidates.items():
        evaluations[candidate_name] = {
            "ae_roundtrip": _evaluate_oracle(
                candidate,
                rows,
                target_cache,
                validation,
                metadata,
                channels,
                bases,
                model,
                shape,
                device,
                int(args.batch_size),
                0,
                "none",
            )
        }
        validation_explained[candidate_name] = {
            branch: _validation_explained_by_rank(
                target_cache[branch][validation],
                candidate[branch][rows],
                cells,
                active,
                bases,
                branch,
                ranks,
                device,
            )
            for branch in ("spectrum", "detail")
        }
        for rank in ranks:
            rank_key = f"rank_{rank}"
            evaluations[candidate_name][rank_key] = {}
            for mode in ("spectrum", "detail", "both"):
                evaluations[candidate_name][rank_key][mode] = _evaluate_oracle(
                    candidate,
                    rows,
                    target_cache,
                    validation,
                    metadata,
                    channels,
                    bases,
                    model,
                    shape,
                    device,
                    int(args.batch_size),
                    rank,
                    mode,
                )
            best_rank = max(
                evaluations[candidate_name][rank_key].items(),
                key=lambda item: float(item[1]["score"]),
            )
            print(
                f"{candidate_name} rank={rank} best={best_rank[0]} score={best_rank[1]['score']:.6f}",
                flush=True,
            )

    flat = []
    for candidate_name, candidate_values in evaluations.items():
        for rank_key, modes in candidate_values.items():
            if rank_key == "ae_roundtrip":
                flat.append((candidate_name, rank_key, "none", modes))
                continue
            for mode, metrics in modes.items():
                flat.append((candidate_name, rank_key, mode, metrics))
    best = max(flat, key=lambda item: float(item[3]["score"]))
    report = {
        "status": "PASS",
        "diagnostic_only": True,
        "leakage_control": {
            "basis_fit_samples": "Fold0-train non-outage only",
            "basis_seed": "OOF spectral priors plus self-excluded Fold0-train references",
            "fold0_target_usage": "oracle coefficients and evaluation only",
            "deployable": False,
        },
        "config": args.config,
        "prediction": args.prediction,
        "prediction_sha256": _sha256(args.prediction),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_bootstrap.PROJECT_ROOT.parent, text=True
        ).strip(),
        "training_samples": int(len(nonoutage_observed)),
        "validation_samples": int(len(validation)),
        "ranks": ranks,
        "basis": basis_report,
        "validation_residual_explained": validation_explained,
        "evaluations": evaluations,
        "best": {
            "candidate": best[0],
            "rank": best[1],
            "mode": best[2],
            "metrics": best[3],
        },
        "crosses_065": bool(float(best[3]["score"]) >= 0.65),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(output_dir / "LOW_RANK_RESIDUAL_ORACLE.json", report)

    rows = []
    for candidate_name, rank_key, mode, metrics in flat:
        rows.append(
            f"| {candidate_name} | {rank_key} | {mode} | {metrics['pas']:.6f} | "
            f"{metrics['pdp']:.6f} | {metrics['nmse']:.6f} | {metrics['score']:.6f} |"
        )
    markdown = f"""# Low-rank Residual Oracle

`DIAGNOSTIC ONLY - NOT DEPLOYABLE`

The PCA bases use only Fold0-train non-outage residuals. Their seed predictions use
OOF spectral priors and references that explicitly exclude the query itself. Fold0
targets are used only to calculate oracle coefficients and final validation metrics.

| Candidate | Rank | Corrected branch | PAS | PDP | NMSE | Score |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Best oracle: `{best[0]} / {best[1]} / {best[2]}` with Score
`{float(best[3]['score']):.6f}`. Crosses strict `0.65`: `{report['crosses_065']}`.
"""
    (output_dir / "LOW_RANK_RESIDUAL_ORACLE.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report["best"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
