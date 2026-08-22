from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import torch

from scheme_e.carrier_transport import CarrierFit, select_transport_candidates
from scheme_e.config import choose_device, load_config, save_json
from scheme_e.diagnostics import (
    aggregate_sample_metrics,
    concatenate_metric_batches,
    sample_metric_batch,
    scale_oracle_predictions,
    target_informed_expert_oracle,
)
from scheme_e.hybrid_training import (
    _normalized_geometry,
    _output_projection_seed,
    _prior_batch,
    _reference_context_batch,
    _transport_batch,
    load_hybrid_checkpoint,
)
from scheme_e.metrics import ChannelMetricAccumulator
from scheme_e.power_safety import apply_outage_policy
from scheme_e.projection import relaxed_output_projection
from scheme_e.reference import build_reference_candidates
from scheme_e.reference_context import select_reference_candidates
from scheme_e.spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE


BEST_SCORE = 0.62705
TARGET_SCORE = 0.65


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=_bootstrap.PROJECT_ROOT.parent, text=True
    ).strip()


def _power_cosine(
    prediction_log: np.ndarray,
    target_log: np.ndarray,
    scale: float,
) -> np.ndarray:
    prediction = np.expm1(
        np.clip(prediction_log.astype(np.float64), 0.0, 20.0)
    ) / float(scale)
    target = np.expm1(np.clip(target_log.astype(np.float64), 0.0, 20.0)) / float(
        scale
    )
    denominator = np.maximum(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1),
        1e-30,
    )
    return np.clip(np.sum(prediction * target, axis=1) / denominator, 0.0, 1.0)


def _teacher_metrics(
    priors: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray | float]:
    nonzero = ~targets["outage"][indices].astype(bool)
    pas = np.zeros(len(indices), dtype=np.float64)
    pdp = np.zeros(len(indices), dtype=np.float64)
    pas[nonzero] = _power_cosine(
        priors["pas_log"][indices[nonzero]],
        targets["pas_log"][indices[nonzero]],
        PAS_LOG_SCALE,
    )
    pdp[nonzero] = _power_cosine(
        priors["pdp_log"][indices[nonzero]],
        targets["pdp_log"][indices[nonzero]],
        PDP_LOG_SCALE,
    )
    power_error = (
        priors["log_power"][indices].astype(np.float64)
        - targets["log_power"][indices].astype(np.float64)
    )
    return {
        "pas": pas,
        "pdp": pdp,
        "pas_mean_nonoutage": float(pas[nonzero].mean()),
        "pdp_mean_nonoutage": float(pdp[nonzero].mean()),
        "power_mae_log10_nonoutage": float(np.abs(power_error[nonzero]).mean()),
        "power_bias_log10_nonoutage": float(power_error[nonzero].mean()),
    }


def _reference_strategy(config: dict, summary: dict) -> dict[str, object]:
    selected = str(summary.get("selected_reference_strategy", "nearest"))
    for candidate in config["hybrid"].get("reference_strategies", []):
        if str(candidate.get("name")) == selected:
            return dict(candidate)
    return {"name": "nearest", "top_k": 1}


def _projection_settings(config: dict, report_path: str | Path | None) -> dict:
    configured = deepcopy(config.get("inference", {}).get("output_projection", {}))
    if configured:
        return configured
    if report_path is None:
        return {}
    selected = _read_json(report_path)["selected"]
    return {
        "iterations": int(selected.get("output_projection_iterations", 0)),
        "strength_by_cell": selected.get(
            "output_projection_strength_by_cell", [0.0, 0.0]
        ),
        "minimum_scale": 0.5,
        "maximum_scale": 2.0,
        "power_source": selected.get("output_projection_power_source", "model"),
        "channel_source": selected.get("output_projection_channel_source", "model"),
    }


def _outage_policy(policy_path: str | Path) -> dict[str, object]:
    policy = _read_json(policy_path)
    return {
        "threshold_by_cell": np.asarray(
            policy["outage_threshold_by_cell"], dtype=np.float32
        ),
        "strength_by_cell": np.asarray(
            policy["soft_outage_strength_by_cell"], dtype=np.float32
        ),
        "report": policy,
    }


def _append_metrics(
    storage: dict[str, list],
    stage: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: object,
    true_outage: torch.Tensor,
    legacy: dict[str, ChannelMetricAccumulator],
) -> None:
    storage[stage].append(sample_metric_batch(prediction, target, shape, true_outage))
    legacy[stage].update(prediction, target, true_outage)


def _collect_variant(
    *,
    name: str,
    config: dict,
    priors: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    channels: np.ndarray,
    validation: np.ndarray,
    observed: np.ndarray,
    checkpoint_path: Path,
    policy: dict[str, object],
    output_projection: dict,
    device: torch.device,
    save_path: Path | None,
    compare_path: Path | None,
    include_scale_oracles: bool,
) -> dict[str, object]:
    model, shape, checkpoint = load_hybrid_checkpoint(config, checkpoint_path, device)
    summary = _read_json(checkpoint_path.parent / "summary.json")
    strategy = _reference_strategy(config, summary)
    transport_config = config["hybrid"].get("transport_seed", {})
    carrier_payload = checkpoint.get("carrier_fit")
    carrier_fit = None
    if carrier_payload is not None:
        carrier_fit = CarrierFit(
            np.asarray(carrier_payload["wave_numbers"], dtype=np.float64),
            np.asarray(carrier_payload["qualities"], dtype=np.float64),
            np.asarray(carrier_payload["pair_counts"], dtype=np.int64),
        )
    transport_count = int(transport_config.get("count", 8)) if carrier_fit else 1
    candidate_count = max(16, transport_count, int(strategy.get("top_k", 1)))
    candidates, distances = build_reference_candidates(
        metadata["train_positions"][validation],
        metadata["train_cells"][validation],
        metadata["train_positions"][observed],
        metadata["train_cells"][observed],
        metadata["outage"][observed],
        top_k=candidate_count,
        target_global_indices=validation,
        observed_global_indices=observed,
    )
    candidate_globals = observed[candidates]
    geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    geometry_std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    if str(strategy.get("name", "nearest")) == "nearest":
        references = candidate_globals[:, 0]
    else:
        references = select_reference_candidates(
            candidate_globals,
            distances,
            _normalized_geometry(
                metadata["train_geometry_features"],
                validation,
                geometry_mean,
                geometry_std,
            ),
            np.clip(
                (metadata["train_geometry_features"] - geometry_mean) / geometry_std,
                -8.0,
                8.0,
            ),
            priors["pas_log"][validation].astype(np.float32),
            priors["pdp_log"][validation].astype(np.float32),
            targets["pas_log"].astype(np.float32),
            targets["pdp_log"].astype(np.float32),
            strategy,
        )
    transport_globals = None
    transport_distances = None
    if carrier_fit is not None:
        local, transport_distances = select_transport_candidates(
            candidates, distances, transport_count
        )
        transport_globals = observed[local]

    target_cells = metadata["train_cells"][validation].astype(np.int64)
    thresholds = np.asarray(policy["threshold_by_cell"], dtype=np.float32)
    strengths = np.asarray(policy["strength_by_cell"], dtype=np.float32)
    target_thresholds = thresholds[np.minimum(target_cells, len(thresholds) - 1)]
    target_strengths = strengths[np.minimum(target_cells, len(strengths) - 1)]
    projection_iterations = int(summary["selected_projection_iterations"])
    output_iterations = int(output_projection.get("iterations", 0))
    output_strengths = np.asarray(
        output_projection.get("strength_by_cell", [0.0]), dtype=np.float32
    )
    target_output_strengths = output_strengths[
        np.minimum(target_cells, len(output_strengths) - 1)
    ]
    power_bounds = checkpoint.get("power_bounds")
    if power_bounds is not None:
        power_bounds = np.asarray(power_bounds, dtype=np.float32)

    output = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            save_path,
            mode="w+",
            dtype=np.complex64,
            shape=(len(validation), *shape.raw_shape),
        )
    comparison = np.load(compare_path, mmap_mode="r") if compare_path else None
    disagreement = np.zeros(len(validation), dtype=np.float64)
    stages: dict[str, list] = defaultdict(list)
    legacy: dict[str, ChannelMetricAccumulator] = defaultdict(
        lambda: ChannelMetricAccumulator(shape)
    )
    batch_size = int(config["hybrid"].get("validation_batch_size", 4))
    model.eval()
    started = time.perf_counter()
    for start in range(0, len(validation), batch_size):
        stop = min(start + batch_size, len(validation))
        indices = validation[start:stop]
        reference_indices = references[start:stop]
        reference = torch.as_tensor(np.asarray(channels[reference_indices]), device=device)
        target = torch.as_tensor(np.asarray(channels[indices]), device=device)
        reference_context = None
        if model.condition_encoder.reference_dim:
            reference_context = _reference_context_batch(
                metadata,
                priors,
                targets,
                indices,
                reference_indices,
                geometry_mean,
                geometry_std,
            )
        inputs = _prior_batch(
            priors,
            metadata,
            indices,
            geometry_mean,
            geometry_std,
            device,
            power_bounds=power_bounds,
            reference_context=reference_context,
        )
        transport_channel = None
        if carrier_fit is not None:
            if transport_globals is None or transport_distances is None:
                raise AssertionError("Transport candidates are missing")
            transport_channel, transport_context = _transport_batch(
                channels,
                metadata,
                indices,
                transport_globals[start:stop],
                transport_distances[start:stop],
                carrier_fit,
                device,
                distance_power=float(transport_config.get("distance_power", 2.0)),
            )
            inputs["transport_context"] = transport_context
        model_outputs = model(
            reference,
            transport_channel=transport_channel,
            projection_iterations=projection_iterations,
            **inputs,
        )
        raw = model_outputs["channel"]
        projected = raw
        if output_iterations > 0:
            seed = _output_projection_seed(
                str(output_projection.get("channel_source", "model")),
                raw,
                reference,
                transport_channel,
            )
            projection_power = (
                inputs["log_power"]
                if str(output_projection.get("power_source", "model")) == "input"
                else model_outputs["power"]
            )
            projected = relaxed_output_projection(
                seed,
                inputs["pas_log"],
                inputs["pdp_log"],
                inputs["ue_log_energy"],
                projection_power,
                shape,
                iterations=output_iterations,
                proxy_count=model.proxy_count,
                strength=torch.as_tensor(
                    target_output_strengths[start:stop], device=device
                ),
                minimum_scale=float(output_projection.get("minimum_scale", 0.5)),
                maximum_scale=float(output_projection.get("maximum_scale", 2.0)),
            )
        final = apply_outage_policy(
            projected,
            inputs["outage_probability"],
            torch.as_tensor(target_thresholds[start:stop], device=device),
            torch.as_tensor(target_strengths[start:stop], device=device),
        )
        true_outage = torch.as_tensor(metadata["outage"][indices], device=device)
        _append_metrics(
            stages,
            "teacher_projected_seed",
            model_outputs["projected_channel"],
            target,
            shape,
            true_outage,
            legacy,
        )
        _append_metrics(stages, "hybrid_raw", raw, target, shape, true_outage, legacy)
        _append_metrics(
            stages,
            "hybrid_post_projection",
            projected,
            target,
            shape,
            true_outage,
            legacy,
        )
        _append_metrics(stages, "final", final, target, shape, true_outage, legacy)
        if include_scale_oracles:
            for oracle_name, oracle_prediction in scale_oracle_predictions(
                final, target
            ).items():
                _append_metrics(
                    stages,
                    oracle_name,
                    oracle_prediction,
                    target,
                    shape,
                    true_outage,
                    legacy,
                )
        if output is not None:
            output[start:stop] = final.detach().cpu().numpy().astype(np.complex64)
        if comparison is not None:
            base = torch.as_tensor(np.asarray(comparison[start:stop]), device=device)
            numerator = (final - base).abs().square().sum(dim=(1, 2, 3)).double()
            denominator = base.abs().square().sum(dim=(1, 2, 3)).double().clamp_min(1e-30)
            disagreement[start:stop] = (numerator / denominator).cpu().numpy()
    if output is not None:
        output.flush()
        del output

    arrays = {stage: concatenate_metric_batches(parts) for stage, parts in stages.items()}
    canonical = {
        stage: aggregate_sample_metrics(values) for stage, values in arrays.items()
    }
    legacy_metrics = {stage: accumulator.compute() for stage, accumulator in legacy.items()}
    return {
        "name": name,
        "shape": shape,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "reference_strategy": strategy,
        "references": references,
        "candidate_distances": distances,
        "arrays": arrays,
        "canonical": canonical,
        "legacy": legacy_metrics,
        "disagreement_nmse": disagreement,
        "predicted_outages": int(
            np.sum(priors["outage_probability"][validation] >= target_thresholds)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _saved_roundtrip(
    path: Path,
    channels: np.ndarray,
    validation: np.ndarray,
    shape: object,
    outage: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    prediction = np.load(path, mmap_mode="r")
    batches = []
    legacy = ChannelMetricAccumulator(shape)
    for start in range(0, len(validation), batch_size):
        stop = min(start + batch_size, len(validation))
        pred = torch.as_tensor(np.asarray(prediction[start:stop]), device=device)
        target = torch.as_tensor(np.asarray(channels[validation[start:stop]]), device=device)
        true_outage = torch.as_tensor(outage[validation[start:stop]], device=device)
        batches.append(sample_metric_batch(pred, target, shape, true_outage))
        legacy.update(pred, target, true_outage)
    arrays = concatenate_metric_batches(batches)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "finite": bool(np.isfinite(prediction).all()),
        "canonical": aggregate_sample_metrics(arrays),
        "legacy": legacy.compute(),
    }


def _metric_subset(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    return aggregate_sample_metrics({name: np.asarray(value)[mask] for name, value in values.items()})


def _format_metrics(metrics: dict[str, object]) -> str:
    return "| %.6f | %.6f | %.6f | %.6f |" % (
        metrics["pas"],
        metrics["pdp"],
        metrics["nmse"],
        metrics["score"],
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    return float(spearmanr(left[finite], right[finite]).statistic)


def _gap_requirements(metrics: dict[str, float]) -> dict[str, float]:
    gap = TARGET_SCORE - float(metrics["score"])
    fixed_nmse_delta = gap / 0.4
    fixed_spectral = 0.4 * float(metrics["pas"]) + 0.4 * float(metrics["pdp"])
    required_nmse = 0.2 / max(TARGET_SCORE - fixed_spectral, 1e-12) - 1.0

    def nmse_for(pas_gain: float, pdp_gain: float) -> float:
        remainder = TARGET_SCORE - fixed_spectral - 0.4 * (pas_gain + pdp_gain)
        return 0.2 / max(remainder, 1e-12) - 1.0

    return {
        "gap": gap,
        "pas_only_target": float(metrics["pas"]) + fixed_nmse_delta,
        "pdp_only_target": float(metrics["pdp"]) + fixed_nmse_delta,
        "nmse_only_target": required_nmse,
        "nmse_if_pas_pdp_gain_0_020": nmse_for(0.02, 0.02),
        "nmse_if_pas_pdp_gain_0_025": nmse_for(0.025, 0.025),
    }


def _write_reports(
    output_dir: Path,
    args: argparse.Namespace,
    config: dict,
    metadata: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    validation: np.ndarray,
    observed: np.ndarray,
    teacher: dict[str, dict],
    variants: dict[str, dict],
    saved: dict[str, object],
    oracle: dict[str, object] | None,
    per_sample: dict[str, np.ndarray],
    result: dict[str, object],
) -> None:
    baseline = variants["base"]
    baseline_metrics = baseline["legacy"]["final"]
    canonical_metrics = baseline["canonical"]["final"]
    gap = _gap_requirements(baseline_metrics)
    fold = int(config["split"]["validation_fold"])
    cells = metadata["train_cells"][validation]

    baseline_md = f"""# Scheme E 0.65 Baseline Audit

## Authority

- Git branch: `{_git('branch', '--show-current')}`
- Git commit used by the audit code: `{_git('rev-parse', 'HEAD')}`
- Fold: `{fold}` (strict spatial Fold0)
- Train / validation: `{len(observed)} / {len(validation)}`
- Validation by BS: `{json.dumps({str(int(c)): int(np.sum(cells == c)) for c in np.unique(cells)})}`
- Config: `{args.config}`
- Checkpoint: `{baseline['checkpoint']}`
- Checkpoint SHA256: `{baseline['checkpoint_sha256']}`
- Base Teacher priors: `{args.base_priors or config['spectral_teacher']['oof_output_path']}`
- Fold0 prediction: `{saved['path']}`
- Prediction SHA256: `{saved['sha256']}`
- Evaluator: `schemeE_spectral_gaussian_hybrid/scheme_e/metrics.py`
- Canonical audit helpers: `schemeE_spectral_gaussian_hybrid/scheme_e/diagnostics.py`

## Reproduction

```bash
cd schemeE_spectral_gaussian_hybrid
python scripts/audit_scheme_e_065.py --config {args.config} --policy {args.policy} --projection-report {args.projection_report}
```

## Recomputed Baseline

| PAS | PDP | NMSE | Score |
|---:|---:|---:|---:|
{_format_metrics(baseline_metrics)}

The row-based canonical recomputation is `{canonical_metrics['score']:.8f}`. The legacy
training evaluator is `{baseline_metrics['score']:.8f}`. Their absolute difference is
`{abs(canonical_metrics['score'] - baseline_metrics['score']):.3e}`.

## Gap To 0.65

- Current gap: `{gap['gap']:.6f}`.
- NMSE fixed: PAS alone must reach `{gap['pas_only_target']:.6f}`.
- NMSE fixed: PDP alone must reach `{gap['pdp_only_target']:.6f}`.
- PAS/PDP fixed: NMSE must fall to `{gap['nmse_only_target']:.6f}`.
- PAS and PDP each +0.020: NMSE must reach `{gap['nmse_if_pas_pdp_gain_0_020']:.6f}`.
- PAS and PDP each +0.025: NMSE must reach `{gap['nmse_if_pas_pdp_gain_0_025']:.6f}`.

Measured facts above are Fold0 offline results, not the official online score. The known
official Scheme E score remains `0.59`.
"""
    (output_dir / "BASELINE_AUDIT.md").write_text(baseline_md, encoding="utf-8")

    rows = []
    for variant_name, variant in variants.items():
        for stage in (
            "teacher_projected_seed",
            "hybrid_raw",
            "hybrid_post_projection",
            "final",
        ):
            metrics = variant["legacy"][stage]
            rows.append(
                f"| {variant_name} | {stage} | {metrics['pas']:.6f} | "
                f"{metrics['pdp']:.6f} | {metrics['nmse']:.6f} | {metrics['score']:.6f} |"
            )
    teacher_rows = []
    for name, values in teacher.items():
        teacher_rows.append(
            f"| {name} | {values['pas_mean_nonoutage']:.6f} | "
            f"{values['pdp_mean_nonoutage']:.6f} | "
            f"{values['power_mae_log10_nonoutage']:.6f} |"
        )
    correlation_text = "Not available: no second Teacher variant was supplied."
    if "adaptive" in variants:
        base_teacher = teacher["base"]
        adaptive_teacher = teacher["adaptive"]
        base_final = variants["base"]["arrays"]["final"]
        adaptive_final = variants["adaptive"]["arrays"]["final"]
        correlation_text = (
            "- Teacher PAS delta -> final PAS delta Spearman: "
            f"`{_safe_spearman(adaptive_teacher['pas'] - base_teacher['pas'], adaptive_final['pas'] - base_final['pas']):.4f}`.\n"
            "- Teacher PDP delta -> final PDP delta Spearman: "
            f"`{_safe_spearman(adaptive_teacher['pdp'] - base_teacher['pdp'], adaptive_final['pdp'] - base_final['pdp']):.4f}`.\n"
            "- Final expert disagreement median NMSE: "
            f"`{np.median(variants['adaptive']['disagreement_nmse']):.6f}`."
        )
    oracle_text = "Not available."
    if oracle is not None:
        om = oracle["metrics"]
        oracle_text = (
            f"DIAGNOSTIC ONLY - NOT DEPLOYABLE. Target-informed selection reaches "
            f"PAS `{om['pas']:.6f}`, PDP `{om['pdp']:.6f}`, NMSE `{om['nmse']:.6f}`, "
            f"Score `{om['score']:.6f}` with counts `{oracle['selection_counts']}`."
        )
    bridge_md = f"""# Metric Bridge Audit

## Metric Definition

The task PDF defines PAS as the mean cosine over all positions, subcarriers and UE
antennas; PDP as the mean cosine over all positions and transmit/receive antenna pairs;
and NMSE as one global error-energy ratio. `metrics.py` implements the same FFT axes and
the official formula `0.4*PAS + 0.4*PDP + 0.2/(1+NMSE)`.

The audit also computes every row explicitly. This tests sample order, BS order, complex
layout, outage handling and batch-size aggregation without changing the production metric.

## Coarse Teacher Space

These PAS/PDP values are measured in the compressed coarse-Teacher feature space. They
are useful diagnostics but are not interchangeable with final channel PAS/PDP.

| Teacher | PAS | PDP | Power MAE (log10) |
|---|---:|---:|---:|
{chr(10).join(teacher_rows)}

## Complex Channel Stages

| Variant | Stage | PAS | PDP | NMSE | Score |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Saved NPY Round Trip

- Shape: `{saved['shape']}`; dtype: `{saved['dtype']}`; finite: `{saved['finite']}`.
- Streaming legacy Score: `{baseline_metrics['score']:.8f}`.
- Reloaded legacy Score: `{saved['legacy']['score']:.8f}`.
- Absolute score delta: `{abs(saved['legacy']['score'] - baseline_metrics['score']):.3e}`.

## Per-sample Transfer

{correlation_text}

## Existing Expert Oracle

{oracle_text}
"""
    (output_dir / "METRIC_BRIDGE_AUDIT.md").write_text(bridge_md, encoding="utf-8")

    final = baseline["arrays"]["final"]
    score = final["sample_score"]
    nonoutage = ~targets["outage"][validation].astype(bool)
    quantiles = np.quantile(score[nonoutage], [0.2, 0.8])
    target_power = final["target_log_power"]
    power_quantiles = np.quantile(target_power[nonoutage], [1 / 3, 2 / 3])
    nearest = per_sample["nearest_distance_m"]
    distance_quantiles = np.quantile(nearest, [1 / 3, 2 / 3])
    density = per_sample["local_density_12m"]
    density_quantiles = np.quantile(density, [1 / 3, 2 / 3])
    groups = {
        "best_20pct": nonoutage & (score >= quantiles[1]),
        "middle_60pct": nonoutage & (score > quantiles[0]) & (score < quantiles[1]),
        "worst_20pct": nonoutage & (score <= quantiles[0]),
        "bs0": cells == 0,
        "bs1": cells == 1,
        "high_power": nonoutage & (target_power >= power_quantiles[1]),
        "mid_power": nonoutage & (target_power > power_quantiles[0]) & (target_power < power_quantiles[1]),
        "low_power": nonoutage & (target_power <= power_quantiles[0]),
        "near_support": nearest <= distance_quantiles[0],
        "mid_support": (nearest > distance_quantiles[0]) & (nearest < distance_quantiles[1]),
        "far_support": nearest >= distance_quantiles[1],
        "high_density": density >= density_quantiles[1],
        "low_density": density <= density_quantiles[0],
        "outage": ~nonoutage,
        "nonoutage": nonoutage,
    }
    slice_rows = []
    for name, mask in groups.items():
        metrics = _metric_subset(final, mask)
        slice_rows.append(
            f"| {name} | {int(mask.sum())} | {metrics['pas']:.5f} | {metrics['pdp']:.5f} | "
            f"{metrics['nmse']:.5f} | {metrics['score']:.5f} |"
        )
    contribution = final["error_energy"] / max(final["target_energy"].sum(), 1e-30)
    ordering = np.argsort(contribution)[::-1]
    top1 = max(1, int(np.ceil(0.01 * len(ordering))))
    top5 = max(1, int(np.ceil(0.05 * len(ordering))))
    total_error = max(float(final["error_energy"].sum()), 1e-30)
    mining_md = f"""# Fold0 Error Mining

## Main Slices

| Slice | Samples | PAS | PDP | NMSE | Score |
|---|---:|---:|---:|---:|---:|
{chr(10).join(slice_rows)}

## Correlations

- Nearest support distance vs sample Score: `{_safe_spearman(nearest, score):.4f}`.
- Local 12 m density vs sample Score: `{_safe_spearman(density, score):.4f}`.
- Target log-power vs sample NMSE: `{_safe_spearman(target_power[nonoutage], final['sample_nmse'][nonoutage]):.4f}`.
- Absolute power error vs sample NMSE: `{_safe_spearman(np.abs(final['prediction_log_power'][nonoutage] - target_power[nonoutage]), final['sample_nmse'][nonoutage]):.4f}`.

## Extreme NMSE Contribution

- Worst 1% of samples contribute `{final['error_energy'][ordering[:top1]].sum() / total_error:.2%}` of total error energy.
- Worst 5% of samples contribute `{final['error_energy'][ordering[:top5]].sum() / total_error:.2%}` of total error energy.

## Interpretation Rule

Distances and densities are computed only from Fold0-visible, same-BS, non-outage support.
The saved NPZ contains the raw per-sample fields so subsequent hypotheses can be checked
without rerunning the network.
"""
    (output_dir / "ERROR_MINING.md").write_text(mining_md, encoding="utf-8")

    save_json(output_dir / "LEVEL0_RESULTS.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict Scheme E Fold0 baseline, metric bridge and oracle audit"
    )
    parser.add_argument("--config", default="configs/v4_fold_best.json")
    parser.add_argument(
        "--policy", default="reports/generated/v4_attempt1_policy.json"
    )
    parser.add_argument(
        "--projection-report",
        default="reports/generated/v4_attempt1_output_projection.json",
    )
    parser.add_argument("--base-priors")
    parser.add_argument("--adaptive-priors")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="../research/scheme_e_065")
    parser.add_argument("--expected-score", type=float, default=BEST_SCORE)
    parser.add_argument("--score-tolerance", type=float, default=5e-4)
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    config["runtime"]["device"] = args.device
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_npz(Path(config["preprocessing"]["artifact_dir"]) / "metadata.npz")
    targets = _load_npz(config["spectral"]["target_path"])
    base_priors_path = args.base_priors or config["spectral_teacher"]["oof_output_path"]
    base_priors = _load_npz(base_priors_path)
    fold = int(config["split"]["validation_fold"])
    available = base_priors["available"].astype(bool)
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    validation = np.flatnonzero(available & validation_mask)
    observed = np.flatnonzero(available & ~validation_mask)
    if not len(validation) or not len(observed):
        raise RuntimeError("Strict Fold0 split is empty")
    channels = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    checkpoint_path = Path(config["hybrid"]["output_dir"]) / "best.pt"
    policy = _outage_policy(args.policy)
    output_projection = _projection_settings(config, args.projection_report)

    prediction_path = output_dir / "FOLD0_BASELINE_PREDICTION.npy"
    variants: dict[str, dict] = {}
    teacher = {"base": _teacher_metrics(base_priors, targets, validation)}
    variants["base"] = _collect_variant(
        name="base",
        config=config,
        priors=base_priors,
        metadata=metadata,
        targets=targets,
        channels=channels,
        validation=validation,
        observed=observed,
        checkpoint_path=checkpoint_path,
        policy=policy,
        output_projection=output_projection,
        device=device,
        save_path=prediction_path,
        compare_path=None,
        include_scale_oracles=True,
    )
    saved = _saved_roundtrip(
        prediction_path,
        channels,
        validation,
        variants["base"]["shape"],
        metadata["outage"],
        device,
        int(config["hybrid"].get("validation_batch_size", 4)),
    )
    baseline_scores = {
        "canonical_in_memory": float(variants["base"]["canonical"]["final"]["score"]),
        "legacy_in_memory": float(variants["base"]["legacy"]["final"]["score"]),
        "canonical_saved_npy": float(saved["canonical"]["score"]),
        "legacy_saved_npy": float(saved["legacy"]["score"]),
    }
    score_deltas = {
        name: abs(score - args.expected_score)
        for name, score in baseline_scores.items()
    }
    if max(score_deltas.values()) > args.score_tolerance:
        failure = {
            "status": "FAIL_BASELINE_REPRODUCTION",
            "expected_score": args.expected_score,
            "score_tolerance": args.score_tolerance,
            "observed_scores": baseline_scores,
            "absolute_deltas": score_deltas,
            "rule": "No downstream diagnostic or training is permitted until this passes.",
        }
        save_json(output_dir / "BASELINE_AUDIT_FAILED.json", failure)
        raise RuntimeError(json.dumps(failure, ensure_ascii=False))

    if args.adaptive_priors:
        adaptive_priors = _load_npz(args.adaptive_priors)
        teacher["adaptive"] = _teacher_metrics(adaptive_priors, targets, validation)
        variants["adaptive"] = _collect_variant(
            name="adaptive",
            config=config,
            priors=adaptive_priors,
            metadata=metadata,
            targets=targets,
            channels=channels,
            validation=validation,
            observed=observed,
            checkpoint_path=checkpoint_path,
            policy=policy,
            output_projection=output_projection,
            device=device,
            save_path=None,
            compare_path=prediction_path,
            include_scale_oracles=False,
        )

    np.save(output_dir / "FOLD0_INDICES.npy", validation.astype(np.int64))

    cells = metadata["train_cells"][validation].astype(np.int64)
    positions = metadata["train_positions"][validation].astype(np.float32)
    nearest = np.zeros(len(validation), dtype=np.float64)
    mean_k = np.zeros(len(validation), dtype=np.float64)
    density_12 = np.zeros(len(validation), dtype=np.int64)
    for cell in np.unique(cells):
        selected = np.flatnonzero(cells == cell)
        support = observed[
            (metadata["train_cells"][observed] == cell)
            & ~metadata["outage"][observed].astype(bool)
        ]
        tree = cKDTree(metadata["train_positions"][support, :2])
        distances, _ = tree.query(positions[selected, :2], k=min(16, len(support)))
        distances = np.asarray(distances).reshape(len(selected), -1)
        nearest[selected] = distances[:, 0]
        mean_k[selected] = distances.mean(axis=1)
        density_12[selected] = np.asarray(
            [len(values) for values in tree.query_ball_point(positions[selected, :2], 12.0)]
        )

    base_final = variants["base"]["arrays"]["final"]
    per_sample: dict[str, np.ndarray] = {
        "sample_id": validation.astype(np.int64),
        "cell_id": cells,
        "position": positions,
        "true_outage": metadata["outage"][validation].astype(bool),
        "outage_probability": base_priors["outage_probability"][validation].astype(np.float32),
        "nearest_distance_m": nearest.astype(np.float32),
        "mean_16nn_distance_m": mean_k.astype(np.float32),
        "local_density_12m": density_12.astype(np.int32),
        "teacher_pas": np.asarray(teacher["base"]["pas"], dtype=np.float32),
        "teacher_pdp": np.asarray(teacher["base"]["pdp"], dtype=np.float32),
    }
    per_sample.update(
        {f"final_{name}": np.asarray(value) for name, value in base_final.items()}
    )
    for stage in ("teacher_projected_seed", "hybrid_raw", "hybrid_post_projection"):
        values = variants["base"]["arrays"][stage]
        for field in ("pas", "pdp", "sample_nmse", "sample_score", "prediction_log_power"):
            per_sample[f"{stage}_{field}"] = np.asarray(values[field])

    oracle = None
    if len(variants) > 1:
        expert_arrays = {name: value["arrays"]["final"] for name, value in variants.items()}
        oracle = target_informed_expert_oracle(expert_arrays)
        per_sample["expert_oracle_selection"] = np.asarray(oracle.pop("selection"))
        per_sample["adaptive_teacher_pas"] = np.asarray(
            teacher["adaptive"]["pas"], dtype=np.float32
        )
        per_sample["adaptive_teacher_pdp"] = np.asarray(
            teacher["adaptive"]["pdp"], dtype=np.float32
        )
        per_sample["adaptive_disagreement_nmse"] = variants["adaptive"][
            "disagreement_nmse"
        ].astype(np.float32)
        for field in ("pas", "pdp", "sample_nmse", "sample_score", "prediction_log_power"):
            per_sample[f"adaptive_final_{field}"] = np.asarray(
                variants["adaptive"]["arrays"]["final"][field]
            )
    np.savez_compressed(output_dir / "PER_SAMPLE_METRICS.npz", **per_sample)

    serializable_variants = {}
    for name, values in variants.items():
        serializable_variants[name] = {
            "checkpoint": values["checkpoint"],
            "checkpoint_sha256": values["checkpoint_sha256"],
            "reference_strategy": values["reference_strategy"],
            "canonical": values["canonical"],
            "legacy": values["legacy"],
            "predicted_outages": values["predicted_outages"],
            "elapsed_seconds": values["elapsed_seconds"],
        }
    result = {
        "status": "PASS",
        "diagnostic_only_oracles": True,
        "fold": fold,
        "training_samples": int(len(observed)),
        "validation_samples": int(len(validation)),
        "config": args.config,
        "policy": args.policy,
        "projection_report": args.projection_report,
        "output_projection": output_projection,
        "baseline_reproduction": {
            "expected_score": args.expected_score,
            "score_tolerance": args.score_tolerance,
            "observed_scores": baseline_scores,
            "absolute_deltas": score_deltas,
        },
        "teacher": {
            name: {key: value for key, value in values.items() if np.isscalar(value)}
            for name, values in teacher.items()
        },
        "variants": serializable_variants,
        "saved_roundtrip": saved,
        "expert_oracle": oracle,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_reports(
        output_dir,
        args,
        config,
        metadata,
        targets,
        validation,
        observed,
        teacher,
        variants,
        saved,
        oracle,
        per_sample,
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
