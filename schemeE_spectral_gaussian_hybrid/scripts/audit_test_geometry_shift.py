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
)
from scheme_e.distribution_audit import (
    domain_classifier_auc,
    feature_shift_report,
    fixed_bin_importance_weights,
    legacy_spatial_block_split,
    same_cell_support_features,
    summarize_features,
    weighted_aggregate_sample_metrics,
)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cell_one_hot(cells: np.ndarray, count: int) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64)
    output = np.zeros((len(cells), int(count)), dtype=np.float64)
    output[np.arange(len(cells)), cells] = 1.0
    return output


def _query_features(
    *,
    support_indices: np.ndarray,
    query_positions: np.ndarray,
    query_cells: np.ndarray,
    query_geometry: np.ndarray,
    geometry_names: list[str],
    metadata: dict[str, np.ndarray],
    cell_count: int,
) -> dict[str, object]:
    support, support_names = same_cell_support_features(
        metadata["train_positions"][support_indices],
        metadata["train_cells"][support_indices],
        query_positions,
        query_cells,
    )
    cell_features = _cell_one_hot(query_cells, cell_count)
    cell_names = [f"cell_{index}" for index in range(cell_count)]
    geometry = np.asarray(query_geometry, dtype=np.float64)
    environment_indices = np.asarray(
        [index for index, name in enumerate(geometry_names) if name.startswith(("local_", "corridor_", "fresnel_", "surface_"))],
        dtype=np.int64,
    )
    link_names = {
        "distance_3d",
        "distance_2d",
        "inverse_distance",
        "azimuth_sin",
        "azimuth_cos",
        "elevation_sin",
    }
    environment_index_set = set(environment_indices.tolist())
    link_environment_indices = np.asarray(
        [
            index
            for index, name in enumerate(geometry_names)
            if index in environment_index_set or name in link_names
        ],
        dtype=np.int64,
    )
    common = np.concatenate([cell_features, support], axis=1)
    return {
        "support": support,
        "support_names": support_names,
        "support_domain": common,
        "support_domain_names": [*cell_names, *support_names],
        "environment_domain": np.concatenate(
            [common, geometry[:, environment_indices]], axis=1
        ),
        "environment_domain_names": [
            *cell_names,
            *support_names,
            *[geometry_names[index] for index in environment_indices],
        ],
        "link_environment_domain": np.concatenate(
            [common, geometry[:, link_environment_indices]], axis=1
        ),
        "link_environment_domain_names": [
            *cell_names,
            *support_names,
            *[geometry_names[index] for index in link_environment_indices],
        ],
        "full_domain": np.concatenate([common, geometry], axis=1),
        "cells": np.asarray(query_cells, dtype=np.int64),
    }


def _dataset_summary(
    features: dict[str, object],
    geometry_names: list[str],
) -> dict[str, object]:
    cells = np.asarray(features["cells"], dtype=np.int64)
    support = np.asarray(features["support"], dtype=np.float64)
    names = list(features["support_names"])
    return {
        "samples": int(len(cells)),
        "cell_counts": {
            str(int(cell)): int(np.sum(cells == cell)) for cell in np.unique(cells)
        },
        "support_features": summarize_features(support, names),
        "domain_feature_count": int(
            np.asarray(features["full_domain"]).shape[1]
        ),
        "geometry_feature_count": int(len(geometry_names)),
    }


@torch.no_grad()
def _evaluate_prediction(
    path: str | Path,
    channels: np.ndarray,
    validation: np.ndarray,
    outage: np.ndarray,
    shape: ChannelShape,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    prediction = np.load(path, mmap_mode="r")
    if tuple(prediction.shape) != (len(validation), *shape.raw_shape):
        raise ValueError(
            f"prediction {path} has shape {prediction.shape}, expected "
            f"{(len(validation), *shape.raw_shape)}"
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
                torch.as_tensor(np.array(channels[selected], copy=True), device=device),
                shape,
                torch.as_tensor(outage[selected].astype(bool), device=device),
            )
        )
        if start == 0 or stop == len(validation) or stop % 200 == 0:
            print(f"metric pass {Path(path).name}: {stop}/{len(validation)}", flush=True)
    arrays = concatenate_metric_batches(parts)
    return aggregate_sample_metrics(arrays), arrays


def _subset_metrics(
    arrays: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, float | int]:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return {"samples": 0}
    return aggregate_sample_metrics(
        {name: np.asarray(values)[selected] for name, values in arrays.items()}
    )


def _metric_slices(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    validation_cells: np.ndarray,
    validation_distance: np.ndarray,
    test_cells: np.ndarray,
    test_distance: np.ndarray,
    edges: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in np.unique(np.concatenate([validation_cells, test_cells])):
        for index in range(len(edges) - 1):
            lower = float(edges[index])
            upper = float(edges[index + 1])
            validation_mask = (
                (validation_cells == cell)
                & (validation_distance >= lower)
                & (validation_distance < upper)
            )
            test_mask = (
                (test_cells == cell)
                & (test_distance >= lower)
                & (test_distance < upper)
            )
            baseline_metrics = _subset_metrics(baseline, validation_mask)
            candidate_metrics = _subset_metrics(candidate, validation_mask)
            gain = None
            if int(baseline_metrics.get("samples", 0)):
                gain = float(candidate_metrics["score"]) - float(
                    baseline_metrics["score"]
                )
            rows.append(
                {
                    "cell": int(cell),
                    "minimum_m": lower,
                    "maximum_m": None if np.isinf(upper) else upper,
                    "fold0_samples": int(validation_mask.sum()),
                    "test_samples": int(test_mask.sum()),
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "gain": gain,
                }
            )
    return rows


def _by_cell(
    arrays: dict[str, np.ndarray], cells: np.ndarray
) -> dict[str, dict[str, float | int]]:
    return {
        str(int(cell)): _subset_metrics(arrays, cells == cell)
        for cell in np.unique(cells)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit observable geometry shift between the original Scheme1 split, "
            "strict Fold0, and the 500 unlabeled test positions"
        )
    )
    parser.add_argument("--config", default="configs/v4_attempt1_structured.json")
    parser.add_argument(
        "--baseline-prediction",
        default="../research/scheme_e_065/FOLD0_QUALITY_GATED_PREDICTION.npy",
    )
    parser.add_argument(
        "--candidate-prediction",
        default=(
            "artifacts/scheme_e_065/l0_022_quality_gated_magnitude/"
            "Fold0_Quality_Gated_Magnitude.npy"
        ),
    )
    parser.add_argument("--expected-baseline", type=float, default=0.631581)
    parser.add_argument("--expected-candidate", type=float, default=0.633628)
    parser.add_argument("--score-tolerance", type=float, default=0.0001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument(
        "--output-dir", default="artifacts/scheme_e_065/l0_023_geometry_shift"
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_023_TEST_GEOMETRY_SHIFT.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    metadata = _load_npz(artifact_dir / "metadata.npz")
    manifest = _read_json(artifact_dir / "manifest.json")
    fold = int(config["split"]["validation_fold"])
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    validation = np.flatnonzero(validation_mask)
    observed = np.flatnonzero(~validation_mask)
    visible_nonoutage = observed[~metadata["outage"][observed].astype(bool)]
    full_nonoutage = np.flatnonzero(~metadata["outage"].astype(bool))
    old_training, old_validation = legacy_spatial_block_split(
        metadata["train_positions"], metadata["train_cells"]
    )
    old_nonoutage = old_training[~metadata["outage"][old_training].astype(bool)]
    if len(old_training) != 3369 or len(old_validation) != 631:
        raise RuntimeError(
            "Legacy Scheme1 split reproduction failed: "
            f"{len(old_training)}/{len(old_validation)}"
        )

    cell_count = int(manifest["setup"]["Q"])
    geometry_names = [str(name) for name in manifest["geometry_feature_names"]]
    strict_features = _query_features(
        support_indices=visible_nonoutage,
        query_positions=metadata["train_positions"][validation],
        query_cells=metadata["train_cells"][validation],
        query_geometry=metadata["train_geometry_features"][validation],
        geometry_names=geometry_names,
        metadata=metadata,
        cell_count=cell_count,
    )
    old_features = _query_features(
        support_indices=old_nonoutage,
        query_positions=metadata["train_positions"][old_validation],
        query_cells=metadata["train_cells"][old_validation],
        query_geometry=metadata["train_geometry_features"][old_validation],
        geometry_names=geometry_names,
        metadata=metadata,
        cell_count=cell_count,
    )
    test_features = _query_features(
        support_indices=full_nonoutage,
        query_positions=metadata["test_positions"],
        query_cells=metadata["test_cells"],
        query_geometry=metadata["test_geometry_features"],
        geometry_names=geometry_names,
        metadata=metadata,
        cell_count=cell_count,
    )

    strict_support_domain = np.asarray(strict_features["support_domain"])
    old_support_domain = np.asarray(old_features["support_domain"])
    test_support_domain = np.asarray(test_features["support_domain"])
    strict_environment_domain = np.asarray(strict_features["environment_domain"])
    old_environment_domain = np.asarray(old_features["environment_domain"])
    test_environment_domain = np.asarray(test_features["environment_domain"])
    strict_link_environment_domain = np.asarray(
        strict_features["link_environment_domain"]
    )
    old_link_environment_domain = np.asarray(old_features["link_environment_domain"])
    test_link_environment_domain = np.asarray(test_features["link_environment_domain"])
    strict_full_domain = np.asarray(strict_features["full_domain"])
    old_full_domain = np.asarray(old_features["full_domain"])
    test_full_domain = np.asarray(test_features["full_domain"])
    full_names = [
        *list(strict_features["support_domain_names"]),
        *geometry_names,
    ]
    strict_nearest = np.asarray(strict_features["support"])[:, 0]
    old_nearest = np.asarray(old_features["support"])[:, 0]
    test_nearest = np.asarray(test_features["support"])[:, 0]

    domain_shift = {
        "strict_fold0_to_test": {
            "support_only_auc": domain_classifier_auc(
                strict_support_domain, test_support_domain
            ),
            "full_geometry_auc": domain_classifier_auc(
                strict_full_domain, test_full_domain
            ),
            "environment_only_auc": domain_classifier_auc(
                strict_environment_domain, test_environment_domain
            ),
            "link_environment_auc": domain_classifier_auc(
                strict_link_environment_domain, test_link_environment_domain
            ),
            "support_shift": feature_shift_report(
                strict_support_domain,
                test_support_domain,
                list(strict_features["support_domain_names"]),
            ),
            "link_environment_shift": feature_shift_report(
                strict_link_environment_domain,
                test_link_environment_domain,
                list(strict_features["link_environment_domain_names"]),
            ),
            "full_geometry_shift": feature_shift_report(
                strict_full_domain, test_full_domain, full_names
            ),
        },
        "scheme1_old_validation_to_test": {
            "support_only_auc": domain_classifier_auc(
                old_support_domain, test_support_domain
            ),
            "full_geometry_auc": domain_classifier_auc(
                old_full_domain, test_full_domain
            ),
            "environment_only_auc": domain_classifier_auc(
                old_environment_domain, test_environment_domain
            ),
            "link_environment_auc": domain_classifier_auc(
                old_link_environment_domain, test_link_environment_domain
            ),
            "support_shift": feature_shift_report(
                old_support_domain,
                test_support_domain,
                list(old_features["support_domain_names"]),
            ),
            "link_environment_shift": feature_shift_report(
                old_link_environment_domain,
                test_link_environment_domain,
                list(old_features["link_environment_domain_names"]),
            ),
            "full_geometry_shift": feature_shift_report(
                old_full_domain, test_full_domain, full_names
            ),
        },
    }

    fold_geometry_audit = []
    for fold_index, fold_mask_raw in enumerate(metadata["validation_masks"]):
        fold_mask = np.asarray(fold_mask_raw, dtype=bool)
        fold_validation = np.flatnonzero(fold_mask)
        fold_visible = np.flatnonzero(
            (~fold_mask) & (~metadata["outage"].astype(bool))
        )
        fold_features = _query_features(
            support_indices=fold_visible,
            query_positions=metadata["train_positions"][fold_validation],
            query_cells=metadata["train_cells"][fold_validation],
            query_geometry=metadata["train_geometry_features"][fold_validation],
            geometry_names=geometry_names,
            metadata=metadata,
            cell_count=cell_count,
        )
        support_auc = domain_classifier_auc(
            np.asarray(fold_features["support_domain"]), test_support_domain
        )
        link_environment_auc = domain_classifier_auc(
            np.asarray(fold_features["link_environment_domain"]),
            test_link_environment_domain,
        )
        fold_nearest = np.asarray(fold_features["support"])[:, 0]
        support_auc_max = max(
            float(support_auc["linear_oof_auc"]),
            float(support_auc["nonlinear_oof_auc"]),
        )
        link_auc_max = max(
            float(link_environment_auc["linear_oof_auc"]),
            float(link_environment_auc["nonlinear_oof_auc"]),
        )
        fold_geometry_audit.append(
            {
                "fold": int(fold_index),
                "samples": int(len(fold_validation)),
                "nearest_median_m": float(np.median(fold_nearest)),
                "nearest_median_gap_to_test_m": abs(
                    float(np.median(fold_nearest)) - float(np.median(test_nearest))
                ),
                "support_only_auc": support_auc,
                "link_environment_auc": link_environment_auc,
                "match_rank_value": float(
                    support_auc_max + 0.5 * link_auc_max
                ),
            }
        )
    fold_geometry_audit.sort(key=lambda item: float(item["match_rank_value"]))
    for rank, item in enumerate(fold_geometry_audit, start=1):
        item["match_rank"] = int(rank)

    strict_auc = domain_shift["strict_fold0_to_test"]["support_only_auc"]
    strict_support_auc = max(
        float(strict_auc["linear_oof_auc"]),
        float(strict_auc["nonlinear_oof_auc"]),
    )
    strict_link_auc_payload = domain_shift["strict_fold0_to_test"][
        "link_environment_auc"
    ]
    strict_link_environment_auc = max(
        float(strict_link_auc_payload["linear_oof_auc"]),
        float(strict_link_auc_payload["nonlinear_oof_auc"]),
    )
    nearest_median_gap = abs(float(np.median(strict_nearest)) - float(np.median(test_nearest)))
    support_representative = bool(
        strict_support_auc <= 0.70 and nearest_median_gap <= 2.0
    )
    propagation_representative = bool(strict_link_environment_auc <= 0.75)

    metrics: dict[str, object] | None = None
    metric_arrays: dict[str, dict[str, np.ndarray]] = {}
    edges = (0.0, 4.0, 6.0, 8.0, 10.0, 12.0, np.inf)
    if not args.geometry_only:
        device = choose_device(args.device)
        shape = ChannelShape.from_setup(manifest["setup"])
        channels = np.load(
            Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
            mmap_mode="r",
        )
        baseline_metrics, baseline_arrays = _evaluate_prediction(
            args.baseline_prediction,
            channels,
            validation,
            metadata["outage"],
            shape,
            device,
            int(args.batch_size),
        )
        candidate_metrics, candidate_arrays = _evaluate_prediction(
            args.candidate_prediction,
            channels,
            validation,
            metadata["outage"],
            shape,
            device,
            int(args.batch_size),
        )
        for observed_score, expected, name in (
            (baseline_metrics["score"], args.expected_baseline, "baseline"),
            (candidate_metrics["score"], args.expected_candidate, "candidate"),
        ):
            if abs(float(observed_score) - float(expected)) > float(args.score_tolerance):
                raise RuntimeError(
                    f"{name} reproduction failed: observed={float(observed_score):.6f}, "
                    f"expected={float(expected):.6f}"
                )
        weights, weight_report = fixed_bin_importance_weights(
            metadata["train_cells"][validation],
            strict_nearest,
            metadata["test_cells"],
            test_nearest,
            edges=edges,
        )
        weighted_baseline = weighted_aggregate_sample_metrics(
            baseline_arrays, weights
        )
        weighted_candidate = weighted_aggregate_sample_metrics(
            candidate_arrays, weights
        )
        direct_gain = float(candidate_metrics["score"]) - float(
            baseline_metrics["score"]
        )
        weighted_gain = float(weighted_candidate["score"]) - float(
            weighted_baseline["score"]
        )
        baseline_by_cell = _by_cell(
            baseline_arrays, metadata["train_cells"][validation]
        )
        candidate_by_cell = _by_cell(
            candidate_arrays, metadata["train_cells"][validation]
        )
        cell_gains = {
            cell: float(candidate_by_cell[cell]["score"])
            - float(baseline_by_cell[cell]["score"])
            for cell in baseline_by_cell
        }
        candidate_gate_signal = bool(
            weighted_gain >= 0.003 and min(cell_gains.values()) >= 0.0
        )
        metrics = {
            "strict_fold0": {
                "quality_gated_v4": baseline_metrics,
                "minor_magnitude_candidate": candidate_metrics,
                "direct_gain": direct_gain,
                "by_cell": {
                    "quality_gated_v4": baseline_by_cell,
                    "minor_magnitude_candidate": candidate_by_cell,
                    "gain": cell_gains,
                },
            },
            "test_geometry_reweighted_fold0": {
                "quality_gated_v4": weighted_baseline,
                "minor_magnitude_candidate": weighted_candidate,
                "gain": weighted_gain,
                "weight_diagnostics": weight_report,
            },
            "fixed_cell_distance_slices": _metric_slices(
                baseline_arrays,
                candidate_arrays,
                metadata["train_cells"][validation],
                strict_nearest,
                metadata["test_cells"],
                test_nearest,
                edges,
            ),
            "candidate_gate_signal": candidate_gate_signal,
        }
        metric_arrays = {
            "baseline": baseline_arrays,
            "candidate": candidate_arrays,
        }

    if support_representative and propagation_representative:
        distribution_decision = "STRICT_FOLD0_SUPPORT_DIFFICULTY_REPRESENTATIVE"
    elif support_representative:
        distribution_decision = (
            "SUPPORT_DIFFICULTY_MATCHES_BUT_LINK_ENVIRONMENT_SHIFT_EXISTS"
        )
    else:
        distribution_decision = "OBSERVABLE_SUPPORT_SHIFT_CONFIRMED"
    candidate_decision = "NOT_EVALUATED"
    if metrics is not None:
        candidate_decision = (
            "ALLOW_INNER_DISTANCE_GATE_PROBE"
            if bool(metrics["candidate_gate_signal"])
            else "KEEP_MINOR_ARTIFACT_ONLY"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "observable_features.npz",
        strict_indices=validation.astype(np.int64),
        old_validation_indices=old_validation.astype(np.int64),
        strict_support=np.asarray(strict_features["support"], dtype=np.float32),
        old_support=np.asarray(old_features["support"], dtype=np.float32),
        test_support=np.asarray(test_features["support"], dtype=np.float32),
        strict_cells=metadata["train_cells"][validation].astype(np.int16),
        old_cells=metadata["train_cells"][old_validation].astype(np.int16),
        test_cells=metadata["test_cells"].astype(np.int16),
        **{
            f"baseline_{name}": value
            for name, value in metric_arrays.get("baseline", {}).items()
        },
        **{
            f"candidate_{name}": value
            for name, value in metric_arrays.get("candidate", {}).items()
        },
    )

    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-023",
        "diagnostic_only": True,
        "uses_test_channel_labels": False,
        "hypothesis": (
            "Observable support and RF-geometry shift may explain the offline/online "
            "score reversal between the original Scheme1 split and strict Fold0."
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_bootstrap.PROJECT_ROOT.parent,
            text=True,
        ).strip(),
        "leakage_control": {
            "test_usage": "positions and precomputed RF geometry only; no test channels exist",
            "fold0_target_usage": "metric evaluation and fixed diagnostic slices only",
            "domain_classifier_target": "validation-versus-test domain label, never channel quality",
            "policy_fitting": "none",
        },
        "datasets": {
            "strict_fold0": _dataset_summary(strict_features, geometry_names),
            "scheme1_old_validation": _dataset_summary(old_features, geometry_names),
            "test": _dataset_summary(test_features, geometry_names),
        },
        "domain_shift": domain_shift,
        "all_fold_geometry_match": fold_geometry_audit,
        "nearest_median_gap_m": {
            "strict_fold0_to_test": nearest_median_gap,
            "scheme1_old_validation_to_test": abs(
                float(np.median(old_nearest)) - float(np.median(test_nearest))
            ),
        },
        "metrics": metrics,
        "decision": {
            "distribution": distribution_decision,
            "minor_candidate": candidate_decision,
            "support_representative_rule": {
                "maximum_support_domain_auc": 0.70,
                "maximum_link_environment_domain_auc": 0.75,
                "maximum_nearest_median_gap_m": 2.0,
                "observed_support_domain_auc": strict_support_auc,
                "observed_link_environment_domain_auc": strict_link_environment_auc,
                "observed_nearest_median_gap_m": nearest_median_gap,
            },
            "candidate_probe_rule": (
                "test-geometry-reweighted gain >=0.003 and nonnegative gain in both cells"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    save_json(report_path, report)

    strict_summary = report["datasets"]["strict_fold0"]["support_features"]["nearest_m"]
    old_summary = report["datasets"]["scheme1_old_validation"]["support_features"]["nearest_m"]
    test_summary = report["datasets"]["test"]["support_features"]["nearest_m"]
    metric_text = "- Channel metric pass: skipped (`--geometry-only`).\n"
    if metrics is not None:
        strict_metric = metrics["strict_fold0"]
        weighted_metric = metrics["test_geometry_reweighted_fold0"]
        metric_text = (
            f"- Strict baseline/candidate: `{float(strict_metric['quality_gated_v4']['score']):.6f}` / "
            f"`{float(strict_metric['minor_magnitude_candidate']['score']):.6f}` "
            f"(gain `{float(strict_metric['direct_gain']):+.6f}`).\n"
            f"- Test-geometry-reweighted baseline/candidate: "
            f"`{float(weighted_metric['quality_gated_v4']['score']):.6f}` / "
            f"`{float(weighted_metric['minor_magnitude_candidate']['score']):.6f}` "
            f"(gain `{float(weighted_metric['gain']):+.6f}`).\n"
        )
    report_path.with_suffix(".md").write_text(
        "# L0-023 Test Geometry Shift Audit\n\n"
        "This report uses test positions and RF geometry only. It never uses test channel labels.\n\n"
        f"- Strict Fold0 nearest support median: `{float(strict_summary['median']):.3f} m`.\n"
        f"- Original Scheme1 validation median: `{float(old_summary['median']):.3f} m`.\n"
        f"- Test nearest support median: `{float(test_summary['median']):.3f} m`.\n"
        f"- Strict-to-test support-domain AUC: `{strict_support_auc:.4f}`.\n"
        f"- Strict-to-test link/environment AUC: `{strict_link_environment_auc:.4f}`.\n"
        f"- Best geometry-matched fold: `{int(fold_geometry_audit[0]['fold'])}` "
        f"(current Fold0 rank `{next(int(item['match_rank']) for item in fold_geometry_audit if int(item['fold']) == fold)}/{len(fold_geometry_audit)}`).\n"
        f"- Distribution decision: `{distribution_decision}`.\n"
        + metric_text
        + f"- Minor candidate decision: `{candidate_decision}`.\n\n"
        "Fold0 numbers are offline diagnostics, not the official online score.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
