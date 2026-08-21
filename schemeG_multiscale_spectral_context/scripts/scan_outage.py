from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_g.config import choose_device, load_config, save_json
from scheme_g.context_data import ContextRepository
from scheme_g.context_training import (
    evaluate_context_thresholds,
    load_context_checkpoint,
    predict_indices,
)
from scheme_g.data import balanced_limit, load_metadata, split_indices
from scheme_g.metrics import official_score


def _slice_predictions(
    predictions: dict[str, np.ndarray], selected: np.ndarray
) -> dict[str, np.ndarray]:
    count = len(next(iter(predictions.values())))
    return {
        name: value[selected] if len(value) == count else value
        for name, value in predictions.items()
    }


def _combine_cell_reports(reports: tuple[dict[str, float], ...]) -> dict[str, object]:
    nonzero = sum(int(report["metric_nonzero_count"]) for report in reports)
    pas = sum(
        float(report["pas"]) * int(report["metric_nonzero_count"]) for report in reports
    ) / max(nonzero, 1)
    pdp = sum(
        float(report["pdp"]) * int(report["metric_nonzero_count"]) for report in reports
    ) / max(nonzero, 1)
    numerator = sum(float(report["nmse_numerator"]) for report in reports)
    denominator = sum(float(report["nmse_denominator"]) for report in reports)
    nmse = numerator / max(denominator, 1e-30)
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "score": official_score(pas, pdp, nmse),
        "by_cell": {
            str(int(report["cell_id"])): {
                "threshold": float(report["threshold"]),
                "soft_strength": float(report["soft_strength"]),
                "spectral_prior_alpha": float(report["spectral_prior_alpha"]),
                "score": float(report["score"]),
            }
            for report in reports
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan conservative outage thresholds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 0.99, 0.999],
    )
    parser.add_argument(
        "--soft-strengths",
        nargs="*",
        type=float,
        default=[0.0, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--spectral-prior-alphas",
        nargs="*",
        type=float,
        default=[0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(config["runtime"]["device"])
    metadata = load_metadata(config)
    training, validation = split_indices(metadata, config)
    if not len(validation):
        raise ValueError("Outage scanning requires a validation fold")
    with np.load(config["encoding"]["output_path"]) as source:
        available = (
            source["available"].astype(bool)
            if "available" in source.files
            else np.ones(len(metadata["train_cells"]), dtype=bool)
        )
    training = training[available[training]]
    validation = validation[available[validation]]
    training = balanced_limit(
        training,
        config["runtime"].get(
            "context_train_limit", config["runtime"].get("train_limit")
        ),
        [metadata["train_cells"]],
        int(config["seed"]) + 3,
    )
    validation = balanced_limit(
        validation,
        config["runtime"].get(
            "context_validation_limit", config["runtime"].get("validation_limit")
        ),
        [metadata["train_cells"]],
        int(config["seed"]) + 4,
    )
    repository = ContextRepository(config, training)
    model, autoencoder, shape, checkpoint = load_context_checkpoint(
        config, args.checkpoint, repository, device
    )
    amp = bool(config["runtime"].get("amp", True))
    predictions = predict_indices(model, repository, validation, device, amp)
    reports = []
    for prior_alpha in args.spectral_prior_alphas:
        metrics_grid = evaluate_context_thresholds(
            model,
            autoencoder,
            repository,
            validation,
            shape,
            device,
            amp,
            args.thresholds,
            int(config["context"].get("validation_decode_batch_size", 8)),
            soft_outage_strength=args.soft_strengths,
            spectral_prior_alpha=prior_alpha,
            prediction_outputs=predictions,
        )
        for metrics in metrics_grid:
            report = {
                "threshold": float(metrics["outage_threshold"]),
                "soft_strength": float(metrics["soft_outage_strength"]),
                "spectral_prior_alpha": float(prior_alpha),
                **metrics,
            }
            reports.append(report)
            print(
                f"threshold={report['threshold']:.6f} "
                f"soft={report['soft_strength']:.2f} "
                f"prior={prior_alpha:.2f} score={report['score']:.6f}",
                flush=True,
            )
    # Prefer the conservative (higher) threshold when validation scores tie.
    best = max(reports, key=lambda report: (report["score"], report["threshold"]))
    cells = metadata["train_cells"][validation]
    cell_reports: dict[int, list[dict[str, float]]] = {}
    for cell_id in sorted(np.unique(cells).tolist()):
        selected = np.flatnonzero(cells == int(cell_id))
        cell_validation = validation[selected]
        cell_predictions = _slice_predictions(predictions, selected)
        candidates: list[dict[str, float]] = []
        for prior_alpha in args.spectral_prior_alphas:
            metrics_grid = evaluate_context_thresholds(
                model,
                autoencoder,
                repository,
                cell_validation,
                shape,
                device,
                amp,
                args.thresholds,
                int(config["context"].get("validation_decode_batch_size", 8)),
                soft_outage_strength=args.soft_strengths,
                spectral_prior_alpha=prior_alpha,
                prediction_outputs=cell_predictions,
            )
            for metrics in metrics_grid:
                candidates.append(
                    {
                        "cell_id": int(cell_id),
                        "threshold": float(metrics["outage_threshold"]),
                        "soft_strength": float(metrics["soft_outage_strength"]),
                        "spectral_prior_alpha": float(prior_alpha),
                        **metrics,
                    }
                )
        cell_reports[int(cell_id)] = candidates
    if len(cell_reports) > 3:
        raise RuntimeError("Per-cell policy scan is bounded to at most three cells")
    best_cellwise = max(
        (
            _combine_cell_reports(tuple(items))
            for items in product(*(cell_reports[cell] for cell in sorted(cell_reports)))
        ),
        key=lambda report: (
            float(report["score"]),
            sum(float(policy["threshold"]) for policy in report["by_cell"].values()),
        ),
    )
    ordered_policies = [
        best_cellwise["by_cell"][str(cell)] for cell in sorted(cell_reports)
    ]
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "best_threshold": float(best["threshold"]),
        "best_soft_strength": float(best["soft_strength"]),
        "best_spectral_prior_alpha": float(best["spectral_prior_alpha"]),
        "best_score": float(best["score"]),
        "best_threshold_by_cell": [
            float(policy["threshold"]) for policy in ordered_policies
        ],
        "best_soft_strength_by_cell": [
            float(policy["soft_strength"]) for policy in ordered_policies
        ],
        "best_spectral_prior_alpha_by_cell": [
            float(policy["spectral_prior_alpha"]) for policy in ordered_policies
        ],
        "best_cellwise_combined": best_cellwise,
        "cellwise_gain_vs_global": float(best_cellwise["score"] - best["score"]),
        "reports": reports,
        "cell_reports": {str(cell): values for cell, values in cell_reports.items()},
    }
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
