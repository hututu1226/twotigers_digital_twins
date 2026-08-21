"""Compare Round 2 test geometry with Scheme F's Fold0 validation geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"p{percentile}": float(np.percentile(values, percentile))
        for percentile in (0, 10, 25, 50, 75, 90, 95, 99, 100)
    }


def connected_components(points: np.ndarray, link_meters: float) -> list[np.ndarray]:
    neighbors = cKDTree(points[:, :2]).query_ball_point(points[:, :2], link_meters)
    unseen = set(range(len(points)))
    components: list[np.ndarray] = []
    while unseen:
        pending = [unseen.pop()]
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            linked = [index for index in neighbors[current] if index in unseen]
            unseen.difference_update(linked)
            pending.extend(linked)
        components.append(np.asarray(component, dtype=np.int64))
    return components


def support_report(
    targets: np.ndarray, references: np.ndarray, radii: tuple[float, ...]
) -> dict[str, object]:
    tree = cKDTree(references[:, :2])
    distances, _ = tree.query(targets[:, :2], k=min(16, len(references)))
    if distances.ndim == 1:
        distances = distances[:, None]
    return {
        "nearest_distance_meters": percentiles(distances[:, 0]),
        "fourth_distance_meters": percentiles(
            distances[:, min(3, distances.shape[1] - 1)]
        ),
        "eighth_distance_meters": percentiles(
            distances[:, min(7, distances.shape[1] - 1)]
        ),
        "support_counts": {
            str(radius): percentiles(
                np.asarray(
                    [
                        len(indices)
                        for indices in tree.query_ball_point(targets[:, :2], radius)
                    ],
                    dtype=np.float64,
                )
            )
            for radius in radii
        },
    }


def cell_report(
    cell_id: int,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_cells: np.ndarray,
    test_cells: np.ndarray,
    validation_mask: np.ndarray,
    outage: np.ndarray,
    log_power: np.ndarray,
    link_meters: float,
    spectral_folds: np.ndarray | None,
) -> dict[str, object]:
    cell_train = np.flatnonzero(train_cells == cell_id)
    cell_test = np.flatnonzero(test_cells == cell_id)
    cell_validation = np.flatnonzero((train_cells == cell_id) & validation_mask)
    cell_observed = np.flatnonzero((train_cells == cell_id) & ~validation_mask)
    components = connected_components(test_positions[cell_test], link_meters)
    component_reports = []
    for component in components:
        points = test_positions[cell_test[component]]
        span = np.ptp(points[:, :2], axis=0)
        component_reports.append(
            {
                "size": int(len(component)),
                "span_x_meters": float(span[0]),
                "span_y_meters": float(span[1]),
                "diameter_proxy_meters": float(np.linalg.norm(span)),
            }
        )
    component_reports.sort(key=lambda item: int(item["size"]), reverse=True)
    nonoutage = cell_train[~outage[cell_train]]
    report = {
        "train_samples": int(len(cell_train)),
        "test_samples": int(len(cell_test)),
        "validation_samples": int(len(cell_validation)),
        "train_xyz_min": train_positions[cell_train].min(axis=0).tolist(),
        "train_xyz_max": train_positions[cell_train].max(axis=0).tolist(),
        "test_xyz_min": test_positions[cell_test].min(axis=0).tolist(),
        "test_xyz_max": test_positions[cell_test].max(axis=0).tolist(),
        "train_outages": int(outage[cell_train].sum()),
        "validation_outages": int(outage[cell_validation].sum()),
        "nonoutage_log_power": percentiles(log_power[nonoutage]),
        "test_support_from_all_train": support_report(
            test_positions[cell_test],
            train_positions[cell_train],
            (2.0, 4.0, 6.0, 10.0),
        ),
        "fold0_support_from_observed": support_report(
            train_positions[cell_validation],
            train_positions[cell_observed],
            (2.0, 4.0, 6.0, 10.0),
        ),
        "test_components_link_meters": float(link_meters),
        "test_component_count": int(len(components)),
        "test_component_sizes": [int(len(component)) for component in components],
        "largest_test_components": component_reports[:10],
    }
    if spectral_folds is not None:
        leaked_distances = []
        clean_distances = []
        for target in cell_validation:
            eligible = cell_train[spectral_folds[cell_train] != spectral_folds[target]]
            hidden = eligible[validation_mask[eligible]]
            observed = eligible[~validation_mask[eligible]]
            target_xy = train_positions[target, :2]
            leaked_distances.append(
                float(
                    np.linalg.norm(
                        train_positions[hidden, :2] - target_xy, axis=1
                    ).min()
                )
                if len(hidden)
                else float("inf")
            )
            clean_distances.append(
                float(
                    np.linalg.norm(
                        train_positions[observed, :2] - target_xy, axis=1
                    ).min()
                )
            )
        leaked = np.asarray(leaked_distances)
        clean = np.asarray(clean_distances)
        report["spectral_oof_fold0_leakage"] = {
            "definition": (
                "A hidden Fold0 point can enter another hidden point's OOF teacher "
                "training set when their spectral folds differ."
            ),
            "hidden_candidate_nearest_distance_meters": percentiles(leaked),
            "observed_candidate_nearest_distance_meters": percentiles(clean),
            "hidden_candidate_is_closer_fraction": float(np.mean(leaked < clean)),
            "hidden_candidate_within_6m_fraction": float(np.mean(leaked <= 6.0)),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--link-meters", type=float, default=6.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.metadata) as source:
        metadata = {key: source[key] for key in source.files}
    validation_mask = metadata["validation_masks"][args.fold].astype(bool)
    cell_ids = sorted(np.unique(metadata["train_cells"]).astype(int).tolist())
    report = {
        "metadata": str(args.metadata.resolve()),
        "validation_fold": int(args.fold),
        "validation_samples": int(validation_mask.sum()),
        "cells": {
            str(cell_id): cell_report(
                cell_id,
                metadata["train_positions"],
                metadata["test_positions"],
                metadata["train_cells"],
                metadata["test_cells"],
                validation_mask,
                metadata["outage"].astype(bool),
                metadata["log_power"],
                args.link_meters,
                metadata.get("spectral_folds"),
            )
            for cell_id in cell_ids
        },
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
