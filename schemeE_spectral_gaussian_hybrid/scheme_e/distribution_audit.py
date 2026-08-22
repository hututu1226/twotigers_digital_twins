from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import official_score


def same_cell_support_features(
    support_positions: np.ndarray,
    support_cells: np.ndarray,
    query_positions: np.ndarray,
    query_cells: np.ndarray,
    *,
    neighbor_count: int = 16,
    radii: Sequence[float] = (6.0, 12.0, 18.0),
) -> tuple[np.ndarray, list[str]]:
    support_positions = np.asarray(support_positions, dtype=np.float64)
    support_cells = np.asarray(support_cells, dtype=np.int64)
    query_positions = np.asarray(query_positions, dtype=np.float64)
    query_cells = np.asarray(query_cells, dtype=np.int64)
    if len(support_positions) != len(support_cells):
        raise ValueError("support positions and cells have different lengths")
    if len(query_positions) != len(query_cells):
        raise ValueError("query positions and cells have different lengths")
    if int(neighbor_count) < 1:
        raise ValueError("neighbor_count must be positive")

    names = ["nearest_m", "mean_4nn_m", "mean_8nn_m", "mean_16nn_m"]
    names.extend(f"density_{float(radius):g}m" for radius in radii)
    output = np.full((len(query_positions), len(names)), np.nan, dtype=np.float64)
    for cell in np.unique(query_cells):
        query_rows = np.flatnonzero(query_cells == cell)
        support_rows = np.flatnonzero(support_cells == cell)
        if not len(support_rows):
            raise ValueError(f"cell {int(cell)} has no visible support")
        tree = cKDTree(support_positions[support_rows, :2])
        count = min(max(int(neighbor_count), 16), len(support_rows))
        distances, _ = tree.query(query_positions[query_rows, :2], k=count)
        distances = np.asarray(distances, dtype=np.float64).reshape(len(query_rows), count)
        output[query_rows, 0] = distances[:, 0]
        for column, requested in enumerate((4, 8, 16), start=1):
            actual = min(requested, count)
            output[query_rows, column] = distances[:, :actual].mean(axis=1)
        for offset, radius in enumerate(radii, start=4):
            output[query_rows, offset] = np.fromiter(
                (
                    len(values)
                    for values in tree.query_ball_point(
                        query_positions[query_rows, :2], float(radius)
                    )
                ),
                dtype=np.float64,
                count=len(query_rows),
            )
    if not np.isfinite(output).all():
        raise RuntimeError("support feature construction left non-finite rows")
    return output, names


def summarize_features(values: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError("feature array and names do not match")
    output: dict[str, dict[str, float]] = {}
    for column, name in enumerate(names):
        selected = array[:, column]
        output[str(name)] = {
            "minimum": float(np.min(selected)),
            "p10": float(np.quantile(selected, 0.10)),
            "p25": float(np.quantile(selected, 0.25)),
            "median": float(np.median(selected)),
            "p75": float(np.quantile(selected, 0.75)),
            "p90": float(np.quantile(selected, 0.90)),
            "maximum": float(np.max(selected)),
            "mean": float(np.mean(selected)),
        }
    return output


def feature_shift_report(
    reference: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
) -> dict[str, object]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("shift feature matrices do not match")
    if left.shape[1] != len(names):
        raise ValueError("shift feature names do not match")
    rows = []
    for column, name in enumerate(names):
        pooled_std = max(float(np.std(np.concatenate([left[:, column], right[:, column]]))), 1e-12)
        rows.append(
            {
                "name": str(name),
                "ks": float(ks_2samp(left[:, column], right[:, column]).statistic),
                "standardized_mean_gap": float(
                    (right[:, column].mean() - left[:, column].mean()) / pooled_std
                ),
                "wasserstein": float(
                    wasserstein_distance(left[:, column], right[:, column])
                ),
            }
        )
    rows.sort(key=lambda item: float(item["ks"]), reverse=True)
    return {
        "maximum_ks": float(rows[0]["ks"]) if rows else 0.0,
        "mean_ks": float(np.mean([row["ks"] for row in rows])) if rows else 0.0,
        "top_features": rows[:12],
    }


def domain_classifier_auc(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    seed: int = 2026,
    folds: int = 5,
) -> dict[str, float | int]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("domain feature matrices do not match")
    features = np.concatenate([left, right], axis=0)
    labels = np.concatenate(
        [np.zeros(len(left), dtype=np.int64), np.ones(len(right), dtype=np.int64)]
    )
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    models = {
        "linear": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                max_iter=2000,
                class_weight="balanced",
                random_state=int(seed),
            ),
        ),
        "nonlinear": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=120,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=int(seed),
            ),
        ),
    }
    result: dict[str, float | int] = {
        "reference_samples": int(len(left)),
        "target_samples": int(len(right)),
        "features": int(features.shape[1]),
    }
    for name, model in models.items():
        probability = cross_val_predict(
            model,
            features,
            labels,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        result[f"{name}_oof_auc"] = float(roc_auc_score(labels, probability))
    return result


def fixed_bin_importance_weights(
    reference_cells: np.ndarray,
    reference_distance: np.ndarray,
    target_cells: np.ndarray,
    target_distance: np.ndarray,
    *,
    edges: Sequence[float] = (0.0, 4.0, 6.0, 8.0, 10.0, 12.0, np.inf),
) -> tuple[np.ndarray, dict[str, object]]:
    reference_cells = np.asarray(reference_cells, dtype=np.int64)
    target_cells = np.asarray(target_cells, dtype=np.int64)
    reference_distance = np.asarray(reference_distance, dtype=np.float64)
    target_distance = np.asarray(target_distance, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.float64)
    if len(edge_array) < 2 or np.any(np.diff(edge_array) <= 0):
        raise ValueError("distance bin edges must increase")
    reference_bin = np.digitize(reference_distance, edge_array[1:-1], right=False)
    target_bin = np.digitize(target_distance, edge_array[1:-1], right=False)
    cells = np.unique(np.concatenate([reference_cells, target_cells]))
    weights = np.zeros(len(reference_cells), dtype=np.float64)
    rows = []
    unmatched_target = 0
    for cell in cells:
        for index in range(len(edge_array) - 1):
            reference_mask = (reference_cells == cell) & (reference_bin == index)
            target_mask = (target_cells == cell) & (target_bin == index)
            reference_count = int(reference_mask.sum())
            target_count = int(target_mask.sum())
            if reference_count:
                weights[reference_mask] = target_count / reference_count
            elif target_count:
                unmatched_target += target_count
            rows.append(
                {
                    "cell": int(cell),
                    "minimum_m": float(edge_array[index]),
                    "maximum_m": None if np.isinf(edge_array[index + 1]) else float(edge_array[index + 1]),
                    "reference_count": reference_count,
                    "target_count": target_count,
                }
            )
    if weights.sum() <= 0:
        raise ValueError("importance weights have zero mass")
    weights *= len(reference_cells) / weights.sum()
    return weights, {
        "edges_m": [None if np.isinf(value) else float(value) for value in edge_array],
        "bins": rows,
        "unmatched_target_samples": int(unmatched_target),
        "effective_reference_samples": float(
            weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-30)
        ),
        "maximum_weight": float(weights.max()),
    }


def weighted_aggregate_sample_metrics(
    values: Mapping[str, np.ndarray], weights: np.ndarray
) -> dict[str, float | int]:
    sample_weights = np.asarray(weights, dtype=np.float64)
    if len(sample_weights) != len(np.asarray(values["target_energy"])):
        raise ValueError("metric arrays and weights have different lengths")
    pas_count = float(np.sum(sample_weights * np.asarray(values["pas_count"], dtype=np.float64)))
    pdp_count = float(np.sum(sample_weights * np.asarray(values["pdp_count"], dtype=np.float64)))
    pas = float(
        np.sum(sample_weights * np.asarray(values["pas_sum"], dtype=np.float64))
        / max(pas_count, 1e-30)
    )
    pdp = float(
        np.sum(sample_weights * np.asarray(values["pdp_sum"], dtype=np.float64))
        / max(pdp_count, 1e-30)
    )
    error = float(
        np.sum(sample_weights * np.asarray(values["error_energy"], dtype=np.float64))
    )
    target = float(
        np.sum(sample_weights * np.asarray(values["target_energy"], dtype=np.float64))
    )
    nmse = error / max(target, 1e-30)
    return {
        "pas": pas,
        "pdp": pdp,
        "nmse": nmse,
        "score": official_score(pas, pdp, nmse),
        "samples": int(len(sample_weights)),
        "effective_samples": float(
            sample_weights.sum() ** 2
            / max(float(np.square(sample_weights).sum()), 1e-30)
        ),
        "weight_sum": float(sample_weights.sum()),
    }


def periodic_cell_holdout_mask(
    positions: np.ndarray,
    cell_ids: np.ndarray,
    *,
    cell_id: int,
    tile_meters: float,
    hole_meters: float,
    phase_x: float,
    phase_y: float,
) -> np.ndarray:
    """Select one cell's periodic square holes at an explicit phase."""

    xy = np.asarray(positions, dtype=np.float64)[:, :2]
    cells = np.asarray(cell_ids, dtype=np.int64)
    if len(xy) != len(cells):
        raise ValueError("positions and cells have different lengths")
    if not 0.0 < float(hole_meters) < float(tile_meters):
        raise ValueError("hole_meters must be positive and smaller than tile_meters")
    local_x = np.mod(xy[:, 0] - float(phase_x), float(tile_meters))
    local_y = np.mod(xy[:, 1] - float(phase_y), float(tile_meters))
    return (
        (cells == int(cell_id))
        & (local_x < float(hole_meters))
        & (local_y < float(hole_meters))
    )


def distribution_signature_distance(
    reference: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compare robust per-feature moments without fitting a domain classifier."""

    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("distribution feature matrices do not match")
    if not len(left) or not len(right):
        raise ValueError("distribution feature matrices must be nonempty")
    pooled = np.concatenate([left, right], axis=0)
    scale = np.maximum(np.std(pooled, axis=0), 1e-6)
    quantiles = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90])
    left_summary = np.concatenate(
        [
            left.mean(axis=0, keepdims=True),
            left.std(axis=0, keepdims=True),
            np.quantile(left, quantiles, axis=0),
        ],
        axis=0,
    )
    right_summary = np.concatenate(
        [
            right.mean(axis=0, keepdims=True),
            right.std(axis=0, keepdims=True),
            np.quantile(right, quantiles, axis=0),
        ],
        axis=0,
    )
    standardized = np.abs(left_summary - right_summary) / scale[None, :]
    return float(np.mean(np.clip(standardized, 0.0, 10.0)))


def test_matched_holdout_gate(
    *,
    samples: int,
    cell_counts: Mapping[int | str, int],
    nearest_median_gap_m: float,
    support_domain_auc: float,
    link_environment_domain_auc: float,
    current_link_environment_auc: float,
    minimum_samples: int = 450,
    maximum_samples: int = 650,
    minimum_samples_per_cell: int = 200,
    maximum_nearest_median_gap_m: float = 2.0,
    maximum_support_domain_auc: float = 0.70,
    minimum_link_auc_reduction: float = 0.10,
) -> tuple[bool, list[str]]:
    """Apply the preregistered promotion gate for a test-matched holdout."""

    reasons: list[str] = []
    if not int(minimum_samples) <= int(samples) <= int(maximum_samples):
        reasons.append("sample_count")
    if not cell_counts or min(int(value) for value in cell_counts.values()) < int(
        minimum_samples_per_cell
    ):
        reasons.append("cell_balance")
    if float(nearest_median_gap_m) > float(maximum_nearest_median_gap_m):
        reasons.append("nearest_support")
    if float(support_domain_auc) > float(maximum_support_domain_auc):
        reasons.append("support_domain_auc")
    required_link_auc = float(current_link_environment_auc) - float(
        minimum_link_auc_reduction
    )
    if float(link_environment_domain_auc) > required_link_auc:
        reasons.append("link_environment_domain_auc")
    return not reasons, reasons


def one_to_one_same_cell_assignment(
    reference_positions: np.ndarray,
    reference_cells: np.ndarray,
    query_positions: np.ndarray,
    query_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each query to a unique same-cell reference by minimum XY distance."""

    reference_xy = np.asarray(reference_positions, dtype=np.float64)[:, :2]
    reference_cell = np.asarray(reference_cells, dtype=np.int64)
    query_xy = np.asarray(query_positions, dtype=np.float64)[:, :2]
    query_cell = np.asarray(query_cells, dtype=np.int64)
    if len(reference_xy) != len(reference_cell):
        raise ValueError("reference positions and cells have different lengths")
    if len(query_xy) != len(query_cell):
        raise ValueError("query positions and cells have different lengths")
    assignment = np.full(len(query_xy), -1, dtype=np.int64)
    distances = np.full(len(query_xy), np.nan, dtype=np.float64)
    for cell in np.unique(query_cell):
        query_rows = np.flatnonzero(query_cell == cell)
        reference_rows = np.flatnonzero(reference_cell == cell)
        if len(reference_rows) < len(query_rows):
            raise ValueError(
                f"cell {int(cell)} has fewer references than queries: "
                f"{len(reference_rows)} < {len(query_rows)}"
            )
        delta = query_xy[query_rows, None, :] - reference_xy[reference_rows][None, :, :]
        cost = np.square(delta).sum(axis=2)
        local_query, local_reference = linear_sum_assignment(cost)
        selected_query = query_rows[local_query]
        selected_reference = reference_rows[local_reference]
        assignment[selected_query] = selected_reference
        distances[selected_query] = np.sqrt(cost[local_query, local_reference])
    if np.any(assignment < 0) or not np.isfinite(distances).all():
        raise RuntimeError("same-cell assignment left unmatched queries")
    if len(np.unique(assignment)) != len(assignment):
        raise RuntimeError("same-cell assignment reused a reference")
    return assignment, distances


def legacy_spatial_block_split(
    positions: np.ndarray,
    cell_labels: np.ndarray,
    *,
    validation_fraction: float = 0.15,
    blocks_per_cell: int = 12,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the original Scheme1 split without loading its cached metadata."""

    positions = np.asarray(positions, dtype=np.float64)
    cell_labels = np.asarray(cell_labels, dtype=np.int64)

    def kmeans(points: np.ndarray, clusters: int, local_seed: int) -> np.ndarray:
        rng = np.random.default_rng(local_seed)
        centers = [points[rng.integers(len(points))]]
        for _ in range(1, clusters):
            distance = np.min(
                np.stack(
                    [np.sum((points - center) ** 2, axis=1) for center in centers]
                ),
                axis=0,
            )
            centers.append(points[int(np.argmax(distance))])
        centers_array = np.asarray(centers, dtype=np.float64)
        labels = np.zeros(len(points), dtype=np.int64)
        for _ in range(50):
            distance = np.sum(
                (points[:, None, :] - centers_array[None, :, :]) ** 2, axis=-1
            )
            next_labels = distance.argmin(axis=1)
            if np.array_equal(labels, next_labels):
                break
            labels = next_labels
            for cluster in range(clusters):
                members = points[labels == cluster]
                if len(members):
                    centers_array[cluster] = members.mean(axis=0)
        return labels

    rng = np.random.default_rng(int(seed))
    validation_parts = []
    for cell in np.unique(cell_labels):
        cell_indices = np.flatnonzero(cell_labels == cell)
        clusters = kmeans(
            positions[cell_indices, :2],
            min(int(blocks_per_cell), len(cell_indices)),
            int(seed) + int(cell),
        )
        target = max(1, int(round(float(validation_fraction) * len(cell_indices))))
        selected: list[int] = []
        for cluster in rng.permutation(int(clusters.max()) + 1):
            selected.extend(cell_indices[clusters == cluster].tolist())
            if len(selected) >= target:
                break
        validation_parts.append(np.asarray(selected, dtype=np.int64))
    validation = np.unique(np.concatenate(validation_parts))
    training = np.setdiff1d(
        np.arange(len(positions), dtype=np.int64), validation, assume_unique=True
    )
    return training.astype(np.int64), validation.astype(np.int64)
