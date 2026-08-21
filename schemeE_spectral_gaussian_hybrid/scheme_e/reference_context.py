from __future__ import annotations

import numpy as np

from .spectral_targets import PAS_LOG_SCALE, PDP_LOG_SCALE


REFERENCE_CONTEXT_DIM = 85


def _power_domain_cosine(
    first_log: np.ndarray,
    second_log: np.ndarray,
    scale: float,
) -> np.ndarray:
    first = np.expm1(np.clip(np.asarray(first_log, dtype=np.float32), 0.0, 20.0))
    second = np.expm1(np.clip(np.asarray(second_log, dtype=np.float32), 0.0, 20.0))
    first /= float(scale)
    second /= float(scale)
    numerator = np.sum(first * second, axis=-1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(first * first, axis=-1, dtype=np.float64)
        * np.sum(second * second, axis=-1, dtype=np.float64)
    )
    return (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)


def build_reference_context(
    target_positions: np.ndarray,
    reference_positions: np.ndarray,
    target_geometry_normalized: np.ndarray,
    reference_geometry_normalized: np.ndarray,
    target_pas_log: np.ndarray,
    target_pdp_log: np.ndarray,
    reference_pas_log: np.ndarray,
    reference_pdp_log: np.ndarray,
    target_log_power: np.ndarray,
    reference_log_power: np.ndarray,
    uncertainty: np.ndarray,
) -> np.ndarray:
    target_positions = np.asarray(target_positions, dtype=np.float32)
    reference_positions = np.asarray(reference_positions, dtype=np.float32)
    target_geometry = np.asarray(target_geometry_normalized, dtype=np.float32)
    reference_geometry = np.asarray(reference_geometry_normalized, dtype=np.float32)
    delta = target_positions - reference_positions
    distance_xy = np.linalg.norm(delta[:, :2], axis=1)
    distance_3d = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(distance_3d[:, None], 1e-6)
    geometry_delta = np.clip(target_geometry - reference_geometry, -8.0, 8.0)
    geometry_rms = np.sqrt(np.mean(geometry_delta * geometry_delta, axis=1))
    pas_cosine = _power_domain_cosine(
        target_pas_log, reference_pas_log, PAS_LOG_SCALE
    )
    pdp_cosine = _power_domain_cosine(
        target_pdp_log, reference_pdp_log, PDP_LOG_SCALE
    )
    scalars = np.column_stack(
        [
            delta / 20.0,
            distance_xy / 20.0,
            distance_3d / 20.0,
            np.log1p(distance_3d) / 4.0,
            direction,
            geometry_rms,
            np.clip(
                np.asarray(target_log_power, dtype=np.float32)
                - np.asarray(reference_log_power, dtype=np.float32),
                -6.0,
                6.0,
            )
            / 6.0,
            pas_cosine,
            pdp_cosine,
            np.asarray(uncertainty, dtype=np.float32),
        ]
    ).astype(np.float32)
    output = np.concatenate([geometry_delta, scalars], axis=1)
    if output.shape[1] != REFERENCE_CONTEXT_DIM:
        raise RuntimeError(
            f"Reference context width changed: {output.shape[1]} != {REFERENCE_CONTEXT_DIM}"
        )
    return output


def select_reference_candidates(
    candidate_indices: np.ndarray,
    distances: np.ndarray,
    target_geometry_normalized: np.ndarray,
    reference_geometry_normalized: np.ndarray,
    target_pas_log: np.ndarray,
    target_pdp_log: np.ndarray,
    reference_pas_log: np.ndarray,
    reference_pdp_log: np.ndarray,
    strategy: dict[str, float | int] | None,
) -> np.ndarray:
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    candidate_distances = np.asarray(distances, dtype=np.float32)
    if candidates.ndim != 2 or candidate_distances.shape != candidates.shape:
        raise ValueError("Reference candidates and distances must have matching 2D shapes")
    if strategy is None or str(strategy.get("name", "nearest")) == "nearest":
        return candidates[:, 0]
    top_k = min(int(strategy.get("top_k", candidates.shape[1])), candidates.shape[1])
    selected = np.empty(len(candidates), dtype=np.int64)
    distance_weight = float(strategy.get("distance_weight", 1.0))
    pas_weight = float(strategy.get("pas_weight", 0.0))
    pdp_weight = float(strategy.get("pdp_weight", 0.0))
    geometry_weight = float(strategy.get("geometry_weight", 0.0))
    for row in range(len(candidates)):
        local = candidates[row, :top_k]
        valid = (local >= 0) & np.isfinite(candidate_distances[row, :top_k])
        local = local[valid]
        if not len(local):
            raise RuntimeError("A target has no valid reference candidate")
        distance = candidate_distances[row, :top_k][valid]
        pas_cosine = _power_domain_cosine(
            np.repeat(target_pas_log[row : row + 1], len(local), axis=0),
            reference_pas_log[local],
            PAS_LOG_SCALE,
        )
        pdp_cosine = _power_domain_cosine(
            np.repeat(target_pdp_log[row : row + 1], len(local), axis=0),
            reference_pdp_log[local],
            PDP_LOG_SCALE,
        )
        geometry_delta = (
            reference_geometry_normalized[local]
            - target_geometry_normalized[row : row + 1]
        )
        geometry_distance = np.sqrt(np.mean(geometry_delta * geometry_delta, axis=1))
        score = (
            distance_weight * distance / 10.0
            + pas_weight * (1.0 - pas_cosine)
            + pdp_weight * (1.0 - pdp_cosine)
            + geometry_weight * geometry_distance
        )
        selected[row] = local[int(np.argmin(score))]
    return selected


def sample_test_matched_references(
    candidates: np.ndarray,
    distances: np.ndarray,
    target_cells: np.ndarray,
    distance_profiles: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    selected = np.empty(len(candidates), dtype=np.int64)
    for row, cell_value in enumerate(np.asarray(target_cells, dtype=np.int64)):
        valid = np.flatnonzero(np.isfinite(distances[row]) & (candidates[row] >= 0))
        if not len(valid):
            raise RuntimeError("A training target has no valid reference candidate")
        profile = np.asarray(distance_profiles[int(cell_value)], dtype=np.float32)
        desired = float(profile[int(rng.integers(0, len(profile)))])
        rank = valid[int(np.argmin(np.abs(distances[row, valid] - desired)))]
        selected[row] = candidates[row, rank]
    return selected
