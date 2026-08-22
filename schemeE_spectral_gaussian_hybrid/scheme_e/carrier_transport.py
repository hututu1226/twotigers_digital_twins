from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree


TRANSPORT_CONTEXT_DIM = 14


@dataclass(frozen=True)
class CarrierFit:
    wave_numbers: np.ndarray
    qualities: np.ndarray
    pair_counts: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "wave_numbers": self.wave_numbers.astype(float).tolist(),
            "qualities": self.qualities.astype(float).tolist(),
            "pair_counts": self.pair_counts.astype(int).tolist(),
        }


def quality_gated_carrier_fit(
    fit: CarrierFit,
    prior_wave_number: float,
    minimum_quality: float = 0.5,
) -> CarrierFit:
    """Fall back to the carrier prior when a per-cell fit is not coherent enough."""
    wave_numbers = np.asarray(fit.wave_numbers, dtype=np.float64)
    qualities = np.asarray(fit.qualities, dtype=np.float64)
    pair_counts = np.asarray(fit.pair_counts, dtype=np.int64)
    if not (wave_numbers.shape == qualities.shape == pair_counts.shape):
        raise ValueError("Carrier fit arrays must have identical shapes")
    if not 0.0 <= float(minimum_quality) <= 1.0:
        raise ValueError("minimum_quality must be between zero and one")
    reliable = np.isfinite(qualities) & (qualities >= float(minimum_quality))
    selected = np.where(reliable, wave_numbers, float(prior_wave_number))
    return CarrierFit(selected, qualities.copy(), pair_counts.copy())


def _alignment_quality(
    wave_numbers: np.ndarray,
    range_deltas: np.ndarray,
    correlations: np.ndarray,
) -> np.ndarray:
    denominator = max(float(np.sum(np.abs(correlations), dtype=np.float64)), 1e-30)
    output = np.empty(len(wave_numbers), dtype=np.float64)
    for start in range(0, len(wave_numbers), 256):
        stop = min(start + 256, len(wave_numbers))
        phase = np.exp(-1j * np.outer(wave_numbers[start:stop], range_deltas))
        output[start:stop] = np.real(phase @ correlations) / denominator
    return output


def _fit_one_wave_number(
    range_deltas: np.ndarray,
    correlations: np.ndarray,
    center: float,
    radius: float,
) -> tuple[float, float]:
    coarse = np.arange(center - radius, center + radius + 0.025, 0.05)
    coarse_quality = _alignment_quality(coarse, range_deltas, correlations)
    coarse_best = float(coarse[int(np.argmax(coarse_quality))])
    fine = np.arange(coarse_best - 0.06, coarse_best + 0.0601, 0.0025)
    fine_quality = _alignment_quality(fine, range_deltas, correlations)
    best = int(np.argmax(fine_quality))
    return float(fine[best]), float(fine_quality[best])


def fit_carrier_transport(
    positions: np.ndarray,
    cells: np.ndarray,
    outage: np.ndarray,
    channels: np.ndarray,
    observed_indices: np.ndarray,
    bs_positions: np.ndarray,
    *,
    seed: int,
    maximum_targets_per_cell: int = 256,
    neighbors: int = 4,
    prior_wave_number: float = -140.33,
    search_radius: float = 12.0,
) -> CarrierFit:
    """Fit one global carrier slope per cell using observed train/train pairs only."""
    positions = np.asarray(positions, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    outage = np.asarray(outage, dtype=np.bool_)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    bs_positions = np.asarray(bs_positions, dtype=np.float64)
    cell_count = int(max(np.max(cells), len(bs_positions) - 1)) + 1
    wave_numbers = np.full(cell_count, float(prior_wave_number), dtype=np.float64)
    qualities = np.zeros(cell_count, dtype=np.float64)
    pair_counts = np.zeros(cell_count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    observed_mask = np.zeros(len(positions), dtype=np.bool_)
    observed_mask[observed_indices] = True
    for cell in range(cell_count):
        sources = np.flatnonzero(observed_mask & ~outage & (cells == cell))
        if len(sources) <= neighbors:
            continue
        target_count = min(int(maximum_targets_per_cell), len(sources))
        targets = np.sort(rng.choice(sources, size=target_count, replace=False))
        tree = cKDTree(positions[sources, :2])
        _, local_neighbors = tree.query(
            positions[targets, :2], k=min(neighbors + 1, len(sources))
        )
        local_neighbors = np.atleast_2d(local_neighbors)
        if len(targets) == 1 and local_neighbors.shape[0] != 1:
            local_neighbors = local_neighbors.T
        table = sources[local_neighbors]
        ranges = np.linalg.norm(positions - bs_positions[cell], axis=1)
        deltas: list[float] = []
        correlations: list[complex] = []
        for target_index, candidates in zip(targets, table, strict=True):
            target = np.asarray(channels[target_index], dtype=np.complex64).reshape(-1)
            candidates = candidates[candidates != target_index][:neighbors]
            for source_index in candidates:
                source = np.asarray(channels[source_index], dtype=np.complex64).reshape(-1)
                deltas.append(float(ranges[target_index] - ranges[source_index]))
                correlations.append(complex(np.vdot(source, target)))
        if not correlations:
            continue
        wave, quality = _fit_one_wave_number(
            np.asarray(deltas, dtype=np.float64),
            np.asarray(correlations, dtype=np.complex128),
            float(prior_wave_number),
            float(search_radius),
        )
        wave_numbers[cell] = wave
        qualities[cell] = quality
        pair_counts[cell] = len(correlations)
    return CarrierFit(wave_numbers, qualities, pair_counts)


def select_transport_candidates(
    candidates: np.ndarray,
    distances: np.ndarray,
    count: int,
    minimum_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the first K support points outside an optional sampled hole radius."""
    candidates = np.asarray(candidates, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    count = max(1, int(count))
    selected = np.empty((len(candidates), count), dtype=np.int64)
    selected_distances = np.empty((len(candidates), count), dtype=np.float32)
    if minimum_distances is None:
        minimum_distances = np.zeros(len(candidates), dtype=np.float32)
    for row in range(len(candidates)):
        valid = np.flatnonzero(
            np.isfinite(distances[row]) & (distances[row] >= minimum_distances[row] - 1e-5)
        )
        if not len(valid):
            valid = np.flatnonzero(np.isfinite(distances[row]))
        chosen = valid[:count]
        if len(chosen) < count:
            chosen = np.pad(chosen, (0, count - len(chosen)), mode="edge")
        selected[row] = candidates[row, chosen]
        selected_distances[row] = distances[row, chosen]
    return selected, selected_distances


def build_transport_seed(
    reference_channels: torch.Tensor,
    target_positions: torch.Tensor,
    source_positions: torch.Tensor,
    cell_ids: torch.Tensor,
    distances: torch.Tensor,
    bs_positions: torch.Tensor,
    wave_numbers: torch.Tensor,
    fit_qualities: torch.Tensor,
    *,
    distance_power: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carrier-align K observed channels and return an IDW seed plus confidence features."""
    if reference_channels.ndim != 5:
        raise ValueError("reference_channels must be [B,K,M,N,S]")
    batch, count = reference_channels.shape[:2]
    if source_positions.shape[:2] != (batch, count):
        raise ValueError("source_positions must match [B,K]")
    cell_ids = cell_ids.long()
    target_bs = bs_positions[cell_ids]
    target_range = torch.linalg.vector_norm(target_positions.float() - target_bs, dim=1)
    source_range = torch.linalg.vector_norm(
        source_positions.float() - target_bs[:, None, :], dim=2
    )
    slopes = wave_numbers[cell_ids]
    phase = slopes[:, None] * (target_range[:, None] - source_range)
    aligned = reference_channels * torch.polar(torch.ones_like(phase), phase)[
        :, :, None, None, None
    ]

    safe_distances = distances.float().clamp_min(1e-3)
    weights = safe_distances.pow(-float(distance_power))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    seed = torch.sum(aligned * weights[:, :, None, None, None], dim=1)

    flat = aligned.flatten(2)
    anchor = flat[:, :1]
    inner = torch.sum(anchor.conj() * flat, dim=2)
    norm = torch.linalg.vector_norm(anchor, dim=2) * torch.linalg.vector_norm(flat, dim=2)
    coherence = inner / norm.clamp_min(1e-12)
    energy = reference_channels.abs().square().mean(dim=(2, 3, 4)).clamp_min(1e-30)
    weighted_energy = torch.sum(weights * energy, dim=1).clamp_min(1e-30)
    seed_energy = seed.abs().square().mean(dim=(1, 2, 3)).clamp_min(1e-30)
    entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-12)), dim=1)
    entropy = entropy / max(float(np.log(max(count, 2))), 1e-6)
    effective = 1.0 / torch.sum(weights.square(), dim=1).clamp_min(1e-12)
    log_energy = torch.log10(energy)
    context = torch.stack(
        [
            safe_distances[:, 0] / 20.0,
            safe_distances.mean(dim=1) / 20.0,
            safe_distances.std(dim=1, unbiased=False) / 20.0,
            safe_distances[:, -1] / 20.0,
            weights.max(dim=1).values,
            entropy,
            effective / float(count),
            fit_qualities[cell_ids],
            coherence.real.mean(dim=1),
            coherence.abs().mean(dim=1),
            coherence.real.amin(dim=1),
            log_energy.mean(dim=1) / 12.0,
            log_energy.std(dim=1, unbiased=False) / 4.0,
            torch.log10(seed_energy / weighted_energy).clamp(-4.0, 1.0) / 4.0,
        ],
        dim=1,
    )
    if context.shape[1] != TRANSPORT_CONTEXT_DIM:
        raise AssertionError("transport context dimension mismatch")
    return seed, context
