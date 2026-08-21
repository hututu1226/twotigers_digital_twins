from __future__ import annotations

import numpy as np
import torch


def fit_power_calibration(
    predicted_log_power: np.ndarray,
    target_log_power: np.ndarray,
    cells: np.ndarray,
    valid_indices: np.ndarray,
    slope_bounds: tuple[float, float] = (0.6, 1.4),
) -> np.ndarray:
    """Fit a robust per-cell affine map around the OOF medians."""
    prediction = np.asarray(predicted_log_power, dtype=np.float32)
    target = np.asarray(target_log_power, dtype=np.float32)
    cell_ids = np.asarray(cells, dtype=np.int64)
    selected = np.asarray(valid_indices, dtype=np.int64)
    low_slope, high_slope = map(float, slope_bounds)
    if not 0.0 < low_slope <= high_slope:
        raise ValueError("Power calibration slope bounds must be positive and ordered")
    cell_count = int(cell_ids.max()) + 1
    parameters = np.empty((cell_count, 3), dtype=np.float32)
    for cell in range(cell_count):
        local = selected[cell_ids[selected] == cell]
        x = prediction[local].astype(np.float64)
        y = target[local].astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) < 3:
            raise RuntimeError(f"Cell {cell} has too few samples for power calibration")
        x_center = float(np.median(x))
        y_center = float(np.median(y))
        centered_x = x - x_center
        centered_y = y - y_center
        radius = np.maximum(
            np.abs(centered_x) / max(float(np.quantile(np.abs(centered_x), 0.9)), 1e-6),
            np.abs(centered_y) / max(float(np.quantile(np.abs(centered_y), 0.9)), 1e-6),
        )
        keep = radius <= 1.0
        denominator = float(np.dot(centered_x[keep], centered_x[keep]))
        slope = (
            float(np.dot(centered_x[keep], centered_y[keep]) / denominator)
            if denominator > 1e-12
            else 1.0
        )
        parameters[cell] = (
            x_center,
            y_center,
            float(np.clip(slope, low_slope, high_slope)),
        )
    return parameters


def apply_power_calibration(
    log_power: np.ndarray,
    ue_log_energy: np.ndarray,
    cells: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    power = np.asarray(log_power, dtype=np.float32)
    ue = np.asarray(ue_log_energy, dtype=np.float32)
    cell_ids = np.asarray(cells, dtype=np.int64)
    calibration = np.asarray(parameters, dtype=np.float32)[cell_ids]
    calibrated = calibration[:, 1] + calibration[:, 2] * (power - calibration[:, 0])
    shift = calibrated - power
    return calibrated.astype(np.float32), (ue + shift[:, None]).astype(np.float32)


def compute_power_bounds(
    log_power: np.ndarray,
    outage: np.ndarray,
    cells: np.ndarray,
    observed_indices: np.ndarray,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> np.ndarray:
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Power quantiles must satisfy 0 <= lower < upper <= 1")
    values = np.asarray(log_power, dtype=np.float32)
    outage_mask = np.asarray(outage, dtype=np.bool_)
    cell_ids = np.asarray(cells, dtype=np.int64)
    observed = np.asarray(observed_indices, dtype=np.int64)
    cell_count = int(cell_ids.max()) + 1
    bounds = np.empty((cell_count, 2), dtype=np.float32)
    for cell in range(cell_count):
        selected = observed[(cell_ids[observed] == cell) & ~outage_mask[observed]]
        if not len(selected):
            raise RuntimeError(f"Cell {cell} has no non-outage power samples")
        low, high = np.quantile(
            values[selected].astype(np.float64), [lower_quantile, upper_quantile]
        )
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise RuntimeError(f"Invalid power bounds for cell {cell}: {low}, {high}")
        bounds[cell] = (float(low), float(high))
    return bounds


def clip_power_priors(
    log_power: np.ndarray,
    ue_log_energy: np.ndarray,
    cells: np.ndarray,
    bounds: np.ndarray,
    maximum_ue_offset: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    power = np.asarray(log_power, dtype=np.float32).copy()
    ue = np.asarray(ue_log_energy, dtype=np.float32).copy()
    cell_ids = np.asarray(cells, dtype=np.int64)
    limits = np.asarray(bounds, dtype=np.float32)[cell_ids]
    original_power = power.copy()
    power = np.clip(power, limits[:, 0], limits[:, 1])
    relative = np.clip(
        ue - original_power[:, None],
        -float(maximum_ue_offset),
        float(maximum_ue_offset),
    )
    ue = np.clip(
        power[:, None] + relative,
        limits[:, :1] - float(maximum_ue_offset),
        limits[:, 1:] + float(maximum_ue_offset),
    )
    return power.astype(np.float32), ue.astype(np.float32)


def torch_power_bounds(
    cells: torch.Tensor,
    bounds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = bounds[cells.long()]
    return selected[:, 0].float(), selected[:, 1].float()


def apply_outage_policy(
    channel: torch.Tensor,
    outage_probability: torch.Tensor,
    threshold: torch.Tensor | float,
    soft_strength: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    probability = outage_probability.float().clamp(0.0, 1.0)
    threshold_tensor = torch.as_tensor(threshold, device=channel.device).float()
    strength_tensor = torch.as_tensor(soft_strength, device=channel.device).float()
    while threshold_tensor.ndim < probability.ndim:
        threshold_tensor = threshold_tensor.unsqueeze(-1)
    while strength_tensor.ndim < probability.ndim:
        strength_tensor = strength_tensor.unsqueeze(-1)
    attenuation = torch.pow((1.0 - probability).clamp_min(1e-4), strength_tensor)
    result = channel * attenuation.sqrt()[:, None, None, None].to(channel.dtype)
    hard = probability >= threshold_tensor
    return result.masked_fill(hard[:, None, None, None], 0.0)
