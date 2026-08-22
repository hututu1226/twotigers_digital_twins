from __future__ import annotations

import torch

from .angle_delay import ChannelShape


def angle_delay_to_complex(
    angle_delay: torch.Tensor, shape: ChannelShape
) -> torch.Tensor:
    """View real angle-delay channels as [B,Mp,N,Mv,Mh,S] complex values."""
    if angle_delay.ndim != 5 or tuple(angle_delay.shape[1:]) != shape.ad_shape:
        raise ValueError(
            f"Expected [B,{','.join(map(str, shape.ad_shape))}], "
            f"got {tuple(angle_delay.shape)}"
        )
    parts = angle_delay.reshape(
        -1, shape.m_p, shape.n, 2, shape.m_v, shape.m_h, shape.s
    )
    return torch.complex(parts[:, :, :, 0], parts[:, :, :, 1])


def complex_to_angle_delay(
    values: torch.Tensor, shape: ChannelShape
) -> torch.Tensor:
    expected = (shape.m_p, shape.n, shape.m_v, shape.m_h, shape.s)
    if values.ndim != 6 or tuple(values.shape[1:]) != expected:
        raise ValueError(
            f"Expected [B,{','.join(map(str, expected))}], got {tuple(values.shape)}"
        )
    parts = torch.stack([values.real, values.imag], dim=3)
    return parts.reshape(-1, *shape.ad_shape)


def angle_delay_log_power(
    angle_delay: torch.Tensor,
    shape: ChannelShape,
    scale: float,
) -> torch.Tensor:
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    power = angle_delay_to_complex(angle_delay, shape).abs().square()
    return torch.log1p(float(scale) * power)


def replace_angle_delay_log_power(
    base: torch.Tensor,
    log_power: torch.Tensor,
    shape: ChannelShape,
    scale: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Apply a full-resolution log-power map while preserving base phase."""
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    base_complex = angle_delay_to_complex(base, shape)
    expected = tuple(base_complex.shape)
    if tuple(log_power.shape) != expected:
        raise ValueError(f"Expected log-power shape {expected}, got {tuple(log_power.shape)}")
    base_magnitude = base_complex.abs()
    base_unit = base_complex / base_magnitude.clamp_min(float(epsilon))
    base_unit = torch.where(
        base_magnitude > float(epsilon), base_unit, torch.ones_like(base_unit)
    )
    magnitude = torch.sqrt(
        torch.expm1(log_power.float().clamp(0.0, 20.0)).clamp_min(0.0)
        / float(scale)
    )
    return complex_to_angle_delay(magnitude * base_unit, shape)


def split_complex_correction(
    base: torch.Tensor,
    corrected: torch.Tensor,
    shape: ChannelShape,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Separate a complex correction into magnitude-only and phase-only variants."""
    base_complex = angle_delay_to_complex(base, shape)
    corrected_complex = angle_delay_to_complex(corrected, shape)
    base_magnitude = base_complex.abs()
    corrected_magnitude = corrected_complex.abs()
    base_unit = base_complex / base_magnitude.clamp_min(float(epsilon))
    corrected_unit = corrected_complex / corrected_magnitude.clamp_min(float(epsilon))
    base_unit = torch.where(
        base_magnitude > float(epsilon), base_unit, corrected_unit
    )
    corrected_unit = torch.where(
        corrected_magnitude > float(epsilon), corrected_unit, base_unit
    )
    return {
        "complex": corrected,
        "magnitude": complex_to_angle_delay(
            corrected_magnitude * base_unit, shape
        ),
        "phase": complex_to_angle_delay(base_magnitude * corrected_unit, shape),
    }


def reconstruct_low_rank_residual(
    residual: torch.Tensor,
    mean: torch.Tensor,
    components: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Return the target-informed projection used only by oracle diagnostics."""
    if residual.ndim != 2:
        raise ValueError("residual must be [samples, features]")
    mean = mean.reshape(1, -1).to(device=residual.device, dtype=residual.dtype)
    components = components.to(device=residual.device, dtype=residual.dtype)
    if residual.shape[1] != mean.shape[1] or components.shape[1] != mean.shape[1]:
        raise ValueError("Residual, mean, and component feature widths differ")
    selected_rank = int(rank)
    if selected_rank < 0 or selected_rank > len(components):
        raise ValueError(
            f"rank must be between 0 and {len(components)}, got {selected_rank}"
        )
    if selected_rank == 0:
        return mean.expand_as(residual)
    selected = components[:selected_rank]
    coefficients = (residual - mean) @ selected.T
    return mean + coefficients @ selected
