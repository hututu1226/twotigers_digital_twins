from __future__ import annotations

import torch

from .angle_delay import ChannelShape
from .spectral_targets import decode_pas_log, decode_pdp_log


def decode_pas_marginals(
    pas_log: torch.Tensor,
    shape: ChannelShape,
    proxy_count: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the mean 2-D PAS into horizontal and vertical marginal targets."""
    _, mean_pas = decode_pas_log(pas_log, shape, proxy_count)
    horizontal = mean_pas.sum(dim=1)
    vertical = mean_pas.sum(dim=2)
    horizontal = horizontal / horizontal.sum(dim=1, keepdim=True).clamp_min(1e-30)
    vertical = vertical / vertical.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return horizontal, vertical


def _axis_projection(
    array: torch.Tensor,
    target: torch.Tensor,
    axis: int,
    aggregate_axes: tuple[int, ...],
    minimum_scale: float,
    maximum_scale: float,
) -> torch.Tensor:
    transformed = torch.fft.fft(array, dim=axis, norm="ortho")
    current = transformed.abs().square().float().sum(dim=aggregate_axes)
    current = current / current.sum(dim=1, keepdim=True).clamp_min(1e-30)
    scale = torch.sqrt(target.clamp_min(1e-30) / current.clamp_min(1e-30))
    scale = scale.clamp(float(minimum_scale), float(maximum_scale))
    view = [len(scale), 1, 1, 1, 1, 1]
    view[axis] = scale.shape[1]
    transformed = transformed * scale.reshape(view).to(transformed.dtype)
    return torch.fft.ifft(transformed, dim=axis, norm="ortho")


def _per_ue_pdp_projection(
    channel: torch.Tensor,
    target_pdp: torch.Tensor,
    minimum_scale: float,
    maximum_scale: float,
) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1, norm="ortho")
    current = delay.abs().square().float().mean(dim=1)
    current = current / current.sum(dim=2, keepdim=True).clamp_min(1e-30)
    scale = torch.sqrt(target_pdp.clamp_min(1e-30) / current.clamp_min(1e-30))
    scale = scale.clamp(float(minimum_scale), float(maximum_scale))
    delay = delay * scale[:, None].to(delay.dtype)
    return torch.fft.fft(delay, dim=-1, norm="ortho")


def _per_ue_energy_projection(
    channel: torch.Tensor,
    ue_log_energy: torch.Tensor,
    minimum_scale: float,
    maximum_scale: float,
) -> torch.Tensor:
    current = channel.abs().square().float().mean(dim=(1, 3))
    target = torch.pow(10.0, ue_log_energy.float())
    scale = torch.sqrt(target.clamp_min(1e-30) / current.clamp_min(1e-30))
    scale = scale.clamp(float(minimum_scale), float(maximum_scale))
    return channel * scale[:, None, :, None].to(channel.dtype)


def alternating_marginal_projection(
    channel: torch.Tensor,
    pas_log: torch.Tensor,
    pdp_log: torch.Tensor,
    ue_log_energy: torch.Tensor,
    shape: ChannelShape,
    *,
    iterations: int = 8,
    proxy_count: int = 24,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
) -> torch.Tensor:
    """Round1-style H/V marginal projection with current per-UE PDP and power targets."""
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    horizontal, vertical = decode_pas_marginals(pas_log, shape, proxy_count)
    target_pdp = decode_pdp_log(pdp_log, shape)
    result = channel
    for _ in range(int(iterations)):
        array = result.reshape(
            -1, shape.m_p, shape.m_v, shape.m_h, shape.n, shape.s
        )
        array = _axis_projection(
            array,
            horizontal,
            axis=3,
            aggregate_axes=(1, 2, 4, 5),
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
        )
        array = _axis_projection(
            array,
            vertical,
            axis=2,
            aggregate_axes=(1, 3, 4, 5),
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
        )
        result = array.reshape_as(channel)
        result = _per_ue_pdp_projection(
            result, target_pdp, minimum_scale, maximum_scale
        )
        result = _per_ue_energy_projection(
            result, ue_log_energy, minimum_scale, maximum_scale
        )
    return result
