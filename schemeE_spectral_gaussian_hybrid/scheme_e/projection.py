from __future__ import annotations

import torch

from .angle_delay import ChannelShape
from .spectral_targets import decode_pas_log, decode_pdp_log


def _pas_projection(
    channel: torch.Tensor,
    target_proxy: torch.Tensor,
    shape: ChannelShape,
    proxy_count: int,
    minimum_scale: float,
    maximum_scale: float,
) -> torch.Tensor:
    array = channel.reshape(-1, shape.m_p, shape.m_v, shape.m_h, shape.n, shape.s)
    beam = torch.fft.fft2(array, dim=(2, 3), norm="ortho")
    power = beam.abs().square().float().mean(dim=(1, 4)).permute(0, 3, 1, 2)
    group_size = shape.s // proxy_count
    current = power.reshape(-1, proxy_count, group_size, shape.m_v, shape.m_h).mean(dim=2)
    current = current / current.sum(dim=(2, 3), keepdim=True).clamp_min(1e-30)
    scale = torch.sqrt(target_proxy.clamp_min(1e-30) / current.clamp_min(1e-30))
    scale = scale.clamp(float(minimum_scale), float(maximum_scale))
    scale = scale.repeat_interleave(group_size, dim=1).permute(0, 2, 3, 1)
    beam = beam * scale[:, None, :, :, None, :].to(beam.dtype)
    return torch.fft.ifft2(beam, dim=(2, 3), norm="ortho").reshape_as(channel)


def _pdp_projection(
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
    delay = delay * scale[:, None, :, :].to(delay.dtype)
    return torch.fft.fft(delay, dim=-1, norm="ortho")


def _energy_projection(
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


def alternating_spectral_projection(
    channel: torch.Tensor,
    pas_log: torch.Tensor,
    pdp_log: torch.Tensor,
    ue_log_energy: torch.Tensor,
    shape: ChannelShape,
    iterations: int = 4,
    proxy_count: int = 24,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
) -> torch.Tensor:
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    target_proxy, _ = decode_pas_log(pas_log, shape, proxy_count)
    target_pdp = decode_pdp_log(pdp_log, shape)
    result = channel
    for _ in range(int(iterations)):
        result = _pas_projection(
            result,
            target_proxy,
            shape,
            proxy_count,
            minimum_scale,
            maximum_scale,
        )
        result = _pdp_projection(result, target_pdp, minimum_scale, maximum_scale)
        result = _energy_projection(result, ue_log_energy, minimum_scale, maximum_scale)
    return result
