from __future__ import annotations

import torch

from .angle_delay import ChannelShape, channel_power
from .spectral_targets import decode_pas_log, decode_pdp_log


def _pas_projection(
    channel: torch.Tensor,
    target_proxy: torch.Tensor,
    target_mean: torch.Tensor,
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

    mean_power = beam.abs().square().float().mean(dim=(1, 4)).mean(dim=-1)
    current_mean = mean_power / mean_power.sum(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-30)
    mean_scale = torch.sqrt(
        target_mean.clamp_min(1e-30) / current_mean.clamp_min(1e-30)
    )
    mean_scale = mean_scale.clamp(float(minimum_scale), float(maximum_scale))
    beam = beam * mean_scale[:, None, :, :, None, None].to(beam.dtype)
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
    target_proxy, target_mean = decode_pas_log(pas_log, shape, proxy_count)
    target_pdp = decode_pdp_log(pdp_log, shape)
    result = channel
    for _ in range(int(iterations)):
        result = _pas_projection(
            result,
            target_proxy,
            target_mean,
            shape,
            proxy_count,
            minimum_scale,
            maximum_scale,
        )
        result = _pdp_projection(result, target_pdp, minimum_scale, maximum_scale)
        result = _energy_projection(result, ue_log_energy, minimum_scale, maximum_scale)
    return result


def relaxed_output_projection(
    channel: torch.Tensor,
    pas_log: torch.Tensor,
    pdp_log: torch.Tensor,
    ue_log_energy: torch.Tensor,
    log_power: torch.Tensor,
    shape: ChannelShape,
    iterations: int = 1,
    proxy_count: int = 24,
    strength: float | torch.Tensor = 1.0,
    minimum_scale: float = 0.5,
    maximum_scale: float = 2.0,
) -> torch.Tensor:
    """Correct decoded PAS/PDP while preserving the model's predicted power."""
    if iterations <= 0:
        return channel
    strength_tensor = torch.as_tensor(
        strength, dtype=channel.real.dtype, device=channel.device
    ).reshape(-1)
    if strength_tensor.numel() == 1:
        strength_tensor = strength_tensor.expand(channel.shape[0])
    if strength_tensor.numel() != channel.shape[0]:
        raise ValueError("strength must be scalar or contain one value per sample")
    strength_tensor = strength_tensor.clamp(0.0, 1.0)
    if not torch.any(strength_tensor > 0):
        return channel

    ue_linear = torch.pow(10.0, ue_log_energy.float()).clamp_min(1e-30)
    ue_linear = ue_linear / ue_linear.mean(dim=1, keepdim=True).clamp_min(1e-30)
    target_ue_log = torch.log10(ue_linear) + log_power.float()[:, None]
    projected = alternating_spectral_projection(
        channel,
        pas_log,
        pdp_log,
        target_ue_log,
        shape,
        iterations=iterations,
        proxy_count=proxy_count,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
    )
    alpha = strength_tensor[:, None, None, None]
    mixed = channel + alpha.to(channel.dtype) * (projected - channel)
    current_power = channel_power(mixed).clamp_min(1e-30)
    target_power = torch.pow(10.0, log_power.float()).clamp_min(1e-30)
    gain = torch.sqrt(target_power / current_power)
    return mixed * gain[:, None, None, None].to(mixed.dtype)
