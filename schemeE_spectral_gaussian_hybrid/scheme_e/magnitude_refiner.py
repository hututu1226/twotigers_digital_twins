from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


def normalize_log_power_grid(
    log_power: torch.Tensor,
    scale: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize each angle-delay power grid to unit mean power."""
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive")
    power = torch.expm1(log_power.float().clamp(0.0, 20.0)) / float(scale)
    mean = power.mean(dim=tuple(range(1, power.ndim)), keepdim=True)
    normalized = power / mean.clamp_min(float(epsilon))
    normalized = torch.where(mean > float(epsilon), normalized, torch.zeros_like(power))
    return torch.log1p(float(scale) * normalized)


def energy_weighted_log_power_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: float,
    emphasis: float = 2.0,
    maximum_weight: float = 12.0,
    beta: float = 0.25,
) -> torch.Tensor:
    """Fit strong multipath bins without letting the many dark bins dominate."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target grids must have equal shapes")
    target = target.float().clamp(0.0, 20.0)
    target_power = torch.expm1(target) / float(scale)
    weights = (1.0 + float(emphasis) * target_power.sqrt()).clamp_max(
        float(maximum_weight)
    )
    error = functional.smooth_l1_loss(
        prediction.float(), target, reduction="none", beta=float(beta)
    )
    return (error * weights).sum() / weights.sum().clamp_min(1e-8)


class LocalMagnitudeBlock3d(nn.Module):
    def __init__(self, channels: int, delay_dilation: int, dropout: float) -> None:
        super().__init__()
        self.delay_dilation = int(delay_dilation)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 3, 5),
            dilation=(1, 1, self.delay_dilation),
            groups=channels,
            bias=False,
        )
        self.mixer = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout3d(float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Conv3d(channels * 2, channels, 1),
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1, 1), 0.1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value)
        delay_padding = 2 * self.delay_dilation
        normalized = functional.pad(
            normalized,
            (delay_padding, delay_padding, 0, 0, 0, 0),
            mode="replicate",
        )
        normalized = functional.pad(
            normalized,
            (0, 0, 1, 1, 1, 1),
            mode="circular",
        )
        update = self.mixer(self.depthwise(normalized))
        return value + self.residual_scale * update


class FullResolutionMagnitudeRefiner(nn.Module):
    """Refine a complete angle-delay log-power grid without a latent bottleneck."""

    def __init__(
        self,
        input_channels: int,
        geometry_dim: int,
        cell_count: int,
        width: int = 32,
        blocks: int = 5,
        dropout: float = 0.03,
        maximum_residual: float = 4.0,
        log_power_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.maximum_residual = float(maximum_residual)
        self.log_power_scale = float(log_power_scale)
        self.stem = nn.Conv3d(self.input_channels + 3, int(width), 1)
        self.geometry = nn.Sequential(
            nn.Linear(int(geometry_dim), int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width) * 2),
        )
        self.station = nn.Embedding(int(cell_count), int(width) * 2)
        dilations = (1, 2, 4, 8)
        self.blocks = nn.Sequential(
            *[
                LocalMagnitudeBlock3d(
                    int(width), dilations[index % len(dilations)], float(dropout)
                )
                for index in range(int(blocks))
            ]
        )
        self.output_norm = nn.GroupNorm(_groups(int(width)), int(width))
        self.output = nn.Conv3d(int(width), self.input_channels, 1)
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        nn.init.zeros_(self.station.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _coordinates(
        value: torch.Tensor,
        delay_start: int,
        total_delay: int,
    ) -> torch.Tensor:
        _, _, vertical, horizontal, delay = value.shape
        v = torch.linspace(-1.0, 1.0, vertical, device=value.device, dtype=value.dtype)
        h = torch.linspace(-1.0, 1.0, horizontal, device=value.device, dtype=value.dtype)
        absolute = torch.arange(
            int(delay_start),
            int(delay_start) + delay,
            device=value.device,
            dtype=value.dtype,
        )
        if int(total_delay) > 1:
            absolute = 2.0 * absolute / float(int(total_delay) - 1) - 1.0
        else:
            absolute.zero_()
        vv, hh, dd = torch.meshgrid(v, h, absolute, indexing="ij")
        return torch.stack([vv, hh, dd], dim=0)[None].expand(
            len(value), -1, -1, -1, -1
        )

    def forward(
        self,
        base_log_power: torch.Tensor,
        geometry: torch.Tensor,
        cell_ids: torch.Tensor,
        *,
        delay_start: int = 0,
        total_delay: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if base_log_power.ndim != 5:
            raise ValueError("base_log_power must be [B,C,V,H,S]")
        if base_log_power.shape[1] != self.input_channels:
            raise ValueError("base_log_power has the wrong channel count")
        if total_delay is None:
            total_delay = int(base_log_power.shape[-1])
        coordinates = self._coordinates(
            base_log_power.float(), int(delay_start), int(total_delay)
        )
        value = self.stem(
            torch.cat([base_log_power.float() / 4.0, coordinates], dim=1)
        )
        scale, bias = (
            self.geometry(geometry.float()) + self.station(cell_ids.long())
        ).chunk(2, dim=1)
        value = value * (1.0 + torch.tanh(scale)[:, :, None, None, None])
        value = value + bias[:, :, None, None, None]
        value = self.blocks(value)
        raw = self.output(functional.gelu(self.output_norm(value)))
        correction = self.maximum_residual * torch.tanh(raw)
        corrected = (base_log_power.float() + correction).clamp(0.0, 20.0)
        return {"log_power": corrected, "correction": correction}
