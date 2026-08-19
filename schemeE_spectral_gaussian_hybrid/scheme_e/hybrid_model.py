from __future__ import annotations

import torch
from torch import nn

from .angle_delay import ChannelShape, channel_to_shape_target, shape_to_channel
from .autoencoder import FactorizedResidualAutoencoder
from .projection import alternating_spectral_projection


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


class ResidualAdapter3d(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv3d(channels * 2, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.pointwise(self.depthwise(self.norm(value)))


class SpectralConditionEncoder(nn.Module):
    def __init__(
        self,
        shape: ChannelShape,
        proxy_count: int,
        geometry_dim: int,
        width: int,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.proxy_count = int(proxy_count)
        self.pas = nn.Sequential(
            nn.Conv2d(self.proxy_count + 1, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, width // 2, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.pdp = nn.Sequential(
            nn.Conv1d(shape.n, width // 2, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(width // 2, width // 2, 7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.geometry = nn.Sequential(
            nn.Linear(int(geometry_dim), width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(width * 2 + shape.n + 3, width * 2),
            nn.LayerNorm(width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
        )

    def forward(
        self,
        pas_log: torch.Tensor,
        pdp_log: torch.Tensor,
        ue_log_energy: torch.Tensor,
        log_power: torch.Tensor,
        uncertainty: torch.Tensor,
        outage_probability: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        pas = pas_log.reshape(
            -1, self.proxy_count + 1, self.shape.m_v, self.shape.m_h
        )
        pdp = pdp_log.reshape(-1, self.shape.n, self.shape.s)
        values = torch.cat(
            [
                self.pas(pas.float()),
                self.pdp(pdp.float()),
                self.geometry(geometry.float()),
                ue_log_energy.float(),
                log_power.float()[:, None],
                uncertainty.float()[:, None],
                outage_probability.float()[:, None],
            ],
            dim=1,
        )
        return self.output(values)


class FullResolutionLatentAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_width: int,
        blocks: int,
        maximum_residual: float,
    ) -> None:
        super().__init__()
        self.maximum_residual = float(maximum_residual)
        self.condition = nn.Linear(condition_width, channels * 2)
        self.blocks = nn.Sequential(
            *[ResidualAdapter3d(channels, 1 if index % 2 == 0 else 2) for index in range(blocks)]
        )
        self.output = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.gate_logit = nn.Parameter(torch.full((channels,), -2.0))

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.condition(condition).chunk(2, dim=1)
        features = latent * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, None, None])
        features = features + 0.1 * shift[:, :, None, None, None]
        residual = torch.tanh(self.output(self.blocks(features))) * self.maximum_residual
        gate = torch.sigmoid(self.gate_logit)[None, :, None, None, None]
        return latent + gate * residual, residual


class SpectralGaussianHybrid(nn.Module):
    def __init__(
        self,
        autoencoder: FactorizedResidualAutoencoder,
        shape: ChannelShape,
        proxy_count: int = 24,
        geometry_dim: int = 71,
        condition_width: int = 192,
        spectrum_blocks: int = 4,
        detail_blocks: int = 6,
        maximum_spectrum_residual: float = 1.0,
        maximum_detail_residual: float = 1.0,
        projection_iterations: int = 4,
        projection_minimum_scale: float = 0.25,
        projection_maximum_scale: float = 4.0,
        train_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.shape = shape
        self.proxy_count = int(proxy_count)
        self.projection_iterations = int(projection_iterations)
        self.projection_minimum_scale = float(projection_minimum_scale)
        self.projection_maximum_scale = float(projection_maximum_scale)
        self.condition_encoder = SpectralConditionEncoder(
            shape, self.proxy_count, geometry_dim, condition_width
        )
        self.spectrum_adapter = FullResolutionLatentAdapter(
            autoencoder.spectrum_shape.channels,
            condition_width,
            spectrum_blocks,
            maximum_spectrum_residual,
        )
        self.detail_adapter = FullResolutionLatentAdapter(
            autoencoder.phase_shape.channels,
            condition_width,
            detail_blocks,
            maximum_detail_residual,
        )
        self.power_head = nn.Sequential(
            nn.Linear(condition_width, condition_width),
            nn.GELU(),
            nn.Linear(condition_width, 1),
        )
        nn.init.zeros_(self.power_head[-1].weight)
        nn.init.zeros_(self.power_head[-1].bias)
        autoencoder.spectrum_encoder.requires_grad_(False)
        autoencoder.phase_encoder.requires_grad_(False)
        autoencoder.decoder.requires_grad_(bool(train_decoder))

    def forward(
        self,
        reference_channel: torch.Tensor,
        pas_log: torch.Tensor,
        pdp_log: torch.Tensor,
        ue_log_energy: torch.Tensor,
        log_power: torch.Tensor,
        uncertainty: torch.Tensor,
        outage_probability: torch.Tensor,
        geometry: torch.Tensor,
        projection_iterations: int | None = None,
    ) -> dict[str, torch.Tensor]:
        iterations = self.projection_iterations if projection_iterations is None else int(projection_iterations)
        projected = alternating_spectral_projection(
            reference_channel,
            pas_log,
            pdp_log,
            ue_log_energy,
            self.shape,
            iterations,
            self.proxy_count,
            self.projection_minimum_scale,
            self.projection_maximum_scale,
        )
        projected_shape, _, _ = channel_to_shape_target(projected, self.shape)
        with torch.set_grad_enabled(self.training):
            spectrum, detail = self.autoencoder.encode(projected_shape)
        spectrum_grid = spectrum.reshape(-1, *self.autoencoder.spectrum_shape.tensor_shape)
        detail_grid = detail.reshape(-1, *self.autoencoder.phase_shape.tensor_shape)
        condition = self.condition_encoder(
            pas_log,
            pdp_log,
            ue_log_energy,
            log_power,
            uncertainty,
            outage_probability,
            geometry,
        )
        adapted_spectrum, spectrum_residual = self.spectrum_adapter(spectrum_grid, condition)
        adapted_detail, detail_residual = self.detail_adapter(detail_grid, condition)
        decoded = self.autoencoder.decode(adapted_spectrum.flatten(1), adapted_detail.flatten(1))
        power_delta = 0.5 * torch.tanh(self.power_head(condition).squeeze(1))
        predicted_power = log_power.float() + power_delta
        channel = shape_to_channel(decoded, predicted_power, self.shape)
        return {
            "channel": channel,
            "projected_channel": projected,
            "spectrum": adapted_spectrum,
            "detail": adapted_detail,
            "spectrum_residual": spectrum_residual,
            "detail_residual": detail_residual,
            "power": predicted_power,
            "power_delta": power_delta,
        }
