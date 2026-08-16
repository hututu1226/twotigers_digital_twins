from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .angle_delay import ChannelShape, normalize_angle_delay


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


class Residual3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.block(value))


class Down3d(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int],
        padding: int | tuple[int, int, int],
    ) -> None:
        super().__init__(
            nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            Residual3d(output_channels),
        )


class Up3d(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int],
        padding: int | tuple[int, int, int],
        residual: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.ConvTranspose3d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not residual,
            )
        ]
        if residual:
            layers.extend(
                [
                    nn.GroupNorm(_groups(output_channels), output_channels),
                    nn.GELU(),
                    Residual3d(output_channels),
                ]
            )
        super().__init__(*layers)


@dataclass(frozen=True)
class StructuredLatentShape:
    channels: int
    angle_vertical: int
    angle_horizontal: int
    delay: int

    @property
    def tensor_shape(self) -> tuple[int, int, int, int]:
        return self.channels, self.angle_vertical, self.angle_horizontal, self.delay

    @property
    def elements(self) -> int:
        return self.channels * self.angle_vertical * self.angle_horizontal * self.delay


class BranchEncoder(nn.Module):
    def __init__(self, input_channels: int, stem_channels: int, latent_channels: int) -> None:
        super().__init__()
        middle_channels = max(stem_channels * 2, latent_channels)
        self.network = nn.Sequential(
            nn.Conv3d(input_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(stem_channels), stem_channels),
            nn.GELU(),
            Residual3d(stem_channels),
            Down3d(stem_channels, middle_channels, 4, 2, 1),
            Down3d(middle_channels, latent_channels, 4, 2, 1),
            Down3d(latent_channels, latent_channels, (3, 3, 4), (1, 1, 4), (1, 1, 0)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class StructuredAngleDelayAutoencoder(nn.Module):
    """Two-branch AE that preserves a 2x4 angle grid and 12 delay bins."""

    def __init__(
        self,
        shape: ChannelShape,
        spectrum_stem_channels: int = 16,
        phase_stem_channels: int = 8,
        spectrum_latent_channels: int = 32,
        phase_latent_channels: int = 16,
        spectrum_log_scale: float = 4.0,
    ) -> None:
        super().__init__()
        if shape.m_v % 4 or shape.m_h % 4 or shape.s % 16:
            raise ValueError("M_V/M_H must be divisible by 4 and S must be divisible by 16")
        self.shape = shape
        self.spectrum_log_scale = float(spectrum_log_scale)
        self.spectrum_shape = StructuredLatentShape(
            int(spectrum_latent_channels), shape.m_v // 4, shape.m_h // 4, shape.s // 16
        )
        self.phase_shape = StructuredLatentShape(
            int(phase_latent_channels), shape.m_v // 4, shape.m_h // 4, shape.s // 16
        )
        complex_channels = shape.ad_channels // 2
        self.spectrum_encoder = BranchEncoder(
            complex_channels, int(spectrum_stem_channels), int(spectrum_latent_channels)
        )
        self.phase_encoder = BranchEncoder(
            shape.ad_channels, int(phase_stem_channels), int(phase_latent_channels)
        )
        latent_channels = int(spectrum_latent_channels + phase_latent_channels)
        decoder_middle = max(int(spectrum_stem_channels * 2), latent_channels)
        self.decoder = nn.Sequential(
            Up3d(latent_channels, latent_channels, (3, 3, 4), (1, 1, 4), (1, 1, 0)),
            Up3d(latent_channels, decoder_middle, 4, 2, 1),
            Up3d(decoder_middle, shape.ad_channels, 4, 2, 1, residual=False),
        )

    @property
    def spectrum_latent_dim(self) -> int:
        return self.spectrum_shape.elements

    @property
    def phase_latent_dim(self) -> int:
        return self.phase_shape.elements

    @property
    def total_latent_dim(self) -> int:
        return self.spectrum_latent_dim + self.phase_latent_dim

    def spectrum_input(self, angle_delay: torch.Tensor) -> torch.Tensor:
        batch = angle_delay.shape[0]
        parts = angle_delay.reshape(
            batch,
            self.shape.m_p * self.shape.n,
            2,
            self.shape.m_v,
            self.shape.m_h,
            self.shape.s,
        )
        power = parts.float().square().sum(dim=2)
        return torch.log1p(self.spectrum_log_scale * power)

    def encode(self, angle_delay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = self.spectrum_encoder(self.spectrum_input(angle_delay))
        phase = self.phase_encoder(angle_delay)
        return spectrum.flatten(1), phase.flatten(1)

    def decode(
        self,
        spectrum_latent: torch.Tensor,
        phase_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spectrum = spectrum_latent.reshape(-1, *self.spectrum_shape.tensor_shape)
        if phase_latent is None:
            phase = spectrum.new_zeros((len(spectrum), *self.phase_shape.tensor_shape))
        else:
            phase = phase_latent.reshape(-1, *self.phase_shape.tensor_shape)
        return normalize_angle_delay(self.decoder(torch.cat([spectrum, phase], dim=1)).float())

    def forward(
        self, angle_delay: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrum, phase = self.encode(angle_delay)
        return self.decode(spectrum, phase), spectrum, phase
