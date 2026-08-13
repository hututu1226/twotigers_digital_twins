from __future__ import annotations

import torch
from torch import nn

from .angle_delay import ChannelShape, normalize_angle_delay


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class EncoderBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv3d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
        )


class DecoderBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, activate: bool = True) -> None:
        layers: list[nn.Module] = [
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
        ]
        if activate:
            layers.extend([nn.GroupNorm(_groups(output_channels), output_channels), nn.GELU()])
        super().__init__(*layers)


class AngleDelayAutoencoder(nn.Module):
    def __init__(self, shape: ChannelShape, base_channels: int = 16, latent_dim: int = 256) -> None:
        super().__init__()
        if shape.m_v % 8 or shape.m_h % 8 or shape.s % 8:
            raise ValueError("Angle-delay dimensions must be divisible by 8")
        self.shape = shape
        self.base_channels = int(base_channels)
        self.latent_dim = int(latent_dim)
        self.compressed_shape = (
            4 * self.base_channels,
            shape.m_v // 8,
            shape.m_h // 8,
            shape.s // 8,
        )
        flattened = 1
        for value in self.compressed_shape:
            flattened *= value
        self.encoder_conv = nn.Sequential(
            EncoderBlock(shape.ad_channels, self.base_channels),
            EncoderBlock(self.base_channels, 2 * self.base_channels),
            EncoderBlock(2 * self.base_channels, 4 * self.base_channels),
        )
        self.to_latent = nn.Linear(flattened, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, flattened)
        self.decoder_conv = nn.Sequential(
            DecoderBlock(4 * self.base_channels, 2 * self.base_channels),
            DecoderBlock(2 * self.base_channels, self.base_channels),
            DecoderBlock(self.base_channels, shape.ad_channels, activate=False),
        )

    def encode(self, angle_delay: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder_conv(angle_delay).flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        feature = self.from_latent(latent).reshape(-1, *self.compressed_shape)
        return normalize_angle_delay(self.decoder_conv(feature).float())

    def forward(self, angle_delay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(angle_delay)
        return self.decode(latent), latent
