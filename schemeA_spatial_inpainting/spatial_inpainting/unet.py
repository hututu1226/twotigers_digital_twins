from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, dropout: float) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
        )


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(output_channels + skip_channels, output_channels, dropout)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, value], dim=1))


class SpatialUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        base_channels: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        widths = [base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels]
        self.encoder0 = ConvBlock(input_channels, widths[0], dropout)
        self.encoder1 = ConvBlock(widths[0], widths[1], dropout)
        self.encoder2 = ConvBlock(widths[1], widths[2], dropout)
        self.bottleneck = ConvBlock(widths[2], widths[3], dropout)
        self.pool = nn.MaxPool2d(2)
        self.decoder2 = UpBlock(widths[3], widths[2], widths[2], dropout)
        self.decoder1 = UpBlock(widths[2], widths[1], widths[1], dropout)
        self.decoder0 = UpBlock(widths[1], widths[0], widths[0], dropout)
        self.latent_head = nn.Conv2d(widths[0], latent_dim, kernel_size=1)
        self.power_head = nn.Conv2d(widths[0], 1, kernel_size=1)
        self.outage_head = nn.Conv2d(widths[0], 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        skip0 = self.encoder0(value)
        skip1 = self.encoder1(self.pool(skip0))
        skip2 = self.encoder2(self.pool(skip1))
        value = self.bottleneck(self.pool(skip2))
        value = self.decoder2(value, skip2)
        value = self.decoder1(value, skip1)
        value = self.decoder0(value, skip0)
        return {
            "latent": self.latent_head(value),
            "power": self.power_head(value),
            "outage_logit": self.outage_head(value),
        }


def pad_to_multiple(value: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = value.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    return functional.pad(value, (0, pad_width, 0, pad_height)), (height, width)


def unpad(value: torch.Tensor, original_shape: tuple[int, int]) -> torch.Tensor:
    return value[..., : original_shape[0], : original_shape[1]]

