from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


def pad_to_multiple(value: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = value.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    return functional.pad(value, (0, pad_width, 0, pad_height)), (height, width)


def unpad(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return value[..., : shape[0], : shape[1]]


class GatedConv2d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.feature = nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding)
        self.gate = nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(_groups(output_channels), output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        feature = functional.gelu(self.norm(self.feature(value)))
        return feature * torch.sigmoid(self.gate(value))


class GatedBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float) -> None:
        super().__init__()
        self.first = GatedConv2d(input_channels, output_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.second = GatedConv2d(output_channels, output_channels)
        self.skip = (
            nn.Conv2d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        return self.second(self.dropout(self.first(value))) + residual


class GatedContextFPN(nn.Module):
    def __init__(self, input_channels: int, base_channels: int, output_channels: int, dropout: float) -> None:
        super().__init__()
        widths = [base_channels, 2 * base_channels, 4 * base_channels, 8 * base_channels]
        self.encoder0 = GatedBlock(input_channels, widths[0], dropout)
        self.encoder1 = GatedBlock(widths[0], widths[1], dropout)
        self.encoder2 = GatedBlock(widths[1], widths[2], dropout)
        self.bottleneck = GatedBlock(widths[2], widths[3], dropout)
        self.pool = nn.MaxPool2d(2)
        self.decoder2 = GatedBlock(widths[3] + widths[2], widths[2], dropout)
        self.decoder1 = GatedBlock(widths[2] + widths[1], widths[1], dropout)
        self.decoder0 = GatedBlock(widths[1] + widths[0], widths[0], dropout)
        self.output = nn.Conv2d(widths[0], output_channels, kernel_size=1)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([value, skip], dim=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        skip0 = self.encoder0(value)
        skip1 = self.encoder1(self.pool(skip0))
        skip2 = self.encoder2(self.pool(skip1))
        value = self.bottleneck(self.pool(skip2))
        value = self.decoder2(self._up(value, skip2))
        value = self.decoder1(self._up(value, skip1))
        value = self.decoder0(self._up(value, skip0))
        return self.output(value)


class EnvironmentEncoder(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
        )


class CellTokenPool(nn.Module):
    def __init__(self, point_channels: int, token_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.token_channels = int(token_channels)
        self.hidden = nn.Sequential(
            nn.Linear(point_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.token = nn.Linear(hidden_channels, token_channels)
        self.gate = nn.Sequential(nn.Linear(hidden_channels, 1), nn.Sigmoid())

    def forward(
        self,
        point_features: torch.Tensor,
        flat_indices: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.hidden(point_features)
        token = self.token(hidden)
        gate = self.gate(hidden)
        size = int(height * width)
        token_sum = token.new_zeros((size, self.token_channels))
        gate_sum = gate.new_zeros((size, 1))
        count = gate.new_zeros((size, 1))
        token_sum.index_add_(0, flat_indices, token * gate)
        gate_sum.index_add_(0, flat_indices, gate)
        count.index_add_(0, flat_indices, torch.ones_like(gate))
        pooled = token_sum / gate_sum.clamp_min(1e-6)
        pooled = pooled.T.reshape(1, self.token_channels, height, width)
        observed = (count > 0).to(pooled.dtype).T.reshape(1, 1, height, width)
        log_count = torch.log1p(count).T.reshape(1, 1, height, width) / math.log(5.0)
        return pooled, observed, log_count


class FourierFeatures(nn.Module):
    def __init__(self, bands: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(int(bands), dtype=torch.float32))

    @property
    def output_channels(self) -> int:
        return 4 * len(self.frequencies)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        angles = math.pi * xy[:, :, None] * self.frequencies[None, None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1).flatten(1)


class ResidualMLP(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(width * 2, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class StationHead(nn.Module):
    def __init__(
        self,
        width: int,
        adapter_width: int,
        spectrum_latent_dim: int,
        phase_latent_dim: int,
    ) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, adapter_width),
            nn.GELU(),
            nn.Linear(adapter_width, width),
        )
        self.spectrum = nn.Linear(width, spectrum_latent_dim)
        self.phase = nn.Linear(width, phase_latent_dim)
        self.power = nn.Linear(width, 1)
        self.outage = nn.Linear(width, 1)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        value = value + self.adapter(value)
        return {
            "spectrum": self.spectrum(value),
            "phase": self.phase(value),
            "power": self.power(value).squeeze(1),
            "outage_logit": self.outage(value).squeeze(1),
        }


def sample_map(feature_map: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    grid = coordinates.reshape(1, -1, 1, 2)
    sampled = functional.grid_sample(
        feature_map, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return sampled[0, :, :, 0].T


class StructuredContextField(nn.Module):
    def __init__(
        self,
        spectrum_latent_dim: int,
        phase_latent_dim: int,
        cell_count: int,
        static_context_channels: int,
        query_numeric_channels: int,
        token_channels: int = 128,
        token_hidden_channels: int = 192,
        context_base_channels: int = 48,
        context_feature_channels: int = 96,
        environment_feature_channels: int = 24,
        station_embedding_channels: int = 16,
        fourier_bands: int = 8,
        query_width: int = 512,
        query_blocks: int = 4,
        adapter_width: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.spectrum_latent_dim = int(spectrum_latent_dim)
        self.phase_latent_dim = int(phase_latent_dim)
        point_channels = spectrum_latent_dim + phase_latent_dim + 4
        self.pool = CellTokenPool(point_channels, token_channels, token_hidden_channels)
        self.context_fpn = GatedContextFPN(
            token_channels + static_context_channels + 2,
            context_base_channels,
            context_feature_channels,
            dropout,
        )
        self.environment_encoder = EnvironmentEncoder(6, environment_feature_channels)
        self.fourier = FourierFeatures(fourier_bands)
        self.station_embedding = nn.Embedding(cell_count, station_embedding_channels)
        query_input = (
            context_feature_channels
            + environment_feature_channels
            + query_numeric_channels
            + self.fourier.output_channels
            + station_embedding_channels
        )
        self.query_input = nn.Sequential(
            nn.Linear(query_input, query_width), nn.LayerNorm(query_width), nn.GELU()
        )
        self.query_blocks = nn.Sequential(
            *[ResidualMLP(query_width, dropout) for _ in range(int(query_blocks))]
        )
        self.station_heads = nn.ModuleList(
            [
                StationHead(
                    query_width,
                    adapter_width,
                    spectrum_latent_dim,
                    phase_latent_dim,
                )
                for _ in range(cell_count)
            ]
        )

    def forward(
        self,
        cell_id: int,
        point_features: torch.Tensor,
        point_flat_indices: torch.Tensor,
        context_static: torch.Tensor,
        environment_bev: torch.Tensor,
        query_context_coordinates: torch.Tensor,
        query_environment_coordinates: torch.Tensor,
        query_numeric: torch.Tensor,
        query_relative_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height, width = context_static.shape[-2:]
        pooled, observed, log_count = self.pool(
            point_features, point_flat_indices, height, width
        )
        context_input = torch.cat(
            [pooled, context_static.unsqueeze(0), observed, log_count], dim=1
        )
        padded, original_shape = pad_to_multiple(context_input)
        context_features = unpad(self.context_fpn(padded), original_shape)
        environment_features = self.environment_encoder(environment_bev.unsqueeze(0))
        context_query = sample_map(context_features, query_context_coordinates)
        environment_query = sample_map(
            environment_features, query_environment_coordinates
        )
        station = self.station_embedding(
            torch.full(
                (len(query_numeric),), int(cell_id), dtype=torch.long, device=query_numeric.device
            )
        )
        query = torch.cat(
            [
                context_query,
                environment_query,
                query_numeric,
                self.fourier(query_relative_xy),
                station,
            ],
            dim=1,
        )
        query = self.query_blocks(self.query_input(query))
        return self.station_heads[int(cell_id)](query)
