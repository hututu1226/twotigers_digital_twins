from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as functional

from .magnitude_refiner import LocalMagnitudeBlock3d


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if int(channels) % count == 0:
            return count
    return 1


class QueryConditionedLocalSetMagnitudeRefiner(nn.Module):
    """Fuse full-resolution local residual maps without a spectral bottleneck."""

    def __init__(
        self,
        input_channels: int,
        geometry_dim: int,
        relative_dim: int,
        cell_count: int,
        width: int = 16,
        blocks: int = 3,
        dropout: float = 0.02,
        maximum_residual: float = 4.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.relative_dim = int(relative_dim)
        self.width = int(width)
        self.maximum_residual = float(maximum_residual)
        self.query_stem = nn.Conv3d(self.input_channels + 3, self.width, 1)
        self.neighbor_stem = nn.Conv3d(self.input_channels * 2, self.width, 1)
        self.geometry = nn.Sequential(
            nn.Linear(int(geometry_dim), self.width),
            nn.LayerNorm(self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width * 2),
        )
        self.station = nn.Embedding(int(cell_count), self.width * 2)
        self.relative = nn.Sequential(
            nn.Linear(self.relative_dim, self.width),
            nn.LayerNorm(self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width * 2 + 1),
        )
        self.query_attention = nn.Conv3d(self.width, self.width, 1, bias=False)
        self.neighbor_attention = nn.Conv3d(self.width, self.width, 1, bias=False)
        dilations = (1, 2, 4, 8)
        self.blocks = nn.Sequential(
            *[
                LocalMagnitudeBlock3d(
                    self.width,
                    dilations[index % len(dilations)],
                    float(dropout),
                )
                for index in range(int(blocks))
            ]
        )
        self.output_norm = nn.GroupNorm(_groups(self.width), self.width)
        self.output = nn.Conv3d(self.width, self.input_channels, 1)
        self.transfer_scale = nn.Parameter(
            torch.zeros(1, self.input_channels, 1, 1, 1)
        )
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
        vertical_axis = torch.linspace(
            -1.0, 1.0, vertical, device=value.device, dtype=value.dtype
        )
        horizontal_axis = torch.linspace(
            -1.0, 1.0, horizontal, device=value.device, dtype=value.dtype
        )
        delay_axis = torch.arange(
            int(delay_start),
            int(delay_start) + delay,
            device=value.device,
            dtype=value.dtype,
        )
        if int(total_delay) > 1:
            delay_axis = 2.0 * delay_axis / float(int(total_delay) - 1) - 1.0
        else:
            delay_axis.zero_()
        vertical_grid, horizontal_grid, delay_grid = torch.meshgrid(
            vertical_axis, horizontal_axis, delay_axis, indexing="ij"
        )
        return torch.stack(
            [vertical_grid, horizontal_grid, delay_grid], dim=0
        )[None].expand(len(value), -1, -1, -1, -1)

    def forward(
        self,
        base_log_power: torch.Tensor,
        neighbor_residual: torch.Tensor,
        neighbor_base_delta: torch.Tensor,
        relative_features: torch.Tensor,
        geometry: torch.Tensor,
        cell_ids: torch.Tensor,
        *,
        delay_start: int = 0,
        total_delay: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if base_log_power.ndim != 5:
            raise ValueError("base_log_power must be [B,C,V,H,S]")
        if neighbor_residual.ndim != 6:
            raise ValueError("neighbor_residual must be [B,K,C,V,H,S]")
        if neighbor_residual.shape != neighbor_base_delta.shape:
            raise ValueError("neighbor residual and base delta shapes must match")
        batch, neighbors = neighbor_residual.shape[:2]
        if batch != len(base_log_power):
            raise ValueError("query and neighbor batch sizes differ")
        if neighbor_residual.shape[2:] != base_log_power.shape[1:]:
            raise ValueError("query and neighbor grid shapes differ")
        if relative_features.shape != (batch, neighbors, self.relative_dim):
            raise ValueError("relative_features has the wrong shape")
        if base_log_power.shape[1] != self.input_channels:
            raise ValueError("base_log_power has the wrong channel count")
        if total_delay is None:
            total_delay = int(base_log_power.shape[-1])

        base = base_log_power.float()
        coordinates = self._coordinates(base, int(delay_start), int(total_delay))
        query = self.query_stem(torch.cat([base / 4.0, coordinates], dim=1))
        query_scale, query_bias = (
            self.geometry(geometry.float()) + self.station(cell_ids.long())
        ).chunk(2, dim=1)
        query = query * (1.0 + torch.tanh(query_scale)[:, :, None, None, None])
        query = query + query_bias[:, :, None, None, None]

        neighbor_input = torch.cat(
            [
                neighbor_residual.float() / self.maximum_residual,
                neighbor_base_delta.float() / 4.0,
            ],
            dim=2,
        ).flatten(0, 1)
        neighbor = self.neighbor_stem(neighbor_input).reshape(
            batch,
            neighbors,
            self.width,
            *base.shape[2:],
        )
        relative = self.relative(relative_features.float())
        neighbor_scale, neighbor_bias, scalar_logit = torch.split(
            relative, [self.width, self.width, 1], dim=2
        )
        neighbor = neighbor * (
            1.0
            + torch.tanh(neighbor_scale)[..., None, None, None]
        )
        neighbor = neighbor + neighbor_bias[..., None, None, None]

        query_attention = functional.normalize(
            self.query_attention(query), dim=1, eps=1e-6
        )
        neighbor_attention = functional.normalize(
            self.neighbor_attention(neighbor.flatten(0, 1)).reshape_as(neighbor),
            dim=2,
            eps=1e-6,
        )
        logits = (
            query_attention[:, None] * neighbor_attention
        ).sum(dim=2) / math.sqrt(float(self.width))
        logits = logits + scalar_logit[..., None, None]
        weights = torch.softmax(logits, dim=1)
        aggregated_feature = torch.sum(neighbor * weights[:, :, None], dim=1)
        aggregated_residual = torch.sum(
            neighbor_residual.float() * weights[:, :, None], dim=1
        )

        value = self.blocks(query + aggregated_feature)
        raw = self.output(functional.gelu(self.output_norm(value)))
        transfer = self.transfer_scale * (
            aggregated_residual / self.maximum_residual
        )
        correction = self.maximum_residual * torch.tanh(raw + transfer)
        corrected = (base + correction).clamp(0.0, 20.0)
        return {
            "log_power": corrected,
            "correction": correction,
            "attention": weights,
        }
