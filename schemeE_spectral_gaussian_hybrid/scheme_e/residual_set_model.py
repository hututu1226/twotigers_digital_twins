from __future__ import annotations

import math

import torch
from torch import nn


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualCoefficientSetEncoder(nn.Module):
    """Predict a low-rank correction while retaining the full seed latent."""

    def __init__(
        self,
        spectrum_shape: tuple[int, int, int, int],
        query_dim: int,
        neighbor_dim: int,
        coefficient_dim: int,
        width: int = 192,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        channels, depth, height, delay = (int(value) for value in spectrum_shape)
        hidden_channels = max(32, width // 4)
        self.spectrum_shape = (channels, depth, height, delay)
        self.spectrum = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(hidden_channels * depth * height * delay, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.query = nn.Sequential(
            nn.Linear(int(query_dim), width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.neighbor = nn.Sequential(
            nn.Linear(int(neighbor_dim), width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.query_key = nn.Linear(width, width, bias=False)
        self.neighbor_key = nn.Linear(width, width, bias=False)
        self.distance_bias = nn.Sequential(
            nn.Linear(1, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(width * 4, width * 2),
            nn.LayerNorm(width * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, int(coefficient_dim)),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        spectrum_latent: torch.Tensor,
        query_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_distance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tuple(spectrum_latent.shape[1:]) != self.spectrum_shape:
            raise ValueError(
                f"Expected spectrum latent {self.spectrum_shape}, got "
                f"{tuple(spectrum_latent.shape[1:])}"
            )
        if neighbor_features.ndim != 3:
            raise ValueError("neighbor_features must be [batch, neighbors, features]")
        spectrum = self.spectrum(spectrum_latent.float())
        query = self.query(query_features.float())
        neighbors = self.neighbor(neighbor_features.float())
        logits = (
            self.neighbor_key(neighbors)
            * self.query_key(query)[:, None]
        ).sum(dim=-1) / math.sqrt(neighbors.shape[-1])
        logits = logits + self.distance_bias(neighbor_distance.float()).squeeze(-1)
        attention = torch.softmax(logits, dim=1)
        attended = torch.sum(attention[:, :, None] * neighbors, dim=1)
        mean = neighbors.mean(dim=1)
        maximum = neighbors.amax(dim=1)
        coefficients = self.output(
            torch.cat([spectrum + query, attended, mean, maximum], dim=1)
        )
        effective_neighbors = 1.0 / attention.square().sum(dim=1).clamp_min(1e-8)
        return {
            "coefficients": coefficients,
            "attention": attention,
            "effective_neighbors": effective_neighbors,
        }
