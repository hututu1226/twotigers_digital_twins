from __future__ import annotations

import math

import torch
from torch import nn


def mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(max(1, layers - 1)):
        modules.extend([nn.Linear(current, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)])
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


def gather_candidate(values: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, route]


class LinkContextEncoder(nn.Module):
    def __init__(
        self,
        token_feature_dim: int,
        hidden_dim: int,
        base_stations: torch.Tensor,
        position_center: torch.Tensor,
        position_scale: torch.Tensor,
        fourier_bands: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fourier_bands = fourier_bands
        self.register_buffer("base_stations", base_stations.float())
        self.register_buffer("position_center", position_center.float())
        self.register_buffer("position_scale", position_scale.float().clamp_min(1.0))
        raw_coordinate_dim = 6
        coordinate_dim = raw_coordinate_dim * (1 + 2 * fourier_bands)
        self.coordinate_encoder = mlp(coordinate_dim, hidden_dim, hidden_dim, layers=3)
        self.token_encoder = mlp(token_feature_dim, hidden_dim, hidden_dim, layers=3)
        self.output = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

    def _fourier(self, value: torch.Tensor) -> torch.Tensor:
        features = [value]
        for band in range(self.fourier_bands):
            frequency = (2.0**band) * math.pi
            features.extend([torch.sin(frequency * value), torch.cos(frequency * value)])
        return torch.cat(features, dim=-1)

    def forward(self, positions: torch.Tensor, map_tokens: torch.Tensor) -> torch.Tensor:
        batch = positions.shape[0]
        normalized = (positions - self.position_center) / self.position_scale
        relative = (positions[:, None, :] - self.base_stations[None, :, :]) / self.position_scale
        absolute = normalized[:, None, :].expand(batch, 2, 3)
        coordinate = torch.cat([absolute, relative], dim=-1)
        query = self.coordinate_encoder(self._fourier(coordinate))
        token_features = self.token_encoder(map_tokens)
        attention = (token_features * query[:, :, None, :]).sum(dim=-1) / math.sqrt(self.hidden_dim)
        attention = torch.softmax(attention, dim=2)
        pooled = (attention[..., None] * token_features).sum(dim=2)
        return self.output(torch.cat([query, pooled], dim=-1))


class ConditionalHeads(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = mlp(2 * hidden_dim, hidden_dim, 2, layers=2)
        self.outage = mlp(hidden_dim, hidden_dim, 1, layers=2)
        self.power = mlp(hidden_dim, hidden_dim, 1, layers=2)

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "gate_logits": self.gate(context.reshape(context.shape[0], -1)),
            "outage_logits": self.outage(context).squeeze(-1),
            "power_z": self.power(context).squeeze(-1),
        }


class ConditionalChannelModel(nn.Module):
    def __init__(self, power_mean: torch.Tensor, power_std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("power_mean", power_mean.float())
        self.register_buffer("power_std", power_std.float().clamp_min(0.1))

    def routed_log_power(
        self, candidate_power_z: torch.Tensor, route: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_z = gather_candidate(candidate_power_z, route)
        log_power = self.power_mean[route] + self.power_std[route] * selected_z
        return selected_z, log_power

