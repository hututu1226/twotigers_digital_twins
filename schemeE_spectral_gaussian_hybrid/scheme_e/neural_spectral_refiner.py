from __future__ import annotations

import torch
from torch import nn


class SpectralNeighborRefiner(nn.Module):
    """Refine a coarse spectral prior from nearby observed spectral tokens."""

    def __init__(
        self,
        latent_dim: int,
        pas_dim: int,
        query_feature_dim: int,
        neighbor_feature_dim: int,
        width: int = 256,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.08,
        maximum_residual: float = 3.0,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.latent_dim = int(latent_dim)
        self.pas_dim = int(pas_dim)
        if not 0 < self.pas_dim < self.latent_dim:
            raise ValueError("pas_dim must split the latent into PAS and PDP sections")
        self.maximum_residual = float(maximum_residual)
        self.query_projection = nn.Sequential(
            nn.Linear(latent_dim + query_feature_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.neighbor_projection = nn.Sequential(
            nn.Linear(latent_dim + neighbor_feature_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.query_type = nn.Parameter(torch.zeros(1, 1, width))
        self.neighbor_type = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, latent_dim)
        self.gate = nn.Sequential(
            nn.Linear(width, max(width // 2, 16)),
            nn.GELU(),
            nn.Linear(max(width // 2, 16), 2),
        )
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.4)

    def forward(
        self,
        base_latent: torch.Tensor,
        query_features: torch.Tensor,
        neighbor_latents: torch.Tensor,
        neighbor_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if base_latent.ndim != 2 or base_latent.shape[1] != self.latent_dim:
            raise ValueError("base_latent has the wrong shape")
        if neighbor_latents.ndim != 3 or neighbor_latents.shape[2] != self.latent_dim:
            raise ValueError("neighbor_latents has the wrong shape")
        if neighbor_features.shape[:2] != neighbor_latents.shape[:2]:
            raise ValueError("neighbor features and latents must share batch/token axes")
        query = self.query_projection(
            torch.cat([base_latent, query_features], dim=1)
        )[:, None]
        neighbors = self.neighbor_projection(
            torch.cat([neighbor_latents, neighbor_features], dim=2)
        )
        tokens = torch.cat(
            [query + self.query_type, neighbors + self.neighbor_type], dim=1
        )
        summary = self.output_norm(self.encoder(tokens)[:, 0])
        residual = self.maximum_residual * torch.tanh(self.output(summary))
        gates = torch.sigmoid(self.gate(summary))
        gated_residual = torch.cat(
            [
                gates[:, :1] * residual[:, : self.pas_dim],
                gates[:, 1:] * residual[:, self.pas_dim :],
            ],
            dim=1,
        )
        return {
            "latent": base_latent + gated_residual,
            "residual": gated_residual,
            "pas_gate": gates[:, 0],
            "pdp_gate": gates[:, 1],
        }
