from __future__ import annotations

import torch
from torch import nn

from ..transforms import (
    ChannelShape,
    angle_delay_to_channel,
    normalize_angle_delay,
    scaled_angle_delay,
)
from .common import ConditionalChannelModel, ConditionalHeads, LinkContextEncoder, gather_candidate


class SparseTokenExpert(nn.Module):
    def __init__(
        self,
        context_dim: int,
        shape: ChannelShape,
        token_count: int,
        model_dim: int,
        heads: int,
        layers: int,
        feedforward_dim: int,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.token_count = token_count
        self.query = nn.Parameter(torch.randn(token_count, model_dim) * 0.02)
        self.condition = nn.Linear(context_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.parameter_head = nn.Linear(model_dim, 6 + shape.ad_channels)
        self.register_buffer("v_grid", torch.linspace(-1.0, 1.0, shape.m_v))
        self.register_buffer("h_grid", torch.linspace(-1.0, 1.0, shape.m_h))
        self.register_buffer("d_grid", torch.linspace(-1.0, 1.0, shape.s))

    @staticmethod
    def _basis(grid: torch.Tensor, center: torch.Tensor, width_raw: torch.Tensor) -> torch.Tensor:
        width = 0.025 + 0.75 * torch.sigmoid(width_raw)
        return torch.exp(-0.5 * ((grid[None, None, :] - center[..., None]) / width[..., None]) ** 2)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        conditioned = self.query[None, :, :] + self.condition(context)[:, None, :]
        parameters = self.parameter_head(self.transformer(conditioned))
        center_v, center_h, center_d = torch.tanh(parameters[..., :3]).unbind(dim=-1)
        width_v, width_h, width_d = parameters[..., 3:6].unbind(dim=-1)
        coefficient = parameters[..., 6:]
        basis_v = self._basis(self.v_grid, center_v, width_v)
        basis_h = self._basis(self.h_grid, center_h, width_h)
        basis_d = self._basis(self.d_grid, center_d, width_d)
        field = torch.einsum(
            "bkc,bkv,bkh,bks->bcvhs", coefficient, basis_v, basis_h, basis_d
        )
        return normalize_angle_delay(field)


class Scheme2Model(ConditionalChannelModel):
    def __init__(
        self,
        shape: ChannelShape,
        token_feature_dim: int,
        hidden_dim: int,
        fourier_bands: int,
        base_stations: torch.Tensor,
        position_center: torch.Tensor,
        position_scale: torch.Tensor,
        power_mean: torch.Tensor,
        power_std: torch.Tensor,
        token_count: int,
        model_dim: int,
        attention_heads: int,
        transformer_layers: int,
        feedforward_dim: int,
    ) -> None:
        super().__init__(power_mean, power_std)
        self.shape = shape
        self.context_encoder = LinkContextEncoder(
            token_feature_dim, hidden_dim, base_stations, position_center, position_scale, fourier_bands
        )
        self.heads = ConditionalHeads(hidden_dim)
        self.experts = nn.ModuleList(
            [
                SparseTokenExpert(
                    hidden_dim, shape, token_count, model_dim, attention_heads,
                    transformer_layers, feedforward_dim
                )
                for _ in range(2)
            ]
        )

    def configure_stage(self, stage: str) -> None:
        if stage != "joint":
            raise ValueError("Scheme2 supports only the joint stage")
        for parameter in self.parameters():
            parameter.requires_grad = True

    def _decode_route(self, context: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        result = context.new_zeros((context.shape[0], *self.shape.ad_shape))
        for cell, expert in enumerate(self.experts):
            indices = torch.nonzero(route == cell, as_tuple=False).flatten()
            if len(indices):
                decoded = expert(context.index_select(0, indices)[:, cell])
                result = result.index_copy(0, indices, decoded)
        return result

    def forward(
        self, positions: torch.Tensor, map_tokens: torch.Tensor, route: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(positions, map_tokens)
        outputs = self.heads(context)
        outputs["predicted_shape"] = self._decode_route(context, route)
        outputs["selected_power_z"], outputs["log_power"] = self.routed_log_power(
            outputs["power_z"], route
        )
        outputs["selected_outage_logits"] = gather_candidate(outputs["outage_logits"], route)
        return outputs

    def generate(
        self, positions: torch.Tensor, map_tokens: torch.Tensor, outage_threshold: float
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(positions, map_tokens)
        outputs = self.heads(context)
        route = outputs["gate_logits"].argmax(dim=-1)
        predicted_shape = self._decode_route(context, route)
        _, log_power = self.routed_log_power(outputs["power_z"], route)
        channel = angle_delay_to_channel(
            scaled_angle_delay(predicted_shape.float(), log_power.float()), self.shape
        )
        outage_probability = torch.sigmoid(gather_candidate(outputs["outage_logits"], route))
        channel = channel.masked_fill(
            (outage_probability >= outage_threshold)[:, None, None, None], 0.0
        )
        return {
            "channel": channel,
            "route": route,
            "outage_probability": outage_probability,
            "log_power": log_power,
        }
