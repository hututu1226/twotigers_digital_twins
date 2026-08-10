from __future__ import annotations

import torch
from torch import nn

from ..transforms import (
    ChannelShape,
    angle_delay_to_channel,
    normalize_angle_delay,
    scaled_angle_delay,
)
from .common import ConditionalChannelModel, ConditionalHeads, LinkContextEncoder, gather_candidate, mlp


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
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
        modules: list[nn.Module] = [
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
        ]
        if activate:
            modules.extend([nn.GroupNorm(_groups(output_channels), output_channels), nn.GELU()])
        super().__init__(*modules)


class AngleDelayAutoencoder(nn.Module):
    def __init__(self, shape: ChannelShape, base_channels: int, latent_dim: int) -> None:
        super().__init__()
        if shape.m_v % 8 or shape.m_h % 8 or shape.s % 8:
            raise ValueError("Angle and delay dimensions must be divisible by 8")
        self.shape = shape
        self.base_channels = base_channels
        self.compressed_shape = (4 * base_channels, shape.m_v // 8, shape.m_h // 8, shape.s // 8)
        flattened = int(torch.tensor(self.compressed_shape).prod().item())
        self.encoder_conv = nn.Sequential(
            EncoderBlock(shape.ad_channels, base_channels),
            EncoderBlock(base_channels, 2 * base_channels),
            EncoderBlock(2 * base_channels, 4 * base_channels),
        )
        self.to_latent = nn.Linear(flattened, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flattened)
        self.decoder_conv = nn.Sequential(
            DecoderBlock(4 * base_channels, 2 * base_channels),
            DecoderBlock(2 * base_channels, base_channels),
            DecoderBlock(base_channels, shape.ad_channels, activate=False),
        )

    def encode(self, angle_delay: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder_conv(angle_delay).flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        feature = self.from_latent(latent).reshape(-1, *self.compressed_shape)
        return normalize_angle_delay(self.decoder_conv(feature))

    def forward(self, angle_delay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(angle_delay)
        return self.decode(latent), latent


class Scheme1Model(ConditionalChannelModel):
    def __init__(
        self,
        shape: ChannelShape,
        token_feature_dim: int,
        hidden_dim: int,
        latent_dim: int,
        base_channels: int,
        fourier_bands: int,
        base_stations: torch.Tensor,
        position_center: torch.Tensor,
        position_scale: torch.Tensor,
        power_mean: torch.Tensor,
        power_std: torch.Tensor,
    ) -> None:
        super().__init__(power_mean, power_std)
        self.shape = shape
        self.context_encoder = LinkContextEncoder(
            token_feature_dim, hidden_dim, base_stations, position_center, position_scale, fourier_bands
        )
        self.heads = ConditionalHeads(hidden_dim)
        self.experts = nn.ModuleList(
            [mlp(hidden_dim, hidden_dim, latent_dim, layers=3) for _ in range(2)]
        )
        self.autoencoder = AngleDelayAutoencoder(shape, base_channels, latent_dim)

    def configure_stage(self, stage: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = stage == "joint"
        if stage == "autoencoder":
            for parameter in self.autoencoder.parameters():
                parameter.requires_grad = True
        elif stage == "predictor":
            for module in [self.context_encoder, self.heads, self.experts]:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        elif stage != "joint":
            raise ValueError(f"Unsupported scheme1 stage: {stage}")

    def forward(
        self,
        positions: torch.Tensor,
        map_tokens: torch.Tensor,
        route: torch.Tensor,
        target_shape: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(positions, map_tokens)
        outputs = self.heads(context)
        candidate_latent = torch.stack(
            [expert(context[:, index]) for index, expert in enumerate(self.experts)], dim=1
        )
        predicted_latent = gather_candidate(candidate_latent, route)
        outputs["predicted_latent"] = predicted_latent
        outputs["predicted_shape"] = self.autoencoder.decode(predicted_latent)
        outputs["selected_power_z"], outputs["log_power"] = self.routed_log_power(
            outputs["power_z"], route
        )
        outputs["selected_outage_logits"] = gather_candidate(outputs["outage_logits"], route)
        if target_shape is not None:
            outputs["target_latent"] = self.autoencoder.encode(target_shape).detach()
        return outputs

    def generate(
        self, positions: torch.Tensor, map_tokens: torch.Tensor, outage_threshold: float
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(positions, map_tokens)
        outputs = self.heads(context)
        route = outputs["gate_logits"].argmax(dim=-1)
        candidate_latent = torch.stack(
            [expert(context[:, index]) for index, expert in enumerate(self.experts)], dim=1
        )
        latent = gather_candidate(candidate_latent, route)
        predicted_shape = self.autoencoder.decode(latent)
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

