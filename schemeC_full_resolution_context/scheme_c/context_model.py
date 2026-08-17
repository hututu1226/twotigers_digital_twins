from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    for count in (8, 4, 2):
        if channels % count == 0:
            return count
    return 1


def pad_to_multiple(
    value: torch.Tensor, multiple: int = 8
) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = value.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    return functional.pad(value, (0, pad_width, 0, pad_height)), (height, width)


def unpad(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return value[..., : shape[0], : shape[1]]


class GatedConv2d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.feature = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.gate = nn.Conv2d(input_channels, output_channels, 3, padding=1)
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
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.dropout(self.first(value))) + self.skip(value)


class GatedContextFPN(nn.Module):
    def __init__(
        self, input_channels: int, base_channels: int, output_channels: int, dropout: float
    ) -> None:
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
        self.output = nn.Conv2d(widths[0], output_channels, 1)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = functional.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
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


class CellTokenPool(nn.Module):
    """Scatter only geometry/power summaries; channel latent stays full resolution."""

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
        pooled = (token_sum / gate_sum.clamp_min(1e-6)).T.reshape(
            1, self.token_channels, height, width
        )
        observed = (count > 0).to(pooled.dtype).T.reshape(1, 1, height, width)
        log_count = torch.log1p(count).T.reshape(1, 1, height, width) / math.log(5.0)
        return pooled, observed, log_count


class FourierFeatures(nn.Module):
    def __init__(self, bands: int) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(int(bands), dtype=torch.float32)
        )

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


class MultiScaleEnvironmentEncoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        def block(input_width: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    input_width,
                    output_channels,
                    3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(_groups(output_channels), output_channels),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(output_channels), output_channels),
                nn.GELU(),
            )

        self.level0 = block(input_channels, 1)
        self.level1 = block(output_channels, 2)
        self.level2 = block(output_channels, 2)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        level0 = self.level0(value)
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        return level0, level1, level2


def sample_map(feature_map: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    grid = coordinates.reshape(1, -1, 1, 2)
    sampled = functional.grid_sample(
        feature_map, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return sampled[0, :, :, 0].T


def sample_pyramid(
    feature_maps: tuple[torch.Tensor, ...], coordinates: torch.Tensor
) -> torch.Tensor:
    return torch.cat([sample_map(level, coordinates) for level in feature_maps], dim=1)


def sample_corridor(feature_map: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    sampled = functional.grid_sample(
        feature_map,
        coordinates.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0].permute(1, 2, 0)
    return torch.cat([sampled.mean(dim=1), sampled.amax(dim=1)], dim=1)


class FullResolutionResidual3d(nn.Module):
    def __init__(self, channels: int, dropout: float, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(channels * 2, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mixed = self.channel_mixer(self.depthwise(self.norm(value)))
        return value + mixed


class FullResolutionLatentCrossAttention(nn.Module):
    """Predict every latent bin from all observed users without a flat bottleneck."""

    def __init__(
        self,
        latent_shape: tuple[int, int, int, int],
        observed_context_channels: int,
        target_context_channels: int,
        token_channels: int,
        attention_heads: int,
        attention_chunk_size: int,
        refinement_blocks: int,
        dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        latent_channels, angle_v, angle_h, delay = latent_shape
        if token_channels % attention_heads:
            raise ValueError("token_channels must be divisible by attention_heads")
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.latent_channels = int(latent_channels)
        self.token_channels = int(token_channels)
        self.attention_heads = int(attention_heads)
        self.head_channels = self.token_channels // self.attention_heads
        self.attention_chunk_size = int(attention_chunk_size)
        self.dropout = float(dropout)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.angle_v_embedding = nn.Parameter(torch.empty(angle_v, token_channels))
        self.angle_h_embedding = nn.Parameter(torch.empty(angle_h, token_channels))
        self.delay_embedding = nn.Parameter(torch.empty(delay, token_channels))
        for embedding in (
            self.angle_v_embedding,
            self.angle_h_embedding,
            self.delay_embedding,
        ):
            nn.init.trunc_normal_(embedding, std=0.02)

        self.latent_projection = nn.Linear(latent_channels, token_channels)
        self.observed_projection = nn.Linear(observed_context_channels, token_channels)
        self.target_projection = nn.Linear(target_context_channels, token_channels)
        self.query = nn.Linear(token_channels, token_channels)
        self.key = nn.Linear(token_channels, token_channels)
        self.value = nn.Linear(token_channels, token_channels)
        self.attention_output = nn.Linear(token_channels, token_channels)
        relation_width = max(16, attention_heads * 4)
        self.relation_bias = nn.Sequential(
            nn.Linear(6, relation_width),
            nn.GELU(),
            nn.Linear(relation_width, attention_heads),
        )
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, token_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(token_channels * 2, token_channels),
        )
        self.refiner = nn.Sequential(
            *[
                FullResolutionResidual3d(
                    token_channels, dropout, dilation=1 if index % 2 == 0 else 2
                )
                for index in range(int(refinement_blocks))
            ]
        )
        self.output = nn.Conv3d(token_channels, latent_channels, 1)

    @property
    def position_count(self) -> int:
        return int(math.prod(self.latent_shape[1:]))

    def _position_embedding(self) -> torch.Tensor:
        return (
            self.angle_v_embedding[:, None, None, :]
            + self.angle_h_embedding[None, :, None, :]
            + self.delay_embedding[None, None, :, :]
        ).reshape(self.position_count, self.token_channels)

    def _attention_chunk(
        self,
        latent: torch.Tensor,
        position: torch.Tensor,
        observed_context: torch.Tensor,
        target_context: torch.Tensor,
        relation_bias: torch.Tensor,
    ) -> torch.Tensor:
        bins, observations = latent.shape[:2]
        targets = target_context.shape[0]
        observed = (
            self.latent_projection(latent)
            + self.observed_projection(observed_context)[None, :, :]
            + position[:, None, :]
        )
        target = self.target_projection(target_context)[None, :, :] + position[:, None, :]
        query = self.query(target).reshape(
            bins, targets, self.attention_heads, self.head_channels
        ).permute(0, 2, 1, 3)
        key = self.key(observed).reshape(
            bins, observations, self.attention_heads, self.head_channels
        ).permute(0, 2, 1, 3)
        value = self.value(observed).reshape(
            bins, observations, self.attention_heads, self.head_channels
        ).permute(0, 2, 1, 3)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=relation_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(
            bins, targets, self.token_channels
        )
        output = target + self.attention_output(attended)
        return output + self.token_mlp(output)

    def forward(
        self,
        observed_latent: torch.Tensor,
        observed_context: torch.Tensor,
        target_context: torch.Tensor,
        observed_relative_xy: torch.Tensor,
        target_relative_xy: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(observed_latent.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Expected observed latent {self.latent_shape}, got {tuple(observed_latent.shape[1:])}"
            )
        delta = target_relative_xy[:, None, :] - observed_relative_xy[None, :, :]
        relation = torch.cat(
            [
                delta,
                delta.abs(),
                torch.linalg.vector_norm(delta, dim=2, keepdim=True),
                (target_relative_xy[:, None, :] * observed_relative_xy[None, :, :]).sum(
                    dim=2, keepdim=True
                ),
            ],
            dim=2,
        )
        relation_bias = self.relation_bias(relation).permute(2, 0, 1).unsqueeze(0)
        latent = observed_latent.flatten(2).permute(2, 0, 1)
        positions = self._position_embedding()
        chunks: list[torch.Tensor] = []
        for start in range(0, self.position_count, self.attention_chunk_size):
            stop = min(start + self.attention_chunk_size, self.position_count)
            arguments = (
                latent[start:stop],
                positions[start:stop],
                observed_context,
                target_context,
                relation_bias,
            )
            if self.gradient_checkpointing and self.training:
                chunk = checkpoint(
                    self._attention_chunk, *arguments, use_reentrant=False
                )
            else:
                chunk = self._attention_chunk(*arguments)
            chunks.append(chunk)
        features = torch.cat(chunks, dim=0).permute(1, 2, 0).reshape(
            len(target_context), self.token_channels, *self.latent_shape[1:]
        )
        if condition is not None:
            if condition.shape != features.shape:
                raise ValueError(
                    f"Condition shape {tuple(condition.shape)} does not match {tuple(features.shape)}"
                )
            features = features + condition
        features = self.refiner(features)
        return self.output(features), features


class FullResolutionContextField(nn.Module):
    """Scheme C field with full-resolution spectrum and detail token branches."""

    def __init__(
        self,
        spectrum_shape: tuple[int, int, int, int],
        phase_shape: tuple[int, int, int, int],
        cell_count: int,
        static_context_channels: int,
        query_numeric_channels: int,
        map_token_channels: int = 32,
        map_hidden_channels: int = 64,
        context_base_channels: int = 32,
        context_feature_channels: int = 64,
        environment_feature_channels: int = 24,
        station_embedding_channels: int = 16,
        fourier_bands: int = 8,
        global_width: int = 256,
        global_blocks: int = 3,
        spectrum_token_channels: int = 64,
        detail_token_channels: int = 48,
        attention_heads: int = 4,
        attention_chunk_size: int = 16,
        refinement_blocks: int = 4,
        dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.spectrum_shape = tuple(int(value) for value in spectrum_shape)
        self.phase_shape = tuple(int(value) for value in phase_shape)
        self.spectrum_latent_dim = int(math.prod(self.spectrum_shape))
        self.phase_latent_dim = int(math.prod(self.phase_shape))
        self.pool = CellTokenPool(4, map_token_channels, map_hidden_channels)
        self.context_fpn = GatedContextFPN(
            map_token_channels + static_context_channels + 2,
            context_base_channels,
            context_feature_channels,
            dropout,
        )
        self.environment_encoder = MultiScaleEnvironmentEncoder(
            6, environment_feature_channels
        )
        self.fourier = FourierFeatures(fourier_bands)
        self.station_embedding = nn.Embedding(cell_count, station_embedding_channels)
        local_environment_channels = 3 * environment_feature_channels
        corridor_channels = 2 * environment_feature_channels
        target_input_channels = (
            context_feature_channels
            + local_environment_channels
            + corridor_channels
            + query_numeric_channels
            + self.fourier.output_channels
            + station_embedding_channels
        )
        observed_input_channels = (
            context_feature_channels
            + local_environment_channels
            + query_numeric_channels
            + self.fourier.output_channels
            + station_embedding_channels
            + 2
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(target_input_channels, global_width),
            nn.LayerNorm(global_width),
            nn.GELU(),
            *[ResidualMLP(global_width, dropout) for _ in range(int(global_blocks))],
        )
        self.observed_encoder = nn.Sequential(
            nn.Linear(observed_input_channels, global_width),
            nn.LayerNorm(global_width),
            nn.GELU(),
            *[ResidualMLP(global_width, dropout) for _ in range(int(global_blocks))],
        )
        branch_arguments = {
            "observed_context_channels": global_width,
            "target_context_channels": global_width,
            "attention_heads": attention_heads,
            "attention_chunk_size": attention_chunk_size,
            "refinement_blocks": refinement_blocks,
            "dropout": dropout,
            "gradient_checkpointing": gradient_checkpointing,
        }
        self.spectrum_field = FullResolutionLatentCrossAttention(
            self.spectrum_shape,
            token_channels=spectrum_token_channels,
            **branch_arguments,
        )
        self.detail_field = FullResolutionLatentCrossAttention(
            self.phase_shape,
            token_channels=detail_token_channels,
            **branch_arguments,
        )
        self.spectrum_to_detail = nn.Conv3d(
            spectrum_token_channels, detail_token_channels, 1
        )
        self.detail_confidence = nn.Conv3d(detail_token_channels, 1, 1)
        output_input = global_width + spectrum_token_channels + detail_token_channels
        output_hidden = max(64, global_width // 2)
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(output_input),
            nn.Linear(output_input, output_hidden),
            nn.GELU(),
            nn.Linear(output_hidden, 2),
        )

    def forward(
        self,
        cell_id: int,
        observed_spectrum: torch.Tensor,
        observed_phase: torch.Tensor,
        observed_power: torch.Tensor,
        observed_outage: torch.Tensor,
        point_features: torch.Tensor,
        point_flat_indices: torch.Tensor,
        context_static: torch.Tensor,
        environment_bev: torch.Tensor,
        observed_context_coordinates: torch.Tensor,
        observed_environment_coordinates: torch.Tensor,
        observed_numeric: torch.Tensor,
        observed_relative_xy: torch.Tensor,
        query_context_coordinates: torch.Tensor,
        query_environment_coordinates: torch.Tensor,
        query_corridor_coordinates: torch.Tensor,
        query_numeric: torch.Tensor,
        query_relative_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height, width = context_static.shape[-2:]
        pooled, observed_mask, log_count = self.pool(
            point_features, point_flat_indices, height, width
        )
        context_input = torch.cat(
            [pooled, context_static.unsqueeze(0), observed_mask, log_count], dim=1
        )
        padded, original_shape = pad_to_multiple(context_input)
        context_features = unpad(self.context_fpn(padded), original_shape)
        environment_features = self.environment_encoder(environment_bev.unsqueeze(0))

        query_context = sample_map(context_features, query_context_coordinates)
        query_environment = sample_pyramid(
            environment_features, query_environment_coordinates
        )
        corridor = sample_corridor(
            environment_features[0], query_corridor_coordinates
        )
        observed_context = sample_map(
            context_features, observed_context_coordinates
        )
        observed_environment = sample_pyramid(
            environment_features, observed_environment_coordinates
        )
        station_query = self.station_embedding(
            torch.full(
                (len(query_numeric),),
                int(cell_id),
                dtype=torch.long,
                device=query_numeric.device,
            )
        )
        station_observed = self.station_embedding(
            torch.full(
                (len(observed_numeric),),
                int(cell_id),
                dtype=torch.long,
                device=observed_numeric.device,
            )
        )
        query_global = self.target_encoder(
            torch.cat(
                [
                    query_context,
                    query_environment,
                    corridor,
                    query_numeric,
                    self.fourier(query_relative_xy),
                    station_query,
                ],
                dim=1,
            )
        )
        observed_global = self.observed_encoder(
            torch.cat(
                [
                    observed_context,
                    observed_environment,
                    observed_numeric,
                    self.fourier(observed_relative_xy),
                    station_observed,
                    observed_power[:, None],
                    observed_outage[:, None],
                ],
                dim=1,
            )
        )
        spectrum, spectrum_features = self.spectrum_field(
            observed_spectrum,
            observed_global,
            query_global,
            observed_relative_xy,
            query_relative_xy,
        )
        detail_condition = functional.interpolate(
            self.spectrum_to_detail(spectrum_features),
            size=self.phase_shape[1:],
            mode="trilinear",
            align_corners=False,
        )
        phase, detail_features = self.detail_field(
            observed_phase,
            observed_global,
            query_global,
            observed_relative_xy,
            query_relative_xy,
            condition=detail_condition,
        )
        confidence = torch.sigmoid(self.detail_confidence(detail_features))
        phase = phase * confidence
        summary = torch.cat(
            [
                query_global,
                spectrum_features.mean(dim=(2, 3, 4)),
                detail_features.mean(dim=(2, 3, 4)),
            ],
            dim=1,
        )
        scalars = self.scalar_head(summary)
        return {
            "spectrum": spectrum.flatten(1),
            "phase": phase.flatten(1),
            "power": scalars[:, 0],
            "outage_logit": scalars[:, 1],
            "detail_confidence": confidence.mean(dim=(1, 2, 3, 4)),
        }
