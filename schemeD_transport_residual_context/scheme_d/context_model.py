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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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


class ConvNeXt2dBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 7, padding=3, groups=channels, bias=False
        )
        self.norm = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels * 4, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.mlp(self.norm(self.depthwise(value)))


class EnvironmentFeaturePyramid(nn.Module):
    """A trainable 1 m BEV backbone with a top-down feature pyramid."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        output_channels: int,
        blocks_per_level: int,
        dropout: float,
    ) -> None:
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 4)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, widths[0], 5, padding=2, bias=False),
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        ConvNeXt2dBlock(width, dropout)
                        for _ in range(int(blocks_per_level))
                    ]
                )
                for width in widths
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(widths[index], widths[index + 1], 3, stride=2, padding=1),
                    nn.GroupNorm(_groups(widths[index + 1]), widths[index + 1]),
                    nn.GELU(),
                )
                for index in range(2)
            ]
        )
        self.lateral = nn.ModuleList(
            [nn.Conv2d(width, output_channels, 1) for width in widths]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(_groups(output_channels), output_channels),
                    nn.GELU(),
                )
                for _ in widths
            ]
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        levels: list[torch.Tensor] = []
        value = self.stem(value)
        for index, stage in enumerate(self.stages):
            value = stage(value)
            levels.append(value)
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        pyramid: list[torch.Tensor] = [self.lateral[index](level) for index, level in enumerate(levels)]
        for index in range(len(pyramid) - 2, -1, -1):
            pyramid[index] = pyramid[index] + functional.interpolate(
                pyramid[index + 1],
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return tuple(self.smooth[index](level) for index, level in enumerate(pyramid))


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


def sample_corridor_sequence(
    feature_map: torch.Tensor, coordinates: torch.Tensor
) -> torch.Tensor:
    sampled = functional.grid_sample(
        feature_map,
        coordinates.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0].permute(1, 2, 0)
    return sampled


class CorridorTransformer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        width: int,
        heads: int,
        layers: int,
        maximum_samples: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("corridor width must be divisible by its attention heads")
        self.maximum_samples = int(maximum_samples)
        self.input = nn.Linear(input_channels, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.empty(1, self.maximum_samples + 1, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.output = nn.LayerNorm(width)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        samples = sequence.shape[1]
        if samples > self.maximum_samples:
            raise ValueError(
                f"corridor has {samples} samples, maximum is {self.maximum_samples}"
            )
        value = self.input(sequence)
        cls = self.cls.expand(len(value), -1, -1)
        value = torch.cat([cls, value], dim=1)
        value = value + self.position[:, : samples + 1]
        return self.output(self.encoder(value)[:, 0])


def pair_relation_features(
    target_relative_xy: torch.Tensor, observed_relative_xy: torch.Tensor
) -> torch.Tensor:
    target = target_relative_xy[:, None, :]
    observed = observed_relative_xy[None, :, :]
    delta = target - observed
    target_radius = torch.linalg.vector_norm(target, dim=2, keepdim=True)
    observed_radius = torch.linalg.vector_norm(observed, dim=2, keepdim=True)
    dot = (target * observed).sum(dim=2, keepdim=True)
    cross = target[..., :1] * observed[..., 1:] - target[..., 1:] * observed[..., :1]
    denominator = (target_radius * observed_radius).clamp_min(1e-6)
    return torch.cat(
        [
            delta,
            delta.abs(),
            torch.linalg.vector_norm(delta, dim=2, keepdim=True),
            target_radius - observed_radius,
            dot,
            cross,
            dot / denominator,
            cross / denominator,
        ],
        dim=2,
    )


class ObservationRouter(nn.Module):
    """Learn which observations are useful without hand-written distance weights."""

    relation_channels = 10

    def __init__(
        self,
        context_channels: int,
        router_width: int,
        pair_width: int,
        top_k: int,
        dropout: float,
        temperature: float = 1.5,
        uniform_mix: float = 0.15,
        route_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.uniform_mix = float(uniform_mix)
        self.route_dropout = float(route_dropout)
        if self.temperature <= 0.0:
            raise ValueError("router temperature must be positive")
        if not 0.0 <= self.uniform_mix < 1.0:
            raise ValueError("router uniform_mix must lie in [0,1)")
        self.query = nn.Linear(context_channels, router_width)
        self.key = nn.Linear(context_channels, router_width)
        self.relation_score = nn.Sequential(
            nn.Linear(self.relation_channels, router_width),
            nn.GELU(),
            nn.Linear(router_width, 1),
        )
        self.pair = nn.Sequential(
            nn.Linear(context_channels * 2 + self.relation_channels, pair_width),
            nn.LayerNorm(pair_width),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(pair_width, pair_width),
            nn.GELU(),
        )
        self.scale = router_width**-0.5

    def forward(
        self,
        observed_context: torch.Tensor,
        target_context: torch.Tensor,
        observed_relative_xy: torch.Tensor,
        target_relative_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        relation = pair_relation_features(target_relative_xy, observed_relative_xy)
        score = (
            self.query(target_context)[:, None, :]
            * self.key(observed_context)[None, :, :]
        ).sum(dim=2) * self.scale
        score = score + self.relation_score(relation).squeeze(2)
        count = min(max(self.top_k, 1), observed_context.shape[0])
        selected_score, indices = torch.topk(score, k=count, dim=1, sorted=False)
        sharp_weights = torch.softmax(
            selected_score.float() / max(self.temperature, 1e-4), dim=1
        )
        if self.training and self.route_dropout > 0.0 and count > 1:
            keep = torch.rand_like(sharp_weights) >= self.route_dropout
            keep.scatter_(1, sharp_weights.argmax(dim=1, keepdim=True), True)
            sharp_weights = sharp_weights * keep
            sharp_weights = sharp_weights / sharp_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights = (1.0 - self.uniform_mix) * sharp_weights + self.uniform_mix / count
        weights = weights.to(selected_score.dtype)
        batch = torch.arange(len(target_context), device=indices.device)[:, None]
        selected_observed = observed_context[indices]
        selected_relation = relation[batch, indices]
        target = target_context[:, None, :].expand(-1, count, -1)
        pair = self.pair(torch.cat([target, selected_observed, selected_relation], dim=2))
        raw_entropy = -(
            weights.float() * weights.float().clamp_min(1e-8).log()
        ).sum(dim=1)
        entropy = raw_entropy / math.log(max(count, 2))
        distance = selected_relation[..., 4]
        return {
            "indices": indices,
            "weights": weights,
            "sharp_weights": sharp_weights.to(selected_score.dtype),
            "pair": pair,
            "entropy": entropy,
            "top1_mass": weights.float().amax(dim=1),
            "effective_neighbors": raw_entropy.exp(),
            "mean_distance": (weights.float() * distance.float()).sum(dim=1),
        }


class FullResolutionResidual3d(nn.Module):
    def __init__(self, channels: int, dropout: float, dilation: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            3,
            padding=0,
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
        dilation = int(self.depthwise.dilation[0])
        normalized = self.norm(value)
        normalized = functional.pad(
            normalized, (dilation, dilation, 0, 0, 0, 0), mode="constant"
        )
        height_mode = "circular" if value.shape[-2] > dilation else "replicate"
        depth_mode = "circular" if value.shape[-3] > dilation else "replicate"
        normalized = functional.pad(
            normalized, (0, 0, dilation, dilation, 0, 0), mode=height_mode
        )
        normalized = functional.pad(
            normalized, (0, 0, 0, 0, dilation, dilation), mode=depth_mode
        )
        mixed = self.channel_mixer(self.depthwise(normalized))
        return value + mixed


class AxialLatentTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("latent token channels must be divisible by attention heads")
        self.delay_norm = nn.LayerNorm(channels)
        self.delay_attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.angle_norm = nn.LayerNorm(channels)
        self.angle_attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(channels)
        self.output_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, angle_v, angle_h, delay = value.shape
        delay_tokens = value.permute(0, 2, 3, 4, 1).reshape(
            batch * angle_v * angle_h, delay, channels
        )
        normalized = self.delay_norm(delay_tokens)
        delay_tokens = delay_tokens + self.delay_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        value = delay_tokens.reshape(batch, angle_v, angle_h, delay, channels)
        angle_tokens = value.permute(0, 3, 1, 2, 4).reshape(
            batch * delay, angle_v * angle_h, channels
        )
        normalized = self.angle_norm(angle_tokens)
        angle_tokens = angle_tokens + self.angle_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = angle_tokens.reshape(
            batch, delay, angle_v, angle_h, channels
        ).permute(0, 2, 3, 1, 4)
        tokens = tokens + self.output_mlp(self.output_norm(tokens))
        return tokens.permute(0, 4, 1, 2, 3)


class GeometryWarpedLatentField(nn.Module):
    """Route, geometrically align, and aggregate every full-resolution latent bin."""

    def __init__(
        self,
        latent_shape: tuple[int, int, int, int],
        observed_context_channels: int,
        target_context_channels: int,
        pair_channels: int,
        cell_count: int,
        token_channels: int,
        attention_heads: int,
        attention_chunk_size: int,
        refinement_blocks: int,
        axial_blocks: int,
        maximum_warp: tuple[float, float, float],
        dropout: float,
        gradient_checkpointing: bool,
        maximum_residual: float,
        route_bias_scale: float,
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
        self.diagnostic_disable_warp = False
        self.diagnostic_route_bias_scale = float(route_bias_scale)
        self.maximum_residual = float(maximum_residual)
        self.register_buffer(
            "maximum_warp",
            torch.tensor(maximum_warp, dtype=torch.float32),
            persistent=True,
        )

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
        self.observed_projection = nn.Linear(pair_channels, token_channels)
        self.target_projection = nn.Linear(target_context_channels, token_channels)
        self.query = nn.Linear(token_channels, token_channels)
        self.key = nn.Linear(token_channels, token_channels)
        self.value = nn.Linear(token_channels, token_channels)
        self.attention_output = nn.Linear(token_channels, token_channels)
        self.warp = nn.Sequential(
            nn.Linear(pair_channels, pair_channels),
            nn.GELU(),
            nn.Linear(pair_channels, 3),
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
        self.axial = nn.ModuleList(
            [
                AxialLatentTransformerBlock(
                    token_channels, attention_heads, dropout
                )
                for _ in range(int(axial_blocks))
            ]
        )
        self.station_film = nn.Embedding(cell_count, token_channels * 2)
        nn.init.zeros_(self.station_film.weight)
        self.output = nn.Conv3d(token_channels, latent_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.residual_gate_logit = nn.Parameter(torch.full((latent_channels,), -1.5))

    @property
    def position_count(self) -> int:
        return int(math.prod(self.latent_shape[1:]))

    def _position_embedding(self) -> torch.Tensor:
        return (
            self.angle_v_embedding[:, None, None, :]
            + self.angle_h_embedding[None, :, None, :]
            + self.delay_embedding[None, None, :, :]
        ).reshape(self.position_count, self.token_channels)

    def _base_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        _, angle_v, angle_h, delay = self.latent_shape
        def axis(size: int) -> torch.Tensor:
            if size == 1:
                return torch.zeros(1, device=device, dtype=dtype)
            return torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)

        z = axis(angle_v)
        y = axis(angle_h)
        x = axis(delay)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1)

    def _warp_latent(
        self,
        observed_latent: torch.Tensor,
        route_indices: torch.Tensor,
        pair: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets, observations = route_indices.shape
        selected = observed_latent.index_select(0, route_indices.reshape(-1)).reshape(
            targets, observations, *self.latent_shape
        )
        offsets = torch.tanh(self.warp(pair).float()) * self.maximum_warp
        if self.diagnostic_disable_warp:
            offsets = torch.zeros_like(offsets)
        _, angle_v, angle_h, delay = self.latent_shape
        normalized = torch.stack(
            [
                offsets[..., 2] * 2.0 / max(delay - 1, 1),
                offsets[..., 1] * 2.0 / max(angle_h - 1, 1),
                offsets[..., 0] * 2.0 / max(angle_v - 1, 1),
            ],
            dim=-1,
        )
        grid = self._base_grid(selected.device, torch.float32)
        grid = grid[None, None] + normalized[:, :, None, None, None, :]
        warped = functional.grid_sample(
            selected.reshape(-1, *self.latent_shape).float(),
            grid.reshape(-1, angle_v, angle_h, delay, 3),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped.to(selected.dtype).reshape_as(selected), offsets

    def _attention_chunk(
        self,
        latent: torch.Tensor,
        position: torch.Tensor,
        pair_features: torch.Tensor,
        target_features: torch.Tensor,
        route_bias: torch.Tensor,
    ) -> torch.Tensor:
        bins, targets, observations = latent.shape[:3]
        observed = (
            self.latent_projection(latent)
            + pair_features[None, :, :, :]
            + position[:, None, None, :]
        )
        target = target_features[None, :, :] + position[:, None, :]
        query = self.query(target).reshape(
            bins, targets, self.attention_heads, self.head_channels
        ).permute(0, 1, 2, 3).unsqueeze(3)
        key = self.key(observed).reshape(
            bins, targets, observations, self.attention_heads, self.head_channels
        ).permute(0, 1, 3, 2, 4)
        value = self.value(observed).reshape(
            bins, targets, observations, self.attention_heads, self.head_channels
        ).permute(0, 1, 3, 2, 4)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=route_bias.to(query.dtype)[None, :, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        ).squeeze(3)
        attended = attended.reshape(bins, targets, self.token_channels)
        output = target + self.attention_output(attended)
        return output + self.token_mlp(output)

    def forward(
        self,
        observed_latent: torch.Tensor,
        target_context: torch.Tensor,
        route: dict[str, torch.Tensor],
        cell_id: int,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(observed_latent.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Expected observed latent {self.latent_shape}, got {tuple(observed_latent.shape[1:])}"
            )
        warped, offsets = self._warp_latent(
            observed_latent, route["indices"], route["pair"]
        )
        latent = warped.flatten(3).permute(3, 0, 1, 2)
        positions = self._position_embedding()
        route_bias = route["weights"].float().clamp_min(1e-8).log()
        route_bias = route_bias - route_bias.mean(dim=1, keepdim=True)
        route_bias = route_bias.clamp(-2.0, 2.0) * float(self.diagnostic_route_bias_scale)
        pair_features = self.observed_projection(route["pair"])
        target_features = self.target_projection(target_context)
        chunks: list[torch.Tensor] = []
        for start in range(0, self.position_count, self.attention_chunk_size):
            stop = min(start + self.attention_chunk_size, self.position_count)
            arguments = (
                latent[start:stop],
                positions[start:stop],
                pair_features,
                target_features,
                route_bias,
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
        for block in self.axial:
            if self.gradient_checkpointing and self.training:
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        film = self.station_film(
            torch.full(
                (len(features),), int(cell_id), dtype=torch.long, device=features.device
            )
        )
        scale, shift = film.chunk(2, dim=1)
        features = features * (1.0 + scale[:, :, None, None, None])
        features = features + shift[:, :, None, None, None]
        weights = route.get("latent_weights", route["weights"]).to(warped.dtype)
        base = (warped * weights[:, :, None, None, None, None]).sum(dim=1)
        raw_residual = torch.tanh(self.output(features)) * self.maximum_residual
        gate = torch.sigmoid(self.residual_gate_logit)[None, :, None, None, None]
        residual = gate * raw_residual
        mean_warp = offsets.abs().mean(dim=(1, 2))
        normalized_warp = offsets.abs() / self.maximum_warp.clamp_min(1e-6)
        warp_saturation = functional.relu(normalized_warp - 0.9).square().mean(dim=(1, 2))
        return base + residual, features, mean_warp, base, residual, warp_saturation


class FullResolutionContextField(nn.Module):
    """Context V3: multi-neighbor transport with bounded residual refinement."""

    def set_diagnostic_ablation(
        self,
        *,
        disable_warp: bool = False,
        route_bias_scale: float = 1.0,
    ) -> None:
        if route_bias_scale < 0.0:
            raise ValueError("route_bias_scale must be non-negative")
        for field in (self.spectrum_field, self.detail_field):
            field.diagnostic_disable_warp = bool(disable_warp)
            field.diagnostic_route_bias_scale = float(route_bias_scale)

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
        environment_base_channels: int = 32,
        environment_feature_channels: int = 24,
        environment_blocks: int = 2,
        corridor_width: int = 96,
        corridor_heads: int = 4,
        corridor_layers: int = 2,
        corridor_maximum_samples: int = 32,
        station_embedding_channels: int = 16,
        fourier_bands: int = 8,
        global_width: int = 256,
        global_blocks: int = 3,
        router_width: int = 128,
        router_top_k: int = 96,
        router_temperature: float = 1.5,
        router_uniform_mix: float = 0.15,
        router_dropout: float = 0.1,
        route_bias_scale: float = 0.15,
        pair_width: int = 128,
        spectrum_token_channels: int = 64,
        detail_token_channels: int = 48,
        attention_heads: int = 4,
        attention_chunk_size: int = 16,
        refinement_blocks: int = 4,
        axial_blocks: int = 2,
        spectrum_maximum_warp: tuple[float, float, float] = (0.75, 1.5, 3.0),
        detail_maximum_warp: tuple[float, float, float] = (1.5, 3.0, 6.0),
        spectrum_maximum_residual: float = 1.0,
        detail_maximum_residual: float = 1.0,
        maximum_power_residual: float = 1.0,
        dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.spectrum_shape = tuple(int(value) for value in spectrum_shape)
        self.phase_shape = tuple(int(value) for value in phase_shape)
        self.spectrum_latent_dim = int(math.prod(self.spectrum_shape))
        self.phase_latent_dim = int(math.prod(self.phase_shape))
        self.maximum_power_residual = float(maximum_power_residual)
        self.pool = CellTokenPool(4, map_token_channels, map_hidden_channels)
        self.context_fpn = GatedContextFPN(
            map_token_channels + static_context_channels + 2,
            context_base_channels,
            context_feature_channels,
            dropout,
        )
        self.environment_encoder = EnvironmentFeaturePyramid(
            6,
            environment_base_channels,
            environment_feature_channels,
            environment_blocks,
            dropout,
        )
        self.corridor_encoder = CorridorTransformer(
            environment_feature_channels,
            corridor_width,
            corridor_heads,
            corridor_layers,
            corridor_maximum_samples,
            dropout,
        )
        self.fourier = FourierFeatures(fourier_bands)
        self.station_embedding = nn.Embedding(cell_count, station_embedding_channels)
        local_environment_channels = 3 * environment_feature_channels
        target_input_channels = (
            context_feature_channels
            + local_environment_channels
            + corridor_width
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
        self.router = ObservationRouter(
            global_width,
            router_width,
            pair_width,
            router_top_k,
            dropout,
            router_temperature,
            router_uniform_mix,
            router_dropout,
        )
        branch_arguments = {
            "observed_context_channels": global_width,
            "target_context_channels": global_width,
            "pair_channels": pair_width,
            "cell_count": cell_count,
            "attention_heads": attention_heads,
            "attention_chunk_size": attention_chunk_size,
            "refinement_blocks": refinement_blocks,
            "axial_blocks": axial_blocks,
            "dropout": dropout,
            "gradient_checkpointing": gradient_checkpointing,
            "route_bias_scale": route_bias_scale,
        }
        self.spectrum_field = GeometryWarpedLatentField(
            self.spectrum_shape,
            token_channels=spectrum_token_channels,
            maximum_warp=spectrum_maximum_warp,
            maximum_residual=spectrum_maximum_residual,
            **branch_arguments,
        )
        self.detail_field = GeometryWarpedLatentField(
            self.phase_shape,
            token_channels=detail_token_channels,
            maximum_warp=detail_maximum_warp,
            maximum_residual=detail_maximum_residual,
            **branch_arguments,
        )
        self.spectrum_to_detail = nn.Conv3d(
            spectrum_token_channels, detail_token_channels, 1
        )
        output_input = global_width + spectrum_token_channels + detail_token_channels
        output_hidden = max(64, global_width // 2)
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(output_input),
            nn.Linear(output_input, output_hidden),
            nn.GELU(),
            nn.Linear(output_hidden, 2),
        )
        with torch.no_grad():
            self.scalar_head[-1].weight[0].zero_()
            self.scalar_head[-1].bias[0].zero_()

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
        corridor = self.corridor_encoder(
            sample_corridor_sequence(
                environment_features[0], query_corridor_coordinates
            )
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
        route = self.router(
            observed_global,
            query_global,
            observed_relative_xy,
            query_relative_xy,
        )
        selected_nonoutage = 1.0 - observed_outage[route["indices"]].float()
        latent_weights = route["weights"].float() * selected_nonoutage
        fallback = latent_weights.sum(dim=1, keepdim=True) <= 1e-8
        latent_weights = latent_weights / latent_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        route["latent_weights"] = torch.where(
            fallback, route["weights"].float(), latent_weights
        ).to(route["weights"].dtype)
        (
            spectrum,
            spectrum_features,
            spectrum_warp,
            spectrum_base,
            spectrum_residual,
            spectrum_warp_saturation,
        ) = self.spectrum_field(
            observed_spectrum,
            query_global,
            route,
            cell_id,
        )
        detail_condition = functional.interpolate(
            self.spectrum_to_detail(spectrum_features),
            size=self.phase_shape[1:],
            mode="trilinear",
            align_corners=False,
        )
        (
            phase,
            detail_features,
            detail_warp,
            detail_base,
            detail_residual,
            detail_warp_saturation,
        ) = self.detail_field(
            observed_phase,
            query_global,
            route,
            cell_id,
            condition=detail_condition,
        )
        summary = torch.cat(
            [
                query_global,
                spectrum_features.mean(dim=(2, 3, 4)),
                detail_features.mean(dim=(2, 3, 4)),
            ],
            dim=1,
        )
        scalars = self.scalar_head(summary)
        selected_power = observed_power[route["indices"]]
        power_base = (
            route["latent_weights"].to(selected_power.dtype) * selected_power
        ).sum(dim=1)
        power = power_base + self.maximum_power_residual * torch.tanh(scalars[:, 0])
        return {
            "spectrum": spectrum.flatten(1),
            "phase": phase.flatten(1),
            "power": power,
            "spectrum_base": spectrum_base.flatten(1),
            "phase_base": detail_base.flatten(1),
            "power_base": power_base,
            "spectrum_residual": spectrum_residual.flatten(1),
            "phase_residual": detail_residual.flatten(1),
            "spectrum_residual_rms": spectrum_residual.float().square().mean(dim=(1, 2, 3, 4)).clamp_min(1e-12).sqrt(),
            "phase_residual_rms": detail_residual.float().square().mean(dim=(1, 2, 3, 4)).clamp_min(1e-12).sqrt(),
            "outage_logit": scalars[:, 1],
            "router_entropy": route["entropy"],
            "router_top1_mass": route["top1_mass"],
            "router_effective_neighbors": route["effective_neighbors"],
            "router_distance": route["mean_distance"],
            "spectrum_warp": spectrum_warp,
            "detail_warp": detail_warp,
            "warp_saturation": 0.5 * (spectrum_warp_saturation + detail_warp_saturation),
            "router_temperature": torch.full_like(route["entropy"], self.router.temperature),
        }
