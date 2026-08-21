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
    def __init__(
        self, input_channels: int, output_channels: int, dropout: float
    ) -> None:
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
        self,
        input_channels: int,
        base_channels: int,
        output_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        widths = [
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
        ]
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

    def __init__(
        self, point_channels: int, token_channels: int, hidden_channels: int
    ) -> None:
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


def latent_statistics_descriptor(
    value: torch.Tensor, dimensions: int, *, normalize: bool = True
) -> torch.Tensor:
    """Pool a latent only for routing/map context, never for reconstruction."""
    if dimensions < 4 or dimensions % 2:
        raise ValueError("descriptor dimensions must be even and at least four")
    width = dimensions // 2
    flattened = value.float().flatten(1)
    mean = functional.adaptive_avg_pool1d(flattened[:, None], width).squeeze(1)
    rms = (
        functional.adaptive_avg_pool1d(flattened.square()[:, None], width)
        .squeeze(1)
        .clamp_min(1e-8)
        .sqrt()
    )
    descriptor = torch.cat([mean, rms], dim=1)
    if normalize:
        descriptor = functional.normalize(descriptor, dim=1, eps=1e-6)
    return descriptor


def spectrum_chart(spectrum: torch.Tensor, dimensions: int) -> torch.Tensor:
    """Smooth-spectrum retrieval chart used by the spectrum router."""
    return latent_statistics_descriptor(spectrum, dimensions, normalize=True)


def channel_chart(
    spectrum: torch.Tensor, phase: torch.Tensor, dimensions: int
) -> torch.Tensor:
    """Compatibility chart retained for diagnostics from earlier schemes."""
    if dimensions < 8 or dimensions % 4:
        raise ValueError(
            "chart dimensions must be a multiple of four and at least eight"
        )
    half = dimensions // 2
    return functional.normalize(
        torch.cat(
            [
                latent_statistics_descriptor(spectrum, half, normalize=False),
                latent_statistics_descriptor(phase, half, normalize=False),
            ],
            dim=1,
        ),
        dim=1,
        eps=1e-6,
    )


class StructuredSpectralPriorEncoder(nn.Module):
    """Map Scheme E PAS/PDP fields onto the full spectrum latent grid."""

    def __init__(
        self,
        proxy_count: int,
        antenna_shape: tuple[int, int],
        pdp_shape: tuple[int, int],
        output_channels: int,
        output_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.proxy_count = int(proxy_count)
        self.antenna_shape = tuple(int(value) for value in antenna_shape)
        self.pdp_shape = tuple(int(value) for value in pdp_shape)
        self.output_shape = tuple(int(value) for value in output_shape)
        self.encoder = nn.Sequential(
            nn.Conv3d(6, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(output_channels), output_channels),
            nn.GELU(),
            FullResolutionResidual3d(output_channels, 0.0, dilation=1),
        )
        self.confidence = nn.Sequential(nn.Linear(1, output_channels), nn.Sigmoid())
        self.gate_logit = nn.Parameter(torch.tensor(-1.0))

    @staticmethod
    def _standardize(value: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
        mean = value.mean(dim=dimensions, keepdim=True)
        std = value.std(dim=dimensions, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (value - mean) / std

    def forward(
        self,
        pas_log: torch.Tensor,
        pdp_log: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(pas_log)
        antenna_v, antenna_h = self.antenna_shape
        users, delays = self.pdp_shape
        expected_pas = (self.proxy_count + 1) * antenna_v * antenna_h
        if pas_log.shape != (batch, expected_pas):
            raise ValueError(
                f"Expected PAS prior {(batch, expected_pas)}, got {tuple(pas_log.shape)}"
            )
        if pdp_log.shape != (batch, users * delays):
            raise ValueError(
                f"Expected PDP prior {(batch, users * delays)}, got {tuple(pdp_log.shape)}"
            )
        pas = pas_log.float().reshape(batch, self.proxy_count + 1, antenna_v, antenna_h)
        proxy = pas[:, : self.proxy_count]
        angle_mean = 0.5 * (proxy.mean(dim=1) + pas[:, -1])
        angle_std = proxy.std(dim=1, unbiased=False)
        angle_mean = self._standardize(angle_mean, (1, 2))
        angle_std = self._standardize(angle_std, (1, 2))
        angle_mean = functional.adaptive_avg_pool2d(
            angle_mean[:, None], self.output_shape[:2]
        )[:, 0]
        angle_std = functional.adaptive_avg_pool2d(
            angle_std[:, None], self.output_shape[:2]
        )[:, 0]

        pdp = pdp_log.float().reshape(batch, users, delays)
        delay_mean = self._standardize(pdp.mean(dim=1), (1,))
        delay_std = self._standardize(pdp.std(dim=1, unbiased=False), (1,))
        delay_mean = functional.adaptive_avg_pool1d(
            delay_mean[:, None], self.output_shape[2]
        )[:, 0]
        delay_std = functional.adaptive_avg_pool1d(
            delay_std[:, None], self.output_shape[2]
        )[:, 0]

        angle_mean_3d = angle_mean[:, :, :, None].expand(
            -1, -1, -1, self.output_shape[2]
        )
        angle_std_3d = angle_std[:, :, :, None].expand_as(angle_mean_3d)
        delay_mean_3d = delay_mean[:, None, None, :].expand_as(angle_mean_3d)
        delay_std_3d = delay_std[:, None, None, :].expand_as(angle_mean_3d)
        joint = angle_mean_3d * delay_mean_3d
        disagreement = angle_std_3d + delay_std_3d
        fields = torch.stack(
            [
                angle_mean_3d,
                angle_std_3d,
                delay_mean_3d,
                delay_std_3d,
                joint,
                disagreement,
            ],
            dim=1,
        )
        features = self.encoder(fields)
        confidence = self.confidence(-uncertainty.float().reshape(batch, 1))
        return (
            torch.sigmoid(self.gate_logit)
            * confidence[:, :, None, None, None]
            * features
        )


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


class FourierOperator3dBlock(nn.Module):
    """Low-mode global mixing without flattening the full latent grid."""

    def __init__(
        self,
        channels: int,
        spatial_shape: tuple[int, int, int],
        maximum_modes: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        angle_v, angle_h, delay = spatial_shape
        modes = (
            min(int(maximum_modes[0]), angle_v),
            min(int(maximum_modes[1]), angle_h),
            min(int(maximum_modes[2]), delay // 2 + 1),
        )
        self.modes = modes
        self.norm = nn.GroupNorm(1, channels)
        self.weight = nn.Parameter(torch.zeros(channels, *modes, 2))
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.output = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(channels * 2, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_dtype = value.dtype
        normalized = self.norm(value).float()
        spectrum = torch.fft.rfftn(normalized, dim=(-3, -2, -1), norm="ortho")
        mixed = torch.zeros_like(spectrum)
        mv, mh, md = self.modes
        weight = torch.view_as_complex(self.weight.contiguous())
        mixed[:, :, :mv, :mh, :md] = spectrum[:, :, :mv, :mh, :md] * weight[None]
        spatial = torch.fft.irfftn(
            mixed,
            s=value.shape[-3:],
            dim=(-3, -2, -1),
            norm="ortho",
        ).to(original_dtype)
        return value + torch.sigmoid(self.gate_logit) * self.output(spatial)


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
                    nn.Conv2d(
                        output_channels, output_channels, 3, padding=1, bias=False
                    ),
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
        pyramid: list[torch.Tensor] = [
            self.lateral[index](level) for index, level in enumerate(levels)
        ]
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
    """Retrieve a small query-specific anchor pool from geometry and radio chart."""

    relation_channels = 10

    def __init__(
        self,
        context_channels: int,
        router_width: int,
        pair_width: int,
        top_k: int,
        dropout: float,
        candidate_top_k: int = 0,
        temperature: float = 1.5,
        uniform_mix: float = 0.0,
        route_dropout: float = 0.1,
        chart_weight: float = 1.0,
        distance_penalty: float = 0.0,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.candidate_top_k = int(candidate_top_k)
        self.register_buffer(
            "temperature_state", torch.tensor(float(temperature)), persistent=True
        )
        self.uniform_mix = float(uniform_mix)
        self.route_dropout = float(route_dropout)
        self.chart_weight = float(chart_weight)
        self.distance_penalty = float(distance_penalty)
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

    @property
    def temperature(self) -> float:
        return float(self.temperature_state.item())

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("router temperature must be positive")
        self.temperature_state.fill_(float(value))

    def forward(
        self,
        observed_context: torch.Tensor,
        target_context: torch.Tensor,
        observed_relative_xy: torch.Tensor,
        target_relative_xy: torch.Tensor,
        observed_chart: torch.Tensor | None = None,
        target_chart: torch.Tensor | None = None,
        observed_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        relation = pair_relation_features(target_relative_xy, observed_relative_xy)
        distance = relation[..., 4]
        score = (
            self.query(target_context)[:, None, :]
            * self.key(observed_context)[None, :, :]
        ).sum(dim=2) * self.scale
        score = score + self.relation_score(relation).squeeze(2)
        score = score - self.distance_penalty * distance
        if observed_chart is not None or target_chart is not None:
            if observed_chart is None or target_chart is None:
                raise ValueError(
                    "observed_chart and target_chart must be supplied together"
                )
            score = score + self.chart_weight * (
                target_chart[:, None, :] * observed_chart[None, :, :]
            ).sum(dim=2)
        valid = torch.ones_like(score, dtype=torch.bool)
        if observed_valid is not None:
            observed_valid = observed_valid.to(device=score.device, dtype=torch.bool)
            if observed_valid.shape != (observed_context.shape[0],):
                raise ValueError("observed_valid must have one value per observation")
            if torch.any(observed_valid):
                valid = observed_valid[None, :].expand_as(score)
        candidate_count = min(
            max(self.candidate_top_k, self.top_k, 1), observed_context.shape[0]
        )
        if candidate_count < observed_context.shape[0]:
            candidate_distance = distance.masked_fill(~valid, torch.inf)
            candidates = torch.topk(
                candidate_distance,
                k=candidate_count,
                dim=1,
                largest=False,
                sorted=False,
            ).indices
            candidate_mask = torch.zeros_like(valid)
            candidate_mask.scatter_(1, candidates, True)
            valid = valid & candidate_mask
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        count = min(max(self.top_k, 1), candidate_count)
        selected_score, indices = torch.topk(score, k=count, dim=1, sorted=False)
        sharp_weights = torch.softmax(
            selected_score.float() / max(self.temperature, 1e-4), dim=1
        )
        if self.training and self.route_dropout > 0.0 and count > 1:
            keep = torch.rand_like(sharp_weights) >= self.route_dropout
            keep.scatter_(1, sharp_weights.argmax(dim=1, keepdim=True), True)
            sharp_weights = sharp_weights * keep
            sharp_weights = sharp_weights / sharp_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        weights = (1.0 - self.uniform_mix) * sharp_weights + self.uniform_mix / count
        weights = weights.to(selected_score.dtype)
        batch = torch.arange(len(target_context), device=indices.device)[:, None]
        selected_observed = observed_context[indices]
        selected_relation = relation[batch, indices]
        target = target_context[:, None, :].expand(-1, count, -1)
        pair = self.pair(
            torch.cat([target, selected_observed, selected_relation], dim=2)
        )
        raw_entropy = -(weights.float() * weights.float().clamp_min(1e-8).log()).sum(
            dim=1
        )
        entropy = raw_entropy / math.log(max(count, 2))
        selected_distance = selected_relation[..., 4]
        return {
            "indices": indices,
            "weights": weights,
            "sharp_weights": sharp_weights.to(selected_score.dtype),
            "pair": pair,
            "entropy": entropy,
            "top1_mass": weights.float().amax(dim=1),
            "effective_neighbors": raw_entropy.exp(),
            "mean_distance": (weights.float() * selected_distance.float()).sum(dim=1),
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
            raise ValueError(
                "latent token channels must be divisible by attention heads"
            )
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
        delay_tokens = (
            delay_tokens
            + self.delay_attention(
                normalized, normalized, normalized, need_weights=False
            )[0]
        )
        value = delay_tokens.reshape(batch, angle_v, angle_h, delay, channels)
        angle_tokens = value.permute(0, 3, 1, 2, 4).reshape(
            batch * delay, angle_v * angle_h, channels
        )
        normalized = self.angle_norm(angle_tokens)
        angle_tokens = (
            angle_tokens
            + self.angle_attention(
                normalized, normalized, normalized, need_weights=False
            )[0]
        )
        tokens = angle_tokens.reshape(batch, delay, angle_v, angle_h, channels).permute(
            0, 2, 3, 1, 4
        )
        tokens = tokens + self.output_mlp(self.output_norm(tokens))
        return tokens.permute(0, 4, 1, 2, 3)


class GeometryWarpedLatentField(nn.Module):
    """Regionally align anchors and select Top-K independently for every latent token."""

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
        token_top_k: int = 2,
        operator_blocks: int = 2,
        operator_modes: tuple[int, int, int] = (4, 6, 8),
        regional_warp: bool = True,
        phase_rotation: bool = False,
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
        self.token_top_k = int(token_top_k)
        if self.token_top_k < 1:
            raise ValueError("token_top_k must be positive")
        self.dropout = float(dropout)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.diagnostic_disable_warp = False
        self.diagnostic_route_bias_scale = float(route_bias_scale)
        self.maximum_residual = float(maximum_residual)
        self.regional_warp = bool(regional_warp)
        self.phase_rotation = bool(phase_rotation)
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
        regional_elements = 3 * (angle_v + angle_h + delay)
        self.regional_warp_field = nn.Linear(pair_channels, regional_elements)
        nn.init.zeros_(self.regional_warp_field.weight)
        nn.init.zeros_(self.regional_warp_field.bias)
        self.regional_warp_gate_logit = nn.Parameter(torch.tensor(-1.5))
        if self.phase_rotation:
            self.phase_rotation_global = nn.Linear(pair_channels, 1)
            self.phase_rotation_field = nn.Linear(
                pair_channels, angle_v + angle_h + delay
            )
            nn.init.zeros_(self.phase_rotation_global.weight)
            nn.init.zeros_(self.phase_rotation_global.bias)
            nn.init.zeros_(self.phase_rotation_field.weight)
            nn.init.zeros_(self.phase_rotation_field.bias)
            self.phase_rotation_gate_logit = nn.Parameter(torch.tensor(-1.5))
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
                AxialLatentTransformerBlock(token_channels, attention_heads, dropout)
                for _ in range(int(axial_blocks))
            ]
        )
        self.operator = nn.ModuleList(
            [
                FourierOperator3dBlock(
                    token_channels,
                    (angle_v, angle_h, delay),
                    operator_modes,
                    dropout,
                )
                for _ in range(int(operator_blocks))
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
        _, angle_v, angle_h, delay = self.latent_shape
        global_offsets = self.warp(pair).float()[:, :, None, None, None, :]
        if self.regional_warp:
            controls = (
                self.regional_warp_field(pair)
                .float()
                .reshape(targets, observations, 3, angle_v + angle_h + delay)
            )
            vertical, horizontal, temporal = torch.split(
                controls, (angle_v, angle_h, delay), dim=3
            )
            regional = (
                vertical[:, :, :, :, None, None]
                + horizontal[:, :, :, None, :, None]
                + temporal[:, :, :, None, None, :]
            ).permute(0, 1, 3, 4, 5, 2)
            global_offsets = (
                global_offsets + torch.sigmoid(self.regional_warp_gate_logit) * regional
            )
        offsets = torch.tanh(global_offsets) * self.maximum_warp
        if self.diagnostic_disable_warp:
            offsets = torch.zeros_like(offsets)
        normalized = torch.stack(
            [
                offsets[..., 2] * 2.0 / max(delay - 1, 1),
                offsets[..., 1] * 2.0 / max(angle_h - 1, 1),
                offsets[..., 0] * 2.0 / max(angle_v - 1, 1),
            ],
            dim=-1,
        )
        grid = self._base_grid(selected.device, torch.float32)
        grid = grid[None, None] + normalized
        warped = (
            functional.grid_sample(
                selected.reshape(-1, *self.latent_shape).float(),
                grid.reshape(-1, angle_v, angle_h, delay, 3),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .to(selected.dtype)
            .reshape_as(selected)
        )
        if self.phase_rotation and self.latent_channels % 2 == 0:
            phase_controls = self.phase_rotation_field(pair).float()
            vertical, horizontal, temporal = torch.split(
                phase_controls, (angle_v, angle_h, delay), dim=2
            )
            phase = (
                vertical[:, :, :, None, None]
                + horizontal[:, :, None, :, None]
                + temporal[:, :, None, None, :]
            )
            phase = (
                phase + self.phase_rotation_global(pair).float()[:, :, None, None, :]
            )
            phase = math.pi * torch.tanh(
                torch.sigmoid(self.phase_rotation_gate_logit) * phase
            )
            real, imaginary = warped.float().chunk(2, dim=2)
            cosine = phase.cos()[:, :, None]
            sine = phase.sin()[:, :, None]
            warped = torch.cat(
                [real * cosine - imaginary * sine, real * sine + imaginary * cosine],
                dim=2,
            ).to(selected.dtype)
        return warped, offsets

    def _attention_chunk(
        self,
        latent: torch.Tensor,
        position: torch.Tensor,
        pair_features: torch.Tensor,
        target_features: torch.Tensor,
        route_bias: torch.Tensor,
        route_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bins, targets, observations = latent.shape[:3]
        observed = (
            self.latent_projection(latent)
            + pair_features[None, :, :, :]
            + position[:, None, None, :]
        )
        target = target_features[None, :, :] + position[:, None, :]
        query = self.query(target).reshape(
            bins, targets, self.attention_heads, self.head_channels
        )
        key = (
            self.key(observed)
            .reshape(
                bins, targets, observations, self.attention_heads, self.head_channels
            )
            .permute(0, 1, 3, 2, 4)
        )
        value = (
            self.value(observed)
            .reshape(
                bins, targets, observations, self.attention_heads, self.head_channels
            )
            .permute(0, 1, 3, 2, 4)
        )
        logits = (query[:, :, :, None, :] * key).sum(dim=4) * self.head_channels**-0.5
        logits = logits + route_bias.to(logits.dtype)[None, :, None, :]
        logits = logits.masked_fill(
            ~route_valid[None, :, None, :], torch.finfo(logits.dtype).min
        )
        selected_count = min(self.token_top_k, observations)
        selected_logits, selected_indices = torch.topk(logits, k=selected_count, dim=3)
        sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        sparse_logits.scatter_(3, selected_indices, selected_logits)
        attention = torch.softmax(sparse_logits.float(), dim=3).to(logits.dtype)
        if self.training and self.dropout > 0.0:
            attention = functional.dropout(attention, self.dropout)
            attention = attention / attention.sum(dim=3, keepdim=True).clamp_min(1e-6)
        attended = (attention[..., None] * value).sum(dim=3)
        attended = attended.reshape(bins, targets, self.token_channels)
        output = target + self.attention_output(attended)
        token_weights = attention.float().mean(dim=2)
        token_base = (token_weights[..., None] * latent.float()).sum(dim=2)
        entropy = -(token_weights * token_weights.clamp_min(1e-8).log()).sum(dim=2)
        return (
            output + self.token_mlp(output),
            token_base,
            entropy.exp(),
            token_weights.amax(dim=2),
        )

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
        route_weights = route.get("latent_weights", route["weights"])
        route_bias = route_weights.float().clamp_min(1e-8).log()
        route_bias = route_bias - route_bias.mean(dim=1, keepdim=True)
        route_bias = route_bias.clamp(-2.0, 2.0) * float(
            self.diagnostic_route_bias_scale
        )
        pair_features = self.observed_projection(route["pair"])
        target_features = self.target_projection(target_context)
        chunks: list[torch.Tensor] = []
        base_chunks: list[torch.Tensor] = []
        effective_chunks: list[torch.Tensor] = []
        top1_chunks: list[torch.Tensor] = []
        route_valid = route.get(
            "latent_valid", torch.ones_like(route_weights, dtype=torch.bool)
        )
        for start in range(0, self.position_count, self.attention_chunk_size):
            stop = min(start + self.attention_chunk_size, self.position_count)
            arguments = (
                latent[start:stop],
                positions[start:stop],
                pair_features,
                target_features,
                route_bias,
                route_valid,
            )
            if self.gradient_checkpointing and self.training:
                chunk = checkpoint(
                    self._attention_chunk, *arguments, use_reentrant=False
                )
            else:
                chunk = self._attention_chunk(*arguments)
            features_chunk, base_chunk, effective_chunk, top1_chunk = chunk
            chunks.append(features_chunk)
            base_chunks.append(base_chunk)
            effective_chunks.append(effective_chunk)
            top1_chunks.append(top1_chunk)
        features = (
            torch.cat(chunks, dim=0)
            .permute(1, 2, 0)
            .reshape(len(target_context), self.token_channels, *self.latent_shape[1:])
        )
        if condition is not None:
            if condition.shape != features.shape:
                raise ValueError(
                    f"Condition shape {tuple(condition.shape)} does not match {tuple(features.shape)}"
                )
            features = features + condition
        features = self.refiner(features)
        for block in self.operator:
            if self.gradient_checkpointing and self.training:
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
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
        base = (
            torch.cat(base_chunks, dim=0)
            .permute(1, 2, 0)
            .reshape(len(target_context), self.latent_channels, *self.latent_shape[1:])
            .to(warped.dtype)
        )
        raw_residual = torch.tanh(self.output(features)) * self.maximum_residual
        gate = torch.sigmoid(self.residual_gate_logit)[None, :, None, None, None]
        residual = gate * raw_residual
        mean_warp = offsets.abs().mean(dim=(1, 2, 3, 4, 5))
        normalized_warp = offsets.abs() / self.maximum_warp.clamp_min(1e-6)
        warp_saturation = (
            functional.relu(normalized_warp - 0.9).square().mean(dim=(1, 2, 3, 4, 5))
        )
        token_effective = torch.cat(effective_chunks, dim=0).mean(dim=0)
        token_top1 = torch.cat(top1_chunks, dim=0).mean(dim=0)
        return (
            base + residual,
            features,
            mean_warp,
            base,
            residual,
            warp_saturation,
            token_effective,
            token_top1,
        )


class BoundedPowerCNP(nn.Module):
    """Per-cell robust power predictor with ordered uncertainty quantiles."""

    def __init__(
        self,
        target_channels: int,
        pair_channels: int,
        cell_count: int,
        width: int,
        maximum_residual: float,
        maximum_absolute_z: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.maximum_residual = float(maximum_residual)
        self.maximum_absolute_z = float(maximum_absolute_z)
        self.query = nn.Linear(target_channels, width)
        self.context = nn.Sequential(
            nn.Linear(pair_channels + 2, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.score = nn.Sequential(nn.Tanh(), nn.Linear(width, 1))
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(target_channels + width),
                    nn.Linear(target_channels + width, width),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(width, 3),
                )
                for _ in range(cell_count)
            ]
        )
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        target: torch.Tensor,
        pair: torch.Tensor,
        route_indices: torch.Tensor,
        observed_power: torch.Tensor,
        observed_outage: torch.Tensor,
        cell_id: int,
    ) -> dict[str, torch.Tensor]:
        selected_power = observed_power[route_indices]
        selected_outage = observed_outage[route_indices].float()
        tokens = self.context(
            torch.cat(
                [pair, selected_power[:, :, None], selected_outage[:, :, None]], dim=2
            )
        )
        logits = self.score(tokens + self.query(target)[:, None, :]).squeeze(2)
        valid = selected_outage < 0.5
        all_invalid = ~valid.any(dim=1, keepdim=True)
        valid = valid | all_invalid
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=1).to(logits.dtype)
        base = (weights * selected_power).sum(dim=1)
        summary = (weights[:, :, None] * tokens).sum(dim=1)
        raw = self.heads[int(cell_id)](torch.cat([target, summary], dim=1))
        median = base + self.maximum_residual * torch.tanh(raw[:, 0])
        median = median.clamp(-self.maximum_absolute_z, self.maximum_absolute_z)
        lower_spread = 0.05 + functional.softplus(raw[:, 1])
        upper_spread = 0.05 + functional.softplus(raw[:, 2])
        q10 = (median - lower_spread).clamp(
            -self.maximum_absolute_z, self.maximum_absolute_z
        )
        q90 = (median + upper_spread).clamp(
            -self.maximum_absolute_z, self.maximum_absolute_z
        )
        entropy = -(weights.float() * weights.float().clamp_min(1e-8).log()).sum(dim=1)
        return {
            "median": median,
            "q10": torch.minimum(q10, median),
            "q90": torch.maximum(q90, median),
            "base": base,
            "weights": weights,
            "effective_neighbors": entropy.exp(),
        }


class FullResolutionContextField(nn.Module):
    """Scheme G: multi-scale local interpolation on the full latent fields."""

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

    def set_router_temperature(self, value: float) -> None:
        for router in (self.router, self.detail_router, self.power_router):
            router.temperature = float(value)

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
        router_candidate_top_k: int = 96,
        router_temperature: float = 1.5,
        router_uniform_mix: float = 0.15,
        router_dropout: float = 0.1,
        router_distance_penalty: float = 0.0,
        detail_router_top_k: int = 8,
        detail_router_candidate_top_k: int = 32,
        detail_router_uniform_mix: float = 0.1,
        detail_router_distance_penalty: float = 1.0,
        power_router_top_k: int = 16,
        power_router_candidate_top_k: int = 32,
        power_router_uniform_mix: float = 0.1,
        power_router_distance_penalty: float = 1.0,
        route_bias_scale: float = 0.15,
        chart_dimensions: int = 64,
        chart_weight: float = 1.0,
        map_spectrum_descriptor_dimensions: int = 32,
        pair_width: int = 128,
        spectrum_token_channels: int = 64,
        detail_token_channels: int = 48,
        attention_heads: int = 4,
        attention_chunk_size: int = 16,
        refinement_blocks: int = 4,
        axial_blocks: int = 2,
        operator_blocks: int = 2,
        operator_modes: tuple[int, int, int] = (4, 6, 8),
        token_top_k: int = 2,
        spectrum_token_top_k: int | None = None,
        detail_token_top_k: int | None = None,
        regional_warp: bool = True,
        detail_phase_rotation: bool = False,
        spectrum_maximum_warp: tuple[float, float, float] = (0.75, 1.5, 3.0),
        detail_maximum_warp: tuple[float, float, float] = (1.5, 3.0, 6.0),
        spectrum_maximum_residual: float = 1.0,
        detail_maximum_residual: float = 1.0,
        maximum_power_residual: float = 1.0,
        maximum_power_z: float = 5.0,
        power_width: int = 192,
        prior_proxy_count: int = 0,
        prior_antenna_shape: tuple[int, int] = (0, 0),
        prior_pdp_shape: tuple[int, int] = (0, 0),
        dropout: float = 0.05,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.spectrum_shape = tuple(int(value) for value in spectrum_shape)
        self.phase_shape = tuple(int(value) for value in phase_shape)
        self.spectrum_latent_dim = int(math.prod(self.spectrum_shape))
        self.phase_latent_dim = int(math.prod(self.phase_shape))
        self.maximum_power_residual = float(maximum_power_residual)
        self.chart_dimensions = int(chart_dimensions)
        self.map_spectrum_descriptor_dimensions = int(
            map_spectrum_descriptor_dimensions
        )
        self.pool = CellTokenPool(
            4 + self.map_spectrum_descriptor_dimensions,
            map_token_channels,
            map_hidden_channels,
        )
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
            candidate_top_k=router_candidate_top_k,
            temperature=router_temperature,
            uniform_mix=router_uniform_mix,
            route_dropout=router_dropout,
            chart_weight=chart_weight,
            distance_penalty=router_distance_penalty,
        )
        self.detail_router = ObservationRouter(
            global_width,
            router_width,
            pair_width,
            detail_router_top_k,
            dropout,
            candidate_top_k=detail_router_candidate_top_k,
            temperature=router_temperature,
            uniform_mix=detail_router_uniform_mix,
            route_dropout=router_dropout,
            chart_weight=0.0,
            distance_penalty=detail_router_distance_penalty,
        )
        self.power_router = ObservationRouter(
            global_width,
            router_width,
            pair_width,
            power_router_top_k,
            dropout,
            candidate_top_k=power_router_candidate_top_k,
            temperature=router_temperature,
            uniform_mix=power_router_uniform_mix,
            route_dropout=router_dropout,
            chart_weight=0.0,
            distance_penalty=power_router_distance_penalty,
        )
        self.query_chart = nn.Sequential(
            nn.LayerNorm(global_width),
            nn.Linear(global_width, global_width),
            nn.GELU(),
            nn.Linear(global_width, self.chart_dimensions),
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
            "operator_blocks": operator_blocks,
            "operator_modes": operator_modes,
            "regional_warp": regional_warp,
        }
        self.spectrum_field = GeometryWarpedLatentField(
            self.spectrum_shape,
            token_channels=spectrum_token_channels,
            maximum_warp=spectrum_maximum_warp,
            maximum_residual=spectrum_maximum_residual,
            phase_rotation=False,
            token_top_k=(
                token_top_k
                if spectrum_token_top_k is None
                else int(spectrum_token_top_k)
            ),
            **branch_arguments,
        )
        self.detail_field = GeometryWarpedLatentField(
            self.phase_shape,
            token_channels=detail_token_channels,
            maximum_warp=detail_maximum_warp,
            maximum_residual=detail_maximum_residual,
            phase_rotation=detail_phase_rotation,
            token_top_k=(
                token_top_k if detail_token_top_k is None else int(detail_token_top_k)
            ),
            **branch_arguments,
        )
        prior_enabled = (
            int(prior_proxy_count) > 0
            and all(int(value) > 0 for value in prior_antenna_shape)
            and all(int(value) > 0 for value in prior_pdp_shape)
        )
        self.prior_encoder = (
            StructuredSpectralPriorEncoder(
                prior_proxy_count,
                prior_antenna_shape,
                prior_pdp_shape,
                spectrum_token_channels,
                self.spectrum_shape[1:],
            )
            if prior_enabled
            else None
        )
        self.spectrum_to_detail = nn.Conv3d(
            spectrum_token_channels, detail_token_channels, 1
        )
        output_input = global_width + spectrum_token_channels + detail_token_channels
        output_hidden = max(64, global_width // 2)
        self.outage_head = nn.Sequential(
            nn.LayerNorm(output_input),
            nn.Linear(output_input, output_hidden),
            nn.GELU(),
            nn.Linear(output_hidden, 1),
        )
        self.power_cnp = BoundedPowerCNP(
            global_width,
            pair_width,
            cell_count,
            int(power_width),
            maximum_power_residual,
            maximum_power_z,
            dropout,
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
        query_prior_pas: torch.Tensor | None = None,
        query_prior_pdp: torch.Tensor | None = None,
        query_prior_uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        height, width = context_static.shape[-2:]
        spectrum_map_descriptor = latent_statistics_descriptor(
            observed_spectrum,
            self.map_spectrum_descriptor_dimensions,
            normalize=False,
        ).to(point_features.dtype)
        pooled, observed_mask, log_count = self.pool(
            torch.cat([point_features, spectrum_map_descriptor], dim=1),
            point_flat_indices,
            height,
            width,
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
        observed_context = sample_map(context_features, observed_context_coordinates)
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
        query_chart = functional.normalize(
            self.query_chart(query_global), dim=1, eps=1e-6
        )
        observed_chart = spectrum_chart(observed_spectrum, self.chart_dimensions)
        nonoutage_observed = observed_outage < 0.5
        spectrum_route = self.router(
            observed_global,
            query_global,
            observed_relative_xy,
            query_relative_xy,
            observed_chart,
            query_chart,
            nonoutage_observed,
        )
        detail_route = self.detail_router(
            observed_global,
            query_global,
            observed_relative_xy,
            query_relative_xy,
            observed_valid=nonoutage_observed,
        )
        power_route = self.power_router(
            observed_global,
            query_global,
            observed_relative_xy,
            query_relative_xy,
            observed_valid=nonoutage_observed,
        )

        def prepare_latent_route(route: dict[str, torch.Tensor]) -> None:
            selected_nonoutage = 1.0 - observed_outage[route["indices"]].float()
            latent_weights = route["weights"].float() * selected_nonoutage
            fallback = latent_weights.sum(dim=1, keepdim=True) <= 1e-8
            latent_weights = latent_weights / latent_weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            route["latent_weights"] = torch.where(
                fallback, route["weights"].float(), latent_weights
            ).to(route["weights"].dtype)
            route["latent_valid"] = (selected_nonoutage > 0.5) | fallback

        prepare_latent_route(spectrum_route)
        prepare_latent_route(detail_route)
        prior_condition = None
        if self.prior_encoder is not None:
            if (
                query_prior_pas is None
                or query_prior_pdp is None
                or query_prior_uncertainty is None
            ):
                raise ValueError("Structured spectral prior inputs are required")
            prior_condition = self.prior_encoder(
                query_prior_pas,
                query_prior_pdp,
                query_prior_uncertainty,
            )
        (
            spectrum,
            spectrum_features,
            spectrum_warp,
            spectrum_base,
            spectrum_residual,
            spectrum_warp_saturation,
            spectrum_token_effective,
            spectrum_token_top1,
        ) = self.spectrum_field(
            observed_spectrum,
            query_global,
            spectrum_route,
            cell_id,
            condition=prior_condition,
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
            detail_token_effective,
            detail_token_top1,
        ) = self.detail_field(
            observed_phase,
            query_global,
            detail_route,
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
        power_result = self.power_cnp(
            query_global,
            power_route["pair"],
            power_route["indices"],
            observed_power,
            observed_outage,
            cell_id,
        )
        return {
            "spectrum": spectrum.flatten(1),
            "phase": phase.flatten(1),
            "power": power_result["median"],
            "power_q10": power_result["q10"],
            "power_q90": power_result["q90"],
            "spectrum_base": spectrum_base.flatten(1),
            "phase_base": detail_base.flatten(1),
            "power_base": power_result["base"],
            "spectrum_residual": spectrum_residual.flatten(1),
            "phase_residual": detail_residual.flatten(1),
            "spectrum_residual_rms": spectrum_residual.float()
            .square()
            .mean(dim=(1, 2, 3, 4))
            .clamp_min(1e-12)
            .sqrt(),
            "phase_residual_rms": detail_residual.float()
            .square()
            .mean(dim=(1, 2, 3, 4))
            .clamp_min(1e-12)
            .sqrt(),
            "outage_logit": self.outage_head(summary).squeeze(1),
            "router_entropy": spectrum_route["entropy"],
            "router_top1_mass": spectrum_route["top1_mass"],
            "router_effective_neighbors": spectrum_route["effective_neighbors"],
            "router_distance": spectrum_route["mean_distance"],
            "detail_router_entropy": detail_route["entropy"],
            "detail_router_top1_mass": detail_route["top1_mass"],
            "detail_router_effective_neighbors": detail_route["effective_neighbors"],
            "detail_router_distance": detail_route["mean_distance"],
            "power_router_effective_neighbors": power_route["effective_neighbors"],
            "power_router_distance": power_route["mean_distance"],
            "spectrum_warp": spectrum_warp,
            "detail_warp": detail_warp,
            "spectrum_token_effective_neighbors": spectrum_token_effective,
            "detail_token_effective_neighbors": detail_token_effective,
            "spectrum_token_top1_mass": spectrum_token_top1,
            "detail_token_top1_mass": detail_token_top1,
            "power_effective_neighbors": power_result["effective_neighbors"],
            "power_uncertainty_width": power_result["q90"] - power_result["q10"],
            "query_chart": query_chart,
            "warp_saturation": 0.5
            * (spectrum_warp_saturation + detail_warp_saturation),
            "router_temperature": torch.full_like(
                spectrum_route["entropy"], self.router.temperature
            ),
        }
