from __future__ import annotations

import torch
import torch.nn.functional as functional

from .angle_delay import ChannelShape
from .metrics import nmse, pas_accuracy, pdp_accuracy


def angle_delay_power(angle_delay: torch.Tensor, shape: ChannelShape) -> torch.Tensor:
    parts = angle_delay.reshape(
        len(angle_delay),
        shape.m_p * shape.n,
        2,
        shape.m_v,
        shape.m_h,
        shape.s,
    )
    return parts.float().square().sum(dim=2)


def energy_weighted_angle_delay_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: ChannelShape,
    emphasis: float,
    maximum_weight: float,
) -> torch.Tensor:
    power = angle_delay_power(target, shape)
    normalized = power / power.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-12)
    weights = (1.0 + float(emphasis) * normalized.sqrt()).clamp_max(float(maximum_weight))
    weights = weights[:, :, None].expand(-1, -1, 2, -1, -1, -1)
    weights = weights.reshape_as(target)
    return ((prediction.float() - target.float()).square() * weights).sum() / weights.sum().clamp_min(1e-12)


def joint_power_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: ChannelShape,
) -> torch.Tensor:
    predicted_power = torch.log1p(angle_delay_power(prediction, shape))
    target_power = torch.log1p(angle_delay_power(target, shape))
    return functional.smooth_l1_loss(predicted_power, target_power)


def complex_coherence_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Encourage the reconstructed complex field to keep the target direction."""
    predicted = prediction.float().flatten(1)
    expected = target.float().flatten(1)
    coherence = functional.cosine_similarity(
        predicted, expected, dim=1, eps=1e-8
    )
    return (1.0 - coherence.clamp(-1.0, 1.0)).mean()


def energy_weighted_complex_direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: ChannelShape,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compare real/imaginary direction per angle-delay bin, weighted by target energy."""
    predicted = prediction.float().reshape(
        len(prediction),
        shape.m_p * shape.n,
        2,
        shape.m_v,
        shape.m_h,
        shape.s,
    )
    expected = target.float().reshape_as(predicted)
    predicted_norm = predicted.square().sum(dim=2).clamp_min(epsilon).sqrt()
    expected_power = expected.square().sum(dim=2)
    expected_norm = expected_power.clamp_min(epsilon).sqrt()
    cosine = (predicted * expected).sum(dim=2) / (
        predicted_norm * expected_norm
    )
    weights = expected_power.sqrt()
    valid = expected_power > epsilon
    weighted = (1.0 - cosine.clamp(-1.0, 1.0)) * weights
    return weighted.masked_fill(~valid, 0.0).sum() / weights.masked_fill(
        ~valid, 0.0
    ).sum().clamp_min(epsilon)


def metric_aligned_channel_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    shape: ChannelShape,
) -> dict[str, torch.Tensor]:
    channel_nmse = nmse(prediction, target)
    losses = {
        "pas": 1.0 - pas_accuracy(prediction, target, shape),
        "pdp": 1.0 - pdp_accuracy(prediction, target),
        "nmse": torch.log1p(channel_nmse),
    }
    # This term is exactly 1 - official_score, including the bounded NMSE term.
    losses["score"] = (
        0.4 * losses["pas"]
        + 0.4 * losses["pdp"]
        + 0.2 * channel_nmse / (1.0 + channel_nmse)
    )
    return losses


def weighted_sum(terms: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    total = next(iter(terms.values())).new_zeros(())
    for name, value in terms.items():
        total = total + float(weights.get(name, 0.0)) * value
    return total
