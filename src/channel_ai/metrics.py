from __future__ import annotations

import torch

from .transforms import ChannelShape


def _array_view(channel: torch.Tensor, shape: ChannelShape) -> torch.Tensor:
    return channel.reshape(-1, shape.m_p, shape.m_v, shape.m_h, shape.n, shape.s)


def pas_spectrum(channel: torch.Tensor, shape: ChannelShape) -> torch.Tensor:
    array = _array_view(channel, shape)
    beam = torch.fft.fft2(array, dim=(2, 3), norm="ortho")
    power = beam.abs().square()
    return power.permute(0, 4, 5, 1, 2, 3).reshape(-1, shape.n, shape.s, shape.m)


def pdp_spectrum(channel: torch.Tensor) -> torch.Tensor:
    delay = torch.fft.ifft(channel, dim=-1, norm="ortho")
    return delay.abs().square()


def cosine_accuracy(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    pred = prediction.reshape(-1, prediction.shape[-1])
    true = target.reshape(-1, target.shape[-1])
    true_norm = torch.linalg.vector_norm(true, dim=-1)
    pred_norm = torch.linalg.vector_norm(pred, dim=-1)
    valid = true_norm > eps
    if not torch.any(valid):
        return prediction.new_tensor(0.0)
    cosine = (pred[valid] * true[valid]).sum(dim=-1)
    cosine = cosine / (pred_norm[valid].clamp_min(eps) * true_norm[valid].clamp_min(eps))
    return cosine.mean()


def pas_accuracy(prediction: torch.Tensor, target: torch.Tensor, shape: ChannelShape) -> torch.Tensor:
    return cosine_accuracy(pas_spectrum(prediction, shape), pas_spectrum(target, shape))


def pdp_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return cosine_accuracy(pdp_spectrum(prediction), pdp_spectrum(target))


def nmse(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    numerator = (prediction - target).abs().square().sum()
    denominator = target.abs().square().sum().clamp_min(eps)
    return numerator / denominator


def composite_score(pas: torch.Tensor, pdp: torch.Tensor, channel_nmse: torch.Tensor) -> torch.Tensor:
    return 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + channel_nmse)
