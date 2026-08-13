from __future__ import annotations

import torch

from .angle_delay import ChannelShape


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


def cosine_accuracy(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    pred = prediction.reshape(-1, prediction.shape[-1])
    true = target.reshape(-1, target.shape[-1])
    pred_norm = torch.linalg.vector_norm(pred, dim=-1)
    true_norm = torch.linalg.vector_norm(true, dim=-1)
    valid = true_norm > epsilon
    if not torch.any(valid):
        return prediction.new_tensor(0.0)
    dot = (pred[valid] * true[valid]).sum(dim=-1)
    cosine = dot / pred_norm[valid].clamp_min(epsilon)
    cosine = cosine / true_norm[valid].clamp_min(epsilon)
    return cosine.mean()


def pas_accuracy(prediction: torch.Tensor, target: torch.Tensor, shape: ChannelShape) -> torch.Tensor:
    return cosine_accuracy(pas_spectrum(prediction, shape), pas_spectrum(target, shape))


def pdp_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return cosine_accuracy(pdp_spectrum(prediction), pdp_spectrum(target))


def nmse(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return (prediction - target).abs().square().sum() / target.abs().square().sum().clamp_min(epsilon)


def official_score(pas: float, pdp: float, channel_nmse: float) -> float:
    return 0.4 * pas + 0.4 * pdp + 0.2 / (1.0 + channel_nmse)


class ChannelMetricAccumulator:
    def __init__(self, shape: ChannelShape) -> None:
        self.shape = shape
        self.pas_sum = 0.0
        self.pdp_sum = 0.0
        self.nonzero_count = 0
        self.nmse_numerator = 0.0
        self.nmse_denominator = 0.0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor, true_outage: torch.Tensor) -> None:
        nonzero = ~true_outage.bool()
        if torch.any(nonzero):
            count = int(nonzero.sum().item())
            self.pas_sum += float(pas_accuracy(prediction[nonzero], target[nonzero], self.shape).cpu()) * count
            self.pdp_sum += float(pdp_accuracy(prediction[nonzero], target[nonzero]).cpu()) * count
            self.nonzero_count += count
        self.nmse_numerator += float((prediction - target).abs().square().sum().cpu())
        self.nmse_denominator += float(target.abs().square().sum().cpu())

    def compute(self) -> dict[str, float]:
        pas = self.pas_sum / max(self.nonzero_count, 1)
        pdp = self.pdp_sum / max(self.nonzero_count, 1)
        channel_nmse = self.nmse_numerator / max(self.nmse_denominator, 1e-30)
        return {
            "pas": pas,
            "pdp": pdp,
            "nmse": channel_nmse,
            "score": official_score(pas, pdp, channel_nmse),
        }
