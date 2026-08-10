from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChannelShape:
    m: int
    m_h: int
    m_v: int
    m_p: int
    n: int
    s: int

    @classmethod
    def from_setup(cls, setup: dict) -> "ChannelShape":
        shape = cls(
            m=int(setup["M"]),
            m_h=int(setup["M_H"]),
            m_v=int(setup["M_V"]),
            m_p=int(setup["M_P"]),
            n=int(setup["N"]),
            s=int(setup["S"]),
        )
        if shape.m != shape.m_h * shape.m_v * shape.m_p:
            raise ValueError("M does not equal M_H * M_V * M_P")
        return shape

    @property
    def ad_channels(self) -> int:
        return 2 * self.m_p * self.n

    @property
    def raw_shape(self) -> tuple[int, int, int]:
        return self.m, self.n, self.s

    @property
    def ad_shape(self) -> tuple[int, int, int, int]:
        return self.ad_channels, self.m_v, self.m_h, self.s


def channel_to_angle_delay(channel: torch.Tensor, shape: ChannelShape) -> torch.Tensor:
    """Convert [B,M,N,S] complex H to [B,2*Mp*N,Mv,Mh,S] real channels."""
    if channel.ndim != 4 or tuple(channel.shape[1:]) != shape.raw_shape:
        raise ValueError(f"Expected [B,{shape.m},{shape.n},{shape.s}], got {tuple(channel.shape)}")
    if not torch.is_complex(channel):
        raise TypeError("channel_to_angle_delay expects a complex tensor")

    # The competition file flattens the BS array as [Mp, Mv, Mh]. The layout is
    # configurable at the project boundary if organizer clarification changes it.
    array = channel.reshape(-1, shape.m_p, shape.m_v, shape.m_h, shape.n, shape.s)
    beam = torch.fft.fft2(array, dim=(2, 3), norm="ortho")
    angle_delay = torch.fft.ifft(beam, dim=-1, norm="ortho")
    angle_delay = angle_delay.permute(0, 1, 4, 2, 3, 5).contiguous()
    stacked = torch.view_as_real(angle_delay)
    stacked = stacked.permute(0, 1, 2, 6, 3, 4, 5).contiguous()
    return stacked.reshape(-1, shape.ad_channels, shape.m_v, shape.m_h, shape.s)


def angle_delay_to_channel(angle_delay: torch.Tensor, shape: ChannelShape) -> torch.Tensor:
    """Invert channel_to_angle_delay and return [B,M,N,S] complex H."""
    if angle_delay.ndim != 5 or tuple(angle_delay.shape[1:]) != shape.ad_shape:
        raise ValueError(
            f"Expected [B,{shape.ad_channels},{shape.m_v},{shape.m_h},{shape.s}], "
            f"got {tuple(angle_delay.shape)}"
        )
    parts = angle_delay.reshape(
        -1, shape.m_p, shape.n, 2, shape.m_v, shape.m_h, shape.s
    )
    parts = parts.permute(0, 1, 2, 4, 5, 6, 3).contiguous()
    complex_ad = torch.view_as_complex(parts)
    complex_ad = complex_ad.permute(0, 1, 3, 4, 2, 5).contiguous()
    frequency_beam = torch.fft.fft(complex_ad, dim=-1, norm="ortho")
    channel_array = torch.fft.ifft2(frequency_beam, dim=(2, 3), norm="ortho")
    return channel_array.reshape(-1, shape.m, shape.n, shape.s)


def channel_power(channel: torch.Tensor) -> torch.Tensor:
    return channel.abs().square().mean(dim=(1, 2, 3))


def normalize_angle_delay(angle_delay: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    # Two real channels represent one complex value, hence the factor of two.
    complex_power = 2.0 * angle_delay.square().mean(dim=(1, 2, 3, 4), keepdim=True)
    return angle_delay / complex_power.clamp_min(eps).sqrt()


def scaled_angle_delay(
    normalized_angle_delay: torch.Tensor, log10_power: torch.Tensor
) -> torch.Tensor:
    amplitude = torch.pow(10.0, 0.5 * log10_power).reshape(-1, 1, 1, 1, 1)
    return normalized_angle_delay * amplitude
