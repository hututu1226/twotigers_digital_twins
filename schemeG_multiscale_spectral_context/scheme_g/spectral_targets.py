from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .angle_delay import ChannelShape


PAS_LOG_SCALE = 1_000.0
PDP_LOG_SCALE = 1_000.0


def spectral_feature_shapes(
    shape: ChannelShape, proxy_count: int = 24
) -> dict[str, tuple[int, ...]]:
    if shape.s % proxy_count:
        raise ValueError(f"S={shape.s} must be divisible by proxy_count={proxy_count}")
    return {
        "pas_proxy": (proxy_count, shape.m_v, shape.m_h),
        "pas_mean": (shape.m_v, shape.m_h),
        "pas_log": ((proxy_count + 1) * shape.m_v * shape.m_h,),
        "pdp": (shape.n, shape.s),
        "pdp_log": (shape.n * shape.s,),
        "ue_log_energy": (shape.n,),
    }


def _normalize_distribution(
    value: torch.Tensor, dimensions: tuple[int, ...]
) -> torch.Tensor:
    return value / value.sum(dim=dimensions, keepdim=True).clamp_min(1e-30)


def channel_spectral_targets(
    channel: torch.Tensor,
    shape: ChannelShape,
    proxy_count: int = 24,
) -> dict[str, torch.Tensor]:
    if channel.ndim != 4 or tuple(channel.shape[1:]) != shape.raw_shape:
        raise ValueError(
            f"Expected [B,{shape.m},{shape.n},{shape.s}], got {tuple(channel.shape)}"
        )
    if not torch.is_complex(channel):
        raise TypeError("channel_spectral_targets expects a complex tensor")
    feature_shapes = spectral_feature_shapes(shape, proxy_count)
    array = channel.reshape(-1, shape.m_p, shape.m_v, shape.m_h, shape.n, shape.s)
    beam = torch.fft.fft2(array, dim=(2, 3), norm="ortho")
    angle_power = beam.abs().square().float().mean(dim=(1, 4)).permute(0, 3, 1, 2)
    group_size = shape.s // proxy_count
    proxy = angle_power.reshape(-1, proxy_count, group_size, shape.m_v, shape.m_h).mean(
        dim=2
    )
    proxy = _normalize_distribution(proxy, (2, 3))
    mean_pas = _normalize_distribution(angle_power.mean(dim=1), (1, 2))
    pas_linear = torch.cat([proxy.flatten(1), mean_pas.flatten(1)], dim=1)
    pas_log = torch.log1p(PAS_LOG_SCALE * pas_linear)

    delay = torch.fft.ifft(channel, dim=-1, norm="ortho")
    delay_power = (
        delay.abs()
        .square()
        .float()
        .reshape(-1, shape.m_p, shape.m_v, shape.m_h, shape.n, shape.s)
    )
    pdp = delay_power.mean(dim=(1, 2, 3))
    pdp = _normalize_distribution(pdp, (2,))
    pdp_log = torch.log1p(PDP_LOG_SCALE * pdp.flatten(1))

    ue_energy = channel.abs().square().float().mean(dim=(1, 3))
    total_power = ue_energy.mean(dim=1)
    outage = total_power <= 1e-30
    ue_log_energy = torch.log10(ue_energy.clamp_min(1e-30))
    ue_log_energy = torch.where(
        outage[:, None], torch.zeros_like(ue_log_energy), ue_log_energy
    )
    log_power = torch.log10(total_power.clamp_min(1e-30))
    log_power = torch.where(outage, torch.zeros_like(log_power), log_power)
    return {
        "pas_log": pas_log.reshape(-1, *feature_shapes["pas_log"]),
        "pdp_log": pdp_log.reshape(-1, *feature_shapes["pdp_log"]),
        "ue_log_energy": ue_log_energy,
        "log_power": log_power,
        "outage": outage,
    }


def decode_pas_log(
    pas_log: torch.Tensor,
    shape: ChannelShape,
    proxy_count: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = spectral_feature_shapes(shape, proxy_count)["pas_log"][0]
    if pas_log.shape[-1] != expected:
        raise ValueError(
            f"Expected PAS feature width {expected}, got {pas_log.shape[-1]}"
        )
    linear = torch.expm1(pas_log.float()).clamp_min(0.0) / PAS_LOG_SCALE
    proxy_elements = proxy_count * shape.m_v * shape.m_h
    proxy = linear[..., :proxy_elements].reshape(-1, proxy_count, shape.m_v, shape.m_h)
    mean_pas = linear[..., proxy_elements:].reshape(-1, shape.m_v, shape.m_h)
    proxy = _normalize_distribution(proxy, (2, 3))
    mean_pas = _normalize_distribution(mean_pas, (1, 2))
    return proxy, mean_pas


def decode_pdp_log(pdp_log: torch.Tensor, shape: ChannelShape) -> torch.Tensor:
    expected = shape.n * shape.s
    if pdp_log.shape[-1] != expected:
        raise ValueError(
            f"Expected PDP feature width {expected}, got {pdp_log.shape[-1]}"
        )
    pdp = torch.expm1(pdp_log.float()).clamp_min(0.0) / PDP_LOG_SCALE
    pdp = pdp.reshape(-1, shape.n, shape.s)
    return _normalize_distribution(pdp, (2,))


def extract_spectral_dataset(
    channel_path: str | Path,
    output_path: str | Path,
    shape: ChannelShape,
    proxy_count: int = 24,
    chunk_size: int = 8,
    device: torch.device | str = "cpu",
    limit: int = 0,
    storage_dtype: str = "float16",
) -> dict[str, object]:
    started = time.perf_counter()
    channels = np.load(channel_path, mmap_mode="r")
    count = min(len(channels), int(limit)) if limit else len(channels)
    feature_shapes = spectral_feature_shapes(shape, proxy_count)
    pas = np.empty((count, feature_shapes["pas_log"][0]), dtype=np.float32)
    pdp = np.empty((count, feature_shapes["pdp_log"][0]), dtype=np.float32)
    ue = np.empty((count, shape.n), dtype=np.float32)
    power = np.empty(count, dtype=np.float32)
    outage = np.empty(count, dtype=np.bool_)
    target_device = torch.device(device)
    for start in range(0, count, int(chunk_size)):
        stop = min(start + int(chunk_size), count)
        block = torch.as_tensor(
            np.array(channels[start:stop], copy=True), device=target_device
        )
        targets = channel_spectral_targets(block, shape, proxy_count)
        pas[start:stop] = targets["pas_log"].cpu().numpy()
        pdp[start:stop] = targets["pdp_log"].cpu().numpy()
        ue[start:stop] = targets["ue_log_energy"].cpu().numpy()
        power[start:stop] = targets["log_power"].cpu().numpy()
        outage[start:stop] = targets["outage"].cpu().numpy()
    dtype = np.float16 if storage_dtype == "float16" else np.float32
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        pas_log=pas.astype(dtype),
        pdp_log=pdp.astype(dtype),
        ue_log_energy=ue.astype(np.float32),
        log_power=power,
        outage=outage,
        proxy_count=np.asarray(proxy_count, dtype=np.int32),
    )
    return {
        "output_path": str(destination),
        "samples": int(count),
        "pas_width": int(pas.shape[1]),
        "pdp_width": int(pdp.shape[1]),
        "outages": int(outage.sum()),
        "storage_dtype": str(dtype),
        "elapsed_seconds": time.perf_counter() - started,
    }
