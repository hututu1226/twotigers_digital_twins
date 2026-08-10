from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from channel_ai.config import load_config
from channel_ai.inference import run_inference
from channel_ai.preprocessing import run_preprocessing
from channel_ai.training import train_from_config


def ensure_preprocessed() -> None:
    manifest = PROJECT_ROOT / "artifacts" / "preprocessed" / "manifest.json"
    if manifest.exists():
        return
    run_preprocessing(
        PROJECT_ROOT / "Round2_Map",
        PROJECT_ROOT / "artifacts" / "preprocessed",
        resolution=4.0,
        link_samples=16,
        local_grid=3,
        local_radius=16.0,
        validation_fraction=0.15,
        blocks_per_cell=12,
        chunk_size=16,
        seed=2026,
    )


def verify_output(path: Path, expected_samples: int) -> None:
    channel = np.load(path, mmap_mode="r")
    expected = (expected_samples, 256, 4, 192)
    if channel.shape != expected:
        raise RuntimeError(f"Expected {expected}, got {channel.shape}")
    if channel.dtype != np.complex64:
        raise RuntimeError(f"Expected complex64, got {channel.dtype}")
    if not np.isfinite(channel).all():
        raise RuntimeError(f"Non-finite values found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both end-to-end CPU smoke tests")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    ensure_preprocessed()
    smoke_dir = PROJECT_ROOT / "artifacts" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for scheme in ("scheme1", "scheme2"):
        config_path = PROJECT_ROOT / "configs" / f"{scheme}_smoke.json"
        config = load_config(config_path)
        checkpoint = train_from_config(config, device_override=args.device)
        output = smoke_dir / f"{scheme}_test_channel_{args.samples}.npy"
        run_inference(config, checkpoint, output, args.device, limit=args.samples)
        verify_output(output, args.samples)
        print(f"PASS {scheme}: {output}")
        gc.collect()
    print("PASS all smoke tests")


if __name__ == "__main__":
    main()

