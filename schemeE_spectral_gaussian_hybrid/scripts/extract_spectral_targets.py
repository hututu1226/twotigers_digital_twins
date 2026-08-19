from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from scheme_e.angle_delay import ChannelShape
from scheme_e.config import choose_device, load_config
from scheme_e.data import load_manifest
from scheme_e.spectral_targets import extract_spectral_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA FFT extraction for Scheme E spectral targets")
    parser.add_argument("--config", default=str(_bootstrap.PROJECT_ROOT / "configs" / "fold0_5090.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["spectral"]["target_path"])
    if output.exists() and not args.force:
        print(json.dumps({"status": "SKIPPED", "reason": "target exists", "output_path": str(output)}, indent=2))
        return
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    device = choose_device(args.device or str(config["runtime"].get("device", "auto")))
    result = extract_spectral_dataset(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy",
        output,
        shape,
        proxy_count=int(config["spectral"].get("proxy_count", 24)),
        chunk_size=int(config["spectral"].get("chunk_size", 8)),
        device=device,
        storage_dtype=str(config["spectral"].get("storage_dtype", "float16")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
