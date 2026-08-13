from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from spatial_inpainting.autoencoder_training import train_autoencoder
from spatial_inpainting.config import load_config
from spatial_inpainting.encoding import encode_training_set
from spatial_inpainting.inference import generate_test_channels
from spatial_inpainting.preprocessing import preprocess_dataset
from spatial_inpainting.spatial_training import train_spatial_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both Scheme A stages on a tiny CPU sample")
    parser.add_argument("--config", default=str(_bootstrap.PROJECT_ROOT / "configs" / "smoke.json"))
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["runtime"]["device"] = args.device
    manifest = Path(config["preprocessing"]["artifact_dir"]) / "manifest.json"
    if not manifest.exists() or args.force_preprocess:
        print("[1/5] preprocessing", flush=True)
        preprocess_dataset(config, force=args.force_preprocess)
    else:
        print("[1/5] preprocessing already exists", flush=True)
    print("[2/5] autoencoder training", flush=True)
    autoencoder = train_autoencoder(config)
    print("[3/5] latent encoding", flush=True)
    encoding = encode_training_set(config)
    print("[4/5] spatial U-Net training", flush=True)
    spatial = train_spatial_model(config)
    print("[5/5] test inference", flush=True)
    inference = generate_test_channels(config)
    output = np.load(inference["output_path"])
    expected = (int(config["runtime"]["test_limit"]), 256, 4, 192)
    if output.shape != expected or output.dtype != np.complex64 or not np.isfinite(output).all():
        raise RuntimeError(
            f"Smoke output invalid: shape={output.shape}, dtype={output.dtype}, finite={np.isfinite(output).all()}"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "autoencoder": autoencoder,
                "encoding": encoding,
                "spatial": spatial,
                "inference": inference,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
