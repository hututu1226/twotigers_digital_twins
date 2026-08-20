from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_f.autoencoder_training import train_autoencoder
from scheme_f.config import load_config
from scheme_f.context_training import train_context_model
from scheme_f.encoding import encode_training_set
from scheme_f.inference import generate_test_channels
from scheme_f.preprocessing import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Scheme F end to end on a tiny sample"
    )
    parser.add_argument(
        "--config", default=str(_bootstrap.PROJECT_ROOT / "configs" / "smoke.json")
    )
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["runtime"]["device"] = args.device
        config["runtime"]["amp"] = args.device != "cpu"
    manifest = Path(config["preprocessing"]["artifact_dir"]) / "manifest.json"
    if not manifest.exists() or args.force_preprocess:
        print("[1/5] dual-resolution preprocessing", flush=True)
        preprocess_dataset(config, force=args.force_preprocess)
    else:
        print("[1/5] preprocessing already exists", flush=True)
    print("[2/5] factorized residual AE v4", flush=True)
    autoencoder = train_autoencoder(config)
    print("[3/5] per-cell latent encoding", flush=True)
    encoding = encode_training_set(config)
    print("[4/5] query-adaptive Top2 transport + operator Context", flush=True)
    context = train_context_model(config)
    print("[5/5] test inference", flush=True)
    inference = generate_test_channels(config)
    output = np.load(inference["output_path"], mmap_mode="r")
    expected_count = int(config["runtime"].get("test_limit") or 500)
    expected = (expected_count, 256, 4, 192)
    finite = bool(np.isfinite(output).all())
    if output.shape != expected or output.dtype != np.complex64 or not finite:
        raise RuntimeError(
            f"Smoke output invalid: shape={output.shape}, dtype={output.dtype}, finite={finite}"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "autoencoder": autoencoder,
                "encoding": encoding,
                "context": context,
                "inference": inference,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
