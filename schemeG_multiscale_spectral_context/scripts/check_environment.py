from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from scheme_g.config import load_config


def _is_lfs_pointer(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size < 1024
        and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Scheme G data and CUDA environment"
    )
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = Path(config["data"]["root"])
    required = [
        "Round2_Setup.json",
        "Round2_Map.ply",
        "Round2_Train_Pos.npy",
        "Round2_Train_Channel.npy",
        "Round2_Test_Pos.npy",
    ]
    missing = [name for name in required if not (data_root / name).is_file()]
    ae_checkpoint = Path(config["context"]["autoencoder_checkpoint"])
    cuda_available = torch.cuda.is_available()
    disk = shutil.disk_usage(Path(config["_project_root"]))
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            if cuda_available
            else None
        ),
        "data_root": str(data_root),
        "missing_data_files": missing,
        "project_free_disk_gib": round(disk.free / 2**30, 2),
        "autoencoder_checkpoint": str(ae_checkpoint),
        "autoencoder_checkpoint_bytes": ae_checkpoint.stat().st_size
        if ae_checkpoint.is_file()
        else 0,
        "autoencoder_is_lfs_pointer": _is_lfs_pointer(ae_checkpoint),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit("Required Round2 data files are missing")
    if not ae_checkpoint.is_file() or ae_checkpoint.stat().st_size < 1_000_000:
        raise SystemExit("Scheme C AE best.pt is missing or still a Git LFS pointer")
    if disk.free < 8 * 2**30:
        raise SystemExit("At least 8 GiB free disk is required")
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")


if __name__ == "__main__":
    main()
