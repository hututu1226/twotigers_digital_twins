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

from scheme_c.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Scheme C data and CUDA environment")
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
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit("Required Round2 data files are missing")
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")


if __name__ == "__main__":
    main()
