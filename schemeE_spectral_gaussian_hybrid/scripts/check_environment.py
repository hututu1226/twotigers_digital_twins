from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from scheme_e.config import load_config


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Scheme E data, dependencies and CUDA")
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--strict-boosters", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = Path(config["data"]["root"])
    required = (
        "Round2_Setup.json",
        "Round2_Map.ply",
        "Round2_Train_Pos.npy",
        "Round2_Train_Channel.npy",
        "Round2_Test_Pos.npy",
    )
    missing = [name for name in required if not (data_root / name).is_file()]
    checkpoint = Path(config["hybrid"]["autoencoder_checkpoint"])
    boosters = {
        name: importlib.util.find_spec(name) is not None for name in ("xgboost", "lightgbm")
    }
    cuda = torch.cuda.is_available()
    disk = shutil.disk_usage(config["_project_root"])
    report = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else None,
        "boosters": boosters,
        "missing_data_files": missing,
        "autoencoder_checkpoint": str(checkpoint),
        "autoencoder_checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "autoencoder_is_lfs_pointer": _is_lfs_pointer(checkpoint),
        "project_free_disk_gib": round(disk.free / 2**30, 2),
    }
    errors: list[str] = []
    if missing:
        errors.append(f"missing Round2 files: {missing}")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000:
        errors.append("AE best.pt is missing, too small, or still a Git LFS pointer")
    if args.require_cuda and not cuda:
        errors.append("CUDA is required but torch.cuda.is_available() is false")
    if args.strict_boosters and not all(boosters.values()):
        errors.append("xgboost and lightgbm are both required for the formal outage ensemble")
    if disk.free < 8 * 2**30:
        errors.append("less than 8 GiB free disk remains")
    if errors:
        report["status"] = "FAILED"
        report["errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
