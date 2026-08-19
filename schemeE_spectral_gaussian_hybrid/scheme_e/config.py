from __future__ import annotations

import contextlib
import json
import os
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def load_config(path: str | Path) -> dict:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    project_root = source.parent.parent if source.parent.name == "configs" else source.parent
    path_fields = (
        ("data", "root"),
        ("preprocessing", "artifact_dir"),
        ("autoencoder", "output_dir"),
        ("encoding", "autoencoder_checkpoint"),
        ("encoding", "output_path"),
        ("spectral", "target_path"),
        ("spectral_teacher", "oof_output_path"),
        ("spectral_teacher", "oof_report_path"),
        ("spectral_teacher", "test_output_path"),
        ("spectral_teacher", "model_path"),
        ("spectral_teacher", "final_report_path"),
        ("hybrid", "autoencoder_checkpoint"),
        ("hybrid", "output_dir"),
        ("hybrid", "initial_checkpoint"),
        ("hybrid_final", "output_dir"),
        ("hybrid_final", "initial_checkpoint"),
        ("inference", "checkpoint"),
        ("inference", "report_path"),
        ("context", "output_dir"),
        ("context", "autoencoder_checkpoint"),
        ("joint", "output_dir"),
        ("joint", "context_checkpoint"),
        ("inference", "context_checkpoint"),
        ("inference", "output_path"),
    )
    for section, key in path_fields:
        if section not in config or key not in config[section]:
            continue
        value = Path(config[section][key])
        if not value.is_absolute():
            config[section][key] = str((project_root / value).resolve())
    config["_config_path"] = str(source)
    config["_project_root"] = str(project_root)
    return config


def save_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    name = requested.lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(name)


def worker_count(configured: int) -> int:
    if configured >= 0:
        return configured
    return min(4, max(0, (os.cpu_count() or 1) - 1))


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    active = device.type == "cuda" and enabled
    try:
        return torch.amp.GradScaler(device.type, enabled=active)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=active)


@contextlib.contextmanager
def evaluating(module: torch.nn.Module) -> Iterator[None]:
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
