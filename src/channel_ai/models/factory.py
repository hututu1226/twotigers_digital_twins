from __future__ import annotations

import numpy as np
import torch

from ..data import load_manifest, load_metadata
from ..transforms import ChannelShape
from .scheme1 import Scheme1Model
from .scheme2 import Scheme2Model


def build_model(config: dict) -> tuple[torch.nn.Module, ChannelShape]:
    artifact_dir = config["data"]["artifacts"]
    manifest = load_manifest(artifact_dir)
    metadata = load_metadata(artifact_dir)
    setup = manifest["setup"]
    shape = ChannelShape.from_setup(setup)
    model_config = config["model"]
    common = {
        "shape": shape,
        "token_feature_dim": int(manifest["map"]["token_feature_dim"]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "fourier_bands": int(model_config.get("fourier_bands", 4)),
        "base_stations": torch.tensor(np.asarray(setup["X"], dtype=np.float32)),
        "position_center": torch.tensor(metadata["position_center"]),
        "position_scale": torch.tensor(metadata["position_scale"]),
        "power_mean": torch.tensor(metadata["power_mean"]),
        "power_std": torch.tensor(metadata["power_std"]),
    }
    if config["scheme"] == "scheme1":
        model = Scheme1Model(
            **common,
            latent_dim=int(model_config["latent_dim"]),
            base_channels=int(model_config["base_channels"]),
        )
    elif config["scheme"] == "scheme2":
        model = Scheme2Model(
            **common,
            token_count=int(model_config["token_count"]),
            model_dim=int(model_config["model_dim"]),
            attention_heads=int(model_config["attention_heads"]),
            transformer_layers=int(model_config["transformer_layers"]),
            feedforward_dim=int(model_config["feedforward_dim"]),
        )
    else:
        raise ValueError(f"Unknown scheme: {config['scheme']}")
    return model, shape

