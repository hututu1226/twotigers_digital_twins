from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from scheme_c.angle_delay import ChannelShape
from scheme_c.autoencoder_training import build_autoencoder
from scheme_c.config import load_config
from scheme_c.context_model import FullResolutionContextField


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Scheme C latent and bottleneck sizes")
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    setup = json.loads(
        (Path(config["data"]["root"]) / "Round2_Setup.json").read_text(
            encoding="utf-8"
        )
    )
    shape = ChannelShape.from_setup(setup)
    autoencoder = build_autoencoder(config, shape)
    section = config["context"]
    context = FullResolutionContextField(
        autoencoder.spectrum_shape.tensor_shape,
        autoencoder.phase_shape.tensor_shape,
        int(setup["Q"]),
        6 + 3 + int(setup["Q"]),
        9,
        map_token_channels=int(section["map_token_channels"]),
        map_hidden_channels=int(section["map_hidden_channels"]),
        context_base_channels=int(section["base_channels"]),
        context_feature_channels=int(section["context_feature_channels"]),
        environment_feature_channels=int(section["environment_feature_channels"]),
        station_embedding_channels=int(section["station_embedding_channels"]),
        fourier_bands=int(section["fourier_bands"]),
        global_width=int(section["global_width"]),
        global_blocks=int(section["global_blocks"]),
        spectrum_token_channels=int(section["spectrum_token_channels"]),
        detail_token_channels=int(section["detail_token_channels"]),
        attention_heads=int(section["attention_heads"]),
        attention_chunk_size=int(section["attention_chunk_size"]),
        refinement_blocks=int(section["refinement_blocks"]),
        dropout=float(section.get("dropout", 0.05)),
        gradient_checkpointing=bool(section.get("gradient_checkpointing", True)),
    )
    total_latent = autoencoder.total_latent_dim
    linear_layers = [module for module in context.modules() if isinstance(module, torch.nn.Linear)]
    flat_latent_layers = [
        {"in": layer.in_features, "out": layer.out_features}
        for layer in linear_layers
        if layer.in_features >= total_latent or layer.out_features >= total_latent
    ]
    report = {
        "autoencoder_architecture": config["autoencoder"].get("architecture", "structured_v2"),
        "autoencoder_parameters": parameter_count(autoencoder),
        "autoencoder_branch_parameters": {
            "spectrum_encoder": parameter_count(autoencoder.spectrum_encoder),
            "detail_encoder": parameter_count(autoencoder.phase_encoder),
            "spectrum_decoder": parameter_count(autoencoder.decoder.spectrum_decoder)
            if hasattr(autoencoder.decoder, "spectrum_decoder")
            else None,
            "detail_decoder": parameter_count(autoencoder.decoder.detail_decoder)
            if hasattr(autoencoder.decoder, "detail_decoder")
            else None,
        },
        "context_parameters": parameter_count(context),
        "spectrum_shape": list(autoencoder.spectrum_shape.tensor_shape),
        "spectrum_elements": autoencoder.spectrum_latent_dim,
        "detail_shape": list(autoencoder.phase_shape.tensor_shape),
        "detail_elements": autoencoder.phase_latent_dim,
        "total_latent_elements": total_latent,
        "full_latent_linear_layers": flat_latent_layers,
        "largest_linear_weight_elements": max(
            layer.in_features * layer.out_features for layer in linear_layers
        ),
        "full_resolution_check": not flat_latent_layers
        and math.prod(context.spectrum_shape) == autoencoder.spectrum_latent_dim
        and math.prod(context.phase_shape) == autoencoder.phase_latent_dim,
    }
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["full_resolution_check"]:
        raise SystemExit("Full-resolution architecture check failed")


if __name__ == "__main__":
    main()
