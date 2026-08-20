from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from scheme_f.angle_delay import ChannelShape
from scheme_f.autoencoder_training import build_autoencoder
from scheme_f.config import load_config
from scheme_f.context_model import FullResolutionContextField


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect Scheme F latent and bottleneck sizes"
    )
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    setup = json.loads(
        (Path(config["data"]["root"]) / "Round2_Setup.json").read_text(encoding="utf-8")
    )
    shape = ChannelShape.from_setup(setup)
    autoencoder = build_autoencoder(config, shape)
    section = config["context"]
    query_numeric_channels = 9
    if bool(section.get("use_rf_geometry", True)):
        query_numeric_channels += 71
    if bool(section.get("spectral_prior", {}).get("enabled", False)):
        query_numeric_channels += int(
            section.get("spectral_prior", {}).get("fallback_channels", 168)
        )
    context = FullResolutionContextField(
        autoencoder.spectrum_shape.tensor_shape,
        autoencoder.phase_shape.tensor_shape,
        int(setup["Q"]),
        6 + 3 + int(setup["Q"]),
        query_numeric_channels,
        map_token_channels=int(section["map_token_channels"]),
        map_hidden_channels=int(section["map_hidden_channels"]),
        context_base_channels=int(section["base_channels"]),
        context_feature_channels=int(section["context_feature_channels"]),
        environment_base_channels=int(section.get("environment_base_channels", 32)),
        environment_feature_channels=int(section["environment_feature_channels"]),
        environment_blocks=int(section.get("environment_blocks", 2)),
        corridor_width=int(section.get("corridor_width", 96)),
        corridor_heads=int(section.get("corridor_heads", 4)),
        corridor_layers=int(section.get("corridor_layers", 2)),
        corridor_maximum_samples=int(section.get("corridor_maximum_samples", 32)),
        station_embedding_channels=int(section["station_embedding_channels"]),
        fourier_bands=int(section["fourier_bands"]),
        global_width=int(section["global_width"]),
        global_blocks=int(section["global_blocks"]),
        router_width=int(section.get("router_width", 128)),
        router_top_k=int(section.get("router_top_k", 96)),
        router_temperature=float(section.get("router_temperature_initial", 1.5)),
        router_uniform_mix=float(section.get("router_uniform_mix", 0.15)),
        router_dropout=float(section.get("router_dropout", 0.1)),
        route_bias_scale=float(section.get("route_bias_scale", 0.15)),
        chart_dimensions=int(section.get("chart_dimensions", 64)),
        chart_weight=float(section.get("chart_weight", 1.0)),
        pair_width=int(section.get("pair_width", 128)),
        spectrum_token_channels=int(section["spectrum_token_channels"]),
        detail_token_channels=int(section["detail_token_channels"]),
        attention_heads=int(section["attention_heads"]),
        attention_chunk_size=int(section["attention_chunk_size"]),
        refinement_blocks=int(section["refinement_blocks"]),
        axial_blocks=int(section.get("axial_blocks", 2)),
        operator_blocks=int(section.get("operator_blocks", 2)),
        operator_modes=tuple(section.get("operator_modes", [4, 6, 8])),
        token_top_k=int(section.get("token_top_k", 2)),
        regional_warp=bool(section.get("regional_warp", True)),
        detail_phase_rotation=bool(section.get("detail_phase_rotation", True)),
        spectrum_maximum_warp=tuple(
            section.get("spectrum_maximum_warp", [0.75, 1.5, 3.0])
        ),
        detail_maximum_warp=tuple(section.get("detail_maximum_warp", [1.5, 3.0, 6.0])),
        spectrum_maximum_residual=float(section.get("spectrum_maximum_residual", 1.0)),
        detail_maximum_residual=float(section.get("detail_maximum_residual", 1.0)),
        maximum_power_residual=float(section.get("maximum_power_residual", 1.0)),
        maximum_power_z=float(section.get("maximum_power_z", 5.0)),
        power_width=int(section.get("power_width", 192)),
        dropout=float(section.get("dropout", 0.05)),
        gradient_checkpointing=bool(section.get("gradient_checkpointing", True)),
    )
    total_latent = autoencoder.total_latent_dim
    linear_layers = [
        module for module in context.modules() if isinstance(module, torch.nn.Linear)
    ]
    flat_latent_layers = [
        {"in": layer.in_features, "out": layer.out_features}
        for layer in linear_layers
        if layer.in_features >= total_latent or layer.out_features >= total_latent
    ]
    report = {
        "autoencoder_architecture": config["autoencoder"].get(
            "architecture", "structured_v2"
        ),
        "context_architecture": section.get("architecture", "unknown"),
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
        "context_branch_parameters": {
            "map_fpn": parameter_count(context.context_fpn),
            "environment": parameter_count(context.environment_encoder)
            + parameter_count(context.corridor_encoder),
            "router": parameter_count(context.router),
            "spectrum_field": parameter_count(context.spectrum_field),
            "detail_field": parameter_count(context.detail_field),
            "power_cnp": parameter_count(context.power_cnp),
        },
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
        "transport_base_is_direct_full_resolution": True,
        "residual_heads_zero_initialized": bool(
            torch.count_nonzero(context.spectrum_field.output.weight) == 0
            and torch.count_nonzero(context.detail_field.output.weight) == 0
        ),
        "router": {
            "top_k": context.router.top_k,
            "temperature": context.router.temperature,
            "uniform_mix": context.router.uniform_mix,
            "route_bias_scale": context.spectrum_field.diagnostic_route_bias_scale,
            "chart_dimensions": context.chart_dimensions,
            "token_top_k": context.spectrum_field.token_top_k,
        },
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
