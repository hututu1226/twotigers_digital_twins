from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import torch

from scheme_e.config import choose_device, count_parameters, load_config
from scheme_e.hybrid_training import _build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Scheme E trainable architecture")
    parser.add_argument("--config", default="configs/fold0_5090.json")
    args = parser.parse_args()
    config = load_config(args.config)
    device = choose_device(str(config["runtime"].get("device", "auto")))
    model, shape = _build_model(config, device)
    forbidden = []
    largest_linear = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            elements = int(module.weight.numel())
            largest_linear = max(largest_linear, elements)
            if module.in_features >= 30_720 or module.out_features >= 30_720:
                forbidden.append(name)
    report = {
        "architecture": (
            "spectral_gaussian_dual_seed_transport_v3"
            if bool(config["hybrid"].get("transport_seed", {}).get("enabled", False))
            else
            "spectral_gaussian_reference_aware_v2"
            if bool(config["hybrid"].get("reference_aware", False))
            else "spectral_gaussian_full_resolution_adapter_v1"
        ),
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "raw_channel_shape": list(shape.raw_shape),
        "spectrum_shape": list(model.autoencoder.spectrum_shape.tensor_shape),
        "detail_shape": list(model.autoencoder.phase_shape.tensor_shape),
        "total_latent_elements": int(
            model.autoencoder.spectrum_shape.elements + model.autoencoder.phase_shape.elements
        ),
        "largest_linear_weight_elements": largest_linear,
        "full_latent_linear_layers": forbidden,
        "full_resolution_check": not forbidden,
        "dual_seed_transport": bool(model.condition_encoder.transport_dim),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if forbidden:
        raise SystemExit("Scheme E contains a forbidden full-latent linear bottleneck")


if __name__ == "__main__":
    main()
