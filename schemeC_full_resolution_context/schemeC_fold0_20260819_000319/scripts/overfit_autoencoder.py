from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from scheme_c.angle_delay import (
    ChannelShape,
    channel_to_shape_target,
    shape_to_channel,
)
from scheme_c.autoencoder import FactorizedResidualAutoencoder
from scheme_c.autoencoder_training import (
    build_autoencoder,
    factorized_autoencoder_batch,
)
from scheme_c.config import (
    autocast_context,
    choose_device,
    load_config,
    make_grad_scaler,
    save_json,
    seed_everything,
)
from scheme_c.data import balanced_limit, load_manifest, load_metadata, split_indices
from scheme_c.losses import weighted_sum
from scheme_c.metrics import ChannelMetricAccumulator


@torch.no_grad()
def evaluate_training_samples(
    model: FactorizedResidualAutoencoder,
    channels: torch.Tensor,
    shape: ChannelShape,
    device: torch.device,
    amp: bool,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    metrics = ChannelMetricAccumulator(shape)
    coarse_metrics = ChannelMetricAccumulator(shape)
    for start in range(0, len(channels), batch_size):
        channel = channels[start : start + batch_size].to(device)
        target_shape, log_power, outage = channel_to_shape_target(channel, shape)
        with autocast_context(device, amp):
            spectrum, detail = model.encode(target_shape)
            prediction_shape = model.decode(spectrum, detail)
            coarse_shape = model.decode(spectrum, None)
        prediction = shape_to_channel(prediction_shape, log_power, shape, outage)
        coarse = shape_to_channel(coarse_shape, log_power, shape, outage)
        metrics.update(prediction, channel, outage)
        coarse_metrics.update(coarse, channel, outage)
    result = metrics.compute()
    coarse = coarse_metrics.compute()
    result["coarse_only_score"] = coarse["score"]
    result["detail_gain"] = result["score"] - coarse["score"]
    model.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prove AE capacity by deliberately overfitting a fixed tiny set"
    )
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--minimum-score", type=float, required=True)
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("samples, steps, and batch-size must be positive")

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = choose_device(config["runtime"]["device"])
    amp = bool(config["runtime"].get("amp", True))
    metadata = load_metadata(config)
    manifest = load_manifest(config)
    shape = ChannelShape.from_setup(manifest["setup"])
    training_indices, _ = split_indices(metadata, config)
    training_indices = training_indices[~metadata["outage"][training_indices]]
    selected = balanced_limit(
        training_indices,
        args.samples,
        [metadata["train_cells"]],
        int(config["seed"]) + 700,
    )
    channel_array = np.load(
        Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r"
    )
    channels = torch.from_numpy(np.array(channel_array[selected], copy=True))
    model = build_autoencoder(config, shape).to(device)
    if not isinstance(model, FactorizedResidualAutoencoder):
        raise TypeError("Capacity gate requires factorized_residual_v4")
    model.set_trainable_stage("joint")
    model.train()
    section = config["autoencoder"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(section.get("weight_decay", 1e-5)),
    )
    scaler = make_grad_scaler(device, amp)
    rng = np.random.default_rng(int(config["seed"]) + args.samples)
    started = time.perf_counter()
    last_metrics: dict[str, float] = {}
    for step in range(args.steps):
        batch_count = min(args.batch_size, len(channels))
        selected_batch = rng.choice(len(channels), size=batch_count, replace=False)
        channel = channels[selected_batch].to(device)
        target_shape, log_power, _ = channel_to_shape_target(channel, shape)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            _, terms = factorized_autoencoder_batch(
                model,
                target_shape,
                log_power,
                channel,
                shape,
                section,
                "joint",
            )
            loss = weighted_sum(terms, section["loss_weights"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite capacity loss at step {step + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(section.get("gradient_clip", 2.0))
        )
        scaler.step(optimizer)
        scaler.update()
        should_report = (
            (step + 1) % max(1, args.report_interval) == 0
            or step == 0
            or step + 1 == args.steps
        )
        if should_report:
            last_metrics = evaluate_training_samples(
                model,
                channels,
                shape,
                device,
                amp,
                args.batch_size,
            )
            print(
                f"capacity samples={len(channels)} step={step + 1}/{args.steps} "
                f"loss={float(loss.detach()):.6f} score={last_metrics['score']:.6f} "
                f"nmse={last_metrics['nmse']:.6f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    passed = float(last_metrics["score"]) >= float(args.minimum_score)
    report = {
        "status": "PASS" if passed else "FAIL",
        "architecture": str(section.get("architecture")),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "total_latent_dim": model.total_latent_dim,
        "samples": len(channels),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "minimum_score": args.minimum_score,
        "metrics": last_metrics,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "selected_indices": selected.tolist(),
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(
            f"AE capacity gate failed: {last_metrics['score']:.6f} < {args.minimum_score:.6f}"
        )


if __name__ == "__main__":
    main()
