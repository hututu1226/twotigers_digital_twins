from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_f.config import choose_device, load_config, seed_everything
from scheme_f.context_data import ContextRepository
from scheme_f.context_diagnostics import run_diagnostics
from scheme_f.context_training import load_context_checkpoint
from scheme_f.data import balanced_limit, load_metadata, split_indices


def project_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose which Scheme F Context component limits validation score"
    )
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--encoded")
    parser.add_argument("--output-dir", default="artifacts/fold0/context_diagnostics")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Balanced validation sample limit; 0 uses the config/full fold",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--outage-threshold", type=float)
    parser.add_argument(
        "--skip-counterfactuals",
        action="store_true",
        help="Skip no-Warp and Router-prior inference ablations",
    )
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    config = load_config(args.config)
    if args.encoded:
        config["encoding"]["output_path"] = str(project_path(config, args.encoded))
    checkpoint_path = project_path(
        config,
        args.checkpoint or config["inference"]["context_checkpoint"],
    )
    output_dir = project_path(config, args.output_dir)
    encoded_path = Path(config["encoding"]["output_path"])
    for label, path in (
        ("Context checkpoint", checkpoint_path),
        ("encoded latent", encoded_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    seed_everything(int(config["seed"]))
    requested_device = args.device or str(config["runtime"].get("device", "auto"))
    device = choose_device(requested_device)
    amp = bool(config["runtime"].get("amp", True)) and device.type == "cuda"
    metadata = load_metadata(config)
    training_indices, validation_indices = split_indices(metadata, config)
    if not len(validation_indices):
        raise ValueError("Diagnostics require a validation fold; use fold0_5090.json")
    with np.load(encoded_path) as source:
        available = (
            source["available"].astype(bool)
            if "available" in source.files
            else np.ones(len(metadata["train_cells"]), dtype=bool)
        )
    training_indices = training_indices[available[training_indices]]
    validation_indices = validation_indices[available[validation_indices]]
    training_indices = balanced_limit(
        training_indices,
        config["runtime"].get(
            "context_train_limit", config["runtime"].get("train_limit")
        ),
        [metadata["train_cells"]],
        int(config["seed"]) + 3,
    )
    configured_limit = config["runtime"].get(
        "context_validation_limit", config["runtime"].get("validation_limit")
    )
    validation_limit = args.limit if args.limit > 0 else configured_limit
    validation_indices = balanced_limit(
        validation_indices,
        validation_limit,
        [metadata["train_cells"], metadata["outage"].astype(np.int8)],
        int(config["seed"]) + 4,
    )
    if not len(validation_indices):
        raise ValueError("No available validation samples remain after filtering")

    repository = ContextRepository(config, training_indices)
    model, autoencoder, shape, checkpoint = load_context_checkpoint(
        config,
        checkpoint_path,
        repository,
        device,
    )
    checkpoint_metrics = checkpoint.get("metrics", {})
    threshold = float(
        args.outage_threshold
        if args.outage_threshold is not None
        else checkpoint_metrics.get(
            "outage_threshold", config["context"].get("outage_threshold", 0.999)
        )
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("The outage threshold must lie in the open interval (0, 1)")

    print(
        json.dumps(
            {
                "device": str(device),
                "checkpoint": str(checkpoint_path),
                "encoded": str(encoded_path),
                "validation_samples": int(len(validation_indices)),
                "outage_threshold": threshold,
                "counterfactuals": not args.skip_counterfactuals,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    report = run_diagnostics(
        config,
        checkpoint_path,
        repository,
        model,
        autoencoder,
        shape,
        checkpoint,
        validation_indices,
        device,
        amp,
        output_dir,
        threshold,
        int(config["context"].get("validation_decode_batch_size", 8)),
        include_counterfactuals=not args.skip_counterfactuals,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "baseline_score": report["baseline"]["metrics"]["score"],
                **report["decision_signals"],
                "elapsed_seconds": report["elapsed_seconds"],
                "report": str(output_dir / "report.json"),
                "summary": str(output_dir / "SUMMARY.md"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
