from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_f.config import choose_device, load_config, save_json
from scheme_f.context_data import ContextRepository
from scheme_f.context_training import evaluate_context_model, load_context_checkpoint
from scheme_f.data import load_metadata, split_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Scheme F separately for each base station"
    )
    parser.add_argument("--config", default="configs/fold0_best.json")
    parser.add_argument("--checkpoint", default="artifacts/fold0/context/best.pt")
    parser.add_argument("--policy", default="artifacts/fold0/context/outage_scan.json")
    parser.add_argument("--output", default="reports/generated/fold0_breakdown.json")
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = load_metadata(config)
    training, validation = split_indices(metadata, config)
    with np.load(config["encoding"]["output_path"]) as source:
        available = source["available"].astype(bool)
    training = training[available[training]]
    validation = validation[available[validation]]
    repository = ContextRepository(config, training)
    device = choose_device(config["runtime"]["device"])
    model, autoencoder, shape, checkpoint = load_context_checkpoint(
        config, args.checkpoint, repository, device
    )
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    threshold = float(policy["best_threshold"])
    soft = float(policy.get("best_soft_strength", 0.0))
    prior = float(policy.get("best_spectral_prior_alpha", 0.0))
    groups: dict[str, dict] = {}
    for name, indices in {
        "all": validation,
        **{
            f"bs{cell_id}": validation[metadata["train_cells"][validation] == cell_id]
            for cell_id in range(repository.cell_count)
        },
    }.items():
        groups[name] = evaluate_context_model(
            model,
            autoencoder,
            repository,
            indices,
            shape,
            device,
            bool(config["runtime"].get("amp", True)),
            threshold,
            int(config["context"].get("validation_decode_batch_size", 8)),
            soft_outage_strength=soft,
            spectral_prior_alpha=prior,
        )
        groups[name]["samples"] = int(len(indices))
    finite = all(
        math.isfinite(float(values[key]))
        for values in groups.values()
        for key in ("pas", "pdp", "nmse", "score")
    )
    maximum_cell_nmse = max(
        float(values["nmse"])
        for name, values in groups.items()
        if name.startswith("bs")
    )
    report = {
        "status": "PASS" if finite and maximum_cell_nmse <= 10.0 else "FAILED",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "policy": {
            "outage_threshold": threshold,
            "soft_outage_strength": soft,
            "spectral_prior_alpha": prior,
        },
        "groups": groups,
        "maximum_cell_nmse": maximum_cell_nmse,
        "submission_warning": maximum_cell_nmse > 3.0,
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Per-base-station validation is non-finite or NMSE exceeds 10")


if __name__ == "__main__":
    main()
