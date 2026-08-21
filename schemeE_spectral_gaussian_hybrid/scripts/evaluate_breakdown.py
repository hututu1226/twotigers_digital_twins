from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from scipy.spatial import cKDTree

from scheme_e.config import choose_device, load_config, save_json
from scheme_e.hybrid_training import evaluate_hybrid, load_hybrid_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Scheme E by BS and support distance")
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--output", default="reports/generated/fold0_breakdown.json")
    args = parser.parse_args()
    config = load_config(args.config)
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    with np.load(artifact_dir / "metadata.npz") as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(config["spectral_teacher"]["oof_output_path"]) as source:
        priors = {name: source[name] for name in source.files}
    fold = int(config["split"]["validation_fold"])
    validation = np.flatnonzero(metadata["validation_masks"][fold])
    observed = np.flatnonzero(~metadata["validation_masks"][fold])
    distance = np.full(len(metadata["train_positions"]), np.inf, dtype=np.float32)
    for cell in np.unique(metadata["train_cells"]):
        target = validation[metadata["train_cells"][validation] == cell]
        support = observed[
            (metadata["train_cells"][observed] == cell) & ~metadata["outage"][observed]
        ]
        distance[target] = cKDTree(metadata["train_positions"][support, :2]).query(
            metadata["train_positions"][target, :2], k=1
        )[0]
    device = choose_device(str(config["runtime"].get("device", "auto")))
    model, shape, checkpoint = load_hybrid_checkpoint(
        config, Path(config["hybrid"]["output_dir"]) / "best.pt", device
    )
    channels = np.load(Path(config["data"]["root"]) / "Round2_Train_Channel.npy", mmap_mode="r")
    threshold = float(checkpoint["outage_threshold"])

    def score(indices: np.ndarray) -> dict:
        if not len(indices):
            return {"samples": 0}
        return evaluate_hybrid(
            model, shape, channels, metadata, priors, indices, observed,
            np.asarray(checkpoint["geometry_mean"]), np.asarray(checkpoint["geometry_std"]),
            device, int(config["hybrid"].get("validation_batch_size", 2)), threshold,
            int(json.loads((Path(config["hybrid"]["output_dir"]) / "summary.json").read_text())["selected_projection_iterations"]),
        )

    report = {
        "all": score(validation),
        "by_cell": {
            str(int(cell)): score(validation[metadata["train_cells"][validation] == cell])
            for cell in np.unique(metadata["train_cells"])
        },
        "by_nearest_support_distance_meters": {
            "0_to_6": score(validation[distance[validation] < 6.0]),
            "6_to_12": score(validation[(distance[validation] >= 6.0) & (distance[validation] < 12.0)]),
            "12_plus": score(validation[distance[validation] >= 12.0]),
        },
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
