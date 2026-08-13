from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from spatial_inpainting.config import load_config, save_json
from spatial_inpainting.data import load_metadata, split_indices


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose spatial smoothness of frozen AE latents")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = load_metadata(config)
    with np.load(config["encoding"]["output_path"]) as source:
        encoded = {key: source[key] for key in source.files}
    training_indices, validation_indices = split_indices(metadata, config)
    available = encoded.get("available", np.ones(len(metadata["train_cells"]), dtype=bool))
    usable = available & ~metadata["outage"]
    latent = (encoded["latent"] - encoded["latent_mean"]) / encoded["latent_std"]
    rng = np.random.default_rng(int(config["seed"]) + 404)
    cells: list[dict] = []
    for cell_id in range(int(metadata["train_cells"].max()) + 1):
        indices = np.flatnonzero(usable & (metadata["train_cells"] == cell_id))
        positions = metadata["train_positions"][indices, :2].astype(np.float64)
        square_distance = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(square_distance, np.inf)
        nearest_local = square_distance.argmin(axis=1)
        nearest_indices = indices[nearest_local]
        nearest_spatial = np.sqrt(square_distance[np.arange(len(indices)), nearest_local])
        nearest_euclidean = np.linalg.norm(latent[indices] - latent[nearest_indices], axis=1)
        nearest_cosine = cosine_similarity(latent[indices], latent[nearest_indices])

        random_local = rng.integers(0, len(indices), size=len(indices))
        same = random_local == np.arange(len(indices))
        random_local[same] = (random_local[same] + 1) % len(indices)
        random_indices = indices[random_local]
        random_spatial = np.linalg.norm(positions - positions[random_local], axis=1)
        random_euclidean = np.linalg.norm(latent[indices] - latent[random_indices], axis=1)
        random_cosine = cosine_similarity(latent[indices], latent[random_indices])
        cells.append(
            {
                "cell_id": cell_id,
                "samples": int(len(indices)),
                "nearest_spatial_meters": summarize(nearest_spatial),
                "nearest_latent_euclidean": summarize(nearest_euclidean),
                "nearest_latent_cosine": summarize(nearest_cosine),
                "random_spatial_meters": summarize(random_spatial),
                "random_latent_euclidean": summarize(random_euclidean),
                "random_latent_cosine": summarize(random_cosine),
                "euclidean_separation_ratio": float(
                    np.mean(random_euclidean) / max(np.mean(nearest_euclidean), 1e-12)
                ),
                "cosine_similarity_gap": float(
                    np.mean(nearest_cosine) - np.mean(random_cosine)
                ),
            }
        )
    report = {
        "encoded_path": config["encoding"]["output_path"],
        "training_samples": int(np.sum(usable[training_indices])),
        "validation_samples": int(np.sum(usable[validation_indices])),
        "interpretation": (
            "A ratio above 1 and a positive cosine gap mean nearby UEs are more similar in latent "
            "space than random same-cell pairs. This is a diagnostic only; no neighbor is used for prediction."
        ),
        "cells": cells,
    }
    if args.output:
        save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
