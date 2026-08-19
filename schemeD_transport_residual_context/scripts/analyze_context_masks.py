from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_d.config import load_config, save_json
from scheme_d.context_data import ContextRepository
from scheme_d.data import load_metadata, split_indices
from scheme_d.preprocessing import preprocess_dataset


def ensure_preprocessing(config: dict) -> bool:
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    metadata_path = artifact_dir / "metadata.npz"
    if metadata_path.is_file():
        return False

    manifest_path = artifact_dir / "manifest.json"
    print(
        f"Preprocessing metadata is missing; building {metadata_path} before mask analysis.",
        flush=True,
    )
    preprocess_dataset(config, force=manifest_path.exists())
    return True


def percentiles(values: list[float]) -> dict[str, float]:
    result = np.percentile(np.asarray(values, dtype=np.float64), [0, 25, 50, 75, 95, 100])
    return {
        name: float(value)
        for name, value in zip(("minimum", "p25", "median", "p75", "p95", "maximum"), result)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Context V3 training-mask support")
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--output", default="artifacts/fold0/context_mask_report.json")
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")

    config = load_config(args.config)
    ensure_preprocessing(config)
    metadata = load_metadata(config)
    training, _ = split_indices(metadata, config)
    repository = ContextRepository.__new__(ContextRepository)
    repository.config = config
    repository.metadata = metadata
    repository.cell_count = int(metadata["train_cells"].max()) + 1
    repository.indices_by_cell = [
        training[metadata["train_cells"][training] == cell_id]
        for cell_id in range(repository.cell_count)
    ]
    repository.test_component_templates = repository._build_test_component_templates()

    section = config["context"]
    rng = np.random.default_rng(int(config["seed"]) + 717)
    targets: list[float] = []
    hidden: list[float] = []
    support: list[float] = []
    guards: list[float] = []
    patterns: Counter[str] = Counter()
    positions = metadata["train_positions"][:, :2]
    for _ in range(args.samples):
        cell_id = int(rng.integers(repository.cell_count))
        sample = repository.sample_spatial_mask(
            rng,
            cell_id,
            float(section["hole_min_meters"]),
            float(section["hole_max_meters"]),
            int(section["minimum_targets"]),
            int(section["maximum_targets"]),
            float(section.get("outage_anchor_probability", 0.25)),
            float(section.get("test_template_probability", 0.65)),
            float(section.get("template_radius_meters", 3.0)),
            float(section.get("observation_guard_min_meters", 3.5)),
            float(section.get("observation_guard_max_meters", 8.5)),
        )
        observed = repository.context_indices(cell_id, sample.hidden)
        nearest = np.linalg.norm(
            positions[sample.targets, None] - positions[observed][None, :], axis=2
        ).min(axis=1)
        targets.append(float(len(sample.targets)))
        hidden.append(float(len(sample.hidden)))
        support.extend(float(value) for value in nearest)
        guards.append(sample.guard_meters)
        patterns[sample.pattern] += 1

    support_summary = percentiles(support)
    minimum_median, maximum_median = section.get("mask_support_median_gate", [4.5, 8.0])
    passed = float(minimum_median) <= support_summary["median"] <= float(maximum_median)
    report = {
        "status": "PASS" if passed else "FAIL",
        "samples": args.samples,
        "target_count": percentiles(targets),
        "hidden_count": percentiles(hidden),
        "nearest_observed_meters": support_summary,
        "guard_meters": percentiles(guards),
        "patterns": dict(sorted(patterns.items())),
        "median_gate": [float(minimum_median), float(maximum_median)],
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit("Context mask support does not match the configured gate")


if __name__ == "__main__":
    main()
