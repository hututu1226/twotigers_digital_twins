from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import _bootstrap  # noqa: F401
import numpy as np

from scheme_e.config import load_config, save_json
from scheme_e.distribution_audit import (
    domain_classifier_auc,
    feature_shift_report,
    one_to_one_same_cell_assignment,
    same_cell_support_features,
    test_matched_holdout_gate,
)


ENVIRONMENT_PREFIXES = ("local_", "corridor_", "fresnel_", "surface_")
LINK_FEATURES = {
    "distance_3d",
    "distance_2d",
    "inverse_distance",
    "azimuth_sin",
    "azimuth_cos",
    "elevation_sin",
}


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: np.array(source[name], copy=True) for name in source.files}


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _query_features(
    metadata: dict[str, np.ndarray],
    *,
    support_indices: np.ndarray,
    query_positions: np.ndarray,
    query_cells: np.ndarray,
    query_geometry: np.ndarray,
    geometry_names: list[str],
    cell_count: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    support, support_names = same_cell_support_features(
        metadata["train_positions"][support_indices],
        metadata["train_cells"][support_indices],
        query_positions,
        query_cells,
    )
    geometry_indices = np.asarray(
        [
            index
            for index, name in enumerate(geometry_names)
            if name in LINK_FEATURES or name.startswith(ENVIRONMENT_PREFIXES)
        ],
        dtype=np.int64,
    )
    geometry = np.asarray(query_geometry, dtype=np.float64)[:, geometry_indices]
    one_hot = np.zeros((len(query_cells), int(cell_count)), dtype=np.float64)
    one_hot[np.arange(len(query_cells)), query_cells.astype(np.int64)] = 1.0
    link_names = [geometry_names[index] for index in geometry_indices]
    return (
        {
            "support": support,
            "support_domain": np.concatenate([one_hot, support], axis=1),
            "link_environment_domain": np.concatenate(
                [one_hot, support, geometry], axis=1
            ),
        },
        [
            *[f"cell_{cell}" for cell in range(int(cell_count))],
            *support_names,
            *link_names,
        ],
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a label-free test-neighborhood validation mask using a "
            "one-to-one same-cell spatial assignment"
        )
    )
    parser.add_argument("--config", default="configs/v4_attempt1_structured.json")
    parser.add_argument("--minimum-samples", type=int, default=450)
    parser.add_argument("--maximum-samples", type=int, default=650)
    parser.add_argument("--minimum-samples-per-cell", type=int, default=200)
    parser.add_argument("--maximum-nearest-gap-m", type=float, default=2.0)
    parser.add_argument("--maximum-support-auc", type=float, default=0.70)
    parser.add_argument("--current-link-auc", type=float, default=0.9962938053097344)
    parser.add_argument("--minimum-link-auc-reduction", type=float, default=0.10)
    parser.add_argument(
        "--output-dir", default="artifacts/scheme_e_065/l0_025_test_neighborhood"
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_025_TEST_NEIGHBORHOOD_HOLDOUT.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    metadata = _load_npz(artifact_dir / "metadata.npz")
    manifest = _read_json(artifact_dir / "manifest.json")
    geometry_names = [str(name) for name in manifest["geometry_feature_names"]]
    cell_count = int(manifest["setup"]["Q"])

    matched_train_indices, assignment_distances = one_to_one_same_cell_assignment(
        metadata["train_positions"],
        metadata["train_cells"],
        metadata["test_positions"],
        metadata["test_cells"],
    )
    validation_mask = np.zeros(len(metadata["train_positions"]), dtype=bool)
    validation_mask[matched_train_indices] = True
    validation = np.flatnonzero(validation_mask)
    visible_nonoutage = np.flatnonzero(
        (~validation_mask) & (~metadata["outage"].astype(bool))
    )
    full_nonoutage = np.flatnonzero(~metadata["outage"].astype(bool))

    validation_features, feature_names = _query_features(
        metadata,
        support_indices=visible_nonoutage,
        query_positions=metadata["train_positions"][validation],
        query_cells=metadata["train_cells"][validation],
        query_geometry=metadata["train_geometry_features"][validation],
        geometry_names=geometry_names,
        cell_count=cell_count,
    )
    test_features, test_feature_names = _query_features(
        metadata,
        support_indices=full_nonoutage,
        query_positions=metadata["test_positions"],
        query_cells=metadata["test_cells"],
        query_geometry=metadata["test_geometry_features"],
        geometry_names=geometry_names,
        cell_count=cell_count,
    )
    if feature_names != test_feature_names:
        raise RuntimeError("validation and test feature names do not match")

    support_auc_payload = domain_classifier_auc(
        validation_features["support_domain"], test_features["support_domain"]
    )
    link_auc_payload = domain_classifier_auc(
        validation_features["link_environment_domain"],
        test_features["link_environment_domain"],
    )
    support_auc = max(
        float(support_auc_payload["linear_oof_auc"]),
        float(support_auc_payload["nonlinear_oof_auc"]),
    )
    link_auc = max(
        float(link_auc_payload["linear_oof_auc"]),
        float(link_auc_payload["nonlinear_oof_auc"]),
    )
    validation_nearest = validation_features["support"][:, 0]
    test_nearest = test_features["support"][:, 0]
    nearest_gap = abs(
        float(np.median(validation_nearest)) - float(np.median(test_nearest))
    )
    cell_counts = {
        str(int(cell)): int(
            np.sum(metadata["train_cells"][validation] == int(cell))
        )
        for cell in np.unique(metadata["test_cells"])
    }
    promoted, failures = test_matched_holdout_gate(
        samples=int(len(validation)),
        cell_counts=cell_counts,
        nearest_median_gap_m=nearest_gap,
        support_domain_auc=support_auc,
        link_environment_domain_auc=link_auc,
        current_link_environment_auc=float(args.current_link_auc),
        minimum_samples=int(args.minimum_samples),
        maximum_samples=int(args.maximum_samples),
        minimum_samples_per_cell=int(args.minimum_samples_per_cell),
        maximum_nearest_median_gap_m=float(args.maximum_nearest_gap_m),
        maximum_support_domain_auc=float(args.maximum_support_auc),
        minimum_link_auc_reduction=float(args.minimum_link_auc_reduction),
    )
    decision = (
        "PROMOTE_ONE_TEST_NEIGHBORHOOD_RETRAIN"
        if promoted
        else "DROP_NEAREST_TEST_NEIGHBORHOOD_HOLDOUT"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "matched_validation_mask.npy", validation_mask)
    np.savez_compressed(
        output_dir / "assignment.npz",
        test_indices=np.arange(len(metadata["test_positions"]), dtype=np.int64),
        train_indices=matched_train_indices.astype(np.int64),
        distances_m=assignment_distances.astype(np.float32),
    )
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-025",
        "diagnostic_only": True,
        "uses_test_channel_labels": False,
        "uses_train_channel_labels": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_bootstrap.PROJECT_ROOT.parent,
            text=True,
        ).strip(),
        "method": (
            "minimum-total-XY-distance one-to-one assignment within each cell"
        ),
        "samples": int(len(validation)),
        "cell_counts": cell_counts,
        "assignment_distance_m": _summary(assignment_distances),
        "validation_nearest_support_m": _summary(validation_nearest),
        "test_nearest_support_m": _summary(test_nearest),
        "nearest_median_gap_to_test_m": nearest_gap,
        "support_only_auc": support_auc_payload,
        "link_environment_auc": link_auc_payload,
        "support_auc_max": support_auc,
        "link_environment_auc_max": link_auc,
        "link_auc_reduction": float(args.current_link_auc) - link_auc,
        "link_environment_shift": feature_shift_report(
            validation_features["link_environment_domain"],
            test_features["link_environment_domain"],
            feature_names,
        ),
        "fixed_gate": {
            "sample_range": [int(args.minimum_samples), int(args.maximum_samples)],
            "minimum_samples_per_cell": int(args.minimum_samples_per_cell),
            "maximum_nearest_median_gap_m": float(args.maximum_nearest_gap_m),
            "maximum_support_domain_auc": float(args.maximum_support_auc),
            "current_link_environment_auc": float(args.current_link_auc),
            "minimum_link_auc_reduction": float(args.minimum_link_auc_reduction),
            "required_maximum_link_environment_auc": float(args.current_link_auc)
            - float(args.minimum_link_auc_reduction),
        },
        "promotion_gate_passed": bool(promoted),
        "promotion_gate_failures": failures,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    save_json(report_path, report)
    report_path.with_suffix(".md").write_text(
        "# L0-025 Test-Neighborhood Holdout Audit\n\n"
        "This audit uses positions and RF geometry only. It never uses train or test channel labels.\n\n"
        f"- Samples: `{len(validation)}`; cells `{cell_counts}`.\n"
        f"- Test-to-train assignment median: `{np.median(assignment_distances):.3f} m`.\n"
        f"- Nearest-support median gap: `{nearest_gap:.3f} m`.\n"
        f"- Support-domain AUC: `{support_auc:.4f}`.\n"
        f"- Link/environment AUC: `{link_auc:.4f}` "
        f"(reduction `{float(args.current_link_auc) - link_auc:.4f}`).\n"
        f"- Gate failures: `{failures}`.\n"
        f"- Decision: `{decision}`.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
