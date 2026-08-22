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
    distribution_signature_distance,
    domain_classifier_auc,
    periodic_cell_holdout_mask,
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


def _link_environment_indices(names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, name in enumerate(names)
            if name in LINK_FEATURES or name.startswith(ENVIRONMENT_PREFIXES)
        ],
        dtype=np.int64,
    )


def _cell_one_hot(cell: int, rows: int, cell_count: int) -> np.ndarray:
    output = np.zeros((int(rows), int(cell_count)), dtype=np.float64)
    output[:, int(cell)] = 1.0
    return output


def _query_features(
    metadata: dict[str, np.ndarray],
    *,
    support_indices: np.ndarray,
    query_positions: np.ndarray,
    query_cells: np.ndarray,
    query_geometry: np.ndarray,
    geometry_indices: np.ndarray,
    cell_count: int,
) -> dict[str, np.ndarray]:
    support, _ = same_cell_support_features(
        metadata["train_positions"][support_indices],
        metadata["train_cells"][support_indices],
        query_positions,
        query_cells,
    )
    one_hot = np.zeros((len(query_cells), int(cell_count)), dtype=np.float64)
    one_hot[np.arange(len(query_cells)), query_cells.astype(np.int64)] = 1.0
    geometry = np.asarray(query_geometry, dtype=np.float64)[:, geometry_indices]
    return {
        "support": support,
        "support_domain": np.concatenate([one_hot, support], axis=1),
        "link_environment_domain": np.concatenate(
            [one_hot, support, geometry], axis=1
        ),
    }


def _phase_candidate(
    metadata: dict[str, np.ndarray],
    test_by_cell: dict[int, dict[str, np.ndarray]],
    *,
    cell: int,
    tile_meters: float,
    hole_meters: float,
    phase_x: float,
    phase_y: float,
    geometry_indices: np.ndarray,
    cell_count: int,
) -> dict[str, object]:
    mask = periodic_cell_holdout_mask(
        metadata["train_positions"],
        metadata["train_cells"],
        cell_id=int(cell),
        tile_meters=float(tile_meters),
        hole_meters=float(hole_meters),
        phase_x=float(phase_x),
        phase_y=float(phase_y),
    )
    query_indices = np.flatnonzero(mask)
    visible = np.flatnonzero(
        (~mask) & (~metadata["outage"].astype(bool))
    )
    features = _query_features(
        metadata,
        support_indices=visible,
        query_positions=metadata["train_positions"][query_indices],
        query_cells=metadata["train_cells"][query_indices],
        query_geometry=metadata["train_geometry_features"][query_indices],
        geometry_indices=geometry_indices,
        cell_count=cell_count,
    )
    target = test_by_cell[int(cell)]
    support_distance = distribution_signature_distance(
        features["support"], target["support"]
    )
    environment_distance = distribution_signature_distance(
        features["link_environment_domain"][:, cell_count + features["support"].shape[1] :],
        target["link_environment_domain"][:, cell_count + target["support"].shape[1] :],
    )
    return {
        "cell": int(cell),
        "phase_x": float(phase_x),
        "phase_y": float(phase_y),
        "samples": int(len(query_indices)),
        "query_indices": query_indices,
        "mask": mask,
        "features": features,
        "support_signature_distance": float(support_distance),
        "environment_signature_distance": float(environment_distance),
        "ranking_value": float(0.5 * support_distance + 0.5 * environment_distance),
    }


def _compact_candidate(candidate: dict[str, object]) -> dict[str, object]:
    return {
        name: candidate[name]
        for name in (
            "cell",
            "phase_x",
            "phase_y",
            "samples",
            "support_signature_distance",
            "environment_signature_distance",
            "ranking_value",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search independent per-cell periodic validation phases that match "
            "the observable test support and RF-geometry distribution"
        )
    )
    parser.add_argument("--config", default="configs/v4_attempt1_structured.json")
    parser.add_argument("--phase-steps", type=int, default=16)
    parser.add_argument("--top-per-cell", type=int, default=6)
    parser.add_argument("--search-minimum-per-cell", type=int, default=160)
    parser.add_argument("--search-maximum-per-cell", type=int, default=360)
    parser.add_argument("--minimum-samples", type=int, default=450)
    parser.add_argument("--maximum-samples", type=int, default=650)
    parser.add_argument("--minimum-samples-per-cell", type=int, default=200)
    parser.add_argument("--maximum-nearest-gap-m", type=float, default=2.0)
    parser.add_argument("--maximum-support-auc", type=float, default=0.70)
    parser.add_argument("--current-link-auc", type=float, default=0.9962938053097344)
    parser.add_argument("--minimum-link-auc-reduction", type=float, default=0.10)
    parser.add_argument(
        "--output-dir", default="artifacts/scheme_e_065/l0_024_matched_holdout"
    )
    parser.add_argument(
        "--report",
        default="../research/scheme_e_065/L0_024_TEST_MATCHED_HOLDOUT.json",
    )
    args = parser.parse_args()
    if int(args.phase_steps) < 2:
        raise ValueError("phase_steps must be at least 2")
    if int(args.top_per_cell) < 1:
        raise ValueError("top_per_cell must be positive")

    started = time.perf_counter()
    config = load_config(args.config)
    artifact_dir = Path(config["preprocessing"]["artifact_dir"])
    metadata = _load_npz(artifact_dir / "metadata.npz")
    manifest = _read_json(artifact_dir / "manifest.json")
    geometry_names = [str(name) for name in manifest["geometry_feature_names"]]
    geometry_indices = _link_environment_indices(geometry_names)
    cell_count = int(manifest["setup"]["Q"])
    cells = sorted(int(cell) for cell in np.unique(metadata["train_cells"]))
    tile_meters = float(config["preprocessing"]["validation_tile_meters"])
    hole_meters = float(config["preprocessing"]["validation_hole_meters"])
    full_nonoutage = np.flatnonzero(~metadata["outage"].astype(bool))

    test_features = _query_features(
        metadata,
        support_indices=full_nonoutage,
        query_positions=metadata["test_positions"],
        query_cells=metadata["test_cells"],
        query_geometry=metadata["test_geometry_features"],
        geometry_indices=geometry_indices,
        cell_count=cell_count,
    )
    test_by_cell: dict[int, dict[str, np.ndarray]] = {}
    for cell in cells:
        selected = metadata["test_cells"] == int(cell)
        test_by_cell[cell] = {
            name: np.asarray(values)[selected]
            for name, values in test_features.items()
        }

    phases = np.arange(int(args.phase_steps), dtype=np.float64)
    phases *= tile_meters / float(args.phase_steps)
    top_candidates: dict[int, list[dict[str, object]]] = {}
    phase_summaries: dict[str, list[dict[str, object]]] = {}
    for cell in cells:
        candidates = []
        for phase_x in phases:
            for phase_y in phases:
                candidate = _phase_candidate(
                    metadata,
                    test_by_cell,
                    cell=cell,
                    tile_meters=tile_meters,
                    hole_meters=hole_meters,
                    phase_x=float(phase_x),
                    phase_y=float(phase_y),
                    geometry_indices=geometry_indices,
                    cell_count=cell_count,
                )
                count = int(candidate["samples"])
                if (
                    int(args.search_minimum_per_cell)
                    <= count
                    <= int(args.search_maximum_per_cell)
                ):
                    candidates.append(candidate)
        if not candidates:
            raise RuntimeError(f"No valid periodic phase candidates for cell {cell}")
        candidates.sort(key=lambda item: float(item["ranking_value"]))
        top_candidates[cell] = candidates[: int(args.top_per_cell)]
        phase_summaries[str(cell)] = [
            _compact_candidate(candidate) for candidate in candidates[:20]
        ]
        print(
            f"cell={cell} valid_phases={len(candidates)} "
            f"best_signature={float(candidates[0]['ranking_value']):.6f}",
            flush=True,
        )

    combinations: list[dict[str, object]] = []
    if len(cells) != 2:
        raise RuntimeError(f"L0-024 expects two cells, found {cells}")
    left_cell, right_cell = cells
    test_support_domain = np.asarray(test_features["support_domain"])
    test_link_domain = np.asarray(test_features["link_environment_domain"])
    test_nearest = np.asarray(test_features["support"])[:, 0]
    for left in top_candidates[left_cell]:
        for right in top_candidates[right_cell]:
            selected_candidates = [left, right]
            validation_mask = np.asarray(left["mask"]) | np.asarray(right["mask"])
            support_domain = np.concatenate(
                [
                    np.asarray(candidate["features"]["support_domain"])
                    for candidate in selected_candidates
                ],
                axis=0,
            )
            link_domain = np.concatenate(
                [
                    np.asarray(candidate["features"]["link_environment_domain"])
                    for candidate in selected_candidates
                ],
                axis=0,
            )
            nearest = np.concatenate(
                [
                    np.asarray(candidate["features"]["support"])[:, 0]
                    for candidate in selected_candidates
                ]
            )
            support_auc_payload = domain_classifier_auc(
                support_domain, test_support_domain
            )
            link_auc_payload = domain_classifier_auc(link_domain, test_link_domain)
            support_auc = max(
                float(support_auc_payload["linear_oof_auc"]),
                float(support_auc_payload["nonlinear_oof_auc"]),
            )
            link_auc = max(
                float(link_auc_payload["linear_oof_auc"]),
                float(link_auc_payload["nonlinear_oof_auc"]),
            )
            nearest_gap = abs(float(np.median(nearest)) - float(np.median(test_nearest)))
            cell_counts = {
                str(int(cell)): int(
                    np.sum(validation_mask & (metadata["train_cells"] == int(cell)))
                )
                for cell in cells
            }
            promoted, failures = test_matched_holdout_gate(
                samples=int(validation_mask.sum()),
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
            combinations.append(
                {
                    "phases": {
                        str(left_cell): [left["phase_x"], left["phase_y"]],
                        str(right_cell): [right["phase_x"], right["phase_y"]],
                    },
                    "samples": int(validation_mask.sum()),
                    "cell_counts": cell_counts,
                    "nearest_median_m": float(np.median(nearest)),
                    "nearest_median_gap_to_test_m": nearest_gap,
                    "support_only_auc": support_auc_payload,
                    "link_environment_auc": link_auc_payload,
                    "support_auc_max": support_auc,
                    "link_environment_auc_max": link_auc,
                    "link_auc_reduction": float(args.current_link_auc) - link_auc,
                    "promotion_gate_passed": bool(promoted),
                    "promotion_gate_failures": failures,
                    "ranking_value": float(link_auc + 0.25 * support_auc + 0.02 * nearest_gap),
                    "validation_mask": validation_mask,
                }
            )
            print(
                f"phases={combinations[-1]['phases']} samples={int(validation_mask.sum())} "
                f"support_auc={support_auc:.4f} link_auc={link_auc:.4f} "
                f"nearest_gap={nearest_gap:.3f} gate={promoted}",
                flush=True,
            )

    combinations.sort(key=lambda item: float(item["ranking_value"]))
    best = combinations[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "matched_validation_mask.npy", best["validation_mask"])
    compact_best = {
        name: value for name, value in best.items() if name != "validation_mask"
    }
    compact_combinations = [
        {name: value for name, value in item.items() if name != "validation_mask"}
        for item in combinations
    ]
    decision = (
        "PROMOTE_ONE_TEST_MATCHED_RETRAIN"
        if bool(best["promotion_gate_passed"])
        else "DROP_PERIODIC_PHASE_MATCHING"
    )
    report = {
        "status": "COMPLETED",
        "experiment_id": "L0-024",
        "diagnostic_only": True,
        "uses_test_channel_labels": False,
        "uses_train_channel_labels": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_bootstrap.PROJECT_ROOT.parent,
            text=True,
        ).strip(),
        "search": {
            "tile_meters": tile_meters,
            "hole_meters": hole_meters,
            "phase_steps": int(args.phase_steps),
            "top_per_cell": int(args.top_per_cell),
            "geometry_feature_count": int(len(geometry_indices)),
            "per_cell_top_phases": phase_summaries,
            "evaluated_combinations": int(len(combinations)),
        },
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
        "best_candidate": compact_best,
        "all_finalists": compact_combinations,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(args.report)
    save_json(report_path, report)
    report_path.with_suffix(".md").write_text(
        "# L0-024 Test-Matched Holdout Search\n\n"
        "This search uses positions and RF geometry only. It never uses test or train channel labels.\n\n"
        f"- Best samples: `{int(best['samples'])}`; cells `{best['cell_counts']}`.\n"
        f"- Nearest-support median gap: `{float(best['nearest_median_gap_to_test_m']):.3f} m`.\n"
        f"- Support-domain AUC: `{float(best['support_auc_max']):.4f}`.\n"
        f"- Link/environment AUC: `{float(best['link_environment_auc_max']):.4f}` "
        f"(reduction `{float(best['link_auc_reduction']):.4f}`).\n"
        f"- Gate failures: `{best['promotion_gate_failures']}`.\n"
        f"- Decision: `{decision}`.\n\n"
        "No model retraining is authorized by this report unless the fixed gate passes.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "best_candidate": compact_best,
                "decision": decision,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
