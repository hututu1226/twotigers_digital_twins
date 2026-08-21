from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from scheme_g.config import save_json


REPO_ROOT = _bootstrap.PROJECT_ROOT.parent
D_PROJECT = REPO_ROOT / "schemeD_transport_residual_context"
if str(D_PROJECT) not in sys.path:
    sys.path.insert(0, str(D_PROJECT))

from scheme_d.config import choose_device, load_config, seed_everything  # noqa: E402
from scheme_d.context_data import ContextRepository  # noqa: E402
from scheme_d.context_training import (  # noqa: E402
    evaluate_context_model,
    load_context_checkpoint,
)
from scheme_d.data import balanced_limit, load_metadata, split_indices  # noqa: E402


def _candidate_roots(explicit: str | None) -> list[Path]:
    roots = [REPO_ROOT]
    if explicit:
        roots.insert(0, Path(explicit))
    for value in (
        "/root/autodl-fs/schemeF_0820_20260820",
        "/root/autodl-fs/twotigers_0819_run",
        "/root/autodl-fs/twotigers_digital_twins",
    ):
        roots.append(Path(value))
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _legacy_paths(root: Path) -> dict[str, Path]:
    return {
        "preprocessed": root
        / "schemeD_transport_residual_context/artifacts/preprocessed_scheme_d",
        "encoded": root
        / "schemeD_transport_residual_context/artifacts/fold0/encoded.npz",
        "checkpoint": root
        / "schemeD_transport_residual_context/artifacts/fold0/context/best.pt",
        "autoencoder": root
        / "schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt",
    }


def _find_root(explicit: str | None) -> tuple[Path | None, dict[str, Path] | None]:
    for root in _candidate_roots(explicit):
        paths = _legacy_paths(root)
        if (
            all(path.is_file() for key, path in paths.items() if key != "preprocessed")
            and (paths["preprocessed"] / "metadata.npz").is_file()
        ):
            return root, paths
    return None, None


def _inferred_temperature(config: dict, epoch: int) -> float:
    section = config["context"]
    initial = float(section.get("router_temperature_initial", 2.0))
    final = float(section.get("router_temperature_final", 0.9))
    anneal = max(int(section.get("router_temperature_anneal_epochs", 300)), 1)
    progress = min(max(epoch, 0) / max(anneal - 1, 1), 1.0)
    return initial + (final - initial) * progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Scheme D Router scan")
    parser.add_argument("--legacy-root")
    parser.add_argument("--limit", type=int, default=192)
    parser.add_argument("--output", default="reports/generated/scheme_d_scan.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    root, paths = _find_root(args.legacy_root)
    if root is None or paths is None:
        report = {
            "status": "SKIPPED",
            "reason": "Scheme D Fold0 checkpoint/cache not found in persistent roots",
            "searched_roots": [
                str(path) for path in _candidate_roots(args.legacy_root)
            ],
        }
        save_json(output, report)
        print(json.dumps(report, indent=2))
        return

    config = load_config(D_PROJECT / "configs/fold0_5090.json")
    config["data"]["root"] = str(root / "Round2_Map")
    config["preprocessing"]["artifact_dir"] = str(paths["preprocessed"])
    config["encoding"]["output_path"] = str(paths["encoded"])
    config["encoding"]["autoencoder_checkpoint"] = str(paths["autoencoder"])
    config["context"]["autoencoder_checkpoint"] = str(paths["autoencoder"])
    config["runtime"]["device"] = args.device
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    metadata = load_metadata(config)
    training, validation = split_indices(metadata, config)
    with np.load(paths["encoded"]) as source:
        available = source["available"].astype(bool)
    training = training[available[training]]
    validation = validation[available[validation]]
    validation = balanced_limit(
        validation,
        args.limit,
        [metadata["train_cells"], metadata["outage"].astype(np.int8)],
        int(config["seed"]) + 91,
    )
    repository = ContextRepository(config, training)
    model, autoencoder, shape, checkpoint = load_context_checkpoint(
        config, paths["checkpoint"], repository, device
    )
    epoch = int(checkpoint.get("epoch", -1))
    inferred_temperature = _inferred_temperature(config, epoch)
    threshold = float(
        checkpoint.get("metrics", {}).get(
            "outage_threshold", config["context"].get("outage_threshold", 0.999)
        )
    )
    definitions = [
        ("reload_bug", 64, 2.0, 0.1, False),
        ("restore_training_temperature", 64, inferred_temperature, 0.1, False),
        ("sparse_top16", 16, 1.0, 0.0, False),
        ("sparse_top8", 8, 0.8, 0.0, False),
        ("sparse_top8_no_warp", 8, 0.8, 0.0, True),
    ]
    reports: list[dict] = []
    for name, top_k, temperature, uniform_mix, disable_warp in definitions:
        model.router.top_k = top_k
        model.router.temperature = temperature
        model.router.uniform_mix = uniform_mix
        model.set_diagnostic_ablation(disable_warp=disable_warp, route_bias_scale=1.0)
        metrics = evaluate_context_model(
            model,
            autoencoder,
            repository,
            validation,
            shape,
            device,
            device.type == "cuda",
            threshold,
            int(config["context"].get("validation_decode_batch_size", 8)),
        )
        reports.append(
            {
                "name": name,
                "top_k": top_k,
                "temperature": temperature,
                "uniform_mix": uniform_mix,
                "disable_warp": disable_warp,
                "metrics": metrics,
            }
        )
        print(f"D scan {name}: score={metrics['score']:.6f}", flush=True)
    best = max(reports, key=lambda item: float(item["metrics"]["score"]))
    report = {
        "status": "PASS",
        "legacy_root": str(root),
        "samples": int(len(validation)),
        "checkpoint_epoch": epoch + 1,
        "checkpoint_recorded_score": checkpoint.get("best_score"),
        "inferred_checkpoint_temperature": inferred_temperature,
        "best_counterfactual": best["name"],
        "best_score": best["metrics"]["score"],
        "reports": reports,
        "interpretation": (
            "Counterfactuals reuse Scheme D weights and are diagnostic only; "
            "they must not be presented as retrained Fold0 results."
        ),
    }
    save_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
