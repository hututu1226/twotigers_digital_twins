from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import _bootstrap  # noqa: F401
from scheme_g.config import save_json


REPO_ROOT = _bootstrap.PROJECT_ROOT.parent
E_PROJECT = REPO_ROOT / "schemeE_spectral_gaussian_hybrid"
if str(E_PROJECT) not in sys.path:
    sys.path.insert(0, str(E_PROJECT))

from scheme_e.config import choose_device, load_config, seed_everything  # noqa: E402
from scheme_e.hybrid_training import evaluate_hybrid, load_hybrid_checkpoint  # noqa: E402


def _candidate_roots(explicit: str | None) -> list[Path]:
    roots = [REPO_ROOT]
    if explicit:
        roots.insert(0, Path(explicit))
    roots.extend(
        [
            Path("/root/autodl-fs/schemeF_0820_20260820"),
            Path("/root/autodl-fs/twotigers_0819_run"),
            Path("/root/autodl-fs/twotigers_digital_twins"),
        ]
    )
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _paths(root: Path) -> dict[str, Path]:
    base = root / "schemeE_spectral_gaussian_hybrid"
    return {
        "metadata": base / "artifacts/preprocessed_scheme_e/metadata.npz",
        "priors": base / "artifacts/fold0/spectral_teacher/oof_priors.npz",
        "checkpoint": base / "artifacts/fold0/hybrid/best.pt",
        "autoencoder": root
        / "schemeC_full_resolution_context/artifacts/fold0/autoencoder/best.pt",
    }


def _find_root(explicit: str | None) -> tuple[Path | None, dict[str, Path] | None]:
    for root in _candidate_roots(explicit):
        paths = _paths(root)
        if all(path.is_file() for path in paths.values()):
            return root, paths
    return None, None


def _balanced(
    indices: np.ndarray, cells: np.ndarray, outage: np.ndarray, limit: int
) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(2097)
    groups: dict[tuple[int, int], list[int]] = {}
    for index in indices:
        groups.setdefault((int(cells[index]), int(outage[index])), []).append(
            int(index)
        )
    for values in groups.values():
        rng.shuffle(values)
    output: list[int] = []
    while len(output) < limit:
        progressed = False
        for key in sorted(groups):
            if groups[key]:
                output.append(groups[key].pop())
                progressed = True
                if len(output) == limit:
                    break
        if not progressed:
            break
    return np.asarray(sorted(output), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Scheme E failure scan")
    parser.add_argument("--legacy-root")
    parser.add_argument("--limit", type=int, default=192)
    parser.add_argument("--output", default="reports/generated/scheme_e_scan.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    output = Path(args.output)
    root, paths = _find_root(args.legacy_root)
    if root is None or paths is None:
        report = {
            "status": "SKIPPED",
            "reason": "Scheme E Fold0 checkpoint/OOF priors not found in persistent roots",
            "searched_roots": [
                str(path) for path in _candidate_roots(args.legacy_root)
            ],
        }
        save_json(output, report)
        print(json.dumps(report, indent=2))
        return

    config = load_config(E_PROJECT / "configs/fold0_5090.json")
    data_root = root / "Round2_Map"
    if not data_root.is_dir():
        data_root = REPO_ROOT / "Round2_Map"
    config["data"]["root"] = str(data_root)
    config["preprocessing"]["artifact_dir"] = str(paths["metadata"].parent)
    config["spectral_teacher"]["oof_output_path"] = str(paths["priors"])
    config["hybrid"]["autoencoder_checkpoint"] = str(paths["autoencoder"])
    config["runtime"]["device"] = args.device
    seed_everything(int(config["seed"]))
    device = choose_device(args.device)
    with np.load(paths["metadata"]) as source:
        metadata = {name: source[name] for name in source.files}
    with np.load(paths["priors"]) as source:
        priors = {name: source[name] for name in source.files}
    available = priors["available"].astype(bool)
    fold = int(config["split"]["validation_fold"])
    validation_mask = metadata["validation_masks"][fold].astype(bool)
    indices = np.arange(len(available), dtype=np.int64)
    observed = indices[available & ~validation_mask]
    validation = _balanced(
        indices[available & validation_mask],
        metadata["train_cells"],
        metadata["outage"],
        args.limit,
    )
    model, shape, checkpoint = load_hybrid_checkpoint(
        config, paths["checkpoint"], device
    )
    channels = np.load(data_root / "Round2_Train_Channel.npy", mmap_mode="r")
    geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    geometry_std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    threshold = float(np.asarray(priors["outage_threshold"]).item())

    variants: dict[str, dict[str, np.ndarray]] = {"baseline": priors}
    oracle_outage = dict(priors)
    oracle_outage["outage_probability"] = metadata["outage"].astype(np.float32)
    variants["oracle_outage"] = oracle_outage
    oracle_power = dict(priors)
    oracle_power["log_power"] = metadata["log_power"].astype(np.float32)
    variants["oracle_power"] = oracle_power
    oracle_both = dict(oracle_power)
    oracle_both["outage_probability"] = metadata["outage"].astype(np.float32)
    variants["oracle_outage_power"] = oracle_both

    bounded = dict(priors)
    bounded_power = np.asarray(priors["log_power"], dtype=np.float32).copy()
    for cell_id in np.unique(metadata["train_cells"]):
        reference = observed[
            (metadata["train_cells"][observed] == cell_id)
            & ~metadata["outage"][observed].astype(bool)
        ]
        low, high = np.quantile(metadata["log_power"][reference], [0.005, 0.995])
        selected = metadata["train_cells"] == cell_id
        bounded_power[selected] = np.clip(bounded_power[selected], low, high)
    bounded["log_power"] = bounded_power
    variants["bounded_gp_power"] = bounded
    bounded_oracle_outage = dict(bounded)
    bounded_oracle_outage["outage_probability"] = metadata["outage"].astype(np.float32)
    variants["bounded_power_oracle_outage"] = bounded_oracle_outage

    reports: dict[str, dict] = {}
    for name, values in variants.items():
        metrics = evaluate_hybrid(
            model,
            shape,
            channels,
            metadata,
            values,
            validation,
            observed,
            geometry_mean,
            geometry_std,
            device,
            int(config["hybrid"].get("validation_batch_size", 2)),
            threshold,
        )
        reports[name] = metrics
        print(f"E scan {name}: score={metrics['score']:.6f} nmse={metrics['nmse']:.6f}")
    baseline = float(reports["baseline"]["score"])
    deltas = {
        name: float(values["score"] - baseline) for name, values in reports.items()
    }
    report = {
        "status": "PASS",
        "legacy_root": str(root),
        "samples": int(len(validation)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) + 1,
        "outage_threshold": threshold,
        "reports": reports,
        "score_deltas": deltas,
        "primary_failure": max(
            ("outage", deltas["oracle_outage"]),
            ("power", deltas["oracle_power"]),
            key=lambda item: item[1],
        )[0],
        "interpretation": (
            "Oracle variants are diagnostic ceilings only. bounded_gp_power is oracle-free "
            "and checks whether the GP tail, rather than spectral shape, causes the failure."
        ),
    }
    save_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
