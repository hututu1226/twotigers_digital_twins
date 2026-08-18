from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def require_file(path: Path, minimum_bytes: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise ValueError(f"Missing or empty required file: {path}")


def read_json(path: Path) -> dict:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_history(path: Path) -> dict:
    require_file(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty training history: {path}")
    if int(records[-1].get("epoch", -1)) < 0:
        raise ValueError(f"Invalid last epoch in {path}")
    return {
        "epochs_recorded": len(records),
        "last_epoch": int(records[-1]["epoch"]) + 1,
    }


def verify_evaluation(path: Path) -> dict:
    report = read_json(path)
    metrics = report.get("metrics", report)
    names = ("pas", "pdp", "nmse", "score")
    values = {name: float(metrics[name]) for name in names}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite evaluation metric: {path}")
    if int(metrics.get("samples", 0)) <= 0:
        raise ValueError(f"Evaluation has no samples: {path}")
    return values


def verify_output(path: Path, expected_count: int) -> dict:
    require_file(path, minimum_bytes=128)
    channel = np.load(path, mmap_mode="r")
    expected_shape = (expected_count, 256, 4, 192)
    if channel.shape != expected_shape:
        raise ValueError(f"Invalid output shape {channel.shape}; expected {expected_shape}")
    if channel.dtype != np.complex64:
        raise ValueError(f"Invalid output dtype {channel.dtype}; expected complex64")
    finite = True
    for start in range(0, len(channel), 10):
        finite = finite and bool(np.isfinite(np.asarray(channel[start : start + 10])).all())
    if not finite:
        raise ValueError(f"NaN or Inf found in {path}")
    return {
        "path": str(path),
        "shape": list(channel.shape),
        "dtype": str(channel.dtype),
        "finite": finite,
        "bytes": path.stat().st_size,
    }


def verify_encoded(path: Path, expected_total: int | None) -> dict:
    require_file(path, minimum_bytes=128)
    with np.load(path) as source:
        spectrum_shape = tuple(int(value) for value in source["spectrum_shape"])
        phase_shape = tuple(int(value) for value in source["phase_shape"])
        spectrum_statistics_shape = tuple(source["spectrum_mean"].shape)
        phase_statistics_shape = tuple(source["phase_mean"].shape)
    total = int(np.prod(spectrum_shape) + np.prod(phase_shape))
    if expected_total is not None and total != expected_total:
        raise ValueError(f"Encoded latent has {total} elements; expected {expected_total}")
    if len(spectrum_statistics_shape) != 2 or spectrum_statistics_shape[0] != 2:
        raise ValueError("Spectrum latent statistics are not separated by base station")
    if len(phase_statistics_shape) != 2 or phase_statistics_shape[0] != 2:
        raise ValueError("Detail latent statistics are not separated by base station")
    return {
        "spectrum_shape": list(spectrum_shape),
        "phase_shape": list(phase_shape),
        "total_latent_elements": total,
        "latent_statistics": "per_cell",
    }


def verify_stage(root: Path, stage: str, require_best: bool) -> dict:
    stage_root = root / stage
    checkpoints = ["final.pt", "last.pt"]
    if require_best:
        checkpoints.append("best.pt")
    for name in checkpoints:
        require_file(stage_root / name, minimum_bytes=1024)
    result = {"history": verify_history(stage_root / "history.jsonl")}
    summary_path = stage_root / "summary.json"
    if summary_path.is_file():
        result["summary"] = read_json(summary_path)
    return result


def verify_fold0(project: Path) -> dict:
    artifact_root = project / "artifacts" / "fold0"
    result = {
        "autoencoder": verify_ae(project),
        "context": verify_stage(artifact_root, "context", require_best=True),
    }
    result["encoded"] = verify_encoded(artifact_root / "encoded.npz", 30720)
    mask_report = read_json(artifact_root / "context_mask_report.json")
    if mask_report.get("status") != "PASS":
        raise ValueError("Context training-mask support gate did not pass")
    result["context_mask_support"] = mask_report
    result["evaluations"] = {
        name: verify_evaluation(artifact_root / name / "evaluation.json")
        for name in ("autoencoder", "context")
    }
    ablation = read_json(artifact_root / "autoencoder" / "ablation.json")
    gate = read_json(artifact_root / "autoencoder" / "quality_gate.json")
    if gate.get("status") != "PASS" or not gate.get("context_training_allowed"):
        raise ValueError("Fold0 AE quality gate did not pass")
    result["autoencoder_ablation"] = {
        "detail_gain": float(ablation["detail_gain"]),
        "shuffle_drop": float(ablation["shuffle_drop"]),
    }
    result["autoencoder_quality_gate"] = gate
    scan = read_json(artifact_root / "context" / "outage_scan.json")
    threshold = float(scan["best_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid Fold0 outage threshold: {threshold}")
    require_file(artifact_root / "stage_gap.json")
    result["best_outage_threshold"] = threshold
    result["output"] = verify_output(
        project / "outputs" / "fold0" / "Round2_Test_Channel.npy", 500
    )
    return result


def verify_final(project: Path) -> dict:
    artifact_root = project / "artifacts" / "final"
    result = {
        name: verify_stage(artifact_root, name, require_best=False)
        for name in ("autoencoder", "context")
    }
    result["encoded"] = verify_encoded(artifact_root / "encoded.npz", 30720)
    mask_report = read_json(artifact_root / "context_mask_report.json")
    if mask_report.get("status") != "PASS":
        raise ValueError("Final Context training-mask support gate did not pass")
    result["context_mask_support"] = mask_report
    output_path = project / "outputs" / "final" / "Round2_Test_Channel.npy"
    result["output"] = verify_output(output_path, 500)
    output_report = read_json(output_path.with_suffix(".json"))
    if int(output_report.get("shape", [0])[0]) != 500:
        raise ValueError("Final output JSON does not describe 500 samples")
    context_checkpoint = artifact_root / "context" / "final.pt"
    if output_path.stat().st_mtime_ns < context_checkpoint.stat().st_mtime_ns:
        raise ValueError("Final output is older than the final Context checkpoint")
    return result


def verify_smoke(project: Path) -> dict:
    artifact_root = project / "artifacts" / "smoke"
    result = {
        name: verify_stage(artifact_root, name, require_best=True)
        for name in ("autoencoder", "context")
    }
    result["encoded"] = verify_encoded(artifact_root / "encoded.npz", None)
    result["output"] = verify_output(
        project / "outputs" / "smoke" / "Round2_Test_Channel.npy", 2
    )
    return result


def verify_capacity(project: Path) -> dict:
    root = project / "artifacts" / "capacity"
    reports = {
        "one_sample": read_json(root / "one_sample.json"),
        "thirty_two_samples": read_json(root / "thirty_two_samples.json"),
    }
    for name, report in reports.items():
        score = float(report["metrics"]["score"])
        minimum = float(report["minimum_score"])
        if report.get("architecture") != "factorized_residual_v4":
            raise ValueError(f"Capacity report {name} uses an old AE architecture")
        if int(report.get("total_latent_dim", 0)) != 30720:
            raise ValueError(f"Capacity report {name} did not test 30,720 latents")
        if report.get("status") != "PASS" or score < minimum:
            raise ValueError(
                f"Capacity report {name} failed: score={score}, minimum={minimum}"
            )
    if int(reports["one_sample"].get("samples", 0)) != 1:
        raise ValueError("One-sample capacity report did not use exactly one sample")
    if int(reports["thirty_two_samples"].get("samples", 0)) != 32:
        raise ValueError("32-sample capacity report did not use exactly 32 samples")
    return reports


def verify_ae(project: Path) -> dict:
    root = project / "artifacts" / "fold0" / "autoencoder"
    checkpoint = root / "best.pt"
    require_file(checkpoint, minimum_bytes=1024)
    result = {
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
        }
    }
    summary = read_json(root / "summary.json")
    if summary.get("architecture") != "factorized_residual_v4":
        raise ValueError("Fold0 AE analysis uses an old architecture")
    latent_elements = int(summary.get("spectrum_latent_dim", 0)) + int(
        summary.get("phase_latent_dim", 0)
    )
    if latent_elements != 30720:
        raise ValueError(
            f"Fold0 AE has {latent_elements} latent elements; expected 30720"
        )

    evaluation = verify_evaluation(root / "evaluation.json")
    ablation = read_json(root / "ablation.json")
    detail_gain = float(ablation["detail_gain"])
    shuffle_drop = float(ablation["shuffle_drop"])
    if not math.isfinite(detail_gain) or not math.isfinite(shuffle_drop):
        raise ValueError("Fold0 AE ablation contains non-finite measurements")

    gate = read_json(root / "quality_gate.json")
    gate_status = str(gate.get("status", ""))
    if gate_status not in {"PASS", "FAIL", "SKIPPED"}:
        raise ValueError(f"Invalid Fold0 AE quality gate status: {gate_status}")
    expected_allowed = gate_status in {"PASS", "SKIPPED"}
    if bool(gate.get("context_training_allowed")) != expected_allowed:
        raise ValueError("Fold0 AE quality gate continuation flag is inconsistent")
    measurements = gate.get("measurements", {})
    comparisons = {
        "score": (float(measurements["score"]), evaluation["score"]),
        "detail_gain": (float(measurements["detail_gain"]), detail_gain),
        "shuffle_drop": (float(measurements["shuffle_drop"]), shuffle_drop),
    }
    for name, (reported, measured) in comparisons.items():
        if not math.isfinite(reported) or abs(reported - measured) > 1e-6:
            raise ValueError(
                f"Fold0 AE gate {name}={reported} does not match evaluation {measured}"
            )

    result.update(
        {
            "summary": summary,
            "evaluation": evaluation,
            "ablation": {
                "detail_gain": detail_gain,
                "shuffle_drop": shuffle_drop,
            },
            "quality_gate": gate,
            "total_latent_elements": latent_elements,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a Scheme C run is complete")
    parser.add_argument(
        "--stage",
        choices=("smoke", "capacity", "ae", "fold0", "final"),
        required=True,
    )
    parser.add_argument(
        "--project", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    project = Path(args.project).resolve()
    try:
        if args.stage == "smoke":
            details = verify_smoke(project)
        elif args.stage == "capacity":
            details = verify_capacity(project)
        elif args.stage == "ae":
            details = verify_ae(project)
        elif args.stage == "fold0":
            details = verify_fold0(project)
        else:
            details = verify_final(project)
    except (KeyError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"status": "FAILED", "stage": args.stage, "error": str(error)},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from None
    report = {"status": "PASS", "stage": args.stage, "details": details}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
