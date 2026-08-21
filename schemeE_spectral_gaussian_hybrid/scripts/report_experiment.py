from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import _bootstrap  # noqa: F401
import torch


def _read(path: Path) -> dict:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_bootstrap.PROJECT_ROOT.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Scheme E machine-readable and Markdown experiment reports")
    parser.add_argument("--stage", choices=("fold0", "final"), required=True)
    parser.add_argument("--output-dir", default=str(_bootstrap.PROJECT_ROOT / "reports" / "generated"))
    args = parser.parse_args()
    root = _bootstrap.PROJECT_ROOT
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read(root / "artifacts" / "preprocessed_scheme_e" / "manifest.json")
    if args.stage == "fold0":
        spectral = _read(root / "artifacts" / "fold0" / "spectral_teacher" / "oof_report.json")
        hybrid = _read(root / "artifacts" / "fold0" / "hybrid" / "summary.json")
        inference = {}
        inspection = {}
        breakdown = _read(root / "reports" / "generated" / "fold0_breakdown.json")
    else:
        spectral = _read(root / "artifacts" / "final" / "spectral_teacher" / "final_report.json")
        hybrid = _read(root / "artifacts" / "final" / "hybrid" / "summary.json")
        inference = _read(root / "reports" / "generated" / "final_inference.json")
        inspection = _read(root / "reports" / "generated" / "final_output_check.json")
        breakdown = _read(root / "reports" / "generated" / "fold0_breakdown.json")
    report = {
        "scheme": "E",
        "stage": args.stage,
        "architecture": "spectral_gaussian_full_resolution_adapter_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "preprocessing": manifest,
        "spectral_teacher": spectral,
        "hybrid": hybrid,
        "inference": inference,
        "output_check": inspection,
        "validation_breakdown": breakdown,
        "test_accuracy_available": False,
        "test_accuracy_note": "Round2 test labels are not provided; only validation accuracy and output integrity can be reported.",
    }
    json_path = output_dir / f"schemeE_{args.stage}_experiment_report.json"
    md_path = output_dir / f"schemeE_{args.stage}_EXPERIMENT_REPORT.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = hybrid.get("best_metrics", {})
    lines = [
        f"# Scheme E {args.stage.upper()} Experiment Report",
        "",
        f"- Git commit: `{report['git_commit']}`",
        f"- Architecture: `{report['architecture']}`",
        f"- GPU: `{report['environment']['gpu'] or 'CPU'}`",
        f"- RF Gaussians: `{manifest.get('rf_gaussians', 'unknown')}`",
        f"- Geometry features: `{manifest.get('geometry_feature_count', 'unknown')}`",
        "",
        "## Validation",
        "",
    ]
    if args.stage == "fold0":
        lines.extend(
            [
                f"- PAS: `{best.get('pas', 'missing')}`",
                f"- PDP: `{best.get('pdp', 'missing')}`",
                f"- NMSE: `{best.get('nmse', 'missing')}`",
                f"- Score: `{best.get('score', 'missing')}`",
                f"- Best epoch: `{hybrid.get('best_epoch', 'missing')}`",
                f"- Selected projection iterations: `{hybrid.get('selected_projection_iterations', 'missing')}`",
                f"- Spectral teacher PAS proxy accuracy: `{spectral.get('pas_accuracy', 'missing')}`",
                f"- Spectral teacher PDP accuracy: `{spectral.get('pdp_accuracy', 'missing')}`",
                "- Per-BS / distance breakdown: `reports/generated/fold0_breakdown.json`",
            ]
        )
    else:
        lines.extend(
            [
                "The final model uses all 4,000 labeled channels, so no held-out score exists at this stage.",
                "Validation evidence is recorded in the Fold0 report.",
                "",
                "## Test Output",
                "",
                f"- Shape: `{inspection.get('shape', 'missing')}`",
                f"- Dtype: `{inspection.get('dtype', 'missing')}`",
                f"- SHA256: `{inspection.get('sha256', 'missing')}`",
                f"- Predicted outages: `{inference.get('predicted_outages', 'missing')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Test labels are unavailable. This report never treats format validation as test accuracy.",
            "The OOF spectral report and Fold0 channel score are the measurable generalization evidence.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
