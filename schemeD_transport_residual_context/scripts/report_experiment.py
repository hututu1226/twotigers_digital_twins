from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import _bootstrap  # noqa: F401
import torch

from scheme_d.reporting import evaluation_metrics


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "MISSING"}


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_bootstrap.PROJECT_ROOT.parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Scheme D experiment report")
    parser.add_argument("--stage", choices=("fold0", "final"), required=True)
    args = parser.parse_args()
    root = _bootstrap.PROJECT_ROOT
    generated = root / "reports" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    evaluation_report = _read(root / "artifacts" / "fold0" / "context" / "evaluation.json")
    evaluation = evaluation_metrics(evaluation_report)
    fold_summary = _read(root / "artifacts" / "fold0" / "context" / "summary.json")
    final_summary = _read(root / "artifacts" / "final" / "context" / "summary.json")
    inference = _read(root / "outputs" / "final" / "Round2_Test_Channel.json")
    report = {
        "scheme": "D",
        "stage": args.stage,
        "architecture": "transport_residual_context_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _commit(),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "fixed_autoencoder": {"checkpoint_source": "Scheme C Fold0", "measured_score": 0.9491},
        "fold0_validation": evaluation,
        "fold0_training": fold_summary,
        "final_training": final_summary if args.stage == "final" else {},
        "test_inference": inference if args.stage == "final" else {},
        "test_accuracy_available": False,
        "test_accuracy_note": "Round2 test labels are unavailable; output integrity is not test accuracy.",
    }
    json_path = generated / f"schemeD_{args.stage}_experiment_report.json"
    md_path = generated / f"schemeD_{args.stage}_EXPERIMENT_REPORT.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Scheme D {args.stage.upper()} Experiment Report", "",
        f"- Git commit: `{report['git_commit']}`",
        f"- GPU: `{report['environment']['gpu'] or 'CPU'}`",
        "- Architecture: `transport_residual_context_v3`",
        "- AE: fixed Scheme C checkpoint, measured Fold0 score `0.9491`", "",
        "## Fold0 Validation", "",
        f"- PAS: `{evaluation.get('pas', 'missing')}`",
        f"- PDP: `{evaluation.get('pdp', 'missing')}`",
        f"- NMSE: `{evaluation.get('nmse', 'missing')}`",
        f"- Score: `{evaluation.get('score', 'missing')}`",
        f"- Effective neighbors: `{evaluation.get('router_effective_neighbors', 'missing')}`",
        f"- Router top-1 mass: `{evaluation.get('router_top1_mass', 'missing')}`",
        f"- Spectrum warp bins: `{evaluation.get('spectrum_warp_bins', 'missing')}`",
        f"- Detail warp bins: `{evaluation.get('detail_warp_bins', 'missing')}`",
        f"- Spectrum residual RMS: `{evaluation.get('spectrum_residual_rms', 'missing')}`",
        f"- Detail residual RMS: `{evaluation.get('phase_residual_rms', 'missing')}`", "",
    ]
    if args.stage == "final":
        lines += [
            "## Final Output", "",
            f"- Shape: `{inference.get('shape', 'missing')}`",
            f"- Predicted outages: `{inference.get('predicted_outages', 'missing')}`",
            "- Test accuracy cannot be computed without labels.", "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
