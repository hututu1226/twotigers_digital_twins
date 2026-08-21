from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def _read(path: str | Path) -> dict:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8")) if source.is_file() else {
        "status": "MISSING",
        "path": str(source),
    }


def main() -> None:
    selection = _read("reports/generated/v2_attempt_selection.json")
    selected_attempt = int(selection.get("selected_attempt", 0) or 0)
    report = {
        "scheme": "E-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_baseline": {
            "scheme": "E-v1",
            "score": 0.59,
            "source": "competition leaderboard feedback supplied by the user",
        },
        "strict_prior": _read(
            "artifacts/v2/fold0/spectral_teacher/strict_report.json"
        ),
        "attempt_selection": selection,
        "selected_policy": _read(
            f"reports/generated/v2_attempt{selected_attempt}_policy.json"
        ),
        "final_config": _read("configs/v2_final_selected.json"),
        "final_teacher": _read(
            "artifacts/v2/final/spectral_teacher/final_report.json"
        ),
        "final_hybrid": _read("artifacts/v2/final/hybrid/summary.json"),
        "inference": _read("reports/generated/v2_final_inference.json"),
        "output_check": _read("reports/generated/v2_final_output_check.json"),
        "test_accuracy_available": False,
        "note": "The 500 test labels are unavailable. Only the official baseline score and strict Fold metrics are accuracy evidence.",
    }
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "schemeE_v2_final_experiment_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_score = selection.get("selected_score", "missing")
    inference = report["inference"]
    check = report["output_check"]
    markdown = "\n".join(
        [
            "# Scheme E-v2 Final Experiment Report",
            "",
            "- E-v1 official baseline: `0.59`",
            f"- Selected strict Fold0 score: `{selected_score}`",
            f"- Selected attempt: `{selection.get('selected_attempt', 'missing')}`",
            f"- Predicted outages: `{inference.get('predicted_outages', 'missing')}`",
            f"- Output shape: `{check.get('shape', 'missing')}`",
            f"- Output dtype: `{check.get('dtype', 'missing')}`",
            f"- SHA256: `{check.get('sha256', 'missing')}`",
            "",
            "The test labels are unavailable, so output validation is not reported as test accuracy.",
            "The strict Fold0 prior excludes every Fold0 validation label from teacher fitting.",
            "",
        ]
    )
    md_path = output_dir / "schemeE_v2_final_EXPERIMENT_REPORT.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": "PASS", "json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
