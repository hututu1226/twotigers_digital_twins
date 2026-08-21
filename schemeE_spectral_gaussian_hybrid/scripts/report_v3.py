from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def _read(path: str | Path) -> dict:
    source = Path(path)
    if source.is_file():
        return json.loads(source.read_text(encoding="utf-8"))
    return {"status": "MISSING", "path": str(source)}


def main() -> None:
    selection = _read("reports/generated/v3_attempt_selection.json")
    selected_attempt = int(selection.get("selected_attempt", 0) or 0)
    report = {
        "scheme": "E-v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_baselines": {
            "round1_program": 0.62,
            "scheme_e_v1": 0.59,
            "scheme_f": 0.52,
            "source": "competition feedback supplied by the user",
        },
        "round1_transfer_probe": {
            "samples": 256,
            "nearest_oracle_power_score": 0.54794,
            "aligned_idw8_oracle_power_score": 0.57027,
            "bs0_nearest": 0.58700,
            "bs0_aligned_idw8": 0.63003,
            "bs1_nearest": 0.50933,
            "bs1_aligned_idw8": 0.52473,
            "note": "Diagnostic only; target power was supplied to isolate seed shape.",
        },
        "attempt_selection": selection,
        "selected_policy": _read(
            f"reports/generated/v3_attempt{selected_attempt}_policy.json"
        ),
        "final_config": _read("configs/v3_final_selected.json"),
        "final_hybrid": _read("artifacts/v3/final/hybrid/summary.json"),
        "inference": _read("reports/generated/v3_final_inference.json"),
        "output_check": _read("reports/generated/v3_final_output_check.json"),
        "test_accuracy_available": False,
    }
    output_dir = Path("reports/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "schemeE_v3_final_experiment_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_score = selection.get("selected_score", "missing")
    inference = report["inference"]
    check = report["output_check"]
    markdown = "\n".join(
        [
            "# Scheme E-v3 Final Experiment Report",
            "",
            "- Round1 official reference: `0.62`",
            "- Scheme E-v1 official baseline: `0.59`",
            f"- Selected strict Fold0 score: `{selected_score}`",
            f"- Selected attempt: `{selection.get('selected_attempt', 'missing')}`",
            f"- Predicted outages: `{inference.get('predicted_outages', 'missing')}`",
            f"- Output shape: `{check.get('shape', 'missing')}`",
            f"- Output dtype: `{check.get('dtype', 'missing')}`",
            f"- SHA256: `{check.get('sha256', 'missing')}`",
            "",
            "The test labels are unavailable, so output validation is not test accuracy.",
            "Carrier slopes are fitted from visible train/train pairs only.",
            "",
        ]
    )
    md_path = output_dir / "schemeE_v3_final_EXPERIMENT_REPORT.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {"status": "PASS", "json": str(json_path), "markdown": str(md_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
