from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        return {"status": "MISSING"}
    return json.loads(source.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate D/E scans into bounded Scheme F edits"
    )
    parser.add_argument("--scheme-d", required=True)
    parser.add_argument("--scheme-e", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    config = read(args.base)
    d_report = read(args.scheme_d)
    e_report = read(args.scheme_e)
    context = config["context"]
    context["output_dir"] = "artifacts/fold0_attempt1/context"
    config["inference"]["context_checkpoint"] = (
        "artifacts/fold0_attempt1/context/best.pt"
    )
    decisions: list[dict[str, object]] = []

    if d_report.get("status") == "PASS":
        reports = {item["name"]: item for item in d_report["reports"]}
        restored = float(reports["restore_training_temperature"]["metrics"]["score"])
        reload_bug = float(reports["reload_bug"]["metrics"]["score"])
        decisions.append(
            {
                "signal": "scheme_d_temperature_restore",
                "score_delta": restored - reload_bug,
                "action": "Scheme F temperature is persisted in state_dict",
            }
        )
        sparse = float(reports["sparse_top8"]["metrics"]["score"])
        no_warp = float(reports["sparse_top8_no_warp"]["metrics"]["score"])
        if no_warp > sparse + 0.003:
            context["spectrum_maximum_warp"] = [0.5, 1.0, 2.0]
            context["detail_maximum_warp"] = [1.0, 2.0, 4.0]
            decisions.append(
                {
                    "signal": "scheme_d_no_warp_better",
                    "score_delta": no_warp - sparse,
                    "action": "halve Scheme F warp range but retain learnable regional transport",
                }
            )

    if e_report.get("status") == "PASS":
        deltas = e_report["score_deltas"]
        outage_gain = float(deltas["oracle_outage"])
        power_gain = float(deltas["oracle_power"])
        bounded_gain = float(deltas["bounded_gp_power"])
        if power_gain >= 0.02:
            context["loss_weights"]["power"] = 0.32
            context["loss_weights"]["power_quantile"] = 0.12
            context["maximum_power_z"] = 4.0
            decisions.append(
                {
                    "signal": "scheme_e_power_oracle",
                    "score_delta": power_gain,
                    "action": "strengthen bounded PowerCNP and tighten normalized output range",
                }
            )
        if outage_gain >= 0.01:
            context["loss_weights"]["outage"] = 0.05
            context["outage_positive_weight"] = 5.0
            decisions.append(
                {
                    "signal": "scheme_e_outage_oracle",
                    "score_delta": outage_gain,
                    "action": "increase outage supervision; inference policy remains OOF-selected",
                }
            )
        decisions.append(
            {
                "signal": "scheme_e_bounded_gp_power",
                "score_delta": bounded_gain,
                "action": "GP power remains a feature only; it never controls final amplitude",
            }
        )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": "PASS",
        "scheme_d_status": d_report.get("status"),
        "scheme_e_status": e_report.get("status"),
        "selected_config": str(destination),
        "decisions": decisions,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
