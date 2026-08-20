from __future__ import annotations


def evaluation_metrics(report: dict[str, object]) -> dict[str, object]:
    metrics = report.get("metrics")
    return metrics if isinstance(metrics, dict) else report
