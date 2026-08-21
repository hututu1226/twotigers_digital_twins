from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from scheme_e.config import load_config
from scheme_e.policy_scan import scan_outage_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Scheme E-v2 per-BS outage policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(config["hybrid"]["output_dir"]) / "best.pt")
    summary_path = Path(args.summary or Path(config["hybrid"]["output_dir"]) / "summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    strategy_name = str(summary.get("selected_reference_strategy", "nearest"))
    strategies = config["hybrid"].get("reference_strategies", [{"name": "nearest", "top_k": 1}])
    strategy = next(
        (dict(value) for value in strategies if str(value.get("name")) == strategy_name),
        {"name": "nearest", "top_k": 1},
    )
    report = scan_outage_policy(
        config,
        checkpoint,
        int(summary["selected_projection_iterations"]),
        strategy,
        args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
