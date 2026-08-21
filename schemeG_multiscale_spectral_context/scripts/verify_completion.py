from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_g.config import load_config
from scheme_g.reporting import evaluation_metrics


def _require(path: Path, minimum: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum:
        raise ValueError(f"missing or incomplete file: {path}")


def _read(path: Path) -> dict:
    _require(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Scheme G formal artifacts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("fold0", "final"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        encoded = Path(config["encoding"]["output_path"])
        _require(encoded, 128)
        with np.load(encoded) as source:
            total = int(
                np.prod(source["spectrum_shape"]) + np.prod(source["phase_shape"])
            )
            available = (
                int(source["available"].sum()) if "available" in source.files else 0
            )
        if total != 30_720:
            raise ValueError(f"encoded latent has {total} elements, expected 30720")
        context_dir = Path(config["context"]["output_dir"])
        checkpoint_name = "best.pt" if args.stage == "fold0" else "final.pt"
        _require(context_dir / checkpoint_name, 1_000_000)
        summary = _read(context_dir / "summary.json")
        details: dict[str, object] = {
            "encoded_samples": available,
            "total_latent_elements": total,
            "context": summary,
            "checkpoint": str(context_dir / checkpoint_name),
        }
        if args.stage == "fold0":
            evaluation_report = _read(context_dir / "evaluation.json")
            evaluation = evaluation_metrics(evaluation_report)
            for name in ("pas", "pdp", "nmse", "score"):
                if not math.isfinite(float(evaluation[name])):
                    raise ValueError(f"non-finite validation {name}")
            details["evaluation"] = evaluation
            details["outage_scan"] = _read(context_dir / "outage_scan.json")
        else:
            output = Path(config["inference"]["output_path"])
            _require(output, 128)
            array = np.load(output, mmap_mode="r")
            if array.shape != (500, 256, 4, 192) or array.dtype != np.complex64:
                raise ValueError(
                    f"invalid final output shape={array.shape}, dtype={array.dtype}"
                )
            if not np.isfinite(array).all():
                raise ValueError("final output contains NaN or Inf")
            details["test_output"] = {
                "shape": list(array.shape),
                "bytes": output.stat().st_size,
            }
        report = {"status": "PASS", "stage": args.stage, "details": details}
    except (KeyError, OSError, ValueError) as error:
        report = {"status": "FAILED", "stage": args.stage, "error": str(error)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
