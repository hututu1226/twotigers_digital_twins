from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from scheme_e.config import load_config


def _require(path: Path, minimum_bytes: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        raise ValueError(f"missing or incomplete file: {path}")


def _read(path: Path) -> dict:
    _require(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_metrics(summary: dict) -> None:
    metrics = summary.get("best_metrics", {})
    for name in ("pas", "pdp", "nmse", "score"):
        if name in metrics and not math.isfinite(float(metrics[name])):
            raise ValueError(f"non-finite {name} in hybrid summary")


def _verify_priors(path: Path, expected: int) -> dict:
    _require(path, 128)
    with np.load(path) as source:
        required = ("pas_log", "pdp_log", "ue_log_energy", "log_power", "outage_probability")
        missing = [name for name in required if name not in source.files]
        if missing:
            raise ValueError(f"prior file misses {missing}: {path}")
        count = len(source["pas_log"])
        finite = all(np.isfinite(source[name]).all() for name in required)
    if count != expected or not finite:
        raise ValueError(f"invalid priors: count={count}, expected={expected}, finite={finite}")
    return {"path": str(path), "samples": count, "finite": finite}


def _verify_output(path: Path, expected: int) -> dict:
    _require(path, 128)
    value = np.load(path, mmap_mode="r")
    shape = (expected, 256, 4, 192)
    if value.shape != shape or value.dtype != np.complex64 or not np.isfinite(value).all():
        raise ValueError(f"invalid output shape={value.shape}, dtype={value.dtype}")
    return {"path": str(path), "shape": list(value.shape), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Scheme E stage completion")
    parser.add_argument("--config", default="configs/fold0_5090.json")
    parser.add_argument("--stage", choices=("fold0", "final", "smoke"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    setup = _read(Path(config["data"]["root"]) / "Round2_Setup.json")
    train_count = int(setup["P_Train"])
    test_count = min(int(setup["P_Test"]), int(config["runtime"].get("test_limit", 0) or setup["P_Test"]))
    try:
        if args.stage in {"fold0", "smoke"}:
            oof = _verify_priors(Path(config["spectral_teacher"]["oof_output_path"]), train_count)
            hybrid_dir = Path(config["hybrid"]["output_dir"])
            _require(hybrid_dir / "best.pt", 1_000_000)
            summary = _read(hybrid_dir / "summary.json")
            _finite_metrics(summary)
            details = {"oof_priors": oof, "hybrid": summary}
        if args.stage in {"final", "smoke"}:
            test_priors = _verify_priors(Path(config["spectral_teacher"]["test_output_path"]), int(setup["P_Test"]))
            hybrid_dir = Path(config["hybrid_final"]["output_dir"])
            _require(hybrid_dir / "best.pt", 1_000_000)
            summary = _read(hybrid_dir / "summary.json")
            _finite_metrics(summary)
            final_details = {
                "test_priors": test_priors,
                "hybrid": summary,
                "test_output": _verify_output(Path(config["inference"]["output_path"]), test_count),
            }
            if args.stage == "smoke":
                details["final"] = final_details
            else:
                details = final_details
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
