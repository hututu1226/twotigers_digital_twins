from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated complex channel NPY")
    parser.add_argument("path")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--report")
    args = parser.parse_args()
    path = Path(args.path)
    array = np.load(path, mmap_mode="r")
    expected = (int(args.samples), 256, 4, 192)
    finite = bool(np.isfinite(array).all())
    if array.shape != expected or array.dtype != np.complex64 or not finite:
        raise RuntimeError(f"Invalid output: shape={array.shape}, dtype={array.dtype}, finite={finite}")
    zero = np.mean(np.abs(array), axis=(1, 2, 3)) <= 1e-30
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    report = {
        "status": "PASS",
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": finite,
        "zero_channels": int(zero.sum()),
        "bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
