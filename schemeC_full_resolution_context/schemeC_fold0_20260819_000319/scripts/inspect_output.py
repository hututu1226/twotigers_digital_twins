from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated channel NPY")
    parser.add_argument("path")
    parser.add_argument("--expected-count", type=int, default=500)
    args = parser.parse_args()
    path = Path(args.path)
    array = np.load(path, mmap_mode="r")
    expected_shape = (args.expected_count, 256, 4, 192)
    finite = bool(np.isfinite(array).all())
    valid = bool(tuple(array.shape) == expected_shape and array.dtype == np.complex64 and finite)
    result = {
        "path": str(path.resolve()),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": finite,
        "exact_zero_samples": int(np.sum(np.max(np.abs(array), axis=(1, 2, 3)) == 0)),
        "size_gib": path.stat().st_size / 1024**3,
        "expected_shape": list(expected_shape),
        "valid": valid,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
