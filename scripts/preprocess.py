from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from channel_ai.preprocessing import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Round 2 positions, channels, and PLY map")
    parser.add_argument("--data-root", default="Round2_Map")
    parser.add_argument("--output-dir", default="artifacts/preprocessed")
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--link-samples", type=int, default=16)
    parser.add_argument("--local-grid", type=int, default=3)
    parser.add_argument("--local-radius", type=float, default=16.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--blocks-per-cell", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = run_preprocessing(
        args.data_root,
        args.output_dir,
        args.resolution,
        args.link_samples,
        args.local_grid,
        args.local_radius,
        args.validation_fraction,
        args.blocks_per_cell,
        args.chunk_size,
        args.seed,
    )
    print(
        f"Preprocessing complete: train={manifest['training_count']} "
        f"validation={manifest['validation_count']} outage={manifest['outage_count']}"
    )


if __name__ == "__main__":
    main()

