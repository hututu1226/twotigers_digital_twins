#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs ../research/scheme_e_065/residual_rank

python scripts/diagnose_low_rank_residual.py \
  --config configs/v4_fold_best.json \
  --prediction ../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy \
  --output-dir ../research/scheme_e_065/residual_rank \
  --ranks 0,8,16,32,64,128 \
  --batch-size 8 \
  --device cuda
