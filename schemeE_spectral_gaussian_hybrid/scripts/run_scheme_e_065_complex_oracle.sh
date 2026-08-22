#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs artifacts/scheme_e_065/l0_010_complex_residual ../research/scheme_e_065

python -u scripts/diagnose_complex_residual_oracle.py \
  --config configs/v4_fold_best.json \
  --cache-dir ../research/scheme_e_065/residual_rank \
  --baseline-prediction ../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy \
  --output-dir artifacts/scheme_e_065/l0_010_complex_residual \
  --report ../research/scheme_e_065/L0_010_COMPLEX_ORACLE.json \
  --ranks 0,8,16,32,64 \
  --pca-oversample 12 \
  --pca-iterations 3 \
  --batch-size 8 \
  --device cuda
