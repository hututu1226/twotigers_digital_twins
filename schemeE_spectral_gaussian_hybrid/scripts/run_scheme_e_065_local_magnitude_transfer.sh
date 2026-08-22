#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs ../research/scheme_e_065

python -u scripts/diagnose_local_magnitude_transfer.py \
  --config configs/v4_fold_best.json \
  --latent-cache ../research/scheme_e_065/residual_rank \
  --map-cache artifacts/scheme_e_065/fullres_log_power_cache \
  --baseline-prediction ../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy \
  --counts 1 4 8 \
  --strengths 0.25 0.5 1.0 \
  --batch-size 4 \
  --report ../research/scheme_e_065/L0_012_LOCAL_MAGNITUDE_TRANSFER.json \
  --device cuda
