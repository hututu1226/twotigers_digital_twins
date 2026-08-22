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
  --counts 8 \
  --strengths 0.25 \
  --aligned \
  --experiment-id L0-013 \
  --batch-size 4 \
  --report ../research/scheme_e_065/L0_013_ALIGNED_MAGNITUDE_TRANSFER.json \
  --device cuda
