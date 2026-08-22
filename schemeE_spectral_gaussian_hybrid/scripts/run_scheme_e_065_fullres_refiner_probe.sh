#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs artifacts/scheme_e_065/l1_004_fullres_refiner ../research/scheme_e_065

python -u scripts/train_fullres_magnitude_refiner.py \
  --config configs/v4_fold_best.json \
  --latent-cache ../research/scheme_e_065/residual_rank \
  --map-cache artifacts/scheme_e_065/fullres_log_power_cache \
  --baseline-prediction ../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy \
  --width 32 \
  --blocks 5 \
  --maximum-residual 4.0 \
  --epochs 80 \
  --patience 10 \
  --validation-interval 2 \
  --minimum-inner-gain 0.004 \
  --batch-size 8 \
  --validation-batch-size 4 \
  --delay-crop 96 \
  --output-dir artifacts/scheme_e_065/l1_004_fullres_refiner \
  --report ../research/scheme_e_065/L1_004_FULLRES_REFINER.json \
  --device cuda
