#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs artifacts/scheme_e_065/l1_residual_probe ../research/scheme_e_065

python scripts/train_residual_coefficient_probe.py \
  --config configs/v4_fold_best.json \
  --cache-dir ../research/scheme_e_065/residual_rank \
  --baseline-prediction ../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy \
  --policy reports/generated/v4_attempt1_policy.json \
  --rank 16 \
  --neighbors 16 \
  --width 192 \
  --epochs 300 \
  --patience 50 \
  --batch-size 96 \
  --validation-batch-size 128 \
  --decode-batch-size 8 \
  --minimum-inner-gain 0.004 \
  --device cuda \
  --output-dir artifacts/scheme_e_065/l1_residual_probe \
  --report ../research/scheme_e_065/L1_RESIDUAL_PROBE.json
