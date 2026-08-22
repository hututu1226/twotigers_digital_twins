#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs artifacts/scheme_e_065/l1_003_magnitude_gp ../research/scheme_e_065

python -u scripts/train_magnitude_gp_probe.py \
  --config configs/v4_fold_best.json \
  --cache-dir ../research/scheme_e_065/residual_rank \
  --baseline-prediction ../research/scheme_e_065/FOLD0_BASELINE_PREDICTION.npy \
  --strict-basis artifacts/scheme_e_065/l0_011_magnitude_residual/train_only_log_power_basis.pt \
  --rank 8 \
  --kernels rq10,rq20,matern20 \
  --alphas 0.5,1.0 \
  --gp-noise 0.01 \
  --feature-length 1.0 \
  --feature-mix 0.5 \
  --minimum-inner-gain 0.004 \
  --output-dir artifacts/scheme_e_065/l1_003_magnitude_gp \
  --report ../research/scheme_e_065/L1_003_MAGNITUDE_GP_PROBE.json \
  --device cuda
