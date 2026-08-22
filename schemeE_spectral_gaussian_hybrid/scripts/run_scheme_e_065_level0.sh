#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs ../research/scheme_e_065

python scripts/audit_scheme_e_065.py \
  --config configs/v4_fold_best.json \
  --policy reports/generated/v4_attempt1_policy.json \
  --projection-report reports/generated/v4_attempt1_output_projection.json \
  --extra-prior v5_base=artifacts/v5/fold0/spectral_teacher/strict_priors.npz \
  --extra-prior local_bank=artifacts/v6/fold0/local_bank_priors.npz \
  --extra-prior adaptive=artifacts/v6/fold0/adaptive_local_bank_priors.npz \
  --expected-score 0.62705 \
  --score-tolerance 0.0005 \
  --device cuda \
  --output-dir ../research/scheme_e_065
