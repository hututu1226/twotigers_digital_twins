#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs reports/generated
python scripts/diagnose_spectral_teacher_ceiling.py \
  --config configs/v4_fold_best.json \
  --policy reports/generated/v4_attempt1_policy.json \
  --output reports/generated/v4_spectral_teacher_ceiling.json
