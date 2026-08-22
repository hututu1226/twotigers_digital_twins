#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs reports/generated
python scripts/diagnose_teacher_predictability.py \
  --config configs/v4_fold_best.json \
  --output reports/generated/v4_teacher_predictability.json
