#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs reports/generated artifacts/v5

run() {
  echo "[$(date '+%F %T')] $*"
  "$@"
}

run python scripts/prepare_v5_config.py
run python -m unittest discover -s tests -v
run python scripts/build_strict_fold_prior.py \
  --config configs/v5_local_teacher.json \
  --fold 0 \
  --output artifacts/v5/fold0/spectral_teacher/strict_priors.npz \
  --report artifacts/v5/fold0/spectral_teacher/strict_report.json
run python scripts/train_hybrid.py \
  --config configs/v5_local_teacher.json --stage fold0
run python scripts/verify_completion.py \
  --config configs/v5_local_teacher.json --stage fold0 \
  --output reports/generated/v5_fold0_completion.json
run python scripts/scan_v2_policy.py \
  --config configs/v5_local_teacher.json \
  --output reports/generated/v5_fold0_policy.json
run python scripts/scan_output_projection.py \
  --config configs/v5_local_teacher.json \
  --policy reports/generated/v5_fold0_policy.json \
  --output reports/generated/v5_fold0_output_projection.json

echo "[$(date '+%F %T')] Scheme E-v5 Fold0 complete"
