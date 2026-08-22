#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs ../research/scheme_e_065

python -u scripts/diagnose_carrier_quality_fallback.py \
  --config configs/v4_fold_best.json \
  --policy reports/generated/v4_attempt1_policy.json \
  --prior-wave-number -140.33 \
  --minimum-quality 0.5 \
  --expected-baseline 0.627089141574626 \
  --report ../research/scheme_e_065/L0_014_CARRIER_QUALITY_FALLBACK.json \
  --device cuda
